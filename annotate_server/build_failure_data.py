"""Export human-reviewed turns and evidence without human annotation fields.

Run from the repository root: python -B annotate_server/build_failure_data.py
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from conversation_numbers import numbers_for
from make_annotation import (conversation_exports, conversation_snapshot, find_turn,
                             gold_evidence, load_input_records, load_output, turn_texts)
from make_result_review import ANNOTATORS, ROOT, RUNS, labelled, read_annotations


def rejected(record):
    """Missing decisions do not count as a vote to remove a turn."""
    codes = set(record.get('failure_types') or
                ([record['failure_type']] if record.get('failure_type') else []))
    return bool(codes) and codes <= {'not_a_failure', 'excluded'} or (
        not codes and record.get('human_label') == 'correct')


def retained_keys(profiles):
    candidates = set.intersection(*(
        {key for key, record in profiles[name].items() if labelled(record)}
        for name in ANNOTATORS))
    removed = {key for key in candidates
               if all(rejected(profiles[name].get(key, {})) for name in ANNOTATORS)}
    return candidates - removed, removed


def file_info(path, root):
    return {'file': path.relative_to(root).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def history_for(conversation, turn):
    """Prior messages only; exclude the current question, answer, and future turns."""
    history, pending = [], []
    ordinal = 0
    for message in conversation:
        pending.append(message)
        if message['role'] == 'assistant':
            ordinal += 1
            if ordinal == turn:
                return history, pending
            history.extend(pending)
            pending = []
    raise ValueError(f'Conversation has no assistant turn {turn}')


def generation_index(folder):
    index = defaultdict(list)
    for filename, digest, export in conversation_exports(conversation_snapshot(folder)):
        source = {'file': 'conversation/' + filename, 'sha256': digest,
                  'rag_name': export.get('rag_name')}
        for conversation in export['conversations']:
            for turn in conversation['turns']:
                index[(conversation['id'], str(turn['dataset_turn']))].append((source, turn))
    return index


def match_generation(index, cid, dataset_turn, question, answer, contexts, rag):
    # Some historical Vector judge inputs contain SelfRAG outputs. Exact content
    # matches preserve that fact instead of substituting today's Vector export.
    matches = [(source, turn) for source, turn in index.get((cid, str(dataset_turn)), [])
               if turn.get('question') == question and turn.get('answer') == answer
               and [p['text'] for p in turn.get('passages', [])] == contexts]
    matches.sort(key=lambda item: (item[0]['rag_name'] != rag, item[0]['file']))
    return ({'status': 'exact_match', 'sources': [source for source, _ in matches],
             'turn': matches[0][1]} if matches else
            {'status': 'unavailable', 'sources': [], 'turn': None})


def experiment_index(folder):
    index = defaultdict(list)
    for filename in ('candidates.jsonl', 'reranked.jsonl'):
        for path in sorted(folder.rglob(filename)):
            source = file_info(path, folder.parent)
            with path.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = (filename, record['method'], record['domain'], record['task_id'])
                    index[key].append({'source': source, 'record': record})
    return index


def match_experiment(index, filename, rag, domain, task_id, question):
    matches = [item for item in index.get((filename, rag, domain, task_id), [])
               if item['record']['question'].strip() == question.strip()]
    if len(matches) > 1:
        raise ValueError(f'Ambiguous {filename} evidence for {rag}/{task_id}')
    return matches[0] if matches else None


def build(root):
    generations = generation_index(root / 'conversation')
    experiments = experiment_index(root / 'conversation')
    gold_path = root / 'judge_run/gold_support.json'
    gold_index = json.loads(gold_path.read_text(encoding='utf-8'))
    methods, summary = {}, {}
    sources = [file_info(gold_path, root)]
    for method, run in RUNS.items():
        folder = root / 'judge_run/runs' / run
        profiles, warnings = read_annotations(root / 'Annotate_result' / method, run)
        if warnings:
            raise ValueError('; '.join(warnings))
        keys, removed = retained_keys(profiles)
        inputs = load_input_records(folder)
        judged, judge_file = load_output(folder, 'mtrag')
        numbers = numbers_for(inputs)
        rows = []
        sources.extend(file_info(path, root) for path in (
            folder / 'input.json', folder / judge_file,
            *sorted((root / 'Annotate_result' / method).glob(f'annotations_{run}_*.json'))))
        for key in sorted(keys, key=lambda k: (
                int(numbers[k.rsplit('#', 1)[0]][1:]), int(k.rsplit('#', 1)[1]))):
            cid, turn = key.rsplit('#', 1)
            turn = int(turn)
            source = inputs[cid]
            question, answer = turn_texts(source['conversation'])[turn - 1]
            history, current = history_for(source['conversation'], turn)
            metadata = source.get('metadata', {})
            dataset_turn = (metadata.get('turn_ids') or list(range(1, turn + 1)))[turn - 1]
            task_id = f'{cid}<::>{dataset_turn}'
            domain = metadata.get('collection') or metadata.get('dataset')
            rag = metadata.get('rag')
            contexts = source.get('contexts')
            selected = contexts[turn - 1] if contexts is not None else None
            gold = source['ground_truth_answers'][turn - 1]
            gold_support = gold_evidence(gold_index, task_id, question, gold)
            generation = match_generation(generations, cid, dataset_turn,
                                          question, answer, selected, rag)
            original = generation['turn'] or {}
            candidates = match_experiment(experiments, 'candidates.jsonl', rag,
                                          domain, task_id, question)
            reranked = match_experiment(experiments, 'reranked.jsonl', rag,
                                        domain, task_id, question)
            gold_task = gold_index['tasks'].get(task_id)
            answerability = source.get('answerability')
            rows.append({
                'case_id': f'{method}:{key}', 'method': method, 'run': run,
                'id': cid, 'conversation_number': numbers[cid], 'turn': turn,
                'dataset_turn': dataset_turn, 'task_id': task_id, 'domain': domain,
                'question': question, 'answer': answer, 'history': history,
                'conversation': history + current,
                'gold': gold,
                'gold_answers': gold_task['gold_answers'] if gold_support['status'] == 'available' else [gold],
                'gold_documents': gold_support['documents'], 'gold_support': gold_support,
                'answerability': answerability[turn - 1] if answerability is not None else None,
                'contexts': selected,
                'selected_passages': original.get('passages') if generation['turn'] else (
                    [{'id': '', 'title': '', 'text': text} for text in selected]
                    if selected is not None else None),
                'generation_trace': generation,
                'original_retrieved_candidates': None,
                'original_reranking_trace': None,
                'retrieved_documents': candidates['record']['candidates'] if candidates else None,
                'retrieval_experiment': candidates, 'reranking_experiment': reranked,
                'input_metadata': metadata,
                'judge_evaluation': find_turn((judged or {}).get(cid), turn),
                'judge_metadata': (judged or {}).get(cid, {}).get('judge'),
                'provenance': {'input': (folder / 'input.json').relative_to(root).as_posix(),
                               'judge': (folder / judge_file).relative_to(root).as_posix(),
                               'gold': gold_path.relative_to(root).as_posix()},
            })
        methods[method] = rows
        summary[method] = {
            'reviewed_turns': len(keys) + len(removed), 'removed_turns': len(removed),
            'omitted_incomplete_turns': len({
                key for profile in profiles.values() for key, record in profile.items()
                if labelled(record)}) - len(keys) - len(removed),
            'retained_turns': len(rows),
            'gold_status': dict(Counter(r['gold_support']['status'] for r in rows)),
            'gold_document_count': sum(len(r['gold_documents']) for r in rows),
            'full_gold_document_count': sum(bool(d.get('full_document')) for r in rows for d in r['gold_documents']),
            'generation_trace_matches': sum(r['generation_trace']['status'] == 'exact_match' for r in rows),
            'generation_backend_mismatches': sum(bool(r['generation_trace']['sources']) and
                r['generation_trace']['sources'][0]['rag_name'] != r['input_metadata'].get('rag') for r in rows),
            'retrieval_experiment_matches': sum(r['retrieval_experiment'] is not None for r in rows),
            'reranking_experiment_matches': sum(r['reranking_experiment'] is not None for r in rows),
        }
    manifest = {'schema_version': 2,
                'format': 'Indented JSON array (UTF-8, two-space indentation).',
                'cohort': 'Intersection of turns with an explicit saved annotation decision from each of PS, KH and WN. Turns with any missing or undecided annotation are omitted; review marks alone do not qualify.',
                'filter': 'Within the fully labelled cohort, remove only when all three annotators chose not_a_failure or excluded (in any combination). A correctness-only correct decision counts as not_a_failure.',
                'human_annotations_in_records': False,
                'history_policy': 'All prior run-input messages; current turn stored separately; no future turns.',
                'retrieval_policy': 'retrieved_documents, retrieval_experiment and reranking_experiment come from a later retrieval experiment. They are not evidence of the candidates or reranking used for the original answer.',
                'gold_document_policy': 'Reference passages retain their full recorded text. full_document is null when the local source-document index is absent or the ID cannot be resolved. An empty gold_documents list can be valid for an unanswerable turn; check gold_support.status.',
                'methods': summary, 'sources': sources}
    return methods, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'Failure_data')
    parser.add_argument('--replace', action='store_true',
                        help='Refresh the generated dataset files in the output folder.')
    args = parser.parse_args()
    methods, manifest = build(args.root.resolve())
    paths = [args.output / f'{method}.json' for method in methods]
    paths.append(args.output / 'manifest.json')
    if any(path.exists() for path in paths) and not args.replace:
        raise SystemExit('Output already exists. Choose a new --output folder or use --replace to refresh it.')
    args.output.mkdir(parents=True, exist_ok=True)
    for method, rows in methods.items():
        path = args.output / f'{method}.json'
        with path.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(rows, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        manifest['methods'][method]['output'] = {
            'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f'{method}: {len(rows)} retained, {manifest["methods"][method]["removed_turns"]} removed')
    (args.output / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
