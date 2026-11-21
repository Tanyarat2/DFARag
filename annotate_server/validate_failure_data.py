"""Validate cohort, evidence boundaries, and hashes in a Failure_data export."""
import argparse
import hashlib
import json
from pathlib import Path

from build_failure_data import history_for, retained_keys
from make_result_review import ANNOTATORS, ROOT, RUNS, labelled, read_annotations


FORBIDDEN = {'annotator', 'annotations', 'failure_type', 'failure_types',
             'failure_taxonomy', 'legacy_failure_types', 'human_label',
             'human_score_band', 'review_complete', 'flagged_for_review',
             'gold_assessment', 'gold_notes', 'annotation_versions'}


def check_keys(value):
    if isinstance(value, dict):
        assert not FORBIDDEN.intersection(value), FORBIDDEN.intersection(value)
        for child in value.values():
            check_keys(child)
    elif isinstance(value, list):
        for child in value:
            check_keys(child)


def validate(root, output):
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    total = 0
    for method, run in RUNS.items():
        profiles, warnings = read_annotations(root / 'Annotate_result' / method, run)
        assert not warnings, warnings
        expected, removed = retained_keys(profiles)
        path = output / (method + '.json')
        with path.open(encoding='utf-8') as stream:
            rows = json.load(stream)
        assert isinstance(rows, list), path
        assert len(rows) == len(expected)
        assert {r['id'] + '#' + str(r['turn']) for r in rows} == expected
        assert len({r['case_id'] for r in rows}) == len(rows)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['methods'][method]['output']['sha256']
        inputs = {r['id']: r for r in json.loads(
            (root / 'judge_run/runs' / run / 'input.json').read_text(encoding='utf-8'))}
        for row in rows:
            check_keys(row)
            key = row['id'] + '#' + str(row['turn'])
            assert all(labelled(profiles[name].get(key, {})) for name in ANNOTATORS), key
            source = inputs[row['id']]
            history, current = history_for(source['conversation'], row['turn'])
            assert row['history'] == history
            assert row['conversation'] == history + current
            assert row['answer'] == current[-1]['content']
            assert row['gold'] == source['ground_truth_answers'][row['turn'] - 1]
            assert row['contexts'] == source['contexts'][row['turn'] - 1]
            assert [p['text'] for p in row['selected_passages']] == row['contexts']
            assert row['gold_support']['status'] == 'available'
            for field in ('retrieval_experiment', 'reranking_experiment'):
                experiment = row[field]['record']
                assert experiment['task_id'] == row['task_id']
                assert experiment['domain'] == row['domain']
                assert experiment['question'].strip() == row['question'].strip()
                assert experiment['inference_stage'] == 'new_retrieval_experiment'
        total += len(rows)
        print(f'{method}: verified {len(rows)} retained, {len(removed)} removed', flush=True)
    for source in manifest['sources']:
        assert hashlib.sha256((root / source['file']).read_bytes()).hexdigest() == source['sha256']
    print(f'PASS: {total} records; cohort, unique IDs, human-label exclusion, hashes, history, original answers/context, gold and experiment matches', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'Failure_data')
    args = parser.parse_args()
    validate(args.root.resolve(), args.output.resolve())
