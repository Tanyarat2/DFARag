"""Run live dynamic attribution on saved Dragonball traces without gold access."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from .evaluator import FailureClassifier, InconclusiveDiagnosis

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class CachedLLM:
    def __init__(self, model, directory):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], timeout=120, max_retries=2)
        self.model, self.directory = model, directory
        self.api_calls = 0
        self.cache_hits = 0
        self.tokens = Counter()

    def __call__(self, system, payload):
        request = dict(model=self.model, messages=[
            dict(role="system", content=system),
            dict(role="user", content=json.dumps(payload, ensure_ascii=False))],
            response_format={"type": "json_object"})
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = self.directory / "requests" / (digest + '.json')
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text())["output"]
        response = self.client.chat.completions.create(**request)
        self.api_calls += 1
        usage = response.usage.model_dump() if response.usage else {}
        for name in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
            self.tokens[name] += usage.get(name, 0)
        raw = response.choices[0].message.content or ''
        # Preserve invalid raw outputs too, rather than losing diagnostic evidence.
        record = dict(request=request, raw_output=raw, response_model=response.model,
                      usage=usage, finish_reason=response.choices[0].finish_reason)
        write(path.with_suffix('.raw.json'), record)
        record['output'] = json.loads(raw)
        write(path, record)
        return record['output']


class DragonballVectorSearch:
    """Lazy retrieval-only reconstruction; no reranker or answer generator needed."""
    def __init__(self, output):
        self.output = output
        self.embedding = None
        self.calls = []

    def load(self):
        import torch
        sys.path.insert(0, str(ROOT / 'rag-error-classification'))
        from clients.local_embedding_client import LocalEmbeddingClient
        from clients.local_numpy_client import LocalNumpyVectorDBClient
        from failure_classification.evidence.dragonball_reconstruction import reconstruct_document_chunks
        from failure_classification.rag.dragonball_rag_adapter import GTE_MODEL_REVISION, GTE_REMOTE_CODE_REVISION
        config = SimpleNamespace(model='Alibaba-NLP/gte-large-en-v1.5', batch_size=20,
                                 device='cuda:0' if torch.cuda.is_available() else 'cpu',
                                 revision=GTE_MODEL_REVISION, code_revision=GTE_REMOTE_CODE_REVISION)
        corpus = ROOT / 'rag-error-classification/data/Dragonball/data/dragonball_corpus_en_processed.jsonl'
        chunks = [chunk for line in corpus.read_text().splitlines() if line.strip()
                  for chunk in reconstruct_document_chunks(json.loads(line), 128, 25)]
        print(f'Building secondary vector index: {len(chunks)} corpus chunks', flush=True)
        self.embedding = LocalEmbeddingClient(config)
        vectors = self.embedding.encode([c['text'] for c in chunks], config.batch_size).numpy()
        self.database = LocalNumpyVectorDBClient()
        self.database.add(vectors, [c['chunk_id'] for c in chunks], chunks)
        self.metadata = dict(corpus=str(corpus.relative_to(ROOT)),
                             corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
                             chunk_count=len(chunks), chunk_size=128, overlap=25,
                             embedding=vars(config), historical_index_equivalence_verified=False)
        write(self.output / 'secondary_index.json', self.metadata)

    def __call__(self, question, top_k):
        if self.embedding is None:
            self.load()
        vector = self.embedding.encode([question], 1).numpy()
        ids, chunks, scores = self.database.search(vector, top_k)
        result = dict(chunks=[c['text'] for c in chunks[0]], exhaustive=False,
                      scope='Full local Dragonball corpus; reconstructed GTE vector index, top-k only; historical identity unverified')
        self.calls.append(dict(question=question, top_k=top_k,
                               chunk_ids=ids[0].tolist(), scores=scores[0].tolist(), result=result))
        write(self.output / 'secondary_searches.json', self.calls)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'result/failure_classification/six_native/luna_first10/classifier_inputs.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'result/failure_classification/dynamic_evaluator/first10')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--model', default='gpt-4o-mini')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    rows = json.loads(args.input.read_text())[:args.limit]
    cases = [dict(case_id=str(r['case_id']), artifacts=dict(
        user_query=r['original_question'], rag_answer=r['original_rag_answer'],
        top_k_chunks=[c['text'] for c in r['retrieved_candidates']],
        prompt_context=[c['text'] for c in r['generation_context']])) for r in rows]
    write(args.output_dir / 'inputs.json', cases)
    llm = CachedLLM(args.model, args.output_dir)
    search = DragonballVectorSearch(args.output_dir)
    classifier = FailureClassifier(llm, search)
    results = []
    for i, case in enumerate(cases, 1):
        print(f"Evaluating {i}/{len(cases)}: {case['case_id']}", flush=True)
        try:
            diagnosis = classifier.evaluate(case['artifacts'])
            record = dict(case_id=case['case_id'], status='classified', diagnosis=diagnosis.model_dump(mode='json'))
        except InconclusiveDiagnosis as exc:
            record = dict(case_id=case['case_id'], status='inconclusive', reason=str(exc))
        except Exception as exc:
            record = dict(case_id=case['case_id'], status='error', error_type=type(exc).__name__, reason=str(exc))
        results.append(record)
        write(args.output_dir / 'results.json', results)
        print(f"  {record['status']}: {record.get('diagnosis', {}).get('attributed_failure', record.get('reason', ''))}", flush=True)
    summary = dict(case_count=len(results), status_counts=dict(Counter(r['status'] for r in results)),
                   failure_counts=dict(Counter(r['diagnosis']['attributed_failure'] for r in results if r['status']=='classified')),
                   model=args.model, api_calls=llm.api_calls, cache_hits=llm.cache_hits,
                   usage=dict(llm.tokens), secondary_search_count=len(search.calls),
                   gold_or_prior_labels_passed_to_evaluator=False,
                   input_source=str(args.input), input_source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                   evaluator_sha256=hashlib.sha256(Path(__file__).with_name('evaluator.py').read_bytes()).hexdigest(),
                   limitations=['Saved reproduced traces; historical execution equivalence unverified.',
                                'This is an execution pilot, not an accuracy benchmark against adjudicated labels.',
                                'Single-label priority policy may mask additional defects.'])
    write(args.output_dir / 'summary.json', summary)
    report = [f"# Dynamic evaluator: {len(results)} cases", "", f"Model: `{args.model}`", "",
              "| Case | Status | Failure |", "| --- | --- | --- |"]
    for record in results:
        report.append(f"| {record['case_id']} | {record['status']} | "
                      f"{record.get('diagnosis', {}).get('attributed_failure', '—')} |")
    for record in results:
        diagnosis = record.get('diagnosis')
        report.extend(['', f"## Case {record['case_id']}", ''])
        if diagnosis:
            report.append(diagnosis['justification'])
            report.extend(['', 'Dynamic questions:', ''])
            report.extend(f"- {question}" for question in diagnosis['dynamic_sub_questions'])
        else:
            report.append(record.get('reason', 'Unknown error'))
    report.extend(['', 'Execution pilot only; no classification accuracy was measured.',
                   'Gold answers and prior labels were excluded from model inputs.',
                   'Historical execution/index equivalence is unverified.', ''])
    (args.output_dir / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    if any(r['status']=='error' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
