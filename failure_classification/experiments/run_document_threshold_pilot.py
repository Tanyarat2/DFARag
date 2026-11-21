"""Run a small live classification pilot without requiring a matching baseline run."""
import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from dotenv import dotenv_values
from failure_classification.errorclassification.data.case_io import ROOT, build_case_input, write_json
from failure_classification.errorclassification.pipeline.classifier import FailureClassifier
from failure_classification.errorclassification.api.client import ClassifierRunner
from failure_classification.errorclassification.data.models import Conversation, ConversationTurn
from failure_classification.errorclassification.prompts.constructor import EVIDENCE_FIRST_PROMPT
from failure_classification.errorclassification.validation.validator import gold_document_coverage, reranking_document_metrics, VALIDATION_POLICY
from failure_classification.errorclassification.taxonomy.labels import TAXONOMY_REVISION
from failure_classification.errorclassification.configuration.config import DEFAULT_CONFIG, load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='YAML settings file')
    parser.add_argument('--input', dest='input_file', type=Path)
    parser.add_argument('--output', dest='output_dir', type=Path)
    parser.add_argument('--model', dest='model_name')
    parser.add_argument('--max-failures', type=int)
    parser.add_argument('--max-passes', type=int)
    parser.add_argument('--method', choices=['b0', 'b1', 'b2'])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--limit', type=int)
    group.add_argument('--all-cases', dest='limit', action='store_const', const=None)
    parser.add_argument('--case-ids', dest='case_ids', nargs='+')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--reuse-cache-from', dest='reuse_cache_from', type=Path)
    parser.add_argument('--api-base-url')
    parser.add_argument('--api-key-env')
    parser.add_argument('--token-limit-parameter', choices=['max_completion_tokens', 'max_tokens'])
    options = vars(parser.parse_args())
    config_path = options.pop('config')
    try:
        settings = load_settings(config_path, overrides=options)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args = argparse.Namespace(**settings.to_dict())
    args.input, args.output, args.model = settings.input_file, settings.output_dir, settings.model_name
    args.reuse_cache_from = settings.reuse_cache_from
    for key, value in dotenv_values(ROOT / '.envrc').items():
        if value and key not in os.environ:
            os.environ[key] = value
    out = args.output
    if out.exists():
        raise ValueError('Use a new output directory to preserve earlier results')
    raw = args.input.read_text(encoding='utf-8')
    rows = json.loads(raw) if args.input.suffix == '.json' else [json.loads(line) for line in raw.splitlines() if line.strip()]
    if args.case_ids:
        rows = [row for row in rows if str(row.get('query_id', row.get('case_id'))) in args.case_ids]
    rows = rows[:args.limit]
    cases = [build_case_input(row) for row in rows]
    write_json(out / 'classifier_inputs.json', cases)
    write_json(out / 'manifest.json', {'model': args.model, 'limit': args.limit,
        'max_passes': args.max_passes, 'max_failures': args.max_failures, 'method': args.method,
        'config_file': str(config_path.resolve()), 'settings': settings.to_dict(), 'input_file': str(args.input.resolve()), 'taxonomy_revision': TAXONOMY_REVISION,
        'validation_policy': VALIDATION_POLICY, 'started_at': datetime.now().isoformat(),
        'note': 'Independent single-turn cases; saved reconstructed evidence, not confirmed historical logs.'})
    write_json(out / 'resolved_config.json', {'error_classification': settings.to_dict()})
    (out / 'classification_prompt.txt').write_text(EVIDENCE_FIRST_PROMPT)
    if args.reuse_cache_from:
        (out / 'case_caches').mkdir(exist_ok=True)
        for case in cases:
            cached = args.reuse_cache_from / 'case_caches' / f'{case["case_id"]}.json'
            if cached.exists():
                shutil.copy2(cached, out / 'case_caches' / cached.name)
    source_root = Path(__file__).resolve().parents[1] / 'errorclassification'
    for source in source_root.rglob('*.py'):
        snapshot = out / 'source_snapshot' / source.relative_to(source_root)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
    def one(case):
        qid = case['case_id']
        runner = ClassifierRunner(args.model, [], [], cache_namespace='independent_generation_pilot_v9',
            cache_path=out / 'case_caches' / f'{qid}.json',
            api_base_url=args.api_base_url, api_key_env=args.api_key_env,
            token_limit_parameter=args.token_limit_parameter)
        try:
            turn = ConversationTurn.from_case_input(case)
            result, audit = FailureClassifier(runner, max_passes=args.max_passes, max_failures=args.max_failures, method=args.method).classify(turn)
            write_json(out / 'audits' / f'{qid}.json', audit)
            write_json(out / 'failures' / f'{qid}.json', result)
            return qid, result, Conversation(str(qid), [turn]).to_dict(), {
                'api_calls': runner.api_calls, 'cache_hits': runner.cache_hits, 'tokens': runner.usage,
                'answerability_requests': result.get('answerability_requests'),
                'total_model_requests': result.get('total_model_requests')}
        finally:
            runner.client.close()
    predictions, conversations, usage, errors = {}, {}, {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, c): c['case_id'] for c in cases}
        for future in as_completed(futures):
            qid = futures[future]
            try:
                qid, result, conversation, counts = future.result()
                predictions[qid], conversations[qid], usage[qid] = result, conversation, counts
                print(f'{len(predictions)}/{len(cases)} case {qid}: {result["supported_failures"]}; primary={result["primary_failure"]}', flush=True)
            except Exception as error:
                errors[qid] = type(error).__name__
                print(f'Case {qid} failed: {type(error).__name__}', flush=True)
            write_json(out / 'predictions_by_query.json', predictions)
            write_json(out / 'conversations.json', [conversations[c['case_id']] for c in cases if c['case_id'] in conversations])
            write_json(out / 'usage.json', usage)
            write_json(out / 'execution_errors.json', errors)
    checks = {}
    lines = ['# Classification run', '',
        f'Model: {args.model}. Completed: {len(predictions)}/{len(cases)}. Saved reconstructed traces.', '',
        '| Case | Gold doc recall | Gold docs lost | Gold docs in top 3 | Computed pipeline flags | Accepted semantic proposals |',
        '|---|---|---|---|---|---|']
    for case in cases:
        qid = case['case_id']
        coverage, metrics = gold_document_coverage(case), reranking_document_metrics(case)
        expected = {'E4': None if coverage['status'] == 'unknown' else coverage['recall'] < 1,
                    'E7': metrics['e7_failure'], 'E8': metrics['e8_failure']}
        result = predictions.get(qid, {})
        accepted = result.get('supported_failures', [])
        checks[qid] = {'coverage': coverage, 'reranking': metrics, 'threshold_flags': expected,
                       'accepted_labels': accepted, 'emitted_pipeline_failures': result.get('pipeline_failures', []),
                       'generation_review': result.get('generation_review'),
                       'unemitted_threshold_failures': [k for k,v in expected.items() if v and k not in result.get('pipeline_failures', [])],
                       'accepted_threshold_conflicts': [k for k,v in expected.items() if v is not True and k in accepted]}
        flags = ', '.join(k for k,v in expected.items() if v) or 'none'
        loss = f'{len(metrics["lost_gold_document_ids"])}/{len(metrics["retrieved_gold_document_ids"])}'
        lines.append(f'| {qid} | {coverage["recall"]} | {loss} | {len(metrics["top_k_gold_document_ids"])} | {flags} | {", ".join(accepted) or "none"} |')
    lines += ['', 'Pipeline flags are computed directly, independently of model proposals and causal primary selection.',
              'Generation review status:']
    for qid, result in predictions.items():
        lines.append(f'- {qid}: {result.get("generation_review", {}).get("status", "unknown")}; '
                     f'answerability={str((result.get("generation_assessment") or {}).get("answerability", "unknown"))}; '
                     f'primary={result.get("primary_failure")}.')
    write_json(out / 'threshold_checks.json', checks)
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    if errors:
        raise SystemExit(f'{len(errors)} cases failed; see execution_errors.json')


if __name__ == '__main__':
    main()
