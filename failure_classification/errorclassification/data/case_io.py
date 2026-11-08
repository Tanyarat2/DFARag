"""Shared input preparation and result export for all classifiers."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

ANNOTATION_FIELDS = {
    'error_category', 'error_stage', 'original_error_category', 'original_error_stage',
    'gold_category', 'gold_category_raw', 'gold_stage_raw', 'gold_failure_type',
    'gold_failure', 'gold_error_category', 'gold_error_stage', 'gold_label_review',
    'primary_failure_type', 'secondary_failure_types', 'candidate_failure_types',
    'review_status', 'review_reason', 'predicted_present', 'failures', 'key_points', 'scores', 'num_rounds', 'max_failures', 'classification_status', 'classifications',
    'answerability_assessment', 'answerability_audit', 'answerability_requests',
    'e9_assessment', 'automatically_assigned_failures', 'pipeline_diagnostics', 'pipeline_failures',
    'generation_assessment', 'generation_review', 'initial_prediction', 'label_assessments', 'supported_failures',
    'model_prediction', 'validation', 'final_attribution', 'method_classifications', 'comparison', 'failure_label', 'stage_target',
}


def without_failure_labels(value):
    if isinstance(value, dict):
        return {key: without_failure_labels(item) for key, item in value.items()
                if key not in ANNOTATION_FIELDS}
    if isinstance(value, list):
        return [without_failure_labels(item) for item in value]
    return copy.deepcopy(value)


def build_case_input(row):
    """Accept prepared classifier inputs or the answer-only annotations dataset."""
    if 'case_id' in row and 'original_rag_answer' in row:
        case = without_failure_labels(row)
        if not isinstance(case['original_rag_answer'], str) or not case['original_rag_answer'].strip():
            raise ValueError('original_rag_answer must contain the evaluated answer')
        return case
    answer = row.get('rag_answer_final')
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Query {row.get('query_id')} needs rag_answer_final; no full-response fallback")
    case = {
        'case_id': str(row['query_id']), 'original_question': row['question'],
        'original_rag_answer': answer, 'gold_answer': row.get('answer'),
        'conversation_history': row.get('conversation_history', []),
        'source_documents': row.get('gold_documents'),
        'indexed_chunks': row.get('indexed_reference_chunks'),
        'retrieved_candidates': row.get('retrieved'),
        'generation_context': row.get('reranked'),
        'subquestions': [], 'probe_observations': [],
        'trace_status': {},
        'pipeline_metadata': {
            'evaluated_answer_field': 'rag_answer_final',
            'rerank_order': row.get('rerank_order'),
            'reference_chunk_provenance': row.get('gold_document_provenance'),
            'note': 'Saved reproduced passages; historical execution equivalence unverified. '
                    'Source documents and indexed chunks cover reference documents only, '
                    'not the full corpus. Gold references are evaluator evidence.',
        },
    }
    for field, expected in [('retrieved_candidates', 8), ('generation_context', 5)]:
        items = case[field]
        case['trace_status'][field] = ('missing' if items is None else
            f"{'complete' if len(items) == expected else 'partial'} "
            f'(saved top-{expected}; {len(items)} observed; historical equivalence unverified)')
    for field in ['source_documents', 'indexed_chunks']:
        case['trace_status'][field] = ('missing' if case[field] is None else
            'reference-document scope only; not a complete corpus/index record')
    return without_failure_labels(case)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


