"""Trace exact reference/fact text through saved evidence without semantic guesses.

`status` describes exact text availability in the declared saved collection.
Semantic absence and question-requiredness are never inferred from a failed match.
"""
from __future__ import annotations
import re

COLLECTIONS = ('source_documents', 'indexed_chunks', 'retrieved_candidates', 'generation_context')


def _reference_facts(case):
    facts, seen = [], set()
    for document in case.get('source_documents') or []:
        if not isinstance(document, dict):
            continue
        for excerpt in document.get('reference_excerpts') or []:
            text = excerpt if isinstance(excerpt, str) else excerpt.get('text') if isinstance(excerpt, dict) else None
            if isinstance(text, str) and text.strip() and text not in seen:
                seen.add(text)
                facts.append({'fact_id': f'reference_{len(facts)+1}', 'requirement': case.get('original_question', ''),
                              'supporting_text': text, 'origin': 'evaluator_reference_excerpt',
                              'requiredness': 'not_independently_verified'})
    return facts


def evidence_availability_map(case, facts=None):
    """Find actual text spans and preserve unknowns and reconstruction scope.

    Optional facts come from the one-call semantic assessment. Their relevance
    remains a model proposal; exact occurrence checks do not adjudicate it.
    """
    records = []
    input_facts = _reference_facts(case) if facts is None else facts if isinstance(facts, list) else []
    for index, fact in enumerate(input_facts):
        if not isinstance(fact, dict):
            continue
        text = fact.get('supporting_text')
        text = text if isinstance(text, str) and text.strip() else ''
        pattern = re.compile(r'\s+'.join(re.escape(token) for token in text.split())) if text else None
        cells = {}
        for field in COLLECTIONS:
            items = case.get(field)
            trace = case.get('trace_status', {}).get(field, 'unknown completeness')
            complete = isinstance(trace, str) and trace.lower().startswith('complete')
            matches = []
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict) or not isinstance(item.get('text'), str):
                    continue
                match = pattern.search(item['text']) if pattern else None
                if match:
                    item_id = item.get('doc_id', item.get('id')) if field == 'source_documents' else item.get('chunk_id')
                    matches.append({'source': field, 'item_id': str(item_id) if item_id is not None else None,
                                    'quote': match.group(0), 'observation': 'Exact reference/fact span occurs in this saved record.'})
            status = 'present' if matches else 'absent' if pattern is not None and isinstance(items, list) and complete else 'unknown'
            cells[field] = {'status': status, 'matching_scope': 'exact_text_with_whitespace_normalization',
                            'semantic_availability': 'not_independently_verified' if matches else 'unknown',
                            'semantic_absence_established': False, 'trace_status': trace,
                            'scope': 'provided saved collection only; not the full corpus or a verified historical run',
                            'evidence': matches[:3]}
        support_ids = {item['item_id'] for item in cells['indexed_chunks']['evidence'] if item['item_id']}
        for field in ('retrieved_candidates', 'generation_context'):
            items = case.get(field)
            observed = {str(item.get('chunk_id')) for item in (items or []) if isinstance(item, dict)}
            cells[field]['indexed_support_ids_observed'] = sorted(support_ids & observed)
            cells[field]['indexed_support_ids_not_observed'] = sorted(support_ids - observed)
            cells[field]['id_absence_scope'] = 'not observed in saved records; equivalent support must still be assessed'
        records.append({'fact_id': str(fact.get('fact_id', f'fact_{index+1}')),
                        'requirement': fact.get('requirement', ''), 'supporting_text': text,
                        'origin': fact.get('origin', 'model_proposed_required_fact'),
                        'requiredness': fact.get('requiredness', 'not_independently_verified'), 'collections': cells})
    return {'policy': 'exact_text_and_membership_observations_v1',
            'warning': 'Reference excerpts are candidate support, not independently verified required facts. Exact-text absence is not semantic absence. Do not infer corpus unanswerability.',
            'facts': records}
