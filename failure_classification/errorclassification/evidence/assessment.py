"""Independent answer-review completeness and deterministic pipeline diagnostics."""
from failure_classification.errorclassification.validation.validator import check_citation, gold_document_coverage, reranking_document_metrics

GENERATION_CODES = tuple(f'E{i}' for i in range(10, 17))
GENERATION_ASSESSMENT_POLICY = 'independent_generation_review_v1'


def generation_assessment_errors(record, case=None):
    """Check the observable assessment contract, not semantic truth."""
    if not isinstance(record, dict):
        return ['Independent generation assessment is missing']
    errors = []
    if isinstance(record.get('reason'), str) and record['reason'].strip().lower().rstrip('.') in {'case-specific assessment', 'case-specific explanation', 'specific evidence-based observation'}:
        errors.append('Generation assessment contains an unfilled placeholder')
    if record.get('answerability') not in ('answerable', 'partially_answerable', 'unanswerable', 'unknown'):
        errors.append('Generation answerability is missing or invalid')
    if record.get('response_behavior') not in ('answered', 'abstained', 'partial_with_limitation', 'unknown'):
        errors.append('Response behavior is missing or invalid')
    if not isinstance(record.get('reason'), str) or not record['reason'].strip():
        errors.append('Generation assessment needs an evidence-based explanation')
    checks = record.get('checks')
    for code in GENERATION_CODES:
        check = checks.get(code) if isinstance(checks, dict) else None
        if (not isinstance(check, dict) or check.get('status') not in ('defect', 'no_defect', 'unknown')
                or not isinstance(check.get('reason'), str) or not check['reason'].strip()
                or check['reason'].strip().lower().rstrip('.') in {'case-specific assessment', 'case-specific explanation', 'specific evidence-based observation'}):
            errors.append(f'{code} needs an independent defect/no_defect/unknown check and reason')
    evidence = record.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        errors.append('Generation assessment needs actual input and answer citations')
    elif case is not None:
        valid = [c for c in evidence if check_citation(c, case) is None]
        if len(valid) != len(evidence):
            errors.append('Generation assessment contains invalid citations; copy exact original spans')
        if not any(c.get('source') == 'original_rag_answer' and c.get('quote') for c in valid):
            errors.append('Generation assessment needs an exact original-answer quote')
        if not any(c.get('source') in {'generation_context', 'generation_history'} for c in valid):
            errors.append('Generation assessment needs actual generator-input evidence')
    return errors


def pipeline_diagnostics(case):
    """Emit threshold findings without turning them into causal answer labels."""
    coverage = gold_document_coverage(case)
    reranking = reranking_document_metrics(case)
    flags = {'E4': None if coverage['status'] == 'unknown' else coverage['recall'] < 1,
             'E7': reranking['e7_failure'], 'E8': reranking['e8_failure']}
    return {'policy': 'deterministic_document_thresholds_v1',
            'findings': {code: {'status': 'unknown' if failed is None else 'failure' if failed else 'pass',
                                 'origin': 'computed_document_membership', 'causal_attribution': False}
                         for code, failed in flags.items()},
            'failures': [code for code, failed in flags.items() if failed is True],
            'document_coverage': coverage, 'reranking': reranking}


def unaddressed_generation_defects(record, assessments):
    """A defect checklist is not itself an attributed failure label."""
    checks = record.get('checks', {}) if isinstance(record, dict) else {}
    checks = checks if isinstance(checks, dict) else {}
    return [f'{code} was marked defective without a supported candidate; attribution remains unresolved'
            for code in GENERATION_CODES if isinstance(checks.get(code), dict)
            and checks[code].get('status') == 'defect'
            and assessments.get(code, {}).get('status') != 'supported']


def generation_evidence_catalog(case):
    """Selectable original spans avoid making the model reproduce long quotes."""
    catalog = {}
    answer = case.get('original_rag_answer')
    if isinstance(answer, str) and answer.strip():
        catalog['answer'] = {'source': 'original_rag_answer', 'item_id': None,
                             'quote': answer, 'observation': 'Selected original answer record'}
    for field, prefix in [('generation_context', 'context'), ('generation_history', 'history')]:
        records = case.get(field)
        records = [records] if isinstance(records, str) else records if isinstance(records, list) else []
        for i, item in enumerate(records, 1):
            text = item if isinstance(item, str) else item.get('text', item.get('content')) if isinstance(item, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            item_id = item.get('chunk_id') if isinstance(item, dict) else None
            citation = {'source': field, 'item_id': str(item_id) if item_id is not None else None,
                        'quote': text, 'observation': 'Selected actual generator input record'}
            if check_citation(citation, case) is None:
                catalog[f'{prefix}_{i}'] = citation
    if case.get('generation_context') == []:
        catalog['context_empty'] = {'source': 'generation_context', 'item_id': None, 'quote': None,
            'observation': 'Observed empty saved generator context'}
    return catalog


def normalize_evidence_response(value, case):
    """Resolve ONLY explicit catalog selections and unambiguous status aliases.

    No missing candidates, references, claims or observations are inferred.
    Original quotes still pass the same evidence validator after resolution.
    """
    import copy
    result, paths = copy.deepcopy(value), []
    if not isinstance(result, dict):
        return result, paths
    catalog = generation_evidence_catalog(case)
    def resolve(record, location):
        if not isinstance(record, dict) or not isinstance(record.get('evidence'), list):
            return
        for i, evidence in enumerate(record['evidence']):
            ref = evidence if isinstance(evidence, str) else evidence.get('ref') if isinstance(evidence, dict) else None
            source_ref = evidence.get('source') if isinstance(evidence, dict) else None
            if isinstance(source_ref, str) and source_ref in catalog:
                selected = catalog[source_ref]
                # The model explicitly named a catalog record as its source.
                # Map the source/ID only; keep its supplied quote for exact validation.
                if evidence.get('item_id') in (None, selected['item_id']):
                    evidence['source'] = selected['source']
                    evidence['item_id'] = selected['item_id']
                    paths.append(f'{location}.evidence[{i}]: catalog source resolved')
            if isinstance(ref, str) and ref in catalog:
                citation = copy.deepcopy(catalog[ref])
                if isinstance(evidence, dict) and isinstance(evidence.get('observation'), str) and evidence['observation'].strip():
                    citation['observation'] = evidence['observation']
                record['evidence'][i] = citation
                paths.append(f'{location}.evidence[{i}]: resolved explicit reference {ref}')
    explicit_findings = []
    for field in ('generation_assessment', 'generation_assessment_update'):
        assessment = result.get(field)
        resolve(assessment, field)
        checks = assessment.get('checks') if isinstance(assessment, dict) else None
        for code, check in checks.items() if isinstance(checks, dict) else []:
            if not isinstance(check, dict):
                continue
            resolve(check, f'{field}.checks.{code}')
            status = check.get('status')
            alias = {'supported': 'defect', 'established': 'defect', 'rejected': 'no_defect',
                     'needs_review': 'unknown', 'insufficient': 'unknown', 'insufficient_evidence': 'unknown'}.get(status) if isinstance(status, str) else None
            if alias:
                check['status'] = alias
                paths.append(f'{field}.checks.{code}.status: {status} -> {alias}')
            # A labeled defect WITH ITS OWN evidence is already an explicit model
            # proposal. Adapt that record; never borrow checklist-level evidence.
            if code in GENERATION_CODES and check.get('status') == 'defect' and check.get('evidence'):
                explicit_findings.append((field, dict(check, code=code, status='supported',
                    key_points=[check.get('reason', '')], origin='explicit_generation_check')))
    for field, finding in explicit_findings:
        target = 'candidate_updates' if field.endswith('_update') else 'candidate_failures'
        records = result.setdefault(target, [])
        if isinstance(records, list) and not any(isinstance(r, dict) and r.get('code') == finding['code'] for r in records):
            records.append(finding)
            paths.append(f'{field}.checks.{finding["code"]}: adapted explicit labeled finding with its own evidence')
    for field in ('candidate_failures', 'candidate_updates'):
        for i, record in enumerate(result.get(field, []) if isinstance(result.get(field), list) else []):
            resolve(record, f'{field}[{i}]')
            if isinstance(record, dict) and record.get('status') == 'established':
                record['status'] = 'supported'
                paths.append(f'{field}[{i}].status: established -> supported')
    resolve(result.get('primary_update'), 'primary_update')
    # Initial primary citations use a separate key; resolve only when explicitly supplied.
    if isinstance(result.get('primary_evidence'), list):
        record = {'evidence': result['primary_evidence']}
        resolve(record, 'primary_evidence')
        result['primary_evidence'] = record['evidence']
    return result, paths
