"""Validate model proposals against the original turn's evidence.

Record checks establish citation consistency and stage prerequisites, not semantic
correctness. All four stages are evaluated; E4 checks complete gold-document coverage.
"""
import copy
import re
from failure_classification.errorclassification.taxonomy.labels import CODE_TO_NAME, CODE_TO_STAGE, STAGES

VALIDATION_POLICY = 'answerability_first_computed_e9_v12'


class EvidenceValidator:
    """One validator instance owns the evidence for one evaluated turn."""

    def __init__(self, case, answerability_assessment=None):
        self.case = case
        self.answerability_assessment = copy.deepcopy(answerability_assessment)

    def validate(self, raw):
        normalized = normalize_prediction(raw)
        declared = normalized['declared_assessments']
        candidates = []
        for code in normalized['proposed'] + normalized['alternatives']:
            record = declared.get(code, {})
            candidates.append({'code': code, 'evidence': normalized['label_evidence'].get(code, []),
                               'key_points': normalized['key_points'].get(code, []),
                               'status': record.get('status', 'needs_review' if code in normalized['alternatives'] else 'supported'),
                               'reason': record.get('reason', 'Unresolved semantic alternative proposed by model' if code in normalized['alternatives'] else ''),
                               'mechanism': record.get('mechanism')})
        assessments = self.validate_all_layers(candidates)
        self.apply_retrieval_answerability(assessments, normalized['raw'])
        self.apply_generation_consistency(assessments, normalized['raw'])
        primary = self.validate_primary(normalized, assessments)
        return normalized, assessments, primary

    def validate_all_layers(self, candidates):
        results = {}
        for validate_stage in (self.validate_chunking, self.validate_retrieval,
                               self.validate_reranking, self.validate_generation):
            results.update(validate_stage(candidates))
        return results

    def validate_chunking(self, candidates):
        return self._validate_candidates(candidates, {'E1', 'E2', 'E3'})

    def validate_retrieval(self, candidates):
        return self._validate_candidates(candidates, {'E4', 'E5', 'E6'})

    def validate_reranking(self, candidates):
        return self._validate_candidates(candidates, {'E7', 'E8'})

    def validate_generation(self, candidates):
        return self._validate_candidates(candidates, {f'E{number}' for number in range(9, 17)})

    def _validate_candidates(self, candidates, codes):
        return {record['code']: self.validate_label(
            record['code'], record.get('evidence'), record.get('key_points'),
            declared_status=record.get('status'), reason=record.get('reason', ''),
            mechanism=record.get('mechanism'))
            for record in candidates if record['code'] in codes}

    def check_prerequisites(self, code, valid, mechanism=None):
        number = int(code[1:])
        if number <= 3:
            return self.check_chunking(code, valid, mechanism)
        if number <= 6:
            return self.check_retrieval(code, valid, mechanism)
        if number <= 8:
            return self.check_reranking(code, valid, mechanism)
        return self.check_generation(code, valid, mechanism)

    def check_chunking(self, code, valid, mechanism=None):
        case = self.case
        _, quoted, errors, missing, semantic_errors = prepare(case, valid, {'source_documents', 'indexed_chunks'})
        for source in ('source_documents', 'indexed_chunks'):
            if source not in quoted:
                errors.append(f'An exact supporting quote from {source} is required')
        return findings(errors, missing, semantic_errors)

    def check_retrieval(self, code, valid, mechanism=None):
        case = self.case
        if code == 'E4':
            sources, quoted, errors, missing, semantic_errors = prepare(
                case, valid, {'retrieved_candidates'})
            if not quoted & {'source_documents', 'indexed_chunks'}:
                errors.append('An exact reference-document or indexed quote is required')
            if errors or missing:
                return findings(errors, missing, semantic_errors)
            coverage = gold_document_coverage(case)
            if coverage['status'] == 'unknown':
                return findings([], ['Gold-document IDs or complete retrieval membership are unavailable'], [])
            if not coverage['missing_document_ids']:
                return findings(['All gold documents were retrieved'], [], [])
            return findings([], [], [])
        _, quoted, errors, missing, semantic_errors = prepare(case, valid, {'retrieved_candidates'})
        if 'retrieved_candidates' not in quoted:
            errors.append('An exact supporting quote from retrieved_candidates is required')
        return findings(errors, missing, semantic_errors)

    def check_reranking(self, code, valid, mechanism=None):
        metrics = reranking_document_metrics(self.case)
        if metrics['status'] == 'unknown':
            return findings([], ['Complete candidate/context document membership is required'], [])
        failed = metrics['e7_failure'] if code == 'E7' else metrics['e8_failure']
        if failed is None:
            return findings([], ['E7 requires retrieved gold documents; E8 requires five ordered reranked results'], [])
        if not failed:
            return findings([f'{code} document threshold is satisfied; no failure'], [], [])
        return findings([], [], [])

    def check_generation(self, code, valid, mechanism=None):
        case = self.case
        sources, quoted, errors, missing, semantic_errors = prepare(case, valid, {'original_rag_answer'})
        if case.get('generation_context') is None and case.get('generation_history') is None:
            missing.append('generation_context or generation_history')
        if not sources & {'generation_context', 'generation_history'}:
            errors.append('Actual generator evidence/history citation required')
        if code in {'E12', 'E13'} and not quoted & {'generation_context', 'generation_history'}:
            errors.append('E12/E13 requires an exact quote of actual supplied support')
        return findings(errors, missing, semantic_errors)

    def validate_label(self, code, citations, points, *, declared_status=None, reason='', mechanism=None):
        case = self.case
        valid, rejected = [], []
        for citation in _list(citations):
            error = check_citation(citation, case)
            if error:
                rejected.append({'citation': copy.deepcopy(citation), 'reason': error})
            else:
                valid.append(copy.deepcopy(citation))
        checks = self.check_prerequisites(code, valid, mechanism)
        errors, missing, semantic_errors = checks['errors'], checks['missing'], checks['semantic_errors']
        clean_points = [point for point in _list(points) if isinstance(point, str) and point.strip()][:3]
        if not clean_points and valid:
            # A factual citation observation is an existing evidence summary, not a
            # newly invented justification. This is a deterministic formatting repair.
            clean_points = list(dict.fromkeys(c['observation'] for c in valid))[:3]
        if missing:
            status = 'insufficient_evidence'
            errors.insert(0, f'Original evidence unavailable: {missing}')
        elif errors:
            status = 'rejected'
        elif declared_status == 'rejected':
            status = 'rejected'
            errors.append(reason or 'Model withdrew this finding')
        elif declared_status == 'insufficient_evidence':
            status = 'insufficient_evidence'
            errors.append(reason or 'Model reports a missing evidentiary prerequisite')
        elif declared_status == 'needs_review' or semantic_errors:
            status = 'needs_review'
            errors.extend(semantic_errors)
            if declared_status == 'needs_review':
                errors.append(reason or 'Model reports a disputed semantic interpretation')
        else:
            status = 'supported'
        return {'status': status, 'stage': CODE_TO_STAGE[code], 'name': CODE_TO_NAME[code],
                'evidence': valid, 'rejected_citations': rejected, 'key_points': clean_points,
                'reasons': errors, 'missing_evidence': missing,
                'mechanism': copy.deepcopy(mechanism),
                'semantic_correctness_verified': False}

    def validate_primary(self, normalized, assessments):
        case = self.case
        raw, code = normalized['raw'], normalized['primary']
        if (code is None or assessments.get(code, {}).get('status') != 'supported') and assessments.get('E9', {}).get('origin') == 'computed_answerability_rule' and assessments['E9']['status'] == 'supported':
            # E9 directly links the affected requested aspect to the observed
            # assertion. It is a valid generation cause when no other supported
            # causal primary was supplied, unlike a document-coverage metric.
            return {'failure': 'E9', 'stage': 'Generation', 'status': 'supported',
                    'reason': raw['e9_assessment']['reason'],
                    'evidence': copy.deepcopy(assessments['E9']['evidence']),
                    'semantic_recheck': False, 'origin': 'computed_answerability_rule',
                    'semantic_correctness_verified': False}
        if code is None:
            return {'failure': None, 'stage': None, 'status': 'needs_review',
                    'reason': 'No valid primary cause proposed', 'semantic_recheck': False}
        if assessments.get(code, {}).get('status') != 'supported':
            return {'failure': None, 'stage': None, 'status': 'rejected',
                    'reason': 'Proposed primary is not independently supported', 'semantic_recheck': False}
        defect = raw.get('answer_defect', {})
        reason = raw.get('primary_reason')
        evidence = raw.get('primary_evidence')
        if not isinstance(defect, dict) or defect.get('status') != 'defect':
            return {'failure': None, 'stage': None, 'status': 'needs_review',
                    'reason': 'An observed original-answer defect is not established', 'semantic_recheck': False}
        if not isinstance(reason, str) or not reason.strip():
            return {'failure': None, 'stage': None, 'status': 'needs_review',
                    'reason': 'Primary causal explanation is absent; a supported label alone does not establish primacy', 'semantic_recheck': False}
        valid = [c for c in _list(evidence) if check_citation(c, case) is None]
        if not any(c['source'] == 'original_rag_answer' and c['quote'] for c in valid):
            return {'failure': None, 'stage': None, 'status': 'needs_review',
                    'reason': 'Primary attribution lacks a valid quote of the affected original answer', 'semantic_recheck': False}
        if isinstance(raw.get('attribution_status'), str) and raw['attribution_status'] in {'needs_review', 'insufficient_evidence'}:
            return {'failure': None, 'stage': None, 'status': 'needs_review',
                    'reason': reason, 'semantic_recheck': raw.get('primary_needs_review') is True and raw.get('attribution_status') == 'needs_review'}
        if raw.get('attribution_status') not in ('established', 'partially_established'):
            return {'failure': None, 'stage': None, 'status': 'needs_review',
                    'reason': 'Model attribution status conflicts with the proposed primary', 'semantic_recheck': False}
        return {'failure': code, 'stage': CODE_TO_STAGE[code], 'status': 'supported',
                'reason': reason, 'evidence': valid, 'semantic_recheck': False,
                'semantic_correctness_verified': False}

    def apply_generation_consistency(self, assessments, raw):
        """E9 is derived from the frozen first assessment, never voted on again."""
        from failure_classification.errorclassification.answerability.assessor import derive_e9, assessment_errors, unknown_assessment
        if self.answerability_assessment and self.answerability_assessment.get('status') == 'completed':
            errors = assessment_errors(self.answerability_assessment, self.case)
            if errors:
                self.answerability_assessment = unknown_assessment('Invalid answerability-first evidence', errors)
        decision = derive_e9(self.answerability_assessment)
        raw['e9_assessment'] = copy.deepcopy(decision)
        raw['answerability_assessment'] = copy.deepcopy(self.answerability_assessment)
        old = assessments.pop('E9', None)
        if decision['status'] == 'failure':
            if not isinstance(raw.get('answer_defect'), dict) or raw['answer_defect'].get('status') != 'defect':
                raw['model_answer_defect'] = copy.deepcopy(raw.get('answer_defect'))
                raw['answer_defect'] = {'status': 'defect', 'summary': decision['reason'],
                                        'origin': 'computed_answerability_rule'}
            evidence = []
            for requirement in decision['affected_requirements']:
                for citation in requirement['evidence']:
                    if citation not in evidence:
                        evidence.append(copy.deepcopy(citation))
            assessments['E9'] = {'status': 'supported', 'stage': 'Generation', 'name': CODE_TO_NAME['E9'],
                'evidence': evidence, 'rejected_citations': [], 'key_points': [decision['reason']],
                'reasons': [], 'missing_evidence': [], 'mechanism': None,
                'origin': 'computed_answerability_rule', 'semantic_correctness_verified': False}
        elif old is not None:
            old['status'] = 'needs_review' if decision['status'] == 'unknown' else 'rejected'
            old['origin'] = 'computed_answerability_rule'
            old['reasons'] = [decision['reason']]
            assessments['E9'] = old
        # Later generation checks may discuss E10–E16, but cannot revise the
        # already-established answerability or make a contradictory E9 vote.
        generation = raw.get('generation_assessment')
        if isinstance(generation, dict):
            first = self.answerability_assessment or {}
            generation['answerability'] = first.get('answerability', 'unknown')
            generation['response_behavior'] = first.get('response_behavior', 'unknown')
            checks = generation.get('checks')
            if isinstance(checks, dict):
                checks['E9'] = {'status': {'failure': 'defect', 'no_failure': 'no_defect', 'unknown': 'unknown'}[decision['status']],
                    'reason': decision['reason'], 'origin': 'computed_answerability_rule'}

    def apply_retrieval_answerability(self, assessments, raw):
        """Keep answerability independent of the gold-document coverage rule."""
        for code in ('E7', 'E8'):
            if code in assessments:
                assessments[code]['document_metrics'] = reranking_document_metrics(self.case)
        if 'E4' in assessments:
            assessments['E4']['document_coverage'] = gold_document_coverage(self.case)


def gold_document_coverage(case):
    """Document membership, counting any chunk from a gold document as coverage.

    Prefer explicit doc_id; legacy dataset chunk IDs encode <doc_id>_<index>.
    Missing or explicitly partial retrieval traces cannot establish absence.
    """
    docs = case.get('gold_source_documents', case.get('source_documents'))
    candidates = case.get('retrieved_candidates')
    gold = {str(d.get('doc_id', d.get('id'))) for d in docs or []
            if isinstance(d, dict) and d.get('doc_id', d.get('id')) is not None}
    trace = str(case.get('trace_status', {}).get('retrieved_candidates', '')).lower()
    observed = set()
    unknown_membership = False
    for item in candidates or []:
        if not isinstance(item, dict):
            unknown_membership = True
            continue
        doc_id = item.get('doc_id')
        if doc_id is None:
            chunk_id = str(item.get('chunk_id', ''))
            match = re.fullmatch(r'(.+)_[0-9]+', chunk_id)
            doc_id = match.group(1) if match else None
        if doc_id is None:
            unknown_membership = True
        else:
            observed.add(str(doc_id))
    missing = gold - observed
    unknown = (not gold or not isinstance(candidates, list) or
               (bool(missing) and (unknown_membership or trace.startswith(('partial', 'missing', 'unknown')))))
    return {'status': 'unknown' if unknown else 'complete' if not missing else 'incomplete',
            'gold_document_ids': sorted(gold), 'retrieved_gold_document_ids': sorted(gold & observed),
            'missing_document_ids': sorted(missing),
            'recall': None if unknown else len(gold & observed) / len(gold)}


def reranking_document_metrics(case):
    """E7 loses >=50% of retrieved gold docs; E8 needs >=2 in top three.

    generation_context is the saved ordered five-result reranking output.
    Distinct documents count once, even if represented by several chunks.
    """
    retrieved = gold_document_coverage(case)
    context = case.get('generation_context')
    selected_case = dict(case, retrieved_candidates=context)
    selected_case['trace_status'] = dict(case.get('trace_status', {}),
        retrieved_candidates=case.get('trace_status', {}).get('generation_context', ''))
    selected = gold_document_coverage(selected_case)
    unknown = retrieved['status'] == 'unknown' or selected['status'] == 'unknown'
    order_unknown = 'unordered' in str(case.get('trace_status', {}).get('generation_context', '')).lower()
    gold_retrieved = set(retrieved['retrieved_gold_document_ids'])
    retained = gold_retrieved & set(selected['retrieved_gold_document_ids'])
    lost = gold_retrieved - retained
    top = gold_document_coverage(dict(selected_case, retrieved_candidates=context[:3] if isinstance(context, list) else None))
    top_ids = gold_retrieved & set(top['retrieved_gold_document_ids'])
    fraction = len(lost) / len(gold_retrieved) if gold_retrieved else None
    return {'status': 'unknown' if unknown else 'observed',
            'retrieved_gold_document_ids': sorted(gold_retrieved),
            'retained_gold_document_ids': sorted(retained),
            'lost_gold_document_ids': sorted(lost), 'lost_fraction': fraction,
            'top_k': 3, 'required_top_k_gold_documents': 2,
            'top_k_gold_document_ids': sorted(top_ids),
            'e7_failure': None if unknown or fraction is None else fraction >= .5,
            'e8_failure': None if unknown or order_unknown or not isinstance(context, list) or len(context) != 5 or top['status'] == 'unknown' else len(top_ids) < 2}


# Response normalization and citation lookup.

def normalize_code(value):
    """Accept codes and exact legacy names; never infer an unknown/mismatched label."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r'(E\d{1,3})(?:\s*:?\s*(.*))?', value.strip(), re.I)
    if not match:
        return None
    code, suffix = match.group(1).upper(), (match.group(2) or '').strip()
    if code not in CODE_TO_NAME:
        return None
    if suffix and suffix.casefold() != CODE_TO_NAME[code].casefold():
        return None
    return code


def _list(value):
    return value if isinstance(value, list) else []


def _merge_map(raw, warnings, name):
    result = {}
    if not isinstance(raw, dict):
        warnings.append(f'{name} is not an object')
        return result
    for label, items in raw.items():
        code = normalize_code(label)
        if code is None:
            warnings.append(f'Ignored invalid {name} key: {label!r}')
            continue
        if not isinstance(items, list):
            warnings.append(f'{code}: {name} is not a list')
            continue
        target = result.setdefault(code, [])
        for item in items:
            if item not in target:
                target.append(copy.deepcopy(item))
    return result


def normalize_prediction(raw):
    warnings, invalid = [], []
    if not isinstance(raw, dict):
        return {'proposed': [], 'key_points': {}, 'label_evidence': {},
                'alternatives': [], 'declared_assessments': {}, 'primary': None,
                'raw': {}, 'invalid_labels': [], 'warnings': ['Output is not an object']}
    proposed = []
    entries = raw.get('supported_failures', [])
    if not isinstance(entries, list):
        warnings.append('supported_failures is not a list')
        entries = []
    for entry in entries:
        code = normalize_code(entry)
        if code is None:
            invalid.append({'value': copy.deepcopy(entry), 'status': 'invalid_label', 'reason': 'Unknown or mismatched E-code/name'})
        elif code not in proposed:
            proposed.append(code)
    points = _merge_map(raw.get('key_points', {}), warnings, 'key_points')
    evidence = _merge_map(raw.get('label_evidence', {}), warnings, 'label_evidence')
    assessments = {}
    for label, record in (raw.get('label_assessments', {}).items() if isinstance(raw.get('label_assessments'), dict) else []):
        code = normalize_code(label)
        if code and isinstance(record, dict):
            assessments[code] = copy.deepcopy(record)
            if code not in proposed:
                proposed.append(code)
            points.setdefault(code, _list(record.get('key_points')))
            evidence.setdefault(code, _list(record.get('evidence')))
    candidates = raw.get('candidate_failures', [])
    if candidates and not isinstance(candidates, list):
        warnings.append('candidate_failures is not a list')
    for candidate in _list(candidates):
        if not isinstance(candidate, dict):
            invalid.append({'value': copy.deepcopy(candidate), 'status': 'invalid_label', 'reason': 'Candidate is not an object'})
            continue
        code = normalize_code(candidate.get('code'))
        if code is None:
            invalid.append({'value': candidate.get('code'), 'status': 'invalid_label', 'reason': 'Unknown or mismatched candidate code'})
            continue
        if code not in proposed:
            proposed.append(code)
        assessments[code] = copy.deepcopy(candidate)
        assessments[code].setdefault('status', 'supported')
        points.setdefault(code, [])
        evidence.setdefault(code, [])
        for field, target in [('key_points', points[code]), ('evidence', evidence[code])]:
            for item in _list(candidate.get(field)):
                if item not in target:
                    target.append(copy.deepcopy(item))
    alternatives = []
    for entry in _list(raw.get('unresolved_alternatives')):
        code = normalize_code(entry)
        if code and code not in proposed and code not in alternatives:
            alternatives.append(code)
        elif not code:
            # Invalid optional alternatives are diagnostic warnings, not grounds
            # for deleting supported findings or making another model call.
            warnings.append(f'Ignored invalid unresolved alternative: {entry!r}')
    primary_raw = raw.get('primary_failure')
    primary = normalize_code(primary_raw)
    if primary_raw is not None and primary is None:
        warnings.append(f'Invalid primary label: {primary_raw!r}')
    return {'proposed': proposed, 'key_points': points, 'label_evidence': evidence,
            'alternatives': alternatives, 'declared_assessments': assessments,
            'primary': primary, 'raw': copy.deepcopy(raw),
            'invalid_labels': invalid, 'warnings': warnings}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def check_citation(citation, case):
    if not isinstance(citation, dict):
        return 'Citation is not an object'
    required = {'source', 'item_id', 'quote', 'observation'}
    if not required <= set(citation):
        return 'Citation lacks source, item_id, quote, or observation'
    source, item_id, quote = citation['source'], citation['item_id'], citation['quote']
    if not isinstance(source, str) or source not in case or case[source] is None:
        return f'Unknown or missing evidence source: {source!r}'
    if not isinstance(citation['observation'], str) or not citation['observation'].strip():
        return 'Citation observation is empty'
    target = case[source]
    if item_id is not None:
        if not isinstance(item_id, str) or not isinstance(target, list):
            return 'Invalid evidence item ID'
        keys = (('chunk_id',) if source in {'indexed_chunks', 'retrieved_candidates', 'generation_context'}
                else ('doc_id', 'id') if source == 'source_documents' else ('probe_id', 'id', 'chunk_id'))
        matches = [item for item in target if isinstance(item, dict)
                   and any(key in item and str(item[key]) == item_id for key in keys)]
        if not matches:
            return f'Item {item_id!r} is absent from {source}'
        target = matches
    if quote is not None:
        if not isinstance(quote, str) or not quote.strip() or not any(quote in text for text in _strings(target)):
            return f'Quote is not an exact substring in {source}/{item_id}'
    return None


def prepare(case, valid, required):
    sources = {citation['source'] for citation in valid}
    quoted = {citation['source'] for citation in valid if citation['quote'] is not None}
    missing = [field for field in sorted(required) if case.get(field) is None]
    errors = []
    if not required <= sources:
        errors.append(f'Required stage evidence sources missing: {sorted(required - sources)}')
    return sources, quoted, errors, missing, []


def findings(errors, missing, semantic_errors):
    return {'errors': errors, 'missing': missing, 'semantic_errors': semantic_errors}


def validate_prediction(raw, case):
    """Convenience entry point for checking a saved model response."""
    return EvidenceValidator(case).validate(raw)


def verify_evidence(value, case):
    """Return substantive label errors without globally short-circuiting on shape."""
    normalized, assessments, _ = validate_prediction(value, case)
    errors = [f'{code}: {reason}' for code, assessment in assessments.items() for reason in assessment['reasons']]
    errors.extend(f'{code}: {item["reason"]}' for code, assessment in assessments.items() for item in assessment['rejected_citations'])
    errors.extend(f'Invalid label: {item["value"]!r}' for item in normalized['invalid_labels'])
    return errors
