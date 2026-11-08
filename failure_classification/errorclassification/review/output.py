"""Check review response structure before applying evidence validation.

Only unambiguous JSON-null spelling is normalized; labels and evidence are never
inferred. A format retry must supply missing candidate records explicitly.
"""
import copy
from failure_classification.errorclassification.validation.validator import normalize_code
from failure_classification.errorclassification.evidence.assessment import generation_assessment_errors

STATUSES = {'supported', 'rejected', 'needs_review', 'insufficient_evidence'}
REVIEW_OUTPUT_POLICY = 'review_json_generation_contract_v2'


def normalize_review_output(value):
    result = copy.deepcopy(value)
    paths = []
    if isinstance(result, dict) and isinstance(result.get('primary_update'), dict):
        primary = result['primary_update']
        if isinstance(primary.get('failure'), str) and primary['failure'].strip().lower() == 'null':
            primary['failure'] = None
            paths.append('primary_update.failure: string null -> JSON null')
    return result, paths


def review_output_errors(value, supported_codes=()):
    if not isinstance(value, dict):
        return ['Review must be one JSON object.']
    errors, candidates = [], set()
    updates = value.get('candidate_updates', [])
    if not isinstance(updates, list):
        errors.append('candidate_updates must be a list of candidate objects.')
        updates = []
    legacy = value.get('label_updates', {})
    if not isinstance(legacy, dict):
        errors.append('label_updates must be an object if supplied.')
        legacy = {}
    updates = updates + [dict(record, code=code) for code, record in legacy.items() if isinstance(record, dict)]
    for i, record in enumerate(updates):
        if not isinstance(record, dict):
            errors.append(f'Candidate {i} must be an object.')
            continue
        code = normalize_code(record.get('code'))
        if code is None:
            errors.append(f'Candidate {i} must name an E1–E16 code.')
        status = record.get('status')
        if not isinstance(status, str) or status not in STATUSES:
            errors.append(f'Candidate {i} has an invalid status.')
        if 'evidence' in record and not isinstance(record['evidence'], list):
            errors.append(f'Candidate {i} evidence must be a list.')
        if code and status == 'supported':
            candidates.add(code)
    primary = value.get('primary_update')
    if primary is not None:
        if not isinstance(primary, dict):
            errors.append('primary_update must be an object or JSON null.')
        else:
            raw = primary.get('failure')
            code = normalize_code(raw)
            if raw is not None and code is None:
                errors.append('primary_update.failure must be E1–E16 or JSON null; no_defect is not a label.')
            status = primary.get('status')
            if not isinstance(status, str) or status not in {'established', 'needs_review'}:
                errors.append('primary_update.status must be established or needs_review.')
            if status == 'established':
                if code is None:
                    errors.append('An established primary requires a valid E-code; use needs_review or no primary update when unresolved.')
                elif code not in candidates and code not in supported_codes:
                    errors.append(f'Primary {code} has no supported candidate record. Supply candidate_updates with its own evidence or leave the primary unresolved.')
            if 'evidence' in primary and not isinstance(primary['evidence'], list):
                errors.append('primary_update.evidence must be a list.')
    defect = value.get('answer_defect_update')
    if defect is not None and (not isinstance(defect, dict)
            or not isinstance(defect.get('status'), str)
            or defect.get('status') not in {'defect', 'no_defect', 'uncertain'}
            or not isinstance(defect.get('summary'), str)):
        errors.append('answer_defect_update must have status defect|no_defect|uncertain and a summary, or be null.')
    generation = value.get('generation_assessment_update')
    if generation is not None:
        errors.extend(generation_assessment_errors(generation))
    return errors
