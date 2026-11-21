"""Plan one bounded repair using available original evidence, never missing logs."""
import copy
from failure_classification.errorclassification.taxonomy.labels import FAILURE_CATEGORIES
from failure_classification.errorclassification.evidence.assessment import generation_assessment_errors, unaddressed_generation_defects
from failure_classification.errorclassification.validation.validator import normalize_code, _list

CODE_STAGE = {category.code: category.stage.value for category in FAILURE_CATEGORIES}


def available_layers(case):
    if not isinstance(case.get('original_question'), str) or not isinstance(case.get('original_rag_answer'), str):
        return []
    layers = []
    if case.get('source_documents') is not None and case.get('indexed_chunks') is not None:
        layers.append('Chunking')
    if case.get('retrieved_candidates') is not None:
        layers.append('Retrieval')
    if case.get('retrieved_candidates') is not None and case.get('generation_context') is not None:
        layers.append('Reranking')
    if case.get('generation_context') is not None or case.get('generation_history') is not None:
        layers.append('Generation')
    return layers


def plan_recheck(case, assessments, primary, *, answer_defect=None, generation_assessment=None):
    layers = available_layers(case)
    frozen = [code for code, record in assessments.items() if record['status'] == 'supported']
    missing = [code for code, record in assessments.items() if record['status'] == 'insufficient_evidence']
    semantic = [code for code, record in assessments.items() if record['status'] == 'needs_review']
    repair = [code for code, record in assessments.items()
              if record['status'] == 'rejected' and not record.get('missing_evidence')
              and CODE_STAGE[code] in layers]
    missing_attribution = (
        isinstance(answer_defect, dict) and answer_defect.get('status') == 'defect'
        and primary.get('failure') is None and bool(layers)
    )
    generation_gaps = (generation_assessment_errors(generation_assessment, case)
                       + unaddressed_generation_defects(generation_assessment, assessments))
    generation_review = 'Generation' in layers and bool(generation_gaps)
    if repair or missing_attribution:
        # Repair may identify another mechanism/layer. For example, an E4 claim
        # can fail because evidence WAS retrieved; generation may be responsible.
        allowed = [code for code, stage in CODE_STAGE.items() if stage in layers
                   and code not in frozen and code not in missing]
        # Document thresholds are computed separately; semantic review uses available evidence.
    else:
        allowed = [code for code in semantic if CODE_STAGE[code] in layers]
    if generation_review:
        allowed = list(dict.fromkeys(allowed + [f'E{i}' for i in range(9, 17)
            if f'E{i}' not in frozen and f'E{i}' not in missing]))
    allowed = [code for code in allowed if code != 'E9']
    primary_review = (bool(primary.get('semantic_recheck'))
                      or bool(repair and primary['failure'] is None) or missing_attribution or generation_review)
    return {'kind': ('evidence_repair' if repair else
                     'missing_attribution' if missing_attribution else 'generation_review' if generation_review else 'semantic_recheck'),
            'rejected_codes_to_reconsider': repair, 'semantic_disputes': semantic,
            'allowed_codes': allowed, 'available_layers': layers,
            'frozen_supported_codes': frozen, 'held_missing_evidence_codes': missing,
            'primary_review_requested': primary_review,
            'missing_attribution': missing_attribution,
            'generation_review_requested': generation_review, 'generation_assessment_gaps': generation_gaps,
            'eligible': bool(allowed or primary_review)}


def apply_review(patch, normalized, assessments, primary, plan, validator):
    """Apply only permitted updates; validate them against the same turn evidence."""
    ignored, applied = [], []
    updates = _list(patch.get('candidate_updates')) if isinstance(patch, dict) else []
    # Retain compatibility with the earlier targeted-patch map contract.
    legacy_updates = patch.get('label_updates', {}) if isinstance(patch, dict) else {}
    if isinstance(legacy_updates, dict):
        updates = updates + [dict(record, code=code) for code, record in legacy_updates.items() if isinstance(record, dict)]
    seen = set()
    for record in updates:
        code = normalize_code(record.get('code')) if isinstance(record, dict) else None
        if code not in plan['allowed_codes'] or code in seen:
            ignored.append(f'Ignored unrequested/invalid/duplicate label update: {code!r}')
            continue
        seen.add(code)
        declared = record.get('status')
        if not isinstance(declared, str) or declared not in {'supported', 'rejected', 'needs_review', 'insufficient_evidence'}:
            ignored.append(f'Ignored invalid status for {code}')
            continue
        assessments[code] = validator.validate_label(code, record.get('evidence'), record.get('key_points'),
                                           declared_status=declared, reason=record.get('reason', ''), mechanism=record.get('mechanism'))
        applied.append(code)
    generation = patch.get('generation_assessment_update') if isinstance(patch, dict) else None
    if isinstance(generation, dict) and 'Generation' in plan['available_layers']:
        normalized['raw']['generation_assessment'] = copy.deepcopy(generation)
    defect = patch.get('answer_defect_update') if isinstance(patch, dict) else None
    if isinstance(defect, dict) and defect.get('status') in ('defect', 'no_defect', 'uncertain') and isinstance(defect.get('summary'), str):
        if defect['status'] == 'no_defect' and any(a['status'] == 'supported' for code, a in assessments.items() if code not in {'E4', 'E7', 'E8'}):
            ignored.append('Ignored no-defect update conflicting with retained supported findings')
        else:
            normalized['raw']['answer_defect'] = copy.deepcopy(defect)
    stage_findings = patch.get('stage_findings', {}) if isinstance(patch, dict) else {}
    if isinstance(stage_findings, dict):
        stages = normalized['raw'].get('stages')
        if not isinstance(stages, dict):
            stages = normalized['raw']['stages'] = {}
        for stage, finding in stage_findings.items():
            if stage in plan['available_layers'] and isinstance(finding, str):
                record = stages.get(stage) if isinstance(stages.get(stage), dict) else {}
                record['findings'] = finding
                stages[stage] = record
    answerability = patch.get('retrieval_answerability_update') if isinstance(patch, dict) else None
    if isinstance(answerability, dict) and 'E4' in plan['allowed_codes']:
        normalized['raw']['retrieval_answerability'] = copy.deepcopy(answerability)
    validator.apply_retrieval_answerability(assessments, normalized['raw'])
    validator.apply_generation_consistency(assessments, normalized['raw'])
    update = patch.get('primary_update') if isinstance(patch, dict) else None
    if plan['primary_review_requested'] and isinstance(update, dict):
        proposal = copy.deepcopy(normalized)
        proposal['primary'] = normalize_code(update.get('failure'))
        proposal['raw'].update(primary_failure=update.get('failure'), primary_reason=update.get('reason'),
                                primary_evidence=update.get('evidence'), attribution_status=update.get('status'))
        primary = validator.validate_primary(proposal, assessments)
    elif update:
        ignored.append('Ignored unrequested primary update')
    return normalized, assessments, primary, applied, ignored
