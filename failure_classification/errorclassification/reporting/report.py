"""Separate model proposals, deterministic record checks, and final attribution.

Also supports saved runs without rewriting their historical prediction files.
"""
import copy
from failure_classification.errorclassification.taxonomy.labels import CODE_TO_NAME, CODE_TO_STAGE, STAGES, TAXONOMY_REVISION
from failure_classification.errorclassification.validation.validator import VALIDATION_POLICY, _list
from failure_classification.errorclassification.evidence.availability import evidence_availability_map
from failure_classification.errorclassification.evidence.assessment import generation_assessment_errors, GENERATION_ASSESSMENT_POLICY, unaddressed_generation_defects


def add_prediction_report(prediction, audit=None):
    """Return an enriched copy; never change accepted labels or rerun a model."""
    from failure_classification.errorclassification.validation.validator import normalize_prediction, normalize_code

    result = copy.deepcopy(prediction)
    audit = audit or []
    initial = result.get('initial_prediction') or (audit[0].get('output', {}) if audit else {})
    normalized = normalize_prediction(initial)
    candidates = {code: {'code': code,
                         'status': normalized['declared_assessments'].get(code, {}).get('status', 'supported'),
                         'key_points': normalized['key_points'].get(code, []),
                         'evidence': normalized['label_evidence'].get(code, [])}
                  for code in normalized['proposed']}
    primary = normalized['primary']
    reason = normalized['raw'].get('primary_reason', '')
    for entry in audit[1:]:
        patch = entry.get('output', {})
        if not isinstance(patch, dict):
            continue
        for candidate in patch.get('candidate_updates', []) or []:
            if not isinstance(candidate, dict):
                continue
            code = normalize_code(candidate.get('code'))
            if code and code in entry.get('applied_codes', []):
                candidates[code] = dict(copy.deepcopy(candidate), code=code)
        update = patch.get('primary_update')
        if entry.get('primary_review_requested') and isinstance(update, dict):
            primary = normalize_code(update.get('failure'))
            reason = update.get('reason', '')
    # Older exports may retain assessments but omit raw model output.
    if not candidates and not initial:
        for code, record in result.get('label_assessments', {}).items():
            candidates[code] = {'code': code, 'status': 'unknown', 'key_points': record.get('key_points', [])}
    result['model_prediction'] = {
        'primary_failure': primary, 'primary_reason': reason,
        'failures': [code for code, record in candidates.items() if record.get('status') != 'rejected'],
        'candidates': list(candidates.values()), 'semantically_verified': False,
    }
    result['validation'] = {
        'scope': 'Citation consistency and observable stage prerequisites; not semantic correctness.',
        'status': result.get('verification_status', 'needs_review'),
        'labels': {code: {'status': record['status'], 'reasons': record.get('reasons', []),
                          'missing_evidence': record.get('missing_evidence', []),
                          'origin': record.get('origin', 'model_proposal'),
                          'rejected_citations': record.get('rejected_citations', []),
                          'semantic_correctness_verified': False}
                   for code, record in result.get('label_assessments', {}).items()},
        'primary': copy.deepcopy(result.get('primary_assessment', {})),
    }
    no_failure = result.get('attribution_status') == 'no_failure'
    result['final_attribution'] = {
        'status': ('no_failure_reported' if no_failure else 'primary_passed_record_checks'
                   if result.get('primary_failure') else 'unresolved'),
        'primary_failure': result.get('primary_failure'), 'failures': result.get('failures', []),
        'semantic_correctness_verified': False,
    }
    return result


def _finalize(normalized, assessments, primary, *, method, passes, max_failures):
    raw = normalized['raw']
    supported = [code for code in CODE_TO_NAME if assessments.get(code, {}).get('status') == 'supported']
    unresolved = [code for code, record in assessments.items() if record['status'] in {'needs_review', 'insufficient_evidence'}]
    rejected = [code for code, record in assessments.items() if record['status'] == 'rejected']
    problems = bool(unresolved or rejected or normalized['invalid_labels'])
    if supported:
        status = 'established' if primary['failure'] and not problems else 'partially_established'
    elif assessments and all(record['status'] == 'insufficient_evidence' for record in assessments.values()) and not normalized['invalid_labels']:
        status = 'insufficient_evidence'
    elif (not assessments and not normalized['invalid_labels']
          and raw.get('attribution_status') == 'insufficient_evidence'
          and _list(raw.get('missing_evidence'))):
        status = 'insufficient_evidence'
    elif (not assessments and not normalized['invalid_labels'] and not normalized['warnings']
          and raw.get('attribution_status') == 'no_failure' and isinstance(raw.get('answer_defect'), dict)
          and raw['answer_defect'].get('status') == 'no_defect'):
        status = 'no_failure'
    else:
        status = 'needs_review'
    stages = {}
    raw_stages = raw.get('stages', {}) if isinstance(raw.get('stages'), dict) else {}
    for stage in STAGES:
        codes = [code for code in supported if CODE_TO_STAGE[code] == stage]
        open_codes = [code for code in unresolved + rejected if CODE_TO_STAGE[code] == stage]
        record = raw_stages.get(stage, {})
        record = record if isinstance(record, dict) else {}
        stage_status = 'established' if codes else ('needs_review' if open_codes else record.get('status', 'needs_review'))
        if not isinstance(stage_status, str) or stage_status not in {'established', 'no_failure_established', 'needs_review', 'insufficient_evidence'}:
            stage_status = 'needs_review'
        if not codes and stage_status == 'established':
            stage_status = 'needs_review'
        if open_codes and all(assessments[code]['status'] == 'insufficient_evidence' for code in open_codes) and not codes:
            stage_status = 'insufficient_evidence'
        stages[stage] = {'status': stage_status, 'findings': str(record.get('findings', 'No interpretable stage finding supplied.')),
                         'failures': codes, 'unresolved_or_rejected': open_codes,
                         'findings_source': 'model_proposal_not_independently_verified'}
    ordered = ([primary['failure']] if primary['failure'] else []) + [code for code in supported if code != primary['failure']]
    # Single-label evaluation uses the validated causal primary, never a fallback
    # to the first independently supported diagnostic finding.
    selected = ([primary['failure']] if primary['failure'] else []) if max_failures == 1 else ordered[:max_failures]
    return {'method': f'evidence_first_{method}_v5', 'validation_policy': VALIDATION_POLICY, 'taxonomy_revision': TAXONOMY_REVISION,
            'answer_defect': raw.get('answer_defect', {'status': 'uncertain', 'summary': 'No interpretable answer-defect assessment.'}),
            'stages': stages, 'supported_failures': supported, 'failures': selected,
            'retrieval_answerability': copy.deepcopy(raw.get('retrieval_answerability')),
            'selection_mode': 'validated_primary_only' if max_failures == 1 else 'supported_label_subset',
            'unresolved_failures': unresolved, 'rejected_failures': rejected,
            'label_assessments': assessments, 'invalid_labels': normalized['invalid_labels'],
            'primary_failure': primary['failure'], 'primary_stage': primary['stage'],
            'primary_assessment': primary, 'attribution_status': status,
            'key_points': {code: assessments[code]['key_points'] for code in supported},
            'label_evidence': {code: assessments[code]['evidence'] for code in supported},
            'unresolved_alternatives': unresolved,
            'missing_evidence': _list(raw.get('missing_evidence')),
            'limitations': _list(raw.get('limitations')), 'normalization_warnings': normalized['warnings'],
            'passes_used': passes, 'max_failures': max_failures,
            'verification_status': ('partially_passed_record_checks' if supported and problems else
                                    'passed_record_checks' if supported or status == 'no_failure' else 'needs_review'),
            'label_names': CODE_TO_NAME, 'scoring_method': 'independent_label_checks_no_vote_scores'}

def build_result(normalized, assessments, primary, *, case, reference_map, initial, plan, audit, method, max_failures):
    """Assemble the result without changing any accepted model judgments."""
    final = _finalize(normalized, assessments, primary, method=method, passes=len(audit), max_failures=max_failures)
    generation = copy.deepcopy(normalized['raw'].get('generation_assessment'))
    generation_errors = (generation_assessment_errors(generation, case)
                         + unaddressed_generation_defects(generation, assessments))
    first = normalized['raw'].get('answerability_assessment') or {}
    first_errors = first.get('validation_errors', [])
    if first.get('status') != 'completed':
        generation_errors = generation_errors + ['Answerability-first assessment is incomplete'] + first_errors
    final['e9_assessment'] = copy.deepcopy(normalized['raw'].get('e9_assessment'))
    final['automatically_assigned_failures'] = ['E9'] if final['e9_assessment'] and final['e9_assessment']['status'] == 'failure' else []
    final['generation_assessment'] = generation
    final['generation_review'] = {'policy': GENERATION_ASSESSMENT_POLICY,
        'status': 'needs_review' if generation_errors else 'completed',
        'errors': generation_errors, 'semantic_correctness_verified': False}
    if generation_errors:
        final['stages']['Generation']['status'] = 'needs_review'
        if final['attribution_status'] == 'no_failure':
            final['attribution_status'] = 'needs_review'
            final['verification_status'] = 'needs_review'
    final['initial_prediction'] = initial
    final['repair_plan'] = plan
    final['repair_attempted'] = len(audit) == 2
    final['format_retries'] = sum(entry.get('format_retries', 0) for entry in audit)
    final['review_output_valid'] = audit[-1].get('review_output_valid') if len(audit) == 2 else None
    defect = final['answer_defect'] if isinstance(final['answer_defect'], dict) else {}
    final['answer_assessment'] = {'status': 'reported_defect' if defect.get('status') == 'defect' else 'no_defect_reported' if defect.get('status') == 'no_defect' else 'uncertain',
                                  'summary': defect.get('summary', ''), 'independently_verified': False}
    final['failure_cause_status'] = 'primary_established' if final['primary_failure'] else 'supported_labels_primary_unresolved' if final['supported_failures'] else 'unresolved'
    final['reference_evidence_map'] = reference_map
    final['required_fact_map'] = evidence_availability_map(case, normalized['raw'].get('required_facts', []))
    final['layer_validation'] = {stage: {'evaluated': True,
        'candidate_codes': [code for code in assessments if CODE_TO_STAGE[code] == stage]} for stage in STAGES}
    final = add_prediction_report(final, audit)
    audit[-1]['final_accepted_prediction'] = copy.deepcopy(final)
    return final
