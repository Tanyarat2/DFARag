"""Independent evidence checks, normalization, attribution and bounded review."""
import copy
import unittest
from failure_classification.tests.answerability_fixtures import AttributionOnlyTestCase
from types import SimpleNamespace
from failure_classification.errorclassification.pipeline.classifier import classify_case
from failure_classification.errorclassification.validation.validator import (
    STAGES, normalize_code, normalize_prediction, verify_evidence, validate_prediction, VALIDATION_POLICY,
)

CASE = {'case_id': 'test', 'original_question': 'Who was appointed in November 2018?',
        'original_rag_answer': 'Emily Anderson',
        'indexed_chunks': [{'chunk_id': '47_11', 'text': 'Michael Evans was appointed in November 2018.'}],
        'retrieved_candidates': [{'chunk_id': '63_10', 'text': 'Emily Anderson was appointed in August 2021.'}],
        'generation_context': [{'chunk_id': '63_10', 'text': 'Emily Anderson was appointed in August 2021.'}],
        'source_documents': [{'doc_id': '47', 'text': 'Michael Evans was appointed in November 2018.'}]}


def citation(source, item_id, quote, observation='Observed evidence.'):
    return {'source': source, 'item_id': item_id, 'quote': quote, 'observation': observation}


def empty_result():
    return {'answer_defect': {'status': 'no_defect', 'summary': 'The supported answer addresses the request.'},
            'stages': {stage: {'status': 'no_failure_established', 'findings': 'No attributable defect established.'} for stage in STAGES},
            'supported_failures': [], 'primary_failure': None, 'primary_stage': None,
            'primary_reason': '', 'primary_evidence': [], 'attribution_status': 'no_failure',
            'key_points': {}, 'label_evidence': {}, 'unresolved_alternatives': [],
            'missing_evidence': [], 'limitations': []}


def retrieval_result():
    value = empty_result()
    label = 'E4 Missed Retrieval'  # Historical full-name format remains accepted.
    value.update(answer_defect={'status': 'defect', 'summary': 'The answer names the wrong person.'},
                 supported_failures=[label], primary_failure=label, primary_stage='Wrong model-supplied stage',
                 primary_reason='Usable indexed November support was missing from the supplied candidates, explaining the wrong answer.',
                 primary_evidence=[citation('original_rag_answer', None, 'Emily Anderson')],
                 attribution_status='established', key_points={label: ['Usable indexed support was not retrieved.']},
                 label_evidence={label: [citation('indexed_chunks', '47_11', 'Michael Evans'),
                                         citation('retrieved_candidates', None, None, 'Saved candidates omit 47_11.')]})
    value['retrieval_answerability'] = {'status': 'insufficient',
        'reason': 'The retrieved Emily/August 2021 passage does not answer the November 2018 appointment question.',
        'missing_requirements': ['Who was appointed in November 2018'],
        'evidence': [citation('retrieved_candidates', '63_10', 'Emily Anderson was appointed in August 2021.')]}
    value['stages']['Retrieval'].update(status='established')
    return value


def add_bad_e13(value):
    value['supported_failures'].append('E13: Misinterpretation')
    value['key_points']['E13'] = ['Misread Michael Evans.']
    value['label_evidence']['E13'] = [citation('original_rag_answer', None, 'Emily Anderson'),
                                     citation('generation_context', '63_10', 'Michael Evans')]
    return value


def run(raw, case=CASE, **kwargs):
    calls = []
    def ask(**request):
        calls.append(request)
        return {'parsed': copy.deepcopy(raw)}
    kwargs.setdefault('max_passes', 1)  # Isolate deterministic validation in these unit checks.
    final, audit = classify_case(case, SimpleNamespace(ask_json=ask), **kwargs)
    return final, audit, calls


class EvidenceFirstTests(AttributionOnlyTestCase):
    def test_reported_defect_without_candidates_gets_one_validated_attribution_review(self):
        case = copy.deepcopy(CASE)
        case['generation_context'] = copy.deepcopy(case['indexed_chunks'])
        raw = empty_result()
        raw.update(answer_defect={'status': 'defect', 'summary': 'The answer names the wrong person.'},
                   candidate_failures=[], attribution_status='established')
        answer_quote = citation('original_rag_answer', None, 'Emily Anderson')
        patch = {'candidate_updates': [{'code': 'E13', 'status': 'supported',
                  'key_points': ['The answer contradicts the supplied appointment evidence.'],
                  'evidence': [answer_quote, citation('generation_context', '47_11',
                              'Michael Evans was appointed in November 2018.')]}],
                 'primary_update': {'failure': 'E13', 'status': 'established',
                                    'reason': 'The generator changed the person in its supplied evidence.',
                                    'evidence': [answer_quote]}}
        outputs = iter([raw, patch])
        calls = []
        def ask(**request):
            calls.append(request)
            return {'parsed': next(outputs)}
        final, audit = classify_case(case, SimpleNamespace(ask_json=ask), max_failures=1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(final['primary_failure'], 'E13')
        self.assertEqual(final['failures'], ['E13'])
        self.assertEqual(final['initial_prediction'], raw)
        self.assertTrue(audit[0]['recheck_plan']['missing_attribution'])
        self.assertEqual(audit[1]['recheck_kind'], 'missing_attribution')
        self.assertIn('without establishing a primary cause', calls[1]['prompt'])

    def test_missing_attribution_does_not_force_a_label_or_bypass_quote_checks(self):
        raw = empty_result()
        raw['answer_defect'] = {'status': 'defect', 'summary': 'Wrong person.'}
        for patch in ({}, {'candidate_updates': [{'code': 'E13', 'status': 'supported',
                      'evidence': [citation('generation_context', '63_10', 'INVENTED QUOTE')]}],
                      'primary_update': {'failure': 'E13', 'status': 'established',
                                         'reason': 'Misreading.', 'evidence': []}}):
            with self.subTest(patch=patch):
                outputs = iter([raw, patch])
                final, audit = classify_case(CASE, SimpleNamespace(ask_json=lambda **kwargs: {'parsed': next(outputs)}))
                self.assertEqual(len(audit), 2)
                self.assertEqual(final['failures'], [])
                self.assertIsNone(final['primary_failure'])

    def test_missing_attribution_respects_missing_layers_and_held_findings(self):
        from failure_classification.errorclassification.review.workflow import plan_recheck
        case = {'original_question': 'Who?', 'original_rag_answer': 'Emily',
                'generation_context': [{'chunk_id': 'c', 'text': 'Michael'}]}
        assessments = {'E13': {'status': 'insufficient_evidence'}, 'E10': {'status': 'supported'}}
        plan = plan_recheck(case, assessments, {'failure': None}, answer_defect={'status': 'defect'})
        self.assertTrue(plan['eligible'])
        self.assertTrue(all(code in {f'E{n}' for n in range(9, 17)} for code in plan['allowed_codes']))
        self.assertNotIn('E13', plan['allowed_codes'])
        self.assertNotIn('E10', plan['allowed_codes'])
        for defect in ({'status': 'no_defect'}, {'status': 'uncertain'}, None):
            self.assertTrue(plan_recheck(case, {}, {'failure': None}, answer_defect=defect)['generation_review_requested'])
        self.assertFalse(plan_recheck({'original_question': 'Who?', 'original_rag_answer': 'Emily'},
                                     {}, {'failure': None}, answer_defect={'status': 'defect'})['eligible'])

    def test_missing_attribution_respects_single_pass_budget(self):
        raw = empty_result()
        raw['answer_defect'] = {'status': 'defect', 'summary': 'Wrong person.'}
        final, _, calls = run(raw, max_passes=1)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(final['primary_failure'])

    def test_rejected_prediction_remains_visible_and_unresolved(self):
        raw = retrieval_result()
        raw['label_evidence']['E4 Missed Retrieval'] = [
            citation('indexed_chunks', '47_11', 'Michael Evans')]
        final, _, _ = run(raw, max_failures=1)
        self.assertEqual(final['failures'], [])
        self.assertEqual(final['model_prediction']['primary_failure'], 'E4')
        self.assertEqual(final['validation']['labels']['E4']['status'], 'rejected')
        self.assertTrue(final['validation']['labels']['E4']['reasons'])
        self.assertEqual(final['final_attribution']['status'], 'unresolved')

    def test_record_checks_do_not_claim_semantic_verification(self):
        final, _, _ = run(retrieval_result(), max_failures=1)
        self.assertEqual(final['final_attribution']['failures'], ['E4'])
        self.assertFalse(final['final_attribution']['semantic_correctness_verified'])
        self.assertEqual(final['final_attribution']['status'], 'primary_passed_record_checks')

    def test_no_failure_without_generation_assessment_is_unresolved(self):
        final, _, _ = run(empty_result())
        self.assertEqual(final['final_attribution']['status'], 'unresolved')
        self.assertEqual(final['generation_review']['status'], 'needs_review')
        self.assertEqual(final['model_prediction']['failures'], [])

    def test_saved_report_applies_only_authorized_model_updates(self):
        from failure_classification.errorclassification.reporting.report import add_prediction_report
        final, _, _ = run(retrieval_result())
        audit = [{'output': retrieval_result()},
                 {'output': {'candidate_updates': [{'code': 'E12', 'status': 'supported'},
                                                   {'code': 'E10', 'status': 'supported'}],
                             'primary_update': {'failure': 'E12', 'reason': 'Updated model attribution'}},
                  'applied_codes': ['E12'], 'primary_review_requested': True}]
        report = add_prediction_report(final, audit)
        self.assertEqual(report['model_prediction']['primary_failure'], 'E12')
        self.assertIn('E12', report['model_prediction']['failures'])
        self.assertNotIn('E10', report['model_prediction']['failures'])
        self.assertEqual(report['failures'], final['failures'])

    def test_equivalent_other_document_does_not_cancel_e4(self):
        case = copy.deepcopy(CASE)
        case['retrieved_candidates'].append({'chunk_id': '99_0',
            'text': 'Michael Evans was appointed in November 2018.'})
        case['generation_context'].append(case['retrieved_candidates'][-1])
        raw = retrieval_result()  # The cited original indexed chunk is still absent.
        raw['retrieval_answerability'] = {'status': 'sufficient',
            'reason': 'An alternative retrieved chunk answers the appointment question.',
            'missing_requirements': [],
            'evidence': [citation('retrieved_candidates', '99_0',
                                 'Michael Evans was appointed in November 2018.')]}
        final, _, _ = run(raw, case, max_failures=1)
        self.assertEqual(final['label_assessments']['E4']['status'], 'supported')
        self.assertEqual(final['failures'], ['E4'])

    def test_e4_does_not_require_answerability_assessment(self):
        raw = retrieval_result()
        del raw['retrieval_answerability']
        final, _, _ = run(raw, max_failures=1)
        self.assertEqual(final['failures'], ['E4'])

    def test_e4_does_not_require_missing_answer_aspect(self):
        raw = retrieval_result()
        raw['retrieval_answerability']['missing_requirements'] = []
        final, _, _ = run(raw, max_failures=1)
        self.assertEqual(final['failures'], ['E4'])

    def test_legacy_codes_and_invalid_codes(self):
        for value in ['E4', 'E4:', 'E4 Missed Retrieval', 'E4: Missed Retrieval', 'e4 missed retrieval']:
            self.assertEqual(normalize_code(value), 'E4')
        for value in ['E99', 'E0', 'E4 Numerical Error', 'E4: some speculative explanation', None, []]:
            self.assertIsNone(normalize_code(value))

    def test_clean_result_uses_one_call_and_python_owns_stage(self):
        final, audit, calls = run(retrieval_result())
        self.assertEqual(len(calls), 1)
        self.assertEqual(final['supported_failures'], ['E4'])
        self.assertEqual(final['primary_failure'], 'E4')
        self.assertEqual(final['primary_stage'], 'Retrieval')
        self.assertEqual(final['attribution_status'], 'established')
        self.assertNotIn('scores', final)
        self.assertEqual(final['stages']['Retrieval']['failures'], ['E4'])

    def test_valid_e4_survives_invalid_e13(self):
        final, audit, calls = run(add_bad_e13(retrieval_result()))
        self.assertEqual(final['supported_failures'], ['E4'])
        self.assertEqual(final['primary_failure'], 'E4')
        self.assertEqual(final['label_assessments']['E13']['status'], 'rejected')
        self.assertEqual(final['attribution_status'], 'partially_established')
        self.assertEqual(len(calls), 1)  # Single-pass validation preserves independent findings.
        self.assertTrue(final['label_assessments']['E13']['rejected_citations'])

    def test_invalid_primary_does_not_delete_supported_label(self):
        raw = retrieval_result()
        raw['primary_failure'] = 'E13'
        final, _, calls = run(raw)
        self.assertEqual(final['supported_failures'], ['E4'])
        self.assertIsNone(final['primary_failure'])
        self.assertEqual(final['primary_assessment']['status'], 'rejected')
        self.assertEqual(len(calls), 1)

    def test_supported_label_is_not_automatically_promoted_to_primary(self):
        raw = retrieval_result()
        raw.pop('primary_reason')
        final, _, _ = run(raw)
        self.assertEqual(final['supported_failures'], ['E4'])
        self.assertIsNone(final['primary_failure'])
        self.assertEqual(final['attribution_status'], 'partially_established')

    def test_fabricated_id_rejects_only_unsupported_finding(self):
        raw = retrieval_result()
        raw['label_evidence']['E4 Missed Retrieval'][0]['item_id'] = 'invented'
        final, _, calls = run(raw)
        self.assertEqual(final['supported_failures'], [])
        self.assertEqual(final['label_assessments']['E4']['status'], 'rejected')
        self.assertTrue(any('absent' in item['reason'] for item in final['label_assessments']['E4']['rejected_citations']))
        self.assertEqual(len(calls), 1)

    def test_bad_extra_citation_does_not_erase_other_valid_evidence_for_same_label(self):
        raw = retrieval_result()
        raw['label_evidence']['E4 Missed Retrieval'].append(citation('indexed_chunks', 'fake', 'Michael Evans'))
        final, _, _ = run(raw)
        self.assertEqual(final['supported_failures'], ['E4'])
        self.assertEqual(final['label_assessments']['E4']['status'], 'supported')
        self.assertEqual(len(final['label_assessments']['E4']['rejected_citations']), 1)

    def test_missing_reranking_trace_is_insufficient_not_invented(self):
        raw = retrieval_result()
        raw['supported_failures'].append('E7')
        raw['label_evidence']['E7'] = [citation('retrieved_candidates', '63_10', 'Emily Anderson')]
        case = copy.deepcopy(CASE)
        case['generation_context'] = None
        final, _, calls = run(raw, case)
        self.assertEqual(final['supported_failures'], ['E4'])
        self.assertEqual(final['label_assessments']['E7']['status'], 'insufficient_evidence')
        self.assertIn('E7', final['unresolved_failures'])
        self.assertEqual(len(calls), 1)  # A second model cannot recover the missing trace.

    def test_present_support_cannot_be_claimed_as_missing(self):
        case = copy.deepcopy(CASE)
        case['retrieved_candidates'].extend(case['indexed_chunks'])
        self.assertTrue(any('All gold documents' in error for error in verify_evidence(retrieval_result(), case)))

    def test_all_invalid_labels_require_review_not_no_failure(self):
        raw = empty_result()
        raw['supported_failures'] = ['E99']
        final, _, calls = run(raw)
        self.assertEqual(final['supported_failures'], [])
        self.assertEqual(final['invalid_labels'][0]['status'], 'invalid_label')
        self.assertEqual(final['attribution_status'], 'needs_review')
        self.assertEqual(len(calls), 1)

    def test_deduplicates_aliases_and_ignores_stage_list_formatting(self):
        raw = retrieval_result()
        raw['supported_failures'].extend(['E4', 'E4: Missed Retrieval'])
        raw['stages']['Retrieval']['failures'] = ['E99']
        final, _, calls = run(raw)
        self.assertEqual(final['supported_failures'], ['E4'])
        self.assertEqual(final['stages']['Retrieval']['failures'], ['E4'])
        self.assertEqual(len(calls), 1)

    def test_targeted_review_cannot_replace_frozen_supported_labels(self):
        raw = retrieval_result()
        raw['unresolved_alternatives'] = ['E10']
        raw['label_evidence']['E10'] = [citation('original_rag_answer', None, 'Emily Anderson'),
                                       citation('generation_context', '63_10', 'Emily Anderson')]
        patch = {'label_updates': {'E4': {'status': 'rejected'},
                                 'E10': {'status': 'supported', 'reason': 'Resolved interpretation.',
                                         'evidence': raw['label_evidence']['E10'], 'key_points': ['Observed defect.']}},
                 'primary_update': None}
        outputs = iter([raw, patch])
        calls = []
        def ask(**request):
            calls.append(request)
            return {'parsed': next(outputs)}
        final, audit = classify_case(CASE, SimpleNamespace(ask_json=ask))
        self.assertEqual(final['supported_failures'], ['E4', 'E10'])
        self.assertEqual(final['primary_failure'], 'E4')
        self.assertEqual(len(calls), 2)
        self.assertIn('E10', audit[1]['target_codes'])
        self.assertNotIn('E4', audit[1]['target_codes'])
        self.assertTrue(any('E4' in item for item in audit[1]['ignored_updates']))
        self.assertEqual(final['initial_prediction'], raw)
        self.assertEqual(audit[-1]['final_accepted_prediction'], final)

    def test_declared_insufficient_label_is_preserved_even_outside_supported_list(self):
        raw = retrieval_result()
        raw['label_assessments'] = {'E7': {'status': 'insufficient_evidence', 'reason': 'No generation trace.'}}
        case = copy.deepcopy(CASE)
        case['generation_context'] = None
        final, _, calls = run(raw, case)
        self.assertEqual(final['label_assessments']['E7']['status'], 'insufficient_evidence')
        self.assertEqual(final['supported_failures'], ['E4'])
        self.assertEqual(len(calls), 1)

    def test_primary_semantic_recheck_is_explicit_and_cannot_invent_labels(self):
        raw = retrieval_result()
        raw['attribution_status'] = 'needs_review'
        raw['primary_needs_review'] = True
        patch = {'label_updates': {}, 'primary_update': {
            'failure': 'E4', 'status': 'established', 'reason': raw['primary_reason'],
            'evidence': raw['primary_evidence']}}
        outputs = iter([raw, patch])
        final, audit = classify_case(CASE, SimpleNamespace(ask_json=lambda **kwargs: {'parsed': next(outputs)}))
        self.assertEqual(final['primary_failure'], 'E4')
        self.assertEqual(final['supported_failures'], ['E4'])
        self.assertEqual(final['passes_used'], 2)
        self.assertTrue(audit[1]['primary_review_requested'])
        raw.pop('primary_needs_review')
        final, _, calls = run(raw)
        self.assertIsNone(final['primary_failure'])
        self.assertEqual(len(calls), 1)

    def test_same_validation_and_recheck_policy_across_b0_b1_b2(self):
        for method in ['b0', 'b1', 'b2']:
            final, _, calls = run(add_bad_e13(retrieval_result()), method=method)
            self.assertEqual(final['supported_failures'], ['E4'])
            self.assertEqual(final['validation_policy'], VALIDATION_POLICY)
            self.assertEqual(len(calls), 1)
        with self.assertRaises(ValueError):
            classify_case(CASE, None, max_passes=3)

    def test_unknown_label_does_not_erase_known_supported_label(self):
        raw = retrieval_result()
        raw['supported_failures'].append('E99')
        final, _, calls = run(raw)
        self.assertEqual(final['supported_failures'], ['E4'])
        self.assertEqual(final['attribution_status'], 'partially_established')
        self.assertEqual(final['invalid_labels'][0]['status'], 'invalid_label')
        self.assertEqual(len(calls), 1)

    def test_b0_b1_b2_share_identical_targeted_review_budget(self):
        prompts = []
        for method in ['b0', 'b1', 'b2']:
            raw = retrieval_result()
            raw['unresolved_alternatives'] = ['E10']
            raw['label_evidence']['E10'] = [citation('original_rag_answer', None, 'Emily Anderson'),
                                           citation('generation_context', '63_10', 'Emily Anderson')]
            outputs = iter([raw, {'label_updates': {}, 'primary_update': None}])
            calls = []
            def ask(**request):
                calls.append(request['prompt'])
                return {'parsed': next(outputs)}
            final, _ = classify_case(CASE, SimpleNamespace(ask_json=ask), method=method)
            self.assertEqual(final['passes_used'], 2)
            self.assertEqual(final['supported_failures'], ['E4'])
            prompts.append(calls)
        self.assertEqual(prompts[0], prompts[1])
        self.assertEqual(prompts[1], prompts[2])

    def test_candidate_records_validate_independently_and_legacy_still_works(self):
        legacy = add_bad_e13(retrieval_result())
        raw = {key: value for key, value in legacy.items() if key not in {'supported_failures', 'key_points', 'label_evidence'}}
        raw['candidate_failures'] = [
            {'code': 'E4', 'evidence': legacy['label_evidence']['E4 Missed Retrieval'], 'key_points': ['Indexed support was missed.']},
            {'code': 'E13', 'evidence': legacy['label_evidence']['E13'], 'key_points': ['Proposed misuse.']},
        ]
        final, _, calls = run(raw)
        self.assertEqual(final['supported_failures'], ['E4'])
        self.assertEqual(final['label_assessments']['E13']['status'], 'rejected')
        self.assertEqual(final['primary_failure'], 'E4')
        self.assertEqual(len(calls), 1)

    def rerank_case(self, order):
        case = copy.deepcopy(CASE)
        case['source_documents'] = [{'doc_id': 'a'}, {'doc_id': 'b'}]
        case['retrieved_candidates'] = [{'chunk_id': f'{d}_0', 'text': d} for d in 'abxyz']
        case['generation_context'] = [{'chunk_id': f'{d}_{i}', 'text': d}
                                      for i, d in enumerate(order)]
        return case

    def test_e7_exact_half_loss_is_supported(self):
        raw = empty_result()
        raw['candidate_failures'] = [{'code': 'E7', 'key_points': ['Half of retrieved gold docs lost.']}]
        final, _, _ = run(raw, self.rerank_case('axxyz'))
        self.assertIn('E7', final['supported_failures'])
        self.assertEqual(final['label_assessments']['E7']['document_metrics']['lost_fraction'], .5)

    def test_e7_different_chunk_same_document_is_retained(self):
        raw = empty_result()
        raw['candidate_failures'] = [{'code': 'E7', 'key_points': ['Proposed loss.']}]
        final, _, _ = run(raw, self.rerank_case('abxyz'))
        self.assertEqual(final['label_assessments']['E7']['status'], 'rejected')

    def test_e8_gold_documents_ranked_four_and_five_fail(self):
        raw = empty_result()
        raw['candidate_failures'] = [{'code': 'E8', 'key_points': ['Gold ranked too low.']}]
        final, _, _ = run(raw, self.rerank_case('xyzab'))
        self.assertIn('E8', final['supported_failures'])

    def test_e8_two_gold_documents_in_top_three_pass(self):
        raw = empty_result()
        raw['candidate_failures'] = [{'code': 'E8', 'key_points': ['Proposed low rank.']}]
        final, _, _ = run(raw, self.rerank_case('axbyz'))
        self.assertEqual(final['label_assessments']['E8']['status'], 'rejected')


    def test_repair_can_replace_wrong_e4_diagnosis_with_generation_failure(self):
        case = copy.deepcopy(CASE)
        case['retrieved_candidates'].extend(case['indexed_chunks'])
        case['generation_context'].extend(case['indexed_chunks'])
        raw = retrieval_result()  # E4 is wrong: its supporting chunk was retrieved.
        patch = {'candidate_updates': [{'code': 'E13', 'status': 'supported',
            'reason': 'Supplied evidence names Michael Evans, but the answer names Emily Anderson.',
            'key_points': ['Actual supplied evidence was misused.'],
            'evidence': [citation('original_rag_answer', None, 'Emily Anderson'),
                         citation('generation_context', '47_11', 'Michael Evans')]}],
            'primary_update': {'failure': 'E13', 'status': 'established',
                'reason': 'The supplied answer-bearing fact was misused.',
                'evidence': [citation('original_rag_answer', None, 'Emily Anderson')]}}
        outputs = iter([raw, patch]); calls = []
        def ask(**request):
            calls.append(request)
            return {'parsed': next(outputs)}
        final, audit = classify_case(case, SimpleNamespace(ask_json=ask))
        self.assertEqual(final['supported_failures'], ['E13'])
        self.assertEqual(final['primary_failure'], 'E13')
        self.assertEqual(final['answer_assessment']['status'], 'reported_defect')
        self.assertEqual(audit[1]['recheck_kind'], 'evidence_repair')
        self.assertEqual(len(calls), 2)
        self.assertEqual(final['initial_prediction'], raw)

    def test_bad_second_pass_cannot_accept_fabricated_citations(self):
        raw = retrieval_result()
        raw['label_evidence']['E4 Missed Retrieval'][0]['item_id'] = 'fake'
        patch = {'candidate_updates': [{'code': 'E4', 'status': 'supported',
                 'evidence': raw['label_evidence']['E4 Missed Retrieval'], 'key_points': ['Same unsupported proposal.']}]}
        outputs = iter([raw, patch]);calls = []
        def ask(**request):
            calls.append(request)
            return {'parsed': next(outputs)}
        final, _ = classify_case(CASE, SimpleNamespace(ask_json=ask))
        self.assertEqual(final['supported_failures'], [])
        self.assertEqual(final['label_assessments']['E4']['status'], 'rejected')
        self.assertEqual(len(calls), 2)
        self.assertEqual(final['attribution_status'], 'needs_review')

    def test_repair_cannot_invent_missing_generation_record_or_rewrite_frozen_label(self):
        case = copy.deepcopy(CASE)
        case['generation_context'] = None
        raw = retrieval_result()
        raw['candidate_failures'] = [{'code': 'E7', 'status': 'supported', 'evidence': [citation('retrieved_candidates', '63_10', 'Emily Anderson')]}]
        final, _, calls = run(raw, case, max_passes=2)
        self.assertEqual(final['supported_failures'], ['E4'])
        self.assertEqual(final['label_assessments']['E7']['status'], 'insufficient_evidence')
        self.assertEqual(len(calls), 1)

    def test_evidence_repair_policy_is_identical_across_methods(self):
        prompts = []
        for method in ['b0', 'b1', 'b2']:
            raw = retrieval_result()
            raw['label_evidence']['E4 Missed Retrieval'][0]['item_id'] = 'fake'
            outputs = iter([raw, {'candidate_updates': []}]); calls=[]
            def ask(**request):
                calls.append(request['prompt'])
                return {'parsed': next(outputs)}
            final, _ = classify_case(CASE, SimpleNamespace(ask_json=ask), method=method)
            self.assertEqual(final['passes_used'], 2)
            prompts.append(calls)
        self.assertEqual(prompts[0], prompts[1])
        self.assertEqual(prompts[1], prompts[2])

    def test_single_failure_selection_uses_only_validated_primary(self):
        final, _, _ = run(retrieval_result(), max_failures=1)
        self.assertEqual(final['failures'], ['E4'])
        self.assertEqual(final['selection_mode'], 'validated_primary_only')
        raw = retrieval_result()
        raw['primary_failure'] = 'E13'  # Unsupported primary; E4 remains a diagnostic finding.
        final, _, _ = run(raw, max_failures=1)
        self.assertEqual(final['supported_failures'], ['E4'])
        self.assertEqual(final['failures'], [])
        self.assertIsNone(final['primary_failure'])
        self.assertEqual(final['attribution_status'], 'partially_established')

    def test_missing_generation_output_gets_bounded_review(self):
        result, audit = classify_case(CASE, SimpleNamespace(ask_json=lambda **kwargs: {'invalid_output': {}}))
        self.assertEqual(result['passes_used'], 2)
        self.assertEqual(result['attribution_status'], 'needs_review')
        self.assertEqual(result['supported_failures'], [])
        self.assertEqual(audit[0]['output'], {})


if __name__ == '__main__':
    unittest.main()
