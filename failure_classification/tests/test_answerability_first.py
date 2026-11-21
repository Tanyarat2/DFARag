"""Exercise the real first-stage orchestration and Python-owned E9 decision."""
import copy
import unittest
from types import SimpleNamespace
from failure_classification.errorclassification.answerability.assessor import (
    AnswerabilityAssessor, finalize_assessment, derive_e9, construct_prompt,
)
from failure_classification.errorclassification.pipeline.classifier import classify_case
from failure_classification.errorclassification.validation.validator import EvidenceValidator


def case():
    return {'case_id': 'cash-flow', 'original_question': 'What was cash flow?',
        'original_rag_answer': 'Cash flow was $50 million.', 'gold_answer': 'Unable to answer',
        'generation_context': [{'chunk_id': 'a_0', 'text': 'Net profit was $50 million.'}],
        'retrieved_candidates': [{'chunk_id': 'a_0', 'text': 'Net profit was $50 million.'}],
        'source_documents': [{'doc_id': 'a', 'text': 'Net profit was $50 million.'}],
        'indexed_chunks': [{'chunk_id': 'a_0', 'text': 'Net profit was $50 million.'}]}


def requirement(c, status='unanswerable', behavior='answered', limitation=False):
    return {'requirement': 'Cash flow', 'answerability': status,
        'reason': 'The supplied financial measure is net profit; it does not establish cash flow.',
        'response_behavior': behavior, 'acknowledged_limitation': limitation,
        'limitation_quote': c['original_rag_answer'] if limitation else None,
        'answer_assertion': c['original_rag_answer'] if behavior == 'answered' else None,
        'evidence': ['answer', 'context_1']}


def assessment(c, **options):
    return finalize_assessment({'requirements': [requirement(c, **options)]}, c)


def generation(c):
    return {'reason': 'The generator used the supplied net-profit measure as cash flow.',
        'evidence': ['answer', 'context_1'],
        'checks': {f'E{i}': {'status': 'defect' if i == 13 else 'unknown',
            'reason': 'Net profit was misapplied as cash flow.' if i == 13 else 'No separate mechanism established.',
            'evidence': ['answer', 'context_1'] if i == 13 else []} for i in range(10, 17)}}


class AnswerabilityFirstTests(unittest.TestCase):
    def test_e9_does_not_require_falsity_or_a_model_e9_proposal(self):
        c = case()
        first = assessment(c)
        _, labels, _ = EvidenceValidator(c, first).validate({})
        self.assertEqual(labels['E9']['status'], 'supported')
        self.assertEqual(labels['E9']['origin'], 'computed_answerability_rule')
        self.assertFalse(derive_e9(first)['proof_of_falsity_required'])

    def test_later_unknown_answerability_cannot_overwrite_first_decision(self):
        c = case()
        raw = {'generation_assessment': {'answerability': 'unknown', 'response_behavior': 'unknown',
                                        'checks': {'E9': {'status': 'no_defect'}}}}
        normalized, labels, _ = EvidenceValidator(c, assessment(c)).validate(raw)
        self.assertEqual(labels['E9']['status'], 'supported')
        self.assertEqual(normalized['raw']['generation_assessment']['answerability'], 'unanswerable')

    def test_later_e9_proposal_cannot_override_answerable_input(self):
        c = case()
        _, labels, _ = EvidenceValidator(c, assessment(c, status='answerable')).validate({
            'candidate_failures': [{'code': 'E9', 'status': 'supported', 'evidence': []}]})
        self.assertEqual(labels['E9']['status'], 'rejected')

    def test_abstention_and_appropriate_limitation_do_not_trigger_e9(self):
        c = case();c['original_rag_answer'] = 'The supplied information does not establish cash flow.'
        for behavior in ('abstained', 'partial_with_limitation'):
            self.assertEqual(derive_e9(assessment(c, behavior=behavior, limitation=True))['status'], 'no_failure')

    def test_unknown_is_not_converted_to_unanswerable(self):
        c = case()
        self.assertEqual(derive_e9(assessment(c, status='unknown'))['status'], 'unknown')

    def test_partial_request_is_decided_per_aspect(self):
        c = case()
        supported = requirement(c, status='answerable')
        supported['requirement'] = 'Net profit'
        missing = requirement(c)
        first = finalize_assessment({'requirements': [supported, missing]}, c)
        self.assertEqual(first['answerability'], 'partially_answerable')
        self.assertEqual(derive_e9(first)['status'], 'failure')
        self.assertEqual(len(derive_e9(first)['affected_requirements']), 1)
        missing.update(response_behavior='abstained', answer_assertion=None, acknowledged_limitation=True,
                       limitation_quote=c['original_rag_answer'])
        first = finalize_assessment({'requirements': [supported, missing]}, c)
        self.assertEqual(derive_e9(first)['status'], 'no_failure')

    def test_placeholders_and_fabricated_references_are_not_completed(self):
        c = case()
        for changed in ({'reason': 'Case-specific assessment'}, {'evidence': ['context_999']}):
            req = requirement(c);req.update(changed)
            first = finalize_assessment({'requirements': [req]}, c)
            self.assertEqual(first['status'], 'needs_review')
            self.assertEqual(derive_e9(first)['status'], 'unknown')

    def test_first_prompt_excludes_gold_answers_and_failure_labels(self):
        c = case();c['gold_answer'] = 'SENTINEL_REFERENCE';c['error_category'] = 'SENTINEL_LABEL'
        prompt = construct_prompt(c)
        self.assertNotIn('SENTINEL_REFERENCE', prompt)
        self.assertNotIn('SENTINEL_LABEL', prompt)

    def test_missing_logs_skip_request_and_remain_unknown(self):
        c = case();c['generation_context'] = None
        runner = SimpleNamespace(ask_json=lambda **kw: self.fail('Must not invent missing logs'))
        first, audit = AnswerabilityAssessor(runner).assess(c)
        self.assertEqual(audit, [])
        self.assertEqual(derive_e9(first)['status'], 'unknown')

    def test_real_orchestration_runs_first_assessment_before_classification(self):
        c = case()
        calls = []
        def ask(**request):
            calls.append(request)
            if request['stage'].startswith('answerability_first'):
                return {'parsed': {'requirements': [requirement(c)]}}
            self.assertIn('FROZEN ANSWERABILITY-FIRST ASSESSMENT', request['prompt'])
            return {'parsed': {'answer_defect': {'status': 'defect', 'summary': 'The supplied financial measure was misused.'},
                'attribution_status': 'needs_review', 'generation_assessment': generation(c)}}
        result, _ = classify_case(c, SimpleNamespace(ask_json=ask), max_passes=1, max_failures=16)
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0]['stage'].startswith('answerability_first'))
        self.assertIn('E9', result['supported_failures'])
        self.assertIn('E13', result['supported_failures'])
        self.assertEqual(result['e9_assessment']['status'], 'failure')
        self.assertEqual(result['answerability_requests'], 1)
        self.assertEqual(result['total_model_requests'], 2)

    def test_malformed_first_assessment_gets_only_one_repair(self):
        c = case();calls = []
        def ask(**request):
            calls.append(request)
            return {'parsed': {'requirements': [dict(requirement(c), reason='Case-specific assessment')]}}
        first, audit = AnswerabilityAssessor(SimpleNamespace(ask_json=ask)).assess(c)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(audit), 2)
        self.assertEqual(first['status'], 'needs_review')

    def test_e9_can_coexist_with_unsupported_content(self):
        c = case()
        from failure_classification.errorclassification.evidence.assessment import normalize_evidence_response
        raw, _ = normalize_evidence_response({'candidate_failures': [{'code': 'E10', 'status': 'supported',
            'reason': 'Cash flow is unsupported', 'evidence': ['answer', 'context_1']}]}, c)
        _, labels, _ = EvidenceValidator(c, assessment(c)).validate(raw)
        self.assertEqual(labels['E9']['status'], 'supported')
        self.assertEqual(labels['E10']['status'], 'supported')

    def test_automatic_e9_is_selected_in_single_label_mode_without_model_nomination(self):
        c = case()
        def ask(**request):
            if request['stage'].startswith('answerability_first'):
                return {'parsed': {'requirements': [requirement(c)]}}
            return {'parsed': {'answer_defect': {'status': 'no_defect', 'summary': 'Incorrect model assessment'},
                'attribution_status': 'no_failure', 'candidate_failures': [],
                'generation_assessment': generation(c)}}
        result, _ = classify_case(c, SimpleNamespace(ask_json=ask), max_passes=1, max_failures=1)
        self.assertEqual(result['primary_failure'], 'E9')
        self.assertEqual(result['failures'], ['E9'])
        self.assertEqual(result['answer_defect']['origin'], 'computed_answerability_rule')

    def test_claimed_limitation_must_exist_in_the_actual_answer(self):
        c = case()
        item = requirement(c, behavior='partial_with_limitation', limitation=True)
        item['limitation_quote'] = 'I do not have that information.'
        first = finalize_assessment({'requirements': [item]}, c)
        self.assertEqual(first['status'], 'needs_review')

    def test_partial_logs_cannot_establish_absence(self):
        c = case();c['trace_status'] = {'generation_context': 'partial saved trace'}
        self.assertEqual(assessment(c)['status'], 'needs_review')

    def test_computed_e9_is_frozen_during_later_review(self):
        c = case()
        calls = []
        def ask(**request):
            calls.append(request['stage'])
            if request['stage'].startswith('answerability_first'):
                return {'parsed': {'requirements': [requirement(c)]}}
            if request['stage'] == 'evidence_first_v5_initial':
                return {'parsed': {'answer_defect': {'status': 'defect', 'summary': 'Unsupported answer'},
                    'generation_assessment': {'reason': 'Case-specific assessment'}}}
            return {'parsed': {'candidate_updates': [{'code': 'E9', 'status': 'rejected'}],
                'generation_assessment_update': generation(c)}}
        result, audit = classify_case(c, SimpleNamespace(ask_json=ask), max_passes=2)
        self.assertEqual(result['e9_assessment']['status'], 'failure')
        self.assertIn('E9', result['supported_failures'])
        self.assertNotIn('E9', audit[0]['recheck_plan']['allowed_codes'])
        self.assertTrue(any('E9' in item for item in audit[-1]['ignored_updates']))

    def test_exact_assertion_is_already_a_valid_answer_reference(self):
        c = case()
        item = requirement(c)
        item['evidence'] = ['context_1']
        first = finalize_assessment({'requirements': [item]}, c)
        self.assertEqual(first['status'], 'completed')
        self.assertEqual(derive_e9(first)['status'], 'failure')

    def test_unanswerability_cannot_be_based_on_only_one_of_multiple_inputs(self):
        c = case();c['generation_context'].append({'chunk_id': 'a_1', 'text': 'Additional financial details.'})
        item = requirement(c)
        first = finalize_assessment({'requirements': [item]}, c)
        self.assertEqual(first['status'], 'needs_review')
        item['evidence'].append('context_2')
        self.assertEqual(finalize_assessment({'requirements': [item]}, c)['status'], 'completed')

    def test_explicit_full_context_selection_supports_scoped_absence(self):
        c = case()
        item = requirement(c);item['evidence'] = ['generation_context']
        self.assertEqual(finalize_assessment({'requirements': [item]}, c)['status'], 'completed')
