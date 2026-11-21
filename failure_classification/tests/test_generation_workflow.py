"""Regression checks for generation review independent of retrieval diagnoses."""
import copy
import unittest
from failure_classification.tests.answerability_fixtures import AttributionOnlyTestCase
from types import SimpleNamespace
from failure_classification.errorclassification.pipeline.classifier import classify_case
from failure_classification.errorclassification.evidence.assessment import pipeline_diagnostics, generation_assessment_errors


def cite(source, item_id, quote, observation='Observed original record'):
    return dict(source=source, item_id=item_id, quote=quote, observation=observation)


def case():
    return dict(case_id='generation', original_question='What was cash flow?',
        original_rag_answer='Cash flow was $50 million.', gold_answer='Unable to answer',
        source_documents=[{'doc_id': 'gold', 'text': 'Net profit was $50 million.'}],
        indexed_chunks=[{'chunk_id': 'gold_0', 'text': 'Net profit was $50 million.'}],
        retrieved_candidates=[{'chunk_id': 'gold_0', 'text': 'Net profit was $50 million.'}],
        generation_context=[{'chunk_id': 'gold_0', 'text': 'Net profit was $50 million.'}]
            + [{'chunk_id': f'other_{i}', 'text': 'Other context.'} for i in range(4)])


def assessment(c, defect=None):
    return dict(answerability='unanswerable' if defect else 'answerable',
        response_behavior='answered', reason='Actual input gives net profit, not cash flow.',
        evidence=[cite('original_rag_answer', None, c['original_rag_answer']),
                  cite('generation_context', 'gold_0', 'Net profit was $50 million.')],
        checks={f'E{i}': {'status': 'defect' if f'E{i}' == defect else 'no_defect',
                         'reason': 'Net profit was used as cash flow.' if f'E{i}' == defect else 'No separate defect established.'}
                for i in range(9, 17)})


class GenerationWorkflowTests(AttributionOnlyTestCase):
    def run_outputs(self, c, outputs, **kwargs):
        replies = iter(outputs)
        calls = []
        def ask(**request):
            calls.append(request)
            return {'parsed': next(replies)}
        result, audit = classify_case(c, SimpleNamespace(ask_json=ask), **kwargs)
        return result, audit, calls

    def test_thresholds_are_emitted_even_when_model_proposes_nothing(self):
        c = case()
        raw = {'answer_defect': {'status': 'no_defect', 'summary': 'Fixture'},
               'attribution_status': 'no_failure', 'generation_assessment': assessment(c)}
        result, _, calls = self.run_outputs(c, [raw])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['pipeline_failures'], ['E8'])
        self.assertEqual(result['supported_failures'], [])
        self.assertIsNone(result['primary_failure'])
        self.assertEqual(result['answer_assessment']['status'], 'no_defect_reported')
        self.assertEqual(result['generation_review']['status'], 'completed')

    def test_supported_retrieval_primary_does_not_suppress_generation_review(self):
        c = case()
        c['source_documents'].append({'doc_id': 'missing', 'text': 'Another reference.'})
        raw = {'answer_defect': {'status': 'defect', 'summary': 'Profit was used as cash flow.'},
               'candidate_failures': [{'code': 'E4', 'status': 'supported',
                    'evidence': [cite('source_documents', 'missing', 'Another reference.'),
                                 cite('retrieved_candidates', None, None, 'Gold document missing is absent.')]}],
               'primary_failure': 'E4', 'primary_reason': 'The missing document caused a wrong answer.',
               'primary_evidence': [cite('original_rag_answer', None, c['original_rag_answer'])],
               'attribution_status': 'established'}
        evidence = [cite('generation_context', 'gold_0', 'Net profit was $50 million.'),
                    cite('original_rag_answer', None, c['original_rag_answer'])]
        patch = {'generation_assessment_update': assessment(c, 'E13'),
                 'candidate_updates': [{'code': 'E13', 'status': 'supported',
                    'key_points': ['A supplied profit quantity was misapplied as cash flow.'], 'evidence': evidence}],
                 'primary_update': {'failure': 'E13', 'status': 'established',
                    'reason': 'The generator misapplied the supplied net-profit figure.', 'evidence': evidence}}
        result, audit, calls = self.run_outputs(c, [raw, patch])
        self.assertEqual(len(calls), 2)
        self.assertTrue(audit[0]['recheck_plan']['generation_review_requested'])
        self.assertEqual(result['primary_failure'], 'E13')
        self.assertEqual(result['generation_review']['status'], 'completed')
        self.assertEqual(result['pipeline_failures'], ['E4', 'E8'])

    def test_invalid_quote_can_be_repaired_without_relaxing_validation(self):
        c = case()
        raw = {'answer_defect': {'status': 'defect', 'summary': 'Wrong relation'},
               'candidate_failures': [{'code': 'E13', 'status': 'supported',
                   'evidence': [cite('generation_context', 'gold_0', 'Fabricated quotation')]}]}
        patch = {'generation_assessment_update': assessment(c, 'E13'),
                 'candidate_updates': [{'code': 'E13', 'status': 'supported',
                   'key_points': ['Net profit misrepresented as cash flow'],
                   'evidence': [cite('generation_context', 'gold_0', 'Net profit was $50 million.'),
                                cite('original_rag_answer', None, 'Cash flow was $50 million.')]}]}
        result, audit, _ = self.run_outputs(c, [raw, patch])
        self.assertEqual(audit[0]['label_assessments']['E13']['status'], 'rejected')
        self.assertIn('E13', result['supported_failures'])
        self.assertEqual(result['generation_review']['status'], 'completed')

    def test_missing_checks_remain_unresolved_after_review_budget(self):
        c = case()
        raw = {'answer_defect': {'status': 'no_defect', 'summary': 'Unverified'}, 'attribution_status': 'no_failure'}
        result, _, calls = self.run_outputs(c, [raw, {}])
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['generation_review']['status'], 'needs_review')
        self.assertEqual(result['final_attribution']['status'], 'unresolved')

    def test_defect_without_candidate_is_not_completed(self):
        c = case()
        raw = {'answer_defect': {'status': 'defect', 'summary': 'Misapplied fact'},
               'generation_assessment': assessment(c, 'E13')}
        result, _, calls = self.run_outputs(c, [raw], max_passes=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['generation_review']['status'], 'needs_review')
        self.assertNotIn('E13', result['supported_failures'])

    def test_invalid_assessment_quote_is_detected(self):
        c = case()
        record = assessment(c)
        record['evidence'][1]['quote'] = 'Invented'
        self.assertTrue(generation_assessment_errors(record, c))

    def test_missing_logs_do_not_force_e9_or_answerability(self):
        c = case()
        c['generation_context'] = None
        raw = {'answer_defect': {'status': 'uncertain', 'summary': 'Missing context'}}
        result, _, calls = self.run_outputs(c, [raw])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['generation_review']['status'], 'needs_review')
        self.assertNotIn('E9', result['supported_failures'])


class EvidenceReferenceTests(AttributionOnlyTestCase):
    def test_explicit_reference_resolves_exact_original_record(self):
        from failure_classification.errorclassification.evidence.assessment import normalize_evidence_response
        c = case()
        raw = {'candidate_updates': [{'code': 'E13', 'status': 'supported',
                                     'evidence': ['answer', {'ref': 'context_1', 'observation': 'Profit supplied'}]}]}
        normalized, paths = normalize_evidence_response(raw, c)
        evidence = normalized['candidate_updates'][0]['evidence']
        self.assertEqual(evidence[0]['quote'], c['original_rag_answer'])
        self.assertEqual(evidence[1]['quote'], c['generation_context'][0]['text'])
        self.assertEqual(evidence[1]['item_id'], 'gold_0')
        self.assertEqual(evidence[1]['observation'], 'Profit supplied')
        self.assertEqual(len(paths), 2)
        self.assertEqual(raw['candidate_updates'][0]['evidence'][0], 'answer')

    def test_unknown_reference_is_not_repaired_or_guessed(self):
        from failure_classification.errorclassification.evidence.assessment import normalize_evidence_response
        raw = {'candidate_updates': [{'code': 'E13', 'status': 'supported', 'evidence': ['context_999']}]}
        normalized, paths = normalize_evidence_response(raw, case())
        self.assertEqual(normalized, raw)
        self.assertEqual(paths, [])

    def test_reference_does_not_create_an_unproposed_candidate(self):
        from failure_classification.errorclassification.evidence.assessment import normalize_evidence_response
        raw = {'generation_assessment_update': {'evidence': ['answer', 'context_1']}, 'candidate_updates': []}
        normalized, _ = normalize_evidence_response(raw, case())
        self.assertEqual(normalized['candidate_updates'], [])

    def test_existing_false_quote_is_not_silently_replaced(self):
        from failure_classification.errorclassification.evidence.assessment import normalize_evidence_response
        raw = {'candidate_updates': [{'code': 'E13', 'evidence': [
            cite('generation_context', 'gold_0', 'False quotation')]}]}
        normalized, paths = normalize_evidence_response(raw, case())
        self.assertEqual(normalized, raw)
        self.assertEqual(paths, [])

    def test_explicit_reference_flows_through_review_and_citation_validation(self):
        c = case()
        initial = {'answer_defect': {'status': 'defect', 'summary': 'Wrong relationship'}}
        g = assessment(c, 'E13')
        g['evidence'] = ['answer', 'context_1']
        patch = {'generation_assessment_update': g,
                 'candidate_updates': [{'code': 'E13', 'status': 'supported',
                    'key_points': ['Net profit used as cash flow'], 'evidence': ['answer', 'context_1']}],
                 'primary_update': {'failure': 'E13', 'status': 'established',
                    'reason': 'The supplied net-profit figure was used as cash flow.', 'evidence': ['answer', 'context_1']}}
        replies = iter([initial, patch])
        result, audit = classify_case(c, SimpleNamespace(ask_json=lambda **kw: {'parsed': next(replies)}))
        self.assertEqual(result['primary_failure'], 'E13')
        self.assertEqual(result['generation_review']['status'], 'completed')
        self.assertTrue(audit[1]['review_response_attempts'][0]['output_normalization'])

    def test_unknown_check_vocabulary_still_requires_review(self):
        from failure_classification.errorclassification.evidence.assessment import normalize_evidence_response
        c = case()
        raw = {'generation_assessment': assessment(c)}
        raw['generation_assessment']['checks']['E10']['status'] = 'maybe-something'
        normalized, _ = normalize_evidence_response(raw, c)
        self.assertTrue(generation_assessment_errors(normalized['generation_assessment'], c))

    def test_explicit_defect_with_own_evidence_is_a_model_proposal(self):
        from failure_classification.errorclassification.evidence.assessment import normalize_evidence_response
        c = case()
        g = assessment(c, 'E13')
        g['checks']['E13']['evidence'] = ['answer', 'context_1']
        normalized, _ = normalize_evidence_response({'generation_assessment_update': g}, c)
        candidate = normalized['candidate_updates'][0]
        self.assertEqual(candidate['code'], 'E13')
        self.assertEqual(candidate['status'], 'supported')
        self.assertEqual(candidate['origin'], 'explicit_generation_check')
        self.assertEqual(candidate['evidence'][1]['quote'], c['generation_context'][0]['text'])

    def test_defect_without_own_evidence_does_not_borrow_top_level_evidence(self):
        from failure_classification.errorclassification.evidence.assessment import normalize_evidence_response
        c = case()
        normalized, _ = normalize_evidence_response({'generation_assessment_update': assessment(c, 'E13')}, c)
        self.assertNotIn('candidate_updates', normalized)

    def test_catalog_source_alias_keeps_false_quote_for_rejection(self):
        from failure_classification.errorclassification.evidence.assessment import normalize_evidence_response
        from failure_classification.errorclassification.validation.validator import check_citation
        raw = {'candidate_updates': [{'code': 'E13', 'evidence': [
            cite('context_1', 'gold_0', 'Invented quotation')]}]}
        normalized, _ = normalize_evidence_response(raw, case())
        evidence = normalized['candidate_updates'][0]['evidence'][0]
        self.assertEqual(evidence['source'], 'generation_context')
        self.assertIsNotNone(check_citation(evidence, case()))

    def test_malformed_reference_source_does_not_crash(self):
        from failure_classification.errorclassification.evidence.assessment import normalize_evidence_response
        raw = {'candidate_updates': [{'code': 'E13', 'evidence': [
            {'source': [], 'quote': 'invalid'}]}]}
        normalized, _ = normalize_evidence_response(raw, case())
        self.assertEqual(normalized, raw)

