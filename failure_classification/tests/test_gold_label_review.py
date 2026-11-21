"""Gold annotation review is post-hoc and distinct from answer support."""
import unittest
from failure_classification.experiments.review_pilot_gold_labels import (
    review_catalog, review_prompt, validate_review, threshold_review,
)

CASE = {'case_id': 'gold-review', 'original_question': 'What was cash flow?',
        'original_rag_answer': '$50 million', 'gold_answer': 'Unable to answer',
        'source_documents': [{'doc_id': 'a'}],
        'retrieved_candidates': [{'chunk_id': 'a_0', 'text': 'Net profit was $50 million.'}],
        'generation_context': [{'chunk_id': 'a_0', 'text': 'Net profit was $50 million.'}]}


class GoldLabelReviewTests(unittest.TestCase):
    def test_applicability_means_failure_support(self):
        for applicability, verdict in [('applies', 'supported'), ('does_not_apply', 'questionable'),
                                       ('unknown', 'insufficient_evidence')]:
            with self.subTest(applicability=applicability):
                raw = {'label_applicability': applicability, 'reason': 'Specific evidence reason',
                       'suggested_failure_type': None, 'evidence': ['answer', 'context_1']}
                result, errors = validate_review(raw, CASE, review_catalog(CASE))
                self.assertEqual(errors, [])
                self.assertEqual(result['verdict'], verdict)

    def test_conflicting_failure_verdict_is_not_silently_accepted(self):
        result, errors = validate_review({'label_applicability': 'does_not_apply', 'verdict': 'supported',
            'reason': 'The answer is supported', 'evidence': ['answer']}, CASE, review_catalog(CASE))
        self.assertTrue(errors)

    def test_unknown_reference_cannot_support_gold_label(self):
        _, errors = validate_review({'verdict': 'supported', 'reason': 'Evidence',
            'evidence': ['context_999']}, CASE, review_catalog(CASE))
        self.assertTrue(errors)

    def test_correct_reference_resolves_actual_text(self):
        result, errors = validate_review({'verdict': 'questionable', 'reason': 'Evidence',
            'evidence': ['context_1']}, CASE, review_catalog(CASE))
        self.assertFalse(errors)
        self.assertEqual(result['evidence'][0]['quote'], CASE['generation_context'][0]['text'])

    def test_threshold_review_does_not_require_a_model_prediction(self):
        self.assertEqual(threshold_review(CASE, 'E4')['verdict'], 'questionable')
        missed = dict(CASE, retrieved_candidates=[])
        self.assertEqual(threshold_review(missed, 'E4')['verdict'], 'supported')

    def test_prompt_identifies_label_support_target_and_updated_e9(self):
        prompt = review_prompt(CASE, 'E9', review_catalog(CASE))
        self.assertIn('DOES THE ORIGINAL RAG RESPONSE COMMIT THE SPECIFIED FAILURE', prompt)
        self.assertIn('No proof of factual falsity is required', prompt)
        self.assertNotIn('predictions_by_query', prompt)
