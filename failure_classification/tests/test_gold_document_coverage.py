"""Gold-document coverage is independent of chunk recall and answer sufficiency."""
import unittest
from failure_classification.errorclassification.validation.validator import gold_document_coverage, reranking_document_metrics


class GoldDocumentCoverageTests(unittest.TestCase):
    def coverage(self, candidates, **extra):
        return gold_document_coverage({'source_documents': [{'doc_id': 'a'}, {'doc_id': 'b'}],
                                       'retrieved_candidates': candidates, **extra})

    def test_any_chunk_from_each_document_counts(self):
        result = self.coverage([{'chunk_id': 'a_9'}, {'chunk_id': 'b_7'}])
        self.assertEqual(result['recall'], 1)
        self.assertEqual(result['missing_document_ids'], [])

    def test_missing_document_and_duplicate_chunks(self):
        result = self.coverage([{'doc_id': 'a', 'chunk_id': 'x'}, {'doc_id': 'a', 'chunk_id': 'y'}])
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['missing_document_ids'], ['b'])

    def test_explicit_document_id_takes_precedence(self):
        self.assertEqual(self.coverage([{'doc_id': 'a', 'chunk_id': 'b_0'}])['missing_document_ids'], ['b'])

    def test_partial_trace_cannot_prove_missing_document(self):
        self.assertEqual(self.coverage([{'chunk_id': 'a_0'}],
            trace_status={'retrieved_candidates': 'partial'})['status'], 'unknown')

    def test_unknown_membership_cannot_prove_missing_document(self):
        self.assertEqual(self.coverage([{'chunk_id': 'unmapped'}])['status'], 'unknown')

    def test_empty_observed_retrieval_has_zero_recall(self):
        self.assertEqual(self.coverage([])['recall'], 0)

    def test_missing_retrieval_or_gold_is_unknown(self):
        self.assertEqual(self.coverage(None)['status'], 'unknown')
        self.assertEqual(gold_document_coverage({'source_documents': [],
            'retrieved_candidates': []})['status'], 'unknown')


class RerankingThresholdTests(unittest.TestCase):
    def metrics(self, retrieved, context, **extra):
        return reranking_document_metrics(dict(
            source_documents=[{'doc_id': d} for d in 'abcd'],
            retrieved_candidates=[{'doc_id': d} for d in retrieved],
            generation_context=[{'doc_id': d} for d in context], **extra))

    def test_loss_below_half_passes(self):
        self.assertFalse(self.metrics('abcd', 'abcxy')['e7_failure'])

    def test_exact_half_and_above_fail(self):
        self.assertTrue(self.metrics('abcd', 'abxyz')['e7_failure'])
        self.assertTrue(self.metrics('abcd', 'axyzz')['e7_failure'])

    def test_denominator_is_retrieved_gold_only(self):
        self.assertTrue(self.metrics('ab', 'axxyz')['e7_failure'])
        self.assertEqual(self.metrics('ab', 'axxyz')['lost_fraction'], .5)

    def test_duplicate_gold_chunks_do_not_satisfy_e8(self):
        self.assertTrue(self.metrics('ab', 'aaxby')['e8_failure'])

    def test_top_three_two_distinct_gold_pass(self):
        self.assertFalse(self.metrics('ab', 'axbyz')['e8_failure'])

    def test_no_retrieved_gold_has_undefined_e7(self):
        self.assertIsNone(self.metrics('xyz', 'xyzxy')['e7_failure'])
        self.assertTrue(self.metrics('xyz', 'xyzxy')['e8_failure'])

    def test_incomplete_reranking_is_unresolved(self):
        self.assertIsNone(self.metrics('ab', 'axy')['e8_failure'])
        result = self.metrics('ab', 'axyzz', trace_status={'generation_context': 'partial'})
        self.assertIsNone(result['e7_failure'])
        self.assertIsNone(result['e8_failure'])
