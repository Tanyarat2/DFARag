import unittest
from unittest.mock import patch
from failure_classification.errorclassification.evidence.availability import evidence_availability_map
from failure_classification.errorclassification.validation.validator import EvidenceValidator


class EvidenceMapperTests(unittest.TestCase):
    def case(self):
        return {'original_question': 'Who was appointed in November?',
                'source_documents': [{'doc_id': '47', 'text': 'Michael Evans was appointed in November.',
                                      'reference_excerpts': ['Michael Evans was appointed in November.']}],
                'indexed_chunks': [{'chunk_id': '47_11', 'text': 'Michael Evans was appointed in November.'}],
                'retrieved_candidates': [{'chunk_id': '63_10', 'text': 'Emily Anderson joined in August.'}],
                'generation_context': None,
                'trace_status': {'retrieved_candidates': 'complete saved top-eight; historical equivalence unverified',
                                 'generation_context': 'missing'}}

    def test_exact_match_and_scoped_absence_preserve_semantic_unknown(self):
        result = evidence_availability_map(self.case())
        cells = result['facts'][0]['collections']
        self.assertEqual(cells['source_documents']['status'], 'present')
        self.assertEqual(cells['indexed_chunks']['status'], 'present')
        self.assertEqual(cells['retrieved_candidates']['status'], 'absent')
        self.assertFalse(cells['retrieved_candidates']['semantic_absence_established'])
        self.assertEqual(cells['retrieved_candidates']['semantic_availability'], 'unknown')
        self.assertEqual(cells['retrieved_candidates']['indexed_support_ids_not_observed'], ['47_11'])
        self.assertEqual(cells['generation_context']['status'], 'unknown')

    def test_partial_trace_is_not_observed_absence(self):
        case = self.case()
        case['trace_status']['retrieved_candidates'] = 'partial saved trace'
        cells = evidence_availability_map(case)['facts'][0]['collections']
        self.assertEqual(cells['retrieved_candidates']['status'], 'unknown')

    def test_semantic_paraphrase_requires_judgment_not_exact_text_guess(self):
        case = self.case()
        case['retrieved_candidates'] = [{'chunk_id': 'alternative', 'text': 'In November the new appointee was Michael Evans.'}]
        cell = evidence_availability_map(case)['facts'][0]['collections']['retrieved_candidates']
        self.assertEqual(cell['status'], 'absent')  # Only the exact reference span is absent.
        self.assertEqual(cell['semantic_availability'], 'unknown')
        self.assertFalse(cell['semantic_absence_established'])

    def test_required_fact_without_supporting_span_remains_visible_as_unknown(self):
        result = evidence_availability_map(self.case(), [{'fact_id': 'profit', 'requirement': 'Requested profit'}])
        self.assertEqual(len(result['facts']), 1)
        self.assertTrue(all(cell['status'] == 'unknown' for cell in result['facts'][0]['collections'].values()))

    def test_all_four_validators_execute_even_without_upstream_candidates(self):
        names = ['validate_chunking', 'validate_retrieval', 'validate_reranking', 'validate_generation']
        validator = EvidenceValidator(self.case())
        patches = [patch.object(validator, name, wraps=getattr(validator, name)) for name in names]
        mocks = [entry.start() for entry in patches]
        try:
            self.assertEqual(validator.validate_all_layers([]), {})
            for mock in mocks:
                mock.assert_called_once()
        finally:
            for entry in patches:
                entry.stop()


if __name__ == '__main__':
    unittest.main()
