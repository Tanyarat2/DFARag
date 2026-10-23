"""Source provenance is matched to the exact generation input, not a boolean."""
import unittest
from make_annotation import generation_evidence


class GenerationEvidenceTests(unittest.TestCase):
    def test_selected_passage_ids_and_boolean_stay_distinct_from_retrieval_pool(self):
        turn = {'question': 'Question?', 'answer': 'Answer.', 'retrieved': True,
                'passages': [{'id': 'doc-1', 'title': 'Title', 'text': 'Evidence.'}]}
        traces = {('hipporag2', 'case', '1'): [turn]}
        result = generation_evidence(traces, 'hipporag2', 'case', '1',
                                     'Question?', 'Answer.', ['Evidence.'])
        self.assertEqual(result['selected_passages'][0]['id'], 'doc-1')
        self.assertIs(result['retrieved_flag'], True)
        self.assertIsNone(result['retrieved_candidates'])
        self.assertIsNone(result['reranking_trace'])

    def test_changed_source_cannot_attach_stale_ids_or_retrieval_flags(self):
        traces = {('hipporag2', 'case', '1'): [{'question': 'Question?', 'answer': 'Old answer.',
                   'retrieved': True, 'passages': [{'id': 'old', 'text': 'Evidence.'}]}]}
        result = generation_evidence(traces, 'hipporag2', 'case', '1',
                                     'Question?', 'New answer.', ['Evidence.'])
        self.assertEqual(result['selected_passages'][0]['id'], '')
        self.assertIsNone(result['retrieved_flag'])

    def test_no_source_keeps_generation_text_without_inventing_ids(self):
        result = generation_evidence({}, 'other', 'case', '1', 'Q', 'A', ['Context'])
        self.assertEqual(result['selected_passages'], [{'id': '', 'title': '', 'text': 'Context'}])
        self.assertIsNone(result['retrieved_candidates'])
