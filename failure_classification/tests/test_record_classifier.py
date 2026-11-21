"""Offline checks of input isolation, evidence rejection, and result persistence."""
import json
import tempfile
import unittest
from failure_classification.tests.answerability_fixtures import AttributionOnlyTestCase
from pathlib import Path

from failure_classification.errorclassification import (
    EvidenceFirstFailureClassifier,
    InputRecord,
    save_classification_result,
)
from failure_classification.errorclassification.evidence.preparation import (
    prepare_classification_evidence,
)


class OfflineModel:
    model = 'offline-test'

    def __init__(self, response):
        self.response = response
        self.prompts = []

    def ask_json(self, **kwargs):
        self.prompts.append(kwargs['prompt'])
        return {'parsed': self.response, 'usage': {}}


class FailureClassifierTests(AttributionOnlyTestCase):
    def setUp(self):
        super().setUp()
        self.record = InputRecord(
            query_id='test-1', question='What is the revenue?', answer='$100 million',
            query_type='Factual', rag_answer='Original full response',
            retrieved=[{'chunk_id': '1_0', 'text': 'Revenue was $100 million.'}],
            rerank_order=[0],
            reranked=[{'chunk_id': '1_0', 'text': 'Revenue was $100 million.'}],
            generated={}, gold_documents=[], indexed_reference_chunks=[],
            gold_document_provenance={}, gold_supporting_excerpt_matches=[],
            rag_answer_final='$200 million',
        )

    def test_prepare_evidence_keeps_answer_and_reference_separate(self):
        case = prepare_classification_evidence(self.record)
        self.assertEqual(case['original_rag_answer'], '$200 million')
        self.assertEqual(case['gold_answer'], '$100 million')
        case['generation_context'][0]['text'] = 'changed'
        self.assertEqual(self.record.reranked[0]['text'], 'Revenue was $100 million.')
        source = self.record.to_dict()
        source.update(error_stage='LEAKED_STAGE', error_category='LEAKED_LABEL')
        clean = prepare_classification_evidence(InputRecord.from_dict(source))
        self.assertNotIn('LEAKED', json.dumps(clean))

    def test_reject_fabricated_citation_without_forcing_a_label(self):
        model = OfflineModel({
            'answer_defect': {'status': 'defect', 'summary': 'Wrong revenue'},
            'candidate_failures': [{
                'code': 'E13', 'status': 'supported',
                'evidence': [{
                    'source': 'generation_context', 'item_id': 'nonexistent',
                    'quote': 'Invented support', 'observation': 'Claims misuse',
                }],
            }],
            'primary_failure': 'E13', 'primary_reason': 'Misread revenue',
            'attribution_status': 'established',
        })
        result = EvidenceFirstFailureClassifier(model, max_passes=1).classify_record(self.record)
        self.assertEqual(result.prediction['failures'], [])
        self.assertIsNone(result.prediction['primary_failure'])
        self.assertEqual(result.prediction['label_assessments']['E13']['status'], 'rejected')
        self.assertEqual(result.prediction['final_attribution']['status'], 'unresolved')
        self.assertEqual(len(model.prompts), 1)

    def test_supported_primary_and_saved_audit(self):
        model = OfflineModel({
            'answer_defect': {'status': 'defect', 'summary': 'Revenue is doubled'},
            'candidate_failures': [{
                'code': 'E16', 'status': 'supported',
                'key_points': ['The supplied quantity is 100, but the answer gives 200.'],
                'evidence': [{
                    'source': 'generation_context', 'item_id': '1_0',
                    'quote': 'Revenue was $100 million.', 'observation': 'Original quantity',
                }, {
                    'source': 'original_rag_answer', 'item_id': None,
                    'quote': '$200 million', 'observation': 'Incorrect quantity',
                }],
            }],
            'primary_failure': 'E16', 'primary_reason': 'Incorrectly represented supplied revenue',
            'primary_evidence': [{
                'source': 'original_rag_answer', 'item_id': None,
                'quote': '$200 million', 'observation': 'Affected answer',
            }],
            'attribution_status': 'established',
        })
        result = EvidenceFirstFailureClassifier(model).classify_record(self.record)
        self.assertEqual(result.prediction['failures'], ['E16'])
        self.assertEqual(set(result.prediction['stages']), {'Chunking', 'Retrieval', 'Reranking', 'Generation'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested/result.json'
            save_classification_result(result, path)
            restored = json.loads(path.read_text())
        self.assertEqual(restored['classifier_model'], 'offline-test')
        self.assertEqual(restored['prediction']['primary_failure'], 'E16')
        self.assertEqual(len(restored['audit']), 2)
        self.assertEqual(restored['prediction']['generation_review']['status'], 'needs_review')


if __name__ == '__main__':
    unittest.main()
