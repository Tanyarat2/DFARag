"""Isolate attribution regressions from the separately tested first-stage request."""
import unittest
from unittest.mock import patch
from failure_classification.errorclassification.answerability.assessor import finalize_assessment, unknown_assessment
from failure_classification.errorclassification.evidence.assessment import generation_evidence_catalog


def supplied_answerable_fixture(_assessor, case):
    catalog = generation_evidence_catalog(case)
    inputs = [key for key in catalog if key != 'answer']
    if not inputs:
        return unknown_assessment('No generator input supplied in this attribution fixture'), []
    record = finalize_assessment({'requirements': [{
        'requirement': case['original_question'], 'answerability': 'answerable',
        'reason': 'This attribution-only test supplies an answerable first-stage fixture.',
        'response_behavior': 'answered', 'acknowledged_limitation': False,
        'answer_assertion': case['original_rag_answer'], 'evidence': ['answer', inputs[0]]}]}, case)
    return record, []


class AttributionOnlyTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        patcher = patch('failure_classification.errorclassification.pipeline.classifier.AnswerabilityAssessor.assess',
                        supplied_answerable_fixture)
        patcher.start()
        self.addCleanup(patcher.stop)
