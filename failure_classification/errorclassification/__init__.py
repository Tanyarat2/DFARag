"""Classify stored RAG records and conversations with one shared engine."""
from failure_classification.errorclassification.data.models import Conversation, ConversationTurn
from failure_classification.errorclassification.pipeline.classifier import FailureClassifier
from failure_classification.errorclassification.data.input_record import InputRecord
from failure_classification.errorclassification.reporting.result import ClassificationResult
from failure_classification.errorclassification.pipeline.record_classifier import EvidenceFirstFailureClassifier
from failure_classification.errorclassification.reporting.storage import save_classification_result

__all__ = [
    "Conversation", "ConversationTurn", "FailureClassifier", "InputRecord",
    "ClassificationResult", "EvidenceFirstFailureClassifier", "save_classification_result",
]
