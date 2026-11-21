"""Readable entry point to the shared, evidence-first classification engine."""
from typing import Any, Protocol

from failure_classification.errorclassification.pipeline.classifier import FailureClassifier

from failure_classification.errorclassification.reporting.result import ClassificationResult
from failure_classification.errorclassification.evidence.preparation import prepare_classification_evidence
from failure_classification.errorclassification.data.input_record import InputRecord


class ClassificationModel(Protocol):
    """Model interface; callers can supply an API runner or an offline stub."""

    model: str

    def ask_json(self, **kwargs: Any) -> dict[str, Any]: ...


class EvidenceFirstFailureClassifier:
    """Classify all four stages using the existing prompts and evidence checks.

    The engine owns proposing, validating, and optionally reviewing findings.
    This class connects InputRecord to that workflow without copying its policy.
    """

    def __init__(
        self,
        model: ClassificationModel,
        *,
        max_passes: int = 2,
        max_failures: int = 1,
    ) -> None:
        self.model_name = model.model
        self.engine = FailureClassifier(
            model, max_passes=max_passes, max_failures=max_failures
        )

    def classify_record(self, record: InputRecord) -> ClassificationResult:
        """Evaluate one original answer and retain its prediction and audit."""
        evidence = prepare_classification_evidence(record)
        prediction, audit = self.engine.classify(evidence)
        return ClassificationResult(
            query_id=record.query_id,
            classifier_model=self.model_name,
            prediction=prediction,
            audit=audit,
        )
