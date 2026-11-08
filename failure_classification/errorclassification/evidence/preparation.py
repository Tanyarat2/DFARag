"""Adapt stored inputs to the shared evidence-first classification contract."""
from typing import Any

from failure_classification.errorclassification.data.case_io import build_case_input

from failure_classification.errorclassification.data.input_record import InputRecord


def prepare_classification_evidence(record: InputRecord) -> dict[str, Any]:
    """Separate reference evidence from retrieved and generator evidence.

    The shared adapter evaluates rag_answer_final, removes prior labels, and
    records the limited scope and unverified historical provenance of passages.
    It never substitutes the reference answer for the evaluated answer.
    """
    return build_case_input(record.to_dict())
