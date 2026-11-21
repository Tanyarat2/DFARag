"""A classification result and its supporting audit for one input record."""
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ClassificationResult:
    query_id: str
    classifier_model: str
    prediction: dict[str, Any]
    audit: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Export a standalone JSON-serializable result."""
        return asdict(self)
