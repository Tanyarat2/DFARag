"""Persist classification results without changing the source dataset."""
import json
from pathlib import Path

from failure_classification.errorclassification.reporting.result import ClassificationResult


def save_classification_result(result: ClassificationResult, path: str | Path) -> None:
    """Write one prediction and its audit as JSON, replacing the file atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    temporary.replace(destination)
