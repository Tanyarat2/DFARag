"""Input records for the new error-classification pipeline.

Example::

    records = InputRecord.load_jsonl()
    print(records[0].question)
    print(records[0].rag_answer_final)
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_INPUT_PATH = (
    Path(__file__).resolve().parents[3]
    / "dataset/singleturn/annotations_with_passages_final_answer.jsonl"
)


@dataclass
class InputRecord:
    """Store one annotated query, its answers, passages, and provenance.

    Nested JSON structures remain dictionaries and lists so that all passage
    metadata and generation variants are retained without changing their schema.
    ``answer`` is the reference answer; ``rag_answer`` and ``rag_answer_final``
    are stored separately, exactly as supplied by the dataset.
    """

    query_id: str
    question: str
    answer: str
    query_type: str
    rag_answer: str
    retrieved: list[dict[str, Any]]
    rerank_order: list[int]
    reranked: list[dict[str, Any]]
    generated: dict[str, Any]
    gold_documents: list[dict[str, Any]]
    indexed_reference_chunks: list[dict[str, Any]]
    gold_document_provenance: dict[str, Any]
    gold_supporting_excerpt_matches: list[dict[str, Any]]
    rag_answer_final: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InputRecord:
        """Create a record, excluding the source's error-classification labels."""
        values = deepcopy(dict(data))
        values.pop("error_stage", None)
        values.pop("error_category", None)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of this record."""
        return asdict(self)

    @classmethod
    def load_jsonl(cls, path: str | Path = DEFAULT_INPUT_PATH) -> list[InputRecord]:
        """Load nonblank JSONL rows, reporting the source line on errors."""
        records = []
        path = Path(path)
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        raise ValueError("expected a JSON object")
                    records.append(cls.from_dict(data))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        return records

    @staticmethod
    def save_jsonl(records: Iterable[InputRecord], path: str | Path) -> None:
        """Write records to a UTF-8 JSONL file."""
        with Path(path).open("w", encoding="utf-8") as destination:
            for record in records:
                destination.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
