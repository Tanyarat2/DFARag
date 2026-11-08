"""Load DragonBall gold-source documents linked to a query's references."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DRAGONBALL_CORPUS_PATH = (
    ROOT
    / "rag-error-classification/data/Dragonball/data/dragonball_corpus_en_processed.jsonl"
)


@lru_cache(maxsize=1)
def _dragonball_documents() -> dict[str, dict[str, Any]]:
    """Read the English DragonBall corpus once, indexed by source document ID."""
    documents: dict[str, dict[str, Any]] = {}
    with DRAGONBALL_CORPUS_PATH.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            documents[str(row["doc_id"])] = row
    return documents


def load_gold_source_documents(reference_evidence: Any) -> list[dict[str, Any]]:
    """Return full DragonBall documents named by gold-reference document IDs.

    These are reference documents for checking source-level answer support. They
    do not establish that the original RAG system indexed or retrieved them.
    """
    if not isinstance(reference_evidence, dict):
        return []
    if reference_evidence.get("source_dataset") != "Dragonball":
        return []

    doc_ids = reference_evidence.get("doc_ids") or []
    if not isinstance(doc_ids, list):
        raise ValueError("DragonBall gold_evidence.doc_ids must be a list")

    corpus = _dragonball_documents()
    result = []
    for doc_id in doc_ids:
        document = corpus.get(str(doc_id))
        if document is None:
            raise ValueError(
                f"DragonBall gold document {doc_id!r} was not found in "
                f"{DRAGONBALL_CORPUS_PATH}"
            )
        result.append(
            {
                "doc_id": doc_id,
                "source_dataset": "Dragonball",
                "doc_title": document.get("doc_title"),
                "metadata": document.get("metadata"),
                "text": document["text"],
            }
        )
    return result
