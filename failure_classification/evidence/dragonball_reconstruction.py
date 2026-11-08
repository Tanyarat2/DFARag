"""Supply scoped Dragonball evidence under an explicit reconstruction assumption.

This module reconstructs word chunks from local data. It does not rerun models,
recover historical traces, or read human failure annotations.
"""

from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RAG_ROOT = ROOT / "rag-error-classification"
CONFIG_PATH = RAG_ROOT / "conf/config-dragonball.yaml"


def reconstruct_document_chunks(
    document: dict[str, Any], chunk_size: int, overlap_size: int
) -> list[dict[str, Any]]:
    """Match FixedLengthChunker; offsets count words, with an exclusive end."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("Dragonball chunk_size must be a positive integer")
    if type(overlap_size) is not int or not 0 <= overlap_size < chunk_size:
        raise ValueError("Dragonball overlap_size must be between zero and chunk_size - 1")
    doc_id = document.get("doc_id", document.get("id"))
    if doc_id is None or not isinstance(document.get("text"), str):
        raise ValueError("A reconstructed document requires doc_id/id and string text")
    doc_id = str(doc_id)
    words = document["text"].split()
    return [
        {
            "chunk_id": f"{doc_id}_{number}", "doc_id": doc_id,
            "para_id": number, "title": document.get("title", ""),
            "text": " ".join(words[start:start + chunk_size]),
            "word_start": start, "word_end": min(start + chunk_size, len(words)),
        }
        for number, start in enumerate(range(0, len(words), chunk_size - overlap_size))
    ]


def _read_jsonl(path: Path, key: str) -> tuple[dict[str, dict[str, Any]], str]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read Dragonball reconstruction input {path}: {exc}") from exc
    rows: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{number}: {exc}") from exc
        if not isinstance(row, dict) or row.get(key) is None:
            raise ValueError(f"Missing {key!r} in {path}:{number}")
        row_id = str(row[key])
        if row_id in rows:
            raise ValueError(f"Duplicate {key} {row_id!r} in {path}:{number}")
        rows[row_id] = row
    return rows, hashlib.sha256(content).hexdigest()


def _configured_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or "${" in value:
        raise ValueError(f"Dragonball config {name} must be a resolved local path")
    return (RAG_ROOT / value).resolve()


@lru_cache(maxsize=1)
def _load_reconstruction_basis() -> dict[str, Any]:
    """Cache local snapshots and their deterministic, unranked chunk inventory."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Dragonball reconstruction needs PyYAML in the project environment") from exc
    try:
        config_bytes = CONFIG_PATH.read_bytes()
        config = yaml.safe_load(config_bytes)
        rag = config["rag"]
        data = rag["data"]
        chunk_size = data["corpus_chunk_size"]
        overlap_size = data["corpus_overlap_size"]
        retrieve_topk = rag["retrieval"]["retrieve_topk"]
        rerank_topk = rag["rerank"]["rerank_topk"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load complete Dragonball reconstruction config {CONFIG_PATH}: {exc}") from exc
    if data.get("corpus_name") != "dragonball" or data.get("query_name") != "dragonball":
        raise ValueError("Reconstruction config must name the Dragonball corpus and queries")
    if data.get("chunking_strategy") != "fixed length" or data.get("load_semantic_chunks") is not False:
        raise ValueError("Dragonball reconstruction only supports configured fixed-length word chunks")
    if (type(retrieve_topk) is not int or retrieve_topk <= 0
            or type(rerank_topk) is not int or not 0 < rerank_topk <= retrieve_topk):
        raise ValueError("Dragonball retrieval/rerank top-k values must be valid positive integers")
    # Validate the chunk parameters even if the source corpus is empty.
    reconstruct_document_chunks({"doc_id": "validation", "text": ""}, chunk_size, overlap_size)
    corpus_path = _configured_path(data.get("corpus_path"), "corpus_path")
    query_path = _configured_path(data.get("query_path"), "query_path")
    corpus, corpus_sha256 = _read_jsonl(corpus_path, "doc_id")
    queries, query_sha256 = _read_jsonl(query_path, "id")
    chunks_by_doc = {
        doc_id: reconstruct_document_chunks(document, chunk_size, overlap_size)
        for doc_id, document in corpus.items()
    }
    chunks_by_id = {chunk["chunk_id"]: chunk for chunks in chunks_by_doc.values() for chunk in chunks}
    model_names = {}
    for stage in ("embedding", "rerank", "generator"):
        model = rag.get(stage, {}).get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"Dragonball config requires a model name for {stage}")
        model_names[stage] = model
    return {
        "corpus": corpus, "queries": queries, "chunks_by_doc": chunks_by_doc,
        "chunks_by_id": chunks_by_id, "chunk_size": chunk_size, "overlap_size": overlap_size,
        "retrieve_topk": retrieve_topk, "rerank_topk": rerank_topk,
        "model_names": model_names,
        "files": {
            "corpus": {"path": str(corpus_path), "sha256": corpus_sha256},
            "queries": {"path": str(query_path), "sha256": query_sha256},
            "config": {"path": str(CONFIG_PATH), "sha256": hashlib.sha256(config_bytes).hexdigest()},
        },
    }


def _validate_collection(
    items: Any, name: str, maximum: int, chunks_by_id: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]] | None:
    if items is None:
        return None
    if not isinstance(items, list) or len(items) > maximum:
        raise ValueError(f"{name} must be a list of at most {maximum} configured chunks")
    result = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("chunk_id"), str):
            raise ValueError(f"Every {name} entry requires a string chunk_id")
        chunk_id = item["chunk_id"]
        if chunk_id in result:
            raise ValueError(f"Duplicate {name} chunk_id {chunk_id!r}")
        expected = chunks_by_id.get(chunk_id)
        if expected is None:
            raise ValueError(f"{name} chunk {chunk_id!r} is absent from the configured reconstructed index")
        if item.get("text") != expected["text"]:
            raise ValueError(f"{name} text for {chunk_id!r} contradicts configured reconstructed chunk text")
        if str(item.get("doc_id")) != expected["doc_id"]:
            raise ValueError(f"{name} document ID for {chunk_id!r} contradicts its reconstructed source")
        result[chunk_id] = item
    return result


def augment_case_input(case_input: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with scoped evidence and the declared reconstruction policy.

    Complete scopes below mean complete for the reconstruction assumption. They
    are never claims that original historical execution logs have been recovered.
    """
    result = copy.deepcopy(case_input)
    # Target annotations belong in evaluation records, never classifier evidence.
    for field in ("original_error_category", "original_error_stage", "gold_category",
                  "gold_category_raw", "gold_stage_raw"):
        result.pop(field, None)
    basis = _load_reconstruction_basis()
    case_id = str(result.get("case_id", ""))
    query = basis["queries"].get(case_id)
    if query is None:
        raise ValueError(f"Dragonball query {case_id!r} is absent from the configured query snapshot")
    if query.get("input") != result.get("original_question"):
        raise ValueError(f"Dragonball query {case_id!r} contradicts the supplied original question")
    reference = result.get("gold_evidence")
    if not isinstance(reference, dict) or reference.get("source_dataset") != "Dragonball":
        raise ValueError("Dragonball reconstruction requires identified Dragonball gold_evidence")
    doc_ids = reference.get("doc_ids")
    if not isinstance(doc_ids, list):
        raise ValueError("Dragonball gold_evidence.doc_ids must be a list")
    doc_ids = [str(doc_id) for doc_id in doc_ids]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Dragonball reference document IDs must be unique")
    try:
        query_doc_ids = query["output"]["provenance"][0]["id"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Dragonball query {case_id!r} lacks reference document provenance") from exc
    if not isinstance(query_doc_ids, list) or set(doc_ids) != {str(d) for d in query_doc_ids}:
        raise ValueError(f"Dragonball reference IDs for {case_id!r} contradict the configured query snapshot")
    supplied_sources = result.get("gold_source_documents") or []
    if not isinstance(supplied_sources, list):
        raise ValueError("gold_source_documents must be a list when supplied")
    supplied_by_id = {}
    for source in supplied_sources:
        if not isinstance(source, dict) or source.get("doc_id") is None:
            raise ValueError("Every gold source requires a document ID")
        doc_id = str(source["doc_id"])
        if doc_id not in doc_ids or doc_id in supplied_by_id:
            raise ValueError(f"Unexpected or duplicate gold source document {doc_id!r}")
        supplied_by_id[doc_id] = source
    source_documents = []
    indexed_chunks = []
    for doc_id in doc_ids:
        source = basis["corpus"].get(doc_id)
        if source is None:
            raise ValueError(f"Dragonball reference document {doc_id!r} is absent from the configured corpus")
        supplied = supplied_by_id.get(doc_id)
        if supplied is not None and supplied.get("text") != source["text"]:
            raise ValueError(f"Gold source text for {doc_id!r} contradicts the configured corpus snapshot")
        source_documents.append(copy.deepcopy(supplied) if supplied is not None else {
            "doc_id": source["doc_id"], "source_dataset": "Dragonball",
            "doc_title": source.get("doc_title"), "metadata": copy.deepcopy(source.get("metadata")),
            "text": source["text"],
        })
        indexed_chunks.extend(copy.deepcopy(basis["chunks_by_doc"][doc_id]))
    candidates = _validate_collection(result.get("retrieved_candidates"), "retrieved_candidates",
                                      basis["retrieve_topk"], basis["chunks_by_id"])
    context = _validate_collection(result.get("generation_context"), "generation_context",
                                   basis["rerank_topk"], basis["chunks_by_id"])
    if context is not None:
        if candidates is None:
            raise ValueError("Cannot validate reconstructed context against missing retrieval candidates")
        for chunk_id, item in context.items():
            if chunk_id not in candidates or item["text"] != candidates[chunk_id]["text"]:
                raise ValueError(f"Generation context chunk {chunk_id!r} contradicts its supplied candidate list")
    result["source_documents"] = source_documents
    result["indexed_chunks"] = indexed_chunks
    trace_status = result.setdefault("trace_status", {})
    trace_status["source_documents"] = "complete (reconstructed reference documents only)"
    trace_status["indexed_chunks"] = "complete (reconstructed chunks of reference documents only)"
    for name, items, maximum in (("retrieved_candidates", candidates, basis["retrieve_topk"]),
                                  ("generation_context", context, basis["rerank_topk"])):
        trace_status[name] = (
            "missing" if items is None else
            f"{'complete' if len(items) == maximum else 'partial'} (supplied reconstructed top-{maximum}; "
            f"{len(items)} observed; historical execution unverified)"
        )
    result["dataset_question_metadata"] = {
        "query_id": case_id, "metadata": copy.deepcopy(query.get("metadata", {})),
        "ambiguous": query.get("metadata", {}).get("ambiguous"),
        "source": copy.deepcopy(basis["files"]["queries"]),
        "ambiguity_provenance": (
            "Original processed-query metadata; preprocessing derives the ambiguity flag from "
            "data/Dragonball/data/ambiguous_queries_by_GPT.csv. This is a GPT-derived flag, "
            "not a human failure annotation or a substitute for examining query scope."
        ),
    }
    metadata = result.setdefault("pipeline_metadata", {})
    metadata["reconstruction_policy"] = {
        "assumption": (
            "For conditional attribution, assume the original RAG used the same processed corpus "
            "and configured full-corpus word index as this local reconstruction. Assume the supplied "
            "top-k retrieval/context reconstructions represent that pipeline. These assumptions "
            "are not recovered historical execution evidence."
        ),
        "scope": "conditional reconstruction, not original historical run logs",
        "files": copy.deepcopy(basis["files"]),
        "chunking": {"strategy": "fixed length", "chunk_size_words": basis["chunk_size"],
                     "overlap_words": basis["overlap_size"], "offset_unit": "word",
                     "word_end_exclusive": True},
        "configured_model_names": copy.deepcopy(basis["model_names"]),
        "retrieve_topk": basis["retrieve_topk"], "rerank_topk": basis["rerank_topk"],
        "full_corpus_document_count": len(basis["corpus"]),
        "full_reconstructed_index_chunk_count": len(basis["chunks_by_id"]),
        "source_documents_scope": {"doc_ids": doc_ids, "reference_documents_only": True},
        "indexed_chunks_scope": {"doc_ids": doc_ids, "reference_documents_only": True,
                                 "chunk_count": len(indexed_chunks)},
        "validation": {
            "candidate_texts_match_configured_word_chunks": candidates is not None,
            "context_is_subset_of_candidates_with_matching_text": context is not None,
            "query_text_and_reference_ids_match_processed_query": True,
        },
        "limitations": [
            "Supplied retrieval ranks, scores, and rerank order are preserved; no embeddings, "
            "retrieval, reranking, or generation were recomputed or verified against configured models.",
            "Only reference-document raw text and chunks are supplied to the classifier. Their absence "
            "of another fact cannot establish corpus-wide unanswerability or absence from the full index.",
            "Configuration and file hashes identify the local reconstruction snapshot; they do not "
            "prove the historical failed run used that snapshot or the same runtime overrides.",
        ],
    }
    return result
