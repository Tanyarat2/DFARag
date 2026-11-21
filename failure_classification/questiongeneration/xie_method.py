from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_JSON_DATA = ROOT / "dataset/singleturn/single_turn_dragonball.json"
DEFAULT_CSV_DATA = ROOT / "dataset/singleturn/rerun_c2.csv"
DEFAULT_OUTPUT_DIR = ROOT / "result/question_generation/b1"
CACHE_PATH = PACKAGE_DIR / "cache/xie_method_cache.json"
MODEL = os.environ.get("XIE_MODEL", "gpt-4o-mini")
MAX_CASES = 377


# These prompt instructions mirror subq_generation.ipynb.
DECOMPOSITION_PROMPT = """Decompose the following complex question into a collection of around 20 sub-questions that you think would be relevant to answer the complex question fully.

Organize the answer in JSON-format, shaped as {"sub-questions": ["sub-question #1", "sub-question #2", ...]}.
Place the JSON-format answer between <answer> and </answer> tags.

Complex question: $QUESTION
Answer in JSON-format:
"""


# This prompt follows type_classification.ipynb.
TYPE_PROMPT = """Based on the sub-question’s relevance and functional role in answering the complex question, classify the sub-question into three types: core, background, and follow-up.

The definitions of these three sub-question types are:
(1) Core sub-questions:
 - They are central to the main topic and directly or partially address the complex question.
 - They are crucial for interpreting the logical reasoning of the complex question and provide essential insights required to answer the complex question.
 - They often involve multiple steps or perspectives, making them fundamental to generating a comprehensive and well-rounded response to the complex question.
(2) Background sub-questions:
 - They are optional when answering the complex question, but they can provide additional context or background information that helps clarify the complex question.
 - Their primary role is to support the understanding of the main topic by offering supplementary evidence or information, though it is not strictly necessary for addressing the core aspects of the complex question.
(3) Follow-up sub-questions:
 - They are not needed to answer the complex question.
 - They often arise after users receive an initial answer and seek further clarification or details.
 - They may explore specific aspects of the response in greater depth, but their answers can sometimes be out-of-scope or beyond the focus of the original question.

Organize the answer in JSON-format, shaped as {"type": "core"}, {"type": "background"}, or {"type": "follow-up"}.
Place the JSON-format answer between <answer> and </answer> tags.

Here are a few examples you can use for reference:

$FEW-SHOT-EXAMPLES

Complex question: $QUESTION
Its sub-question: $SUB-QUESTION
Answer in JSON-format:
"""


# The notebook's displayed schema has a punctuation typo; this retains its
# top-level JSON-list output and the same answered/unanswered judgments.
COVERAGE_PROMPT = """You are given a passage and a list of questions. For each question, determine if there exists any part of the passage that can answer the question.
If the question is answered, identify the text span that answers the question; if the question is unanswered, just return "None".

Organize your response in JSON-format, shaped as the following:

[
    {
        "question": QUESTION_1,
        "judgment": "answered",
        "text_span": "YOUR_IDENTIFIED_TEXT_SPAN"
    },
    {
        "question": QUESTION_2,
        "judgment": "unanswered",
        "text_span": "None"
    },
    ...
]

Place the JSON-format response between <answer> and </answer> tags.

Here are a few examples you can use for reference:

$FEW-SHOT-EXAMPLES

Passage: $PASSAGE
List of questions: $LIST_OF_QUESTIONS
Response in JSON-format:
"""


JSON_COLUMNS = {
    "reconstructed_retrieved_top8",
    "reconstructed_rerank_order",
    "reconstructed_reranked_top5",
    "gold_evidence",
}
ALLOWED_TYPES = {"core", "background", "follow-up"}


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parse_json_response(raw: str) -> Any:
    cleaned = raw.replace("<answer>", "").replace("</answer>", "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    return json.loads(cleaned)


def _coverage_list(value: Any) -> Any:
    """Keep the notebook's list response, while tolerating a JSON wrapper."""
    if isinstance(value, dict) and isinstance(value.get("response"), list):
        return value["response"]
    return value


def _is_decomposition(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("sub-questions"), list)
        and bool(value["sub-questions"])
        and all(isinstance(q, str) and q.strip() for q in value["sub-questions"])
    )


def _is_type(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") in ALLOWED_TYPES


def _is_coverage(value: Any, questions: list[str]) -> bool:
    items = _coverage_list(value)
    if not isinstance(items, list):
        return False
    by_question = {item.get("question"): item for item in items if isinstance(item, dict)}
    return (
        len(by_question) == len(questions)
        and all(question in by_question for question in questions)
        and all(
            by_question[question].get("judgment") in {"answered", "unanswered"}
            and isinstance(by_question[question].get("text_span"), str)
            for question in questions
        )
    )


def _majority_vote(values: list[Any]) -> str:
    labels = [str(value) for value in values if value is not None and str(value) != "nan"]
    if not labels:
        raise ValueError("Cannot choose a majority label from an empty annotation row")
    counts = Counter(labels)
    # np.unique in the notebook sorts labels first, then np.argmax chooses the
    # first maximum. Sorting by label reproduces that tie-breaking behavior.
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def load_few_shot_examples(
    type_workbook: Path = ROOT / "annotation_merged_v2.xlsx",
    coverage_workbook: Path = ROOT / "Answerability Annotation.xlsx",
) -> tuple[str, str]:
    """Load the same annotation examples used by the type and coverage notebooks."""
    import pandas as pd

    if not type_workbook.exists():
        raise FileNotFoundError(f"Type few-shot workbook not found: {type_workbook}")
    if not coverage_workbook.exists():
        raise FileNotFoundError(f"Coverage few-shot workbook not found: {coverage_workbook}")

    # Type-classification notebook examples, using four annotators' majority vote.
    type_sheets = pd.read_excel(type_workbook, sheet_name=None)
    type_examples = []
    for sheet_name in [f"q{i}" for i in range(1, 11)]:
        frame = type_sheets[sheet_name]
        question = str(frame["Question"].iloc[0])
        for _, row in frame.iterrows():
            label = _majority_vote(
                [row.get(name) for name in ("PERSON-1", "PERSON-2", "PERSON-3", "PERSON-4")]
            )
            example = (
                "Complex question: $QUESTION\nIts sub-question: $SUB-QUESTION\n"
                'Answer in JSON-format: {"type": "$LABEL"}'
            )
            type_examples.append(
                example.replace("$QUESTION", question)
                .replace("$SUB-QUESTION", str(row["Sub-Question"]))
                .replace("$LABEL", label)
            )

    # Coverage notebook examples: five articles, each annotated by two systems.
    coverage_sheets = pd.read_excel(coverage_workbook, sheet_name=None)
    coverage_examples = []
    for sheet_name in [f"Article{i}" for i in range(1, 6)]:
        frame = coverage_sheets[sheet_name]
        questions = frame["Sub-Question"].tolist()
        for system in ("You.com", "Perplexity.ai"):
            passage = str(frame[system].iloc[0]).replace("[", "").replace("]", "")
            judgments = []
            for _, row in frame.iterrows():
                question = row["Sub-Question"]
                judgment = str(row[f"{system}-judgment"]).strip()
                if judgment not in {"answered", "unanswered"}:
                    raise ValueError(f"Unexpected coverage label {judgment!r} in {sheet_name}/{system}")
                raw_passage = str(row[system])
                left, right = raw_passage.find("["), raw_passage.find("]")
                if judgment == "answered" and left >= 0 and right > left:
                    text_span = raw_passage[left + 1 : right]
                else:
                    text_span = "None"
                judgments.append(
                    {"question": question, "judgment": judgment, "text_span": text_span}
                )
            example = (
                "Passage: $PASSAGE\nList of questions: $LIST_OF_QUESTIONS\n"
                "Response in JSON-format: $RESPONSE_IN_JSON_FORMAT"
            )
            coverage_examples.append(
                example.replace("$PASSAGE", passage)
                .replace("$LIST_OF_QUESTIONS", json.dumps(questions, indent=4, ensure_ascii=False))
                .replace(
                    "$RESPONSE_IN_JSON_FORMAT",
                    json.dumps(judgments, indent=4, ensure_ascii=False),
                )
            )

    return "\n\n".join(type_examples), "\n\n".join(coverage_examples)


def load_dataset(path: Path, limit: int) -> list[dict[str, Any]]:
    """Load either the converted JSON records or the original CSV records."""
    if limit < 1 or limit > MAX_CASES:
        raise ValueError(f"limit must be between 1 and {MAX_CASES}")

    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON array of rows in {path}")
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            for column in JSON_COLUMNS:
                value = row.get(column)
                if isinstance(value, str) and value.strip():
                    row[column] = json.loads(value)
                elif value == "":
                    row[column] = None
    else:
        raise ValueError("Input must be a .json or .csv file")

    if len(rows) < limit:
        raise ValueError(f"Requested {limit} cases, but {path} contains only {len(rows)} rows")

    cases = []
    for index, row in enumerate(rows[:limit], start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} is not a JSON object")
        case_id = row.get("query_id")
        question = row.get("question")
        answer = row.get("original_rag_answer")
        docs = row.get("reconstructed_retrieved_top8")
        if case_id is None or not isinstance(question, str) or not question.strip():
            raise ValueError(f"Row {index} needs query_id and question")
        if docs is not None and not isinstance(docs, list):
            raise ValueError(f"Row {index}: reconstructed_retrieved_top8 must be a list or null")
        documents = []
        for doc_index, doc in enumerate(docs or []):
            if not isinstance(doc, dict) or not isinstance(doc.get("text"), str):
                raise ValueError(f"Row {index}: retrieved chunk {doc_index} needs a text field")
            documents.append(doc)
        cases.append(
            {
                "query_id": str(case_id),
                "question": question,
                "rag_answer": answer if isinstance(answer, str) else "",
                "retrieved_documents": documents,
            }
        )
    ids = [case["query_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Selected rows contain duplicate query_id values")
    return cases


class NotebookStyleRunner:
    """Direct chat calls with the notebooks' system message, temperature, and cache."""

    def __init__(self, model: str, type_examples: str, coverage_examples: str):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not loaded; source the project-root .envrc first")
        self.client = OpenAI(api_key=api_key, max_retries=2, timeout=120)
        self.model = model
        self.type_examples = type_examples
        self.coverage_examples = coverage_examples
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.cache = {}
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.api_calls = 0
        self.cache_hits = 0

    def _save_cache(self) -> None:
        temp_path = CACHE_PATH.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(CACHE_PATH)

    def ask_json(
        self,
        *,
        stage: str,
        prompt: str,
        validator: Callable[[Any], bool],
        max_completion_tokens: int,
    ) -> dict[str, Any]:
        cache_key = _hash_json(
            {"stage": stage, "model": self.model, "temperature": 0, "prompt": prompt}
        )
        cached = self.cache.get(cache_key)
        if cached and validator(cached.get("parsed")):
            self.cache_hits += 1
            return {
                **cached,
                "request_hash": cache_key,
                "cache_hit": True,
                "error": None,
            }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_completion_tokens=max_completion_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content or ""
            usage_obj = response.usage
            usage = {
                "prompt_tokens": int(usage_obj.prompt_tokens) if usage_obj else 0,
                "completion_tokens": int(usage_obj.completion_tokens) if usage_obj else 0,
                "total_tokens": int(usage_obj.total_tokens) if usage_obj else 0,
            }
            self.api_calls += 1
            for key in self.usage:
                self.usage[key] += usage[key]
            parsed = _parse_json_response(raw)
            if not validator(parsed):
                return {
                    "parsed": None,
                    "raw_excerpt": raw[:2500],
                    "usage": usage,
                    "request_hash": cache_key,
                    "cache_hit": False,
                    "error": "JSON was parsed but did not match this notebook stage's expected structure",
                }
            entry = {"parsed": parsed, "raw": raw, "usage": usage}
            self.cache[cache_key] = entry
            self._save_cache()
            return {
                **entry,
                "request_hash": cache_key,
                "cache_hit": False,
                "error": None,
            }
        except Exception as exc:
            # Do not include credential-like strings from exception messages in saved JSON.
            message = re.sub(r"sk-[^\s'\"},]+", "[REDACTED]", str(exc))[:500]
            return {
                "parsed": None,
                "raw_excerpt": None,
                "usage": {},
                "request_hash": cache_key,
                "cache_hit": False,
                "error": f"{type(exc).__name__}: {message}",
            }

    def decompose(self, question: str) -> dict[str, Any]:
        prompt = DECOMPOSITION_PROMPT.replace("$QUESTION", question)
        result = self.ask_json(
            stage="decomposition",
            prompt=prompt,
            validator=_is_decomposition,
            max_completion_tokens=2500,
        )
        result["sub_questions"] = (
            result["parsed"]["sub-questions"] if result["parsed"] is not None else []
        )
        return result

    def classify_type(self, question: str, sub_question: str) -> dict[str, Any]:
        prompt = (
            TYPE_PROMPT.replace("$QUESTION", question)
            .replace("$SUB-QUESTION", sub_question)
            .replace("$FEW-SHOT-EXAMPLES", self.type_examples)
        )
        result = self.ask_json(
            stage="subquestion_type",
            prompt=prompt,
            validator=_is_type,
            max_completion_tokens=128,
        )
        result["type"] = result["parsed"]["type"] if result["parsed"] is not None else "None"
        return result

    def measure_coverage(self, passage: str, questions: list[str]) -> dict[str, Any]:
        prompt = (
            COVERAGE_PROMPT.replace("$PASSAGE", passage)
            .replace("$LIST_OF_QUESTIONS", json.dumps(questions, indent=4, ensure_ascii=False))
            .replace("$FEW-SHOT-EXAMPLES", self.coverage_examples)
        )
        result = self.ask_json(
            stage="subquestion_coverage",
            prompt=prompt,
            validator=lambda value: _is_coverage(value, questions),
            max_completion_tokens=3500,
        )
        if result["parsed"] is not None:
            result["parsed"] = _coverage_list(result["parsed"])
        return result


def _judgment_map(judgments: Any, questions: list[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(judgments, list):
        return {}
    return {
        item["question"]: item
        for item in judgments
        if isinstance(item, dict) and item.get("question") in questions
    }


def _coverage_status(item: dict[str, Any] | None) -> str:
    if item is None:
        return "unknown"
    if item.get("judgment") == "answered":
        return "covered"
    if item.get("judgment") == "unanswered":
        return "uncovered"
    return "unknown"


def aggregate_retrieved_coverage(
    coverage_by_doc: dict[str, dict[str, Any]], questions: list[str]
) -> tuple[dict[str, bool], dict[str, str]]:
    """Return notebook-style booleans plus explicit covered/uncovered/unknown status."""
    boolean_coverage = {}
    status_by_question = {}
    for question in questions:
        per_doc_items = []
        any_answered = False
        for doc_result in coverage_by_doc.values():
            by_question = _judgment_map(doc_result.get("judgments"), questions)
            item = by_question.get(question)
            if item is not None:
                per_doc_items.append(item)
                any_answered = any_answered or item.get("judgment") == "answered"
        boolean_coverage[question] = any_answered
        if any_answered:
            status_by_question[question] = "covered"
        elif not coverage_by_doc or len(per_doc_items) < len(coverage_by_doc):
            status_by_question[question] = "unknown"
        elif all(item.get("judgment") == "unanswered" for item in per_doc_items):
            status_by_question[question] = "uncovered"
        else:
            status_by_question[question] = "unknown"
    return boolean_coverage, status_by_question


def _doc_output_key(doc: dict[str, Any], used: set[str]) -> str:
    doc_id = str(doc.get("doc_id") if doc.get("doc_id") is not None else "unknown_doc")
    if doc_id not in used:
        used.add(doc_id)
        return doc_id
    chunk_id = str(doc.get("chunk_id") if doc.get("chunk_id") is not None else len(used) + 1)
    key = f"{doc_id}::{chunk_id}"
    suffix = 2
    while key in used:
        key = f"{doc_id}::{chunk_id}::{suffix}"
        suffix += 1
    used.add(key)
    return key


def _run_coverage_for_case(
    case: dict[str, Any],
    typed_subquestions: list[dict[str, Any]],
    runner: NotebookStyleRunner,
) -> dict[str, Any]:
    question = case["question"]
    rag_answer = case["rag_answer"] or ""
    subquestions = [item["sub_question"] for item in typed_subquestions]
    errors = []

    # Like subq_coverage_measurement.ipynb, check the final RAG answer once.
    answer_result = runner.measure_coverage(rag_answer, subquestions) if rag_answer else {
        "parsed": None,
        "error": "original RAG answer is empty",
        "request_hash": None,
        "cache_hit": False,
    }
    answer_judgments = answer_result.get("parsed") or []
    if answer_result.get("error"):
        errors.append({"source": "original_rag_answer", "error": answer_result["error"]})
    answer_by_question = _judgment_map(answer_judgments, subquestions)
    answer_coverage = {
        subq: answer_by_question.get(subq, {}).get("judgment") == "answered"
        for subq in subquestions
    }
    answer_coverage_status = {
        subq: _coverage_status(answer_by_question.get(subq)) for subq in subquestions
    }

    # Like the notebook, send each retrieved passage independently.
    coverage_by_doc = {}
    used_keys: set[str] = set()
    for doc in case["retrieved_documents"]:
        doc_id = _doc_output_key(doc, used_keys)
        result = runner.measure_coverage(doc["text"], subquestions)
        judgments = result.get("parsed")
        if result.get("error"):
            errors.append({"source": f"retrieved_documents.{doc_id}", "error": result["error"]})
        record = {
            "rank": doc.get("rank"),
            "title": doc.get("title"),
            "judgments": judgments,
        }
        # These identifiers supplement the notebook's doc-level record because
        # this Dragonball CSV stores retrieved chunks rather than titled docs.
        if doc.get("doc_id") is not None:
            record["doc_id"] = str(doc["doc_id"])
        if doc.get("chunk_id") is not None:
            record["chunk_id"] = str(doc["chunk_id"])
        if result.get("error"):
            record["error"] = result["error"]
        coverage_by_doc[doc_id] = record

    # Preserve the notebook's boolean aggregation and add a separate tri-state
    # field so malformed/missing results do not have to be mistaken for false.
    retrieved_by_question, retrieved_status = aggregate_retrieved_coverage(
        coverage_by_doc, subquestions
    )

    return {
        "query_id": case["query_id"],
        "question": question,
        "rag_answer": rag_answer,
        "sub_questions_with_types": typed_subquestions,
        "answer_judgments": answer_judgments,
        "answer_coverage": answer_coverage,
        "answer_coverage_status": answer_coverage_status,
        "retrieved_coverage": retrieved_by_question,
        "retrieved_coverage_status": retrieved_status,
        "coverage_by_doc": coverage_by_doc,
        "coverage_errors": errors,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def run_xie_method(
    *,
    input_path: Path = DEFAULT_JSON_DATA if DEFAULT_JSON_DATA.exists() else DEFAULT_CSV_DATA,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit: int = 1,
    model: str = MODEL,
    type_workbook: Path = ROOT / "annotation_merged_v2.xlsx",
    coverage_workbook: Path = ROOT / "Answerability Annotation.xlsx",
) -> dict[str, Any]:
    """Run generation, type classification, then answer/document coverage."""
    cases = load_dataset(input_path, limit)
    type_examples, coverage_examples = load_few_shot_examples(type_workbook, coverage_workbook)
    runner = NotebookStyleRunner(model, type_examples, coverage_examples)
    output_dir.mkdir(parents=True, exist_ok=True)

    decompositions = []
    typed_dataset = []
    coverage_results = []
    all_errors = []

    for index, case in enumerate(cases, start=1):
        print(f"Xie case {index}/{len(cases)} ({case['query_id']}): sub-question generation", flush=True)
        decomposition = runner.decompose(case["question"])
        questions = decomposition["sub_questions"]
        if decomposition.get("error"):
            all_errors.append({"query_id": case["query_id"], "stage": "decomposition", "error": decomposition["error"]})
        decomposition_record = {"query_id": case["query_id"], "question": case["question"], "sub_questions": questions}
        decompositions.append(decomposition_record)
        _write_json(output_dir / "decompositions.json", decompositions)

        typed_subquestions = []
        for sub_index, sub_question in enumerate(questions, start=1):
            print(
                f"  Type classification {sub_index}/{len(questions)}",
                flush=True,
            )
            result = runner.classify_type(case["question"], sub_question)
            typed_subquestions.append({"sub_question": sub_question, "type": result["type"]})
            if result.get("error"):
                all_errors.append(
                    {
                        "query_id": case["query_id"],
                        "stage": "type_classification",
                        "sub_question": sub_question,
                        "error": result["error"],
                    }
                )

        typed_record = {
            "query_id": case["query_id"],
            "question": case["question"],
            "sub_questions_with_types": typed_subquestions,
        }
        typed_dataset.append(typed_record)
        _write_json(output_dir / "decompositions_with_type.json", typed_dataset)

        print(f"  Coverage: original answer and {len(case['retrieved_documents'])} retrieved chunks", flush=True)
        coverage_record = _run_coverage_for_case(case, typed_subquestions, runner)
        coverage_results.append(coverage_record)
        all_errors.extend(
            {"query_id": case["query_id"], "stage": "coverage", **error}
            for error in coverage_record["coverage_errors"]
        )
        _write_json(output_dir / "coverage_results.json", coverage_results)

    summary = {
        "run_mode": "notebook_style_xie_method",
        "input_file": str(input_path.resolve()),
        "selected_query_ids": [case["query_id"] for case in cases],
        "model": model,
        "temperature": 0,
        "calls_made_this_run": runner.api_calls,
        "cache_hits_this_run": runner.cache_hits,
        "api_token_usage_this_run": runner.usage,
        "error_count": len(all_errors),
        "errors": all_errors,
        "outputs": {
            "decompositions": str((output_dir / "decompositions.json").resolve()),
            "typed_decompositions": str((output_dir / "decompositions_with_type.json").resolve()),
            "coverage": str((output_dir / "coverage_results.json").resolve()),
        },
        "notes": [
            "Prompts, few-shot sources, and per-passage checks follow the three project notebooks.",
            "Retrieved passages come from reconstructed_retrieved_top8 in the selected dataset row.",
            "The notebook model is gpt-4-turbo; this run uses the configured model and defaults to gpt-4o-mini.",
            "answer_coverage and retrieved_coverage preserve the notebook's boolean outputs; the *_status fields retain unknown separately.",
        ],
    }
    _write_json(output_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_JSON_DATA if DEFAULT_JSON_DATA.exists() else DEFAULT_CSV_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=1, help="Cases to run (1-377; defaults to 1).")
    parser.add_argument("--model", default=MODEL, help="OpenAI model; defaults to XIE_MODEL or gpt-4o-mini.")
    parser.add_argument("--type-workbook", type=Path, default=ROOT / "annotation_merged_v2.xlsx")
    parser.add_argument("--coverage-workbook", type=Path, default=ROOT / "Answerability Annotation.xlsx")
    args = parser.parse_args()
    print(json.dumps(run_xie_method(input_path=args.input, output_dir=args.output_dir, limit=args.limit, model=args.model, type_workbook=args.type_workbook, coverage_workbook=args.coverage_workbook), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
