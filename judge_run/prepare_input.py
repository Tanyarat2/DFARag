#!/usr/bin/env python3
"""conversation.json (nested 'input' format) -> work/input.json (format read by the judges).

Keeps only what the judges need: id, conversation, ground_truth_answers, and contexts
(the selected passages the RAG generator actually saw). Per-turn answerability and
exact histories are preserved when recorded in input, or otherwise in metadata.
Missing evidence stays unknown; a recorded empty passage list stays empty.

    python judge_run/prepare_input.py [--src conversation.json] [--out work] [--limit N]

A RAG export (top-level "conversations" with per-turn question/answer/passages, e.g.
conversation_raptor.json) is also accepted; it has no gold answers, so they are read from
the official MTRAG reference tasks (--reference) by task_id. When the original
reference file is unavailable, the local gold_support.json index is used.
"""

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE = ROOT / "mtrag_three_rag" / "_data" / "reference.jsonl"
GOLD_SUPPORT = pathlib.Path(__file__).resolve().with_name("gold_support.json")

ANSWERABILITY_VALUES = {"ANSWERABLE", "UNANSWERABLE", "PARTIAL", "CONVERSATIONAL",
                        "UNKNOWN", "YES", "NO"}


def selected_contexts(metadata, turns, record_id):
    """Extract generator context only; never substitute pre-selection retrieval."""
    retrieved = metadata.get("retrieved_chunks_by_turn")
    if retrieved is None:
        return None
    if not isinstance(retrieved, list) or len(retrieved) != turns:
        raise ValueError("retrieved_chunks_by_turn must have %d entries in %s" % (turns, record_id))
    contexts = []
    for turn, trace in enumerate(retrieved, 1):
        if trace is None:
            contexts.append(None)
            continue
        if not isinstance(trace, dict):
            raise ValueError("retrieved_chunks_by_turn[%d] must be an object in %s" % (turn, record_id))
        documents = trace.get("final_context_documents")
        if documents is None:
            contexts.append(None)
            continue
        if not isinstance(documents, list) or not all(
                isinstance(d, dict) and isinstance(d.get("text"), str) for d in documents):
            raise ValueError("final_context_documents at turn %d must be a list of text documents in %s" %
                             (turn, record_id))
        contexts.append([d["text"] for d in documents])
    return contexts


def preserve_optional_turn_fields(source_input, metadata, target, turns, record_id):
    """Copy recorded fields without fabricating answerability or generator history."""
    for field in ("answerability", "histories"):
        source = source_input if field in source_input else metadata
        if field not in source:
            continue
        values = source[field]
        if not isinstance(values, list) or len(values) != turns:
            raise ValueError("%s must have %d entries in %s" % (field, turns, record_id))
        if field == "answerability":
            if not all(value is None or isinstance(value, str) and value.strip().upper() in ANSWERABILITY_VALUES
                       for value in values):
                raise ValueError("answerability entries must be supported labels or null in %s" % record_id)
        else:
            if not all(isinstance(history, list) and all(
                    isinstance(message, dict) and message.get("role") in ("system", "user", "assistant", "tool")
                    and isinstance(message.get("content"), str) for message in history) for history in values):
                raise ValueError("histories entries must be lists of role/content messages in %s" % record_id)
        target[field] = values


def load_reference(reference_path=None):
    """Read original tasks or the existing gold index without making helper files."""
    path = pathlib.Path(reference_path) if reference_path is not None else (
        DEFAULT_REFERENCE if DEFAULT_REFERENCE.is_file() else GOLD_SUPPORT)
    if not path.is_file():
        raise ValueError("Gold answers not found. Supply --reference with an MTRAG reference file "
                         "or gold_support.json. Missing: %s" % path)
    reference = {}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), dict):
            raise ValueError("Reference JSON must be a gold_support.json index: %s" % path)
        for task_id, task in payload["tasks"].items():
            reference[task_id] = {
                "task_id": task_id, "_question": task["question"],
                "targets": [{"text": text} for text in task.get("gold_answers", [])],
            }
    else:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                task = json.loads(line)
                task_id = task["task_id"]
                if task_id in reference:
                    raise ValueError("Duplicate reference task: %s" % task_id)
                questions = [m["text"] for m in task.get("input", [])
                             if m.get("speaker") == "user"]
                if questions:
                    task["_question"] = questions[-1]
                reference[task_id] = task
    print("Gold answers: %s" % path)
    return reference


def records_from_rag_export(export, reference_path=None, limit=None):
    """Judge input from a RAG export; gold answers come from MTRAG reference targets."""
    reference = load_reference(reference_path)
    records = []
    conversations = export["conversations"]
    for conv in conversations[:limit] if limit else conversations:
        turns = conv["turns"]
        if not turns or any(not t.get("answered") or not t.get("retrieved") for t in turns):
            raise ValueError("conversation %s has unanswered or unretrieved turns" % conv["id"])
        conversation, golds, contexts, answerability = [], [], [], []
        for turn in turns:
            task = reference.get(turn["task_id"])
            if task is None or not task.get("targets"):
                raise ValueError("no MTRAG reference answer for %s" % turn["task_id"])
            if ("_question" in task and
                    task["_question"].strip() != turn["question"].strip()):
                raise ValueError("Reference question does not match %s. Supply matching gold answers "
                                 "with --reference." % turn["task_id"])
            labels = turn.get("answerability")
            if labels is None:
                # Some RAG exports omit this field even though the official task
                # records it. Use the matching reference task, never infer it.
                labels = task.get("Answerability")
            if (not isinstance(labels, list) or len(labels) != 1 or
                    not isinstance(labels[0], str) or
                    labels[0].strip().upper() not in ANSWERABILITY_VALUES):
                raise ValueError("answerability must be a one-item supported-label list "
                                 "in the export or reference for %s" % turn["task_id"])
            conversation += [{"role": "user", "content": turn["question"]},
                             {"role": "assistant", "content": turn["answer"]}]
            golds.append(task["targets"][0]["text"])
            contexts.append([p["text"] for p in turn["passages"]])
            answerability.append(labels[0])
        records.append({
            "id": conv["id"], "conversation": conversation, "ground_truth_answers": golds,
            "contexts": contexts, "answerability": answerability,
            "metadata": {"dataset": conv.get("domain"), "rag": export.get("rag_name"),
                         "turn_ids": [str(t["dataset_turn"]) for t in turns]}})
    return records


def prepare_records(source, reference=None, limit=None):
    src = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
    if isinstance(src, dict) and "conversations" in src:
        return records_from_rag_export(src, reference, limit)
    inp = []
    for r in src[:limit] if limit else src:
        i, m = r["input"], r["input"]["metadata"]
        n = sum(x["role"] == "assistant" for x in i["conversation"])
        if n != len(i["ground_truth_answers"]):
            raise ValueError("turn count mismatch in %s" % r["id"])
        ctx = selected_contexts(m, n, r["id"])
        record = {"id": r["id"], "conversation": i["conversation"],
                  "ground_truth_answers": i["ground_truth_answers"],
                  "metadata": {k: m[k] for k in ("dataset", "rag", "collection", "turn_ids") if k in m}}
        if ctx is not None:
            record["contexts"] = ctx
        preserve_optional_turn_fields(i, m, record, n, r["id"])
        inp.append(record)
    return inp


def write_input(records, folder):
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "input.json").open("w", encoding="utf-8") as stream:
        json.dump(records, stream, ensure_ascii=False, indent=1)
    print("%d conversations, %d turns -> %s" % (
        len(records), sum(len(r["ground_truth_answers"]) for r in records), folder / "input.json"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=pathlib.Path, default=ROOT / "conversation.json")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "work")
    ap.add_argument("--limit", type=int, help="first N conversations only (pilot)")
    ap.add_argument("--reference", type=pathlib.Path,
                    help="gold reference JSONL or gold_support.json; auto-detected when omitted")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    try:
        records = prepare_records(args.src, args.reference, args.limit)
        write_input(records, args.out)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        ap.exit(1, "Preparation failed: %s\n" % exc)


if __name__ == "__main__":
    main()
