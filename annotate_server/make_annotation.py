#!/usr/bin/env python3
"""Build a self-contained mtRAG annotation sheet for RAG turns.

    python annotate_server/make_annotation.py --work-dir judge_run/runs/pilot5

Reads input.json and published or recovery judge output from the run folder. Existing result JSON is
never overwritten. The human binary label is stored separately from the
judge's binary screening label and the paper's E1–E16 failure annotation.
"""

import argparse
import functools
import hashlib
import html
import json
import math
import pathlib
import warnings

from corpus_store import CORPUS_DIR, CorpusIndex, corpus_metadata
from failure_taxonomy import TAXONOMY, EXTRA_TYPES, taxonomy_data

JUDGE_TITLES = {"mtrag": "mtRAG (RB_llm + RL_F + IDK)"}
DEFAULT_THRESHOLDS = {"f": 1.0, "mtrag": 0.8, "rb_llm_min_rating": 6.0}


def rb_band(score):
    """Project-only display bands on the raw 1-10 RB_llm rating."""
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    if score <= 5:
        return "low"
    if score <= 8:
        return "mid"
    return "high"


def output_paths(work_dir, name):
    """Canonical output first, followed by recovery snapshots (read-only)."""
    work_dir = pathlib.Path(work_dir)
    canonical = work_dir / ("out.%s.json" % name)
    return [canonical] + sorted(work_dir.glob("out.%s.recovery_*.json" % name))


def load_output(work_dir, name):
    """Prefer published output; otherwise use the most complete valid snapshot."""
    paths = output_paths(work_dir, name)
    inputs = load_input_records(pathlib.Path(work_dir))
    best = None
    for path in paths:
        if not path.is_file():
            continue
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list) or any(
                    not isinstance(record, dict) or not isinstance(record.get("id"), str)
                    or not isinstance(record.get("evaluation", {}).get("turn_evaluations", []), list)
                    for record in records):
                raise ValueError("Expected judge records with conversation IDs and turn evaluations")
            indexed = {record["id"]: record for record in records}
            if len(indexed) != len(records):
                raise ValueError("Duplicate conversation IDs")
            if path != paths[0] and inputs and any(
                    record["id"] not in inputs or
                    {key: value for key, value in record.get("input", inputs[record["id"]]).items() if key != "id"} !=
                    {key: value for key, value in inputs[record["id"]].items() if key != "id"}
                    for record in records):
                raise ValueError("Recovery output does not match this run's input")
        except (ValueError, OSError) as exc:
            if path == paths[0]:
                raise
            warnings.warn("Skipping recovery output %s: %s" % (path.name, exc))
            continue
        if path == paths[0]:
            return indexed, path.name
        judged_turns = sum(turn.get("label") in ("correct", "incorrect")
                           for record in records
                           for turn in record.get("evaluation", {}).get("turn_evaluations", [])
                           if isinstance(turn, dict))
        rank = (judged_turns, len(indexed), path.name)
        if best is None or rank > best[0]:
            best = (rank, indexed, path.name)
    return (best[1], best[2]) if best else (None, None)


def load(work_dir, name):
    return load_output(work_dir, name)[0]


def turn_texts(conversation):
    result, questions = [], []
    for message in conversation:
        if message["role"] == "user":
            questions.append(message["content"])
        elif message["role"] == "assistant":
            result.append(("\n".join(questions), message["content"]))
            questions = []
    return result


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "%.3f" % value
    return str(value)


def not_run(status="not_run", headline="not run"):
    return {"status": status, "label": None, "headline": headline, "reasoning": "",
            "metrics": [], "notes": []}


def score_steps(scores, label, band, cutoff):
    """Explain saved mtRAG scores and the displayed gate without changing result files."""
    rb = scores.get("RB_llm")
    f = scores.get("RL_F")
    phi = scores.get("phi")
    conditioned = scores.get("conditioned_score")
    idk = scores.get("idk")
    answerability = scores.get("answerability")
    if not isinstance(rb, (int, float)) or isinstance(rb, bool):
        return []
    rb_norm = rb / 10
    steps = [
        "RB_llm: the LLM reads this turn's question, previous conversation, "
        "reference answer, RAG answer and selected passages. It gives one holistic "
        "1-10 rating for faithfulness, appropriateness and completeness; these "
        "criteria have no fixed numeric weights. %s/10 = %s (project level: %s)." %
        (fmt(rb), fmt(rb_norm), band or "n/a")]
    statements = scores.get("statements") or []
    if f is None:
        steps.append("RL_F: unavailable (no factual statements were extracted).")
        steps.append("phi: RL_F is unavailable, so phi = RB_llm/10 = %s." % fmt(phi))
    else:
        supported = sum(item.get("verdict") == "yes" for item in statements)
        detail = "%d/%d supported statements = " % (supported, len(statements)) if statements else ""
        steps.append("RL_F: %s%s. Each statement is checked against the selected "
                     "passages; the reference answer is not used for this metric." %
                     (detail, fmt(f)))
        if f <= 0:
            steps.append("phi: RL_F is 0, so the harmonic mean is 0.")
        else:
            steps.append("phi: 2 / (1/%s + 1/%s) = %s. This two-metric "
                         "per-turn combination is a project adaptation." %
                         (fmt(rb_norm), fmt(f), fmt(phi)))
    if answerability is None:
        rule = "answerability is unavailable, so the score stays at phi"
    elif answerability in ("UNANSWERABLE", "CONVERSATIONAL"):
        rule = ("the question is marked %s; IDK=%s, so the score is 1 only "
                "for IDK=1, otherwise 0" % (answerability, fmt(idk)))
    else:
        rule = ("the question is marked %s; IDK=%s, so the score is 0 "
                "for IDK=1, otherwise phi" % (answerability, fmt(idk)))
    steps.append("Conditioned score: %s; %s -> %s." % (rule, fmt(phi), fmt(conditioned)))
    steps.append("Screening label: conditioned score %s %s %s -> %s. "
                 "The cutoff and three RB_llm levels are project choices." %
                 (str(conditioned), ">=" if conditioned >= cutoff else "<",
                  str(cutoff), label))
    return steps


def scored_view(turn, cutoff=DEFAULT_THRESHOLDS["mtrag"]):
    if turn is None:
        return not_run()
    if turn["label"] == "not_judged":
        return {"status": "not_judged", "label": None, "headline": "not judged (earlier failure)",
                "reasoning": turn.get("reasoning", ""), "metrics": [], "notes": []}
    scores = turn.get("scores") or {}
    conditioned = scores.get("conditioned_score")
    has_score = (isinstance(conditioned, (int, float)) and
                 not isinstance(conditioned, bool) and math.isfinite(conditioned))
    label = ("correct" if conditioned >= cutoff else "incorrect") if has_score else None
    explanation = ("mtRAG conditioned score %s %s %s: %s." %
                   (str(conditioned), ">=" if label == "correct" else "<", str(cutoff), label)
                   if has_score else "No valid mtRAG conditioned score is saved; this turn cannot be screened.")
    answerability = scores.get("answerability")
    if label == "incorrect":
        if answerability in ("UNANSWERABLE", "CONVERSATIONAL") and scores.get("idk") != 1:
            explanation += (" The recorded question is %s and IDK=%s; this rule requires "
                            "a full insufficient-information response (IDK=1)." %
                            (answerability, fmt(scores.get("idk"))))
        elif answerability in ("ANSWERABLE", "PARTIAL") and scores.get("idk") == 1:
            explanation += " The recorded question is answerable/partial, but the answer fully abstains (IDK=1)."
    verdicts = scores.get("metric_verdicts") or {}
    band = rb_band(scores.get("RB_llm"))
    headline = "RB_llm %s/10 (%s); conditioned score %s" % (
        fmt(scores.get("RB_llm")), band or "unavailable", str(conditioned) if has_score else "unavailable")
    metrics = [
        ["RB_llm (1-10)", fmt(scores.get("RB_llm")), verdicts.get("RB_llm") or ""],
        ["RB_llm level (project bins)", band or "n/a", ""],
        ["RL_F (faithfulness)", fmt(scores.get("RL_F")), verdicts.get("RL_F") or ""],
        ["IDK judge", fmt(scores.get("idk")), verdicts.get("IDK") or ""],
        ["answerability", fmt(scores.get("answerability")), ""],
        ["phi (harmonic mean)", fmt(scores.get("phi")), ""],
        ["conditioned score", str(conditioned) if has_score else "n/a", label or ""]]
    notes = ["[%s] %s" % (item.get("verdict", "?"), item.get("statement", ""))
             for item in scores.get("statements") or []]
    return {"status": "judged", "label": label, "saved_label": turn["label"], "headline": headline,
            "cutoff": cutoff, "screening_reason": explanation,
            "conditioned_score": conditioned if has_score else None,
            "score_band": band, "reasoning": turn.get("reasoning", ""),
            "statements": scores.get("statements") or [],
            "metrics": metrics, "notes": notes,
            "score_steps": score_steps(scores, label, band, cutoff) if has_score else []}


def find_turn(record, turn):
    if record is None:
        return None
    for item in record.get("evaluation", {}).get("turn_evaluations", []):
        if item["turn"] == turn:
            return item
    return None


def load_input_records(work_dir):
    path = work_dir / "input.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        return {record["id"]: record for record in json.load(stream)}


def gold_evidence(index, task_id, question, gold):
    task = index.get("tasks", {}).get(task_id)
    if task is None:
        return {"status": "missing", "task_id": task_id, "documents": [],
                "message": "Original reference evidence is not available for this task."}
    if task.get("question", "").strip() != question.strip() or gold not in task.get("gold_answers", []):
        return {"status": "mismatch", "task_id": task_id, "documents": [],
                "message": "Reference question or gold answer differs from this run; evidence was not attached."}
    return {"status": "available", "task_id": task_id,
            "documents": task["documents"], "source": index.get("source", {}),
            "message": "Original reference-task passages; relevance labels are from the dataset."}


@functools.lru_cache(maxsize=2)
def conversation_passage_archive(snapshot):
    """Index recorded passages by collection; never claim these are a full index."""
    collections = {}
    for filename, _mtime, _size in snapshot:
        path = pathlib.Path(filename)
        try:
            raw = path.read_bytes()
            export = json.loads(raw)
        except (OSError, ValueError):
            continue
        if not isinstance(export, dict) or not isinstance(export.get("conversations"), list):
            continue
        provenance = {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()}
        for conversation in export["conversations"]:
            collection = str(conversation.get("domain") or "unspecified")
            store = collections.setdefault(collection, {"documents": {}, "sources": []})
            if provenance not in store["sources"]:
                store["sources"].append(provenance)
            for turn in conversation.get("turns", []):
                for passage in turn.get("passages", []):
                    if not isinstance(passage, dict) or not isinstance(passage.get("text"), str):
                        continue
                    text = passage["text"]
                    doc_id = str(passage.get("id") or passage.get("document_id") or "")
                    key = (doc_id, text)
                    if key not in store["documents"]:
                        store["documents"][key] = {
                            "id": doc_id, "title": str(passage.get("title") or "Untitled"),
                            "text": text, "source_file": path.name,
                            "observed_task": str(turn.get("task_id") or ""),
                        }
    return {collection: {"kind": "conversation_passages", "documents": list(store["documents"].values()),
                         "sources": store["sources"],
                         "message": "Passage snapshots recorded in conversation exports, not the complete original corpus or candidate pool."}
            for collection, store in collections.items()}


def load_passage_archive(folder):
    snapshot = tuple((str(p.resolve()), p.stat().st_mtime_ns, p.stat().st_size)
                     for p in sorted(pathlib.Path(folder).glob("*.json")))
    return conversation_passage_archive(snapshot)


@functools.lru_cache(maxsize=2)
def generation_traces(snapshot):
    """Keep source passage IDs for the final context; a retrieval flag is not a pool."""
    traces = {}
    for filename, _mtime, _size in snapshot:
        path = pathlib.Path(filename)
        try:
            export = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(export, dict):
            continue
        for conversation in export.get("conversations", []):
            for turn in conversation.get("turns", []):
                key = (export.get("rag_name"), conversation.get("id"), str(turn.get("dataset_turn")))
                traces.setdefault(key, []).append(turn)
    return traces


def generation_evidence(traces, rag, cid, turn_id, question, answer, selected):
    """Attach original IDs only when the source question, answer, and context match."""
    result = {"selected_passages": [
        {"id": "", "title": "", "text": text} for text in selected or []],
        "retrieved_flag": None, "retrieved_candidates": None, "reranking_trace": None}
    for turn in traces.get((rag, cid, str(turn_id)), []):
        passages = turn.get("passages", [])
        if (turn.get("question") != question or turn.get("answer") != answer or
                [item.get("text") for item in passages] != selected):
            continue
        result["selected_passages"] = [{key: item.get(key, "") for key in
                                        ("id", "title", "text")} for item in passages]
        flag = turn.get("retrieved")
        result["retrieved_flag"] = flag if isinstance(flag, bool) else None
        break
    return result


def build_data(work_dir, live=False):
    work_dir = pathlib.Path(work_dir)
    evidence_path = pathlib.Path(__file__).resolve().parent.parent / "judge_run" / "gold_support.json"
    evidence_index = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else {}
    inputs = load_input_records(work_dir)
    judged, judged_source = load_output(work_dir, "mtrag")
    if not inputs and not judged:
        raise SystemExit("No input.json or out.mtrag.json found in %s" % work_dir)
    ids = list(inputs)
    for cid in judged or {}:
        if cid not in ids:
            ids.append(cid)
    thresholds = dict(DEFAULT_THRESHOLDS)
    for record in (judged or {}).values():
        thresholds.update({k: v for k, v in record.get("judge", {}).get("thresholds", {}).items()
                           if k in thresholds and v is not None})
        break
    # Human screening always uses the requested 0.8 cutoff. Saved outputs are untouched.
    thresholds["mtrag"] = DEFAULT_THRESHOLDS["mtrag"]
    rows = []
    conversation_folder = pathlib.Path(__file__).resolve().parent.parent / "conversation"
    trace_snapshot = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                           for path in sorted(conversation_folder.glob("*.json")))
    traces = generation_traces(trace_snapshot)
    for cid in ids:
        source = inputs.get(cid)
        if source is None:
            source = judged[cid].get("input", judged[cid])
        texts = turn_texts(source["conversation"])
        golds = source.get("ground_truth_answers") or []
        contexts = source.get("contexts")
        answerability = source.get("answerability")
        for k, (question, answer) in enumerate(texts, 1):
            if judged is None:
                view = not_run()
            elif cid not in judged:
                view = not_run("record_missing", "record missing from output")
            else:
                turn = find_turn(judged[cid], k)
                view = scored_view(turn, thresholds["mtrag"]) if turn is not None else not_run(
                    "turn_missing", "turn missing from output")
            selected = None if contexts is None or k - 1 >= len(contexts) else contexts[k - 1]
            gold = golds[k - 1] if k - 1 < len(golds) else ""
            turn_ids = source.get("metadata", {}).get("turn_ids") or []
            task_id = "%s<::>%s" % (cid, turn_ids[k - 1] if k - 1 < len(turn_ids) else k)
            rows.append({
                "key": "%s#%d" % (cid, k), "id": cid, "turn": k, "n_turns": len(texts),
                "question": question, "answer": answer,
                "gold": gold,
                "corpus_collection": str(source.get("metadata", {}).get("collection") or
                                         source.get("metadata", {}).get("dataset") or "unspecified"),
                "gold_support": gold_evidence(evidence_index, task_id, question, gold),
                "answerability": (str(answerability[k - 1]).upper()
                                  if answerability and k - 1 < len(answerability) and answerability[k - 1]
                                  else "not recorded"),
                "contexts": None if selected is None else list(selected),
                **generation_evidence(traces, source.get("metadata", {}).get("rag"), cid,
                                      turn_ids[k - 1] if k - 1 < len(turn_ids) else k,
                                      question, answer, selected),
                "judges": {"mtrag": view},
            })
    coverage = {"mtrag": {
        "records": len(judged or {}),
        "turns": sum(row["judges"]["mtrag"]["status"] == "judged" for row in rows),
        "source": judged_source}}
    collections = sorted({row['corpus_collection'] for row in rows})
    official_stores = {collection: corpus_metadata(collection) for collection in collections}
    archive = (load_passage_archive(pathlib.Path(__file__).resolve().parent.parent / 'conversation')
               if any(store is None for store in official_stores.values()) else {})
    corpus = {collection: official_stores[collection] or archive.get(collection, {
                  'kind': 'missing', 'documents': [], 'sources': [],
                  'message': 'No conversation passage archive is available for this collection.'})
              for collection in collections}
    first_questions = {}
    for row in rows:
        first_questions.setdefault(row['id'], row['question'])
    for collection in corpus:
        official = official_stores[collection]
        if official is None:
            continue
        corpus[collection] = official
        if live:
            continue
        index = CorpusIndex(CORPUS_DIR / (collection + '.sqlite3'))
        documents = {}
        match_ids = {}
        for row in rows:
            if row['corpus_collection'] != collection:
                continue
            result = index.search('', row['question'] + ' ' + first_questions[row['id']], limit=10)
            match_ids[row['key']] = [match['doc']['id'] for match in result['matches']]
            for match in result['matches']:
                documents[match['doc']['id']] = match['doc']
        official.update(kind='ibm_corpus_snapshot', documents=list(documents.values()), match_ids=match_ids)
        official['message'] = ('Offline snapshot of the top 10 IBM corpus matches per turn; '
                               'search covers this embedded subset. The live server searches the full collection. '
                               'Matches retain BM25 search order without answer-based reranking.')
    return {"run": work_dir.name, "rows": rows, "judge_titles": JUDGE_TITLES,
            "corpus": corpus,
            **taxonomy_data(), "thresholds": thresholds,
            "coverage": coverage, "total_records": len(ids),
            "criteria_html": criteria_html(thresholds)}


def criteria_html(thresholds):
    """Explain the source of each score and distinguish paper from project choices."""
    cutoff = html.escape(fmt(thresholds["mtrag"]))
    return (
        "<section><h3>mtRAG-style evaluation in this project</h3>"
        "<p><b>RB_llm (1–10):</b> An LLM judge reads the current question, previous "
        "conversation, reference answer, RAG answer, and selected passages. It rates "
        "faithfulness, appropriateness, and completeness with the mtRAG prompt adapted "
        "from RAD-Bench. The code parses Rating: [[n]] and stores both n and n/10. "
        "This is one holistic rating; the three criteria have no fixed numeric "
        "weights, and 7/10 does not mean seven of ten claims are correct. "
        "The paper used several judge models and their median; this run uses the "
        "configured model once per turn.</p>"
        "<p><b>RL_F (0–1):</b> RAGAS faithfulness splits the answer into statements, "
        "checks each against selected passages, then divides supported statements by "
        "all statements. It does not consult the reference answer.</p>"
        "<p><b>IDK:</b> The LLM decides whether the answer says there is insufficient "
        "information: yes=1, partial=0.5, no=0. The local prompt uses fewer examples "
        "than the IBM implementation.</p>"
        "<p><b>phi (0–1):</b> Here phi = 2 / (1/(RB_llm/10) + 1/RL_F), the harmonic "
        "mean of two metrics when RL_F is defined and both are positive; otherwise "
        "it is 0 if either is 0, or RB_llm/10 when RL_F is undefined. This two-metric "
        "per-turn composite is <b>our adaptation</b>. In the mtRAG paper, phi denotes "
        "one of the individual metrics RB_alg, RB_llm, or RL_F. The mtRAG leaderboard "
        "uses a harmonic mean of all three metrics for system ranking.</p>"
        "<p><b>Conditioned score (0–1):</b> If answerability is ANSWERABLE/PARTIAL "
        "and IDK=yes, score=0; otherwise score=phi. If UNANSWERABLE/CONVERSATIONAL, "
        "score=1 only when IDK=yes, otherwise 0. If answerability is absent, score=phi. "
        "The paper describes conditioning each metric separately for answerable, partial, "
        "and unanswerable turns; the conversational and unknown cases are local choices. "
        "The judge's existing binary screening label is <b>correct iff conditioned "
        "score ≥ " + cutoff + "</b>; this cutoff is project-set.</p>"
        "<p><b>Three levels on this page:</b> RB_llm ≤5 = low; &gt;5 to ≤8 = medium; "
        "&gt;8 = high. These are project display/annotation bands, not thresholds from "
        "the mtRAG paper and not the binary failure gate.</p>"
        "<p>Sources: <a href='https://arxiv.org/html/2501.03468#S6.SS3' "
        "target='_blank' rel='noreferrer'>mtRAG paper §6.3</a>; "
        "<a href='https://github.com/IBM/mt-rag-benchmark/blob/main/scripts/evaluation/judge_utils.py' "
        "target='_blank' rel='noreferrer'>IBM judge prompt and rating parser</a>; "
        "<a href='https://ibm.github.io/mt-rag-benchmark/MTRAGEval/' "
        "target='_blank' rel='noreferrer'>official leaderboard formula</a>.</p>"
        "<p>Judge scores screen RAG answers; a human should verify failed turns "
        "before failure attribution.</p></section>"
    )


PAGE = pathlib.Path(__file__).with_name("annotation_template.html").read_text(encoding="utf-8")


def render_page(data):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # keep the JSON from ever closing the script element or opening an HTML comment
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return PAGE.replace("__DATA__", payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", help="default: <work-dir>/annotate.html")
    args = parser.parse_args(argv)
    work_dir = pathlib.Path(args.work_dir)
    output = pathlib.Path(args.output) if args.output else work_dir / "annotate.html"
    if output.exists():
        raise SystemExit("%s already exists; refusing to overwrite." % output)
    data = build_data(work_dir)
    output.write_text(render_page(data), encoding="utf-8")
    print("wrote %s (%d turns)" % (output, len(data["rows"])))


if __name__ == "__main__":
    main()
