#!/usr/bin/env python3
"""mtRAG-style turn-by-turn RAG-answer judge.

Uses RB_llm, RL_F (RAGAS faithfulness), IDK, and answerability conditioning.
The two-metric harmonic mean and binary cutoff are this project's adaptations.

Output retains the existing judge-result schema for downstream attribution
(failed turn == label "incorrect"). Raw scores are kept on every turn.
Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "qwen3.8-27b-q4_k_m"   # legacy direct-CLI default; run.py judge selects Qwen3-32B
PROMPT_VERSION = "paper-ragas-mtrag-v1"  # retained so previous mtRAG results can resume

# --------------------------------------------------------------------------- #
# Prompts.  "PAPER" = wording taken from the paper / IBM mt-rag-benchmark repo.
#           "OURS"  = wrapper we added (JSON output format), not from the paper.
# --------------------------------------------------------------------------- #

# RAGAS faithfulness step 1 (statement extraction). PAPER: first sentence. OURS: JSON format line.
RAGAS_STATEMENTS = (
    "Given a question and answer, create one or more statements from each sentence in the "
    "given answer.\nquestion: {question}\nanswer: {answer}\n\n"
    'Return only JSON of the form {{"statements": ["...", "..."]}}.'
)

# RAGAS faithfulness step 2 (verification). PAPER: first sentence. OURS: JSON format line.
RAGAS_VERIFY = (
    "Consider the given context and following statements, then determine whether they are "
    "supported by the information present in the context.\n"
    "context: {context}\n{statements}\n\n"
    'Return only JSON: a list "verdicts" with one object per statement, in order, each '
    '{{"statement": "...", "reason": "...", "verdict": "yes" or "no"}}.'
)

# mtRAG RB_llm: format_conversation_radbench() from IBM/mt-rag-benchmark judge_utils.py (PAPER).

RB_LLM = """
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question given the provided document and a reference answer.

Your evaluation should assess the faithfulness, appropriateness, and completeness. Your evaluation should focus on the assistant's answer to the question of the current turn.  You will be given the assistant's answer and a reference answer. You will also be given the user questions and assistant's answers of the previous turns of the conversation. You should consider how well the assistant's answer captures the key information, knowledge points mentioned in the reference answer, and how it respects or builds upon the focus and knowledge points from the previous turns.

[Faithfulness]: You are given the full conversation, the question of the current turn, the assistant's answer, and documents. You should evaluate how faithful is the assistant's answer to the information in the document and previous conversation.
[Appropriateness]: You should evaluate if the assistant's answer is relevant to the question of the current turn and if it addresses all the issues raised by the question without adding extra information.
[Completeness]: You should evaluate whether the assistant's answer is complete with information from the documents.

Begin your evaluation by comparing the assistant's answer against the reference answer in this turn. Be as objective as possible, and provide a detailed justification for your rating. After providing your explanation, you must rate the response on a scale of 1 to 10, strictly following this format: "Rating: [[rating]]", for example: "Rating: [[5]]".

[The Start of Previous Conversation]
{previous_conversation}
[The End of Previous Conversation]

[The Start of Current Turn Question]
{current_question}
[The End of Current Turn Question]

[The Start of Reference Answer]
{reference_answer}
[The End of Reference Answer]

[The Start of Assistant's Answer]
{response}
[The End of Assistant's Answer]

[The Start of Document]
{document}
[The End of Document]
"""

# mtRAG IDK judge: instruction paragraph verbatim from format_idk_judge() (PAPER). The repo
# prompt also has 15 few-shot examples that we did NOT retrieve; we keep only the single
# example shown to us (OURS: zero-/one-shot instead of 15-shot).
IDK_JUDGE = """Read an inquiry and a response to it.  Decide if the response contains, in part or in whole, an indication that there is not sufficient information to fully answer the inquiry.  Use only one of the following labels: yes, no, partial.  Answer "yes" if the response indicates that the inquiry cannot be answered, for example, due to lack of information (in a reference document or text).  Answer "partial" if the response indicates insufficient information (in a reference document or text) to answer part of the inquiry, but still attempts to answer other part of the inquiry that can be answered.  Answer "no" if the response does not indicate a lack of information preventing answering part of the inquiry, even if some reasoning steps or general knowledge are needed to address the inquiry.

[Inquiry]
how do i enroll my wife in medical insurance?

[Response]
She can continue medical and dental coverage under COBRA by calling the Benefits Center.

[Output]
no

[Inquiry]
{inquiry}

[Response]
{response}

[Output]
"""

NO_ANSWER_ANSWERABILITY = {"UNANSWERABLE", "CONVERSATIONAL"}


def verdict(ok: bool) -> str:
    return "correct" if ok else "incorrect"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", type=pathlib.Path, required=True, help=".json/.jsonl (see prepare_input.py)")
    p.add_argument("--output", type=pathlib.Path)
    p.add_argument("--resume-from", type=pathlib.Path,
                   help="reuse completed records from a partial output; write the merged result to a NEW --output")
    p.add_argument("--turn-checkpoints", action="store_true",
                   help="save and reuse completed turns in input-folder/turn_checkpoints, with exact input/settings checks")
    p.add_argument("--mode", choices=["mtrag"], default="mtrag",
                   help="mtRAG-style judge for every turn (single and multi-turn)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key-env", default=None, metavar="ENVVAR",
                   help="name of an env var holding a bearer key (e.g. OPENAI_API_KEY when --base-url is "
                        "https://api.openai.com/v1); the key itself is never taken on the command line")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--no-max-tokens", action="store_true",
                   help="omit request max_tokens for every mtRAG call; provider/model limits still apply")
    p.add_argument("--reasoning-max-tokens", type=int,
                   help="OpenRouter thinking-token budget for every mtRAG call; requires explicit --max-tokens")
    p.add_argument("--adaptive-max-tokens", type=int,
                   help="retry budget/length failures with doubled reasoning and completion budgets, "
                        "up to this completion ceiling; requires --reasoning-max-tokens")
    p.add_argument("--idk-max-tokens", type=int, default=512,
                   help="initial completion-token budget for the mtRAG IDK call; retries double it")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--limit", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-at-first-failure", action="store_true",
                   help="judge turns in order and label every turn after the first incorrect one "
                        "'not_judged'; default judges every turn")
    g = p.add_argument_group("PASS/FAIL thresholds (the papers define scores, not a pass/fail cut)")
    g.add_argument("--f-threshold", type=float, default=1.0,
                   help="mtRAG's RL_F per-metric verdict threshold")
    g.add_argument("--mtrag-threshold", type=float, default=0.8,
                   help="mtrag mode: turn is correct iff IDK-conditioned score >= this")
    g.add_argument("--rb-llm-min-rating", type=float, default=6.0,
                   help="mtrag per-metric verdict for RB_llm (1-10): correct iff rating >= this (OURS)")
    args = p.parse_args(argv)
    # Keep historical judge.thresholds keys for old mtRAG output/resume compatibility.
    args.vrag_accept_min = 4
    args.ar_threshold = None
    explicit_output = args.output is not None
    if args.workers < 1 or args.retries < 1 or args.max_tokens < 64 or args.idk_max_tokens < 64:
        p.error("workers/retries must be positive; max-tokens and idk-max-tokens >= 64")
    supplied_options = argv if argv is not None else sys.argv[1:]
    if args.no_max_tokens and any(
            token == option or token.startswith(option + "=")
            for token in supplied_options
            for option in ("--max-tokens", "--idk-max-tokens")):
        p.error("--no-max-tokens cannot be combined with --max-tokens or --idk-max-tokens")
    has_explicit_max_tokens = any(
        token == "--max-tokens" or token.startswith("--max-tokens=")
        for token in supplied_options)
    if args.reasoning_max_tokens is not None:
        if (args.reasoning_max_tokens < 1 or not has_explicit_max_tokens or
                args.max_tokens <= args.reasoning_max_tokens or args.no_max_tokens):
            p.error("--reasoning-max-tokens requires explicit --max-tokens larger than the reasoning budget")
        if any(token == "--idk-max-tokens" or token.startswith("--idk-max-tokens=")
               for token in supplied_options):
            p.error("--reasoning-max-tokens cannot be combined with --idk-max-tokens")
    if args.adaptive_max_tokens is not None and (
            args.reasoning_max_tokens is None or args.no_max_tokens or
            args.adaptive_max_tokens < args.max_tokens):
        p.error("--adaptive-max-tokens requires --reasoning-max-tokens and a ceiling >= --max-tokens")
    if args.adaptive_max_tokens is not None:
        # Adaptive recovery always preserves completed turns across passes.
        args.turn_checkpoints = True
    args.api_key = None
    if args.api_key_env:
        args.api_key = os.environ.get(args.api_key_env)
        if not args.api_key:
            p.error("--api-key-env %s is not set in the environment" % args.api_key_env)
    if not args.input.is_file():
        p.error("input file does not exist: %s" % args.input)
    if args.output is None and not args.dry_run:
        args.output = args.input.with_name(args.input.stem + ".mtrag-judged" + (
            args.input.suffix if args.input.suffix in (".json", ".jsonl") else ".jsonl"))
    if args.output is not None and args.output.resolve() == args.input.resolve():
        p.error("output must not overwrite the input file")
    if args.resume_from:
        if args.dry_run:
            p.error("--resume-from cannot be combined with --dry-run")
        if not args.resume_from.is_file():
            p.error("resume file does not exist: %s" % args.resume_from)
        if not explicit_output or args.output.exists():
            p.error("--resume-from requires a new, non-existing --output")
        if args.resume_from.resolve() == args.output.resolve():
            p.error("resume output must not overwrite the previous result")
    return args


# ------------------------------- input ------------------------------------- #

def load_records(path: pathlib.Path) -> Tuple[str, List[Dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        recs = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        kind = "jsonl"
    elif path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        recs, kind = (payload, "json-array") if isinstance(payload, list) else ([payload], "json-object")
    else:
        raise ValueError("input must be .json or .jsonl")
    seen = set()
    for i, r in enumerate(recs, 1):
        r.setdefault("id", str(i))
        r.setdefault("metadata", {})
        if str(r["id"]) in seen:
            raise ValueError("duplicate record id: %s" % r["id"])
        seen.add(str(r["id"]))
        r["_turns"] = split_turns(r, i)
    if not recs:
        raise ValueError("input contains no records")
    return kind, recs


def split_turns(rec: Dict[str, Any], pos: int) -> List[Dict[str, Any]]:
    """conversation -> [{question, answer, history:[(q,a)..]}] plus per-turn contexts/gold/answerability."""
    turns: List[Dict[str, Any]] = []
    pending: List[str] = []
    for m in rec.get("conversation") or []:
        if m.get("role") == "user":
            pending.append(str(m.get("content", "")))
        elif m.get("role") == "assistant":
            if not pending:
                raise ValueError("record %d: assistant reply before any user message" % pos)
            turns.append({"question": "\n".join(pending), "answer": str(m.get("content", ""))})
            pending = []
    if not turns:
        raise ValueError("record %d: conversation has no assistant turns" % pos)
    n = len(turns)
    ctxs = rec.get("contexts")      # mtRAG needs these selected passages (checked in check_inputs)
    if ctxs is not None and not (isinstance(ctxs, list) and len(ctxs) == n and all(
            isinstance(c, list) and all(isinstance(x, str) for x in c) for c in ctxs)):
        raise ValueError("record %d: 'contexts' must be a list of %d lists of passage strings" % (pos, n))
    golds = rec.get("ground_truth_answers")
    if golds is not None and not (isinstance(golds, list) and len(golds) == n):
        raise ValueError("record %d: ground_truth_answers must have %d entries" % (pos, n))
    ans = rec.get("answerability")
    if ans is not None and not (isinstance(ans, list) and len(ans) == n):
        raise ValueError("record %d: answerability must have %d entries" % (pos, n))
    # optional per-turn history EXACTLY as the RAG was given it ([{role, content}, ...]); when the
    # RAG was fed MTRAG's original history (mtrag_three_rag), that -- not the RAG's own earlier
    # answers -- is the "previous conversation" a fair judge should see.
    hists = rec.get("histories")
    if hists is not None and not (isinstance(hists, list) and len(hists) == n):
        raise ValueError("record %d: 'histories' must have %d entries" % (pos, n))
    for i, t in enumerate(turns):
        t["context"] = ctxs[i] if ctxs is not None else None
        t["gold"] = golds[i] if golds else None
        t["answerability"] = str(ans[i]).upper() if ans and ans[i] else None
        t["history"] = (history_pairs(hists[i]) if hists is not None
                        else [(u["question"], u["answer"]) for u in turns[:i]])
    return turns


def history_pairs(msgs: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """[{role, content}, ...] -> [(user, assistant), ...]; a trailing unanswered user message is dropped."""
    pairs: List[Tuple[str, str]] = []
    pending: List[str] = []
    for m in msgs:
        if m.get("role") == "user":
            pending.append(str(m.get("content", "")))
        elif m.get("role") == "assistant" and pending:
            pairs.append(("\n".join(pending), str(m.get("content", ""))))
            pending = []
    return pairs


# ------------------------------- LLM I/O ----------------------------------- #

def _post(url: str, payload: Dict[str, Any], timeout: float, api_key: Optional[str] = None) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _add_usage(acc: Dict[str, int], body: Dict[str, Any]) -> None:
    for k, v in (body.get("usage") or {}).items():
        if isinstance(v, int):
            acc[k] = acc.get(k, 0) + v


def ask(args: argparse.Namespace, prompt: str, parse: Callable[[str], Any], usage: Dict[str, int],
        schema: Optional[Dict[str, Any]] = None, max_tokens: Optional[int] = None,
        step: str = "LLM call") -> Any:
    """One chat call; retried (token budget doubled) on transport/format/parse errors."""
    last: Optional[Exception] = None
    last_response_usage: Dict[str, Any] = {}
    base = max_tokens or args.max_tokens
    no_max_tokens = getattr(args, "no_max_tokens", False)
    reasoning_max_tokens = getattr(args, "reasoning_max_tokens", None)
    adaptive_max_tokens = getattr(args, "adaptive_max_tokens", None)
    for attempt in range(1, args.retries + 1):
        completion_ceiling = args.max_tokens
        reasoning_ceiling = reasoning_max_tokens
        if adaptive_max_tokens is not None:
            completion_ceiling = min(args.max_tokens * 2 ** (attempt - 1), adaptive_max_tokens)
            reasoning_ceiling = min(reasoning_max_tokens * 2 ** (attempt - 1),
                                    completion_ceiling - 1)
        payload: Dict[str, Any] = {"model": args.model, "temperature": 0, "seed": 0,
                                   "messages": [{"role": "user", "content": prompt}]}
        if not no_max_tokens:
            # Fixed budgets remain the default. Explicit adaptive mode increases
            # both budgets only on retries, never above its requested ceiling.
            payload["max_tokens"] = (completion_ceiling if reasoning_max_tokens is not None
                                     else base * (2 ** (attempt - 1)))
        if reasoning_max_tokens is not None:
            payload["reasoning"] = {"max_tokens": reasoning_ceiling}
        if adaptive_max_tokens is not None:
            usage["max_requested_completion_tokens"] = max(
                usage.get("max_requested_completion_tokens", 0), completion_ceiling)
            usage["max_requested_reasoning_tokens"] = max(
                usage.get("max_requested_reasoning_tokens", 0), reasoning_ceiling)
        if schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "out", "strict": True, "schema": schema}}
        try:
            body = _post(args.base_url.rstrip("/") + "/chat/completions", payload, args.timeout, args.api_key)
            _add_usage(usage, body)
            last_response_usage = body.get("usage") or {}
            reported_completion = last_response_usage.get("completion_tokens")
            reported_reasoning = (last_response_usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens")
            if reasoning_max_tokens is not None:
                if isinstance(reported_completion, int) and not isinstance(reported_completion, bool):
                    usage["completion_budget_verified_calls"] = usage.get(
                        "completion_budget_verified_calls", 0) + 1
                else:
                    usage["completion_budget_unverified_calls"] = usage.get(
                        "completion_budget_unverified_calls", 0) + 1
                if isinstance(reported_reasoning, int) and not isinstance(reported_reasoning, bool):
                    usage["reasoning_tokens"] = usage.get("reasoning_tokens", 0) + reported_reasoning
                    usage["reasoning_budget_verified_calls"] = usage.get(
                        "reasoning_budget_verified_calls", 0) + 1
                else:
                    usage["reasoning_budget_unverified_calls"] = usage.get(
                        "reasoning_budget_unverified_calls", 0) + 1
                if (isinstance(reported_completion, int) and not isinstance(reported_completion, bool) and
                        reported_completion > completion_ceiling):
                    error = ValueError if adaptive_max_tokens is not None else RuntimeError
                    raise error(
                        "%s provider exceeded requested completion ceiling %d: reported %d; "
                        "last_response_usage=%s" %
                        (step, completion_ceiling, reported_completion,
                         json.dumps(last_response_usage)))
            if (reasoning_max_tokens is not None and isinstance(reported_reasoning, int) and
                    reported_reasoning > reasoning_ceiling):
                error = ValueError if adaptive_max_tokens is not None else RuntimeError
                raise error(
                    "%s provider exceeded requested reasoning budget %d: reported %d; "
                    "last_response_usage=%s" %
                    (step, reasoning_ceiling, reported_reasoning,
                     json.dumps(last_response_usage)))
            choice = body["choices"][0]
            if choice.get("finish_reason") != "stop":
                if (no_max_tokens or reasoning_max_tokens is not None) and choice.get("finish_reason") == "length":
                    error = ValueError if adaptive_max_tokens is not None else RuntimeError
                    raise error(
                        "%s reached the provider/model output limit%s; last_response_usage=%s" %
                        (step, " with max_tokens omitted" if no_max_tokens else
                         " with completion ceiling %d" % completion_ceiling,
                         json.dumps(last_response_usage)))
                raise ValueError("generation did not stop cleanly: %s" % choice.get("finish_reason"))
            text = re.sub(r"<think>.*?</think>", "", choice["message"]["content"], flags=re.S).strip()
            return parse(text)
        except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as exc:
            last = exc
            if attempt < args.retries:
                if adaptive_max_tokens is not None:
                    print("%s: attempt %d/%d failed (%s); retrying with budgets up to %d tokens." % (
                        step, attempt, args.retries, type(exc).__name__,
                        min(completion_ceiling * 2, adaptive_max_tokens)), file=sys.stderr, flush=True)
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(
        "%s failed after %d attempts: %s; last_requested_max_tokens=%s; "
        "last_response_usage=%s" %
        (step, args.retries, last,
         "provider_default" if no_max_tokens else
         completion_ceiling if reasoning_max_tokens is not None else base * (2 ** (args.retries - 1)),
         json.dumps(last_response_usage))
    ) from last


# -------------------- RAGAS faithfulness used by mtRAG --------------------- #

def _statements_schema() -> Dict[str, Any]:
    return {"type": "object", "properties": {"statements": {"type": "array", "items": {"type": "string"}}},
            "required": ["statements"], "additionalProperties": False}


def _verdicts_schema(n: int) -> Dict[str, Any]:
    item = {"type": "object", "properties": {
        "statement": {"type": "string"}, "reason": {"type": "string"},
        "verdict": {"type": "string", "enum": ["yes", "no"]}},
        "required": ["statement", "reason", "verdict"], "additionalProperties": False}
    # no minItems/maxItems: OpenAI strict json_schema may reject them; the count is checked in parse_verdicts
    return {"type": "object", "properties": {"verdicts": {"type": "array", "items": item}},
            "required": ["verdicts"], "additionalProperties": False}


def faithfulness(args, question: str, answer: str, contexts: List[str], usage) -> Dict[str, Any]:
    """RAGAS faithfulness F = |V| / |S|. Uses question, answer, context only (no gold)."""
    def parse_statements(t: str) -> List[str]:
        s = [x.strip() for x in json.loads(t)["statements"] if isinstance(x, str) and x.strip()]
        return s
    stmts = ask(args, RAGAS_STATEMENTS.format(question=question, answer=answer),
                parse_statements, usage, _statements_schema(), step="RAGAS statement extraction")
    if not stmts:
        return {"score": None, "statements": [], "note": "no statements extracted"}

    def parse_verdicts(t: str) -> List[Dict[str, str]]:
        v = json.loads(t)["verdicts"]
        if len(v) != len(stmts) or any(x.get("verdict") not in ("yes", "no") for x in v):
            raise ValueError("verdicts do not match statements")
        return v
    numbered = "\n".join("statement: %s" % s for s in stmts)
    verdicts = ask(args, RAGAS_VERIFY.format(context="\n\n".join(contexts), statements=numbered),
                   parse_verdicts, usage, _verdicts_schema(len(stmts)),
                   max_tokens=args.max_tokens * 2, step="RAGAS support verification")
    yes = sum(v["verdict"] == "yes" for v in verdicts)
    return {"score": yes / len(stmts),
            "statements": [{"statement": s, "verdict": v["verdict"], "reason": v["reason"]}
                           for s, v in zip(stmts, verdicts)]}


# ------------------------------- mtRAG ------------------------------------- #

def format_history(history: List[Tuple[str, str]]) -> str:
    return "\n".join("user: %s\nassistant: %s" % qa for qa in history) or "(none)"


def parse_rating(text: str) -> float:
    m = re.findall(r"Rating:\s*\[\[\s*(\d+(?:\.\d+)?)\s*\]\]", text)
    if not m:
        raise ValueError("no 'Rating: [[n]]' in judge output")
    r = float(m[-1])
    if not 1 <= r <= 10:
        raise ValueError("rating out of range: %s" % r)
    return r


def parse_idk(text: str) -> float:
    """Repo's first_token_idk: yes -> 1, no -> 0, anything else (incl. partial) -> 0.5."""
    m = re.match(r"\s*([A-Za-z]+)", text)
    tok = m.group(1).lower() if m else ""
    return 1.0 if tok == "yes" else 0.0 if tok == "no" else 0.5


def harmonic(values: List[float]) -> float:
    return 0.0 if any(v <= 0 for v in values) else len(values) / sum(1.0 / v for v in values)


def idk_conditioned(answerability: Optional[str], idk: float, phi: float) -> float:
    """Paper-style IDK correction applied to this project's composite phi.

    unanswerable/conversational: IDK yes -> 1, otherwise 0.
    answerable/partial:          IDK yes -> 0, otherwise the metric value phi.
    answerability unknown:       phi (no conditioning).
    Conversational/unknown handling is a project adaptation (paper excludes it)."""
    if answerability is None:
        return phi
    if answerability in NO_ANSWER_ANSWERABILITY:
        return 1.0 if idk == 1.0 else 0.0
    return 0.0 if idk == 1.0 else phi


def judge_turn_mtrag(args, turn: Dict[str, Any], usage) -> Dict[str, Any]:
    if turn["gold"] is None:
        raise ValueError("mtrag mode needs ground_truth_answers (RB_llm is reference-based)")
    prompt = RB_LLM.format(
        previous_conversation=format_history(turn["history"]), current_question=turn["question"],
        reference_answer=turn["gold"], response=turn["answer"],
        document="\n\n".join(turn["context"]))
    rb_text, rb = ask(args, prompt, lambda t: (t, parse_rating(t)), usage)
    rb_norm = rb / 10.0                      # IBM judge_utils.extract_rating also normalises by /10.
    f = faithfulness(args, turn["question"], turn["answer"], turn["context"], usage)   # RL_F
    idk = ask(args, IDK_JUDGE.format(inquiry=turn["question"], response=turn["answer"]),
              parse_idk, usage, max_tokens=args.idk_max_tokens)
    parts = [rb_norm] + ([f["score"]] if f["score"] is not None else [])
    phi = harmonic(parts)                    # OURS: RB_alg omitted (needs bert-score/torch)
    score = idk_conditioned(turn["answerability"], idk, phi)
    label = "correct" if score >= args.mtrag_threshold else "incorrect"
    a = turn["answerability"]
    idk_ok = None if a is None else verdict(idk == 1.0 if a in NO_ANSWER_ANSWERABILITY else idk != 1.0)
    scores = {"RB_llm": rb, "RB_llm_norm": rb_norm, "RL_F": f["score"], "idk": idk,
              "answerability": a, "phi": phi, "conditioned_score": score,
              "statements": f["statements"],
              "metric_verdicts": {
                  "RB_llm": verdict(rb >= args.rb_llm_min_rating),
                  "RL_F": None if f["score"] is None else verdict(f["score"] >= args.f_threshold),
                  "IDK": idk_ok,             # did the RAG abstain exactly when it should have
                  "conditioned": label}}
    reasoning = ("mtRAG: RB_llm=%.0f/10, RL_F=%s, IDK=%s, answerability=%s -> conditioned score %.3f "
                 "(threshold %.2f). RB_llm justification: %s" % (
                     rb, "n/a" if f["score"] is None else "%.3f" % f["score"], idk,
                     turn["answerability"] or "unknown", score, args.mtrag_threshold,
                     re.sub(r"\s+", " ", rb_text)[:400]))
    return {"label": label, "scores": scores, "reasoning": reasoning}


# ------------------------------- driver ------------------------------------ #

def turn_checkpoint(args, rec):
    """Locate a checkpoint scoped to this exact record and judging policy."""
    if not getattr(args, "turn_checkpoints", False):
        return None, {}
    settings = {key: getattr(args, key, None) for key in (
        "model", "base_url", "max_tokens", "reasoning_max_tokens", "adaptive_max_tokens",
        "no_max_tokens", "idk_max_tokens", "stop_at_first_failure", "f_threshold",
        "mtrag_threshold", "vrag_accept_min", "rb_llm_min_rating", "ar_threshold")}
    identity = {"record": rec, "settings": settings, "prompt_version": PROMPT_VERSION}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    path = args.input.parent / "turn_checkpoints" / (digest + ".json")
    if not path.is_file():
        return path, {}
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        raise ValueError("Invalid turn checkpoint for record %s" % rec["id"])
    turns = saved.get("turn_evaluations")
    usage = saved.get("usage")
    elapsed = saved.get("elapsed_seconds")
    valid = (saved.get("version") == 1 and saved.get("fingerprint") == digest and
             saved.get("id") == str(rec["id"]) and isinstance(turns, list) and
             len(turns) <= len(rec["_turns"]) and isinstance(usage, dict) and
             all(isinstance(key, str) and isinstance(value, (int, float)) and value >= 0
                 for key, value in usage.items()) and
             isinstance(elapsed, (int, float)) and elapsed >= 0)
    first_incorrect = None
    if valid:
        for k, turn in enumerate(turns, 1):
            if not isinstance(turn, dict) or turn.get("turn") != k:
                valid = False
                break
            label = turn.get("label")
            if first_incorrect is not None and args.stop_at_first_failure:
                if label != "not_judged":
                    valid = False
                    break
            elif (label not in ("correct", "incorrect") or
                  not isinstance(turn.get("scores"), dict) or
                  not isinstance(turn.get("reasoning"), str)):
                valid = False
                break
            if label == "incorrect" and first_incorrect is None:
                first_incorrect = k
    if not valid:
        raise ValueError("Invalid turn checkpoint for record %s; refusing to reuse it." % rec["id"])
    return path, saved


def save_turn_checkpoint(path, rec, evals, usage, elapsed):
    if path is None:
        return
    payload = {"version": 1, "fingerprint": path.stem, "id": str(rec["id"]),
               "turn_evaluations": evals, "usage": usage,
               "elapsed_seconds": round(elapsed, 3)}
    write_output(path, "json-object", [payload])


def evaluate_one(index: int, rec: Dict[str, Any], args) -> Tuple[int, Dict[str, Any]]:
    started = time.monotonic()
    checkpoint_path, saved = turn_checkpoint(args, rec)
    usage = dict(saved.get("usage", {}))
    previous_elapsed = saved.get("elapsed_seconds", 0)
    evals = list(saved.get("turn_evaluations", []))
    saved_count = len(evals)
    first_failed = (next((e["turn"] for e in evals if e["label"] == "incorrect"), None)
                    if args.stop_at_first_failure else None)
    for k, turn in enumerate(rec["_turns"], 1):
        if k <= saved_count:
            print("id=%s turn %d/%d: reusing saved %s" % (
                rec["id"], k, len(rec["_turns"]), evals[k - 1]["label"]), file=sys.stderr, flush=True)
            continue
        if first_failed is not None:
            evals.append({"turn": k, "label": "not_judged",
                          "reasoning": "Not judged: an earlier turn (%d) was incorrect." % first_failed})
            save_turn_checkpoint(checkpoint_path, rec, evals, usage,
                                 previous_elapsed + time.monotonic() - started)
            continue
        try:
            print("id=%s turn %d/%d: judging" % (rec["id"], k, len(rec["_turns"])),
                  file=sys.stderr, flush=True)
            r = judge_turn_mtrag(args, turn, usage)
        except Exception as exc:
            # Retain completed turns and reported usage from failed attempts.
            save_turn_checkpoint(checkpoint_path, rec, evals, usage,
                                 previous_elapsed + time.monotonic() - started)
            raise RuntimeError(
                "record %s turn %d (%s) failed: %s; api_reported_usage_so_far=%s" %
                (rec["id"], k, "mtrag", exc, json.dumps(usage))
            ) from exc
        evals.append({"turn": k, "label": r["label"], "reasoning": r["reasoning"], "scores": r["scores"]})
        save_turn_checkpoint(checkpoint_path, rec, evals, usage,
                             previous_elapsed + time.monotonic() - started)
        print("id=%s turn %d/%d: %s" % (rec["id"], k, len(rec["_turns"]), r["label"]),
              file=sys.stderr, flush=True)
        if getattr(args, "stop_at_first_failure", False) and r["label"] == "incorrect":
            first_failed = k
    conv = "correct" if all(e["label"] in ("correct", "not_judged") for e in evals) else "incorrect"
    first_incorrect = next((e["turn"] for e in evals if e["label"] == "incorrect"), None)
    judge_info = {"model": args.model, "prompt_version": PROMPT_VERSION,
                  "thresholds": {"f": args.f_threshold, "mtrag": args.mtrag_threshold,
                                 "vrag_accept_min": args.vrag_accept_min,
                                 "rb_llm_min_rating": args.rb_llm_min_rating,
                                 "ar": args.ar_threshold},
                  "elapsed_seconds": round(previous_elapsed + time.monotonic() - started, 3), "usage": usage}
    if args.idk_max_tokens != 512:
        judge_info["idk_max_tokens"] = args.idk_max_tokens
    if args.no_max_tokens:
        judge_info["max_tokens_mode"] = "provider_default"
    if args.max_tokens != 1024:
        judge_info["completion_max_tokens"] = args.max_tokens
    if getattr(args, "reasoning_max_tokens", None) is not None:
        judge_info["reasoning_max_tokens"] = args.reasoning_max_tokens
    if getattr(args, "adaptive_max_tokens", None) is not None:
        judge_info["adaptive_max_tokens"] = args.adaptive_max_tokens
    if getattr(args, "stop_at_first_failure", False):
        judge_info["turn_rule"] = "stop_at_first_failure"
    return index, {
        "id": rec["id"],
        "input": {k: rec[k] for k in ("conversation", "ground_truth_answers", "contexts", "answerability",
                                       "histories", "metadata") if k in rec},
        "evaluation": {"turn_evaluations": evals, "conversation_label": conv, "method": "mtrag",
                       "first_failed_turn": first_incorrect},
        "judge": judge_info,
    }


def write_output(path: pathlib.Path, kind: str, results: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as h:
        if kind == "jsonl":
            for r in results:
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            json.dump(results[0] if kind == "json-object" else results, h, ensure_ascii=False, indent=2)
            h.write("\n")
    tmp.replace(path)


def check_inputs(records: List[Dict[str, Any]]) -> None:
    """Require selected context and gold answers before any mtRAG API calls."""
    for r in records:
        if any(t["context"] is None for t in r["_turns"]):
            raise SystemExit("record %s: mtRAG needs 'contexts' (selected passages per turn)" % r["id"])
        if any(t["gold"] is None for t in r["_turns"]):
            raise SystemExit("record %s: mtRAG needs ground_truth_answers" % r["id"])


def load_completed_for_resume(args, records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Reuse only complete records from the same input and judging configuration."""
    if not args.resume_from:
        return {}
    with args.resume_from.open(encoding="utf-8") as stream:
        if args.resume_from.suffix.lower() == ".jsonl":
            saved = [json.loads(line) for line in stream if line.strip()]
        else:
            payload = json.load(stream)
            saved = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(item, dict) for item in saved):
        raise ValueError("resume file must contain judged records")
    input_by_id = {str(rec["id"]): rec for rec in records}
    expected_thresholds = {
        "f": args.f_threshold, "mtrag": args.mtrag_threshold,
        "vrag_accept_min": args.vrag_accept_min,
        "rb_llm_min_rating": args.rb_llm_min_rating, "ar": args.ar_threshold,
    }
    completed = {}
    for item in saved:
        rid = str(item.get("id"))
        if rid in completed or rid not in input_by_id:
            raise ValueError("resume file has duplicate or unknown record ID: %s" % rid)
        rec = input_by_id[rid]
        expected_input = {k: rec[k] for k in (
            "conversation", "ground_truth_answers", "contexts", "answerability",
            "histories", "metadata") if k in rec}
        judge = item.get("judge") or {}
        evaluation = item.get("evaluation") or {}
        turns = evaluation.get("turn_evaluations") or []
        expected_turns = list(range(1, len(rec["_turns"]) + 1))
        actual_turns = [turn.get("turn") for turn in turns if isinstance(turn, dict)]
        labels = [turn.get("label") for turn in turns if isinstance(turn, dict)]
        first_incorrect = next((i for i, label in enumerate(labels, 1)
                                if label == "incorrect"), None)
        expected_conversation_label = "incorrect" if first_incorrect else "correct"
        expected_rule = "stop_at_first_failure" if args.stop_at_first_failure else None
        if args.stop_at_first_failure:
            expected_labels = (["correct"] * (first_incorrect - 1) + ["incorrect"] +
                               ["not_judged"] * (len(labels) - first_incorrect)
                               if first_incorrect else ["correct"] * len(labels))
            valid_turn_rule = labels == expected_labels
        else:
            valid_turn_rule = "not_judged" not in labels
        if (item.get("input") != expected_input or
                evaluation.get("method") != "mtrag" or
                judge.get("model") != args.model or
                judge.get("prompt_version") != PROMPT_VERSION or
                judge.get("thresholds") != expected_thresholds or
                judge.get("idk_max_tokens", 512) != args.idk_max_tokens or
                judge.get("reasoning_max_tokens") != args.reasoning_max_tokens or
                # Explicit adaptive mode may retain fixed-budget results from
                # its identical baseline. Each record keeps its original metadata.
                judge.get("adaptive_max_tokens") not in (
                    None, getattr(args, "adaptive_max_tokens", None)) or
                judge.get("completion_max_tokens", 1024) != args.max_tokens or
                judge.get("max_tokens_mode", "bounded") != (
                    "provider_default" if args.no_max_tokens else "bounded") or
                judge.get("turn_rule") != expected_rule or
                actual_turns != expected_turns or
                len(turns) != len(expected_turns) or
                any(label not in ("correct", "incorrect", "not_judged") for label in labels) or
                not valid_turn_rule or
                evaluation.get("conversation_label") != expected_conversation_label or
                evaluation.get("first_failed_turn") != first_incorrect):
            raise ValueError("resume record %s differs from input, method, settings, or turn count" % rid)
        completed[rid] = item
    return completed


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    kind, records = load_records(args.input)
    if args.limit:
        records = records[: args.limit]
    check_inputs(records)
    completed = load_completed_for_resume(args, records)
    if args.dry_run:
        r = records[0]
        t = r["_turns"][-1]
        prompts: Dict[str, Any] = {"mode": "mtrag", "record_id": r["id"], "turn": len(r["_turns"])}
        prompts["faithfulness_step1"] = RAGAS_STATEMENTS.format(question=t["question"], answer=t["answer"])
        prompts["RB_llm"] = RB_LLM.format(
            previous_conversation=format_history(t["history"]), current_question=t["question"],
            reference_answer=t["gold"], response=t["answer"], document="\n\n".join(t["context"]))
        prompts["IDK"] = IDK_JUDGE.format(inquiry=t["question"], response=t["answer"])
        json.dump(prompts, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    print("Judging %d missing record(s); reusing %d completed record(s) with %s (%d worker(s))" %
          (len(records) - len(completed), len(completed), args.model, args.workers),
          file=sys.stderr, flush=True)
    ordered: List[Optional[Dict[str, Any]]] = [completed.get(str(r["id"])) for r in records]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(evaluate_one, i, r, args): r["id"] for i, r in enumerate(records)
                if ordered[i] is None}
        for done, fut in enumerate(as_completed(futs), 1):
            try:
                i, res = fut.result()
            except Exception as exc:        # keep going; report at the end (never silently drop)
                failures.append((futs[fut], str(exc)))
                print("[%d/%d] id=%s FAILED: %s" % (done, len(futs), futs[fut], exc), file=sys.stderr)
                continue
            ordered[i] = res
            # Save complete conversations as they finish, including on a partial
            # run. Failed records never receive invented scores or labels.
            write_output(args.output, kind, [r for r in ordered if r is not None])
            print("[%d/%d] id=%s label=%s" % (done, len(futs), res["id"],
                  res["evaluation"]["conversation_label"]), file=sys.stderr, flush=True)
    results = [r for r in ordered if r is not None]
    if results:
        write_output(args.output, kind, results)
    bad = sum(r["evaluation"]["conversation_label"] == "incorrect" for r in results)
    print("Done: correct=%d incorrect=%d failed_records=%d output=%s" % (
        len(results) - bad, bad, len(failures), args.output), file=sys.stderr)
    for rid, msg in failures:
        print("  FAILED %s: %s" % (rid, msg), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
