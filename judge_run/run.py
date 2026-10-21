#!/usr/bin/env python3
"""One-folder entry point for the mtRAG answer-quality judge.

All active judging code and prompts are in judge_run. Run artifacts are kept
under judge_run/runs/<name>; exports in conversation/ remain read-only sources.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import List, Optional

from prepare_input import prepare_records, write_input
from judge_mtrag import split_turns

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
STAGES = ("judge", "prepare", "preview", "mtrag")
QWEN3 = "qwen/qwen3-32b"


def run_dir_for(name: str, runs_dir: pathlib.Path) -> pathlib.Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("run name must use only letters, digits, underscores, and hyphens")
    return runs_dir / name


def require_file(path: pathlib.Path, instruction: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit("Missing %s. %s" % (path, instruction))


def require_new(path: pathlib.Path) -> None:
    if path.exists():
        raise SystemExit("Already exists: %s. Use a new --name to preserve this result." % path)


def api_arguments() -> List[str]:
    url = os.environ.get("JUDGE_BASE_URL")
    model = os.environ.get("JUDGE_MODEL")
    key_name = os.environ.get("JUDGE_API_KEY_ENV")
    if not url or not model or not key_name or not os.environ.get(key_name):
        raise SystemExit(
            "Set JUDGE_BASE_URL, JUDGE_MODEL, JUDGE_API_KEY_ENV, and its API key "
            "in this PowerShell session before calling a judge."
        )
    return ["--base-url", url, "--model", model, "--api-key-env", key_name]


def validate_contexts(records: list) -> None:
    if not records:
        raise SystemExit("No conversations to judge.")
    if any(record.get("contexts") is None for record in records):
        raise SystemExit("mtRAG needs recorded selected contexts for every record.")
    if any(any(context is None for context in record["contexts"]) for record in records):
        raise SystemExit("mtRAG cannot read null context entries; selected passages must be recorded.")
    seen = set()
    try:
        for pos, record in enumerate(records, 1):
            record_id = str(record["id"])
            if record_id in seen:
                raise ValueError("Duplicate conversation ID: %s" % record_id)
            seen.add(record_id)
            turns = split_turns(record, pos)
            if any(turn["gold"] is None for turn in turns):
                raise ValueError("Missing gold answers in conversation %s" % record_id)
    except (ValueError, KeyError, TypeError) as exc:
        raise SystemExit("Invalid judge input: %s" % exc) from None


def resolve_source(value: str) -> pathlib.Path:
    """Accept a file path or a filename/stem in the project's conversation folder."""
    path = pathlib.Path(value)
    if path.is_file():
        return path.resolve()
    candidate = ROOT / "conversation" / path
    if not candidate.suffix:
        candidate = candidate.with_suffix(".json")
    require_file(candidate, "Give a JSON filename from conversation/ or a full file path.")
    return candidate.resolve()


def archive_run(folder: pathlib.Path) -> None:
    # Only move this run to a sibling folder; never follow a run-directory link.
    if folder.is_symlink() or folder.resolve().parent != folder.parent.resolve():
        raise SystemExit("Cannot archive a linked run directory: %s" % folder)
    backup = folder.with_name(folder.name + "_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.rename(backup)
    print("Previous run saved: %s" % backup, flush=True)


def run_mtrag(input_path: pathlib.Path, output_path: pathlib.Path,
              idk_max_tokens: Optional[int] = None, no_max_tokens: bool = False,
              max_tokens: Optional[int] = None,
              reasoning_max_tokens: Optional[int] = None,
              api_args: Optional[List[str]] = None) -> None:
    records = json.loads(input_path.read_text(encoding="utf-8"))
    validate_contexts(records)
    try:
        subprocess.run(
            [sys.executable, str(HERE / "judge_mtrag.py"),
             "--input", str(input_path), "--output", str(output_path), "--mode", "mtrag",
             *(["--idk-max-tokens", str(idk_max_tokens)] if idk_max_tokens is not None else []),
             *(["--no-max-tokens"] if no_max_tokens else []),
             *(["--max-tokens", str(max_tokens)] if max_tokens is not None else []),
             *(["--reasoning-max-tokens", str(reasoning_max_tokens)]
               if reasoning_max_tokens is not None else []),
             *(api_arguments() if api_args is None else api_args)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        # The mtRAG judge already printed the failed record and its cause.
        raise SystemExit(exc.returncode) from None


def main(argv: Optional[List[str]] = None, runs_dir: Optional[pathlib.Path] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--name", help="run folder name; judge defaults to source name + model")
    parser.add_argument("--src", help="judge/prepare: JSON path or filename in conversation/")
    parser.add_argument("--reference", type=pathlib.Path,
                        help="judge/prepare: optional gold reference; otherwise auto-detected")
    parser.add_argument("--model", help="judge: model ID (default qwen/qwen3-32b; alias qwen3)")
    parser.add_argument("--base-url", help="judge: API URL (default OpenRouter)")
    parser.add_argument("--api-key-env", help="judge: key variable (default OPENROUTER_API_KEY)")
    parser.add_argument("--fresh", action="store_true",
                        help="judge: archive the existing run folder before starting a new run")
    parser.add_argument("--dry-run", action="store_true",
                        help="judge: validate source and show settings without writing files or API calls")
    parser.add_argument("--limit", type=int, help="first N conversations; judge/prepare only")
    parser.add_argument("--idk-max-tokens", type=int,
                        help="initial IDK completion-token budget for mtrag; retries double it")
    parser.add_argument("--no-max-tokens", action="store_true",
                        help="omit request max_tokens for mtrag; provider/model limits still apply")
    parser.add_argument("--max-tokens", type=int,
                        help="completion-token ceiling per mtRAG request when reasoning is budgeted")
    parser.add_argument("--reasoning-max-tokens", type=int,
                        help="OpenRouter thinking-token budget per mtRAG request (requires --max-tokens)")
    args = parser.parse_args(argv)
    judging = args.stage in ("judge", "mtrag")
    if args.src is not None and args.stage not in ("judge", "prepare"):
        parser.error("--src is used with judge or prepare")
    if args.reference is not None and args.stage not in ("judge", "prepare"):
        parser.error("--reference is used with judge or prepare")
    if args.stage != "judge" and any((args.model, args.base_url, args.api_key_env, args.fresh, args.dry_run)):
        parser.error("--model, --base-url, --api-key-env, --fresh and --dry-run are used with judge")
    if args.stage == "judge" and not args.src:
        parser.error("judge needs --src, e.g. --src conversation_vector")
    if args.limit is not None and (args.stage not in ("prepare", "judge") or args.limit < 1):
        parser.error("--limit must be positive and is used with judge or prepare")
    if args.idk_max_tokens is not None and (not judging or args.idk_max_tokens < 64):
        parser.error("--idk-max-tokens must be >= 64 and is used with judge or mtrag")
    if args.no_max_tokens and not judging:
        parser.error("--no-max-tokens is used with judge or mtrag")
    if args.no_max_tokens and args.idk_max_tokens is not None:
        parser.error("--no-max-tokens cannot be combined with --idk-max-tokens")
    if args.max_tokens is not None and (not judging or args.max_tokens < 64):
        parser.error("--max-tokens must be >= 64 and is used with judge or mtrag")
    if args.reasoning_max_tokens is not None and (not judging or args.reasoning_max_tokens < 1):
        parser.error("--reasoning-max-tokens must be positive and is used with judge or mtrag")
    if args.no_max_tokens and args.max_tokens is not None:
        parser.error("--no-max-tokens cannot be combined with --max-tokens")
    if args.max_tokens is not None and args.reasoning_max_tokens is None:
        parser.error("--max-tokens here requires --reasoning-max-tokens")
    if args.reasoning_max_tokens is not None and (
            args.max_tokens is None or args.no_max_tokens or
            args.max_tokens <= args.reasoning_max_tokens):
        parser.error("--reasoning-max-tokens requires --max-tokens larger than the reasoning budget")
    if args.reasoning_max_tokens is not None and args.idk_max_tokens is not None:
        parser.error("--reasoning-max-tokens cannot be combined with --idk-max-tokens")
    if args.stage == "judge":
        source = resolve_source(args.src)
        model = QWEN3 if args.model in (None, "qwen3") else args.model
        suffix = "q3_32b" if model == QWEN3 else re.sub(r"[^A-Za-z0-9_-]", "_", model.split("/")[-1])
        args.name = args.name or (source.stem + "_" + suffix)[:64]
        # Match the previous Qwen3 pilot. Other models keep the ordinary budgets.
        if (model == QWEN3 and not args.no_max_tokens and args.idk_max_tokens is None and
                args.max_tokens is None and args.reasoning_max_tokens is None):
            args.reasoning_max_tokens, args.max_tokens = 4096, 8192
    else:
        args.name = args.name or "pilot3"
    try:
        run_dir = run_dir_for(args.name, runs_dir if runs_dir is not None else HERE / "runs")
    except ValueError as exc:
        parser.error(str(exc))
    input_path = run_dir / "input.json"

    if args.stage == "judge":
        try:
            records = prepare_records(source, args.reference, args.limit)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise SystemExit("Preparation failed: %s" % exc) from None
        validate_contexts(records)
        if run_dir.exists() and not args.fresh:
            raise SystemExit("Run already exists: %s. Add --fresh to archive it, or choose another --name." % run_dir)
        base_url = args.base_url or "https://openrouter.ai/api/v1"
        key_env = args.api_key_env or "OPENROUTER_API_KEY"
        print("Source: %s\nModel: %s\nRun: %s\nThreshold: 0.8\n"
              "Reasoning budget: %s; completion budget: %s" % (
                  source, model, run_dir, args.reasoning_max_tokens,
                  "provider default" if args.no_max_tokens else (args.max_tokens or 1024)), flush=True)
        if args.dry_run:
            print("Validated %d conversations, %d turns. No files changed; no API calls." % (
                len(records), sum(len(r["ground_truth_answers"]) for r in records)))
            return
        if not os.environ.get(key_env):
            raise SystemExit("Set %s in this terminal before judging. No files changed." % key_env)
        if run_dir.exists():
            archive_run(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        write_input(records, run_dir)
        run_mtrag(input_path, run_dir / "out.mtrag.json", args.idk_max_tokens,
                  args.no_max_tokens, args.max_tokens, args.reasoning_max_tokens,
                  ["--base-url", base_url, "--model", model, "--api-key-env", key_env])
        return

    if args.stage == "prepare":
        require_new(run_dir)
        try:
            records = prepare_records(resolve_source(args.src) if args.src else ROOT / "conversation.json",
                                      args.reference, args.limit)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise SystemExit("Preparation failed: %s" % exc) from None
        write_input(records, run_dir)
        return

    require_file(input_path, "Run: python run.py prepare --name %s --limit 3" % args.name)

    if args.stage == "preview":
        output_path = run_dir / "preview.json"
        require_new(output_path)
        model = os.environ.get("JUDGE_MODEL")
        if not model:
            raise SystemExit("Set JUDGE_MODEL before preview so it matches the judge run.")
        command = [sys.executable, str(HERE / "judge_mtrag.py"),
                   "--input", str(input_path), "--dry-run", "--mode", "mtrag", "--model", model]
        result = subprocess.run(command, capture_output=True)
        if result.returncode:
            raise SystemExit(result.stderr.decode("utf-8", errors="replace"))
        with output_path.open("xb") as stream:
            stream.write(result.stdout)
        print("Preview saved: %s (no API call)" % output_path)
        return

    output_path = run_dir / "out.mtrag.json"
    require_new(output_path)
    run_mtrag(input_path, output_path, args.idk_max_tokens, args.no_max_tokens,
              args.max_tokens, args.reasoning_max_tokens)


if __name__ == "__main__":
    main()
