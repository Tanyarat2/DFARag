#!/usr/bin/env python3
"""Wait for the current vector judge, then run SelfRAG and RAPTOR in the background.

Run from a terminal containing OPENROUTER_API_KEY, or enter it at the masked
prompt. Credentials are passed only through the child environment, never files.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import termios
import time
import tty
from datetime import datetime

import judge_mtrag
from prepare_input import prepare_records, write_input


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
MODEL = "qwen/qwen3-32b"
BASE_URL = "https://openrouter.ai/api/v1"
VECTOR = "convos_vector_q3_32b"
REMAINING = ("selfrag", "raptor")


def masked_api_key():
    """Read an ASCII API key with star feedback; always restore terminal settings."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError("Run from your terminal to enter the API key securely.")
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    key = bytearray()
    escape_mode = 0
    try:
        # Disable terminal echo before accepting any pasted or typed characters.
        tty.setcbreak(fd, when=termios.TCSANOW)
        sys.stdout.write("OpenRouter API key (* hides each character): ")
        sys.stdout.flush()
        while True:
            char = os.read(fd, 1)
            if not char or char == b"\x04":
                raise EOFError("API key entry cancelled; no queue started.")
            if char == b"\x03":
                raise KeyboardInterrupt
            if char in (b"\r", b"\n"):
                return key.decode("ascii").strip()
            if escape_mode:
                if escape_mode == 1:
                    escape_mode = 2 if char in (b"[", b"O") else 0
                elif 0x40 <= char[0] <= 0x7e:
                    escape_mode = 0
                continue
            if char == b"\x1b":
                escape_mode = 1
            elif char in (b"\x7f", b"\x08"):
                if key:
                    key.pop()
                    sys.stdout.write("\b \b")
            elif char == b"\x15":  # Ctrl+U clears the entry.
                sys.stdout.write("\b \b" * len(key))
                key.clear()
            elif 0x20 <= char[0] <= 0x7e:
                key.extend(char)
                sys.stdout.write("*")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        sys.stdout.write("\n")
        sys.stdout.flush()


def process_identity(pid):
    try:
        fields = Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except (OSError, IndexError):
        return None


def validate_completed(name):
    """Require all input records and turns under the same Qwen3 settings."""
    folder = RUNS / name
    output = folder / "out.mtrag.json"
    if not output.is_file():
        raise ValueError("%s has no completed output; queue stopped." % name)
    args = judge_mtrag.parse_args([
        "--input", str(folder / "input.json"), "--output", str(output),
        "--model", MODEL, "--base-url", BASE_URL,
        "--reasoning-max-tokens", "4096", "--max-tokens", "8192",
    ])
    # Reuse the judge's strict resume validation without invoking the API or
    # writing anything. Set resume_from after parsing because output is read-only.
    args.resume_from = output
    _, records = judge_mtrag.load_records(args.input)
    judge_mtrag.check_inputs(records)
    completed = judge_mtrag.load_completed_for_resume(args, records)
    if len(completed) != len(records):
        raise ValueError("%s completed only %d/%d records; queue stopped." % (
            name, len(completed), len(records)))
    print("Verified complete: %s (%d conversations, %d turns)" % (
        name, len(records), sum(len(r["_turns"]) for r in records)), flush=True)


def recovery_arguments(folder, ceiling):
    return ["--input", str(folder / "input.json"), "--mode", "mtrag",
            "--turn-checkpoints",
            "--base-url", BASE_URL, "--model", MODEL,
            "--reasoning-max-tokens", "4096", "--max-tokens", "8192",
            "--adaptive-max-tokens", str(ceiling)]


def recovery_state(folder, ceiling):
    """Find the most complete validated output without modifying prior files."""
    args = judge_mtrag.parse_args(recovery_arguments(folder, ceiling))
    _, records = judge_mtrag.load_records(args.input)
    judge_mtrag.check_inputs(records)
    best_path, best_count = None, 0
    canonical = folder / "out.mtrag.json"
    candidates = ([canonical] if canonical.is_file() else []) + sorted(
        folder.glob("out.mtrag.recovery_*.json"))
    for candidate in candidates:
        args.resume_from = candidate
        completed = judge_mtrag.load_completed_for_resume(args, records)
        if best_path is None or len(completed) > best_count:
            best_path, best_count = candidate, len(completed)
    return best_path, best_count, len(records)


def publish_recovery(folder, source):
    canonical = folder / "out.mtrag.json"
    if source == canonical:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if canonical.exists():
        shutil.copy2(canonical, folder / ("out.mtrag.before_recovery_%s.json" % stamp))
    temporary = folder / "out.mtrag.publish.tmp"
    shutil.copyfile(source, temporary)
    temporary.replace(canonical)


def complete_run(rag, ceiling, recovery_passes):
    name = "convos_%s_q3_32b" % rag
    folder = RUNS / name
    expected = prepare_records(HERE.parent / "conversation" / ("conversation_%s.json" % rag))
    input_path = folder / "input.json"
    if folder.exists():
        if not input_path.is_file() or json.loads(input_path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Existing input differs from source for %s; no files overwritten." % name)
    else:
        folder.mkdir()
        write_input(expected, folder)
    source, count, total = recovery_state(folder, ceiling)
    for attempt in range(1, recovery_passes + 1):
        if count == total:
            break
        print("%s: %d/%d complete; recovery pass %d/%d" % (
            name, count, total, attempt, recovery_passes), flush=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = folder / ("out.mtrag.recovery_%s_%d.json" % (stamp, attempt))
        command = [sys.executable, "-B", str(HERE / "judge_mtrag.py"),
                   *recovery_arguments(folder, ceiling), "--output", str(output),
                   "--api-key-env", "OPENROUTER_API_KEY"]
        if source is not None:
            command.extend(["--resume-from", str(source)])
        process = subprocess.run(command, cwd=HERE, check=False)
        if process.returncode not in (0, 1):
            raise ValueError("%s exited with code %d; queue stopped." % (name, process.returncode))
        source, count, total = recovery_state(folder, ceiling)
    if count != total:
        raise ValueError("%s still has %d/%d missing conversations after %d passes. "
                         "Partial progress is saved; rerun the recovery queue to continue." % (
                             name, total - count, total, recovery_passes))
    publish_recovery(folder, source)
    print("Verified complete: %s (%d/%d conversations)" % (name, count, total), flush=True)


def worker(watched, complete_all=False, adaptive_max_tokens=32768, recovery_passes=3, rags=None):
    selected = rags or ("vector", *REMAINING)
    print("Queue: %s; model=%s" % (" -> ".join(selected), MODEL), flush=True)
    for rag in (() if complete_all else REMAINING):
        name = "convos_%s_q3_32b" % rag
        if (RUNS / name).exists():
            raise ValueError("Run already exists: %s; queue will preserve it." % name)
    if watched:
        print("Waiting for vector process(es): %s" % ", ".join(
            str(pid) for pid, _ in watched), flush=True)
    while any(process_identity(pid) == identity for pid, identity in watched):
        time.sleep(15)
    if complete_all:
        for rag in selected:
            complete_run(rag, adaptive_max_tokens, recovery_passes)
        print("Queue finished: all conversations in the selected RAGs are complete.", flush=True)
        return
    validate_completed(VECTOR)
    for rag in REMAINING:
        name = "convos_%s_q3_32b" % rag
        print("Starting %s at %s" % (name, datetime.now().isoformat()), flush=True)
        subprocess.run([
            sys.executable, "-B", str(HERE / "run.py"), "judge",
            "--src", "conversation_%s" % rag, "--name", name,
            "--base-url", BASE_URL, "--model", MODEL,
            "--api-key-env", "OPENROUTER_API_KEY",
        ], cwd=HERE, check=True)
        validate_completed(name)
    print("Queue finished successfully.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--complete-all", action="store_true",
                        help="recover vector gaps, then complete SelfRAG and RAPTOR, retaining saved results")
    parser.add_argument("--rags", nargs='+', choices=('vector', 'selfrag', 'raptor', 'hipporag'),
                        help="with --complete-all, judge/recover only these backends in order")
    parser.add_argument("--adaptive-max-tokens", type=int, default=32768,
                        help="complete-all: completion ceiling on retries (default 32768)")
    parser.add_argument("--recovery-passes", type=int, default=3,
                        help="complete-all: maximum passes over missing conversations per run (default 3)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--watch", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rags and not args.complete_all:
        parser.error("--rags requires --complete-all")
    if args.adaptive_max_tokens < 8192 or args.recovery_passes < 1:
        parser.error("--adaptive-max-tokens must be >= 8192 and --recovery-passes must be positive")
    if args.worker:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY is missing.")
        worker([(int(value.split(":")[0]), value.split(":")[1]) for value in args.watch],
               args.complete_all, args.adaptive_max_tokens, args.recovery_passes, args.rags)
        return

    # Hold a lock before detaching so launching twice cannot duplicate paid runs.
    RUNS.mkdir(exist_ok=True)
    with (RUNS / ".remaining_judges.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("A remaining-judges queue is already running.") from None
        for rag in (() if args.complete_all else REMAINING):
            if (RUNS / ("convos_%s_q3_32b" % rag)).exists():
                raise ValueError("A destination run already exists; no queue started.")
        env = os.environ.copy()
        if not env.get("OPENROUTER_API_KEY"):
            env["OPENROUTER_API_KEY"] = masked_api_key()
        if not env.get("OPENROUTER_API_KEY"):
            raise ValueError("No API key entered; no queue started.")
        print("API key received (%d characters)." % len(env["OPENROUTER_API_KEY"]), flush=True)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
                   "--lock-fd", str(lock.fileno())]
        if args.complete_all:
            command.extend(["--complete-all", "--adaptive-max-tokens", str(args.adaptive_max_tokens),
                            "--recovery-passes", str(args.recovery_passes)])
            if args.rags:
                command.extend(["--rags", *args.rags])
        for pid in args.wait_pid:
            identity = process_identity(pid)
            if identity is not None:
                command.extend(["--watch", "%d:%s" % (pid, identity)])
        log_path = RUNS / "remaining_judges.log"
        with log_path.open("a") as log:
            process = subprocess.Popen(
                command, cwd=HERE, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(lock.fileno(),))
        print("Background queue PID: %d" % process.pid)
        print("Log: %s" % log_path)
        print("You may close this terminal. Keep the vector job running until it finishes.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAPI key entry cancelled; no queue started.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, EOFError, subprocess.CalledProcessError) as exc:
        print("Queue stopped: %s" % exc, file=sys.stderr, flush=True)
        sys.exit(1)
