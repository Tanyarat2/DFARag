"""Recovery and adaptive-budget checks; all model calls use synthetic responses."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

import judge_mtrag as judge
import queue_remaining_judges as queue


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / queue.VECTOR
        self.folder.mkdir()
        self.records = [{
            "id": str(i), "conversation": [
                {"role": "user", "content": "Question %d?" % i},
                {"role": "assistant", "content": "Answer"}],
            "contexts": [["Evidence"]], "ground_truth_answers": ["Gold"],
            "answerability": ["ANSWERABLE"],
        } for i in range(3)]
        self.input = self.folder / "input.json"
        self.input.write_text(json.dumps(self.records))
        self.output = self.folder / "out.mtrag.json"
        self.base = ["--input", str(self.input), "--output", str(self.output),
                     "--model", queue.MODEL, "--reasoning-max-tokens", "4096", "--max-tokens", "8192"]

    def args(self, adaptive=True, ceiling=32768):
        return judge.parse_args(self.base + (["--adaptive-max-tokens", str(ceiling)] if adaptive else []))

    def test_direct_cli_default_output_uses_mtrag_name(self):
        args = judge.parse_args(["--input", str(self.input)])
        self.assertEqual(args.output, self.input.with_name("input.mtrag-judged.json"))

    def test_hipporag_queue_runs_only_selected_backend(self):
        with mock.patch.object(queue, 'complete_run') as complete:
            queue.worker([], complete_all=True, rags=['hipporag'])
        complete.assert_called_once_with('hipporag', 32768, 3)

    @staticmethod
    def response(reasoning=100, completion=200, finish="stop"):
        return {"choices": [{"finish_reason": finish, "message": {"content": "OK"}}],
                "usage": {"completion_tokens": completion,
                          "completion_tokens_details": {"reasoning_tokens": reasoning}}}

    def evaluated(self, index, adaptive=False):
        _, records = judge.load_records(self.input)
        with mock.patch.object(judge, "judge_turn_mtrag", return_value={
                "label": "correct", "reasoning": "Supported", "scores": {}}):
            return judge.evaluate_one(index, records[index], self.args(adaptive))[1]

    def test_adaptive_retries_rejected_usage_with_larger_budgets(self):
        responses = [self.response(reasoning=9135, completion=8192),
                     self.response(reasoning=9000, completion=16384), self.response()]
        usage = {}
        with mock.patch.object(judge, "_post", side_effect=responses) as post, mock.patch.object(judge.time, "sleep"):
            self.assertEqual(judge.ask(self.args(), "Prompt", str, usage), "OK")
        payloads = [call.args[1] for call in post.call_args_list]
        self.assertEqual([p["max_tokens"] for p in payloads], [8192, 16384, 32768])
        self.assertEqual([p["reasoning"]["max_tokens"] for p in payloads], [4096, 8192, 16384])
        self.assertEqual(usage["max_requested_completion_tokens"], 32768)

    def test_adaptive_retries_length_response(self):
        with mock.patch.object(judge, "_post", side_effect=[
                self.response(finish="length"), self.response()]) as post, mock.patch.object(judge.time, "sleep"):
            self.assertEqual(judge.ask(self.args(), "Prompt", str, {}), "OK")
        self.assertEqual(post.call_count, 2)

    def test_fixed_budget_keeps_fail_fast_behavior(self):
        with mock.patch.object(judge, "_post", return_value=self.response(reasoning=9135)) as post:
            with self.assertRaisesRegex(RuntimeError, "exceeded requested reasoning budget"):
                judge.ask(self.args(adaptive=False), "Prompt", str, {})
        self.assertEqual(post.call_count, 1)

    def test_adaptive_never_exceeds_cap(self):
        args = self.args(ceiling=16384)
        args.retries = 4
        with mock.patch.object(judge, "_post", return_value=self.response(finish="length")) as post, mock.patch.object(judge.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "failed after 4 attempts"):
                judge.ask(args, "Prompt", str, {})
        self.assertEqual([c.args[1]["max_tokens"] for c in post.call_args_list], [8192, 16384, 16384, 16384])
        self.assertTrue(all(c.args[1]["reasoning"]["max_tokens"] < 16384 for c in post.call_args_list))

    def test_adaptive_requires_explicit_baseline_budgets(self):
        with self.assertRaises(SystemExit):
            judge.parse_args(["--input", str(self.input), "--adaptive-max-tokens", "32768"])

    def test_adaptive_resume_preserves_fixed_results_and_provenance(self):
        baseline = self.evaluated(0)
        self.output.write_text(json.dumps([baseline]))
        args = self.args()
        args.resume_from = self.output
        _, records = judge.load_records(self.input)
        reused = judge.load_completed_for_resume(args, records)
        self.assertEqual(reused["0"], baseline)
        self.assertNotIn("adaptive_max_tokens", reused["0"]["judge"])
        self.assertEqual(self.evaluated(1, adaptive=True)["judge"]["adaptive_max_tokens"], 32768)

    def test_fixed_resume_rejects_adaptive_results(self):
        self.output.write_text(json.dumps([self.evaluated(0, adaptive=True)]))
        args = self.args(adaptive=False)
        args.resume_from = self.output
        _, records = judge.load_records(self.input)
        with self.assertRaisesRegex(ValueError, "differs"):
            judge.load_completed_for_resume(args, records)

    def test_changed_adaptive_ceiling_is_rejected_on_resume(self):
        self.output.write_text(json.dumps([self.evaluated(0, adaptive=True)]))
        args = self.args(ceiling=16384)
        args.resume_from = self.output
        _, records = judge.load_records(self.input)
        with self.assertRaisesRegex(ValueError, "differs"):
            judge.load_completed_for_resume(args, records)

    def test_first_record_saved_before_later_failure(self):
        first = self.evaluated(0)
        saved = threading.Event()
        real_write = judge.write_output

        def evaluate(index, record, args):
            if index == 0:
                return 0, first
            if not saved.wait(timeout=3):
                raise AssertionError("First completed record was not saved promptly")
            self.assertTrue(self.output.is_file())
            raise RuntimeError("Synthetic failure")

        def checkpoint(*args):
            real_write(*args)
            saved.set()

        with mock.patch.object(judge, "evaluate_one", side_effect=evaluate), mock.patch.object(judge, "write_output", side_effect=checkpoint):
            code = judge.main(self.base)
        self.assertEqual(code, 1)
        self.assertTrue(saved.is_set())
        self.assertEqual(json.loads(self.output.read_text()), [first])

    def test_queue_recovers_only_missing_records_and_preserves_original(self):
        original = self.evaluated(0)
        self.output.write_text(json.dumps([original]))
        original_bytes = self.output.read_bytes()
        calls = []

        def run(command, **kwargs):
            source = Path(command[command.index("--resume-from") + 1])
            target = Path(command[command.index("--output") + 1])
            results = json.loads(source.read_text())
            calls.append(command)
            results.append(self.evaluated(len(results), adaptive=True))
            target.write_text(json.dumps(results))
            return subprocess.CompletedProcess(command, 0 if len(results) == 3 else 1)

        with mock.patch.object(queue, "RUNS", self.root), mock.patch.object(queue, "prepare_records", return_value=copy.deepcopy(self.records)), mock.patch.object(queue.subprocess, "run", side_effect=run):
            queue.complete_run("vector", 32768, 3)
        results = json.loads(self.output.read_text())
        self.assertEqual([r["id"] for r in results], ["0", "1", "2"])
        self.assertEqual(results[0], original)
        self.assertEqual(len(calls), 2)
        backup = list(self.folder.glob("out.mtrag.before_recovery_*.json"))
        self.assertEqual(len(backup), 1)
        self.assertEqual(backup[0].read_bytes(), original_bytes)

    def test_queue_stops_after_bounded_recovery_passes(self):
        self.output.write_text(json.dumps([self.evaluated(0)]))
        original_bytes = self.output.read_bytes()
        with mock.patch.object(queue, "RUNS", self.root), mock.patch.object(queue, "prepare_records", return_value=copy.deepcopy(self.records)), mock.patch.object(queue.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)) as run:
            with self.assertRaisesRegex(ValueError, "still has 2/3 missing"):
                queue.complete_run("vector", 32768, 2)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(self.output.read_bytes(), original_bytes)

    def checkpoint_record(self):
        record = copy.deepcopy(self.records[0])
        record["conversation"] = [message for i in range(3) for message in (
            {"role": "user", "content": "Question %d?" % i},
            {"role": "assistant", "content": "Answer %d" % i})]
        record["contexts"] = [["Evidence"]] * 3
        record["ground_truth_answers"] = ["Gold"] * 3
        record["answerability"] = ["ANSWERABLE"] * 3
        self.input.write_text(json.dumps([record]))
        _, loaded = judge.load_records(self.input)
        args = self.args()
        args.turn_checkpoints = True
        return args, loaded[0]

    def test_retry_reuses_completed_turns_and_accounts_for_failed_attempt_usage(self):
        args, record = self.checkpoint_record()
        args.api_key = "synthetic-secret-never-save"

        def first_attempt(args, turn, usage):
            usage["completion_tokens"] = usage.get("completion_tokens", 0) + 5
            if turn["question"] == "Question 1?":
                raise ValueError("Synthetic provider failure")
            return {"label": "correct", "reasoning": "Original judgment", "scores": {"value": 0.9}}

        with mock.patch.object(judge, "judge_turn_mtrag", side_effect=first_attempt) as call:
            with self.assertRaisesRegex(RuntimeError, "turn 2"):
                judge.evaluate_one(0, record, args)
        self.assertEqual(call.call_count, 2)
        checkpoint, saved = judge.turn_checkpoint(args, record)
        self.assertEqual(len(saved["turn_evaluations"]), 1)
        self.assertEqual(saved["usage"]["completion_tokens"], 10)
        self.assertNotIn("synthetic-secret-never-save", checkpoint.read_text())
        original_turn = copy.deepcopy(saved["turn_evaluations"][0])

        def retry(args, turn, usage):
            usage["completion_tokens"] += 5
            return {"label": "incorrect", "reasoning": "New judgment", "scores": {"value": 0.2}}

        with mock.patch.object(judge, "judge_turn_mtrag", side_effect=retry) as call:
            _, result = judge.evaluate_one(0, record, args)
        self.assertEqual(call.call_count, 2)
        self.assertEqual([c.args[1]["question"] for c in call.call_args_list], ["Question 1?", "Question 2?"])
        self.assertEqual(result["evaluation"]["turn_evaluations"][0], original_turn)
        self.assertEqual(result["judge"]["usage"]["completion_tokens"], 20)
        self.assertEqual(len(result["evaluation"]["turn_evaluations"]), 3)

    def test_complete_turn_checkpoint_reused_without_model_calls(self):
        args, record = self.checkpoint_record()
        with mock.patch.object(judge, "judge_turn_mtrag", return_value={
                "label": "correct", "reasoning": "Supported", "scores": {}}):
            _, first = judge.evaluate_one(0, record, args)
        with mock.patch.object(judge, "judge_turn_mtrag") as call:
            _, resumed = judge.evaluate_one(0, record, args)
        call.assert_not_called()
        self.assertEqual(resumed["evaluation"], first["evaluation"])

    def test_input_or_policy_change_does_not_reuse_turn_checkpoint(self):
        args, record = self.checkpoint_record()
        with mock.patch.object(judge, "judge_turn_mtrag", return_value={
                "label": "correct", "reasoning": "Supported", "scores": {}}):
            judge.evaluate_one(0, record, args)
        old_path, _ = judge.turn_checkpoint(args, record)
        modified = copy.deepcopy(record)
        modified["contexts"][0] = ["Different evidence"]
        new_path, saved = judge.turn_checkpoint(args, modified)
        self.assertNotEqual(new_path, old_path)
        self.assertEqual(saved, {})
        args.adaptive_max_tokens = 16384
        new_path, saved = judge.turn_checkpoint(args, record)
        self.assertNotEqual(new_path, old_path)
        self.assertEqual(saved, {})

    def test_corrupt_turn_checkpoint_is_rejected(self):
        args, record = self.checkpoint_record()
        with mock.patch.object(judge, "judge_turn_mtrag", return_value={
                "label": "correct", "reasoning": "Supported", "scores": {}}):
            judge.evaluate_one(0, record, args)
        path, saved = judge.turn_checkpoint(args, record)
        saved["turn_evaluations"][0]["turn"] = 99
        path.write_text(json.dumps(saved))
        with mock.patch.object(judge, "judge_turn_mtrag") as call:
            with self.assertRaisesRegex(ValueError, "Invalid turn checkpoint"):
                judge.evaluate_one(0, record, args)
        call.assert_not_called()

    def test_stop_rule_is_preserved_in_turn_checkpoint(self):
        args, record = self.checkpoint_record()
        args.stop_at_first_failure = True
        with mock.patch.object(judge, "judge_turn_mtrag", return_value={
                "label": "incorrect", "reasoning": "Unsupported", "scores": {}}) as call:
            _, first = judge.evaluate_one(0, record, args)
        self.assertEqual(call.call_count, 1)
        with mock.patch.object(judge, "judge_turn_mtrag") as call:
            _, resumed = judge.evaluate_one(0, record, args)
        call.assert_not_called()
        self.assertEqual(resumed["evaluation"], first["evaluation"])


if __name__ == "__main__":
    unittest.main()
