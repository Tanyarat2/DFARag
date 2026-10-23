"""Startup and preservation checks; never call a judge API."""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_input
import run as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="judge-runner-")
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        (self.root / "conversation").mkdir()
        self.source = self.root / "conversation" / "sample.json"
        self.export = {"rag_name": "vector", "conversations": [{
            "id": "abc", "domain": "clapnq", "turns": [{
                "task_id": "abc<::>1", "dataset_turn": 1, "question": "Question?",
                "answer": "Answer", "answered": True, "retrieved": True,
                "passages": [{"text": "Selected evidence"}], "answerability": ["ANSWERABLE"],
            }],
        }]}
        self.source.write_text(json.dumps(self.export), encoding="utf-8")
        self.gold = self.root / "gold_support.json"
        self.gold.write_text(json.dumps({"tasks": {"abc<::>1": {
            "question": "Question?", "gold_answers": ["Gold"], "documents": [],
        }}}), encoding="utf-8")
        self.runs = self.root / "runs"
        self.old = self.runs / "sample_q3_32b"
        for patch in (mock.patch.object(runner, "ROOT", self.root),
                      mock.patch.object(prepare_input, "DEFAULT_REFERENCE", self.root / "missing.jsonl"),
                      mock.patch.object(prepare_input, "GOLD_SUPPORT", self.gold),
                      mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True)):
            patch.start()
            self.addCleanup(patch.stop)

    def invoke(self, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main(list(args), runs_dir=self.runs)

    def existing_run(self):
        self.old.mkdir(parents=True)
        (self.old / "out.mtrag.json").write_bytes(b"old judge output")
        (self.old / "annotations").mkdir()
        (self.old / "annotations" / "PS.json").write_bytes(b"old human annotations")

    def test_one_command_prepares_and_uses_previous_qwen3_settings(self):
        with mock.patch.object(runner, "run_mtrag") as judge:
            self.invoke("judge", "--src", "sample")
        records = json.loads((self.old / "input.json").read_text(encoding="utf-8"))
        self.assertEqual(records[0]["ground_truth_answers"], ["Gold"])
        self.assertEqual(records[0]["contexts"], [["Selected evidence"]])
        args = judge.call_args.args
        self.assertEqual(args[:2], (self.old / "input.json", self.old / "out.mtrag.json"))
        self.assertEqual(args[4:6], (8192, 4096))
        self.assertEqual(args[6], ["--base-url", "https://openrouter.ai/api/v1",
                                   "--model", "qwen/qwen3-32b", "--api-key-env", "OPENROUTER_API_KEY"])
        self.assertFalse((self.old / "reference.jsonl").exists())

    def test_source_filename_and_explicit_path(self):
        self.assertEqual(runner.resolve_source("sample.json"), self.source.resolve())
        self.assertEqual(runner.resolve_source(str(self.source)), self.source.resolve())

    def test_nested_source_and_ambiguous_filename(self):
        folder = self.root / 'conversation' / 'vector'
        folder.mkdir()
        nested = folder / self.source.name
        self.source.rename(nested)
        self.assertEqual(runner.resolve_source('sample'), nested.resolve())
        self.assertEqual(runner.resolve_source('vector/sample.json'), nested.resolve())
        other = self.root / 'conversation' / 'other'
        other.mkdir()
        (other / nested.name).write_text(nested.read_text(encoding='utf-8'), encoding='utf-8')
        with self.assertRaisesRegex(SystemExit, 'Ambiguous conversation filename'):
            runner.resolve_source('sample')

    def test_child_command_receives_model_and_budgets(self):
        with mock.patch.object(runner.subprocess, "run") as child:
            self.invoke("judge", "--src", "sample")
        command = child.call_args.args[0]
        self.assertEqual(pathlib.Path(command[1]).name, "judge_mtrag.py")
        self.assertEqual(command[command.index("--model") + 1], "qwen/qwen3-32b")
        self.assertEqual(command[command.index("--reasoning-max-tokens") + 1], "4096")
        self.assertEqual(command[command.index("--max-tokens") + 1], "8192")
        self.assertEqual(command[command.index("--mode") + 1], "mtrag")

    def test_dry_run_never_writes_or_calls_judge(self):
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(runner, "run_mtrag") as judge:
            self.invoke("judge", "--src", "sample", "--dry-run")
        judge.assert_not_called()
        self.assertFalse(self.runs.exists())

    def test_prepare_and_preview_use_shared_input_and_renamed_judge_without_api(self):
        with mock.patch.object(runner.subprocess, "run") as child:
            self.invoke("prepare", "--src", "sample", "--name", "prepared")
        child.assert_not_called()
        folder = self.runs / "prepared"
        before = (folder / "input.json").read_bytes()
        records = json.loads(before)
        self.assertEqual(records[0]["contexts"], [["Selected evidence"]])
        self.assertEqual(records[0]["ground_truth_answers"], ["Gold"])
        with mock.patch.dict("os.environ", {"JUDGE_MODEL": runner.QWEN3}):
            self.invoke("preview", "--name", "prepared")
        preview = json.loads((folder / "preview.json").read_text())
        self.assertEqual(preview["mode"], "mtrag")
        self.assertEqual(preview["record_id"], "abc")
        self.assertEqual((folder / "input.json").read_bytes(), before)
        self.assertFalse((folder / "out.mtrag.json").exists())

    def test_invalid_prepare_does_not_create_partial_run(self):
        self.export["conversations"][0]["turns"][0]["question"] = "Mismatched question"
        self.source.write_text(json.dumps(self.export))
        with self.assertRaisesRegex(SystemExit, "Preparation failed"):
            self.invoke("prepare", "--src", "sample", "--name", "invalid")
        self.assertFalse((self.runs / "invalid").exists())

    def test_fresh_archives_judge_and_human_annotations(self):
        self.existing_run()
        with mock.patch.object(runner, "run_mtrag"):
            self.invoke("judge", "--src", "sample", "--fresh")
        backups = list(self.runs.glob("sample_q3_32b_backup_*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "out.mtrag.json").read_bytes(), b"old judge output")
        self.assertEqual((backups[0] / "annotations/PS.json").read_bytes(), b"old human annotations")
        self.assertTrue((self.old / "input.json").is_file())

    def test_existing_run_requires_explicit_fresh(self):
        self.existing_run()
        with self.assertRaisesRegex(SystemExit, "--fresh"), mock.patch.object(runner, "run_mtrag") as judge:
            self.invoke("judge", "--src", "sample")
        judge.assert_not_called()
        self.assertEqual((self.old / "out.mtrag.json").read_bytes(), b"old judge output")

    def test_missing_key_does_not_archive_existing_run(self):
        self.existing_run()
        with mock.patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(SystemExit, "OPENROUTER_API_KEY"):
            self.invoke("judge", "--src", "sample", "--fresh")
        self.assertTrue((self.old / "out.mtrag.json").exists())
        self.assertEqual(list(self.runs.glob("*_backup_*")), [])

    def test_mismatched_reference_preserves_existing_run(self):
        self.existing_run()
        self.export["conversations"][0]["turns"][0]["question"] = "Different task"
        self.source.write_text(json.dumps(self.export), encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "does not match"):
            self.invoke("judge", "--src", "sample", "--fresh")
        self.assertTrue((self.old / "out.mtrag.json").exists())

    def test_missing_reference_task_is_rejected(self):
        self.gold.write_text('{"tasks": {}}', encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "no MTRAG reference"):
            self.invoke("judge", "--src", "sample")
        self.assertFalse(self.runs.exists())

    def test_duplicate_conversations_do_not_archive_existing_run(self):
        self.existing_run()
        self.export["conversations"] *= 2
        self.source.write_text(json.dumps(self.export), encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "Duplicate conversation ID"):
            self.invoke("judge", "--src", "sample", "--fresh")
        self.assertTrue((self.old / "out.mtrag.json").exists())

    def test_explicit_missing_reference_does_not_silently_fall_back(self):
        with self.assertRaisesRegex(SystemExit, "Gold answers not found"):
            self.invoke("judge", "--src", "sample", "--reference", str(self.root / "absent.jsonl"))

    def test_original_reference_supplies_missing_answerability(self):
        self.export["conversations"][0]["turns"][0].pop("answerability")
        reference = self.root / "original.jsonl"
        reference.write_text(json.dumps({"task_id": "abc<::>1", "Answerability": ["PARTIAL"],
                                        "targets": [{"text": "Original gold"}],
                                        "input": [{"speaker": "user", "text": "Question?"}]}), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            records = prepare_input.records_from_rag_export(self.export, reference)
        self.assertEqual(records[0]["ground_truth_answers"], ["Original gold"])
        self.assertEqual(records[0]["answerability"], ["PARTIAL"])

    def test_legacy_mtrag_uses_existing_input(self):
        self.old.mkdir(parents=True)
        (self.old / "input.json").write_text("[]", encoding="utf-8")
        with mock.patch.object(runner, "run_mtrag") as judge:
            self.invoke("mtrag", "--name", "sample_q3_32b", "--reasoning-max-tokens", "4096",
                        "--max-tokens", "8192")
        self.assertEqual(judge.call_args.args[4:6], (8192, 4096))
        self.assertEqual(len(judge.call_args.args), 6)


if __name__ == "__main__":
    unittest.main()
