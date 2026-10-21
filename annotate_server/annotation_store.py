"""Separate, atomic annotation files for the three human reviewers."""

import datetime
import json
import os
import pathlib
import re
import tempfile
import threading

ANNOTATORS = ("PS", "KH", "WN")
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
FIELDS = {"human_label", "human_score_band", "failure_type", "failure_types", "notes", "review_complete",
          "gold_assessment", "gold_notes", "evidence"}
CHOICES = {
    "human_label": {"", "correct", "incorrect"},
    "human_score_band": {"", "low", "mid", "high"},
    "failure_type": {"", "knowledge_boundary", "chunking", "retrieval", "context_selection",
                     "grounding", "response_coverage", "not_a_failure", "outside_taxonomy",
                     "insufficient_evidence"},
    "gold_assessment": {"", "supported", "questionable", "incorrect"},
}


def validate_changes(changes, valid_keys):
    if not isinstance(changes, dict):
        raise ValueError("changes must be an object keyed by turn.")
    for key, entry in changes.items():
        if key not in valid_keys:
            raise ValueError("An annotation turn does not belong to this run.")
        if not isinstance(entry, dict) or set(entry) - FIELDS:
            raise ValueError("Annotation fields are invalid.")
        for field, value in entry.items():
            if field == "review_complete":
                if not isinstance(value, bool):
                    raise ValueError("review_complete must be true or false.")
            elif field == "failure_types":
                allowed = CHOICES["failure_type"] - {""}
                if (not isinstance(value, list) or
                        any(not isinstance(item, str) or item not in allowed for item in value) or
                        len(set(value)) != len(value)):
                    raise ValueError("failure_types must be a list of distinct valid failure choices.")
                if len(value) > 1 and set(value) & {"not_a_failure", "outside_taxonomy", "insufficient_evidence"}:
                    raise ValueError("Alternative outcomes must be selected on their own.")
                legacy = value[0] if len(value) == 1 else ""
                if "failure_type" in entry and entry["failure_type"] != legacy:
                    raise ValueError("failure_type must match the single selection, or be empty for multiple choices.")
            elif field == "evidence":
                if not isinstance(value, list) or any(
                    not isinstance(item, dict) or set(item) != {"source", "locator", "quote"} or
                    any(not isinstance(item[name], str) for name in ("source", "locator", "quote"))
                    for item in value
                ):
                    raise ValueError("Evidence needs source, locator, and quote strings.")
            elif not isinstance(value, str) or field in CHOICES and value not in CHOICES[field]:
                raise ValueError("Invalid annotation value for " + field + ".")


class AnnotationStore:
    def __init__(self, runs_dir):
        self.runs_dir = pathlib.Path(runs_dir).resolve()
        self.lock = threading.RLock()

    def path(self, run, annotator):
        if not RUN_NAME.fullmatch(run) or annotator not in ANNOTATORS:
            raise ValueError("Choose a valid run and annotator (PS, KH, or WN).")
        work_dir = (self.runs_dir / run).resolve()
        if not work_dir.is_relative_to(self.runs_dir) or not work_dir.is_dir():
            raise FileNotFoundError("Run not found.")
        target = (work_dir / "annotations" / (annotator + ".json")).resolve()
        if not target.is_relative_to(work_dir):
            raise ValueError("Annotation path must stay within the run folder.")
        return target

    def read(self, run, annotator):
        with self.lock:
            path = self.path(run, annotator)
            if not path.exists():
                return {"schema": 1, "run": run, "annotator": annotator, "annotations": {}}
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise RuntimeError("Saved annotations could not be read; the file was left unchanged.") from exc
            if (not isinstance(data, dict) or data.get("run") != run or
                    data.get("annotator") != annotator or not isinstance(data.get("annotations"), dict)):
                raise RuntimeError("Saved annotation ownership is invalid; the file was left unchanged.")
            return data

    def patch(self, run, annotator, changes, valid_keys):
        validate_changes(changes, valid_keys)
        with self.lock:
            data = self.read(run, annotator)
            if not changes:
                return data
            data["annotations"].update(changes)
            data["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            target = self.path(run, annotator)
            target.parent.mkdir(exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                                 prefix=annotator + ".", suffix=".tmp", delete=False) as stream:
                    temporary = pathlib.Path(stream.name)
                    json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write("\n")
                os.replace(temporary, target)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
            return data
