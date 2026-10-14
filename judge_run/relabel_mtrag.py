"""Apply a new screening cutoff to saved mtRAG scores, preserving a byte-for-byte backup.

No model calls are made. Runs with skipped turns need a separate judging run and
are rejected here because the missing scores cannot be reconstructed.
"""

import argparse
import json
import math
import pathlib
import re


def relabel(path, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    raw = path.read_bytes()
    records = json.loads(raw.decode("utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("expected a non-empty array of judged records")
    if all(r["judge"]["thresholds"]["mtrag"] == threshold for r in records):
        print("Already uses threshold %s: %s" % (threshold, path))
        return
    for record in records:
        evaluation = record["evaluation"]
        turns = evaluation["turn_evaluations"]
        if record["judge"].get("turn_rule") == "stop_at_first_failure":
            raise ValueError("cannot relabel a stop-at-first-failure run")
        for turn in turns:
            score = turn.get("scores", {}).get("conditioned_score")
            if (turn["label"] not in ("correct", "incorrect") or
                    isinstance(score, bool) or not isinstance(score, (int, float)) or
                    not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("all turns must have completed, valid conditioned scores")
    backup = path.with_name(path.stem + ".before-threshold-" +
                            str(threshold).replace(".", "p") + path.suffix)
    for record in records:
        judge = record["judge"]
        previous = judge["thresholds"]["mtrag"]
        judge["thresholds"]["mtrag"] = threshold
        judge["screening_relabel"] = {"previous_threshold": previous,
                                      "new_threshold": threshold,
                                      "source_backup": backup.name}
        evaluation = record["evaluation"]
        for turn in evaluation["turn_evaluations"]:
            label = "correct" if turn["scores"]["conditioned_score"] >= threshold else "incorrect"
            turn["label"] = label
            turn["scores"]["metric_verdicts"]["conditioned"] = label
            turn["reasoning"] = re.sub(r"\(threshold [0-9.]+\)",
                                       "(threshold %.2f)" % threshold, turn.get("reasoning", ""))
        first = next((t["turn"] for t in evaluation["turn_evaluations"]
                      if t["label"] == "incorrect"), None)
        evaluation["first_failed_turn"] = first
        evaluation["conversation_label"] = "incorrect" if first is not None else "correct"
    with backup.open("xb") as stream:
        stream.write(raw)
    temporary = path.with_name(path.name + ".relabel.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(records, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)
    turns = [t for r in records for t in r["evaluation"]["turn_evaluations"]]
    bad = sum(t["label"] == "incorrect" for t in turns)
    print("%s: correct=%d incorrect=%d; backup=%s" %
          (path.parent.name, len(turns) - bad, bad, backup.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    relabel(args.input, args.threshold)


if __name__ == "__main__":
    main()
