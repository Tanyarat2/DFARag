"""Import original MTRAG reference evidence for judging and annotation, without altering runs."""

import argparse
import hashlib
import json
import pathlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent / "gold_support.json")
    args = parser.parse_args()
    raw = args.reference.read_bytes()
    tasks = {}
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        task_id = task["task_id"]
        if task_id in tasks:
            raise ValueError("Duplicate reference task: " + task_id)
        questions = [m["text"] for m in task["input"] if m["speaker"] == "user"]
        tasks[task_id] = {
            "question": questions[-1] if questions else "",
            "gold_answers": [t["text"] for t in task.get("targets", [])],
            "documents": [{
                "id": c.get("document_id", ""), "title": c.get("title", ""),
                "text": c["text"],
                "relevance": sorted({v.get("value", "unknown") for v in
                                     c.get("feedback", {}).get("relevant", {}).values()}),
            } for c in task.get("contexts", [])],
        }
    payload = {
        "source": {"file": args.reference.name,
                   "sha256": hashlib.sha256(raw).hexdigest(),
                   "dataset_url": "https://github.com/IBM/mt-rag-benchmark/tree/main/mtrag-human"},
        "tasks": tasks,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("Imported %d reference tasks to %s" % (len(tasks), args.output))


if __name__ == "__main__":
    main()
