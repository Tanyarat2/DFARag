"""Build an offline, read-only comparison of exported annotator results."""
import argparse
import hashlib
import json
from pathlib import Path

from make_annotation import build_data
from failure_taxonomy import DEFINITIONS
from conversation_numbers import registry

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"Vector": "convos_vector_q3_32b", "HippoRag": "convos_hipporag_q3_32b",
        "Raptor": "convos_raptor_q3_32b", "SelfRAG": "convos_selfrag_q3_32b"}
ANNOTATORS = ("PS", "KH", "WN")


def labelled(record):
    # Review completion and flags are independent of a saved decision.
    return bool(record.get("failure_types") or record.get("failure_type") or
                record.get("human_label"))


def skip_incorrect(record):
    types = record.get('failure_types') or [record.get('failure_type')]
    return len(types) == 1 and types[0] in ('not_a_failure', 'excluded')


def annotation_version(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def read_annotations(folder, run):
    profiles, warnings = {name: {} for name in ANNOTATORS}, []
    for path in sorted(folder.glob("*.json")):
        if not path.name.startswith("annotations_" + run + "_"):
            warnings.append(f"Skipped {folder.name}/{path.name}: filename belongs to another run.")
            continue
        name = path.stem.rsplit("_", 1)[-1]
        if name not in profiles:
            warnings.append(f"Skipped {path.name}: unknown annotator.")
            continue
        for record in json.loads(path.read_text(encoding="utf-8-sig")):
            if record.get("annotator", name) != name:
                raise ValueError(f"Annotator mismatch in {path}")
            key = f'{record["id"]}#{int(record["turn"])}'
            if key in profiles[name]:
                raise ValueError(f"Duplicate annotation {key} in {path}")
            profiles[name][key] = record
    return profiles, warnings


def build_review(results, runs, load_data=None):
    rows, warnings, sources = [], [], []
    for method, run in RUNS.items():
        profiles, messages = read_annotations(results / method, run)
        warnings.extend(messages)
        data = load_data(runs / run) if load_data else build_data(runs / run, live=True)
        sources.append({"method": method, "judge": data["coverage"]["mtrag"]["source"]})
        first_incorrect = {}
        targets = {name: {} for name in ANNOTATORS}
        for row in sorted(data['rows'], key=lambda row: row['turn']):
            if row['judges']['mtrag']['label'] == 'incorrect':
                first_incorrect[row['id']] = min(first_incorrect.get(row['id'], row['turn']), row['turn'])
                for name in ANNOTATORS:
                    if not skip_incorrect(profiles[name].get(row['key'], {})):
                        targets[name].setdefault(row['id'], row['turn'])
        for row in sorted(data["rows"], key=lambda row: (
                int(row["conversation_number"][1:]), row["turn"])):
            annotations = {name: profiles[name].get(row["key"], {}) for name in ANNOTATORS}
            count = sum(labelled(record) for record in annotations.values())
            is_first = first_incorrect.get(row['id']) == row['turn']
            queue_annotators = [name for name in ANNOTATORS
                                if targets[name].get(row['id']) == row['turn']]
            if not count and not queue_annotators:
                continue
            rows.append({**{k: row[k] for k in (
                "id", "conversation_number", "turn", "question", "answer", "gold", "gold_support",
                "retrieval_context", "selected_passages", "contexts", "corpus_collection")},
                "key": method + ":" + row["key"], "method": method,
                "judge_label": row["judges"]["mtrag"]["label"],
                "first_incorrect": is_first,
                "queue_annotators": queue_annotators,
                "annotations": annotations, "labelled_count": count,
                "annotation_versions": {name: annotation_version(annotations[name] or None)
                                        for name in ANNOTATORS}})
    return {"rows": rows, "warnings": warnings, "sources": sources,
            "annotators": ANNOTATORS,
            "labels": {code: value["name"] for code, value in DEFINITIONS.items()}}


def annotation_snapshot(data):
    snapshot = {"rows": [{key: row[key] for key in (
        'key', 'id', 'turn', 'method', 'conversation_number', 'annotations',
        'annotation_versions', 'labelled_count', 'judge_label', 'first_incorrect', 'queue_annotators')} for row in data['rows']],
        "warnings": data['warnings'], "sources": data['sources']}
    snapshot['version'] = annotation_version(snapshot)
    return snapshot


def render(data):
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(
        "\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return Path(__file__).with_name("result_review_template.html").read_text(
        encoding="utf-8").replace("__REVIEW_DATA__", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "Annotate_result")
    parser.add_argument("--runs", type=Path, default=ROOT / "judge_run" / "runs")
    parser.add_argument("--output", type=Path, default=ROOT / "Annotate_result" / "review.html")
    args = parser.parse_args()
    data = build_review(args.results, args.runs)
    args.output.write_text(render(data), encoding="utf-8")
    args.output.with_name('conversation_numbers.json').write_text(
        json.dumps(registry(), indent=2) + '\n', encoding='utf-8')
    print(f"Created {args.output}")
    for method in RUNS:
        rows = [r for r in data["rows"] if r["method"] == method]
        print(f"{method}: {sum(r['first_incorrect'] for r in rows)} first incorrect, "
              f"{sum(r['labelled_count'] > 0 for r in rows)} annotated, {len(rows)} review turns")
    for warning in data["warnings"]:
        print(warning)


if __name__ == "__main__":
    main()
