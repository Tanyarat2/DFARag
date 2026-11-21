# RAG failure classification

This folder contains the current classification and question-generation workflows. You can copy or
push the whole `failure_classification/` folder into another project. Datasets,
API credentials, caches, saved results, old baselines and unrelated historical tools are
not part of this source package.

## What each folder does

| Folder | Purpose | Where to start |
|---|---|---|
| `errorclassification/` | Classifier components grouped into named folders, each with its own README | `pipeline/classifier.py` |
| `questiongeneration/` | Your restored B1 subquestions, B2 follow-ups and diagnostic evaluation | `README.md` |
| `rag/` | Default Dragonball RAG adapter used by dynamic question generation | `dragonball_rag_adapter.py` |
| `evidence/` | Source/reconstruction helpers required by diagnostic search | `dragonball_reconstruction.py` |
| `conf/` | Editable model, failure-limit, path and run settings | `config-classification.yaml` |
| `experiments/` | The three current tools: run cases, review gold labels, build the viewer | `run_document_threshold_pilot.py` |
| `tests/` | Offline regression tests for this current workflow | Run the command below |

All E1–E16 classification components live in `errorclassification/`. Use `FailureClassifier` for
prepared cases/conversations, or `EvidenceFirstFailureClassifier` for dataset
`InputRecord` objects; both use the same engine and rules.

## Install and run

From the project containing this folder:

```bash
python -m pip install -r failure_classification/requirements.txt
```

Alternatively, install the package with `python -m pip install ./failure_classification`.
Set `OPENAI_API_KEY` and, if needed, `OPENAI_BASE_URL` in your environment. An
existing parent-project `.envrc` is also supported; credentials are not included
in this folder.

Edit `conf/config-classification.yaml` to choose the model, `max_failures`,
`max_passes`, input/output paths, case limit/IDs, workers and API connection.
Then run:

```bash
python -m failure_classification.experiments.run_document_threshold_pilot \
  --config failure_classification/conf/config-classification.yaml
```

The shipped config is also the default when `--config` is omitted. Relative
paths inside YAML start at the YAML file's directory. Paths passed as CLI flags
start at your working directory. CLI flags override YAML settings:

```bash
python -m failure_classification.experiments.run_document_threshold_pilot \
  --config failure_classification/conf/config-classification.yaml \
  --model gpt-4o-mini --max-failures 1 --max-passes 2 \
  --input /path/to/annotations.jsonl --output /path/to/results/run1
```

`max_failures: 1` selects the primary only; larger values allow that many
supported labels. `max_passes` controls attribution passes (1 or 2); the
answerability-first request is separate. Set `limit: null` or use `--all-cases`
for the full input. Each run writes `resolved_config.json` and records its
settings in the manifest. Use a new output directory for each run.

Review the original gold failure labels separately, then build the viewer:

```bash
python -m failure_classification.experiments.review_pilot_gold_labels \
  --run-dir /path/to/results/run1 \
  --input /path/to/annotations.jsonl

python -m failure_classification.experiments.build_document_threshold_viewer \
  --run-dir /path/to/results/run1
```

Open `/path/to/results/run1/viewer.html`. An older viewer such as `index5.html`
is optional historical context via `--historical-viewer`; it is not required.
The viewer is a local HTML file; `--site-directory` is optional for hosting.
Gold-label reviews are model-assisted judgments, distinct from classifier
predictions and not independently confirmed ground truth.

## Input data

The runner accepts JSONL or a JSON list. Original annotation rows supply:

- `query_id`, `question`, `answer` (reference answer), `rag_answer_final` (evaluated answer).
- `retrieved`: initial candidate chunks; `reranked`: actual generator-context chunks.
- `gold_documents`: reference documents; `indexed_reference_chunks`: indexed evidence.
- Optional history and provenance fields.

Chunks use `chunk_id` and `text`, with optional `doc_id`; reference documents
use `doc_id` and `text`. Missing evidence (`null`) is different from an observed
empty set (`[]`). The classifier can also accept prepared `case_id` /
`original_question` / `original_rag_answer` inputs directly.

Original `error_category` and `error_stage` are used only by the separate gold
review. They are removed from classification input. The runner stores the input
path in its manifest so review can locate it; `--input` can override that path.

## How classification works

1. Assess each requested aspect once using actual generator evidence and the
   existing RAG answer. Gold answers and failure labels are excluded from this step.
2. Python computes E9 when an unanswerable aspect is substantively answered
   without acknowledging insufficient information. This decision is frozen.
3. Evaluate chunking, retrieval, reranking, and E10–E16 using their own evidence.
4. Validate citations and perform one bounded attribution review if necessary.
5. Save predictions, first-stage assessments, audit records and document metrics.

E9 can coexist with E10/E13. E4/E7/E8 document thresholds are computed separately
from the causal primary. Unknown evidence remains unresolved. The workflow
prevents contradictory second E9 votes; it does not independently establish
that the model interpreted all source evidence correctly.

## Tests

Run from the parent project, or from anywhere after installing the package:

```bash
python -m unittest discover -s failure_classification/tests -t . -q
```

The tests use offline model stubs; they do not make live API calls.

## Folder organization

Inside `errorclassification/`, code is separated into `pipeline`, `answerability`,
`prompts`, `taxonomy`, `evidence`, `validation`, `review`, `data`, `reporting`,
`api`, and `configuration`. Each README explains the files in that folder.
Use the public imports shown in the classifier README; old direct flat-module
imports now use the corresponding subfolder path.

## Question generation restored

`questiongeneration/` was moved to the project-level legacy archive during the
previous cleanup. It is restored as active source, together with the `rag/` and
`evidence/` helpers its dynamic workflow needs. See `questiongeneration/README.md`
for B1 and B2 commands, inputs and optional runtime dependencies. Install Excel
example support with `python -m pip install './failure_classification[excel]'`.
The built-in Dragonball adapter requires the separate RAG project and its assets;
a custom RAG adapter can be supplied instead.

Classification consumes supplied evidence/diagnostic records; it does not
silently run question generation. The six-category dynamic evaluator and the
E1–E16 classifier are separate workflows.

## Historical archive

Older baselines/comparisons, annotation app, configs, caches and saved results
remain outside this source folder in
`../legacy/failure_classification/project_cleanup_20261108/`.
Its manifest preserves original paths and hashes. Restored question-generation
files are active here; archive copies remain as historical snapshots.
`../legacy/failure_classification/organization_20261108/` records the module moves
and the restoration. `.gitignore` excludes secrets and generated outputs.
