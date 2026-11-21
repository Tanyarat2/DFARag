# Failure data

This dataset contains **278 turns labelled by all three annotators**, stored as
an indented JSON array in one file per RAG method. Each turn is a separate object
with two-space indentation for easy reading and folding in an editor.
It is an export of existing evidence; building it makes
no model calls and does not regenerate answers or retrieval results.

| File | Labelled by all three | Removed turns | Retained turns |
| --- | ---: | ---: | ---: |
| `Vector.json` | 101 | 13 | 88 |
| `HippoRag.json` | 74 | 5 | 69 |
| `Raptor.json` | 24 | 0 | 24 |
| `SelfRAG.json` | 104 | 7 | 97 |
| Total | 303 | 25 | 278 |

## Selection

Start with the intersection of turns that have an explicit saved annotation
decision from **each of PS, KH, and WN** in `Annotate_result/<method>/`. Omit a
turn if any annotator has a missing or undecided annotation. Review marks alone
do not qualify a turn. This omits 22 incompletely labelled turns from the earlier
export: Vector 9, HippoRag 6, Raptor 1, and SelfRAG 6.

Within that fully labelled cohort, remove a turn only if **all three annotators**
marked it `not_a_failure` or `excluded`, in any combination. A correctness-only
`correct` decision also counts as `not_a_failure`.

All three annotators must provide a decision; their labels do not have to agree.
Disputed turns are retained. Insufficient-evidence and outside-taxonomy decisions
also remain unless all three people meet the removal rule. This dataset is a
screened review cohort, not an adjudicated set of confirmed failure types.

Records contain no annotator identities, failure labels, notes, review flags,
human correctness judgments, or human gold assessments. Machine judge results
remain under `judge_evaluation` and `judge_metadata`.

## Record contents

| Fields | Contents |
| --- | --- |
| `case_id`, `method`, `run`, `id`, `conversation_number` | Stable method/conversation identifiers and source run |
| `turn`, `dataset_turn`, `task_id`, `domain` | Turn position and reference-task identifiers |
| `question`, `answer` | Original evaluated question and RAG answer |
| `history` | All messages preceding this turn, from the evaluated run input |
| `conversation` | History plus the current question and answer; excludes future turns |
| `gold`, `gold_answers`, `answerability` | Run gold answer, reference answer variants, and recorded answerability |
| `gold_documents`, `gold_support` | Reference passages, IDs, titles, relevance, source details, and evidence status |
| `contexts`, `selected_passages` | Exact original generator context, with passage metadata and order |
| `generation_trace` | Exact matching source turn, including recorded history, rewritten query, model, retrieval flag, and available generation settings |
| `retrieved_documents` | Candidate document text and metadata from the later retrieval experiment |
| `retrieval_experiment`, `reranking_experiment` | Complete matching experiment records and source file hashes |
| `original_retrieved_candidates`, `original_reranking_trace` | `null`: these original generation traces are unavailable |
| `input_metadata`, `judge_evaluation`, `judge_metadata`, `provenance` | Other recorded run metadata, machine evaluation, and source paths |

Retrieval and reranking experiments were performed separately from original answer
generation. Their documents are included for inspection, but they do not establish
which candidates or reranker produced the original answer. The actual original
selected context is in `contexts` and `selected_passages`.

Gold documents contain the complete reference passage text saved in
`judge_run/gold_support.json`. Full parent source documents are not installed
locally, so their `full_document` fields are `null`. An empty reference-document
list can be valid for an unanswerable task. All 278 exported turns have
`gold_support.status` equal to `available`.

`manifest.json` records selection rules, source hashes, output hashes, counts,
and evidence coverage. Each retained turn has a matching original generation
trace and matching retrieval and reranking experiment records.

## Rebuild

From the repository root:

```powershell
python -B annotate_server/build_failure_data.py --output Failure_data_new
```

Choose a new output folder to preserve the existing snapshot. The builder refuses
to replace existing dataset files unless `--replace` is explicitly passed. To
refresh this folder, run `python -B annotate_server/build_failure_data.py --replace`.
Source runs, conversation exports, and human
annotation files are read without modification.

Validate an export against its source files:

```powershell
python -B annotate_server/validate_failure_data.py
```

Validation passed for all 278 records: a saved decision from all three annotators,
cohort membership, unique IDs, absence of
human annotation fields, source/output hashes, history boundaries, original
answers and contexts, gold evidence, and experiment record identity.
