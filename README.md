# Dynamic follow-up questions for RAG failure attribution

This project investigates whether dynamic follow-up questions identify the failure
type of an original RAG answer more accurately than static sub-question generation.
The design follows the [project report](doc/Report_Dynamic_Follow_up_Questions_for_Attributing_Failures_of_RAG.pdf)
and [presentation](doc/PPT_Dynamic_Follow_up_Questions_for_Attributing_Failures_of_RAG.pdf).

The implemented tools prepare conversation exports, judge answer quality, and
collect independent human annotations. B0/B1/B2 below describes the proposed
attribution experiment; live probing and the final comparison are not implemented.

## Quick start

Run these commands from the repository root with Python 3. The tools use the
standard library; no package installation is required. Review and offline checks
make no judge API calls.

Start human review:

```bash
python3 -B annotate_server/annotation_server.py
```

Open <http://127.0.0.1:8765/>. With VS Code Remote SSH, forward remote port **8765**
to local port **8765**. Select a run and your PS, KH, or WN profile. Judge assessments
and gold evidence start collapsed for each turn and reviewer. Live annotations
save separately under `judge_run/runs/<name>/annotations/`.
The sidebar separates failure-label progress (only each conversation's first
judge-marked incorrect turn) from review progress (all turns). **Reviewed this turn**
marks any turn independently; selecting one or more failure checkboxes on the
first incorrect turn counts a conversation label and turns its header green.
The compact sidebar gives conversations readable names from their first questions
alongside the original IDs. Search by name, conversation number, ID, or question.

Watch judge progress from another terminal:

```bash
tail -n 100 -f judge_run/runs/remaining_judges.log
```

Generate a file to send to another annotator:

```bash
python3 -B annotate_server/make_annotation.py --work-dir judge_run/runs/convos_vector_q3_32b
```

Send the generated `judge_run/runs/convos_vector_q3_32b/annotate.html`. Your friend
opens it in a browser without a server, chooses their profile, and returns an
**Export JSON** file. Offline drafts live in the browser, not in the HTML file;
export regularly. The updated `annotate_share_ibm.html` in that run includes the new
layout and official corpus excerpts. The generator refuses to overwrite existing pages; use `--output`
with a new filename when regenerating.

See the [annotation guide](annotate_server/README.md) for the review interface,
exports/imports, and comparing independent annotations. See the
[judging guide](judge_run/README.md) for preparation, API configuration, token
budgets, recovery, and the authoritative score definitions.

## Repository layout and maintenance

| Path | Purpose |
| --- | --- |
| `conversation/` | Original RAG exports; retain for provenance |
| `corpora/` | Local IBM passage indexes and [corpus setup/provenance](corpora/README.md) |
| `doc/` | Project report and presentation |
| `judge_run/` | Judge CLI, preparation, recovery, relabelling, and tests |
| `judge_run/gold_support.json` | Shared gold answers and reference documents |
| `judge_run/runs/<name>/` | Inputs, canonical results, recovery files, checkpoints, and human annotations |
| `annotate_server/` | Live server, static HTML generator/template, annotation storage, and tests |

Judging code stays in `judge_run/`; human review code stays in `annotate_server/`.
`judge_run/judge_mtrag.py` owns mtRAG scoring and resume/checkpoint validation.
Both judge and prepare stages call the shared `prepare_input.py` functions.
Live page summaries and HTTP routes stay in `annotation_server.py`; official and
archive corpus search share `corpus_store.py` and one stop-word list.
Recovery scripts and their tests are source files and must stay visible to Git.
Python caches, generated annotation HTML, queue logs, and lock files are ignored.
Ignoring generated files does not remove them from disk.

Keep the remaining modules separate by purpose: input/reference preparation,
paid judging/recovery, saved-score relabelling, annotation storage, HTML generation,
and corpus indexing each have independent callers or command-line workflows.
Relabelling and reference import remain explicit tools, rather than side effects
of starting the judge or review server. The test files cover those boundaries.

Retain historical runs, threshold backups, recovery outputs, and turn checkpoints:
they preserve paid work and provenance. Older pilot runs are not interchangeable
with complete runs. Keep per-annotator files and original exports, even when names
or bytes look similar. Only Python caches are disposable automatically.

## Offline checks

```bash
python3 -B -m unittest discover -s judge_run -p 'test_*.py' -v
python3 -B -m unittest discover -s annotate_server -p 'test_*.py' -v
```

The tests use temporary runs and synthetic judge responses. Browser checks run
when a compatible browser is installed and otherwise skip. If your environment
routes local requests through a proxy, set `NO_PROXY=127.0.0.1,localhost` and
`no_proxy=127.0.0.1,localhost` before running the checks.

## Research question and experimental conditions

**Can dynamic follow-up questions attribute failures of RAG more accurately than
static sub-question generation?**

Every condition investigates the same original failed turn. The failure classifier,
taxonomy, original evidence and human reference labels stay fixed.

| Condition | Additional diagnostic evidence | Role |
| --- | --- | --- |
| B0 | None: original Q-A log, conversation history, selected context and permitted reference evidence only | No-probing baseline |
| B0+B1 | Static core/background/follow-up sub-questions and coverage checks against the original context and answer | Xie-style static coverage baseline from the report |
| B0+B2 | Follow-up questions selected sequentially from the original problem and previous probe retrievals and answers | Proposed adaptive diagnosis |

Static coverage is an adaptation of [Xie et al.](https://arxiv.org/abs/2410.15531).
Sequential questioning is inspired by [LLM-as-an-Interviewer](https://arxiv.org/abs/2412.10424).
Extending these ideas to the project's six RAG failure types is the proposed
contribution, not an established improvement over B1.

## B2: sequential diagnostic follow-ups

1. Confirm an observable issue in the original answer: a contradiction, unsupported
   claim, wrong entity, missing essential facet or inappropriate refusal. A low judge
   score alone does not establish a factual error, and not every gold-answer detail
   is mandatory.
2. Keep competing hypotheses across all six categories. Do not commit to a failure
   type before probing. Allow the screening decision or reference answer to be wrong.
3. Generate one focused, non-leading question that distinguishes plausible
   explanations. Do not give the expected answer to the RAG system.
4. Ask the original RAG backend and record the question, retrieved document IDs and
   full text, answer, conversation state and backend/index configuration.
5. Use that observation to choose the next question within a predefined probe budget.
   Retain unresolved hypotheses; two questions do not guarantee coverage of all causes.
6. Give the original evidence and raw probe results to the same final classifier used
   by B0 and B1. Planner hypotheses are provisional and must not become gold labels.

A corrected second answer demonstrates recoverability under that probe. It does
not establish why the original answer failed. Original traces remain necessary.

| Failure type | Evidence needed to support attribution |
| --- | --- |
| `knowledge_boundary` | Original corpus/verified conversation scope and evidence that the request was unsupported |
| `chunking` | Original source evidence and indexed chunks showing loss of a usable evidence unit |
| `retrieval` | Usable original indexed evidence and the original retrieved candidate pool |
| `context_selection` | Original candidates compared with context actually supplied to generation |
| `grounding` | Original answer compared with selected evidence and the established conversational request |
| `response_coverage` | Essential user-request facets and available evidence omitted by a supported answer |

Use `supported`, `ruled_out` or `unresolved` evidence statuses. A fresh retrieval
result cannot replace missing original candidate logs. An absent selected passage
alone does not distinguish knowledge boundary, chunking, retrieval and selection.

## Gold evidence and comparison controls

The report's Appendix H describes **gold-informed offline diagnosis**: the B2
generator can inspect reference answers/documents and diagnostic signals without
disclosing the expected answer to the chatbot. This is different from a gold-blind
generator. Human failure-type labels must never be passed into generation or
classification. Document the permitted evidence policy for every condition.

The report's B1 coverage baseline does not require live RAG probes, while B2 does.
This comparison tests the complete frameworks. To isolate the benefit of dynamic
question selection, additionally compare fixed static probes with adaptive probes
using the same original history policy, backend, index, allowed operations and
maximum probe budget. Report actual calls and token costs, including internal LLM
calls. If gold access differs, report that difference and add a matched-information
or gold-blind comparison before claiming an effect due to adaptation alone.

An optional batch-versus-adaptive B2 ablation uses the same permitted evidence but
generates all questions upfront in one condition and one question after each probe
result in the other. Budgets such as 2, 4 and 6 probes can test how much interaction
is needed; the budget must be fixed before evaluation.

## Human reference labels and evaluation

Humans annotate the **original failure**, independently of B0/B1/B2 predictions.
Separate human labels for B1 and B2 follow-up answers are not needed for the main
research question. Three independent annotators and adjudication are proposed in
the report. Preserve and report pre-adjudication agreement; consensus alone does
not establish independent reliability.

Use one fixed, adjudicated primary failure type per case for the main single-label
comparison. Secondary types can be recorded separately. If the target becomes all
failure causes, both model outputs and evaluation must instead be consistently
multi-label. Record unresolved evidence and false-positive screening cases explicitly.

The annotation UI supports multiple failure causes per turn in `failure_types`,
including migration of older single-cause labels. It does not infer a primary
cause. Using these annotations in the planned single-label comparison requires
an explicit adjudicated primary label; a multi-label evaluation must retain the
complete selected set for both human and model predictions.

Do not confuse the **gold answer** (answer correctness), **human gold failure label**
(attribution correctness), or **conditioned score** (screening).

| Metric | What to measure |
| --- | --- |
| Failure-type accuracy | Fraction of predictions matching the same human primary label |
| Macro-F1 | Unweighted F1 across the six fixed failure categories; also report class support |
| Per-type precision/recall/F1 and confusion matrix | Which causes improve or remain confused |
| Stage accuracy | Retrieval-side versus generation-side classification, as a secondary measure |
| Paired uncertainty | B2-minus-B1 accuracy/F1 intervals; use conversation-cluster resampling for related turns |
| Efficiency | Actual probes, total model calls, token use and elapsed time |

The RQ is supported by reliable improvement in attribution accuracy/F1 on a frozen
held-out set, not by more questions, longer explanations or correct second answers.
Keep related turns and RAG variants of a source conversation in the same data split.
Tune prompts and selection thresholds on development data, then freeze them.

The report evaluates **all failed turns**. Keeping only the **first failed turn per
conversation** is an alternative pilot policy with a different case population.
Choose and document one policy before the main comparison; apply it equally to all
conditions. The report's preliminary annotation agreement is low, so guideline
refinement and verification of ambiguous labels are needed before final scoring.

## Current data and implementation limits

The current `conversation/conversation_vector.json` is the corrected Vector export
merged from upstream commit `df49376`; it identifies its backend as **`vector`**.
The earlier export at commit `b0624c8` was identical to SelfRAG and labelled
**`selfrag`**. The saved `judge_run/runs/convos_vector_q3_32b/input.json` and its
completed judgments still refer to that earlier export. Preserve this completed
work and its annotations as historical data; do not treat it as judgments of the
corrected Vector answers. Judge the corrected export under a new run name.
Filenames alone are not backend provenance.

Current files provide questions, RAG answers, conversation history, selected context,
and a reference evidence index. The review UI displays the question and answer above
the evidence columns and highlights likely matching sentences. Selected chunks keep
recorded order; corpus matches keep search order without answer-based reranking. Install IBM's complete passage collections with
`python3 -B annotate_server/corpus_store.py --download`; see the [corpus guide](corpora/README.md).
Live review searches the full collection; static review embeds top-10-per-turn
excerpts. Lexical overlap does not prove support or which chunk the model used.
These benchmark corpora do not establish the original RAG's indexed corpus/version
or pre-selection candidate pool. Some upstream causes remain unresolved without
those original traces.
