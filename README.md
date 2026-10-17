# Dynamic follow-up questions for RAG failure attribution

This project investigates whether dynamic follow-up questions identify the failure
type of an original RAG answer more accurately than static sub-question generation.
The design follows the [project report](doc/Report_Dynamic_Follow_up_Questions_for_Attributing_Failures_of_RAG.pdf)
and [presentation](doc/PPT_Dynamic_Follow_up_Questions_for_Attributing_Failures_of_RAG.pdf).

The working implementation currently provides answer-quality judging and human
annotation. The B0/B1/B2 experiment below is the proposed research design; this
README does not claim that live probing or the final comparison has been completed.
`DFARag/` is reserved for attribution work. No attribution command is required to
use the existing judge and annotation tools.

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

The provided `conversation/conversation_vector.json` and current prepared inputs
record `rag_name`/`metadata.rag` as **`selfrag`**, despite their filenames. The export
records a Qwen3-8B generator. Verify the actual original backend and index before
live probing; filenames alone are not backend provenance.

Current files provide questions, RAG answers, conversation history, selected context
and a reference evidence index. They do not by themselves provide the complete
original corpus/index or a separate pre-selection candidate pool. Some upstream
causes therefore remain unresolved without additional traces.

## Judge and annotation workflow

Run commands from `judge_run`. This folder judges each turn with the project's
mtRAG-style judge and lets a person review/annotate the result. Existing run JSON
files are retained; the live server reads the current `out.mtrag.json` without
modifying prior results. This step selects failed RAG cases for later attribution;
it does not assign one of the six failure causes.

## Commands (PowerShell)

Create a new run name so previous outputs are not overwritten:

```powershell
New-Item -ItemType Directory -Path runs/conversation_new -ErrorAction Stop
Copy-Item runs/conversation_vector_q25/input.json runs/conversation_new/input.json
```

This reuses the prepared input so judge models evaluate the same cases.
The `prepare` stage defaults to `../conversation.json`, which is not supplied in
this checkout. For a new RAG export, call `prepare_input.py` with an explicit source
and the original MTRAG reference JSONL file. The reference JSONL is needed for gold
answers; `gold_support.json` is a display index, not that reference file.

Set the approved model and API key environment variables in the same session,
then run mtRAG. The key itself must be set in `OPENROUTER_API_KEY` and is not
stored in the run files:

```powershell
$env:JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
$env:JUDGE_MODEL = "qwen/qwen-2.5-72b-instruct"
$env:JUDGE_API_KEY_ENV = "OPENROUTER_API_KEY"
python -B run.py preview --name conversation_new
python run.py mtrag --name conversation_new
python -B annotation_server.py --runs-dir .\runs --port 18766
```

If a thinking model stops the short IDK call at `finish_reason=length`, the
IDK completion budget can be raised for that run without changing the prompts
or score rule, for example:

```powershell
python -B run.py mtrag --name conversation_vector_qwen35b --idk-max-tokens 4096
```

The default is 512. Failed IDK requests retry with twice the previous budget,
so 4096 can request up to 16384 on the third attempt and use more credits.
This option is recorded in `out.mtrag.json` and cannot be mixed with a
different IDK budget when resuming completed records. A failed record has no
saved turn-level checkpoint, so rerunning it repeats its earlier API calls.

For a run that leaves the completion limit to the provider/model, use:

```powershell
python -B run.py mtrag --name conversation_vector_qwen35b --no-max-tokens
```

This omits `max_tokens` for every mtRAG API call, including RB_llm, both RAGAS
steps, and IDK. It cannot be combined with `--idk-max-tokens`. The provider and
model still have their own output limits; reasoning tokens count toward those
limits and are billed. If a call ends with `finish_reason=length`, the judge
stops that call immediately instead of retrying the same unlimited request.
The selected mode is recorded in the output so resume cannot mix it with a
bounded run. The ordinary command without this flag keeps the prior budgets.

For an explicitly budgeted thinking-model run, keep thinking on while limiting
each request to 4096 reasoning tokens and 8192 completion tokens in total:

```powershell
python -B run.py mtrag --name conversation_new_budgeted --reasoning-max-tokens 4096 --max-tokens 8192
```

This sends `reasoning.max_tokens` and `max_tokens` to OpenRouter for every
mtRAG call, including the RAGAS and IDK steps. The 8192 completion ceiling
stays fixed across retries; a `length` response is not retried with a higher
ceiling. The judge also rejects a response if its reported reasoning-token use
exceeds 4096, or if reported completion use exceeds 8192. The output usage
records how many calls reported token counts and how many could not be verified.
Provider support must be checked with a small pilot: a provider may ignore a
reasoning budget, and a completed answer can still hit its total
completion limit. Both budgets are recorded in the result, and runs with
different budgets cannot be merged by `--resume-from`. Use a new run name;
do not overwrite the previous unbounded result.

Open `http://127.0.0.1:18766/run/conversation_new`. The live page regenerates
from the latest JSON when refreshed. Alternatively,
`python run.py annotate --name conversation_new` writes a static
`runs/conversation_new/annotate.html` once and refuses to overwrite one that
already exists. Export human annotations with the page's JSON or CSV button.
After changing `make_annotation.py`, restart the annotation server so the live
server imports the new UI code; a previously generated static HTML file does
not update on refresh.
The page shows selected context beside original reference-task passages, with
document IDs and relevance labels. It reads `judge_run/gold_support.json` and only
attaches evidence whose task ID, question and reference answer match the run.
The summary badge shows the conditioned score; human `human_label` and failure type
are exported separately from judge screening labels.

## Per-turn score and provenance

| Value | Inputs and calculation | Source |
| --- | --- | --- |
| `RB_llm` | The LLM receives the prior conversation, current question, reference answer, RAG answer and selected passages. It rates faithfulness, appropriateness and completeness from 1 to 10. The parsed score is also divided by 10. | [mtRAG §6.3](https://arxiv.org/html/2501.03468#S6.SS3), [IBM prompt/parser](https://github.com/IBM/mt-rag-benchmark/blob/main/scripts/evaluation/judge_utils.py). |
| `RL_F` | A RAGAS-style faithfulness step splits the answer into statements and checks each against selected passages. Score = supported statements / all statements, from 0 to 1. The reference answer is not used in this step. | [mtRAG §6.3.2](https://arxiv.org/html/2501.03468#S6.SS3), [RAGAS](https://arxiv.org/abs/2309.15217). |
| `idk` | An LLM detects whether the answer signals insufficient information: yes=1, partial=0.5, no=0. | [mtRAG §6.3.3](https://arxiv.org/html/2501.03468#S6.SS3), [IBM IDK prompt](https://github.com/IBM/mt-rag-benchmark/blob/main/scripts/evaluation/judge_utils.py). |
| `phi` | **Project adaptation:** harmonic mean of `RB_llm/10` and `RL_F`: `2 / (1/(RB_llm/10) + 1/RL_F)`. If `RL_F` is undefined, use `RB_llm/10`; if a defined component is zero, use zero. | Inspired by mtRAG metrics; [official leaderboard](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/) uses the harmonic mean of **three** metrics for system ranking. |
| `conditioned_score` | If ANSWERABLE/PARTIAL and IDK=yes, 0; otherwise `phi`. If UNANSWERABLE/CONVERSATIONAL, 1 only for IDK=yes, otherwise 0. If answerability is missing, `phi`. | mtRAG conditions **each metric individually** for answerable/partial/unanswerable turns. Combining first and treating conversational/missing answerability as above are local adaptations. |

The project's existing binary screening rule is `correct` iff
`conditioned_score >= 0.8`; this cutoff is not claimed to be the paper's.
Exactly 0.8 is correct. Existing results can be relabelled without API calls using
`python -B relabel_mtrag.py --input runs/<name>/out.mtrag.json` from `judge_run`.
The script backs up the original bytes before updating labels and threshold metadata;
numeric scores remain unchanged. It rejects runs with skipped turns.
The `RB_llm` per-metric verdict uses `>= 6`. The page also shows these
**project-only three levels on the raw RB_llm rating**: low `<=5`, medium
`>5 and <=8`, high `>8`. Exactly 5 is low; exactly 8 is medium. They
do not change `correct/incorrect` or failure-attribution eligibility.
Historical `vrag_accept_min` and `ar` keys remain in the mtRAG JSON threshold
metadata solely so old mtRAG runs can be resumed with the same schema; neither
starts a vRAG or standalone RAGAS judge.

The paper's notation `phi` denotes an individual metric among
`RB_alg`, `RB_llm` and `RL_F`; it is not the two-metric harmonic mean in
this implementation. The paper uses several judge models and their median
for RB_llm. This implementation uses the configured model once per turn,
omits RB_alg, and uses a shortened IDK few-shot prompt. Report this as an
**mtRAG-style project adaptation**, not an exact replication.

## Files and output

- `run.py`: mtRAG-only entry point.
- `prepare_input.py`: copies conversation, gold answers and selected
  context into a new run's `input.json`; no RAG regeneration.
- `judge_paper.py`: mtRAG score, including its required RAGAS faithfulness
  component, API transport and exact-configuration resume.
- `make_annotation.py`: HTML view of mtRAG and binary human annotation.
- `annotation_server.py`: local live run picker and annotation page.
- `import_gold_support.py`: imports reference-task passages for side-by-side review.
- `relabel_mtrag.py`: applies a screening cutoff to completed scores with a backup.
- `runs/<name>/out.mtrag.json`: per-turn scores, binary screening labels,
  reasoning and provenance. Other historical `out.*.json` files remain
  in `runs/` for audit; they are not read by the current UI.

The human export stores the current binary choice in `human_label`. It retains
`human_score_band` only for historical three-level annotations. Failure type is
selected separately from the locked six-category taxonomy. Do not equate
a quality band with a failure cause.
