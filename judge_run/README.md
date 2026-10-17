# mtRAG judging in one folder

Run commands from `judge_run`. This folder judges each turn with the project's
mtRAG-style judge and lets a person review/annotate the result. Existing run JSON
files are retained; the live server reads the current `out.mtrag.json` without
modifying prior results. This step selects failed RAG cases for later attribution;
it does not assign one of the six failure causes.

## Commands (PowerShell)

Create a new run name so previous outputs are not overwritten:

```powershell
python run.py prepare --name conversation_new --limit 3
python run.py preview --name conversation_new
```

`prepare` reads `../conversation.json` and writes
`runs/conversation_new/input.json`. For a different source, use
`python prepare_input.py --src ..\conversation1.json --out runs/conversation1`,
then continue with `--name conversation1`. The preparer preserves selected
passages, reference answers, answerability and recorded history. `preview`
writes `preview.json` without any paid API call.

Set the approved model and API key environment variables in the same session,
then run mtRAG. The key itself must be set in `OPENROUTER_API_KEY` and is not
stored in the run files:

```powershell
$env:JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
$env:JUDGE_MODEL = "qwen/qwen3-30b-a3b"
$env:JUDGE_API_KEY_ENV = "OPENROUTER_API_KEY"
python run.py mtrag --name conversation_new
python run.py serve
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

Open `http://127.0.0.1:8765/run/conversation_new`. The live page regenerates
from the latest JSON when refreshed. Alternatively,
`python run.py annotate --name conversation_new` writes a static
`runs/conversation_new/annotate.html` once and refuses to overwrite one that
already exists. Export human annotations with the page's JSON or CSV button.
After changing `make_annotation.py` or `annotation_template.html`, restart `python run.py serve` so the live
server imports the new UI code; a previously generated static HTML file does
not update on refresh.

### Annotation workspace

The review queue groups turns under their conversation ID. Green turns have an
mtRAG conditioned score **>= 0.8**; red turns have a score **< 0.8**. Exactly 0.8
passes, and turns without a valid score stay neutral. The displayed label follows
this cutoff even when an older saved label used a different threshold. The result
files are not rewritten. Incorrect turns show the score/conditioning reason,
saved judge explanation, and any unsupported claims. **Judge guideline** and
**Failure guideline** at the top left open the scoring rules and failure evidence
requirements. Judge assessments remain visible.

Select **PS**, **KH**, or **WN** in the annotator dropdown. Everyone sees the same
conversations and can review the same turn independently. Switching profiles
keeps the selected turn and restores that person's labels, notes, gold review,
citations, and progress. The annotation bar stays on the right on desktop and
stacks below the comparison on narrow displays.

The live server autosaves changed turns into separate files:

```text
runs/<name>/annotations/PS.json
runs/<name>/annotations/KH.json
runs/<name>/annotations/WN.json
```

Changes merge by turn within the selected profile; they do not replace another
annotator's file or alter judge outputs. Browser backups use distinct
`annot:<run>:PS`, `annot:<run>:KH`, and `annot:<run>:WN` keys. Offline changes remain
pending and retry when the profile is reopened or another edit is made. The save
status in the right bar identifies the annotator and whether the work reached the
server. Static HTML saves only in that browser; JSON exports remain a portable
backup. Earlier shared browser annotations are retained at their original key;
**Copy earlier labels to PS/KH/WN** explicitly copies them into the chosen profile
without overwriting its existing work.

The comparison shows three evidence columns together: **RAG answer + selected
context**, **gold answer + gold documents**, and **corpus evidence**. Each column
scrolls independently. On a narrow display, scroll the comparison horizontally.
Conversation history and detailed mtRAG checks have separate views. Passage search
highlights literal matches; it does not predict relevance.

Corpus evidence is loaded automatically from RAG conversation exports in
`conversation/*.json`, separated by each export's conversation `domain` and the
run's recorded collection/dataset. The archive retains document IDs, titles,
source filenames, a recorded task ID, and file hashes. Exact ID/text duplicates
are merged; different text under the same ID is retained. These are **recorded
passage snapshots, not a complete original corpus or retrieval candidate pool**.
Archive absence cannot establish knowledge-boundary or retrieval failure.
Question-word suggestions and corpus search are text-match navigation aids.

**Load corpus file** can temporarily use a local JSON array, a JSON object with a
`documents` array, or JSONL in the current collection. Each document needs a
`text`, `contents`, or `content` string; `document_id`/`id`/`_id` and `title` are
optional. Check the original corpus version yourself. Loaded files stay in memory
until reload; citations retain their source and document locator in exports.

Choose **Correct** or **Incorrect** in the annotation panel. For an incorrect
answer, click one or more failure-cause checkbox cards. Each card includes a short
description; the selected count and **Clear choices** make the set easy to review.
Selected causes have expandable evidence guides, with links to relevant panels
and missing original traces. These guides
do not automatically assign a failure. Original corpus scope, indexed chunks,
and retrieval candidates are not provided by the current annotation data, so use
**Insufficient evidence** when those traces are needed to establish the cause.
**Outside taxonomy** and **Insufficient evidence** are alternative outcomes: choosing
one replaces the cause selections, and choosing a cause replaces the alternative.
Marking the answer Correct records Not a failure and clears the failure causes.

All chosen values are stored in the canonical `failure_types` array. Earlier
`failure_type` strings load as one-item lists. JSON includes the complete array;
CSV stores it as a JSON-encoded cell. The legacy `failure_type` field contains a
value only when exactly one choice is selected; it stays empty for multiple
choices. No primary cause is inferred from selection order. Autosave, offline
backup, annotator switching, and imports preserve the complete selection.

The separate **Gold answer check** records whether the reference is supported,
questionable, or appears incorrect, with an explanation in `gold_notes`. This
assessment (`gold_assessment`) does not change the RAG-answer label, saved gold,
or judge scores. Both fields round-trip through JSON and CSV exports/imports.

Select an exact quote in an answer or passage and click **Cite selection** or
**Cite passage / selection** to attach it. Without a selection, the complete text
is attached. Citations retain their source, passage locator, and quote in the new
`evidence` export field (a JSON array; JSON-encoded in CSV). Existing annotation
fields and historical score bands are preserved. Export filenames include the
annotator, such as `annotations_<run>_PS.json`. Older JSON exports can still be
imported: records tagged PS/KH/WN go to their own profiles, and records without an
annotator go to the selected profile. Unmatched turns or unknown annotator names
are skipped and reported. Matching imports replace that profile's annotation for
the imported turn.

**Reviewed** means a correctness decision plus one or more compatible failure causes (or an alternative outcome) for
incorrect answers. Older labels without a cause remain labelled and can be found
with **Unreviewed only**. Evidence citations and explanatory notes are optional.
Use **Next unreviewed** to continue. Export JSON to retain a portable backup.

For descriptive three-band agreement with older three-level human exports, run:

```powershell
python calibrate.py --work-dir runs/conversation_new --labels annotations_conversation_new.json
```

`calibrate.py` compares historical human `human_score_band` values with the
judge's RB_llm band. The current annotation page asks for a binary
`human_label` (`correct` or `incorrect`), so new exports do not supply the
three-level human ratings needed by `calibrate.py`. It does not change the
binary failure gate or estimate the accuracy of the later Xie-vs-Proposed
failure attribution.

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
Exactly 0.8 passes. To apply this cutoff to an existing all-turn run without
calling the model again, run `python -B relabel_mtrag.py --input runs/<name>/out.mtrag.json`.
This preserves the original bytes in `out.mtrag.before-threshold-0p8.json` and
updates screening labels, conversation labels, first failed turns and threshold
metadata. Numeric scores, prompts and API usage are unchanged. Runs with skipped
turns cannot be relabelled because their missing scores must first be judged.
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

The annotation page shows selected RAG context and original reference-task
passages in side-by-side evidence columns. Document IDs and dataset relevance
labels are included. This display uses `judge_run/gold_support.json`, a separate
evidence index; it does not modify judge inputs, saved scores or human annotations.
Reference evidence is attached only when the task ID, question (ignoring outer
whitespace) and gold answer match. Missing or mismatched evidence is shown explicitly.

To build the index from an original MTRAG reference JSONL file, run from `judge_run`:

```powershell
python -B import_gold_support.py --reference path\to\reference.jsonl
```

The importer refuses to overwrite an existing index. Restart the annotation server
after changing the UI code, then refresh the browser.

- `run.py`: mtRAG-only entry point.
- `prepare_input.py`: copies conversation, gold answers and selected
  context into a new run's `input.json`; no RAG regeneration.
- `judge_paper.py`: mtRAG score, including its required RAGAS faithfulness
  component, API transport and exact-configuration resume.
- `make_annotation.py`: HTML view of mtRAG and binary human annotation.
- `annotation_template.html`: self-contained review UI embedded by the generator.
- `annotation_server.py`: local live run picker and annotation page.
- `annotation_store.py`: validated, atomic per-annotator saves under each run.
- `calibrate.py`: descriptive three-level human/model agreement.
- `runs/<name>/out.mtrag.json`: per-turn scores, binary screening labels,
  reasoning and provenance. Other historical `out.*.json` files remain
  in `runs/` for audit; they are not read by the current UI.

The human export stores the current binary choice in `human_label`. It retains
`human_score_band` only for historical three-level annotations. Failure causes are
selected separately from the locked six-category taxonomy. Do not equate
a quality band with a failure cause.

## UI checks

From the repository root, run:

```powershell
python -m unittest discover -s judge_run -p 'test_annotation*.py' -v
```

The checks cover safe data embedding, missing evidence, and preservation of judge
statement reasons. When Chrome or Edge is installed, an isolated headless browser
also checks desktop and mobile annotation workflows, legacy imports, citations,
exports, filters, top-left guidelines, and the right annotation bar. No judge API calls are made.
The checks also verify the exact 0.8 boundary, unchanged saved outputs, grouped
conversation IDs, corpus isolation/deduplication, and independent gold review.
Persistence checks use temporary run folders to verify separate PS/KH/WN files,
HTTP validation, server reload, offline-draft recovery, and profile-aware imports.
They also cover multiple failure selections, deselection, alternative outcomes,
legacy single-cause migration, and complete array round-trips through autosave and exports.
