# mtRAG judging

Run commands from `judge_run`. This folder prepares inputs, judges turns with
the project's mtRAG-style judge, and recovers incomplete runs. Run data stays in
`judge_run/runs/`. Human review tools live separately in
[annotate_server](../annotate_server/README.md). Judging selects failed RAG cases
for later attribution; it does not assign one of the six failure causes.

## Start a new judge run

Use Python 3 with the standard library. Commands in this guide run from
`judge_run/`; from the repository root, prefix scripts with `judge_run/`.
Set `OPENROUTER_API_KEY` in the terminal that starts a paid judge run. Human review,
input preparation, previews, and the offline tests make no judge API calls.

In Bash, validate a new run before starting it:

```bash
python3 -B run.py judge --src conversation_vector --name vector_new --dry-run
python3 -B run.py judge --src conversation_vector --name vector_new
```

Choose a new name for each experiment. Existing inputs and results remain under
`runs/`. The current complete-run names are `convos_vector_q3_32b`,
`convos_selfrag_q3_32b`, and `convos_raptor_q3_32b`; the similarly named
`conversation_vector_*` folders contain earlier pilot runs.

The corrected Vector source merged from upstream commit `df49376` identifies its
backend as `vector`. The earlier Vector file was a duplicate SelfRAG export, and
`runs/convos_vector_q3_32b/input.json` still records that earlier `selfrag` input.
Its completed judgments and annotations are historical results, not evaluations
of the corrected Vector answers. Keep them and use a new run name (such as
`vector_corrected_q3_32b`) for the current Vector source. The recovery queue rejects
an existing run whose input differs from the current source; do not overwrite its
input to bypass that check.

### Imported HippoRAG2 run

`conversation/conversation_hipporag.json` combines the Cloud, FiQA, and Govt
exports: **81 conversations / 618 turns**. There is no ClapNQ output. Original
files, hashes, Git revision and the repeatable conversion command are documented
in [conversation/README.md](../conversation/README.md). All turns have been
validated against the existing gold references and prepared under
`runs/convos_hipporag_q3_32b/input.json`.

Start the existing resumable judge queue for HippoRAG only, from the repository root:

```bash
python3 -B judge_run/queue_remaining_judges.py --complete-all --rags hipporag
```

It uses `qwen/qwen3-32b`, threshold 0.8, per-turn checkpoints and the same adaptive
4096/8192 baseline budgets as the other completed runs (retry ceiling 32768).
If `OPENROUTER_API_KEY` is unset, enter it at the masked terminal prompt. Never
paste the key into chat or source files. The launcher detaches and logs to
`judge_run/runs/remaining_judges.log`. Rerun the same command to recover incomplete
work. Existing Vector, SelfRAG and RAPTOR results are not rejudged. Without `--rags`,
`--complete-all` keeps its original three-backend behavior.

The prepared input alone does not mean judging has run. After full verification,
the queue publishes `runs/convos_hipporag_q3_32b/out.mtrag.json` for annotation.

### PowerShell

Judge an entire file from `conversation/` with one command (from `judge_run`):

```powershell
python -B run.py judge --src conversation_vector --fresh
```

This reads `../conversation/conversation_vector.json`, prepares every turn, and
judges it with `qwen/qwen3-32b` through OpenRouter. Only `OPENROUTER_API_KEY`
needs to be set in this terminal. Defaults match the Qwen3 pilot: 4096 reasoning
tokens, 8192 total completion tokens per request, and a correct/incorrect cutoff
of 0.8. The default run name is `conversation_vector_q3_32b`.

`--fresh` moves an existing run (including human annotation files) to a sibling
`<name>_backup_<timestamp>` folder before preparing the new one. Omit it for a
new run name. Source, gold references, selected passages and API-key presence are
checked before archiving. Browser drafts are keyed by run name; export any
browser-only drafts before reusing that name. No existing run is moved during
`--dry-run`.

Change `--src` to another JSON filename or stem in `conversation/`, or pass a full
file path. Set `--name` to choose a different output folder. For example:

```powershell
python -B run.py judge --src conversation_vector --name conversation_vector_all_q3_32b
python -B run.py judge --src conversation_vector --fresh --dry-run
```

Gold answers are loaded from the original reference JSONL when available, or from
the existing `gold_support.json` index. No per-run reference file or manual input
preparation is needed. Cached references must match both task ID and question;
new tasks need matching gold answers supplied with `--reference`. The index does
not contain answerability labels, so exports using it must record answerability.
Use `--model`, `--base-url`, and `--api-key-env` for another judge/API. These options
belong to `judge`; the older prepared-input `mtrag` workflow below still uses
`JUDGE_MODEL`, `JUDGE_BASE_URL`, and `JUDGE_API_KEY_ENV`.

### Existing prepared-input workflow

Create a new run name so previous outputs are not overwritten:

```powershell
python run.py prepare --src conversation_vector --name conversation_new --limit 3
python run.py preview --name conversation_new
```

`prepare --src conversation_vector` reads `../conversation/conversation_vector.json`
and writes `runs/conversation_new/input.json`. Always supply `--src`: the legacy
default `../conversation.json` is not included in this checkout. The preparer
preserves selected passages, reference answers, answerability and recorded history. `preview`
writes `preview.json` without any paid API call.

Set the approved model and API key environment variables in the same session,
then run mtRAG. The key itself must be set in `OPENROUTER_API_KEY` and is not
stored in the run files:

```powershell
$env:JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
$env:JUDGE_MODEL = "qwen/qwen3-30b-a3b"
$env:JUDGE_API_KEY_ENV = "OPENROUTER_API_KEY"
python run.py mtrag --name conversation_new
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
different IDK budget when resuming completed records. Without
`judge_mtrag.py --turn-checkpoints`, rerunning a failed record repeats its earlier
API calls. The completion queue below enables turn checkpoints automatically.

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

### Recover missing conversations and complete all three RAGs

For an existing Qwen3-32B run, start the completion queue in another Bash
terminal on Linux. The queue uses Linux process identities and POSIX terminal/lock
APIs. To continue after all previous judge processes have exited:

```bash
python3 -B queue_remaining_judges.py --complete-all
```

Supply each active vector process ID with `--wait-pid` so recovery
waits until the current process has finished writing its output:

```bash
python3 -B queue_remaining_judges.py --complete-all --wait-pid 12345
```

Replace `12345` with the active vector process ID. Stop an older waiting queue
before replacing it; the queue lock rejects duplicate launchers. If
`OPENROUTER_API_KEY` is unset, a masked prompt displays one `*` per character.
The queue detaches from the terminal and logs to `runs/remaining_judges.log`.
An existing foreground vector job still requires its original terminal to stay
connected until it finishes.

Follow the latest log (from `judge_run/`):

```bash
tail -n 100 -f runs/remaining_judges.log
```

Press Ctrl+C to stop following the log; the detached queue keeps running.

This mode completes `convos_vector_q3_32b`, then `convos_selfrag_q3_32b`, then
`convos_raptor_q3_32b`. Each request starts with the existing 4096 reasoning / 8192
completion budgets. On retries, the budgets can increase to 8192/16384 and
16384/32768. Larger retries can cost more. Configure the total completion ceiling
with `--adaptive-max-tokens` (default 32768).

Only missing conversations are judged. Original fixed-budget results with the
same inputs and baseline settings may be reused when adaptive mode is explicitly
selected. They retain their original per-record provenance. New records store
`judge.adaptive_max_tokens` and the highest requested budgets in usage metadata;
adaptive results cannot be resumed as fixed-budget results or under a different
adaptive ceiling. All other input and scoring-setting checks remain strict.

The judge prints turn progress and saves completed conversations immediately.
This queue also enables `--turn-checkpoints`: after every completed turn, it
atomically saves a checkpoint under the run's `turn_checkpoints/` directory.
Recovery preserves those judgments and calls the judge only for remaining turns.
Checkpoints require an exact match of the full conversation, selected evidence,
history, scoring settings, model, prompt version, API URL and token-budget policy.
Reported usage from failed attempts is retained as well. The original running
process cannot retroactively produce checkpoints for turns it already discarded.
Recovery files use `out.mtrag.recovery_*.json`; any prior
canonical output is backed up before a fully verified result replaces it.
The queue requires every conversation and turn before advancing to the next
RAG. After `--recovery-passes` passes (default 3), persistent failures stop the
queue with saved partial progress. Rerun the same completion queue to continue;
do not use `--fresh` to repair missing conversations.

To review saved results, follow the [annotation server instructions](../annotate_server/README.md).
Use <http://127.0.0.1:8765/> for live review. The server and static generator read
the canonical `out.mtrag.json`, not individual recovery files.
The question and RAG answer appear above the evidence columns. The review UI can
search the [official IBM passage corpora](../corpora/README.md) and highlight
matching text. Selected chunks retain recorded order; corpus search retains BM25
order without answer-based reranking. These display aids do not change
judge scores, original context order, or saved human labels.

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

- `run.py`: judging entry point with `judge`, `prepare`, `preview`, and `mtrag` stages.
- `prepare_input.py`: shared preparation functions used by both `run.py judge` and `run.py prepare`, plus a standalone CLI.
- `judge_mtrag.py`: mtRAG scoring, API transport, checkpoints, and configuration-checked resume.
- `queue_remaining_judges.py`: recovers missing records and completes Vector, SelfRAG, and RAPTOR.
- `relabel_mtrag.py`: updates screening labels from saved scores, retaining a backup.
- `import_gold_support.py`: imports the shared reference index from an original MTRAG JSONL file.
- `gold_support.json`: shared gold answers and reference passages for preparation and annotation.
- `runs/<name>/input.json`: prepared judge input.
- `runs/<name>/out.mtrag.json`: per-turn scores, labels, reasoning, and provenance.
- `runs/<name>/turn_checkpoints/`: saved progress for runs with turn checkpoints enabled.
- `runs/<name>/annotations/`: independent PS, KH, and WN human annotations; written by the review server.
- `runs/remaining_judges.log`: queue progress; follows new runs as the log grows.

The scoring module and every launcher/import use `judge_mtrag.py`. For the direct
CLI, run `python3 -B judge_mtrag.py --help`. When no `--output` is supplied, it
writes `<input-stem>.mtrag-judged.json` (or `.jsonl`); `run.py` and the queue still
write the canonical `out.mtrag.json`.

The judge supports only mtRAG. The former constant-returning mode dispatcher and
its unused arguments are removed; `--mode mtrag` remains accepted for existing
commands. Prompt-version strings, result schemas, scoring settings, and checkpoint
fingerprints remain unchanged so completed results can still be resumed.

The remaining tools have distinct jobs: keep preparation separate from paid API
execution, recovery separate from new experiments, and relabelling/reference import
as explicit operations. `run.py prepare` calls the shared preparer directly instead
of starting a second Python process.

Keep original inputs, canonical outputs, recovery/threshold backups, checkpoints,
and human annotations. They retain provenance and completed paid work. Generated
`annotate*.html` pages can be recreated with the annotation generator, but keep any
snapshot already distributed to reviewers. Python caches are disposable. Git ignores
caches, generated review pages, queue logs, and the queue lock; it includes the
recovery launcher and recovery tests.

To build the shared reference index from an original MTRAG reference file:

```bash
python3 -B import_gold_support.py --reference /path/to/reference.jsonl
```

The importer refuses to overwrite an existing index. Annotation tools, their HTML
UI, storage, and tests live in `../annotate_server/`. Existing human annotations
remain under each run's `annotations/` directory.

## Offline checks

From the repository root:

```bash
python3 -B -m unittest discover -s judge_run -p 'test_*.py' -v
```

Runner and recovery tests use temporary files and synthetic model responses;
no judge API calls are made. If local HTTP checks are intercepted by a proxy, set
`NO_PROXY=127.0.0.1,localhost` and `no_proxy=127.0.0.1,localhost`.
