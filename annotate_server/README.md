# Annotation server

This folder contains the live annotation server, HTML generator/template, annotation
storage, and their tests. Judging and recovery commands live in
[../judge_run](../judge_run/README.md).

Use Python 3 and the standard library; no package installation or judge API key is
needed. Choose a live server for autosave on this machine or a static HTML page for
reviewers who should work without a server.

## Start the server

From `annotate_server`:

```bash
python3 -B annotation_server.py
```

Open <http://127.0.0.1:8765/> and select a run. From the repository root, use:

```bash
python3 -B annotate_server/annotation_server.py
```

The server uses port **8765**. With VS Code Remote SSH, forward remote port 8765
to local port 8765 and open <http://127.0.0.1:8765/>.

The default run directory is `judge_run/runs`, resolved relative to the script,
so it works from any working directory. To use another run directory:

```bash
python3 -B annotation_server.py --runs-dir /path/to/runs
```

The server reads each run's current `out.mtrag.json`; recovery files are published
there by the judge queue after full completion. Refresh the page to load new
results. Human edits autosave under the selected run's `annotations/` directory.
No judge API key is needed. Restart the server after changing the Python or HTML
UI files, then refresh the browser.

The live page initially downloads the full turn list and the first conversation.
Other conversations load when selected, including their answers, selected context,
gold evidence, and full turn history. Corpus search runs on the server and returns
up to 30 matching passages, so the annotation form is available while matches load.
Repeated navigation reuses downloaded conversations and the existing sidebar.

The server compresses responses for browsers that support gzip and caches the last
two runs. Changes to judge inputs, results, gold evidence, or conversation archives
invalidate the cache on the next request. Human annotations load separately;
autosave and exports include annotations for every turn, including unopened turns.
Static HTML exports embed all conversations and a BM25 search snapshot of official
corpus passages (top 10 per turn), or the recorded passage archive as a fallback.

## Install the IBM benchmark corpus

From the repository root:

```bash
python3 -B annotate_server/corpus_store.py --download
```

This explicitly downloads and indexes IBM's four passage-level collections at a
pinned Git revision. Restart the server, then refresh the browser. The corpus panel
uses the official collection matching each turn. See [corpus setup and provenance](../corpora/README.md)
for counts, source revision, ranking, and offline snapshot limits. Review requests
use local indexes and make no external API calls.

## Generate a static annotation page

From `annotate_server`:

```bash
python3 -B make_annotation.py --work-dir ../judge_run/runs/conversation_new
```

This writes `annotate.html` inside the run folder and refuses to overwrite an
existing file. Use `--output /path/to/new.html` to choose another output path.
Static pages save drafts in the browser; export JSON or CSV for a portable copy.
Regenerate to a new file when the UI changes.

The tools read shared gold evidence from `judge_run/gold_support.json` and passage
snapshots from `conversation/*.json`. Reference evidence is attached only when
the task ID, question, and gold answer match. Build the index using
`judge_run/import_gold_support.py` as described in the judging README.

## Send a review file to a friend

From the repository root, generate a page for the same run you are reviewing:

```bash
python3 -B annotate_server/make_annotation.py --work-dir judge_run/runs/convos_vector_q3_32b
```

Send only the resulting `judge_run/runs/convos_vector_q3_32b/annotate.html`. The
updated `annotate_share_ibm.html` in that run includes the new layout and IBM corpus
excerpts; `annotate_share.html` remains an earlier review snapshot.
Your friend opens the downloaded file in Chrome or Edge, chooses their profile
(for example KH while you use PS), and annotates the agreed set of turns. They do
not need Python, the source files, or a running server. Neither person's server
needs to stay running for offline review.

Each reviewer should click **Export JSON** regularly and return their own
`annotations_<run>_<annotator>.json`. Browser drafts are not written into the HTML
file and do not travel when that file is copied. You can import the returned JSON
into the live page; profile tags route records to the correct reviewer. Imports
replace annotations for the matched turns within that profile, so export a backup
before importing revisions.

For agreement analysis, keep the run, page snapshot, and reviewed turn set fixed.
Match exports by conversation ID and turn; compare only pairs with both labels.
Use pairwise Cohen's kappa for Correct/Incorrect. Multiple failure causes require
separate comparisons per cause. Preserve independent exports before adjudication.
The current app exports the required fields but does not calculate kappa yet.

## Annotation workspace

The compact sidebar leaves more space for scrolling conversations and turns.
Each conversation shows a readable name from its first question, a stable number
such as **C01**, and its original ID (hover to see a long ID in full). Search by
name or number to find every turn in that conversation, or search by ID or question.
Names and numbers are navigation aids; annotation exports still use the original
conversation ID and turn. Sticky headers keep the selected turn visible when navigating.

The review queue groups turns under their conversation ID. Green turns have an
mtRAG conditioned score **>= 0.8**; red turns have a score **< 0.8**. Exactly 0.8
passes, and turns without a valid score stay neutral. The displayed label follows
this cutoff even when an older saved label used a different threshold. The result
files are not rewritten. Every turn has a collapsed **Show mtRAG judge assessment**
toggle containing the score/conditioning reason, saved judge explanation, all
statement checks, and score calculations. The toggle starts closed for each turn
and annotator; its state is remembered during the page session and resets on reload.
The question panel does not repeat the judge badge. Queue labels and colors still
show judge results. **Judge guideline** and
**Failure guideline** at the top left open the scoring rules and failure evidence
requirements.

Select **PS**, **KH**, or **WN** in the annotator dropdown. Everyone sees the same
conversations and can review the same turn independently. Switching profiles
keeps the selected turn and restores that person's labels, notes, gold review,
citations, and progress. The annotation bar stays on the right on desktop and
stacks below the comparison on narrow displays.

The sidebar tracks two independent measures for the selected reviewer:

- **Failure labels:** one target per conversation, at its earliest turn whose mtRAG
  conditioned score is below 0.8. Selecting at least one failure checkbox on that
  turn counts it once, including Outside taxonomy or Insufficient evidence.
  A Correct/Incorrect decision and the review checkbox are not required for this
  counter. Correct alone (including its automatic Not a failure value) does not
  count. Later incorrect turns do not add labels. Conversations
  with no judge-marked incorrect turns are excluded from this denominator.
- **Review queue:** reviewed turns across the entire run. Use **Reviewed this turn**
  to mark any turn, including a later or unlabelled turn. Every reviewed card shows
  **✓ Reviewed**. Checking review completion alone never creates a failure label.

Each target has a **First incorrect** badge. Its conversation header and target
card turn green when at least one failure checkbox is selected on that first
incorrect turn, and a **✓ Labelled** mark appears beside the conversation ID.
Clearing its last failure choice removes the label count and completion mark.
Conversations whose every turn has a valid correct mtRAG score
also have green headers, with **All turns correct**; they do not count as failure
labels or receive the annotation-completion mark. Missing scores remain neutral.
Colors and completion marks use every turn in the conversation, including hidden
or filtered turns. Collapsed groups hide their turn containers and have no empty
box or separator border.

Counts and target selection always use the full run, even when search or filters hide the
first incorrect turn. Review marks and both counters are separate for PS, KH, and WN.

The live server autosaves changed turns into separate files:

```text
judge_run/runs/<name>/annotations/PS.json
judge_run/runs/<name>/annotations/KH.json
judge_run/runs/<name>/annotations/WN.json
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

The **RAG answer appears directly above Find text** in the comparison panel; the
question stays in its own panel above. The complete
answer visible without an inner scroll box. Green highlights mark complete RAG-answer
sentences likely to match sentences in selected chunks; hover for the original chunk numbers.

The three evidence columns show **selected / retrieved chunks**, **gold answer +
gold documents**, and **corpus evidence**. Selected chunks retain their recorded order, original chunk numbers and citation
locators. No comparison panel reranks evidence using the RAG answer.
Green highlights in selected passages, gold answers/documents, and corpus passages
mark whole sentences with substantial content-word overlap with a RAG-answer sentence.
Matching compares sentences individually, suppresses weak generic-word overlap,
and rejects pairs whose explicit numbers have no values in common. Yellow marks
show literal search matches, including within highlighted sentences. These likely
matches do not establish semantic support or actual model usage; paraphrases without
shared words may not be highlighted. Gold evidence keeps
its original order. Each column scrolls independently. On a narrow display, scroll
the comparison horizontally.
Gold answers and supporting documents start collapsed for each turn and annotator.
Use **Show gold answer + gold documents** to reveal them or click again to hide
them. The toggle stays as you set it while working on that turn in this page;
reloading the page hides the gold evidence again.
Conversation history has a separate view; detailed mtRAG checks are inside the
judge toggle.

When installed, corpus evidence comes from IBM's passage-level corpus and is
searched with BM25 using the current and first questions. Search results retain
BM25 order with no answer-based reranking. The live page searches the complete
collection; static pages contain a labelled top-10-per-turn search snapshot.
Archive fallback and user-loaded files retain recorded order when filtered.

Without an installed official index, the panel falls back to RAG conversation
exports in `conversation/*.json`, separated by each export's conversation `domain`
and the run's recorded collection/dataset. The archive retains document IDs, titles,
source filenames, a recorded task ID, and file hashes. Exact ID/text duplicates
are merged; different text under the same ID is retained. These are **recorded
passage snapshots, not a complete original corpus or retrieval candidate pool**.
Archive absence cannot establish knowledge-boundary or retrieval failure.
The **Clear search** button clears manual corpus search and restores question
matches. Benchmark membership alone does not establish that a passage
existed in the original RAG's index or reached its retrieval candidate pool.

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

**Reviewed this turn** stores an independent boolean `review_complete` in autosave,
JSON, and CSV. It can be checked or cleared without changing correctness or failure
causes. For older annotations without this flag, a complete correctness/failure
decision still counts as reviewed; an explicit checkbox choice overrides that
default. A complete decision means Correct with no conflicting cause, or Incorrect
with one or more compatible causes/alternative outcomes. Evidence citations and
notes remain optional. **Unreviewed only** and **Next unreviewed** use turn-review
status, independently of conversation label completion. Export JSON regularly.

## Files

- `annotation_server.py`: live run picker, page summaries/cache, corpus-search routing, and save endpoints.
- `annotation_store.py`: validated, atomic per-annotator storage.
- `corpus_store.py`: pinned IBM corpus import, provenance, SQLite FTS5 search, archive fallback search, and shared search terms/stop words.
- `make_annotation.py`: combines run results and evidence into a review page.
- `annotation_template.html`: HTML, CSS, and browser annotation logic.
- `test_annotation.py`, `test_annotation_store.py`, `test_annotation_live.py`, `test_corpus_store.py`: offline UI, progressive loading, highlighting and evidence order, corpus imports, HTTP, and persistence checks.

Live page-summary helpers stay in `annotation_server.py`; all corpus search is
consolidated in `corpus_store.py`. The static generator remains usable independently.
Keep `annotation_store.py` separate for validated atomic writes, and keep the HTML
template separate so the live server and static generator share one interface.
Judge results come from `judge_run/judge_mtrag.py`; the review server does not call
the judge. Keep annotations under each run's
`annotations/` directory and retain distributed HTML snapshots. Generated pages
and Python caches are ignored by Git; caches can be deleted without losing work.

## Offline checks

From the repository root:

```bash
python3 -B -m unittest discover -s annotate_server -p 'test_*.py' -v
```

Checks use temporary run folders and make no judge API calls. They cover evidence,
score boundaries, safe embedding, separate PS/KH/WN files, HTTP validation, and
preservation of saved judge outputs. The original browser workflows run when Chrome or Edge
is installed; the progressive-loading workflow uses Firefox. Unavailable browser
checks are skipped. Firefox checks use an isolated temporary profile and verify
that delayed corpus results do not block annotation or saved reviewer data.
They also check collapsed judge assessments across turns and reviewer profiles.
If a proxy intercepts local HTTP requests, set `NO_PROXY=127.0.0.1,localhost` and
`no_proxy=127.0.0.1,localhost` before running the checks.
