# IBM MTRAG passage corpora

The annotation server searches the passage-level collections from
[IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark/tree/53fef6f5917c7e10c436f697e69a955dbc795d81/corpora/passage_level).
IBM recommends this granularity for alignment with its reference passages; see its
[corpus guide](https://github.com/IBM/mt-rag-benchmark/blob/53fef6f5917c7e10c436f697e69a955dbc795d81/corpora/README.md).

## Download and index

From the repository root:

```bash
python3 -B annotate_server/corpus_store.py --download
```

This downloads the four ZIPs at Git revision
`53fef6f5917c7e10c436f697e69a955dbc795d81` and builds one local SQLite FTS5 index
per collection. Python's SQLite build must support FTS5. Downloads are explicit;
review requests do not access the internet or call a model. Existing indexes are
retained. To rebuild, stop the server and move the index to a backup location
before rerunning the command. Restart the server after installation.

The imported records at this revision contain these passage counts (counted from
the actual files, rather than copied from the upstream summary table):

| Collection | Local index | Passages |
| --- | --- | ---: |
| ClapNQ | `clapnq.sqlite3` | 183,408 |
| Cloud | `cloud.sqlite3` | 72,442 |
| FiQA | `fiqa.sqlite3` | 61,022 |
| Govt | `govt.sqlite3` | 49,607 |

The indexes preserve passage IDs, titles, full text, source filenames, Git revision,
download URL, and archive SHA-256. Temporary ZIPs are removed after import.
Indexes are generated files and ignored by Git. Follow the upstream dataset's
license and attribution terms when redistributing passages.

## Review search and ordering

The live server searches only the current turn's collection. BM25 searches the
current and first questions and returns up to 30 passages in its original search
order. There is no answer-based reranking or candidate reranking stage. Manual
search requires all entered words. Search results are navigation aids, not a model
attribution trace or a semantic entailment check.

Both indexed IBM search and the conversation-archive fallback are implemented in
`annotate_server/corpus_store.py`, with shared stop words.
The HTTP server routes requests to these search classes; it does not maintain a
second ranking implementation. Archive fallback and browser searches filter
passages in their recorded order. Offline snapshots preserve BM25 search order.

Selected generator chunks retain their recorded order, chunk numbers and citation
locators. Gold answers and documents
retain their original order and highlight complete sentences likely to match a
RAG-answer sentence. Sentence matching uses substantial content-word overlap and
excludes pairs with conflicting explicit numeric values; it does not verify entailment.
Neither RAG answers, gold evidence nor judge explanations affect corpus search order.

Static HTML embeds the top 10 official matches per turn, deduplicated across the
run. Its corpus search covers that offline subset; the live server searches the
full collection. The original conversation passage archive is used as a labelled
fallback when an official collection has not been installed.

The benchmark corpus does not establish which corpus/index revision the original
RAG used. Original retrieved candidates, reranker scores, and prompt traces remain
separate evidence requirements for failure attribution.
