# Conversation exports

`vector/conversation_vector.json`, `selfrag/conversation_selfrag.json`,
`raptor/conversation_raptor.json`,
and `conversation_hipporag.json` use the same `conversations` / `turns` export schema.
Keep IDs, answers, recorded history, selected passage text and passage order intact.
The corrected Vector export now identifies its backend as `vector`. Before upstream
commit `df49376`, that filename contained the same `selfrag` export as SelfRAG.
The existing `convos_vector_q3_32b` judge run retains those earlier inputs and
results. Use a new run name to judge the corrected Vector export; its answers and
selected passage lists differ from the earlier source.

HippoRAG2 was imported from [Tanyarat2/DFARag](https://github.com/Tanyarat2/DFARag/tree/df493766e85770cd5ea48e6592e6f69f223f1438/conversation)
at commit `df493766e85770cd5ea48e6592e6f69f223f1438`.

| Original file | Conversations | Turns |
| --- | ---: | ---: |
| `hipporag2-cloud/conversation.json` | 26 | 205 |
| `hipporag2-fiqa/conversation.json` | 27 | 199 |
| `hipporag2-govt/conversation.json` | 28 | 214 |
| Combined `conversation_hipporag.json` | 81 | 618 |

No ClapNQ output was present. The domain files retain their exact upstream bytes.
The combined file preserves every conversation and turn unchanged and records
source hashes, the Git revision, and each original export's provenance. These are
existing Qwen3-8B RAG outputs; import does not regenerate answers or rerun retrieval.

To reproduce the combination from the repository root, choose a new output file:

```bash
python3 -B judge_run/import_hipporag.py --revision df493766e85770cd5ea48e6592e6f69f223f1438 --output /tmp/conversation_hipporag.json
```

The importer validates task IDs, gold-question matches, answerability, selected
context and turn counts before creating its output. Prepared judge data is at
`judge_run/runs/convos_hipporag_q3_32b/input.json`. See the
[judge guide](../judge_run/README.md) to start resumable judging.

## Retrieval/reranking experiment artifacts

Backend/domain folders contain `candidates.jsonl` and `reranked.jsonl` from
`new_retrieval_experiment` (commit `93157a0`). These files record new retrieval and
reranking results without answer generation. Keep them separate from the original
generation contexts in the conversation exports; matching a task ID alone does
not establish that the original saved answer used these new candidates.
