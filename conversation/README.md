# Conversation exports

`conversation_vector.json`, `conversation_selfrag.json`, `conversation_raptor.json`,
and `conversation_hipporag.json` use the same `conversations` / `turns` export schema.
Keep IDs, answers, recorded history, selected passage text and passage order intact.
Vector and SelfRAG currently contain identical exports labelled `selfrag`; verify
their actual backend provenance before treating them as independent conditions.

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
