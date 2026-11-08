# Dynamic failure attribution

`evaluator.py` implements artifact inspection → dynamic probe generation → ordered
triage → targeted secondary index search. It requires Python 3.10+ and Pydantic 2.
It does not require gold answers or modify existing question generators.

```python
from failure_classification.questiongeneration.evaluator import FailureClassifier

# llm(system_prompt: str, payload: dict) -> parsed JSON dict
# search(question: str, top_k: int) -> SearchResult or equivalent dict
result = FailureClassifier(llm, search).evaluate({
    "user_query": "What was Acme's revenue in 2023?",
    "top_k_chunks": ["..."],
    "prompt_context": ["..."],
    "rag_answer": "...",
})
print(result.model_dump_json(indent=2))
```

The LLM adapter must send the system prompt separately from the JSON-serialized
payload and return a parsed JSON object. Each stage specifies its own output
schema in the prompt; Pydantic validates the response. Malformed output and service
errors propagate to the caller. The caller may apply its own retry policy.

The search adapter must execute a new vector query on the **same corpus index**,
using the same embedding model as that index, and return:

```python
{
    "chunks": ["actual retrieved passage text"],
    "exhaustive": False,
    "scope": "original-index, snapshot identifier, top 10 candidates",
}
```

Retrieved hits are checked for semantic relevance by the LLM. A nonempty result
alone does not establish retrieval failure. Set `exhaustive=True` only if the
backend returns all corpus passages for review or independently guarantees that
its returned passages exhaust possible relevant evidence for that question.
Ordinary approximate or exact top-k vector retrieval is not exhaustive, even if
it returns zero hits. Index/service errors must raise exceptions.

CLI, with your adapters in an importable `my_adapters.py`:

```bash
python -m failure_classification.questiongeneration.evaluator \
  --input artifacts.json --output diagnosis.json \
  --llm my_adapters:llm --search my_adapters:search --top-k 10
```

Output contains exactly `extracted_constraints`, `identified_delta`,
`dynamic_sub_questions`, `attributed_failure`, and `justification`.
The first established category wins in the requested order: grounding, response
coverage, context selection, chunking, retrieval, knowledge boundary. This is a
single-label attribution policy; multiple defects may coexist. Grounding means
lack of support in the passed context, not necessarily real-world falsity.

Secondary searches run only for dynamic questions lacking initial retrieved
support. Recovery of any missing target establishes retrieval failure; knowledge
boundary requires exhaustive unsuccessful review of every probed missing target.
It describes those targets, not every fact in the user query.

`InconclusiveDiagnosis` is raised when no failure is established or secondary
search cannot establish corpus absence. Successful inputs cannot truthfully be
forced into the six failure-only labels. Plain string chunks lack source and
adjacency metadata, so chunking attribution is conservative and requires quoted
boundary evidence from at least two chunks. Semantic judgments remain LLM
assessments; substring checks validate citation integrity, not entailment.

Validation:

```bash
python -m unittest discover -s failure_classification/tests -t . -p 'test_dynamic_evaluator.py' -v
```

These tests use deterministic adapters and exercise all six branches plus
non-exhaustive misses, invalid citations and backend errors. Live model and index
integration requires your adapters and credentials; it is not exercised here.

## Run the saved first 10 cases

Use the project's Python environment (which includes the API and embedding
clients) and load your credentials, then run:

```bash
set -a
source .envrc
set +a
/disk/sp2026rageval/kh/miniconda3/envs/ragec310/bin/python \
  -m failure_classification.questiongeneration.run_dynamic_evaluator \
  --limit 10 --model gpt-5.6-luna \
  --output-dir result/failure_classification/dynamic_evaluator/first10_luna
```

The runner strips reference answers, reference documents and prior labels before
constructing the four input artifacts. It stores inputs, cached raw/parsed model
responses, per-case diagnoses, summary and a readable report. Secondary vector
search is initialized only if triage needs it, using the local corpus and the
configured pinned GTE embedding model; historical index equivalence is unverified.
A run with any execution errors exits nonzero and preserves successful cases.
Triage validation allows two correction attempts after the initial response;
invalid quotes remain errors if the model cannot correct them. Other failures
propagate to the runner's per-case error records.
