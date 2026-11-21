# Classification components

Start at `pipeline/classifier.py` (`FailureClassifier.classify`). Each folder has
its own README explaining the individual files inside it.

| Folder | What it does | Main files |
|---|---|---|
| `pipeline/` | Runs the workflow and adapts dataset records | `classifier.py`, `record_classifier.py` |
| `answerability/` | Makes the first evidence assessment and computes E9 | `assessor.py` |
| `prompts/` | Builds model prompts | `constructor.py` |
| `taxonomy/` | Defines E1–E16, stages and overlap rules | `labels.py` |
| `evidence/` | Prepares evidence and tracks actual text availability | `preparation.py`, `assessment.py`, `availability.py` |
| `validation/` | Checks citations, prerequisites and deterministic rules | `validator.py` |
| `review/` | Plans and applies bounded reviews | `workflow.py`, `output.py` |
| `data/` | Loads dataset rows and stores conversations/turns | `input_record.py`, `case_io.py`, `models.py` |
| `reporting/` | Builds and saves results/audits | `report.py`, `result.py`, `storage.py` |
| `api/` | Calls model APIs and manages request caches | `client.py` |
| `configuration/` | Reads YAML settings and CLI overrides | `config.py` |

Public imports remain short and stable:

```python
from failure_classification.errorclassification import (
    FailureClassifier, Conversation, ConversationTurn,
    InputRecord, EvidenceFirstFailureClassifier, save_classification_result,
)
```

Question generation is restored separately at `../questiongeneration/`.

## Answerability and E9

The first model request receives the original question, actual generator
context/history, evidence policy, and the existing answer. It excludes gold
answers and failure labels. Each requested aspect has answerability, a concrete
reason, response behavior, an exact assertion or limitation quote, and source
references. Placeholder reasons, missing records and invalid citations cannot
produce a completed assessment.

Python derives E9 from those same records: an unanswerable requested aspect was
substantively answered without acknowledging insufficient information. Falsity
is not required. Correct abstention or a supported partial answer with a clear
limitation is not E9. Later classification/review cannot overwrite the first
assessment or cast a second E9 vote. Computed E9 is emitted even when the model
omits it; when no other supported primary exists, it can supply the primary.
E10 and E13 remain separate mechanisms that may coexist with E9.

`answerability_assessment`, `answerability_audit`, `e9_assessment` and
`automatically_assigned_failures` make this visible. E10–E16 are independently
checked against actual generator evidence. Record checks do not independently
verify the semantic correctness of the model's judgments.

## Document thresholds

- E4: at least one chunk from every gold document must be retrieved.
- E7: at least 50% of distinct retrieved gold documents are lost after reranking.
- E8: fewer than two distinct retrieved gold documents occur in the top three
  entries of the ordered five-result reranking output.

`pipeline_diagnostics` and `pipeline_failures` record these findings separately
from answer quality and causal primary selection. With only one gold document,
the configured E8 requirement cannot be satisfied; this is retained explicitly.

## Results and request counts

`model_prediction` preserves model proposals, `validation` records checks and
origins, and `final_attribution` shows selected labels or unresolved status.
Missing evidence or no accepted label does not mean the answer is correct.
Historical result files are not rewritten.

The first assessment uses one request and at most one format/evidence repair.
Attribution uses one initial pass and at most one review, with at most one review
format correction: at most five requests, excluding transport retries.
`passes_used` counts attribution passes; `answerability_requests` and
`total_model_requests` expose the complete workflow. Missing generator records
skip the first request and leave answerability unknown.

See the parent README for dataset input, runner, gold review, viewer and tests.

## Dataset-record API

Record loading and result saving are part of this same package:

```python
from failure_classification.errorclassification import (
    InputRecord, EvidenceFirstFailureClassifier, save_classification_result,
)
from failure_classification.errorclassification.api.client import ClassifierRunner

records = InputRecord.load_jsonl("/path/to/annotations.jsonl")
runner = ClassifierRunner(model="gpt-4o-mini", type_examples=[], coverage_examples=[])
try:
    classifier = EvidenceFirstFailureClassifier(runner, max_passes=2, max_failures=16)
    result = classifier.classify_record(records[0])
    save_classification_result(result, "/path/to/results/first_record.json")
finally:
    runner.client.close()
```

Use `FailureClassifier` directly for prepared case dictionaries or conversations.
Both interfaces call the same engine. The former separate adapter folder has
been merged here; update imports to `failure_classification.errorclassification`.
