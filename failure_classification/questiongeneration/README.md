# Question generation

This is your restored question-generation code. It was previously moved to the
project-level legacy archive during cleanup; it is active here again. The archive
copy is retained as historical reference, not used at runtime.

| File | Purpose |
|---|---|
| `xie_method.py` | B1: generate subquestions, classify their type, and check answer/retrieval coverage following the notebooks |
| `dynamic_followup_q.py` | B2: generate/review follow-up questions and collect actual RAG responses |
| `diagnostic_verification.py` | Validate diagnostic questions and their original-answer connection |
| `evaluator.py` | Separate dynamic diagnostic evaluator using an LLM and search adapter |
| `run_dynamic_evaluator.py` | Run that evaluator on saved Dragonball traces |
| `build_dynamic_evaluator_viewer.py` | Display saved dynamic-evaluator results |

The E1–E16 classifier is in `errorclassification/`. These generators are separate
entry points: classification does not silently generate new questions. The
six-category dynamic evaluator is also distinct from the E1–E16 classifier.

## B1 / notebook-style generation

Install Excel support if using the notebook annotation examples:

```bash
python -m pip install './failure_classification[excel]'
python -m failure_classification.questiongeneration.xie_method \
  --input /path/to/single_turn_dragonball.json \
  --output-dir /path/to/results/b1 --limit 10 --model gpt-4o-mini \
  --type-workbook /path/to/annotation_merged_v2.xlsx \
  --coverage-workbook '/path/to/Answerability Annotation.xlsx'
```

Input data and annotation workbooks remain external assets; they are not bundled
into the source folder.

## B2 / dynamic follow-up generation

```bash
python -m failure_classification.questiongeneration.dynamic_followup_q \
  --input /path/to/single_turn_dragonball.json \
  --output-dir /path/to/results/b2 --limit 10 --model gpt-4o-mini \
  --review-model gpt-5.6-luna --rag-adapter my_adapter:ask_rag
```

The adapter is an importable function accepting a question and returning an
actual RAG response (optionally with traces). The built-in default adapter is
`failure_classification.rag.dragonball_rag_adapter:ask_rag`; it requires the
separate `rag-error-classification` repository, its model/runtime dependencies,
and corpus/config files. A custom adapter can be used when moving this package
to another project. No RAG backend, corpus, model weights or credentials are bundled.

See `README_dynamic_evaluator.md` for the separate dynamic evaluator's interface.
Tests use offline stubs and do not exercise live model/index inference.
