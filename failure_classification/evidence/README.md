# Diagnostic source helpers

- `dragonball_reconstruction.py`: reconstruct source chunks for diagnostic search
  and preserve explicit scope/provenance limits; used by the dynamic evaluator.
- `gold_documents.py`: load source documents from local benchmark provenance.

These are dataset/reconstruction helpers for question-generation diagnostics.
The classifier's citation catalogs and availability checks are separately grouped
under `errorclassification/evidence/`.
