"""Shared annotation vocabulary from Leung et al., EACL 2026, Section 4.

Error names/codes follow the paper; evidence checks are project annotation aids.
Older project labels are retained as history, never guessed into paper labels.
"""

PAPER_URL = "https://aclanthology.org/2026.eacl-long.147/"
TAXONOMY_VERSION = "ragec-e1-e16-2026"
STAGES = ["Chunking", "Retrieval", "Reranking", "Generation"]
LEGACY_TYPES = {
    "knowledge_boundary": "Knowledge boundary", "chunking": "Chunking",
    "retrieval": "Retrieval", "context_selection": "Context selection",
    "grounding": "Grounding", "response_coverage": "Response coverage",
}
EXTRA_TYPES = ["not_a_failure", "outside_taxonomy", "insufficient_evidence"]


def definition(code, name, stage, short, check, need, look, missing=""):
    return {"code": code, "name": name, "title": code + " · " + name,
            "stage": stage, "short": short, "check": check,
            "need": need, "look": look, "missing": missing}


CHUNKS_MISSING = "The original indexed chunk boundaries are not supplied here. Selected passages alone cannot establish a chunking error."
RETRIEVAL_MISSING = "The original retrieval candidate pool is not supplied here. Corpus search now is new evidence, not the original retrieval trace."
RERANK_MISSING = "Compare original candidates before and after reranking. Selected context alone cannot establish a reranker error; do not use these labels for a pipeline without a reranker."
DEFINITIONS = {
    "E1": definition("E1", "Overchunking", "Chunking", "Chunks too small or fragmented",
        "Inspect where a needed evidence unit was cut apart and whether that fragmentation prevents answering.",
        ["Original source and indexed chunks", "The lost evidence relationship"],
        ["question", "gold", "selected", "corpus"], CHUNKS_MISSING),
    "E2": definition("E2", "Underchunking", "Chunking", "Chunks too large and mixed",
        "Identify unrelated topics inside the indexed chunk and how they obscure the required evidence.",
        ["Original source and indexed chunk size", "Relevant and distracting portions of that chunk"],
        ["question", "gold", "selected", "corpus"], CHUNKS_MISSING),
    "E3": definition("E3", "Context Mismatch", "Chunking", "Chunk boundaries break contextual links",
        "Check whether a boundary separates an entity, definition, or antecedent from the text that depends on it.",
        ["Adjacent original chunks and source text", "The definition or reference lost at the boundary"],
        ["question", "gold", "selected", "corpus"], CHUNKS_MISSING),
    "E4": definition("E4", "Missed Retrieval", "Retrieval", "Needed evidence absent from retrieved candidates",
        "Locate usable evidence in the original index, then verify that it was absent from the original candidate pool.",
        ["Relevant indexed evidence", "Original retrieved candidate IDs and text"],
        ["question", "gold", "selected", "corpus"], RETRIEVAL_MISSING),
    "E5": definition("E5", "Low Relevance", "Retrieval", "Candidates only loosely related",
        "Compare the original retrieved candidates with the precise information requested; cite the relevance gap.",
        ["Original retrieved candidates", "The requested fact or facet that candidates fail to address"],
        ["question", "selected", "gold", "corpus"], RETRIEVAL_MISSING),
    "E6": definition("E6", "Semantic Drift", "Retrieval", "Keyword matches miss query intent",
        "Identify matching terms in the original candidates and show how their meaning differs from the user's request.",
        ["Original retrieved candidates", "Query intent and the misleading keyword match"],
        ["question", "selected", "history"], RETRIEVAL_MISSING),
    "E7": definition("E7", "Low Recall", "Reranking", "Reranker drops needed candidates",
        "Find a relevant original candidate that the reranker excluded from the context sent to generation.",
        ["Original candidates before reranking", "Reranker ranks or cutoff and final selected context"],
        ["selected", "gold", "corpus"], RERANK_MISSING),
    "E8": definition("E8", "Low Precision", "Reranking", "Reranker promotes irrelevant candidates",
        "Identify an irrelevant candidate promoted by the original reranker into the generator's context.",
        ["Original candidates and reranker output", "The irrelevant selected passage and its effect"],
        ["question", "selected", "answer"], RERANK_MISSING),
    "E9": definition("E9", "Abstention Failure", "Generation", "Answers when it should abstain",
        "Establish that the request is ambiguous or unanswerable from the corpus, and identify the unjustified answer.",
        ["Verified corpus scope or unresolved ambiguity", "The answer that should have acknowledged the limit"],
        ["question", "answer", "gold", "corpus"],
        "An unsupported selected passage alone does not prove that the corpus lacks an answer. An unnecessary refusal is not this paper's E9."),
    "E10": definition("E10", "Fabricated Content", "Generation", "Invented content in an answerable response",
        "Check answerability, identify the unsupported claim, and compare it with the selected context and verified evidence.",
        ["The exact invented claim", "Evidence for answerability and absence of support"],
        ["question", "answer", "selected", "gold", "corpus"],
        "A lexical mismatch or a judge flag alone is not evidence of fabrication."),
    "E11": definition("E11", "Parametric Overreliance", "Generation", "Internal knowledge overrides retrieved evidence",
        "Show that the answer substitutes outside model knowledge for the evidence supplied to generation.",
        ["The substituted claim and selected evidence", "Evidence distinguishing outside knowledge from invention"],
        ["answer", "selected", "gold"],
        "Do not guess the model's internal reasoning. If the record cannot distinguish this from fabrication, record the uncertainty."),
    "E12": definition("E12", "Incomplete Answer", "Generation", "Supported answer misses essential details",
        "Identify a required facet omitted from an otherwise supported answer and cite the available evidence for it.",
        ["An essential requested facet", "Selected evidence that could answer it"],
        ["question", "answer", "selected", "gold"],
        "Being shorter than the gold answer is not enough; establish that the omitted detail was needed."),
    "E13": definition("E13", "Misinterpretation", "Generation", "Misreads or misrepresents retrieved content",
        "Pair the mistaken answer claim with the passage it misreads, including entity or relationship confusion.",
        ["The exact answer claim", "The relevant selected passage and correct interpretation"],
        ["answer", "selected", "history"]),
    "E14": definition("E14", "Contextual Misalignment", "Generation", "Factual response answers a different question",
        "Verify the response facts, then show how they fail to address the actual request in its conversation context.",
        ["The requested information", "Supported response facts that address something else"],
        ["question", "answer", "selected", "history"]),
    "E15": definition("E15", "Chronological Inconsistency", "Generation", "Wrong timeline or event ordering",
        "Compare dates, event order, and time references in the response with the relevant evidence.",
        ["The erroneous temporal claim", "Evidence for the correct dates or ordering"],
        ["question", "answer", "selected", "history"]),
    "E16": definition("E16", "Numerical Error", "Generation", "Wrong calculation, quantity, or numerical reading",
        "Check quantities, units, operands, and arithmetic against the evidence; record the correct calculation when possible.",
        ["The erroneous number or calculation", "Source values, units, and the supported result"],
        ["answer", "selected", "gold"]),
}
TAXONOMY = list(DEFINITIONS)

# Plain-language annotation aids; examples illustrate labels, not predictions.
ANNOTATION_AIDS = {
    "E1": {"label": "Chunks are too small", "example": "A system name is in one chunk, but its explanation is cut into another."},
    "E2": {"label": "Chunks are too large", "example": "One chunk mixes many topics, hiding the evidence needed to answer."},
    "E3": {"label": "Chunk boundaries break a link", "example": "One chunk names a country; the next only says 'this country'."},
    "E4": {"label": "Needed evidence was not retrieved", "example": "The original index has an answer, but the retrieved list does not include it."},
    "E5": {"label": "Retrieved passages are not relevant enough", "example": "The question asks when a report was published, but the passages only discuss its contents."},
    "E6": {"label": "Matching words have the wrong meaning", "example": "The search finds the same words, but they refer to a different topic."},
    "E7": {"label": "Reranking removed needed evidence", "example": "Search found the answer, but its low rank kept it out of the final context."},
    "E8": {"label": "Reranking kept irrelevant evidence", "example": "A passage that does not answer the question is ranked high and sent to the model."},
    "E9": {"label": "Answered when it should not", "example": "The corpus has no funding details, but the answer claims the company raised no money."},
    "E10": {"label": "Made up unsupported details", "example": "The question is answerable, but the answer adds a medical history absent from the evidence."},
    "E11": {"label": "Used model knowledge instead of evidence", "example": "The answer uses a general assumption that the supplied documents do not support."},
    "E12": {"label": "Left out an important part", "example": "The question asks for two reasons. Evidence gives both, but the answer gives only one."},
    "E13": {"label": "Misread the evidence", "example": "The passage describes two people, but the answer mixes up their details."},
    "E14": {"label": "True facts, but not an answer to the question", "example": "The question asks how a crime happened, but the answer only gives the prison sentence."},
    "E15": {"label": "Got the time or event order wrong", "example": "The answer puts December before October in the same year."},
    "E16": {"label": "Got numbers, units, or calculations wrong", "example": "The evidence says 30 million baht, but the answer adds the figures as 50 million."},
}
STAGE_GUIDES = {
    "Chunking": {"label": "Split documents", "short": "Evidence loses meaning when text is split",
                 "evidence": "Check the original documents and the chunks stored in the index.",
                 "missing": "These exports do not include all original chunk boundaries. More evidence is needed to confirm this stage."},
    "Retrieval": {"label": "Find passages", "short": "Search misses or finds the wrong evidence",
                  "evidence": "Check the original index and the retrieved list before reranking.",
                  "missing": "The original retrieved list is not included here. A new corpus search cannot replace it."},
    "Reranking": {"label": "Rank passages", "short": "Ranking drops useful passages or keeps wrong ones",
                  "evidence": "Compare candidates before and after reranking with the final selected context.",
                  "missing": "Only use this stage if the system has a reranker and its before-and-after logs are available."},
    "Generation": {"label": "Write the answer", "short": "The answer uses the evidence incorrectly",
                   "evidence": "Check the question, answer, selected context, and relevant conversation history.",
                   "missing": "Choose this stage only when evidence supports it. Missing retrieval logs alone do not prove a generation error."},
}


# Short comparison notes for the annotation cards; paper definitions stay unchanged.
COMPARISON_NOTES = {
    "E1": "Check whether splitting a passage removed evidence needed to answer.",
    "E2": "Check whether a large chunk mixes useful evidence with unrelated topics.",
    "E3": "Check whether a chunk boundary separates a name or reference from its explanation.",
    "E4": "Find the needed evidence in Gold or Corpus, then check whether the original search missed it.",
    "E5": "Compare the question with the retrieved passages. Do they cover the exact information requested?",
    "E6": "Check whether matching words refer to a different meaning or topic.",
    "E7": "Check whether useful evidence was retrieved but removed before generation.",
    "E8": "Check whether reranking kept irrelevant passages in the final context.",
    "E9": "Check whether the corpus can answer the question. Should the model have said it could not answer?",
    "E10": "Find a claim in the answer that the selected context and verified evidence do not support.",
    "E11": "Check whether the answer replaces supplied evidence with outside knowledge. Do not guess the model’s reasoning.",
    "E12": "Find a requested detail that is supported by selected context but missing from the answer.",
    "E13": "Compare the answer’s claim with the selected passage. Check names, facts, and relationships.",
    "E14": "Check whether the answer addresses the actual question, even if its facts are true.",
    "E15": "Compare dates and the order of events in the answer with the evidence.",
    "E16": "Compare numbers and units with the evidence, then check the calculation."
}
for _code, _note in COMPARISON_NOTES.items():
    ANNOTATION_AIDS[_code]["compare_note"] = _note


def taxonomy_data():
    """Payload shared by live pages and standalone HTML exports."""
    return {"failure_types": TAXONOMY, "extra_types": EXTRA_TYPES,
            "failure_definitions": DEFINITIONS, "failure_stages": STAGES,
            "failure_taxonomy": TAXONOMY_VERSION, "failure_paper_url": PAPER_URL,
            "legacy_failure_labels": LEGACY_TYPES, "failure_annotation_aids": ANNOTATION_AIDS,
            "failure_stage_guides": STAGE_GUIDES}
