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
EXTRA_TYPES = ["not_a_failure", "outside_taxonomy", "insufficient_evidence", "excluded"]


def definition(code, name, stage, short, check, need, look, missing=""):
    return {"code": code, "name": name, "title": code + " · " + name,
            "stage": stage, "short": short, "check": check,
            "need": need, "look": look, "missing": missing}


CHUNKS_MISSING = 'Original chunk boundaries are missing. Selected passages alone cannot confirm a chunking error.'
RETRIEVAL_MISSING = 'The original retrieved list is missing. A new corpus search cannot show what the original search returned.'
RERANK_MISSING = 'Use E7 or E8 only if the system used a reranker and its before-and-after records show the ranking error.'
DEFINITIONS = {
    'E1': definition('E1',
        'Overchunking',
        'Chunking',
        'Chunks are too small',
        'Small or broken chunks leave out meaning needed to answer. Compare the chunk with its source and nearby chunks.',
        ['The needed text in Gold or Corpus', 'The original chunks showing where that text was cut apart'],
        ['question', 'gold', 'selected', 'corpus'],
        CHUNKS_MISSING),
    'E2': definition('E2',
        'Underchunking',
        'Chunking',
        'Chunks are too large',
        'A large chunk mixes several topics. Unrelated text hides the needed facts or distracts the model.',
        ['The large indexed chunk and its source', 'The useful facts and unrelated text mixed inside it'],
        ['question', 'gold', 'selected', 'corpus'],
        CHUNKS_MISSING),
    'E3': definition('E3',
        'Context Mismatch',
        'Chunking',
        'Chunk boundaries break a link',
        'A split separates a name, definition, or explanation from text that depends on it. The chunks lose the connection needed to find or understand the answer.',
        ['The source and the two chunks on either side of the split', 'The name or explanation that the next chunk needs'],
        ['question', 'gold', 'selected', 'corpus'],
        CHUNKS_MISSING),
    'E4': definition('E4',
        'Missed Retrieval',
        'Retrieval',
        'Needed evidence was not retrieved',
        'Usable answer evidence exists in the index, but search missed some needed evidence. Missing a gold document alone is not enough if other retrieved passages fully answer the question.',
        ['Usable answer evidence in Gold or Corpus that was indexed', 'The retrieved passages before reranking, with the needed evidence missing'],
        ['question', 'gold', 'selected', 'corpus'],
        RETRIEVAL_MISSING),
    'E5': definition('E5',
        'Low Relevance',
        'Retrieval',
        'Retrieved passages are not relevant enough',
        'The retrieved passages are only loosely related to the question. They discuss the topic but give little help with the exact fact requested.',
        ['The exact information requested', 'Retrieved passages that discuss the topic without giving that information'],
        ['question', 'selected', 'gold', 'corpus'],
        RETRIEVAL_MISSING),
    'E6': definition('E6',
        'Semantic Drift',
        'Retrieval',
        "Matching words miss the question's meaning",
        'Search matches names or words but misses what the question is asking for. A word can have the same meaning and still match the wrong information.',
        ['The names or words that search matched', 'The requested meaning or detail that the matched passages fail to provide'],
        ['question', 'selected', 'history'],
        RETRIEVAL_MISSING),
    'E7': definition('E7',
        'Low Recall',
        'Reranking',
        'Reranking removed needed evidence',
        'Search retrieved a needed passage, but reranking put it below the cutoff and left it out of Selected context.',
        ['The needed passage in Retrieved context before reranking', 'Its reranking result and its absence from Selected context'],
        ['selected', 'gold', 'corpus'],
        RERANK_MISSING),
    'E8': definition('E8',
        'Low Precision',
        'Reranking',
        'Reranking kept irrelevant evidence',
        'Reranking places irrelevant passages high enough to enter Selected context. Those passages distract the model or lead it to a wrong conclusion.',
        ['The question and the irrelevant passage', 'Its reranking result and inclusion in Selected context'],
        ['question', 'selected', 'answer'],
        RERANK_MISSING),
    'E9': definition('E9',
        'Abstention Failure',
        'Generation',
        'Answered when it should not',
        'The corpus cannot answer the question, or the question is too unclear. The model still gives an unjustified answer instead of saying it cannot answer or asking for clarification.',
        ['The question, Gold answer, and relevant corpus evidence', 'The claim made despite missing information or an unclear question'],
        ['question', 'answer', 'gold', 'corpus'],
        'Missing selected evidence alone does not prove that the corpus has no answer. An unnecessary refusal is not E9.'),
    'E10': definition('E10',
        'Fabricated Content',
        'Generation',
        'Made up unsupported details',
        'The question is answerable, but the answer adds an invented claim unsupported by Selected context or other verified evidence. The claim can still be about the same topic.',
        ['The exact unsupported claim in the RAG answer', 'Selected context and verified Gold or Corpus evidence'],
        ['question', 'answer', 'selected', 'gold', 'corpus'],
        'Different wording or a judge flag alone does not prove that a claim was made up.'),
    'E11': definition('E11',
        'Parametric Overreliance',
        'Generation',
        'Used model knowledge instead of evidence',
        'The answer relies on general model knowledge or assumptions instead of the supplied evidence. That knowledge may be true. Do not choose this only because a claim is missing from Selected context.',
        ['The outside fact or assumption used in the answer', 'The supplied evidence and why it does not justify using that fact here'],
        ['answer', 'selected', 'gold'],
        'If the record cannot distinguish model knowledge from invented content, flag the turn and explain what is unclear.'),
    'E12': definition('E12',
        'Incomplete Answer',
        'Generation',
        'Left out an important part',
        'An otherwise supported answer leaves out an essential part of the question that Selected context could answer. A shorter answer than Gold is not automatically incomplete.',
        ['The essential detail requested by the question', 'Evidence for that detail in Selected context and its absence from the answer'],
        ['question', 'answer', 'selected', 'gold'],
        'Check whether the missing detail is needed. Extra detail in Gold is not always required.'),
    'E13': definition('E13',
        'Misinterpretation',
        'Generation',
        'Misread the evidence',
        'The answer misunderstands, incorrectly combines, or misstates Selected context. It may directly answer the question but get the facts or relationships wrong.',
        ['The mistaken claim in the RAG answer', 'The selected passage and what it actually means'],
        ['answer', 'selected', 'history']),
    'E14': definition('E14',
        'Contextual Misalignment',
        'Generation',
        'True facts, but not an answer to the question',
        'The answer gives supported facts but misses what the question asks. It can discuss the same person or topic while answering the wrong aspect.',
        ['The exact information requested by the question', 'The answer and selected evidence showing its facts are true but do not answer that request'],
        ['question', 'answer', 'selected', 'history']),
    'E15': definition('E15',
        'Chronological Inconsistency',
        'Generation',
        'Got the time or event order wrong',
        'The answer gives a wrong date, puts events in the wrong order, or mixes up past, present, and future.',
        ['The date, order, or time claim in the RAG answer', 'Selected context and Gold evidence for the correct timeline'],
        ['question', 'answer', 'selected', 'history']),
    'E16': definition('E16',
        'Numerical Error',
        'Generation',
        'Got numbers, units, or calculations wrong',
        'The answer misreads a quantity, uses a wrong unit, or makes a calculation error. Check the source values and the calculation.',
        ['The number, unit, or calculation in the RAG answer', 'The source values in Selected context and the supported result'],
        ['answer', 'selected', 'gold']),
}
TAXONOMY = list(DEFINITIONS)

# Plain-language annotation aids; examples illustrate labels, not predictions.
ANNOTATION_AIDS = {'E1': {'label': 'Chunks are too small',
        'example': 'One chunk says a company launched a platform, but another chunk holds the words online '
                   'booking platform. The retrieved fragment cannot identify it.'},
 'E2': {'label': 'Chunks are too large',
        'example': 'One chunk mixes asset controls, market risks, and company plans. The answer uses the '
                   'risk discussion to answer a question about asset controls.'},
 'E3': {'label': 'Chunk boundaries break a link',
        'example': 'Chunk A names the Philippines. Chunk B says the country faces the Pacific Ocean, but '
                   'does not name it. Search misses Chunk B because the link was split.'},
 'E4': {'label': 'Needed evidence was not retrieved',
        'example': 'A question asks about two companies. Evidence for both is indexed, but search returns '
                   'evidence for only one.'},
 'E5': {'label': 'Retrieved passages are not relevant enough',
        'example': 'The question asks when a report was published. Search returns passages about events in '
                   'the report, without its publication year.'},
 'E6': {'label': "Matching words miss the question's meaning",
        'example': "The question asks for two defendants' prison sentences. Search finds both cases, but the "
                   'passages about one defendant discuss the case without giving the sentence.'},
 'E7': {'label': 'Reranking removed needed evidence',
        'example': "Search finds both people's sentences. Reranking drops the passage for one person, so the "
                   'model only receives the other sentence.'},
 'E8': {'label': 'Reranking kept irrelevant evidence',
        'example': 'A high-ranked passage mentions New York but says nothing about a stock exchange. The '
                   'model uses it to guess New York Stock Exchange.'},
 'E9': {'label': 'Answered when it should not',
        'example': 'There is no evidence about fundraising. The model says the company raised no money '
                   'instead of saying the amount is unknown.'},
 'E10': {'label': 'Made up unsupported details',
         'example': "The records support a patient's medical history, but the answer adds a disease that the "
                    'evidence does not support.'},
 'E11': {'label': 'Used model knowledge instead of evidence',
         'example': "The records do not give a company's total liabilities. The model uses general financial "
                    'rules to calculate a total anyway.'},
 'E12': {'label': 'Left out an important part',
         'example': 'The question asks for two reasons. Selected context gives both, but the answer gives '
                    'only one.'},
 'E13': {'label': 'Misread the evidence',
         'example': 'The passage says Mina founded Alpha and Niran became its CEO. The answer says Niran '
                    'founded Alpha.'},
 'E14': {'label': 'True facts, but not an answer to the question',
         'example': 'The question asks who founded Alpha. The answer correctly names its CEO, but does not '
                    'name its founder.'},
 'E15': {'label': 'Got the time or event order wrong',
         'example': 'The answer says an event in December happened before an event in October of the same '
                    'year.'},
 'E16': {'label': 'Got numbers, units, or calculations wrong',
         'example': 'The evidence supports a total of 30 million, but the answer incorrectly adds figures to '
                    'get 50 million.'}}
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
COMPARISON_NOTES = {'E1': 'Look for evidence lost because chunks are too small.',
 'E2': 'Look for a large chunk whose mixed topics hide or distract from the answer.',
 'E3': 'Check whether a split breaks the link between a name and its explanation.',
 'E4': 'Check for missing needed evidence, not just a missing gold document.',
 'E5': 'Check whether the retrieved passages help with the exact question.',
 'E6': 'Check whether matching names or words missed the requested meaning or detail.',
 'E7': 'Find needed evidence in Retrieved context that reranking left out of Selected context.',
 'E8': 'Check whether reranking gave irrelevant passages priority in Selected context.',
 'E9': 'Check whether the model should have said it could not answer or asked for clarification.',
 'E10': 'Find an invented claim, even if it sounds relevant to the question.',
 'E11': 'Look for outside knowledge replacing evidence. If you cannot tell, flag the turn rather than guess.',
 'E12': 'Find a needed detail available in Selected context but missing from the answer.',
 'E13': 'Check whether the answer changed the meaning of the selected evidence.',
 'E14': 'Check whether true facts answer the requested question or only a different aspect.',
 'E15': 'Check dates, event order, and whether facts belong to the past, present, or future.',
 'E16': 'Check the source numbers, units, and calculation against the answer.'}
for _code, _note in COMPARISON_NOTES.items():
    ANNOTATION_AIDS[_code]["compare_note"] = _note


def taxonomy_data():
    """Payload shared by live pages and standalone HTML exports."""
    return {"failure_types": TAXONOMY, "extra_types": EXTRA_TYPES,
            "failure_definitions": DEFINITIONS, "failure_stages": STAGES,
            "failure_taxonomy": TAXONOMY_VERSION, "failure_paper_url": PAPER_URL,
            "legacy_failure_labels": LEGACY_TYPES, "failure_annotation_aids": ANNOTATION_AIDS,
            "failure_stage_guides": STAGE_GUIDES}
