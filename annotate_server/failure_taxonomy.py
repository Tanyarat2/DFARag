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
    "E1": {"label": "แบ่ง chunk เล็กเกินไป", "example": "ชื่อระบบอยู่ chunk หนึ่ง แต่คำอธิบายถูกตัดไปอีก chunk จนตอบไม่ได้"},
    "E2": {"label": "แบ่ง chunk ใหญ่เกินไป", "example": "รวมหลายหัวข้อใน chunk เดียว จนข้อมูลที่ไม่เกี่ยวข้องกลบข้อมูลที่ต้องใช้"},
    "E3": {"label": "ตัดความหมายที่ต้องเชื่อมกัน", "example": "ชื่อประเทศอยู่ก่อนจุดตัด ส่วน chunk ถัดไปมีเพียงคำว่า ‘ประเทศนี้’"},
    "E4": {"label": "ค้นไม่เจอหลักฐานที่มีอยู่", "example": "มี chunk ที่ตอบได้ใน index เดิม แต่ไม่อยู่ในรายการที่ retrieve มา"},
    "E5": {"label": "ค้นได้ข้อมูลที่เกี่ยวเพียงกว้าง ๆ", "example": "ถามปีที่เผยแพร่รายงาน แต่ค้นได้แต่เนื้อหาทั่วไปในรายงาน"},
    "E6": {"label": "คำตรงกัน แต่คนละความหมาย", "example": "คำค้นตรงกับเอกสาร แต่เนื้อหาไม่ตรงเจตนาหรือประเด็นที่ถาม"},
    "E7": {"label": "reranker ตัดหลักฐานสำคัญออก", "example": "retrieve เจอคำตอบแล้ว แต่ถูกจัดอันดับต่ำจนไม่ถูกส่งให้โมเดล"},
    "E8": {"label": "reranker เลือกข้อมูลที่ไม่เกี่ยว", "example": "chunk ที่ไม่ตอบคำถามได้อันดับสูงและถูกส่งไปใช้สร้างคำตอบ"},
    "E9": {"label": "ควรงดตอบ แต่กลับตอบผิด", "example": "ไม่มีข้อมูลเงินระดมทุนใน corpus แต่ตอบยืนยันว่า ‘ไม่ได้ระดมทุน’"},
    "E10": {"label": "แต่งข้อมูลที่ไม่มีหลักฐาน", "example": "คำถามมีคำตอบ แต่โมเดลเพิ่มประวัติโรคที่ไม่ปรากฏในหลักฐาน"},
    "E11": {"label": "ใช้ความรู้ภายในแทนหลักฐาน", "example": "ใช้ข้อสมมติจากความรู้ทั่วไปมาตอบ ทั้งที่เอกสารไม่รองรับข้อสมมตินั้น"},
    "E12": {"label": "ตอบถูกบางส่วน แต่ขาดสิ่งสำคัญ", "example": "ถามเหตุผลสองข้อ มีหลักฐานครบ แต่ตอบถูกมาเพียงข้อเดียว"},
    "E13": {"label": "ตีความหลักฐานผิด", "example": "เอกสารกล่าวถึงคนสองคน แต่คำตอบนำข้อมูลของอีกคนมาปนกัน"},
    "E14": {"label": "ข้อมูลจริง แต่ตอบไม่ตรงคำถาม", "example": "ถามว่าก่อเหตุอย่างไร แต่ตอบเฉพาะโทษจำคุกที่ได้รับ"},
    "E15": {"label": "เวลา หรือลำดับเหตุการณ์ผิด", "example": "ตอบว่าเดือนธันวาคมเกิดก่อนเดือนตุลาคมในปีเดียวกัน"},
    "E16": {"label": "ตัวเลข หน่วย หรือคำนวณผิด", "example": "หลักฐานระบุ 30 ล้านบาท แต่คำตอบรวมตัวเลขผิดเป็น 50 ล้านบาท"},
}
STAGE_GUIDES = {
    "Chunking": {"label": "การแบ่งเอกสาร", "short": "ข้อมูลเสียความหมายตอนแบ่ง chunk",
                 "evidence": "ดูเอกสารต้นฉบับและขอบเขต chunks ที่ index ไว้จริง",
                 "missing": "ข้อมูลชุดนี้ไม่มีขอบเขต chunks เดิมครบ ต้องมีหลักฐานเพิ่มเพื่อยืนยัน"},
    "Retrieval": {"label": "การค้นหา", "short": "ค้นไม่เจอ หรือค้นได้ข้อมูลไม่ตรง",
                  "evidence": "ดู index เดิมและรายการ retrieve ก่อน rerank",
                  "missing": "ข้อมูลชุดนี้ไม่มีรายการ retrieve ก่อน rerank การค้น corpus ตอนนี้ใช้แทน log เดิมไม่ได้"},
    "Reranking": {"label": "การจัดอันดับ", "short": "เลือกหรือตัด candidates ผิดหลังค้นเจอ",
                  "evidence": "เทียบรายการก่อน–หลัง rerank กับ Selected context",
                  "missing": "ต้องมี reranker ในระบบนั้น และมี log ก่อน–หลัง rerank ซึ่งข้อมูลชุดนี้ไม่ได้แนบมา"},
    "Generation": {"label": "การสร้างคำตอบ", "short": "หลักฐานกับคำตอบแสดงข้อผิดพลาดตอนตอบ",
                   "evidence": "ดูคำถาม คำตอบ Selected context และประวัติสนทนาที่เกี่ยวข้อง",
                   "missing": "เลือกเมื่อมีหลักฐานรองรับ ห้ามสรุปเป็น Generation เพียงเพราะไม่มี retrieval log"},
}


def taxonomy_data():
    """Payload shared by live pages and standalone HTML exports."""
    return {"failure_types": TAXONOMY, "extra_types": EXTRA_TYPES,
            "failure_definitions": DEFINITIONS, "failure_stages": STAGES,
            "failure_taxonomy": TAXONOMY_VERSION, "failure_paper_url": PAPER_URL,
            "legacy_failure_labels": LEGACY_TYPES, "failure_annotation_aids": ANNOTATION_AIDS,
            "failure_stage_guides": STAGE_GUIDES}
