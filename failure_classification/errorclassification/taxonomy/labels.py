"""E1–E16 definitions, stages, and accepted labels. Edit definitions here."""
TAXONOMY_REVISION = 'e9_unanswerable_substantive_answer_20261108'


E1_Overchunking = r'''- E1 Overchunking: Documents are split into excessively small chunks, losing
  important context and making individual chunks incomplete or ambiguous.'''


E2_Underchunking = r'''- E2 Underchunking: Chunks are too large and mix multiple topics, confusing or
  diluting important information.'''


E3_Context_Mismatch = r'''- E3 Context Mismatch: Arbitrary chunk boundaries disrupt logical structure
  and separate information from the contextual links needed to interpret it.'''


E4_Missed_Retrieval = r'''- E4 Missed Retrieval: Initial retrieval must return ALL gold/reference
  documents. Assign E4 when at least one gold document is absent from the complete
  retrieved candidate set (gold-document recall below 100%). Any retrieved chunk
  from a gold document counts as retrieving that document; all gold chunks are
  not required. Compare document IDs, not wording or answer sufficiency.
  Equivalent evidence from a different document does not cancel E4. Report the
  gold IDs, retrieved gold IDs, missing gold IDs, and document recall. Missing
  gold IDs or incomplete retrieval records leave absence unresolved.
  This is a coverage failure and need not imply the answer is incorrect.'''


E5_Low_Relevance = r'''- E5 Low Relevance: Initial retrieval returns chunks with weak substantive
  relevance to the original information need. They may share a topic, entity,
  or broad subject with the question but do not meaningfully address its
  requested facts or relationships. Cite the actual retrieval results and the
  unmet request; an incorrect final answer alone does not establish E5.
  A same-company, same-year sustainability passage does not answer a revenue
  question, but missing revenue alone does not prove E6 semantic drift.
  E6 requires an identifiable entity, timeframe, relationship, interpretation,
  or information-need mismatch. Do not redundantly add E5 to E4 for the same
  missing-evidence symptom without independent support.'''


E6_Semantic_Drift = r'''- E6 Semantic Drift: Retrieved chunks match keywords or surface wording but
  fail to align with the query's intended meaning. Identify a specific mismatch
  in entity, timeframe, relationship, interpretation, or information need.
  A passage on the correct topic that lacks the requested answer is not by
  itself semantic drift.'''


E7_Low_Recall = r'''- E7 Low Recall: At least 50% of the distinct gold documents present in
  initial retrieved candidates are absent from the final reranked context.
  Denominator: retrieved gold documents, not all gold documents or chunks.
  Any surviving chunk from a document retains that document. Exactly 50% loss
  triggers E7. No retrieved gold documents leaves this ratio undefined.'''


E8_Low_Precision = r'''- E8 Low Precision: Among the top K=3 entries in the ordered five-result
  reranking output, fewer than ceil(0.5*K)=2 distinct retrieved gold documents
  appear. At least two gold documents in the top three passes this criterion.
  Count document IDs once; duplicate chunks do not count as different documents.
  Gold documents ranked fourth or fifth do not satisfy top-three coverage.
  This threshold is independent of answer correctness or a harmful-selection
  mechanism. Incomplete or unordered reranking evidence leaves E8 unresolved.'''


E9_Abstention_Failure = r'''- E9 Abstention Failure: The requested answer, or a necessary requested aspect,
  cannot be established from the permitted evidence, but the generator supplies
  a substantive answer to that aspect instead of acknowledging that the
  information is unavailable or insufficient, abstaining, or asking for clarification.
  Proof that the asserted answer is false is NOT required.

  Establish three findings:
  1. Identify the question or requested aspect that is unanswerable from
     sufficiently complete actual generator context and reliable permitted history,
     respecting any permission to use outside knowledge.
  2. Show that acknowledging insufficient information, abstention, or clarification
     was warranted for that aspect.
  3. Identify the substantive answer asserted for that unanswerable aspect.

  Assess answerability and response behavior separately. Missing logs mean
  unknown, not unanswerable. A gold 'Unable to answer' annotation is evaluator
  guidance, not automatic proof that the actual permitted input cannot answer.
  A substantive answer may be plausible or even true and still be E9 under a
  source-grounded evidence policy if the permitted evidence cannot establish it.

  For partially answerable requests, identify the unanswerable aspect AND the
  assertion answering it. Correctly answering supported parts while clearly
  acknowledging the unavailable part is not E9. Optional unsupported background
  does not automatically establish failure to abstain on the requested answer.
  Appropriate abstention is not a failure. Refusal despite sufficient usable
  evidence is not E9. Missing generator support does not prove corpus-wide absence.

  E9 assesses the failure to acknowledge an unanswerable request. E10 separately
  assesses unsupported factual content; E13 assesses misuse of supplied facts.
  E9 may coexist with E10, E13 or another label when that label's own mechanism
  is also established. Neither label follows automatically from the other.'''


E10_Fabricated_Content = r'''- E10 Fabricated Content: Identify a factual claim in the original answer that
  is unsupported by the evidence and reliable history actually permitted to the
  generator. The required answer can be present while the response adds invented
  details, such as an unsupported launch location or attendance figure. Compare
  each disputed claim with the complete allowed input; a fact absent only from
  the gold answer is not enough. Respect any explicit permission to use outside
  knowledge. E10 concerns unsupported factual content, whereas E9 concerns the
  decision to answer an unanswerable aspect without acknowledging the limitation.
  E9 and E10 may describe the same assertion from these different perspectives
  when both prerequisites are established; neither automatically implies the other.'''


E11_Parametric_Overreliance = r'''- E11 Parametric Overreliance: The response substitutes the model's internal
  knowledge for the supplied source evidence. Require additional observable
  evidence establishing that reliance; a wrong answer, contradiction, familiar
  historical fact, or unsupported claim alone cannot reveal its internal origin.
  A later model explanation of its own behavior is not an authoritative trace.
  If the information's origin cannot be established, leave E11 unresolved and
  assess an observable E10 or E13 defect on its own evidence. Respect the task's
  actual policy on permitted outside knowledge.'''


E12_Incomplete_Answer = r'''- E12 Incomplete Answer: The response correctly addresses part of the original
  request but omits another essential requested detail that the generator had
  sufficient usable evidence to provide. Identify the exact requested aspect,
  quote its support in actual generation context/history, and show the omission.
  For example, revenue and profit were both supplied, but only revenue was
  answered. A missing fact available only in gold or an unselected chunk does
  not establish this generation failure; investigate E4/E7 or another upstream
  loss. Do not demand optional background details. A partial supported answer
  that clearly explains an unavailable aspect is not E12 merely for omitting it.'''


E13_Misinterpretation = r'''- E13 Misinterpretation: The generator misreads, transforms, binds, or applies
  information actually supplied to it. Cite the exact supplied fact and the
  response's conflicting interpretation: for example, reversing buyer and seller,
  attributing one person's achievement to another, resolving a reference wrongly,
  or applying a fact about one event to a different event. Gold-only support
  cannot prove the generator misread a passage it never received. Prefer E13 to
  E10 when the specific misuse of a real supplied fact is demonstrated; do not
  infer misuse merely because an unsupported answer differs from gold.'''


E14_Contextual_Misalignment = r'''- E14 Contextual Misalignment: The response supplies supported, potentially
  correct facts but addresses a different information need from the original
  question. Identify the requested relation or question type and the relation
  actually answered. For example, the user asks why an acquisition occurred,
  the supplied evidence explains the motive, but the answer gives only its date
  and price. E12 answers part of the requested need while omitting another part;
  E14 substitutes a different need. An incorrect factual claim is not by itself
  this category, and an ambiguous question does not justify inventing a user
  constraint from the gold reference.'''


E15_Chronological_Inconsistency = r'''- E15 Chronological Inconsistency: The response contradicts the timeline,
  ordering, or past/present/future relationships established by supplied evidence.
  Identify the events and quote the dates or ordering needed to check them.
  For example, the context shows a 2020 launch before a 2021 acquisition, but
  the answer says the launch happened afterward. A date appearing in a wrong
  answer is not automatically E15: applying a real date to the wrong entity or
  event may instead be E13. Select the demonstrated temporal mechanism.'''


E16_Numerical_Error = r'''- E16 Numerical Error: The response incorrectly calculates, copies, scales,
  converts, or represents numerical information available in supplied evidence.
  Quote the operands or original quantity and identify the erroneous result,
  including relevant units and conditions. For example, an increase from 80 to
  100 is 20, not 30; reporting 100 million as 1 billion also qualifies. A number
  appearing in an unsupported claim does not automatically establish E16: if
  no supplied quantity or valid calculation grounds the numerical check,
  investigate E10 or another mechanism rather than assuming arithmetic failed.'''


DEFINITIONS = "\n".join((
    E1_Overchunking,
    E2_Underchunking,
    E3_Context_Mismatch,
    E4_Missed_Retrieval,
    E5_Low_Relevance,
    E6_Semantic_Drift,
    E7_Low_Recall,
    E8_Low_Precision,
    E9_Abstention_Failure,
    E10_Fabricated_Content,
    E11_Parametric_Overreliance,
    E12_Incomplete_Answer,
    E13_Misinterpretation,
    E14_Contextual_Misalignment,
    E15_Chronological_Inconsistency,
    E16_Numerical_Error,
))


OVERLAP_RULES = r'''OVERLAP RULES

- E1 versus E2 versus E3: Demonstrate the particular preparation defect and
  its effect on the original requirement. Prefer E3 for a specifically broken
  boundary/link, E1 for context lost by excessive splitting, and E2 for
  demonstrated harmful topic mixing. Size alone is insufficient.
- E4 checks all-gold-document coverage independently of answer sufficiency.
  E5 assesses weak substantive relevance; E6 requires a specific semantic
  mismatch. Do not infer E5/E6 merely from a missing gold document.
- E7 versus E8: E7 checks loss of at least half the retrieved gold documents
  across reranking; E8 checks at least two distinct retrieved gold documents
  in the top three of five reranked entries. Both may independently apply.
  Neither requires demonstrating an answer defect or semantic evidence loss.
- E10 versus E13: Prefer E13 when a documented fact is transformed or applied
  incorrectly, such as using the partner of a June event to answer a question
  about a September event. Prefer E10 for invented factual content lacking
  support in the actual permitted input. Do not relabel every misapplication
  as both categories merely because its resulting claim is unsupported.
- E9 versus E10/E13: E9 is the decision to answer an unanswerable requested
  aspect without acknowledging insufficient information; falsity is not required.
  E10 is unsupported factual content; E13 is demonstrated misuse of supplied facts.
  E9 can coexist with either when each has its own evidence. Prefer E13 to E10
  for the same supplied-fact misuse, while retaining an independently established E9.
- E9 versus upstream failures: E9 evaluates the generator's handling of
  insufficient usable input. A separately demonstrated earlier evidence loss
  can be the primary upstream failure while E9 is an additional observed
  generation failure. Do not require corpus-wide absence for E9 or infer an
  upstream failure solely because the generator should have abstained.
  An appropriate scoped abstention is acceptable. A refusal despite sufficient
  evidence is not E9 as defined; do not force a label whose definition is unmet.
- E11 versus E10: Do not infer internal-knowledge reliance solely from an
  unsupported statement. If the source of the claim is unknown, E10 may be
  supported while E11 remains unresolved. Respect any explicit permission to
  use outside knowledge rather than treating all such use as automatically bad.
- E12 versus E14: E12 leaves an essential requested detail out of an otherwise
  correct answer; E14 answers a different information need despite factual
  correctness. Wrong values are not just omissions. Both a supported omission
  and a separate incorrect assertion can be additional distinct findings.
- E15 versus E13 versus E16: Prefer E15 for a demonstrated timeline/order
  error and E16 for arithmetic or quantity errors. Selecting a real fact about
  the wrong event or relation is E13; a date appearing in that relation does
  not alone make it E15 or E16. Choose the specific evidenced mechanism, not
  an inferred internal process.'''


from dataclasses import dataclass
from enum import Enum


class RagFailureStage(str, Enum):
    DOCUMENT_PREPROCESSING = 'Chunking'
    RETRIEVAL_FAILURES = 'Retrieval'
    RERANKING_FAILURES = 'Reranking'
    ANSWER_GENERATION = 'Generation'
    UNKNOWN = 'unknown'

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class FailureCategory:
    code: str
    name: str
    description: str
    stage: RagFailureStage

    @property
    def label(self):
        return f'{self.code} {self.name}'


def _category(code, name, definition, stage):
    return FailureCategory(code, name, definition.split(': ', 1)[1], stage)


OVERCHUNKING = _category(
    'E1', 'Overchunking', E1_Overchunking,
    RagFailureStage.DOCUMENT_PREPROCESSING,
)

UNDERCHUNKING = _category(
    'E2', 'Underchunking', E2_Underchunking,
    RagFailureStage.DOCUMENT_PREPROCESSING,
)

CONTEXT_MISMATCH = _category(
    'E3', 'Context Mismatch', E3_Context_Mismatch,
    RagFailureStage.DOCUMENT_PREPROCESSING,
)

MISSED_RETRIEVAL = _category(
    'E4', 'Missed Retrieval', E4_Missed_Retrieval,
    RagFailureStage.RETRIEVAL_FAILURES,
)

LOW_RELEVANCE = _category(
    'E5', 'Low Relevance', E5_Low_Relevance,
    RagFailureStage.RETRIEVAL_FAILURES,
)

SEMANTIC_DRIFT = _category(
    'E6', 'Semantic Drift', E6_Semantic_Drift,
    RagFailureStage.RETRIEVAL_FAILURES,
)

LOW_RECALL = _category(
    'E7', 'Low Recall', E7_Low_Recall,
    RagFailureStage.RERANKING_FAILURES,
)

LOW_PRECISION = _category(
    'E8', 'Low Precision', E8_Low_Precision,
    RagFailureStage.RERANKING_FAILURES,
)

ABSTENTION_FAILURE = _category(
    'E9', 'Abstention Failure', E9_Abstention_Failure,
    RagFailureStage.ANSWER_GENERATION,
)

FABRICATED_CONTENT = _category(
    'E10', 'Fabricated Content', E10_Fabricated_Content,
    RagFailureStage.ANSWER_GENERATION,
)

PARAMETRIC_OVERRELIANCE = _category(
    'E11', 'Parametric Overreliance', E11_Parametric_Overreliance,
    RagFailureStage.ANSWER_GENERATION,
)

INCOMPLETE_ANSWER = _category(
    'E12', 'Incomplete Answer', E12_Incomplete_Answer,
    RagFailureStage.ANSWER_GENERATION,
)

MISINTERPRETATION = _category(
    'E13', 'Misinterpretation', E13_Misinterpretation,
    RagFailureStage.ANSWER_GENERATION,
)

CONTEXTUAL_MISALIGNMENT = _category(
    'E14', 'Contextual Misalignment', E14_Contextual_Misalignment,
    RagFailureStage.ANSWER_GENERATION,
)

CHRONOLOGICAL_INCONSISTENCY = _category(
    'E15', 'Chronological Inconsistency', E15_Chronological_Inconsistency,
    RagFailureStage.ANSWER_GENERATION,
)

NUMERICAL_ERROR = _category(
    'E16', 'Numerical Error', E16_Numerical_Error,
    RagFailureStage.ANSWER_GENERATION,
)

FAILURE_CATEGORIES = (
    OVERCHUNKING,
    UNDERCHUNKING,
    CONTEXT_MISMATCH,
    MISSED_RETRIEVAL,
    LOW_RELEVANCE,
    SEMANTIC_DRIFT,
    LOW_RECALL,
    LOW_PRECISION,
    ABSTENTION_FAILURE,
    FABRICATED_CONTENT,
    PARAMETRIC_OVERRELIANCE,
    INCOMPLETE_ANSWER,
    MISINTERPRETATION,
    CONTEXTUAL_MISALIGNMENT,
    CHRONOLOGICAL_INCONSISTENCY,
    NUMERICAL_ERROR,
)

FAILURE_TYPES = tuple(category.label for category in FAILURE_CATEGORIES)


STAGES = ('Chunking', 'Retrieval', 'Reranking', 'Generation')
CODE_TO_NAME = {category.code: category.name for category in FAILURE_CATEGORIES}
CODE_TO_STAGE = {category.code: category.stage.value for category in FAILURE_CATEGORIES}
