import unittest

from failure_classification.questiongeneration.evaluator import (
    Artifacts, Failure, FailureClassifier, InconclusiveDiagnosis,
)


class DynamicEvaluatorTests(unittest.TestCase):
    def run_case(self, category, exhaustive=True, recovered=False):
        artifacts = Artifacts(user_query="What was Acme's revenue in 2023?",
                              top_k_chunks=["Acme 2023", "revenue was 10"],
                              prompt_context=["revenue was 10"], rag_answer="Unknown")
        calls = []
        citation = {"chunk_index": 0, "quote": "revenue was 10"}

        def llm(prompt, payload):
            if "dynamic_sub_questions" in prompt:
                return dict(extracted_constraints=["Acme revenue in 2023"],
                            identified_delta="Revenue missing or unsupported",
                            dynamic_sub_questions=["What was Acme's revenue in 2023?"])
            if "fragmented_relationship" in prompt:
                return dict(
                    ungrounded_claims=["Revenue was 20"] if category == "GROUNDING_FAILURE" else [],
                    omitted_requirements=["revenue"] if category == "RESPONSE_COVERAGE_FAILURE" else [],
                    prompt_support=[citation] if category == "RESPONSE_COVERAGE_FAILURE" else [],
                    retrieved_support=[dict(chunk_index=1, quote="revenue was 10")]
                    if category == "CONTEXT_SELECTION_FAILURE" else [],
                    fragmented_relationship=category == "CHUNKING_FAILURE",
                    boundary_evidence=[dict(chunk_index=0, quote="Acme 2023"),
                                       dict(chunk_index=1, quote="revenue was 10")]
                    if category == "CHUNKING_FAILURE" else [], explanation="Observed stage evidence")
            supported = payload["chunks"] == ["Acme revenue in 2023 was 10"]
            return dict(supported=supported,
                        evidence=[dict(chunk_index=0, quote=payload["chunks"][0])] if supported else [],
                        explanation="Semantic assessment")

        def search(question, top_k):
            calls.append(question)
            return dict(chunks=["Acme revenue in 2023 was 10"] if recovered else [],
                        exhaustive=exhaustive, scope="test corpus")

        result = FailureClassifier(llm, search).evaluate(artifacts)
        self.assertEqual(result.attributed_failure.value, category)
        self.assertEqual(bool(calls), category in {"RETRIEVAL_FAILURE", "KNOWLEDGE_BOUNDARY_FAILURE"})
        self.assertEqual(set(result.model_dump()), {"extracted_constraints", "identified_delta",
                         "dynamic_sub_questions", "attributed_failure", "justification"})

    def test_six_decision_branches(self):
        for failure in Failure:
            with self.subTest(failure=failure):
                self.run_case(failure.value, recovered=failure == Failure.RETRIEVAL_FAILURE)

    def test_bounded_search_miss_is_inconclusive(self):
        with self.assertRaises(InconclusiveDiagnosis):
            self.run_case("KNOWLEDGE_BOUNDARY_FAILURE", exhaustive=False)

    def test_fabricated_quote_rejected(self):
        from failure_classification.questiongeneration.evaluator import Citation, validate_citations
        with self.assertRaises(ValueError):
            validate_citations([Citation(chunk_index=0, quote="invented")], ["actual"])

    def test_invalid_triage_is_repaired_with_bounded_retry(self):
        attempts = []
        def llm(prompt, payload):
            if "dynamic_sub_questions" in prompt:
                return dict(extracted_constraints=["x"], identified_delta="x missing",
                            dynamic_sub_questions=["What is x?"])
            attempts.append(payload)
            return dict(ungrounded_claims=[], omitted_requirements=["x"],
                        prompt_support=[dict(chunk_index=0, quote="invented" if len(attempts)==1 else "x is 1")],
                        retrieved_support=[], fragmented_relationship=False,
                        boundary_evidence=[], explanation="x omitted")
        def search(*args):
            self.fail("Coverage failure should not search")
        result = FailureClassifier(llm, search).evaluate(dict(
            user_query="What is x?", top_k_chunks=["x is 1"],
            prompt_context=["x is 1"], rag_answer="Unknown"))
        self.assertEqual(result.attributed_failure, Failure.RESPONSE_COVERAGE_FAILURE)
        self.assertEqual(len(attempts), 2)
        self.assertIn("validation_error", attempts[1])

    def test_search_errors_propagate(self):
        from failure_classification.questiongeneration.evaluator import CounterfactualSearchModule, ProbePlan
        def search(*args):
            raise ConnectionError("index unavailable")
        module = CounterfactualSearchModule(lambda *args: {}, search)
        with self.assertRaises(ConnectionError):
            module.diagnose(ProbePlan(extracted_constraints=["x"], identified_delta="x",
                                     dynamic_sub_questions=["x?"]),
                            Artifacts(user_query="x", top_k_chunks=[], prompt_context=[], rag_answer=""))


if __name__ == "__main__":
    unittest.main()
