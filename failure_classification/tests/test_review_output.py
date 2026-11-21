"""Review contracts, bounded format correction, and evidence validation regressions."""
import copy
import unittest
from failure_classification.tests.answerability_fixtures import AttributionOnlyTestCase
from types import SimpleNamespace
from failure_classification.errorclassification.pipeline.classifier import classify_case
from failure_classification.errorclassification.api.client import ClassifierRunner
from failure_classification.errorclassification.review.output import normalize_review_output, review_output_errors

CASE = {'case_id': 'format-test', 'original_question': 'Which year?',
        'original_rag_answer': '2021', 'generation_context': [{'chunk_id': 'c', 'text': 'Report year 2018.'}]}
INITIAL = {'answer_defect': {'status': 'defect', 'summary': 'The answer changes the year.'},
           'candidate_failures': [], 'primary_failure': None, 'attribution_status': 'needs_review'}
ANSWER = {'source': 'original_rag_answer', 'item_id': None, 'quote': '2021', 'observation': 'Wrong year.'}
CONTEXT = {'source': 'generation_context', 'item_id': 'c', 'quote': 'Report year 2018.', 'observation': 'Supplied year.'}
PATCH = {'candidate_updates': [{'code': 'E13', 'status': 'supported',
          'evidence': [ANSWER, CONTEXT], 'key_points': ['Supplied year was misread.']}],
         'primary_update': {'failure': 'E13', 'status': 'established',
          'reason': 'The answer changes the year in its supplied context.', 'evidence': [ANSWER]}}


def classify_with(replies):
    responses = iter(replies)
    calls = []
    def ask(**request):
        calls.append(request)
        return copy.deepcopy(next(responses))
    result, audit = classify_case(CASE, SimpleNamespace(ask_json=ask), max_failures=1)
    return result, audit, calls


class ReviewOutputTests(AttributionOnlyTestCase):
    def test_malformed_json_gets_one_format_retry_and_preserves_audit_and_usage(self):
        broken = {'parsed': None, 'invalid_output': '{"candidate_updates": [',
                  'error_type': 'invalid_json', 'validation_error': 'Missing closing bracket',
                  'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}}
        fixed = {'parsed': PATCH, 'usage': {'prompt_tokens': 20, 'completion_tokens': 5, 'total_tokens': 25}}
        result, audit, calls = classify_with([{'parsed': INITIAL}, broken, fixed])
        self.assertEqual(len(calls), 3)
        self.assertEqual(result['passes_used'], 2)
        self.assertEqual(result['format_retries'], 1)
        self.assertEqual(result['primary_failure'], 'E13')
        self.assertEqual(audit[1]['usage']['total_tokens'], 38)
        self.assertEqual(audit[1]['review_response_attempts'][0]['output'], broken['invalid_output'])
        self.assertIn('Missing closing bracket', calls[-1]['prompt'])
        self.assertTrue(calls[1]['json_mode'])

    def test_two_bad_patches_do_not_loop_or_get_partially_applied(self):
        invalid = {'parsed': {'candidate_updates': PATCH['candidate_updates'],
                             'primary_update': {'failure': 'no_defect', 'status': 'established'}}}
        result, audit, calls = classify_with([{'parsed': INITIAL}, invalid, invalid])
        self.assertEqual(len(calls), 3)
        self.assertEqual(result['failures'], [])
        self.assertFalse(result['review_output_valid'])
        self.assertEqual(audit[1]['applied_codes'], [])
        self.assertTrue(audit[1]['ignored_updates'])
        self.assertIn('no_defect', ' '.join(audit[1]['review_response_attempts'][0]['errors']))

    def test_primary_without_candidate_requests_explicit_candidate_instead_of_inventing_one(self):
        invalid = {'primary_update': PATCH['primary_update']}
        self.assertTrue(any('no supported candidate' in e for e in review_output_errors(invalid)))
        result, audit, calls = classify_with([{'parsed': INITIAL}, {'parsed': invalid}, {'parsed': PATCH}])
        self.assertEqual(result['primary_failure'], 'E13')
        self.assertIn('candidate_updates', calls[-1]['prompt'])
        self.assertEqual(audit[1]['review_response_attempts'][0]['output'], invalid)

    def test_existing_supported_primary_does_not_need_duplicate_candidate(self):
        self.assertEqual(review_output_errors({'primary_update': PATCH['primary_update']}, ['E13']), [])

    def test_safe_string_null_is_normalized_without_inventing_a_label(self):
        raw = {'primary_update': {'failure': 'null', 'status': 'needs_review'}}
        normalized, paths = normalize_review_output(raw)
        self.assertIsNone(normalized['primary_update']['failure'])
        self.assertEqual(raw['primary_update']['failure'], 'null')
        self.assertTrue(paths)
        result, audit, calls = classify_with([{'parsed': INITIAL}, {'parsed': raw}])
        self.assertEqual(len(calls), 2)
        self.assertIsNone(result['primary_failure'])
        self.assertTrue(audit[1]['review_response_attempts'][0]['output_normalization'])

    def test_format_correction_cannot_bypass_evidence_validation(self):
        patch = copy.deepcopy(PATCH)
        patch['candidate_updates'][0]['evidence'][1]['quote'] = 'INVENTED TEXT'
        result, _, calls = classify_with([{'parsed': INITIAL}, {'parsed': patch}])
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['failures'], [])
        self.assertEqual(result['label_assessments']['E13']['status'], 'rejected')

    def test_valid_unresolved_patch_does_not_trigger_format_retry(self):
        result, _, calls = classify_with([{'parsed': INITIAL}, {'parsed': {}}])
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['format_retries'], 0)
        self.assertIsNone(result['primary_failure'])

    def test_unknown_codes_and_established_null_are_rejected(self):
        for value in ['no_defect', 'E99', 'null', None]:
            with self.subTest(value=value):
                self.assertTrue(review_output_errors({'primary_update': {'failure': value, 'status': 'established'}}))

    def test_malformed_status_types_are_reported_without_crashing(self):
        self.assertTrue(review_output_errors({'candidate_updates': [{'code': 'E13', 'status': []}]}))
        self.assertTrue(review_output_errors({'primary_update': {'failure': None, 'status': []}}))
        self.assertTrue(review_output_errors({'answer_defect_update': {'status': [], 'summary': ''}}))

    def test_json_mode_is_opt_in_and_has_a_distinct_cache_key(self):
        runner = ClassifierRunner.__new__(ClassifierRunner)
        runner.model = 'test-model'
        args = runner.build_chat_completion_arguments('JSON please', 100, None, json_mode=True)
        self.assertEqual(args['response_format'], {'type': 'json_object'})
        self.assertNotIn('response_format', runner.build_chat_completion_arguments('JSON please', 100, None))
        self.assertNotEqual(runner.request_hash(stage='test', prompt='JSON please'),
                            runner.request_hash(stage='test', prompt='JSON please', json_mode=True))


if __name__ == '__main__':
    unittest.main()
