import json
import tempfile
import unittest
from pathlib import Path

from failure_classification.questiongeneration.dynamic_followup_q import run_dynamic_method


def draft(question):
    return json.dumps({'claim_verification_questions': [question],
                       'missing_information_questions': []})


class RetryTests(unittest.TestCase):
    def run_cases(self, generate, review, ask, rows=None):
        with tempfile.TemporaryDirectory() as directory:
            records = run_dynamic_method(
                rows or [{'query_id': '2159', 'question': 'Which reporting year?',
                          'original_rag_answer': 'No year is specified.'}],
                generate, ask, output_dir=directory, review_question=review)
            saved = json.loads((Path(directory) / 'diagnostic_results.json').read_text())
            self.assertEqual(records, saved)
            return records

    def test_rejections_accumulate_and_next_case_continues(self):
        outputs = iter([draft('Which reporting year?'), draft('Which events?'),
                        draft('What reporting year?'), draft('Which year is identified?'),
                        draft('Which month?'), json.dumps({
                            'claim_verification_questions': [], 'missing_information_questions': []})])
        prompts = []
        asked = []

        def generate(prompt):
            prompts.append(prompt)
            return next(outputs)

        def review(payload):
            question = payload['candidate']['question']
            accepted = question in {'Which reporting year?', 'Which month?'}
            return {'accepted': accepted, 'stop': False,
                    'reason': 'Accepted' if accepted else 'Rejected draft: ' + question,
                    'factual_target': 'Reporting year'}

        def ask(question):
            asked.append(question)
            return '2018. [65_13]'

        records = self.run_cases(generate, review, ask, [
            {'query_id': '2159', 'question': 'Which reporting year?',
             'original_rag_answer': 'No year is specified.'},
            {'query_id': 'next', 'question': 'Which month?',
             'original_rag_answer': 'Unknown.'}])
        self.assertEqual(asked, ['Which reporting year?', 'Which month?'])
        self.assertIn('Which events?', prompts[3])
        self.assertIn('What reporting year?', prompts[3])
        self.assertEqual(records[0]['termination_reason'], 'draft_retries_exhausted')
        self.assertTrue(records[0]['diagnostics_partial'])
        self.assertEqual(len(records[0]['turns']), 1)
        self.assertEqual(records[1]['status'], 'complete')

    def test_reviewer_stop_preserves_probe(self):
        outputs = iter([draft('Which reporting year?'), draft('What reporting year?')])
        def review(payload):
            stop = bool(payload['previous_diagnostic_turns'])
            return {'accepted': not stop, 'stop': stop,
                    'reason': 'Core information need answered; no distinct target remains.'
                              if stop else 'First probe', 'factual_target': 'Reporting year'}
        records = self.run_cases(lambda prompt: next(outputs), review, lambda q: '2018')
        self.assertEqual(len(records[0]['turns']), 1)
        self.assertIn('Core information need', records[0]['stop_reason'])
        self.assertNotIn('diagnostics_partial', records[0])

    def test_api_error_is_not_swallowed(self):
        def generate(prompt):
            raise ConnectionError('API unavailable')
        with self.assertRaises(ConnectionError):
            self.run_cases(generate, lambda p: None, lambda q: 'Unused')


if __name__ == '__main__':
    unittest.main()
