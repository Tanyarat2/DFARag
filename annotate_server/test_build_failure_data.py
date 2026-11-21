"""Check cohort filtering and history boundaries for the label-free export."""
import unittest

from build_failure_data import history_for, match_generation, retained_keys


class FailureDataTests(unittest.TestCase):
    def test_all_three_must_label_before_unanimous_rejection_filter(self):
        profiles = {
            'PS': {'remove': {'failure_types': ['not_a_failure']},
                   'keep': {'failure_types': ['not_a_failure']},
                   'missing': {'failure_types': ['excluded']},
                   'undecided': {'failure_types': ['E9']},
                   'review_only': {'review_complete': True}},
            'KH': {'remove': {'failure_types': ['excluded']},
                   'keep': {'failure_types': ['E9']},
                   'undecided': {'failure_types': ['E9']},
                   'missing': {'failure_types': ['excluded']}},
            'WN': {'remove': {'human_label': 'correct'},
                   'undecided': {'failure_types': [], 'review_complete': True},
                   'keep': {'failure_types': ['excluded']}},
        }
        self.assertEqual(retained_keys(profiles), ({'keep'}, {'remove'}))

    def test_history_excludes_current_and_future_messages(self):
        messages = [{'role': role, 'content': f'{turn}-{role}'}
                    for turn in range(1, 4) for role in ('user', 'assistant')]
        history, current = history_for(messages, 2)
        self.assertEqual(history, messages[:2])
        self.assertEqual(current, messages[2:4])
        self.assertEqual(history_for(messages, 1)[0], [])

    def test_historical_backend_match_requires_exact_context_and_answer(self):
        source = {'file': 'selfrag.json', 'rag_name': 'selfrag'}
        turn = {'question': 'q', 'answer': 'a', 'passages': [{'text': 'ctx'}]}
        index = {('c', '1'): [(source, turn)]}
        result = match_generation(index, 'c', 1, 'q', 'a', ['ctx'], 'vector')
        self.assertEqual(result['sources'], [source])
        self.assertEqual(match_generation(index, 'c', 1, 'q', 'other', ['ctx'], 'vector')['status'], 'unavailable')
        self.assertEqual(match_generation(index, 'c', 1, 'q', 'a', ['wrong'], 'vector')['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
