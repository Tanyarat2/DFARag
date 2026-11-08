"""Turn identity, evidence isolation, persistence, and classifier integration."""
import copy
import json
import tempfile
import unittest
from failure_classification.tests.answerability_fixtures import AttributionOnlyTestCase
from pathlib import Path
from types import SimpleNamespace

from failure_classification.errorclassification import Conversation, ConversationTurn, FailureClassifier
from failure_classification.errorclassification.pipeline.classifier import classify_case


def case(case_id='2159'):
    return {'case_id': case_id, 'original_question': 'Which year?',
            'original_rag_answer': '2018', 'gold_answer': '2018',
            'source_documents': [{'doc_id': 'gold', 'text': 'Reference-only text'}],
            'indexed_chunks': [{'chunk_id': 'gold_1', 'text': 'Indexed-only text'}],
            'retrieved_candidates': [{'chunk_id': 'other_1', 'text': 'Year 2018'}],
            'generation_context': [{'chunk_id': 'other_1', 'text': 'Year 2018'}],
            'conversation_history': [], 'pipeline_metadata': {'note': 'saved run'}}


def runner(model):
    def ask_json(**request):
        # A previous diagnosis must never become model input on reclassification.
        assert 'DO_NOT_FEED_BACK' not in request['prompt']
        return {'parsed': {'answer_defect': {'status': 'no_defect', 'summary': 'Supported year'},
                           'attribution_status': 'no_failure', 'candidate_failures': [],
                           'primary_failure': None, 'required_facts': [],
                           'generation_assessment': {
                               'answerability': 'answerable', 'response_behavior': 'answered',
                               'reason': 'The input supplies the requested year.',
                               'evidence': [
                                   {'source': 'original_rag_answer', 'item_id': None, 'quote': '2018', 'observation': 'Answer'},
                                   {'source': 'generation_context', 'item_id': 'other_1', 'quote': 'Year 2018', 'observation': 'Support'}],
                               'checks': {f'E{i}': {'status': 'no_defect', 'reason': 'No defect observed.'} for i in range(9, 17)}}}}
    return SimpleNamespace(model=model, ask_json=ask_json)


class ConversationModelTests(AttributionOnlyTestCase):
    def test_input_roundtrip_keeps_reference_and_execution_evidence_separate(self):
        original = case()
        turn = ConversationTurn.from_case_input(original)
        self.assertEqual(turn.to_case_input(), original)
        self.assertEqual(turn.reference_evidence['source_documents'][0]['doc_id'], 'gold')
        self.assertEqual(turn.retrieved_chunks[0]['chunk_id'], 'other_1')
        original['generation_context'][0]['text'] = 'modified source'
        self.assertEqual(turn.generation_context[0]['text'], 'Year 2018')
        exported = turn.to_case_input()
        exported['retrieved_candidates'].clear()
        self.assertEqual(len(turn.retrieved_chunks), 1)

    def test_missing_and_empty_collections_remain_distinct(self):
        original = case()
        del original['generation_context']
        original['retrieved_candidates'] = None
        turn = ConversationTurn.from_case_input(original)
        self.assertEqual(turn.to_case_input(), original)
        original['generation_context'] = []
        self.assertEqual(ConversationTurn.from_case_input(original).to_case_input()['generation_context'], [])

    def test_multi_turn_save_and_load_preserves_order_ids_history_and_model_results(self):
        first = ConversationTurn.from_case_input(case('first'))
        second_case = case('second')
        second_case['turn_id'] = 'followup'
        second_case['conversation_history'] = [{'role': 'user', 'content': first.question},
                                               {'role': 'assistant', 'content': first.answer}]
        second = ConversationTurn.from_case_input(second_case)
        first.record_classification('mini', {'model_prediction': {'failures': ['E4']},
                                             'final_attribution': {'status': 'unresolved'}}, [{'pass': 1}])
        conversation = Conversation('conversation-1', [first, second])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'conversation.json'
            conversation.save(path)
            loaded = Conversation.load(path)
        self.assertEqual(loaded.to_dict(), conversation.to_dict())
        self.assertEqual([turn.turn_id for turn in loaded.turns], ['first', 'followup'])
        self.assertEqual(loaded.turns[1].to_case_input(), second_case)
        self.assertEqual(loaded.turns[1].classifications, {})

    def test_classification_attaches_to_the_correct_turn_and_model(self):
        turn = ConversationTurn.from_case_input(case())
        other = ConversationTurn.from_case_input(case('other'))
        turn.record_classification('old-model', {'note': 'DO_NOT_FEED_BACK'}, [])
        mini, audit = classify_case(turn, runner('mini'), max_passes=1, max_failures=1)
        classify_case(turn, runner('luna'), max_passes=1, max_failures=1)
        self.assertEqual(turn.classifications['mini']['prediction'], mini)
        self.assertEqual(turn.classifications['mini']['audit'], audit)
        self.assertEqual(set(turn.classifications), {'old-model', 'mini', 'luna'})
        self.assertEqual(other.classifications, {})
        mini['final_attribution']['status'] = 'changed outside turn'
        self.assertEqual(turn.classifications['mini']['prediction']['final_attribution']['status'], 'no_failure_reported')

    def test_dictionary_and_class_inputs_send_identical_prompts(self):
        prompts = []
        def ask_json(**request):
            prompts.append(request['prompt'])
            return runner('fake').ask_json(**request)
        for value in [case(), ConversationTurn.from_case_input(case())]:
            classify_case(value, SimpleNamespace(model='fake', ask_json=ask_json), max_passes=1)
        self.assertEqual(prompts[0], prompts[1])

    def test_previous_labels_are_removed_from_imported_inputs(self):
        original = case()
        original['final_attribution'] = {'primary_failure': 'E4'}
        original['retrieved_candidates'][0]['error_category'] = 'E13'
        turn = ConversationTurn.from_case_input(original)
        self.assertNotIn('final_attribution', turn.to_case_input())
        self.assertNotIn('error_category', turn.retrieved_chunks[0])

    def test_classifier_runs_a_conversation_and_stores_each_turn(self):
        conversation = Conversation('chat', [ConversationTurn.from_case_input(case('first')),
                                             ConversationTurn.from_case_input(case('second'))])
        classifier = FailureClassifier(runner('mini'), max_passes=1, max_failures=1)
        results = classifier.run(conversation)
        self.assertEqual(list(results), ['first', 'second'])
        for turn in conversation.turns:
            self.assertEqual(turn.classifications['mini']['prediction'], results[turn.turn_id])
            self.assertEqual(len(turn.classifications['mini']['audit']), 1)
        results['first']['final_attribution']['status'] = 'modified outside storage'
        self.assertEqual(conversation.turns[1].classifications['mini']['prediction']['final_attribution']['status'],
                         'no_failure_reported')


    def test_duplicate_turns_invalid_evidence_and_identity_overrides_are_rejected(self):
        turn = ConversationTurn.from_case_input(case())
        with self.assertRaisesRegex(ValueError, 'Duplicate turn'):
            Conversation('one', [turn, copy.deepcopy(turn)])
        for value in [None, '', '   ']:
            with self.assertRaises(ValueError):
                ConversationTurn(value, 'Question', 'Answer')
        with self.assertRaisesRegex(ValueError, 'retrieved_chunks'):
            ConversationTurn('one', 'Question', 'Answer', retrieved_chunks=['not a chunk object'])
        with self.assertRaisesRegex(ValueError, 'metadata cannot override'):
            ConversationTurn('one', 'Question', 'Answer', metadata={'generation_context': []})
        with self.assertRaisesRegex(ValueError, 'reference_evidence accepts only'):
            ConversationTurn('one', 'Question', 'Answer', reference_evidence={'retrieved_candidates': []})
        with self.assertRaisesRegex(ValueError, 'schema version'):
            Conversation.from_dict({'schema_version': 99})


if __name__ == '__main__':
    unittest.main()
