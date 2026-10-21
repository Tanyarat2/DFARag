"""Corpus identity, atomic import, ranking, and offline snapshot checks; no network."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import corpus_store
import make_annotation
from corpus_store import CorpusIndex, import_archive


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [
            {'id': 'passage-1', 'title': 'Venus', 'text': 'Venus is a planet.'},
            {'id': 'passage-2', 'title': 'Rotation', 'text': 'Venus rotates westward.'},
            {'id': 'passage-3', 'title': 'Mars', 'text': 'Mars has two moons.'},
        ]
        self.archive = self.make_archive(self.rows)
        self.output = self.root / 'clapnq.sqlite3'
        import_archive(self.archive, self.output, 'clapnq', revision='fixture-revision')

    def make_archive(self, rows):
        path = self.root / 'fixture.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('fixture.jsonl', '\n'.join(json.dumps(row) for row in rows))
        return path

    def test_import_preserves_ids_text_and_source_provenance(self):
        index = CorpusIndex(self.output)
        metadata = index.metadata()
        self.assertEqual(metadata['document_count'], 3)
        self.assertEqual(metadata['revision'], 'fixture-revision')
        self.assertEqual(metadata['sha256'], hashlib.sha256(self.archive.read_bytes()).hexdigest())
        result = index.search('passage 2', '', '')
        self.assertEqual(result['match_count'], 1)
        self.assertEqual(result['document_count'], 3)
        self.assertEqual(result['matches'][0]['doc']['text'], self.rows[1]['text'])
        self.assertEqual(result['matches'][0]['doc']['id'], 'passage-2')
        self.assertEqual(result['matches'][0]['doc']['source_file'], 'clapnq.jsonl')

    def test_question_search_order_ignores_answer_and_operators_are_literal(self):
        index = CorpusIndex(self.output)
        result = index.search('', 'Which planet?', 'Venus rotates westward.')
        self.assertEqual(result['matches'][0]['doc']['id'], 'passage-1')
        self.assertEqual(result, index.search('', 'Which planet?', 'Mars has two moons.'))
        self.assertNotIn('answer_overlap', result['matches'][0])
        self.assertEqual(index.search('Venus moons', '', '')['match_count'], 0)
        self.assertEqual(index.search('" OR * NOT ()', '', '')['match_count'], 0)
        self.assertEqual(index.search('', 'the and', '')['matches'], [])
        self.assertIn('4.5', corpus_store.terms('The age is 4.5 billion years.'))
        self.assertNotIn('5', corpus_store.terms('The age is 4.5 billion years.'))

    def test_failed_import_retains_existing_index(self):
        before = self.output.read_bytes()
        self.make_archive([{'id': 'bad', 'text': 42}])
        with self.assertRaises(ValueError):
            import_archive(self.archive, self.output, 'clapnq')
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(list(self.root.glob('*.sqlite3')), [self.output])

    def test_collection_isolation_and_live_vs_offline_payload(self):
        work = self.root / 'pilot'
        work.mkdir()
        (work / 'input.json').write_text(json.dumps([{
            'id': 'case', 'conversation': [
                {'role': 'user', 'content': 'Which direction does Venus rotate?'},
                {'role': 'assistant', 'content': 'Venus rotates westward.'}],
            'contexts': [['Venus rotates westward.']],
            'ground_truth_answers': ['Westward.'], 'metadata': {'collection': 'clapnq'}}]))
        with mock.patch.object(corpus_store, 'CORPUS_DIR', self.root), \
                mock.patch.object(make_annotation, 'CORPUS_DIR', self.root), \
                mock.patch.object(make_annotation, 'load_passage_archive', return_value={}):
            self.assertIsNone(corpus_store.corpus_metadata('fiqa'))
            live = make_annotation.build_data(work, live=True)['corpus']['clapnq']
            self.assertEqual(live['kind'], 'ibm_corpus')
            self.assertEqual(live['documents'], [])
            self.assertEqual(live['document_count'], 3)
            snapshot = make_annotation.build_data(work)['corpus']['clapnq']
            self.assertEqual(snapshot['kind'], 'ibm_corpus_snapshot')
            self.assertEqual(snapshot['match_ids']['case#1'][0], 'passage-1')
            self.assertIn('offline', snapshot['message'].lower())
            self.assertEqual(snapshot['sources'][0]['revision'], 'fixture-revision')
            self.assertTrue(all(doc['id'] in {'passage-1', 'passage-2'} for doc in snapshot['documents']))


if __name__ == '__main__':
    unittest.main()
