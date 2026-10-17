"""Persistence and HTTP checks; all test annotations stay in temporary folders."""

import json
import pathlib
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from annotation_server import make_handler
from annotation_store import AnnotationStore


def make_run(root, name='pilot'):
    work = pathlib.Path(root) / name
    work.mkdir()
    (work / 'input.json').write_text(json.dumps([{
        'id': 'case', 'conversation': [
            {'role': 'user', 'content': 'First question'}, {'role': 'assistant', 'content': 'First answer'},
            {'role': 'user', 'content': 'Second question'}, {'role': 'assistant', 'content': 'Second answer'},
            {'role': 'user', 'content': 'Third question'}, {'role': 'assistant', 'content': 'Third answer'}]}]), encoding='utf-8')
    return work


class AnnotationStoreTests(unittest.TestCase):
    def test_multiple_causes_and_legacy_single_cause_can_both_be_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            make_run(folder)
            store = AnnotationStore(folder)
            valid = {'case#1', 'case#2'}
            store.patch('pilot', 'PS', {'case#1': {'human_label': 'incorrect', 'failure_type': '',
                                                'failure_types': ['retrieval', 'grounding']}}, valid)
            store.patch('pilot', 'KH', {'case#1': {'human_label': 'incorrect', 'failure_type': 'grounding'}}, valid)
            store.patch('pilot', 'PS', {'case#2': {'failure_types': ['response_coverage'],
                                                'failure_type': 'response_coverage'}}, valid)
            data = AnnotationStore(folder).read('pilot', 'PS')['annotations']
            self.assertEqual(data['case#1']['failure_types'], ['retrieval', 'grounding'])
            self.assertEqual(data['case#1']['failure_type'], '')
            self.assertEqual(store.read('pilot', 'KH')['annotations']['case#1']['failure_type'], 'grounding')

    def test_invalid_multiple_cause_records_do_not_overwrite_saved_work(self):
        with tempfile.TemporaryDirectory() as folder:
            make_run(folder)
            store = AnnotationStore(folder)
            store.patch('pilot', 'PS', {'case#1': {'notes': 'Keep'}}, {'case#1'})
            before = store.path('pilot', 'PS').read_bytes()
            for entry in [{'failure_types': 'grounding'}, {'failure_types': ['unknown']},
                          {'failure_types': ['grounding', 'grounding']}, {'failure_types': [{}]},
                          {'failure_types': ['grounding', 'not_a_failure']},
                          {'failure_types': ['grounding', 'insufficient_evidence']},
                          {'failure_types': ['grounding', 'outside_taxonomy']},
                          {'failure_types': ['grounding', 'retrieval'], 'failure_type': 'grounding'},
                          {'failure_types': [], 'failure_type': 'grounding'}]:
                with self.subTest(entry=entry), self.assertRaises(ValueError):
                    store.patch('pilot', 'PS', {'case#1': entry}, {'case#1'})
            self.assertEqual(store.path('pilot', 'PS').read_bytes(), before)

    def test_three_reviewers_can_label_the_same_turn_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            work = make_run(folder)
            original = (work / 'input.json').read_bytes()
            store = AnnotationStore(folder)
            for name, label in [('PS', 'correct'), ('KH', 'incorrect'), ('WN', 'incorrect')]:
                store.patch('pilot', name, {'case#1': {'human_label': label, 'notes': name}}, {'case#1'})
            reloaded = AnnotationStore(folder)
            for name in ('PS', 'KH', 'WN'):
                saved = reloaded.read('pilot', name)
                self.assertEqual(saved['annotations']['case#1']['notes'], name)
                self.assertEqual(saved['annotator'], name)
                self.assertTrue((work / 'annotations' / (name + '.json')).is_file())
            self.assertEqual((work / 'input.json').read_bytes(), original)

    def test_turn_patches_preserve_other_turns_and_clear_decisions(self):
        with tempfile.TemporaryDirectory() as folder:
            make_run(folder)
            store = AnnotationStore(folder)
            valid = {'case#1', 'case#2'}
            store.patch('pilot', 'PS', {'case#1': {'human_label': 'correct'}}, valid)
            store.patch('pilot', 'PS', {'case#2': {'notes': 'Keep this note'}}, valid)
            store.patch('pilot', 'PS', {'case#1': {}}, valid)
            data = store.read('pilot', 'PS')['annotations']
            self.assertEqual(data['case#1'], {})
            self.assertEqual(data['case#2']['notes'], 'Keep this note')

    def test_invalid_destinations_and_records_do_not_change_saved_data(self):
        with tempfile.TemporaryDirectory() as folder:
            make_run(folder)
            store = AnnotationStore(folder)
            store.patch('pilot', 'PS', {'case#1': {'notes': 'Original'}}, {'case#1'})
            before = store.path('pilot', 'PS').read_bytes()
            for run, name, changes in [('../outside', 'PS', {}), ('pilot', '../PS', {}),
                                       ('pilot', 'XX', {}), ('pilot', 'PS', {'other#1': {}}),
                                       ('pilot', 'PS', {'case#1': {'annotator': 'KH'}}),
                                       ('pilot', 'PS', {'case#1': {'human_label': 'maybe'}}),
                                       ('pilot', 'PS', {'case#1': {'evidence': [{'quote': 'Missing source'}]}})]:
                with self.subTest(run=run, name=name, changes=changes), self.assertRaises(ValueError):
                    store.patch(run, name, changes, {'case#1'})
            self.assertEqual(store.path('pilot', 'PS').read_bytes(), before)

    def test_corrupt_or_wrong_owner_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            make_run(folder)
            store = AnnotationStore(folder)
            path = store.path('pilot', 'PS')
            path.parent.mkdir()
            for original in ['broken JSON', json.dumps({'run': 'pilot', 'annotator': 'KH', 'annotations': {}})]:
                path.write_text(original, encoding='utf-8')
                with self.assertRaises(RuntimeError):
                    store.patch('pilot', 'PS', {'case#1': {'notes': 'New'}}, {'case#1'})
                self.assertEqual(path.read_text(encoding='utf-8'), original)

    def test_http_profiles_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            work = make_run(folder)
            handler = make_handler(pathlib.Path(folder))
            handler.log_message = lambda *args: None
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = 'http://127.0.0.1:%d' % server.server_port

            def request(path, payload=None, headers=None):
                data = None if payload is None else json.dumps(payload).encode()
                request_headers = {'Content-Type': 'application/json', **(headers or {})}
                req = urllib.request.Request(base + path, data=data, headers=request_headers)
                try:
                    response = urllib.request.urlopen(req, timeout=5)
                except urllib.error.HTTPError as exc:
                    response = exc
                with response:
                    return response.status, json.loads(response.read())

            try:
                self.assertEqual(request('/api/annotations/pilot/PS')[1]['annotations'], {})
                for name in ('PS', 'KH', 'WN'):
                    status, result = request('/api/annotations/pilot/' + name,
                                             {'changes': {'case#2': {'notes': name, 'human_label': 'correct'}}})
                    self.assertEqual(status, 200)
                    self.assertEqual(result['annotator'], name)
                self.assertEqual(request('/api/annotations/pilot/KH')[1]['annotations']['case#2']['notes'], 'KH')
                self.assertEqual(request('/api/annotations/pilot/PS', {'changes': {'other#1': {}}})[0], 400)
                self.assertEqual(request('/api/annotations/pilot/XX')[0], 400)
                self.assertEqual(request('/api/annotations/missing/PS')[0], 404)
                self.assertEqual(request('/api/annotations/pilot/PS', {'changes': {}}, {'Origin': 'https://unrelated.example'})[0], 403)
                self.assertEqual(request('/api/annotations/pilot/PS', {'changes': {}}, {'Content-Type': 'text/plain'})[0], 415)
                page = urllib.request.urlopen(base + '/run/pilot', timeout=5).read().decode()
                self.assertIn('"annotation_api": "/api/annotations/pilot"', page)
                self.assertNotIn('id="blind"', page)
                self.assertNotIn('id="coverage"', page)
                self.assertEqual(len(list((work / 'annotations').glob('*.json'))), 3)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
