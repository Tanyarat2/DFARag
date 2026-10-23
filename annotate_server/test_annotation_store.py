"""Persistence and HTTP checks; all test annotations stay in temporary folders."""

import gzip
import errno
import json
import pathlib
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

import annotation_server

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
    def test_both_loopback_addresses_share_one_port_and_annotations(self):
        with tempfile.TemporaryDirectory() as folder:
            make_run(folder)
            handler = make_handler(pathlib.Path(folder))
            handler.log_message = lambda *args: None
            servers = annotation_server.create_servers('localhost', 0, handler)
            threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
            for thread in threads:
                thread.start()
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                port = servers[0].server_port
                self.assertTrue(all(server.server_port == port for server in servers))
                self.assertEqual(servers[0].server_address[0], '127.0.0.1')
                request = urllib.request.Request('http://127.0.0.1:%d/api/annotations/pilot/PS' % port,
                    data=json.dumps({'changes': {'case#1': {'notes': 'Shared loopback annotation'}}}).encode(),
                    headers={'Content-Type': 'application/json'})
                with opener.open(request, timeout=3) as response:
                    self.assertTrue(json.load(response)['saved'])
                for server in servers:
                    host = '[::1]' if ':' in server.server_address[0] else '127.0.0.1'
                    with opener.open('http://%s:%d/' % (host, port), timeout=3) as response:
                        self.assertIn('pilot', response.read().decode())
                    with opener.open('http://%s:%d/api/annotations/pilot/PS' % (host, port), timeout=3) as response:
                        self.assertEqual(json.load(response)['annotations']['case#1']['notes'], 'Shared loopback annotation')
            finally:
                for server in servers:
                    server.shutdown()
                    server.server_close()
                for thread in threads:
                    thread.join(timeout=2)

    def test_unsupported_ipv6_keeps_ipv4_and_port_conflict_closes_it(self):
        for error, supported in [(errno.EAFNOSUPPORT, False), (errno.EADDRINUSE, True)]:
            with self.subTest(error=error), mock.patch.object(annotation_server, 'ThreadingHTTPServer') as ipv4, \
                    mock.patch.object(annotation_server, 'IPv6HTTPServer', side_effect=OSError(error, 'Fixture bind error')):
                ipv4.return_value.server_port = 8765
                if supported:
                    with self.assertRaises(OSError):
                        annotation_server.create_servers('localhost', 8765, object())
                    ipv4.return_value.server_close.assert_called_once()
                else:
                    self.assertEqual(annotation_server.create_servers('localhost', 8765, object()), [ipv4.return_value])
                    ipv4.return_value.server_close.assert_not_called()

    def test_independent_review_status_roundtrips_and_rejects_non_booleans(self):
        with tempfile.TemporaryDirectory() as folder:
            make_run(folder)
            store = AnnotationStore(folder)
            store.patch('pilot', 'PS', {'case#1': {'review_complete': True}}, {'case#1'})
            self.assertIs(store.read('pilot', 'PS')['annotations']['case#1']['review_complete'], True)
            self.assertNotIn('human_label', store.read('pilot', 'PS')['annotations']['case#1'])
            store.patch('pilot', 'PS', {'case#1': {'review_complete': False}}, {'case#1'})
            before = store.path('pilot', 'PS').read_bytes()
            self.assertIs(store.read('pilot', 'PS')['annotations']['case#1']['review_complete'], False)
            for invalid in ('true', 1, None):
                with self.assertRaises(ValueError):
                    store.patch('pilot', 'PS', {'case#1': {'review_complete': invalid}}, {'case#1'})
            self.assertEqual(store.path('pilot', 'PS').read_bytes(), before)

    def test_uncertain_correctness_saves_separately_from_incorrect(self):
        with tempfile.TemporaryDirectory() as folder:
            make_run(folder)
            store = AnnotationStore(folder)
            for owner, label in [('PS', 'uncertain'), ('KH', 'incorrect')]:
                store.patch('pilot', owner, {'case#1': {
                    'human_label': label, 'failure_types': ['insufficient_evidence'],
                    'failure_type': 'insufficient_evidence', 'review_complete': True}}, {'case#1'})
                self.assertEqual(AnnotationStore(folder).read('pilot', owner)
                                 ['annotations']['case#1']['human_label'], label)
            before = store.path('pilot', 'PS').read_bytes()
            with self.assertRaises(ValueError):
                store.patch('pilot', 'PS', {'case#1': {'human_label': 'maybe'}}, {'case#1'})
            self.assertEqual(store.path('pilot', 'PS').read_bytes(), before)

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

    def test_paper_errors_roundtrip_with_legacy_history(self):
        from failure_taxonomy import DEFINITIONS, TAXONOMY, TAXONOMY_VERSION
        self.assertEqual(TAXONOMY, ['E%d' % n for n in range(1, 17)])
        self.assertEqual([sum(d['stage'] == stage for d in DEFINITIONS.values())
                          for stage in ('Chunking', 'Retrieval', 'Reranking', 'Generation')], [3, 3, 2, 8])
        with tempfile.TemporaryDirectory() as folder:
            make_run(folder)
            store = AnnotationStore(folder)
            for code in TAXONOMY:
                entry = {'failure_types': [code], 'failure_type': code,
                         'failure_taxonomy': TAXONOMY_VERSION,
                         'legacy_failure_types': ['retrieval', 'grounding']}
                store.patch('pilot', 'PS', {'case#1': entry}, {'case#1'})
                self.assertEqual(AnnotationStore(folder).read('pilot', 'PS')['annotations']['case#1'], entry)
            before = store.path('pilot', 'PS').read_bytes()
            for invalid in ({'failure_types': ['E17']}, {'failure_types': ['E1', 'E1']},
                            {'failure_types': ['E16', 'insufficient_evidence']},
                            {'legacy_failure_types': ['E1']}, {'legacy_failure_types': 'grounding'},
                            {'legacy_failure_types': ['grounding', 'grounding']},
                            {'failure_taxonomy': 'unknown'}):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    store.patch('pilot', 'PS', {'case#1': invalid}, {'case#1'})
            self.assertEqual(store.path('pilot', 'PS').read_bytes(), before)

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

    def test_cached_pages_compress_refresh_and_reuse_data_for_saves(self):
        with tempfile.TemporaryDirectory() as folder:
            work = make_run(folder)
            handler = make_handler(pathlib.Path(folder))
            handler.log_message = lambda *args: None
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = 'http://127.0.0.1:%d' % server.server_port
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

            def page(encoding):
                req = urllib.request.Request(base + '/run/pilot', headers={'Accept-Encoding': encoding})
                with opener.open(req, timeout=5) as response:
                    body = response.read()
                    self.assertEqual(len(body), int(response.headers['Content-Length']))
                    self.assertEqual(response.headers['Vary'], 'Accept-Encoding')
                    self.assertEqual(response.headers.get('Content-Encoding'),
                                     'gzip' if encoding in ('gzip', 'gzip, deflate') else None)
                    if response.headers.get('Content-Encoding') == 'gzip':
                        self.assertIn('gzip', encoding)
                        body = gzip.decompress(body)
                    else:
                        self.assertIsNone(response.headers.get('Content-Encoding'))
                    return body.decode()

            try:
                with mock.patch.object(annotation_server.annotation, 'build_data',
                                       wraps=annotation_server.annotation.build_data) as build:
                    compressed = page('gzip, deflate')
                    self.assertIn('First question', compressed)
                    self.assertEqual(page('identity'), compressed)
                    self.assertEqual(page('gzip;q=0, *;q=1'), compressed)
                    self.assertEqual(build.call_count, 1)
                    req = urllib.request.Request(base + '/api/annotations/pilot/PS',
                        data=json.dumps({'changes': {'case#1': {'notes': 'Saved once'}}}).encode(),
                        headers={'Content-Type': 'application/json'})
                    with opener.open(req, timeout=5) as response:
                        self.assertTrue(json.load(response)['saved'])
                    self.assertEqual(build.call_count, 1)
                    self.assertEqual(AnnotationStore(folder).read('pilot', 'PS')['annotations']['case#1']['notes'], 'Saved once')
                    (work / 'out.mtrag.json').write_text(json.dumps([{
                        'id': 'case', 'evaluation': {'turn_evaluations': [
                            {'turn': 1, 'label': 'correct', 'scores': {'conditioned_score': 0.95}}
                        ]}}]), encoding='utf-8')
                    refreshed = page('gzip')
                    payload = json.loads(refreshed.split('<script id="data" type="application/json">', 1)[1].split('</script>', 1)[0])
                    self.assertEqual(payload['rows'][0]['judges']['mtrag']['conditioned_score'], .95)
                    self.assertNotEqual(refreshed, compressed)
                    self.assertEqual(build.call_count, 2)
                    records = json.loads((work / 'input.json').read_text())
                    records[0]['conversation'][0]['content'] = 'Changed question'
                    (work / 'input.json').write_text(json.dumps(records), encoding='utf-8')
                    self.assertIn('Changed question', page('gzip'))
                    self.assertEqual(build.call_count, 3)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

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
                payload = json.loads(page.split('<script id="data" type="application/json">', 1)[1].split('</script>', 1)[0])
                self.assertEqual(payload['annotation_api'], '/api/annotations/pilot')
                self.assertNotIn('id="blind"', page)
                self.assertNotIn('id="coverage"', page)
                self.assertEqual(len(list((work / 'annotations').glob('*.json'))), 3)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
