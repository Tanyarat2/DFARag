"""Direct review edits preserve exports and back up originals before replacement."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from annotation_server import make_handler
from make_result_review import RUNS, annotation_version
from result_store import ResultStore, ReviewConflict
from test_agreement_results import profiles_fixture, write_exports

ROW = {'key': 'case#1', 'id': 'case', 'turn': 1, 'conversation_number': 'C01'}


def record():
    return {'id': 'case', 'turn': 1, 'annotator': 'WN', 'conversation_number': 'C01',
            'failure_types': ['E4'], 'failure_type': 'E4', 'notes': 'Original notes',
            'review_complete': False, 'flagged_for_review': True,
            'gold_assessment': 'questionable', 'gold_notes': 'Original gold check',
            'evidence': [{'source': 'RAG answer', 'locator': 'answer', 'quote': 'Evidence quote'}],
            'legacy_failure_types': ['grounding'], 'custom_field': {'keep': 7}}


def prepare(root):
    results, runs = root / 'results', root / 'runs'
    results.mkdir(); runs.mkdir()
    profiles = profiles_fixture()
    profiles['WN']['case#1'] = record()
    write_exports(results, profiles)
    for run in RUNS.values():
        work = runs / run
        work.mkdir()
        inputs, judged = [], []
        for cid, n in [('case', 3), ('other', 1)]:
            messages = []
            for i in range(n):
                messages.extend([{'role': 'user', 'content': f'{cid} question {i + 1}'},
                                 {'role': 'assistant', 'content': 'Saved answer'}])
            inputs.append({'id': cid, 'conversation': messages, 'metadata': {'collection': 'review-test'}})
            judged.append({'id': cid, 'evaluation': {'turn_evaluations': [
                {'turn': i + 1, 'label': 'incorrect', 'scores': {'conditioned_score': .3}}
                for i in range(n)]}})
        (work / 'input.json').write_text(json.dumps(inputs), encoding='utf-8')
        (work / 'out.mtrag.json').write_text(json.dumps(judged), encoding='utf-8')
    return results, runs, profiles


class ResultStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ResultStore(Path(self.temp.name))
        self.path = self.store.path('Vector', 'WN')
        self.path.parent.mkdir()
        self.other = {'id': 'other', 'turn': 1, 'annotator': 'WN', 'failure_types': ['E9']}
        self.path.write_text(json.dumps([record(), self.other]), encoding='utf-8')

    def test_update_preserves_fields_other_turns_and_original_backup(self):
        before = self.path.read_bytes()
        saved = self.store.save('Vector', 'WN', ROW,
            {'failure_types': ['E5', 'E9'], 'failure_type': '', 'notes': 'Reviewed'}, annotation_version(record()))
        rows = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertEqual(rows[1], self.other)
        for key in ('evidence', 'legacy_failure_types', 'custom_field', 'flagged_for_review', 'gold_notes'):
            self.assertEqual(rows[0][key], record()[key])
        self.assertEqual(rows[0]['failure_types'], ['E5', 'E9'])
        self.assertEqual(rows[0]['failure_type'], '')
        self.assertEqual(rows[0]['conversation_number'], 'C01')
        self.assertEqual((self.store.root / saved['backup']).read_bytes(), before)
        self.assertEqual(saved['version'], annotation_version(rows[0]))

    def test_stale_decision_cannot_overwrite_newer_file(self):
        changed = dict(record(), notes='Changed outside the review')
        self.path.write_text(json.dumps([changed, self.other]), encoding='utf-8')
        before = self.path.read_bytes()
        with self.assertRaises(ReviewConflict) as conflict:
            self.store.save('Vector', 'WN', ROW, {'notes': 'Stale edit'}, annotation_version(record()))
        self.assertEqual(conflict.exception.record, changed)
        self.assertEqual(self.path.read_bytes(), before)

    def test_other_turn_changes_are_merged_and_new_profile_can_be_created(self):
        other = dict(self.other, notes='Newer unrelated decision')
        self.path.write_text(json.dumps([record(), other]), encoding='utf-8')
        self.store.save('Vector', 'WN', ROW, {'notes': 'Changed'}, annotation_version(record()))
        self.assertEqual(json.loads(self.path.read_text(encoding='utf-8'))[1], other)
        saved = self.store.save('Raptor', 'KH', ROW, {'failure_types': ['E6']}, annotation_version(None))
        self.assertEqual(saved['record']['annotator'], 'KH')
        self.assertEqual(saved['record']['failure_type'], 'E6')
        self.assertIsNone(saved['backup'])

    def test_invalid_labels_ownership_and_corrupt_exports_are_not_written(self):
        before = self.path.read_bytes()
        for changes in ({'failure_types': ['E99']}, {'failure_types': ['E5', 'excluded']},
                        {'id': 'forged'}, {'human_label': 'correct', 'failure_types': ['E4']}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.store.save('Vector', 'WN', ROW, changes, annotation_version(record()))
            self.assertEqual(self.path.read_bytes(), before)
        self.path.write_text('{', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.store.save('Vector', 'WN', ROW, {'notes': 'Unsafe'}, annotation_version(record()))
        self.assertEqual(self.path.read_text(), '{')
        with self.assertRaises(ValueError):
            self.store.path('../outside', 'WN')

    def test_failed_write_keeps_original_and_a_backup(self):
        before = self.path.read_bytes()
        atomic = self.store._atomic
        def write(path, raw):
            if path == self.path:
                raise OSError('Synthetic disk failure')
            return atomic(path, raw)
        with mock.patch.object(self.store, '_atomic', side_effect=write):
            with self.assertRaises(OSError):
                self.store.save('Vector', 'WN', ROW, {'notes': 'Changed'}, annotation_version(record()))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(len(list((self.store.root / '.review_backups').rglob('*.json'))), 1)


CHECKS = r"""<pre id="test-result"></pre><script>
(async()=>{
 const result=document.getElementById('test-result'),assert=(ok,message)=>{if(!ok)throw Error(message);};
 const waitFor=async(predicate,label)=>{for(let i=0;i<120;i++){if(predicate())return;await new Promise(resolve=>setTimeout(resolve,100));}throw Error('Timed out: '+label+' / '+$('edit-feedback').textContent);};
 const choose=code=>{for(const i of document.querySelectorAll('#error-options input'))i.checked=i.value===code;$('error-options').dispatchEvent(new Event('change',{bubbles:true}));};
 const submit=$('decision-form').onsubmit;let pendingSave;
 $('decision-form').onsubmit=event=>{pendingSave=submit(event);return pendingSave;};
 const save=async()=>{$('decision-form').requestSubmit();await pendingSave;};
 try{
  assert(D.review_api==='/api/review','Live review API');
  document.querySelector('.edit-decision[data-annotator="WN"]').click();
  assert($('decision-editor').open && $('edit-flagged').checked,'Editor opens with saved fields');
  const previous=index;document.body.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));assert(index===previous,'Modal blocks turn navigation');
  choose('E6');$('edit-notes').value='Reviewed in browser';await save();
  await waitFor(()=>!$('decision-editor').open && !saving,'first save');
  let row=D.rows.find(r=>r.method==='Vector'&&r.id==='case'&&r.turn===1);
  assert(row.annotations.WN.failure_types.join(',')==='E6','Saved labels visible');
  assert(row.annotations.WN.evidence[0].quote==='Evidence quote' && row.annotations.WN.custom_field.keep===7,'Hidden fields preserved');
  const metrics=await (await fetch('/api/results')).json();
  assert(metrics.methods.Vector.rows[0].decisions.WN.labels[0]==='E6','Agreement reads saved export');
  document.querySelector('.edit-decision[data-annotator="WN"]').click();
  await fetch('/test-replace',{method:'POST'});assert(editState,'Editor stays active after reopening');choose('E7');await save();
  await waitFor(()=>!$('edit-reload').hidden && !saving,'conflict response');
  assert($('decision-editor').open && $('edit-save').disabled,'Conflict keeps draft open');
  assert(document.querySelector('#error-options input[value="E7"]').checked,'Conflict preserves unsaved choices');
  $('edit-reload').click();assert(document.querySelector('#error-options input[value="E5"]').checked,'Load latest decision');
  assert($('edit-notes').value==='External revision','Latest notes loaded');choose('E8');await save();
  await waitFor(()=>!$('decision-editor').open && !saving,'resolved save');
  row=D.rows.find(r=>r.method==='Vector'&&r.id==='case'&&r.turn===1);
  assert(row.annotations.WN.failure_types[0]==='E8','Save after conflict resolution');
  assert(document.documentElement.scrollWidth<=window.innerWidth,'No horizontal page overflow');
  result.dataset.status='pass';result.textContent='PASS';
 }catch(error){dirty=false;result.dataset.status='fail';result.textContent=error.stack;}
})();
</script>"""


class ReviewHttpTests(unittest.TestCase):
    def test_http_page_save_conflict_and_results_refresh(self):
        with tempfile.TemporaryDirectory() as folder:
            results, runs, _ = prepare(Path(folder))
            handler = make_handler(runs, results)
            handler.log_message = lambda *args: None
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            base = f'http://127.0.0.1:{server.server_port}'
            def request(path, data=None, origin=None):
                headers = {'Content-Type': 'application/json'}
                if origin: headers['Origin'] = origin
                req = urllib.request.Request(base + path, headers=headers,
                    data=json.dumps(data).encode() if data is not None else None)
                try: response = opener.open(req, timeout=10)
                except urllib.error.HTTPError as error: response = error
                with response:
                    body = response.read().decode('utf-8')
                    return response.status, body if path == '/review' else json.loads(body)
            try:
                status, page = request('/review');self.assertEqual(status, 200)
                self.assertIn('"review_api": "/api/review"', page)
                _, snapshot = request('/api/review/annotations')
                row = next(r for r in snapshot['rows'] if r['method'] == 'Vector' and r['id'] == 'case' and r['turn'] == 1)
                payload = {'id': 'case', 'turn': 1, 'expected_version': row['annotation_versions']['WN'],
                           'changes': {'failure_types': ['E6'], 'notes': 'Changed'}}
                status, saved = request('/api/review/Vector/WN', payload)
                self.assertEqual(status, 200)
                self.assertTrue(saved['saved'])
                self.assertTrue((results / saved['backup']).is_file())
                self.assertEqual(saved['record']['evidence'], record()['evidence'])
                self.assertEqual(request('/api/review/Vector/WN', payload)[0], 409)
                _, metrics = request('/api/results')
                self.assertEqual(metrics['methods']['Vector']['rows'][0]['decisions']['WN']['labels'], ['E6'])
                _, refreshed = request('/api/review/annotations')
                self.assertNotEqual(snapshot['version'], refreshed['version'])
                self.assertEqual(request('/api/review/Vector/WN', payload, 'https://unrelated.example')[0], 403)
                self.assertEqual(request('/api/review/Other/WN', payload)[0], 400)
                payload['id'] = 'missing'
                self.assertEqual(request('/api/review/Vector/WN', payload)[0], 404)
                self.assertEqual(request('/api/review/turn?method=Vector&id=case&turn=1')[0], 200)
                export_path = results / 'Vector' / f'annotations_{RUNS["Vector"]}_WN.json'
                records = json.loads(export_path.read_text(encoding='utf-8'))
                records.append({'id': 'case', 'turn': 3, 'annotator': 'WN', 'failure_types': ['E5']})
                export_path.write_text(json.dumps(records), encoding='utf-8')
                _, latest = request('/api/review/annotations')
                self.assertTrue(any(r['method'] == 'Vector' and r['turn'] == 3 for r in latest['rows']))
                status, detail = request('/api/review/turn?method=Vector&id=case&turn=3')
                self.assertEqual(status, 200)
                self.assertEqual(detail['row']['answer'], 'Saved answer')
            finally:
                server.shutdown();server.server_close();thread.join(timeout=5)

    def test_browser_edit_save_and_resolve_replaced_export(self):
        browser = next((p for p in [shutil.which('chromium'),
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'] if p and Path(p).is_file()), None)
        if not browser: self.skipTest('No browser installed')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            results, runs, profiles = prepare(root)
            parent = make_handler(runs, results)
            class Harness(parent):
                def respond(self, status, body, content_type='text/html; charset=utf-8', compressed=None):
                    if self.path == '/review' and status == 200:
                        body = body.decode('utf-8') if isinstance(body, bytes) else body
                        body = body.replace('</body>', CHECKS + '</body>');compressed = None
                    super().respond(status, body, content_type, compressed)
                def do_POST(self):
                    if self.path == '/test-replace':
                        path = results / 'Vector' / f'annotations_{RUNS["Vector"]}_WN.json'
                        records = json.loads(path.read_text(encoding='utf-8'))
                        records[0].update(failure_types=['E5'], failure_type='E5', notes='External revision')
                        path.write_text(json.dumps(records), encoding='utf-8')
                        self.respond_json(200, {'ok': True})
                    else: super().do_POST()
                def log_message(self, *args): pass
            server = ThreadingHTTPServer(('127.0.0.1', 0), Harness)
            thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
            try:
                for width in (1440, 600):
                    write_exports(results, copy.deepcopy(profiles))
                    result = subprocess.run([browser, '--headless', '--disable-gpu', '--no-first-run',
                        '--no-default-browser-check', '--disable-background-networking', '--disable-extensions',
                        '--disable-sync', '--user-data-dir=' + str(root / f'profile-{width}'),
                        f'--window-size={width},1100', '--timeout=25000', '--virtual-time-budget=16000',
                        '--dump-dom', f'http://127.0.0.1:{server.server_port}/review'],
                        capture_output=True, encoding='utf-8', errors='replace', timeout=40)
                    report = result.stdout.split('<pre id="test-result"', 1)[-1].split('</pre>', 1)[0]
                    self.assertTrue('data-status="pass"' in report,
                                    f'Width {width}: {report[-1800:]} {result.stderr[-500:]}')
                self.assertGreaterEqual(len(list((results / '.review_backups').rglob('*.json'))), 4)
            finally:
                server.shutdown();server.server_close();thread.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
