"""Check agreement arithmetic, the supplied reference, live file reloads, and UI."""
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

from agreement_results import ANNOTATORS, RUNS, binary_scores, compare, decision
from annotation_server import make_handler
from annotation_store import AnnotationStore


def profiles_fixture():
    return {name: {'case#1': {'failure_types': ['E4']},
                   'case#2': {'failure_types': ['E5']},
                   'other#1': {'failure_types': ['not_a_failure']}}
            for name in ANNOTATORS}


def write_exports(root, profiles):
    for method, run in RUNS.items():
        folder = root / method
        folder.mkdir(exist_ok=True)
        for name, annotations in profiles.items():
            rows = []
            for key, record in annotations.items():
                cid, turn = key.rsplit('#', 1)
                rows.append(dict(record, id=cid, turn=int(turn), annotator=name))
            (folder / f'annotations_{run}_{name}.json').write_text(json.dumps(rows), encoding='utf-8')


CHECKS = r"""<pre id="test-result"></pre><script>
(async()=>{
 const result=document.getElementById('test-result');
 const assert=(ok,message)=>{if(!ok)throw Error(message);};
 const waitFor=async predicate=>{for(let i=0;i<100;i++){if(predicate())return;await new Promise(resolve=>setTimeout(resolve,100));}throw Error('Timed out waiting for live update');};
 try{
  await waitFor(()=>data&&document.querySelector('.comparison tbody tr'));
  assert(data.methods.HippoRag.total===3,'Initial sample');
  assert(document.querySelector('#report>.panel tbody').children.length===16,'All sixteen E rows');
  assert(document.querySelectorAll('.comparison tbody tr').length===3,'Comparison records');
  const search=$('row-search');search.value='C01';search.dispatchEvent(new Event('input'));
  assert(document.querySelectorAll('.comparison tbody tr').length===2,'Conversation search');
  const previous=data.version;
  await fetch('/test-edit',{method:'POST'});
  await waitFor(()=>data.version!==previous);
  assert(data.methods.HippoRag.counts.exact===2,'Polling updated calculations');
  assert($('row-search').value==='C01','Row search survives updates');
  $('method').value='Raptor';$('method').dispatchEvent(new Event('change'));
  assert(document.querySelectorAll('.comparison tbody tr').length===2,'Method switch');
  $('source').value='live';$('source').dispatchEvent(new Event('change'));
  await waitFor(()=>data&&data.source==='live');
  assert(data.methods.Vector.total===1,'Live saves are separate');
  assert(document.documentElement.scrollWidth<=window.innerWidth,'Responsive page width');
  assert(!/[\u2014\u00b7]/.test(document.body.innerText),'No em dash or middle dot');
  result.dataset.status='pass';result.textContent='PASS';
 }catch(error){result.dataset.status='fail';result.textContent=error.stack;}
})();
</script>"""


class AgreementTests(unittest.TestCase):
    def test_binary_kappa_and_undefined_cases(self):
        scores = binary_scores([True, True, False, False], [True, False, True, False])
        self.assertEqual(scores['agreement'], .5)
        self.assertEqual(scores['kappa'], 0)
        self.assertEqual(scores['frequency'], 3)
        self.assertEqual(binary_scores([True, False], [False, True])['kappa'], -1)
        self.assertEqual(binary_scores([True, False], [True, False])['kappa'], 1)
        self.assertIsNone(binary_scores([False, False], [False, False])['kappa'])
        self.assertIsNone(binary_scores([True, True], [True, True])['kappa'])
        self.assertIsNone(binary_scores([], [])['agreement'])

    def test_special_and_missing_are_pair_specific(self):
        profiles = profiles_fixture()
        profiles['PS']['case#1'] = {'failure_types': ['excluded']}
        del profiles['KH']['case#2']
        result = compare(profiles)
        self.assertEqual([pair['n'] for pair in result['pairs']], [1, 2, 2])
        self.assertEqual(result['counts']['special'], 1)
        self.assertEqual(result['counts']['incomplete'], 1)
        self.assertEqual(result['counts']['exact'], 1)
        self.assertEqual(result['shared_label'], 1)
        self.assertTrue(decision({'human_label': 'correct'})['eligible'])
        self.assertFalse(decision({'review_complete': True})['eligible'])
        self.assertFalse(decision({'failure_types': ['E5'], 'human_label': 'correct'})['eligible'])

    def test_cohort_joins_id_and_turn_instead_of_row_order(self):
        profiles = profiles_fixture()
        profiles['KH'] = dict(reversed(list(profiles['KH'].items())))
        profiles['PS']['extra#1'] = {'failure_types': ['E9']}
        self.assertEqual(compare(profiles)['counts']['exact'], 3)
        union = compare(profiles, 'union')
        self.assertEqual(union['total'], 4)
        self.assertEqual(union['counts']['incomplete'], 1)

    def test_supplied_seventy_turn_example(self):
        rows = json.loads(Path(__file__).with_name('test_fixtures').joinpath('agreement_reference.json').read_text())
        profiles = {name: {} for name in ANNOTATORS}
        aliases = {'NF': 'not_a_failure', 'EX': 'excluded', 'IE': 'insufficient_evidence', 'OT': 'outside_taxonomy'}
        for cid, turn, *labels in rows:
            for name, label in zip(ANNOTATORS, labels):
                profiles[name][f'{cid}#{turn}'] = {'failure_types': [aliases.get(code, code) for code in label.split(', ')]}
        result = compare(profiles)
        self.assertEqual([p['n'] for p in result['pairs']], [66, 65, 66])
        self.assertEqual([round(p['weighted_mean'], 3) for p in result['pairs']], [.592, .458, .485])
        self.assertEqual(result['counts'], {'exact': 8, 'shared': 43, 'none': 14, 'special': 5, 'incomplete': 0})
        self.assertEqual(result['shared_e'], 49)
        self.assertEqual(result['shared_label'], 51)
        self.assertEqual(result['below_all'], ['E3', 'E5', 'E6', 'E7', 'E8', 'E10', 'E12', 'E14'])
        self.assertAlmostEqual(result['pairs'][0]['scores']['E3']['kappa'], .653, places=3)
        self.assertAlmostEqual(result['pairs'][2]['scores']['E12']['kappa'], -.040, places=3)

    def test_http_reloads_files_and_separates_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            results, runs = root / 'results', root / 'runs'
            results.mkdir(); runs.mkdir()
            profiles = profiles_fixture()
            write_exports(results, profiles)
            run = RUNS['Vector']
            (runs / run).mkdir()
            AnnotationStore(runs).patch(run, 'WN', {'live#1': {'failure_types': ['E9']}}, {'live#1'})
            handler = make_handler(runs, results)
            handler.log_message = lambda *args: None
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            base = f'http://127.0.0.1:{server.server_port}'
            def get(path):
                with opener.open(base + path, timeout=5) as response:
                    self.assertEqual(response.headers['Cache-Control'], 'no-store')
                    return response.read()
            try:
                self.assertIn(b'Live annotation results', get('/results'))
                before = json.loads(get('/api/results'))
                profiles['WN']['case#1']['failure_types'] = ['E6']
                write_exports(results, profiles)
                after = json.loads(get('/api/results'))
                self.assertNotEqual(before['version'], after['version'])
                self.assertEqual(after['methods']['HippoRag']['counts']['exact'], 2)
                live = json.loads(get('/api/results?source=live'))
                self.assertEqual(live['methods']['Vector']['total'], 1)
                self.assertEqual(live['methods']['Vector']['rows'][0]['id'], 'live')
                with self.assertRaises(urllib.error.HTTPError) as invalid:
                    get('/api/results?source=unknown')
                self.assertEqual(invalid.exception.code, 400)
                (results / 'HippoRag' / f'annotations_{RUNS["HippoRag"]}_WN.json').write_text('{')
                with self.assertRaises(urllib.error.HTTPError) as malformed:
                    get('/api/results')
                self.assertEqual(malformed.exception.code, 500)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=5)

    def test_browser_live_polling_desktop_and_mobile(self):
        browser = next((p for p in [shutil.which('chromium'),
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'] if p and Path(p).is_file()), None)
        if not browser:
            self.skipTest('No browser installed')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            results, runs = root / 'results', root / 'runs'
            results.mkdir(); runs.mkdir()
            profiles = profiles_fixture()
            write_exports(results, profiles)
            (runs / RUNS['Vector']).mkdir()
            AnnotationStore(runs).patch(RUNS['Vector'], 'WN', {'live#1': {'failure_types': ['E9']}}, {'live#1'})
            parent = make_handler(runs, results)
            class Harness(parent):
                def do_GET(self):
                    if self.path == '/results':
                        from agreement_results import results_page
                        self.respond(200, results_page().replace('</body>', CHECKS + '</body>'))
                    else:
                        super().do_GET()
                def do_POST(self):
                    if self.path == '/test-edit':
                        profiles['WN']['case#1']['failure_types'] = ['E6']
                        write_exports(results, profiles)
                        self.respond_json(200, {'ok': True})
                    else:
                        super().do_POST()
                def log_message(self, *args):
                    pass
            server = ThreadingHTTPServer(('127.0.0.1', 0), Harness)
            thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
            try:
                for width in (1440, 600):
                    profiles['WN']['case#1']['failure_types'] = ['E4']
                    write_exports(results, profiles)
                    result = subprocess.run([browser, '--headless', '--disable-gpu', '--no-first-run',
                        '--no-default-browser-check', '--disable-background-networking', '--disable-extensions',
                        '--disable-sync', '--user-data-dir=' + str(root / f'profile-{width}'),
                        f'--window-size={width},1100', '--timeout=20000', '--virtual-time-budget=12000',
                        '--dump-dom', f'http://127.0.0.1:{server.server_port}/results'],
                        capture_output=True, encoding='utf-8', errors='replace', timeout=35)
                    report = result.stdout.split('<pre id="test-result"', 1)[-1].split('</pre>', 1)[0]
                    self.assertTrue('data-status="pass"' in report,
                                    f'Width {width}: {report[-1200:]} {result.stderr[-500:]}')
            finally:
                server.shutdown();server.server_close();thread.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
