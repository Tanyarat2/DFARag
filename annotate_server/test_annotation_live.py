"""Progressive live loading: small pages, complete details, and independent saves."""

import copy
import gzip
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

import make_annotation as annotation
from annotation_server import RunPages, available_runs, make_handler
from corpus_store import CorpusSearch
from annotation_store import AnnotationStore


def make_run(root):
    work = pathlib.Path(root) / 'pilot'
    work.mkdir()
    records = []
    results = []
    for cid, count in [('case', 3), ('other', 2)]:
        conversation = []
        for k in range(count):
            conversation.extend([
                {'role': 'user', 'content': 'Which direction does Venus rotate?'},
                {'role': 'assistant', 'content': 'Venus rotates westward.' if cid == 'case'
                 else 'Second conversation private answer %d' % (k + 1)}])
        records.append({'id': cid, 'conversation': conversation,
                        'ground_truth_answers': ['Venus rotates westward.'] * count,
                        'contexts': [['Venus rotates westward.']] * count,
                        'metadata': {'collection': 'planets'}})
        results.append({'id': cid, 'evaluation': {'turn_evaluations': [
            {'turn': k + 1, 'label': 'incorrect' if cid == 'case' and k == 0 else 'correct',
             'reasoning': 'Saved explanation for browser disclosure checks.',
             'scores': {'conditioned_score': .4 if cid == 'case' and k == 0 else .95,
                        'statements': [{'statement': 'Judge claim for inspection.',
                                        'verdict': 'no', 'reason': 'Judge claim reason.'}]}}
            for k in range(count)]}})
    (work / 'input.json').write_text(json.dumps(records), encoding='utf-8')
    (work / 'out.mtrag.json').write_text(json.dumps(results), encoding='utf-8')
    documents = [{'id': 'doc-%d' % i, 'title': 'Venus', 'text': 'Venus rotates westward. ' * 200,
                  'source_file': 'planets.json'} for i in range(35)]
    archive = {'planets': {'kind': 'conversation_passages', 'documents': documents,
                           'sources': [], 'message': 'Recorded passage archive.'}}
    return work, archive


class ProgressiveTests(unittest.TestCase):
    def test_small_page_details_search_and_saving_unopened_turns(self):
        with tempfile.TemporaryDirectory() as folder:
            work, archive = make_run(folder)
            with mock.patch.object(annotation, 'load_passage_archive', return_value=archive):
                handler = make_handler(pathlib.Path(folder))
                handler.log_message = lambda *args: None
                server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                base = 'http://127.0.0.1:%d' % server.server_port

                def get(path):
                    try:
                        response = opener.open(base + path, timeout=5)
                    except urllib.error.HTTPError as exc:
                        response = exc
                    with response:
                        return response.status, response.read()

                try:
                    status, body = get('/run/pilot')
                    self.assertEqual(status, 200)
                    html = body.decode()
                    payload = html.split('<script id="data" type="application/json">', 1)[1].split('</script>', 1)[0]
                    data = json.loads(payload)
                    self.assertEqual(len(data['rows']), 5)
                    self.assertNotIn('Second conversation private answer', html)
                    self.assertEqual(sum(bool(row.get('_summary')) for row in data['rows']), 2)
                    self.assertEqual(data['corpus']['planets']['documents'], [])
                    self.assertEqual(data['corpus']['planets']['document_count'], 35)
                    self.assertEqual(data['rows'][-1]['judges']['mtrag']['conditioned_score'], .95)
                    self.assertLess(len(body), 125_000)
                    self.assertLess(len(gzip.compress(body)), 35_000)
                    status, body = get('/api/conversation/pilot?id=other')
                    self.assertEqual(status, 200)
                    details = json.loads(body)['rows']
                    self.assertEqual(len(details), 2)
                    self.assertEqual(details[0]['answer'], 'Second conversation private answer 1')
                    self.assertEqual(details[1]['contexts'], ['Venus rotates westward.'])
                    self.assertEqual(get('/api/conversation/pilot?id=missing')[0], 404)
                    status, body = get('/api/corpus/pilot?key=other%232&q=Venus')
                    search = json.loads(body)
                    self.assertEqual(status, 200)
                    self.assertEqual(search['document_count'], 35)
                    self.assertEqual(search['match_count'], 35)
                    self.assertEqual(len(search['matches']), 30)
                    status, body = get('/api/corpus/pilot?key=case%231&q=doc-17')
                    search = json.loads(body)
                    self.assertEqual(search['match_count'], 1)
                    self.assertEqual(search['matches'][0]['i'], 17)
                    self.assertEqual(search['matches'][0]['doc'], archive['planets']['documents'][17])
                    self.assertEqual(get('/api/corpus/pilot?key=missing')[0], 404)
                    request = urllib.request.Request(base + '/api/annotations/pilot/PS',
                        data=json.dumps({'changes': {'other#2': {'notes': 'Saved without opening'}}}).encode(),
                        headers={'Content-Type': 'application/json'})
                    with opener.open(request, timeout=5) as response:
                        self.assertTrue(json.load(response)['saved'])
                    self.assertEqual(AnnotationStore(folder).read('pilot', 'PS')['annotations']['other#2']['notes'],
                                     'Saved without opening')
                    results = json.loads((work / 'out.mtrag.json').read_text())
                    results[1]['evaluation']['turn_evaluations'][0]['scores']['conditioned_score'] = .4
                    (work / 'out.mtrag.json').write_text(json.dumps(results), encoding='utf-8')
                    refreshed = json.loads(get('/api/conversation/pilot?id=other')[1])['rows']
                    self.assertEqual(refreshed[0]['judges']['mtrag']['conditioned_score'], .4)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)

    def test_corpus_word_matching_order_and_unicode(self):
        docs = [{'id': 'a', 'title': 'Venus', 'text': 'spins westward'},
                {'id': 'b', 'title': 'Planets', 'text': 'Venus spins westward'},
                {'id': 'c', 'title': 'ดาว', 'text': 'ภาษาไทย'}]
        search = CorpusSearch(docs)
        self.assertEqual([m['i'] for m in search.search('venus', '')['matches']], [0, 1])
        self.assertEqual(search.search('venus missing', '')['match_count'], 0)
        self.assertEqual(search.search('', 'What is Venus?')['match_count'], 2)
        self.assertEqual(search.search('ภาษาไทย', '')['matches'][0]['i'], 2)
        self.assertEqual(search.search('', 'the and')['match_count'], 3)

    def test_firefox_progressive_workflow(self):
        binary = pathlib.Path('/snap/firefox/current/usr/lib/firefox/firefox')
        firefox = str(binary) if binary.is_file() else shutil.which('firefox')
        if not firefox:
            self.skipTest('Firefox is unavailable')
        with tempfile.TemporaryDirectory(prefix='annotation-live-browser-') as folder:
            base = pathlib.Path(folder)
            work, archive = make_run(base)
            AnnotationStore(base).patch('pilot', 'PS', {'other#2': {'notes': 'Earlier saved note'}}, {'other#2'})
            finished = threading.Event()
            corpus_gate = threading.Event()
            profile_gate = threading.Event()
            outcomes = []
            with mock.patch.object(annotation, 'load_passage_archive', return_value=archive):
                parent = make_handler(base)
                page = RunPages().get(work)['payload'].decode()
                page = page.replace('</body>', PROGRESSIVE_CHECKS + '</body>')

                class Harness(parent):
                    def do_GET(self):
                        if self.path == '/workflow':
                            self.respond(200, page)
                        else:
                            if self.path == '/api/annotations/pilot/PS':
                                profile_gate.wait(timeout=5)
                            if self.path.startswith('/api/corpus/'):
                                corpus_gate.wait(timeout=5)
                            super().do_GET()

                    def do_POST(self):
                        if self.path == '/test/unblock-profile':
                            profile_gate.set()
                            self.respond_json(200, {'ok': True})
                        elif self.path == '/test/unblock':
                            corpus_gate.set()
                            self.respond_json(200, {'ok': True})
                        elif self.path == '/test/result':
                            outcomes.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                            self.respond_json(200, {'ok': True})
                            finished.set()
                        else:
                            super().do_POST()

                    def log_message(self, *args):
                        pass

                server = ThreadingHTTPServer(('127.0.0.1', 0), Harness)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                profile = base / 'profile'
                profile.mkdir()
                process = subprocess.Popen([firefox, '--headless', '--no-remote', '--profile', str(profile),
                    'http://127.0.0.1:%d/workflow' % server.server_port],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    self.assertTrue(finished.wait(timeout=35), 'Firefox did not report its workflow result.')
                    self.assertEqual(outcomes[0]['status'], 'pass', outcomes[0].get('error'))
                    self.assertGreaterEqual(outcomes[0]['checks'], 15)
                    self.assertEqual(AnnotationStore(base).read('pilot', 'PS')['annotations']['other#1']['notes'],
                                     'Progressive autosave')
                finally:
                    profile_gate.set()
                    corpus_gate.set()
                    process.terminate()
                    try:
                        process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)


    def test_firefox_guided_desktop_and_mobile_layout(self):
        from test_annotation import fixture
        binary = pathlib.Path('/snap/firefox/current/usr/lib/firefox/firefox')
        firefox = str(binary) if binary.is_file() else shutil.which('firefox')
        if not firefox:
            self.skipTest('Firefox is unavailable')
        # A regular Firefox window has a 500px minimum; both desktop and mobile CSS paths are exercised.
        for width in (1280, 500):
            with self.subTest(width=width), tempfile.TemporaryDirectory(prefix='guided-layout-') as folder:
                finished, outcomes = threading.Event(), []
                page = annotation.render_page(fixture()).replace('</body>', LAYOUT_CHECKS + '</body>')

                from http.server import BaseHTTPRequestHandler

                class Handler(BaseHTTPRequestHandler):
                    def do_GET(self):
                        payload = page.encode()
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/html; charset=utf-8')
                        self.send_header('Content-Length', str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)

                    def do_POST(self):
                        outcomes.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                        self.send_response(200)
                        self.end_headers()
                        finished.set()

                    def log_message(self, *args):
                        pass

                server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                profile = pathlib.Path(folder) / 'profile'
                profile.mkdir()
                process = subprocess.Popen([firefox, '--headless', '--no-remote', '--profile', str(profile),
                    'http://127.0.0.1:%d/' % server.server_port],
                    env=dict(os.environ, MOZ_HEADLESS_WIDTH=str(width), MOZ_HEADLESS_HEIGHT='1000'),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    self.assertTrue(finished.wait(timeout=20), 'Browser did not report layout checks.')
                    self.assertEqual(outcomes[0]['status'], 'pass', outcomes[0].get('error'))
                    self.assertEqual(outcomes[0]['width'], width)
                finally:
                    process.terminate()
                    try:
                        process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)


PROGRESSIVE_CHECKS = r"""
<script>
(async()=>{
  let checks=0;
  const assert=(ok,message)=>{if(!ok)throw Error(message);checks++;};
  const wait=async test=>{const start=Date.now();while(!test()){if(Date.now()-start>10000)throw Error('Timed out waiting for live data.');await new Promise(resolve=>setTimeout(resolve,25));}};
  try{
    assert(document.querySelector('#question .question')&&document.querySelector('#answer .prose'),'Question and answer are visible while saved annotations are delayed');
    assert(getProfile(currentAnnotator).loading&&document.getElementById('annotationPanel').inert,'Annotation controls stay locked until saved labels arrive');
    assert(!document.querySelector('.queue').inert&&!document.querySelector('.workspace').inert,'Reading and conversation navigation remain available while the profile loads');
    assert(document.getElementById('expJson').disabled&&document.getElementById('imp').disabled,'Exports and imports cannot overwrite or export an unloaded annotation profile');
    openTurn('case#2');assert(activeKey==='case#2'&&document.getElementById('annotationPanel').inert,'Users can browse another turn before saved labels arrive');
    cite(D.rows[1],'Question','question',D.rows[1].question,document.querySelector('#question .question'));
    assert(!entry(D.rows[1]).evidence?.length,'Citation edits wait for the saved annotation profile');
    openTurn('case#1');
    await fetch('/test/unblock-profile',{method:'POST'});
    await annotationReady;
    assert(!document.getElementById('annotationPanel').inert&&!document.getElementById('expJson').disabled,'Editing and exports become available after loading saved labels');
    const visible=node=>!!node&&node.getClientRects().length>0;
    assert(document.querySelectorAll('input[name="humanLabel"]').length===3,'Correctness has correct, incorrect, and explicit uncertainty decisions');
    assert(!visible(document.getElementById('failureChoices'))&&!visible(document.getElementById('reviewNotes')),'Fresh turns show only the correctness decision, without 16 competing labels');
    document.querySelector('input[value="incorrect"]').click();await Promise.resolve();
    assert(visible(document.querySelector('.stage-picker'))&&![...document.querySelectorAll('.failure-stage')].some(visible),'Incorrect reveals stage choices without choosing a stage automatically');
    document.querySelector('[data-failure-stage="Generation"]').click();
    assert(document.querySelector('.failure-stage:not([hidden])').dataset.stage==='Generation'&&[...document.querySelectorAll('.failure-choice-grid input')].filter(visible).length===8,'Only the chosen stage types are visible');
    assert(document.querySelectorAll('.failure-stage[data-stage="Generation"] .failure-subheading').length===3,'Generation errors have three smaller groups');
    document.querySelector('#chosenStage button').click();assert(visible(document.querySelector('.stage-picker')),'Reviewer can reopen the compact stage picker');
    document.querySelector('[data-failure-stage="Retrieval"]').click();
    assert([...document.querySelectorAll('.failure-choice-grid input')].filter(visible).length===3&&!getFailureTypes(entry(D.rows[0])).length,'Browsing stages does not assign any failure');
    assert(document.getElementById('stageEvidence').textContent.includes('ก่อน rerank')&&document.getElementById('stageEvidence').textContent.includes('ไม่ได้'),'Stage guide explains required original traces');
    delete A['case#1'];failureStageViews.delete(stageViewKey(D.rows[0]));save();queue();renderReview();await Promise.resolve();
    assert(JSON.stringify(D.failure_types)===JSON.stringify(Array.from({length:16},(_,i)=>'E'+(i+1))),'Paper codes E1 through E16 appear exactly once in order');
    assert(document.querySelectorAll('.failure-stage').length===4&&document.querySelectorAll('.failure-choice-grid input').length===16,'All 16 errors are grouped into four pipeline stages');
    assert(JSON.stringify([...document.querySelectorAll('.failure-stage')].map(s=>s.querySelectorAll('input').length))==='[3,3,2,8]','Stage boundaries follow the paper');
    for(const code of D.failure_types){document.querySelector('[data-failure-type="'+code+'"]').click();assert(document.getElementById('failureGuide').textContent.includes(guides[code].check),'Evidence guide for '+code);document.getElementById('clearFailureChoices').click();}
    assert(document.getElementById('failureGuideContent').children.length===16&&document.getElementById('failureGuidelines').textContent.includes('ขั้นตอนแรก'),'Guideline covers all errors and the paper protocol');

    const unopened=document.querySelector('.queue-group[data-id="other"]');
    assert(!unopened.open&&unopened.querySelectorAll('.queue-item').length===0,'Collapsed conversations do not build hidden turn cards during startup');
    assert(corpusRequests.has(JSON.stringify(['case#1',''])),'Corpus request runs separately from the form and starts before profile loading finishes');
    const sidebar=document.querySelector('.queue'),queueList=document.getElementById('queue');
    assert(queueList.clientHeight>sidebar.clientHeight*.45,'Compact sidebar leaves most of its height for scrolling conversations and turns: '+queueList.clientHeight+'/'+sidebar.clientHeight);
    const firstGroup=queueList.querySelector('.queue-group[data-id="case"]');
    assert(firstGroup.querySelectorAll('.queue-item').length===3,'Expanded conversation has all of its turn cards');
    assert(firstGroup.querySelector('.conversation-name').textContent.includes('Venus')&&firstGroup.querySelector('.case-id').textContent==='case','Readable first-question name appears with the original conversation ID');
    const laterQuestion=D.rows[2].question;D.rows[2].question='A follow-up without the original topic';
    document.getElementById('search').value=conversationInfo('case').name;applyFilters();
    assert(filtered().some(r=>r.key==='case#3'),'Searching the conversation name also finds its differently worded follow-up turns');
    document.getElementById('search').value='C01';applyFilters();
    assert(filtered().length===3&&filtered().every(r=>r.id==='case'),'Stable conversation number finds only its own turns');
    D.rows[2].question=laterQuestion;document.getElementById('search').value='';applyFilters();
    openTurn('case#3');openTurn('case#1');
    assert(queueButtons.get('case#1').getBoundingClientRect().top>=queueList.querySelector('.queue-group[data-id="case"] > summary').getBoundingClientRect().bottom,'Navigating back keeps the first turn below its sticky conversation header');
    assert(document.getElementById('notes'),'Annotation form is available before corpus matches finish');
    assert(D.rows.filter(r=>r._summary).length===2,'Other conversation starts as summaries');
    assert(document.querySelector('#evidence > #answer')&&document.getElementById('answer').nextElementSibling===document.getElementById('evidenceSearch'),'RAG answer appears directly above the Find text field');
    assert(document.getElementById('answer').getBoundingClientRect().bottom<document.querySelector('.comparison-track').getBoundingClientRect().top,'RAG answer is above all comparison columns');
    assert(!document.querySelector('#evidence-selected #answer'),'Answer is not repeated inside a scrolling evidence column');
    assert(getComputedStyle(document.querySelector('#answer .prose')).maxHeight==='none','Top answer has no inner scrolling limit');
    assert(document.querySelector('#answer mark.answer-match').title.includes('selected chunk(s) 1'),'Answer highlights identify the original selected chunk');
    assert(document.querySelector('#evidence-selected mark.answer-match'),'Selected evidence highlights overlap with the RAG answer');
    document.getElementById('goldDisclosure').querySelector('summary').click();
    assert(document.querySelector('#reference mark.answer-match'),'Gold answer highlights overlap with the RAG answer');
    const numericHighlight=highlighted('The age is 1.5 billion years.','','The age is 4.5 billion years.');
    assert(!numericHighlight.querySelector('mark.answer-match'),'Sentences with conflicting decimal values are not marked as likely matches');
    const sentenceText='Venus rotates westward. Saturn has rings.',sentenceHighlight=highlighted(sentenceText,'','Venus rotates slowly westward.');
    assert(sentenceHighlight.querySelectorAll('mark.answer-match').length===1&&sentenceHighlight.querySelector('mark.answer-match').textContent==='Venus rotates westward.','Likely matching sentence is highlighted in full, including punctuation');
    assert(sentenceHighlight.textContent===sentenceText,'Sentence highlighting preserves every character of the evidence');
    assert(!highlighted('The planet has an ancient volcano.','','The planet has cold oceans.').querySelector('mark.answer-match'),'A single generic shared word does not highlight an unrelated sentence');
    const searchedSentence=highlighted(sentenceText,'westward','Venus rotates slowly westward.');
    assert(searchedSentence.querySelector('mark.search-match').textContent==='westward'&&searchedSentence.textContent===sentenceText,'Literal search highlighting works inside a matched sentence');
    const current=D.rows[0],originalContexts=current.contexts;
    current.contexts=['Saturn has rings.','Venus rotates westward.'];renderReview();
    assert(document.querySelector('#evidence-selected .passage h3').textContent==='Selected chunk 1','Selected chunks retain recorded order even when a later chunk matches the answer');
    assert(document.querySelectorAll('#evidence-selected .passage')[1].querySelector('mark.answer-match'),'Later matching chunk keeps its sentence highlight without moving');
    document.querySelector('#evidence-selected .passage button').click();
    assert(entry(current).evidence.at(-1).locator==='selected:1','Citations retain the original first-chunk locator');
    const highlightedAnswer=document.querySelector('#answer .prose'),range=document.createRange(),walker=document.createTreeWalker(highlightedAnswer,NodeFilter.SHOW_TEXT),firstText=walker.nextNode();
    range.setStart(firstText,0);range.setEnd(firstText,5);window.getSelection().removeAllRanges();window.getSelection().addRange(range);
    cite(current,'RAG answer','answer',current.answer,highlightedAnswer);window.getSelection().removeAllRanges();
    assert(entry(current).evidence.at(-1).quote==='Venus','Exact selections across highlighted answer text can be cited');
    current.contexts=originalContexts;renderReview();
    const judge=()=>document.getElementById('judgeDisclosure');
    assert(!judge().open,'Incorrect turn starts with the judge collapsed');
    assert(!judge().classList.contains('incorrect')&&!judge().classList.contains('correct'),'Closed judge panel has no verdict color');
    assert(judge().querySelector('summary').textContent==='Show mtRAG judge assessment','Closed toggle title does not reveal the verdict or score');
    assert(!judge().querySelector('.judge-content').checkVisibility(),'Collapsed assessment content is not visible');
    assert(!document.querySelector('#question .judges'),'Question panel does not reveal the score');
    assert(!document.querySelector('[data-tab="judge"]'),'Judge checks share the same disclosure');
    judge().querySelector('summary').click();await new Promise(resolve=>setTimeout(resolve,0));
    assert(judge().open&&judge().querySelector('.judge-content').checkVisibility(),'Judge content appears when expanded');
    assert(judge().textContent.includes('Saved explanation for browser disclosure checks.')&&judge().textContent.includes('Judge claim reason.'),'Saved explanation and full statement reasons are inside the toggle');
    assert(judge().textContent.includes('Scores & calculation')&&judge().textContent.includes('0.4'),'Score and calculations are inside the toggle');
    renderReview();assert(judge().open,'Rerender preserves the current turn disclosure choice');
    openTurn('case#2');assert(!judge().open,'Another correct turn starts collapsed');
    openTurn('case#1');assert(judge().open,'Returning to a turn remembers its disclosure choice');
    const originalJudge=D.rows[2].judges.mtrag;D.rows[2].judges.mtrag={status:'not_run',headline:'Judge not run.'};
    openTurn('case#3');assert(!judge().open,'Unjudged turns also start collapsed');
    judge().querySelector('summary').click();assert(judge().textContent.includes('Judge not run.'),'Unjudged status remains available inside the toggle');
    D.rows[2].judges.mtrag=originalJudge;openTurn('case#1');
    await fetch('/test/unblock',{method:'POST'});
    await wait(()=>document.querySelector('#evidence-corpus .column-body').textContent.includes('35 matching passages'));
    assert(document.querySelectorAll('#evidence-corpus .passage').length===30,'Remote corpus results are bounded');
    let loads=0;const originalFetch=window.fetch;
    window.fetch=(url,options)=>{if(String(url).startsWith(D.detail_api))loads++;return originalFetch(url,options);};
    const firstQueueNode=document.querySelector('.queue-item');
    openTurn('other#1');openTurn('other#2');openTurn('other#1');
    await ensureConversation('other');await new Promise(resolve=>setTimeout(resolve,0));
    assert(loads===1,'Rapid navigation reuses one conversation request');
    assert(document.querySelector('.queue-item')===firstQueueNode,'Lazy conversation load retains the sidebar');
    assert(document.querySelector('#answer').textContent.includes('Second conversation private answer 1'),'Selected conversation answer arrives');
    assert(!judge().open,'Lazy-loaded correct conversation starts collapsed');
    openTurn('other#2');
    assert(document.getElementById('notes').value==='Earlier saved note','Previously saved notes load for a previously unopened turn');
    assert(loads===1,'Another turn in the loaded conversation needs no request');
    document.querySelector('[data-tab="history"]').click();
    assert(document.getElementById('evidenceBody').textContent.includes('Second conversation private answer 1'),'Complete conversation history arrives together');
    openTurn('other#1');
    const search=document.getElementById('corpusSearch');search.value='doc-17';search.dispatchEvent(new Event('input'));
    await wait(()=>document.querySelector('#evidence-corpus .column-body').textContent.includes('1 matching passages'));
    assert(document.querySelectorAll('#evidence-corpus .passage').length===1,'Live corpus search shows only its matches');
    assert(document.querySelector('#evidence-corpus .passage').textContent.includes('doc-17'),'Live search keeps document provenance');
    document.querySelector('input[value="incorrect"]').click();await Promise.resolve();
    document.querySelector('[data-failure-type="E10"]').click();
    const notes=document.getElementById('notes');notes.value='Progressive autosave';notes.dispatchEvent(new Event('input'));
    await flushProfile('PS');
    const saved=await (await fetch('/api/annotations/pilot/PS')).json();
    assert(saved.annotations['other#1'].notes==='Progressive autosave','Autosave works after a lazy load');
    assert(saved.annotations['other#2'].notes==='Earlier saved note','Autosave preserves other turns');
    assert(rowsOut().some(row=>row.id==='other'&&row.turn===1&&row.notes==='Progressive autosave'),'Exports include annotations for loaded conversations');
    judge().querySelector('summary').click();await new Promise(resolve=>setTimeout(resolve,0));
    await switchAnnotator('KH');assert(activeKey==='other#1'&&!hasAnnotation(A['other#1']),'Reviewer switching remains independent');
    assert(!judge().open,'Another reviewer starts with the assessment collapsed');
    await switchAnnotator('PS');assert(A['other#1'].notes==='Progressive autosave','Saved reviewer data survives switching');
    assert(judge().open,'Original reviewer keeps their own disclosure choice');
    const later=D.rows.find(r=>r.key==='case#2');later.judges.mtrag={...later.judges.mtrag,conditioned_score:.3,label:'incorrect'};
    A['case#1']={review_complete:true};A['case#2']={human_label:'incorrect',failure_types:['E4'],review_complete:true};save();queue();
    assert(document.getElementById('labelProgress').textContent==='0 / 1','Review-only first failure and a labelled later failure do not count as a labelled conversation');
    assert(document.querySelector('.queue-item[data-key="case#2"]').dataset.reviewed==='true','A non-first incorrect turn has its own reviewed checkmark');
    assert(document.querySelector('.queue-group[data-id="case"]').dataset.labelled==='false','Conversation stays incomplete until its first incorrect turn is labelled');
    assert(!document.querySelector('.queue-group[data-id="case"] .conversation-complete'),'Review marks and labels on later failures do not show the conversation completion mark');
    openTurn('case#2');document.getElementById('reviewComplete').click();
    assert(!reviewed(later)&&annotationComplete(later),'Review status can be cleared independently of a complete failure annotation');
    A['case#1']={review_complete:false};openTurn('case#1');
    document.querySelector('[data-failure-type="E10"]').click();
    assert(document.getElementById('labelProgress').textContent==='1 / 1'&&document.querySelector('.queue-group[data-id="case"] .conversation-complete'),'One checked failure cause on the first incorrect turn is enough for the label and conversation mark');
    assert(!A['case#1'].human_label&&!reviewed(D.rows[0]),'Checkbox-only labelling requires neither a correctness decision nor a reviewed mark');
    document.querySelector('[data-failure-type="E4"]').click();
    assert(document.getElementById('labelProgress').textContent==='1 / 1','Multiple checked causes still count the conversation once');
    document.querySelector('[data-failure-type="E10"]').click();document.querySelector('[data-failure-type="E4"]').click();
    assert(document.getElementById('labelProgress').textContent==='0 / 1'&&!document.querySelector('.queue-group[data-id="case"] .conversation-complete'),'Removing the last checked failure choice removes its count and mark');
    document.querySelector('[data-failure-type="insufficient_evidence"]').click();
    assert(document.getElementById('labelProgress').textContent==='1 / 1','Checked alternative outcomes also count as failure choices');
    document.querySelector('input[value="correct"]').click();
    assert(document.getElementById('labelProgress').textContent==='0 / 1'&&!document.querySelector('.queue-group[data-id="case"] .conversation-complete'),'Correct alone and its automatic Not a failure value do not count as a checked failure choice');
    A['case#1']={human_label:'incorrect',failure_types:['E10']};save();queue();
    assert(document.getElementById('labelProgress').textContent==='1 / 1','First incorrect turn counts its conversation once');
    assert(document.querySelector('.queue-group[data-id="case"]').dataset.labelled==='true'&&document.querySelector('.queue-item[data-key="case#1"]').dataset.labelled==='true','Completed first failure marks the conversation and target as labelled');
    const completedStatus=document.querySelector('.queue-item[data-key="case#1"] .turn-status');
    assert(completedStatus.textContent.includes('Incorrect')&&completedStatus.title.includes('mtRAG: Incorrect'),'Labelling preserves the original visible judge verdict');
    assert(getComputedStyle(completedStatus).color===getComputedStyle(document.querySelector('.queue-item[data-key="case#2"] .turn-status')).color,'Labelled and unlabelled incorrect verdicts retain the same red text');
    assert(getComputedStyle(document.querySelector('.queue-item[data-key="case#1"]')).backgroundColor===getComputedStyle(document.querySelector('.queue-item[data-key="case#2"]')).backgroundColor,'Labelling does not change an incorrect turn card from red to green');
    assert(document.querySelector('.queue-item[data-key="case#2"] .turn-status').textContent.includes('Incorrect'),'Later unlabelled incorrect cards retain the judge verdict');
    assert(document.querySelector('.queue-group[data-id="case"] .conversation-ident .conversation-complete').textContent==='✓ Labelled','First-failure completion has an explicit mark beside the conversation ID');
    const allCorrectGroup=document.querySelector('.queue-group[data-id="other"]');
    assert(allCorrectGroup.dataset.allCorrect==='true'&&allCorrectGroup.textContent.includes('All turns correct'),'All-correct conversations are green without contributing failure labels');
    assert(!allCorrectGroup.querySelector('.conversation-complete'),'All-correct color does not falsely mark a first-failure annotation complete');
    assert(getComputedStyle(allCorrectGroup.querySelector('summary')).backgroundColor===getComputedStyle(document.querySelector('.queue-group[data-id="case"] > summary')).backgroundColor,'All-correct and completed-first-failure headers both use green');
    const unscored=D.rows.find(r=>r.key==='other#2'),savedJudge=unscored.judges.mtrag;
    unscored.judges.mtrag={status:'not_run'};queue();
    assert(document.querySelector('.queue-group[data-id="other"]').dataset.allCorrect==='false','A missing judge score never counts as an all-correct conversation');
    unscored.judges.mtrag=savedJudge;queue();
    const closedGroup=document.querySelector('.queue-group[data-id="case"]');closedGroup.open=false;
    assert(getComputedStyle(closedGroup.querySelector('.queue-turns')).display==='none'&&getComputedStyle(closedGroup).borderBottomWidth==='0px','Collapsed conversations have no empty turn box or separator border');
    const scrollingList=document.getElementById('queue'),savedHeight=scrollingList.style.height,savedFlex=scrollingList.style.flex;
    const clippedGroup=document.querySelector('.queue-group[data-id="other"]');clippedGroup.open=false;
    scrollingList.style.flex='none';scrollingList.style.height=(closedGroup.offsetHeight+45)+'px';scrollingList.scrollTop=0;observeQueueEdges();
    await wait(()=>clippedGroup.dataset.clipped==='true');
    assert(getComputedStyle(clippedGroup).visibility==='hidden','Partially cut-off closed conversation cards do not leave a rectangular preview at the list edge');
    scrollingList.scrollTop=scrollingList.scrollHeight;
    await wait(()=>clippedGroup.dataset.clipped==='false');
    assert(getComputedStyle(clippedGroup).visibility==='visible','Scrolling the clipped conversation fully into view reveals its complete card');
    scrollingList.style.height=savedHeight;scrollingList.style.flex=savedFlex;observeQueueEdges();
    document.getElementById('fTodo').checked=true;applyFilters();
    assert(document.getElementById('labelProgress').textContent==='1 / 1','Filtering out the first failure does not change the conversation label count');
    assert(document.querySelector('.queue-group[data-id="case"]').dataset.labelled==='true','Filtered conversation uses the original first failure, not its first visible turn');
    assert(document.querySelector('.queue-group[data-id="case"]').dataset.allCorrect==='false'&&document.querySelector('.queue-group[data-id="case"] .conversation-complete'),'Filtered cards keep the full-conversation color and completion mark');
    document.getElementById('fTodo').checked=false;applyFilters();openTurn('case#2');
    await flushProfile('PS');
    const storedReview=await (await fetch('/api/annotations/pilot/PS')).json();
    assert(storedReview.annotations['case#2'].review_complete===false,'Explicit unreviewed flag saves to the server');
    assert(rowsOut().find(r=>r.id==='case'&&r.turn===2).review_complete===false,'JSON exports preserve independent review state');
    await switchAnnotator('KH');
    assert(document.getElementById('labelProgress').textContent==='0 / 1'&&!document.getElementById('reviewComplete').checked,'Reviewer profiles have independent label counts and review marks');
    profiles.delete('PS');await switchAnnotator('PS');
    assert(document.getElementById('labelProgress').textContent==='1 / 1'&&!document.getElementById('reviewComplete').checked,'Server reload restores separate conversation labels and per-turn review state');
    await document.getElementById('imp').onchange({target:{files:[new File([JSON.stringify(rowsOut())],'review-roundtrip.json')],value:''}});
    assert(A['case#2'].review_complete===false,'JSON import preserves an explicit unreviewed flag');
    openTurn('case#3');document.getElementById('reviewComplete').click();
    assert(reviewed(D.rows.find(r=>r.key==='case#3'))&&!labelled(D.rows.find(r=>r.key==='case#3')),'Any later turn can be marked reviewed without a correctness label');
    assert(document.getElementById('labelProgress').textContent==='1 / 1','Reviewing a later turn does not add another conversation label');
    openTurn('case#1');document.getElementById('clearFailureChoices').click();
    assert(document.getElementById('labelProgress').textContent==='0 / 1'&&document.querySelector('.queue-group[data-id="case"]').dataset.labelled==='false','Clearing the first failure cause removes conversation completion');
    assert(!document.querySelector('.queue-group[data-id="case"] .conversation-complete'),'Clearing the first-failure annotation also removes its ID completion mark');
    assert(document.querySelector('.queue-item[data-key="case#1"] .turn-status').textContent.includes('Incorrect'),'Clearing the final failure choice preserves the original judge verdict on its card');
    // Historical labels stay separate from the new scientific categories.
    const oldRecords=[{id:'case',turn:1,annotator:'WN',human_label:'incorrect',failure_type:'grounding',notes:'Original label',evidence:[{source:'RAG answer',locator:'answer',quote:'Venus'}]},
      {id:'case',turn:2,annotator:'WN',human_label:'incorrect',failure_types:['retrieval','response_coverage'],review_complete:false},
      {id:'case',turn:3,annotator:'WN',failure_type:'chunking'}];
    await document.getElementById('imp').onchange({target:{files:[new File([JSON.stringify(oldRecords)],'older-labels.json')],value:''}});
    await flushProfile('WN');profiles.delete('WN');await switchAnnotator('WN');openTurn('case#1');await Promise.resolve();
    assert(JSON.stringify(A['case#1'].legacy_failure_types)==='["grounding"]'&&!A['case#1'].failure_types.length,'Old single category survives import and server reload without a guessed conversion');
    assert(JSON.stringify(A['case#2'].legacy_failure_types)==='["retrieval","response_coverage"]'&&!A['case#2'].failure_types.length,'Old multi-category arrays survive intact');
    assert(A['case#1'].notes==='Original label'&&A['case#1'].evidence[0].quote==='Venus','Taxonomy migration preserves notes and citations');
    assert(reviewed(D.rows[0])&&!reviewed(D.rows[1]),'Historical completed reviews and explicit unreviewed flags remain independent');
    assert(document.getElementById('labelProgress').textContent==='0 / 1'&&document.querySelector('.legacy-failures').textContent.includes('Grounding'),'Legacy history is visible and does not count as a new-taxonomy label');
    document.querySelector('[data-failure-type="E13"]').click();await flushProfile('WN');profiles.delete('WN');await switchAnnotator('WN');
    assert(document.querySelector('[data-failure-type="E13"]').checked&&A['case#1'].legacy_failure_types[0]==='grounding','New paper label and earlier history save together');
    assert(document.getElementById('labelProgress').textContent==='1 / 1','One new failure checkbox completes the first-incorrect label');
    let exported;download=(name,text,type)=>exported={name,text,type};document.getElementById('expJson').click();
    const exportedRows=JSON.parse(exported.text),first=exportedRows.find(r=>r.turn===1);
    assert(first.failure_taxonomy==='ragec-e1-e16-2026'&&first.failure_type==='E13'&&JSON.stringify(first.failure_types)==='["E13"]'&&first.legacy_failure_types[0]==='grounding','JSON identifies the taxonomy and preserves current plus historical labels');
    document.getElementById('expCsv').click();assert(exported.text.includes('legacy_failure_types')&&exported.text.includes('ragec-e1-e16-2026')&&exported.text.includes('""grounding""')&&exported.text.includes('""E13""'),'CSV preserves taxonomy, current labels, and old labels');
    await document.getElementById('imp').onchange({target:{files:[new File([JSON.stringify(exportedRows)],'new-taxonomy.json')],value:''}});
    assert(A['case#1'].failure_type==='E13'&&A['case#1'].legacy_failure_types[0]==='grounding','New JSON import preserves both label vocabularies');
    assert(!getProfile('PS').annotations['case#1'].legacy_failure_types?.length,'Legacy import remains isolated by reviewer');
    await switchAnnotator('PS');
    // Guided completion preserves old multi-stage work until the reviewer resolves it.
    await switchAnnotator('KH');
    A['case#2']={human_label:'incorrect',failure_types:['E4','E13'],notes:'Keep these notes',review_complete:false};openTurn('case#2');await Promise.resolve();
    assert(document.querySelectorAll('.cause-chip').length===2&&finishProblem(D.rows[1]),'Earlier cross-stage selections stay visible and need an explicit decision');
    document.querySelector('[data-failure-stage="Chunking"]').click();
    assert(JSON.stringify(getFailureTypes(A['case#2']))==='["E4","E13"]','Changing the stage view never silently clears existing labels');
    await finishAndNext(D.rows[1]);assert(activeKey==='case#2'&&A['case#2'].review_complete===false,'Save and next does not complete unresolved cross-stage labels');
    document.querySelector('.cause-chip[aria-label="Remove E4"]').click();
    assert(!finishProblem(D.rows[1])&&A['case#2'].notes==='Keep these notes','Removing a conflicting selection preserves explanation and permits completion');
    failureStageViews.delete(stageViewKey(D.rows[1]));refreshFailureChoices(D.rows[1]);
    const example=document.querySelector('[data-help-type="E13"]');assert(!example.hidden&&example.textContent.includes('Compare:'),'Selected failure shows comparison notes inline');
    openGuideline('failure');assert(document.getElementById('failureGuidelines').textContent.includes(D.failure_annotation_aids.E13.example),'Detailed examples are available in the top-left guideline');document.getElementById('failureGuidelines').close();
    // Reset only this fixture profile to test completion and navigation deterministically.
    A['case#1']={};A['case#2']={review_complete:true};A['case#3']={review_complete:true};A['other#1']={};A['other#2']={review_complete:true};openTurn('case#1');await Promise.resolve();
    await finishAndNext(D.rows[0]);assert(activeKey==='case#1'&&!reviewed(D.rows[0]),'An empty decision cannot be completed accidentally');
    document.querySelector('input[value="uncertain"]').click();await Promise.resolve();
    assert(A['case#1'].human_label==='uncertain'&&JSON.stringify(getFailureTypes(A['case#1']))==='["insufficient_evidence"]'&&visible(document.getElementById('failureChoices')),'Optional uncertain review preserves its value without hiding failure annotation');
    document.getElementById('notes').value='Missing evidence to decide';document.getElementById('notes').dispatchEvent(new Event('input'));
    await finishAndNext(D.rows[0]);assert(activeKey==='other#1','Save and next goes to the next unreviewed turn, skipping completed turns');
    const completed=await (await fetch('/api/annotations/pilot/KH')).json();
    assert(completed.annotations['case#1'].human_label==='uncertain'&&completed.annotations['case#1'].review_complete===true,'Server saves uncertain decisions with explicit review completion');
    profiles.delete('KH');await switchAnnotator('KH');openTurn('case#1');await Promise.resolve();
    assert(document.querySelector('input[value="uncertain"]').checked&&A['case#1'].notes==='Missing evidence to decide','Uncertain decision and notes survive server reload');
    const unknownExport=rowsOut();await document.getElementById('imp').onchange({target:{files:[new File([JSON.stringify(unknownExport)],'uncertain.json')],value:''}});
    assert(A['case#1'].human_label==='uncertain','JSON roundtrip preserves uncertainty without changing it to incorrect');
    openTurn('other#1');document.querySelector('input[value="correct"]').click();await Promise.resolve();
    assert(visible(document.getElementById('failureChoices'))&&JSON.stringify(getFailureTypes(A['other#1']))==='["not_a_failure"]','Optional Correct review keeps failure annotation available');
    await finishAndNext(D.rows.find(r=>r.key==='other#1'));assert(activeKey==='other#1'&&reviewed(D.rows.find(r=>r.key==='other#1')),'Final save completes the last turn without wrapping to a reviewed turn');
    // Offline failure keeps the saved draft and current turn visible for retry.
    A['other#1']={human_label:'incorrect',failure_types:['E13'],review_complete:false};renderReview();await Promise.resolve();
    const workingFetch=window.fetch;window.fetch=(url,options)=>String(url)===D.annotation_api+'/KH'&&options?.method==='POST'?Promise.resolve(new Response(JSON.stringify({error:'Fixture offline'}),{status:503,headers:{'Content-Type':'application/json'}})):workingFetch(url,options);
    await finishAndNext(D.rows.find(r=>r.key==='other#1'));
    assert(activeKey==='other#1'&&!getProfile('KH').connected&&getProfile('KH').dirty.has('other#1'),'Failed server save retains the turn and retryable local draft');
    assert(JSON.parse(localStorage.getItem(profileKey('KH')))['other#1'].failure_types[0]==='E13','Offline draft retains the selected failure');
    window.fetch=workingFetch;await flushProfile('KH');assert(getProfile('KH').connected&&!getProfile('KH').dirty.size,'Retry syncs the completed local review');
    await switchAnnotator('PS');assert(A['other#1'].notes==='Progressive autosave','Guided workflow keeps other annotators independent');
    await fetch('/test/result',{method:'POST',body:JSON.stringify({status:'pass',checks})});
  }catch(error){await fetch('/test/result',{method:'POST',body:JSON.stringify({status:'fail',checks,error:error.message+'\n'+error.stack})});}
})();
</script>
"""


LAYOUT_CHECKS = r"""
<script>
(async()=>{
  const assert=(ok,message)=>{if(!ok)throw Error(message);};
  const frames=()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
  try{
    await annotationReady;openTurn('case#2');await Promise.resolve();
    document.querySelector('input[value="incorrect"]').click();await Promise.resolve();
    document.querySelector('[data-failure-stage="Generation"]').click();await frames();
    const panel=document.getElementById('annotationPanel'),finish=document.getElementById('finishReview');
    assert(document.documentElement.scrollWidth<=window.innerWidth+1,'The page fits the viewport without horizontal overflow');
    assert(panel.getBoundingClientRect().width<=window.innerWidth,'The annotation panel fits on screen');
    assert(document.querySelectorAll('.failure-stage:not([hidden])').length===1,'Only the chosen failure stage is exposed');
    assert(document.getElementById('stageFilter').hidden,'Chosen stage filter collapses to a compact summary');
    if(window.innerWidth>950){
      assert(finish.getBoundingClientRect().bottom<=window.innerHeight-8,'Save and next stays inside the visible desktop viewport');
      assert(getComputedStyle(panel).position==='sticky','Desktop annotation stays beside the evidence');
    }else{
      panel.scrollIntoView();await frames();
      assert(panel.getBoundingClientRect().top>=-1&&panel.getBoundingClientRect().top<40,'Mobile can scroll directly to the annotation form');
      assert(getComputedStyle(panel).position==='static','Mobile form scrolls with the page');
    }
    document.querySelector('[data-failure-type="E13"]').click();await frames();
    const guide=document.querySelector('[data-help-type="E13"]');assert(!guide.hidden&&guide.textContent.includes('Compare:')&&!document.querySelector('dialog[open]'),'Failure selection shows comparison notes without opening a popup');
    await fetch('/result',{method:'POST',body:JSON.stringify({status:'pass',width:window.innerWidth})});
  }catch(error){await fetch('/result',{method:'POST',body:JSON.stringify({status:'fail',error:error.message+'\n'+error.stack})});}
})();
</script>
"""


class RecoveryOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work, self.archive = make_run(self.temp.name)
        self.canonical = self.work / 'out.mtrag.json'
        self.records = json.loads(self.canonical.read_text(encoding='utf-8'))

    def snapshot(self, suffix, records):
        path = self.work / ('out.mtrag.recovery_%s.json' % suffix)
        path.write_text(json.dumps(records), encoding='utf-8')
        return path

    def test_nested_conversations_refresh_archive_and_generation_ids(self):
        folder = pathlib.Path(self.temp.name) / 'conversation'
        nested = folder / 'vector' / 'cloud'
        nested.mkdir(parents=True)
        path = nested / 'conversation.json'
        export = {'rag_name': 'vector', 'conversations': [{'id': 'case', 'domain': 'cloud',
            'turns': [{'dataset_turn': 1, 'task_id': 'case<::>1', 'question': 'Question?',
                       'answer': 'Answer.', 'retrieved': True,
                       'passages': [{'id': 'original', 'text': 'Original evidence.'}]}]}]}
        path.write_text(json.dumps(export), encoding='utf-8')
        first = annotation.load_passage_archive(folder)
        self.assertEqual(first['cloud']['documents'][0]['id'], 'original')
        traces = annotation.generation_traces(annotation.conversation_snapshot(folder))
        self.assertEqual(traces[('vector', 'case', '1')][0]['passages'][0]['id'], 'original')
        export['conversations'][0]['turns'][0]['passages'][0] = {'id': 'updated', 'text': 'New, longer evidence.'}
        path.write_text(json.dumps(export), encoding='utf-8')
        self.assertEqual(annotation.load_passage_archive(folder)['cloud']['documents'][0]['id'], 'updated')

    def test_best_recovery_is_shown_with_its_source_and_missing_records(self):
        self.canonical.unlink()
        self.snapshot('1', [])
        source = self.snapshot('2', self.records[:1])
        original = source.read_bytes()
        with mock.patch.object(annotation, 'load_passage_archive', return_value=self.archive):
            data = annotation.build_data(self.work, live=True)
        self.assertEqual(data['coverage']['mtrag']['source'], source.name)
        self.assertEqual(data['coverage']['mtrag']['records'], 1)
        self.assertEqual(data['rows'][-1]['judges']['mtrag']['status'], 'record_missing')
        self.assertEqual(source.read_bytes(), original)
        self.assertFalse(self.canonical.exists())
        self.assertEqual(available_runs(pathlib.Path(self.temp.name))[0]['judges'], ['mtrag'])

    def test_published_output_remains_authoritative(self):
        self.snapshot('later', self.records * 0)
        loaded, source = annotation.load_output(self.work, 'mtrag')
        self.assertEqual(source, 'out.mtrag.json')
        self.assertEqual(len(loaded), 2)

    def test_saved_input_may_omit_the_outer_conversation_id(self):
        self.canonical.unlink()
        inputs = annotation.load_input_records(self.work)
        records = copy.deepcopy(self.records)
        for record in records:
            record['input'] = {key: value for key, value in inputs[record['id']].items() if key != 'id'}
        source = self.snapshot('1', records)
        loaded, selected = annotation.load_output(self.work, 'mtrag')
        self.assertEqual(selected, source.name)
        self.assertEqual(len(loaded), 2)

    def test_invalid_or_wrong_input_recovery_cannot_hide_valid_results(self):
        self.canonical.unlink()
        valid = self.snapshot('1', self.records)
        self.snapshot('2', []).write_text('{unfinished', encoding='utf-8')
        wrong = copy.deepcopy(self.records)
        wrong[0]['input'] = {'conversation': []}
        self.snapshot('3', wrong)
        with self.assertWarns(UserWarning):
            loaded, source = annotation.load_output(self.work, 'mtrag')
        self.assertEqual(source, valid.name)
        self.assertEqual(len(loaded), 2)

    def test_cache_refreshes_when_a_better_recovery_arrives(self):
        self.canonical.unlink()
        self.snapshot('1', self.records[:1])
        pages = RunPages()
        with mock.patch.object(annotation, 'load_passage_archive', return_value=self.archive):
            first = pages.get(self.work)
            self.assertEqual(first['rows']['other#1']['judges']['mtrag']['status'], 'record_missing')
            self.snapshot('2', self.records)
            second = pages.get(self.work)
        self.assertIsNot(first, second)
        self.assertEqual(second['rows']['other#1']['judges']['mtrag']['status'], 'judged')




if __name__ == '__main__':
    unittest.main()
