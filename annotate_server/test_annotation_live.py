"""Progressive live loading: small pages, complete details, and independent saves."""

import gzip
import json
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
from annotation_server import RunPages, make_handler
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
                    self.assertLess(len(body), 120_000)
                    self.assertLess(len(gzip.compress(body)), 30_000)
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
                            if self.path.startswith('/api/corpus/'):
                                corpus_gate.wait(timeout=5)
                            super().do_GET()

                    def do_POST(self):
                        if self.path == '/test/unblock':
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


PROGRESSIVE_CHECKS = r"""
<script>
(async()=>{
  let checks=0;
  const assert=(ok,message)=>{if(!ok)throw Error(message);checks++;};
  const wait=async test=>{const start=Date.now();while(!test()){if(Date.now()-start>10000)throw Error('Timed out waiting for live data.');await new Promise(resolve=>setTimeout(resolve,25));}};
  try{
    await annotationReady;
    assert(corpusRequests.size===1,'Corpus request runs separately from the form');
    const sidebar=document.querySelector('.queue'),queueList=document.getElementById('queue');
    assert(queueList.clientHeight>sidebar.clientHeight*.45,'Compact sidebar leaves most of its height for scrolling conversations and turns');
    const firstGroup=queueList.querySelector('.queue-group[data-id="case"]');
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
    document.querySelector('[data-failure-type="grounding"]').click();
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
    A['case#1']={review_complete:true};A['case#2']={human_label:'incorrect',failure_types:['retrieval'],review_complete:true};save();queue();
    assert(document.getElementById('labelProgress').textContent==='0 / 1','Review-only first failure and a labelled later failure do not count as a labelled conversation');
    assert(document.querySelector('.queue-item[data-key="case#2"]').dataset.reviewed==='true','A non-first incorrect turn has its own reviewed checkmark');
    assert(document.querySelector('.queue-group[data-id="case"]').dataset.labelled==='false','Conversation stays incomplete until its first incorrect turn is labelled');
    assert(!document.querySelector('.queue-group[data-id="case"] .conversation-complete'),'Review marks and labels on later failures do not show the conversation completion mark');
    openTurn('case#2');document.getElementById('reviewComplete').click();
    assert(!reviewed(later)&&annotationComplete(later),'Review status can be cleared independently of a complete failure annotation');
    A['case#1']={review_complete:false};openTurn('case#1');
    document.querySelector('[data-failure-type="grounding"]').click();
    assert(document.getElementById('labelProgress').textContent==='1 / 1'&&document.querySelector('.queue-group[data-id="case"] .conversation-complete'),'One checked failure cause on the first incorrect turn is enough for the label and conversation mark');
    assert(!A['case#1'].human_label&&!reviewed(D.rows[0]),'Checkbox-only labelling requires neither a correctness decision nor a reviewed mark');
    document.querySelector('[data-failure-type="retrieval"]').click();
    assert(document.getElementById('labelProgress').textContent==='1 / 1','Multiple checked causes still count the conversation once');
    document.querySelector('[data-failure-type="grounding"]').click();document.querySelector('[data-failure-type="retrieval"]').click();
    assert(document.getElementById('labelProgress').textContent==='0 / 1'&&!document.querySelector('.queue-group[data-id="case"] .conversation-complete'),'Removing the last checked failure choice removes its count and mark');
    document.querySelector('[data-failure-type="insufficient_evidence"]').click();
    assert(document.getElementById('labelProgress').textContent==='1 / 1','Checked alternative outcomes also count as failure choices');
    document.querySelector('input[value="correct"]').click();
    assert(document.getElementById('labelProgress').textContent==='0 / 1'&&!document.querySelector('.queue-group[data-id="case"] .conversation-complete'),'Correct alone and its automatic Not a failure value do not count as a checked failure choice');
    A['case#1']={human_label:'incorrect',failure_types:['grounding']};save();queue();
    assert(document.getElementById('labelProgress').textContent==='1 / 1','First incorrect turn counts its conversation once');
    assert(document.querySelector('.queue-group[data-id="case"]').dataset.labelled==='true'&&document.querySelector('.queue-item[data-key="case#1"]').dataset.labelled==='true','Completed first failure marks the conversation and target turn green');
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
    await fetch('/test/result',{method:'POST',body:JSON.stringify({status:'pass',checks})});
  }catch(error){await fetch('/test/result',{method:'POST',body:JSON.stringify({status:'fail',checks,error:error.message+'\n'+error.stack})});}
})();
</script>
"""


if __name__ == '__main__':
    unittest.main()
