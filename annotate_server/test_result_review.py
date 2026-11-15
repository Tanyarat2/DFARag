"""Result matching and browser navigation checks for the offline review page."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import make_result_review as review


BROWSER_CHECKS = r"""<script>
const report=document.createElement('pre');report.id='test-result';
try{
 const assert=(value,message)=>{if(!value)throw Error(message);};
 assert(document.querySelectorAll('.overview-card').length===3,'All annotator totals visible together');
 assert(!document.querySelector('.overview details')&&!document.querySelector('#overview-person'),'Dashboard has no collapsed panels or person selector');
 const dashboardRects=[...document.querySelectorAll('.overview-card')].map(card=>card.getBoundingClientRect());
 assert(dashboardRects.every(rect=>rect.width>0&&rect.height>0),'All three dashboards are rendered visibly');
 if(window.innerWidth>650)assert(dashboardRects[0].left<dashboardRects[1].left&&dashboardRects[1].left<dashboardRects[2].left&&Math.abs(dashboardRects[0].top-dashboardRects[2].top)<2,'Annotator dashboards sit side by side');
 assert(!/[\u0e00-\u0e7f]/.test(document.querySelector('.overview').innerText),'All dashboard interface text is English');
 assert(document.querySelectorAll('.download-dashboard').length===3,'Each annotator can download their dashboard');
 for(const name of D.annotators){
  const button=document.querySelector('[data-download-annotator="'+name+'"]'),originalClick=HTMLAnchorElement.prototype.click,originalURL=URL.createObjectURL;let blob,filename;
  try{HTMLAnchorElement.prototype.click=function(){filename=this.download;};URL.createObjectURL=value=>{blob=value;return originalURL.call(URL,value);};button.click();}
  finally{HTMLAnchorElement.prototype.click=originalClick;URL.createObjectURL=originalURL;}
  assert(filename===name+'_annotation_dashboard.png'&&blob?.type==='image/png','Download exports a named PNG for the selected person');
  const bytes=Uint8Array.from(atob(dashboardImageCanvas(name).toDataURL('image/png').split(',')[1]),character=>character.charCodeAt(0)),header=new DataView(bytes.buffer);
  assert(bytes[0]===137&&bytes[1]===80&&bytes[2]===78&&bytes[3]===71&&bytes.length>5000,'Download contains a real PNG image');
  assert(header.getUint32(16)===1200&&header.getUint32(20)>2000,'PNG includes the full dashboard at high resolution');
  assert(!button.disabled&&button.textContent==='Download PNG','Download button is ready for another export');
 }
 for(const name of D.annotators)assert(Number(document.querySelector('[data-overview-annotator="'+name+'"]').textContent)===D.overview.annotators[name].total,'Overall total matches exported decisions');
 for(const name of D.annotators){const card=document.querySelector('.overview-card[data-person="'+name+'"]');assert(card.querySelectorAll('.distribution-row').length===16,'Every annotator has E1 to E16');for(const row of card.querySelectorAll('.distribution-row'))assert(Number(row.dataset.count)===D.overview.annotators[name].label_counts[row.dataset.code],'Chart counts belong to that annotator');}
 const overviewText=$('overview-cards').textContent;
 assert($('coverage').value==='all'&&$('scope').value==='annotated','Shared annotated turns visible by default');
 assert(filtered.every(row=>row.labelled_count===D.annotators.length),'Turns missing an annotator are hidden initially');
 const requestedRow=D.rows.find(row=>row.method==='HippoRag'&&row.conversation_number==='C58'&&row.turn===5);
 assert(requestedRow&&openRequestedTurn(new URLSearchParams({method:'HippoRag',conversation:'C58',turn:'5'})),'C58 HippoRAG is accessible through a direct link');
 assert(filtered[index].key===requestedRow.key&&$('coverage').value==='any'&&$('scope').value==='everything','Direct links reveal requested turns even with empty decisions');
 $('method').value='';$('search').value='';$('scope').value='annotated';$('coverage').value='all';applyFilters();window.scrollTo(0,0);
 $('coverage').value='any';$('coverage').dispatchEvent(new Event('change'));
 $('scope').value='everything';$('scope').dispatchEvent(new Event('change'));
 assert(filtered.length===D.rows.length && filtered.length>0,'Every imported review turn is available');
 const numbersById=new Map();for(const row of D.rows){assert(/^C\d{2,}$/.test(row.conversation_number),'Every turn has a conversation number');assert(!numbersById.has(row.id)||numbersById.get(row.id)===row.conversation_number,'Same ID has the same number across methods');numbersById.set(row.id,row.conversation_number);}
 assert(document.querySelector('.conversation-number').textContent===filtered[index].conversation_number,'Conversation number visible');
 const searchNumber=filtered[index].conversation_number;$('search').value=searchNumber;$('search').dispatchEvent(new Event('input'));
 assert(filtered.length>0 && filtered.every(row=>row.conversation_number===searchNumber),'Search by conversation number');
 $('search').value='';$('search').dispatchEvent(new Event('input'));
 assert(document.querySelectorAll('.annotator').length===3,'Three decision cards');
 assert([...document.querySelectorAll('.edit-decision')].every(button=>button.disabled),'Offline snapshot cannot write source files');
 assert(document.querySelectorAll('.evidence-grid .panel').length===3,'Three evidence sections');
 assert(!document.querySelector('.evidence-grid details'),'Passages and sources have no toggles');
 assert([...document.querySelectorAll('.passage .prose')].every(node=>node.getBoundingClientRect().height>0),'Passage text expanded');
 assert(matchesGold({id:'gold-1',text:'Changed formatting'},[{id:'gold-1',text:'Reference'}]),'Gold ID match');
 assert(matchesGold({text:'Exact\n gold text'},[{id:'reference',text:'Exact gold text'}]),'Gold text fallback ignores whitespace');
 assert(!matchesGold({title:'Same title',text:'Different passage'},[{title:'Same title',text:'Reference passage'}]),'Titles alone do not create gold matches');
 assert(!matchesGold({text:''},[{text:''}]),'Empty text does not match');
 const excerpt='The reference contains enough specific words for a reliable exact match.';
 const embedded='Extra context.\n'+excerpt.replace('specific words','specific\n  words')+' More context.';
 const excerptRanges=goldTextRanges(embedded,[{text:excerpt}]);
 assert(excerptRanges.length===1,'Gold excerpt found inside longer passage');
 const marked=highlightedPassage(embedded,excerptRanges);
 assert(marked.querySelector('mark.gold-text') && marked.textContent===embedded,'Inline highlights preserve original text');
 assert(goldTextRanges('Unrelated words with no shared evidence.',[{text:excerpt}]).length===0,'Unrelated text not highlighted');
 assert(goldTextRanges('common words',[{text:'Only common words repeated elsewhere.'}]).length===0,'Short generic overlap not highlighted');
 assert(goldTextRanges(embedded,[{text:excerpt},{text:excerpt}]).length===1,'Overlapping highlights merged');
 const unsafe='<script>alert(1)<'+ '/script> '+excerpt;
 const safeNode=highlightedPassage(unsafe,goldTextRanges(unsafe,[{text:excerpt}]));
 assert(!safeNode.querySelector('script') && safeNode.textContent===unsafe,'Highlighted content stays plain text');
 const answerPanels=document.querySelectorAll('.reading>.answer');
 assert(answerPanels.length===2,'RAG and gold answer panels');
 assert(answerPanels[1].querySelector('h2').textContent==='Gold Answer','Gold heading');
 assert(answerPanels[1].querySelector('.prose').textContent===clean(filtered[index].gold||'Gold answer not recorded.'),'Gold answer content');
 if(window.innerWidth>620){const rag=answerPanels[0].getBoundingClientRect(),gold=answerPanels[1].getBoundingClientRect();assert(Math.abs(rag.top-gold.top)<2 && rag.right<gold.left,'Answers side by side');}
 assert(!/[\u2014\u00b7]/.test(document.body.innerText),'Display punctuation');
 const first=filtered[index].key;
 document.body.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));
 assert(index===1 && filtered[index].key!==first,'Right arrow');
 document.body.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowLeft',bubbles:true}));
 assert(index===0,'Left arrow');
 document.body.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowLeft',bubbles:true}));
 assert(index===0,'Start boundary');
 $('search').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));
 assert(index===0,'Typing preserves position');
 $('next').click();assert(index===1,'Next button');
 $('prev').click();assert(index===0,'Previous button');
 for(const method of ['Vector','HippoRag','Raptor']){
  $('method').value=method;$('method').dispatchEvent(new Event('change'));
  assert(filtered.length===D.rows.filter(r=>r.method===method).length && filtered.length>0,method+' includes all labelled turns');
 }
 $('method').value='Raptor';$('method').dispatchEvent(new Event('change'));
 assert(filtered.length>0,'Raptor visible by default');
 assert(document.querySelectorAll('.annotator').length===D.annotators.length,'All Raptor decision cards visible');
 $('coverage').value='all';$('coverage').dispatchEvent(new Event('change'));
 assert(filtered.length===D.rows.filter(r=>r.method==='Raptor'&&r.labelled_count===D.annotators.length).length,'Strict filter requires all Raptor annotators');
 $('coverage').value='any';$('coverage').dispatchEvent(new Event('change'));
 assert(filtered.length===D.rows.filter(r=>r.method==='Raptor').length,'Partial Raptor queue');
 assert(document.querySelectorAll('.annotator').length===3,'Missing reviewers visible');
 $('search').value='no-match-XYZ';$('search').dispatchEvent(new Event('input'));
 assert(filtered.length===0,'Search');
 assert($('overview-cards').textContent===overviewText,'Queue filters do not change overall totals');
 $('search').value='';$('search').dispatchEvent(new Event('input'));
 $('method').value='';$('method').dispatchEvent(new Event('change'));
 $('coverage').value='all';$('coverage').dispatchEvent(new Event('change'));
 move(99999);assert($('next').disabled,'End boundary');move(-99999);
 const rects=[...document.querySelectorAll('.annotator')].map(c=>c.getBoundingClientRect());
 assert(document.querySelector('.annotators').getBoundingClientRect().bottom<=document.querySelector('.question').getBoundingClientRect().top,'Question follows annotator decisions');
 if(window.innerWidth>650)assert(rects[0].left<rects[1].left && rects[1].left<rects[2].left,'Side by side');
 assert(document.documentElement.scrollWidth<=window.innerWidth,'No horizontal page overflow');
 $('coverage').value='any';$('coverage').dispatchEvent(new Event('change'));
 const hasBothMatches=r=>r.gold_support?.status==='available' && r.retrieval_context?.documents.some(doc=>matchesGold(doc,r.gold_support.documents)) && r.selected_passages.some(doc=>matchesGold(doc,r.gold_support.documents));
 index=filtered.findIndex(hasBothMatches);assert(index>=0,'Real turn has gold matches in retrieved and selected');render();
 const row=filtered[index],goldDocs=row.gold_support.documents,panels=document.querySelectorAll('.evidence-grid>.panel');
 assert([...panels].map(panel=>panel.querySelector('h3').textContent).join(',')==='Selected Context,Retrieved Passages,Gold Documents','Selected context precedes retrieved passages');
 for(const [panelIndex,docs] of [[0,row.selected_passages],[1,row.retrieval_context.documents],[2,goldDocs]]){
  const passages=[...panels[panelIndex].querySelectorAll('.passage')];
  assert(passages.length===docs.length,'All passages rendered');
  for(const [i,passage] of passages.entries()){
   const expected=matchesGold(docs[i],goldDocs);
   const ranges=panelIndex===2?[]:goldTextRanges(passageText(docs[i]),goldDocs);
   assert(passage.classList.contains('gold-match')===expected,'Only gold documents highlighted');
   assert(!!passage.querySelector('.gold-badge')===(expected||ranges.length>0),'Gold badge matches document or excerpt');
   assert(passage.querySelectorAll('mark.gold-text').length===ranges.length,'Inline gold highlights rendered');
   assert(passage.querySelector('.prose').textContent===clean(passageText(docs[i])||'No passage text recorded.'),'Passage text preserved');
  }
 }
 assert(panels[0].querySelector('mark.gold-text') && panels[1].querySelector('mark.gold-text'),'Gold text visibly highlighted in retrieved and selected');
 assert(!document.querySelector('.evidence-grid details'),'Matched passages have no toggles');
 report.dataset.status='pass';report.textContent='PASS';
}catch(error){report.dataset.status='fail';report.textContent=error.stack;}
document.body.append(report);
</script>"""


class ResultReviewTests(unittest.TestCase):
    def test_overview_counts_only_turns_annotated_by_all_three_even_when_labels_differ(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for method, run in review.RUNS.items():
                target = root / method
                target.mkdir()
                records = {'PS': [{'id': 'case', 'turn': 1, 'failure_types': ['E4', 'E9', 'E4']},
                                  {'id': 'case', 'turn': 2, 'notes': 'Draft', 'review_complete': True},
                                  {'id': 'case', 'turn': 3, 'failure_types': ['excluded']}],
                           'KH': [{'id': 'case', 'turn': 1, 'failure_types': ['not_a_failure']},
                                  {'id': 'case', 'turn': 4, 'human_label': 'correct'}],
                           'WN': [{'id': 'case', 'turn': 1, 'failure_type': 'E12'},
                                  {'id': 'case', 'turn': 5, 'flagged_for_review': True}]}
                # SelfRAG has no WN export. It must still appear with zero.
                for name, items in records.items():
                    if method == 'SelfRAG' and name == 'WN':
                        continue
                    (target / f'annotations_{run}_{name}.json').write_text(json.dumps(items))
            data = review.build_review(root, root, lambda _: {
                'rows': [], 'coverage': {'mtrag': {'source': 'out.mtrag.json'}}})
            overview = data['overview']
            self.assertEqual({name: person['total'] for name, person in overview['annotators'].items()},
                             {'PS': 3, 'KH': 3, 'WN': 3})
            for person in overview['annotators'].values():
                self.assertEqual(person['by_method']['SelfRAG'], 0)
            self.assertEqual(overview['total_annotations'], 9)
            self.assertEqual(overview['annotated_turns'], 3)
            self.assertEqual(overview['all_annotators'], 3)
            self.assertEqual(overview['annotators']['PS']['error_turns'], 3)
            self.assertEqual(overview['annotators']['PS']['label_counts']['E4'], 3)
            self.assertEqual(overview['annotators']['PS']['label_counts']['E9'], 3)
            self.assertEqual(overview['annotators']['PS']['outcomes']['excluded'], 0)
            self.assertEqual(overview['annotators']['KH']['error_turns'], 0)
            self.assertEqual(overview['annotators']['KH']['outcomes']['not_a_failure'], 3)
            self.assertEqual(overview['annotators']['WN']['label_counts']['E12'], 3)
            self.assertEqual(review.annotation_snapshot(data)['overview'], overview)

    def test_no_failure_and_excluded_advance_each_annotator_independently(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / 'SelfRAG'
            target.mkdir()
            run = review.RUNS['SelfRAG']
            profiles = {'PS': [(1, 'not_a_failure'), (3, 'excluded')],
                        'KH': [(1, 'E12')], 'WN': [(1, 'not_a_failure')]}
            for name, decisions in profiles.items():
                (target / f'annotations_{run}_{name}.json').write_text(json.dumps([
                    {'id': 'case', 'turn': turn, 'annotator': name, 'failure_types': [code]}
                    for turn, code in decisions]))
            rows = []
            for turn in (4, 3, 2, 1):
                row = dict.fromkeys(('question', 'answer', 'gold', 'gold_support',
                    'retrieval_context', 'selected_passages', 'contexts', 'corpus_collection'))
                row.update(id='case', key=f'case#{turn}', turn=turn, conversation_number='C01',
                           judges={'mtrag': {'label': 'correct' if turn == 2 else 'incorrect'}})
                rows.append(row)
            with mock.patch.dict(review.RUNS, {'SelfRAG': run}, clear=True):
                data = review.build_review(root, root, lambda _: {
                    'rows': rows, 'coverage': {'mtrag': {'source': 'out.mtrag.json'}}})
            self.assertEqual([(r['turn'], r['queue_annotators']) for r in data['rows']],
                             [(1, ['KH']), (3, ['WN']), (4, ['PS'])])
            self.assertEqual(data['rows'][0]['annotations']['PS']['failure_types'], ['not_a_failure'])
            self.assertEqual(data['rows'][2]['labelled_count'], 0)

    def test_selfrag_keeps_all_saved_decisions_regardless_of_judge_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / 'SelfRAG'
            target.mkdir()
            run = review.RUNS['SelfRAG']
            records = [{'id': 'case', 'turn': turn, 'annotator': 'KH',
                        'failure_types': ['E12']} for turn in (1, 2)]
            (target / f'annotations_{run}_KH.json').write_text(json.dumps(records))
            rows = []
            for turn, label in ((1, 'incorrect'), (2, 'correct'), (3, 'incorrect')):
                row = dict.fromkeys(('question', 'answer', 'gold', 'gold_support',
                    'retrieval_context', 'selected_passages', 'contexts', 'corpus_collection'))
                row.update(id='case', key=f'case#{turn}', turn=turn,
                           conversation_number='C01', judges={'mtrag': {'label': label}})
                rows.append(row)
            # An unannotated conversation still contributes its earliest incorrect turn.
            for turn, label in ((1, 'correct'), (3, 'incorrect'), (2, 'incorrect')):
                rows.append(dict(rows[0], id='unannotated', key=f'unannotated#{turn}',
                                 conversation_number='C10', turn=turn,
                                 judges={'mtrag': {'label': label}}))
            with mock.patch.dict(review.RUNS, {'SelfRAG': run}, clear=True):
                data = review.build_review(root, root, lambda _: {
                    'rows': list(reversed(rows)), 'coverage': {'mtrag': {'source': 'out.mtrag.json'}}})
            self.assertEqual([(r['conversation_number'],r['turn']) for r in data['rows']], [('C01',1),('C01',2),('C10',2)])
            self.assertEqual([r['judge_label'] for r in data['rows']], ['incorrect', 'correct', 'incorrect'])
            self.assertEqual([r['first_incorrect'] for r in data['rows']], [True, False, True])
            self.assertEqual(data['rows'][2]['labelled_count'], 0)
            self.assertEqual(data['rows'][1]['annotations']['WN'], {})
            self.assertEqual(review.annotation_snapshot(data)['rows'][1]['judge_label'], 'correct')

    def test_review_mark_is_not_a_decision(self):
        self.assertFalse(review.labelled({'review_complete': True}))
        self.assertFalse(review.labelled({'flagged_for_review': True}))
        self.assertTrue(review.labelled({'failure_types': ['excluded']}))
        self.assertTrue(review.labelled({'human_label': 'correct'}))

    def test_run_mismatch_is_not_attached(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'annotations_wrong_PS.json').write_text('[]')
            (path / 'annotations_right_KH.json').write_text(json.dumps([
                {'id': 'case', 'turn': 2, 'failure_types': ['E9'], 'annotator': 'KH'}]))
            profiles, warnings = review.read_annotations(path, 'right')
            self.assertEqual(len(warnings), 1)
            self.assertEqual(profiles['PS'], {})
            self.assertEqual(profiles['KH']['case#2']['failure_types'], ['E9'])

    def test_safe_embedding(self):
        page = review.render({'question': '</script><script>alert(1)</script>'})
        self.assertNotIn('</script><script>alert(1)', page)
        self.assertIn(r'\u003c/script>', page)

    def test_exported_page_in_browser(self):
        browser = next((p for p in [shutil.which('chromium'),
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe']
            if p and Path(p).is_file()), None)
        if not browser:
            self.skipTest('No browser installed')
        page = (review.ROOT / 'Annotate_result' / 'review.html').read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            for width in (1440, 600):
                path = base / f'check-{width}.html'
                path.write_text(page.replace('</body>', BROWSER_CHECKS + '</body>'), encoding='utf-8')
                result = subprocess.run([browser, '--headless', '--disable-gpu',
                    '--no-first-run', '--no-default-browser-check', '--disable-extensions',
                    '--disable-background-networking', '--disable-sync',
                    '--user-data-dir=' + str(base / f'profile-{width}'),
                    f'--window-size={width},1100', '--timeout=15000',
                    '--virtual-time-budget=3000', '--dump-dom', path.as_uri()],
                    capture_output=True, encoding='utf-8', errors='replace', timeout=40)
                report = result.stdout.split('<pre id="test-result"', 1)[-1].split('</pre>', 1)[0]
                self.assertTrue('data-status="pass"' in report,
                              f'Width {width}: {report[-1500:]} {result.stderr[-500:]}')


if __name__ == '__main__':
    unittest.main()
