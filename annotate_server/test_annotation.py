"""Offline checks for annotation data, safe embedding, and the review workflow.

Run: python -m unittest discover -s annotate_server -p test_annotation.py -v
The browser test uses an isolated temporary headless Chrome/Edge profile when
an installed browser is available; it never uses a personal browser session.
"""

import copy
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

import make_annotation as annotation
from annotation_server import make_handler


def fixture():
    view = annotation.scored_view({
        "label": "incorrect", "reasoning": "Check the direction of rotation.",
        "scores": {"RB_llm": 4, "RL_F": 0, "phi": 0,
                   "conditioned_score": 0, "idk": 0,
                   "answerability": "ANSWERABLE",
                   "statements": [{"statement": "Venus rotates eastward.",
                                   "verdict": "no", "reason": "The passage says westward."}]}})
    rows = []
    for i in range(1, 4):
        rows.append({"key": "case#%s" % i, "id": "case", "turn": i, "n_turns": 3,
                     "question": "Which direction does Venus rotate?" if i < 3 else "What about Earth?",
                     "corpus_collection": "planets",
                     "answer": "Venus rotates eastward.", "gold": "Venus rotates westward.",
                     "answerability": "ANSWERABLE", "contexts": ["Venus rotates westward.", "Earth rotates eastward."],
                     "gold_support": {"status": "available", "task_id": "case<::>%s" % i,
                                      "message": "Verified reference passages.",
                                      "documents": [{"id": "venus", "title": "Venus", "text": "Venus rotates westward.", "relevance": ["relevant"]}]},
                     "judges": {"mtrag": copy.deepcopy(view)}})
    rows[2]["contexts"] = None
    rows[2]["gold_support"] = {"status": "missing", "documents": [], "message": "Reference evidence unavailable."}
    rows[2]["judges"]["mtrag"] = annotation.not_run()
    return {"run": "annotation_ui_test", "rows": rows,
            "corpus": {"planets": {"kind": "conversation_passages", "documents": [
                {"id": "venus-corpus", "title": "Venus", "text": "Venus rotates westward.", "source_file": "conversation.json"},
                {"id": "earth-corpus", "title": "Earth", "text": "Earth rotates eastward.", "source_file": "conversation.json"}
            ], "sources": [{"file": "conversation.json"}], "message": "Recorded passage archive, not the complete original corpus."}},
            **annotation.taxonomy_data(),
            "judge_titles": annotation.JUDGE_TITLES,
            "coverage": {"mtrag": {"turns": 2}}, "criteria_html": "<p>Scoring guide.</p>"}


BROWSER_CHECKS = r"""
<pre id="testResult"></pre>
<script>
(async()=>{
  const result=document.getElementById('testResult');
  let checks=0;
  function assert(ok,message){if(!ok)throw Error(message);checks++;}
  function click(selector){const node=document.querySelector(selector);assert(!!node,'Missing '+selector);node.click();}
  function choose(value){const r=D.rows.find(r=>r.key===activeKey);if(value==='not_a_failure'){setFailureTypes(entry(r),[value]);save();queue();refreshFailureChoices(r);return;}$('clearFailureChoices').click();const stage=guides[value]?.stage;if(D.failure_stages.includes(stage)){document.querySelector('[data-failure-stage="'+stage+'"]').click();assert(!document.querySelector('.failure-stage[data-stage="'+stage+'"]').hidden,'Chosen stage exposes its failure types');}document.querySelector('[data-failure-type="'+value+'"]').click();const note=document.querySelector('[data-help-type="'+value+'"]');assert(!note.hidden&&note.textContent.includes('Compare: '+comparisonSources(value).map(where=>comparisonLabels[where]).join(' | ')),'Selecting a failure shows a note directly below its card');assert(!document.querySelector('dialog[open]'),'Selecting a failure never opens a popup');assert(note.querySelectorAll('button,a,summary,input').length===0,'Comparison hints are plain text without controls');}

  function checkVisible(){assert(document.body.scrollWidth<=document.body.clientWidth,'Horizontal layout overflow');}
  try{
    await Promise.resolve();
    assert(!$('failureChoices').hidden&&!$('failureChoices').disabled,'Failure annotation is available immediately, including previously reviewed turns');
    assert(document.querySelectorAll('.failure-stage:not([hidden]) [data-failure-type]').length===16,'All sixteen paper failure types are shown by default');
    assert(!$('optionalCorrectness').open,'Correctness check is optional and collapsed');
    assert($('failureChoices').querySelectorAll('.failure-inline-help:not([hidden])').length===16,'All sixteen plain comparison hints are visible before selection');
    const originalHint=document.querySelector('[data-help-type="E13"]').textContent;document.querySelector('[data-failure-type="E13"]').click();document.querySelector('[data-failure-type="E13"]').click();assert(document.querySelector('[data-help-type="E13"]').textContent===originalHint&&!document.querySelector('[data-help-type="E13"]').hidden,'Selecting and clearing a failure does not toggle its hint');
    const evidenceRow={contexts:['Venus rotates westward.'],gold_support:{status:'available',documents:[{id:'gold-1',text:'Venus rotates westward.'}]}};
    const evidenceFacts=passageFacts(evidenceRow,{id:'gold-1',text:'Venus rotates westward.'});
    assert(evidenceFacts.used&&evidenceFacts.goldDocument&&evidenceFacts.goldRanges.length,'Exact selected gold passage has all three evidence badges');
    const containing=passageFacts(evidenceRow,{id:'other',text:'An introduction. Venus   rotates westward. More text.'});
    assert(!containing.used&&!containing.goldDocument&&containing.goldRanges.length,'Containing an excerpt does not imply the same gold document or generation context');
    const wrong=passageFacts(evidenceRow,{id:'other',text:'Venus rotates eastward.'});
    assert(!wrong.goldRanges.length&&!wrong.used&&!wrong.goldDocument,'Similar words with different evidence do not create gold badges');
    assert(!goldExcerptRanges({...evidenceRow,gold_support:{...evidenceRow.gold_support,status:'mismatch'}},'Venus rotates westward.').length,'Mismatched reference tasks cannot highlight gold text');
    const badgeCard=passage(evidenceRow,'Selected context','selected:1','Chunk','Venus rotates westward.','','',{id:'gold-1',used:true});
    assert(badgeCard.querySelectorAll('.evidence-badge').length===3&&badgeCard.querySelector('mark.gold-excerpt'),'Badges and exact gold highlight are rendered');
    assert(!badgeCard.textContent.includes('Retrieved'),'Final context alone never creates a Retrieved badge');
    assert(badgeCard.querySelector('.prose').textContent==='Venus rotates westward.','Highlighting preserves original evidence text for citations');
    assert(A['case#1'].human_label==='correct','Legacy binary label migration');
    assert(!A['case#1'].human_score_band,'Migrated binary field removed');
    assert(A['case#2'].human_score_band==='mid','Historical band preserved');
    assert($('queue').querySelectorAll('.queue-item').length===3,'Full queue visible');
    assert($('queue').querySelectorAll('.queue-group').length===1,'Turns grouped by conversation ID');
    assert($('queue').querySelector('.case-id').textContent==='case','Conversation ID appears once per group');
    const other={...D.rows[2],key:'other#1',id:'other',turn:1,n_turns:1};D.rows.push(other);queue();
    assert($('queue').querySelectorAll('.queue-group').length===2,'Separate IDs create separate groups');D.rows.pop();queue();
    assert($('review').querySelectorAll('#answer,#reference').length===2,'Answer comparison');
    assert($('review').querySelectorAll('.evidence-column').length===3,'Three simultaneous evidence columns');
    assert($('queue').querySelector('[data-judge=incorrect] .turn-status').textContent.includes('Incorrect'),'mtRAG turn status');
    assert(!$('judgeDisclosure').open,'Judge assessment starts collapsed');
    assert($('judgeDisclosure').querySelector('.judge-content').textContent.includes('0.8'),'mtRAG cutoff retained inside disclosure');
    assert(!document.querySelector('#question .judges'),'Question does not repeat judge results');
    assert($('progress').textContent==='1 / 3','Initial reviewed progress');
    checkVisible();

    const queueNode=$('queue').querySelector('.queue-item'),groupNode=$('queue').querySelector('.queue-group');
    openTurn('case#2');await Promise.resolve();
    assert($('queue').querySelector('.queue-item')===queueNode&&$('queue').querySelector('.queue-group')===groupNode,'Turn navigation reuses queue nodes');
    assert($('queue').querySelectorAll('[aria-current=true]').length===1&&$('queue').querySelector('[aria-current=true]').dataset.key==='case#2','Turn navigation updates the selected queue item');
    click('input[value="incorrect"]');await Promise.resolve();
    assert(!reviewed(D.rows[1]),'Incorrect needs a cause');
    for(const type of D.failure_types){choose(type);assert(document.querySelector('[data-help-type="'+type+'"]').textContent.includes('Compare: '+comparisonSources(type).map(where=>comparisonLabels[where]).join(' | ')),'Guide '+type);}
    choose('E4');
    assert(document.querySelector('[data-help-type="E4"]').textContent.includes('Retrieved context not recorded.'),'Retrieval evidence limitation');
    assert(document.querySelector('[data-help-type="E4"] .comparison-note').textContent==='Compare: Question | Gold | Retrieved context | Selected context | Corpus','Retrieval hints distinguish the original retrieved pool from final selection');
    const rerankHint=document.querySelector('[data-help-type="E7"]').textContent;assert(rerankHint.includes('Retrieved context')&&rerankHint.includes('Selected context'),'Reranking hints include both before and after selection');
    assert(!document.querySelector('[data-help-type="E13"] .comparison-note').textContent.includes('Retrieved context'),'Generation-only comparisons do not imply original retrieval evidence exists');
    const e4Hint=document.querySelector('[data-help-type="E4"]');assert(!e4Hint.hidden&&!e4Hint.querySelector('button,a,summary'),'Comparison hint is permanently visible plain text');
    assert(reviewed(D.rows[1]),'Incorrect and cause reviewed');
    choose('not_a_failure');
    assert(reviewed(D.rows[1]),'No failure can be annotated independently of the optional correctness check');
    assert($('failureGuide').textContent.includes('judge result is unchanged'),'No failure does not rewrite the saved judge result');
    choose('E10');
    assert(document.querySelectorAll('.failure-choice-grid input[type="checkbox"]').length===16,'Sixteen visible checkbox choices replace the dropdown');
    click('[data-failure-type="E4"]');click('[data-failure-type="E12"]');
    assert(A['case#2'].failure_types.length===3&&A['case#2'].failure_type==='','Several causes selected without an implicit primary');
    assert($('failureCount').textContent==='3 selected','Selected cause count');
    assert($('failureChoices').querySelectorAll('.failure-help:not([hidden])').length===16,'Evidence hints stay visible for all sixteen failure types');
    click('[data-failure-type="E4"]');assert(A['case#2'].failure_types.length===2&&!getFailureTypes(A['case#2']).includes('E4'),'Individual causes can be deselected');
    click('[data-failure-type="insufficient_evidence"]');assert(JSON.stringify(getFailureTypes(A['case#2']))==='["insufficient_evidence"]','Unresolved outcome replaces confirmed causes');
    click('[data-failure-type="E10"]');assert(JSON.stringify(getFailureTypes(A['case#2']))==='["E10"]','A confirmed cause replaces an alternative outcome');
    click('#clearFailureChoices');assert(!reviewed(D.rows[1])&&!getFailureTypes(A['case#2']).length,'Clear choices resets completion');
    click('[data-failure-type="E10"]');click('[data-failure-type="E4"]');click('[data-failure-type="E12"]');
    $('notes').value='Wrong direction; the selected passage says westward.';
    $('notes').dispatchEvent(new Event('input'));

    const text=$('answer').querySelector('.prose');const range=document.createRange();
    const firstText=document.createTreeWalker(text,NodeFilter.SHOW_TEXT).nextNode();range.setStart(firstText,0);range.setEnd(firstText,5);
    const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);
    cite(D.rows[1],'RAG answer','answer',D.rows[1].answer,text);
    assert(A['case#2'].evidence[0].quote==='Venus','Exact selected quote retained');
    selection.removeAllRanges();
    click('[data-tab="compare"]');
    click('#evidence-selected .passage button');
    assert(A['case#2'].evidence[1].quote==='Venus rotates westward.','Whole passage citation');
    click('#evidence-selected .passage button');
    assert(A['case#2'].evidence.length===2,'Duplicate citation avoided');
    assert($('citations').children.length===2,'Citations shown');
    $('evidenceSearch').value='westward';$('evidenceSearch').dispatchEvent(new Event('input'));
    assert($('evidence-selected').querySelectorAll('.passage').length===1,'Evidence search filters selected passages');
    assert($('evidence-selected').querySelector('mark.search-match').textContent==='westward','Evidence search match highlighted');
    $('evidenceSearch').value='';$('evidenceSearch').dispatchEvent(new Event('input'));
    assert($('evidence-gold').textContent.includes('Dataset relevance: relevant'),'Reference relevance visible');
    assert($('evidence-corpus').textContent.includes('not the complete original corpus'),'Archive limitations visible');
    assert(!$('corpusDisclosure').open,'Corpus starts collapsed');
    click('#corpusDisclosure > summary');
    assert($('corpusDisclosure').open,'Corpus opens with its search controls');
    assert(!$('retrievedDisclosure').open,'Retrieved toggle starts closed');
    const row=D.rows[1];
    row.retrieval_context={documents:[{id:'retrieved-venus',title:'Venus',text:'Venus rotates westward.',retrieval_rank:1},{id:'retrieved-earth',title:'Earth',text:'Earth rotates eastward.',retrieval_rank:2}],source:{file:'test/candidates.jsonl',inference_stage:'new_retrieval_experiment'}};
    renderEvidence(row);click('#retrievedDisclosure > summary');
    assert($('retrievedDisclosure').open,'Retrieved context opens independently');
    assert($('retrievedDisclosure').querySelectorAll('.passage').length===2,'Retrieved chunks shown as separate cards');
    assert(!$('retrievedDisclosure').querySelector('.evidence-badge.used'),'Candidate matching selected text does not claim original generator use');
    assert($('retrievedDisclosure').textContent.includes('Before rerank'),'Retrieved context identifies its pipeline stage');
    const hint=document.createElement('div');renderInlineComparison(row,'E4',hint);
    assert(!hint.textContent.includes('not recorded'),'Available retrieval is reflected in failure hints');
    delete row.retrieval_context;renderEvidence(row);
    click('#corpusDisclosure > summary');
    if(!$('corpusDisclosure').open)click('#corpusDisclosure > summary');

    click('#corpusDisclosure > summary');
    assert(!$('corpusDisclosure').open,'Corpus closes independently from Gold');
    click('#corpusDisclosure > summary');
    $('corpusSearch').value='earth-corpus';$('corpusSearch').dispatchEvent(new Event('input'));
    assert($('evidence-corpus').textContent.includes('Earth rotates eastward.'),'Corpus search by original document ID');
    assert(!$('evidence-corpus').querySelector('.column-body').textContent.includes('Venus rotates westward.'),'Corpus search filters evidence');
    click('#evidence-corpus .passage button');
    assert(A['case#2'].evidence[2].source==='Conversation passage archive','Corpus citation retains archive provenance');
    document.querySelectorAll('#citations .citation button')[2].click();
    $('goldAssessment').value='incorrect';$('goldAssessment').dispatchEvent(new Event('change'));
    $('goldNotes').value='The corpus evidence contradicts the gold answer.';$('goldNotes').dispatchEvent(new Event('input'));
    assert(A['case#2'].gold_assessment==='incorrect'&&A['case#2'].human_label==='incorrect','Gold review is independent');
    click('[data-tab="history"]');assert($('evidenceBody').textContent.includes('Turn 1 · User'),'Earlier conversation visible');
    click('#judgeDisclosure>summary');assert($('judgeDisclosure').open&&$('judgeDisclosure').textContent.includes('The passage says westward.'),'Judge toggle reveals statement reasoning');

    assert(!$('blind'),'Hide judge assessment removed');
    assert(!$('coverage')&&!document.querySelector('.queue a[href="#guidelines"]'),'Sidebar footer removed');
    click('#judgeGuideButton');assert($('judgeGuidelines').open&&$('judgeGuidelines').textContent.includes('0.8'),'Judge guideline opens from top left');$('judgeGuidelines').close();
    click('#failureGuideButton');assert($('failureGuidelines').open&&$('failureGuideContent').children.length===16&&$('failureGuideContent').textContent.includes(D.failure_annotation_aids.E13.example),'Top-left guideline popup contains all sixteen types and detailed examples');$('failureGuidelines').close();
    if(window.innerWidth>950){const form=$('annotationPanel').getBoundingClientRect(),reading=document.querySelector('.reading').getBoundingClientRect();assert(form.left>=reading.right,'Annotation panel is on the right');assert(getComputedStyle($('annotationPanel')).position==='sticky','Right annotation bar stays visible');}
    click('[data-tab="compare"]');
    assert($('notes').value.includes('Wrong direction'),'Notes survive re-render');
    assert(JSON.parse(localStorage.getItem(KEY))['case#2'].evidence.length===2,'Citations persisted');

    let exported;
    download=(name,text,type)=>exported={name,text,type};
    click('#expJson');const out=JSON.parse(exported.text);
    const second=out.find(o=>o.turn===2);
    assert(second.evidence[0].quote==='Venus'&&second.notes.includes('Wrong direction'),'JSON evidence export');
    assert(second.human_score_band==='mid','Historical band export');
    assert(second.failure_types.length===3&&second.failure_types.includes('E10')&&second.failure_types.includes('E4')&&second.failure_types.includes('E12')&&second.failure_type==='','JSON exports all causes without a primary label');
    assert(exported.name.endsWith('_PS.json')&&out.every(o=>o.annotator==='PS'),'Exports identify PS');
    assert(second.gold_assessment==='incorrect'&&second.gold_notes.includes('contradicts'),'Gold review exported');
    click('#expCsv');assert(exported.text.includes('evidence')&&exported.text.includes('westward'),'CSV evidence export');
    assert(exported.text.includes('failure_types')&&exported.text.includes('""E4""')&&exported.text.includes('""E12""'),'CSV exports multiple causes as an array');
    await $('imp').onchange({target:{files:[new File([JSON.stringify(out)],'roundtrip.json')],value:''}});
    await Promise.resolve();assert(A['case#2'].evidence.length===2,'JSON import roundtrip');
    assert(A['case#2'].failure_types.length===3&&A['case#2'].failure_type==='','Multiple causes survive JSON import');
    assert(A['case#2'].gold_assessment==='incorrect'&&A['case#2'].gold_notes===second.gold_notes,'Gold review import roundtrip');
    await $('imp').onchange({target:{files:[new File(['not json'],'bad.json')],value:''}});
    assert(A['case#2'].notes===second.notes,'Malformed import preserves data');
    await $('imp').onchange({target:{files:[new File([JSON.stringify([{id:'case',turn:3,human_score_band:'incorrect',notes:'Legacy import'}])],'old.json')],value:''}});
    assert(A['case#3'].human_label==='incorrect','Older JSON imports');

    $('fTodo').checked=true;$('fTodo').dispatchEvent(new Event('change'));await Promise.resolve();
    assert(activeKey==='case#3'&&$('queue').querySelectorAll('.queue-item').length===1,'Unreviewed filter shows remaining cause');
    assert($('evidenceBody').textContent.includes('not recorded'),'Missing context distinguished');
    choose('E1');assert($('failureGuide').textContent.includes('Missing for this turn: selected context'),'Missing evidence guidance');
    $('fTodo').checked=false;$('fTodo').dispatchEvent(new Event('change'));
    $('fFail').checked=true;$('fFail').dispatchEvent(new Event('change'));
    assert($('queue').querySelectorAll('.queue-item').length===2,'Incorrect judge filter');
    $('search').value='not present';$('search').dispatchEvent(new Event('input'));
    assert($('review').textContent.includes('No turns match'),'Empty filter state');
    $('search').value='';$('search').dispatchEvent(new Event('input'));
    $('fFail').checked=false;$('fFail').dispatchEvent(new Event('change'));
    openTurn('case#2');await Promise.resolve();click('input[value="correct"]');await Promise.resolve();
    assert(A['case#2'].failure_type==='not_a_failure'&&!$('failureChoices').disabled,'Optional Correct review still leaves failure annotation available');
    assert(A['case#2'].notes===second.notes&&A['case#2'].evidence.length===2,'Decision changes preserve notes and evidence');
    delete A['case#2'].failure_types;A['case#2'].failure_type='E4';renderReview();await Promise.resolve();
    assert(document.querySelector('[data-help-type="E4"]').textContent.includes('Compare:'),'Earlier optional correctness labels never hide failure guidance');
    choose('not_a_failure');await Promise.resolve();
    assert(A['case#2'].failure_type==='not_a_failure','Older conflicting label can be corrected');
    document.querySelector('.work-heading .eyebrow').textContent='1534A095279F2CB888FB0BEA17BD70DA';
    document.querySelector('.case-id').textContent='1534A095279F2CB888FB0BEA17BD70DA';
    await loadCorpusFile(D.rows[1],new File(['{"document_id":"local-1","text":"Venus rotates westward."}\n{"document_id":"local-2","text":"Earth rotates eastward."}'],'corpus.jsonl'));
    assert(corpusFor(D.rows[1]).documents.length===2&&corpusFor(D.rows[1]).kind==='local_file','Local JSONL corpus loaded');
    await loadCorpusFile(D.rows[1],new File(['{"documents":[{"id":"single","contents":"Original source text."}]}'],'corpus.json'));
    assert(corpusFor(D.rows[1]).documents[0].id==='single','JSON corpus wrapper accepted');
    await loadCorpusFile(D.rows[1],new File(['{"documents":[{"id":"invalid"}]}'],'bad-corpus.json'));
    assert(corpusFor(D.rows[1]).documents[0].id==='single','Invalid corpus import preserves prior corpus');
    const ps=JSON.parse(JSON.stringify(A)),selectedTurn=activeKey;
    await switchAnnotator('KH');await Promise.resolve();
    assert(activeKey===selectedTurn,'Switching annotator keeps the selected conversation and turn');
    assert(!hasAnnotation(A['case#2']),'KH starts with separate data for the same turn');
    click('input[value="incorrect"]');await Promise.resolve();choose('E4');$('notes').value='KH independent notes';$('notes').dispatchEvent(new Event('input'));
    assert($('progress').textContent==='1 / 3','KH has independent progress');
    await switchAnnotator('WN');await Promise.resolve();
    assert(!hasAnnotation(A['case#2'])&&!hasAnnotation(A['case#3']),'WN starts separately; shared legacy work is not assigned automatically');
    copyLegacyAnnotations();await Promise.resolve();
    assert(A['case#3'].notes==='Earlier shared work','Earlier shared work can be assigned explicitly');
    assert(JSON.parse(localStorage.getItem(BASE_KEY))['case#3'].notes==='Earlier shared work','Original shared data retained');
    click('input[value="correct"]');await Promise.resolve();$('notes').value='WN independent notes';$('notes').dispatchEvent(new Event('input'));
    await switchAnnotator('PS');await Promise.resolve();
    assert(A['case#2'].notes===ps['case#2'].notes&&A['case#2'].evidence.length===2&&A['case#2'].gold_notes===ps['case#2'].gold_notes,'PS labels, notes, evidence, and gold review restored');
    assert(JSON.parse(localStorage.getItem(BASE_KEY+':KH'))['case#2'].notes==='KH independent notes','KH browser storage is separate');
    assert(JSON.parse(localStorage.getItem(BASE_KEY+':WN'))['case#2'].notes==='WN independent notes','WN browser storage is separate');
    const mixed=[{id:'case',turn:2,annotator:'KH',notes:'KH imported',human_label:'correct'},{id:'case',turn:2,annotator:'WN',notes:'WN imported',human_label:'incorrect',failure_type:'grounding'},{id:'case',turn:2,annotator:'UNKNOWN',notes:'Do not assign'}];
    await $('imp').onchange({target:{files:[new File([JSON.stringify(mixed)],'mixed.json')],value:''}});await Promise.resolve();
    assert(A['case#2'].notes===ps['case#2'].notes,'Tagged import does not overwrite selected PS work');
    assert(getProfile('KH').annotations['case#2'].notes==='KH imported'&&getProfile('WN').annotations['case#2'].notes==='WN imported','Tagged imports route to their own profiles');
    assert(JSON.stringify(getProfile('WN').annotations['case#2'].legacy_failure_types)==='["grounding"]'&&!getProfile('WN').annotations['case#2'].failure_types.length,'Older single-cause imports are preserved without guessing an E1–E16 label');
    profiles.delete('KH');await switchAnnotator('KH');await Promise.resolve();
    assert(A['case#2'].notes==='KH imported','Profile reload restores persisted KH data');
    assert(rowsOut().every(o=>o.annotator==='KH'),'KH export only identifies KH');
    await switchAnnotator('PS');await Promise.resolve();
    checkVisible();
    result.textContent='PASS '+checks+' checks';result.dataset.status='pass';
  }catch(error){result.textContent='FAIL '+error.stack;result.dataset.status='fail';}
})();
</script>
"""

LIVE_PROFILE_CHECKS = r"""
<pre id="testResult"></pre><script>
(async()=>{
  const result=document.getElementById('testResult');let checks=0;
  function assert(ok,message){if(!ok)throw Error(message);checks++;}
  function label(value){document.querySelector('input[value="'+value+'"]').click();}
  function note(text){$('notes').value=text;$('notes').dispatchEvent(new Event('input'));}
  async function saved(owner){return (await (await fetch(D.annotation_api+'/'+owner)).json()).annotations;}
  try{
    await annotationReady;await Promise.resolve();openTurn('case#2');await Promise.resolve();
    const originalJudge=JSON.stringify(D.rows[1].judges);
    assert(!A['case#2'].human_label&&!$('failureChoices').hidden,'A new turn starts with failure types ready and no correctness decision');
    document.querySelector('[data-failure-type="E10"]').click();
    assert(!A['case#2'].human_label&&!finishProblem(D.rows[1]),'A failure can be completed without an inferred correctness label');
    await finishAndNext(D.rows[1]);
    assert((await saved('PS'))['case#2'].review_complete===true&&!(await saved('PS'))['case#2'].human_label,'Save and next persists a failure-only review');
    assert(JSON.stringify(D.rows[1].judges)===originalJudge,'Failure-only review never changes the saved judge assessment');
    openTurn('case#2');await Promise.resolve();document.querySelector('[data-failure-type="E4"]').click();note('PS server note');await flushProfile('PS');
    assert((await saved('PS'))['case#2'].notes==='PS server note','PS saved to server');
    assert((await saved('PS'))['case#2'].failure_types.length===2&&(await saved('PS'))['case#2'].failure_type==='','Multiple failure causes saved to server');
    await switchAnnotator('KH');await Promise.resolve();
    assert(activeKey==='case#2'&&!hasAnnotation(A['case#2']),'KH opens same turn with empty independent record');
    label('correct');await Promise.resolve();note('KH server note');await flushProfile('KH');
    await switchAnnotator('WN');await Promise.resolve();assert(!hasAnnotation(A['case#2']),'WN has independent record');
    label('correct');await Promise.resolve();note('WN server note');await flushProfile('WN');
    await switchAnnotator('PS');await Promise.resolve();assert(A['case#2'].notes==='PS server note'&&!A['case#2'].human_label,'Failure-only PS review restored from server without adding a correctness decision');
    assert((await saved('KH'))['case#2'].notes==='KH server note'&&(await saved('WN'))['case#2'].notes==='WN server note','Other annotator files preserved');
    await switchAnnotator('KH');profiles.delete('PS');localStorage.removeItem(profileKey('PS'));localStorage.removeItem(profileKey('PS')+':pending');
    await switchAnnotator('PS');await Promise.resolve();assert(A['case#2'].notes==='PS server note','Server restores work after browser cache removal');
    assert(A['case#2'].failure_types.length===2&&document.querySelector('[data-failure-type="E10"]').checked&&document.querySelector('[data-failure-type="E4"]').checked,'Multiple checked causes restored from server');
    const realFetch=globalThis.fetch;globalThis.fetch=(url,options)=>options?.method==='POST'?Promise.reject(Error('Offline test')):realFetch(url,options);
    note('PS offline draft');await flushProfile('PS');
    assert(getProfile('PS').dirty.has('case#2'),'Failed server save remains pending');
    assert(JSON.parse(localStorage.getItem(profileKey('PS')))['case#2'].notes==='PS offline draft','Offline backup retained in PS storage');
    globalThis.fetch=realFetch;
    await switchAnnotator('KH');await switchAnnotator('PS');await Promise.resolve();await flushProfile('PS');
    assert((await saved('PS'))['case#2'].notes==='PS offline draft','Offline draft synchronizes when connection returns');
    assert((await saved('PS'))['case#2'].failure_types.length===2,'Offline synchronization preserves every cause');
    assert((await saved('KH'))['case#2'].notes==='KH server note','Retry does not overwrite KH');
    assert($('saveStatus').textContent.includes('Saved to server')&&$('saveStatus').textContent.includes('PS'),'Save status identifies server and owner');
    result.textContent='PASS '+checks+' live profile checks';result.dataset.status='pass';
  }catch(error){result.textContent='FAIL '+error.stack;result.dataset.status='fail';}
})();
</script>
"""


class AnnotationTests(unittest.TestCase):
    def test_static_cli_from_another_directory_loads_shared_gold_and_preserves_files(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        index = json.loads((root / 'judge_run' / 'gold_support.json').read_text(encoding='utf-8'))
        task_id, task = next((key, value) for key, value in index['tasks'].items()
                             if value.get('gold_answers') and value.get('documents'))
        conversation_id, turn_id = task_id.rsplit('<::>', 1)
        with tempfile.TemporaryDirectory() as folder:
            work = pathlib.Path(folder)
            input_path = work / 'input.json'
            original = json.dumps([{
                'id': conversation_id,
                'conversation': [{'role': 'user', 'content': task['question']},
                                 {'role': 'assistant', 'content': 'Recorded RAG answer'}],
                'ground_truth_answers': [task['gold_answers'][0]],
                'metadata': {'turn_ids': [turn_id]},
            }])
            input_path.write_text(original, encoding='utf-8')
            command = [sys.executable, '-B', str(root / 'annotate_server' / 'make_annotation.py'),
                       '--work-dir', str(work)]
            result = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = work / 'annotate.html'
            page = output.read_text(encoding='utf-8')
            payload = page.split('<script id="data" type="application/json">', 1)[1].split('</script>', 1)[0]
            evidence = json.loads(payload)['rows'][0]['gold_support']
            self.assertEqual(evidence['status'], 'available')
            self.assertEqual(evidence['documents'], task['documents'])
            saved_page = output.read_bytes()
            repeated = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn('refusing to overwrite', repeated.stderr)
            self.assertEqual(output.read_bytes(), saved_page)
            self.assertEqual(input_path.read_text(encoding='utf-8'), original)

    def test_browser_server_profile_roundtrip(self):
        candidates = [shutil.which('google-chrome'), shutil.which('chromium'),
                      r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                      r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe']
        browser = next((p for p in candidates if p and pathlib.Path(p).is_file()), None)
        if not browser:
            self.skipTest('No headless Chrome/Edge installed')
        with tempfile.TemporaryDirectory() as folder:
            base = pathlib.Path(folder)
            work = base / 'annotation_ui_test'
            work.mkdir()
            messages = []
            for turn in range(3):
                messages.extend([{'role': 'user', 'content': 'Question %d' % turn},
                                 {'role': 'assistant', 'content': 'Answer %d' % turn}])
            (work / 'input.json').write_text(json.dumps([{'id': 'case', 'conversation': messages}]), encoding='utf-8')
            data = fixture()
            data['annotation_api'] = '/api/annotations/annotation_ui_test'
            page = annotation.render_page(data).replace('</body>', LIVE_PROFILE_CHECKS + '</body>')
            parent = make_handler(base)

            class Harness(parent):
                def do_GET(self):
                    if self.path == '/workflow':
                        self.respond(200, page)
                    else:
                        super().do_GET()

                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(('127.0.0.1', 0), Harness)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                completed = subprocess.run([browser, '--headless', '--disable-gpu', '--no-first-run',
                                            '--no-default-browser-check', '--disable-background-networking',
                                            '--disable-extensions', '--disable-sync',
                                            '--user-data-dir=' + str(base / 'browser-profile'),
                                            '--window-size=1440,1100', '--virtual-time-budget=15000',
                                            '--dump-dom', 'http://127.0.0.1:%d/workflow' % server.server_port],
                                           capture_output=True, encoding='utf-8', errors='replace', timeout=35)
                result = completed.stdout.split('<pre id="testResult"', 1)[-1].split('</pre>', 1)[0]
                self.assertIn('data-status="pass"', result, result[-2500:] + completed.stderr[-1000:])
                for owner in ('PS', 'KH', 'WN'):
                    saved = json.loads((work / 'annotations' / (owner + '.json')).read_text(encoding='utf-8'))
                    self.assertEqual(saved['annotator'], owner)
                self.assertIn('offline draft', json.loads((work / 'annotations/PS.json').read_text())['annotations']['case#2']['notes'])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_screening_boundary_and_invalid_scores(self):
        for score, label in [(0, 'incorrect'), (.799999, 'incorrect'), (.8, 'correct'),
                             (.800001, 'correct'), (1, 'correct'), (None, None),
                             ('0.9', None), (True, None), (float('nan'), None), (float('inf'), None)]:
            with self.subTest(score=score):
                turn = {'label': 'incorrect', 'scores': {'conditioned_score': score}}
                view = annotation.scored_view(turn)
                self.assertEqual(view['label'], label)
                self.assertEqual(view['saved_label'], 'incorrect')
                if label is not None:
                    self.assertIn(str(score), view['screening_reason'])
                json.dumps(view, allow_nan=False)

    def test_fixed_cutoff_does_not_rewrite_saved_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder)
            record = {'id': 'case', 'conversation': [{'role': 'user', 'content': 'Question?'},
                                                     {'role': 'assistant', 'content': 'Answer.'}]}
            (path / 'input.json').write_text(json.dumps([record]), encoding='utf-8')
            saved = [{'id': 'case', 'judge': {'thresholds': {'mtrag': .95}}, 'evaluation': {
                'turn_evaluations': [{'turn': 1, 'label': 'incorrect', 'scores': {'conditioned_score': .8}}]}}]
            output = path / 'out.mtrag.json'
            output.write_text(json.dumps(saved), encoding='utf-8')
            before = output.read_bytes()
            data = annotation.build_data(path)
            self.assertEqual(data['thresholds']['mtrag'], .8)
            self.assertEqual(data['rows'][0]['judges']['mtrag']['label'], 'correct')
            self.assertEqual(output.read_bytes(), before)

    def test_conditioning_explanation(self):
        view = annotation.scored_view({'label': 'incorrect', 'scores': {
            'conditioned_score': 0, 'answerability': 'UNANSWERABLE', 'idk': 0}})
        self.assertIn('requires a full insufficient-information response', view['screening_reason'])
        view = annotation.scored_view({'label': 'incorrect', 'scores': {
            'conditioned_score': 0, 'answerability': 'ANSWERABLE', 'idk': 1}})
        self.assertIn('fully abstains', view['screening_reason'])

    def test_passage_archive_separates_collections_and_preserves_variants(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / 'conversation.json'
            one = {'id': 'same-id', 'title': 'Source', 'text': 'Version one.'}
            two = {**one, 'text': 'Version two.'}
            export = {'conversations': [
                {'domain': 'one', 'turns': [{'task_id': 'task1', 'passages': [one, one, two]}]},
                {'domain': 'two', 'turns': [{'task_id': 'task2', 'passages': [one]}]}]}
            path.write_text(json.dumps(export), encoding='utf-8')
            store = annotation.load_passage_archive(folder)
            self.assertEqual(len(store['one']['documents']), 2)
            self.assertEqual(len(store['two']['documents']), 1)
            self.assertEqual(store['one']['kind'], 'conversation_passages')
            self.assertIn('not the complete original corpus', store['one']['message'])
            self.assertEqual(store['one']['documents'][0]['observed_task'], 'task1')
            self.assertEqual(len(store['one']['sources'][0]['sha256']), 64)
            export['conversations'][0]['turns'][0]['passages'].append({**one, 'id': 'new', 'text': 'New passage.'})
            path.write_text(json.dumps(export), encoding='utf-8')
            self.assertEqual(len(annotation.load_passage_archive(folder)['one']['documents']), 3)

    def test_structured_statement_reasons(self):
        view = fixture()["rows"][0]["judges"]["mtrag"]
        self.assertEqual(view["statements"][0]["reason"], "The passage says westward.")
        self.assertEqual(view["label"], "incorrect")
        self.assertEqual(view["conditioned_score"], 0)

    def test_safe_json_embedding(self):
        data = fixture()
        data["rows"][0]["answer"] = '</script><script>alert("bad")</script>\u2028&'
        page = annotation.render_page(data)
        payload = page.split('<script id="data" type="application/json">', 1)[1].split('</script>', 1)[0]
        self.assertNotIn('<script>', payload)
        self.assertEqual(json.loads(payload), data)

    def test_build_with_missing_evidence_and_no_judge(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder)
            record = {"id": "missing", "conversation": [{"role": "user", "content": "Question?"},
                                                         {"role": "assistant", "content": "Answer."}]}
            (path / 'input.json').write_text(json.dumps([record]), encoding='utf-8')
            row = annotation.build_data(path)["rows"][0]
            self.assertIsNone(row["contexts"])
            self.assertEqual(row["gold_support"]["status"], "missing")
            self.assertEqual(row["judges"]["mtrag"]["status"], "not_run")

    def test_browser_workflow(self):
        candidates = [shutil.which('google-chrome'), shutil.which('chromium'),
                      r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                      r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe']
        browser = next((p for p in candidates if p and pathlib.Path(p).is_file()), None)
        if not browser:
            self.skipTest('No headless Chrome/Edge installed')
        with tempfile.TemporaryDirectory() as folder:
            base = pathlib.Path(folder)
            for width in (1440, 390):
                page = annotation.render_page(fixture())
                seed = '<script>localStorage.setItem("annot:annotation_ui_test:PS", JSON.stringify({"case#1":{"human_score_band":"correct"},"case#2":{"human_score_band":"mid"}}));localStorage.setItem("annot:annotation_ui_test",JSON.stringify({"case#3":{"human_label":"incorrect","notes":"Earlier shared work"}}));</script>'
                # Windows Chrome can enforce a 500px minimum layout viewport even
                # for a 390px headless screenshot. Constrain the root to exercise
                # an actual 390px content width under the same mobile CSS rules.
                if width == 390:
                    seed += '<style>html{width:390px}</style>'
                page = page.replace('<script>\n\'use strict\';', seed + '<script>\n\'use strict\';', 1)
                path = base / ('test-%s.html' % width)
                path.write_text(page.replace('</body>', BROWSER_CHECKS + '</body>'), encoding='utf-8')
                completed = subprocess.run([browser, '--headless', '--disable-gpu', '--no-first-run',
                                            '--no-default-browser-check', '--disable-background-networking',
                                            '--disable-extensions', '--disable-sync',
                                            '--user-data-dir=' + str(base / ('profile-%s' % width)),
                                            '--window-size=%s,1100' % width,
                                            '--timeout=15000', '--virtual-time-budget=5000', '--dump-dom', path.as_uri()],
                                           capture_output=True, encoding='utf-8', errors='replace', timeout=35)
                result = completed.stdout.split('<pre id="testResult"', 1)[-1].split('</pre>', 1)[0]
                self.assertIn('data-status="pass"', result, 'Width %s: %s\n%s' % (width, result[-2500:], completed.stderr[-1000:]))


class GenerationEvidenceTests(unittest.TestCase):
    def test_selected_passage_ids_and_boolean_stay_distinct_from_retrieval_pool(self):
        turn = {'question': 'Question?', 'answer': 'Answer.', 'retrieved': True,
                'passages': [{'id': 'doc-1', 'title': 'Title', 'text': 'Evidence.'}]}
        traces = {('hipporag2', 'case', '1'): [turn]}
        result = annotation.generation_evidence(traces, 'hipporag2', 'case', '1',
                                     'Question?', 'Answer.', ['Evidence.'])
        self.assertEqual(result['selected_passages'][0]['id'], 'doc-1')
        self.assertIs(result['retrieved_flag'], True)
        self.assertIsNone(result['retrieved_candidates'])
        self.assertIsNone(result['reranking_trace'])

    def test_changed_source_cannot_attach_stale_ids_or_retrieval_flags(self):
        traces = {('hipporag2', 'case', '1'): [{'question': 'Question?', 'answer': 'Old answer.',
                   'retrieved': True, 'passages': [{'id': 'old', 'text': 'Evidence.'}]}]}
        result = annotation.generation_evidence(traces, 'hipporag2', 'case', '1',
                                     'Question?', 'New answer.', ['Evidence.'])
        self.assertEqual(result['selected_passages'][0]['id'], '')
        self.assertIsNone(result['retrieved_flag'])

    def test_no_source_keeps_generation_text_without_inventing_ids(self):
        result = annotation.generation_evidence({}, 'other', 'case', '1', 'Q', 'A', ['Context'])
        self.assertEqual(result['selected_passages'], [{'id': '', 'title': '', 'text': 'Context'}])
        self.assertIsNone(result['retrieved_candidates'])


class RetrievalContextTests(unittest.TestCase):
    def test_candidates_match_backend_collection_task_and_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)/'hippo'/'cloud'/'candidates.jsonl'
            path.parent.mkdir(parents=True)
            record={'method':'hipporag2','domain':'cloud','task_id':'case<::>1','question':'Q?',
                    'inference_stage':'new_retrieval_experiment','original_answers_reused':False,
                    'candidates':[{'id':'doc','text':'Before rerank','retrieval_rank':1}]}
            path.write_text(json.dumps(record)+'\n',encoding='utf-8')
            records=annotation.retrieval_records(annotation.retrieval_snapshot(tmp))
            result=annotation.retrieval_evidence(records,'hipporag2','cloud','case<::>1','Q?')
            self.assertEqual(result['documents'][0]['id'],'doc')
            self.assertEqual(result['source']['inference_stage'],'new_retrieval_experiment')
            self.assertIsNone(annotation.retrieval_evidence(records,'vector','cloud','case<::>1','Q?'))
            self.assertIsNone(annotation.retrieval_evidence(records,'hipporag2','fiqa','case<::>1','Q?'))
            self.assertIsNone(annotation.retrieval_evidence(records,'hipporag2','cloud','case<::>1','Different?'))
            path.write_text(json.dumps({**record,'candidates':[]})+'\n',encoding='utf-8')
            refreshed=annotation.retrieval_records(annotation.retrieval_snapshot(tmp))
            self.assertEqual(annotation.retrieval_evidence(refreshed,'hipporag2','cloud','case<::>1','Q?')['documents'],[])

    def test_ambiguous_pool_is_not_attached(self):
        records={('hipporag2','cloud','case<::>1'):[{'question':'Q?'},{'question':'Q?'}]}
        self.assertIsNone(annotation.retrieval_evidence(records,'hipporag2','cloud','case<::>1','Q?'))


if __name__ == '__main__':
    unittest.main()
