"""Build a portable HTML viewer for the live dynamic evaluator pilot."""
from __future__ import annotations

import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

STYLE = """
:root{--ink:#182c40;--muted:#627185;--line:#dce4ed;--blue:#245ac4;--paper:#fff;--wash:#f3f6fa;font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink);background:var(--wash)}
*{box-sizing:border-box}body{margin:0;font-size:15px;line-height:1.6}header{background:#fff;border-bottom:1px solid var(--line);padding:30px max(24px,calc((100vw - 1440px)/2))}h1{margin:0;font-size:clamp(25px,3vw,36px);letter-spacing:-.7px}h2{font-size:22px;margin:0 0 14px}h3{font-size:14px;margin:0 0 9px;color:var(--muted)}p{margin:0 0 12px}.eyebrow{text-transform:uppercase;font-size:12px;letter-spacing:1.4px;font-weight:750;color:var(--blue);margin-bottom:6px}.subtitle{color:var(--muted);max-width:980px;margin:10px 0 0}main{max-width:1488px;padding:24px;margin:auto}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}.metric{padding:16px 20px;background:white;border:1px solid var(--line);border-radius:12px}.metric strong{display:block;font-size:29px;line-height:1.3}.metric span{color:var(--muted);font-size:13px}.layout{display:grid;grid-template-columns:250px minmax(0,1fr);gap:22px;align-items:start}aside{position:sticky;top:16px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:17px}input,select{width:100%;padding:10px;border:1px solid #bdcbdc;border-radius:7px;background:white;color:var(--ink);font:inherit;margin:5px 0 12px}label{font-size:13px;font-weight:650}nav{max-height:54vh;overflow:auto}nav a{display:block;text-decoration:none;color:var(--ink);border-top:1px solid var(--line);padding:10px 0;font-size:12px}nav strong{display:block;font-size:14px;color:var(--blue)}a{color:var(--blue)}.case{scroll-margin-top:16px;background:#fff;border:1px solid var(--line);border-radius:13px;padding:24px;margin-bottom:20px;overflow-wrap:anywhere}.case-head{display:flex;gap:12px;justify-content:space-between;align-items:center;margin-bottom:16px}.case-head h2{margin:0}.badge{display:inline-block;font-size:12px;font-weight:700;padding:5px 10px;border-radius:30px;background:#fce9e8;color:#993b39}.badge.inconclusive{background:#fff2d6;color:#825600}.question{font-size:19px;font-weight:650;margin-bottom:18px}.box{background:#f5f8fc;border:1px solid #e1e8f1;border-radius:9px;padding:16px;margin:12px 0}.answer{border-left:3px solid #7195d7}.readable{white-space:pre-wrap;overflow-wrap:anywhere}.delta{background:#edf4ff;border-color:#cddffd}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.grid .box{margin:0}.section{margin-top:20px;padding-top:18px;border-top:1px solid var(--line)}ol,ul{margin:0;padding-left:22px}li+li{margin-top:9px}details{border:1px solid var(--line);border-radius:8px;margin:10px 0;background:white}summary{cursor:pointer;padding:12px;font-weight:650;font-size:14px}details[open]>summary{border-bottom:1px solid var(--line)}.detail-body{padding:15px}.chunk{border:1px solid var(--line);border-radius:7px;margin:10px 0;padding:13px;font-size:13px}.chunk-label{font-size:12px;color:var(--muted);font-weight:650;margin-bottom:7px}.small{font-size:12px;color:var(--muted)}.note{color:var(--muted);font-size:13px;padding:14px 0}.empty{padding:25px;background:#fff;border:1px solid var(--line);border-radius:12px}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;margin:0}.legend{display:flex;flex-wrap:wrap;gap:5px 15px;font-size:12px;color:var(--muted);margin-bottom:18px}.legend span{border-bottom:2px solid #dce4ed}.pill{font-size:11px;color:#32634e;background:#e7f5ed;padding:3px 7px;border-radius:5px;margin-left:8px}[hidden]{display:none!important}
@media(max-width:950px){.layout{grid-template-columns:1fr}aside{position:static}nav{display:flex;gap:15px;max-height:150px;flex-wrap:wrap}nav a{width:120px}.grid{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}@media(max-width:500px){main{padding:12px}.case{padding:16px}.case-head{align-items:start;flex-direction:column}header{padding:24px 16px}.badge{font-size:11px}}@media print{aside,.metrics{display:none}.layout{display:block}.case{break-inside:avoid}header,main{padding:10px}}
"""


def text(value):
    return escape(str(value), quote=True)


def list_html(values, ordered=False):
    tag = 'ol' if ordered else 'ul'
    return f'<{tag}>' + ''.join(f'<li>{text(v)}</li>' for v in values) + f'</{tag}>' if values else '<p class="small">None recorded.</p>'


def build(source: Path, target: Path):
    cases = json.loads((source / 'inputs.json').read_text())
    results = {r['case_id']: r for r in json.loads((source / 'results.json').read_text())}
    summary = json.loads((source / 'summary.json').read_text())
    plans = {}
    for path in sorted((source / 'requests').glob('*.json')):
        if path.name.endswith('.raw.json'):
            continue
        record = json.loads(path.read_text())
        output = record.get('output', {})
        if 'dynamic_sub_questions' not in output or 'attributed_failure' in output:
            continue
        payload = json.loads(record['request']['messages'][1]['content'])
        if 'user_query' in payload:
            plans[payload['user_query']] = output
    cards, links = [], []
    for case in cases:
        cid, artifacts = case['case_id'], case['artifacts']
        result = results[cid]
        diagnosis = result.get('diagnosis', {})
        plan = diagnosis or plans.get(artifacts['user_query'], {})
        status = result['status']
        failure = diagnosis.get('attributed_failure', 'INCONCLUSIVE' if status == 'inconclusive' else 'ERROR')
        search_text = json.dumps([cid, artifacts, plan, result], ensure_ascii=False).lower()
        links.append(f'<a href="#case-{text(cid)}" data-case-link="{text(cid)}"><strong>{text(cid)}</strong>{text(failure.replace("_", " ").title())}</a>')
        counts = (len(artifacts['top_k_chunks']), len(artifacts['prompt_context']))
        chunk_groups = []
        for key, title in [('prompt_context', 'Passed context · C_prompt'), ('top_k_chunks', 'Retrieved candidates · C_retrieved')]:
            chunks = []
            for index, passage in enumerate(artifacts[key]):
                passed = key == 'top_k_chunks' and passage in artifacts['prompt_context']
                badge = '<span class="pill">Passed to generator</span>' if passed else ''
                chunks.append(f'<div class="chunk"><div class="chunk-label">Chunk {index + 1} · zero-based index {index}{badge}</div><div class="readable">{text(passage)}</div></div>')
            chunk_groups.append(f'<details><summary>{text(title)} · {len(chunks)} chunks</summary><div class="detail-body">{"".join(chunks)}</div></details>')
        reason = diagnosis.get('justification', result.get('reason', 'No justification recorded.'))
        no_failure_note = '<p class="small">No failure was established for this case. Inconclusive is not a seventh failure category.</p>' if status == 'inconclusive' else ''
        cards.append(f'''<article class="case" id="case-{text(cid)}" data-case-id="{text(cid)}" data-status="{text(status)}" data-search="{text(search_text)}">
<div class="case-head"><h2>Case {text(cid)}</h2><span class="badge {text(status)}">{text(failure.replace('_', ' '))}</span></div>
<p class="question">{text(artifacts['user_query'])}</p>
<div class="box answer"><h3>Evaluated RAG answer</h3><div class="readable">{text(artifacts['rag_answer'])}</div></div>
<div class="grid"><div class="box"><h3>Extracted query constraints</h3>{list_html(plan.get('extracted_constraints', []))}</div>
<div class="box delta"><h3>Identified delta</h3><div class="readable">{text(plan.get('identified_delta', 'Not recorded'))}</div></div></div>
<section class="section"><h3>Dynamic verification questions · {len(plan.get('dynamic_sub_questions', []))}</h3>{list_html(plan.get('dynamic_sub_questions', []), True)}</section>
<section class="section"><h3>Attribution justification</h3><div class="readable">{text(reason)}</div>{no_failure_note}</section>
<section class="section"><h3>Execution artifacts · {counts[0]} retrieved / {counts[1]} passed</h3>{''.join(chunk_groups)}</section>
<details><summary>Structured diagnosis JSON</summary><div class="detail-body"><pre>{text(json.dumps(result, ensure_ascii=False, indent=2))}</pre></div></details>
</article>''')
    counts = Counter(r['status'] for r in results.values())
    metrics = [(len(cases), 'Evaluated cases'), (counts['classified'], 'Classified'),
               (counts['inconclusive'], 'Inconclusive'), (counts['error'], 'Execution errors')]
    metrics_html = ''.join(f'<div class="metric"><strong>{n}</strong><span>{label}</span></div>' for n, label in metrics)
    legend = ''.join(f'<span>{label}</span>' for label in ['Grounding', 'Response coverage', 'Context selection', 'Chunking', 'Retrieval', 'Knowledge boundary'])
    script = """const cards=Array.from(document.querySelectorAll('[data-case-id]'));const search=document.getElementById('search');const status=document.getElementById('status');function filter(){const needle=search.value.toLocaleLowerCase().trim();let count=0;cards.forEach(card=>{const visible=(!status.value||card.dataset.status===status.value)&&(!needle||card.dataset.search.includes(needle));card.hidden=!visible;const link=document.querySelector('[data-case-link="'+card.dataset.caseId+'"]');if(link)link.hidden=!visible;if(visible)count++;});document.getElementById('count').textContent=count+' of '+cards.length+' cases';document.getElementById('empty').hidden=count!==0;}search.addEventListener('input',filter);status.addEventListener('change',filter);filter();"""
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dynamic RAG failure attribution · First 10 cases</title><style>{STYLE}</style></head>
<body><header><div class="eyebrow">Check first · Dynamic probing</div><h1>RAG failure attribution</h1><p class="subtitle">First 10 saved cases · Model: {text(summary['model'])}. Inspect the evaluated answer, constraint gaps, dynamic questions, and the exact context available to the generator.</p></header>
<main><section class="metrics" aria-label="Run summary">{metrics_html}</section>
<div class="legend" aria-label="Six failure categories">{legend}</div>
<div class="layout"><aside><label for="search">Search cases and passages</label><input id="search" type="search" placeholder="ID, entity, question…"><label for="status">Status</label><select id="status"><option value="">All statuses</option><option value="classified">Classified</option><option value="inconclusive">Inconclusive</option><option value="error">Execution error</option></select><p class="small" id="count">{len(cases)} cases</p><nav aria-label="Case navigation">{''.join(links)}</nav></aside>
<div><div class="note">Gold answers and prior failure labels were excluded from evaluator inputs. All nine classified cases received Grounding Failure under the ordered single-label policy. This run measures execution, not classification accuracy. No secondary index searches were triggered.</div>
<div id="empty" class="empty" hidden>No cases match your filters.</div>{''.join(cards)}
<details><summary>Run metadata and limitations</summary><div class="detail-body"><pre>{text(json.dumps(summary, ensure_ascii=False, indent=2))}</pre></div></details>
<p class="note">Saved reproduced traces; historical execution and index equivalence are unverified. Each case shows the final-answer text actually evaluated. Secondary search and corpus-wide absence were not tested by these 10 cases.</p></div></div></main><script>{script}</script></body></html>'''
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding='utf-8')
    print(f'Built {target}: {len(cases)} cases, {len(html):,} characters')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'result/failure_classification/dynamic_evaluator/first10_luna')
    parser.add_argument('--output', type=Path, default=ROOT / 'dataset/singleturn/view_annotations/index8.html')
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == '__main__':
    main()
