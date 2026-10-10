# SPDX-License-Identifier: CC-BY-4.0
# Copyright 2026 LAION / Project Alexandria contributors.
"""Static English overview plus five complete paper readers, six views each."""
import hashlib
import html
import json
from pathlib import Path

from common import ROOT, digest, jsonl, load, write
from ku_pipeline import KU_CONDITIONS

SITE = ROOT/'outputs/site'
LIVE = 'https://projects.laion.ai/project-alexandria/ku-examples/'
HF12 = 'laion/Alexandria-Gemma-4-12B-it-Qwen27-Knowledge-Units-LoRA-r128'
HF4 = 'laion/Alexandria-Gemma-4-E4B-it-Qwen27-Knowledge-Units-LoRA-r128'
TEACHER = 'Qwen/Qwen3.8-27B-FP8'

def escape(value): return html.escape(str(value), quote=True)
def link(url, text): return f'<a href="{escape(url)}">{escape(text)}</a>'
def label(condition):
    name = condition.removesuffix('_bt')
    words = 500 if name.endswith('500') else 1000
    if name.startswith('qwen_'): return 'Qwen 3.8 27B FP8 · teacher', words
    model = 'Gemma 4 E4B IT' if name.startswith('gemma4_') else 'Gemma 4 12B IT'
    return model+(' · KU LoRA rank 128' if '_r128_' in name else ' · no adapter'), words
def score(value): return f'{100*value["accuracy"]:.1f}% ({value["correct"]}/200)'
def slug(condition):
    condition = condition.removesuffix('_bt')
    return condition.replace('_r128', '').replace('_', '-')
def pretty(value):
    if isinstance(value, str): return escape(value)
    return escape(json.dumps(value, ensure_ascii=False, indent=2))
def attrs(value):
    if not value: return '<p class="muted">No attributes recorded.</p>'
    return '<dl class="attributes">'+''.join(f'<dt>{escape(k)}</dt><dd>{pretty(v)}</dd>' for k,v in value.items())+'</dl>'
def page(title, body, prefix=''):
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)} · Alexandria</title><link rel="stylesheet" href="{prefix}reader.css"></head><body><header class="site-header"><a href="{prefix}index.html" class="brand">PROJECT ALEXANDRIA <span>Knowledge Unit reader</span></a><nav><a href="{prefix}index.html#results">Results</a><a href="{prefix}index.html#examples">Five papers</a><a href="{prefix}evaluation.pdf">PDF report</a><a href="https://arxiv.org/abs/2502.19413v2">Alexandria paper</a></nav></header><main>{body}</main><footer>Measured on 20 disjoint papers · No thinking · Factual KUs in source order · English → German → English repair · 10 October 2026</footer></body></html>'''

CSS = '''
:root{--ink:#163044;--muted:#5b6b77;--border:#dce5eb;--bg:#f4f7f9;--teal:#087e78;--blue:#287aad}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}.site-header{background:white;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;gap:20px;padding:20px max(24px,calc((100vw - 1220px)/2));align-items:center}.brand{font-size:11px;font-weight:800;letter-spacing:.12em;color:var(--teal)}.brand span{display:block;font-size:13px;letter-spacing:0;color:var(--ink)}nav{display:flex;gap:20px;font-size:12px}main{max-width:1220px;margin:35px auto;padding:0 24px}h1{font-size:38px;line-height:1.16;letter-spacing:-.035em;max-width:1000px;margin:10px 0 20px}h2{font-size:24px;line-height:1.3;letter-spacing:-.02em;margin:32px 0 12px}h3{font-size:16px;line-height:1.4}.eyebrow{text-transform:uppercase;letter-spacing:.1em;font-size:11px;color:var(--teal);font-weight:750}.lead{font-size:17px;max-width:1000px;color:var(--muted)}.muted{color:var(--muted);font-size:13px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}.card,.panel{background:white;border:1px solid var(--border);padding:18px 22px;border-radius:12px}.card strong{font-size:30px;display:block;line-height:1.3}.card span{font-size:12px;color:var(--muted)}.pipeline{display:flex;flex-wrap:wrap;gap:10px;margin:24px 0}.pipeline span{border:1px solid var(--border);border-radius:8px;background:white;padding:9px 14px;font-size:13px}.pipeline b{align-self:center;color:var(--teal)}.table-scroll{overflow-x:auto;border:1px solid var(--border);border-radius:10px;background:white}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #edf1f4;vertical-align:top}th{font-size:11px;color:var(--muted);background:#edf3f7}.numeric{white-space:nowrap;font-variant-numeric:tabular-nums}.accent{color:var(--teal);font-weight:750}.badge{display:inline-block;font-size:11px;background:#e2f2ee;padding:3px 8px;border-radius:5px;color:#17655d;margin-right:5px}.warning{background:#fff3df;color:#885c1d}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:18px 0}.papers{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}.paper-card:last-child{grid-column:1/-1}.paper-card h3{font-size:18px;margin:8px 0 12px}.paper-card a.button{display:inline-block;margin-top:14px;background:var(--teal);color:white;padding:8px 14px;border-radius:6px;font-size:13px}.metadata{display:grid;grid-template-columns:160px 1fr;gap:5px 15px;font-size:13px;margin:14px 0}.metadata dt{color:var(--muted)}.metadata dd{margin:0;overflow-wrap:anywhere}.views{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.views a{border:1px solid var(--border);background:white;border-radius:7px;padding:7px 11px;font-size:12px}.views .current{border-color:var(--teal);background:#e6f4f0;color:var(--teal)}.toc{position:sticky;top:0;z-index:2;background:#f4f7f9ee;border:1px solid var(--border);padding:10px 16px;border-radius:8px;font-size:12px;display:flex;gap:12px;flex-wrap:wrap}.unit{margin:28px 0;background:white;border:1px solid var(--border);border-radius:12px;padding:26px;scroll-margin-top:65px}.unit h2{margin:0 0 7px}.context{border-left:3px solid var(--teal);padding:8px 16px;background:#f1f8f5;font-size:16px;line-height:1.7;margin:16px 0}.entities{display:grid;grid-template-columns:1fr 1fr;gap:15px}.entity{border:1px solid var(--border);border-radius:9px;padding:14px 17px;min-width:0}.entity h3{margin:0 0 3px;font-size:16px;overflow-wrap:anywhere}.type{font-size:11px;color:var(--muted)}.attributes{display:grid;grid-template-columns:minmax(100px,35%) 1fr;gap:5px 12px;margin:13px 0;font-size:12px}.attributes dt{color:var(--muted);overflow-wrap:anywhere}.attributes dd{margin:0;overflow-wrap:anywhere;white-space:pre-wrap}.relations{font-size:12px;padding-left:18px}.relations li{margin:7px 0;overflow-wrap:anywhere}.relations code{font-size:11px;color:var(--teal)}.adjacent{display:grid;grid-template-columns:1fr 1fr;gap:15px;border-top:1px solid var(--border);margin-top:22px;padding-top:15px;font-size:12px;color:var(--muted)}details{font-size:12px;margin:12px 0}summary{cursor:pointer;color:var(--blue)}pre{background:#f2f5f7;padding:13px;border-radius:7px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}footer{max-width:1220px;margin:45px auto 20px;padding:20px 24px;border-top:1px solid var(--border);color:var(--muted);font-size:11px}.note{font-size:12px;max-width:1080px;color:var(--muted)}@media(max-width:800px){.site-header{display:block}nav{margin-top:15px;flex-wrap:wrap;gap:12px}main{padding:0 16px}.cards,.columns,.papers,.entities{grid-template-columns:1fr}.paper-card:last-child{grid-column:auto}h1{font-size:29px}.metadata{grid-template-columns:1fr}.unit{padding:18px}.adjacent{grid-template-columns:1fr}.toc{position:static}}@media print{.site-header nav,.toc{display:none}body{background:white}main{margin:0;padding:0}.unit,.entity{break-inside:avoid}.views{display:none}.entities{display:block}.entity{margin:10px 0}a{color:inherit}}
'''

def build_reader(example, all_for_paper, metrics, audits):
    docid, c = example['document_id'], example['condition']; result = example['result']; source = example['source_metadata']
    directory = SITE/docid; directory.mkdir(parents=True, exist_ok=True)
    filename = slug(c)+'.html'; json_name = slug(c)+'.json'
    write(directory/json_name, example)
    model, words = label(c)
    status = 'Complete extraction' if example['status']=='generated' else 'Partial extraction · original failure retained'
    metadata = [('Source title',source['title']),('Source authors','; '.join(source['authors']) or 'Unavailable in registry'),
                ('KU document title',result['title']),('KU document author field',result.get('author') or 'Not populated by the original extractor'),
                ('Style annotation','No dedicated document-style field in this schema'),
                ('KU sequence',f"{example['chunks_generated']} of {example['chunks_expected']} expected chunks"),
                ('Attribution provenance','Source registry metadata shown for attribution; not added to the QA context')]
    meta = '<dl class="metadata">'+''.join(f'<dt>{escape(k)}</dt><dd>{escape(v)}</dd>' for k,v in metadata)+'</dl>'
    views = '<div class="views">'+''.join(f'<a class="{"current" if e["condition"]==c else ""}" href="{slug(e["condition"])}.html">{escape(label(e["condition"])[0])} · {label(e["condition"])[1]:,} words</a>' for e in all_for_paper)+'</div>'
    a = audits[(c,docid)]['audit']['categories']['narrative']
    q = metrics['scores'][c]
    intro = f'''<p class="eyebrow">Full-paper Knowledge Units · saved output</p><h1>{escape(source['title'])}</h1>
<p class="lead">{escape(model)} · {words:,}-word source chunks · guarded TranslateGemma 4B back-translation</p>
<span class="badge">{escape(status)}</span><span class="badge warning">Longest literal run: {a['longest_contiguous_match_words']} words</span>
<div class="panel">{meta}<p class="muted">This condition scores <strong>{score(q)}</strong> across all 20 evaluation papers. The example page does not imply that every claim in this individual paper is correct.</p></div>{views}
<p class="note">Every saved KU, entity, attribute and relationship is shown below, in original chunk order. Navigation follows the source sequence rather than publication dates. Adjacent summaries are reader navigation, not lookahead supplied to extraction. All KUs were generated before the paraphrasing stage. {link(json_name,'Download complete factual KU JSON')}.</p>'''
    if source['doi']: intro += '<p class="muted">Original paper: '+link('https://doi.org/'+source['doi'], source['doi'])+'</p>'
    units = result['knowledge_units']
    toc = '<div class="toc">Jump to KU: '+''.join(link(f'#ku{u["chunk_index"]}',str(u['chunk_index']+1)) for u in units)+'</div>'
    sections = []
    for i,u in enumerate(units):
        start,end = u['source']['start_word']+1,u['source']['end_word']
        entities = []
        for e in u['entities']:
            relationships = '<ul class="relations">'+''.join('<li><code>'+escape(r['predicate'])+'</code> → '+escape(r['target'])+(attrs(r['attributes']) if r['attributes'] else '')+(f'<span class="muted"> [target ID: {escape(r["target_id"])}]</span>' if r.get('target_id') else '')+'</li>' for r in e['relationships'])+'</ul>' if e['relationships'] else '<p class="muted">No relationships recorded.</p>'
            aliases = '<p class="muted">Aliases: '+', '.join(escape(x) for x in e['aliases'])+'</p>' if e['aliases'] else ''
            entities.append(f'<article class="entity"><h3>{escape(e["name"])}</h3><span class="type">{escape(e["entity_type"])} · ID: {escape(e["entity_id"])}</span>{aliases}{attrs(e["attributes"])}<h4>Relationships</h4>{relationships}</article>')
        adjacent = []
        for j,name in [(i-1,'Previous KU'),(i+1,'Next KU')]:
            adjacent.append('<div>'+link(f'#ku{units[j]["chunk_index"]}',name)+': '+escape(units[j]['context_summary'])+'</div>' if 0<=j<len(units) else '<div>'+name+': '+('Start of paper' if j<0 else 'End of paper')+'</div>')
        fingerprint = '<details><summary>Source positions, MinHash and extraction warnings</summary><pre>'+pretty(dict(source=u['source'],extraction_warnings=u['extraction_warnings']))+'</pre></details>'
        sections.append(f'<section class="unit" id="ku{u["chunk_index"]}"><h2>KU {u["chunk_index"]+1} <span class="muted">· source words {start:,}–{end:,}</span></h2><p class="muted">{len(u["entities"])} entities · {sum(len(e["relationships"]) for e in u["entities"])} relationships · {min(10,u["chunk_index"])} preceding KUs available during extraction</p><h3>Recorded context summary</h3><p class="context">{escape(u["context_summary"])}</p><div class="entities">'+''.join(entities)+'</div>'+fingerprint+'<div class="adjacent">'+''.join(adjacent)+'</div></section>')
    (directory/filename).write_text(page(source['title'],intro+toc+''.join(sections),'../'))
    return dict(document_id=docid,condition=c,page=docid+'/'+filename,json=docid+'/'+json_name,
                knowledge_units=len(units),entities=sum(len(u['entities']) for u in units),complete=example['status']=='generated')

def main():
    metrics = load(ROOT/'outputs/metrics.json'); copy = load(ROOT/'outputs/copy_overlap.json')
    examples = [json.loads(x) for x in (ROOT/'outputs/public_demo_examples.jsonl').read_text().splitlines()]
    audits = {(r['condition'],r['document_id']): r for r in [json.loads(x) for x in (ROOT/'outputs/copy_overlap_details.jsonl').read_text().splitlines()]}
    SITE.mkdir(parents=True,exist_ok=True); (SITE/'reader.css').write_text(CSS)
    bypaper = {}
    for e in examples: bypaper.setdefault(e['document_id'],[]).append(e)
    manifest = []
    for docid, variants in bypaper.items():
        variants.sort(key=lambda e: (0 if e['condition'].startswith('gemma12') else 1 if e['condition'].startswith('gemma4') else 2, label(e['condition'])[1]))
        for e in variants: manifest.append(build_reader(e,variants,metrics,audits))
    body = '<p class="eyebrow">Models, measurements and five complete examples</p><h1>Scientific facts, extracted as Knowledge Units</h1><p class="lead">Read the complete factual extraction of five scientific papers, from the first source chunk to the last. Compare 500- and 1,000-word chunks, Gemma LoRA students and the Qwen-27B teacher. This demo adds guarded back-translation to reduce copied wording and measures whether question-answering quality survives.</p>'
    body += '<div class="pipeline"><span>Full paper</span><b>→</b><span>Sequential 500 / 1,000-word KUs</span><b>→</b><span>Detect copied phrases</span><b>→</b><span>English → German → English</span><b>→</b><span>Fidelity &amp; copying guards</span><b>→</b><span>Frozen QA evaluation</span></div>'
    cards = []
    for c,title in [('original','Full original paper'),('qwen_ku500_bt','Qwen teacher · 500-word KUs + BT'),('gemma12_r128_ku500_bt','Gemma12 LoRA · 500-word KUs + BT'),('no_context','No paper context')]:
        cards.append('<div class="card"><span>'+title+'</span><strong>'+f'{100*metrics["scores"][c]["accuracy"]:.1f}%'+'</strong><span>'+str(metrics['scores'][c]['correct'])+' / 200 correct</span></div>')
    body += '<div class="cards">'+''.join(cards)+'</div><p class="note">20 disjoint papers, 200 frozen four-choice MCQs per condition, the same Qwen2.5-7B answerer. All 28 contexts were freshly judged in this repair study. No thinking, no retraining, and no QA-based repair selection. Failed generations remain wrong in the denominator.</p>'
    body += '<h2 id="weights">Weights and chunk-size profiles</h2><div class="columns">'
    for repo,model in [(HF12,'Gemma 4 12B IT'),(HF4,'Gemma 4 E4B IT')]:
        body += f'<div class="panel"><h3>{model} · KU LoRA rank 128</h3><p>{link("https://huggingface.co/"+repo,"Adapter repository & weights")}</p><p>{link("https://huggingface.co/"+repo+"/blob/main/profiles/ku500.json","500-word inference profile")} · {link("https://huggingface.co/"+repo+"/blob/main/profiles/ku1000.json","1,000-word inference profile")}</p><p class="muted">One adapter per base model, trained on 1,000-word teacher chunks for one epoch. These links apply the same weights at two inference chunk sizes.</p></div>'
    body += '</div><p class="note">Teacher: '+link('https://huggingface.co/'+TEACHER,TEACHER)+' · Translation: '+link('https://huggingface.co/google/translategemma-4b-it','TranslateGemma 4B')+'.</p>'
    body += '<h2 id="results">QA accuracy: before and after guarded back-translation</h2><div class="table-scroll"><table><thead><tr><th>KU generator</th><th>Source chunk</th><th>Raw KUs</th><th>Guarded BT</th><th>Change</th><th>Paired 95% interval</th><th>Invalid after BT</th></tr></thead><tbody>'
    for c in KU_CONDITIONS:
        a,b = metrics['scores'][c],metrics['scores'][c+'_bt']; d = metrics['paired_backtranslation_minus_raw'][c]; low,high=d['confidence_interval_95']; name,words=label(c)
        body += f'<tr><td>{escape(name)}</td><td class="numeric">{words:,} words</td><td class="numeric">{score(a)}</td><td class="numeric accent">{score(b)}</td><td class="numeric">{100*d["point"]:+.1f} pp</td><td class="numeric">{100*low:+.1f} to {100*high:+.1f} pp</td><td>{b["invalid"]}</td></tr>'
    body += '</tbody></table></div><p class="note">Paired intervals use 10,000 paper-bootstrap draws. An interval containing zero does not establish a reliable accuracy change. Repair and batching parameters were fixed before this QA run. Questions were teacher-authored and automatically source-validated; they are not independently human-curated.</p>'
    body += '<h2>Full-paper, no-context and summary reference conditions</h2><div class="table-scroll"><table><thead><tr><th>Context</th><th>QA accuracy</th><th>95% interval</th><th>Invalid slots</th><th>Weights</th></tr></thead><tbody>'
    extra = {'no_context':('No paper context','Qwen/Qwen2.5-7B-Instruct'),'original':('Complete original paper / ground-truth context','Qwen/Qwen2.5-7B-Instruct'),
             'qwen_summary':('Qwen 27B summary',TEACHER),'gemma4_base_summary':('Gemma E4B summary · no adapter','google/gemma-4-E4B-it'),
             'gemma4_r128_summary':('Gemma E4B KU LoRA used for summary generation',HF4),'gemma12_base_summary':('Gemma12 summary · no adapter','google/gemma-4-12B-it'),
             'gemma12_r128_summary':('Gemma12 KU LoRA used for summary generation',HF12),'gemma12_previous_summary_lora':('Gemma12 earlier summary LoRA · rank128','laion/Alexandria-Gemma-4-12B-it-Qwen27-Summaries-LoRA-r128')}
    for c,(name,weights) in extra.items():
        s=metrics['scores'][c]; low,high=s['confidence_interval_95']; body += f'<tr><td>{name}</td><td class="numeric">{score(s)}</td><td class="numeric">{100*low:.1f}–{100*high:.1f}%</td><td>{s["invalid"]}</td><td>{link("https://huggingface.co/"+weights,"Model / adapter")}</td></tr>'
    body += '</tbody></table></div><p class="note">Original paper means the actual source context, not an assumed perfect answer score. These summary rows were not back-translated in this KU study.</p>'
    body += '<h2>Copying and fidelity checks</h2><div class="table-scroll"><table><thead><tr><th>KU generator</th><th>Words</th><th>Raw coverage ≥6</th><th>BT coverage ≥6</th><th>BT longest run</th><th>BT ≤5 pass</th><th>BT ≥7 flagged</th></tr></thead><tbody>'
    for c in KU_CONDITIONS:
        a,b=copy['conditions'][c]['narrative'],copy['conditions'][c+'_bt']['narrative']; name,words=label(c)
        body += f'<tr><td>{escape(name)}</td><td>{words:,}</td><td>{100*a["mean_covered_fraction_six_plus"]:.2f}%</td><td>{100*b["mean_covered_fraction_six_plus"]:.2f}%</td><td>{b["maximum_contiguous_match_words"]} words</td><td>{b["strict_five_word_pass"]} / 20</td><td>{b["violation_seven_plus"]} / 20</td></tr>'
    body += '</tbody></table></div>'
    q=metrics['quality'];perf=metrics['translation'];allocation=metrics['allocation']
    body += f'<div class="columns"><div class="panel"><h3>Conservative repair acceptance</h3><p>{q["inserted_windows"]:,} windows inserted across {q["changed_documents"]} of 200 KU document versions. Raw candidates had {q["raw_attempts_with_critical_flags"]:,} conservative number / unit / formula flags. <strong>{q["guarded_global_critical_failures"]} retained document-level critical-value flags</strong> after guard/rollback.</p><p class="muted">Acceptance requires both NLI directions ≥0.9, unchanged critical-value signatures and no copied run over five words inside the repaired window. Rejected changes retain the original field. These automated checks do not certify every scientific claim.</p></div><div class="panel"><h3>Measured compute</h3><p>Translation: {perf["active_translation_seconds"]:.1f} active seconds / {perf["active_translation_gpu_hours"]:.4f} active GPU-hours for 200 KU document versions. Complete experiment: {allocation.get("reserved_gpu_hours",0):.4f} reserved GPU-hours on four GH200 GPUs.</p><p class="muted">Active translation excludes cold load, semantic checks, QA and idle roles. Reserved allocation includes them. These are measured experimental costs, not a large-corpus throughput guarantee.</p></div></div><p class="note">Primary copying counts retain every factual field, including protected technical names, relation targets and bibliographic information occurring inside factual attributes, exactly as in the preceding KU benchmark. Titles at document level are metadata. Preserving names can leave literal matches; no output is removed to improve the score.</p>'
    body += '<h2>Context, attribution and style: what is actually present?</h2><div class="panel"><p>The Alexandria paper describes contextual summaries, entities, attributes, relationships and sentence MinHash. This experiment uses the repository’s sequential schema: at most ten preceding KU summaries/entity names support consistent naming. <strong>It does not supply future source chunks.</strong> The recorded <code>context_summary</code> is a local contextual description under the current prompt; a separate summary of the entire preceding narrative is not guaranteed.</p><p>Each reader exposes source-registry title/authors alongside the actual KU metadata. Authors may also appear inside entity attributes. The original document-author envelope is empty; registry attribution is added for display only. There is no dedicated style annotation field. Navigation shows the preceding and following KU summaries as a reader aid, rather than generated lookahead annotations. Short factual descriptors are visible where the extractor recorded them.</p></div>'
    body += '<div class="table-scroll"><table><thead><tr><th>Generator / chunk</th><th>Exact title /20</th><th>Author envelope /20</th><th>Author attributes /20</th><th>Style attributes /20</th><th>Context / generated KUs</th><th>MinHash / generated KUs</th></tr></thead><tbody>'
    for c in KU_CONDITIONS:
        m=metrics['metadata'][c];name,words=label(c); body += f'<tr><td>{escape(name)} · {words:,}</td><td>{m["exact_document_title_matches"]}</td><td>{m["nonempty_document_authors"]}</td><td>{m["author_attributes_present"]}</td><td>{m["explicit_style_attributes_present"]}</td><td>{m["context_summary_present_units"]} / {m["units"]}</td><td>{m["sentence_minhash_present_units"]} / {m["units"]}</td></tr>'
    body += '</tbody></table></div><h2 id="examples">Five complete papers</h2><p class="muted">Selected before this repair evaluation, without QA-based selection. Each paper provides six complete saved views: Gemma12 LoRA, Gemma E4B LoRA and Qwen27, each at 500 and 1,000 words. Partial failures, if any, remain explicit.</p><div class="papers">'
    for docid,versions in bypaper.items():
        default=next(e for e in versions if e['condition']=='gemma12_r128_ku500_bt');source=default['source_metadata'];body += f'<article class="panel paper-card"><span class="eyebrow">{escape(docid)}</span><h3>{escape(source["title"])}</h3><p class="muted">{escape("; ".join(source["authors"]))}</p><p class="muted">{len(default["result"]["knowledge_units"])} sequential KUs at 500 words · six model/chunk views</p><a class="button" href="{docid}/{slug(default["condition"])}.html">Read complete Knowledge Units →</a></article>'
    body += '</div><h2>Downloads and reproducibility</h2><p>'+link('evaluation.pdf','Evaluation PDF')+' · '+link('examples.pdf','Complete Gemma12 examples PDF')+' · '+link('metrics.json','All QA scores and protocol')+' · '+link('copy_overlap.json','Copy audit')+' · '+link('manifest.json','Example inventory and checksums')+'</p><p class="note">'+link('https://arxiv.org/abs/2502.19413v2','Project Alexandria paper')+' · '+link('https://github.com/LAION-AI/project-alexandria','Code and evaluation repository')+'. Demo KUs preserve saved outputs without editorial rewriting. Original paper bodies, source-containing prompts, model reasoning and MCQs are not included in this reader.</p>'
    if (ROOT/'outputs/bibliographic_copy_diagnostic.json').exists():
        diag=load(ROOT/'outputs/bibliographic_copy_diagnostic.json')
        write(SITE/'bibliographic_copy_diagnostic.json',diag)
        body+='<p class="note">Embedded author/title/citation attributes and exact document-title strings are also measured as metadata in a '+link('bibliographic_copy_diagnostic.json','separate bibliographic copying diagnostic')+'. The historical primary all-factual-string audit above remains unchanged; this additional classification does not change repairs or QA scores.</p>'
    (SITE/'index.html').write_text(page('Knowledge Unit pipeline, back-translation and QA',body))
    write(SITE/'metrics.json',metrics);write(SITE/'copy_overlap.json',copy)
    write(SITE/'manifest.json',dict(demo_papers=5,views=len(manifest),examples=manifest,
          saved_KU_content_unchanged=True,primary_QA_conditions=28,live_url=LIVE,
          source_metrics_sha256=digest((ROOT/'outputs/metrics.json').read_bytes()),
          notice='Reader attribution and adjacent navigation are display-only; no new facts added to model outputs or QA contexts.'))
    print('BUILT_SITE',len(manifest),'full paper model/chunk readers',flush=True)

if __name__=='__main__':main()
