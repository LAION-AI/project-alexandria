"""Local-only reproducible report for the larger insertion system."""
from pathlib import Path
from collections import Counter
import json,html,hashlib,shutil,subprocess,re
ROOT=Path('/e/scratch/reformo/schuhmann1/scientific-post-gemma-insertion-system-v3-97-20261024')
HERE=Path(__file__).resolve().parent
HOME=Path('/e/home/jusers/schuhmann1/jupiter')
def load(p):return json.loads(p.read_text())
def lines(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def esc(s):return html.escape(str(s))
def main():
 r=load(ROOT/'outputs/results_and_scaling.json');assert r['complete'];protocol=load(ROOT/'inputs/protocol.json')
 old=load(ROOT.parent/'scientific-post-gemma-safe-insertions-guarded-97-20261024/outputs/results_and_scaling.json')['conditions']['grammar_insert']
 labels={'baseline':'Frozen preceding pipeline','legacy_grammar':'Earlier narrow grammar rules','system_greedy':'Larger catalog, immediate-gain search','system_joint':'Larger catalog, joint planning','system_core':'Conservative catalog, joint planning'}
 body_rows='';records={};allrows=[]
 for c in ['baseline','legacy_grammar','system_greedy','system_joint','system_core']:
  if c in r['conditions']:v=r['conditions'][c];m=v['insertion'];counts=v['meaning_counts'];cpu=v['extra_cpu_core_hours_38m'];n=v['overlap']['narrative']
  elif c=='legacy_grammar':v=old;m=v['insertion'];counts=v['meaning_counts'];cpu=v['extra_cpu_core_hours_38m'];n=v['overlap']['narrative']
  else:
   m=load(ROOT/'outputs/baseline_complete.json');n=load(ROOT/'outputs/overlap/copy_overlap.json')['conditions'][c]['narrative'];counts={};cpu=0
  q=r['qa']['scores'][c];bad=counts.get('major_meaning_change',0)+counts.get('unintelligible',0)
  original=m.get('original_word_coverage_fixed_denominator',m['mean_coverage']);fixed=m.get('coverage_with_fixed_original_denominator',m['mean_coverage'])
  if c=='baseline':original=m['mean_coverage'];fixed=m['mean_coverage']
  allrows.append(dict(condition=c,label=labels[c],mean_coverage=n['mean_covered_fraction_six_plus'],fixed_denominator_coverage=fixed,original_word_fixed_denominator_coverage=original,
   copied_words=m['absolute_copied_words'],added_words=m.get('inserted_words',0),qa=q['correct'],meaning_counts=counts,cpu_core_hours_38m=cpu,strict_passes=n['strict_five_word_pass']))
  body_rows+=f'<tr><th>{labels[c]}</th><td>{n["mean_covered_fraction_six_plus"]:.4%}</td><td>{fixed:.4%}</td><td>{original:.4%}</td><td>{q["correct"]}/970<br>{q["accuracy"]:.2%}</td><td>{esc(counts) if counts else "—"}</td><td>{n["strict_five_word_pass"]}/97</td><td>{m.get("inserted_words",0):,}</td><td>{cpu:,.0f}</td></tr>'
 cards=lines(ROOT/'outputs/reviews/post.jsonl');unique={c['pair_id']:c for c in cards if c['condition']!='controls'}
 selected=sorted(unique.values(),key=lambda c:(c['judgment']['severity'] in {'unchanged_or_equivalent','awkward_only'},c['pair_id']))[:16]
 samples=''
 for c in selected:
  samples+='<article><h3>'+esc(c['condition'])+' · '+esc(c['document_id'])+' · '+esc(c['judgment']['severity'])+'</h3><div class="two"><div><b>Before</b><p>'+esc(c['before'])+'</p></div><div><b>After</b><p>'+esc(c['after'])+'</p></div></div><p class="small">Automatic reviewer: '+esc(c['judgment']['reason'])+'</p></article>'
 joint_examples=[]
 for p in sorted((ROOT/'traces/system_joint').glob('*.jsonl')):
  for t in lines(p):
   if t['search']['zero_gain_intermediate_plans']:
    joint_examples.append(dict(uid=p.stem,path=t['path'],before=t['before'],after=t['after'],search=t['search'],edits=t['edits']))
    break
  if len(joint_examples)>=6:break
 evidence=''
 for t in joint_examples:evidence+='<details><summary>'+esc(t['uid'])+' · '+esc(t['path'])+'</summary><div class="two"><div><b>Before</b><p>'+esc(t['before'])+'</p></div><div><b>Joint plan</b><p>'+esc(t['after'])+'</p></div></div><pre>'+esc(json.dumps(t['search'],indent=2))+'</pre></details>'
 rule_tables=''
 for c,v in r['conditions'].items():
  m=v['insertion'];rule_tables+='<h3>'+labels[c]+'</h3><div class="two"><div><b>Accepted rules</b><pre>'+esc(json.dumps(m['rule_counts'],indent=2))+'</pre></div><div><b>Search counters</b><pre>'+esc(json.dumps(m['search'],indent=2))+'</pre></div></div>'
 controls=''.join('<tr><td>'+esc(c['before'])+'</td><td>'+esc(c['after'])+'</td><td>'+esc(c['expected'])+'</td><td>'+esc(c['judgment']['severity'])+'</td></tr>' for c in r['meaning_controls'])
 job=load(ROOT/'outputs/status.json')['job_id']
 v1=ROOT.parent/'scientific-post-gemma-insertion-system-97-20261024'
 v2=ROOT.parent/'scientific-post-gemma-insertion-system-v2-97-20261024'
 ids=[load(v1/'outputs/status.json')['job_id'],load(v2/'outputs/status.json')['job_id'],job]
 proc=subprocess.run(['sacct','-X','-j',','.join(ids),'--noheader','--parsable2','--format=JobID,State,ElapsedRaw,AllocTRES'],check=True,capture_output=True,text=True)
 jobs=[]
 for line in proc.stdout.strip().splitlines():
  a=line.split('|');assert a[1]=='COMPLETED',a
  match=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',a[3]);gpus=int(match.group(1)) if match else 4
  jobs.append(dict(job_id=a[0],state=a[1],elapsed_seconds=int(a[2]),allocated_gh200_gpus=gpus,reserved_gpu_hours=int(a[2])*gpus/3600))
 allocation=dict(jobs=jobs,total_reserved_gpu_hours=sum(x['reserved_gpu_hours'] for x in jobs),scope='All three system test versions, including CPU work, model loading/idle and actual full-node reservations. Deployed new stages remain CPU-only.')
 write=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
 write(ROOT/'outputs/allocation_accounting.json',allocation);write(HERE/'comparison.json',allrows)
 glossary_count=sum(len(load(p)['definitions']) for p in (ROOT/'outputs/glossaries').glob('*.json'))
 c=r['conditions']['system_core'];j=r['conditions']['system_joint'];g=r['conditions']['system_greedy'];jb=j['insertion'];cb=c['insertion']
 css='''*{box-sizing:border-box}body{margin:0;background:#f3f4ec;color:#183d34;font:16px/1.65 system-ui,sans-serif}main{max-width:1350px;margin:auto;padding:36px 24px 80px}h1{font:clamp(34px,5vw,61px)/1.1 Georgia,serif}h2{font:30px Georgia,serif;margin-top:40px}.lead{font-size:20px;max-width:1000px}.note{background:#fff0c8;border-radius:12px;padding:20px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.card,article{background:white;border:1px solid #d2dece;border-radius:14px;padding:20px;margin:18px 0}.big{font-size:32px;display:block}.scroll{overflow:auto;background:white;border:1px solid #d2dece;border-radius:12px}table{border-collapse:collapse;width:100%;min-width:1100px}td,th{padding:13px;text-align:left;vertical-align:top;border-bottom:1px solid #d2dece}thead{background:#e2ebdd}.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}p,td{overflow-wrap:anywhere}.small{font-size:14px;color:#5a6e63}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}details{border-top:1px solid #d2dece;padding:14px 0}summary{cursor:pointer;font-weight:650}a{color:#10624a}@media(max-width:800px){.two,.cards{grid-template-columns:1fr}}'''
 body=f'''<h1>A larger insertion system for scientific summaries</h1><p class="lead">Thirty rule families, typed unit explanations, paper-defined acronym glosses, and a planner that can combine individually zero-gain steps. Same 97 full papers and 970 original questions. 24 October 2026.</p>
 <div class="cards"><div class="card">Full system with joint planning<b class="big">{j['overlap']['narrative']['mean_covered_fraction_six_plus']:.3%}</b>mean narrative coverage in literal source matches of six or more words</div><div class="card">Conservative system<b class="big">{c['overlap']['narrative']['mean_covered_fraction_six_plus']:.3%}</b>{c['qa']['correct']}/970 QA; automatic meaning labels shown below</div><div class="card">Zero-gain intermediate steps<b class="big">{jb['search']['zero_gain_intermediate_plans']}</b>committed component plans containing an individually insufficient insertion</div></div>
 <p class="note"><b>Interpret the result precisely.</b> Added function words and explanatory parentheses can break a literal sequence while retaining all original wording in order. This is surface elaboration, not an independently written conceptual paraphrase. Mean copied-word coverage below 5% is separate from eliminating every passage longer than five words.</p>
 <h2>All frozen arms</h2><div class="scroll"><table><thead><tr><th>Condition</th><th>Mean covered words≥6</th><th>All covered words / original denominator</th><th>Original covered words / original denominator</th><th>QA</th><th>Automatic meaning labels</th><th>Entire narrative≤5</th><th>Added words</th><th>Extra CPU coreh /38M</th></tr></thead><tbody>{body_rows}</tbody></table></div>
 <p>The mean is the unweighted average of 97 per-summary narrative fractions. Fixed original denominators expose padding. Each committed plan must reduce original covered word positions, total covered positions, six-gram hits and fraction; merely appending text fails. Explicit quotes and bibliography are audited separately. Source hashes and primary/punctuation-split evidence are retained. Failed proposals and rollbacks remain visible; no paper or QA slot is dropped.</p>
 <h2>The rule system</h2><p>The catalog contains {protocol['rule_family_count']} families: complementizers and quantifiers; explicit purpose/instrument/process phrases; unambiguous physical quantity elaborations; temporal phrases; textual and past-tense reduced relatives; exact unit-name parentheses; and initialism definitions obtained from the same paper. Version 2 recognizes PDF-wrapped and reverse definitions, and standard aliases whose full term occurs in that paper; explicit conflicting definitions take precedence. The core arm excludes acronym glosses and both relative-clause families.</p><p>Unit names use a finite exact dictionary and never rescale a quantity. Single-letter symbols need context: h can be Planck’s constant and g can denote gravitational acceleration. Known percent units such as mol% remain whole; other slash/dot/product/power/per compounds are protected. Version 3 adds the percent glyph, chemically contextual molarity, rpm with rotation/centrifugation cues, and daltons. M parameters and modulo expressions are excluded. Original formula/quote/code spans cannot be interrupted. A dimension qualifier cannot capture a coordinated quantity in a different dimension. Typed parenthetical unit explanations may resemble functions to a generic regex; they are explicitly recognized as explanatory language.</p><p>The source parser found {glossary_count} eligible one-to-five-word definition/terminology entries across the 97 sources. Forward/reverse initialisms use constrained letter alignment; a standard alias requires its full term in the paper. Ambiguous/multiple definitions and mathematical/numeric glosses are rejected. Every inventory records explicit versus reference-grounded provenance. This is a finite heuristic, not a complete entity-linking model. Every paper’s definition inventory and source hash are saved.</p>
 <h2>How joint planning differs from the old search</h2><p>The planner starts from the original full-source six-gram matches and groups connected copied-word intervals. It generates candidates on the original fragment, preventing repeated self-expansion. Within each component it keeps up to 24 candidate plans and explores up to six insertions. An intermediate state may leave copied-word coverage unchanged. Only the final plan must make a strict improvement. Overlapping explanations of the same quantity are incompatible; each summary has a 3% added-word budget.</p><p>The matched catalog in the greedy arm provides the search ablation. Its actual copied-word coverage is {g['overlap']['narrative']['mean_covered_fraction_six_plus']:.4%}, versus {j['overlap']['narrative']['mean_covered_fraction_six_plus']:.4%} for joint planning. Search statistics, not the illustrative fixture, establish how much this helps on the 97 real papers.</p><p>A targeted 20-word fixture verifies the former failure: neither legal unit-name insertion alone reduces covered original words, but both together break the middle segment. The greedy planner leaves the fixture unchanged; joint planning finds the two-insertion solution. {protocol['smoke_checks']} catalog/provenance/adversarial checks pass.</p>
 <h2>Actual plans with zero-gain intermediate steps</h2>{evidence if evidence else '<p>No such component plan was committed on the real cohort; the counterexample works, but it does not establish corpus-level gains.</p>'}
 <h2>Rule and search counts</h2>{rule_tables}
 <h2>Meaning and QA evaluation</h2><p>The unchanged Qwen2.5-7B student answers every original MCQ under fixed prompts, parser, sampling and paired seeds. Exact same contexts reuse saved predictions; this run needed {r['qa']['unique_calls']} new student calls. A pinned 27B teacher compares all final original/edited sentence pairs with thinking disabled, using exact-pair caching. It judges scope, certainty, quantity, reference, causal meaning and grammar. Automatic labels are not human equivalence guarantees.</p><p>Numbers, unit quantities, Greek variables, operators, acronyms, modal words, negation and an explicit epistemic-word inventory must be preserved. Every original character and normalized word remains in order; original mathematical spans remain contiguous. Those checks do not prove general semantic equivalence. The 97-paper cohort has been reused repeatedly and this is exploratory; future generalization needs fresh papers.</p>
 <h2>Reviewer controls, including failures</h2><div class="scroll"><table><thead><tr><th>Before</th><th>After</th><th>Expected</th><th>Actual label</th></tr></thead><tbody>{controls}</tbody></table></div>
 <h2>Inspect actual edits and failure flags</h2><p class="small">Illustrative pairs sorted to expose non-equivalent labels first. Complete pairs and raw judgments remain in the runtime artifacts. This selection is not an error-rate sample.</p>{samples}
 <h2>Compute and reproduction</h2><p>The additional stage is CPU-only at deployment, assuming its work overlaps the existing generator/repair. CPU extrapolations include warm source indexing, candidate generation/search and primary audits; they exclude bulk I/O and worker startup. The frozen baseline still includes the existing one-Gemma repair and earlier CPU stages. The preceding generator-plus-Gemma-repair planning estimate remains 73,699 GH200 GPUh/38M at 85% useful capacity.</p><p>Offline benchmark reservation: {esc(allocation)}. All earlier outputs remain available in the v1 and v2 runtime directories; the corresponding local HTML snapshots end in V1_97 and V2_97.</p><pre>cd {esc(ROOT)}
/e/fscratch/reformo/schuhmann1/scientific-procedural-synonyms-20261012/env/bin/python code/repair_saved.py \\
  --summary saved-baseline.json --source full-paper.txt \\
  --mode system_core --out elaborated.json</pre><p>Other reproducible modes: system_greedy and system_joint. <a href="{esc(ROOT/'outputs/results_and_scaling.json')}">Full scores and timings</a> · <a href="{esc(ROOT/'outputs/overlap/COPY_OVERLAP.md')}">Full-source overlap audit</a> · <a href="{esc(ROOT/'inputs/protocol.json')}">Frozen rules and code hashes</a>.</p>'''
 dest=HOME/'POST_GEMMA_INSERTION_SYSTEM_97_2026-10-24.html';dest.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Alexandria insertion system</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>')
 for p in (ROOT/'code').glob('*.py'):(HERE/'code').mkdir(exist_ok=True);shutil.copy2(p,HERE/'code'/p.name)
 for name in ['vendor','resources']:
  if not (HERE/name).exists():(HERE/name).symlink_to(ROOT/name,target_is_directory=True)
 for p in [ROOT/'outputs/results_and_scaling.json',ROOT/'inputs/protocol.json',ROOT/'outputs/allocation_accounting.json']:shutil.copy2(p,HERE/p.name)
 write(HERE/'artifact_paths.json',dict(runtime_root=str(ROOT),home_html=str(dest),published_to_github=False));write(HERE/'joint_plan_examples.json',joint_examples)
 print('REPORT',dest);print('TABLE',json.dumps(allrows));print('ALLOCATION',allocation)
if __name__=='__main__':main()
