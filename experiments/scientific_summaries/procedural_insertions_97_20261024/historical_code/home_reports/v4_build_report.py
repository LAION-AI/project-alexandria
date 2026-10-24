"""Build the local v4 report from completed, frozen benchmark artifacts."""
import hashlib,html,json,re,shutil,subprocess
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
HOME=HERE.parent.parent
ROOT=Path('/e/scratch/reformo/schuhmann1/scientific-post-gemma-insertion-system-v4-97-20261024')
PREVIOUS=ROOT.with_name('scientific-post-gemma-insertion-system-v3-97-20261024')
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def esc(s):return html.escape(str(s))
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def main():
 r=read(ROOT/'outputs/results_and_scaling.json');assert r['complete']
 protocol=read(ROOT/'inputs/protocol.json')
 assert all(hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==value for name,value in protocol['code_sha256'].items())
 previous=read(PREVIOUS/'outputs/results_and_scaling.json')
 overlap=read(ROOT/'outputs/overlap/copy_overlap.json')
 rows=[]
 for c,label in [('baseline','Frozen pipeline before insertion system'),('legacy_grammar','Earlier narrow grammar'),('v3_joint','Previous best: v3 joint'),('system_greedy','v4: expanded catalog, greedy'),('system_joint','v4: expanded catalog, joint'),('system_core','v4: conservative catalog, joint')]:
  if c=='v3_joint':v=previous['conditions']['system_joint'];m=v['insertion'];n=v['overlap']['narrative'];q=v['qa'];meaning=v['meaning_counts']
  elif c in r['conditions']:v=r['conditions'][c];m=v['insertion'];n=v['overlap']['narrative'];q=v['qa'];meaning=v['meaning_counts']
  else:
   base=ROOT if c=='baseline' else PREVIOUS
   m=read(base/f'outputs/{c}_complete.json') if (base/f'outputs/{c}_complete.json').exists() else {}
   if c=='legacy_grammar':m=read(ROOT.parent/'scientific-post-gemma-safe-insertions-guarded-97-20261024/outputs/grammar_insert_complete.json')
   n=overlap['conditions'][c]['narrative'];q=r['qa']['scores'][c];meaning={}
  fixed=m.get('original_word_coverage_fixed_denominator',n['mean_covered_fraction_six_plus'])
  rows.append(dict(condition=c,label=label,coverage=n['mean_covered_fraction_six_plus'],original_fixed_coverage=fixed,pooled_coverage=n['pooled_covered_fraction_six_plus'],qa_correct=q['correct'],qa_total=q['total'],qa_accuracy=q['accuracy'],meaning_counts=meaning,strict_passes=n['strict_five_word_pass'],copied_words=m.get('absolute_copied_words'),added_words=m.get('inserted_words'),rollbacks=m.get('rollbacks')))
 write(HERE/'comparison.json',rows)
 job=read(ROOT/'outputs/status.json')['job_id']
 p=subprocess.run(['sacct','-X','-j',str(job),'--noheader','--parsable2','--format=JobID,State,ElapsedRaw,AllocTRES'],check=True,capture_output=True,text=True)
 a=p.stdout.strip().splitlines()[0].split('|');assert a[1]=='COMPLETED',a
 gpu=int(re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',a[3]).group(1))
 accounting=dict(job_id=a[0],state=a[1],elapsed_seconds=int(a[2]),allocated_gpus=gpu,reserved_gpu_hours=int(a[2])*gpu/3600,production_added_gpu_hours=0,scope='Full allocation, including CPU stage, loading, QA/review, idle GPUs and final audits. Production insertions are CPU-only; the preceding generator and Gemma repair still cost GPUs.')
 write(ROOT/'outputs/allocation_accounting.json',accounting);write(HERE/'allocation_accounting.json',accounting)
 table=''.join(f'<tr><th>{esc(v["label"])}</th><td>{v["coverage"]:.4%}</td><td>{v["original_fixed_coverage"]:.4%}</td><td>{v["pooled_coverage"]:.4%}</td><td>{v["qa_correct"]}/{v["qa_total"]}<br>{v["qa_accuracy"]:.2%}</td><td>{v["strict_passes"]}/97</td><td>{esc(v["meaning_counts"])}</td></tr>' for v in rows)
 controls=''.join(f'<tr><td>{esc(c["before"])}</td><td>{esc(c["after"])}</td><td>{esc(c["expected"])}</td><td>{esc(c["judgment"]["severity"])}</td></tr>' for c in r['meaning_controls'])
 cards=lines(ROOT/'outputs/reviews/post.jsonl');cards=[c for c in cards if c['condition']=='system_joint']
 flags=[c for c in cards if c['judgment']['severity'] not in ['unchanged_or_equivalent','awkward_only']]
 examples=flags+[c for c in cards if c not in flags][:10]
 samples=''.join(f'<article><h3>{esc(c["document_id"])} · {esc(c["judgment"]["severity"])}</h3><div class="two"><div><b>Before</b><p>{esc(c["before"])}</p></div><div><b>After</b><p>{esc(c["after"])}</p></div></div><p class="small">Automatic reviewer: {esc(c["judgment"]["reason"])}</p></article>' for c in examples)
 metrics=''
 for c,v in r['conditions'].items():
  m=v['insertion']
  metrics+=f'<h3>{esc(c)}</h3><p>{m["changed_summaries"]}/97 changed; {m["rollbacks"]} rolled back; {m["inserted_words"]:,} added words. {m["absolute_original_copied_words"]:,} original word positions remain covered, versus {m["baseline_copied_words"]:,} before the system.</p><div class="two"><pre>{esc(json.dumps(m["rule_counts"],indent=2))}</pre><pre>{esc(json.dumps(m["search"],indent=2))}</pre></div><p>Warm CPU time, including source indexing and primary audits: {m["cpu_seconds_including_index"]:.2f} core-seconds/97 papers, linearly extrapolated to {v["extra_cpu_core_hours_38m"]:,.0f} CPU core-hours/38M. This excludes bulk I/O and worker startup, and assumes similar source lengths.</p>'
 old=previous['conditions']['system_joint']['insertion'];new=r['conditions']['system_joint']['insertion']
 saved=old['absolute_original_copied_words']-new['absolute_original_copied_words']
 css='body{margin:0;background:#f3f5ee;color:#173e34;font:16px/1.65 system-ui,sans-serif}main{max-width:1350px;margin:auto;padding:40px 24px 80px}h1{font:clamp(34px,5vw,57px)/1.15 Georgia,serif}h2{font:30px Georgia,serif;margin-top:40px}.lead{font-size:20px;max-width:1050px}.note{background:#fff0c8;padding:20px;border-radius:14px}.scroll{overflow:auto;background:white;border-radius:14px}table{border-collapse:collapse;width:100%;min-width:1080px}th,td{padding:13px;text-align:left;vertical-align:top;border-bottom:1px solid #d6dfd0}thead{background:#e1ebda}.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{background:white;padding:22px;border-radius:14px;margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}.small{font-size:14px;color:#52685e}a{color:#08674c}td,p{overflow-wrap:anywhere}@media(max-width:750px){.two{grid-template-columns:1fr}}'
 body=f'''<h1>Expanded procedural repair: version 4</h1>
 <p class="lead">A frozen comparison on all 97 scientific papers and all 970 original questions. Exact-value unit explanations, paper-grounded acronym definitions and bounded grammatical insertions. All original wording remains in order.</p>
 <p class="note"><b>What the percentage means.</b> A word is covered if it belongs to a contiguous source match of at least six normalized words. The headline percentage is the unweighted mean of 97 narrative fractions. A separate fixed-denominator column counts covered <em>original</em> words divided by the original narrative length, so padding alone cannot deliver an improvement. Pooled corpus coverage is also shown. A mean below 5% does not mean every summary has no long copied passage.</p>
 <h2>Results</h2><div class="scroll"><table><thead><tr><th>Frozen condition</th><th>Mean coverage ≥6</th><th>Original coverage / original length</th><th>Pooled corpus coverage</th><th>QA accuracy</th><th>Entire narrative ≤5 words</th><th>Automatic meaning labels</th></tr></thead><tbody>{table}</tbody></table></div>
 <p>The joint v4 arm removes {saved:,} additional original covered word positions versus v3 joint. All six reported conditions remain visible; no arm is chosen by its QA score. Exact numbers, bootstrap intervals and QA predictions are retained in the JSON artifacts.</p>
 <h2>What changed</h2><p>The catalog has {protocol['rule_family_count']} families. This version adds bounded duration phrases without changing their bounds, fixed postnominal “of interest” relatives, and omitted object relative markers before an explicit passive clause. The reduced-relative noun vocabulary is expanded. No certainty, frequency, causality, exclusivity or uncertainty fillers are added.</p>
 <p>Additional exact unit names include femtoseconds, picoseconds, terahertz, picograms, pressure and context-checked parts-per-million/billion notation. A finite list of compound units such as mg/mL, g/L and m/s is explained as a whole, with no conversion or rescaling. Molarity, speed and ambiguous single-letter symbols retain positive-context checks. Pure numeric percentage parentheses remain contiguous and receive their explanation outside the original parentheses; prose parentheses use square explanatory brackets. Existing adjacent explanations are not repeated.</p>
 <p>Hyphenated initialisms such as ICP-OES can now align with their source definitions. Singular/plural variants are consolidated only for a fixed set of regular technical head nouns; genuinely conflicting definitions still prevent expansion. Each glossary entry records its source and provenance. The conservative arm omits acronym glosses and complex reduced relatives.</p>
 <h2>Selection and integrity</h2><p>Candidates are generated once on each original fragment. The component-wise joint planner can combine individually zero-gain insertions, but commits only a plan that decreases original covered words, total covered words, source six-gram hits and covered fraction. A 3% whole-summary added-word limit applies. Every original character and normalized word must remain in order, original math/quote/code spans must remain contiguous, and numerical, unit, operator, Greek-variable, acronym, modal, negation and epistemic signatures must agree. Evidence quotes and metadata are left untouched and audited separately.</p>
 <p>{protocol['smoke_checks']} targeted checks passed before code hashes were frozen and the full job was submitted. Every final summary was checked against its own full untruncated source using the established whitespace-word metric and a separate punctuation-split diagnostic. All 97 source hashes and all original question/gold slots are retained. Rollbacks remain visible.</p>
 <h2>Meaning and QA</h2><p>The pinned Qwen2.5-7B student uses the original prompts, parser, sampling and paired seeds on every original MCQ. Only exact-context predictions are reused; this run generated {r['qa']['unique_calls']} new student answers. The same pinned 27B reviewer judges all changed sentence/window pairs, reusing only exact pairs. Labels are automatic and do not prove human equivalence; all non-equivalent joint-arm flags are shown below.</p>
 <p>This 97-paper benchmark has been repeatedly reused for development. The small QA differences are exploratory and do not establish a generalization gain. Glosses and function words change literal strings while retaining the original wording; they are surface elaborations, not independently written conceptual paraphrases. Meaningful reuse and appropriate attribution still matter.</p>
 <h2>Rule counts and timings</h2>{metrics}
 <h2>Reviewer controls and limitations</h2><div class="scroll"><table><thead><tr><th>Before</th><th>After</th><th>Expected</th><th>Actual automatic label</th></tr></thead><tbody>{controls}</tbody></table></div><p>The reviewer is known to undercall some removed hedges or added certainty, and to miss one mixed-dimension scope error in the negative controls. Exact protections therefore remain mandatory. All control failures are displayed.</p>
 <h2>Inspect changes and all joint-arm flags</h2>{samples}
 <h2>Compute and reproduction</h2><p>The new production stage uses CPUs and adds no model inference. It follows the existing generated summary, one Gemma repair and preceding procedural stages; their GPU cost remains. The complete benchmark reservation was {accounting['reserved_gpu_hours']:.3f} GPU-hours ({gpu} allocated GH200 GPUs × {accounting['elapsed_seconds']} seconds), including startup and idle allocation.</p>
 <pre>/e/fscratch/reformo/schuhmann1/scientific-procedural-synonyms-20261012/env/bin/python \\
 {ROOT}/code/repair_saved.py \\
 --summary saved-baseline.json --source full-paper.txt \\
 --mode system_joint --out repaired.json</pre>
 <p><a href="{ROOT}/outputs/results_and_scaling.json">Scores and timings</a> · <a href="{ROOT}/outputs/overlap/COPY_OVERLAP.md">Full-source overlap report</a> · <a href="{ROOT}/inputs/protocol.json">Frozen protocol and code hashes</a> · <a href="{HOME}/POST_GEMMA_INSERTION_SYSTEM_V3_97_2026-10-24.html">Previous v3 report</a>. Local artifacts only; no GitHub publication.</p>'''
 dest=HOME/'POST_GEMMA_INSERTION_SYSTEM_97_2026-10-24.html'
 if dest.exists() and not (HOME/'POST_GEMMA_INSERTION_SYSTEM_V3_97_2026-10-24.html').exists():shutil.copy2(dest,HOME/'POST_GEMMA_INSERTION_SYSTEM_V3_97_2026-10-24.html')
 dest.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scientific summary insertion system v4</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>')
 (HERE/'code').mkdir(exist_ok=True)
 for p in (ROOT/'code').glob('*.py'):shutil.copy2(p,HERE/'code'/p.name)
 for name in ['vendor','resources']:
  if not (HERE/name).exists():(HERE/name).symlink_to((ROOT/name).resolve(),target_is_directory=True)
 for p in [ROOT/'outputs/results_and_scaling.json',ROOT/'inputs/protocol.json']:shutil.copy2(p,HERE/p.name)
 write(HERE/'artifact_paths.json',dict(runtime_root=str(ROOT),home_html=str(dest),published_to_github=False))
 print('REPORT',dest);print('COMPARISON',json.dumps(rows));print('ACCOUNTING',accounting)

if __name__=='__main__':main()
