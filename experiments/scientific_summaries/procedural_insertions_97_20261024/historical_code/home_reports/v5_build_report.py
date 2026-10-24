"""Local report: protected terminology and expanded insertion system, v5."""
from pathlib import Path
from collections import Counter
import hashlib,html,json,re,shutil,subprocess

HERE=Path(__file__).resolve().parent
HOME=HERE.parent.parent
ROOT=Path('/e/scratch/reformo/schuhmann1/scientific-post-gemma-insertion-system-v5-97-20261024')
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def lines(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def esc(s):return html.escape(str(s))

def main():
 result=read(ROOT/'outputs/results_and_scaling.json');assert result['complete']
 protocol=read(ROOT/'inputs/protocol.json')
 assert all(hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==value for name,value in protocol['code_sha256'].items())
 previous=read(ROOT.with_name('scientific-post-gemma-insertion-system-v4-97-20261024')/'outputs/results_and_scaling.json')
 rows=[]
 def add(label,condition,r):
  v=r['conditions'][condition];n=v['overlap']['narrative'];m=v['insertion'];q=v['qa']
  rows.append(dict(label=label,condition=condition,mean_coverage=n['mean_covered_fraction_six_plus'],fixed_original_coverage=m['original_word_coverage_fixed_denominator'],pooled_coverage=n['pooled_covered_fraction_six_plus'],qa_correct=q['correct'],qa_total=q['total'],qa_accuracy=q['accuracy'],meaning_counts=v['meaning_counts'],strict_passes=n['strict_five_word_pass'],copied_original_positions=m['absolute_original_copied_words'],added_words=m['inserted_words'],rollbacks=m['rollbacks'],cpu_core_hours_38m=v['extra_cpu_core_hours_38m']))
 base=read(ROOT/'outputs/baseline_complete.json');qa=result['qa']['scores']['baseline']
 rows.append(dict(label='Frozen pipeline before insertion system',condition='baseline',mean_coverage=base['mean_coverage'],fixed_original_coverage=base['mean_coverage'],pooled_coverage=base['pooled_coverage'],qa_correct=qa['correct'],qa_total=qa['total'],qa_accuracy=qa['accuracy'],meaning_counts={},strict_passes=base['strict_passes'],copied_original_positions=base['absolute_copied_words'],added_words=0,rollbacks=base['rollbacks'],cpu_core_hours_38m=0))
 for version in ['v3','v4']:
  old=read(ROOT.with_name(f'scientific-post-gemma-insertion-system-{version}-97-20261024')/'outputs/results_and_scaling.json');add(f'{version}: joint planning','system_joint',old)
 for c,label in [('system_greedy','v5: greedy planning'),('system_joint','v5: joint planning'),('system_core','v5: conservative catalog')]:add(label,c,result)
 write(HERE/'comparison.json',rows)
 job=read(ROOT/'outputs/status.json')['job_id']
 out=subprocess.run(['sacct','-X','-j',str(job),'--noheader','--parsable2','--format=JobID,State,ElapsedRaw,AllocTRES'],check=True,capture_output=True,text=True).stdout.strip().splitlines()[0].split('|')
 assert out[1]=='COMPLETED',out;gpus=int(re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',out[3]).group(1))
 accounting=dict(job_id=job,state=out[1],elapsed_seconds=int(out[2]),allocated_gh200_gpus=gpus,reserved_gpu_hours=int(out[2])*gpus/3600,production_added_gpu_hours=0,scope='Actual full allocation, including startup, CPU processing, QA/review, idle GPUs and final audits. Added deployment stage is CPU-only, after the existing generator and Gemma repair.')
 write(ROOT/'outputs/allocation_accounting.json',accounting);write(HERE/'allocation_accounting.json',accounting)
 table=''.join(f'<tr><th>{esc(v["label"])}</th><td>{v["mean_coverage"]:.4%}</td><td>{v["fixed_original_coverage"]:.4%}</td><td>{v["pooled_coverage"]:.4%}</td><td>{v["qa_correct"]}/{v["qa_total"]}<br>{v["qa_accuracy"]:.2%}</td><td>{v["strict_passes"]}/97</td><td>{esc(v["meaning_counts"])}</td></tr>' for v in rows)
 cards=[c for c in lines(ROOT/'outputs/reviews/post.jsonl') if c['condition']=='system_joint']
 flags=[c for c in cards if c['judgment']['severity'] not in {'unchanged_or_equivalent','awkward_only'}]
 selected=flags+[c for c in cards if c not in flags][:12]
 samples=''.join(f'<article><h3>{esc(c["document_id"])} · {esc(c["judgment"]["severity"])}</h3><div class="two"><div><b>Before</b><p>{esc(c["before"])}</p></div><div><b>After</b><p>{esc(c["after"])}</p></div></div><p class="small">Automatic reviewer: {esc(c["judgment"]["reason"])}</p></article>' for c in selected)
 controls=''.join(f'<tr><td>{esc(c["before"])}</td><td>{esc(c["after"])}</td><td>{esc(c["expected"])}</td><td>{esc(c["judgment"]["severity"])}</td></tr>' for c in result['meaning_controls'])
 details=''
 for c,v in result['conditions'].items():
  m=v['insertion'];n=v['overlap']['narrative']
  details+=f'<h3>{esc(c)}</h3><p>{m["changed_summaries"]}/97 summaries changed; {m["rollbacks"]} rollbacks; {m["inserted_words"]:,} added narrative words. {m["absolute_original_copied_words"]:,} original covered positions remain out of {m["baseline_copied_words"]:,} before the insertion system. There are {m["narrative_passages_6plus"]:,} disjoint ≥6-word passages; the maximum match is {n["maximum_contiguous_match_words"]} words.</p><div class="two"><div><b>Accepted rule counts</b><pre>{esc(json.dumps(m["rule_counts"],indent=2))}</pre></div><div><b>Search counters</b><pre>{esc(json.dumps(m["search"],indent=2))}</pre></div></div><p>Warm CPU work including source indexing and primary audits: {m["cpu_seconds_including_index"]:.3f} core-seconds/97 papers, linearly extrapolated to {v["extra_cpu_core_hours_38m"]:,.0f} CPU core-hours/38M. Bulk I/O and worker startup are excluded; source length and copy density affect scaling.</p>'
 old=previous['conditions']['system_joint']['insertion'];new=result['conditions']['system_joint']['insertion']
 reduction=old['absolute_original_copied_words']-new['absolute_original_copied_words']
 j=result['conditions']['system_joint'];n=j['overlap']['narrative']
 css='body{margin:0;background:#f4f5ee;color:#173d34;font:16px/1.65 system-ui,sans-serif}main{max-width:1360px;margin:auto;padding:40px 24px 80px}h1{font:clamp(32px,5vw,57px)/1.15 Georgia,serif}h2{font:30px Georgia,serif;margin-top:40px}.lead{font-size:20px;max-width:1060px}.note{background:#fff0ca;padding:22px;border-radius:14px}.scroll{overflow:auto;background:white;border-radius:14px}table{border-collapse:collapse;width:100%;min-width:1080px}th,td{padding:13px;text-align:left;vertical-align:top;border-bottom:1px solid #d5dfd0}thead{background:#e1ebdb}.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{background:white;padding:22px;border-radius:14px;margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}.small{font-size:14px;color:#50665d}a{color:#08674d}p,td{overflow-wrap:anywhere}@media(max-width:750px){.two{grid-template-columns:1fr}}'
 body=f'''<h1>Version 5: rejected exploratory provenance rule</h1><p class="note"><b>Do not deploy this version.</b> Manual review and the automatic reviewer found an invalid from-complement rewrite in a biomedical sentence. Version 6 adds a governor safeguard and stricter handling of mixed-case material names. This page preserves all failed-run outputs and scores.</p><p class="lead">Version 5: {protocol['rule_family_count']} finite rule families, paper-grounded definitions, exact measurement annotations and grammatical insertions. Same 97 papers, full untruncated sources and 970 original questions per condition.</p>
 <p class="note"><b>Read the metric precisely.</b> A narrative word is covered if it belongs to a contiguous match of at least six normalized source words. The headline is the unweighted mean of 97 narrative fractions. Fixed original denominators prevent padding from masquerading as progress; pooled corpus coverage is also reported. Added glosses and grammar markers change literal strings while retaining original wording. This is surface elaboration, not independently authored conceptual paraphrasing.</p>
 <h2>Frozen comparisons</h2><div class="scroll"><table><thead><tr><th>Condition</th><th>Mean covered words ≥6</th><th>Original covered / original length</th><th>Pooled coverage</th><th>QA accuracy</th><th>Entire narrative ≤5</th><th>Automatic meaning labels</th></tr></thead><tbody>{table}</tbody></table></div>
 <p>Joint v5 versus joint v4: {reduction:+,} fewer original word positions remain covered by long matches. The joint v5 mean is {n['mean_covered_fraction_six_plus']:.4%}; its fixed-original-denominator coverage is {new['original_word_coverage_fixed_denominator']:.4%}. These are distinct from eliminating every long passage, which {n['strict_five_word_pass']}/97 narratives achieve.</p>
 <h2>What was improved</h2><p><b>Terminology protection.</b> Multiword definitions explicitly present in the current paper, together with a finite list of technical expressions such as “regions of interest”, “degrees of freedom”, “confidence interval”, “standard error” and “false discovery rate”, may no longer be split by insertions. This addresses an actual v4 meaning flag. Original formula, code and quotation spans retain their separate protection.</p>
 <p><b>Definitions and text extraction.</b> Unicode NFKC normalization handles PDF ligatures before definition matching and duplicate-gloss detection. Mixed-case, plural and hyphenated initialisms can align with source definitions. A mixed-case initialism still requires all its letters in order and a minimum of matching word initials; ambiguity blocks expansion. Additional fixed reference aliases require their canonical full term to occur in this same source, and explicit source conflicts take precedence. Source hashes, inventory and provenance remain saved.</p>
 <p><b>Grammatical coverage.</b> The methodological gerund vocabulary now includes examining, combining, selecting, clustering, normalizing and other concrete operations. Purpose constructions remain restricted to verbs expressing design/use/engineering. A fixed set of head nouns permits provenance relatives, such as “journals that are from …”. Singular numeric reference labels may become “Figure number 1”; the identifier is never split or respelled. Bare parallel passives and infinitives may repeat an already present auxiliary or “to”; intervening negation, objects, adverbs and hedges make those rules ineligible.</p>
 <p><b>Measurements.</b> Scientific notation and guarded numeric ranges remain exactly written as before. Unit glosses include angstroms, context-checked flux/current/energy/force/length symbols, spaced percent units and enumerated inverse units. Range annotations use plural unit names. Compact K/m/Å require explicit physical cues. Compound units remain whole. No value is converted or rescaled, and uncertain symbols remain ineligible.</p>
 <h2>Planner and preservation checks</h2><p>Candidates are generated once on the original fragment. Connected copied intervals are searched jointly, admitting individually zero-gain intermediate steps. A committed plan must strictly reduce covered original words, covered total words, matching source six-grams and covered fraction. Each summary has a 3% added-word cap. Every original character and normalized word must remain in order; numerical, unit, Greek-variable, operator, acronym, modal, negation and epistemic signatures must agree. Evidence quotes and bibliographic metadata are untouched and audited separately.</p>
 <p>{protocol['smoke_checks']} targeted tests passed before the 14 source-file hashes were frozen. All cohort source-copy audits ran on the compute allocation using the established whitespace-word normalization and a separate punctuation-split diagnostic. All 97 sources and original 970 gold/question slots are retained. Failures and rollbacks remain visible.</p>
 <h2>QA and meaning</h2><p>The pinned Qwen2.5-7B student answers the unchanged MCQs under the original prompts, parser, sampling and paired seeds. Exact-context caching is the only prediction reuse; this job generated {result['qa']['unique_calls']} new student answers. No summaries are edited or selected using their new QA scores. A pinned 27B reviewer judges every final changed sentence/window pair and reuses only exact pairs. Its judgments are automatic, and known control failures remain shown below.</p>
 <p>The cohort has been reused repeatedly for exploratory development. Small QA differences do not demonstrate a generalization improvement. Exact guards and automatic reviewer labels do not guarantee semantic equivalence. All joint-arm non-equivalent/unparsed flags are displayed below, alongside examples labeled equivalent.</p>
 <h2>Actual rules, search and CPU costs</h2>{details}
 <h2>Reviewer controls, including failures</h2><div class="scroll"><table><thead><tr><th>Original</th><th>Edited</th><th>Expected</th><th>Actual automatic label</th></tr></thead><tbody>{controls}</tbody></table></div>
 <h2>All joint-arm flags and sample edits</h2>{samples}
 <h2>Reservation and reproduction</h2><p>The additional deployment repair stage is CPU-only and follows the existing generator, one Gemma repair and earlier procedural stages. The complete test reservation used {accounting['reserved_gpu_hours']:.3f} GPU-hours: {gpus} allocated GH200 GPUs × {accounting['elapsed_seconds']} seconds, including loading, idle allocation, QA/review and final audits.</p>
 <pre>/e/fscratch/reformo/schuhmann1/scientific-procedural-synonyms-20261012/env/bin/python \\
 {ROOT}/code/repair_saved.py \\
 --summary saved-baseline.json --source full-paper.txt \\
 --mode system_joint --out repaired.json</pre>
 <p><a href="{ROOT}/outputs/results_and_scaling.json">Full scores and timings</a> · <a href="{ROOT}/outputs/overlap/COPY_OVERLAP.md">Full-source audit</a> · <a href="{ROOT}/inputs/protocol.json">Frozen protocol and code hashes</a> · <a href="{HOME}/POST_GEMMA_INSERTION_SYSTEM_V4_97_2026-10-24.html">Version 4 report</a>. Local files only; no GitHub publication.</p>'''
 dest=HOME/'POST_GEMMA_INSERTION_SYSTEM_V5_97_2026-10-24.html'
 snapshot=HOME/'POST_GEMMA_INSERTION_SYSTEM_V4_97_2026-10-24.html'
 if not snapshot.exists():shutil.copy2(HOME/'POST_GEMMA_INSERTION_SYSTEM_97_2026-10-24.html',snapshot)
 dest.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Procedural scientific summary repair v5</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>')
 (HERE/'code').mkdir(exist_ok=True)
 for f in (ROOT/'code').glob('*.py'):shutil.copy2(f,HERE/'code'/f.name)
 for name in ['vendor','resources']:
  if not (HERE/name).exists():(HERE/name).symlink_to((ROOT/name).resolve(),target_is_directory=True)
 for f in [ROOT/'outputs/results_and_scaling.json',ROOT/'inputs/protocol.json']:shutil.copy2(f,HERE/f.name)
 write(HERE/'artifact_paths.json',dict(runtime_root=str(ROOT),home_html=str(dest),published_to_github=False))
 print('REPORT',dest);print('TABLE',json.dumps(rows));print('ALLOCATION',accounting)

if __name__=='__main__':main()
