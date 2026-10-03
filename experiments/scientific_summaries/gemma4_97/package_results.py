"""Package measured Gemma results and extend the shared GH200 comparison."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'ornith_dflash_97'))
from package_results import archive,table,write_csv,write_json,compressed_copy

MODELS=('gemma4_e4b','gemma4_12b')
NAMES={'gemma4_e4b':'Gemma 4 E4B IT','gemma4_12b':'Gemma 4 12B IT'}
KEYS=('factual_accuracy','coverage','clarity','faithfulness','scientific_precision')

def load(path):
    return json.loads(path.read_text())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    root=parser.parse_args().run_dir.resolve()
    manifest=load(root/'inputs/experiment_manifest.json')
    report=load(root/'outputs/report.json');qa=load(root/'outputs/qa-results.json')
    audit=load(root/'outputs/final_audit.json')
    if not audit['passed'] or audit['conditions']!=14:
        raise ValueError('Fourteen complete conditions must pass the final audit')
    artifact=HERE/'artifacts';artifact.mkdir(exist_ok=True)
    for name in ['report.json','final_audit.json','performance_recipe.json','complete.json','code_snapshot.json',
                 'qa_code_snapshot.json','schema_smoke_code_snapshot.json','recovery_intervention.json','schema_reconciliation.json']:
        shutil.copyfile(root/'outputs'/name,artifact/name)
    for model in MODELS:
        shutil.copyfile(root/'outputs'/('first_pass-'+model+'.json'),artifact/('first_pass-'+model+'.json'))
    completion=load(root/'outputs/complete.json');job=completion['job_id'];allocations=[]
    for allocation in completion['allocation_job_ids']:
        accounting=subprocess.check_output(['sacct','-X','-j',allocation,'--format=JobIDRaw,State,ElapsedRaw,AllocTRES','--parsable2','--noheader'],text=True)
        row=next(line.split('|') for line in accounting.splitlines() if line.split('|')[0]==allocation)
        gpus=int(next(x.split('=')[1] for x in row[3].split(',') if x.startswith('gres/gpu=')))
        allocations.append(dict(job_id=allocation,state=row[1],elapsed_seconds=int(row[2]),allocated_gpus=gpus,
            allocated_gpu_hours=int(row[2])*gpus/3600))
    if allocations[-1]['state']!='COMPLETED':raise ValueError('QA resume allocation is not complete')
    bill=dict(job_id=job,state=allocations[-1]['state'],elapsed_seconds=sum(a['elapsed_seconds'] for a in allocations),
        allocated_gpus=gpus,allocations=allocations,allocated_gpu_hours=sum(a['allocated_gpu_hours'] for a in allocations),
        note='Complete scheduler allocation includes startup, benchmarks, both models, recovery, QA and idle GPUs; overlapping steps counted once.')
    write_json(artifact/'gpu_accounting.json',bill)
    compressed_copy(root/'outputs/qa-results.json',artifact/'qa-results.json.gz')
    provenance=HERE/'provenance';provenance.mkdir(exist_ok=True)
    for name in ['experiment_manifest.json','judge-weights-manifest.json','judge-config.json','token_preflight.json','submission.json','qa_submission.json']+[
            model+suffix for model in MODELS for suffix in ['-weights-manifest.json','-config.json']]:
        shutil.copyfile(root/'inputs'/name,provenance/name)
    shutil.copytree(root/'code',provenance/'source_code',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(root/'rescue_code',provenance/'schema_rescue_and_qa_code',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    archive([(f,str(f.relative_to(root))) for folder in ['logs','outputs/server_accounting']
        for f in (root/folder).rglob('*') if f.is_file()],artifact/'runtime_logs.tar.gz')
    with (root/'inputs/paper_names.tsv').open(newline='') as handle:names=list(csv.reader(handle,delimiter='\t'))
    with (HERE/'paper_names.tsv').open('w',newline='') as handle:
        csv.writer(handle,delimiter='\t',quoting=csv.QUOTE_ALL,lineterminator='\n').writerows(names)
    throughput=[];best={}
    for model in MODELS:
        result=load(root/'outputs/throughput'/(model+'.json'))
        if not result.get('complete'):raise ValueError('Incomplete throughput '+model)
        shutil.copyfile(root/'outputs/throughput'/(model+'.json'),artifact/('throughput-'+model+'.json'))
        for b in result['batches']:
            throughput.append(dict(model=model,runtime='ar',phase=b['phase'],batch=b['concurrency'],cache=b['cache_state'],
                seconds=b['seconds'],input_tokens=b['input_tokens'],output_tokens=b['completion_tokens'],
                output_tokens_per_second=b['output_tokens_per_second'],mean_request_seconds=b['request_latency_mean_seconds'],
                gpu_memory_used_total_utilization=b['gpu_memory']))
        directory=root/'outputs/throughput'/model
        archive([(f,str(f.relative_to(directory))) for f in directory.rglob('*') if f.is_file()],artifact/('throughput-traces-'+model+'.tar.gz'))
        best[model]={phase:max((b for b in result['batches'] if b['phase']==phase and b['cache_state']==state),
            key=lambda b:b['output_tokens_per_second']) for phase,state in [('generation','cold'),('correction','warm')]}
    write_csv(HERE/'throughput.csv',throughput)
    scores=[dict(condition=k,model=v['name'],correct=v['correct'],total=v['total'],accuracy=v['accuracy'],
        ci95_low=v['ci95'][0],ci95_high=v['ci95'][1],invalid=v['invalid']) for k,v in report['models'].items()]
    write_csv(HERE/'scores.csv',scores)
    timings=[dict(model=model,phase=k,**v) for model in MODELS for k,v in report['generation'][model]['phases'].items()]
    write_csv(HERE/'phase_timings.csv',timings)
    paper_scores=[];transitions={model:collections.Counter() for model in MODELS}
    for doc in qa['documents']:
        for label,value in doc['conditions'].items():
            paper_scores.append(dict(document_id=doc['document_id'],condition=label,
                correct=sum(next(iter(r['predictions'].values()))==r['gold'] for r in value['rows']),
                total=10,invalid=sum(next(iter(r['predictions'].values())) is None for r in value['rows']),
                latest_condition_seconds=value['elapsed_seconds'],context_sha256=value['context_sha256']))
        for model in MODELS:
            for x,y in zip(doc['conditions'][model+'_raw']['rows'],doc['conditions'][model+'_corrected']['rows']):
                a=next(iter(x['predictions'].values()))==x['gold'];b=next(iter(y['predictions'].values()))==y['gold']
                transitions[model][('correct' if a else 'wrong')+'_to_'+('correct' if b else 'wrong')]+=1
    write_csv(HERE/'paper_scores.csv',paper_scores)
    write_json(artifact/'correction_transitions.json',transitions)
    self_rows=[];stats={};assessments=[];cohort_times={}
    for model in MODELS:
        directory=root/'outputs/cohorts'/model/'documents'
        starts=[];ends=[]
        for folder in sorted(directory.iterdir()):
            if not folder.is_dir():continue
            doc=load(folder/'document.json')
            ends.append((folder/'document.json').stat().st_mtime)
            for call in doc['attempts']:
                if call.get('trace_file'):
                    stamp=int(Path(call['trace_file']).name.split('-')[0])/1e9
                    starts.append(stamp-call['elapsed_seconds'])
            if not doc.get('raw') or not doc.get('corrected'):raise ValueError('Missing summary')
            archive([(f,str(f.relative_to(folder))) for f in folder.rglob('*') if f.is_file()],artifact/'paper_traces'/model/(folder.name+'.tar.gz'))
            for phase in ['raw','corrected']:
                assessment=doc.get(phase+'_quality_assessment',{})
                assessments.append(dict(model=model,document_id=folder.name,phase=phase,assessment=assessment))
                self_rows.append(dict(model=model,document_id=folder.name,phase=phase,
                    review_failed=bool(assessment.get('review_failed')),verdict=assessment.get('verdict','unavailable'),
                    issues=len(assessment.get('issues',[])),missing_central_facts=len(assessment.get('missing_central_facts',[])),
                    **{k:assessment.get('scores',{}).get(k,'') for k in KEYS}))
        first=load(root/'outputs'/('first_pass-'+model+'.json'))
        cohort_times[model]=dict(first_pass_seconds=report['generation'][model]['completion']['elapsed_seconds'],
            first_pass_raw=first['raw'],first_pass_corrected=first['corrected'],
            recovered_papers=report['generation'][model]['completion']['source_recovered_papers'],
            full_source_only_trace_span_seconds=max(ends)-min(starts),
            span_definition='First recorded model request start (trace timestamp minus recorded latency) through last source-only document write; includes recovery, excludes startup/throughput/QA and initial token-count preflight.')
        for phase in ['raw','corrected']:
            rows=[r for r in self_rows if r['model']==model and r['phase']==phase]
            stats[model+'_'+phase]=dict(papers=len(rows),failed_reviews=sum(r['review_failed'] for r in rows),
                verdicts=dict(collections.Counter(r['verdict'] for r in rows)),issues=sum(r['issues'] for r in rows),
                missing_central_facts=sum(r['missing_central_facts'] for r in rows),
                scores={k:dict(n=len(v:=[r[k] for r in rows if r[k]!='']),mean=statistics.mean(v) if v else None,
                    histogram=dict(collections.Counter(v))) for k in KEYS})
    write_csv(HERE/'self_audit_scores.csv',self_rows)
    write_json(artifact/'quality_assessments.json',assessments)
    compressed_copy(artifact/'quality_assessments.json',artifact/'quality_assessments.json.gz')
    (artifact/'quality_assessments.json').unlink()
    write_json(artifact/'self_audit_aggregate.json',stats)
    write_json(artifact/'cohort_timings.json',cohort_times)
    sections=[
        '# Findings and results: Gemma 4 E4B IT and 12B IT on JUPITER',
        f'**Complete: both models, 97 held-out papers each, 970 immutable MCQs per condition; Slurm job {job}.** No LoRA or speculative draft is used.',
        '## Models and shared protocol',
        '[Gemma 4 E4B IT](https://huggingface.co/google/gemma-4-E4B-it) is the official small checkpoint requested as 4B IT: '
        '4.5B effective parameters, approximately 8B including embeddings. [Gemma 4 12B IT](https://huggingface.co/google/gemma-4-12B-it) '
        'is the 11.95B unified checkpoint. Both use unquantized **BF16**, native vLLM Gemma implementations, '
        'tensor parallel size 1, two independent one-GPU replicas per model, and the native `gemma4` reasoning parser. '
        'Only text is supplied. The 12B staged training checkpoint is reused read-only after full checksum verification.',
        table(['Checkpoint','Pinned revision'],[[manifest['models'][m]['id'],manifest['models'][m]['revision']] for m in MODELS]),
        f"Frozen protocol repository revision `{manifest['commit']}`; testset SHA256 `{manifest['testset_sha256']}`; "
        f"initial system-prompt SHA256 `{manifest['prompt_sha256']}`.",
        'The same original 19-field summary prompt, full untruncated dataset text, temperature 0.2, top-p 0.95, '
        '16,000 initial output tokens, 65,536 context and **thinking disabled** are used as in the '
        '[9B](../ornith_dflash_97/README.md) and [35B](../ornith35_dflash_97/README.md) runs. These sampling settings '
        'preserve the comparison protocol and differ from the Gemma model-card default temperature 1.0 / top-k 64. '
        'Generation, same-model source-only audit, full semantic correction, repository V3 field/quote repair, '
        'and post-correction audit use no questions, options, gold answers, QA rationale or QA evidence. '
        'Bounded recovery applies only to unusable outputs: up to 32,768 generation / 24,576 correction output tokens, '
        'presence penalty 1.5; original failures and all requests are retained. Optional citation rankings may be '
        'discarded only after bounded model repairs with full provenance. No papers are selected using QA scores.',
        '### Conditional finite-schema recovery and protocol difference',
        'The original common-protocol controller failed its completeness gate after both standard recovery '
        'rounds: E4B had 96 raw / 83 corrected artifacts and 12B had 97 raw / 94 corrected artifacts. '
        'Repeated full-output retries sometimes reached the token limit; E4B also emitted format commentary '
        'inside the scientific claims array. Additional **source-only** format rescue restores 14 E4B and '
        '3 12B corrected outputs, including the one missing E4B raw artifact. No valid existing raw or corrected '
        'artifact is replaced. The original failed proposals and the original failed scheduler allocation are retained.',
        'For the missing raw artifact, an already complete 19-field model proposal is reused after removing '
        'an explicitly non-scientific schema-note string from the claims array; all scientific claim objects '
        'and other fields are preserved. Corrected rescue starts with an existing V3 partial draft or complete '
        'source-only semantic correction. If no usable semantic correction exists, finite JSON-schema decoding '
        'requests the same 19 fields, at most 12 entries per grounded sequence / 5 claims, with 8,192 output '
        'tokens and no optional citation ranking. Temperature 0.2 / top-p 0.95 / thinking disabled are retained; '
        'presence penalty is 1.5 on schema-constrained rescue calls. Literal anchoring runs **after** field '
        'restoration with an enum of only supplied source-fragment indices or `drop`, followed by the unchanged '
        'strict validator and source-only self-audit. Every removed entry and original format note is preserved.',
        'This extra fallback is a **documented postprocessing difference** from the earlier Ornith/Qwen runs. '
        'The comparison measures deployed pipelines, not an isolated model or exactly identical recovery recipe. '
        'The throughput probes were completed before this fallback and measure ordinary autoregressive '
        'JSON-object decoding; complete workflow times and accounting include the failed retry loops and rescue. '
        'The QA phase is resumed in a second allocation after all 97 raw/corrected outputs are verified, '
        'without regenerating any baseline answers or using QA to choose repairs.',
        'The fixed answerer is **Qwen2.5-7B-Instruct BF16** revision `a09a35458c702b33eeacc393d103063234e8bc28`, '
        'vLLM 0.30.0: temperature 0.5, top-p 0.95, 100 output tokens, frequency/presence penalties 1.05, '
        'thinking disabled, four concurrent requests, at most five formatting attempts. The repository '
        'historical ASCII sanitizer and case-sensitive semicolon parser are unchanged; invalid answers count wrong. '
        'Ten previously completed conditions are reused only after exact source, context, question, option order, '
        'prompt and judge/protocol checks. **3,880 new answers** are scored; **13,580 total answers** are audited. '
        'Confidence intervals use 10,000 document-cluster bootstrap draws with seed 250219413.',
        'All 97 evaluation papers and outputs remain outside the 1,000-paper training workspace. The throughput '
        'tuning set is the same 64 frozen non-holdout sources. Synthetic MCQs were authored by gpt-6-luna; '
        'the length-filtered convenience sample has no human benchmark validation or completeness check against publisher PDFs.',
        '## All accuracy scores',
        table(['Condition','Correct / 970','Accuracy','95% paper-bootstrap CI','Invalid'],[
            [v['name'],v['correct'],f"{100*v['accuracy']:.2f}%",f"{100*v['ci95'][0]:.2f}–{100*v['ci95'][1]:.2f}%",v['invalid']]
            for v in report['models'].values()]),
        '## Paired Gemma QA differences',
        table(['Comparison','Difference (percentage points)','95% paired paper-bootstrap CI'],[
            [k,f"{100*v['difference']:+.2f}",f"{100*v['ci95'][0]:+.2f} to {100*v['ci95'][1]:+.2f}"]
            for k,v in report['paired_differences'].items() if k.startswith('gemma4_')]),
        '## Correction answer transitions',
        table(['Model','Wrong → correct','Correct → wrong','Correct → correct','Wrong → wrong'],[
            [NAMES[m],transitions[m]['wrong_to_correct'],transitions[m]['correct_to_wrong'],transitions[m]['correct_to_correct'],transitions[m]['wrong_to_wrong']] for m in MODELS]),
        '## Throughput: every measured batch',
        'One GH200 per measured runtime, BF16 native autoregressive decoding, continuous batching, prefix cache, '
        'chunked prefill (8,192 max batched tokens), compilation/CUDA graphs, 64 scheduled sequences and GPU '
        'memory utilization 0.90. Runtime uses Python 3.11.15, vLLM 0.30.0, PyTorch 2.13.0 and Transformers 5.18.0 '
        'on CUDA 13. Exact commands, selected kernels, GPU memory and before/after metrics are archived. '
        'All probes are bounded to 512 output tokens. Tokens/s divides actual emitted output tokens by batch wall '
        'time including prefill. Cold resets the prefix cache; warm repeats the whole input but hits depend on '
        'capacity and eviction. A first correction caching only the source can be slower. These probes do not '
        'measure complete-summary pipeline throughput.',
        table(['Model','Batch','Generation cold tok/s','Generation repeat tok/s','Correction cold tok/s','Correction repeat tok/s'],[
            [NAMES[m],batch]+[f"{next(r['output_tokens_per_second'] for r in throughput if r['model']==m and r['batch']==batch and r['phase']==p and r['cache']==c):.2f}"
            for p,c in [('generation','cold'),('generation','warm'),('correction','cold'),('correction','warm')]]
            for m in MODELS for batch in [1,4,8,16,32,64]]),
        'The production selector caps concurrent complete-paper workflows at 32 per replica. Each paper stays '
        'on its assigned replica across generation, review and correction. It chooses batches using the tuning '
        'probes, before any evaluation QA. The saved recipe and actual cohort config identify exact concurrency.',
        '## Complete-output request times and token usage',
        table(['Model','Phase','Calls','Output tokens','Input tokens','Mean seconds','Median seconds','P95 seconds'],[
            [NAMES[r['model']],r['phase'],r['calls'],r['completion_tokens'],r['prompt_tokens'],
             f"{r['mean_request_seconds']:.2f}",f"{r['median_request_seconds']:.2f}",f"{r['p95_request_seconds']:.2f}"] for r in timings]),
        'These are individual full-output request latencies at deployed concurrency, including queued work, '
        'not sums of GPU active time or complete cohort wall times. First-pass timers precede source recovery; '
        'reconciled counts in complete.json do not extend the original timer. Use scheduler accounting for '
        'the complete benchmark bill and trace timestamps for recovery intervals.',
        '## Cohort times including source-only recovery',
        table(['Model','First-pass seconds','First-pass raw / 97','First-pass corrected / 97','Recovered papers','Complete source-only trace span seconds'],[
            [NAMES[m],f"{v['first_pass_seconds']:.2f}",v['first_pass_raw'],v['first_pass_corrected'],v['recovered_papers'],
             f"{v['full_source_only_trace_span_seconds']:.2f}"] for m,v in cohort_times.items()]),
        'The complete trace span runs from the first recorded model request start (response trace timestamp '
        'minus recorded request latency) to the last source-only document write. It includes generation, '
        'audits, correction and all bounded recovery, but excludes model startup, throughput probes, initial '
        'token-count preflight and QA. It is an observed workflow span, not summed GPU active time. '
        'Reconciled first-pass completion files retain the original pre-recovery clock; the separate '
        'first-pass counts here prevent treating that clock as time to finish all 97 corrected papers.',
        '## Narrative lengths',
        table(['Condition','Mean words','Median words','Minimum','Maximum'],[
            [k,f"{v['mean_words']:.2f}",v['median_words'],v['min_words'],v['max_words']]
            for k,v in report['summary_lengths'].items()]),
        '## Source-only self-assessments',
        table(['Condition','Valid / 97','Failed / 97','Pass','Needs correction','Issues','Missing central facts'],[
            [k,97-v['failed_reviews'],v['failed_reviews'],v['verdicts'].get('pass',0),v['verdicts'].get('needs_correction',0),v['issues'],v['missing_central_facts']]
            for k,v in stats.items()]),
        table(['Condition','Rubric','Valid n','Mean / 5','Histogram'],[
            [label,key,value['n'],f"{value['mean']:.3f}" if value['mean'] is not None else 'unavailable',json.dumps(value['histogram'],sort_keys=True)]
            for label,s in stats.items() for key,value in s['scores'].items()]),
        'Self-assessments are fallible same-model judgments, not independent human factual labels. Failed '
        'audits are unavailable; means exclude failed assessments without imputing scores. All papers remain '
        'in QA. Literal quote/schema validation does not prove semantic entailment. Any actually emitted '
        'reasoning fields are preserved; thinking was disabled and hidden reasoning is not invented.',
        '## Allocation, audit and evidence',
        f"The complete resumed benchmark uses **{bill['allocated_gpu_hours']:.4f} allocated GPU-hours** "
        f"across **{bill['elapsed_seconds']} seconds of summed scheduler runtime**. Each allocation uses four GPUs. "
        'The original controller allocation failed its completeness guard; the resumed QA allocation completed. '
        'Source-only schema-rescue steps overlap the original allocation and are counted once.',
        table(['Slurm job','State','Allocated seconds','GPUs','GPU-hours'],[
            [a['job_id'],a['state'],a['elapsed_seconds'],a['allocated_gpus'],f"{a['allocated_gpu_hours']:.4f}"] for a in allocations]),
        f"All four new cohorts contain 97 outputs each. Final audit verifies **{audit['qa_prompt_hash_checks']:,} QA prompt hashes**, "
        f"**{audit['prediction_parser_checks']:,} parsed predictions**, immutable gold/option order and every corrected literal evidence ledger.",
        '## Conclusions',
        '\n\n'.join(f"**{NAMES[m]}:** raw {report['models'][m+'_raw']['correct']}/970 "
        f"({100*report['models'][m+'_raw']['accuracy']:.2f}%); corrected {report['models'][m+'_corrected']['correct']}/970 "
        f"({100*report['models'][m+'_corrected']['accuracy']:.2f}%). Best cold-generation probe "
        f"{best[m]['generation']['output_tokens_per_second']:.2f} tok/s (batch {best[m]['generation']['concurrency']}); "
        f"best repeat-input correction {best[m]['correction']['output_tokens_per_second']:.2f} tok/s "
        f"(batch {best[m]['correction']['concurrency']})." for m in MODELS),
        'On this QA-retention metric, **Ornith 9B remains ahead of both Gemma pipelines**: '
        '93.81% raw / 93.20% corrected versus E4B 85.15% / 84.64% and 12B 86.60% / 86.80%. '
        'Each paired Gemma-minus-Ornith-9B confidence interval is below zero. This is not a '
        'length-controlled model-capability comparison: mean raw narratives contain 775 words for E4B, '
        '918 for 12B and 2,548 for Ornith 9B. Gemma also uses the documented additional schema rescue.',
        'Neither Gemma correction arm shows a clear QA improvement: E4B changes by −0.52 percentage '
        'points (paired CI −1.75 to +0.72), and 12B by +0.21 (−0.31 to +0.82). The 12B-minus-E4B '
        'raw difference is +1.44 points (−0.93 to +3.71), so this sample does not establish a clear '
        'ordering between the two Gemma pipelines. These conclusions concern answerability under '
        'the fixed student and synthetic questions; they do not establish human factual superiority.',
        'E4B has the highest cold-generation probe rate among the measured single-target AR arms, '
        'but large batches do not prevent schema/quote failures or long retries. The full source-only '
        'trace spans are approximately 53.5 minutes for E4B and 50.6 minutes for 12B, including the '
        'original unsuccessful recovery and final reconciliation. The entire two-model experiment '
        'uses 4.6689 allocated GPU-hours. Starting with finite schemas may reduce these recovery '
        'costs, but a complete fresh cohort under that revised recipe has not been timed.',
        'Compare paired confidence intervals, narrative lengths, invalids and self-audit failure rates before '
        'choosing a deployment. Correction can change answerability and factual reliability differently. '
        'These complete deployed pipelines are not length-matched, and no independent human faithfulness '
        'study or trained-LoRA gain is established. See the [combined GH200 comparison](../ORNITH_DFLASH_GH200_RESULTS.md).',
        '## Files and verification',
        '`scores.csv` contains fourteen scores; `paper_scores.csv` contains every paper/condition and time; '
        '`throughput.csv` contains 48 probes; `phase_timings.csv` contains all full-output phase aggregates; '
        '`self_audit_scores.csv` includes 388 valid/failed audits; `paper_names.tsv` identifies all 97 papers. '
        'All 194 per-model paper archives retain raw/corrected summaries, audit findings, correction proposals, '
        'actual requests/responses, emitted reasoning fields and failed attempts/recovery. Full QA, runtime '
        'logs, metrics, code/model fingerprints and accounting are preserved. Weights/caches are excluded. '
        'Machine-specific HPC paths in snapshots require adaptation before reproducing inference elsewhere.',
        '```bash\npython3 experiments/scientific_summaries/gemma4_97/verify_results.py\n```',
        'Rebuild with `package_results.py --run-dir /path/to/run`. All packaged files have SHA256 checksums. '
        'Upstream model/source attribution and terms remain unchanged.'
    ]
    (HERE/'README.md').write_text('\n\n'.join(sections)+'\n')
    overview=HERE.parent/'ORNITH_DFLASH_GH200_RESULTS.md'
    base=overview.read_text().split('\n## Gemma 4 E4B IT and 12B IT')[0]
    base=base.replace('# Ornith DFlash: measured GH200 accuracy and throughput','# Scientific summaries: measured GH200 accuracy and throughput')
    base=base.replace('[35B-A3B](ornith35_dflash_97/README.md).',
        '[35B-A3B](ornith35_dflash_97/README.md), [Gemma E4B / 12B](gemma4_97/README.md).')
    base=base.replace('QA DFlash configuration','QA decoding')
    lines=base.splitlines();position=next(i for i,line in enumerate(lines) if line.startswith('| Generator |'))+2
    while position<len(lines) and lines[position].startswith('| '):position+=1
    rows=[]
    for m in MODELS:
        a=report['models'][m+'_raw'];b=report['models'][m+'_corrected']
        rows.append(f"| {NAMES[m]} | BF16 | {a['correct']}/970 · {100*a['accuracy']:.2f}% | "
            f"{b['correct']}/970 · {100*b['accuracy']:.2f}% | {report['summary_lengths'][m+'_raw']['mean_words']:.0f} / "
            f"{report['summary_lengths'][m+'_corrected']['mean_words']:.0f} | ar |")
    lines=[line for line in lines[:position] if not any(line.startswith('| '+NAMES[m]+' | BF16 |') for m in MODELS)]+rows+lines[position:]
    position=next(i for i,line in enumerate(lines) if line.startswith('| Target | Runtime |'))+2
    while position<len(lines) and lines[position].startswith('| '):position+=1
    rows=[f"| {NAMES[m]} | ar | {best[m]['generation']['output_tokens_per_second']:.2f} | "
          f"{best[m]['generation']['concurrency']} | {best[m]['correction']['output_tokens_per_second']:.2f} | "
          f"{best[m]['correction']['concurrency']} |" for m in MODELS]
    lines=[line for line in lines[:position] if not any(line.startswith('| '+NAMES[m]+' | ar |') for m in MODELS)]+rows+lines[position:]
    base='\n'.join(lines)
    extra=['## Gemma 4 E4B IT and 12B IT','',
        'Same frozen 97 papers, fixed student and initial source-only generation/correction protocol; '
        'Gemma adds documented finite-schema rescue after common-protocol failures. '
        '[Full English findings and evidence](gemma4_97/README.md). BF16, autoregressive decoding, no LoRA.','',
        table(['Model','Raw QA','Corrected QA','Mean words raw / corrected','Cold generation tok/s (batch)','Repeat correction tok/s (batch)'],[
            [NAMES[m],f"{report['models'][m+'_raw']['correct']}/970 · {100*report['models'][m+'_raw']['accuracy']:.2f}%",
             f"{report['models'][m+'_corrected']['correct']}/970 · {100*report['models'][m+'_corrected']['accuracy']:.2f}%",
             f"{report['summary_lengths'][m+'_raw']['mean_words']:.0f} / {report['summary_lengths'][m+'_corrected']['mean_words']:.0f}",
             f"{best[m]['generation']['output_tokens_per_second']:.2f} ({best[m]['generation']['concurrency']})",
             f"{best[m]['correction']['output_tokens_per_second']:.2f} ({best[m]['correction']['concurrency']})"] for m in MODELS]),'',
        'The small checkpoint is **E4B** (4.5B effective / approximately 8B including embeddings). '
        'Probe throughput is measured separately from complete-output timings. The Gemma report includes '
        'paired confidence intervals against each Ornith model and Qwen27B, all batch observations, '
        'self-audit failures and complete scheduler GPU-hours. Previous ten QA conditions are reused only '
        'after exact immutable source/question/context/prompt/protocol validation.']
    extra+=['','## Findings across all five pipelines','',
        '**Ornith 9B leads both Gemma pipelines on raw and corrected QA retention**, with paired confidence '
        'intervals below zero for each Gemma-minus-9B comparison. Gemma narratives are substantially shorter '
        '(775 / 918 mean raw words versus 2,548 for Ornith 9B), and Gemma uses an additional source-only '
        'finite-schema fallback after format failures. These are complete-pipeline results, not a '
        'length-matched capability comparison.','',
        'Neither Gemma correction change has a paired interval excluding zero. E4B has stronger measured '
        'AR cold-generation throughput, while 9B DFlash4 retains the strongest repeat-input correction '
        'probe rate. Token rates alone do not capture retry and repair costs. The Gemma experiment costs '
        '4.6689 allocated GPU-hours including its failed original controller and successful QA resumption.']
    overview.write_text(base.rstrip()+'\n\n'+'\n'.join(extra)+'\n')
    paths=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts)
    (HERE/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(HERE))+'\n' for p in paths))
    print(json.dumps(dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths),scores=14,qa_answers=13580,
        paper_traces=194,throughput_observations=len(throughput),destination=str(HERE))))

if __name__=='__main__':main()
