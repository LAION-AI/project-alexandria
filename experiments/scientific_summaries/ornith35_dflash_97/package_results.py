"""Package the 35B run and the measured 9B/35B comparison, without inference."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'ornith_dflash_97'))
from package_results import archive,table,write_csv,write_json,compressed_copy

def load(path):
    return json.loads(path.read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args();root=args.run_dir.resolve()
    report=load(root/'outputs/report.json');manifest=load(root/'inputs/experiment_manifest.json')
    qa=load(root/'outputs/qa-results.json');audit=load(root/'outputs/final_audit.json')
    if not audit['passed'] or audit['conditions']!=10:
        raise ValueError('All ten fixed-cohort conditions must pass the audit')
    artifact=HERE/'artifacts';artifact.mkdir(exist_ok=True)
    for name in ['report.json','final_audit.json','performance_recipe.json','complete.json','first_pass.json','code_snapshot.json']:
        shutil.copyfile(root/'outputs'/name,artifact/name)
    job=load(root/'outputs/complete.json')['job_id']
    accounting=subprocess.check_output(['sacct','-X','-j',job,'--format=JobIDRaw,State,ElapsedRaw,AllocTRES','--parsable2','--noheader'],text=True)
    row=next(line.split('|') for line in accounting.splitlines() if line.split('|')[0]==job)
    gpu_count=int(next(x.split('=')[1] for x in row[3].split(',') if x.startswith('gres/gpu=')))
    bill=dict(job_id=job,state=row[1],elapsed_seconds=int(row[2]),allocated_gpus=gpu_count,
              allocated_gpu_hours=int(row[2])*gpu_count/3600,
              note='Scheduler allocation includes startup, benchmarks, generation, recovery, QA and idle GPU time; overlapping steps counted once.')
    write_json(artifact/'gpu_accounting.json',bill)
    compressed_copy(root/'outputs/qa-results.json',artifact/'qa-results.json.gz')
    provenance=HERE/'provenance';provenance.mkdir(exist_ok=True)
    for name in ['experiment_manifest.json','target-weights-manifest.json','draft-weights-manifest.json','judge-weights-manifest.json',
                 'target-config.json','draft-config.json','judge-config.json']:
        shutil.copyfile(root/'inputs'/name,provenance/name)
    shutil.copytree(root/'code',provenance/'source_code',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    archive([(f,str(f.relative_to(root))) for folder in ['logs','outputs/server_accounting']
             for f in (root/folder).rglob('*') if f.is_file()],artifact/'runtime_logs.tar.gz')
    with (root/'inputs/paper_names.tsv').open(newline='') as f:names=list(csv.reader(f,delimiter='\t'))
    with (HERE/'paper_names.tsv').open('w',newline='') as f:
        csv.writer(f,delimiter='\t',quoting=csv.QUOTE_ALL,lineterminator='\n').writerows(names)
    throughput=[]
    for mode in ['ar','dflash4','dflash8','dflash15']:
        result=load(root/'outputs/throughput'/(mode+'.json'))
        if not result.get('complete'):raise ValueError('Incomplete throughput runtime '+mode)
        shutil.copyfile(root/'outputs/throughput'/(mode+'.json'),artifact/('throughput-'+mode+'.json'))
        for b in result['batches']:
            throughput.append(dict(runtime=mode,phase=b['phase'],batch=b['concurrency'],cache=b['cache_state'],
                seconds=b['seconds'],input_tokens=b['input_tokens'],output_tokens=b['completion_tokens'],
                output_tokens_per_second=b['output_tokens_per_second'],mean_request_seconds=b['request_latency_mean_seconds'],
                gpu_memory_used_total_utilization=b['gpu_memory']))
        directory=root/'outputs/throughput'/mode
        archive([(f,str(f.relative_to(directory))) for f in directory.rglob('*') if f.is_file()],
                artifact/('throughput-traces-'+mode+'.tar.gz'))
    write_csv(HERE/'throughput.csv',throughput)
    speculative={}
    for mode in ['dflash4','dflash8','dflash15']:
        metrics=(root/'outputs/throughput'/mode/'correction-64-after.prom').read_text()
        def counter(name):
            return sum(float(v) for v in re.findall(r'^vllm:'+name+r'\{[^\n]*\} ([0-9.eE+\-]+)$',metrics,re.M))
        drafts=counter('spec_decode_num_drafts_total')
        tokens=counter('spec_decode_num_draft_tokens_total')
        accepted=counter('spec_decode_num_accepted_tokens_total')
        speculative[mode]=dict(drafts=drafts,draft_tokens=tokens,accepted_tokens=accepted,
            acceptance_fraction=accepted/tokens,accepted_per_draft=accepted/drafts,
            scope='Cumulative counters across the complete benchmark, including kernel warmup; not a production acceptance estimate.')
    write_json(artifact/'speculative_metrics.json',speculative)
    write_json(artifact/'metadata_notes.json',dict(
        production_replicas=4,
        inherited_recipe_text='The original recipe production_note says three replicas; the actual commands, cohort config and orchestrator use four.',
        warm_round='A repeated full input is measured; full cache residence is not guaranteed. The source report used the phrase fully warmed correction, which overstates cache guarantees.',
        accounting='complete.json is Python orchestration time; gpu_accounting.json is the complete scheduler allocation and determines billed GPU-hours.'))
    def best(mode,phase,cache):
        return max(r['output_tokens_per_second'] for r in throughput if r['runtime']==mode and r['phase']==phase and r['cache']==cache)
    generation_ratio=best('dflash4','generation','cold')/best('ar','generation','cold')
    correction_ratio=best('dflash4','correction','warm')/best('ar','correction','warm')
    scores=[dict(condition=k,model=v['name'],correct=v['correct'],total=v['total'],accuracy=v['accuracy'],
                 ci95_low=v['ci95'][0],ci95_high=v['ci95'][1],invalid=v['invalid']) for k,v in report['models'].items()]
    write_csv(HERE/'scores.csv',scores)
    timings=[dict(model='ornith35_dflash',phase=k,**v) for k,v in report['generation']['ornith35_dflash']['phases'].items()]
    write_csv(HERE/'phase_timings.csv',timings)
    paper_scores=[];transitions=collections.Counter()
    for doc in qa['documents']:
        for label,value in doc['conditions'].items():
            paper_scores.append(dict(document_id=doc['document_id'],condition=label,
                correct=sum(next(iter(r['predictions'].values()))==r['gold'] for r in value['rows']),
                total=10,invalid=sum(next(iter(r['predictions'].values())) is None for r in value['rows']),
                latest_condition_seconds=value['elapsed_seconds'],context_sha256=value['context_sha256']))
        for x,y in zip(doc['conditions']['ornith35_raw']['rows'],doc['conditions']['ornith35_corrected']['rows']):
            a=next(iter(x['predictions'].values()))==x['gold'];b=next(iter(y['predictions'].values()))==y['gold']
            transitions[('correct' if a else 'wrong')+'_to_'+('correct' if b else 'wrong')]+=1
    write_csv(HERE/'paper_scores.csv',paper_scores)
    write_json(artifact/'correction_transitions.json',dict(transitions))
    self_rows=[];self_stats={};assessments=[]
    keys=['factual_accuracy','coverage','clarity','faithfulness','scientific_precision']
    directory=root/'outputs/cohorts/ornith35_dflash/documents'
    for folder in sorted(directory.iterdir()):
        if not folder.is_dir():continue
        document=load(folder/'document.json')
        if not document.get('raw') or not document.get('corrected'):raise ValueError('Missing summary')
        archive([(f,str(f.relative_to(folder))) for f in folder.rglob('*') if f.is_file()],
                artifact/'paper_traces'/(folder.name+'.tar.gz'))
        for phase in ['raw','corrected']:
            assessment=document.get(phase+'_quality_assessment',{})
            assessments.append(dict(document_id=folder.name,phase=phase,assessment=assessment))
            self_rows.append(dict(document_id=folder.name,phase=phase,review_failed=bool(assessment.get('review_failed')),
                verdict=assessment.get('verdict','unavailable'),issues=len(assessment.get('issues',[])),
                missing_central_facts=len(assessment.get('missing_central_facts',[])),
                **{k:assessment.get('scores',{}).get(k,'') for k in keys}))
    write_csv(HERE/'self_audit_scores.csv',self_rows)
    write_json(artifact/'quality_assessments.json',assessments)
    compressed_copy(artifact/'quality_assessments.json',artifact/'quality_assessments.json.gz')
    (artifact/'quality_assessments.json').unlink()
    for phase in ['raw','corrected']:
        rows=[r for r in self_rows if r['phase']==phase]
        self_stats[phase]=dict(papers=len(rows),failed_reviews=sum(r['review_failed'] for r in rows),
            verdicts=dict(collections.Counter(r['verdict'] for r in rows)),issues=sum(r['issues'] for r in rows),
            missing_central_facts=sum(r['missing_central_facts'] for r in rows),
            scores={k:dict(n=len(v:=[r[k] for r in rows if r[k]!='']),mean=statistics.mean(v) if v else None,
                histogram=dict(collections.Counter(v))) for k in keys})
    write_json(artifact/'self_audit_aggregate.json',self_stats)
    raw=report['models']['ornith35_raw'];corrected=report['models']['ornith35_corrected']
    winner=report['new_model_qa_speculative_configuration']
    sections=[
        '# Findings and results: Ornith-1.5-35B-A3B + DFlash on JUPITER',
        f"**Complete: 97 held-out papers, 970 immutable MCQs; Slurm job {job}.** "
        f"Raw accuracy **{raw['correct']}/970 ({100*raw['accuracy']:.2f}%)**; corrected accuracy "
        f"**{corrected['correct']}/970 ({100*corrected['accuracy']:.2f}%)**. QA summaries were produced with **{winner}**. No LoRA is used.",
        '## Model and protocol',
        '[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) is the '
        '35B mixture-of-experts target (approximately 3B parameters active per token). '
        '[Ornith-1.5-35B-A3B-DFlash](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-DFlash) '
        'is its separate approximately 0.386B draft. Both use **BF16**; inference uses one '
        'target/draft pair per GH200, tensor parallel size 1. The draft is not a standalone summarizer.',
        f"Target revision: `{manifest['target_revision']}`. Draft revision: `{manifest['draft_revision']}`. "
        f"Repository protocol revision: `{manifest['commit']}`. Testset SHA256: `{manifest['testset_sha256']}`.",
        'The [same frozen test set](../data/README.md), original 19-field prompt, full untruncated '
        'source, no-thinking mode, temperature 0.2, top-p 0.95, 16,000 initial output tokens '
        'and 65,536 maximum context are used. Source-only same-model audits, full semantic '
        'correction, V3 field/quote repair and bounded source-only recovery follow the '
        '[completed 9B protocol](../ornith_dflash_97/README.md). Failed long outputs may use '
        '32,768 generation / 24,576 correction output tokens with presence penalty 1.5. '
        'Unusable optional citation rankings may be discarded only after bounded repairs, '
        'with original values/provenance retained. Scientific narrative is validated separately.',
        'Generators/reviewers/correctors receive no questions, options, gold keys, QA rationales '
        'or QA evidence. No papers are dropped for quality. All evaluation outputs remain '
        'outside the 1,000-paper training workspace and must not enter LoRA training.',
        'The fixed student is **Qwen2.5-7B-Instruct BF16**, pinned revision '
        '`a09a35458c702b33eeacc393d103063234e8bc28`, vLLM 0.30.0, temperature 0.5, '
        'top-p 0.95, 100 output tokens, frequency/presence penalties 1.05, no thinking, '
        'four concurrent requests, five formatting attempts maximum. Exact historical '
        'ASCII sanitizer, semicolon parser, source/question/option order and prompt hashes '
        'are preserved. Eight prior comparison conditions are reused after exact validation; '
        '**1,940 new answers** are scored. All **9,700** combined answers are audited.',
        'The synthetic questions were authored by gpt-6-luna. This is a convenience sample '
        'of relatively short dataset texts, with no independent completeness check against '
        'publisher PDFs or human benchmark validation. Confidence intervals use 10,000 '
        'document-cluster bootstrap draws, seed 250219413, keeping all ten questions per '
        'paper together. Paired differences use the same paper clusters.',
        '## All accuracy scores',
        table(['Condition','Correct / 970','Accuracy','95% paper-bootstrap CI','Invalid'],[
            [v['name'],v['correct'],f"{100*v['accuracy']:.2f}%",f"{100*v['ci95'][0]:.2f}–{100*v['ci95'][1]:.2f}%",v['invalid']]
            for v in report['models'].values()]),
        '### Paired comparisons involving the new 35B model',
        table(['Comparison','Difference (percentage points)','95% paired CI'],[
            [k,f"{100*v['difference']:+.2f}",f"{100*v['ci95'][0]:+.2f} to {100*v['ci95'][1]:+.2f}"]
            for k,v in report['paired_differences'].items() if 'ornith35' in k]),
        '### Answer changes after correction',
        table(['Correct → correct','Wrong → wrong','Wrong → correct','Correct → wrong'],[
            [transitions[k] for k in ['correct_to_correct','wrong_to_wrong','wrong_to_correct','correct_to_wrong']]]),
        'These are fixed-student answer transitions, not independent counts of factual errors '
        'in summaries. A small difference with a paired interval including zero is uncertain.',
        '## Every measured throughput configuration',
        'One GH200 per runtime. Native vLLM DFlash / MoE kernels, continuous batching, '
        'prefix caching, chunked prefill (8,192 max batched tokens), compilation/CUDA graphs, '
        '64 maximum scheduled sequences and GPU memory utilization 0.90. GPU-reported '
        'memory is 97,871 MiB. Exact actual server commands/backend selections and '
        'before/after metrics are archived.',
        'All 512-token probes use 64 frozen **non-holdout** source-only papers, disjoint '
        'from evaluation. Tokens/s is actual emitted output tokens divided by batch '
        'wall time, including the relevant prefill. The warm round repeats its entire '
        'input; actual cache hits depend on capacity and eviction. It can be more optimistic than a first correction caching only the source '
        'prefix. These are short probes, not complete-summary timings. `ar` is the same '
        '35B target without a draft; DFlash numbers denote configured speculative tokens.',
        table(['Runtime','Batch','Generation cold tok/s','Generation warm tok/s','Correction cold tok/s','Correction warm tok/s'],[
            [mode,batch]+[f"{next(r['output_tokens_per_second'] for r in throughput if r['runtime']==mode and r['batch']==batch and r['phase']==phase and r['cache']==cache):.2f}"
                          for phase,cache in [('generation','cold'),('generation','warm'),('correction','cold'),('correction','warm')]]
            for mode in ['ar','dflash4','dflash8','dflash15'] for batch in [1,4,8,16,32,64]]),
        'The production selector chooses the strongest measured **DFlash** configuration '
        'by the harmonic mean of best cold-generation / repeat-input correction throughput; '
        'standalone AR is reported separately and not silently selected for DFlash QA. '
        'No complete-cohort standalone AR QA arm was run for this new model, so the '
        'benchmark does not establish paired AR/DFlash accuracy equivalence.',
        '## Complete-output API request timing and token work',
        table(['Phase','Calls','Prompt tokens','Output tokens','Mean s','Median s','P95 s'],[
            [r['phase'],r['calls'],r['prompt_tokens'],r['completion_tokens'],f"{r['mean_request_seconds']:.2f}",
             f"{r['median_request_seconds']:.2f}",f"{r['p95_request_seconds']:.2f}"] for r in timings]),
        'These are individual API-request latencies and include failures/retries and '
        'concurrent scheduling. Adding them does not produce cohort wall time. Repeated '
        'prompt tokens are not equivalent to uncached prefill. The first-pass counts are '
        'retained separately from recovered final counts; full allocation accounting '
        'also includes startup and idle GPU time.',
        '## Length and quality-audit limitations',
        table(['Condition','Mean words','Median','Minimum','Maximum'],[
            [k,f"{v['mean_words']:.2f}",v['median_words'],v['min_words'],v['max_words']]
            for k,v in report['summary_lengths'].items()]),
        'These pipelines share the initial prompt but **summary lengths are not equalized**. '
        'The comparison measures deployed pipelines with different model architectures '
        'and sizes, rather than an isolated parameter-count effect. Qwen27B is FP8; both '
        'Ornith models are BF16. Faster output per second is not the same as faster '
        'papers per second when narrative/retry lengths differ.',
        table(['Self-audit','Scored / 97','Failed','Pass','Needs correction','Issues','Missing facts'],[
            [k,v['scores']['factual_accuracy']['n'],v['failed_reviews'],v['verdicts'].get('pass',0),
             v['verdicts'].get('needs_correction',0),v['issues'],v['missing_central_facts']] for k,v in self_stats.items()]),
        table(['Self-audit']+[k.replace('_',' ')+' / 5' for k in keys],[
            [phase]+[f"{v['scores'][k]['mean']:.3f}" if v['scores'][k]['mean'] is not None else 'unavailable' for k in keys]
            for phase,v in self_stats.items()]),
        'Same-model audits are fallible and are **not independent human labels**. Means '
        'use valid scored audits only; failed assessments are unavailable. Mechanical '
        'schema/literal-quote validation does not prove entailment or establish a clean '
        'semantic audit. Actual emitted reasoning fields are preserved when present; '
        'thinking was disabled and no hidden reasoning is invented.',
        '## Allocation and evidence',
        f"Slurm job **{job}**: **{bill['elapsed_seconds']} seconds**, **{gpu_count} allocated GPUs**, "
        f"**{bill['allocated_gpu_hours']:.4f} allocated GPU-hours**, final state **{bill['state']}**. "
        'This includes benchmark/model startup, generation, recovery, QA and idle capacity; '
        'it is not a per-model active-inference or 60-million-paper bill.',
        f"All 97 raw and 97 corrected new summaries are present. The final audit verifies "
        f"**{audit['qa_prompt_hash_checks']:,} QA prompt hashes**, **{audit['prediction_parser_checks']:,} parsed predictions**, "
        'source checksums, immutable questions and corrected literal evidence ledgers. '
        'All per-paper outputs, findings/correction proposals, actual request/response '
        'journals, retry/recovery histories, benchmark metrics, server logs, code and '
        'model fingerprints are preserved. The earlier 9B/Qwen evidence remains in its '
        'separate linked package.',
        '## Conclusions',
        f"**DFlash does not speed up this measured 35B workload.** The strongest speculative "
        f"configuration, DFlash4, reaches {100*generation_ratio:.1f}% of best standalone AR generation "
        f"throughput and {100*correction_ratio:.1f}% of repeat-input correction throughput, comparing "
        'each runtime at its own strongest measured batch. This result differs from the 9B correction speedup. '
        f"DFlash4 accepts {100*speculative['dflash4']['acceptance_fraction']:.2f}% of proposed draft tokens "
        f"({speculative['dflash4']['accepted_per_draft']:.3f} accepted tokens per draft). Acceptance alone "
        'does not imply a speedup. Server logs select the native Triton MoE backend on Hopper; '
        'larger speculative windows and reduced KV-cache capacity can add overhead, but no kernel/cache '
        'ablation was run to isolate the cause. All counters are retained in `speculative_metrics.json`.',
        f"The 35B model scores **{100*raw['accuracy']:.2f}% raw** and **{100*corrected['accuracy']:.2f}% corrected**. "
        'Read paired confidence intervals and output lengths together before attributing '
        'differences to model size. The [combined GH200 comparison](../ORNITH_DFLASH_GH200_RESULTS.md) '
        'puts the two Ornith models beside Qwen27B and separates production QA settings '
        'from the best short-probe throughput. No trained-LoRA gain, human factual '
        'superiority, or full-output speedup beyond the measured workload is established.',
        '## Files and CPU verification',
        '`scores.csv` contains all ten scores; `paper_scores.csv` contains every paper/condition '
        'score and timing; `throughput.csv` contains all 96 probe observations; '
        '`phase_timings.csv` contains complete-output call aggregates; `self_audit_scores.csv` '
        'includes all 194 valid/failed assessments; `paper_names.tsv` identifies all papers. '
        '`artifacts/` contains full QA, model traces, audits, runtime logs, metrics and '
        'accounting. `provenance/` holds pinned identities and source snapshots. '
        'Model weights/caches are excluded.',
        '```bash\npython3 experiments/scientific_summaries/ornith35_dflash_97/verify_results.py\n```',
        'Rebuild from the preserved run with `package_results.py --run-dir /path/to/run`. '
        'Original HPC scripts retain machine-specific paths and require adaptation '
        'before inference elsewhere. Upstream source/model attribution and terms remain '
        'unchanged. All packaged files have SHA256 checksums.'
    ]
    (HERE/'README.md').write_text('\n\n'.join(sections)+'\n')
    old=load(HERE.parent/'ornith_dflash_97/artifacts/report.json')
    overview=['# Ornith DFlash: measured GH200 accuracy and throughput','',
        'Frozen 97 papers / 970 MCQs per condition; fixed BF16 Qwen2.5-7B student. No LoRA. '
        'Full protocols, evidence and limitations: [9B](ornith_dflash_97/README.md), [35B-A3B](ornith35_dflash_97/README.md).','',
        '| Generator | Precision | Raw QA | Corrected QA | Mean words raw / corrected | QA DFlash configuration |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for name,raw_key,corrected_key,length_key,precision,runtime in [
        ('Ornith-1.5-9B','ornith_raw','ornith_corrected','ornith_dflash','BF16','dflash8'),
        ('Ornith-1.5-35B-A3B','ornith35_raw','ornith35_corrected','ornith35_dflash','BF16',winner),
        ('Qwen3.8-27B','qwen38_raw','qwen38_corrected','qwen38_fp8','FP8','none')]:
        a=report['models'][raw_key];b=report['models'][corrected_key]
        x=report['summary_lengths'][length_key+'_raw'];y=report['summary_lengths'][length_key+'_corrected']
        overview.append(f"| {name} | {precision} | {a['correct']}/970 · {100*a['accuracy']:.2f}% | "
            f"{b['correct']}/970 · {100*b['accuracy']:.2f}% | {x['mean_words']:.0f} / {y['mean_words']:.0f} | {runtime} |")
    overview+=['','## Best measured output throughput per GH200','',
        'Cold-source generation and repeat-input correction, 512-token probes on non-holdout papers. Warm-round cache reuse is subject to capacity and eviction. '
        'Both phases select their best measured batch independently; no output-length or '
        'full-pipeline speed equivalence is implied.','',
        '| Target | Runtime | Generation tok/s | Batch | Correction tok/s | Batch |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for target,base in [('Ornith-1.5-9B',HERE.parent/'ornith_dflash_97'),('Ornith-1.5-35B-A3B',HERE)]:
        for mode in ['ar','dflash4','dflash8','dflash15']:
            bs=load(base/'artifacts'/('throughput-'+mode+'.json'))['batches']
            g=max((b for b in bs if b['phase']=='generation' and b['cache_state']=='cold'),key=lambda b:b['output_tokens_per_second'])
            c=max((b for b in bs if b['phase']=='correction' and b['cache_state']=='warm'),key=lambda b:b['output_tokens_per_second'])
            overview.append(f"| {target} | {mode} | {g['output_tokens_per_second']:.2f} | {g['concurrency']} | {c['output_tokens_per_second']:.2f} | {c['concurrency']} |")
    overview+=['','## Paired 35B versus 9B QA differences','',
        '| Comparison | Difference (percentage points) | 95% paired paper-bootstrap CI |',
        '| --- | ---: | ---: |']
    for key in ['ornith35_raw_minus_ornith_raw','ornith35_corrected_minus_ornith_corrected']:
        v=report['paired_differences'][key]
        overview.append(f"| {key} | {100*v['difference']:+.2f} | {100*v['ci95'][0]:+.2f} to {100*v['ci95'][1]:+.2f} |")
    overview+=['','Source-only correction can change question-answerability and factual reliability differently. '
        'Output lengths are not matched; model sizes, architectures and precision differ. '
        'Self-audits are not independent human labels. Prior eight-condition QA is reused '
        'only after exact source/question/context/prompt validation. See complete reports '
        'for all confidence intervals, invalids, timing, recovery and allocation hours. '
        'The existing [60M 9B planning estimate](ornith_dflash_97/SCALING_60M.md) remains '
        'a 9B estimate; it is not silently relabelled as a measured 35B cost.']
    overview+=['',f"**35B throughput finding:** best DFlash4 reaches {100*generation_ratio:.1f}% of best AR "
        f"generation throughput and {100*correction_ratio:.1f}% of best repeat-input correction throughput. "
        'The 35B experiment provides no measured DFlash speedup. Its raw and corrected QA differences '
        'versus 9B both have paired confidence intervals crossing zero.']
    (HERE.parent/'ORNITH_DFLASH_GH200_RESULTS.md').write_text('\n'.join(overview)+'\n')
    paths=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts)
    (HERE/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(HERE))+'\n' for p in paths))
    print(json.dumps(dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths),scores=10,qa_answers=9700,paper_traces=97,
                         throughput_observations=len(throughput),destination=str(HERE))))

if __name__=='__main__':
    main()
