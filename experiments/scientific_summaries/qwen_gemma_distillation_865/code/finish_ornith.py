"""Combine actual Ornith AR outputs without dropping failed paper identifiers."""
from common import ROOT,load,write,jsonl
import statistics
from transformers import AutoTokenizer

cohort=load(ROOT/'inputs/frozen_cohort.json')
rows=[load(ROOT/'outputs/ornith/documents'/p['document_id']/'result.json') for p in cohort['papers']]
assert len(rows)==len({r['document_id'] for r in rows})==865
tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/ornith-1.5-9b',local_files_only=True)
for row in rows:
    row['native_narrative_tokens']=len(tokenizer.encode(row['judge_context'],add_special_tokens=False))
workers=[load(ROOT/'outputs/ornith'/('worker'+str(i)+'.json')) for i in range(8)]
phase_ids={w['job_id'] for w in workers}
assert len(phase_ids)==1, 'Worker timing records must belong to the same allocation'
phase_id=next(iter(phase_ids))
completion=prompt=calls=0
new_completion=new_prompt=new_calls=0
retained_completion=retained_calls=0
for folder in ['ornith/documents','ornith_initial_attempts','ornith_retry_attempts','ornith_resume_attempts']:
    for path in (ROOT/'outputs'/folder).rglob('attempt*.json'):
        record=load(path)
        usage=record.get('response',{}).get('usage',{})
        completion+=usage.get('completion_tokens',0);prompt+=usage.get('prompt_tokens',0);calls+=1
        if record.get('job_id')==phase_id:
            new_completion+=usage.get('completion_tokens',0);new_prompt+=usage.get('prompt_tokens',0);new_calls+=1
        else:
            retained_completion+=usage.get('completion_tokens',0);retained_calls+=1
seconds=sum(w['seconds'] for w in workers)
performance=dict(model='ornith-ai/Ornith-1.5-9B',model_revision='489cb97981b8654bcfcf30ce1f94ed1b62e07b53',
    precision='BF16',speculative_decoding=False,thinking=True,papers=865,concurrency_per_gpu=16,
    completion_tokens_all_attempts=completion,prompt_tokens_all_attempts=prompt,api_calls=calls,
    summed_active_gpu_seconds=seconds,active_generation_gpu_hours=seconds/3600,
    completion_tokens_per_second_per_gpu=new_completion/seconds if seconds else None,
    measured_phase_job_id=phase_id,
    resumed_phase_completion_tokens=new_completion,resumed_phase_prompt_tokens=new_prompt,
    resumed_phase_api_calls=new_calls,retained_previous_completion_tokens=retained_completion,
    retained_previous_api_calls=retained_calls,thinking_token_budget_resumed_phase=16384,
    initial_uncapped_papers=sum(r.get('thinking_token_budget') is None for r in rows if r['status']=='generated'),
    successful_papers_by_thinking_budget={str(budget) if budget is not None else 'uncapped':
        sum(r.get('thinking_token_budget')==budget for r in rows if r['status']=='generated')
        for budget in [None,16384,12288]},
    throughput_scope='Latest completed allocation only: numerator uses recorded job IDs and denominator uses that allocation worker timings. Earlier successful outputs are retained under their actual recipes. All completed calls, including archived failed retries, enter all-attempt totals. Interrupted in-flight calls have no complete usage record; full Slurm GPU-hours are reported separately.',
    mean_native_narrative_tokens=statistics.mean(r['native_narrative_tokens'] for r in rows),
    mean_narrative_words=statistics.mean(len(r['judge_context'].split()) for r in rows),
    failed_generation_papers=sum(r['status']!='generated' for r in rows),
    failed_papers_retained=True,setup_and_previous_failed_allocation_excluded=True,workers=workers)
write(ROOT/'outputs/ornith/performance.json',performance)
jsonl(ROOT/'outputs/ornith/papers_and_summaries.jsonl',rows)
write(ROOT/'outputs/ornith/complete.json',dict(papers=865,
    generated=sum(r['status']=='generated' for r in rows),failed=sum(r['status']!='generated' for r in rows),
    speculative_decoding=False,model='ornith-ai/Ornith-1.5-9B'))
