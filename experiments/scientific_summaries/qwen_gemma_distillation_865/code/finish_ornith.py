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
completion=prompt=calls=0
for item in cohort['papers']:
    for path in (ROOT/'outputs/ornith/documents'/item['document_id']).glob('attempt*.json'):
        usage=load(path)['response'].get('usage',{})
        completion+=usage.get('completion_tokens',0);prompt+=usage.get('prompt_tokens',0);calls+=1
seconds=sum(w['seconds'] for w in workers)
performance=dict(model='ornith-ai/Ornith-1.5-9B',model_revision='489cb97981b8654bcfcf30ce1f94ed1b62e07b53',
    precision='BF16',speculative_decoding=False,thinking=True,papers=865,concurrency_per_gpu=16,
    completion_tokens_all_attempts=completion,prompt_tokens_all_attempts=prompt,api_calls=calls,
    summed_active_gpu_seconds=seconds,active_generation_gpu_hours=seconds/3600,
    completion_tokens_per_second_per_gpu=completion/seconds,
    mean_native_narrative_tokens=statistics.mean(r['native_narrative_tokens'] for r in rows),
    mean_narrative_words=statistics.mean(len(r['judge_context'].split()) for r in rows),
    failed_generation_papers=sum(r['status']!='generated' for r in rows),
    failed_papers_retained=True,setup_and_previous_failed_allocation_excluded=True,workers=workers)
write(ROOT/'outputs/ornith/performance.json',performance)
jsonl(ROOT/'outputs/ornith/papers_and_summaries.jsonl',rows)
write(ROOT/'outputs/ornith/complete.json',dict(papers=865,
    generated=sum(r['status']=='generated' for r in rows),failed=sum(r['status']!='generated' for r in rows),
    speculative_decoding=False,model='ornith-ai/Ornith-1.5-9B'))
