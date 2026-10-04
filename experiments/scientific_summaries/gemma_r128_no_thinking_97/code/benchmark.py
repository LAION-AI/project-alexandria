"""Matched no-thinking QA and deployment optimizations; all tuning is outside holdout."""
import concurrent.futures as cf
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

from common import ROOT,SOURCE,TRAINING_ROOT,TRAINING_PY,load,write,digest,jsonl
from inference import Server,generate_paper,post

REFERENCE=Path('/e/fscratch/reformo/schuhmann1/scientific-distillation-865-20261004')
PINNED=Path('/e/fscratch/reformo/schuhmann1/scientific-ornith-dflash-eval-97/inputs/project-alexandria-5aac4b5ba2a78b20637e8ab960fe79d01ecd769a')
VENDOR=PINNED/'experiments/scientific_summaries'
sys.path.insert(0,str(SOURCE/'code'))
from run import summary_system
sys.path.insert(0,str(PINNED/'src'));sys.path.insert(0,str(VENDOR))
import evaluate
from project_alexandria.backends import OpenAICompatibleBackend

OUT=ROOT/'outputs/evaluation'
NAMES={'thinking_reference':'Rank 128 / live BF16 LoRA / thinking (cached)',
       'no_thinking_matched':'Rank 128 / live BF16 LoRA / no thinking / concurrency 16',
       'no_thinking_merged':'Rank 128 / merged BF16 / no thinking / tuned deployment',
       'no_thinking_fp8':'Rank 128 / merged FP8 / no thinking / tuned deployment'}
ARMS=list(NAMES)[1:]
LOCK=threading.Lock()
STATES={a:dict(state='waiting') for a in ARMS}


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


def state(label,**data):
    with LOCK:
        STATES[label].update(data)
        write(ROOT/'outputs/arm_status.json',STATES)
    print(label,data,flush=True)


def prepare():
    test=VENDOR/'data/testset.json'
    assert digest(test)=='a0d5e5f99a0025c6cd8a5140a39a07a220ded6f37f04994500549886ffd83261'
    papers=load(test)['papers'];original=load(VENDOR/'data/papers.json')
    assert len(papers)==97 and all(len(p['questions'])==10 for p in papers)
    training={p['document_id'] for p in load(ROOT/'inputs/frozen_cohort.json')['papers']}
    assert len(training)==865 and not training&{p['document_id'] for p in papers}
    probes=load(ROOT/'inputs/optimization_papers.json')
    assert len(probes)==64 and {p['document_id'] for p in probes}<=training
    for p in papers:
        assert sha(p['fulltext'])==p['fulltext_sha256']
        assert evaluate.question_signature(evaluate.questions_for(original[p['original_paper_index']],p['original_paper_index']))==evaluate.question_signature(p['questions'])
    trained=load(REFERENCE/'outputs/train-r128/result.json')
    runtime=load(ROOT/'inputs/runtime.json')
    for name in ['vllm','torch','transformers','tokenizers']:
        assert importlib.metadata.version(name)==runtime['packages'][name], 'Pinned inference runtime changed'
    assert digest(ROOT/'models/gemma-4-12b-it/chat_template.jinja')==runtime['base_chat_template_sha256']==runtime['adapter_chat_template_sha256']
    assert trained['completed_epochs']==1 and trained['source_papers']==865 and trained['lora_rank']==128
    assert load(REFERENCE/'outputs/evaluation/complete.json')['audit_passed']
    old=load(REFERENCE/'outputs/evaluation/protocol.json')
    prompts=load(ROOT/'inputs/reference_prompts.json')
    assert sha(summary_system(prompts))==old['generator_prompt_sha256']
    protocol=dict(old)
    protocol.update(thinking=False,primary_comparison='Same rank-128 adapter, BF16 live LoRA, identical prompts/sampling/seeds/context/budget/retries; thinking disabled',
        reference_generation=REFERENCE.as_posix(),reference_qa_sha256=digest(REFERENCE/'outputs/evaluation/qa-results.json'),
        reference_qa_accuracy=897/970,reference_thinking=True,reference_reused_without_new_billing=True,
        qa_execution='Fixed pinned BF16 judge on GPU 3; score every ready summary without changing the held-out cohort',
        qa_replicas=1,qa_gpu_indices=[3],qa_concurrency_per_replica=4,
        adapter_training=trained,optimization_cohort_sha256=digest(ROOT/'inputs/optimization_papers.json'),
        optimization_selection='Only 64 non-held-out training papers; fixed-length 512-token probes; no QA/gold used to choose deployment',
        optimization_probe_costs_not_used_for_full_summary_scaling=True,production_capacity_assumption=.85,
        primary_concurrency=16,full_output_budget=24576,runtime=runtime,
        generation_parameters_changed=True,only_primary_generator_parameter_changed='enable_thinking: true to false',
        job_id=os.environ.get('SLURM_JOB_ID'))
    write(OUT/'protocol.json',protocol)
    for p in papers:
        row=load(REFERENCE/'outputs/evaluation/generation/gemma12_r128'/p['document_id']/'result.json')
        write(OUT/'generation/thinking_reference'/p['document_id']/'result.json',row)
    perf=load(REFERENCE/'outputs/evaluation/gemma12_r128-generation-performance.json')
    perf['cached_reference']=True
    write(OUT/'thinking_reference-generation-performance.json',perf)
    oldqa=load(REFERENCE/'outputs/evaluation/qa-results.json')
    if not (OUT/'qa-results.json').exists():
        write(OUT/'qa-results.json',dict(labels=NAMES,documents=[dict(document_id=d['document_id'],
            fulltext_sha256=d['fulltext_sha256'],questions=d['questions'],
            conditions={'thinking_reference':d['conditions']['gemma12_r128']}) for d in oldqa['documents']]))
    state('no_thinking_matched',state='prepared')
    return papers,probes,prompts


def reasoning_tokens(response):
    usage=response.get('usage') or {}
    return (usage.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0


def assert_no_thinking(response):
    message=response['choices'][0]['message']
    assert not (message.get('reasoning_content') or message.get('reasoning')), 'Unexpected returned thinking text'
    assert reasoning_tokens(response)==0, 'Unexpected emitted thinking tokens'


def probe(server,papers,prompts,concurrency,tag,model):
    # A cold-source sweep; the shared system prefix can still warm naturally.
    request=urllib.request.Request(server.endpoint+'/reset_prefix_cache',data=b'',method='POST')
    with urllib.request.urlopen(request,timeout=60) as response:
        assert json.load(response)['success'], 'Cannot reset a busy source prefix cache'
    start=time.monotonic()
    def task(p):
        payload=dict(model=model,messages=[dict(role='system',content=summary_system(prompts)),
            dict(role='user',content=prompts['summary-user'].replace('{paper_text}',p['fulltext']))],
            temperature=1.,top_p=.95,top_k=20,min_p=0.,presence_penalty=0.,repetition_penalty=1.,
            seed=int.from_bytes(hashlib.sha256((p['document_id']+'-0').encode()).digest()[:4],'big')%2147483647,
            max_tokens=512,min_tokens=512,ignore_eos=True,
            chat_template_kwargs=dict(enable_thinking=False,preserve_thinking=True,reasoning_effort='medium'),
            response_format={'type':'json_object'})
        tick=time.monotonic();response=post(server.endpoint,'/v1/chat/completions',payload)
        assert_no_thinking(response)
        write(ROOT/'outputs/probes'/tag/('c'+str(concurrency))/p['document_id'],
              dict(request=payload,response=response,seconds=time.monotonic()-tick))
        return response['usage']
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:usage=list(pool.map(task,papers))
    elapsed=time.monotonic()-start
    tokens=sum(u['completion_tokens'] for u in usage)
    record=dict(config=tag,concurrency=concurrency,papers=len(papers),elapsed_seconds=elapsed,
        completion_tokens=tokens,prompt_tokens=sum(u['prompt_tokens'] for u in usage),
        completion_tokens_per_second=tokens/elapsed,scope='fixed 512-token partial outputs, not full summaries',
        prefix_cache_reset_before_sweep=True,held_out_papers_used=0)
    write(ROOT/'outputs/probes'/tag/('c'+str(concurrency)+'.json'),record)
    return record


def cohort(server,label,papers,prompts,concurrency,model):
    state(label,state='generating',concurrency=concurrency)
    fresh=[p for p in papers if not (OUT/'generation'/label/p['document_id']/'result.json').exists()]
    start=time.monotonic()
    def task(p):
        folder=OUT/'generation'/label/p['document_id']
        source=dict(document_id=p['document_id'],fulltext=p['fulltext'],source_sha256=p['fulltext_sha256'])
        # Transport failures stop this arm and remain explicit; no silent paper omission.
        row=generate_paper(server.endpoint,source,summary_system(prompts),
            prompts['summary-user'].replace('{paper_text}',p['fulltext']),folder,model=model,thinking=False)
        for a in folder.glob('attempt*.json'):assert_no_thinking(load(a)['response'])
        assert not row.get('reasoning_content')
        return row
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:list(pool.map(task,fresh))
    elapsed=time.monotonic()-start
    rows=[load(OUT/'generation'/label/p['document_id']/'result.json') for p in papers]
    path=OUT/(label+'-generation-performance.json')
    segments=load(path).get('generation_segments',[]) if path.exists() else []
    if fresh:segments.append(dict(job_id=os.environ.get('SLURM_JOB_ID'),elapsed_seconds=elapsed,new_papers=len(fresh)))
    seconds=sum(s['elapsed_seconds'] for s in segments)
    attempts=[load(a) for p in papers for a in (OUT/'generation'/label/p['document_id']).glob('attempt*.json')]
    assert all(reasoning_tokens(a['response'])==0 for a in attempts)
    completion=sum(a['response'].get('usage',{}).get('completion_tokens',0) for a in attempts)
    write(path,dict(papers=97,successful_papers=sum(r['status']=='generated' for r in rows),
        elapsed_seconds=seconds,active_gpus=1,active_generation_gpu_hours=seconds/3600,
        completion_tokens_all_attempts=completion,prompt_tokens_all_attempts=sum(a['response'].get('usage',{}).get('prompt_tokens',0) for a in attempts),
        reasoning_tokens_all_attempts=0,thinking=False,api_calls=len(attempts),
        output_tokens_per_second=completion/seconds,mean_native_narrative_tokens=statistics.mean(r.get('native_narrative_tokens',0) for r in rows),
        concurrency=concurrency,restarters_and_server_setup_excluded=True,generation_segments=segments,
        includes_full_outputs_tokenization_prefill_and_format_retries=True))
    jsonl(OUT/(label+'-summaries.jsonl'),rows)
    state(label,state='complete')


def primary(papers,prompts):
    label='no_thinking_matched';server=None
    try:
        server=Server(ROOT/'models/gemma-4-12b-it',19600,label,parser='gemma4',gpu=0,adapter=ROOT/'models/r128-adapter')
        cohort(server,label,papers,prompts,16,'trained')
    except Exception as e:
        state(label,state='failed',error=repr(e));raise
    finally:
        if server:server.close()


def tuned(label,gpu,papers,probes,prompts,configs):
    candidates=[];errors=[];port=19601+gpu
    for index,config in enumerate(configs):
        tag=label+'-candidate'+str(index);server=None
        try:
            state(label,state='tuning',config=config)
            server=Server(ROOT/'models/merged-r128-bf16',port,tag,parser='gemma4',gpu=gpu,cache_reset=True,**config)
            for c in [16,32,64]:
                result=probe(server,probes,prompts,c,tag,'summary-model')
                candidates.append(dict(config=config,concurrency=c,probe=result,tag=tag))
        except Exception as e:
            errors.append(dict(config=config,error=repr(e),tag=tag))
            print('Unsupported/failed optimization',tag,repr(e),flush=True)
        finally:
            if server:server.close()
    if not candidates:
        write(ROOT/'outputs/optimization'/(label+'.json'),dict(state='unavailable',errors=errors))
        state(label,state='unavailable',errors=errors)
        return
    chosen=max(candidates,key=lambda c:c['probe']['completion_tokens_per_second'])
    write(ROOT/'outputs/optimization'/(label+'.json'),dict(state='selected',selection=chosen,
        candidates=candidates,unsupported_candidates=errors,selection_uses_held_out_qa=False,
        probe_speed_is_not_full_summary_cost=True))
    server=None
    try:
        server=Server(ROOT/'models/merged-r128-bf16',port,label,parser='gemma4',gpu=gpu,**chosen['config'])
        cohort(server,label,papers,prompts,chosen['concurrency'],'summary-model')
    except Exception as e:state(label,state='failed',error=repr(e));raise
    finally:
        if server:server.close()


def qa_loop(papers,done):
    server=Server(ROOT/'models/Qwen2.5-7B-Instruct',19603,'fixed-qa',gpu=3,judge=True)
    try:
        backend=OpenAICompatibleBackend('qwen25',base_url=server.endpoint+'/v1',api_key='',max_tokens=100,
            temperature=.5,concurrency=4,thinking=False,frequency_penalty=1.05,presence_penalty=1.05,timeout=600)
        evaluate.CONDITIONS=('qwen_summary',)
        results=load(OUT/'qa-results.json');docs={d['document_id']:d for d in results['documents']}
        while True:
            progress=False
            for p in papers:
                d=docs[p['document_id']]
                for label in ARMS:
                    path=OUT/'generation'/label/p['document_id']/'result.json'
                    if label in d['conditions'] or not path.exists():continue
                    generated=load(path);context=generated['judge_context'];start=time.monotonic()
                    if generated['status']=='generated':
                        proxy=dict(document_id=p['document_id'],viewer_config=p['source']['viewer_config'],
                            fulltext=p['fulltext'],existing_summary=p['existing_summary'],qwen_summary=context)
                        value=evaluate.run_document(proxy,p['questions'],backend,context_limit=32768)
                        value['generation_failed']=False
                    else:
                        value=dict(generation_failed=True,rows=[dict(question_index=q['question_index'],gold=q['answer'],
                            predictions={'qwen_summary':None},responses={'qwen_summary':dict(generation_failed=True,
                            error=generated.get('error'),prompt_sha256=sha(evaluate.historical_answer_prompt(q['formatted_question'],'')))}) for q in p['questions']])
                    value.update(context_sha256=sha(context),elapsed_seconds=time.monotonic()-start)
                    d['conditions'][label]=value;write(OUT/'qa-results.json',results)
                    print('QA',label,p['document_id'],flush=True);progress=True
            if done.is_set() and not progress:break
            if not progress:time.sleep(2)
    finally:server.close()


def bootstrap(values):
    rng=random.Random(250219413)
    samples=sorted(sum(values[rng.randrange(len(values))] for _ in values)/(10*len(values)) for _ in range(10000))
    return [samples[249],samples[9749]]


def finalize(papers):
    qa=load(OUT/'qa-results.json');by_id={d['document_id']:d for d in qa['documents']}
    available=['thinking_reference']+[a for a in ARMS if STATES[a]['state']=='complete']
    assert 'no_thinking_matched' in available
    scores={};performance={};per={}
    for label in available:
        values=[];correct=invalid=0;contexts=[]
        for p in papers:
            d=by_id[p['document_id']];assert d['questions']==p['questions'] and d['fulltext_sha256']==p['fulltext_sha256']
            generated=load(OUT/'generation'/label/p['document_id']/'result.json');contexts.append(generated)
            condition=d['conditions'][label];assert condition['context_sha256']==sha(generated['judge_context']) and len(condition['rows'])==10
            n=0
            for row,q in zip(condition['rows'],p['questions']):
                assert row['gold']==q['answer'] and row['question_index']==q['question_index']
                if not condition['generation_failed']:
                    assert row['responses']['qwen_summary']['prompt_sha256']==sha(evaluate.historical_answer_prompt(q['formatted_question'],generated['judge_context']))
                prediction=next(iter(row['predictions'].values()));n+=prediction==q['answer'];invalid+=prediction is None
            values.append(n);correct+=n
        per[label]=values
        performance[label]=load(OUT/(label+'-generation-performance.json'))
        scores[label]=dict(name=NAMES[label],correct=correct,total=970,accuracy=correct/970,ci95=bootstrap(values),
            invalid_predictions=invalid,failed_generation_papers=sum(c['status']!='generated' for c in contexts),
            mean_narrative_words=statistics.mean(len(c['judge_context'].split()) for c in contexts),
            mean_native_narrative_tokens=performance[label]['mean_native_narrative_tokens'])
    assert scores['thinking_reference']['correct']==897
    differences={}
    for label in available[1:]:
        differences[label+'_minus_thinking']=dict(difference=(scores[label]['correct']-897)/970,
            ci95=bootstrap([a-b for a,b in zip(per[label],per['thinking_reference'])]))
    for label in available[2:]:
        differences[label+'_minus_matched_no_thinking']=dict(difference=(scores[label]['correct']-scores['no_thinking_matched']['correct'])/970,
            ci95=bootstrap([a-b for a,b in zip(per[label],per['no_thinking_matched'])]))
    scaling=[]
    for label in available:
        perf=performance[label];success=97-scores[label]['failed_generation_papers']
        if not success:continue
        for count in [38_000_000,60_000_000,98_000_000]:
            measured=perf['elapsed_seconds']/97*count/3600
            scaling.append(dict(condition=label,papers=count,measured_rate_gpu_hours_per_attempted_paper=measured,
                gpu_hours_at_85pct_capacity=measured/.85,
                planning_gpu_hours_for_successful_outputs=measured/.85*97/success,
                observed_successful_papers=success,observed_attempted_papers=97,
                startup_qa_training_semantic_correction_and_source_acquisition_excluded=True))
    report=dict(papers=97,questions_per_condition=970,models=scores,generation_performance=performance,
        paired_differences=differences,scaling=scaling,arm_status=STATES,
        protocol=load(OUT/'protocol.json'),audit_passed=True,
        conclusions=dict(qa_measures_summary_answerability_with_fixed_judge=True,
            independent_human_factual_superiority_not_established=True,
            length_confound_reported=True,optimization_selection_has_no_held_out_qa=True))
    write(OUT/'report.json',report)
    lines=['# Gemma 4 12B IT rank-128: thinking ablation and inference optimization','',
        'Same Qwen-distilled adapter, 97 frozen papers and 970 immutable MCQs. '
        'No semantic correction. Format failures count as ten wrong answers. '
        'The thinking reference is reused verbatim from the audited completed experiment.','',
        '| Condition | Correct / 970 | QA accuracy | Failed papers | Mean narrative words | Mean narrative tokens | Generation seconds | Completion tokens/s/GPU |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label in available:
        s=scores[label];p=performance[label]
        lines.append(f"| {s['name']} | {s['correct']} | {100*s['accuracy']:.2f}% | {s['failed_generation_papers']} | {s['mean_narrative_words']:.0f} | {s['mean_native_narrative_tokens']:.0f} | {p['elapsed_seconds']:.1f} | {p['output_tokens_per_second']:.1f} |")
    lines+=['','## Paired QA differences (95% paper-bootstrap intervals)','']
    for key,v in differences.items():lines.append(f"- {key}: {100*v['difference']:+.2f} percentage points (95% CI {100*v['ci95'][0]:+.2f} to {100*v['ci95'][1]:+.2f}).")
    lines+=['','## Scaling on GH200 GPUs','',
        '| Condition | Papers | Measured-rate GPU-hours | Planning GPU-hours (85% useful capacity and observed output yield) |',
        '| --- | ---: | ---: | ---: |']
    for s in scaling:lines.append(f"| {NAMES[s['condition']]} | {s['papers']:,} | {s['measured_rate_gpu_hours_per_attempted_paper']:,.0f} | {s['planning_gpu_hours_for_successful_outputs']:,.0f} |")
    lines+=['','These extrapolations use actual complete generation, including prefill, native tokenization and all bounded format retries. '
        'They exclude QA, training, server startup, semantic correction and source acquisition. '
        'The 85% capacity factor is a planning assumption. Output-yield normalization assumes a comparable future paper mix; '
        'it does not guarantee recovery of difficult failed papers. Use four independent model replicas per Jupiter node to '
        'avoid paying for idle GPUs. Short capped probes select deployment settings and are never used as full-summary costs.','',
        '## Optimization protocol','',
        '64 deterministic non-held-out training papers across ten domains; concurrency 16/32/64; '
        '512-token partial outputs with cold source prefix caches. Runtime selection is made before held-out QA. '
        'Merged BF16 and FP8 are evaluated on all 97 papers because merging and quantization can change outputs. '
        'Unsupported backend/KV-cache configurations are retained in optimization logs. '
        'The original baseline already uses FlashAttention 4, CUDA graphs, prefix caching and chunked prefill.','',
        'QA tests answerability under the pinned Qwen2.5-7B student and synthetic MCQs. It is not an independent human factuality assessment. '
        'Summary length and formatting failure rate are reported alongside accuracy.','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    write(OUT/'complete.json',dict(audit_passed=True,conditions=len(available),scored_answers=970*len(available),
        job_id=os.environ.get('SLURM_JOB_ID'),unsupported_arms=[a for a in ARMS if a not in available]))


def main():
    papers,probes,prompts=prepare()
    done=threading.Event();errors=[]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        qa=pool.submit(qa_loop,papers,done)
        baseline=pool.submit(primary,papers,prompts)
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                PYTHONPATH=str(TRAINING_ROOT/'python_packages'),TOKENIZERS_PARALLELISM='false')
            with (ROOT/'logs/merge.log').open('a') as log:
                result=subprocess.run([str(TRAINING_PY),str(ROOT/'code/merge_adapter.py')],env=env,stdout=log,stderr=log)
            if result.returncode:raise RuntimeError('Adapter merge failed; see logs/merge.log')
            bf16=pool.submit(tuned,'no_thinking_merged',1,papers,probes,prompts,[
                dict(max_batched_tokens=8192),dict(max_batched_tokens=16384),
                dict(max_batched_tokens=16384,attention_backend='FLASHINFER')])
            fp8=pool.submit(tuned,'no_thinking_fp8',2,papers,probes,prompts,[
                dict(max_batched_tokens=16384,quantization='fp8_per_tensor'),
                dict(max_batched_tokens=16384,quantization='fp8_per_tensor',kv_cache_dtype='fp8')])
            for future in [bf16,fp8]:
                try:future.result()
                except Exception as e:errors.append(repr(e))
        except Exception as e:
            errors.append(repr(e))
            for label in ARMS[1:]:state(label,state='unavailable',error=repr(e))
        finally:
            try:baseline.result()
            except Exception as e:errors.append(repr(e))
            done.set()
        qa.result()
    write(ROOT/'outputs/runtime_errors.json',dict(errors=errors))
    finalize(papers)
    from publish import publish
    publish()


if __name__=='__main__':main()
