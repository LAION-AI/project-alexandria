"""Evaluate four Gemma conditions, auditing all ten reused baseline conditions."""
import copy
import hashlib
import random
import statistics
import time
from common import ROOT,VENDOR,load,write,digest
import evaluate
from copy_overlap_eval import audit_cohort_root
from project_alexandria.backends import OpenAICompatibleBackend

def bootstrap(values):
    rng=random.Random(evaluate.SEED)
    samples=sorted(sum(values[rng.randrange(len(values))] for _ in values)/(10*len(values)) for _ in range(10000))
    return [samples[249],samples[9749]]

def all_contexts():
    manifest=load(ROOT/'inputs/experiment_manifest.json')
    old=__import__('pathlib').Path(manifest['comparison_run'])
    old9=__import__('pathlib').Path(manifest['comparison_9b_run'])
    papers=load(VENDOR/'data/testset.json')['papers']
    contexts={p['document_id']:dict(no_context='',original=p['fulltext'],summary=p['existing_summary']) for p in papers}
    for item in load(VENDOR/'summary_runs/qwen27b/summaries.json')['documents']:
        contexts[item['document_id']]['qwen38_reference']=item['judge_context']
    for base,model,prefix in [(old9,'ornith_dflash','ornith'),(old9,'qwen38_fp8','qwen38'),(old,'ornith35_dflash','ornith35'),(ROOT,'gemma4_e4b','gemma4_e4b'),(ROOT,'gemma4_12b','gemma4_12b')]:
        for phase in ('raw','corrected'):
            items=load(base/'outputs/cohorts'/model/(phase+'-summaries.json'))['documents']
            if len(items)!=97 or {d['document_id'] for d in items}!=set(contexts):
                raise ValueError('Incomplete fixed-cohort cache')
            for item in items:
                contexts[item['document_id']][prefix+'_'+phase]=item['judge_context']
    return contexts

def validate_checkpoint(results,contexts):
    bundle={p['document_id']:p for p in load(VENDOR/'data/testset.json')['papers']}
    for doc in results['documents']:
        paper=bundle[doc['document_id']]
        if doc['questions']!=paper['questions'] or doc['fulltext_sha256']!=paper['fulltext_sha256']:
            raise ValueError('Reused QA source or immutable questions changed')
        for label,condition in doc['conditions'].items():
            context=contexts[doc['document_id']][label]
            if condition.get('generation_failed') or condition['context_sha256']!=digest(context) or len(condition['rows'])!=10:
                raise ValueError('Stale reused QA context')
            canonical=label if label in ('no_context','original','summary') else 'qwen_summary'
            for row,question in zip(condition['rows'],paper['questions']):
                if row['gold']!=question['answer'] or row['responses'][canonical]['prompt_sha256']!=digest(
                        evaluate.historical_answer_prompt(question['formatted_question'],context)):
                    raise ValueError('Reused QA prompt/gold mismatch')

def score_all(endpoint):
    manifest=load(ROOT/'inputs/experiment_manifest.json')
    old=__import__('pathlib').Path(manifest['comparison_run'])
    papers=load(VENDOR/'data/testset.json')['papers']
    original=load(VENDOR/'data/papers.json')
    for p in papers:
        questions=evaluate.questions_for(original[p['original_paper_index']],p['original_paper_index'])
        if evaluate.question_signature(questions)!=evaluate.question_signature(p['questions']):
            raise ValueError('Exported question option order changed')
    expected_protocol=load(old/'outputs/qa-results.json')['protocol']
    results=load(ROOT/'outputs/qa-results.json') if (ROOT/'outputs/qa-results.json').exists() else load(old/'outputs/qa-results.json')
    if results['judge_revision']!=load(ROOT/'inputs/judge-hf-api.json')['sha'] or results['protocol']!=expected_protocol:
        raise ValueError('Fixed judge/runtime/decoding changed')
    if set(expected_protocol)!=set(load(old/'outputs/qa-results.json')['protocol']):
        raise ValueError('Protocol inventory changed')
    labels=dict(results['labels'])
    for model in ('gemma4_e4b','gemma4_12b'):
        for phase in ('raw','corrected'):
            labels[model+'_'+phase]=manifest['models'][model]['id']+' BF16 / '+phase+' summary'
    results['labels']=labels
    results['baseline_provenance']=dict(path=str(old/'outputs/qa-results.json'),
        sha256=hashlib.sha256((old/'outputs/qa-results.json').read_bytes()).hexdigest(),
        reused_conditions=[k for k in labels if not k.startswith('gemma4_')],
        note='All reused source/context/prompt hashes and immutable gold/order checked; no baseline re-generation.')
    contexts=all_contexts()
    validate_checkpoint(results,contexts)
    write(ROOT/'outputs/qa-results.json',results)
    backend=OpenAICompatibleBackend('qwen25',base_url=endpoint+'/v1',api_key='',max_tokens=100,
        temperature=.5,concurrency=4,thinking=False,frequency_penalty=1.05,presence_penalty=1.05,timeout=600)
    by_id={d['document_id']:d for d in results['documents']}
    for paper in papers:
        document=by_id[paper['document_id']]
        for label in ('gemma4_e4b_raw','gemma4_e4b_corrected','gemma4_12b_raw','gemma4_12b_corrected'):
            if label in document['conditions']:
                continue
            started=time.monotonic()
            context=contexts[paper['document_id']][label]
            proxy=dict(document_id=paper['document_id'],viewer_config=paper['source']['viewer_config'],
                       fulltext=paper['fulltext'],existing_summary=paper['existing_summary'],qwen_summary=context)
            evaluate.CONDITIONS=('qwen_summary',)
            value=evaluate.run_document(proxy,paper['questions'],backend,context_limit=32768)
            value['context_sha256']=digest(context)
            value['elapsed_seconds']=time.monotonic()-started
            document['conditions'][label]=value
            results['elapsed_seconds']+=value['elapsed_seconds']
            write(ROOT/'outputs/qa-results.json',results)
            print('QA',paper['document_id'],label,flush=True)
    validate_checkpoint(results,contexts)
    report=load(old/'outputs/report.json')
    report.pop('new_target',None)
    report.pop('new_model_qa_speculative_configuration',None)
    report['baseline_provenance']=results['baseline_provenance']
    report['new_models']=manifest['models']
    report['new_model_performance_recipe']=load(ROOT/'outputs/performance_recipe.json')
    report.pop('gpu_accounting',None)
    report.pop('final_audit',None)
    counts={}
    for label,name in labels.items():
        per=[];invalid=0
        for d in results['documents']:
            rows=d['conditions'][label]['rows']
            per.append(sum(next(iter(r['predictions'].values()))==r['gold'] for r in rows))
            invalid+=sum(next(iter(r['predictions'].values())) is None for r in rows)
        counts[label]=per
        report['models'][label]=dict(name=name,correct=sum(per),total=970,accuracy=sum(per)/970,
                                   invalid=invalid,failed_generation_papers=0,ci95=bootstrap(per))
    for model in ('gemma4_e4b','gemma4_12b'):
        for left,right in [(model+'_corrected',model+'_raw'),(model+'_raw','ornith_raw'),
                           (model+'_corrected','ornith_corrected'),(model+'_raw','ornith35_raw'),
                           (model+'_corrected','ornith35_corrected'),(model+'_raw','qwen38_raw'),
                           (model+'_corrected','qwen38_corrected'),(model+'_raw','original')]:
            values=[a-b for a,b in zip(counts[left],counts[right])]
            report['paired_differences'][left+'_minus_'+right]=dict(difference=sum(values)/970,ci95=bootstrap(values))
        phases={}
        for path in sorted((ROOT/'outputs/cohorts'/model/'documents').glob('*/document.json')):
            doc=load(path)
            for call in doc['attempts']:
                phase=call.get('phase','structural_repair')
                group=phases.setdefault(phase,dict(calls=0,completion_tokens=0,prompt_tokens=0,request_seconds=[]))
                group['calls']+=1;group['completion_tokens']+=call['usage'].get('completion_tokens',0)
                group['prompt_tokens']+=call['usage'].get('prompt_tokens',0);group['request_seconds'].append(call['elapsed_seconds'])
        for value in phases.values():
            times=sorted(value.pop('request_seconds'))
            value.update(mean_request_seconds=statistics.mean(times),median_request_seconds=statistics.median(times),
                         p95_request_seconds=times[min(len(times)-1,int(.95*len(times)))])
        report['generation'][model]=dict(completion=load(ROOT/'outputs/cohorts'/model/'complete.json'),phases=phases)
        for phase in ('raw','corrected'):
            values=load(ROOT/'outputs/cohorts'/model/(phase+'-summaries.json'))['documents']
            lengths=[len(d['judge_context'].split()) for d in values]
            sources={p['document_id']:len(p['fulltext'].split()) for p in papers}
            report['summary_lengths'][model+'_'+phase]=dict(mean_words=statistics.mean(lengths),
                median_words=statistics.median(lengths),min_words=min(lengths),max_words=max(lengths),
                mean_context_to_source_word_ratio=statistics.mean(len(d['judge_context'].split())/sources[d['document_id']] for d in values))
    values=[a-b for a,b in zip(counts['gemma4_12b_raw'],counts['gemma4_e4b_raw'])]
    report['paired_differences']['gemma4_12b_raw_minus_gemma4_e4b_raw']=dict(difference=sum(values)/970,ci95=bootstrap(values))
    report['copy_overlap']=audit_cohort_root(ROOT,papers,['gemma4_e4b', 'gemma4_12b'])
    write(ROOT/'outputs/report.json',report)
    lines=['# Gemma 4 E4B IT and 12B IT: 97-paper scientific-summary results','',
        'Same frozen 97 papers / 970 MCQs; fixed BF16 Qwen2.5-7B student. Ten previous conditions reused after exact hashes; four new conditions scored.','',
        '| Condition | Correct / 970 | Accuracy | 95% paper-bootstrap CI | Invalid |','| --- | ---: | ---: | ---: | ---: |']
    for v in report['models'].values():
        lines.append(f"| {v['name']} | {v['correct']} | {100*v['accuracy']:.2f}% | {100*v['ci95'][0]:.2f}–{100*v['ci95'][1]:.2f}% | {v['invalid']} |")
    lines+=['','## Paired differences','']
    for k,v in report['paired_differences'].items():
        lines.append(f"- {k}: {100*v['difference']:+.2f} percentage points, CI {100*v['ci95'][0]:+.2f} to {100*v['ci95'][1]:+.2f}.")
    lines+=['','## Measured throughput per GH200 GPU','',
        '512-token probes on 64 non-holdout sources. Cold generation / repeat-input correction (cache hits depend on capacity and eviction); full-paper request latencies are separate.','',
        '| Runtime | Batch | Generation cold tok/s | Generation warm tok/s | Correction cold tok/s | Correction warm tok/s |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode in ('gemma4_e4b','gemma4_12b'):
        batches=load(ROOT/'outputs/throughput'/(mode+'.json'))['batches']
        for concurrency in (1,4,8,16,32,64):
            values=[next(x['output_tokens_per_second'] for x in batches if x['phase']==p and x['cache_state']==c and x['concurrency']==concurrency)
                    for p,c in [('generation','cold'),('generation','warm'),('correction','cold'),('correction','warm')]]
            lines.append('| '+mode+' | '+str(concurrency)+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines+=['','## Interpretation','',
        'These scores measure complete deployed summary pipelines. Output lengths are not equalized; BF16 Gemma and Ornith targets differ in architecture and size; Qwen27B is FP8. Same-model audit scores are not human factual labels. All evaluation outputs are held out from LoRA training.']
    (ROOT/'outputs/report.md').write_text('\n'.join(lines)+'\n')
    return report
