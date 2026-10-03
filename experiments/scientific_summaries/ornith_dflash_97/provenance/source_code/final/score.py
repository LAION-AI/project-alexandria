import hashlib
import random
import statistics
import time
from common import ROOT,VENDOR,Client,load,write
import evaluate
from project_alexandria.backends import OpenAICompatibleBackend

def bootstrap(values):
    rng=random.Random(evaluate.SEED)
    samples=sorted(sum(values[rng.randrange(len(values))] for _ in values)/(10*len(values)) for _ in range(10000))
    return [samples[249],samples[9749]]

def score_all(endpoint):
    bundle=load(VENDOR/'data/testset.json')
    papers=bundle['papers']
    original=load(VENDOR/'data/papers.json')
    # Verify exported evaluated option order against the pinned repository exactly.
    for paper in papers:
        generated=evaluate.questions_for(original[paper['original_paper_index']],paper['original_paper_index'])
        if evaluate.question_signature(generated)!=evaluate.question_signature(paper['questions']):
            raise ValueError('Immutable exported option order differs from repository')
    backend=OpenAICompatibleBackend('qwen25',base_url=endpoint+'/v1',api_key='',max_tokens=100,
        temperature=.5,concurrency=4,thinking=False,frequency_penalty=1.05,presence_penalty=1.05,timeout=600)
    caches={}
    labels={
        'no_context':'No context', 'original':'Full paper text', 'summary':'Existing Gemini summary',
        'qwen38_reference':'Qwen3.8-27B AutoRound mixed INT4 / repository repaired summary',
        'ornith_raw':'Ornith-1.5-9B BF16 + DFlash / raw summary',
        'ornith_corrected':'Ornith-1.5-9B BF16 + DFlash / corrected summary',
        'qwen38_raw':'Qwen3.8-27B official FP8 / raw summary',
        'qwen38_corrected':'Qwen3.8-27B official FP8 / corrected summary'}
    reference=load(VENDOR/'summary_runs/qwen27b/summaries.json')
    caches['qwen38_reference']={d['document_id']:d for d in reference['documents']}
    for model,prefix in [('ornith_dflash','ornith'),('qwen38_fp8','qwen38')]:
        for condition in ('raw','corrected'):
            cache=load(ROOT/'outputs/cohorts'/model/(condition+'-summaries.json'))
            caches[prefix+'_'+condition]={d['document_id']:d for d in cache['documents']}
    results=load(ROOT/'outputs/qa-results.json') if (ROOT/'outputs/qa-results.json').exists() else dict(
        judge_model='Qwen/Qwen2.5-7B-Instruct',judge_revision=load(ROOT/'inputs/judge-hf-api.json')['sha'],
        protocol=dict(temperature=.5,top_p=.95,max_tokens=100,frequency_penalty=1.05,presence_penalty=1.05,
            concurrency=4,attempts=5,thinking=False,precision='BF16',runtime='vLLM 0.30.0',
            max_model_len=32768,parser='historical_semicolon_v1',legacy_ascii_sanitizer=True,
            option_order='exported immutable order',bootstrap_seed=evaluate.SEED,bootstrap_resamples=10000),
        labels=labels,documents=[],elapsed_seconds=0)
    by_id={d['document_id']:d for d in results['documents']}
    for paper in papers:
        identifier=paper['document_id']
        document=by_id.get(identifier,dict(document_id=identifier,fulltext_sha256=paper['fulltext_sha256'],
            questions=paper['questions'],conditions={}))
        if document['questions']!=paper['questions'] or document['fulltext_sha256']!=paper['fulltext_sha256']:
            raise ValueError('Checkpoint source or questions changed')
        for condition in labels:
            missing = condition in caches and identifier not in caches[condition]
            expected_context = '' if condition=='no_context' else (paper['fulltext'] if condition=='original'
                else (paper['existing_summary'] if condition=='summary' else
                (None if missing else caches[condition][identifier]['judge_context'])))
            if condition in document['conditions']:
                previous=document['conditions'][condition]
                if missing and previous.get('generation_failed'):
                    continue
                if not missing and not previous.get('generation_failed'):
                    expected_sha=hashlib.sha256(expected_context.encode()).hexdigest()
                    canonical=condition if condition in ('no_context','original','summary') else 'qwen_summary'
                    if previous.get('context_sha256')==expected_sha and all(
                        row['responses'][canonical]['prompt_sha256']==hashlib.sha256(
                        evaluate.historical_answer_prompt(question['formatted_question'],expected_context).encode()).hexdigest()
                        for row,question in zip(previous['rows'],paper['questions'])):
                        continue
                document.setdefault('superseded_conditions',[]).append(dict(condition=condition,record=previous))
            started=time.monotonic()
            if missing:
                # Do not let a failed summary obtain prior-knowledge credit from an empty context.
                value=dict(generation_failed=True,rows=[dict(question_index=q['question_index'],
                    gold=q['answer'],predictions={'qwen_summary':None},responses={}) for q in paper['questions']])
            else:
                canonical=condition if condition in ('no_context','original','summary') else 'qwen_summary'
                evaluate.CONDITIONS=(canonical,)
                context='' if canonical=='no_context' else (paper['fulltext'] if canonical=='original'
                    else (paper['existing_summary'] if canonical=='summary' else caches[condition][identifier]['judge_context']))
                if condition in caches and caches[condition][identifier]['fulltext_sha256']!=paper['fulltext_sha256']:
                    raise ValueError('Summary source changed')
                proxy=dict(document_id=identifier,viewer_config=paper['source']['viewer_config'],
                    fulltext=paper['fulltext'],existing_summary=paper['existing_summary'],qwen_summary=context)
                value=evaluate.run_document(proxy,paper['questions'],backend,context_limit=32768)
                value['context_sha256']=hashlib.sha256(context.encode()).hexdigest()
            value['elapsed_seconds']=time.monotonic()-started
            document['conditions'][condition]=value
            results['elapsed_seconds']+=value['elapsed_seconds']
            by_id[identifier]=document
            results['documents']=[by_id[p['document_id']] for p in papers if p['document_id'] in by_id]
            write(ROOT/'outputs/qa-results.json',results)
            print('QA',identifier,condition,len(results['documents']),flush=True)
    report=dict(papers=97,questions=970,models={},paired_differences={},generation={},performance_recipe=load(ROOT/'outputs/performance_recipe.json'))
    per_document={}
    for condition,label in labels.items():
        counts=[]
        invalid=0
        failed=0
        for document in results['documents']:
            item=document['conditions'][condition]
            rows=item['rows']
            counts.append(sum(next(iter(r['predictions'].values()))==r['gold'] for r in rows))
            invalid+=sum(next(iter(r['predictions'].values())) is None for r in rows)
            failed+=bool(item.get('generation_failed'))
        per_document[condition]=counts
        report['models'][condition]=dict(name=label,correct=sum(counts),total=970,accuracy=sum(counts)/970,
            invalid=invalid,failed_generation_papers=failed,ci95=bootstrap(counts))
    for left,right in [('ornith_corrected','ornith_raw'),('qwen38_corrected','qwen38_raw'),
        ('ornith_raw','qwen38_raw'),('ornith_corrected','qwen38_corrected'),
        ('ornith_corrected','qwen38_reference'),('ornith_corrected','original')]:
        differences=[l-r for l,r in zip(per_document[left],per_document[right])]
        report['paired_differences'][left+'_minus_'+right]=dict(difference=sum(differences)/970,ci95=bootstrap(differences))
    for model in ('ornith_dflash','qwen38_fp8'):
        completion=load(ROOT/'outputs/cohorts'/model/'complete.json')
        phases={}
        for path in sorted((ROOT/'outputs/cohorts'/model/'documents').glob('*/document.json')):
            doc=load(path)
            for call in doc['attempts']:
                phase=call.get('phase','structural_repair')
                group=phases.setdefault(phase,dict(calls=0,completion_tokens=0,request_seconds=[],prompt_tokens=0))
                group['calls']+=1
                group['completion_tokens']+=call['usage'].get('completion_tokens',0)
                group['prompt_tokens']+=call['usage'].get('prompt_tokens',0)
                group['request_seconds'].append(call['elapsed_seconds'])
        for phase,value in phases.items():
            times=sorted(value.pop('request_seconds'))
            value.update(mean_request_seconds=statistics.mean(times),median_request_seconds=statistics.median(times),
                p95_request_seconds=times[min(len(times)-1,int(.95*len(times)))])
        report['generation'][model]=dict(completion=completion,phases=phases)
    write(ROOT/'outputs/report.json',report)
    lines=['# Scientific summary evaluation: Ornith + DFlash','',
        '97 frozen papers, 970 immutable MCQs; Qwen2.5-7B-Instruct BF16 judge, historical repository prompt/parser.',
        '', '| Model / condition | Correct | Accuracy | 95% paper bootstrap CI |',
        '|---|---:|---:|---:|']
    for value in report['models'].values():
        lines.append('| '+value['name']+' | '+str(value['correct'])+'/970 | '+f"{100*value['accuracy']:.2f}% | {100*value['ci95'][0]:.2f}–{100*value['ci95'][1]:.2f}% |")
    lines+=['','## Paired differences','']
    for key,value in report['paired_differences'].items():
        lines.append('- '+key+': '+f"{100*value['difference']:+.2f} percentage points, CI {100*value['ci95'][0]:+.2f} to {100*value['ci95'][1]:+.2f}.")
    lines+=['','## Interpretation limits','',
        '- Source-only self-audits are generated by the same model; the QA evaluator is fixed and separate.',
        '- New Ornith and Qwen FP8 runs share the initial prompt and correction policy; model size and precision differ.',
        '- Repository Qwen INT4 uses a different historical repair policy and is a separate reference.',
        '- 512-token throughput probes choose runtime/batching; actual complete-paper latencies are in report.json.',
        '- Test paper outputs are evaluation artifacts and never enter the 1000-paper training dataset.']
    (ROOT/'outputs/report.md').write_text('\n'.join(lines)+'\n')
    return report
