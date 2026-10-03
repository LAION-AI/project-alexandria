"""Source-only recovery of mechanical failures, then reconcile changed QA prompts.

Recovery never reads MCQs, predictions, answer keys or QA scores. It keeps raw
summaries unchanged when available and repairs only unusable corrected artifacts.
"""
import concurrent.futures
import copy
import hashlib
import json
import shutil
import time
from common import (ROOT, PROMPT, Client, load, write, digest, source_user, correction_user,
    parse_summary, validate_summary, invalid_fields, narrative_context, parse_json)
from run_cohort import review
from orchestrate import start_server, stop, OWNED, status
from summary_runtime import REPAIR_SYSTEM, ANCHOR_SYSTEM, quote_tasks
from summary_repair_v3 import FIELD_USER_TEMPLATE, field_example, normalize_shapes, _apply_anchors

class RecoveryClient(Client):
    def post(self,route,payload,timeout=1800):
        if route=='/v1/chat/completions' and payload.get('max_tokens',0)>4096:
            # A remaining failed document repeats an identical sentence until
            # 32768 tokens. Use the model-card anti-repetition setting only for
            # recovery of unusable long outputs; both models use this same policy.
            payload['presence_penalty']=1.5
        return super().post(route,payload,timeout)

def recover(paper, model, endpoint):
    path=ROOT/'outputs/cohorts'/model/'documents'/paper['document_id']/'document.json'
    result=load(path)
    if result.get('raw') and result.get('corrected') and result.get('corrected_quality_assessment'):
        return result
    backup=path.with_name('before-source-recovery-'+str(time.time_ns())+'.json')
    shutil.copyfile(path,backup)
    client=RecoveryClient(endpoint,'summary-model',path.parent/'calls')
    journal=result['attempts']
    seed=int(digest(paper['document_id']+'|source-recovery')[:8],16)%2147483000
    result.setdefault('source_recovery_history',[]).append(dict(prior_error=result.get('error'),
        backup_file=str(backup.relative_to(ROOT)),policy='bounded_single_task_literal_anchor_recovery_v1'))
    started=time.monotonic()
    try:
        if result.get('raw') and result.get('corrected'):
            result['corrected_quality_assessment']=review(paper,result['corrected']['summary'],client,journal,'source_recovery_corrected_review')
            result.pop('error',None)
            result['elapsed_seconds']+=time.monotonic()-started
            write(path,result)
            return result
        if not result.get('raw'):
            for attempt in range(3):
                user=source_user(paper)+'\n\nFORMAT_RECOVERY\nEarlier drafts exceeded the output budget or had malformed fields. Return a complete 19-field object. Use only literal proof fragments of at most five words, never long copied passages. Avoid repeated narrative statements and invented details. Aim for 2500–4000 narrative words when the source supports that detail; a short source may require less. Do not add filler to meet field lengths.'
                budget=min(32768,client.context_limit-client.token_count([{'role':'system','content':PROMPT},{'role':'user','content':user}]))
                record=client.generate(PROMPT,user,budget,seed+attempt,phase='source_recovery_generation')
                journal.append(record)
                try:
                    state=parse_summary(record)
                    break
                except (ValueError,KeyError,TypeError) as error:
                    record['validation_error']=str(error)
            else:
                raise ValueError('Source recovery generation did not produce usable schema')
            result['raw']=dict(summary=copy.deepcopy(state),judge_context=narrative_context(state),
                quote_validation_errors=invalid_fields(state,paper['fulltext']),raw_generation_trace=record.get('trace_file'),
                elapsed_seconds=time.monotonic()-started)
        raw=result['raw']['summary']
        failure=result.get('structural_repair_failure')
        if failure and failure.get('draft_summary'):
            state=copy.deepcopy(failure['draft_summary'])
        else:
            audit=result.get('raw_quality_assessment')
            if not audit:
                audit=review(paper,raw,client,journal,'source_recovery_raw_review')
                result['raw_quality_assessment']=audit
            for attempt in range(2):
                user=correction_user(paper,raw,audit)+'\nKeep proof fragments at most five words and avoid duplicate narrative facts. Return the complete object; do not add filler.'
                budget=min(24576,client.context_limit-client.token_count([{'role':'system','content':PROMPT},{'role':'user','content':user}]))
                record=client.generate(PROMPT,user,budget,
                    seed+30+attempt,phase='source_recovery_semantic_correction')
                journal.append(record)
                try:
                    state=parse_summary(record)
                    break
                except (ValueError,KeyError,TypeError) as error:
                    record['validation_error']=str(error)
            else:
                raise ValueError('Source recovery semantic correction failed')
        state,_=normalize_shapes(state)
        # Quote tasks are handled independently. One invalid index can no longer
        # discard valid decisions for seven other statements in the same batch.
        tasks=quote_tasks(state,paper['fulltext'])
        for task in reversed(tasks):
            public={k:v for k,v in task.items() if k!='path'}
            public['id']='q0'
            public['candidates']=[dict(index=i,**candidate) for i,candidate in enumerate(task['candidates'])]
            prompt='BEGIN_PAPER\n'+paper['fulltext']+'\nEND_PAPER\n\nTASKS\n'+json.dumps([public],ensure_ascii=False)
            prompt+='\nReturn exactly {"q0": INTEGER_INDEX_OR_DROP}. Choose only an index explicitly provided. If none supports the statement, choose "drop". Never invent an index.'
            for attempt in range(2):
                record=client.generate(ANCHOR_SYSTEM,prompt,512,seed+100+len(journal),phase='source_recovery_single_anchor')
                journal.append(record)
                try:
                    choice,_=parse_json(record['response'])
                    if record['finish_reason']=='length':
                        raise ValueError('Incomplete anchor choice')
                    single=dict(task,id='q0')
                    state=_apply_anchors(state,[single],choice,[],[])
                    state,_=normalize_shapes(state)
                    break
                except (ValueError,KeyError,TypeError) as error:
                    record['validation_error']=str(error)
        citations=state.get('top_influential_citations')
        if isinstance(citations,list) and len(citations)>5:
            result.setdefault('source_recovery_normalizations',[]).append(dict(
                operation='retain_first_five_citations_in_model_rank_order',original_count=len(citations),
                original_entries=copy.deepcopy(citations)))
            state['top_influential_citations']=citations[:5]
        # Citation errors are not part of the repository's narrative quote-task
        # inventory. Repair these and remaining shape errors directly, one field
        # at a time, always retaining the complete source and exact final validator.
        for _ in range(3):
            errors=invalid_fields(state,paper['fulltext'])
            if not errors:
                break
            for field,error in errors.items():
                user=FIELD_USER_TEMPLATE.format(source=paper['fulltext'],field=field,
                    value=json.dumps(state[field],ensure_ascii=False),error=error,
                    example=json.dumps(field_example(field),ensure_ascii=False))
                if field=='top_influential_citations':
                    user+='\nUse at most five entries. Citation keys must be literal contiguous bibliography text from this source. Remove an entry or a quote that cannot be verified; do not invent a replacement. If no grounded entries remain, use the empty string.'
                record=client.generate(REPAIR_SYSTEM,user,4096,seed+400+len(journal),phase='source_recovery_field_repair')
                journal.append(record)
                try:
                    patch,_=parse_json(record['response'])
                    if record['finish_reason']=='length' or not isinstance(patch,dict) or set(patch)!={field}:
                        raise ValueError('Invalid single-field repair')
                    state[field]=patch[field]
                    state,_=normalize_shapes(state)
                    citations=state.get('top_influential_citations')
                    if isinstance(citations,list) and len(citations)>5:
                        result.setdefault('source_recovery_normalizations',[]).append(dict(
                            operation='retain_first_five_citations_in_model_rank_order',original_count=len(citations),
                            original_entries=copy.deepcopy(citations)))
                        state['top_influential_citations']=citations[:5]
                except (ValueError,KeyError,TypeError) as error:
                    record['validation_error']=str(error)
        summary,context,spans=validate_summary(json.dumps(state,ensure_ascii=False),paper['fulltext'])
        result['corrected']=dict(summary=summary,judge_context=context,evidence_spans=spans,
            elapsed_seconds=time.monotonic()-started,changed=summary!=raw,source_recovery=True)
        result['corrected_quality_assessment']=review(paper,summary,client,journal,'source_recovery_corrected_review')
        result.pop('error',None)
    except Exception as error:
        result['error']=repr(error)
    result['complete']=True
    result['elapsed_seconds']+=time.monotonic()-started
    write(path,result)
    print('SOURCE_RECOVERY',model,paper['document_id'],bool(result.get('raw')),bool(result.get('corrected')),result.get('error',''),flush=True)
    return result

def rebuild(model,sources):
    directory=ROOT/'outputs/cohorts'/model
    docs=[load(directory/'documents'/p['document_id']/'document.json') for p in sources]
    for condition in ('raw','corrected'):
        path=directory/(condition+'-summaries.json')
        cache=load(path)
        shutil.copyfile(path,path.with_name(condition+'-summaries.before-recovery-'+str(time.time_ns())+'.json'))
        cache['documents']=[dict(document_id=d['document_id'],fulltext_sha256=d['fulltext_sha256'],**d[condition]) for d in docs if d.get(condition)]
        cache['failures']=[dict(document_id=d['document_id'],error=d.get('error')) for d in docs if not d.get(condition)]
        cache['config']['source_recovery_policy']='bounded_single_task_literal_anchor_recovery_v1'
        cache['config']['recovery_output_budget_policy']='Both models: generation up to 32768, correction up to 24576, bounded by full untruncated source context; only after unusable output'
        cache['config']['recovery_anti_repetition']='Both models: presence_penalty=1.5 on unusable-output recovery generation/correction; actual request preserved in traces'
        write(path,cache)
    completion=load(directory/'complete.json')
    completion.update(raw=sum(bool(d.get('raw')) for d in docs),corrected=sum(bool(d.get('corrected')) for d in docs),
        source_recovered_papers=sum(bool(d.get('source_recovery_history')) for d in docs))
    write(directory/'complete.json',completion)

def reconcile_checkpoint():
    # This is deliberately after generation/recovery; only the evaluator reads QA.
    path=ROOT/'outputs/qa-results.json'
    if not path.exists():
        return
    checkpoint=load(path)
    shutil.copyfile(path,path.with_name('qa-results.before-recovery-'+str(time.time_ns())+'.json'))
    caches={}
    for model,prefix in [('ornith_dflash','ornith'),('qwen38_fp8','qwen38')]:
        for condition in ('raw','corrected'):
            caches[prefix+'_'+condition]={d['document_id']:d for d in load(ROOT/'outputs/cohorts'/model/(condition+'-summaries.json'))['documents']}
    invalidated=[]
    for document in checkpoint['documents']:
        for condition,cache in caches.items():
            old=document['conditions'].get(condition)
            new=cache.get(document['document_id'])
            same_failure=old and old.get('generation_failed') and new is None
            same_context=old and new and old.get('context_sha256')==digest(new['judge_context'])
            if old and not same_failure and not same_context:
                del document['conditions'][condition]
                invalidated.append(dict(document_id=document['document_id'],condition=condition))
    write(path,checkpoint)
    write(ROOT/'outputs/qa-recovery-reconciliation.json',dict(invalidated=invalidated,
        reason='Context hashes changed after source-only recovery; all other fixed prompts retained'))

def main():
    sources=load(ROOT/'inputs/eval_sources_only.json')
    pending={}
    for model in ('ornith_dflash','qwen38_fp8'):
        pending[model]=[p for p in sources if not all(load(ROOT/'outputs/cohorts'/model/'documents'/p['document_id']/'document.json').get(c) for c in ('raw','corrected','corrected_quality_assessment'))]
    write(ROOT/'outputs/recovery_code_snapshot.json',dict(sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),pending={k:len(v) for k,v in pending.items()}))
    status('source_only_mechanical_recovery',pending={k:len(v) for k,v in pending.items()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        servers={model:executor.submit(start_server,i,'dflash8' if model=='ornith_dflash' else 'qwen38','summary-model')
            for i,model in enumerate(('ornith_dflash','qwen38_fp8')) if pending[model]}
        judge_future=executor.submit(start_server,2,'judge','qwen25')
        def supplemental_tuning():
            path=ROOT/'outputs/throughput/dflash4.json'
            if path.exists() and load(path).get('complete'):
                return load(path)
            from benchmark import run_benchmark
            server=start_server(3,'dflash4','ornith')
            try:
                return run_benchmark(server['endpoint'],'dflash4',3)
            finally:
                stop(server)
        tuning_future=executor.submit(supplemental_tuning)
        repairs=[]
        for model,future in servers.items():
            server=future.result()
            repairs.append((server,[executor.submit(recover,p,model,server['endpoint']) for p in pending[model]]))
        for server,futures in repairs:
            for future in futures:
                future.result()
            stop(server)
        judge=judge_future.result()
        try:
            tuning_result=tuning_future.result()
        except Exception as error:
            tuning_result=dict(mode='dflash4',complete=False,error=repr(error))
            write(ROOT/'outputs/supplemental_tuning_failure.json',tuning_result)
    for model in pending:
        rebuild(model,sources)
    reconcile_checkpoint()
    status('qa_reconciliation_after_recovery')
    from score import score_all
    report=score_all(judge['endpoint'])
    report['supplemental_throughput_benchmark']=tuning_result
    report['qa_speculative_configuration']='dflash8'
    write(ROOT/'outputs/report.json',report)
    lines=['','## Measured output throughput per GH200 GPU','',
        '512-token probes on non-holdout papers. Generation uses a cold source prefix; correction uses a cached full input. Complete-paper latencies are reported separately. QA used DFlash8.',
        '', '| Runtime | Batch | Generation tokens/s | Correction tokens/s |','|---|---:|---:|---:|']
    for mode in ('ar','dflash4','dflash8','dflash15'):
        path=ROOT/'outputs/throughput'/(mode+'.json')
        if not path.exists() or not load(path).get('complete'):
            continue
        rows=load(path)['batches']
        for batch in (1,32,64):
            gen=next(x for x in rows if x['phase']=='generation' and x['cache_state']=='cold' and x['concurrency']==batch)
            corr=next(x for x in rows if x['phase']=='correction' and x['cache_state']=='warm' and x['concurrency']==batch)
            lines.append('| '+mode+' | '+str(batch)+' | '+str(round(gen['output_tokens_per_second']))+' | '+str(round(corr['output_tokens_per_second']))+' |')
    with (ROOT/'outputs/report.md').open('a') as handle:
        handle.write('\n'.join(lines)+'\n')
    stop(judge)
    status('complete',papers=97,questions=970,report=str(ROOT/'outputs/report.md'),source_recovery=True)
    write(ROOT/'outputs/finalized.json',dict(papers=97,questions=970,completion={model:load(ROOT/'outputs/cohorts'/model/'complete.json') for model in pending}))

if __name__=='__main__':
    try:
        main()
    finally:
        for item in list(reversed(OWNED)):
            stop(item)
