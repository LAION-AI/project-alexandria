import argparse
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import (ROOT, Client, PROMPT, REVIEW_SYSTEM, source_user, review_user,
    correction_user, parse_summary, validate_review, invalid_fields, validate_summary,
    narrative_context, repair_document, digest, load, write)
import json

def review(paper, summary, client, journal, phase):
    seed = int(digest(paper['document_id']+'|'+phase)[:8],16)%2147483000
    for attempt in range(2):
        record = client.generate(REVIEW_SYSTEM,review_user(paper,summary),4096,seed+attempt,phase=phase)
        journal.append(record)
        try:
            return validate_review(record,paper)
        except (ValueError,KeyError,TypeError) as error:
            record['validation_error']=str(error)
    return dict(review_failed=True, errors=[r.get('validation_error') for r in journal if r['phase']==phase])

def process(paper, endpoint, model):
    directory=ROOT/'outputs/cohorts'/model/'documents'/paper['document_id']
    final=directory/'document.json'
    if final.exists():
        saved=load(final)
        if saved['fulltext_sha256']!=paper['fulltext_sha256']:
            raise ValueError('Saved source changed')
        if saved.get('complete'):
            return saved
    client=Client(endpoint,'summary-model',directory/'calls')
    started=time.monotonic()
    result=dict(document_id=paper['document_id'],fulltext_sha256=paper['fulltext_sha256'],
        source=paper['source'], model=model, raw=None, corrected=None, attempts=[])
    journal=result['attempts']
    seed=int(digest(paper['document_id'])[:8],16)%2147483000
    try:
        for attempt in range(3):
            record=client.generate(PROMPT,source_user(paper),16000,seed+attempt,phase='generation')
            journal.append(record)
            try:
                state=parse_summary(record)
                break
            except (ValueError,KeyError,TypeError) as error:
                record['validation_error']=str(error)
        else:
            raise ValueError('No complete schema-shaped raw summary after three attempts')
        mechanical=invalid_fields(state,paper['fulltext'])
        raw=dict(summary=copy.deepcopy(state),judge_context=narrative_context(state),
            quote_validation_errors=mechanical, elapsed_seconds=time.monotonic()-started,
            raw_generation_trace=record.get('trace_file'))
        result['raw']=raw
        write(final,result)
        audit=review(paper,state,client,journal,'raw_review')
        result['raw_quality_assessment']=audit
        feedback=dict(semantic_audit=audit,mechanical_validation_errors=mechanical)
        correction_started=time.monotonic()
        for attempt in range(2):
            record=client.generate(PROMPT,correction_user(paper,state,feedback),16000,
                seed+500+attempt,phase='semantic_correction')
            journal.append(record)
            try:
                revised=parse_summary(record)
                break
            except (ValueError,KeyError,TypeError) as error:
                record['validation_error']=str(error)
        else:
            raise ValueError('Semantic correction failed twice')
        errors=invalid_fields(revised,paper['fulltext'])
        if errors:
            failure=dict(document_id=paper['document_id'],fulltext_sha256=paper['fulltext_sha256'],
                draft_summary=revised,failed=True,attempts=[copy.deepcopy(record)],validation_errors=errors,
                repair_protocol='field_local_strict_grounding_v3',elapsed_seconds=0)
            repaired=repair_document(paper,client,failure)
            journal.extend(repaired.get('attempts',[])[1:])
            if repaired.get('failed'):
                result['structural_repair_failure']=repaired
                raise ValueError('Repository V3 quote/shape repair incomplete')
            revised=repaired['summary']
            result['structural_repair_provenance']={k:v for k,v in repaired.items() if k not in ('attempts','summary','judge_context')}
        summary,context,spans=validate_summary(json.dumps(revised,ensure_ascii=False),paper['fulltext'])
        result['corrected']=dict(summary=summary,judge_context=context,evidence_spans=spans,
            elapsed_seconds=time.monotonic()-correction_started,
            changed=summary!=state, semantic_correction_trace=record.get('trace_file'))
        write(final,result)
        result['corrected_quality_assessment']=review(paper,summary,client,journal,'corrected_review')
    except Exception as error:
        result['error']=repr(error)
    result['complete']=True
    result['elapsed_seconds']=time.monotonic()-started
    write(final,result)
    return result

def run_cohort(endpoints,model,concurrency):
    sources=load(ROOT/'inputs/eval_sources_only.json')
    directory=ROOT/'outputs/cohorts'/model
    clients=len(endpoints)
    started=time.monotonic()
    # A fixed paper stays on its replica through generation, audit and correction.
    # This lets correction reuse the exact generation source prefix.
    assignments=[sources[i::clients] for i in range(clients)]
    saved=[]
    def replica(index):
        results=[]
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures={executor.submit(process,p,endpoints[index],model):p for p in assignments[index]}
            for future in as_completed(futures):
                result=future.result()
                results.append(result)
                print('PAPER',model,result['document_id'],bool(result.get('raw')),bool(result.get('corrected')),result.get('error',''),flush=True)
                write(directory/('replica-'+str(index)+'.json'),dict(completed=len(results),total=len(assignments[index]),
                    raw=sum(bool(x.get('raw')) for x in results),corrected=sum(bool(x.get('corrected')) for x in results)))
        return results
    with ThreadPoolExecutor(max_workers=clients) as executor:
        for results in executor.map(replica,range(clients)):
            saved.extend(results)
    ordered={d['document_id']:d for d in saved}
    for condition in ('raw','corrected'):
        documents=[]
        failures=[]
        for paper in sources:
            result=ordered[paper['document_id']]
            if result.get(condition):
                documents.append(dict(document_id=paper['document_id'],fulltext_sha256=paper['fulltext_sha256'],
                    **result[condition]))
            else:
                failures.append(dict(document_id=paper['document_id'],error=result.get('error','missing output')))
        write(directory/(condition+'-summaries.json'),dict(config=dict(model=model,condition=condition,
            temperature=.2,top_p=.95,thinking=False, max_tokens=16000,concurrency_per_replica=concurrency,
            replicas=clients,system_prompt_sha256=digest(PROMPT)),documents=documents,failures=failures,
            elapsed_seconds=time.monotonic()-started))
    write(directory/'complete.json',dict(model=model,papers=97,raw=sum(bool(x.get('raw')) for x in saved),
        corrected=sum(bool(x.get('corrected')) for x in saved),elapsed_seconds=time.monotonic()-started,
        documents=[x['document_id'] for x in sources]))
    return saved

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--endpoint',action='append',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--concurrency',type=int,default=8)
    args=parser.parse_args()
    run_cohort(args.endpoint,args.model,args.concurrency)
