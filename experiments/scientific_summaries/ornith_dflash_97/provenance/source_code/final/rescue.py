"""Source-only recovery of incomplete test outputs; QA answers never used for recovery."""
import copy
import concurrent.futures
import hashlib
import json
import os
import shutil
import time
from common import *
from run_cohort import review
from orchestrate import start_server,stop,OWNED,status,START
from score import score_all
from summary_runtime import ANCHOR_SYSTEM

SHORT_QUOTE = {'type':'string','pattern':r'^\s*\S+(?:\s+\S+){0,4}\s*$'}
GROUNDED_SCHEMA = {'anyOf':[{'type':'string','enum':['']},{'type':'array','minItems':1,
    'items':{'type':'object','minProperties':1,'maxProperties':1,'additionalProperties':SHORT_QUOTE}}]}
CLAIM_SCHEMA = {'type':'object','properties':dict(description={'type':'string'},
    supporting_evidence=GROUNDED_SCHEMA,contradicting_evidence=GROUNDED_SCHEMA,implications=GROUNDED_SCHEMA),
    'required':['description','supporting_evidence','contradicting_evidence','implications'],'additionalProperties':False}
SUMMARY_PROPERTIES = {key:({'type':'string'} if key in KEYS[:5] else GROUNDED_SCHEMA) for key in KEYS}
SUMMARY_PROPERTIES['claims']={'anyOf':[{'type':'string','enum':['']},{'type':'array','minItems':1,'items':CLAIM_SCHEMA}]}
SUMMARY_PROPERTIES['top_influential_citations']={'anyOf':[{'type':'string','enum':['']},{'type':'array','minItems':1,'maxItems':5,
    'items':{'type':'object','properties':{'quotes':{'type':'array','minItems':1,'items':SHORT_QUOTE}},'required':['quotes'],
    'minProperties':2,'maxProperties':2,'additionalProperties':{'type':'string'}}}]}
SUMMARY_SCHEMA={'type':'object','properties':SUMMARY_PROPERTIES,'required':list(KEYS),'additionalProperties':False}

class RecoveryClient(Client):
    def post(self,route,payload,timeout=1800):
        if route=='/v1/chat/completions':
            system=payload['messages'][0]['content']
            schema=None
            if system==PROMPT:
                schema=SUMMARY_SCHEMA
            elif system==ANCHOR_SYSTEM:
                tasks=json.loads(payload['messages'][1]['content'].rsplit('\nTASKS\n',1)[1])
                properties={task['id']:{'enum':list(range(len(task['candidates'])))+['drop']} for task in tasks}
                schema={'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
            if schema:
                payload['response_format']={'type':'json_schema','json_schema':{'name':'source_only_recovery',
                    'strict':True,'schema':schema}}
        return super().post(route,payload,timeout)

def recover(paper,endpoint,model):
    directory=ROOT/'outputs/cohorts'/model/'documents'/paper['document_id']
    path=directory/'document.json'
    doc=load(path)
    if doc.get('raw') and doc.get('corrected'):
        return doc
    backup=directory/'before-source-only-recovery.json'
    if not backup.exists():
        shutil.copyfile(path,backup)
    client=RecoveryClient(endpoint,'summary-model',directory/'recovery_calls')
    journal=doc['attempts']
    seed=int(digest(paper['document_id']+'|source-only-recovery')[:8],16)%2147483000
    started=time.monotonic()
    doc['failure_history']=[doc.get('error')]
    provenance=[]
    try:
        if not doc.get('raw'):
            for attempt in range(3):
                record=client.generate(PROMPT,source_user(paper),24576,seed+attempt,phase='generation_recovery')
                journal.append(record)
                try:
                    raw=parse_summary(record)
                    break
                except (ValueError,KeyError,TypeError) as error:
                    record['validation_error']=str(error)
            else:
                raise ValueError('Source-only generation recovery exhausted')
            doc['raw']=dict(summary=raw,judge_context=narrative_context(raw),
                quote_validation_errors=invalid_fields(raw,paper['fulltext']),
                raw_generation_trace=record.get('trace_file'))
        raw=doc['raw']['summary']
        failure=doc.get('structural_repair_failure')
        if failure and failure.get('draft_summary'):
            draft=copy.deepcopy(failure['draft_summary'])
        else:
            audit=review(paper,raw,client,journal,'raw_recovery_review')
            doc['raw_quality_assessment']=audit
            record=client.generate(PROMPT,correction_user(paper,raw,audit),16000,seed+100,phase='semantic_correction_recovery')
            journal.append(record)
            draft=parse_summary(record)
        # Citation metadata is excluded from the student context. Remove only entries
        # that fail the source's literal citation/proof requirements, recording each.
        citations=draft.get('top_influential_citations')
        if isinstance(citations,list):
            from evaluate import evidence_span
            valid=[]
            for entry in citations:
                try:
                    if not isinstance(entry,dict) or len(entry)!=2 or 'quotes' not in entry:
                        raise ValueError('Malformed citation ranking entry')
                    key=next(k for k in entry if k!='quotes')
                    evidence_span(paper['fulltext'],key)
                    if not isinstance(entry['quotes'],list) or not entry['quotes']:
                        raise ValueError('Missing citation influence proof')
                    for quote in entry['quotes']:
                        _,_,exact=evidence_span(paper['fulltext'],quote)
                        if len(exact.split())>5:
                            raise ValueError('Citation proof longer than five words')
                    valid.append(entry)
                except (ValueError,KeyError,TypeError,StopIteration) as error:
                    provenance.append(dict(operation='removed_unverifiable_citation_metadata',entry=entry,
                        reason=str(error),student_context_includes_citation_rankings=False))
            for entry in valid[5:]:
                provenance.append(dict(operation='removed_lower_ranked_citation_beyond_schema_limit',
                    entry=entry,reason='Schema-v4 specifies at most five ranked citations',
                    student_context_includes_citation_rankings=False))
            draft['top_influential_citations']=valid[:5] or ''
        errors=invalid_fields(draft,paper['fulltext'])
        if errors:
            current=dict(document_id=paper['document_id'],fulltext_sha256=paper['fulltext_sha256'],
                draft_summary=draft,failed=True,attempts=copy.deepcopy(journal[-1:]),validation_errors=errors,
                repair_protocol='field_local_strict_grounding_v3',elapsed_seconds=0)
            repaired=repair_document(paper,client,current)
            journal.extend(repaired.get('attempts',[])[1:])
            if repaired.get('failed'):
                doc['structural_repair_failure']=repaired
                raise ValueError('Source-only recovery still fails strict validation')
            draft=repaired['summary']
        summary,context,spans=validate_summary(json.dumps(draft,ensure_ascii=False),paper['fulltext'])
        doc['corrected']=dict(summary=summary,judge_context=context,evidence_spans=spans,
            changed=summary!=raw,source_only_recovery=True,recovery_seconds=time.monotonic()-started)
        doc['corrected_quality_assessment']=review(paper,summary,client,journal,'corrected_recovery_review')
        doc.pop('error',None)
    except Exception as error:
        doc['error']=repr(error)
    doc['source_only_recovery_provenance']=provenance
    doc['source_only_recovery_seconds']=time.monotonic()-started
    doc['elapsed_seconds']+=doc['source_only_recovery_seconds']
    write(path,doc)
    print('RECOVERY',model,paper['document_id'],bool(doc.get('raw')),bool(doc.get('corrected')),doc.get('error',''),flush=True)
    return doc

def refresh(model):
    sources=load(ROOT/'inputs/eval_sources_only.json')
    directory=ROOT/'outputs/cohorts'/model
    documents={p['document_id']:load(directory/'documents'/p['document_id']/'document.json') for p in sources}
    for condition in ('raw','corrected'):
        cache=load(directory/(condition+'-summaries.json'))
        cache['documents']=[dict(document_id=p['document_id'],fulltext_sha256=p['fulltext_sha256'],
            **documents[p['document_id']][condition]) for p in sources if documents[p['document_id']].get(condition)]
        cache['failures']=[dict(document_id=p['document_id'],error=documents[p['document_id']].get('error'))
            for p in sources if not documents[p['document_id']].get(condition)]
        cache['config']['source_only_recovery_policy']='Schema-v4 constrained generation (24,576 output tokens for incomplete raw drafts), valid anchor-choice enums, V3 retry, remove literally unverifiable citation rankings (excluded from QA context)'
        cache['source_only_recovery_papers']=[p['document_id'] for p in sources if documents[p['document_id']].get('source_only_recovery_seconds') is not None]
        write(directory/(condition+'-summaries.json'),cache)
    completion=load(directory/'complete.json')
    completion['raw']=sum(bool(d.get('raw')) for d in documents.values())
    completion['corrected']=sum(bool(d.get('corrected')) for d in documents.values())
    completion['source_only_recovered_papers']=[p['document_id'] for p in sources if documents[p['document_id']].get('source_only_recovery_seconds') is not None]
    write(directory/'complete.json',completion)
    return completion

def main():
    write(ROOT/'outputs/recovery-code-snapshot.json',dict(files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'code').glob('*') if p.is_file()},job_id=os.getenv('SLURM_JOB_ID')))
    sources=load(ROOT/'inputs/eval_sources_only.json')
    pending={}
    for model in ('ornith_dflash','qwen38_fp8'):
        pending[model]=[p for p in sources if not load(ROOT/'outputs/cohorts'/model/'documents'/p['document_id']/'document.json').get('corrected')
            or not load(ROOT/'outputs/cohorts'/model/'documents'/p['document_id']/'document.json').get('raw')]
    status('source_only_recovery',pending={m:len(ps) for m,ps in pending.items()})
    if not any(pending.values()):
        status('complete',papers=97,questions=970,recovery_needed=False)
        return
    if (ROOT/'outputs/qa-results.json').exists():
        shutil.copyfile(ROOT/'outputs/qa-results.json',ROOT/'outputs/qa-results.before-source-only-recovery.json')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        servers={}
        futures={model:executor.submit(start_server,i,'dflash8' if model=='ornith_dflash' else 'qwen38','summary-model')
            for i,(model,items) in enumerate(pending.items()) if items}
        judge_future=executor.submit(start_server,2,'judge','qwen25')
        for model,future in futures.items():
            servers[model]=future.result()
        for model,server in servers.items():
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
                list(workers.map(lambda p:recover(p,server['endpoint'],model),pending[model]))
            stop(server)
            print('COHORT',refresh(model),flush=True)
        judge=judge_future.result()
        status('qa_reconcile_after_source_only_recovery')
        score_all(judge['endpoint'])
        stop(judge)
    status('complete',papers=97,questions=970,recovery_needed=True)
    write(ROOT/'outputs/recovery-complete.json',dict(job_id=os.getenv('SLURM_JOB_ID'),
        elapsed_seconds=time.monotonic()-START,allocated_gpus=4,allocated_gpu_hours=4*(time.monotonic()-START)/3600,
        completion={m:load(ROOT/'outputs/cohorts'/m/'complete.json') for m in pending}))

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('recovery_failed',error=repr(error))
        raise
    finally:
        for item in list(reversed(OWNED)):
            stop(item)
