"""Source-only format rescue after observed schema-note/repetition failures.

Produces separate candidates while the original driver is running. Never reads QA.
Uses original model proposals and finite, schema-constrained anchor selections.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import time
from concurrent.futures import ThreadPoolExecutor
from common import (ROOT,Client,PROMPT,REVIEW_SYSTEM,source_user,correction_user,parse_summary,
    load,write,digest,invalid_fields,validate_summary,narrative_context,parse_json,KEYS)
from summarize import GROUNDED
from summary_runtime import ANCHOR_SYSTEM,quote_tasks
from summary_repair_v3 import normalize_shapes,structural_fields,_apply_anchors

def sequence_schema():
    return {'anyOf':[{'const':''},{'type':'array','minItems':1,'maxItems':12,'items':{
        'type':'object','minProperties':1,'maxProperties':1,'additionalProperties':{'type':'string'}}}]}

def summary_schema():
    properties={k:{'type':'string'} for k in KEYS[:5]}
    properties.update({k:sequence_schema() for k in GROUNDED})
    claim={'type':'object','properties':dict(description={'type':'string'},
        **{k:sequence_schema() for k in ['supporting_evidence','contradicting_evidence','implications']}),
        'required':['description','supporting_evidence','contradicting_evidence','implications'],'additionalProperties':False}
    properties['claims']={'anyOf':[{'const':''},{'type':'array','minItems':1,'maxItems':5,'items':claim}]}
    # Optional bibliographic ranking is omitted after the existing bounded repairs failed.
    properties['top_influential_citations']={'const':''}
    return {'type':'object','properties':properties,'required':list(KEYS),'additionalProperties':False}

class StructuredClient(Client):
    def generate_schema(self,system,user,budget,seed,phase,schema):
        self.next_schema=schema
        try:return self.generate(system,user,budget,seed,phase=phase)
        finally:self.next_schema=None
    def post(self,route,payload,timeout=1800):
        if route=='/v1/chat/completions' and getattr(self,'next_schema',None):
            payload['response_format']={'type':'json_schema','json_schema':dict(name='source_only_rescue',strict=True,schema=self.next_schema)}
            payload['presence_penalty']=1.5
        return super().post(route,payload,timeout)

def remove_format_notes(obj):
    obj=copy.deepcopy(obj);removed=[]
    claims=obj.get('claims')
    if isinstance(claims,list):
        keep=[]
        for entry in claims:
            if isinstance(entry,str) and re.match(r'^(data_and_code_availability|ethical_considerations|robustness_and_ablation_notes|key_figures_tables|top_influential_citations)_(notes|placeholder)',entry):
                removed.append(dict(operation='remove_non_scientific_schema_note_from_claims',original_value=entry))
            else:keep.append(entry)
        obj['claims']=keep
    obj,changes=normalize_shapes(obj)
    return obj,removed+changes

def complete_proposal(folder,phases):
    for path in reversed(sorted((folder/'calls').glob('*.json'))):
        record=load(path)['record']
        if record['phase'] not in phases or record['finish_reason']=='length':continue
        try:
            obj,_=parse_json(record['response'])
            if not isinstance(obj,dict) or set(obj)!=set(KEYS):continue
            obj={k:obj[k] for k in KEYS};obj,notes=remove_format_notes(obj)
            if structural_fields(obj):continue
            return obj,dict(trace_file=str(path.relative_to(ROOT)),normalizations=notes)
        except (ValueError,KeyError,TypeError):continue
    return None,None

def rescue(paper,model,endpoint):
    started=time.monotonic()
    original=ROOT/'outputs/cohorts'/model/'documents'/paper['document_id']
    saved=load(original/'document.json')
    directory=ROOT/'outputs/schema_rescue'/model/paper['document_id']
    output=directory/'candidate.json'
    if output.exists() and load(output).get('passed'):return load(output)
    client=StructuredClient(endpoint,'summary-model',directory/'calls')
    journal=[];seed=int(digest(paper['document_id']+'|schema-rescue')[:8],16)%2147483000
    result=dict(document_id=paper['document_id'],model=model,fulltext_sha256=paper['fulltext_sha256'],
        passed=False,attempts=journal,policy='source_only_finite_schema_rescue_v1')
    try:
        raw=saved.get('raw')
        if not raw:
            proposal,provenance=complete_proposal(original,{'generation','source_recovery_generation'})
            if proposal is None:raise ValueError('No complete source-only raw proposal available')
            raw=dict(summary=proposal,judge_context=narrative_context(proposal),
                quote_validation_errors=invalid_fields(proposal,paper['fulltext']),format_rescue=provenance,
                elapsed_seconds=0,raw_generation_trace=provenance['trace_file'])
        result['raw']=raw
        audit=saved.get('raw_quality_assessment') or dict(review_failed=True,errors=['Unavailable before schema rescue'])
        result['raw_quality_assessment']=audit
        partial=(saved.get('structural_repair_failure') or {}).get('draft_summary')
        if isinstance(partial,dict) and set(partial)==set(KEYS):
            partial,notes=remove_format_notes({k:partial[k] for k in KEYS})
        if isinstance(partial,dict) and set(partial)==set(KEYS) and not structural_fields(partial):
            proposal,provenance=partial,dict(existing_v3_partial_draft=True,normalizations=notes)
        else:
            proposal,provenance=complete_proposal(original,{'semantic_correction','source_recovery_semantic_correction'})
        if proposal is None:
            for attempt in range(2):
                user=correction_user(paper,raw['summary'],audit)+'\nFORMAT_RESCUE\nUse the supplied JSON schema exactly. Do not include instructions or format commentary in claims. Unreported fields are empty strings. Preserve supported scientific detail without filler. Optional citation ranking may be empty after the earlier failed repairs.'
                record=client.generate_schema(PROMPT,user,8192,seed+attempt,'schema_rescue_semantic_correction',summary_schema())
                journal.append(record)
                try:proposal=parse_summary(record);break
                except (ValueError,KeyError,TypeError) as error:record['validation_error']=str(error)
            else:raise ValueError('Schema-constrained semantic correction failed')
            provenance={'trace_file':record['trace_file']}
        result['semantic_correction_provenance']=provenance
        state=copy.deepcopy(proposal)
        # Anchor AFTER any field rewrite. Finite legal indices prevent malformed selections.
        for number,task in enumerate(reversed(quote_tasks(state,paper['fulltext']))):
            public={k:v for k,v in task.items() if k!='path'};public['id']='q0'
            public['candidates']=[dict(index=i,**candidate) for i,candidate in enumerate(task['candidates'])]
            schema={'type':'object','properties':{'q0':{'enum':list(range(len(public['candidates'])))+['drop']}},'required':['q0'],'additionalProperties':False}
            user='BEGIN_PAPER\n'+paper['fulltext']+'\nEND_PAPER\n\nTASKS\n'+json.dumps([public],ensure_ascii=False)
            user+='\nChoose only a supplied supporting source fragment; otherwise choose drop. Do not invent an index.'
            record=client.generate_schema(ANCHOR_SYSTEM,user,512,seed+100+number,'schema_rescue_anchor',schema)
            journal.append(record);choices,_=parse_json(record['response'])
            if record['finish_reason']=='length':raise ValueError('Incomplete finite anchor output')
            task=dict(task,id='q0')
            anchors=[];removals=[];state=_apply_anchors(state,[task],choices,anchors,removals)
            result.setdefault('anchor_decisions',[]).append(dict(anchors=anchors,removals=removals))
            state,_=normalize_shapes(state)
        errors=invalid_fields(state,paper['fulltext'])
        if errors and set(errors)=={'top_influential_citations'}:
            result['discarded_optional_citations']=copy.deepcopy(state['top_influential_citations'])
            state['top_influential_citations']=''
        state,context,spans=validate_summary(json.dumps(state,ensure_ascii=False),paper['fulltext'])
        result['corrected']=dict(summary=state,judge_context=context,evidence_spans=spans,
            changed=state!=raw['summary'],source_recovery=True,format_rescue=True,
            elapsed_seconds=time.monotonic()-started)
        from run_cohort import review
        result['corrected_quality_assessment']=review(paper,state,client,journal,'schema_rescue_corrected_review')
        result['passed']=True
    except Exception as error:result['error']=repr(error)
    result['elapsed_seconds']=time.monotonic()-started
    write(output,result)
    print('SCHEMA_RESCUE',model,paper['document_id'],result['passed'],result.get('error',''),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--endpoint',required=True)
    parser.add_argument('--document-id')
    args=parser.parse_args()
    sources=load(ROOT/'inputs/eval_sources_only.json')
    pending=[p for p in sources if p['document_id']==args.document_id] if args.document_id else [p for p in sources
        if not load(ROOT/'outputs/cohorts'/args.model/'documents'/p['document_id']/'document.json').get('corrected')]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(lambda p:rescue(p,args.model,args.endpoint),pending))
    if not all(r['passed'] for r in results):raise SystemExit(1)
