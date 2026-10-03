"""Merge verified source-only candidates after the original Slurm driver stops."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from common import ROOT,load,write,validate_summary,narrative_context

def main():
    active=subprocess.check_output(['squeue','-j','2166049','-h'],text=True).strip()
    if active:raise RuntimeError('Original generation driver still active; no concurrent document mutation allowed')
    sources=load(ROOT/'inputs/eval_sources_only.json')
    counts={}
    for model in ['gemma4_e4b','gemma4_12b']:
        replaced=[]
        for paper in sources:
            folder=ROOT/'outputs/cohorts'/model/'documents'/paper['document_id']
            path=folder/'document.json';doc=load(path)
            rescue_dir=ROOT/'outputs/schema_rescue'/model/paper['document_id']
            if not (doc.get('raw') and doc.get('corrected')):
                candidate=load(rescue_dir/'candidate.json')
                if not candidate['passed'] or candidate['fulltext_sha256']!=paper['fulltext_sha256']:
                    raise ValueError('Unverified format-recovery candidate')
                shutil.copy2(path,folder/('before-schema-rescue-'+str(time.time_ns())+'.json'))
                if not doc.get('raw'):
                    doc['raw']=candidate['raw']
                    doc['raw_quality_assessment']=candidate['raw_quality_assessment']
                if not doc.get('corrected'):
                    doc['corrected']=candidate['corrected']
                    doc['corrected_quality_assessment']=candidate['corrected_quality_assessment']
                doc.setdefault('source_recovery_history',[]).append(dict(policy=candidate['policy'],
                    candidate_file=str((rescue_dir/'candidate.json').relative_to(ROOT)),prior_error=doc.get('error')))
                doc['schema_rescue_provenance']={k:v for k,v in candidate.items() if k not in ['raw','corrected','attempts']}
                doc['attempts'].extend(candidate['attempts'])
                replaced.append(paper['document_id'])
            if rescue_dir.exists():
                shutil.copytree(rescue_dir,folder/'schema_rescue',dirs_exist_ok=True)
            # Preserve every completed call, including ones held in memory when the old driver stopped.
            seen={c.get('trace_file') for c in doc['attempts'] if c.get('trace_file')}
            for call in sorted((folder/'calls').glob('*.json')):
                trace=str(call.relative_to(ROOT))
                if trace not in seen:
                    record=load(call)['record'];record['trace_file']=trace
                    record['journal_reconciled_after_controller_intervention']=True
                    doc['attempts'].append(record);seen.add(trace)
            if narrative_context(doc['raw']['summary'])!=doc['raw']['judge_context']:
                raise ValueError('Raw narrative differs')
            _,context,ledger=validate_summary(json.dumps(doc['corrected']['summary'],ensure_ascii=False),paper['fulltext'])
            expected=[(x['path'],x['quote'],x['start'],x['end']) for x in ledger]
            actual=[(x['path'],x['quote'],x['start'],x['end']) for x in doc['corrected']['evidence_spans']]
            if context!=doc['corrected']['judge_context'] or expected!=actual:
                raise ValueError('Corrected narrative or literal source ledger differs')
            doc.pop('error',None);doc['complete']=True
            write(path,doc)
        directory=ROOT/'outputs/cohorts'/model
        docs=[load(directory/'documents'/p['document_id']/'document.json') for p in sources]
        replaced=[d['document_id'] for d in docs if d.get('schema_rescue_provenance')]
        for phase in ['raw','corrected']:
            cache=load(directory/(phase+'-summaries.json'))
            shutil.copy2(directory/(phase+'-summaries.json'),directory/(phase+'-summaries.before-schema-rescue.json'))
            cache['documents']=[dict(document_id=d['document_id'],fulltext_sha256=d['fulltext_sha256'],**d[phase]) for d in docs]
            cache['failures']=[]
            cache['config']['schema_rescue_policy']='source_only_finite_schema_rescue_v1'
            cache['config']['schema_rescue_note']='Conditional finite JSON schemas and anchoring after field restoration, after common-protocol failures; no QA-guided changes.'
            write(directory/(phase+'-summaries.json'),cache)
        completion=load(directory/'complete.json')
        completion.update(raw=97,corrected=97,source_recovered_papers=sum(bool(d.get('source_recovery_history')) for d in docs),
            schema_rescued_papers=len(replaced),schema_rescued_documents=replaced)
        write(directory/'complete.json',completion);counts[model]=completion
    write(ROOT/'outputs/schema_reconciliation.json',dict(passed=True,models=counts,qa_used=False,
        source_code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'rescue_code').glob('*.py')}))
    write(ROOT/'outputs/performance_recipe.json',dict(models={m:load(ROOT/'outputs'/('performance_recipe-'+m+'.json'))
        for m in ['gemma4_e4b','gemma4_12b']},precision='BF16',thinking=False))
    print('Reconciled both 97-paper raw/corrected cohorts; literal evidence and source hashes verified.',flush=True)

if __name__=='__main__':main()
