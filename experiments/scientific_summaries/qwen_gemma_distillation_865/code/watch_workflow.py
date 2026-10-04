"""Read-only job tracking and bounded HF upload submission after local login."""
import datetime
import html
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from huggingface_hub import get_token
from common import ROOT,load,write


def optional(path):
    return load(path) if path.exists() else None


def persist():
    destination=Path('/e/data1/datasets/playground/mmlaion/schuhmann1/scientific-distillation-865-20261004')
    destination.mkdir(parents=True,exist_ok=True)
    phases=[]
    if (ROOT/'outputs/release_ready.json').exists():phases.append(('release',ROOT/'release',destination/'release'))
    for rank in [64,128]:
        folder=ROOT/'outputs'/('train-r'+str(rank))
        if (folder/'result.json').exists():phases.append(('train-r'+str(rank),folder,destination/'outputs'/folder.name))
    for phase in ['ornith','evaluation']:
        folder=ROOT/'outputs'/phase
        if (folder/'complete.json').exists():phases.append((phase,folder,destination/'outputs'/phase))
    state=optional(ROOT/'outputs/persistence.json') or dict(destination=str(destination),completed_phases=[])
    for phase,source,target in phases:
        if phase in state['completed_phases']:continue
        target.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-a','--exclude=checkpoint-step*/','--exclude=optimizer.pt','--exclude=*.tmp',
                        '--exclude=.cache/',str(source)+'/',str(target)+'/'],check=True,stdout=subprocess.DEVNULL)
        state['completed_phases'].append(phase)
        write(ROOT/'outputs/persistence.json',state)
    for folder in ['code','inputs','training']:
        target=destination/folder;target.mkdir(exist_ok=True)
        subprocess.run(['rsync','-a','--exclude=__pycache__/','--exclude=*.tmp',str(ROOT/folder)+'/',str(target)+'/'],
                       check=True,stdout=subprocess.DEVNULL)
    for name in ['job.json','STATUS.json','MONITOR.html','README.md','monitor_process.json']:
        if (ROOT/name).exists():
            subprocess.run(['rsync','-a',str(ROOT/name),str(destination/name)],check=True,stdout=subprocess.DEVNULL)
    target=destination/'outputs';target.mkdir(exist_ok=True)
    for name in ['allocation_accounting.json','hf_upload.json','persistence.json','release_ready.json','core_ready.json']:
        if (ROOT/'outputs'/name).exists():
            subprocess.run(['rsync','-a',str(ROOT/'outputs'/name),str(target/name)],check=True,stdout=subprocess.DEVNULL)


def track():
    metadata=load(ROOT/'job.json')
    generation_job=metadata.get('evaluation_generation_job')
    labels=['gemma12_base','gemma12_r64','gemma12_r128']
    if generation_job and not metadata.get('old_eval_stopped_after_generation'):
        complete=all((ROOT/'outputs/evaluation'/(label+'-generation-performance.json')).exists() for label in labels)
        qa_complete=(ROOT/'outputs/evaluation/complete.json').exists()
        if complete and (qa_complete or not metadata.get('qa_on_existing_allocation')):
            subprocess.run(['scancel',generation_job],check=True)
            metadata['old_eval_stopped_after_generation']=True
            metadata['old_eval_stop_reason']='All three 97-paper generator cohorts are saved. Dedicated concurrent QA continues; avoid duplicate QA in the former sequential process.'
            if metadata.get('qa_on_existing_allocation'):
                subprocess.run(['scancel',metadata['evaluation_job']],check=True)
                metadata['old_eval_stop_reason']='All 2910 QA answers audited and final report saved by concurrent QA on the existing allocation; stop former sequential orchestration and unused rescue job.'
            write(ROOT/'job.json',metadata)
    keys=['train_rank64_job','train_rank128_job','ornith_job','evaluation_job']
    ids=[metadata[k] for k in keys]
    queue=subprocess.run(['squeue','-h','-j',','.join(ids),'-o','%i|%T|%M|%R'],text=True,capture_output=True,check=True)
    live={row.split('|')[0]:row.split('|')[1:] for row in queue.stdout.splitlines()}
    jobs={key:dict(id=metadata[key],live=live.get(metadata[key])) for key in keys}
    accounting_ids=ids+['2171755','2171756','2171770','2158056','2167404']
    if generation_job:accounting_ids.append(generation_job)
    if metadata.get('failed_ornith_resume_job'):accounting_ids.append(metadata['failed_ornith_resume_job'])
    accounting_ids+=metadata.get('failed_ornith_resume_jobs',[])
    accounting_ids=list(dict.fromkeys(accounting_ids))
    if metadata.get('hf_upload_job'):accounting_ids.append(metadata['hf_upload_job'])
    accounting=subprocess.check_output(['sacct','-n','-X','-j',','.join(accounting_ids),
                    '--format=JobID,JobName,State,ElapsedRaw,AllocTRES','--parsable2'],text=True)
    accounts=[]
    for line in accounting.splitlines():
        fields=line.split('|')
        if len(fields)<5:continue
        job,name,state,elapsed,tres=fields[:5]
        allocations=dict(item.split('=',1) for item in tres.split(',') if '=' in item)
        gpu=int(allocations.get('gres/gpu','0'))
        accounts.append(dict(job_id=job,name=name,state=state,elapsed_seconds=int(elapsed),allocated_gpus=gpu,
                             allocation_gpu_hours=int(elapsed)*gpu/3600))
        for value in jobs.values():
            if value['id']==job:value['accounting_state']=state
    write(ROOT/'outputs/allocation_accounting.json',dict(jobs=accounts,includes_cancelled_and_failed_attempts=True))
    progress={}
    for rank in [64,128]:
        folder=ROOT/'outputs'/('train-r'+str(rank))
        timings=folder/'timings.jsonl'
        last=None
        if timings.exists():
            lines=timings.read_text().splitlines()
            if lines:
                try:last=json.loads(lines[-1])
                except json.JSONDecodeError:pass
        progress[str(rank)]=dict(latest_step=last,result=optional(folder/'result.json'))
    ornith_results=list((ROOT/'outputs/ornith/documents').glob('*/result.json'))
    ornith_generated=sum(load(path)['status']=='generated' for path in ornith_results)
    ornith=dict(attempted_papers=len(ornith_results),target_papers=865,
                generated_papers=ornith_generated,failed_papers=len(ornith_results)-ornith_generated,
                missing_papers=865-len(ornith_results),
                complete=optional(ROOT/'outputs/ornith/complete.json'))
    ready=optional(ROOT/'outputs/release_ready.json')
    hf=optional(ROOT/'outputs/hf_upload.json') or dict(status='awaiting_hf_login')
    if ready and get_token() and hf.get('status')=='awaiting_hf_login' and not metadata.get('hf_upload_job'):
        job=subprocess.check_output(['sbatch','--parsable',str(ROOT/'code/upload_hf.sbatch')],text=True).strip()
        metadata['hf_upload_job']=job
        write(ROOT/'job.json',metadata)
        hf=dict(status='queued',job_id=job)
        write(ROOT/'outputs/hf_upload.json',hf)
    if metadata.get('hf_upload_execution')=='login_process' and hf.get('status') not in ['complete','failed']:
        if not Path('/proc/'+str(metadata['hf_upload_pid'])).exists():
            hf.update(status='failed',error='Owned login upload process ended before completion')
            write(ROOT/'outputs/hf_upload.json',hf)
    elif metadata.get('hf_upload_job') and hf.get('status') not in ['complete','failed']:
        active=subprocess.check_output(['squeue','-h','-j',metadata['hf_upload_job'],'-o','%T'],text=True).strip()
        if not active:
            accounting=subprocess.check_output(['sacct','-n','-X','-j',metadata['hf_upload_job'],'--format=State','--parsable2'],text=True).strip()
            if accounting.startswith(('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY')):
                hf.update(status='failed',slurm_state=accounting)
                write(ROOT/'outputs/hf_upload.json',hf)
    evaluation=optional(ROOT/'outputs/evaluation/complete.json')
    qa_checkpoint=optional(ROOT/'outputs/evaluation/qa-live-results.json')
    qa_progress=dict(matched_papers=0,scored_answers=0,target_answers=2910,models={})
    if qa_checkpoint:
        qa_progress['matched_papers']=sum(len(d['conditions'])==3 for d in qa_checkpoint['documents'])
        for document in qa_checkpoint['documents']:
            for label,condition in document['conditions'].items():
                score=qa_progress['models'].setdefault(label,dict(correct=0,total=0))
                for row in condition['rows']:
                    score['total']+=1
                    score['correct']+=next(iter(row['predictions'].values()))==row['gold']
        qa_progress['scored_answers']=sum(v['total'] for v in qa_progress['models'].values())
    gpu_done=all(v['result'] and v['result']['status']=='complete' for v in progress.values()) and bool(ornith['complete']) and bool(evaluation)
    failure=any(v.get('accounting_state','').startswith(('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY'))
                and not (key=='evaluation_job' and evaluation) for key,v in jobs.items())
    state='complete' if gpu_done and hf.get('status')=='complete' else 'gpu_work_complete_awaiting_hf' if gpu_done else 'requires_attention' if failure else 'running'
    snapshot=dict(state=state,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  teacher_stopped=True,training_papers=865,external_test_papers=97,external_test_mcqs=970,
                  generator_only=True,epochs=1,peak_learning_rate=2e-5,jobs=jobs,
                  training=progress,ornith=ornith,release=ready,huggingface=hf,evaluation=evaluation,
                  qa_progress=qa_progress,
                  monitor_host=socket.gethostname(),monitor_pid=os.getpid())
    write(ROOT/'STATUS.json',snapshot)
    rows=[]
    for rank,v in progress.items():
        step=v['latest_step'] or {}
        rows.append(f"<tr><td>Gemma 12B rank {rank}</td><td>{step.get('step',0)}/109</td><td>{step.get('loss','—')}</td><td>{round(step.get('remaining_compute_seconds',0)/60,1)} min remaining compute</td></tr>")
    ornith_state=jobs['ornith_job'].get('live')
    ornith_state=ornith_state[0] if ornith_state else jobs['ornith_job'].get('accounting_state','unknown')
    rows.append(f"<tr><td>Ornith 9B native AR</td><td>{ornith_generated}/865 generated; {ornith['failed_papers']} failed; {ornith['missing_papers']} pending results</td><td>no DFlash</td><td>{'complete' if ornith['complete'] else ornith_state}</td></tr>")
    rows.append(f"<tr><td>97-paper evaluation</td><td>{qa_progress['scored_answers']}/2910 answers; {qa_progress['matched_papers']}/97 matched papers</td><td>matched baseline/r64/r128</td><td>{'complete' if evaluation else 'concurrent QA / remaining generation'}</td></tr>")
    rows.append(f"<tr><td>HF dataset</td><td>865 papers</td><td>{html.escape(hf['status'])}</td><td>{html.escape(hf.get('url','Login required'))}</td></tr>")
    (ROOT/'MONITOR.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>865-paper generator distillation</title><style>body{font:16px system-ui;margin:30px;color:#16212b}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #ccd4da;text-align:left}code{font-size:12px}</style><h1>865-paper Qwen → Gemma generator distillation</h1><p>Updated '+html.escape(snapshot['updated_at'])+'</p><p>Teacher stopped. One epoch, rank 64 and 128, generator only. Peak LR 2e-5 after initial rank-128 divergence at 1e-4.</p><table><tr><th>Task</th><th>Progress</th><th>Details</th><th>Status</th></tr>'+''.join(rows)+'</table><p>Training: genuine source-only generator reasoning and matching original answer; corrected final summaries and correction traces are published separately. Frozen test: 97 papers/970 MCQs, zero training overlap.</p><pre>'+html.escape(json.dumps(jobs,indent=2))+'</pre></html>')
    return state


def main():
    deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        try:
            state=track()
            persist()
            from publish_local_results import publish
            publish()
            if state=='complete':return
        except Exception as e:
            # Avoid echoing exception messages that can contain service credentials.
            print('Monitor retry:',type(e).__name__,flush=True)
        time.sleep(30)


if __name__=='__main__':main()
