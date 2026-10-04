"""Upload the frozen release from a login host and persist independently queued GPU work."""
import datetime
import html
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from common import ROOT,load,write
from huggingface_hub import get_token
from upload_hf import main as upload

DEST=Path('/e/data1/datasets/playground/mmlaion/schuhmann1/ornith-distillation-736-20261004')
GIT=Path('/e/home/jusers/schuhmann1/jupiter/project-alexandria/experiments/scientific_summaries/ornith_gemma_distillation_736')


def optional(path):return load(path) if path.exists() else None


def track():
    meta=load(ROOT/'job.json')
    keys=['prepare_job','train_rank64_job','train_rank128_job','evaluation_job']
    ids=[meta[k] for k in keys]
    queue=subprocess.check_output(['squeue','-h','-j',','.join(ids),'-o','%i|%T|%M|%R'],text=True)
    live={line.split('|')[0]:line.split('|')[1:] for line in queue.splitlines()}
    jobs={key:dict(id=meta[key],live=live.get(meta[key])) for key in keys}
    accounting_ids=ids+meta['original_generation_jobs']
    accounting=subprocess.check_output(['sacct','-n','-X','-j',','.join(accounting_ids),
        '--format=JobID,JobName,State,ElapsedRaw,AllocTRES','--parsable2'],text=True)
    records=[]
    for line in accounting.splitlines():
        fields=line.split('|')
        if len(fields)<5:continue
        job,name,state,elapsed,tres=fields[:5]
        allocation=dict(x.split('=',1) for x in tres.split(',') if '=' in x)
        gpus=int(allocation.get('gres/gpu','0'))
        records.append(dict(job_id=job,name=name,state=state,elapsed_seconds=int(elapsed),allocated_gpus=gpus,
            allocation_gpu_hours=int(elapsed)*gpus/3600,phase='original_generation' if job in meta['original_generation_jobs'] else 'new_distillation'))
        for value in jobs.values():
            if value['id']==job:value['accounting_state']=state
    write(ROOT/'outputs/allocation_accounting.json',dict(jobs=records,includes_failed_attempts=True,
        updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    training={}
    for rank in [64,128]:
        folder=ROOT/'outputs'/('train-r'+str(rank));timings=folder/'timings.jsonl';last=None
        if timings.exists():
            lines=timings.read_text().splitlines()
            if lines:last=__import__('json').loads(lines[-1])
        training[str(rank)]=dict(latest_step=last,result=optional(folder/'result.json'))
    hf=optional(ROOT/'outputs/hf_upload.json') or dict(status='awaiting_preparation')
    if (ROOT/'outputs/release_ready.json').exists() and hf['status']!='complete':
        if not get_token():hf=dict(status='awaiting_hf_login')
        else:
            # One initial upload and two bounded retries; never overwrite a different release.
            tries=int(meta.get('hf_upload_attempts',0))
            if tries<3:
                meta['hf_upload_attempts']=tries+1;write(ROOT/'job.json',meta)
                try:upload()
                except Exception as e:
                    hf=optional(ROOT/'outputs/hf_upload.json') or {}
                    hf.update(status='failed',error_type=type(e).__name__,attempt=tries+1)
                    write(ROOT/'outputs/hf_upload.json',hf)
                else:hf=load(ROOT/'outputs/hf_upload.json')
    evaluation=optional(ROOT/'outputs/evaluation/complete.json')
    checkpoint=optional(ROOT/'outputs/evaluation/qa-live-results.json')
    qa=dict(matched_papers=0,scored_answers=0,target_answers=2910,reused_baseline_answers=970)
    if checkpoint:
        qa['matched_papers']=sum(len(d['conditions'])==3 for d in checkpoint['documents'])
        qa['scored_answers']=sum(len(c['rows']) for d in checkpoint['documents'] for c in d['conditions'].values())
    done=all(v['result'] and v['result']['status']=='complete' for v in training.values()) and bool(evaluation) and hf['status']=='complete'
    failure=any(v.get('accounting_state','').startswith(('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY')) for v in jobs.values())
    status=dict(state='complete' if done else 'requires_attention' if failure else 'running',
        updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),frozen_papers=736,teacher='ornith-ai/Ornith-1.5-9B',
        generator_only=True,epochs=1,peak_learning_rate=2e-5,jobs=jobs,training=training,huggingface=hf,
        evaluation=evaluation,qa_progress=qa,monitor_host=socket.gethostname(),monitor_pid=os.getpid())
    write(ROOT/'STATUS.json',status)
    rows=''.join('<tr><td>'+html.escape(k)+'</td><td>'+html.escape(v['id'])+'</td><td>'+html.escape(v['live'][0] if v['live'] else v.get('accounting_state','unknown'))+'</td></tr>' for k,v in jobs.items())
    (ROOT/'MONITOR.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Ornith 736 → Gemma generator LoRAs</title><style>body{font:16px system-ui;margin:30px}td,th{padding:10px;border:1px solid #aaa}table{border-collapse:collapse}</style><h1>Ornith 736 → Gemma 4 12B IT</h1><p>736 frozen raw summaries and actual reasoning; generator-only ranks 64/128, one epoch. Original generation stopped by user.</p><p>HF: '+html.escape(hf['status'])+'</p><p>QA: '+str(qa['scored_answers'])+'/2910 answer records; baseline 970 reused verbatim.</p><table><tr><th>Task</th><th>Job</th><th>State</th></tr>'+rows+'</table><p>Updated '+html.escape(status['updated_at'])+'</p></html>')
    return status


def persist():
    DEST.mkdir(parents=True,exist_ok=True);GIT.mkdir(parents=True,exist_ok=True)
    for name in ['code','inputs']:
        subprocess.run(['rsync','-a','--exclude=__pycache__/','--exclude=*.tmp',str(ROOT/name)+'/',str(DEST/name)+'/'],check=True,stdout=subprocess.DEVNULL)
    for name in ['README.md','job.json','STATUS.json','MONITOR.html','monitor_process.json']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,DEST/name)
    for name in ['hf_upload.json','allocation_accounting.json']:
        path=ROOT/'outputs'/name
        if path.exists():
            (DEST/'outputs').mkdir(exist_ok=True);shutil.copy2(path,DEST/'outputs'/name);shutil.copy2(path,GIT/name)
    for rank in [64,128]:
        folder=ROOT/'outputs'/('train-r'+str(rank))
        if (folder/'result.json').exists():
            target=DEST/'outputs'/folder.name
            if not (target/'result.json').exists():
                subprocess.run(['rsync','-a','--exclude=checkpoint-step*/','--exclude=optimizer.pt',str(folder)+'/',str(target)+'/'],check=True,stdout=subprocess.DEVNULL)
            result=load(folder/'result.json');result.pop('trainable_module_names',None)
            write(GIT/('training-r'+str(rank)+'.json'),result);shutil.copy2(folder/'timings.jsonl',GIT/('training-r'+str(rank)+'-timings.jsonl'))
    folder=ROOT/'outputs/evaluation'
    if (folder/'complete.json').exists():
        target=DEST/'outputs/evaluation'
        if not (target/'complete.json').exists():
            subprocess.run(['rsync','-a','--exclude=*.tmp',str(folder)+'/',str(target)+'/'],check=True,stdout=subprocess.DEVNULL)
        target=GIT/'evaluation';target.mkdir(exist_ok=True)
        for path in folder.iterdir():
            if path.is_file() and path.name!='qa-live-results.json':shutil.copy2(path,target/path.name)


def main():
    deadline=time.monotonic()+48*3600
    while time.monotonic()<deadline:
        try:
            status=track();persist()
            if status['state']=='complete':return
        except Exception as e:print('Monitor retry:',type(e).__name__,flush=True)
        time.sleep(30)


if __name__=='__main__':main()
