"""Monitor/persist queued work; optionally push final evidence with a credential kept only in RAM."""
import base64
import datetime
import os
from pathlib import Path
import socket
import subprocess
import time
from common import ROOT,load,write
from publish import publish,REPO

CHECKOUT=REPO.parents[2]
BRANCH='findings/ornith-dflash-97-20261003'


def safe_git_env(token=None):
    env=dict(os.environ)
    for key in list(env):
        if key=='GIT_CURL_VERBOSE' or key.startswith('GIT_TRACE'):env.pop(key,None)
    env.update(GIT_TERMINAL_PROMPT='0',GIT_CONFIG_COUNT='2',
        GIT_CONFIG_KEY_0='http.https://github.com/.extraHeader',
        GIT_CONFIG_VALUE_0='Authorization: Basic '+base64.b64encode(('x-access-token:'+token).encode()).decode() if token else '',
        GIT_CONFIG_KEY_1='credential.helper',GIT_CONFIG_VALUE_1='')
    return env


def push(token):
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=CHECKOUT,text=True).strip()
    if branch!=BRANCH:raise RuntimeError('Checkout changed branch; final results are saved locally')
    relative=str(REPO.relative_to(CHECKOUT))
    subprocess.run(['git','add','--',relative],cwd=CHECKOUT,check=True,capture_output=True)
    staged=subprocess.run(['git','diff','--cached','--quiet','--',relative],cwd=CHECKOUT)
    if staged.returncode==1:
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com',
            'commit','--only','-m','Report rank-128 Gemma no-thinking quality, optimized throughput and GPU-hour scaling','--',relative],
            cwd=CHECKOUT,check=True,capture_output=True)
    env=safe_git_env(token)
    result=subprocess.run(['git','push','--quiet','origin','HEAD:refs/heads/'+BRANCH],cwd=CHECKOUT,env=env,capture_output=True)
    if result.returncode:raise RuntimeError('GitHub push failed (credential/output suppressed)')
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=CHECKOUT,env=env,text=True).split()[0]
    assert local==remote
    write(ROOT/'outputs/github_publish.json',dict(status='complete',commit=local,branch=BRANCH))


def track():
    job=load(ROOT/'job.json')['benchmark_job']
    queue=subprocess.check_output(['squeue','-h','-j',job,'-o','%T|%M|%R'],text=True).strip()
    data=subprocess.check_output(['sacct','-n','-X','-j',job,'--format=JobID,State,ElapsedRaw,AllocTRES','--parsable2'],text=True)
    records=[]
    for line in data.splitlines():
        fields=line.split('|')
        if len(fields)<4:continue
        jid,state,elapsed,tres=fields[:4];allocation=dict(x.split('=',1) for x in tres.split(',') if '=' in x)
        gpus=int(allocation.get('gres/gpu','0'))
        records.append(dict(job_id=jid,state=state,elapsed_seconds=int(elapsed),allocated_gpus=gpus,
            allocation_gpu_hours=int(elapsed)*gpus/3600))
    write(ROOT/'outputs/allocation_accounting.json',dict(jobs=records,full_node_idle_time_included=True))
    out=ROOT/'outputs/evaluation';completed=(out/'complete.json').exists()
    qa=load(out/'qa-results.json') if (out/'qa-results.json').exists() else {'documents':[]}
    generation={label:len(list((out/'generation'/label).glob('*/result.json'))) for label in
        ['no_thinking_matched','no_thinking_merged','no_thinking_fp8']}
    scored={label:sum(len(d['conditions'].get(label,{}).get('rows',[])) for d in qa['documents']) for label in generation}
    failed=any(r['state'].startswith(('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY')) for r in records)
    status=dict(state='complete' if completed else 'requires_attention' if failed else 'running' if queue.startswith('RUNNING') else 'queued',
        job_id=job,queue=queue,accounting=records,generation_papers=generation,scored_new_answers=scored,
        target_new_answers_per_arm=970,cached_thinking_reference_correct=897,cached_reference_total=970,
        updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),monitor_pid=os.getpid())
    write(ROOT/'STATUS.json',status)
    return status


def main():
    token=None
    fd=os.environ.pop('BENCHMARK_PUBLISH_CREDENTIAL_FD',None)
    if fd is not None:
        with os.fdopen(int(fd),'r') as handle:token=handle.read().strip()
    write(ROOT/'monitor_process.json',dict(host=socket.gethostname(),pid=os.getpid(),credentials_on_disk=False,
        final_push_enabled=bool(token),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    deadline=time.monotonic()+48*3600;last_publish=0
    while time.monotonic()<deadline:
        try:
            status=track()
            if time.monotonic()-last_publish>300 or status['state'] in ['complete','requires_attention']:
                publish();last_publish=time.monotonic()
            if status['state']=='complete':
                if token:
                    for _ in range(3):
                        try:push(token);break
                        except Exception as e:
                            print('Final publication retry:',type(e).__name__,flush=True);time.sleep(30)
                return
            if status['state']=='requires_attention':return
        except Exception as e:print('Monitor retry:',type(e).__name__,flush=True)
        time.sleep(30)


if __name__=='__main__':main()
