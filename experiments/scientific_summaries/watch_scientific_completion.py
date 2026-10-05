"""Finish overlap audits and publish already authorized Ornith/FP8 evaluations."""
import base64
import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from postprocess_copy_overlap import update

BASE=Path('/e/fscratch/reformo/schuhmann1')
CHECKOUT=Path(__file__).resolve().parents[2]
BRANCH='findings/ornith-dflash-97-20261003'
RUNS={'ornith-distillation-736-20261004':'ornith_gemma_distillation_736',
      'scientific-gemma-r128-fp8-20261005':'gemma_r128_fp8_97'}
STATE=BASE/'scientific-copy-overlap-audit-20261005/completion_watch.json'


def load(path):return json.loads(path.read_text())


def env_for_git(token):
    env=dict(os.environ)
    for key in list(env):
        if key=='GIT_CURL_VERBOSE' or key.startswith('GIT_TRACE'):env.pop(key,None)
    env.update(GIT_TERMINAL_PROMPT='0',GIT_CONFIG_COUNT='2',
        GIT_CONFIG_KEY_0='http.https://github.com/.extraHeader',
        GIT_CONFIG_VALUE_0='Authorization: Basic '+base64.b64encode(('x-access-token:'+token).encode()).decode(),
        GIT_CONFIG_KEY_1='credential.helper',GIT_CONFIG_VALUE_1='')
    return env


def push_paths(paths,token):
    assert subprocess.check_output(['git','branch','--show-current'],cwd=CHECKOUT,text=True).strip()==BRANCH
    subprocess.run(['git','add','--',*paths],cwd=CHECKOUT,check=True,capture_output=True)
    if subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=CHECKOUT).returncode==1:
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com',
            'commit','--only','-m','Publish completed scientific summary evaluation with mandatory source-copy audits','--',*paths],
            cwd=CHECKOUT,check=True,capture_output=True)
    env=env_for_git(token)
    result=subprocess.run(['git','push','--quiet','origin','HEAD:refs/heads/'+BRANCH],cwd=CHECKOUT,env=env,capture_output=True)
    if result.returncode:raise RuntimeError('GitHub push failed; credential-bearing output suppressed')
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=CHECKOUT,env=env,text=True).split()[0]
    assert local==remote
    return local


def persist(name,folder):
    root=BASE/name;target=CHECKOUT/'experiments/scientific_summaries'/folder
    overlap=update(root)
    assert overlap is not None
    for p in (root/'outputs/evaluation').iterdir():
        if p.is_file() and p.suffix in ['.json','.jsonl','.md','.csv'] and p.name!='qa-live-results.json':
            (target/'evaluation').mkdir(parents=True,exist_ok=True);shutil.copy2(p,target/'evaluation'/p.name)
    for p in (root/'code').iterdir():
        if p.is_file() and p.suffix in ['.py','.sbatch']:
            (target/'code').mkdir(exist_ok=True);shutil.copy2(p,target/'code'/p.name)
    for filename in ['README.md','job.json']:
        if (root/filename).exists():shutil.copy2(root/filename,target/filename)
    data1=Path('/e/data1/datasets/playground/mmlaion/schuhmann1')/name/'outputs/evaluation'
    data1.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--copy-links','--exclude=*.tmp',str(root/'outputs/evaluation')+'/',str(data1)+'/'],check=True)
    return str(target.relative_to(CHECKOUT))


def main():
    fd=int(os.environ.pop('SCIENTIFIC_PUBLISH_CREDENTIAL_FD'))
    with os.fdopen(fd,'r') as handle:token=handle.read().strip()
    done={};deadline=time.monotonic()+48*3600
    while time.monotonic()<deadline:
        for name,folder in RUNS.items():
            if name in done:continue
            root=BASE/name
            if not (root/'outputs/evaluation/complete.json').exists():continue
            try:
                assert load(root/'outputs/evaluation/complete.json')['audit_passed']
                path=persist(name,folder);commit=push_paths([path],token)
                done[name]=dict(commit=commit,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            except Exception as e:print('Completion publication retry:',type(e).__name__,flush=True)
        STATE.write_text(json.dumps(dict(host=socket.gethostname(),pid=os.getpid(),completed=done,
            pending=[n for n in RUNS if n not in done],credentials_on_disk=False,
            updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
        if len(done)==len(RUNS):return
        time.sleep(30)


if __name__=='__main__':main()
