"""Credential stays in RAM; publish scoped completed results when the owned job finishes."""
import base64
import datetime
import getpass
import os
from pathlib import Path
import subprocess
import time
from common import ROOT,load,write
from critical_values import VERSION as CRITICAL_VERSION

REPO=Path('/e/home/jusers/schuhmann1/jupiter/project-alexandria')
SCOPE='experiments/scientific_summaries/backtranslation_200_20261005'

def main():
    token=getpass.getpass('GitHub token (hidden): ')
    while not (ROOT/'outputs/complete.json').exists() or load(ROOT/'outputs/complete.json').get('critical_values_version')!=CRITICAL_VERSION:
        write(ROOT/'publish_watch.json',dict(state='waiting_for_complete_results',
              updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              credential_storage='RAM only; not written to files',
              runtime_failure=load(ROOT/'outputs/failure.json') if (ROOT/'outputs/failure.json').exists() else None))
        time.sleep(30)
    assert load(ROOT/'outputs/complete.json')['audit_passed']
    subprocess.run(['git','add','--',SCOPE],cwd=REPO,check=True)
    dirty=subprocess.run(['git','diff','--cached','--name-only'],cwd=REPO,capture_output=True,text=True,check=True).stdout.splitlines()
    assert all(p.startswith(SCOPE+'/') for p in dirty), 'Refuse unrelated staged changes'
    if dirty:
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com',
            'commit','-m','Report 200-summary backtranslation throughput, numerical fidelity, source overlap and QA'],cwd=REPO,check=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_TRACE') and k!='GIT_CURL_VERBOSE'}
    env.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
               GIT_CONFIG_VALUE_0='Authorization: Basic '+base64.b64encode(('x-access-token:'+token).encode()).decode())
    token=None
    result=subprocess.run(['git','push','origin','HEAD'],cwd=REPO,env=env,capture_output=True,text=True)
    if result.returncode:
        write(ROOT/'publish_watch.json',dict(state='push_failed',exit_code=result.returncode));raise RuntimeError('Push failed')
    local=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()
    branch=subprocess.run(['git','branch','--show-current'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()
    remote=subprocess.run(['git','ls-remote','origin','refs/heads/'+branch],cwd=REPO,env=env,capture_output=True,text=True,check=True).stdout.split()[0]
    assert local==remote
    write(ROOT/'publish_watch.json',dict(state='published',commit=local,remote_verified=True,
          updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('Published and remote-verified',local,flush=True)

if __name__=='__main__':main()
