"""Complete packaging/publication after the submitted benchmark finishes."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/e/home/jusers/schuhmann1/jupiter/project-alexandria')
JOB='2166049'

def main():
    lock=(ROOT/'outputs/finalizer.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for _ in range(1440):
        result=subprocess.run(['sacct','-X','-j',JOB,'--format=State','--noheader','--parsable2'],text=True,capture_output=True)
        state=result.stdout.strip().strip('|')
        if state=='COMPLETED':
            subprocess.run(['python3',str(ROOT/'code/finalize_results.py')],check=True)
            environment=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_ASKPASS='/bin/false',SSH_ASKPASS='/bin/false')
            publish=subprocess.run(['git','push','-u','origin','findings/ornith-dflash-97-20261003'],cwd=REPO,env=environment)
            path=ROOT/'outputs/publication_status.json';status=json.loads(path.read_text())
            status.update(github_push='pushed' if publish.returncode==0 else 'authentication unavailable; complete local commit retained')
            path.write_text(json.dumps(status,indent=2))
            print('FINAL',json.dumps(status),flush=True)
            return
        if state in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'):
            raise RuntimeError('Benchmark did not complete: '+state+'; inspect owned Slurm logs')
        time.sleep(45)
    raise TimeoutError('Benchmark did not finish within eighteen hours')

if __name__=='__main__':main()
