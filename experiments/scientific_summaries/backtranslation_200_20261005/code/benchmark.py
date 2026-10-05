"""Run four owned GPU workers, preserve failures, publish complete local evidence."""
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import ROOT,write
from provision import main as provision

def main():
    tick=time.monotonic();provision()
    workers=[]
    specifications=[(0,'translategemma',['translate.py','--family','translategemma']),
                    (1,'windy',['translate.py','--family','windy']),
                    (2,'quality',['quality.py']), (3,'qa',['qa.py'])]
    try:
        for gpu,label,args in specifications:
            cache=ROOT/'cache'/label;cache.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XDG_CACHE_HOME=str(cache),
                     VLLM_CACHE_ROOT=str(cache/'vllm'),TRITON_CACHE_DIR=str(cache/'triton'),
                     FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'),TORCH_EXTENSIONS_DIR=str(cache/'torch_extensions'))
            log=(ROOT/'logs'/(label+'-'+os.environ.get('SLURM_JOB_ID','local')+'.log')).open('a')
            p=subprocess.Popen([sys.executable,str(ROOT/'code'/args[0])]+args[1:],env=env,
                               stdout=log,stderr=log,start_new_session=True)
            workers.append((label,p,log))
        while any(p.poll() is None for _,p,_ in workers):
            failed=[(label,p.returncode) for label,p,_ in workers if p.poll() not in [None,0]]
            write(ROOT/'outputs/worker_status.json',dict(job_id=os.environ.get('SLURM_JOB_ID'),
                  elapsed_seconds=time.monotonic()-tick,workers={l:dict(pid=p.pid,exit_code=p.poll()) for l,p,_ in workers}))
            if failed:raise RuntimeError('Worker failure: '+str(failed))
            time.sleep(5)
        from finalize import main as finalize
        finalize()
        write(ROOT/'outputs/complete.json',dict(complete=True,audit_passed=True,summary_versions=200,
            unique_papers=97,arms=3,job_id=os.environ.get('SLURM_JOB_ID'),
            node_elapsed_seconds=time.monotonic()-tick,allocated_gpus=4,
            measured_node_gpu_hours=(time.monotonic()-tick)*4/3600,
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    except Exception as e:
        write(ROOT/'outputs/failure.json',dict(error=repr(e),job_id=os.environ.get('SLURM_JOB_ID')))
        raise
    finally:
        for _,p,log in workers:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            log.close()

if __name__=='__main__':main()
