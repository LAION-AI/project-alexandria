"""Run four owned GPU workers, preserve failures, publish complete local evidence."""
import datetime
import argparse
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import ROOT,write,load
from provision import main as provision
from critical_values import VERSION as CRITICAL_VERSION

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quality-only',action='store_true');args=parser.parse_args()
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    tick=time.monotonic();provision()
    workers=[]
    specifications=[(0,'translategemma',['translate.py','--family','translategemma']),
                    (1,'windy',['translate.py','--family','windy']),
                    (2,'quality',['quality.py']), (3,'qa',['qa.py'])]
    if all((ROOT/'outputs/translation'/arm/'complete.json').exists() for arm in ['windy_greedy','windy_beam4']):
        # Reuse immutable completed translations; give the free GPU to a second judge.
        os.environ['QA_WORKERS']='2'
        specifications=[(0,'translategemma',['translate.py','--family','translategemma']),
                        (1,'qa-beam4',['qa.py','--arms','windy_beam4']),
                        (2,'quality',['quality.py']),
                        (3,'qa',['qa.py','--arms','windy_greedy','translategemma'])]
    if args.quality_only:
        os.environ['QA_WORKERS']='3'
        specifications=[(0,'qa-translategemma',['qa.py','--arms','translategemma']),
                        (1,'qa-beam4',['qa.py','--arms','windy_beam4']),
                        (2,'quality',['quality.py']),
                        (3,'qa',['qa.py','--arms','windy_greedy'])]
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
        from finalize import main as finalize, REPO, DURABLE
        finalize()
        main_seconds=time.monotonic()-tick
        slurm_start=os.environ.get('SLURM_JOB_START_TIME')
        node_seconds=time.time()-int(slurm_start) if slurm_start else main_seconds
        failed=load(ROOT/'job.json').get('failed_benchmark_jobs',[])
        failed_gpu_hours=sum(j.get('allocated_gpu_hours',0) for j in failed)
        previous=load(ROOT/'job.json').get('completed_generation_job',{})
        accounting=dict(complete=True,audit_passed=True,critical_values_version=CRITICAL_VERSION,summary_versions=200,
            unique_papers=97,arms=3,job_id=os.environ.get('SLURM_JOB_ID'),
            main_process_elapsed_seconds=main_seconds,node_elapsed_seconds=node_seconds,allocated_gpus=4,
            measured_node_gpu_hours=node_seconds*4/3600,
            node_time_includes_module_startup_when_slurm_start_epoch_available=bool(slurm_start),
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            failed_initial_provision_job=2181277,failed_initial_provision_gpu_hours=74*4/3600,
            failed_benchmark_jobs=failed,failed_benchmark_gpu_hours=failed_gpu_hours,
            completed_generation_job=previous,
            total_measured_and_failed_gpu_hours=node_seconds*4/3600+74*4/3600+failed_gpu_hours+previous.get('allocated_gpu_hours',0))
        write(REPO/'allocation_accounting.json',accounting)
        write(DURABLE/'outputs/complete.json',accounting)
        file=DURABLE/'outputs/complete.json'
        manifest=load(ROOT/'outputs/evidence_manifest.json')
        manifest['files']=[v for v in manifest['files'] if v['path']!='outputs/complete.json']
        manifest['files'].append(dict(path='outputs/complete.json',bytes=file.stat().st_size,
            sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        for path in [ROOT/'outputs/evidence_manifest.json',DURABLE/'evidence_manifest.json',REPO/'evidence_manifest.json']:
            write(path,manifest)
        # Readiness marker is last: the publisher cannot race the evidence export.
        write(ROOT/'outputs/complete.json',accounting)
    except Exception as e:
        write(ROOT/'outputs/failure.json',dict(error=repr(e),job_id=os.environ.get('SLURM_JOB_ID')))
        raise
    finally:
        for _,p,log in workers:
            # The owned worker's process group can outlive an exited parent.
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            if p.poll() is None:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            log.close()

if __name__=='__main__':main()
