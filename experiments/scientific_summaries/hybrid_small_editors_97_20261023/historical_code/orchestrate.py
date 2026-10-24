"""Bounded node allocation, independent workers, incremental QA and checkpoints."""
import os,signal,subprocess,sys,time
from common import ROOT,CONDITIONS,cohort,condition_file,load,write

def main():
 assert os.environ.get('SLURM_JOB_ID');started=time.monotonic();owned=[];failures={};stopped=set()
 def interrupted(signum,frame):raise InterruptedError('Allocation signal '+str(signum))
 signal.signal(signal.SIGTERM,interrupted)
 if all((ROOT/f'outputs/{name}.json').exists() and load(ROOT/f'outputs/{name}.json').get('complete') for name in ['qa_results','semantic_results']):
  print('ALREADY_COMPLETE; no GPU workers started',flush=True);return
 assert (ROOT/'outputs/model_download_complete.json').exists(),'Download before allocating GPUs'
 subprocess.run([sys.executable,str(ROOT/'code/prepare.py')],check=True)
 roles=[('qwen08',0,'repair.py',['qwen08'],'hybrid_qwen08_r3_complete'),('gemma_e2b',1,'repair.py',['gemma_e2b'],'hybrid_gemma_e2b_r3_complete'),
  ('qa',2,'evaluate.py',['qa'],'qa_results'),('review',3,'evaluate.py',['review'],'semantic_results')]
 failed_file=ROOT/'outputs/failed_roles.json'
 if failed_file.exists():failed_file.unlink()
 try:
  for name,gpu,script,args,marker in roles:
   complete=ROOT/f'outputs/{marker}.json'
   if complete.exists() and load(complete).get('complete'):continue
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));cache_name={'qwen08':'qwen','qa':'qa','review':'review','gemma_e2b':'gemma_e2b'}[name]
   for variable,suffix in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('XDG_CACHE_HOME','xdg'),('FLASHINFER_WORKSPACE_BASE','flashinfer')]:env[variable]=str(ROOT/'cache'/cache_name/suffix)
   env['VLLM_MAX_NUM_BAD_WORDS']='64';env['VLLM_MAX_BAD_WORDS_TOTAL_TOKENS']='512';env['VLLM_USE_FLASHINFER_SAMPLER']='0'
   logpath=ROOT/f'logs/{name}.log'
   if logpath.exists():logpath.replace(logpath.with_name(f'{name}-before-{os.environ["SLURM_JOB_ID"]}.log'))
   stream=logpath.open('w');process=subprocess.Popen([sys.executable,'-u',str(ROOT/'code'/script),*args],env=env,stdout=stream,stderr=stream,start_new_session=True)
   owned.append((name,process,stream,complete))
  last_render=0;last_complete={}
  while any(p.poll() is None and name not in stopped for name,p,_,_ in owned):
   states={}
   for name,p,stream,marker in owned:
    if name in stopped:states[name]='complete; GPU process released';continue
    done=marker.exists() and load(marker).get('complete')
    if done:
     last_complete.setdefault(name,time.monotonic())
     if p.poll() is None and time.monotonic()-last_complete[name]>3:
      os.killpg(p.pid,signal.SIGTERM);stopped.add(name)
     states[name]='complete';continue
    code=p.poll();states[name]=code if code is not None else 'running'
    with (ROOT/f'logs/{name}.log').open('rb') as src:
     src.seek(0,2);src.seek(max(0,src.tell()-8192));tail=src.read()
    if code not in [None,0] or b'\nTraceback (most recent call last):\n' in tail:
     failures[name]=dict(exit_code=code,error='Uncaught worker failure; independent completed work retained')
     write(failed_file,failures)
     if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
     stopped.add(name)
   write(ROOT/'outputs/status.json',dict(phase='running',job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,roles=states,failures=failures))
   if time.monotonic()-last_render>25:
    subprocess.run([sys.executable,str(ROOT/'code/report.py')],check=False);last_render=time.monotonic()
   time.sleep(2)
  if failures:raise RuntimeError('Failed roles: '+str(list(failures)))
  from copy_overlap_eval import audit_conditions
  rows=cohort();contexts={c:{r['document_id']:load(condition_file(c,r['uid'])) for r in rows} for c in CONDITIONS}
  audit_conditions(rows,contexts,ROOT/'outputs/overlap')
  elapsed=time.monotonic()-started
  write(ROOT/'outputs/status.json',dict(phase='complete',job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=elapsed,allocated_gpu_hours_including_all_evaluation=4*elapsed/3600))
  subprocess.run([sys.executable,str(ROOT/'code/report.py')],check=True)
 except BaseException as error:
  write(ROOT/'outputs/status.json',dict(phase='failed',job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,error=repr(error),failures=failures))
  subprocess.run([sys.executable,str(ROOT/'code/report.py')],check=False);raise
 finally:
  for name,p,stream,_ in owned:
   if p.poll() is None:
    try:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=8)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    except ProcessLookupError:pass
   stream.close()
if __name__=='__main__':main()
