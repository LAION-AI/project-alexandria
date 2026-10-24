import os,signal,subprocess,sys,time,faulthandler
from common import ROOT,write,load

def main():
 assert os.environ.get('SLURM_JOB_ID');start=time.monotonic();owned=[]
 subprocess.run([sys.executable,str(ROOT/'code/finish_review.py'),'prepare'],check=True)
 from finish_review import ensure_overlap
 ensure_overlap()
 if load(ROOT/'outputs/review_resume_plan.json')['unique_pending']==0:
  subprocess.run([sys.executable,str(ROOT/'code/finish_review.py'),'aggregate'],check=True);return
 print('PRELOAD_SHARED_LIBRARIES_START',flush=True)
 faulthandler.dump_traceback_later(60,repeat=True)
 preload_start=time.monotonic()
 import torch
 print('PRELOAD_TORCH_DONE',round(time.monotonic()-preload_start,2),flush=True)
 from transformers import AutoTokenizer
 from vllm import LLM,SamplingParams
 print('PRELOAD_VLLM_DONE',round(time.monotonic()-preload_start,2),flush=True)
 faulthandler.cancel_dump_traceback_later()
 try:
  for lane in range(4):
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(lane))
   for variable,suffix in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('XDG_CACHE_HOME','xdg'),('FLASHINFER_WORKSPACE_BASE','flashinfer')]:env[variable]=str(ROOT/'cache/review'/suffix)
   marker=ROOT/f'outputs/review_shard_{lane}_complete.json'
   # A prior interruption may have left a marker for a different assignment.
   if marker.exists():marker.unlink()
   stream=(ROOT/f'logs/review-resume-{os.environ["SLURM_JOB_ID"]}-{lane}.log').open('w')
   p=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/finish_review.py'),str(lane)],env=env,stdout=stream,stderr=stream,start_new_session=True)
   owned.append((lane,p,stream,marker))
  finished=set();last_report=0
  while len(finished)<len(owned):
   for lane,p,stream,marker in owned:
    if lane in finished:continue
    if marker.exists() and load(marker).get('complete'):
     finished.add(lane)
     if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
    elif p.poll() is not None:raise RuntimeError('Review shard failed: '+str(lane))
   write(ROOT/'outputs/status.json',dict(phase='meaning_review_running',job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-start,
    finished_shards=len(finished),total_shards=4,editors_and_qa_already_complete=True))
   if time.monotonic()-last_report>25:subprocess.run([sys.executable,str(ROOT/'code/report.py')],check=False);last_report=time.monotonic()
   time.sleep(2)
  subprocess.run([sys.executable,str(ROOT/'code/finish_review.py'),'aggregate'],check=True)
 except BaseException as error:
  write(ROOT/'outputs/status.json',dict(phase='meaning_review_interrupted',job_id=os.environ['SLURM_JOB_ID'],error=repr(error),elapsed_seconds=time.monotonic()-start));raise
 finally:
  for _,p,stream,_ in owned:
   if p.poll() is None:
    try:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    except ProcessLookupError:pass
   stream.close()
if __name__=='__main__':main()
