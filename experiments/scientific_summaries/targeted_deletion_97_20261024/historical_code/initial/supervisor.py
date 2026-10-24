"""One bounded allocation; CPU deletion, three reviewers, one QA engine."""
import os,signal,subprocess,sys,time
from common import ROOT,load,write

NLP='/e/fscratch/reformo/schuhmann1/scientific-procedural-synonyms-20261012/env/bin/python'


def main():
    owned=[];start=time.monotonic()
    def launch(name,script,args=(),gpu=None,nlp=False):
        env=dict(os.environ)
        if nlp:env.pop('PYTHONPATH',None)
        env['CUDA_VISIBLE_DEVICES']=str(gpu) if gpu is not None else ''
        for key,suffix in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('XDG_CACHE_HOME','xdg'),('FLASHINFER_WORKSPACE_BASE','flashinfer')]:
            env[key]=str(ROOT/'cache'/name/suffix)
        stream=(ROOT/f'logs/{name}-{os.environ["SLURM_JOB_ID"]}.log').open('w')
        p=subprocess.Popen([NLP if nlp else sys.executable,'-u',str(ROOT/'code'/script),*args],env=env,stdout=stream,stderr=stream,start_new_session=True)
        owned.append((name,p,stream));return p
    def monitor(path):
        while not path.exists():
            for name,p,_ in owned:
                if p.poll() not in [None,0]:raise RuntimeError(name+' failed '+str(p.returncode))
            time.sleep(1)
    try:
        launch('deletion','postprocess.py',nlp=True)
        print('PRELOAD_START',flush=True)
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM
        print('PRELOAD_COMPLETE',time.monotonic()-start,flush=True)
        for lane,gpu in enumerate([0,2,3]):launch('review'+str(lane),'review.py',[str(lane)],gpu=gpu)
        launch('qa','evaluate.py',['qa'],gpu=1)
        write(ROOT/'outputs/status.json',dict(phase='deletion_and_evaluation',job_id=os.environ['SLURM_JOB_ID']))
        monitor(ROOT/'outputs/cpu_complete.json')
        for lane in range(3):monitor(ROOT/f'outputs/post_review_lane_{lane}_complete.json')
        monitor(ROOT/'outputs/qa_results.json')
        for name,p,_ in owned:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        from finish import main as finish
        finish()
        write(ROOT/'outputs/status.json',dict(phase='complete',job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-start))
        print('ALL_DELETION_RESULTS_COMPLETE',flush=True)
    except BaseException as error:
        write(ROOT/'outputs/failure.json',dict(error=repr(error),elapsed_seconds=time.monotonic()-start));raise
    finally:
        for _,p,stream in owned:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                except ProcessLookupError:pass
            stream.close()


if __name__=='__main__':main()
