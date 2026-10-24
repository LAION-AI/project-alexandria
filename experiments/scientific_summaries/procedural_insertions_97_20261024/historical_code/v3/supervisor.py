import os,signal,subprocess,sys,time
from common import ROOT,load,write
NLP='/e/fscratch/reformo/schuhmann1/scientific-procedural-synonyms-20261012/env/bin/python'


def main():
    owned=[];start=time.monotonic()
    def launch(name,script,gpu=None,nlp=False):
        env=dict(os.environ)
        if nlp:env.pop('PYTHONPATH',None)
        env['CUDA_VISIBLE_DEVICES']='' if gpu is None else str(gpu)
        cache_name='review0' if name=='review' else name
        for k,s in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('XDG_CACHE_HOME','xdg'),('FLASHINFER_WORKSPACE_BASE','flashinfer')]:env[k]=str(ROOT/'cache'/cache_name/s)
        log=(ROOT/f'logs/{name}-{os.environ["SLURM_JOB_ID"]}.log').open('w')
        args=[NLP if nlp else sys.executable,'-u',str(ROOT/'code'/script)]
        if name=='qa':args+=['qa']
        p=subprocess.Popen(args,env=env,stdout=log,stderr=log,start_new_session=True);owned.append((name,p,log))
    def wait(path):
        while not path.exists():
            for name,p,_ in owned:
                if p.poll() not in [None,0]:raise RuntimeError(name+' failed '+str(p.returncode))
            time.sleep(1)
    try:
        launch('deletion','postprocess.py',nlp=True)
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM
        launch('review','review.py',gpu=0);launch('qa','evaluate.py',gpu=1)
        wait(ROOT/'outputs/cpu_complete.json');wait(ROOT/'outputs/review_complete.json');wait(ROOT/'outputs/qa_results.json')
        for _,p,_ in owned:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        from finish import main as finish
        finish()
        write(ROOT/'outputs/status.json',dict(phase='complete',job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-start))
    except BaseException as e:
        write(ROOT/'outputs/failure.json',dict(error=repr(e)));raise
    finally:
        for _,p,stream in owned:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                except ProcessLookupError:pass
            stream.close()


if __name__=='__main__':main()
