import concurrent.futures
import os
import signal
import subprocess
import sys
import time
import urllib.request
from common import ROOT,load,write
from benchmark import run_benchmark,select
from run_cohort import run_cohort

SERVER_BASE=18490
OWNED=[]
START=time.monotonic()

def status(phase,**fields):
    write(ROOT/'outputs/status.json',dict(phase=phase,job_id=os.environ.get('SLURM_JOB_ID'),
        elapsed_seconds=time.monotonic()-START,updated_unix=time.time(),**fields))
    print('STATUS',phase,fields,flush=True)

def start_server(gpu,mode,alias='ornith'):
    port=SERVER_BASE+gpu
    directory=ROOT/'cache'/('gpu'+str(gpu))
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HOME=str(ROOT/'cache/hf'),HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',PYTHONUNBUFFERED='1',
        VLLM_SERVER_DEV_MODE='1',XDG_CACHE_HOME=str(directory),TRITON_CACHE_DIR=str(directory/'triton'),
        VLLM_CACHE_ROOT=str(directory/'vllm'),TORCH_EXTENSIONS_DIR=str(directory/'torch_extensions'),
        FLASHINFER_WORKSPACE_BASE=str(directory/'flashinfer'))
    if mode=='qwen38':
        model='/e/fscratch/reformo/schuhmann1/scientific-distillation-1000/models/Qwen3.8-27B-FP8'
    elif mode=='judge':
        model=str(ROOT/'models/Qwen2.5-7B-Instruct')
    else:
        model=str(ROOT/'models/Ornith-1.5-35B-A3B')
    argv=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',model,
        '--served-model-name',*([alias] if mode=='judge' else ['ornith','summary-model']),'--host','127.0.0.1','--port',str(port),'--tensor-parallel-size','1',
        '--max-model-len','32768' if mode=='judge' else '65536',
        '--max-num-seqs','4' if mode=='judge' else ('8' if mode=='qwen38' else '64'),
        '--max-num-batched-tokens','8192','--gpu-memory-utilization','0.90',
        '--generation-config','vllm','--enable-prefix-caching','--enable-chunked-prefill']
    if mode!='judge':
        argv+=['--reasoning-parser','qwen3','--language-model-only']
    if mode.startswith('dflash'):
        tokens=int(mode.removeprefix('dflash'))
        argv+=['--speculative-config',__import__('json').dumps(dict(method='dflash',model=str(ROOT/'models/Ornith-1.5-35B-A3B-DFlash'),num_speculative_tokens=tokens))]
    log=(ROOT/'logs'/('server-'+mode+'-gpu'+str(gpu)+'-'+str(time.time_ns())+'.log')).open('w')
    process=subprocess.Popen(argv,env=environment,stdout=log,stderr=log,start_new_session=True)
    item=dict(process=process,log=log,argv=argv,gpu=gpu,mode=mode,start=time.monotonic(),endpoint='http://127.0.0.1:'+str(port))
    OWNED.append(item)
    for _ in range(360):
        if process.poll() is not None:
            log.flush()
            raise RuntimeError('Owned server exited: '+mode+'; '+log.name)
        try:
            with urllib.request.urlopen(item['endpoint']+'/health',timeout=5) as response:
                if response.status==200:
                    return item
        except (OSError,ValueError):
            pass
        time.sleep(5)
    raise RuntimeError('Server did not become healthy within 30 minutes: '+mode)

def stop(item):
    process=item['process']
    if process.poll() is None:
        if os.getpgid(process.pid)!=process.pid:
            raise RuntimeError('Owned server process group changed')
        os.killpg(process.pid,signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait(timeout=10)
    item['log'].close()
    measurement=dict(mode=item['mode'],gpu=item['gpu'],seconds=time.monotonic()-item['start'],
        command=item['argv'],log_file=item['log'].name,exit_code=process.returncode)
    write(ROOT/'outputs/server_accounting'/(item['mode']+'-'+str(item['gpu'])+'-'+str(time.time_ns())+'.json'),measurement)
    OWNED.remove(item)

def main():
    from recovery import recover,rebuild
    if not (ROOT/'inputs/models_ready.json').exists():
        raise RuntimeError('Pinned model staging incomplete')
    manifest=load(ROOT/'inputs/experiment_manifest.json')
    write(ROOT/'outputs/code_snapshot.json',dict(files={str(p.relative_to(ROOT)):__import__('hashlib').sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'code').glob('*') if p.is_file()},job_id=os.getenv('SLURM_JOB_ID')))
    status('loading_benchmark_servers',target=manifest['target_model'],runtimes=manifest['runtimes'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(start_server,i,mode) for i,mode in enumerate(manifest['runtimes'])]
        servers=[future.result() for future in futures]
        status('throughput_benchmark',runtimes=manifest['runtimes'])
        results=list(executor.map(lambda s:run_benchmark(s['endpoint'],s['mode'],s['gpu']),servers))
        recipe=select(results)
        winner=recipe['winner']['mode']
        kept=next(s for s in servers if s['mode']==winner)
        for server in list(servers):
            if server is not kept:
                stop(server)
        status('loading_production_replicas',recipe=recipe,replicas=4)
        other_gpus=[i for i in range(4) if i!=kept['gpu']]
        production=[kept]+list(executor.map(lambda i:start_server(i,winner,'summary-model'),other_gpus))
        # start_server supplies both benchmark and cohort aliases; an alias may be duplicated,
        # which is supported by vLLM. All requests save the exact served name used.
        status('generating_eval_summaries',papers=97,replicas=4,concurrency_per_replica=recipe['production_concurrency'])
        sources=load(ROOT/'inputs/eval_sources_only.json')
        model='ornith35_dflash'
        first=run_cohort([s['endpoint'] for s in production],model,recipe['production_concurrency'])
        write(ROOT/'outputs/first_pass.json',dict(raw=sum(bool(d.get('raw')) for d in first),
            corrected=sum(bool(d.get('corrected')) for d in first),papers=97))
        for round_index in range(2):
            pending=[p for p in sources if not all(load(ROOT/'outputs/cohorts'/model/'documents'/p['document_id']/'document.json').get(k)
                for k in ('raw','corrected','corrected_quality_assessment'))]
            if not pending:
                break
            status('source_only_recovery',round=round_index+1,pending=len(pending))
            def recover_replica(index):
                papers=pending[index::len(production)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as repairs:
                    return list(repairs.map(lambda p:recover(p,model,production[index]['endpoint']),papers))
            list(executor.map(recover_replica,range(len(production))))
        rebuild(model,sources)
        completion=load(ROOT/'outputs/cohorts'/model/'complete.json')
        if completion['raw']!=97 or completion['corrected']!=97:
            raise RuntimeError('Incomplete source-only cohort after bounded recovery: '+str(completion))
        for server in production:
            stop(server)
    status('loading_fixed_qa_judge')
    judge=start_server(0,'judge','qwen25')
    status('qa_evaluation',new_conditions=2,papers=97,questions_per_condition=970)
    from score import score_all
    score_all(judge['endpoint'])
    stop(judge)
    from audit_final import main as audit
    audit()
    status('complete',papers=97,questions=970,report=str(ROOT/'outputs/report.md'))
    write(ROOT/'outputs/complete.json',dict(job_id=os.getenv('SLURM_JOB_ID'),elapsed_seconds=time.monotonic()-START,
        allocated_gpus=4,allocated_gpu_hours=4*(time.monotonic()-START)/3600,
        new_model_conditions=2,comparison_conditions=10))

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=repr(error))
        raise
    finally:
        for item in list(reversed(OWNED)):
            stop(item)
