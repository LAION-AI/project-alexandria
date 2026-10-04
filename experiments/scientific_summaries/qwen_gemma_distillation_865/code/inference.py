"""Owned native vLLM servers and fully recorded source-only generator requests."""
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from common import ROOT, SOURCE, write

sys.path.insert(0, str(SOURCE / 'code'))
from run import json_object
from reference_validator import KEYS, narrative_context, validate_summary


def post(endpoint, route, payload, timeout=1800):
    request=urllib.request.Request(endpoint+route, json.dumps(payload,ensure_ascii=False).encode(),
                                   {'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=timeout) as response:
        return json.load(response)


class Server:
    def __init__(self, model, port, name, parser=None, gpu=None, adapter=None, judge=False):
        folder=ROOT/'cache'/name
        folder.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                 OMP_NUM_THREADS='8',PYTHONUNBUFFERED='1',XDG_CACHE_HOME=str(folder),
                 VLLM_CACHE_ROOT=str(folder/'vllm'),TRITON_CACHE_DIR=str(folder/'triton'),
                 TORCH_EXTENSIONS_DIR=str(folder/'torch_extensions'))
        env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
        env['MAX_JOBS']='8'
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        self.command=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',str(model),
            '--served-model-name','qa-model' if judge else 'summary-model','--host','127.0.0.1','--port',str(port),
            '--dtype','bfloat16','--tensor-parallel-size','1','--max-model-len','32768' if judge else '65536',
            '--max-num-seqs','4' if judge else '64','--max-num-batched-tokens','8192',
            '--gpu-memory-utilization','0.90','--generation-config','vllm','--enable-prefix-caching','--enable-chunked-prefill']
        if not judge:self.command+=['--language-model-only','--reasoning-parser',parser]
        if adapter:self.command+=['--enable-lora','--max-lora-rank','128','--lora-modules','trained='+str(adapter)]
        assert '--speculative-config' not in self.command
        self.endpoint='http://127.0.0.1:'+str(port)
        self.log=(ROOT/'logs'/(name+'-server-'+os.environ.get('SLURM_JOB_ID','local')+'.log')).open('w')
        write(ROOT/'outputs/server_commands'/(name+'.json'),dict(command=self.command,speculative_decoding=False,
              job_id=os.environ.get('SLURM_JOB_ID'),gpu=gpu,adapter=str(adapter) if adapter else None))
        self.process=subprocess.Popen(self.command,env=env,stdout=self.log,stderr=self.log,start_new_session=True)
        try:
            for _ in range(360):
                if self.process.poll() is not None:raise RuntimeError('Server exited; see '+self.log.name)
                try:
                    with urllib.request.urlopen(self.endpoint+'/health',timeout=5) as response:
                        if response.status==200:return
                except OSError:pass
                time.sleep(5)
            raise RuntimeError('Server startup timeout')
        except Exception:
            self.close()
            raise

    def close(self):
        if self.process.poll() is None:
            assert os.getpgid(self.process.pid)==self.process.pid
            os.killpg(self.process.pid,signal.SIGTERM)
            try:self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGKILL);self.process.wait(timeout=10)
        self.log.close()


def generate_paper(endpoint,paper,system,user,folder,model='summary-model',thinking=True):
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/'result.json'
    if target.exists():return json.loads(target.read_text())
    start=time.monotonic()
    base_user=user
    error=''
    for attempt in range(3):
        current_user=base_user+('\n\nFormat error from the preceding attempt: '+error if error else '')
        messages=[dict(role='system',content=system),dict(role='user',content=current_user)]
        kwargs=dict(enable_thinking=thinking,preserve_thinking=True,reasoning_effort='medium')
        count=post(endpoint,'/tokenize',dict(model=model,messages=messages,add_generation_prompt=True,
                                           chat_template_kwargs=kwargs))['count']
        if count+24576>65536:raise ValueError('Full untruncated source plus output budget exceeds context')
        seed=int.from_bytes(hashlib.sha256((paper['document_id']+'-'+str(attempt)).encode()).digest()[:4],'big')%2147483647
        payload=dict(model=model,messages=messages,temperature=1.0,top_p=.95,top_k=20,min_p=0.0,
                     presence_penalty=0.0,repetition_penalty=1.0,max_tokens=24576,seed=seed,
                     chat_template_kwargs=kwargs,response_format={'type':'json_object'})
        tick=time.monotonic()
        response=post(endpoint,'/v1/chat/completions',payload)
        record=dict(request=payload,response=response,seconds=time.monotonic()-tick,input_tokens_preflight=count)
        write(folder/('attempt'+str(attempt)+'.json'),record)
        choice=response['choices'][0];message=choice['message']
        try:
            if choice['finish_reason']!='stop':raise ValueError('Generation did not finish: '+str(choice['finish_reason']))
            summary=json_object(message['content'])
            if not isinstance(summary,dict) or set(summary)!=set(KEYS):raise ValueError('Need all 19 summary fields')
            summary={k:summary[k] for k in KEYS}
            context=narrative_context(summary)
            if not context.strip():raise ValueError('Empty summary narrative')
            narrative_tokens=post(endpoint,'/tokenize',dict(model=model,prompt=context,add_special_tokens=False))['count']
            source_error=None
            try:validate_summary(json.dumps(summary,ensure_ascii=False),paper['fulltext'])
            except ValueError as e:source_error=str(e)
            result=dict(document_id=paper['document_id'],domain=paper.get('domain'),source_sha256=paper.get('source_sha256'),
                        status='generated',summary=summary,judge_context=context,
                        reasoning_content=message.get('reasoning_content',message.get('reasoning')),
                        strict_source_schema_validation_passed=source_error is None,
                        source_validation_error=source_error,attempts=attempt+1,
                        native_narrative_tokens=narrative_tokens,last_usage=response.get('usage',{}),
                        elapsed_seconds=time.monotonic()-start,semantic_correction=False)
            write(target,result)
            return result
        except (ValueError,KeyError,TypeError) as e:error=str(e)
    result=dict(document_id=paper['document_id'],domain=paper.get('domain'),status='generation_failed',
                error=error,judge_context='',elapsed_seconds=time.monotonic()-start,attempts=3)
    write(target,result)
    return result
