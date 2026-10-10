# SPDX-License-Identifier: CC-BY-4.0
# Copyright 2026 LAION / Project Alexandria contributors.
import hashlib,json,os,sys,threading,time
from pathlib import Path
ROOT=Path(os.environ.get('ALEXANDRIA_KU_DISTILL_RUN',str(Path(__file__).resolve().parents[1])))
OLD=ROOT.parent/'scientific-qwen27-ku1000-97-20261008'
sys.path[:0]=[str(ROOT/'vendor/src'),str(ROOT/'vendor/audit'),str(ROOT/'vendor')]
INFERENCE_PY=sys.executable
TRAIN_PY=sys.executable
MODELS={
 'gemma4':dict(id='google/gemma-4-E4B-it',revision='ee0ef6023621cff504d758262d4e04895a5af4a2',path=str(ROOT/'models'/'gemma-4-E4B-it')),
 'gemma12':dict(id='google/gemma-4-12B-it',revision='707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7',path=str(ROOT/'models'/'gemma-4-12b-it')),
 'qwen27':dict(id='Qwen/Qwen3.8-27B-FP8',revision='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',path=str(ROOT/'models'/'Qwen3.8-27B-FP8')),
 'judge':dict(id='Qwen/Qwen2.5-7B-Instruct',revision='a09a35458c702b33eeacc393d103063234e8bc28',path=str(ROOT/'models'/'Qwen2.5-7B-Instruct'))}
LEGACY_ADAPTER=str(ROOT/'models/legacy-summary-adapter')
TARGET_MODULES=r'.*language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$'
def load(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+f'.tmp-{os.getpid()}-{threading.get_ident()}');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sources(cohort='new20'):
 papers=load(ROOT/'inputs/new20_sources.json') if cohort=='new20' else load(OLD/'inputs/sources_only.json')
 assert len(papers)==(20 if cohort=='new20' else 97)
 assert all(digest(p['fulltext'])==p['fulltext_sha256'] for p in papers)
 return papers

def post(endpoint,route,payload):
 import urllib.request
 req=urllib.request.Request(endpoint+route,data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=1800) as response:return json.load(response)

class Backend:
 def __init__(self,endpoint,alias,phase,temperature=.1,json_output=True):
  self.endpoint,self.model_name,self.phase,self.temperature,self.json_output=endpoint,alias,phase,temperature,json_output
  self.local=threading.local();self.max_tokens=8192
 def generate(self,system,prompt,max_tokens=8192,response_format=None):
  messages=[dict(role='system',content=system),dict(role='user',content=prompt)]
  count=post(self.endpoint,'/tokenize',dict(model=self.model_name,messages=messages,add_generation_prompt=True,chat_template_kwargs={'enable_thinking':False}))['count']
  if count+max_tokens>32768:raise ValueError(f'No-truncation context overflow: {count}+{max_tokens}>32768')
  payload=dict(model=self.model_name,messages=messages,max_tokens=max_tokens,temperature=self.temperature,top_p=.95,chat_template_kwargs={'enable_thinking':False})
  if self.json_output:payload.update(response_format=response_format or {'type':'json_object'},seed=getattr(self.local,'seed',20261009))
  if self.phase=='qa':payload.update(frequency_penalty=1.05,presence_penalty=1.05)
  tick=time.monotonic()
  for attempt in range(3):
   try:
    response=post(self.endpoint,'/v1/chat/completions',payload)
    choice=response['choices'][0];text=choice['message'].get('content')
    usage=response.get('usage',{});reason=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens',0)
    assert not reason and not choice['message'].get('reasoning_content') and not choice['message'].get('reasoning'),'Unexpected thinking output'
    trace=dict(request=payload,response=response,elapsed_seconds=time.monotonic()-tick,document_id=getattr(self.local,'document_id',None),condition=getattr(self.local,'condition',None),job_id=os.environ.get('SLURM_JOB_ID'))
    path=ROOT/'traces'/self.phase/f'{time.time_ns()}-{threading.get_ident()}.json';write(path,trace)
    self.local.last=dict(trace_file=str(path.relative_to(ROOT)),usage=usage,finish_reason=choice.get('finish_reason'),response=text,prompt_tokens=count,elapsed_seconds=trace['elapsed_seconds'])
    if not isinstance(text,str) or not text.strip():raise ValueError('Empty final content')
    return text
   except Exception:
    if attempt==2:raise
    time.sleep(2**attempt)

ARMS=[]
read_jsonl=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
sha=digest
def jsonl(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
