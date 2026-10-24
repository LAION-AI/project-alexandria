import hashlib,json,os,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
LEGACY=Path('/e/scratch/reformo/schuhmann1/scientific-unrepaired-audit-qwen08-20261023')
sys.path[:0]=[str(ROOT/'vendor'),str(ROOT/'vendor/src')]
MODELS={'qwen08':ROOT/'models/Qwen3.5-0.8B','gemma_e2b':ROOT/'models/gemma-4-E2B-it'}
CONDITIONS=['baseline','safe_again','lexicon_v1','lexicon_v2']
def load(p):return json.loads(Path(p).read_text())
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+f'.tmp-{os.getpid()}-{threading.get_ident()}')
 temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def read_jsonl(p):
 with Path(p).open() as stream:return [json.loads(line) for line in stream if line.strip()]
def jsonl(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+f'.tmp-{os.getpid()}')
 with temp.open('w') as stream:
  for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
 temp.replace(p)
def cohort():return read_jsonl(ROOT/'inputs/cohort97.jsonl')
def condition_file(c,uid):return ROOT/f'outputs/repaired/{c}/{uid}.json'
def named_values(text):
 import re
 # Exact acronyms and chemical/collision spellings such as Au+Au. A separate
 # numerical/formula signature is checked too. This is not complete entity NER.
 return set(re.findall(r'\b[A-Z][A-Z0-9]{1,}(?:[-_][A-Za-z0-9]+)*\b|\b[A-Z][a-z]?(?:\+[A-Z][a-z]?)+\b',text))
def negations(text):
 import re
 value=re.sub(r"n['’]t\b",' not',text.casefold())
 return len(re.findall(r'\b(?:not|no|never|without|neither|nor)\b',value))
def nonoverlapping_passages(index,text):
 from ngram_overlap import tokenize
 _,words,_=tokenize(text);runs=set()
 for i in range(max(0,len(words)-4)):
  for j in index.five.get(tuple(words[i:i+5]),()):
   if i and j and words[i-1]==index.words[j-1]:continue
   length=5
   while i+length<len(words) and j+length<len(index.words) and words[i+length]==index.words[j+length]:length+=1
   if length>=6:runs.add((i,i+length))
 chosen=[]
 for a,b in sorted(runs,key=lambda p:(-(p[1]-p[0]),p[0])):
  if not any(a<d and b>c for c,d in chosen):chosen.append((a,b))
 return [b-a for a,b in chosen]
