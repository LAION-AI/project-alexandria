"""Frozen full-source audit and an explicitly unsafe first-sense synonym diagnostic."""
import concurrent.futures,functools,json,os,re,statistics,time
from collections import Counter,defaultdict
from common import ROOT,load,write,read_jsonl,jsonl,digest
from ngram_overlap import SourceIndex,fragments,tokenize,merged_length,audit_summary
from passages import select

LEXROOT='/e/fscratch/reformo/schuhmann1/scientific-procedural-synonyms-20261012'

def init():
 global splitter,editor,protected,blocked_lexnames
 import nltk,procedural
 nltk.data.path.insert(0,str(ROOT/'resources/nltk_data'))
 splitter=nltk.tokenize.PunktTokenizer('english')
 protected=set(procedural.PROTECTED)
 blocked_lexnames=set(procedural.BLOCK_LEXNAMES)
 protected.update(word for phrase in procedural.COLLOCATIONS for word in phrase.split())
 # Diagnostic: remove scientific sense/collocation protections, keeping ordinary
 # spelling/frequency/POS/inflection and numeric/formula checks. No NLI gating.
 procedural.PROTECTED=set();procedural.BLOCK_LEXNAMES=set()
 editor=procedural.LexicalRepair();editor.collocations=[]

def matches(index,text):
 norm,words,pos=tokenize(text);unique={}
 for i in range(max(0,len(words)-4)):
  for j in index.five.get(tuple(words[i:i+5]),()):
   if i and j and words[i-1]==index.words[j-1]:continue
   n=5
   while i+n<len(words) and j+n<len(index.words) and words[i+n]==index.words[j+n]:n+=1
   if n>=6:unique.setdefault((i,i+n),(j,j+n))
 # Longest-first, non-overlapping target occurrences. Not sliding six-grams.
 chosen=[]
 for (a,b),(j,k) in sorted(unique.items(),key=lambda v:(-(v[0][1]-v[0][0]),v[0][0],v[1][0])):
  if any(a<d and b>c for c,d,_,_ in chosen):continue
  chosen.append((a,b,j,k))
 lengths=[b-a for a,b,_,_ in chosen]
 covered=merged_length(unique.keys())
 examples=[dict(words=b-a,text=norm[pos[a][0]:pos[b-1][1]],source_start=j,source_end=k) for a,b,j,k in chosen]
 six=[]
 for a,b,_,_ in chosen:
  for i in range(a,b-5):six.append(norm[pos[i][0]:pos[i+5][1]])
 return dict(words=len(words),covered=covered,lengths=lengths,count=len(chosen),maximal_count=len(unique),examples=examples,six_phrases=list(dict.fromkeys(six)))

def process(row):
 if not isinstance(row['summary'],dict):return dict(uid=row['uid'],group=row['group'],failure=True)
 index=SourceIndex(row['fulltext']);assert index.sha256==row['source_sha256']
 categories={k:dict(words=0,covered=0,lengths=[],count=0,maximal_count=0,sentences=0,affected_sentences=0) for k in ['narrative','evidence','metadata']}
 examples=[]
 for text,path,cat in fragments(row['summary']):
  m=matches(index,text);c=categories[cat]
  for key in ['words','covered','count','maximal_count']:c[key]+=m[key]
  c['lengths']+=m['lengths']
  spans=list(splitter.span_tokenize(text));c['sentences']+=len(spans)
  c['affected_sentences']+=sum(matches(index,text[a:b])['count']>0 for a,b in spans)
  if cat=='narrative':examples.extend(dict(path=path,**e) for e in m['examples'])
 result=dict(uid=row['uid'],document_id=row['document_id'],group=row['group'],categories=categories,examples=sorted(examples,key=lambda x:-x['words'])[:5])
 if row['group']!='gemma_r128_fp8':return result
 repaired,traces=editor.repair_document(row,index,'synonyms_broad')
 write(ROOT/f'outputs/repaired/naive/{row["uid"]}.json',repaired)
 pairs=[]
 if not repaired['repair']['rolled_back']:
  for t in traces:
   if not t['edits']:continue
   before=splitter.tokenize(t['original']);after=splitter.tokenize(t['repaired'])
   if len(before)!=len(after):before=[t['original']];after=[t['repaired']]
   for i,(a,b) in enumerate(zip(before,after)):
    if a!=b:pairs.append(dict(uid=row['uid'],path=t['path'],sentence=i,before=a,after=b,condition='naive',unit='sentence' if len(before)>1 or len(splitter.tokenize(a))==1 else 'fragment'))
 jsonl(ROOT/f'traces/naive/{row["uid"]}.jsonl',traces)
 windows=select(row['summary'],row['fulltext'])
 for i,w in enumerate(windows):
  tokens=editor.nltk.word_tokenize(w['text'],preserve_line=True);tags=editor.nltk.pos_tag(tokens)
  all_words={word for word,tag in tags if (tag.startswith('NN') or tag.startswith('JJ')) and re.fullmatch('[A-Za-z]{3,}',word)}
  # Protected diagnostic keeps technical/risky vocabulary, proper names, and
  # words without a common same-POS dictionary alternative out of the ban list.
  guarded={word for word,tag in tags if tag in ['NN','NNS','JJ'] and re.fullmatch('[a-z]{3,}',word) and word not in protected and editor.freq(word,'en')>=4.0 and any(info['lexname'] not in blocked_lexnames for _,info in editor.lookup(word,tag,'synonyms_broad'))}
  w.update(uid=row['uid'],window_index=i,ban_all=sorted(all_words),ban_guarded=sorted(guarded),ban_phrases=matches(index,w['text'])['six_phrases'])
 result.update(pairs=pairs,windows=windows,naive_repair=repaired['repair'])
 return result

def quantile(values,p):
 if not values:return None
 x=sorted(values);i=(len(x)-1)*p;a=int(i);return x[a]+(x[min(a+1,len(x)-1)]-x[a])*(i-a)

def dist(values):
 return dict(n=len(values),mean=statistics.mean(values) if values else None,median=quantile(values,.5),p90=quantile(values,.9),p95=quantile(values,.95),maximum=max(values,default=0))

def main():
 assert os.environ.get('SLURM_JOB_ID'),'Run the full analysis on compute'
 if (ROOT/'outputs/cpu_complete.json').exists():print('CPU_CHECKPOINT_REUSED',flush=True);return
 tick=time.monotonic();rows=read_jsonl(ROOT/'inputs/cohort.jsonl')
 cached=load(ROOT/'outputs/cached_full_audit.json') if (ROOT/'outputs/cached_full_audit.json').exists() else None
 work=[r for r in rows if r['group']=='gemma_r128_fp8'] if cached else rows
 with concurrent.futures.ProcessPoolExecutor(max_workers=12,initializer=init) as pool:results=list(pool.map(process,work,chunksize=3))
 report=dict(cached['groups']) if cached else {}
 for group in ['gemma_r128_fp8','gemma_r128_bf16','qwen27_raw_generator']:
  selected=[r for r in results if r['group']==group];good=[r for r in selected if not r.get('failure')];cats={}
  if not selected:continue
  for cat in ['narrative','evidence','metadata']:
   cs=[r['categories'][cat] for r in good];lengths=[n for c in cs for n in c['lengths']]
   cats[cat]=dict(passage_lengths_words=dist(lengths),passages_per_summary=dist([c['count'] for c in cs]),maximal_match_count_including_overlap=sum(c['maximal_count'] for c in cs),
    longest_match_per_summary=dist([max(c['lengths'],default=0) for c in cs]),mean_covered_fraction=statistics.mean(c['covered']/c['words'] if c['words'] else 0 for c in cs),
    pooled_covered_fraction=sum(c['covered'] for c in cs)/max(1,sum(c['words'] for c in cs)),summary_word_count=dist([c['words'] for c in cs]),
    sentence_units=sum(c['sentences'] for c in cs),affected_sentence_units=sum(c['affected_sentences'] for c in cs),strict_passes=sum(not c['count'] for c in cs),
    length_histogram={label:sum(lo<=n<=hi for n in lengths) for label,lo,hi in [('6',6,6),('7–9',7,9),('10–14',10,14),('15–19',15,19),('20–29',20,29),('30+',30,100000)]})
  report[group]=dict(slots=len(selected),auditable=len(good),failures=len(selected)-len(good),categories=cats)
 assert abs(report['gemma_r128_fp8']['categories']['narrative']['mean_covered_fraction']-.15227428470594992)<1e-12
 primary=[r for r in results if r['group']=='gemma_r128_fp8']
 pairs=[p for r in primary for p in r['pairs']];windows=[w for r in primary for w in r['windows']]
 for i,p in enumerate(pairs):p['pair_id']=digest(json.dumps(p,sort_keys=True))
 jsonl(ROOT/'inputs/naive_sentence_pairs.jsonl',pairs);jsonl(ROOT/'inputs/windows.jsonl',windows)
 jsonl(ROOT/'outputs/raw_per_summary.jsonl',[{k:v for k,v in r.items() if k not in ['windows','pairs']} for r in results])
 write(ROOT/'outputs/raw_audit.json',dict(groups=report,unique_papers=962,summary_versions=1000,seconds=time.monotonic()-tick,
  passage_definition='Maximal normalized source matches >=6 words; canonical source occurrence per target interval; longest-first non-overlapping target intervals. Means describe this declared counting rule, not all overlapping six-grams.',
  naive=dict(changed_sentence_units=len(pairs),changed_summaries=len({p['uid'] for p in pairs}),edits=sum(r['naive_repair']['edit_count'] for r in primary),rollbacks=sum(r['naive_repair']['rolled_back'] for r in primary))))
 write(ROOT/'outputs/cpu_complete.json',dict(complete=True,windows=len(windows),naive_pairs=len(pairs)))
 print('CPU_COMPLETE',len(windows),len(pairs),flush=True)

if __name__=='__main__':main()
