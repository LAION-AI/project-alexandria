"""Deterministic lexical overlap repair; lexical guards do not prove semantic equivalence."""
import re,time,json,functools
from collections import Counter
from common import ROOT,load,write,digest
from ngram_overlap import SourceIndex,tokenize,fragments,audit_summary,VERSION as AUDIT_VERSION
from passages import patch
from critical_values import check,VERSION as CRITICAL_VERSION
from reference_validator import narrative_context,validate_summary

# Protect scientific quantities, polarity/uncertainty and common technical collocations.
PROTECTED=set('significant significance statistically statistical insignificant positive negative normal abnormal control controls controlled random randomized randomised blind blinded double sample samples sampling population populations sensitivity specificity precision accuracy recall bias unbiased variance deviation error errors risk odds ratio correlation causal causality associated association independent dependent null hypothesis hypotheses absolute relative total mean median average minimum maximum optimal optimum efficient effective efficacy probability likelihood mass weight force energy work power charge current resistance flux strain stress pressure volume temperature frequency velocity speed acceleration density momentum entropy field spin quantum chemical biological genetic genomic molecular neural temporal spatial clinical medical magnetic electric electrical nuclear experimental empirical theoretical critical stable unstable convergent divergent continuous discrete finite infinite linear nonlinear uniform nonuniform convex concave constant variable stochastic deterministic novel uncertain possible probable certain robust weak strong high low increased decreased greater lower less more fewer many much all every none neither either only exactly approximately about up down rise fall increase decrease reduction limit confidence interval'.split())
COLLOCATIONS=['neural network','confidence interval','randomized controlled','statistical significance','standard deviation','standard error','type i','type ii','false positive','false negative','positive predictive','negative predictive','relative risk','absolute risk','normal distribution','linear regression','control group','treatment group','experimental group','energy gap','magnetic field','machine learning','deep learning','sample size','effect size','null hypothesis','causal effect','critical point','phase transition','gene expression','black hole','white dwarf','brown dwarf','dark matter','dark energy','heavy metal','light curve','cross section','event horizon','general relativity','special relativity','active learning','open access','peer review']
BLOCK_LEXNAMES={'noun.animal','noun.body','noun.feeling','noun.food','noun.location','noun.person','noun.phenomenon','noun.plant','noun.quantity','noun.state','noun.substance','noun.time'}

class LexicalRepair:
 def __init__(self):
  import nltk
  from nltk.corpus.reader.wordnet import WordNetCorpusReader
  from nltk.tokenize import TreebankWordTokenizer
  from wordfreq import zipf_frequency
  from lemminflect import getInflection
  nltk.data.path.insert(0,str(ROOT/'resources/nltk_data'));self.nltk=nltk;self.split=TreebankWordTokenizer();self.freq=zipf_frequency;self.inflect=getInflection
  self.wn=WordNetCorpusReader(str(ROOT/'resources/oewn2025/oewn2025'),None)
  self.lookup=functools.lru_cache(maxsize=50000)(self._lookup)
  self.collocations=[tokenize(x)[1] for x in COLLOCATIONS]

 def _lookup(self,word,tag,mode):
  pos='n' if tag in ['NN','NNS'] else 'a';lemma=self.wn.morphy(word,pos)
  if not lemma or lemma in PROTECTED:return ()
  synsets=list(dict.fromkeys(self.wn.synsets(lemma,pos)))
  if not synsets:return ()
  if mode=='synonyms_strict' and len(synsets)!=1:return ()
  # Broad arm deliberately tests the common shortcut of choosing the first sense.
  synset=synsets[0]
  if synset.lexname() in BLOCK_LEXNAMES:return ()
  values=[]
  for candidate in synset.lemma_names():
   if not re.fullmatch('[a-z]+',candidate) or candidate==lemma or candidate in PROTECTED:continue
   inverse=list(dict.fromkeys(self.wn.synsets(candidate,pos)))
   if mode=='synonyms_strict' and (len(inverse)!=1 or inverse[0]!=synset):continue
   freq=self.freq(candidate,'en')
   if freq<3.5 or freq<self.freq(lemma,'en')-1.5:continue
   forms=self.inflect(candidate,tag=tag) if tag=='NNS' else (candidate,)
   for form in forms or ():
    if not re.fullmatch('[a-z]+',form) or form in PROTECTED or tokenize(form)[1]==tokenize(word)[1]:continue
    values.append((form,dict(original_lemma=lemma,candidate_lemma=candidate,synset=synset.name(),definition=synset.definition(),lexname=synset.lexname(),source_senses=len(synsets),candidate_senses=len(inverse),zipf_frequency=freq)))
  return tuple(sorted(values,key=lambda v:(-v[1]['zipf_frequency'],v[0])))

 def repair_fragment(self,text,index,mode):
  original=text;current=text;edits=[];edited_positions=set();skip=None
  for iteration in range(1):
   _,words,_=tokenize(current);six=index.grams(6)
   hits=[i for i in range(len(words)-5) if tuple(words[i:i+6]) in six]
   if not hits:break
   # Map normalized word positions back to original whitespace token offsets.
   raw=[]
   for m in re.finditer(r'\S+',current):
    values=tokenize(m.group())[1]
    if values:
     if len(values)!=1:skip='normalization_offset_ambiguity';break
     raw.append((m.start(),m.end(),m.group(),values[0]))
   if skip or [r[3] for r in raw]!=words:skip='normalization_offset_ambiguity';break
   spans=list(self.split.span_tokenize(current));tags=self.nltk.pos_tag([current[a:b] for a,b in spans]);tagged={(a,b):tag for (a,b),(_,tag) in zip(spans,tags)}
   protected=set()
   for phrase in self.collocations:
    for i in range(len(words)-len(phrase)+1):
     if words[i:i+len(phrase)]==phrase:protected.update(range(i,i+len(phrase)))
   covered=Counter(j for i in hits for j in range(i,i+6));options=[]
   for j,count in covered.items():
    if j in protected or j in edited_positions:continue
    a,b,surface,norm=raw[j];m=re.fullmatch(r'([^A-Za-z]*)([a-z]+)([^A-Za-z]*)',surface)
    if not m:continue
    word=m.group(2);wa=a+len(m.group(1));wb=wa+len(word);tag=tagged.get((wa,wb))
    if tag not in ['NN','NNS','JJ'] or word in PROTECTED:continue
    for form,info in self.lookup(word,tag,mode):
     options.append((-count,-info['zipf_frequency'],j,wa,wb,word,form,tag,info))
   accepted=False
   for _,_,j,wa,wb,word,form,tag,info in sorted(options,key=lambda v:v[:3]):
    candidate=current[:wa]+form+current[wb:];cw=tokenize(candidate)[1]
    if len(cw)!=len(words):continue
    next_hits=[i for i in range(len(cw)-5) if tuple(cw[i:i+6]) in six]
    if len(next_hits)>=len(hits):continue
    critical=check(original,candidate)
    if not critical['passed']:continue
    edits.append(dict(old=word,new=form,tag=tag,word_position=j,hit_sixgrams_before=len(hits),hit_sixgrams_after=len(next_hits),lexical=info))
    current=candidate;edited_positions.add(j);accepted=True;break
   if not accepted:break
  final_words=tokenize(current)[1]
  residual=sum(tuple(final_words[i:i+6]) in index.grams(6) for i in range(max(0,len(final_words)-5)))
  return current,dict(edits=edits,residual_sixgrams=residual,skip_reason=skip,original=text,repaired=current,semantics_verified=False)

 def repair_document(self,row,index,mode):
  if not isinstance(row['summary'],dict):
   return dict(uid=row['uid'],document_id=row['document_id'],group=row['group'],source_sha256=row['source_sha256'],condition=mode,status='input_unparseable',summary=None,judge_context='',audit=None,repair=dict(input_error=row.get('input_error'),semantics_verified=False)),[]
  tick=time.monotonic();replacements={};trace=[]
  for text,path,category in fragments(row['summary']):
   if category!='narrative':continue
   words=tokenize(text)[1]
   if not any(tuple(words[i:i+6]) in index.grams(6) for i in range(max(0,len(words)-5))):continue
   value,t=self.repair_fragment(text,index,mode);t.update(path=path,uid=row['uid'],condition=mode);trace.append(t)
   if value!=text:replacements[(path,text)]=value
  repair_seconds=time.monotonic()-tick
  summary=row['summary'];error=None;rollback=False
  try:
   result=patch(summary,replacements);context=narrative_context(result)
   assert [(s,p,c) for s,p,c in fragments(summary) if c!='narrative']==[(s,p,c) for s,p,c in fragments(result) if c!='narrative'],'Protected proof/metadata changed'
   critical=check(row['judge_context'],context);json_critical=check(json.dumps(summary,ensure_ascii=False),json.dumps(result,ensure_ascii=False))
   assert critical['passed'] and json_critical['passed'],'Assembled numeric/unit/formula signatures changed'
   try:validate_summary(json.dumps(summary,ensure_ascii=False),row['fulltext']);valid_original=True
   except (ValueError,KeyError,TypeError):valid_original=False
   if valid_original:validate_summary(json.dumps(result,ensure_ascii=False),row['fulltext'])
   old=audit_summary(row['fulltext'],summary,index=index);audit=audit_summary(row['fulltext'],result,index=index)
   assert audit['categories']['narrative']['longest_contiguous_match_words']<=old['categories']['narrative']['longest_contiguous_match_words'],'Longest source match increased'
  except (AssertionError,ValueError,KeyError,TypeError) as e:
   result=summary;context=row['judge_context'];audit=audit_summary(row['fulltext'],summary,index=index);rollback=True;error=repr(e)
  return dict(uid=row['uid'],document_id=row['document_id'],group=row['group'],source_sha256=row['source_sha256'],condition=mode,status='generated',summary=result,judge_context=context,audit=audit,
   repair=dict(seconds=repair_seconds,critical_values=check(row['judge_context'],context),rolled_back=rollback,error=error,changed=context!=row['judge_context'],edit_count=sum(len(t['edits']) for t in trace),edited_fragments=len(replacements),semantics_verified=False)),trace
