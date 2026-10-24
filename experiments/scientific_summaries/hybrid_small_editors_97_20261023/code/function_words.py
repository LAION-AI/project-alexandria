"""QA-blind, finite phrase rules; no dictionary substitutions of technical words.

This is an additional ablation, not a claim that function words are harmless.
Every changed sentence is retained for an independent automatic meaning review.
"""
import concurrent.futures,json,os,re,statistics,time
from collections import Counter
from common import ROOT,write,read_jsonl,jsonl,digest
from ngram_overlap import SourceIndex,fragments,tokenize,merged_length,audit_summary
from critical_values import check,LATEX
from passages import patch
from reference_validator import narrative_context,validate_summary
from cpu_audit import matches

FUNCTION=['function_connectives','function_contractions','function_prepositions','function_coordination']

# These are fixed grammatical expressions, not a general synonym lexicon.
CONNECTIVES=[
 ('for_example',r'\bfor example\b','for instance'),
 ('for_instance',r'\bfor instance\b','for example'),
 ('in_addition',r'\bin addition\b(?!\s+to\b)','additionally'),
 ('additionally',r'\badditionally\b','in addition'),
 ('furthermore',r'\bfurthermore\b','moreover'),
 ('moreover',r'\bmoreover\b','furthermore'),
 ('however_comma',r'\bhowever(?=\s*[,;])','nevertheless'),
 ('nevertheless_comma',r'\bnevertheless(?=\s*[,;])','however'),
 ('therefore',r'\btherefore\b','thus'),
 ('hence_comma',r'\bhence(?=\s*[,;])','therefore'),
 ('although',r'\balthough\b','though'),
 ('in_order_to',r'\bin order to\b','to'),
 ('so_as_to',r'\bso as to\b','to'),
]

# Negation is retained as an ordinary English contraction, never deleted.
CONTRACTIONS=[
 (a.replace(' ','_'),r'\b'+a+r'\b',b) for a,b in [
 ('is not',"isn't"),('are not',"aren't"),('was not',"wasn't"),
 ('were not',"weren't"),('do not',"don't"),('does not',"doesn't"),
 ('did not',"didn't"),('has not',"hasn't"),('have not',"haven't"),
 ('had not',"hadn't"),('will not',"won't"),('would not',"wouldn't"),
 ('should not',"shouldn't"),('could not',"couldn't"),('must not',"mustn't"),
 ]
]

# No bare in/on/at/by/to/for swaps. They encode relations and quantities.
PREPOSITIONS=[
 ('with_regard_to',r'\bwith regard to\b','regarding'),
 ('with_respect_to',r'\bwith respect to\b','in relation to'),
 ('in_relation_to',r'\bin relation to\b','with respect to'),
 ('in_comparison_with',r'\bin comparison with\b','compared with'),
 ('compared_with',r'\bcompared with\b','in comparison with'),
 ('because_of',r'\bbecause of\b','owing to'),
 ('owing_to',r'\bowing to\b','because of'),
 ('by_means_of',r'\bby means of\b(?!\s+using\b)','using'),
 ('prior_to',r'\bprior to\b','before'),
 ('subsequent_to',r'\bsubsequent to\b','after'),
 ('among',r'\bamong\b','amongst'),
 ('amongst',r'\bamongst\b','among'),
 ('toward',r'\btoward\b','towards'),
 ('towards',r'\btowards\b','toward'),
]
COORDINATION=[
 ('and_noun_or_adjective',r'\band\b','as well as'),
 ('but_contrast',r'\bbut\b','yet'),
 ('also_adverb',r'\balso\b','additionally'),
]
RULES={
 FUNCTION[0]:CONNECTIVES,
 FUNCTION[1]:CONNECTIVES+CONTRACTIONS,
 FUNCTION[2]:CONNECTIVES+CONTRACTIONS+PREPOSITIONS,
 FUNCTION[3]:CONNECTIVES+CONTRACTIONS+PREPOSITIONS+COORDINATION,
}

def init():
 global splitter,nltk_tools,word_splitter
 import nltk
 nltk.data.path.insert(0,str(ROOT/'resources/nltk_data'))
 splitter=nltk.tokenize.PunktTokenizer('english')
 nltk_tools=nltk;word_splitter=nltk.tokenize.TreebankWordTokenizer()

def coordination_allowed(rule,text,match):
 """Local POS/context heuristics, deliberately not a guarantee of grammar."""
 if rule not in {r[0] for r in COORDINATION}:return True
 a,b=match.span();left=text[:a];right=text[b:]
 clause=re.split(r'[.;:!?\n]',left)[-1].casefold()
 if re.search(r'\b(?:both|between|either|neither|whether)\b|\bnot\s+only\b',clause):return False
 if any(s in clause for s in ['=','<','>','∧','∨','&']):return False
 if re.search(r'\b(?:trial\s+and\s+error|necessary\s+and\s+sufficient|if\s+and\s+only\s+if)\b',text,re.I):return False
 if rule=='also_adverb':return not re.search(r'\bnot\s*$',left,re.I)
 if rule=='but_contrast':
  return not re.search(r'\b(?:all|anything|nothing|not|only)\b',clause)
 # Restrict 'and' to simple common-noun/adjective coordination. Proper names,
 # clause conjunctions, explicit correlatives and nearby plural-subject verbs
 # remain unchanged. Residual agreement/idiom errors are measured afterwards.
 spans=list(word_splitter.span_tokenize(text));tags=nltk_tools.pos_tag([text[c:d] for c,d in spans])
 preceding=[(word,tag) for (c,d),(word,tag) in zip(spans,tags) if d<=a]
 following=[(word,tag) for (c,d),(word,tag) in zip(spans,tags) if c>=b]
 if not preceding or not following:return False
 valid={'NN','NNS','JJ','JJR','JJS'}
 if preceding[-1][1] not in valid or following[0][1] not in valid:return False
 # Stop at the first verb/punctuation after the second coordinate.
 for word,tag in following[:8]:
  if tag.startswith('VB'):
   if word.casefold() in {'are','were','have'}:return False
   break
  if tag in {'.',',',':'}:break
 return True

def surface_case(original,replacement):
 if original.isupper():return replacement.upper()
 if original[0].isupper():return replacement[0].upper()+replacement[1:]
 return replacement

def six_state(text,index):
 _,words,_=tokenize(text);grams=index.grams(6)
 hits=[i for i in range(max(0,len(words)-5)) if tuple(words[i:i+6]) in grams]
 return words,hits,merged_length((i,i+6) for i in hits)

def repair_fragment(text,index,condition):
 original=text;edits=[];rejections=Counter();tick=time.monotonic()
 for iteration in range(128):
  words,hits,covered=six_state(text,index)
  if not hits:break
  # The audit normalizes Unicode. Map only when original whitespace-token
  # boundaries agree exactly, avoiding unsafe offsets in PDF/math fragments.
  raw=[]
  for match in re.finditer(r'\S+',text):
   values=tokenize(match.group())[1]
   if len(values)>1:break
   if values:raw.append((match.start(),match.end(),values[0]))
  if [r[2] for r in raw]!=words:
   rejections['normalization_offset_ambiguity']+=1;break
  matched_spans=[(raw[i][0],raw[i+5][1]) for i in hits]
  protected=[m.span() for m in LATEX.finditer(text)]
  protected += [m.span() for m in re.finditer(r'`[^`]*`|"[^"\n]*"|“[^”\n]*”',text)]
  options=[]
  for priority,(rule,pattern,replacement) in enumerate(RULES[condition]):
   for match in re.finditer(pattern,text,re.I):
    a,b=match.span()
    if not any(a<d and b>c for c,d in matched_spans):continue
    if any(a<d and b>c for c,d in protected):
     rejections['protected_math_or_literal']+=1;continue
    if not coordination_allowed(rule,text,match):
     rejections['coordination_grammar_guard']+=1;continue
    value=surface_case(match.group(),replacement)
    candidate=text[:a]+value+text[b:]
    _,new_hits,new_covered=six_state(candidate,index)
    # Require absolute progress, not a lower fraction from a longer denominator.
    if len(new_hits)>=len(hits) or new_covered>=covered:continue
    if not check(original,candidate)['passed']:
     rejections['critical_signature']+=1;continue
    options.append((-(len(hits)-len(new_hits)),new_covered,priority,a,b,rule,value,candidate,match.group(),len(new_hits)))
  if not options:break
  best=min(options,key=lambda o:o[:5]);_,_,_,a,b,rule,value,candidate,old,remaining=best
  edits.append(dict(rule=rule,old=old,new=value,start=a,end=b,sixgrams_before=len(hits),sixgrams_after=remaining))
  text=candidate
 return text,dict(original=original,repaired=text,edits=edits,rejections=dict(rejections),seconds=time.monotonic()-tick)

def process(row):
 index=SourceIndex(row['fulltext']);assert index.sha256==row['source_sha256']
 baseline=audit_summary(row['fulltext'],row['summary'],index=index,expanded_diagnostic=False)
 output=[]
 for condition in FUNCTION:
  tick=time.monotonic();replacements={};traces=[]
  for text,path,category in fragments(row['summary']):
   if category!='narrative':continue
   if not six_state(text,index)[1]:continue
   value,trace=repair_fragment(text,index,condition);trace.update(uid=row['uid'],path=path,condition=condition)
   traces.append(trace)
   if value!=text:replacements[(path,text)]=value
  rollback=False;error=None
  try:
   summary=patch(row['summary'],replacements);context=narrative_context(summary)
   assert check(row['judge_context'],context)['passed'],'Document signature changed'
   assert check(json.dumps(row['summary'],ensure_ascii=False),json.dumps(summary,ensure_ascii=False))['passed'],'Serialized signature changed'
   assert [(t,p,c) for t,p,c in fragments(summary) if c!='narrative']==[(t,p,c) for t,p,c in fragments(row['summary']) if c!='narrative'],'Evidence/metadata changed'
   try:validate_summary(json.dumps(row['summary'],ensure_ascii=False),row['fulltext']);original_valid=True
   except (ValueError,KeyError,TypeError):original_valid=False
   if original_valid:validate_summary(json.dumps(summary,ensure_ascii=False),row['fulltext'])
   audit=audit_summary(row['fulltext'],summary,index=index,expanded_diagnostic=False)
   assert audit['categories']['narrative']['longest_contiguous_match_words']<=baseline['categories']['narrative']['longest_contiguous_match_words'],'Longest match increased'
  except (AssertionError,ValueError,KeyError,TypeError) as e:
   summary=row['summary'];context=row['judge_context'];audit=baseline;rollback=True;error=repr(e)
  pairs=[]
  if not rollback:
   for trace in traces:
    if not trace['edits']:continue
    before=splitter.tokenize(trace['original']);after=splitter.tokenize(trace['repaired'])
    unit='sentence'
    if len(before)!=len(after):before=[trace['original']];after=[trace['repaired']];unit='fragment'
    for i,(a,b) in enumerate(zip(before,after)):
     if a!=b:
      pair=dict(uid=row['uid'],path=trace['path'],sentence=i,before=a,after=b,condition=condition,unit=unit)
      pair['pair_id']=digest(json.dumps(pair,sort_keys=True));pairs.append(pair)
  passage_lengths=[n for text,path,category in fragments(summary) if category=='narrative' for n in matches(index,text)['lengths']]
  record=dict(uid=row['uid'],document_id=row['document_id'],group=row['group'],condition=condition,summary=summary,judge_context=context,audit=audit,
   narrative_nonoverlapping_passages_6plus=len(passage_lengths),narrative_passage_lengths_words=passage_lengths,
   source_sha256=row['source_sha256'],rolled_back=rollback,error=error,critical=check(row['judge_context'],context),semantics_verified=False)
  write(ROOT/f'outputs/repaired/{condition}/{row["uid"]}.json',record)
  jsonl(ROOT/f'traces/{condition}/{row["uid"]}.jsonl',traces)
  rules=Counter(e['rule'] for t in traces for e in t['edits']) if not rollback else Counter()
  output.append(dict(condition=condition,uid=row['uid'],changed=context!=row['judge_context'],rollback=rollback,
   audit=audit['categories']['narrative'],pairs=pairs,rules=dict(rules),passage_lengths=passage_lengths,seconds=time.monotonic()-tick))
 return output

def run():
 assert os.environ.get('SLURM_JOB_ID'),'Full-corpus work belongs on compute'
 if all((ROOT/f'outputs/{c}_complete.json').exists() for c in FUNCTION):return
 tick=time.monotonic();rows=[r for r in read_jsonl(ROOT/'inputs/cohort.jsonl') if r['group']=='gemma_r128_fp8']
 write(ROOT/'inputs/function_words_protocol.json',dict(conditions=FUNCTION,rules=RULES,inputs=97,
  acceptance='Strict reduction of both overlapping six-gram count and absolute copied word count; immutable-original numbers/units/formulas; protected math/literal spans; unchanged evidence/metadata; document rollback on validation failure.',
  semantic_filter=False,meaning_review='Every actually changed sentence; automatic reviewer plus selected examples; no claim of guaranteed equivalence.',
  prohibited='No arbitrary article/determiner, negation, quantifier, or bare in/on/at/by/to/for swaps. No character substitutions or punctuation-only tricks.'))
 with concurrent.futures.ProcessPoolExecutor(max_workers=12,initializer=init) as pool:
  results=[item for group in pool.map(process,rows,chunksize=2) for item in group]
 wall=time.monotonic()-tick
 for condition in FUNCTION:
  rs=[r for r in results if r['condition']==condition];pairs=[p for r in rs for p in r['pairs']]
  jsonl(ROOT/f'inputs/{condition}_sentence_pairs.jsonl',pairs)
  counts=Counter()
  for r in rs:counts.update(r['rules'])
  write(ROOT/f'outputs/{condition}_performance.json',dict(cpu_only=True,cpu_worker_seconds=sum(r['seconds'] for r in rs),
   shared_condition_wall_seconds=wall,conditions=len(FUNCTION),workers=12,repair_gpu_hours=0,rule_counts=dict(counts)))
  lengths=[n for r in rs for n in r['passage_lengths']]
  write(ROOT/f'outputs/{condition}_complete.json',dict(complete=True,condition=condition,papers=len(rs),changed_summaries=sum(r['changed'] for r in rs),
   rollbacks=sum(r['rollback'] for r in rs),mean_coverage=statistics.mean(r['audit']['covered_word_fraction_by_minimum_run']['6'] for r in rs),
   strict_passes=sum(r['audit']['strict_five_word_limit_passed'] for r in rs),changed_sentence_units=len(pairs),edits=sum(counts.values()),rule_counts=dict(counts),
   narrative_nonoverlapping_passages_6plus=len(lengths),mean_passages_per_summary=len(lengths)/len(rs),mean_passage_length_words=statistics.mean(lengths) if lengths else 0))
  print('FUNCTION_DONE',condition,'pairs',len(pairs),'edits',sum(counts.values()),flush=True)

def main():
 import fcntl
 # The short counting allocation and the full QA allocation share results.
 # Serialize them so they cannot produce competing completion checkpoints.
 with (ROOT/'outputs/.function_words.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  run()

if __name__=='__main__':main()
