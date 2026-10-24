"""Same residuals, adaptive batched paraphrasing, immutable-original guards."""
import json,os,re,statistics,time
from collections import Counter
from common import ROOT,MODELS,cohort,load,write,read_jsonl,jsonl,condition_file,digest,nonoverlapping_passages,named_values,negations
from ngram_overlap import SourceIndex,audit_summary,fragments,tokenize
from passages import assemble
from critical_values import check
from reference_validator import narrative_context,validate_summary

SYSTEM=('Paraphrase scientific prose in English while preserving every scientific claim. '
 'Keep scientific terminology, entity names, numbers, unit symbols, formulas, uncertainty and negation unchanged. '
 'Use different sentence structure rather than replacing precise technical terms. '
 'Do not omit facts, add facts, explain your work, or insert filler. Return only the rewritten TEXT.')

def phrases(index,text):
 normalized,words,positions=tokenize(text);six=index.grams(6)
 hits=[i for i in range(max(0,len(words)-5)) if tuple(words[i:i+6]) in six]
 # A short, fixed text budget prevents huge logits masks. Full-source checking
 # still catches unmasked spelling/case variants and any new source copying.
 selected=[]
 for i in sorted(hits,key=lambda i:(abs(i-len(words)/2),i)):
  value=normalized[positions[i][0]:positions[i+5][1]]
  if value not in selected:selected.append(value)
  if len(selected)==6:break
 return selected

def bounded_bans(tokenizer,base):
 selected=[];total=0
 for phrase in base:
  for form in dict.fromkeys([phrase,phrase[0].upper()+phrase[1:]]):
   # vLLM may encode both leading-space variants. Count both as an upper bound.
   cost=sum(len(tokenizer.encode(prefix+form,add_special_tokens=False)) for prefix in ['', ' '])
   if total+cost<=256 and form not in selected:selected.append(form);total+=cost
 return selected,total

def finalize(model,round_count,rows,windows,current):
 condition=f'hybrid_{model}_r{round_count}';records=[];final_edits=[];tick=time.monotonic()
 for row in rows:
  index=SourceIndex(row['fulltext']);ws=[w for w in windows if w['uid']==row['uid']]
  texts=[current[w['pair_key']] for w in ws];changed=[a!=w['text'] for w,a in zip(ws,texts)]
  original=audit_summary(row['fulltext'],row['safe_summary'],index=index,expanded_diagnostic=False)
  rollback=False;error=None
  try:
   summary=assemble(row['safe_summary'],ws,texts,accepted=changed);context=narrative_context(summary)
   assert check(row['safe_context'],context)['passed'],'Document number/unit/formula signature changed'
   assert check(json.dumps(row['safe_summary'],ensure_ascii=False),json.dumps(summary,ensure_ascii=False))['passed'],'Serialized signature changed'
   assert [(t,p,c) for t,p,c in fragments(summary) if c!='narrative']==[(t,p,c) for t,p,c in fragments(row['safe_summary']) if c!='narrative'],'Evidence/metadata changed'
   try:validate_summary(json.dumps(row['safe_summary'],ensure_ascii=False),row['fulltext']);original_valid=True
   except (ValueError,TypeError,KeyError):original_valid=False
   if original_valid:validate_summary(json.dumps(summary,ensure_ascii=False),row['fulltext'])
   audit=audit_summary(row['fulltext'],summary,index=index,expanded_diagnostic=False)
   old=original['categories']['narrative'];new=audit['categories']['narrative']
   assert new['longest_contiguous_match_words']<=old['longest_contiguous_match_words'],'Assembled longest match increased'
   assert new['covered_words_by_minimum_run']['6']<=old['covered_words_by_minimum_run']['6'],'Assembled copied word count increased'
  except (AssertionError,ValueError,TypeError,KeyError) as e:
   summary=row['safe_summary'];context=row['safe_context'];audit=original;rollback=True;error=repr(e)
  lengths=[n for text,path,cat in fragments(summary) if cat=='narrative' for n in nonoverlapping_passages(index,text)]
  record=dict(uid=row['uid'],document_id=row['document_id'],source_sha256=row['source_sha256'],condition=condition,summary=summary,judge_context=context,status='generated',audit=audit,
   passage_lengths=lengths,rolled_back=rollback,error=error,changed_from_safe=context!=row['safe_context'],critical=check(row['safe_context'],context))
  write(condition_file(condition,row['uid']),record);records.append(record)
  if not rollback:
   for w,after in zip(ws,texts):
    if after!=w['text']:final_edits.append(dict(uid=row['uid'],window_index=w['window_index'],path=w['path'],condition=condition,before=w['text'],after=after,
     pair_id=digest(w['text']+'\0'+after),unit='final accepted source-copy window; up to two adjacent sentence spans'))
 jsonl(ROOT/f'outputs/final_edits/{condition}.jsonl',final_edits)
 lengths=[n for r in records for n in r['passage_lengths']]
 write(ROOT/f'outputs/{condition}_complete.json',dict(complete=True,condition=condition,papers=97,mean_coverage=statistics.mean(r['audit']['categories']['narrative']['covered_word_fraction_by_minimum_run']['6'] for r in records),
  strict_passes=sum(r['audit']['categories']['narrative']['strict_five_word_limit_passed'] for r in records),narrative_passages_6plus=len(lengths),mean_passages_per_summary=len(lengths)/97,
  mean_passage_length_words=statistics.mean(lengths) if lengths else 0,changed_from_safe=sum(r['changed_from_safe'] for r in records),rollbacks=sum(r['rolled_back'] for r in records),
  final_edited_windows=len(final_edits),finalization_seconds=time.monotonic()-tick))

def main(model):
 if (ROOT/f'outputs/hybrid_{model}_r3_complete.json').exists():return
 from transformers import AutoTokenizer
 from vllm import LLM,SamplingParams
 from nli import NLI
 setup_start=time.monotonic();tokenizer=AutoTokenizer.from_pretrained(MODELS[model],local_files_only=True)
 nli=NLI()
 engine=LLM(model=str(MODELS[model]),language_model_only=True,dtype='bfloat16',max_model_len=4096,max_num_seqs=256,max_num_batched_tokens=16384,
  gpu_memory_utilization=.75,enable_prefix_caching=True,enable_chunked_prefill=True,generation_config='vllm',disable_log_stats=True,seed=20261023)
 setup_seconds=time.monotonic()-setup_start
 if model=='qwen08':
  controls=[('The result was not statistically significant.','The result was statistically significant.'),
   ('The mass was 5 mg.','The mass was 5 g.'),('The treatment reduced mortality.','Mortality decreased following the treatment.')]
  scores,control_perf=nli.score([p for a,b in controls for p in [(a,b),(b,a)]])
  write(ROOT/'outputs/nli_control_examples.json',dict(cutoff=.95,performance=control_perf,cases=[dict(before=a,after=b,critical=check(a,b),
   scores=scores[2*i:2*i+2],nli_accepts=all(s is not None and s['entailment']>=.95 for s in scores[2*i:2*i+2])) for i,(a,b) in enumerate(controls)]))
 rows=cohort();windows=read_jsonl(ROOT/'inputs/windows.jsonl');indexes={r['uid']:SourceIndex(r['fulltext']) for r in rows}
 current={w['pair_key']:w['text'] for w in windows};performance=[];next_round=0
 checkpoint=ROOT/f'outputs/checkpoints/{model}.json'
 if checkpoint.exists():
  state=load(checkpoint);current=state['current'];performance=state['performance'];next_round=state['next_round']
 # One bounded real-input smoke batch validates both the model and masks before
 # the full batch. Its answers are retained for round one, never regenerated.
 for attempt in range(next_round,3):
  round_start=time.monotonic();requests=[];params=[];candidates=[];excluded=Counter()
  for w in windows:
   text=current[w['pair_key']];index=indexes[w['uid']];before=index.fragment(text,w['path'],'narrative')
   if before['longest_contiguous_match_words']<=5:continue
   copied=phrases(index,text);banned,token_budget=bounded_bans(tokenizer,copied)
   user=('Rewrite TEXT with different sentence structure while preserving its complete meaning. '
    'Do not use punctuation tricks, artificial spelling, or meaningless additions to break a match.\n'
    'CONTEXT (do not rewrite): '+w['preceding_context']+'\n'
    'COPIED WORD SEQUENCES TO REPHRASE: '+json.dumps(copied,ensure_ascii=False)+'\n'
    'TEXT: '+text)
   ids=tokenizer.apply_chat_template([dict(role='system',content=SYSTEM),dict(role='user',content=user)],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
   assert isinstance(ids,list) and ids and all(isinstance(t,int) for t in ids)
   if len(ids)+384>4096:excluded['input_overlength_without_truncation']+=1;continue
   seed=20261023+1009*attempt+int(w['pair_key'][:6],16)
   sp=SamplingParams(temperature=[.35,.65,.95][attempt],top_p=.95,top_k=64,max_tokens=384,seed=seed,bad_words=banned)
   requests.append(dict(prompt_token_ids=ids));params.append(sp)
   candidates.append(dict(w,current=text,attempt=attempt,prompt_tokens=len(ids),banned_phrases=banned,ban_token_upper_bound=token_budget,overlap_before=before))
  order=sorted(range(len(requests)),key=lambda i:len(requests[i]['prompt_token_ids']))
  generation_start=time.monotonic();calls=[]
  if attempt==0 and order:
   smoke=order[:8];tick=time.monotonic();out=engine.generate([requests[i] for i in smoke],[params[i] for i in smoke],use_tqdm=False)
   calls.extend(zip(smoke,out));order=order[8:]
   write(ROOT/f'outputs/{model}_smoke.json',dict(passed=True,examples=8,seconds=time.monotonic()-tick,max_mask_token_upper_bound=max(candidates[i]['ban_token_upper_bound'] for i in smoke)))
  if order:
   out=engine.generate([requests[i] for i in order],[params[i] for i in order],use_tqdm=False);calls.extend(zip(order,out))
  generation_seconds=time.monotonic()-generation_start;eligible=[]
  assess_start=time.monotonic()
  for i,out in calls:
   r=candidates[i];o=out.outputs[0];after=o.text.strip();r.update(candidate=after,output_tokens=len(o.token_ids),finish_reason=o.finish_reason,reasons=[],accepted=False)
   if o.finish_reason!='stop' or not after:r['reasons'].append('unfinished_or_empty')
   if any(t in after for t in ['<think>','</think>','<|channel>','<channel|>']) or after.lstrip().startswith(('{','```')):r['reasons'].append('not_plain_final_prose')
   original_words=len(tokenize(r['text'])[1]);after_words=len(tokenize(after)[1])
   if not .75*max(1,original_words)<=after_words<=1.35*max(1,original_words):r['reasons'].append('length_ratio')
   r['critical']=check(r['text'],after)
   if not r['critical']['passed']:r['reasons'].append('critical_values')
   if named_values(r['text'])!=named_values(after):r['reasons'].append('acronym_or_chemical_identifier_change')
   if negations(r['text'])!=negations(after):r['reasons'].append('negation_count_changed')
   new=indexes[r['uid']].fragment(after,r['path'],'narrative');r['overlap_after']=new
   if new['covered_words_by_minimum_run']['6']>=r['overlap_before']['covered_words_by_minimum_run']['6'] or new['longest_contiguous_match_words']>r['overlap_before']['longest_contiguous_match_words']:r['reasons'].append('no_safe_absolute_copy_reduction')
   if not r['reasons']:eligible.append(r)
  scores,nli_perf=nli.score([pair for r in eligible for pair in [(r['text'],r['candidate']),(r['candidate'],r['text'])]])
  for i,r in enumerate(eligible):
   a,b=scores[2*i:2*i+2];r.update(nli_forward=a,nli_backward=b)
   if a is None or b is None:r['reasons'].append('nli_overlength_without_truncation')
   elif min(a['entailment'],b['entailment'])<.95:r['reasons'].append('bidirectional_entailment_below_0.95')
   else:r['accepted']=True;current[r['pair_key']]=r['candidate']
  assess_seconds=time.monotonic()-assess_start
  rejection=Counter(v for r in candidates for v in r.get('reasons',[]));rejection.update(excluded)
  jsonl(ROOT/f'traces/{model}/round-{attempt}.jsonl',candidates)
  performance.append(dict(round=attempt+1,requests=len(candidates),accepted=sum(r.get('accepted',False) for r in candidates),generation_seconds=generation_seconds,
   assess_seconds=assess_seconds,nli=nli_perf,input_tokens=sum(r['prompt_tokens'] for r in candidates),output_tokens=sum(r.get('output_tokens',0) for r in candidates),
   wall_seconds=time.monotonic()-round_start,rejections=dict(rejection)))
  if attempt in [0,2]:
   finalize(model,attempt+1,rows,windows,current)
   write(ROOT/f'outputs/hybrid_{model}_r{attempt+1}_performance.json',dict(complete=True,setup_seconds=setup_seconds,rounds=performance,
    warm_gpu_worker_seconds=sum(r['wall_seconds'] for r in performance),generation_seconds=sum(r['generation_seconds'] for r in performance),nli_seconds=sum(r['nli']['seconds'] for r in performance),
    max_concurrent_sequences=256,max_batched_tokens=16384,all_per_request_mask_tokens_at_most=256,scope='Warm single-GH200 repair worker, including prompt preparation, generation and candidate guards; final assembly/source audits and one-time model setup recorded separately.'))
  write(checkpoint,dict(next_round=attempt+1,current=current,performance=performance))
  print('REPAIR_ROUND_DONE',model,attempt+1,performance[-1]['requests'],performance[-1]['accepted'],flush=True)
 write(ROOT/f'outputs/{model}_worker_complete.json',dict(complete=True,model=model,setup_seconds=setup_seconds))

if __name__=='__main__':
 import sys
 main(sys.argv[1])
