# SPDX-License-Identifier: CC-BY-4.0
# Copyright 2026 LAION / Project Alexandria contributors.
import concurrent.futures,threading,time
from common import ROOT,OLD,Backend,digest,load,sources,write
def path(cohort,condition,docid):return ROOT/"outputs/generation"/condition/(docid+".json")
from project_alexandria.experiments.mcq import historical_answer_prompt,extract_historical_choice
SYSTEM='You are a very smart very intelligence assistant who is very helpful.'

def score(p,questions,cohort,condition,backend):
 dest=ROOT/'outputs/qa'/cohort/condition/(p['document_id']+'.json')
 if dest.exists():return
 if condition=='no_context':context='';failed=False
 elif condition=='original':context=p['fulltext'];failed=False
 else:
  row=load(path(cohort,condition,p['document_id']));assert row['source_sha256']==p['fulltext_sha256'];context=row['judge_context'];failed=row['status']!='generated'
 def answer(q):
  backend.local.document_id=p['document_id'];backend.local.condition=condition
  record=dict(question_index=q['question_index'],gold=q['answer'],prediction=None,attempts=[],generation_failed=failed)
  if failed:return record
  prompt=historical_answer_prompt(q['formatted_question'],context);record['prompt_sha256']=digest(prompt)
  for attempt in range(5):
   try:
    text=backend.generate(SYSTEM,prompt,100);record['attempts'].append(dict(backend.local.last));record['prediction']=extract_historical_choice(text)
    if record['prediction'] is not None:break
   except ValueError as e:record['error']=str(e);break
  return record
 tick=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(answer,questions))
 assert len(rows)==10
 write(dest,dict(document_id=p['document_id'],source_sha256=p['fulltext_sha256'],condition=condition,cohort=cohort,rows=rows,context_sha256=digest(context),elapsed_seconds=time.monotonic()-tick))
 print('QA_COMPLETE',cohort,condition,p['document_id'],sum(r['prediction']==r['gold'] for r in rows),flush=True)
