"""Shared inputs, frozen before QA; no gold answers are read by this stage."""
import os,time,statistics
from common import ROOT,cohort,write,jsonl,condition_file,nonoverlapping_passages,named_values,digest
from ngram_overlap import SourceIndex,audit_summary,fragments
from passages import select

def main():
 assert os.environ.get('SLURM_JOB_ID'),'Full-source work belongs on compute'
 if (ROOT/'outputs/preparation_complete.json').exists():return
 tick=time.monotonic();rows=cohort();windows=[];summaries={c:[] for c in ['raw','safe_rules']}
 for row in rows:
  index=SourceIndex(row['fulltext']);assert row['source_sha256']==index.sha256
  for c,prefix in [('raw','raw'),('safe_rules','safe')]:
   summary=row[prefix+'_summary'];context=row[prefix+'_context'];audit=audit_summary(row['fulltext'],summary,index=index,expanded_diagnostic=False)
   lengths=[n for text,path,cat in fragments(summary) if cat=='narrative' for n in nonoverlapping_passages(index,text)]
   record=dict(uid=row['uid'],document_id=row['document_id'],source_sha256=row['source_sha256'],condition=c,summary=summary,judge_context=context,
    status='generated',audit=audit,passage_lengths=lengths,rolled_back=False,changed_from_raw=context!=row['raw_context'])
   write(condition_file(c,row['uid']),record);summaries[c].append(record)
  for i,w in enumerate(select(row['safe_summary'],row['fulltext'])):
   w.update(uid=row['uid'],document_id=row['document_id'],window_index=i,protected_names=sorted(named_values(w['text'])),
    pair_key=digest(row['uid']+str(i)+w['text']))
   windows.append(w)
 jsonl(ROOT/'inputs/windows.jsonl',windows)
 for c,records in summaries.items():
  lengths=[n for r in records for n in r['passage_lengths']]
  write(ROOT/f'outputs/{c}_complete.json',dict(complete=True,papers=97,condition=c,mean_coverage=statistics.mean(r['audit']['categories']['narrative']['covered_word_fraction_by_minimum_run']['6'] for r in records),
   strict_passes=sum(r['audit']['categories']['narrative']['strict_five_word_limit_passed'] for r in records),narrative_passages_6plus=len(lengths),mean_passages_per_summary=len(lengths)/97,
   mean_passage_length_words=statistics.mean(lengths),changed_from_raw=sum(r['changed_from_raw'] for r in records),rollbacks=0))
 write(ROOT/'outputs/preparation_complete.json',dict(complete=True,papers=97,windows=len(windows),seconds=time.monotonic()-tick))
 print('PREPARATION_COMPLETE',len(windows),flush=True)
if __name__=='__main__':main()
