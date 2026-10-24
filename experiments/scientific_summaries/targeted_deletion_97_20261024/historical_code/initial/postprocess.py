"""Freeze all deletion outputs before running the original QA slots."""
import concurrent.futures,json,os,time,statistics
from collections import Counter
from common import ROOT,CONDITIONS,cohort,load,write,jsonl,digest,condition_file,nonoverlapping_passages
from ngram_overlap import SourceIndex,fragments,audit_summary
from passages import patch
from critical_values import check
from reference_validator import narrative_context,validate_summary
from lexicon import immutable
import deletion

PARENT=ROOT.parent/'scientific-post-gemma-structural-20261024'


def process(row):
    tick=time.process_time();index=SourceIndex(row['fulltext']);index_cpu=time.process_time()-tick
    assert index.sha256==row['source_sha256']
    base=load(PARENT/f'outputs/repaired/lexicon_v2/{row["uid"]}.json')
    baseline=dict(base,condition='baseline')
    write(condition_file('baseline',row['uid']),baseline)
    values=[]
    for level,c in enumerate(CONDITIONS[1:]):
        tick=time.process_time();traces=[];changes={};rollback=False;error=None
        for text,path,cat in fragments(base['summary']):
            if cat!='narrative':continue
            after,edits=deletion.repair(text,index,level)
            if after!=text:
                changes[path,text]=after
                traces.append(dict(path=path,before=text,after=after,edits=edits))
        try:
            summary=patch(base['summary'],changes);context=narrative_context(summary)
            assert immutable(base['judge_context'],context),'Scientific value signature changed'
            assert check(json.dumps(base['summary'],ensure_ascii=False),json.dumps(summary,ensure_ascii=False))['passed']
            nonprose=lambda s:[(t,p,k) for t,p,k in fragments(s) if k!='narrative']
            assert nonprose(base['summary'])==nonprose(summary),'Evidence or metadata changed'
            try:validate_summary(json.dumps(base['summary'],ensure_ascii=False),row['fulltext']);valid=True
            except (ValueError,TypeError,KeyError):valid=False
            if valid:validate_summary(json.dumps(summary,ensure_ascii=False),row['fulltext'])
            audit=audit_summary(row['fulltext'],summary,index=index,expanded_diagnostic=False)
            assert audit['categories']['narrative']['longest_contiguous_match_words']<=base['audit']['categories']['narrative']['longest_contiguous_match_words']
        except (AssertionError,ValueError,TypeError,KeyError) as e:
            summary=base['summary'];context=base['judge_context'];audit=base['audit'];rollback=True;error=repr(e)
        pairs=[]
        if not rollback:
            for trace in traces:
                before=deletion.splitter.tokenize(trace['before']);after=deletion.splitter.tokenize(trace['after'])
                if len(before)!=len(after):before=[trace['before']];after=[trace['after']]
                for a,b in zip(before,after):
                    if a!=b:pairs.append(dict(uid=row['uid'],document_id=row['document_id'],condition=c,path=trace['path'],
                        before=a,after=b,pair_id=digest(a+'\0'+b)))
        record=dict(uid=row['uid'],document_id=row['document_id'],condition=c,status='generated',source_sha256=index.sha256,
            summary=summary,judge_context=context,audit=audit,rolled_back=rollback,error=error)
        write(condition_file(c,row['uid']),record);jsonl(ROOT/f'traces/{c}/{row["uid"]}.jsonl',traces)
        narrative=audit['categories']['narrative']
        lengths=[n for t,p,k in fragments(summary) if k=='narrative' for n in nonoverlapping_passages(index,t)]
        values.append(dict(condition=c,uid=row['uid'],words=narrative['words'],covered=narrative['covered_words_by_minimum_run']['6'],
            coverage=narrative['covered_word_fraction_by_minimum_run']['6'],strict=narrative['strict_five_word_limit_passed'],
            passage_lengths=lengths,changed=context!=base['judge_context'],rollbacks=rollback,
            deleted_words=base['audit']['categories']['narrative']['words']-narrative['words'],pairs=pairs,
            rules=dict(Counter(e['rule'] for t in traces for e in t['edits'])) if not rollback else {},
            cpu_seconds=time.process_time()-tick,index_cpu_seconds=index_cpu))
    return values


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Full audit must run on compute'
    if (ROOT/'outputs/cpu_complete.json').exists():return
    start=time.monotonic();rows=cohort()
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,initializer=deletion.init) as pool:
        results=[]
        for i,group in enumerate(pool.map(process,rows,chunksize=1),1):
            results.extend(group)
            if i%10==0:print('CPU_PAPERS',i,flush=True)
    pairs=[]
    for c in CONDITIONS[1:]:
        records=[r for r in results if r['condition']==c];pairs.extend(p for r in records for p in r['pairs'])
        counts=Counter()
        for r in records:counts.update(r['rules'])
        write(ROOT/f'outputs/{c}_complete.json',dict(complete=True,papers=97,mean_coverage=statistics.mean(r['coverage'] for r in records),
            pooled_coverage=sum(r['covered'] for r in records)/sum(r['words'] for r in records),
            strict_passes=sum(r['strict'] for r in records),narrative_passages_6plus=sum(len(r['passage_lengths']) for r in records),
            changed_summaries=sum(r['changed'] for r in records),rollbacks=sum(r['rollbacks'] for r in records),
            final_edited_windows=sum(len(r['pairs']) for r in records),deleted_words=sum(r['deleted_words'] for r in records),rule_counts=dict(counts),
            cpu_seconds_including_index=sum(r['cpu_seconds']+r['index_cpu_seconds'] for r in records),shared_wall_seconds=time.monotonic()-start))
    jsonl(ROOT/'outputs/per_summary.jsonl',[{k:v for k,v in r.items() if k!='pairs'} for r in results])
    jsonl(ROOT/'inputs/post_review.jsonl',pairs)
    write(ROOT/'outputs/cpu_complete.json',dict(complete=True,conditions=4,papers=97,review_cards=len(pairs),seconds=time.monotonic()-start))
    print('DELETION_CPU_COMPLETE',len(pairs),flush=True)


if __name__=='__main__':main()
