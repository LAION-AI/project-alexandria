"""CPU-only post-Gemma repair; all original papers remain in every condition."""
import concurrent.futures
import json
import os
import statistics
import time
from collections import Counter
from common import ROOT, CONDITIONS, cohort, condition_file, load, write, jsonl, digest, nonoverlapping_passages
from ngram_overlap import SourceIndex, fragments, audit_summary
from passages import patch, sentence_spans
from critical_values import check
from reference_validator import narrative_context, validate_summary
from lexicon import repair, immutable

OLD=ROOT.parent/'scientific-hybrid-qwen08-gemma-e2b-97-20261023'


def init():
    global banks
    from function_words import RULES
    safe=[dict(id='safe_'+name,pattern=pattern,replacement=replacement,blocked_context_regex=[])
          for name,pattern,replacement in RULES['function_prepositions']]
    banks={'safe_again':safe,'lexicon_v1':safe+load(ROOT/'inputs/learned_rules_v1.json')['rules'],
           'lexicon_v2':safe+load(ROOT/'inputs/learned_rules_v2.json')['rules']}


def process(row):
    common_start=time.process_time()
    index=SourceIndex(row['fulltext'])
    assert index.sha256==row['source_sha256']
    base=load(OLD/f'outputs/repaired/hybrid_gemma_e2b_r1/{row["uid"]}.json')
    common_cpu=time.process_time()-common_start
    results=[]
    for condition,rules in banks.items():
        tick=time.process_time();wall=time.monotonic();replacements={};traces=[]
        for text,path,category in fragments(base['summary']):
            if category!='narrative':continue
            value,edits=repair(text,index,rules)
            if value!=text:
                replacements[path,text]=value
                traces.append(dict(path=path,before=text,after=value,edits=edits))
        rollback=False;error=None
        try:
            summary=patch(base['summary'],replacements);context=narrative_context(summary)
            assert immutable(base['judge_context'],context),'Document immutable values'
            assert check(json.dumps(base['summary'],ensure_ascii=False),json.dumps(summary,ensure_ascii=False))['passed'],'Serialized signature'
            assert [(t,p,c) for t,p,c in fragments(base['summary']) if c!='narrative']==[(t,p,c) for t,p,c in fragments(summary) if c!='narrative'],'Protected evidence/metadata'
            try:validate_summary(json.dumps(base['summary'],ensure_ascii=False),row['fulltext']);valid=True
            except (ValueError,TypeError,KeyError):valid=False
            if valid:validate_summary(json.dumps(summary,ensure_ascii=False),row['fulltext'])
            audit=audit_summary(row['fulltext'],summary,index=index,expanded_diagnostic=False)
            assert audit['categories']['narrative']['longest_contiguous_match_words']<=base['audit']['categories']['narrative']['longest_contiguous_match_words']
        except (AssertionError,ValueError,KeyError,TypeError) as e:
            rollback=True;error=repr(e);summary=base['summary'];context=base['judge_context'];audit=base['audit']
        pairs=[]
        if not rollback:
            for trace in traces:
                before=[trace['before'][a:b] for a,b in sentence_spans(trace['before'],max_words=10000)]
                after=[trace['after'][a:b] for a,b in sentence_spans(trace['after'],max_words=10000)]
                if len(before)!=len(after):before=[trace['before']];after=[trace['after']]
                for a,b in zip(before,after):
                    if a!=b:pairs.append(dict(uid=row['uid'],document_id=row['document_id'],condition=condition,path=trace['path'],before=a,after=b,pair_id=digest(a+'\0'+b)))
        lengths=[n for t,p,c in fragments(summary) if c=='narrative' for n in nonoverlapping_passages(index,t)]
        record=dict(uid=row['uid'],document_id=row['document_id'],source_sha256=index.sha256,condition=condition,summary=summary,
                    judge_context=context,audit=audit,status='generated',rolled_back=rollback,error=error)
        write(condition_file(condition,row['uid']),record)
        jsonl(ROOT/f'traces/{condition}/{row["uid"]}.jsonl',traces)
        results.append(dict(condition=condition,document_id=row['document_id'],coverage=audit['categories']['narrative']['covered_word_fraction_by_minimum_run']['6'],
            strict=audit['categories']['narrative']['strict_five_word_limit_passed'],passage_lengths=lengths,rollback=rollback,pairs=pairs,
            rules=dict(Counter(e['rule'] for t in traces for e in t['edits'])) if not rollback else {},
            cpu_seconds=time.process_time()-tick,common_index_cpu_seconds=common_cpu,wall_seconds=time.monotonic()-wall,changed=context!=base['judge_context']))
    return results


def main():
    assert os.environ.get('SLURM_JOB_ID')
    if (ROOT/'inputs/post_review.jsonl').exists():return
    started=time.monotonic();rows=cohort()
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,initializer=init) as pool:
        results=[item for group in pool.map(process,rows,chunksize=2) for item in group]
    pairs=[]
    for condition in CONDITIONS[1:]:
        records=[r for r in results if r['condition']==condition]
        pairs.extend(p for r in records for p in r['pairs'])
        lengths=[n for r in records for n in r['passage_lengths']]
        counts=Counter()
        for r in records:counts.update(r['rules'])
        write(ROOT/f'outputs/{condition}_complete.json',dict(complete=True,papers=97,condition=condition,mean_coverage=statistics.mean(r['coverage'] for r in records),
            strict_passes=sum(r['strict'] for r in records),narrative_passages_6plus=len(lengths),changed_summaries=sum(r['changed'] for r in records),
            rollbacks=sum(r['rollback'] for r in records),final_edited_windows=sum(len(r['pairs']) for r in records),rule_counts=dict(counts),
            cpu_seconds_including_index=sum(r['cpu_seconds']+r['common_index_cpu_seconds'] for r in records),
            cpu_seconds_repair_and_audit=sum(r['cpu_seconds'] for r in records),shared_wall_seconds=time.monotonic()-started))
        print('POSTPROCESS_COMPLETE',condition,len(lengths),len([p for p in pairs if p['condition']==condition]),flush=True)
    jsonl(ROOT/'inputs/post_review.jsonl',pairs)


if __name__=='__main__':main()
