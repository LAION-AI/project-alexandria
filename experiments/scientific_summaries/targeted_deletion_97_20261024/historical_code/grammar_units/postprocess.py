"""Five frozen arms; same 97 full sources, no QA-informed output selection."""
import concurrent.futures,json,os,time,statistics
from collections import Counter
from common import ROOT,CONDITIONS,cohort,load,write,jsonl,digest,condition_file,nonoverlapping_passages
from ngram_overlap import SourceIndex,fragments,audit_summary
from passages import patch
from reference_validator import narrative_context,validate_summary
import number_style as numeric
import grammar_units as editor

ARTICLE_PARENT=ROOT.parent/'scientific-post-gemma-article-caseguard-97-20261024'
NUMBER_PARENT=ROOT.parent/'scientific-post-gemma-number-style-guarded-97-20261024'

def protected(s):return [(t,p,k) for t,p,k in fragments(s) if k!='narrative']

def process(row):
    tick=time.process_time();index=SourceIndex(row['fulltext']);index_cpu=time.process_time()-tick
    assert index.sha256==row['source_sha256']
    article=load(ARTICLE_PARENT/f'outputs/repaired/delete_articles_case_guard/{row["uid"]}.json')
    previous=load(NUMBER_PARENT/f'outputs/repaired/numbers_percent_guarded/{row["uid"]}.json')
    write(condition_file('baseline',row['uid']),dict(previous,condition='baseline'))
    tick=time.process_time();number_changes={};number_traces=[]
    for text,path,cat in fragments(article['summary']):
        if cat!='narrative':continue
        after,edits=numeric.repair(text,index,'numbers_percent_guarded')
        assert numeric.verify_trace(text,after,edits)
        if after!=text:
            number_changes[path,text]=after;number_traces.append(dict(path=path,before=text,after=after,edits=edits))
    identifier_summary=patch(article['summary'],number_changes)
    identifier_context=narrative_context(identifier_summary)
    assert numeric.invariant_non_numeric(article['judge_context'],identifier_context)
    assert protected(article['summary'])==protected(identifier_summary)
    numeric_cpu=time.process_time()-tick
    jsonl(ROOT/f'traces/identifier_numeric/{row["uid"]}.jsonl',number_traces)
    values=[]
    for c in CONDITIONS[1:]:
        tick=time.process_time();traces=[];changes={};rollback=False;error=None
        if c!='identifier_guard':
            for text,path,cat in fragments(identifier_summary):
                if cat!='narrative':continue
                after,edits=editor.repair(text,index,c)
                assert editor.verify_trace(text,after,edits)
                if after!=text:
                    changes[path,text]=after;traces.append(dict(path=path,before=text,after=after,edits=edits))
        try:
            summary=patch(identifier_summary,changes);context=narrative_context(summary)
            assert editor.invariant(identifier_context,context)
            assert protected(previous['summary'])==protected(summary)
            try:validate_summary(json.dumps(previous['summary'],ensure_ascii=False),row['fulltext']);valid=True
            except (ValueError,TypeError,KeyError):valid=False
            if valid:validate_summary(json.dumps(summary,ensure_ascii=False),row['fulltext'])
            audit=audit_summary(row['fulltext'],summary,index=index,expanded_diagnostic=False)
            assert audit['categories']['narrative']['longest_contiguous_match_words']<=article['audit']['categories']['narrative']['longest_contiguous_match_words']
        except (AssertionError,ValueError,TypeError,KeyError) as e:
            summary=previous['summary'];context=previous['judge_context'];audit=previous['audit'];rollback=True;error=repr(e)
        # Review every visible final change versus the previous numeric pipeline,
        # including identifier restorations, never just successful proposals.
        # A citation label and its justification can have the same displayed
        # path. Preserve occurrence order; a dict keyed by path loses one.
        old_frag=list(fragments(previous['summary']))
        new_frag=list(fragments(summary))
        assert len(old_frag)==len(new_frag)
        pairs=[]
        for (before_text,old_path,old_cat),(text,path,cat) in zip(old_frag,new_frag):
            assert (old_path,old_cat)==(path,cat)
            if cat!='narrative':continue
            if before_text==text:continue
            before=editor.splitter.tokenize(before_text);after=editor.splitter.tokenize(text)
            if len(before)!=len(after):before=[before_text];after=[text]
            for a,b in zip(before,after):
                if a!=b:pairs.append(dict(uid=row['uid'],document_id=row['document_id'],condition=c,path=path,before=a,after=b,pair_id=digest(a+'\0'+b)))
        record=dict(uid=row['uid'],document_id=row['document_id'],condition=c,status='generated',source_sha256=index.sha256,
            summary=summary,judge_context=context,audit=audit,rolled_back=rollback,error=error,
            numeric_preservation_mode='Recomputed from guarded articles: exact Decimal value/sign/precision; acronym-adjacent IDs protected; exact unit aliases and narrowly optional reporting-that only. Per-stage provenance retained.')
        write(condition_file(c,row['uid']),record);jsonl(ROOT/f'traces/{c}/{row["uid"]}.jsonl',traces)
        n=audit['categories']['narrative'];base_n=previous['audit']['categories']['narrative']
        lengths=[x for t,p,k in fragments(summary) if k=='narrative' for x in nonoverlapping_passages(index,t)]
        values.append(dict(condition=c,uid=row['uid'],words=n['words'],covered=n['covered_words_by_minimum_run']['6'],coverage=n['covered_word_fraction_by_minimum_run']['6'],
            baseline_words=base_n['words'],baseline_covered=base_n['covered_words_by_minimum_run']['6'],strict=n['strict_five_word_limit_passed'],passage_lengths=lengths,
            changed=context!=previous['judge_context'],rollback=rollback,pairs=pairs,
            rules=dict(Counter(e['rule'] for t in traces for e in t['edits'])) if not rollback else {},
            numeric_roundtrips=sum(len(t['edits']) for t in number_traces),numeric_cpu_seconds=numeric_cpu,
            cpu_seconds=time.process_time()-tick,index_cpu_seconds=index_cpu))
    return values


def main():
    assert os.environ.get('SLURM_JOB_ID')
    if (ROOT/'outputs/cpu_complete.json').exists():return
    start=time.monotonic();results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,initializer=editor.init) as pool:
        for i,group in enumerate(pool.map(process,cohort(),chunksize=1),1):
            results.extend(group)
            if i%10==0:print('GRAMMAR_UNIT_PAPERS',i,flush=True)
    pairs=[]
    for c in CONDITIONS[1:]:
        records=[r for r in results if r['condition']==c];pairs.extend(p for r in records for p in r['pairs']);counts=Counter()
        for r in records:counts.update(r['rules'])
        write(ROOT/f'outputs/{c}_complete.json',dict(complete=True,papers=97,mean_coverage=statistics.mean(r['coverage'] for r in records),
            pooled_coverage=sum(r['covered'] for r in records)/sum(r['words'] for r in records),
            strict_passes=sum(r['strict'] for r in records),narrative_passages_6plus=sum(len(r['passage_lengths']) for r in records),
            changed_summaries=sum(r['changed'] for r in records),rollbacks=sum(r['rollback'] for r in records),final_edited_windows=sum(len(r['pairs']) for r in records),
            words=sum(r['words'] for r in records),baseline_words=sum(r['baseline_words'] for r in records),absolute_copied_words=sum(r['covered'] for r in records),baseline_copied_words=sum(r['baseline_covered'] for r in records),
            coverage_with_fixed_original_denominator=statistics.mean(r['covered']/r['baseline_words'] for r in records),rule_counts=dict(counts),numeric_roundtrips=sum(r['numeric_roundtrips'] for r in records),
            cpu_seconds_including_index=sum(r['cpu_seconds']+r['numeric_cpu_seconds']+r['index_cpu_seconds'] for r in records),
            cpu_seconds_grammar_unit_stage=sum(r['cpu_seconds'] for r in records),shared_wall_seconds=time.monotonic()-start))
    jsonl(ROOT/'outputs/per_summary.jsonl',[{k:v for k,v in r.items() if k!='pairs'} for r in results]);jsonl(ROOT/'inputs/post_review.jsonl',pairs)
    write(ROOT/'outputs/cpu_complete.json',dict(complete=True,papers=97,conditions=len(CONDITIONS)-1,review_cards=len(pairs),seconds=time.monotonic()-start))
    print('GRAMMAR_UNITS_CPU_COMPLETE',len(pairs),flush=True)


if __name__=='__main__':main()
