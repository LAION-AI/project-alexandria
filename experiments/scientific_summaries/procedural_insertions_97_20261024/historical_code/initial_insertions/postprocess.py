import concurrent.futures,json,os,time,statistics
from collections import Counter
from common import ROOT,CONDITIONS,cohort,load,write,jsonl,digest,condition_file,nonoverlapping_passages
from ngram_overlap import SourceIndex,fragments,audit_summary,tokenize
from passages import patch
from reference_validator import narrative_context,validate_summary
import insertions as editor

PARENT=ROOT.parent/'scientific-post-gemma-grammar-units-97-20261024'
def protected(s):return [(t,p,k) for t,p,k in fragments(s) if k!='narrative']

def origin_coverage(before,after,edits,index):
    origins=list(range(len(tokenize(before)[1])));current=before
    for e in edits:
        assert current==e['before'];k=len(tokenize(current[:e['start']])[1]);n=len(tokenize(e['replacement'])[1])
        origins=origins[:k]+[None]*n+origins[k:];current=e['after']
    assert current==after
    words=tokenize(after)[1];assert len(words)==len(origins)
    assert [words[i] for i,x in enumerate(origins) if x is not None]==tokenize(before)[1]
    _,covered=editor.covered_positions(words,index)
    return sum(origins[i] is not None for i in covered)

def process(row):
    tick=time.process_time();index=SourceIndex(row['fulltext']);index_cpu=time.process_time()-tick
    assert index.sha256==row['source_sha256']
    base=load(PARENT/f'outputs/repaired/combined/{row["uid"]}.json')
    write(condition_file('baseline',row['uid']),dict(base,condition='baseline'))
    values=[]
    for c in CONDITIONS[1:]:
        tick=time.process_time();traces=[];changes={};rollback=False;error=None;original_covered=0
        for text,path,cat in fragments(base['summary']):
            if cat!='narrative':continue
            after,edits=editor.repair(text,index,c);assert editor.verify_trace(text,after,edits)
            original_covered+=origin_coverage(text,after,edits,index)
            if after!=text:
                changes[path,text]=after;traces.append(dict(path=path,before=text,after=after,edits=edits))
        try:
            summary=patch(base['summary'],changes);context=narrative_context(summary)
            assert editor.guard(base['judge_context'],context)
            assert protected(base['summary'])==protected(summary)
            old=list(fragments(base['summary']));new=list(fragments(summary));assert len(old)==len(new)
            try:validate_summary(json.dumps(base['summary'],ensure_ascii=False),row['fulltext']);valid=True
            except (ValueError,TypeError,KeyError):valid=False
            if valid:validate_summary(json.dumps(summary,ensure_ascii=False),row['fulltext'])
            audit=audit_summary(row['fulltext'],summary,index=index,expanded_diagnostic=False)
            n=audit['categories']['narrative'];base_n=base['audit']['categories']['narrative']
            assert n['longest_contiguous_match_words']<=base_n['longest_contiguous_match_words']
            assert n['words']<=base_n['words']*1.03,'Whole-summary added-word limit exceeded'
            assert n['covered_words_by_minimum_run']['6']<=base_n['covered_words_by_minimum_run']['6']
        except (AssertionError,ValueError,TypeError,KeyError) as e:
            summary=base['summary'];context=base['judge_context'];audit=base['audit'];rollback=True;error=repr(e)
            original_covered=base['audit']['categories']['narrative']['covered_words_by_minimum_run']['6']
        pairs=[]
        old=list(fragments(base['summary']));new=list(fragments(summary));assert len(old)==len(new)
        for (a,p,k),(b,q,l) in zip(old,new):
            assert (p,k)==(q,l)
            if k!='narrative' or a==b:continue
            aa=editor.splitter.tokenize(a);bb=editor.splitter.tokenize(b)
            if len(aa)!=len(bb):aa=[a];bb=[b]
            for x,y in zip(aa,bb):
                if x!=y:pairs.append(dict(uid=row['uid'],document_id=row['document_id'],condition=c,path=p,before=x,after=y,pair_id=digest(x+'\0'+y)))
        record=dict(uid=row['uid'],document_id=row['document_id'],condition=c,status='generated',source_sha256=index.sha256,
            summary=summary,judge_context=context,audit=audit,rolled_back=rollback,error=error,
            content_preservation='Strict insertions only; every original normalized word retained in order; exact numeric/unit/math signature, modals, negation and acronyms preserved. Automatic meaning review separately required.')
        write(condition_file(c,row['uid']),record);jsonl(ROOT/f'traces/{c}/{row["uid"]}.jsonl',traces)
        n=audit['categories']['narrative'];base_n=base['audit']['categories']['narrative']
        lengths=[x for t,p,k in fragments(summary) if k=='narrative' for x in nonoverlapping_passages(index,t)]
        values.append(dict(condition=c,uid=row['uid'],words=n['words'],covered=n['covered_words_by_minimum_run']['6'],coverage=n['covered_word_fraction_by_minimum_run']['6'],
            baseline_words=base_n['words'],baseline_covered=base_n['covered_words_by_minimum_run']['6'],original_covered_words=original_covered,
            strict=n['strict_five_word_limit_passed'],passage_lengths=lengths,changed=context!=base['judge_context'],rollback=rollback,error=error,pairs=pairs,
            rules=dict(Counter(e['rule'] for t in traces for e in t['edits'])) if not rollback else {},
            inserted_words=n['words']-base_n['words'],cpu_seconds=time.process_time()-tick,index_cpu_seconds=index_cpu))
    return values

def main():
    assert os.environ.get('SLURM_JOB_ID')
    if (ROOT/'outputs/cpu_complete.json').exists():return
    start=time.monotonic();results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,initializer=editor.init) as pool:
        for i,group in enumerate(pool.map(process,cohort(),chunksize=1),1):
            results.extend(group)
            if i%10==0:print('INSERTION_PAPERS',i,flush=True)
    pairs=[]
    for c in CONDITIONS[1:]:
        records=[r for r in results if r['condition']==c];pairs.extend(p for r in records for p in r['pairs']);counts=Counter()
        for r in records:counts.update(r['rules'])
        write(ROOT/f'outputs/{c}_complete.json',dict(complete=True,papers=97,mean_coverage=statistics.mean(r['coverage'] for r in records),
            pooled_coverage=sum(r['covered'] for r in records)/sum(r['words'] for r in records),strict_passes=sum(r['strict'] for r in records),
            narrative_passages_6plus=sum(len(r['passage_lengths']) for r in records),changed_summaries=sum(r['changed'] for r in records),rollbacks=sum(r['rollback'] for r in records),
            final_edited_windows=sum(len(r['pairs']) for r in records),words=sum(r['words'] for r in records),baseline_words=sum(r['baseline_words'] for r in records),
            inserted_words=sum(r['inserted_words'] for r in records),absolute_copied_words=sum(r['covered'] for r in records),baseline_copied_words=sum(r['baseline_covered'] for r in records),
            absolute_original_copied_words=sum(r['original_covered_words'] for r in records),
            coverage_with_fixed_original_denominator=statistics.mean(r['covered']/r['baseline_words'] for r in records),
            original_word_coverage_fixed_denominator=statistics.mean(r['original_covered_words']/r['baseline_words'] for r in records),
            rule_counts=dict(counts),cpu_seconds_including_index=sum(r['cpu_seconds']+r['index_cpu_seconds'] for r in records),shared_wall_seconds=time.monotonic()-start))
    controls=[
        ('The treatment reduced mortality.','The treatment maybe reduced mortality.','major_meaning_change'),
        ('The treatment may reduce mortality.','The treatment reduces mortality.','major_meaning_change'),
        ('All samples were tested.','Only some samples were tested.','major_meaning_change'),
        ('The model did not converge.','The model did converge.','major_meaning_change'),
        ('The treatment reduced error by 5%.','The treatment reduced error by 50%.','major_meaning_change'),
        ('The treatment reduced mortality.','The treatment clearly reduced mortality.','major_meaning_change'),
        ('All the samples were collected from healthy volunteers.','All of the samples were collected from healthy volunteers.','unchanged_or_equivalent'),
        ('The instrument was used to measure the response.','The instrument was used in order to measure the response.','unchanged_or_equivalent'),
        ('This study evaluated the model on held-out data.','This particular study evaluated the model on held-out data.','unchanged_or_equivalent'),
    ]
    controls=[dict(condition='controls',uid='control-'+str(i),document_id='control-'+str(i),path='control',before=a,after=b,expected=expected,pair_id=digest(a+'\0'+b)) for i,(a,b,expected) in enumerate(controls)]
    jsonl(ROOT/'outputs/per_summary.jsonl',[{k:v for k,v in r.items() if k!='pairs'} for r in results]);jsonl(ROOT/'inputs/post_review.jsonl',pairs+controls)
    write(ROOT/'outputs/cpu_complete.json',dict(complete=True,papers=97,conditions=len(CONDITIONS)-1,review_cards=len(pairs),review_controls=len(controls),seconds=time.monotonic()-start))
    print('INSERTION_CPU_COMPLETE',len(pairs),flush=True)

if __name__=='__main__':main()
