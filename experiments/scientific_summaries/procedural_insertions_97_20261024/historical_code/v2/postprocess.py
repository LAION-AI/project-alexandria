import concurrent.futures,json,os,time,statistics,math
from collections import Counter
from common import ROOT,CONDITIONS,cohort,load,write,jsonl,digest,condition_file,nonoverlapping_passages
from ngram_overlap import SourceIndex,fragments,audit_summary,tokenize
from passages import patch
from reference_validator import narrative_context,validate_summary
import insertions as editor

PARENT=ROOT.parent/'scientific-post-gemma-grammar-units-97-20261024'
LEGACY=ROOT.parent/'scientific-post-gemma-safe-insertions-guarded-97-20261024'
def protected(s):return [(t,p,k) for t,p,k in fragments(s) if k!='narrative']

def process(row):
    tick=time.process_time();index=SourceIndex(row['fulltext']);editor.prepare_index(index);index_cpu=time.process_time()-tick
    assert index.sha256==row['source_sha256']
    base=load(PARENT/f'outputs/repaired/combined/{row["uid"]}.json')
    write(condition_file('baseline',row['uid']),dict(base,condition='baseline'))
    old=load(LEGACY/f'outputs/repaired/grammar_insert/{row["uid"]}.json');write(condition_file('legacy_grammar',row['uid']),dict(old,condition='legacy_grammar'))
    write(ROOT/f'outputs/glossaries/{row["uid"]}.json',dict(source_sha256=index.sha256,definitions=index.insertion_glossary,provenance=index.insertion_glossary_provenance))
    values=[];base_n=base['audit']['categories']['narrative']
    for c in CONDITIONS[2:]:
        tick=time.process_time();traces=[];changes={};rollback=False;error=None;original_covered=0;stats=Counter();rule_candidates=Counter()
        remaining=math.floor(base_n['words']*.03)
        for text,path,cat in fragments(base['summary']):
            if cat!='narrative':continue
            after,plans,search=editor.repair(text,index,c,max_add_words=remaining)
            assert editor.verify_trace(text,after,plans,index)
            stats.update({k:v for k,v in search.items() if k!='rule_candidates'});rule_candidates.update(search['rule_candidates'])
            if after!=text:
                remaining-=len(tokenize(after)[1])-len(tokenize(text)[1]);assert remaining>=0
                changes[path,text]=after;traces.extend(dict(t,path=path) for t in plans)
                original_covered+=plans[0]['metrics_after']['original_covered']
            else:
                _,covered=editor.legacy.covered_positions(tokenize(text)[1],index);original_covered+=len(covered)
        try:
            summary=patch(base['summary'],changes);context=narrative_context(summary)
            assert editor.scientific_guard(base['judge_context'],context)
            assert protected(base['summary'])==protected(summary)
            old_frag=list(fragments(base['summary']));new_frag=list(fragments(summary));assert len(old_frag)==len(new_frag)
            try:validate_summary(json.dumps(base['summary'],ensure_ascii=False),row['fulltext']);valid=True
            except (ValueError,TypeError,KeyError):valid=False
            if valid:validate_summary(json.dumps(summary,ensure_ascii=False),row['fulltext'])
            audit=audit_summary(row['fulltext'],summary,index=index,expanded_diagnostic=False);n=audit['categories']['narrative']
            assert n['longest_contiguous_match_words']<=base_n['longest_contiguous_match_words'],'New longer source match'
            assert n['words']<=base_n['words']*1.03,'Document added-word budget'
            assert n['covered_words_by_minimum_run']['6']<=base_n['covered_words_by_minimum_run']['6']
        except (AssertionError,ValueError,TypeError,KeyError) as e:
            summary=base['summary'];context=base['judge_context'];audit=base['audit'];rollback=True;error=repr(e)
            original_covered=base_n['covered_words_by_minimum_run']['6']
        pairs=[];old_frag=list(fragments(base['summary']));new_frag=list(fragments(summary));assert len(old_frag)==len(new_frag)
        for (a,p,k),(b,q,l) in zip(old_frag,new_frag):
            assert (p,k)==(q,l)
            if k!='narrative' or a==b:continue
            aa=editor.splitter.tokenize(a);bb=editor.splitter.tokenize(b)
            if len(aa)!=len(bb):aa=[a];bb=[b]
            for x,y in zip(aa,bb):
                if x!=y:pairs.append(dict(uid=row['uid'],document_id=row['document_id'],condition=c,path=p,before=x,after=y,pair_id=digest(x+'\0'+y)))
        record=dict(uid=row['uid'],document_id=row['document_id'],condition=c,status='generated',source_sha256=index.sha256,
            summary=summary,judge_context=context,audit=audit,rolled_back=rollback,error=error,
            content_preservation='All original characters/words retained in order. Numeric/unit/Greek/operator/modal/negation/epistemic signatures equal; original formula spans unbroken. Inserted unit parentheses are typed metalinguistic explanations, not algebra. Full-source definition and edit provenance retained.')
        write(condition_file(c,row['uid']),record);jsonl(ROOT/f'traces/{c}/{row["uid"]}.jsonl',traces)
        n=audit['categories']['narrative'];lengths=[x for t,p,k in fragments(summary) if k=='narrative' for x in nonoverlapping_passages(index,t)]
        values.append(dict(condition=c,uid=row['uid'],words=n['words'],covered=n['covered_words_by_minimum_run']['6'],coverage=n['covered_word_fraction_by_minimum_run']['6'],
            baseline_words=base_n['words'],baseline_covered=base_n['covered_words_by_minimum_run']['6'],original_covered_words=original_covered,
            strict=n['strict_five_word_limit_passed'],passage_lengths=lengths,changed=context!=base['judge_context'],rollback=rollback,error=error,pairs=pairs,
            rules=dict(Counter(e['rule'] for t in traces for e in t['edits'])) if not rollback else {},search=dict(stats),rule_candidates=dict(rule_candidates),
            inserted_words=n['words']-base_n['words'],cpu_seconds=time.process_time()-tick,index_cpu_seconds=index_cpu))
    return values

def main():
    assert os.environ.get('SLURM_JOB_ID')
    if (ROOT/'outputs/cpu_complete.json').exists():return
    start=time.monotonic();results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,initializer=editor.init) as pool:
        for i,group in enumerate(pool.map(process,cohort(),chunksize=1),1):
            results.extend(group)
            if i%10==0:print('SYSTEM_PAPERS',i,flush=True)
    pairs=[]
    for c in CONDITIONS[2:]:
        records=[r for r in results if r['condition']==c];pairs.extend(p for r in records for p in r['pairs']);counts=Counter();stats=Counter();candidate_counts=Counter()
        for r in records:counts.update(r['rules']);stats.update(r['search']);candidate_counts.update(r['rule_candidates'])
        write(ROOT/f'outputs/{c}_complete.json',dict(complete=True,papers=97,mean_coverage=statistics.mean(r['coverage'] for r in records),
            pooled_coverage=sum(r['covered'] for r in records)/sum(r['words'] for r in records),strict_passes=sum(r['strict'] for r in records),narrative_passages_6plus=sum(len(r['passage_lengths']) for r in records),
            changed_summaries=sum(r['changed'] for r in records),rollbacks=sum(r['rollback'] for r in records),final_edited_windows=sum(len(r['pairs']) for r in records),
            words=sum(r['words'] for r in records),baseline_words=sum(r['baseline_words'] for r in records),inserted_words=sum(r['inserted_words'] for r in records),
            absolute_copied_words=sum(r['covered'] for r in records),baseline_copied_words=sum(r['baseline_covered'] for r in records),absolute_original_copied_words=sum(r['original_covered_words'] for r in records),
            coverage_with_fixed_original_denominator=statistics.mean(r['covered']/r['baseline_words'] for r in records),original_word_coverage_fixed_denominator=statistics.mean(r['original_covered_words']/r['baseline_words'] for r in records),
            rule_counts=dict(counts),candidate_counts=dict(candidate_counts),search=dict(stats),cpu_seconds_including_index=sum(r['cpu_seconds']+r['index_cpu_seconds'] for r in records),shared_wall_seconds=time.monotonic()-start))
    controls=[('The treatment reduced mortality.','The treatment maybe reduced mortality.','major_meaning_change'),
        ('The treatment may reduce mortality.','The treatment reduces mortality.','major_meaning_change'),('All samples were tested.','Only some samples were tested.','major_meaning_change'),
        ('The model did not converge.','The model did converge.','major_meaning_change'),('The treatment reduced error by 5%.','The treatment reduced error by 50%.','major_meaning_change'),
        ('The treatment reduced mortality.','The treatment clearly reduced mortality.','major_meaning_change'),
        ('All the samples were collected from healthy volunteers.','All of the samples were collected from healthy volunteers.','unchanged_or_equivalent'),
        ('The instrument was used to measure the response.','The instrument was used in order to measure the response.','unchanged_or_equivalent'),
        ('The sample contained 5 mg of the compound.','The sample contained 5 mg (milligrams) of the compound.','unchanged_or_equivalent'),
        ('The MRI data were processed using a model.','The MRI (magnetic resonance imaging) data were processed using a model.','unchanged_or_equivalent'),
        ('The system operated at 400 K and 1000 atm.','The system operated at a temperature of 400 K and 1000 atm.','major_meaning_change')]
    controls=[dict(condition='controls',uid='control-'+str(i),document_id='control-'+str(i),path='control',before=a,after=b,expected=e,pair_id=digest(a+'\0'+b)) for i,(a,b,e) in enumerate(controls)]
    jsonl(ROOT/'outputs/per_summary.jsonl',[{k:v for k,v in r.items() if k!='pairs'} for r in results]);jsonl(ROOT/'inputs/post_review.jsonl',pairs+controls)
    write(ROOT/'outputs/cpu_complete.json',dict(complete=True,papers=97,conditions=len(CONDITIONS)-2,review_cards=len(pairs),review_controls=len(controls),seconds=time.monotonic()-start))
    print('SYSTEM_CPU_COMPLETE',len(pairs),flush=True)

if __name__=='__main__':main()
