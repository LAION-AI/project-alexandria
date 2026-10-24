"""Repair review pair bookkeeping without changing any summary, QA or audit.

This one-off recovery script requires the preserved initial diagnostic and all
four correct exact-pair judgments to exist. It performs no model inference or
new full-source audit. Future runs use the corrected producer directly.
"""
import hashlib,time
from collections import Counter
from common import ROOT,CONDITIONS,cohort,load,write,jsonl,read_jsonl,digest,condition_file
from ngram_overlap import fragments
import grammar_units as editor


def checksums():
    paths=list((ROOT/'outputs/repaired').rglob('*.json'))
    paths+=list((ROOT/'outputs/qa').glob('*.json'))
    paths+=list((ROOT/'outputs/overlap').glob('*'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def main():
    editor.init();before_hashes=checksums();pairs=[]
    for c in CONDITIONS[1:]:
        cards=[]
        for row in cohort():
            old=list(fragments(load(condition_file('baseline',row['uid']))['summary']))
            new=list(fragments(load(condition_file(c,row['uid']))['summary']))
            assert len(old)==len(new)
            for (a,p,k),(b,q,l) in zip(old,new):
                assert (p,k)==(q,l)
                if k!='narrative' or a==b:continue
                aa=editor.splitter.tokenize(a);bb=editor.splitter.tokenize(b)
                if len(aa)!=len(bb):aa=[a];bb=[b]
                for x,y in zip(aa,bb):
                    if x!=y:cards.append(dict(uid=row['uid'],document_id=row['document_id'],condition=c,path=p,before=x,after=y,pair_id=digest(x+'\0'+y)))
        complete=load(ROOT/f'outputs/{c}_complete.json');complete['final_edited_windows']=len(cards)
        complete['review_alignment_corrected']=True;write(ROOT/f'outputs/{c}_complete.json',complete)
        pairs.extend(cards)
    assert len(pairs)==10 and len({p['pair_id'] for p in pairs})==4
    cache={p['pair_id'] for p in read_jsonl(ROOT/'inputs/review_cache.jsonl')}
    assert all(p['pair_id'] in cache for p in pairs),'Recovery must not require a new model call'
    jsonl(ROOT/'inputs/post_review.jsonl',pairs)
    cpu=load(ROOT/'outputs/cpu_complete.json');cpu['review_cards']=len(pairs);cpu['review_alignment_corrected']=True
    write(ROOT/'outputs/cpu_complete.json',cpu)
    from review import main as review
    review()
    assert checksums()==before_hashes
    from finish import main as finish
    finish(reuse_overlap=True)
    assert checksums()==before_hashes
    original=load(ROOT/'outputs/diagnostics/review_alignment_initial/review_complete.json')
    proof=dict(passed=True,correct_final_cards=len(pairs),unique_actual_changed_sentences=4,
               initial_wrong_cards=60,initial_gpu_pairs=original['new_unique_pairs'],
               additional_gpu_calls=0,summary_qa_audit_files_unchanged=True,
               preserved_files_sha256=before_hashes,
               explanation='Displayed citation paths are not unique. Pair fragments by occurrence order and assert path/category equality. 15 false label/justification comparisons in each of4 arms were never visible summary edits; raw cards and labels remain in the diagnostic.')
    write(ROOT/'outputs/review_alignment_correction.json',proof)
    protocol=load(ROOT/'inputs/protocol.json')
    protocol['review_pairing_correction']=dict(scope='Review metadata only; no summary/QA/audit changes, no new GPU calls.',
       frozen_original_protocol='outputs/diagnostics/review_alignment_initial/inputs_protocol.json',
       corrected_code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'code').glob('*.py'))})
    write(ROOT/'inputs/protocol.json',protocol)
    print('PASS review alignment; all summary/QA/audit hashes unchanged;10 cards,4 unique actual edits')


if __name__=='__main__':main()
