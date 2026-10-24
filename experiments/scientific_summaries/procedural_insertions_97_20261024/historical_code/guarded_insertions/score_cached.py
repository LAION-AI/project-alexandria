"""Score only exact previously evaluated contexts; fail on any cache miss."""
import copy
import numpy as np
from common import ROOT,CONDITIONS,cohort,load,write,digest,condition_file

def main():
    cache={}
    for path in sorted((ROOT/'inputs/qa_cache').glob('*.json')):
        for a in load(path):cache[a['document_id'],a['question_index'],a['context_sha256']]=a
    papers={p['document_id']:p for p in load(ROOT/'inputs/testset.json')['papers']};rows=cohort();results={}
    for c in CONDITIONS:
        answers=[]
        for row in rows:
            context=load(condition_file(c,row['uid']))['judge_context'];sha=digest(context)
            for q in papers[row['document_id']]['questions']:
                key=row['document_id'],q['question_index'],sha
                assert key in cache,('Unseen context requires actual student inference',c,key)
                answer=copy.deepcopy(cache[key]);assert answer['gold']==q['answer']
                answer['reused_identical_context']=True;answers.append(answer)
        assert len(answers)==970
        write(ROOT/f'outputs/qa/{c}.json',answers);results[c]=answers
    counts=np.array([[sum(a['prediction']==a['gold'] for a in results[c] if a['document_id']==r['document_id']) for c in CONDITIONS] for r in rows])
    rng=np.random.default_rng(20261023);bootstrap=counts[rng.integers(0,97,(10000,97))].mean(axis=1)/10;scores={}
    for i,c in enumerate(CONDITIONS):
        scores[c]=dict(correct=int(counts[:,i].sum()),total=970,accuracy=float(counts[:,i].sum()/970),invalid=sum(a['prediction'] is None for a in results[c]),
            confidence_interval_95=np.quantile(bootstrap[:,i],[.025,.975]).tolist(),difference_vs_raw_pp=float((counts[:,i]-counts[:,0]).sum()/970*100),
            paired_difference_vs_raw_95ci_pp=np.quantile((bootstrap[:,i]-bootstrap[:,0])*100,[.025,.975]).tolist(),difference_vs_safe_pp=float((counts[:,i]-counts[:,1]).sum()/970*100),
            paired_difference_vs_safe_95ci_pp=np.quantile((bootstrap[:,i]-bootstrap[:,1])*100,[.025,.975]).tolist())
    write(ROOT/'outputs/qa_results.json',dict(complete=True,papers=97,questions_per_condition=970,scores=scores,unique_calls=0,exact_cache_only=True,
        protocol='Every retained context is byte-identical to an actually evaluated initial insertion arm or baseline. Unseen contexts fail; no prediction inference or imputation.',
        temperature=.5,top_p=.95,presence_penalty=1.05,frequency_penalty=1.05,max_tokens=100,format_attempts=5,bootstrap='10,000 whole-paper resamples',numpy_version=np.__version__))
    print('EXACT_CACHED_QA_COMPLETE',[(c,x['correct']) for c,x in scores.items()],flush=True)

if __name__=='__main__':main()
