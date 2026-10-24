import json
from collections import Counter
from common import ROOT,CONDITIONS,cohort,load,write,read_jsonl,condition_file,digest
from copy_overlap_eval import audit_conditions
from ngram_overlap import fragments


def main():
    rows=cohort();qa=load(ROOT/'outputs/qa_results.json')
    qa['protocol']='Original970 slots and frozen prompts/parser/sampling/seeds. Local baseline is guarded articles913/970. Exact context/pair caching only. No QA-based output changes.'
    write(ROOT/'outputs/qa_results.json',qa)
    contexts={c:{r['document_id']:load(condition_file(c,r['uid'])) for r in rows} for c in CONDITIONS}
    overlap=audit_conditions(rows,contexts,ROOT/'outputs/overlap')
    gold={(p['document_id'],q['question_index']):q['answer'] for p in load(ROOT/'inputs/testset.json')['papers'] for q in p['questions']}
    for c in CONDITIONS:
        answers=load(ROOT/f'outputs/qa/{c}.json');assert len(answers)==970
        for a in answers:
            assert a['gold']==gold[a['document_id'],a['question_index']]
            assert a['context_sha256']==digest(contexts[c][a['document_id']]['judge_context'])
        for row in rows:
            protected=lambda s:[(t,p,k) for t,p,k in fragments(s) if k!='narrative']
            assert protected(contexts['baseline'][row['document_id']]['summary'])==protected(contexts[c][row['document_id']]['summary'])
    c=CONDITIONS[1];m=load(ROOT/f'outputs/{c}_complete.json');reviews=read_jsonl(ROOT/'outputs/reviews/post.jsonl');counts=dict(Counter(r['judgment']['severity'] for r in reviews));n=len(reviews)
    entry=dict(overlap=overlap['conditions'][c],qa=qa['scores'][c],deletion=m,meaning_counts=counts,reviewed_changed_windows=n,
        equivalent_or_awkward_rate=(counts.get('unchanged_or_equivalent',0)+counts.get('awkward_only',0))/n if n else None,
        major_or_unintelligible_rate=(counts.get('major_meaning_change',0)+counts.get('unintelligible',0))/n if n else None,
        extra_cpu_core_hours_38m=m['cpu_seconds_including_index']/97*38_000_000/3600,extra_runtime_gpu_hours=0)
    result=dict(complete=True,papers=97,mcqs_per_condition=970,conditions={c:entry},qa=qa,review_work=load(ROOT/'outputs/review_complete.json'),
        integrity=dict(passed=True,qa_slots=1940,protected_evidence_metadata=True,numeric_preservation='Exact Decimal value/sign/precision per edit, whole-math-sentence and quantity-context protections, nonnumeric text provenance. All full-source audits retained.'),
        limitations='Test-informed response to automatic meaning failures; reused97 benchmark, not independent validation. Notation changes are not semantic paraphrases. Automatic reviewer has known undercalled and false-positive controls.')
    write(ROOT/'outputs/results_and_scaling.json',result);write(ROOT/'outputs/integrity_checks.json',result['integrity'])
    previous=load(ROOT.parent/'scientific-post-gemma-number-style-97-20261024/outputs/combined_comparison.json');combined=dict(previous['conditions']);combined[c]=entry
    write(ROOT/'outputs/combined_comparison.json',dict(complete=True,conditions=combined,numeric_local_baseline='delete_articles_case_guard',original_results_preserved=True,limitations=result['limitations']))
    print('GUARDED_NUMBER_STYLE_COMPLETE',entry['qa']['correct'],counts,flush=True)


if __name__=='__main__':main()
