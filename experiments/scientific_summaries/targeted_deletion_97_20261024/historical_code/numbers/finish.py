import json
from collections import Counter
from common import ROOT,CONDITIONS,cohort,load,write,read_jsonl,condition_file,digest
from copy_overlap_eval import audit_conditions
from ngram_overlap import fragments


def main():
    rows=cohort();qa=load(ROOT/'outputs/qa_results.json')
    qa['protocol']='All970 original slots, fixed historical prompts/parser/sampling and paired seeds. Local baseline is the already case-guarded article summaries (913/970), not the earlier9.04% baseline. Reuse only exact context answers.'
    write(ROOT/'outputs/qa_results.json',qa)
    contexts={c:{r['document_id']:load(condition_file(c,r['uid'])) for r in rows} for c in CONDITIONS}
    overlap=audit_conditions(rows,contexts,ROOT/'outputs/overlap')
    gold={(p['document_id'],q['question_index']):q['answer'] for p in load(ROOT/'inputs/testset.json')['papers'] for q in p['questions']}
    for c in CONDITIONS:
        answers=load(ROOT/f'outputs/qa/{c}.json');assert len(answers)==970
        assert len({(a['document_id'],a['question_index']) for a in answers})==970
        for a in answers:
            assert a['gold']==gold[a['document_id'],a['question_index']]
            assert a['context_sha256']==digest(contexts[c][a['document_id']]['judge_context'])
        for r in rows:
            protected=lambda s:[(t,p,k) for t,p,k in fragments(s) if k!='narrative']
            assert protected(contexts['baseline'][r['document_id']]['summary'])==protected(contexts[c][r['document_id']]['summary'])
    reviews=read_jsonl(ROOT/'outputs/reviews/post.jsonl');entries={}
    for c in CONDITIONS[1:]:
        m=load(ROOT/f'outputs/{c}_complete.json');rs=[r for r in reviews if r['condition']==c];counts=dict(Counter(r['judgment']['severity'] for r in rs));n=len(rs)
        entries[c]=dict(overlap=overlap['conditions'][c],qa=qa['scores'][c],deletion=m,meaning_counts=counts,reviewed_changed_windows=n,
            equivalent_or_awkward_rate=(counts.get('unchanged_or_equivalent',0)+counts.get('awkward_only',0))/n if n else None,
            major_or_unintelligible_rate=(counts.get('major_meaning_change',0)+counts.get('unintelligible',0))/n if n else None,
            extra_cpu_core_hours_38m=m['cpu_seconds_including_index']/97*38_000_000/3600,extra_runtime_gpu_hours=0)
    result=dict(complete=True,papers=97,mcqs_per_condition=970,conditions=entries,qa=qa,
        integrity=dict(passed=True,qa_slots=2910,protected_evidence_metadata=True,numeric_preservation='Every accepted numeric/unit edit has exact roundtrip, sign and precision checks plus exact nonnumeric-text provenance; formula/operator guards. Old digit-only signature intentionally not used for digit-word changes.'),
        review_work=load(ROOT/'outputs/review_complete.json'),limitations='Exploratory97 benchmark. Numeric notation changes are not semantic paraphrases. Controlled grammar and finite range; automatic judgments may undercall errors. All failed/rolled-back records retained.')
    write(ROOT/'outputs/results_and_scaling.json',result);write(ROOT/'outputs/integrity_checks.json',result['integrity'])
    previous=load(ROOT.parent/'scientific-post-gemma-adjective-adverb-only-97-20261024/outputs/combined_comparison.json')
    combined=dict(previous['conditions']);combined.update(entries)
    write(ROOT/'outputs/combined_comparison.json',dict(complete=True,conditions=combined,numeric_local_baseline='delete_articles_case_guard',original_results_preserved=True,limitations=result['limitations']))
    print('NUMBER_STYLE_COMPLETE',{c:r['qa']['correct'] for c,r in entries.items()},flush=True)


if __name__=='__main__':main()
