from collections import Counter
from common import ROOT,CONDITIONS,cohort,load,write,read_jsonl,condition_file,digest
from ngram_overlap import fragments
from copy_overlap_eval import audit_conditions

def main():
    rows=cohort();qa=load(ROOT/'outputs/qa_results.json')
    qa['protocol']='Unchanged original970 slots, Qwen2.5 student/prompt/parser/sampling/paired seeds. Baseline is prior combined grammar/units pipeline912/970. No QA-based rewrite or selection. Exact-context caching only.'
    qa['difference_aliases']={'difference_vs_raw_pp':'versus baseline','difference_vs_safe_pp':'versus grammar_insert'}
    write(ROOT/'outputs/qa_results.json',qa)
    contexts={c:{r['document_id']:load(condition_file(c,r['uid'])) for r in rows} for c in CONDITIONS}
    overlap=audit_conditions(rows,contexts,ROOT/'outputs/overlap')
    gold={(p['document_id'],q['question_index']):q['answer'] for p in load(ROOT/'inputs/testset.json')['papers'] for q in p['questions']}
    for c in CONDITIONS:
        answers=load(ROOT/f'outputs/qa/{c}.json');assert len(answers)==970 and len({(a['document_id'],a['question_index']) for a in answers})==970
        for a in answers:
            assert a['gold']==gold[a['document_id'],a['question_index']]
            assert a['context_sha256']==digest(contexts[c][a['document_id']]['judge_context'])
        for row in rows:
            protected=lambda s:[(t,p,k) for t,p,k in fragments(s) if k!='narrative']
            assert protected(contexts['baseline'][row['document_id']]['summary'])==protected(contexts[c][row['document_id']]['summary'])
    reviews=read_jsonl(ROOT/'outputs/reviews/post.jsonl');result={}
    for c in CONDITIONS[1:]:
        m=load(ROOT/f'outputs/{c}_complete.json');cards=[r for r in reviews if r['condition']==c];assert len(cards)==m['final_edited_windows']
        for card in cards:
            row=next(r for r in rows if r['uid']==card['uid']);a=list(fragments(contexts['baseline'][row['document_id']]['summary']));b=list(fragments(contexts[c][row['document_id']]['summary']))
            assert any(p==q==card['path'] and k==l=='narrative' and x!=y and card['before'] in x and card['after'] in y for (x,p,k),(y,q,l) in zip(a,b))
        counts=dict(Counter(r['judgment']['severity'] for r in cards));n=len(cards)
        result[c]=dict(overlap=overlap['conditions'][c],qa=qa['scores'][c],insertion=m,meaning_counts=counts,reviewed_changed_windows=n,
            equivalent_or_awkward_rate=(counts.get('unchanged_or_equivalent',0)+counts.get('awkward_only',0))/n if n else None,
            major_or_unintelligible_rate=(counts.get('major_meaning_change',0)+counts.get('unintelligible',0))/n if n else None,
            extra_cpu_core_hours_38m=m['cpu_seconds_including_index']/97*38_000_000/3600,extra_runtime_gpu_hours=0,
            below_five_percent_reported=overlap['conditions'][c]['narrative']['mean_covered_fraction_six_plus']<.05,
            below_five_percent_fixed_denominator=m['coverage_with_fixed_original_denominator']<.05)
    controls=[c for c in reviews if c['condition']=='controls']
    value=dict(complete=True,papers=97,mcqs_per_condition=970,conditions=result,qa=qa,review_work=load(ROOT/'outputs/review_complete.json'),meaning_controls=controls,
        integrity=dict(passed=True,qa_slots=len(CONDITIONS)*970,protected_evidence_metadata=True,original_word_sequence_preserved=True,scientific_signatures_preserved=True),
        limitations='Exploratory on the reused97 benchmark. Exact lexical and scientific signatures do not prove semantic equivalence. Automatic reviewer labels are not human error rates. Inserting function words changes literal wording but does not create independent semantic paraphrases. All original matches/QA/outputs retained.')
    write(ROOT/'outputs/results_and_scaling.json',value);write(ROOT/'outputs/integrity_checks.json',value['integrity'])
    print('INSERTIONS_FINISHED',[(c,e['qa']['correct'],e['meaning_counts']) for c,e in result.items()],flush=True)

if __name__=='__main__':main()
