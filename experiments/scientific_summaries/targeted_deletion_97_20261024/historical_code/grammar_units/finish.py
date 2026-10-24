import json
from collections import Counter
from common import ROOT,CONDITIONS,cohort,load,write,read_jsonl,condition_file,digest
from copy_overlap_eval import audit_conditions
from ngram_overlap import fragments


def main(reuse_overlap=False):
    rows=cohort();qa=load(ROOT/'outputs/qa_results.json')
    qa['protocol']='Original970 slots, frozen prompts/parser/sampling/seeds. Baseline is previous guarded numeric pipeline912/970. Arm1 is identifier-protected recomputation from guarded articles; grammar/unit arms add to Arm1. Exact-context/pair cache only. No QA-based selection.'
    qa['difference_aliases']={'difference_vs_raw_pp':'versus baseline (previous guarded numbers)', 'difference_vs_safe_pp':'versus identifier_guard'}
    write(ROOT/'outputs/qa_results.json',qa)
    contexts={c:{r['document_id']:load(condition_file(c,r['uid'])) for r in rows} for c in CONDITIONS}
    # Review-only metadata correction may reuse the completed compute-node
    # audit when every repaired file/context is byte-identical. The correction
    # helper records hashes before/after and preserves the initial raw labels.
    overlap=load(ROOT/'outputs/overlap/copy_overlap.json') if reuse_overlap else audit_conditions(rows,contexts,ROOT/'outputs/overlap')
    gold={(p['document_id'],q['question_index']):q['answer'] for p in load(ROOT/'inputs/testset.json')['papers'] for q in p['questions']}
    for c in CONDITIONS:
        answers=load(ROOT/f'outputs/qa/{c}.json');assert len(answers)==970
        assert len({(a['document_id'],a['question_index']) for a in answers})==970
        for a in answers:
            assert a['gold']==gold[a['document_id'],a['question_index']]
            assert a['context_sha256']==digest(contexts[c][a['document_id']]['judge_context'])
        for row in rows:
            protected=lambda s:[(t,p,k) for t,p,k in fragments(s) if k!='narrative']
            assert protected(contexts['baseline'][row['document_id']]['summary'])==protected(contexts[c][row['document_id']]['summary'])
    reviews=read_jsonl(ROOT/'outputs/reviews/post.jsonl');conditions={}
    from collections import defaultdict
    expected=defaultdict(list)
    # Include occurrence order because multiple citation fragments share paths.
    for c in CONDITIONS[1:]:
        for row in rows:
            old=list(fragments(contexts['baseline'][row['document_id']]['summary']))
            new=list(fragments(contexts[c][row['document_id']]['summary']))
            assert len(old)==len(new)
            for (a,p,k),(b,q,l) in zip(old,new):
                assert (p,k)==(q,l)
                if k=='narrative' and a!=b:expected[c,row['uid']].append((p,a,b))
    for card in reviews:
        allowed=expected[card['condition'],card['uid']]
        assert any(card['path']==p and card['before'] in a and card['after'] in b for p,a,b in allowed),'Review pair is not a visible summary edit'
    for c in CONDITIONS[1:]:
        m=load(ROOT/f'outputs/{c}_complete.json');cards=[r for r in reviews if r['condition']==c];counts=dict(Counter(r['judgment']['severity'] for r in cards));n=len(cards)
        conditions[c]=dict(overlap=overlap['conditions'][c],qa=qa['scores'][c],deletion=m,meaning_counts=counts,reviewed_changed_windows=n,
            equivalent_or_awkward_rate=(counts.get('unchanged_or_equivalent',0)+counts.get('awkward_only',0))/n if n else None,
            major_or_unintelligible_rate=(counts.get('major_meaning_change',0)+counts.get('unintelligible',0))/n if n else None,
            extra_cpu_core_hours_38m=m['cpu_seconds_including_index']/97*38_000_000/3600,
            grammar_unit_only_cpu_core_hours_38m=m['cpu_seconds_grammar_unit_stage']/97*38_000_000/3600,extra_runtime_gpu_hours=0)
    result=dict(complete=True,papers=97,mcqs_per_condition=970,conditions=conditions,qa=qa,review_work=load(ROOT/'outputs/review_complete.json'),
        integrity=dict(passed=True,qa_slots=len(CONDITIONS)*970,protected_evidence_metadata=True,numeric_preservation='Exact Decimal numeric roundtrips; acronym/model ID protection; exact case-sensitive same-magnitude unit aliases; only verified reporting-clause that deletion. All full-source audits retained.'),
        limitations='Test-informed exploratory extension on reused97 benchmark, not independent validation. Grammar and notation changes are not substantive paraphrases. Automatic meaning review has known false positives/undercalls. Identifier protection can correctly restore an overlap and therefore slightly increase copying.',
        review_alignment_correction='Duplicate citation paths initially caused15 unchanged label/justification mismatches per arm. Corrected using aligned occurrence order; all wrong cards/raw labels retained under outputs/diagnostics/review_alignment_initial. Summary/QA/copy-audit files unchanged; only actual edit pairs count.')
    write(ROOT/'outputs/results_and_scaling.json',result);write(ROOT/'outputs/integrity_checks.json',result['integrity'])
    previous=load(ROOT.parent/'scientific-post-gemma-number-style-guarded-97-20261024/outputs/combined_comparison.json');combined=dict(previous['conditions'])
    for c,entry in conditions.items():combined['grammar_units_'+c]=entry
    write(ROOT/'outputs/combined_comparison.json',dict(complete=True,conditions=combined,local_baseline='numbers_percent_guarded',original_results_preserved=True,limitations=result['limitations']))
    print('GRAMMAR_UNITS_COMPLETE',[(c,e['qa']['correct'],e['meaning_counts']) for c,e in conditions.items()],flush=True)


if __name__=='__main__':main()
