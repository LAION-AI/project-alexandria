import json,html
from pathlib import Path
from collections import Counter
from common import ROOT,CONDITIONS,cohort,load,write,read_jsonl,condition_file,digest
from copy_overlap_eval import audit_conditions
from ngram_overlap import fragments
from critical_values import check


def main():
    rows=cohort();qa=load(ROOT/'outputs/qa_results.json')
    qa['protocol']='Original970 slots, historical prompts/parser/sampling and paired question seeds20261023. Cached answers reused only by exact paper/question/context hash. No output selection after QA.'
    write(ROOT/'outputs/qa_results.json',qa)
    contexts={c:{r['document_id']:load(condition_file(c,r['uid'])) for r in rows} for c in CONDITIONS}
    overlap=audit_conditions(rows,contexts,ROOT/'outputs/overlap')
    gold={(p['document_id'],q['question_index']):q['answer'] for p in load(ROOT/'inputs/testset.json')['papers'] for q in p['questions']}
    protected=lambda s:[(t,p,k) for t,p,k in fragments(s) if k!='narrative']
    for c in CONDITIONS:
        answers=load(ROOT/f'outputs/qa/{c}.json');assert len(answers)==970
        assert len({(a['document_id'],a['question_index']) for a in answers})==970
        for a in answers:
            assert a['gold']==gold[a['document_id'],a['question_index']]
            assert a['context_sha256']==digest(contexts[c][a['document_id']]['judge_context'])
        for row in rows:
            a=contexts['baseline'][row['document_id']]['summary'];b=contexts[c][row['document_id']]['summary']
            assert protected(a)==protected(b)
            assert check(json.dumps(a,ensure_ascii=False),json.dumps(b,ensure_ascii=False))['passed']
    c=CONDITIONS[1];metrics=load(ROOT/f'outputs/{c}_complete.json');reviews=read_jsonl(ROOT/'outputs/reviews/post.jsonl')
    counts=dict(Counter(x['judgment']['severity'] for x in reviews));n=len(reviews)
    entry=dict(overlap=overlap['conditions'][c],qa=qa['scores'][c],deletion=metrics,meaning_counts=counts,reviewed_changed_windows=n,
        equivalent_or_awkward_rate=(counts.get('unchanged_or_equivalent',0)+counts.get('awkward_only',0))/n if n else None,
        major_or_unintelligible_rate=(counts.get('major_meaning_change',0)+counts.get('unintelligible',0))/n if n else None,
        extra_cpu_core_hours_38m=metrics['cpu_seconds_including_index']/97*38_000_000/3600,extra_runtime_gpu_hours=0)
    result=dict(complete=True,papers=97,mcqs_per_condition=970,conditions={c:entry},qa=qa,
        integrity=dict(passed=True,unique_qa_slots=1940,protected_evidence_metadata=True,numeric_formula_signatures_preserved=True),
        review_work=load(ROOT/'outputs/review_complete.json'),
        limitations='Articles and fillers retained. Rules fixed before new QA. Reused exploratory97 benchmark, automatic reviewer has known undercalled controls, existing Gemma errors not certified fixed.')
    write(ROOT/'outputs/results_and_scaling.json',result);write(ROOT/'outputs/integrity_checks.json',result['integrity'])
    previous=load(ROOT.parent/'scientific-post-gemma-article-caseguard-97-20261024/outputs/combined_comparison.json')
    combined=dict(previous['conditions']);combined[c]=entry
    write(ROOT/'outputs/combined_comparison.json',dict(complete=True,conditions=combined,original_results_preserved=True,limitations=result['limitations']))
    report=Path('/e/home/jusers/schuhmann1/jupiter/POST_GEMMA_ADJECTIVE_ADVERB_ONLY_97_2026-10-24.html')
    table=''
    for name,v in combined.items():
        ov=v['overlap']['narrative'];m=v['meaning_counts'];bad=m.get('major_meaning_change',0)+m.get('unintelligible',0)
        table+=f'<tr><th>{html.escape(name)}</th><td>{ov["mean_covered_fraction_six_plus"]:.2%}</td><td>{ov["mean_covered_fraction_seven_plus"]:.2%}</td><td>{v["qa"]["correct"]}/970</td><td>{bad}/{v["reviewed_changed_windows"]}</td><td>{v["extra_cpu_core_hours_38m"]:,.0f}</td></tr>'
    report.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Article versus modifier deletion</title><style>body{font:16px/1.65 system-ui;max-width:1250px;margin:30px auto;padding:24px;background:#f4f6f1;color:#183c35}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}.scroll{overflow:auto}</style><h1>Article versus adjective/adverb deletion</h1><p>Same97 saved post-Gemma scientific summaries and970 original MCQs. The new arm deletes only selected adjectives/adverbs within residual≥6-word source matches. Articles and filler phrases remain. This is separate from the earlier combined article+modifier arm. Every condition starts from the same baseline, and all final outputs retain source-copy flags.</p><div class="scroll"><table><tr><th>Condition</th><th>Mean copied words≥6</th><th>Mean copied words≥7</th><th>QA</th><th>Major/unintelligible new windows</th><th>Extra CPU coreh/38M</th></tr>'+table+'</table></div><p>The frozen Qwen2.5-7B student answers every original MCQ. A frozen27B teacher automatically judges every changed sentence/window; exact previous pairs are cached. These judgments can undercall scientific errors, and high QA does not certify fluent or faithful wording. All97 full-source audits include whitespace-word primary decisions and a separate punctuation-split diagnostic. Existing evidence quotes, metadata and numeric/formula signatures are unchanged.</p><p>Results: <code>'+str(ROOT/'outputs/results_and_scaling.json')+'</code>. Full evidence: <code>outputs/overlap/</code>. Deployed deletion is CPU-only; QA/reviewer costs are offline measurements.</p></html>')
    print('ADJECTIVE_ADVERB_ONLY_COMPLETE',entry['qa']['correct'],counts,flush=True)


if __name__=='__main__':main()
