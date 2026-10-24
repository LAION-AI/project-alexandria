"""Full-source overlap, untouched-QA integrity, quality and production CPU costs."""
import html
import json
import statistics
from collections import Counter
from common import ROOT, CONDITIONS, cohort, condition_file, load, write, read_jsonl, digest
from copy_overlap_eval import audit_conditions


def main():
    rows=cohort();qa=load(ROOT/'outputs/qa_results.json');reviews=[r for lane in range(3) for r in read_jsonl(ROOT/f'outputs/reviews/post-{lane}.jsonl')]
    contexts={c:{r['document_id']:load(condition_file(c,r['uid'])) for r in rows} for c in CONDITIONS}
    audit_conditions(rows,contexts,ROOT/'outputs/overlap')
    testset={p['document_id']:p for p in load(ROOT/'inputs/testset.json')['papers']}
    for c in CONDITIONS:
        answers=load(ROOT/f'outputs/qa/{c}.json');assert len(answers)==970
        for answer in answers:
            assert digest(contexts[c][answer['document_id']]['judge_context'])==answer['context_sha256']
            q=next(q for q in testset[answer['document_id']]['questions'] if q['question_index']==answer['question_index'])
            assert q['answer']==answer['gold']
        for row in rows:
            protected=lambda s:[(t,p,cat) for t,p,cat in __import__('ngram_overlap').fragments(s) if cat!='narrative']
            assert protected(contexts[c][row['document_id']]['summary'])==protected(contexts['baseline'][row['document_id']]['summary'])
    result={'complete':True,'baseline':'Safe CPU rules, then one Gemma E2B round; all previous results preserved.',
        'training_mining':load(ROOT/'outputs/mining_complete.json'),'dictionary_diagnostic':load(ROOT/'outputs/dictionary_diagnostic.json'),
        'luna_refinement_received':(ROOT/'inputs/luna_refinement_approved.json').exists(),'conditions':{},'qa':qa,
        'semantic_scope':'New CPU changes relative to Gemma baseline only; its preexisting36 severe/unintelligible judgments among1450 windows are not certified repaired.',
        'limits':'Repeated97 exploratory benchmark. Rules were selected using865 training papers and document-level training validation; historical test failures informed global safety blacklists. Automatic judgments are not human truth; overlapping source words are not legal determinations.'}
    table=''
    for c in CONDITIONS:
        r=load(ROOT/f'outputs/{c}_complete.json');cs=[p for p in reviews if p['condition']==c]
        meaning=dict(Counter(p['judgment']['severity'] for p in cs))
        cpu=r.get('cpu_seconds_including_index',0)/97*38_000_000/3600
        entry=dict(overlap=r,qa=qa['scores'][c],final_edit_meaning_counts=meaning,final_edit_review_windows=len(cs),
            procedural_cpu_core_hours_38m=cpu,generation_plus_one_gemma_repair_gpu_hours_38m=73699.0932093638,
            extra_runtime_gpu_hours_for_procedural_stage=0)
        result['conditions'][c]=entry
        bad=meaning.get('major_meaning_change',0)+meaning.get('unintelligible',0)
        table+=f'<tr><th>{html.escape(c)}</th><td>{r["mean_coverage"]:.2%}</td><td>{r["narrative_passages_6plus"]}</td><td>{qa["scores"][c]["correct"]}/970 ({qa["scores"][c]["accuracy"]:.2%})</td><td>{bad}/{len(cs)}</td><td>{cpu:,.0f}</td></tr>'
    write(ROOT/'outputs/results_and_scaling.json',result)
    write(ROOT/'outputs/integrity_checks.json',dict(passed=True,papers=97,conditions=4,qa_slots=3880,protected_evidence_metadata=True))
    output=__import__('pathlib').Path('/e/home/jusers/schuhmann1/jupiter/POST_GEMMA_STRUCTURAL_LEXICON_97_2026-10-24.html')
    output.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Post-Gemma procedural lexicon</title><style>body{font:16px/1.65 system-ui;background:#f4f5ee;color:#183d35;margin:40px auto;padding:20px;max-width:1200px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:15px;text-align:left;border-bottom:1px solid #ddd}h1{font:40px Georgia}.scroll{overflow:auto}</style><h1>Structural CPU repair after one Gemma E2B round</h1><p>Same97 saved summaries and970 original MCQs. Finite scientific phrase rules proposed with Luna, selected and refined using the865 training papers. No fresh Gemma generation. All failed/rolled-back papers remain in the denominator.</p><div class="scroll"><table><tr><th>Condition</th><th>Mean copied coverage ≥6</th><th>Source-match passages</th><th>QA</th><th>Major/unintelligible newly edited windows</th><th>Extra CPU core-hours /38M</th></tr>'+table+'</table></div><p>GPU inference estimate remains73,699 GH200 GPUh /38M including original generation and one Gemma E2B repair. The additional deployed stage is CPU-only; offline rule discovery and QA/reviewer costs are separate. CPU projection includes measured indexing, rule selection/guards and source audit. No claim of zero semantic errors or untouched-test generalization is made.</p><p>Machine-readable evidence: <code>'+str(ROOT/'outputs/results_and_scaling.json')+'</code>. All full-source primary and punctuation-split copying diagnostics are retained in <code>outputs/overlap/</code>.</p></html>')
    print('FINAL_REPORT',str(output),flush=True)


if __name__=='__main__':main()
