import html,json
from pathlib import Path
from collections import Counter
from common import ROOT,CONDITIONS,cohort,load,write,read_jsonl,condition_file,digest
from copy_overlap_eval import audit_conditions
from ngram_overlap import fragments
from critical_values import check


def main():
    rows=cohort();qa=load(ROOT/'outputs/qa_results.json')
    qa['protocol']='All original970 slots; same historical prompts/parser/sampling/seeds. Frozen baseline and exact paper/question/context prediction cache from preserved previous run. All new contexts evaluated. No QA-based output selection.'
    qa['inherited_statistic_key_mapping']={'vs_raw':'vs baseline','vs_safe':'vs delete_articles_case_guard (second condition)'}
    write(ROOT/'outputs/qa_results.json',qa)
    contexts={c:{r['document_id']:load(condition_file(c,r['uid'])) for r in rows} for c in CONDITIONS}
    overlap=audit_conditions(rows,contexts,ROOT/'outputs/overlap')
    test={p['document_id']:{q['question_index']:q for q in p['questions']} for p in load(ROOT/'inputs/testset.json')['papers']}
    protected=lambda s:[(t,p,k) for t,p,k in fragments(s) if k!='narrative']
    for c in CONDITIONS:
        answers=load(ROOT/f'outputs/qa/{c}.json');assert len(answers)==970
        assert len({(x['document_id'],x['question_index']) for x in answers})==970
        for a in answers:
            assert a['gold']==test[a['document_id']][a['question_index']]['answer']
            assert a['context_sha256']==digest(contexts[c][a['document_id']]['judge_context'])
        for row in rows:
            original=contexts['baseline'][row['document_id']]['summary'];edited=contexts[c][row['document_id']]['summary']
            assert protected(original)==protected(edited)
            assert check(json.dumps(original,ensure_ascii=False),json.dumps(edited,ensure_ascii=False))['passed']
    c=CONDITIONS[1];metrics=load(ROOT/f'outputs/{c}_complete.json');reviews=read_jsonl(ROOT/'outputs/reviews/post.jsonl')
    counts=dict(Counter(x['judgment']['severity'] for x in reviews));n=len(reviews)
    entry=dict(overlap=overlap['conditions'][c],qa=qa['scores'][c],deletion=metrics,meaning_counts=counts,reviewed_changed_windows=n,
        equivalent_or_awkward_rate=(counts.get('unchanged_or_equivalent',0)+counts.get('awkward_only',0))/n,
        major_or_unintelligible_rate=(counts.get('major_meaning_change',0)+counts.get('unintelligible',0))/n,
        extra_cpu_core_hours_38m=metrics['cpu_seconds_including_index']/97*38_000_000/3600,extra_runtime_gpu_hours=0)
    result=dict(complete=True,papers=97,mcqs_per_condition=970,conditions={c:entry},qa=qa,
        integrity=dict(passed=True,unique_qa_slots=1940,all97_full_sources=True,protected_evidence_metadata=True,numeric_formula_signatures_preserved=True),
        review_work=load(ROOT/'outputs/review_complete.json'),
        limits='Case/grammar bug correction after previous automatic meaning audit, not independent held-out validation. Cached reviewer controls undercalled two severe qualifier losses; automatic judgments are not human labels. Pre-existing Gemma mistakes not certified fixed.')
    write(ROOT/'outputs/results_and_scaling.json',result);write(ROOT/'outputs/integrity_checks.json',result['integrity'])
    previous=load(ROOT.parent/'scientific-post-gemma-deletion-97-20261024/outputs/results_and_scaling.json')
    combined=dict(previous['conditions']);combined[c]=entry
    write(ROOT/'outputs/combined_comparison.json',dict(complete=True,conditions=combined,
        six_word_sanity_check=load(ROOT.parent/'scientific-post-gemma-deletion-97-20261024/outputs/metric_sanity_check.json'),
        original_results_preserved=True,case_guard_revision='Lowercase articles, or sentence-initial capitalized articles followed by noun/adjective; preserve An identifier',
        limitations=result['limits']))
    table=''
    for name,v in combined.items():
        ov=v['overlap']['narrative'];meaning=v['meaning_counts'];n=v['reviewed_changed_windows'];bad=meaning.get('major_meaning_change',0)+meaning.get('unintelligible',0)
        table+=f'<tr><th>{html.escape(name)}</th><td>{ov["mean_covered_fraction_six_plus"]:.2%}</td><td>{ov["mean_covered_fraction_seven_plus"]:.2%}</td><td>{ov["strict_five_word_pass"]}/97</td><td>{v["qa"]["correct"]}/970</td><td>{bad}/{n}</td><td>{meaning.get("awkward_only",0)}/{n}</td><td>{v["extra_cpu_core_hours_38m"]:,.0f}</td></tr>'
    report=Path('/e/home/jusers/schuhmann1/jupiter/POST_GEMMA_TARGETED_DELETION_97_2026-10-24.html')
    old=report.read_text();start=old.index('<div class="scroll">');end=old.index('</table></div>',start)+len('</table></div>')
    updated='<div class="scroll"><table><tr><th>Condition</th><th>Mean copied coverage ≥6</th><th>Mean copied coverage ≥7</th><th>Strict≤5 summaries</th><th>QA</th><th>Major/unintelligible new windows</th><th>Awkward only</th><th>Extra CPU coreh/38M</th></tr>'+table+'</table></div>'
    old=old[:start]+updated+old[end:]
    note='<section><h2>Case-aware article guard</h2><p>The original article arm produced9 severe/unintelligible automatic judgments, all caused by removing the molecular abbreviation <b>An</b> from one paper. Its tag was incorrectly DT. The follow-up arm preserves mid-sentence capitalized identifiers and requires a following noun/adjective. Only article deletion is enabled. Earlier outputs, failures and scores remain preserved. This is an exploratory bug correction informed by the97-paper audit, not independent validation.</p><p>All97 final outputs received new full-source audits. Old QA predictions are reused only for identical paper/question/context hashes; old meaning judgments only for identical original/edited pairs. New cases are judged. Reviewer control tests undercalled removal of <i>randomized</i> and <i>may</i> as minor rather than major, so the reported severe counts can underestimate risk.</p><h3>Does the test incorrectly count deleted words?</h3><p>No. A six-word source sentence was tested with each of its six words individually deleted. Every resulting five-word sentence passed. The matcher requires consecutive words, with no gaps. Remaining violations belong to other unchanged or insufficiently shortened source matches. Exactly six words are reported separately from seven or more.</p><p>Updated results: <code>'+str(ROOT/'outputs/combined_comparison.json')+'</code>. New full-source evidence: <code>'+str(ROOT/'outputs/overlap')+'</code>.</p></section>'
    old=old.replace('<h2>Examples, including failures</h2>',note+'<h2>Original-run examples, including preserved failures</h2>')
    report.write_text(old)
    print('CASE_GUARD_COMPLETE',entry['qa']['correct'],counts,flush=True)


if __name__=='__main__':main()
