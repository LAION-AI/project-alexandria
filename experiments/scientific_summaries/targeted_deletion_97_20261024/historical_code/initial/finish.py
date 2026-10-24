"""Final frozen-output QA integrity, meaning, source audits and CPU scaling."""
import html,json,statistics,time
from collections import Counter
from pathlib import Path
from common import ROOT,CONDITIONS,cohort,load,write,read_jsonl,digest,condition_file
from ngram_overlap import fragments
from copy_overlap_eval import audit_conditions
from critical_values import check


def main():
    rows=cohort();qa=load(ROOT/'outputs/qa_results.json')
    qa['protocol']='Frozen historical prompts, parser and original970 MCQs. Saved post-Gemma/lexicon baseline predictions reused with exact context hashes; unchanged contexts share answers. Question-specific paired seeds20261023, identical sampling/retry settings. No summaries changed after seeing QA.'
    qa['inherited_statistic_key_mapping']={'vs_raw':'vs baseline','vs_safe':'vs delete_fillers'}
    write(ROOT/'outputs/qa_results.json',qa)
    reviewed=[r for lane in range(3) for r in read_jsonl(ROOT/f'outputs/reviews/post-{lane}.jsonl')]
    contexts={c:{r['document_id']:load(condition_file(c,r['uid'])) for r in rows} for c in CONDITIONS}
    overlap=audit_conditions(rows,contexts,ROOT/'outputs/overlap')
    questions={p['document_id']:{q['question_index']:q for q in p['questions']} for p in load(ROOT/'inputs/testset.json')['papers']}
    protected=lambda s:[(t,p,k) for t,p,k in fragments(s) if k!='narrative']
    signatures={c:dict(numeric_checked_papers=0,formula_checked_papers=0,passed_papers=0) for c in CONDITIONS}
    for c in CONDITIONS:
        answers=load(ROOT/f'outputs/qa/{c}.json');assert len(answers)==970
        assert len({(a['document_id'],a['question_index']) for a in answers})==970
        for answer in answers:
            assert answer['gold']==questions[answer['document_id']][answer['question_index']]['answer']
            assert digest(contexts[c][answer['document_id']]['judge_context'])==answer['context_sha256']
        for row in rows:
            base=contexts['baseline'][row['document_id']]['summary'];edited=contexts[c][row['document_id']]['summary']
            assert protected(base)==protected(edited)
            signature=check(json.dumps(base,ensure_ascii=False),json.dumps(edited,ensure_ascii=False))
            assert signature['passed']
            signatures[c]['passed_papers']+=1
            signatures[c]['numeric_checked_papers']+=signature['numeric_content_present']
            signatures[c]['formula_checked_papers']+=signature['formula_content_present']
    result=dict(complete=True,papers=97,mcqs_per_condition=970,baseline='Saved Gemma E2B first repair plus structural lexicon v2',conditions={},
        qa=qa,review_controls=[r for r in reviewed if r['condition']=='review_control'],
        integrity=dict(passed=True,unique_qa_slots=4850,protected_evidence_metadata=True,scientific_signatures=signatures),
        scope='Meaning judgments cover every newly changed visible sentence/fragment relative to the saved baseline. Pre-existing Gemma repair errors are not re-certified.',
        limitations='Exploratory reused97-paper set; automatic reviewer, not human truth. POS tagging can err. Deletion breaks literal strings but is not a guarantee of independent paraphrasing.',
        scaling='Warm CPU core-time includes source index, deletion candidate search, POS tagging, guards, main source audit, saved-output handling and passage diagnostics. Worker/tagger setup excluded and amortized. Separate expanded audit, offline QA/reviewer, original generation and original repair excluded. Shared source index and warmed variants can affect comparative CPU timings.')
    table='';excerpts=[]
    for c in CONDITIONS:
        metrics=load(ROOT/f'outputs/{c}_complete.json');meaning=[r for r in reviewed if r['condition']==c]
        counts=dict(Counter(r['judgment']['severity'] for r in meaning));n=len(meaning)
        cpu=metrics.get('cpu_seconds_including_index',0)/97*38_000_000/3600 if c!='baseline' else 0
        condition_overlap=overlap['conditions'][c]
        record=dict(overlap=condition_overlap,qa=qa['scores'][c],deletion=metrics,meaning_counts=counts,reviewed_changed_windows=n,
            equivalent_or_awkward_rate=(counts.get('unchanged_or_equivalent',0)+counts.get('awkward_only',0))/n if n else None,
            major_or_unintelligible_rate=(counts.get('major_meaning_change',0)+counts.get('unintelligible',0))/n if n else None,
            extra_cpu_core_hours_38m=cpu,extra_runtime_gpu_hours=0)
        result['conditions'][c]=record
        cov=condition_overlap['narrative']['mean_covered_fraction_six_plus'];score=qa['scores'][c]
        severe=counts.get('major_meaning_change',0)+counts.get('unintelligible',0)
        table+=f'<tr><th>{html.escape(c)}</th><td>{cov:.2%}</td><td>{metrics["narrative_passages_6plus"]}</td><td>{condition_overlap["narrative"]["strict_five_word_pass"]}/97</td><td>{metrics.get("deleted_words",0):,}</td><td>{score["correct"]}/970</td><td>{severe}/{n}</td><td>{counts.get("awkward_only",0)}/{n}</td><td>{cpu:,.0f}</td></tr>'
        for label in ['unchanged_or_equivalent','awkward_only','major_meaning_change','unintelligible']:
            chosen=sorted([r for r in meaning if r['judgment']['severity']==label],key=lambda r:r['pair_id'])[:3]
            for r in chosen:
                excerpts.append(dict(r))
    write(ROOT/'outputs/results_and_scaling.json',result)
    write(ROOT/'outputs/integrity_checks.json',result['integrity'])
    jsonl_path=ROOT/'outputs/selected_examples.jsonl'
    from common import jsonl
    jsonl(jsonl_path,excerpts)
    content=''.join('<section><h3>'+html.escape(r['condition']+' — '+r['judgment']['severity'])+'</h3><p><b>Before:</b> '+html.escape(r['before'])+'</p><p><b>After:</b> '+html.escape(r['after'])+'</p><p>'+html.escape(r['judgment'].get('reason',''))+'</p></section>' for r in excerpts)
    report=Path('/e/home/jusers/schuhmann1/jupiter/POST_GEMMA_TARGETED_DELETION_97_2026-10-24.html')
    report.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Targeted deletion after Gemma repair</title><style>body{font:16px/1.65 system-ui;max-width:1400px;margin:30px auto;padding:24px;background:#f3f6f1;color:#193c36}h1{font:42px Georgia}table{width:100%;border-collapse:collapse;background:white}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow:auto}section{background:white;padding:16px;margin:18px 0;border-radius:12px}</style><h1>Can deleting low-information words reduce source copying?</h1><p>Same97 saved scientific summaries, full sources, and970 original MCQs per condition. Baseline includes one Gemma E2B repair and the structural CPU phrase bank. Each deletion arm starts independently from this baseline. Only residual ≥6-word overlaps are targeted; no summary regeneration. Maximum15% deletion per fragment, with at least one-word allowance for short fragments. Source coverage must decrease in both absolute words and fraction; no increased longest source match. Evidence and metadata remain unchanged.</p><p>Arms: sentence-initial discourse/metatext; additionally a/an/the tagged as determiners as determiners; additionally eight whitelisted nontechnical evaluative adjectives; additionally adjectives and adverbs passing scientific/scope protections. Comparative and superlative modifiers are not deleted. Broad POS deletion is an experimental diagnostic, not a safety certification.</p><div class="scroll"><table><tr><th>Condition</th><th>Mean copied coverage ≥6</th><th>Source-match passages</th><th>Strict≤5 summaries</th><th>Deleted words</th><th>QA</th><th>Major/unintelligible new windows</th><th>Awkward only</th><th>Extra CPU coreh/38M</th></tr>'+table+'</table></div><p>Every changed visible sentence/fragment is assessed by the frozen27B FP8 automatic reviewer with thinking disabled. All970 QA slots per condition are retained. Reviewer judgments are not human labels, and QA can miss factual errors or poor grammar. Existing baseline errors are not certified repaired. Removing a word can break an exact string without substantially rephrasing its content.</p><p>The deployed deletion stage needs no GPU. CPU estimates are warm core-time extrapolations; source-index reuse and warmed variants affect comparisons. Worker startup, expanded diagnostic, QA/reviewer, original generation and original Gemma repair are separate. Machine-readable results: <code>'+str(ROOT/'outputs/results_and_scaling.json')+'</code>. Full primary and punctuation-split source-copy evidence: <code>outputs/overlap/</code>.</p><h2>Examples, including failures</h2>'+content+'</html>')
    print('DELETION_FINAL_REPORT',report,flush=True)


if __name__=='__main__':main()
