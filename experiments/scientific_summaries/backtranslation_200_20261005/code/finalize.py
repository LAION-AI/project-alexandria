"""English results, paired paper-cluster intervals and durable evidence manifest."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics
import time
from common import ROOT, ARMS, NAMES, load, read_jsonl, write

REPO=Path('/e/home/jusers/schuhmann1/jupiter/project-alexandria/experiments/scientific_summaries/backtranslation_200_20261005')
DURABLE=Path('/e/data1/datasets/playground/mmlaion/schuhmann1/scientific-backtranslation-200-20261005')

def correct(value):
    return sum(r['predictions']['qwen_summary']==r['gold'] for r in value['rows'])

def paired_ci(documents):
    groups=defaultdict(lambda:[0,0])
    for d in documents:
        groups[d['document_id']][0]+=correct(d['repaired'])-correct(d['original'])
        groups[d['document_id']][1]+=10
    values=list(groups.values());rng=random.Random(250219413);scores=[]
    for _ in range(5000):
        sample=rng.choices(values,k=len(values))
        scores.append(sum(v[0] for v in sample)/sum(v[1] for v in sample))
    scores.sort();return [scores[125],scores[4874]]

def main():
    protocol=load(ROOT/'inputs/protocol.json');rows={};lines=[
        '# Local back-translation benchmark: 200 scientific summaries', '',
        '200 existing no-thinking summary versions of 97 held-out papers: 97 tuned merged FP8, '
        '97 tuned merged BF16 and six live BF16 rank-128 Gemma 4 12B outputs. '
        'This is not 200 independent papers. All original and repaired outputs remain in the evaluation. '
        'Translation never receives questions, gold answers or reasoning traces.', '',
        'Only narrative windows containing a source-copy run of at least six normalized words '
        'are translated English → German → English, once. Preceding context is included where available. '
        'Metadata and explicit proof quotes are retained verbatim and audited separately.', '',
        '## Complete round-trip throughput', '',
        '| Model / decoding | Chosen batch | Windows | Successful round trips | Seconds | Windows/s | Summary versions/hour | Native output tokens/s | Active translation GPU hours |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        perf=load(ROOT/'outputs/translation'/arm/'performance.json')
        quality=load(ROOT/'outputs/quality'/arm/'report.json')
        qa=load(ROOT/'outputs/qa'/arm/'qa-results.json');documents=qa['documents']
        assert len(documents)==200 and len({d['document_id'] for d in documents})==97
        original=sum(correct(d['original']) for d in documents);after=sum(correct(d['repaired']) for d in documents)
        ci=paired_ci(documents)
        rows[arm]=dict(performance=perf,quality=quality,qa=dict(original_correct=original,
            repaired_correct=after,total=2000,original_accuracy=original/2000,repaired_accuracy=after/2000,
            difference=(after-original)/2000,paired_paper_cluster_ci95=ci,
            invalid_repaired=sum(r['predictions']['qwen_summary'] is None for d in documents for r in d['repaired']['rows']),
            execution=load(ROOT/'outputs/qa'/arm/'complete.json')))
        lines.append(f"| {NAMES[arm]} | {perf['batch_size']} | {perf['window_count']} | {perf['successful_roundtrips']} | {perf['roundtrip_seconds']:.2f} | {perf['windows_per_second']:.2f} | {perf['summaries_per_hour']:.1f} | {perf['combined_native_output_tokens_per_second']:.1f} | {perf['active_translation_gpu_hours']:.5f} |")
    lines+=['', 'Times include tokenization, both translation directions and complete outputs for all selected windows. '
        'They exclude model loading, compilation, the disjoint-training-paper batch sweep, quality checks and QA. '
        'Summary versions/hour depends on this cohort’s copy-window density; these are repairs, not newly generated summaries. '
        'Native token rates use different model tokenizers and are not a direct work-normalized comparison. '
        'One GPU per translating worker; two Windy arms run sequentially on their worker. '
        'TranslateGemma uses BF16, vLLM continuous batching, chunked prefill, prefix caching, language-only inference and default graph/attention selection. '
        'Windy uses FP16, SDPA, token-length sorting and dynamic batch padding. No source or candidate is silently truncated.', '',
        '## Numbers, units and formulas before the acceptance filter', '',
        '| Arm | Numeric windows | Suspect numeric changes | Suspect unit changes | Formula windows | Suspect formula/variable changes | Raw summaries with suspect changes | Guarded summaries with suspect changes |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,r in rows.items():
        q=r['quality'];lines.append(f"| {NAMES[arm]} | {q['numeric_windows']} | {q['numeric_signature_changed']} | {q['unit_signature_changed']} | {q['formula_windows']} | {q['formula_signature_changed']} | {q['raw_summaries_with_suspect_critical_change']} | {q['guarded_summaries_with_suspect_critical_change']} |")
    lines+=['', 'Critical signatures preserve numeric signs, decimal/scientific values, associated units, superscripts/subscripts, '
        'Greek variable names, detected mathematical expressions and inequalities. Cosmetic spacing, trailing decimal zeros and '
        'supported Unicode/LaTeX variants are normalized. General algebraic equivalence, unit conversions, spelled-out numbers and '
        'all possible scientific notation are not fully parsed. A signature mismatch is a conservative review flag, not a human-labeled error. '
        'Zero suspect changes in guarded outputs is guaranteed by rejecting/rolling back mismatching candidates; it does not mean the '
        'translator itself preserved everything. Original scientific claims are not independently certified by this check.', '',
        '## Accepted changes and meaning checks', '',
        '| Arm | Selected windows | Accepted windows inserted | Changed summaries | Global rollbacks | NLI below cutoff | QC seconds |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,r in rows.items():
        q=r['quality'];lines.append(f"| {NAMES[arm]} | {q['selected_windows']} | {q['actually_inserted_windows']} | {q['changed_summaries']} | {q['rolled_back_summaries']} | {q['nli_pairwise_below_cutoff']} | {q['total_quality_seconds']:.2f} |")
    lines+=['', 'Acceptance was fixed before QA: successful generation, unchanged critical signatures, bidirectional NLI entailment '
        'of at least 0.90, and no candidate-window source run above five words. Full assembled narratives are checked again. '
        'Rejected passages retain their original text and copy flags. NLI compares original and repaired summary passages; '
        'it is a heuristic, not a proof of scientific equivalence or source grounding. Overlength NLI pairs are rejected rather than truncated. '
        'The small positive/negative control set is diagnostic and not a scientific-domain accuracy estimate.', '',
        '## Source-copy overlap', '',
        '| Arm / output | Narrative ≤5 | Exactly 6 | ≥7 | Longest run | Mean coverage ≥6 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm,r in rows.items():
        for label,v in r['quality']['copy_overlap']['conditions'].items():
            n=v['narrative'];lines.append(f"| {NAMES[arm]} / {label} | {n['strict_five_word_pass']} | {n['borderline_six_only']} | {n['violation_seven_plus']} | {n['maximum_contiguous_match_words']} | {100*n['mean_covered_fraction_six_plus']:.3f}% |")
    lines+=['', 'Audit 2.1 uses normalized whitespace-delimited words against each paper’s complete untruncated source. '
        'Five is allowed, six borderline, seven or more flagged. Expanded punctuation-token diagnostics, cumulative coverage, '
        'metadata and evidence-quote checks are retained in the evidence. No paper is removed for overlap.', '',
        '## Fixed historical QA', '',
        '| Arm | Original correct / 2,000 | Guarded correct / 2,000 | Original accuracy | Guarded accuracy | Difference (percentage points) | Paired 95% paper-cluster interval |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for arm,r in rows.items():
        q=r['qa'];ci=q['paired_paper_cluster_ci95'];lines.append(f"| {NAMES[arm]} | {q['original_correct']} | {q['repaired_correct']} | {100*q['original_accuracy']:.2f}% | {100*q['repaired_accuracy']:.2f}% | {100*q['difference']:+.2f} | [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] |")
    lines+=['', 'The pinned 97-paper/970-MCQ test set, historical ASCII prompt and answer parser are unchanged. '
        'Each of the 200 versions is scored on its paper’s ten questions (2,000 slots per arm). '
        'Intervals resample 97 paper clusters, retaining all versions together. These are correlated versions, not 2,000 independent trials. '
        'Exact unchanged contexts reuse cached answers; changed contexts are answered by the same Qwen2.5-7B-Instruct judge '
        'with temperature 0.5, top-p 0.95, 100 output tokens, frequency/presence penalties 1.05 and up to four invalid-answer retries. '
        'Native vLLM batch scheduling differs from earlier HTTP concurrency four, and the stochastic judge adds sampling noise.', '',
        '## Conclusions', '']
    best=max(rows,key=lambda a:rows[a]['quality']['actually_inserted_windows'])
    lines.append(f"{NAMES[best]} inserted the most guarded repair windows ({rows[best]['quality']['actually_inserted_windows']}). "
        'Speed alone is insufficient: raw signature failures, acceptance coverage, residual whole-summary copying and QA must be considered together.')
    for arm,r in rows.items():
        n=r['quality']['copy_overlap']['conditions']['guarded_backtranslation']['narrative']
        lines.append(f"{NAMES[arm]} leaves {n['violation_seven_plus']}/200 narratives with runs of at least seven words; "
                     f"{n['strict_five_word_pass']}/200 meet the strict narrative five-word limit.")
    lines+=['', 'A single round trip with conservative rejection should only be adopted if the measured reduction and useful acceptance rate '
        'justify its overhead. These results do not establish that repeatedly translating until zero overlap would preserve scientific meaning. '
        'The table describes this fixed experiment; it is not a quality guarantee or a production-scale cost claim.', '',
        '## Provenance and evidence', '',
        'Pinned model IDs/revisions: `model_pins.json`. Exact selection, acceptance rules and code hashes: `protocol.json`. '
        'All batch sweep measurements, raw German/English candidates, numerical/formula signatures, NLI scores, rejected candidates, '
        'original/repaired QA and source-copy evidence are retained without editing the published training datasets.', '',
        f'Durable evidence directory: `{DURABLE}`. `evidence_manifest.json` lists SHA256 and byte lengths. '
        'Model weights and credentials are not included in the evidence export. Allocated node GPU hours and failed preparation overhead '
        'are recorded separately from active translation time.', '']
    results='\n'.join(lines)
    write(ROOT/'outputs/report.json',dict(protocol=protocol,arms=rows))
    (ROOT/'outputs/RESULTS.md').write_text(results)
    # Keep complete source/input and raw output evidence on the shared durable filesystem.
    DURABLE.mkdir(parents=True,exist_ok=True)
    for directory in ['inputs','outputs','code','logs']:
        shutil.copytree(ROOT/directory,DURABLE/directory,dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.tmp'))
    manifest=[]
    for p in sorted(DURABLE.rglob('*')):
        if not p.is_file() or p.name=='evidence_manifest.json':continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        manifest.append(dict(path=str(p.relative_to(DURABLE)),bytes=p.stat().st_size,sha256=h.hexdigest()))
    write(ROOT/'outputs/evidence_manifest.json',dict(files=manifest))
    shutil.copy2(ROOT/'outputs/evidence_manifest.json',DURABLE/'evidence_manifest.json')
    REPO.mkdir(parents=True,exist_ok=True);(REPO/'RESULTS.md').write_text(results)
    for name in ['model_pins.json','protocol.json','runtime.json']:
        shutil.copy2(ROOT/'inputs'/name,REPO/name)
    for name in ['report.json','evidence_manifest.json','nli_control_examples.json','qa_protocol.json']:
        shutil.copy2(ROOT/'outputs'/name,REPO/name)
    target=REPO/'evidence';target.mkdir(exist_ok=True)
    for arm in ARMS:
        folder=target/arm;folder.mkdir(exist_ok=True)
        for name in ['report.json','copy_overlap.json','copy_overlap.csv','COPY_OVERLAP.md']:
            shutil.copy2(ROOT/'outputs/quality'/arm/name,folder/name)
        for directory,name in [('optimization',arm+'.json'),('translation',arm+'/performance.json'),('qa',arm+'/complete.json')]:
            shutil.copy2(ROOT/'outputs'/directory/name,folder/(directory+'.json'))
        for name in ['window_quality.jsonl','copy_overlap_details.jsonl','summaries.jsonl']:
            with (ROOT/'outputs/quality'/arm/name).open('rb') as source, gzip.open(folder/(name+'.gz'),'wb',compresslevel=6) as out:
                shutil.copyfileobj(source,out)
        with (ROOT/'outputs/qa'/arm/'qa-results.json').open('rb') as source,gzip.open(folder/'qa-results.json.gz','wb',compresslevel=6) as out:
            shutil.copyfileobj(source,out)
    print('FINAL RESULTS AND DURABLE EVIDENCE READY',flush=True)

if __name__=='__main__':main()
