"""Package the completed JUPITER run and render findings; no inference is performed."""
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def compressed_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as out:
        with gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0) as compressed:
            with source.open('rb') as original:
                shutil.copyfileobj(original, compressed)


def archive(files, destination):
    """Keep complete trace files, with relative names and reproducible archive headers."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as out:
        with gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0) as compressed:
            with tarfile.open(mode='w|', fileobj=compressed) as tar:
                for path, name in sorted(files, key=lambda pair: pair[1]):
                    data = path.read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    report = load(root / 'outputs/report.json')
    qa = load(root / 'outputs/qa-results.json')
    manifest = load(root / 'inputs/experiment_manifest.json')
    assert report['papers'] == 97 and report['questions'] == 970
    assert report['final_audit']['passed']
    artifact = HERE / 'artifacts'
    artifact.mkdir(exist_ok=True)
    for name in ['report.json', 'final_audit.json', 'gpu_accounting.json', 'performance_recipe.json',
                 'summary_lengths.json', 'finalized.json', 'final_sources.json', 'code_snapshot.json']:
        shutil.copyfile(root / 'outputs' / name, artifact / name)
    compressed_copy(root / 'outputs/qa-results.json', artifact / 'qa-results.json.gz')
    for path in sorted((root / 'outputs').glob('qa-results.before-recovery*.json')):
        compressed_copy(path, artifact / (path.name + '.gz'))
    with (root / 'inputs/paper_names.tsv').open(newline='') as stream:
        names = list(csv.reader(stream, delimiter='\t'))
    with (HERE / 'paper_names.tsv').open('w', newline='') as stream:
        csv.writer(stream, delimiter='\t', quoting=csv.QUOTE_ALL, lineterminator='\n').writerows(names)
    provenance = HERE / 'provenance'
    provenance.mkdir(exist_ok=True)
    for name in ['experiment_manifest.json', 'target-weights-manifest.json', 'draft-weights-manifest.json',
                 'judge-weights-manifest.json', 'qwen38-teacher-manifest.json', 'runtime-manifest.json',
                 'target-config.json', 'draft-config.json', 'judge-config.json']:
        shutil.copyfile(root / 'inputs' / name, provenance / name)
    for stage in ['primary', 'final']:
        shutil.copytree(root / 'outputs/code_snapshots' / stage,
                        provenance / 'source_code' / stage, dirs_exist_ok=True)
    archive([(p, str(p.relative_to(root))) for p in (root / 'logs').rglob('*') if p.is_file()] +
            [(p, str(p.relative_to(root))) for p in (root / 'outputs/server_accounting').rglob('*') if p.is_file()],
            artifact / 'runtime_logs.tar.gz')

    throughput = []
    for mode in ['ar', 'dflash4', 'dflash8', 'dflash15']:
        bench = load(root / f'outputs/throughput/{mode}.json')
        shutil.copyfile(root / f'outputs/throughput/{mode}.json', artifact / f'throughput-{mode}.json')
        for batch in bench['batches']:
            throughput.append(dict(runtime=mode, phase=batch['phase'], batch=batch['concurrency'],
                                   cache=batch['cache_state'], seconds=batch['seconds'],
                                   input_tokens=batch['input_tokens'], output_tokens=batch['completion_tokens'],
                                   output_tokens_per_second=batch['output_tokens_per_second'],
                                   mean_request_seconds=batch['request_latency_mean_seconds'],
                                   gpu_memory_used_total_utilization=batch['gpu_memory']))
        directory = root / f'outputs/throughput/{mode}'
        if directory.exists():
            archive([(p, str(p.relative_to(directory))) for p in directory.rglob('*') if p.is_file()],
                    artifact / f'throughput-traces-{mode}.tar.gz')
    write_csv(HERE / 'throughput.csv', throughput)
    phase_rows = [dict(model=model, phase=phase, **values)
                  for model, value in report['generation'].items() for phase, values in value['phases'].items()]
    write_csv(HERE / 'phase_timings.csv', phase_rows)
    score_rows = [dict(condition=condition, model=values['name'], correct=values['correct'],
                       total=values['total'], accuracy=values['accuracy'], ci95_low=values['ci95'][0],
                       ci95_high=values['ci95'][1], invalid=values['invalid'],
                       failed_generation_papers=values['failed_generation_papers'])
                  for condition, values in report['models'].items()]
    write_csv(HERE / 'scores.csv', score_rows)

    papers = {p['document_id']: p for p in load(root / 'inputs/eval_sources_only.json')}
    paper_rows, qa_timing = [], []
    flips = {model: collections.Counter() for model in ['ornith', 'qwen38']}
    for doc in qa['documents']:
        identifier = doc['document_id']
        for condition, value in doc['conditions'].items():
            rows = value['rows']
            correct = sum(next(iter(row['predictions'].values())) == row['gold'] for row in rows)
            invalid = sum(next(iter(row['predictions'].values())) is None for row in rows)
            paper_rows.append(dict(document_id=identifier, title=papers[identifier]['source']['source_title'],
                                   condition=condition, correct=correct, total=len(rows), invalid=invalid,
                                   fulltext_sha256=doc['fulltext_sha256'], context_sha256=value['context_sha256']))
            qa_timing.append(dict(document_id=identifier, condition=condition,
                                  latest_condition_seconds=value['elapsed_seconds'],
                                  format_attempts=sum(len(next(iter(row['responses'].values()))['attempts']) for row in rows)))
        for model in flips:
            for raw, corrected in zip(doc['conditions'][model + '_raw']['rows'],
                                      doc['conditions'][model + '_corrected']['rows']):
                left = next(iter(raw['predictions'].values())) == raw['gold']
                right = next(iter(corrected['predictions'].values())) == corrected['gold']
                flips[model][('correct' if left else 'wrong') + '_to_' + ('correct' if right else 'wrong')] += 1
    write_csv(HERE / 'paper_scores.csv', paper_rows)
    write_csv(HERE / 'qa_timings.csv', qa_timing)
    write_json(artifact / 'correction_transitions.json', {model: dict(value) for model, value in flips.items()})
    write_json(artifact / 'qa_timing_notes.json', dict(
        cumulative_scoring_intervals_seconds=qa['elapsed_seconds'],
        latest_retained_intervals_seconds=sum(r['latest_condition_seconds'] for r in qa_timing),
        note='Cumulative timer includes superseded condition evaluations. Latest intervals exclude superseded '
             'conditions. Both exclude judge loading, checkpoint I/O between conditions, and queue waits. '
             'Neither is total end-to-end elapsed experiment time.'))

    assessments, audit_rows, audit_stats = [], [], {}
    score_keys = ['factual_accuracy', 'coverage', 'clarity', 'faithfulness', 'scientific_precision']
    for model in ['ornith_dflash', 'qwen38_fp8']:
        documents = []
        for folder in sorted((root / f'outputs/cohorts/{model}/documents').iterdir()):
            if not folder.is_dir():
                continue
            doc = load(folder / 'document.json')
            documents.append(doc)
            archive([(p, str(p.relative_to(folder))) for p in folder.rglob('*') if p.is_file()],
                    artifact / 'paper_traces' / model / (folder.name + '.tar.gz'))
            for phase in ['raw', 'corrected']:
                audit = doc.get(phase + '_quality_assessment', {})
                assessments.append(dict(model=model, document_id=folder.name, phase=phase, assessment=audit))
                audit_rows.append(dict(model=model, document_id=folder.name, phase=phase,
                                       review_failed=bool(audit.get('review_failed')),
                                       verdict=audit.get('verdict', 'unavailable'),
                                       issues=len(audit.get('issues', [])),
                                       missing_central_facts=len(audit.get('missing_central_facts', [])),
                                       **{k: audit.get('scores', {}).get(k, '') for k in score_keys}))
        assert len(documents) == 97
        for phase in ['raw', 'corrected']:
            group = [r for r in audit_rows if r['model'] == model and r['phase'] == phase]
            audit_stats[model + '_' + phase] = dict(
                papers=len(group), failed_reviews=sum(r['review_failed'] for r in group),
                verdicts=dict(collections.Counter(r['verdict'] for r in group)),
                issues=sum(r['issues'] for r in group), missing_central_facts=sum(r['missing_central_facts'] for r in group),
                scores={k: dict(n=len(v := [r[k] for r in group if r[k] != '']),
                                mean=sum(v) / len(v), histogram=dict(collections.Counter(v))) for k in score_keys})
    write_csv(HERE / 'self_audit_scores.csv', audit_rows)
    write_json(artifact / 'self_audit_aggregate.json', audit_stats)
    write_json(artifact / 'quality_assessments.json', assessments)
    compressed_copy(artifact / 'quality_assessments.json', artifact / 'quality_assessments.json.gz')
    (artifact / 'quality_assessments.json').unlink()

    best = {}
    for mode in ['ar', 'dflash4', 'dflash8', 'dflash15']:
        for phase, cache in [('generation', 'cold'), ('correction', 'warm')]:
            best[(mode, phase)] = max((r for r in throughput if r['runtime'] == mode and
                                      r['phase'] == phase and r['cache'] == cache),
                                     key=lambda r: r['output_tokens_per_second'])
    sections = [
        '# Findings and results: Ornith-1.5-9B + DFlash on 97 scientific papers',
        '**Completed on JUPITER, 2026-10-03.** This report covers the complete BF16 Ornith / official FP8 '
        'Qwen3.8-27B generation, self-audit, correction and fixed-student QA experiment: 97 papers, '
        '970 immutable questions, eight conditions, and 7,760 scored answers. No LoRA is used.',
        '## Findings',
        '- Raw Ornith summaries achieve **910/970 (93.81%)**, versus **862/970 (88.87%)** for new Qwen3.8-27B '
        'FP8 summaries. The paired difference is **+4.95 percentage points**, 95% paper-bootstrap CI **+3.20 to +6.70**.\n'
        '- Corrected Ornith summaries achieve **904/970 (93.20%)**, versus **871/970 (89.79%)** for corrected Qwen. '
        'The paired difference is **+3.40 points**, CI **+1.34 to +5.36**.\n'
        '- Ornith correction changes six answers from wrong to correct and twelve from correct to wrong: '
        'a net **−6/970 (−0.62 points)**. Its CI, **−1.65 to +0.31**, includes zero. '
        'This is an observed small QA decrease, not conclusive evidence of worse factual reliability.\n'
        '- Qwen correction changes twelve answers from wrong to correct and three in the other direction: '
        '**+9/970 (+0.93 points)**, CI **0.00 to +1.96**. Correction is therefore not uniformly helpful or harmful.\n'
        '- Ornith summaries are substantially longer: **2,548 words raw / 2,302 corrected**, versus '
        '**1,199 / 1,187** for Qwen. Output length was not matched; greater retained detail may contribute '
        'to the QA advantage.\n'
        '- The best DFlash4 short-probe recipe reaches **1,028 output tokens/s for cold-source generation '
        '(batch 64)** and **2,673 tokens/s for fully cached correction (batch 32)** on one GH200. '
        'Standalone Ornith reaches **1,029 / 2,085 tokens/s**, respectively. DFlash brings approximately '
        '**28% more correction throughput** when each runtime uses its best measured batch, and essentially '
        'no cold-generation gain in this workload.\n'
        '- The actual 97-paper QA summaries were produced with **DFlash8**, selected before the supplemental '
        'DFlash4 benchmark. DFlash4 throughput and DFlash8 QA must be identified separately.\n'
        '- Total allocated compute, including benchmarking, loading, recovery, QA, idle allocation time '
        'and unsuccessful allocated starts, is **4.3611 GPU-hours**.',
        '## 1. Frozen evaluation and model identities',
        'The [released test set](../data/README.md) contains **50 arXiv and 47 Bethgelab papers**, with ten '
        'synthetic `gpt-6-luna`-authored MCQs each. It is a length-filtered convenience sample, not a '
        'human-validated or random scientific-paper benchmark. We retain all 97 papers and the exported '
        'question/option order. The three original excluded records remain excluded. Source text is the '
        'dataset text; completeness against publisher PDFs was not established.',
        table(['Role', 'Checkpoint', 'Precision', 'Pinned revision'], [
            ['Generator / corrector', '[ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B)', 'BF16', '`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`'],
            ['Speculative draft', '[ornith-ai/Ornith-1.5-9B-DFlash](https://huggingface.co/ornith-ai/Ornith-1.5-9B-DFlash)', 'BF16', '`21be3446a606afa67e20c33e68ca37396499248f`'],
            ['New comparison generator / corrector', '[Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)', 'Official FP8', '`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`'],
            ['Fixed QA student', '[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)', 'BF16', '`a09a35458c702b33eeacc393d103063234e8bc28`'],
            ['Historical summary reference', 'Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound', 'Mixed INT4', '`4756e3e4871aefd8d7cd5b0f6155ae5490451c1e`']]),
        'The DFlash link is a draft checkpoint (about 1.2B parameters), **not a standalone summarizer**. '
        'It needs the 9B target, which verifies its proposals. The standalone throughput arm uses that '
        'same BF16 target without a draft. There is no separate standalone-target QA arm in this 97-paper '
        'experiment; its QA equivalence was not measured here. The older RTX3090 / Q8 runs and the '
        '[paired 20-paper pilot](../ORNITH_DFLASH.md) are separate experiments.',
        f"Evaluation implementation pinned to repository commit `{manifest['commit']}`.\n\n"
        f"Test-set SHA256: `{manifest['testset_sha256']}`.\n\n"
        f"Effective initial summary-prompt SHA256: `{manifest['prompt_sha256']}`.",
        '## 2. Generation, correction and QA protocol',
        'Both new generators receive the same initial 19-field summary prompt and complete source text, '
        'temperature **0.2**, top-p **0.95**, and **thinking disabled**. The initial output budget is '
        '**16,000 tokens**, maximum context **65,536 tokens**. Sources are preflighted with the serving '
        'tokenizer and never truncated. Generator, reviewer and corrector never receive QA questions, '
        'choices, gold keys, rationales, existing summaries or QA evidence.',
        'The pipeline saves raw generation, a source-only same-model audit, a full semantic correction '
        'where needed, bounded V3 schema/source-quote repairs, and a corrected same-model audit. Raw '
        'means before semantic correction, with logged lossless shape normalization where necessary. '
        'The student context renders substantive narrative fields and excludes proof quotes, metadata '
        'and citation rankings. Literal quote alignment checks are mechanical evidence checks, not '
        'proof that every claim follows from its quote.',
        'Unusable/truncated/repetitive outputs receive bounded source-only recovery under a shared '
        'conditional policy. Recovery may use **32,000 generation / 24,000 correction output tokens** '
        'within the unchanged context limit. Observed repeated-sentence failures led to a documented '
        '**presence penalty of 1.5** on long-output recovery requests. Exact per-call settings and '
        'prior failures are retained. **13 Ornith papers and 2 Qwen papers** needed source recovery. '
        'Recovery decisions did not consult QA correctness; only changed context hashes were rejudged. '
        'No papers were dropped for quality. For `arxiv-500010`, an unusable optional bibliography '
        'ranking was set to an empty string after repeated formatting failures; the scientific '
        'narrative, original metadata, repair history and provenance remain saved.',
        'QA imports the pinned repository `run_document`, `historical_answer_prompt`, ASCII sanitizer '
        'and case-sensitive semicolon parser. Fixed student: BF16 Qwen2.5-7B-Instruct, **temperature 0.5**, '
        '**top-p 0.95**, **100 output tokens**, **frequency/presence penalties 1.05**, no thinking, '
        '**four concurrent requests**, maximum context **32,768**, up to **five formatting attempts**. '
        'Remaining invalid answers count as incorrect. All conditions are re-evaluated with this '
        'student, rather than silently mixing old cached scores. Confidence intervals use '
        '**10,000 document-cluster bootstrap draws, seed 250219413**; paired comparisons keep each '
        'paper\'s ten questions together.',
        '## 3. All QA scores',
        table(['Condition', 'Correct / 970', 'Accuracy', '95% paper-bootstrap CI', 'Invalid'], [
            [v['name'], v['correct'], f"{100*v['accuracy']:.2f}%",
             f"{100*v['ci95'][0]:.2f}–{100*v['ci95'][1]:.2f}%", v['invalid']]
            for v in report['models'].values()]),
        'All eight conditions retain 97 papers and 970 questions. All four new-model conditions '
        'have zero missing summaries and zero invalid final QA answers. The no-context condition '
        'has **four invalid answers**, counted wrong. The historical INT4 summary scores **870/970 '
        '(89.69%) in this rerun**; its older repository score was **872/970 (89.90%)**. Frozen inputs '
        'do not make stochastic sampling deterministic, and runtime differences may also contribute. '
        'The historical INT4 repair policy differs from the new full semantic-correction policy.',
        '### Paired differences',
        table(['Comparison', 'Difference (percentage points)', '95% paired paper-bootstrap CI'], [
            [k.replace('_minus_', ' − '), f"{100*v['difference']:+.2f}",
             f"{100*v['ci95'][0]:+.2f} to {100*v['ci95'][1]:+.2f}"]
            for k, v in report['paired_differences'].items()]),
        '### What changed after correction?',
        table(['Model', 'Correct → correct', 'Wrong → wrong', 'Wrong → correct', 'Correct → wrong', 'Net gain'], [
            [model, values['correct_to_correct'], values['wrong_to_wrong'], values['wrong_to_correct'],
             values['correct_to_wrong'], values['wrong_to_correct'] - values['correct_to_wrong']]
            for model, values in flips.items()]),
        'These are changes in the fixed student\'s answers, not directly counted factual errors in '
        'summaries. A corrected narrative can change the student\'s response even when neither '
        'narrative is demonstrably false. The experiment does not identify the cause of each transition.',
        '## 4. Narrative lengths',
        table(['Condition', 'Mean words', 'Median', 'Minimum', 'Maximum', 'Mean context/source word ratio'], [
            [k, f"{v['mean_words']:.2f}", v['median_words'], v['min_words'], v['max_words'],
             f"{v['mean_context_to_source_word_ratio']:.4f}"] for k, v in report['summary_lengths'].items()]),
        'Word counts refer to rendered narrative student contexts. Ratios are the mean of '
        'per-paper context/source ratios, not the ratio of two corpus totals. Ornith raw summaries '
        'average about **2.13×** the Qwen raw length. Correction reduces mean Ornith length by '
        'about **9.64%**; it reduces Qwen length by about **1.03%**. No controlled length-matched '
        'ablation was performed.',
        '## 5. Self-audit scores and failures',
        '**These are same-model source-only assessments, not independent human quality labels.** '
        'An audit can report high scores while listing unresolved issues; the saved normalized '
        'verdict preserves that inconsistency. Failed/invalid audits are explicitly unavailable, '
        'not imputed as perfect scores and not grounds for dropping papers from QA.',
        table(['Condition', 'Scored / 97', 'Failed audit', 'Pass', 'Needs correction', 'Flagged issues', 'Missing facts'], [
            [k, v['scores']['factual_accuracy']['n'], v['failed_reviews'], v['verdicts'].get('pass', 0),
             v['verdicts'].get('needs_correction', 0), v['issues'], v['missing_central_facts']]
            for k, v in audit_stats.items()]),
        table(['Condition', 'Factual accuracy / 5', 'Coverage / 5', 'Clarity / 5', 'Faithfulness / 5', 'Scientific precision / 5'], [
            [k] + [f"{v['scores'][s]['mean']:.3f}" for s in score_keys] for k, v in audit_stats.items()]),
        'Means use only valid scored audits: denominators differ substantially (**52/60 Ornith '
        'raw/corrected; 91/94 Qwen raw/corrected**). They cannot establish that one model is more '
        'faithful than another. Corrected audits still flag issues in **22 Ornith and 6 Qwen** '
        'papers; mechanical source-span validation does not imply an independent semantic pass. '
        'All per-paper grades, score histograms, failures, issues and proposed corrections are '
        'included in the artifacts.',
        '## 6. Hardware, runtime and optimizations',
        'JUPITER allocations provide four GH200 GPUs per node; **97,871 MiB** GPU memory was '
        'reported by NVML for each GPU used here (approximately 95.6 GiB). Runtime: '
        '**vLLM 0.30.0, PyTorch 2.13.0, Transformers 5.18.0, huggingface-hub 1.33.0**, '
        'Python 3.11 on ARM aarch64, CUDA 13. Model revisions and full-file SHA256 fingerprints '
        'are included. Native vLLM DFlash is used.',
        'Optimizations: one TP=1 model replica per GPU; continuous batching; up to 64 requests; '
        'automatic prefix caching; chunked prefill with **8,192 max batched tokens**; '
        'torch compilation/CUDA graphs and native attention/GDN kernels; local compiler caches; '
        'GPU memory utilization **0.90**. The main cohort used three Ornith DFlash8 replicas '
        '(32 concurrent papers each) and one Qwen FP8 replica (eight concurrent papers), followed '
        'by the fixed QA student. The source text was never shortened to improve speed.',
        '### Probe design and every measured batch',
        'Performance tuning uses **64 already accepted, non-holdout training papers**, not the '
        '97 evaluation papers. Every probe caps output at **512 tokens** and reports actual '
        'emitted output tokens divided by batch wall time. Batches **1, 4, 8, 16, 32 and 64** '
        'are measured in both cold and warm cache states for both phases: **96 observations**. '
        'These are short probes, not complete summaries; requests often finish at the output cap. '
        'Warm correction repeats the **entire correction input** after warming it, making it '
        'more optimistic than a first correction that reuses only the source prefix.',
        table(['Runtime', 'Batch', 'Generation cold tok/s', 'Generation warm tok/s',
               'Correction cold tok/s', 'Correction warm tok/s'], [
            [mode, batch] + [f"{next(r['output_tokens_per_second'] for r in throughput if r['runtime']==mode and r['batch']==batch and r['phase']==phase and r['cache']==cache):.2f}"
                            for phase, cache in [('generation','cold'),('generation','warm'),('correction','cold'),('correction','warm')]]
            for mode in ['ar','dflash4','dflash8','dflash15'] for batch in [1,4,8,16,32,64]]),
        '`ar` means standalone Ornith target without speculative decoding; `dflash4/8/15` '
        'denote the configured number of speculative tokens. At matched correction batch 32, '
        'DFlash4 improves warm throughput by about **55%** over AR. Comparing each runtime\'s '
        'best warm correction batch gives about **28%**. At batch 1, DFlash4 cold generation '
        'is slower than AR (**112 vs 140 tokens/s**). DFlash therefore does not provide a '
        'universal speedup.',
        'Larger batches are not always faster: DFlash8/15 at batch 64 show a pronounced warm '
        'correction slowdown. Prefix-cache pressure and speculative verification overhead '
        'are plausible contributors; metrics and cache counters are archived. Reported '
        'DFlash8 proposal acceptance is approximately **26–30%**; the shorter DFlash4 proposal '
        'achieves approximately **46.5%**. Acceptance alone is not end-to-end throughput. '
        'The initial selector chose the fastest **DFlash** candidate (8 versus 15); standalone '
        'AR had a higher initial combined probe score. DFlash4 was measured afterward and '
        'did not retroactively change the QA production configuration.',
        '## 7. Complete-output request timings and token work',
        'The following timers describe **individual API calls**, including retries and contention '
        'with concurrent requests. They are not end-to-end latency per paper. The generation '
        'and semantic-correction phases use full output budgets; their observed medians for '
        'Ornith are **100.78 s** and **94.10 s**, respectively. Adding phase means or medians '
        'does not produce measured wall time for a concurrently batched cohort.',
        table(['Model', 'Phase', 'Calls', 'Prompt tokens', 'Output tokens', 'Mean s', 'Median s', 'P95 s'], [
            [r['model'], r['phase'], r['calls'], r['prompt_tokens'], r['completion_tokens'],
             f"{r['mean_request_seconds']:.2f}", f"{r['median_request_seconds']:.2f}", f"{r['p95_request_seconds']:.2f}"]
            for r in phase_rows]),
        'Token counts include unsuccessful/repeated calls in the saved phase journals. Prompt '
        'tokens include repeated source prefixes; they do not measure uncached prefill work. '
        'The saved first-pass cohort timers are **643.04 s for Ornith on three GPUs** and '
        '**1,728.21 s for Qwen on one GPU**. The first pass produced only **90 raw / 84 corrected '
        'Ornith outputs** and **96 raw / 95 corrected Qwen outputs**. Completion counts were '
        'updated after recovery while those original timers remained unchanged: **neither '
        'timer is the elapsed time to finish all 97 papers**. GPU counts and output lengths '
        'also differ, so these timers are not a fair standalone speed ratio.',
        f"The QA checkpoint accumulates **{qa['elapsed_seconds']:.2f} s** of condition-scoring intervals, "
        f"including superseded contexts. The sum for latest retained conditions is "
        f"**{sum(r['latest_condition_seconds'] for r in qa_timing):.2f} s**. These exclude judge loading, "
        'checkpoint I/O between conditions and scheduler waits. Per-paper/condition times '
        'and format-attempt counts are in `qa_timings.csv`.',
        '## 8. Scheduler allocation and GPU-hours',
        table(['Slurm job', 'Outcome', 'Allocated seconds', 'GPUs', 'GPU-hours'], [
            [j['job_id'], j['state'], j['elapsed_seconds'], j['allocated_gpus'],
             f"{j['allocated_gpu_hours']:.6f}"] for j in report['gpu_accounting']['jobs']]),
        '**Total: 4.361111 allocated GPU-hours** (all listed jobs, four GPUs per allocation). '
        'This is an allocation bill that includes startup and idle GPUs; it is not a '
        'per-model active compute estimate. Overlapping Slurm steps are counted only once '
        'within their allocation. The first recovery allocation was canceled after an '
        'observed long repetitive output. The final 54-second failed job was a redundant '
        'guarded recovery that refused to overwrite an already completed document and '
        'performed no model work. Its 0.06 GPU-hours are still included. A separately '
        'canceled queued duplicate had zero allocation time. No validated 60-million-paper '
        'cost projection or LoRA training-cost measurement follows from this small run.',
        '## 9. Verification and preserved evidence',
        '- **97/97 raw and 97/97 corrected summaries for each new model**: 388 narrative contexts.\n'
        '- All **194 corrected** summaries pass schema/source-span checks: **14,078 Ornith '
        'and 8,600 Qwen** literal proof spans. Raw proof violations remain recorded.\n'
        '- All **7,760 QA prompt hashes and parsed predictions** pass the final audit; source '
        'hashes, immutable questions and gold keys match the frozen bundle.\n'
        '- Evaluation artifacts remain separate from the 1,000-paper training workspace. '
        '**Never use these 97 papers, their outputs, questions or answers as held-out '
        'LoRA training data.** Training-overlap screening/export guards are a separate pipeline.\n'
        '- All final raw/corrected document records, actual model requests/responses, '
        'source-only findings/proposed corrections, failures and recovery histories are '
        'preserved in per-paper trace archives. QA responses and prior checkpoints, '
        'benchmark calls/metrics, server logs, code snapshots and model-file hashes are included.\n'
        '- Thinking was disabled for this evaluation. Actual emitted reasoning fields are '
        'retained when present; no hidden reasoning trace is fabricated.',
        '## 10. Conclusions and next comparison',
        '1. **Use raw Ornith as the current QA baseline for this cohort.** It has the highest '
        'new-model summary score and avoids semantic-correction cost. This is a practical '
        'choice for question-answerability, not proof of universal factual superiority.\n'
        '2. **Do not assume automatic full-summary correction improves QA.** Ornith shows '
        'a small uncertain decline and Qwen a small improvement. Targeted correction of '
        'independently verified errors is a candidate to test, not an established result.\n'
        '3. **For this measured throughput workload, DFlash4 is the strongest correction '
        'candidate.** Use batch 32 for cached correction and batch 64 for cold generation '
        'as initial settings, then validate complete outputs on representative deployment '
        'lengths. Fully warmed 512-token rates must not be used as full-pipeline speed claims.\n'
        '4. **Ornith provides a strong baseline without LoRA.** No trained-LoRA versus base '
        'comparison is reported here, so this does not demonstrate that LoRA is unnecessary. '
        'Future LoRA evaluation must preserve holdout exclusion and the same fixed QA protocol.\n'
        '5. **Run a length-matched comparison and an independent factual audit before '
        'claiming model-level superiority.** Current results compare complete pipelines '
        'with different model sizes/precisions, output lengths, retry frequencies and '
        'self-audit availability. One synthetic convenience cohort and one stochastic '
        'evaluation run do not establish performance across scientific domains.',
        '## Files and reproduction',
        table(['File', 'Contents'], [
            ['[scores.csv](scores.csv)', 'All eight scores, confidence intervals and invalid counts'],
            ['[paper_scores.csv](paper_scores.csv)', 'All 776 paper/condition scores and source/context hashes'],
            ['[paper_names.tsv](paper_names.tsv)', 'All 97 names, paper identifiers and DOIs'],
            ['[throughput.csv](throughput.csv)', 'All 96 runtime/batch/cache/phase observations'],
            ['[phase_timings.csv](phase_timings.csv)', 'Every saved generation/review/correction/repair phase aggregate'],
            ['[qa_timings.csv](qa_timings.csv)', 'Latest condition intervals and format-attempt counts'],
            ['[self_audit_scores.csv](self_audit_scores.csv)', 'All 388 same-model audit rows, including failures'],
            ['[artifacts/report.json](artifacts/report.json)', 'Original complete numerical report, lengths and paired CIs'],
            ['[artifacts/](artifacts/)', 'Full QA, audit findings, per-paper traces, benchmark traces, logs, verification and accounting'],
            ['[provenance/](provenance/)', 'Pinned model-file fingerprints, configs, run protocol, primary/final source snapshots'],
            ['[SHA256SUMS](SHA256SUMS)', 'Integrity checksums for every packaged file']]),
        'Verify the packaged results on CPU, without network or model inference, from the repository root:\n\n'
        '```bash\npython3 experiments/scientific_summaries/ornith_dflash_97/verify_results.py\n```\n\n'
        'To rebuild this package from the preserved JUPITER run directory:\n\n'
        '```bash\npython3 experiments/scientific_summaries/ornith_dflash_97/package_results.py \\\n'
        '  --run-dir /path/to/scientific-ornith-dflash-eval-97\n```\n\n'
        'Original HPC source snapshots contain machine-specific paths and are preserved as '
        'provenance; adapt paths and stage the pinned repository/model revisions before '
        'running inference elsewhere. Trace `.tar.gz` archives use relative paths. Model '
        'weights and caches are not committed. Third-party source data/models retain their '
        'upstream terms and attribution; this package does not change them.'
    ]
    sections.insert(2, 'See also the [60-million-paper GPU-hour planning table](SCALING_60M.md), '
                    'with and without correction, explicit cache assumptions and sensitivity ranges.')
    (HERE / 'README.md').write_text('\n\n'.join(sections) + '\n')
    from scaling_60m import render
    render(HERE)
    paths = sorted(p for p in HERE.rglob('*') if p.is_file() and p.name != 'SHA256SUMS' and '__pycache__' not in p.parts)
    (HERE / 'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' +
                                          str(p.relative_to(HERE)) + '\n' for p in paths))
    print(json.dumps(dict(files=len(paths), bytes=sum(p.stat().st_size for p in paths),
                          scores=len(score_rows), paper_scores=len(paper_rows), throughput=len(throughput),
                          audit_rows=len(audit_rows), destination=str(HERE))))


if __name__ == '__main__':
    main()
