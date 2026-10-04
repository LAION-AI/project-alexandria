"""Estimate 60M-paper costs from preserved Gemma measurements; no GPU inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import tarfile


HERE = Path(__file__).resolve().parent
MODELS = {'gemma4_e4b': ('Gemma 4 E4B IT', 'gemma-4-E4B-it'),
          'gemma4_12b': ('Gemma 4 12B IT', 'gemma-4-12B-it')}
GENERATION_PHASES = {'generation', 'source_recovery_generation'}


def load(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                       '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                      ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def measure_lengths(tokenizer_root):
    # CPU-only; local_files_only prevents fetching weights or contacting a model server.
    from transformers import AutoTokenizer
    import tokenizers
    import transformers

    manifest = load(HERE / 'provenance/experiment_manifest.json')
    report = load(HERE / 'artifacts/report.json')
    rows, lengths, tokenizer_metadata = [], {}, {}
    for model, (_, directory) in MODELS.items():
        path = tokenizer_root / directory
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        tokenizer_metadata[model] = dict(
            model_id=manifest['models'][model]['id'],
            revision=manifest['models'][model]['revision'],
            tokenizer_class=type(tokenizer).__name__,
            file_sha256={name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                         for name in ['tokenizer.json', 'tokenizer.model', 'tokenizer_config.json']
                         if (path / name).is_file()})
        archives = sorted((HERE / 'artifacts/paper_traces' / model).glob('*.tar.gz'))
        if len(archives) != 97:
            raise ValueError('Expected 97 complete paper archives per model')
        model_rows = []
        for archive in archives:
            with tarfile.open(archive, 'r:gz') as handle:
                document = json.load(handle.extractfile('document.json'))
            row = dict(model=model, document_id=document['document_id'])
            for phase in ['raw', 'corrected']:
                context = document[phase]['judge_context']
                serialized = json.dumps(document[phase]['summary'], ensure_ascii=False)
                row[phase + '_narrative_tokens'] = len(tokenizer.encode(context, add_special_tokens=False))
                row[phase + '_summary_json_tokens'] = len(tokenizer.encode(serialized, add_special_tokens=False))
                row[phase + '_narrative_words'] = len(context.split())
                row[phase + '_context_sha256'] = digest(context)
                row[phase + '_summary_json_sha256'] = digest(serialized)
            rows.append(row)
            model_rows.append(row)
        for phase in ['raw', 'corrected']:
            condition = model + '_' + phase
            lengths[condition] = dict(papers=97)
            for kind in ['narrative_tokens', 'summary_json_tokens', 'narrative_words']:
                values = [row[phase + '_' + kind] for row in model_rows]
                lengths[condition][kind] = dict(mean=statistics.mean(values),
                                               median=statistics.median(values),
                                               minimum=min(values), maximum=max(values), total=sum(values))
            if abs(lengths[condition]['narrative_words']['mean'] -
                   report['summary_lengths'][condition]['mean_words']) > 1e-9:
                raise ValueError('Tokenized narrative does not match the evaluated narrative')
    write_csv(HERE / 'summary_token_lengths.csv', rows)
    write_json(HERE / 'summary_token_lengths.json', dict(
        description='Final evaluated narrative and full 19-field JSON lengths, not cumulative inference work.',
        add_special_tokens=False,
        summary_json_serialization='json.dumps(summary, ensure_ascii=False); default separators and preserved field order',
        runtime=dict(transformers=transformers.__version__, tokenizers=tokenizers.__version__),
        tokenizer_metadata=tokenizer_metadata, conditions=lengths))


def render(papers=60_000_000):
    report = load(HERE / 'artifacts/report.json')
    lengths = load(HERE / 'summary_token_lengths.json')['conditions']
    workload, estimates, rates = {}, [], {}
    for model in MODELS:
        phases = report['generation'][model]['phases']
        generation = [value for key, value in phases.items() if key in GENERATION_PHASES]
        post = [value for key, value in phases.items() if key not in GENERATION_PHASES]
        work = dict(source_papers=97,
                    generation_calls=sum(value['calls'] for value in generation),
                    post_generation_calls=sum(value['calls'] for value in post),
                    generation_input_tokens_per_paper=sum(value['prompt_tokens'] for value in generation) / 97,
                    generation_output_tokens_per_paper=sum(value['completion_tokens'] for value in generation) / 97,
                    post_generation_input_tokens_per_paper=sum(value['prompt_tokens'] for value in post) / 97,
                    post_generation_output_tokens_per_paper=sum(value['completion_tokens'] for value in post) / 97)
        workload[model] = work
        batches = load(HERE / ('artifacts/throughput-' + model + '.json'))['batches']
        selected = {phase: max((b for b in batches if b['phase'] == phase and b['cache_state'] == cache),
                               key=lambda b: b['output_tokens_per_second'])
                    for phase, cache in [('generation', 'cold'), ('correction', 'warm')]}
        g = selected['generation']['output_tokens_per_second']
        d = selected['correction']['output_tokens_per_second']
        rates[model] = {phase: dict(tokens_per_second=value['output_tokens_per_second'],
                                   batch=value['concurrency'], cache_state=value['cache_state'])
                        for phase, value in selected.items()}

        def estimate(corrected, cache, prefill, capacity, factor=1.0):
            # Cold generation already includes prefill, so it is not added again.
            seconds = work['generation_output_tokens_per_paper'] / (g * factor)
            if corrected:
                seconds += work['post_generation_output_tokens_per_paper'] / (d * factor)
                seconds += work['post_generation_input_tokens_per_paper'] * (1 - cache) / prefill
            return papers * seconds / capacity / 3600

        for corrected in [False, True]:
            estimates.append(dict(
                model=model, workflow='generation + audits/correction/repairs' if corrected else 'generation only',
                corrected=corrected,
                central_cache75_gpu_hours=estimate(corrected, .75, 25_000, .85),
                central_no_post_input_cache_gpu_hours=estimate(corrected, 0, 25_000, .85),
                fast_scenario_gpu_hours=estimate(corrected, .90, 40_000, .90),
                cautious_scenario_gpu_hours=estimate(corrected, 0, 15_000, .80, .65)))
    result = dict(papers=papers, hardware='JUPITER GH200, one BF16 Gemma target per GPU, native AR',
                  assumptions=dict(
                      central=dict(useful_capacity=.85, additional_uncached_prefill_tokens_per_second=25_000,
                                   post_input_cache_fractions=[.75, 0], throughput_factor=1.0),
                      fast=dict(useful_capacity=.90, additional_uncached_prefill_tokens_per_second=40_000,
                                post_input_cache_fraction=.90, throughput_factor=1.0),
                      cautious=dict(useful_capacity=.80, additional_uncached_prefill_tokens_per_second=15_000,
                                    post_input_cache_fraction=0, throughput_factor=.65)),
                  measured_rates=rates, observed_workload=workload, summary_lengths=lengths, estimates=estimates,
                  interpretation='Planning sensitivity scenarios, not measurements at 60M scale or confidence intervals.')
    write_json(HERE / 'scaling_60m.json', result)
    write_csv(HERE / 'scaling_60m.csv', estimates)

    def hours(value):
        return f'{round(value / 1000) * 1000:,}'

    sections = [
        '# Planning estimate: 60 million Gemma scientific summaries on JUPITER',
        '**English planning table, based on 97 held-out papers per model and the completed 2026-10-04 GH200 experiment.** '
        'Both checkpoints use BF16 native autoregressive decoding, no LoRA and no speculative draft. '
        'These figures extrapolate measured work and probe throughput; 60 million papers have not been processed.',
        '## Summary lengths and central GPU-hour estimates',
        table(['Model', 'Mean final narrative tokens: raw', 'Mean final narrative tokens: corrected',
               '60M summaries only: GPU-hours', '60M summaries + full correction: GPU-hours'], [
            [MODELS[m][0], f"{lengths[m + '_raw']['narrative_tokens']['mean']:,.0f}",
             f"{lengths[m + '_corrected']['narrative_tokens']['mean']:,.0f}",
             hours(next(r['central_cache75_gpu_hours'] for r in estimates if r['model'] == m and not r['corrected'])),
             hours(next(r['central_cache75_gpu_hours'] for r in estimates if r['model'] == m and r['corrected'])) + '–' +
             hours(next(r['central_no_post_input_cache_gpu_hours'] for r in estimates if r['model'] == m and r['corrected']))]
            for m in MODELS]),
        'The full-correction interval uses central assumptions with **75% to 0% post-generation input reuse**; '
        'it is a cache sensitivity range, not a statistical interval. All central figures reserve 15% of modeled capacity '
        'for overhead and imperfect utilization. Generation-only costs include the recorded failed-generation retries and '
        'generation recovery. Full correction also includes both source-only self-audits, semantic correction, '
        'quote/field repairs, failed attempts and finite-schema rescue.',
        '## Final artifact lengths',
        table(['Model', 'Variant', 'Mean narrative tokens', 'Median narrative tokens',
               'Mean full summary JSON tokens', 'Mean narrative words'], [
            [MODELS[m][0], phase,
             f"{lengths[m + '_' + phase]['narrative_tokens']['mean']:,.2f}",
             lengths[m + '_' + phase]['narrative_tokens']['median'],
             f"{lengths[m + '_' + phase]['summary_json_tokens']['mean']:,.2f}",
             f"{lengths[m + '_' + phase]['narrative_words']['mean']:,.2f}"]
            for m in MODELS for phase in ['raw', 'corrected']]),
        'Token counts use each checkpoint\'s pinned native Gemma tokenizer, **without BOS/EOS or chat-template tokens**. '
        'The narrative is exactly the text supplied to the QA answerer. The full JSON includes all 19 fields, '
        'proof quotations and metadata, serialized with json.dumps(summary, ensure_ascii=False). It is a canonical '
        'artifact length; original generation whitespace can differ. Neither artifact-length column includes failed attempts '
        'or audits. Their additional token work is counted separately below. The two tokenizer.json files have the same SHA256.',
        '## Wider planning sensitivity',
        table(['Model', 'Workflow', 'Fast scenario: GPU-hours', 'Cautious scenario: GPU-hours'], [
            [MODELS[r['model']][0], 'Summaries + full correction' if r['corrected'] else 'Summaries only',
             hours(r['fast_scenario_gpu_hours']), hours(r['cautious_scenario_gpu_hours'])] for r in estimates]),
        'Fast: measured rates, 90% useful capacity, 90% post-input reuse, 40,000 additional uncached prefill tokens/s. '
        'Cautious: 65% of measured rates, 80% useful capacity, no post-input reuse, 15,000 additional uncached prefill tokens/s. '
        'The prefill capacities, cache fractions and utilization allowances are explicit assumptions, '
        'shared with the [previous Ornith 9B estimate](../ornith_dflash_97/SCALING_60M.md); they are not measured whole-pipeline hit rates.',
        '## Observed inference work per finished paper',
        table(['Model', 'Work', 'Input tokens / paper', 'Output tokens / paper', 'Calls across 97 papers'], [
            [MODELS[m][0], label, f"{workload[m][prefix + '_input_tokens_per_paper']:,.2f}",
             f"{workload[m][prefix + '_output_tokens_per_paper']:,.2f}", workload[m][prefix + '_calls']]
            for m in MODELS for label, prefix in [('Generation and its recovery', 'generation'),
                                                  ('Additional audits, correction and repairs', 'post_generation')]]),
        'Generation phases are generation and source_recovery_generation; every other recorded phase is assigned to '
        'post-generation work. These are API usage counts including failed outputs and all recorded recovery, divided '
        'by 97 completed papers. Generation-only estimates reuse the generation/recovery work observed inside the full '
        'workflow; a separate complete generation-only cohort was not timed. Some regeneration was triggered by a '
        'downstream correction failure, so the raw-only retry allowance can be conservative.',
        '## Measured throughput used for the estimate',
        table(['Model', 'Cold-generation output tok/s', 'Repeat-input correction output tok/s', 'Batch'], [
            [MODELS[m][0], f"{rates[m]['generation']['tokens_per_second']:,.2f}",
             f"{rates[m]['correction']['tokens_per_second']:,.2f}",
             str(rates[m]['generation']['batch']) + ' / ' + str(rates[m]['correction']['batch'])] for m in MODELS]),
        'Each rate is measured on **one GH200** with actual 512-token probes and includes batch wall time. '
        'The generation probe starts cold; its prefill is already included. The correction probe repeats the full input '
        'with prefix caching, but capacity and eviction determine actual hits. Its rate is a proxy for all heterogeneous '
        'post-generation work, including very short repair requests; it is not an aggregate full-workflow measurement. '
        'The added uncached-prefill term approximates further cache misses and can be conservative because the repeat probe '
        'already includes residual prefill. Peak prefill and output rates are not assumed to occur simultaneously.',
        '## Calculation',
        '\n'.join(['`' * 3 + 'text',
            'N = 60,000,000',
            'GPUh_raw = N / (3600 * useful_capacity) * O_generation / (cold_generation_rate * rate_factor)',
            'GPUh_full = N / (3600 * useful_capacity) * [',
            '    O_generation / (cold_generation_rate * rate_factor)',
            '  + O_post / (repeat_correction_rate * rate_factor)',
            '  + I_post * (1 - post_input_cache_fraction) / assumed_prefill_rate',
            ']', '`' * 3]),
        '## Interpretation and limits',
        '**12B is cheaper under the observed workload despite lower generation tokens/s.** E4B consumed many more '
        'tokens in repeated or malformed generations and many more repair prompts. Its 775-word average raw narrative '
        'therefore does not imply a small generation bill. Starting with finite-schema decoding may lower those costs, '
        'but a complete fresh cohort under that revised recipe has not been measured and no such savings are counted here.',
        'Neither Gemma correction arm showed a clear QA gain on the fixed 970-MCQ test: E4B 85.15% raw / 84.64% corrected; '
        '12B 86.60% raw / 86.80% corrected. This does not establish equivalent factual reliability without correction.',
        'These estimates exclude QA evaluation, PDF acquisition/OCR, CPU-only processing, LoRA training and storage/I/O '
        'beyond the generic capacity allowance. Do not multiply the complete benchmark\'s 4.6689 GPU-hours by 60M/97: '
        'that mixed allocation includes both models, probe sweeps, QA, startup and idle GPUs. Source texts in this sample '
        'are relatively short; different paper lengths, output targets, retry behavior and cache pressure can substantially '
        'change the cost. Small-call latency, scheduler and CPU orchestration bottlenecks are not modeled individually. '
        'The scenario limits are assumptions, not established bounds on a production run.',
        '## Recompute and evidence',
        '\n'.join(['`' * 3 + 'bash',
            '# Standard Python: recalculate costs from the preserved token counts.',
            'python3 experiments/scientific_summaries/gemma4_97/scaling_60m.py',
            '# CPU-only tokenizer recount; requires Transformers 5.18 and local pinned tokenizer files.',
            'python experiments/scientific_summaries/gemma4_97/scaling_60m.py --tokenizer-root /path/to/local/models',
            '`' * 3]),
        'Exact figures and assumptions: [scaling_60m.json](scaling_60m.json) / [scaling_60m.csv](scaling_60m.csv). '
        'Per-paper lengths and tokenizer hashes: [summary_token_lengths.csv](summary_token_lengths.csv) / '
        '[summary_token_lengths.json](summary_token_lengths.json). Inputs: [phase timings](phase_timings.csv), '
        '[throughput](throughput.csv), full preserved paper archives and [findings/results](README.md).'
    ]
    (HERE / 'SCALING_60M.md').write_text('\n\n'.join(sections) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-root', type=Path,
                        help='CPU-only recount from local pinned tokenizers; omit to use preserved counts')
    args = parser.parse_args()
    if args.tokenizer_root:
        measure_lengths(args.tokenizer_root)
    result = render()
    print(json.dumps(dict(summary_lengths=result['summary_lengths'], estimates=result['estimates']), indent=2))
