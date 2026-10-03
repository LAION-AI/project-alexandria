"""Recompute planning estimates from measured token work and throughput; no inference."""
import csv
import json
from pathlib import Path


def render(here, papers=60_000_000):
    report = json.loads((here / 'artifacts/report.json').read_text())
    phases = report['generation']['ornith_dflash']['phases']
    raw_keys = ['generation', 'source_recovery_generation']
    raw_input = sum(phases[k]['prompt_tokens'] for k in raw_keys) / 97
    raw_output = sum(phases[k]['completion_tokens'] for k in raw_keys) / 97
    post_input = sum(v['prompt_tokens'] for k, v in phases.items() if k not in raw_keys) / 97
    post_output = sum(v['completion_tokens'] for k, v in phases.items() if k not in raw_keys) / 97
    assumptions = dict(papers=papers, precision='BF16 Ornith target and draft',
                       prefill_tokens_per_second=25_000, useful_capacity=0.85,
                       cache_fraction_of_post_generation_prompt_tokens=0.75,
                       fast=dict(prefill_tokens_per_second=40_000, useful_capacity=0.90,
                                 post_input_cache_fraction=0.90, throughput_factor=1.0),
                       cautious=dict(prefill_tokens_per_second=15_000, useful_capacity=0.80,
                                     post_input_cache_fraction=0.0, throughput_factor=0.65))
    rows = []
    for mode in ['ar', 'dflash4']:
        benchmark = json.loads((here / f'artifacts/throughput-{mode}.json').read_text())['batches']
        generation = max((b for b in benchmark if b['phase']=='generation' and b['cache_state']=='cold'),
                         key=lambda b: b['output_tokens_per_second'])
        correction = max((b for b in benchmark if b['phase']=='correction' and b['cache_state']=='warm'),
                         key=lambda b: b['output_tokens_per_second'])
        g, c = generation['output_tokens_per_second'], correction['output_tokens_per_second']

        def estimate(corrected, cache, prefill, capacity, factor=1):
            # Cold generation throughput already includes generation prefill: never add it twice.
            seconds = raw_output / (g * factor)
            if corrected:
                seconds += post_output / (c * factor) + post_input * (1-cache) / prefill
            return papers * seconds / capacity / 3600

        for corrected in [False, True]:
            rows.append(dict(runtime=mode, workflow='generation + audit/correction/repair' if corrected else 'generation only',
                             corrected=corrected, generation_batch=generation['concurrency'],
                             correction_batch=correction['concurrency'], generation_tokens_per_second=g,
                             post_generation_tokens_per_second=c,
                             central_cache75_gpu_hours=estimate(corrected, .75, 25_000, .85),
                             central_no_post_input_cache_gpu_hours=estimate(corrected, 0, 25_000, .85),
                             fast_scenario_gpu_hours=estimate(corrected, .90, 40_000, .90),
                             cautious_scenario_gpu_hours=estimate(corrected, 0, 15_000, .80, .65)))
    workload = dict(source_papers=97, generation_calls=sum(phases[k]['calls'] for k in raw_keys),
                    generation_input_tokens_per_paper=raw_input, generation_output_tokens_per_paper=raw_output,
                    post_generation_input_tokens_per_paper=post_input,
                    post_generation_output_tokens_per_paper=post_output,
                    all_pipeline_calls=sum(v['calls'] for v in phases.values()),
                    full_input_tokens_per_paper=raw_input+post_input,
                    full_output_tokens_per_paper=raw_output+post_output)
    result = dict(assumptions=assumptions, observed_workload=workload, estimates=rows,
                  interpretation='Planning scenarios, not measured 60M-paper performance or statistical confidence intervals.',
                  caveats=['All post-generation phases share the warm correction decode rate as a planning proxy.',
                           'Prefill rate and cache fractions are assumptions, not measured full-pipeline capacities.',
                           'Cold 512-token generation rate embeds a higher prefill/output ratio than complete summaries.',
                           'AR reuses the observed DFlash8 token/retry workload; no complete AR workload was evaluated.',
                           'Longer sources, repetition, small repair-call latency and cache eviction can change the result.'])
    (here / 'scaling_60m.json').write_text(json.dumps(result, indent=2) + '\n')
    with (here / 'scaling_60m.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n');writer.writeheader();writer.writerows(rows)

    def rounded(value):
        return f'{round(value / 1000):,},000'

    lines = [
        '# Planning estimate: 60 million scientific summaries on JUPITER',
        '**English planning table, based on the completed 97-paper GH200 experiment (2026-10-03).** '
        'These are GPU-hour estimates, not measurements at 60-million-paper scale. The target is '
        '**Ornith-1.5-9B BF16**, optionally with its BF16 DFlash draft. No FP8 target speedup is assumed.',
        '## Estimated GPU-hours',
        '| Runtime | Workflow | Central: 75% post-input cache | Central: no post-input cache | Fast–cautious scenarios |',
        '| --- | --- | ---: | ---: | ---: |'
    ]
    lines += [f"| {'Standalone Ornith 9B' if r['runtime']=='ar' else 'Ornith 9B + DFlash4'} | "
              f"{'Summaries + audits, correction and repairs' if r['corrected'] else 'Summaries only'} | "
              f"{rounded(r['central_cache75_gpu_hours'])} | {rounded(r['central_no_post_input_cache_gpu_hours'])} | "
              f"{rounded(r['fast_scenario_gpu_hours'])}–{rounded(r['cautious_scenario_gpu_hours'])} |" for r in rows]
    lines += [
        'A practical initial budget for DFlash4 is approximately **0.4 million GPU-hours for summaries '
        'only**, or **0.6–0.9 million GPU-hours for the full audit/correction/repair workflow**. '
        'The cautious full-workflow scenario exceeds **1.3 million GPU-hours**. Capacity and failure '
        'rates should be validated on a larger representative run before reserving a production budget.',
        '“With correction” includes initial summaries **plus both source-only self-audits, semantic '
        'correction, source-quote/shape repairs and recovery**, as actually recorded. “Without '
        'correction” includes raw generation and its failed-generation retries/recovery, with '
        '**no quality audits or semantic/quote repair**. It therefore does not deliver the same '
        'evidence-validation guarantee as the corrected pipeline.',
        '## Observed work per paper',
        '| Work | Prompt tokens / paper | Output tokens / paper |',
        '| --- | ---: | ---: |',
        f'| Initial generation and generation recovery | {raw_input:,.2f} | {raw_output:,.2f} |',
        f'| Additional audits, correction and repairs | {post_input:,.2f} | {post_output:,.2f} |',
        f'| Complete recorded pipeline | {raw_input+post_input:,.2f} | {raw_output+post_output:,.2f} |',
        f'These counts include **{workload["generation_calls"]} generation calls** and '
        f'**{workload["all_pipeline_calls"]} total calls** across 97 papers, including failed attempts '
        'and recovery. The average generated-token work is much larger than the final narrative '
        'length. Repeated source text in repair prompts accounts for much of the input volume. '
        'This is the Ornith DFlash8 workload; using it for standalone AR and DFlash4 assumes the '
        'same retry/repair volume. That assumption has not been established in complete paired runs.',
        '## Throughput and explicit assumptions',
        '- Cold-source generation: **1,029.11 output tokens/s** standalone AR, **1,027.61** DFlash4; '
        'both best measured at batch **64**. These 512-token rates already include generation '
        'prefill; **generation prefill is not added again**. Cold probes have a greater '
        'prefill/output ratio than full summaries and may be conservative for long generation.\n'
        '- Fully warmed correction: **2,085.00 output tokens/s** standalone AR at batch **64**, '
        '**2,672.51** DFlash4 at batch **32**. This decode rate is used as a planning proxy for '
        '**all post-generation phases**, including short audit/anchor/repair calls. It is not '
        'a measured aggregate throughput for that heterogeneous work.\n'
        '- Central useful capacity: **85%** of the modeled capacity, reserving 15% for overhead '
        'and imperfect utilization. Central additional uncached prefill: **25,000 tokens/s '
        'per GPU**, an explicit assumption.\n'
        '- Central cache scenario: **75% of post-generation prompt tokens** reused; alternative '
        'column assumes **0%**. These cache fractions are **not measured whole-pipeline hit '
        'rates**. Generation stays cold in both columns.\n'
        '- Fast scenario: measured decode rates, **40,000** additional prefill tokens/s, '
        '**90%** useful capacity and **90%** post-input cache.\n'
        '- Cautious scenario: **65%** of measured decode rates, **15,000** additional prefill '
        'tokens/s, **80%** useful capacity and **no** post-input cache.',
        '## Calculation',
        'Let `N = 60,000,000`, `O_gen` and `O_post` be observed output tokens per paper, '
        '`I_post` additional prompt tokens per paper, `G_cold` measured cold generation '
        'tokens/s, `D_warm` measured cached correction tokens/s, `P` assumed uncached prefill '
        'tokens/s, `c` post-input cache fraction, `u` useful capacity and `f` throughput factor.',
        '```text\nGPUh_summary = N / (3600 × u) × O_gen / (G_cold × f)\n\n'
        'GPUh_full = N / (3600 × u) × [\n'
        '    O_gen / (G_cold × f)\n'
        '  + O_post / (D_warm × f)\n'
        '  + I_post × (1 − c) / P\n]\n```',
        'The extra prefill term is needed because the correction benchmark fully cached its '
        'input. This additive work approximation does not claim that peak prefill and decode '
        'rates are attained concurrently. It also does not model every small-call latency. '
        'The scenario range is a sensitivity analysis, **not a confidence interval**.',
        '## Scope and conclusions',
        'Raw Ornith achieved **93.81% QA accuracy**, corrected Ornith **93.20%** on this cohort; '
        'the −0.62-point paired difference is statistically uncertain. Skipping full correction '
        'therefore saves substantial compute without an observed QA penalty here. It does '
        '**not** establish equal factual reliability or source-evidence validity.',
        'DFlash4 improves modeled post-generation decode cost, while measured cold generation '
        'is essentially unchanged. Repeated repair prefill, cache reuse and retry volume '
        'matter more than a headline draft speedup. The next useful measurement is a larger '
        'complete-output throughput run with cold unique papers and measured aggregate cache reuse.',
        'Estimates exclude **QA evaluation, PDF extraction/OCR, data acquisition, CPU-only '
        'processing, storage/I/O beyond the generic capacity allowance, and LoRA training**. '
        'Scheduler waiting does not consume allocated GPU-hours but affects calendar time. '
        'Do not multiply the experiment\'s 4.3611 total allocated GPU-hours by 60M/97: that '
        'allocation also contains benchmarking, a second generator, all QA controls, startup '
        'and idle GPUs. Likewise, the older repository FP8 scaling assumptions and RTX3090/Q8 '
        'workload describe different experiments.',
        'These source papers are relatively short dataset texts. Longer deployment papers, '
        'different output lengths, repetitive generations, large-context cache eviction, '
        'small-request scheduling overhead and different failure rates can materially change '
        'the cost. Future batching/repair reductions are not counted as proven savings.',
        '## Recompute and audit',
        '```bash\npython3 experiments/scientific_summaries/ornith_dflash_97/scaling_60m.py\n```',
        'Exact unrounded figures and assumptions are in [scaling_60m.json](scaling_60m.json) '
        'and [scaling_60m.csv](scaling_60m.csv). The measured source data are '
        '[phase_timings.csv](phase_timings.csv), [throughput.csv](throughput.csv), and '
        '[the full findings/results report](README.md).'
    ]
    # Table rows must remain consecutive, while prose sections have blank-line separation.
    text = '\n\n'.join(lines)
    text = text.replace('|\n\n|', '|\n|')
    (here / 'SCALING_60M.md').write_text(text + '\n')
    return result


if __name__ == '__main__':
    result = render(Path(__file__).resolve().parent)
    print(json.dumps(result['estimates'], indent=2))
