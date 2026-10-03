# Planning estimate: 60 million scientific summaries on JUPITER

**English planning table, based on the completed 97-paper GH200 experiment (2026-10-03).** These are GPU-hour estimates, not measurements at 60-million-paper scale. The target is **Ornith-1.5-9B BF16**, optionally with its BF16 DFlash draft. No FP8 target speedup is assumed.

## Estimated GPU-hours

| Runtime | Workflow | Central: 75% post-input cache | Central: no post-input cache | Fast–cautious scenarios |
| --- | --- | ---: | ---: | ---: |
| Standalone Ornith 9B | Summaries only | 366,000 | 366,000 | 346,000–599,000 |
| Standalone Ornith 9B | Summaries + audits, correction and repairs | 572,000 | 845,000 | 476,000–1,432,000 |
| Ornith 9B + DFlash4 | Summaries only | 367,000 | 367,000 | 346,000–600,000 |
| Ornith 9B + DFlash4 | Summaries + audits, correction and repairs | 547,000 | 821,000 | 453,000–1,391,000 |

A practical initial budget for DFlash4 is approximately **0.4 million GPU-hours for summaries only**, or **0.6–0.9 million GPU-hours for the full audit/correction/repair workflow**. The cautious full-workflow scenario exceeds **1.3 million GPU-hours**. Capacity and failure rates should be validated on a larger representative run before reserving a production budget.

“With correction” includes initial summaries **plus both source-only self-audits, semantic correction, source-quote/shape repairs and recovery**, as actually recorded. “Without correction” includes raw generation and its failed-generation retries/recovery, with **no quality audits or semantic/quote repair**. It therefore does not deliver the same evidence-validation guarantee as the corrected pipeline.

## Observed work per paper

| Work | Prompt tokens / paper | Output tokens / paper |
| --- | ---: | ---: |
| Initial generation and generation recovery | 13,662.96 | 19,221.44 |
| Additional audits, correction and repairs | 464,505.93 | 12,215.08 |
| Complete recorded pipeline | 478,168.89 | 31,436.53 |

These counts include **164 generation calls** and **3597 total calls** across 97 papers, including failed attempts and recovery. The average generated-token work is much larger than the final narrative length. Repeated source text in repair prompts accounts for much of the input volume. This is the Ornith DFlash8 workload; using it for standalone AR and DFlash4 assumes the same retry/repair volume. That assumption has not been established in complete paired runs.

## Throughput and explicit assumptions

- Cold-source generation: **1,029.11 output tokens/s** standalone AR, **1,027.61** DFlash4; both best measured at batch **64**. These 512-token rates already include generation prefill; **generation prefill is not added again**. Cold probes have a greater prefill/output ratio than full summaries and may be conservative for long generation.
- Fully warmed correction: **2,085.00 output tokens/s** standalone AR at batch **64**, **2,672.51** DFlash4 at batch **32**. This decode rate is used as a planning proxy for **all post-generation phases**, including short audit/anchor/repair calls. It is not a measured aggregate throughput for that heterogeneous work.
- Central useful capacity: **85%** of the modeled capacity, reserving 15% for overhead and imperfect utilization. Central additional uncached prefill: **25,000 tokens/s per GPU**, an explicit assumption.
- Central cache scenario: **75% of post-generation prompt tokens** reused; alternative column assumes **0%**. These cache fractions are **not measured whole-pipeline hit rates**. Generation stays cold in both columns.
- Fast scenario: measured decode rates, **40,000** additional prefill tokens/s, **90%** useful capacity and **90%** post-input cache.
- Cautious scenario: **65%** of measured decode rates, **15,000** additional prefill tokens/s, **80%** useful capacity and **no** post-input cache.

## Calculation

Let `N = 60,000,000`, `O_gen` and `O_post` be observed output tokens per paper, `I_post` additional prompt tokens per paper, `G_cold` measured cold generation tokens/s, `D_warm` measured cached correction tokens/s, `P` assumed uncached prefill tokens/s, `c` post-input cache fraction, `u` useful capacity and `f` throughput factor.

```text
GPUh_summary = N / (3600 × u) × O_gen / (G_cold × f)

GPUh_full = N / (3600 × u) × [
    O_gen / (G_cold × f)
  + O_post / (D_warm × f)
  + I_post × (1 − c) / P
]
```

The extra prefill term is needed because the correction benchmark fully cached its input. This additive work approximation does not claim that peak prefill and decode rates are attained concurrently. It also does not model every small-call latency. The scenario range is a sensitivity analysis, **not a confidence interval**.

## Scope and conclusions

Raw Ornith achieved **93.81% QA accuracy**, corrected Ornith **93.20%** on this cohort; the −0.62-point paired difference is statistically uncertain. Skipping full correction therefore saves substantial compute without an observed QA penalty here. It does **not** establish equal factual reliability or source-evidence validity.

DFlash4 improves modeled post-generation decode cost, while measured cold generation is essentially unchanged. Repeated repair prefill, cache reuse and retry volume matter more than a headline draft speedup. The next useful measurement is a larger complete-output throughput run with cold unique papers and measured aggregate cache reuse.

Estimates exclude **QA evaluation, PDF extraction/OCR, data acquisition, CPU-only processing, storage/I/O beyond the generic capacity allowance, and LoRA training**. Scheduler waiting does not consume allocated GPU-hours but affects calendar time. Do not multiply the experiment's 4.3611 total allocated GPU-hours by 60M/97: that allocation also contains benchmarking, a second generator, all QA controls, startup and idle GPUs. Likewise, the older repository FP8 scaling assumptions and RTX3090/Q8 workload describe different experiments.

These source papers are relatively short dataset texts. Longer deployment papers, different output lengths, repetitive generations, large-context cache eviction, small-request scheduling overhead and different failure rates can materially change the cost. Future batching/repair reductions are not counted as proven savings.

## Recompute and audit

```bash
python3 experiments/scientific_summaries/ornith_dflash_97/scaling_60m.py
```

Exact unrounded figures and assumptions are in [scaling_60m.json](scaling_60m.json) and [scaling_60m.csv](scaling_60m.csv). The measured source data are [phase_timings.csv](phase_timings.csv), [throughput.csv](throughput.csv), and [the full findings/results report](README.md).
