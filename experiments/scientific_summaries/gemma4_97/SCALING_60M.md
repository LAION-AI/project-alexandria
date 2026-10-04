# Planning estimate: 60 million Gemma scientific summaries on JUPITER

**English planning table, based on 97 held-out papers per model and the completed 2026-10-04 GH200 experiment.** Both checkpoints use BF16 native autoregressive decoding, no LoRA and no speculative draft. These figures extrapolate measured work and probe throughput; 60 million papers have not been processed.

## Summary lengths and central GPU-hour estimates

| Model | Mean final narrative tokens: raw | Mean final narrative tokens: corrected | 60M summaries only: GPU-hours | 60M summaries + full correction: GPU-hours |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT | 1,128 | 1,105 | 210,000 | 318,000–499,000 |
| Gemma 4 12B IT | 1,330 | 1,324 | 85,000 | 163,000–200,000 |

The full-correction interval uses central assumptions with **75% to 0% post-generation input reuse**; it is a cache sensitivity range, not a statistical interval. All central figures reserve 15% of modeled capacity for overhead and imperfect utilization. Generation-only costs include the recorded failed-generation retries and generation recovery. Full correction also includes both source-only self-audits, semantic correction, quote/field repairs, failed attempts and finite-schema rescue.

## Final artifact lengths

| Model | Variant | Mean narrative tokens | Median narrative tokens | Mean full summary JSON tokens | Mean narrative words |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 E4B IT | raw | 1,128.37 | 1078 | 2,544.18 | 775.46 |
| Gemma 4 E4B IT | corrected | 1,105.44 | 1071 | 2,009.06 | 759.30 |
| Gemma 4 12B IT | raw | 1,330.42 | 1319 | 2,198.49 | 918.26 |
| Gemma 4 12B IT | corrected | 1,324.35 | 1298 | 2,047.06 | 914.43 |

Token counts use each checkpoint's pinned native Gemma tokenizer, **without BOS/EOS or chat-template tokens**. The narrative is exactly the text supplied to the QA answerer. The full JSON includes all 19 fields, proof quotations and metadata, serialized with json.dumps(summary, ensure_ascii=False). It is a canonical artifact length; original generation whitespace can differ. Neither artifact-length column includes failed attempts or audits. Their additional token work is counted separately below. The two tokenizer.json files have the same SHA256.

## Wider planning sensitivity

| Model | Workflow | Fast scenario: GPU-hours | Cautious scenario: GPU-hours |
| --- | --- | --- | --- |
| Gemma 4 E4B IT | Summaries only | 198,000 | 343,000 |
| Gemma 4 E4B IT | Summaries + full correction | 258,000 | 849,000 |
| Gemma 4 12B IT | Summaries only | 80,000 | 139,000 |
| Gemma 4 12B IT | Summaries + full correction | 146,000 | 333,000 |

Fast: measured rates, 90% useful capacity, 90% post-input reuse, 40,000 additional uncached prefill tokens/s. Cautious: 65% of measured rates, 80% useful capacity, no post-input reuse, 15,000 additional uncached prefill tokens/s. The prefill capacities, cache fractions and utilization allowances are explicit assumptions, shared with the [previous Ornith 9B estimate](../ornith_dflash_97/SCALING_60M.md); they are not measured whole-pipeline hit rates.

## Observed inference work per finished paper

| Model | Work | Input tokens / paper | Output tokens / paper | Calls across 97 papers |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT | Generation and its recovery | 15,163.75 | 14,746.64 | 182 |
| Gemma 4 E4B IT | Additional audits, correction and repairs | 307,450.19 | 5,919.37 | 3094 |
| Gemma 4 12B IT | Generation and its recovery | 8,427.26 | 2,992.36 | 101 |
| Gemma 4 12B IT | Additional audits, correction and repairs | 61,657.35 | 6,154.80 | 609 |

Generation phases are generation and source_recovery_generation; every other recorded phase is assigned to post-generation work. These are API usage counts including failed outputs and all recorded recovery, divided by 97 completed papers. Generation-only estimates reuse the generation/recovery work observed inside the full workflow; a separate complete generation-only cohort was not timed. Some regeneration was triggered by a downstream correction failure, so the raw-only retry allowance can be conservative.

## Measured throughput used for the estimate

| Model | Cold-generation output tok/s | Repeat-input correction output tok/s | Batch |
| --- | --- | --- | --- |
| Gemma 4 E4B IT | 1,376.85 | 2,425.43 | 64 / 64 |
| Gemma 4 12B IT | 691.32 | 1,818.44 | 64 / 64 |

Each rate is measured on **one GH200** with actual 512-token probes and includes batch wall time. The generation probe starts cold; its prefill is already included. The correction probe repeats the full input with prefix caching, but capacity and eviction determine actual hits. Its rate is a proxy for all heterogeneous post-generation work, including very short repair requests; it is not an aggregate full-workflow measurement. The added uncached-prefill term approximates further cache misses and can be conservative because the repeat probe already includes residual prefill. Peak prefill and output rates are not assumed to occur simultaneously.

## Calculation

```text
N = 60,000,000
GPUh_raw = N / (3600 * useful_capacity) * O_generation / (cold_generation_rate * rate_factor)
GPUh_full = N / (3600 * useful_capacity) * [
    O_generation / (cold_generation_rate * rate_factor)
  + O_post / (repeat_correction_rate * rate_factor)
  + I_post * (1 - post_input_cache_fraction) / assumed_prefill_rate
]
```

## Interpretation and limits

**12B is cheaper under the observed workload despite lower generation tokens/s.** E4B consumed many more tokens in repeated or malformed generations and many more repair prompts. Its 775-word average raw narrative therefore does not imply a small generation bill. Starting with finite-schema decoding may lower those costs, but a complete fresh cohort under that revised recipe has not been measured and no such savings are counted here.

Neither Gemma correction arm showed a clear QA gain on the fixed 970-MCQ test: E4B 85.15% raw / 84.64% corrected; 12B 86.60% raw / 86.80% corrected. This does not establish equivalent factual reliability without correction.

These estimates exclude QA evaluation, PDF acquisition/OCR, CPU-only processing, LoRA training and storage/I/O beyond the generic capacity allowance. Do not multiply the complete benchmark's 4.6689 GPU-hours by 60M/97: that mixed allocation includes both models, probe sweeps, QA, startup and idle GPUs. Source texts in this sample are relatively short; different paper lengths, output targets, retry behavior and cache pressure can substantially change the cost. Small-call latency, scheduler and CPU orchestration bottlenecks are not modeled individually. The scenario limits are assumptions, not established bounds on a production run.

## Recompute and evidence

```bash
# Standard Python: recalculate costs from the preserved token counts.
python3 experiments/scientific_summaries/gemma4_97/scaling_60m.py
# CPU-only tokenizer recount; requires Transformers 5.18 and local pinned tokenizer files.
python experiments/scientific_summaries/gemma4_97/scaling_60m.py --tokenizer-root /path/to/local/models
```

Exact figures and assumptions: [scaling_60m.json](scaling_60m.json) / [scaling_60m.csv](scaling_60m.csv). Per-paper lengths and tokenizer hashes: [summary_token_lengths.csv](summary_token_lengths.csv) / [summary_token_lengths.json](summary_token_lengths.json). Inputs: [phase timings](phase_timings.csv), [throughput](throughput.csv), full preserved paper archives and [findings/results](README.md).
