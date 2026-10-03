# Findings and results: Ornith-1.5-35B-A3B + DFlash on JUPITER

**Complete: 97 held-out papers, 970 immutable MCQs; Slurm job 2165579.** Raw accuracy **897/970 (92.47%)**; corrected accuracy **899/970 (92.68%)**. QA summaries were produced with **dflash4**. No LoRA is used.

## Model and protocol

[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) is the 35B mixture-of-experts target (approximately 3B parameters active per token). [Ornith-1.5-35B-A3B-DFlash](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-DFlash) is its separate approximately 0.386B draft. Both use **BF16**; inference uses one target/draft pair per GH200, tensor parallel size 1. The draft is not a standalone summarizer.

Target revision: `10fbf86fed7ecee4a061f8b499a618f46001cac1`. Draft revision: `942d20fee84d04c9ecbf18e9c6fc4144914a1d5d`. Repository protocol revision: `5aac4b5ba2a78b20637e8ab960fe79d01ecd769a`. Testset SHA256: `a0d5e5f99a0025c6cd8a5140a39a07a220ded6f37f04994500549886ffd83261`.

The [same frozen test set](../data/README.md), original 19-field prompt, full untruncated source, no-thinking mode, temperature 0.2, top-p 0.95, 16,000 initial output tokens and 65,536 maximum context are used. Source-only same-model audits, full semantic correction, V3 field/quote repair and bounded source-only recovery follow the [completed 9B protocol](../ornith_dflash_97/README.md). Failed long outputs may use 32,768 generation / 24,576 correction output tokens with presence penalty 1.5. Unusable optional citation rankings may be discarded only after bounded repairs, with original values/provenance retained. Scientific narrative is validated separately.

Generators/reviewers/correctors receive no questions, options, gold keys, QA rationales or QA evidence. No papers are dropped for quality. All evaluation outputs remain outside the 1,000-paper training workspace and must not enter LoRA training.

The fixed student is **Qwen2.5-7B-Instruct BF16**, pinned revision `a09a35458c702b33eeacc393d103063234e8bc28`, vLLM 0.30.0, temperature 0.5, top-p 0.95, 100 output tokens, frequency/presence penalties 1.05, no thinking, four concurrent requests, five formatting attempts maximum. Exact historical ASCII sanitizer, semicolon parser, source/question/option order and prompt hashes are preserved. Eight prior comparison conditions are reused after exact validation; **1,940 new answers** are scored. All **9,700** combined answers are audited.

The synthetic questions were authored by gpt-6-luna. This is a convenience sample of relatively short dataset texts, with no independent completeness check against publisher PDFs or human benchmark validation. Confidence intervals use 10,000 document-cluster bootstrap draws, seed 250219413, keeping all ten questions per paper together. Paired differences use the same paper clusters.

## All accuracy scores

| Condition | Correct / 970 | Accuracy | 95% paper-bootstrap CI | Invalid |
| --- | --- | --- | --- | --- |
| No context | 596 | 61.44% | 57.94–64.95% | 4 |
| Full paper text | 942 | 97.11% | 95.98–98.14% | 0 |
| Existing Gemini summary | 838 | 86.39% | 83.92–88.76% | 0 |
| Qwen3.8-27B AutoRound mixed INT4 / repository repaired summary | 870 | 89.69% | 87.53–91.75% | 0 |
| Ornith-1.5-9B BF16 + DFlash / raw summary | 910 | 93.81% | 91.86–95.57% | 0 |
| Ornith-1.5-9B BF16 + DFlash / corrected summary | 904 | 93.20% | 91.24–94.95% | 0 |
| Qwen3.8-27B official FP8 / raw summary | 862 | 88.87% | 86.49–91.13% | 0 |
| Qwen3.8-27B official FP8 / corrected summary | 871 | 89.79% | 87.63–91.86% | 0 |
| Ornith-1.5-35B-A3B BF16 + DFlash / raw summary | 897 | 92.47% | 90.31–94.43% | 0 |
| Ornith-1.5-35B-A3B BF16 + DFlash / corrected summary | 899 | 92.68% | 90.62–94.54% | 0 |

### Paired comparisons involving the new 35B model

| Comparison | Difference (percentage points) | 95% paired CI |
| --- | --- | --- |
| ornith35_corrected_minus_ornith35_raw | +0.21 | -0.41 to +0.82 |
| ornith35_raw_minus_ornith_raw | -1.34 | -2.89 to +0.21 |
| ornith35_corrected_minus_ornith_corrected | -0.52 | -2.37 to +1.24 |
| ornith35_raw_minus_qwen38_raw | +3.61 | +1.75 to +5.46 |
| ornith35_corrected_minus_qwen38_corrected | +2.89 | +0.93 to +4.85 |
| ornith35_raw_minus_original | -4.64 | -6.70 to -2.78 |

### Answer changes after correction

| Correct → correct | Wrong → wrong | Wrong → correct | Correct → wrong |
| --- | --- | --- | --- |
| 892 | 66 | 7 | 5 |

These are fixed-student answer transitions, not independent counts of factual errors in summaries. A small difference with a paired interval including zero is uncertain.

## Every measured throughput configuration

One GH200 per runtime. Native vLLM DFlash / MoE kernels, continuous batching, prefix caching, chunked prefill (8,192 max batched tokens), compilation/CUDA graphs, 64 maximum scheduled sequences and GPU memory utilization 0.90. GPU-reported memory is 97,871 MiB. Exact actual server commands/backend selections and before/after metrics are archived.

All 512-token probes use 64 frozen **non-holdout** source-only papers, disjoint from evaluation. Tokens/s is actual emitted output tokens divided by batch wall time, including the relevant prefill. The warm round repeats its entire input; actual cache hits depend on capacity and eviction. It can be more optimistic than a first correction caching only the source prefix. These are short probes, not complete-summary timings. `ar` is the same 35B target without a draft; DFlash numbers denote configured speculative tokens.

| Runtime | Batch | Generation cold tok/s | Generation warm tok/s | Correction cold tok/s | Correction warm tok/s |
| --- | --- | --- | --- | --- | --- |
| ar | 1 | 201.31 | 207.40 | 178.58 | 209.64 |
| ar | 4 | 374.10 | 536.85 | 357.33 | 520.65 |
| ar | 8 | 453.37 | 771.47 | 473.12 | 736.98 |
| ar | 16 | 644.09 | 1187.31 | 607.12 | 1069.15 |
| ar | 32 | 873.79 | 1559.54 | 735.55 | 1438.99 |
| ar | 64 | 1082.18 | 1151.90 | 779.04 | 780.38 |
| dflash4 | 1 | 123.99 | 122.91 | 105.11 | 117.80 |
| dflash4 | 4 | 182.72 | 263.93 | 310.18 | 286.72 |
| dflash4 | 8 | 347.79 | 398.08 | 483.61 | 635.12 |
| dflash4 | 16 | 469.40 | 540.93 | 497.49 | 548.82 |
| dflash4 | 32 | 514.47 | 643.27 | 659.37 | 667.24 |
| dflash4 | 64 | 560.68 | 541.47 | 723.85 | 716.27 |
| dflash8 | 1 | 142.52 | 146.12 | 183.94 | 143.03 |
| dflash8 | 4 | 203.74 | 358.64 | 413.78 | 383.50 |
| dflash8 | 8 | 451.94 | 531.37 | 591.01 | 494.17 |
| dflash8 | 16 | 473.74 | 496.19 | 586.16 | 595.91 |
| dflash8 | 32 | 464.56 | 522.56 | 633.02 | 605.97 |
| dflash8 | 64 | 510.73 | 486.92 | 633.12 | 631.95 |
| dflash15 | 1 | 114.60 | 99.60 | 166.46 | 178.02 |
| dflash15 | 4 | 211.71 | 311.09 | 443.18 | 433.70 |
| dflash15 | 8 | 244.62 | 244.61 | 443.02 | 444.92 |
| dflash15 | 16 | 349.21 | 330.72 | 455.07 | 466.09 |
| dflash15 | 32 | 388.67 | 388.94 | 446.50 | 501.39 |
| dflash15 | 64 | 361.01 | 407.82 | 493.34 | 488.66 |

The production selector chooses the strongest measured **DFlash** configuration by the harmonic mean of best cold-generation / repeat-input correction throughput; standalone AR is reported separately and not silently selected for DFlash QA. No complete-cohort standalone AR QA arm was run for this new model, so the benchmark does not establish paired AR/DFlash accuracy equivalence.

## Complete-output API request timing and token work

| Phase | Calls | Prompt tokens | Output tokens | Mean s | Median s | P95 s |
| --- | --- | --- | --- | --- | --- | --- |
| generation | 127 | 1043753 | 737512 | 95.48 | 66.41 | 212.04 |
| raw_review | 100 | 1043840 | 11337 | 3.35 | 2.43 | 8.44 |
| semantic_correction | 101 | 1286010 | 440917 | 46.12 | 42.26 | 78.78 |
| field_repair | 21 | 171912 | 2774 | 2.57 | 1.35 | 6.89 |
| source_anchor_selection | 523 | 9593504 | 36678 | 2.07 | 1.72 | 3.90 |
| corrected_review | 88 | 888232 | 7216 | 1.98 | 1.69 | 4.17 |
| source_recovery_semantic_correction | 6 | 75102 | 30365 | 31.67 | 29.11 | 60.27 |
| source_recovery_single_anchor | 315 | 2228464 | 3174 | 0.57 | 0.46 | 0.92 |
| source_recovery_corrected_review | 9 | 94053 | 738 | 1.03 | 0.86 | 2.13 |
| source_recovery_generation | 2 | 17131 | 9161 | 44.39 | 44.39 | 59.63 |
| source_recovery_raw_review | 2 | 21513 | 164 | 1.54 | 1.54 | 2.37 |
| source_recovery_field_repair | 12 | 98239 | 8076 | 3.95 | 2.54 | 22.39 |

These are individual API-request latencies and include failures/retries and concurrent scheduling. Adding them does not produce cohort wall time. Repeated prompt tokens are not equivalent to uncached prefill. The first-pass counts are retained separately from recovered final counts; full allocation accounting also includes startup and idle GPU time.

## Length and quality-audit limitations

| Condition | Mean words | Median | Minimum | Maximum |
| --- | --- | --- | --- | --- |
| ornith_dflash_raw | 2547.97 | 2273 | 1018 | 7440 |
| ornith_dflash_corrected | 2302.34 | 2111 | 790 | 5975 |
| qwen38_fp8_raw | 1198.89 | 1153 | 766 | 1914 |
| qwen38_fp8_corrected | 1186.56 | 1153 | 751 | 1914 |
| ornith35_dflash_raw | 1557.20 | 1421 | 766 | 5186 |
| ornith35_dflash_corrected | 1532.34 | 1406 | 744 | 4882 |

These pipelines share the initial prompt but **summary lengths are not equalized**. The comparison measures deployed pipelines with different model architectures and sizes, rather than an isolated parameter-count effect. Qwen27B is FP8; both Ornith models are BF16. Faster output per second is not the same as faster papers per second when narrative/retry lengths differ.

| Self-audit | Scored / 97 | Failed | Pass | Needs correction | Issues | Missing facts |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 95 | 2 | 90 | 5 | 7 | 0 |
| corrected | 97 | 0 | 97 | 0 | 0 | 0 |

| Self-audit | factual accuracy / 5 | coverage / 5 | clarity / 5 | faithfulness / 5 | scientific precision / 5 |
| --- | --- | --- | --- | --- | --- |
| raw | 4.979 | 5.000 | 5.000 | 4.979 | 4.968 |
| corrected | 5.000 | 5.000 | 5.000 | 5.000 | 4.990 |

Same-model audits are fallible and are **not independent human labels**. Means use valid scored audits only; failed assessments are unavailable. Mechanical schema/literal-quote validation does not prove entailment or establish a clean semantic audit. Actual emitted reasoning fields are preserved when present; thinking was disabled and no hidden reasoning is invented.

## Allocation and evidence

Slurm job **2165579**: **2278 seconds**, **4 allocated GPUs**, **2.5311 allocated GPU-hours**, final state **COMPLETED**. This includes benchmark/model startup, generation, recovery, QA and idle capacity; it is not a per-model active-inference or 60-million-paper bill.

All 97 raw and 97 corrected new summaries are present. The final audit verifies **9,700 QA prompt hashes**, **9,700 parsed predictions**, source checksums, immutable questions and corrected literal evidence ledgers. All per-paper outputs, findings/correction proposals, actual request/response journals, retry/recovery histories, benchmark metrics, server logs, code and model fingerprints are preserved. The earlier 9B/Qwen evidence remains in its separate linked package.

## Conclusions

**DFlash does not speed up this measured 35B workload.** The strongest speculative configuration, DFlash4, reaches 51.8% of best standalone AR generation throughput and 49.8% of repeat-input correction throughput, comparing each runtime at its own strongest measured batch. This result differs from the 9B correction speedup. DFlash4 accepts 65.28% of proposed draft tokens (2.586 accepted tokens per draft). Acceptance alone does not imply a speedup. Server logs select the native Triton MoE backend on Hopper; larger speculative windows and reduced KV-cache capacity can add overhead, but no kernel/cache ablation was run to isolate the cause. All counters are retained in `speculative_metrics.json`.

The 35B model scores **92.47% raw** and **92.68% corrected**. Read paired confidence intervals and output lengths together before attributing differences to model size. The [combined GH200 comparison](../ORNITH_DFLASH_GH200_RESULTS.md) puts the two Ornith models beside Qwen27B and separates production QA settings from the best short-probe throughput. No trained-LoRA gain, human factual superiority, or full-output speedup beyond the measured workload is established.

## Files and CPU verification

`scores.csv` contains all ten scores; `paper_scores.csv` contains every paper/condition score and timing; `throughput.csv` contains all 96 probe observations; `phase_timings.csv` contains complete-output call aggregates; `self_audit_scores.csv` includes all 194 valid/failed assessments; `paper_names.tsv` identifies all papers. `artifacts/` contains full QA, model traces, audits, runtime logs, metrics and accounting. `provenance/` holds pinned identities and source snapshots. Model weights/caches are excluded.

```bash
python3 experiments/scientific_summaries/ornith35_dflash_97/verify_results.py
```

Rebuild from the preserved run with `package_results.py --run-dir /path/to/run`. Original HPC scripts retain machine-specific paths and require adaptation before inference elsewhere. Upstream source/model attribution and terms remain unchanged. All packaged files have SHA256 checksums.
