# Findings and results: Gemma 4 E4B IT and 12B IT on JUPITER

**Complete: both models, 97 held-out papers each, 970 immutable MCQs per condition; Slurm job 2166972.** No LoRA or speculative draft is used.

## Models and shared protocol

[Gemma 4 E4B IT](https://huggingface.co/google/gemma-4-E4B-it) is the official small checkpoint requested as 4B IT: 4.5B effective parameters, approximately 8B including embeddings. [Gemma 4 12B IT](https://huggingface.co/google/gemma-4-12B-it) is the 11.95B unified checkpoint. Both use unquantized **BF16**, native vLLM Gemma implementations, tensor parallel size 1, two independent one-GPU replicas per model, and the native `gemma4` reasoning parser. Only text is supplied. The 12B staged training checkpoint is reused read-only after full checksum verification.

| Checkpoint | Pinned revision |
| --- | --- |
| google/gemma-4-E4B-it | ee0ef6023621cff504d758262d4e04895a5af4a2 |
| google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 |

Frozen protocol repository revision `5aac4b5ba2a78b20637e8ab960fe79d01ecd769a`; testset SHA256 `a0d5e5f99a0025c6cd8a5140a39a07a220ded6f37f04994500549886ffd83261`; initial system-prompt SHA256 `3643ce1ef340c4e63f451b85f9db75e8203bfc691e315e47523b3d08077b9bed`.

The same original 19-field summary prompt, full untruncated dataset text, temperature 0.2, top-p 0.95, 16,000 initial output tokens, 65,536 context and **thinking disabled** are used as in the [9B](../ornith_dflash_97/README.md) and [35B](../ornith35_dflash_97/README.md) runs. These sampling settings preserve the comparison protocol and differ from the Gemma model-card default temperature 1.0 / top-k 64. Generation, same-model source-only audit, full semantic correction, repository V3 field/quote repair, and post-correction audit use no questions, options, gold answers, QA rationale or QA evidence. Bounded recovery applies only to unusable outputs: up to 32,768 generation / 24,576 correction output tokens, presence penalty 1.5; original failures and all requests are retained. Optional citation rankings may be discarded only after bounded model repairs with full provenance. No papers are selected using QA scores.

### Conditional finite-schema recovery and protocol difference

The original common-protocol controller failed its completeness gate after both standard recovery rounds: E4B had 96 raw / 83 corrected artifacts and 12B had 97 raw / 94 corrected artifacts. Repeated full-output retries sometimes reached the token limit; E4B also emitted format commentary inside the scientific claims array. Additional **source-only** format rescue restores 14 E4B and 3 12B corrected outputs, including the one missing E4B raw artifact. No valid existing raw or corrected artifact is replaced. The original failed proposals and the original failed scheduler allocation are retained.

For the missing raw artifact, an already complete 19-field model proposal is reused after removing an explicitly non-scientific schema-note string from the claims array; all scientific claim objects and other fields are preserved. Corrected rescue starts with an existing V3 partial draft or complete source-only semantic correction. If no usable semantic correction exists, finite JSON-schema decoding requests the same 19 fields, at most 12 entries per grounded sequence / 5 claims, with 8,192 output tokens and no optional citation ranking. Temperature 0.2 / top-p 0.95 / thinking disabled are retained; presence penalty is 1.5 on schema-constrained rescue calls. Literal anchoring runs **after** field restoration with an enum of only supplied source-fragment indices or `drop`, followed by the unchanged strict validator and source-only self-audit. Every removed entry and original format note is preserved.

This extra fallback is a **documented postprocessing difference** from the earlier Ornith/Qwen runs. The comparison measures deployed pipelines, not an isolated model or exactly identical recovery recipe. The throughput probes were completed before this fallback and measure ordinary autoregressive JSON-object decoding; complete workflow times and accounting include the failed retry loops and rescue. The QA phase is resumed in a second allocation after all 97 raw/corrected outputs are verified, without regenerating any baseline answers or using QA to choose repairs.

The fixed answerer is **Qwen2.5-7B-Instruct BF16** revision `a09a35458c702b33eeacc393d103063234e8bc28`, vLLM 0.30.0: temperature 0.5, top-p 0.95, 100 output tokens, frequency/presence penalties 1.05, thinking disabled, four concurrent requests, at most five formatting attempts. The repository historical ASCII sanitizer and case-sensitive semicolon parser are unchanged; invalid answers count wrong. Ten previously completed conditions are reused only after exact source, context, question, option order, prompt and judge/protocol checks. **3,880 new answers** are scored; **13,580 total answers** are audited. Confidence intervals use 10,000 document-cluster bootstrap draws with seed 250219413.

All 97 evaluation papers and outputs remain outside the 1,000-paper training workspace. The throughput tuning set is the same 64 frozen non-holdout sources. Synthetic MCQs were authored by gpt-6-luna; the length-filtered convenience sample has no human benchmark validation or completeness check against publisher PDFs.

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
| google/gemma-4-E4B-it BF16 / raw summary | 826 | 85.15% | 82.68–87.53% | 0 |
| google/gemma-4-E4B-it BF16 / corrected summary | 821 | 84.64% | 82.06–87.22% | 0 |
| google/gemma-4-12B-it BF16 / raw summary | 840 | 86.60% | 84.23–88.97% | 0 |
| google/gemma-4-12B-it BF16 / corrected summary | 842 | 86.80% | 84.43–89.07% | 0 |

## Paired Gemma QA differences

| Comparison | Difference (percentage points) | 95% paired paper-bootstrap CI |
| --- | --- | --- |
| gemma4_e4b_corrected_minus_gemma4_e4b_raw | -0.52 | -1.75 to +0.72 |
| gemma4_e4b_raw_minus_ornith_raw | -8.66 | -10.72 to -6.49 |
| gemma4_e4b_corrected_minus_ornith_corrected | -8.56 | -10.93 to -6.08 |
| gemma4_e4b_raw_minus_ornith35_raw | -7.32 | -9.48 to -5.05 |
| gemma4_e4b_corrected_minus_ornith35_corrected | -8.04 | -10.41 to -5.57 |
| gemma4_e4b_raw_minus_qwen38_raw | -3.71 | -5.98 to -1.44 |
| gemma4_e4b_corrected_minus_qwen38_corrected | -5.15 | -7.53 to -2.78 |
| gemma4_e4b_raw_minus_original | -11.96 | -14.23 to -9.79 |
| gemma4_12b_corrected_minus_gemma4_12b_raw | +0.21 | -0.31 to +0.82 |
| gemma4_12b_raw_minus_ornith_raw | -7.22 | -9.28 to -5.26 |
| gemma4_12b_corrected_minus_ornith_corrected | -6.39 | -8.56 to -4.33 |
| gemma4_12b_raw_minus_ornith35_raw | -5.88 | -8.25 to -3.51 |
| gemma4_12b_corrected_minus_ornith35_corrected | -5.88 | -8.14 to -3.71 |
| gemma4_12b_raw_minus_qwen38_raw | -2.27 | -4.43 to -0.21 |
| gemma4_12b_corrected_minus_qwen38_corrected | -2.99 | -5.26 to -0.82 |
| gemma4_12b_raw_minus_original | -10.52 | -12.68 to -8.35 |
| gemma4_12b_raw_minus_gemma4_e4b_raw | +1.44 | -0.93 to +3.71 |

## Correction answer transitions

| Model | Wrong → correct | Correct → wrong | Correct → correct | Wrong → wrong |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT | 14 | 19 | 807 | 130 |
| Gemma 4 12B IT | 6 | 4 | 836 | 124 |

## Throughput: every measured batch

One GH200 per measured runtime, BF16 native autoregressive decoding, continuous batching, prefix cache, chunked prefill (8,192 max batched tokens), compilation/CUDA graphs, 64 scheduled sequences and GPU memory utilization 0.90. Runtime uses Python 3.11.15, vLLM 0.30.0, PyTorch 2.13.0 and Transformers 5.18.0 on CUDA 13. Exact commands, selected kernels, GPU memory and before/after metrics are archived. All probes are bounded to 512 output tokens. Tokens/s divides actual emitted output tokens by batch wall time including prefill. Cold resets the prefix cache; warm repeats the whole input but hits depend on capacity and eviction. A first correction caching only the source can be slower. These probes do not measure complete-summary pipeline throughput.

| Model | Batch | Generation cold tok/s | Generation repeat tok/s | Correction cold tok/s | Correction repeat tok/s |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 E4B IT | 1 | 86.86 | 90.83 | 68.06 | 72.11 |
| Gemma 4 E4B IT | 4 | 243.66 | 279.68 | 196.06 | 233.48 |
| Gemma 4 E4B IT | 8 | 375.85 | 550.05 | 341.02 | 454.61 |
| Gemma 4 E4B IT | 16 | 592.59 | 944.43 | 524.33 | 815.32 |
| Gemma 4 E4B IT | 32 | 971.29 | 1769.54 | 785.11 | 1501.50 |
| Gemma 4 E4B IT | 64 | 1376.85 | 2998.56 | 1008.90 | 2425.43 |
| Gemma 4 12B IT | 1 | 58.09 | 62.69 | 47.64 | 52.90 |
| Gemma 4 12B IT | 4 | 162.33 | 209.02 | 131.79 | 181.19 |
| Gemma 4 12B IT | 8 | 248.74 | 410.32 | 213.49 | 356.31 |
| Gemma 4 12B IT | 16 | 374.84 | 728.96 | 306.59 | 652.63 |
| Gemma 4 12B IT | 32 | 549.88 | 1304.81 | 413.40 | 1167.53 |
| Gemma 4 12B IT | 64 | 691.32 | 2049.09 | 488.25 | 1818.44 |

The production selector caps concurrent complete-paper workflows at 32 per replica. Each paper stays on its assigned replica across generation, review and correction. It chooses batches using the tuning probes, before any evaluation QA. The saved recipe and actual cohort config identify exact concurrency.

## Complete-output request times and token usage

| Model | Phase | Calls | Output tokens | Input tokens | Mean seconds | Median seconds | P95 seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 E4B IT | generation | 162 | 1221011 | 1303336 | 127.71 | 54.39 | 305.90 |
| Gemma 4 E4B IT | raw_review | 136 | 75026 | 1218555 | 9.25 | 6.96 | 15.05 |
| Gemma 4 E4B IT | semantic_correction | 87 | 241772 | 976707 | 49.73 | 46.30 | 74.65 |
| Gemma 4 E4B IT | field_repair | 121 | 11159 | 827971 | 2.23 | 1.17 | 7.02 |
| Gemma 4 E4B IT | source_anchor_selection | 576 | 32450 | 10540436 | 1.60 | 1.49 | 2.88 |
| Gemma 4 E4B IT | source_recovery_single_anchor | 1156 | 10044 | 8418958 | 0.16 | 0.14 | 0.31 |
| Gemma 4 E4B IT | source_recovery_field_repair | 317 | 44737 | 2375083 | 1.91 | 1.47 | 4.98 |
| Gemma 4 E4B IT | source_recovery_corrected_review | 59 | 38737 | 543104 | 9.09 | 7.34 | 19.19 |
| Gemma 4 E4B IT | corrected_review | 62 | 32282 | 457341 | 11.37 | 10.72 | 20.20 |
| Gemma 4 E4B IT | source_recovery_generation | 20 | 209413 | 167548 | 156.51 | 39.55 | 518.89 |
| Gemma 4 E4B IT | source_recovery_raw_review | 18 | 8738 | 160784 | 5.72 | 5.28 | 9.39 |
| Gemma 4 E4B IT | source_recovery_semantic_correction | 20 | 58162 | 226322 | 33.67 | 31.70 | 56.33 |
| Gemma 4 E4B IT | schema_rescue_anchor | 514 | 4546 | 3815611 | 0.20 | 0.17 | 0.39 |
| Gemma 4 E4B IT | schema_rescue_corrected_review | 27 | 14274 | 250285 | 6.33 | 5.61 | 11.53 |
| Gemma 4 E4B IT | schema_rescue_semantic_correction | 1 | 2252 | 11511 | 28.30 | 28.30 | 28.30 |
| Gemma 4 12B IT | generation | 101 | 290259 | 817444 | 84.29 | 75.60 | 111.30 |
| Gemma 4 12B IT | raw_review | 98 | 9678 | 821747 | 3.56 | 3.03 | 5.63 |
| Gemma 4 12B IT | semantic_correction | 104 | 341884 | 1097362 | 90.19 | 72.96 | 276.97 |
| Gemma 4 12B IT | field_repair | 95 | 5428 | 634918 | 3.10 | 0.83 | 20.02 |
| Gemma 4 12B IT | source_anchor_selection | 134 | 5913 | 2035191 | 2.81 | 2.60 | 6.76 |
| Gemma 4 12B IT | corrected_review | 85 | 7508 | 688824 | 3.83 | 3.33 | 7.11 |
| Gemma 4 12B IT | source_recovery_field_repair | 19 | 5830 | 161556 | 5.57 | 6.01 | 9.01 |
| Gemma 4 12B IT | source_recovery_corrected_review | 9 | 747 | 86380 | 1.83 | 1.73 | 2.35 |
| Gemma 4 12B IT | source_recovery_semantic_correction | 15 | 212197 | 140676 | 265.35 | 458.38 | 479.46 |
| Gemma 4 12B IT | source_recovery_single_anchor | 25 | 205 | 169440 | 0.31 | 0.22 | 0.60 |
| Gemma 4 12B IT | schema_rescue_semantic_correction | 3 | 7216 | 27458 | 47.94 | 46.18 | 53.37 |
| Gemma 4 12B IT | schema_rescue_anchor | 19 | 161 | 96540 | 0.35 | 0.29 | 0.51 |
| Gemma 4 12B IT | schema_rescue_corrected_review | 3 | 249 | 20671 | 2.08 | 2.04 | 2.25 |

These are individual full-output request latencies at deployed concurrency, including queued work, not sums of GPU active time or complete cohort wall times. First-pass timers precede source recovery; reconciled counts in complete.json do not extend the original timer. Use scheduler accounting for the complete benchmark bill and trace timestamps for recovery intervals.

## Cohort times including source-only recovery

| Model | First-pass seconds | First-pass raw / 97 | First-pass corrected / 97 | Recovered papers | Complete source-only trace span seconds |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 E4B IT | 878.01 | 86 | 50 | 47 | 3208.75 |
| Gemma 4 12B IT | 859.15 | 97 | 85 | 12 | 3035.26 |

The complete trace span runs from the first recorded model request start (response trace timestamp minus recorded request latency) to the last source-only document write. It includes generation, audits, correction and all bounded recovery, but excludes model startup, throughput probes, initial token-count preflight and QA. It is an observed workflow span, not summed GPU active time. Reconciled first-pass completion files retain the original pre-recovery clock; the separate first-pass counts here prevent treating that clock as time to finish all 97 corrected papers.

## Narrative lengths

| Condition | Mean words | Median words | Minimum | Maximum |
| --- | --- | --- | --- | --- |
| ornith_dflash_raw | 2547.97 | 2273 | 1018 | 7440 |
| ornith_dflash_corrected | 2302.34 | 2111 | 790 | 5975 |
| qwen38_fp8_raw | 1198.89 | 1153 | 766 | 1914 |
| qwen38_fp8_corrected | 1186.56 | 1153 | 751 | 1914 |
| ornith35_dflash_raw | 1557.20 | 1421 | 766 | 5186 |
| ornith35_dflash_corrected | 1532.34 | 1406 | 744 | 4882 |
| gemma4_e4b_raw | 775.46 | 753 | 569 | 1106 |
| gemma4_e4b_corrected | 759.30 | 729 | 566 | 1071 |
| gemma4_12b_raw | 918.26 | 906 | 617 | 1211 |
| gemma4_12b_corrected | 914.43 | 896 | 617 | 1211 |

## Source-only self-assessments

| Condition | Valid / 97 | Failed / 97 | Pass | Needs correction | Issues | Missing central facts |
| --- | --- | --- | --- | --- | --- | --- |
| gemma4_e4b_raw | 42 | 55 | 0 | 42 | 57 | 43 |
| gemma4_e4b_corrected | 51 | 46 | 0 | 51 | 86 | 55 |
| gemma4_12b_raw | 96 | 1 | 95 | 1 | 2 | 1 |
| gemma4_12b_corrected | 97 | 0 | 96 | 1 | 2 | 1 |

| Condition | Rubric | Valid n | Mean / 5 | Histogram |
| --- | --- | --- | --- | --- |
| gemma4_e4b_raw | factual_accuracy | 42 | 4.952 | {"4": 2, "5": 40} |
| gemma4_e4b_raw | coverage | 42 | 4.929 | {"4": 3, "5": 39} |
| gemma4_e4b_raw | clarity | 42 | 4.976 | {"4": 1, "5": 41} |
| gemma4_e4b_raw | faithfulness | 42 | 4.952 | {"4": 2, "5": 40} |
| gemma4_e4b_raw | scientific_precision | 42 | 4.905 | {"4": 4, "5": 38} |
| gemma4_e4b_corrected | factual_accuracy | 51 | 4.863 | {"4": 7, "5": 44} |
| gemma4_e4b_corrected | coverage | 51 | 4.843 | {"4": 8, "5": 43} |
| gemma4_e4b_corrected | clarity | 51 | 5.000 | {"5": 51} |
| gemma4_e4b_corrected | faithfulness | 51 | 4.863 | {"4": 7, "5": 44} |
| gemma4_e4b_corrected | scientific_precision | 51 | 4.843 | {"4": 8, "5": 43} |
| gemma4_12b_raw | factual_accuracy | 96 | 4.979 | {"3": 1, "5": 95} |
| gemma4_12b_raw | coverage | 96 | 5.000 | {"5": 96} |
| gemma4_12b_raw | clarity | 96 | 5.000 | {"5": 96} |
| gemma4_12b_raw | faithfulness | 96 | 4.979 | {"3": 1, "5": 95} |
| gemma4_12b_raw | scientific_precision | 96 | 4.969 | {"2": 1, "5": 95} |
| gemma4_12b_corrected | factual_accuracy | 97 | 4.969 | {"2": 1, "5": 96} |
| gemma4_12b_corrected | coverage | 97 | 5.000 | {"5": 97} |
| gemma4_12b_corrected | clarity | 97 | 5.000 | {"5": 97} |
| gemma4_12b_corrected | faithfulness | 97 | 4.969 | {"2": 1, "5": 96} |
| gemma4_12b_corrected | scientific_precision | 97 | 4.969 | {"2": 1, "5": 96} |

Self-assessments are fallible same-model judgments, not independent human factual labels. Failed audits are unavailable; means exclude failed assessments without imputing scores. All papers remain in QA. Literal quote/schema validation does not prove semantic entailment. Any actually emitted reasoning fields are preserved; thinking was disabled and hidden reasoning is not invented.

## Allocation, audit and evidence

The complete resumed benchmark uses **4.6689 allocated GPU-hours** across **4202 seconds of summed scheduler runtime**. Each allocation uses four GPUs. The original controller allocation failed its completeness guard; the resumed QA allocation completed. Source-only schema-rescue steps overlap the original allocation and are counted once.

| Slurm job | State | Allocated seconds | GPUs | GPU-hours |
| --- | --- | --- | --- | --- |
| 2166049 | FAILED | 3684 | 4 | 4.0933 |
| 2166972 | COMPLETED | 518 | 4 | 0.5756 |

All four new cohorts contain 97 outputs each. Final audit verifies **13,580 QA prompt hashes**, **13,580 parsed predictions**, immutable gold/option order and every corrected literal evidence ledger.

## Conclusions

**Gemma 4 E4B IT:** raw 826/970 (85.15%); corrected 821/970 (84.64%). Best cold-generation probe 1376.85 tok/s (batch 64); best repeat-input correction 2425.43 tok/s (batch 64).

**Gemma 4 12B IT:** raw 840/970 (86.60%); corrected 842/970 (86.80%). Best cold-generation probe 691.32 tok/s (batch 64); best repeat-input correction 1818.44 tok/s (batch 64).

Compare paired confidence intervals, narrative lengths, invalids and self-audit failure rates before choosing a deployment. Correction can change answerability and factual reliability differently. These complete deployed pipelines are not length-matched, and no independent human faithfulness study or trained-LoRA gain is established. See the [combined GH200 comparison](../ORNITH_DFLASH_GH200_RESULTS.md).

## Files and verification

`scores.csv` contains fourteen scores; `paper_scores.csv` contains every paper/condition and time; `throughput.csv` contains 48 probes; `phase_timings.csv` contains all full-output phase aggregates; `self_audit_scores.csv` includes 388 valid/failed audits; `paper_names.tsv` identifies all 97 papers. All 194 per-model paper archives retain raw/corrected summaries, audit findings, correction proposals, actual requests/responses, emitted reasoning fields and failed attempts/recovery. Full QA, runtime logs, metrics, code/model fingerprints and accounting are preserved. Weights/caches are excluded. Machine-specific HPC paths in snapshots require adaptation before reproducing inference elsewhere.

```bash
python3 experiments/scientific_summaries/gemma4_97/verify_results.py
```

Rebuild with `package_results.py --run-dir /path/to/run`. All packaged files have SHA256 checksums. Upstream model/source attribution and terms remain unchanged.
