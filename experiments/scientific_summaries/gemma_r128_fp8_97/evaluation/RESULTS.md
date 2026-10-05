# Gemma 4 12B IT rank-128: thinking ablation and inference optimization

Same Qwen-distilled adapter, 97 frozen papers and 970 immutable MCQs. No semantic correction. Format failures count as ten wrong answers. The thinking reference is reused verbatim from the audited completed experiment.

| Condition | Correct / 970 | QA accuracy | Failed papers | Mean narrative words | Mean narrative tokens | Generation seconds | Completion tokens/s/GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rank 128 / live BF16 LoRA / thinking (cached) | 897 | 92.47% | 2 | 2469 | 3753 | 4302.4 | 500.9 |
| Rank 128 / live BF16 LoRA / no thinking / concurrency 16 | 906 | 93.40% | 0 | 2096 | 3242 | 1397.2 | 428.9 |
| Rank 128 / merged BF16 / no thinking / tuned deployment | 911 | 93.92% | 0 | 1945 | 3002 | 745.9 | 676.8 |
| Rank 128 / merged FP8 / no thinking / tuned deployment | 914 | 94.23% | 0 | 2000 | 3097 | 545.4 | 939.9 |

## Paired QA differences (95% paper-bootstrap intervals)

- no_thinking_matched_minus_thinking: +0.93 percentage points (95% CI -1.86 to +4.33).
- no_thinking_merged_minus_thinking: +1.44 percentage points (95% CI -1.24 to +4.85).
- no_thinking_fp8_minus_thinking: +1.75 percentage points (95% CI -1.03 to +5.15).
- no_thinking_merged_minus_matched_no_thinking: +0.52 percentage points (95% CI -0.82 to +1.96).
- no_thinking_fp8_minus_matched_no_thinking: +0.82 percentage points (95% CI -0.62 to +2.37).

## Scaling on GH200 GPUs

**Paraphrase-compliance qualification:** these figures count structurally finished outputs. All generated no-thinking narratives exceed the six-word source-copy tolerance. The cost of enforcing the five-word paraphrasing limit is not yet measured.

| Condition | Papers | Measured-rate GPU-hours | Planning GPU-hours (85% useful capacity and observed output yield) |
| --- | ---: | ---: | ---: |
| Rank 128 / live BF16 LoRA / thinking (cached) | 38,000,000 | 468,193 | 562,411 |
| Rank 128 / live BF16 LoRA / thinking (cached) | 60,000,000 | 739,251 | 888,017 |
| Rank 128 / live BF16 LoRA / thinking (cached) | 98,000,000 | 1,207,444 | 1,450,428 |
| Rank 128 / live BF16 LoRA / no thinking / concurrency 16 | 38,000,000 | 152,043 | 178,874 |
| Rank 128 / live BF16 LoRA / no thinking / concurrency 16 | 60,000,000 | 240,068 | 282,433 |
| Rank 128 / live BF16 LoRA / no thinking / concurrency 16 | 98,000,000 | 392,111 | 461,308 |
| Rank 128 / merged BF16 / no thinking / tuned deployment | 38,000,000 | 81,171 | 95,495 |
| Rank 128 / merged BF16 / no thinking / tuned deployment | 60,000,000 | 128,164 | 150,782 |
| Rank 128 / merged BF16 / no thinking / tuned deployment | 98,000,000 | 209,335 | 246,277 |
| Rank 128 / merged FP8 / no thinking / tuned deployment | 38,000,000 | 59,347 | 69,820 |
| Rank 128 / merged FP8 / no thinking / tuned deployment | 60,000,000 | 93,705 | 110,241 |
| Rank 128 / merged FP8 / no thinking / tuned deployment | 98,000,000 | 153,052 | 180,061 |

These extrapolations use actual complete generation, including prefill, native tokenization and all bounded format retries. They exclude QA, training, server startup, semantic correction and source acquisition. The 85% capacity factor is a planning assumption. Output-yield normalization assumes a comparable future paper mix; it does not guarantee recovery of difficult failed papers. Use four independent model replicas per Jupiter node to avoid paying for idle GPUs. Short capped probes select deployment settings and are never used as full-summary costs.

## Optimization protocol

64 deterministic non-held-out training papers across ten domains; concurrency 16/32/64; 512-token partial outputs with cold source prefix caches. Runtime selection is made before held-out QA. Merged BF16 and FP8 are evaluated on all 97 papers because merging and quantization can change outputs. Unsupported backend/KV-cache configurations are retained in optimization logs. The original baseline already uses FlashAttention 4, CUDA graphs, prefix caching and chunked prefill.

QA tests answerability under the pinned Qwen2.5-7B student and synthetic MCQs. It is not an independent human factuality assessment. Summary length and formatting failure rate are reported alongside accuracy.

## Source-copy overlap audit

Exact normalized contiguous word overlap against each complete paper source. Five words are allowed, six are borderline, seven or more are flagged. Narrative prose, evidence quotes and bibliographic metadata are reported separately. Coverage counts each summary word once per threshold within each fragment. Separate statements are never concatenated to create matches.

| Condition | Audited outputs | Narrative ≤5 | Narrative exactly 6 | Narrative ≥7 | Longest narrative match | Mean narrative coverage ≥6 | Evidence ≥7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| thinking_reference | 95 | 0 | 0 | 95 | 42 | 19.31% | 2 |
| no_thinking_matched | 97 | 0 | 0 | 97 | 61 | 14.91% | 34 |
| no_thinking_merged | 97 | 0 | 0 | 97 | 34 | 15.74% | 36 |
| no_thinking_fp8 | 97 | 0 | 0 | 97 | 32 | 15.23% | 34 |

Primary counts use whitespace-delimited words after Unicode normalization and punctuation stripping; mathematical expressions, decimal numbers and hyphenated terms are not split into artificial extra words. A separate punctuation-split diagnostic is retained in the JSONL. NFKC normalization, casefolding, soft-hyphen removal and PDF line-end word joining are used. Offsets refer to normalized text and word positions. Technical names and ordinary scientific phrases can match literally; lexical overlap alone does not establish plagiarism. All output flags remain visible and no paper is removed from QA. Reasoning traces, instructions and reviewer verdicts are not summary prose. Bibliographic names and titles are included in the metadata diagnostics, not silently mixed into paraphrase statistics.
