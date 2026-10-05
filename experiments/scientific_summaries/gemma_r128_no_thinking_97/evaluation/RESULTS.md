# Gemma 4 12B IT rank-128: thinking ablation and inference optimization

Same Qwen-distilled adapter, 97 frozen papers and 970 immutable MCQs. No semantic correction. Format failures count as ten wrong answers. The thinking reference is reused verbatim from the audited completed experiment.

| Condition | Correct / 970 | QA accuracy | Failed papers | Mean narrative words | Mean narrative tokens | Generation seconds | Completion tokens/s/GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rank 128 / live BF16 LoRA / thinking (cached) | 897 | 92.47% | 2 | 2469 | 3753 | 4302.4 | 500.9 |
| Rank 128 / live BF16 LoRA / no thinking / concurrency 16 | 906 | 93.40% | 0 | 2096 | 3242 | 1397.2 | 428.9 |
| Rank 128 / merged BF16 / no thinking / tuned deployment | 911 | 93.92% | 0 | 1945 | 3002 | 745.9 | 676.8 |

## Paired QA differences (95% paper-bootstrap intervals)

- no_thinking_matched_minus_thinking: +0.93 percentage points (95% CI -1.86 to +4.33).
- no_thinking_merged_minus_thinking: +1.44 percentage points (95% CI -1.24 to +4.85).
- no_thinking_merged_minus_matched_no_thinking: +0.52 percentage points (95% CI -0.82 to +1.96).

## Scaling on GH200 GPUs

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

These extrapolations use actual complete generation, including prefill, native tokenization and all bounded format retries. They exclude QA, training, server startup, semantic correction and source acquisition. The 85% capacity factor is a planning assumption. Output-yield normalization assumes a comparable future paper mix; it does not guarantee recovery of difficult failed papers. Use four independent model replicas per Jupiter node to avoid paying for idle GPUs. Short capped probes select deployment settings and are never used as full-summary costs.

## Optimization protocol

64 deterministic non-held-out training papers across ten domains; concurrency 16/32/64; 512-token partial outputs with cold source prefix caches. Runtime selection is made before held-out QA. Merged BF16 and FP8 are evaluated on all 97 papers because merging and quantization can change outputs. Unsupported backend/KV-cache configurations are retained in optimization logs. The original baseline already uses FlashAttention 4, CUDA graphs, prefix caching and chunked prefill.

QA tests answerability under the pinned Qwen2.5-7B student and synthetic MCQs. It is not an independent human factuality assessment. Summary length and formatting failure rate are reported alongside accuracy.
