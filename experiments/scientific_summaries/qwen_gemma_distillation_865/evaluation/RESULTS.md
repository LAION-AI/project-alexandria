# Gemma 4 12B IT: Qwen generator-reasoning distillation

865 training papers; ranks 64 and 128, one epoch each. Summary generation only. Matched untrained baseline uses exactly the same prompt, thinking mode and sampling. 97 frozen evaluation papers / 970 immutable MCQs per condition.

| Model | Correct / 970 | QA accuracy | 95% paper-bootstrap CI | Failed generation papers | Mean narrative words |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 12B IT / no LoRA, thinking enabled | 850 | 87.63% | 85.26–89.90% | 0 | 1070 |
| Gemma 4 12B IT / rank 64, one epoch | 853 | 87.94% | 82.68–92.68% | 7 | 2198 |
| Gemma 4 12B IT / rank 128, one epoch | 897 | 92.47% | 89.07–95.15% | 2 | 2469 |

## Paired QA differences

- gemma12_r64_minus_base: +0.31 percentage points; paired CI -5.36 to +5.46.
- gemma12_r128_minus_base: +4.85 percentage points; paired CI +0.93 to +8.14.

## Generation time and native summary length

| Model | Generation seconds | Completion tokens/s/GPU | Mean narrative tokens | API calls |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 12B IT / no LoRA, thinking enabled | 1559.9 | 638.1 | 1539.1 | 103 |
| Gemma 4 12B IT / rank 64, one epoch | 5181.3 | 491.7 | 3345.5 | 157 |
| Gemma 4 12B IT / rank 128, one epoch | 4302.4 | 500.9 | 3753.1 | 132 |

Completion throughput includes actual thinking and JSON output from all format attempts. Time includes generator requests and tokenization, excluding server startup and QA. Three model servers run concurrently on three GPUs; Slurm allocates one four-GPU node.

## One-epoch training

| Rank | Peak learning rate | Trainable parameters | Compute seconds | Worker seconds including setup/save | Compute GPU-hours (8 GPUs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 2e-05 | 262,275,072 | 2652.8 | 2692.8 | 5.90 |
| 128 | 2e-05 | 524,550,144 | 2688.6 | 2741.6 | 5.97 |

Both ranks were restarted from the base checkpoint at the same 2e-5 learning rate after rank 128 diverged during an initial 1e-4 attempt. Cancelled-attempt GPU time is recorded separately in the allocation accounting; it is not hidden in successful compute time.

The targets are original source-only Qwen generator outputs and their actual reasoning, not corrected final summaries with mismatched reasoning. The earlier 86.60% Gemma raw score used different prompt/thinking/sampling settings and is a historical reference, not the matched no-LoRA control. QA measures answerability under the fixed student and synthetic MCQs; it does not establish independent human factual superiority. Every paper and failed output is retained.
