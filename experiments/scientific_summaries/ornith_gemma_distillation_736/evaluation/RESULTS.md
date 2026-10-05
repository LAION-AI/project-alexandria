# Gemma 4 12B IT: Ornith generator-reasoning distillation

736 training papers; ranks 64 and 128, one epoch each. Summary generation only. Matched untrained baseline uses exactly the same prompt, thinking mode and sampling. 97 frozen evaluation papers / 970 immutable MCQs per condition.

| Model | Correct / 970 | QA accuracy | 95% paper-bootstrap CI | Failed generation papers | Mean narrative words |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 12B IT / no LoRA, thinking enabled | 850 | 87.63% | 85.26–89.90% | 0 | 1070 |
| Gemma 4 12B IT / Ornith-distilled rank 64, one epoch | 874 | 90.10% | 86.49–93.09% | 2 | 1679 |
| Gemma 4 12B IT / Ornith-distilled rank 128, one epoch | 755 | 77.84% | 70.52–84.64% | 16 | 1364 |

## Paired QA differences

- gemma12_r64_minus_base: +2.47 percentage points; paired CI -0.93 to +5.36.
- gemma12_r128_minus_base: -9.79 percentage points; paired CI -17.32 to -2.68.

## Generation time and native summary length

| Model | Generation seconds | Completion tokens/s/GPU | Mean narrative tokens | API calls |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 12B IT / no LoRA, thinking enabled | 1559.9 | 638.1 | 1539.1 | 103 |
| Gemma 4 12B IT / Ornith-distilled rank 64, one epoch | 5022.9 | 490.7 | 2511.1 | 150 |
| Gemma 4 12B IT / Ornith-distilled rank 128, one epoch | 6841.0 | 471.8 | 2048.9 | 174 |

Completion throughput includes actual thinking and JSON output from all format attempts. Time includes generator requests and tokenization, excluding server startup and QA. The baseline output and QA responses are reused verbatim from the audited matched Qwen study. Two LoRA generators and two QA replicas share one four-GPU node; the baseline generation time is historical, not newly billed.

## One-epoch training

| Rank | Peak learning rate | Trainable parameters | Compute seconds | Worker seconds including setup/save | Compute GPU-hours (8 GPUs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 2e-05 | 262,275,072 | 3177.3 | 3221.1 | 7.06 |
| 128 | 2e-05 | 524,550,144 | 3194.6 | 3247.5 | 7.10 |

Both ranks train from the same base checkpoint at a 2e-5 peak learning rate. The earlier Qwen-teacher experiment had a cancelled 1e-4 attempt; it is not part of this Ornith study. GPU time is recorded separately in the allocation accounting; it is not hidden in successful compute time.

The targets are original source-only Ornith generator outputs and their actual reasoning, not corrected final summaries with mismatched reasoning. The earlier 86.60% Gemma raw score used different prompt/thinking/sampling settings and is a historical reference, not the matched no-LoRA control. QA measures answerability under the fixed student and synthetic MCQs; it does not establish independent human factual superiority. Every paper and failed output is retained. These targets are unreviewed raw Ornith outputs: all 736 failed the stricter source-evidence schema check. Summary length and failure rates must be considered alongside QA accuracy.

## Source-copy overlap audit

Exact normalized contiguous word overlap against each complete paper source. Five words are allowed, six are borderline, seven or more are flagged. Narrative prose, evidence quotes and bibliographic metadata are reported separately. Coverage counts each summary word once per threshold within each fragment. Separate statements are never concatenated to create matches.

| Condition | Audited outputs | Narrative ≤5 | Narrative exactly 6 | Narrative ≥7 | Longest narrative match | Mean narrative coverage ≥6 | Evidence ≥7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemma12_base | 97 | 0 | 2 | 95 | 50 | 22.29% | 8 |
| gemma12_r64 | 95 | 0 | 0 | 95 | 143 | 26.87% | 83 |
| gemma12_r128 | 81 | 0 | 0 | 81 | 184 | 22.09% | 72 |

Primary counts use whitespace-delimited words after Unicode normalization and punctuation stripping; mathematical expressions, decimal numbers and hyphenated terms are not split into artificial extra words. A separate punctuation-split diagnostic is retained in the JSONL. NFKC normalization, casefolding, soft-hyphen removal and PDF line-end word joining are used. Offsets refer to normalized text and word positions. Technical names and ordinary scientific phrases can match literally; lexical overlap alone does not establish plagiarism. All output flags remain visible and no paper is removed from QA. Reasoning traces, instructions and reviewer verdicts are not summary prose. Bibliographic names and titles are included in the metadata diagnostics, not silently mixed into paraphrase statistics.
