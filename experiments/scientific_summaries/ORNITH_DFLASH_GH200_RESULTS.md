# Ornith DFlash: measured GH200 accuracy and throughput

Frozen 97 papers / 970 MCQs per condition; fixed BF16 Qwen2.5-7B student. No LoRA. Full protocols, evidence and limitations: [9B](ornith_dflash_97/README.md), [35B-A3B](ornith35_dflash_97/README.md).

| Generator | Precision | Raw QA | Corrected QA | Mean words raw / corrected | QA DFlash configuration |
| --- | --- | ---: | ---: | ---: | --- |
| Ornith-1.5-9B | BF16 | 910/970 · 93.81% | 904/970 · 93.20% | 2548 / 2302 | dflash8 |
| Ornith-1.5-35B-A3B | BF16 | 897/970 · 92.47% | 899/970 · 92.68% | 1557 / 1532 | dflash4 |
| Qwen3.8-27B | FP8 | 862/970 · 88.87% | 871/970 · 89.79% | 1199 / 1187 | none |

## Best measured output throughput per GH200

Cold-source generation and repeat-input correction, 512-token probes on non-holdout papers. Warm-round cache reuse is subject to capacity and eviction. Both phases select their best measured batch independently; no output-length or full-pipeline speed equivalence is implied.

| Target | Runtime | Generation tok/s | Batch | Correction tok/s | Batch |
| --- | --- | ---: | ---: | ---: | ---: |
| Ornith-1.5-9B | ar | 1029.11 | 64 | 2085.00 | 64 |
| Ornith-1.5-9B | dflash4 | 1027.61 | 64 | 2672.51 | 32 |
| Ornith-1.5-9B | dflash8 | 881.95 | 64 | 2255.22 | 32 |
| Ornith-1.5-9B | dflash15 | 768.45 | 32 | 1995.75 | 32 |
| Ornith-1.5-35B-A3B | ar | 1082.18 | 64 | 1438.99 | 32 |
| Ornith-1.5-35B-A3B | dflash4 | 560.68 | 64 | 716.27 | 64 |
| Ornith-1.5-35B-A3B | dflash8 | 510.73 | 64 | 631.95 | 64 |
| Ornith-1.5-35B-A3B | dflash15 | 388.67 | 32 | 501.39 | 32 |

## Paired 35B versus 9B QA differences

| Comparison | Difference (percentage points) | 95% paired paper-bootstrap CI |
| --- | ---: | ---: |
| ornith35_raw_minus_ornith_raw | -1.34 | -2.89 to +0.21 |
| ornith35_corrected_minus_ornith_corrected | -0.52 | -2.37 to +1.24 |

Source-only correction can change question-answerability and factual reliability differently. Output lengths are not matched; model sizes, architectures and precision differ. Self-audits are not independent human labels. Prior eight-condition QA is reused only after exact source/question/context/prompt validation. See complete reports for all confidence intervals, invalids, timing, recovery and allocation hours. The existing [60M 9B planning estimate](ornith_dflash_97/SCALING_60M.md) remains a 9B estimate; it is not silently relabelled as a measured 35B cost.

**35B throughput finding:** best DFlash4 reaches 51.8% of best AR generation throughput and 49.8% of best repeat-input correction throughput. The 35B experiment provides no measured DFlash speedup. Its raw and corrected QA differences versus 9B both have paired confidence intervals crossing zero.
