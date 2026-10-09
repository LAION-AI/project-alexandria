# Complete Knowledge Unit comparison: 1,000-word and 500-word chunks

**Follow-up completed:** 2026-10-10T00:49:08 Europe/Berlin. **Primary cohort:** the same 20 disjoint papers and frozen 200 questions. All 18 context conditions were freshly judged in this follow-up.

## Explanation for new readers

A Knowledge Unit (KU) records facts as a contextual description plus entities, attributes and relationships. The source paper is processed in consecutive chunks, then the resulting KUs form a factual context for a separate question-answering model. Chunk size is a maximum count of source words per call, not output length. The last chunk may be shorter. Every source word is covered once.

The rank128 LoRA adapters add learned low-rank matrices to Gemma E4B and Gemma12. Both were previously trained for one epoch on 391 Qwen27 completions from 97 papers, extracted with 1,000-word chunks. This follow-up applies those fixed adapters to 500-word chunks without retraining. Their evaluation papers are separate from those 97 training papers.

QA accuracy is the fraction of four-choice questions solved by the same Qwen2.5-7B-Instruct answerer using each representation. The generators do not answer the questions themselves. The new source-only questions and gold labels were frozen before any initial evaluation-context generation. The answering prompt, sampling, ASCII filter, parser and invalid-answer retry policy are unchanged.

## 1. All measured KU generators and adapters

| Generator | Adapter | 1,000-word KUs | 500-word KUs | 500 minus 1,000 | Paired 95% interval |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen 27B FP8 | None | **88.00% (176/200)** | **89.50% (179/200)** | +1.50 pp | -3.00 to +5.50 pp |
| Gemma 4 E4B IT | None | **64.50% (129/200)** | **75.50% (151/200)** | +11.00 pp | -0.50 to +21.50 pp |
| Gemma 4 E4B IT | KU LoRA rank128 | **80.00% (160/200)** | **82.50% (165/200)** | +2.50 pp | -10.00 to +12.50 pp |
| Gemma 4 12B IT | None | **63.50% (127/200)** | **75.00% (150/200)** | +11.50 pp | +0.00 to +21.50 pp |
| Gemma 4 12B IT | KU LoRA rank128 | **85.00% (170/200)** | **88.50% (177/200)** | +3.50 pp | -1.00 to +8.00 pp |

All figures retain the full 200-question denominator. Percentage-point changes are absolute changes in accuracy. Paired intervals resample all ten questions together within each paper (10,000 bootstrap samples, seed 20261009).

### Original paper and no-context controls

| Context | Correct / 200 | QA accuracy | 95% interval | Invalid slots |
| --- | ---: | ---: | --- | ---: |
| No paper context | 89 | 44.50% | 40.50 to 49.00% | 5 |
| Complete original paper | 182 | 91.00% | 87.50 to 94.50% | 0 |

### Generation failures and invalid QA answers

| KU condition | Failed papers / 20 | Invalid QA slots / 200 |
| --- | ---: | ---: |
| qwen_ku1000 | 0 | 0 |
| qwen_ku500 | 0 | 0 |
| gemma4_base_ku1000 | 0 | 0 |
| gemma4_base_ku500 | 1 | 10 |
| gemma4_r128_ku1000 | 0 | 0 |
| gemma4_r128_ku500 | 1 | 10 |
| gemma12_base_ku1000 | 1 | 10 |
| gemma12_base_ku500 | 1 | 10 |
| gemma12_r128_ku1000 | 0 | 0 |
| gemma12_r128_ku500 | 0 | 0 |

A paper with any failed KU chunk retains all ten QA slots as wrong/invalid, even when partial KUs exist. Existing 1,000-word outputs, including failures, were reused unchanged. No failed output was regenerated after observing scores.

## 2. Summary references on the same papers

| Generator | Adapter | Correct / 200 | QA accuracy | 95% interval | Invalid slots |
| --- | --- | ---: | ---: | --- | ---: |
| Qwen 27B FP8 | None | 164 | 82.00% | 76.00 to 87.50% | 0 |
| Gemma 4 E4B IT | None | 126 | 63.00% | 50.50 to 74.00% | 20 |
| Gemma 4 E4B IT | KU LoRA rank128 | 135 | 67.50% | 54.00 to 79.50% | 20 |
| Gemma 4 12B IT | None | 148 | 74.00% | 68.00 to 79.50% | 0 |
| Gemma 4 12B IT | KU LoRA rank128 | 161 | 80.50% | 74.00 to 86.50% | 0 |
| Gemma 4 12B IT | Earlier summary LoRA rank128 | 178 | 89.00% | 83.50 to 93.50% | 0 |

These summary texts and all 1,000-word/Qwen contexts are cached from the earlier completed experiment; only the four Gemma 500-word conditions are newly generated. Every condition receives fresh QA judgements here, allowing a direct within-run chunk comparison. The answerer samples at temperature 0.5, so scores of unchanged contexts may differ slightly from the [initial report](../README.md).

## 3. Adapter improvements and other paired comparisons

| Comparison | Change | 95% interval |
| --- | ---: | --- |
| qwen_ku500_minus_qwen_ku1000 | +1.50 pp | -3.00 to +5.50 pp |
| gemma4_base_ku500_minus_gemma4_base_ku1000 | +11.00 pp | -0.50 to +21.50 pp |
| gemma4_r128_ku500_minus_gemma4_r128_ku1000 | +2.50 pp | -10.00 to +12.50 pp |
| gemma4_r128_ku500_minus_gemma4_base_ku500 | +7.00 pp | -7.50 to +20.00 pp |
| gemma4_r128_ku1000_minus_gemma4_base_ku1000 | +15.50 pp | +8.50 to +23.00 pp |
| gemma12_base_ku500_minus_gemma12_base_ku1000 | +11.50 pp | +0.00 to +21.50 pp |
| gemma12_r128_ku500_minus_gemma12_r128_ku1000 | +3.50 pp | -1.00 to +8.00 pp |
| gemma12_r128_ku500_minus_gemma12_base_ku500 | +13.50 pp | +4.50 to +25.00 pp |
| gemma12_r128_ku1000_minus_gemma12_base_ku1000 | +21.50 pp | +13.00 to +31.50 pp |
| gemma4_r128_ku500_minus_gemma12_r128_ku500 | -6.00 pp | -17.50 to +3.00 pp |
| gemma4_r128_ku1000_minus_gemma12_r128_ku1000 | -5.00 pp | -10.51 to +0.50 pp |
| gemma12_r128_ku500_minus_gemma12_previous_summary_lora | -0.50 pp | -6.00 to +5.00 pp |

The fresh cohort contains only twenty papers. Intervals including zero do not establish a reliable ordering of their compared conditions. Questions were authored by the Qwen teacher and source-validated automatically by Qwen2.5; this is not an independent human-curated question set.

## 4. Original 97-paper teacher benchmark

The following reference scores come from the 9 October run, not the fresh answering run above. New student-adapter results are never reported on these 97 now-training papers. The two cohorts have different questions and must not be pooled.

| Context | Correct / 970 | QA accuracy |
| --- | ---: | ---: |
| No context | 599 | 61.75% |
| Complete original paper | 938 | 96.70% |
| Qwen 27B KUs, 1,000 words | 878 | 90.52% |
| Qwen 27B KUs, 500 words | 898 | 92.58% |
| Earlier Gemma12 direct summary LoRA, FP8 | 913 | 94.12% |

## 5. Copying audit and scientific limitations

Every new 500-word representation is checked against its full source, with the same audit version as the cached representations. Up to five normalized consecutive whitespace words are allowed, exactly six is borderline and seven or more is flagged. Factual KU fields are separate fragments; bibliographic metadata and explicit summary evidence are separate categories. The punctuation-split diagnostic remains separate.

| KU condition | Audited | Narrative ≤5 | Six only | Narrative ≥7 | Longest run | Mean coverage ≥6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen_ku1000 | 20 | 0 | 0 | 20 | 22 words | 10.32% |
| qwen_ku500 | 20 | 0 | 0 | 20 | 31 words | 10.97% |
| gemma4_base_ku1000 | 20 | 0 | 0 | 20 | 41 words | 21.27% |
| gemma4_base_ku500 | 20 | 0 | 0 | 20 | 37 words | 22.51% |
| gemma4_r128_ku1000 | 20 | 0 | 0 | 20 | 26 words | 14.19% |
| gemma4_r128_ku500 | 20 | 0 | 0 | 20 | 37 words | 14.80% |
| gemma12_base_ku1000 | 20 | 1 | 2 | 17 | 18 words | 8.35% |
| gemma12_base_ku500 | 20 | 0 | 0 | 20 | 20 words | 8.16% |
| gemma12_r128_ku1000 | 20 | 0 | 0 | 20 | 37 words | 11.18% |
| gemma12_r128_ku500 | 20 | 0 | 0 | 20 | 35 words | 13.47% |

Failed partial outputs retain explicit status flags in the audit. QA usefulness and compliance with the copying rule are different measurements. This experiment applies no back-translation or semantic correction. Details include source hashes and cumulative copying from multiple spans.

## 6. Runtime and observed allocation cost

The follow-up allocation used **17.97 minutes**, four GH200 GPUs and **1.1978 reserved GPU-hours**. The initial distillation/evaluation allocations used **4.1000 GPU-hours**, including failed attempts. Together these allocations used **5.2978 GPU-hours**. Training is already included in the initial total; this follow-up does not retrain. Queue waits allocate no GPU time. These are complete experimental allocation costs, not production summarization throughput estimates.

Four Gemma generation servers run in parallel; the first freed GPU becomes the fixed QA judge while remaining generators finish. Generation uses the same sequential extraction prompt, temperature 0.1, top-p 0.95, disabled thinking and up to 16 concurrent documents per server. The 8,192-token budget has one predeclared 16,384-token format retry. Chunks within a paper are sequential. All model revisions, training settings and frozen-question hashes match the earlier run.

## 7. Inspection and reproducibility

- [Metrics, settings, frozen hashes and allocation accounting](metrics.json).
- [All 3,600 primary QA predictions and gold labels](qa_predictions.jsonl.gz).
- [Copy audit](copy_overlap.json), [per-paper CSV](copy_overlap.csv) and [detailed source-hash evidence](copy_overlap_details.jsonl.gz).
- [Artifact checksums](artifact_manifest.json).
- [Earlier overview with training details and original 97-paper references](../README.md).
