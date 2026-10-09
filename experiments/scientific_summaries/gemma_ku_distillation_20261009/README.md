# Knowledge Units: 1,000-word and 500-word chunks, base models and LoRA adapters

**Results completed:** 9 October 2026. **Overview updated:** 10 October 2026. All accuracies below are measured results from that completed run.

**Completed 500-word Gemma follow-up:** [Full 500/1,000-word KU table with base models, adapters and original-paper controls](ku500_followup_20261010/README.md). It freshly judges all contexts on the same frozen 20-paper cohort.

## What this experiment measures

A Knowledge Unit (KU) is a structured record of facts from a paper: a short contextual description plus entities, attributes and relationships. A paper is processed in consecutive chunks, and its KUs are combined into a factual context for a separate question-answering model. A summary is a prose-oriented representation of the same paper.

**“1,000-word” and “500-word” describe the maximum number of source words in an extraction call.** They do not describe the length of the generated KUs or summaries. The final chunk can be shorter. Every source word is covered once; the original available paper text is kept untruncated. Previously generated KU descriptions/entity names from up to ten preceding chunks provide naming continuity.

A LoRA is a set of additional learned low-rank matrices attached to a base model. Here, both new adapters are **rank-128 Gemma adapters**, trained for one epoch to imitate Qwen-generated KUs. All new generation conditions disable thinking. No back-translation or semantic correction is applied in this comparison.

**QA accuracy measures how often the same Qwen2.5-7B-Instruct answerer solves four-choice questions using each context.** The generators do not answer these evaluation questions themselves. This compares how useful the representations are to that fixed answerer.

## At a glance

- On 20 new papers, the KU LoRA improves Gemma E4B from **65.50% to 81.00%**, and Gemma12 from **63.50% to 85.00%**. Both paired paper-bootstrap intervals for improvement are above zero.
- Qwen 500-word KUs reach **92.58%** versus **90.52%** for 1,000-word KUs on the original 97-paper panel. On the fresh 20-paper panel, both reach **88.50%**.
- The earlier Gemma12 summary-trained LoRA reaches **89.50%** on the fresh panel; full original text reaches **90.50%**. These are reference conditions, distinct from the new KU-trained adapters.
- Source copying remains unresolved: both new KU adapters contain a run of at least seven copied normalized words in **all 20** fresh papers.

## 1. Fair adapter comparison: 20 new papers, 200 questions

The new adapters train on 391 Qwen27 completions extracted with 1,000-word source chunks from the original 97 papers. Those 97 papers are therefore training data for the new adapters. **All new-adapter scores use a separate 20-paper cohort**, screened against earlier paper sources using identifiers, near-title matching, full-text hashes and shared-text checks. This is a stratified convenience sample, not a uniformly random sample of all scientific literature.

The new cohort has ten questions per paper. Qwen27 authors the questions from source text alone, Qwen2.5 independently checks the gold answers and their selected original-source evidence, and all 200 questions are frozen before evaluation-context generation. Answer positions are balanced. This differs from the original Luna-authored 970-question benchmark, so absolute accuracies across the two panels should not be pooled.

### Knowledge Units and original-paper baselines

| Representation / generator | Adapter | Source chunk | Correct / total | QA accuracy | 95% interval | Invalid slots |
| --- | --- | --- | ---: | ---: | --- | ---: |
| No paper context | — | — | 90 / 200 | **45.00%** | 40.50–50.00% | 4 |
| Complete original paper | — | Whole paper | 181 / 200 | **90.50%** | 87.00–93.50% | 0 |
| Qwen 27B FP8 KUs | None | 1,000 words | 177 / 200 | **88.50%** | 85.00–92.00% | 0 |
| Qwen 27B FP8 KUs | None | 500 words | 177 / 200 | **88.50%** | 86.00–91.00% | 0 |
| Gemma 4 E4B IT KUs | None | 1,000 words | 131 / 200 | **65.50%** | 58.50–72.50% | 0 |
| Gemma 4 E4B IT KUs | KU LoRA, rank 128 | 1,000 words | 162 / 200 | **81.00%** | 75.00–86.50% | 0 |
| Gemma 4 12B IT KUs | None | 1,000 words | 127 / 200 | **63.50%** | 53.50–72.00% | 10 |
| Gemma 4 12B IT KUs | KU LoRA, rank 128 | 1,000 words | 170 / 200 | **85.00%** | 80.50–89.00% | 0 |

One Gemma12 base KU paper has a failed extraction chunk; all ten QA slots for that paper remain wrong/invalid in the primary 200-question denominator. Both new KU adapters generate complete outputs for all 20 papers. The four invalid no-context answers are also retained as wrong.

**Coverage of chunk-size experiments:** Qwen27 has measured 500-word and 1,000-word conditions. Gemma E4B, Gemma12 and both new KU adapters have measured **1,000-word conditions only**. At the time of this initial run their 500-word conditions were **not evaluated**. The linked follow-up now supplies those measurements; no score was imputed.

### Summary references on the same 20 papers

| Representation / generator | Adapter | Source chunk | Correct / total | QA accuracy | 95% interval | Invalid slots |
| --- | --- | --- | ---: | ---: | --- | ---: |
| Qwen 27B FP8 summary | None | Whole paper | 164 / 200 | **82.00%** | 76.00–87.50% | 0 |
| Gemma 4 E4B IT summary | None | Whole paper | 132 / 200 | **66.00%** | 53.50–76.50% | 20 |
| Gemma 4 E4B IT summary | KU LoRA, rank 128 | Whole paper | 134 / 200 | **67.00%** | 54.00–79.00% | 20 |
| Gemma 4 12B IT summary | None | Whole paper | 148 / 200 | **74.00%** | 68.00–79.50% | 0 |
| Gemma 4 12B IT summary | KU LoRA, rank 128 | Whole paper | 162 / 200 | **81.00%** | 74.00–87.50% | 0 |
| Gemma 4 12B IT summary | Earlier summary LoRA, rank 128 | Whole paper | 179 / 200 | **89.50%** | 84.00–94.00% | 0 |

The E4B base and E4B KU-LoRA summary conditions each have two generation failures, accounting for 20 wrong/invalid slots per condition. No failed outputs are discarded or regenerated after observing QA scores. These summary calls use the existing full-paper summary prompt; the new adapters were trained for KU generation, not specifically for summary generation.

**Follow-up completed:** All four Gemma 500-word conditions are now measured in the [matched comparison](ku500_followup_20261010/README.md).

## 2. Original benchmark: 97 papers, 970 questions

This panel compares the Qwen teacher chunk sizes and original reference contexts. **The newly trained Gemma KU adapters are not evaluated here**, because these papers supply their training targets. All conditions below are freshly scored in the 9 October run; Qwen1,000 KUs and the earlier direct Gemma FP8 summaries reuse their unchanged generated contexts.

| Representation / generator | Adapter | Source chunk | Correct / total | QA accuracy | 95% interval | Invalid slots |
| --- | --- | --- | ---: | ---: | --- | ---: |
| No paper context | — | — | 599 / 970 | **61.75%** | 58.14–65.26% | 3 |
| Complete original paper | — | Whole paper | 938 / 970 | **96.70%** | 95.57–97.84% | 0 |
| Qwen 27B FP8 KUs | None | 1,000 words | 878 / 970 | **90.52%** | 88.66–92.37% | 0 |
| Qwen 27B FP8 KUs | None | 500 words | 898 / 970 | **92.58%** | 90.72–94.33% | 0 |
| Gemma 4 12B IT, FP8 summary | Earlier summary LoRA, rank 128 | Whole paper | 913 / 970 | **94.12%** | 92.37–95.77% | 0 |

The answerer samples at temperature 0.5, so fresh scores of cached texts can differ slightly from earlier reports. For example, the separate [8 October KU1,000 report](../qwen27_ku1000_97_20261008/RESULTS.md) reported 90.62%, whereas this fresh answering run reports 90.52%. The context was not rewritten to obtain a different score.

## 3. Paired changes and uncertainty

Intervals resample entire papers, keeping all ten questions together. The bootstrap uses 10,000 samples and seed 20261009. Values below are percentage-point changes, not relative-percent increases.

| Panel | Paired comparison | Change | 95% interval |
| --- | --- | ---: | --- |
| 20 fresh papers | Gemma E4B KU LoRA minus base | +15.50 pp | +10.00 to +21.50 pp |
| 20 fresh papers | Gemma12 KU LoRA minus base | +21.50 pp | +13.50 to +31.00 pp |
| 20 fresh papers | Qwen500 minus Qwen1,000 | +0.00 pp | -4.50 to +4.50 pp |
| 97 original papers | Qwen500 minus Qwen1,000 | +2.06 pp | +0.21 to +3.92 pp |
| 20 fresh papers | E4B KU LoRA minus 12B KU LoRA | -4.00 pp | -9.50 to +1.00 pp |
| 20 fresh papers | 12B KU LoRA minus earlier summary LoRA | -4.50 pp | -9.50 to +0.50 pp |

The improvements over the corresponding Gemma KU baselines are clear in this sample. The 12B-versus-E4B adapter comparison and the new-12B-KU-versus-earlier-summary-adapter comparison have intervals including zero; this small cohort does not establish a reliable ordering for those pairs. The new questions were authored by the same Qwen model used as a teacher/generator, which limits how independently its representations are tested. Source/gold validation is automated rather than a human verification of every question.

## 4. Training and inference settings

| Setting | New Gemma E4B KU adapter | New Gemma12 KU adapter |
| --- | --- | --- |
| Base model | `google/gemma-4-E4B-it` | `google/gemma-4-12B-it` |
| Rank / alpha / dropout | 128 / 256 / 0.05 | 128 / 256 / 0.05 |
| Training epochs / optimizer steps | 1 / 49 | 1 / 49 |
| Training examples / source papers | 391 / 97 | 391 / 97 |
| Learning rate / optimizer | 2e-5 / AdamW | 2e-5 / AdamW |
| Microbatch / gradient accumulation | 1 / 8 | 1 / 8 |
| Precision / attention | BF16 / flex attention | BF16 / flex attention |
| Training worker wall time | 8.12 minutes | 12.73 minutes |
| Supervision | Assistant-only KU completion tokens | Assistant-only KU completion tokens |

The recorded Qwen teacher traces contain no reasoning tokens; no reasoning targets were fabricated. The teacher is `Qwen/Qwen3.8-27B-FP8` (revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`). Gemma E4B revision: `ee0ef6023621cff504d758262d4e04895a5af4a2`; Gemma12 revision: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. The earlier summary-trained adapter was trained on 865 Qwen summary examples.

KU generation reuses the repository sequential few-shot extraction prompt at temperature 0.1 and top-p 0.95. It permits an 8,192-token completion and one predeclared complete-JSON format retry at 16,384 tokens. Summary references use the existing 19-field summary prompt, temperature 1.0, top-p 0.95, a 16,384-token budget and at most three structure-only attempts. There are up to 16 concurrent documents per generation server; chunks within each KU document remain sequential.

The fixed answerer is `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`. It uses the historical answering prompt, ASCII filter and semicolon answer parser: temperature 0.5, top-p 0.95, frequency/presence penalties 1.05, 100 output tokens and up to four retries after an invalid answer. Judge concurrency is four and the context limit is 32,768 tokens. There is no QA-driven tuning of generation. New question/option strings express mathematical notation in English words so the fixed ASCII filter does not erase option distinctions. That historical filter still applies to answering contexts.

## 5. Complete-source copying audit

Each generated representation is checked against its own full untruncated source. The primary rule allows up to five consecutive normalized whitespace words; exactly six is borderline, and seven or more is flagged. The audit measures cumulative coverage from multiple copied spans as well as the longest span. KU factual fields are separate fragments, so joining graph fields cannot create artificial matches. Summary narrative, explicit evidence quotes and bibliographic metadata are measured separately; reasoning, questions and prompts are excluded. The punctuation-split diagnostic remains separate in the detailed audit.

| Panel / KU condition | Audited outputs | Narrative ≤5 | Six only | Narrative ≥7 | Longest run | Mean coverage in runs ≥6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh20 / Qwen 27B FP8, None, 1,000 words | 20 | 0 | 0 | 20 | 22 words | 10.32% |
| Fresh20 / Qwen 27B FP8, None, 500 words | 20 | 0 | 0 | 20 | 31 words | 10.97% |
| Fresh20 / Gemma 4 E4B IT, None, 1,000 words | 20 | 0 | 0 | 20 | 41 words | 21.27% |
| Fresh20 / Gemma 4 E4B IT, KU LoRA, rank 128, 1,000 words | 20 | 0 | 0 | 20 | 26 words | 14.19% |
| Fresh20 / Gemma 4 12B IT, None, 1,000 words | 20 | 1 | 2 | 17 | 18 words | 8.35% |
| Fresh20 / Gemma 4 12B IT, KU LoRA, rank 128, 1,000 words | 20 | 0 | 0 | 20 | 37 words | 11.18% |
| Original97 / Qwen 27B FP8, None, 1,000 words | 97 | 0 | 1 | 96 | 24 words | 9.56% |
| Original97 / Qwen 27B FP8, None, 500 words | 97 | 0 | 0 | 97 | 29 words | 9.70% |

The Gemma12 base row includes the partial output from its one failed paper, flagged as failed in the per-paper audit. Both new adapters fail the strict five-word rule on every fresh paper. E4B copying coverage decreases after KU training; Gemma12 coverage increases in this sample. QA gains do not establish compliance with the copying rule. This run includes no paraphrasing/back-translation stage. Summary evidence and metadata diagnostics, source hashes and failure statuses are included in the linked audit artifacts.

## 6. Actual compute used

| Allocation | Outcome | Wall time | Allocated GPUs | Reserved GPU-hours |
| --- | --- | ---: | ---: | ---: |
| 2248709 | Both trainings completed; inference startup failed | 18.90 min | 4 | 1.2600 |
| 2250191 | Inference worked; unfrozen question authoring failed | 9.78 min | 4 | 0.6522 |
| 2253395 | Generation, QA and copy audit completed | 32.82 min | 4 | 2.1878 |

Total reserved compute is **4.1000 GPU-hours** across all three allocations. The successful final evaluation allocation uses **2.1878 GPU-hours**. Training is already included in the first allocation and must not be added again. Batch/step rows are excluded from Slurm sums to prevent double counting. Queue waits allocate no GPU time. These figures include startup, question authoring, generation, QA, audit and idle roles; they are not production-generation throughput or large-scale cost estimates.

## 7. Evidence files

- [Machine-readable metrics and model/settings metadata](metrics.json).
- [Per-question gold/prediction records](qa_predictions.jsonl.gz): all 765 paper-condition records and 7,650 primary QA slots; no source-containing prompts are included.
- [Copy audit by condition](copy_overlap.json), [per-paper copy CSV](copy_overlap.csv), and [detailed copy evidence with source hashes](copy_overlap_details.jsonl.gz).
- [Artifact checksums](artifact_manifest.json).
- [Existing 97-paper dataset and usage documentation](../data/README.md).

The original-paper panel has 970 questions per condition and the fresh-paper panel has 200 per condition. Failed papers and invalid answers are retained in those denominators. The full source-containing inference traces and final adapter weights are preserved in experiment storage. The data exports here support inspection of reported scores and paper-level paired differences.
