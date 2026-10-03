# Findings and results: Ornith-1.5-9B + DFlash on 97 scientific papers

**Completed on JUPITER, 2026-10-03.** This report covers the complete BF16 Ornith / official FP8 Qwen3.8-27B generation, self-audit, correction and fixed-student QA experiment: 97 papers, 970 immutable questions, eight conditions, and 7,760 scored answers. No LoRA is used.

See also the [60-million-paper GPU-hour planning table](SCALING_60M.md), with and without correction, explicit cache assumptions and sensitivity ranges.

## Findings

- Raw Ornith summaries achieve **910/970 (93.81%)**, versus **862/970 (88.87%)** for new Qwen3.8-27B FP8 summaries. The paired difference is **+4.95 percentage points**, 95% paper-bootstrap CI **+3.20 to +6.70**.
- Corrected Ornith summaries achieve **904/970 (93.20%)**, versus **871/970 (89.79%)** for corrected Qwen. The paired difference is **+3.40 points**, CI **+1.34 to +5.36**.
- Ornith correction changes six answers from wrong to correct and twelve from correct to wrong: a net **−6/970 (−0.62 points)**. Its CI, **−1.65 to +0.31**, includes zero. This is an observed small QA decrease, not conclusive evidence of worse factual reliability.
- Qwen correction changes twelve answers from wrong to correct and three in the other direction: **+9/970 (+0.93 points)**, CI **0.00 to +1.96**. Correction is therefore not uniformly helpful or harmful.
- Ornith summaries are substantially longer: **2,548 words raw / 2,302 corrected**, versus **1,199 / 1,187** for Qwen. Output length was not matched; greater retained detail may contribute to the QA advantage.
- The best DFlash4 short-probe recipe reaches **1,028 output tokens/s for cold-source generation (batch 64)** and **2,673 tokens/s for fully cached correction (batch 32)** on one GH200. Standalone Ornith reaches **1,029 / 2,085 tokens/s**, respectively. DFlash brings approximately **28% more correction throughput** when each runtime uses its best measured batch, and essentially no cold-generation gain in this workload.
- The actual 97-paper QA summaries were produced with **DFlash8**, selected before the supplemental DFlash4 benchmark. DFlash4 throughput and DFlash8 QA must be identified separately.
- Total allocated compute, including benchmarking, loading, recovery, QA, idle allocation time and unsuccessful allocated starts, is **4.3611 GPU-hours**.

## 1. Frozen evaluation and model identities

The [released test set](../data/README.md) contains **50 arXiv and 47 Bethgelab papers**, with ten synthetic `gpt-6-luna`-authored MCQs each. It is a length-filtered convenience sample, not a human-validated or random scientific-paper benchmark. We retain all 97 papers and the exported question/option order. The three original excluded records remain excluded. Source text is the dataset text; completeness against publisher PDFs was not established.

| Role | Checkpoint | Precision | Pinned revision |
| --- | --- | --- | --- |
| Generator / corrector | [ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) | BF16 | `489cb97981b8654bcfcf30ce1f94ed1b62e07b53` |
| Speculative draft | [ornith-ai/Ornith-1.5-9B-DFlash](https://huggingface.co/ornith-ai/Ornith-1.5-9B-DFlash) | BF16 | `21be3446a606afa67e20c33e68ca37396499248f` |
| New comparison generator / corrector | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | Official FP8 | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Fixed QA student | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | BF16 | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Historical summary reference | Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound | Mixed INT4 | `4756e3e4871aefd8d7cd5b0f6155ae5490451c1e` |

The DFlash link is a draft checkpoint (about 1.2B parameters), **not a standalone summarizer**. It needs the 9B target, which verifies its proposals. The standalone throughput arm uses that same BF16 target without a draft. There is no separate standalone-target QA arm in this 97-paper experiment; its QA equivalence was not measured here. The older RTX3090 / Q8 runs and the [paired 20-paper pilot](../ORNITH_DFLASH.md) are separate experiments.

Evaluation implementation pinned to repository commit `5aac4b5ba2a78b20637e8ab960fe79d01ecd769a`.

Test-set SHA256: `a0d5e5f99a0025c6cd8a5140a39a07a220ded6f37f04994500549886ffd83261`.

Effective initial summary-prompt SHA256: `3643ce1ef340c4e63f451b85f9db75e8203bfc691e315e47523b3d08077b9bed`.

## 2. Generation, correction and QA protocol

Both new generators receive the same initial 19-field summary prompt and complete source text, temperature **0.2**, top-p **0.95**, and **thinking disabled**. The initial output budget is **16,000 tokens**, maximum context **65,536 tokens**. Sources are preflighted with the serving tokenizer and never truncated. Generator, reviewer and corrector never receive QA questions, choices, gold keys, rationales, existing summaries or QA evidence.

The pipeline saves raw generation, a source-only same-model audit, a full semantic correction where needed, bounded V3 schema/source-quote repairs, and a corrected same-model audit. Raw means before semantic correction, with logged lossless shape normalization where necessary. The student context renders substantive narrative fields and excludes proof quotes, metadata and citation rankings. Literal quote alignment checks are mechanical evidence checks, not proof that every claim follows from its quote.

Unusable/truncated/repetitive outputs receive bounded source-only recovery under a shared conditional policy. Recovery may use **32,000 generation / 24,000 correction output tokens** within the unchanged context limit. Observed repeated-sentence failures led to a documented **presence penalty of 1.5** on long-output recovery requests. Exact per-call settings and prior failures are retained. **13 Ornith papers and 2 Qwen papers** needed source recovery. Recovery decisions did not consult QA correctness; only changed context hashes were rejudged. No papers were dropped for quality. For `arxiv-500010`, an unusable optional bibliography ranking was set to an empty string after repeated formatting failures; the scientific narrative, original metadata, repair history and provenance remain saved.

QA imports the pinned repository `run_document`, `historical_answer_prompt`, ASCII sanitizer and case-sensitive semicolon parser. Fixed student: BF16 Qwen2.5-7B-Instruct, **temperature 0.5**, **top-p 0.95**, **100 output tokens**, **frequency/presence penalties 1.05**, no thinking, **four concurrent requests**, maximum context **32,768**, up to **five formatting attempts**. Remaining invalid answers count as incorrect. All conditions are re-evaluated with this student, rather than silently mixing old cached scores. Confidence intervals use **10,000 document-cluster bootstrap draws, seed 250219413**; paired comparisons keep each paper's ten questions together.

## 3. All QA scores

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

All eight conditions retain 97 papers and 970 questions. All four new-model conditions have zero missing summaries and zero invalid final QA answers. The no-context condition has **four invalid answers**, counted wrong. The historical INT4 summary scores **870/970 (89.69%) in this rerun**; its older repository score was **872/970 (89.90%)**. Frozen inputs do not make stochastic sampling deterministic, and runtime differences may also contribute. The historical INT4 repair policy differs from the new full semantic-correction policy.

### Paired differences

| Comparison | Difference (percentage points) | 95% paired paper-bootstrap CI |
| --- | --- | --- |
| ornith_corrected − ornith_raw | -0.62 | -1.65 to +0.31 |
| qwen38_corrected − qwen38_raw | +0.93 | +0.00 to +1.96 |
| ornith_raw − qwen38_raw | +4.95 | +3.20 to +6.70 |
| ornith_corrected − qwen38_corrected | +3.40 | +1.34 to +5.36 |
| ornith_corrected − qwen38_reference | +3.51 | +1.44 to +5.57 |
| ornith_corrected − original | -3.92 | -5.57 to -2.37 |

### What changed after correction?

| Model | Correct → correct | Wrong → wrong | Wrong → correct | Correct → wrong | Net gain |
| --- | --- | --- | --- | --- | --- |
| ornith | 898 | 54 | 6 | 12 | -6 |
| qwen38 | 859 | 96 | 12 | 3 | 9 |

These are changes in the fixed student's answers, not directly counted factual errors in summaries. A corrected narrative can change the student's response even when neither narrative is demonstrably false. The experiment does not identify the cause of each transition.

## 4. Narrative lengths

| Condition | Mean words | Median | Minimum | Maximum | Mean context/source word ratio |
| --- | --- | --- | --- | --- | --- |
| ornith_dflash_raw | 2547.97 | 2273 | 1018 | 7440 | 0.7799 |
| ornith_dflash_corrected | 2302.34 | 2111 | 790 | 5975 | 0.7013 |
| qwen38_fp8_raw | 1198.89 | 1153 | 766 | 1914 | 0.3625 |
| qwen38_fp8_corrected | 1186.56 | 1153 | 751 | 1914 | 0.3590 |

Word counts refer to rendered narrative student contexts. Ratios are the mean of per-paper context/source ratios, not the ratio of two corpus totals. Ornith raw summaries average about **2.13×** the Qwen raw length. Correction reduces mean Ornith length by about **9.64%**; it reduces Qwen length by about **1.03%**. No controlled length-matched ablation was performed.

## 5. Self-audit scores and failures

**These are same-model source-only assessments, not independent human quality labels.** An audit can report high scores while listing unresolved issues; the saved normalized verdict preserves that inconsistency. Failed/invalid audits are explicitly unavailable, not imputed as perfect scores and not grounds for dropping papers from QA.

| Condition | Scored / 97 | Failed audit | Pass | Needs correction | Flagged issues | Missing facts |
| --- | --- | --- | --- | --- | --- | --- |
| ornith_dflash_raw | 52 | 45 | 33 | 19 | 84 | 14 |
| ornith_dflash_corrected | 60 | 37 | 38 | 22 | 86 | 27 |
| qwen38_fp8_raw | 91 | 6 | 89 | 2 | 5 | 0 |
| qwen38_fp8_corrected | 94 | 3 | 88 | 6 | 20 | 0 |

| Condition | Factual accuracy / 5 | Coverage / 5 | Clarity / 5 | Faithfulness / 5 | Scientific precision / 5 |
| --- | --- | --- | --- | --- | --- |
| ornith_dflash_raw | 4.904 | 4.923 | 4.308 | 4.827 | 4.212 |
| ornith_dflash_corrected | 4.917 | 4.950 | 4.267 | 4.883 | 4.150 |
| qwen38_fp8_raw | 4.978 | 4.934 | 5.000 | 4.989 | 4.978 |
| qwen38_fp8_corrected | 4.936 | 4.915 | 5.000 | 4.957 | 4.936 |

Means use only valid scored audits: denominators differ substantially (**52/60 Ornith raw/corrected; 91/94 Qwen raw/corrected**). They cannot establish that one model is more faithful than another. Corrected audits still flag issues in **22 Ornith and 6 Qwen** papers; mechanical source-span validation does not imply an independent semantic pass. All per-paper grades, score histograms, failures, issues and proposed corrections are included in the artifacts.

## 6. Hardware, runtime and optimizations

JUPITER allocations provide four GH200 GPUs per node; **97,871 MiB** GPU memory was reported by NVML for each GPU used here (approximately 95.6 GiB). Runtime: **vLLM 0.30.0, PyTorch 2.13.0, Transformers 5.18.0, huggingface-hub 1.33.0**, Python 3.11 on ARM aarch64, CUDA 13. Model revisions and full-file SHA256 fingerprints are included. Native vLLM DFlash is used.

Optimizations: one TP=1 model replica per GPU; continuous batching; up to 64 requests; automatic prefix caching; chunked prefill with **8,192 max batched tokens**; torch compilation/CUDA graphs and native attention/GDN kernels; local compiler caches; GPU memory utilization **0.90**. The main cohort used three Ornith DFlash8 replicas (32 concurrent papers each) and one Qwen FP8 replica (eight concurrent papers), followed by the fixed QA student. The source text was never shortened to improve speed.

### Probe design and every measured batch

Performance tuning uses **64 already accepted, non-holdout training papers**, not the 97 evaluation papers. Every probe caps output at **512 tokens** and reports actual emitted output tokens divided by batch wall time. Batches **1, 4, 8, 16, 32 and 64** are measured in both cold and warm cache states for both phases: **96 observations**. These are short probes, not complete summaries; requests often finish at the output cap. Warm correction repeats the **entire correction input** after warming it, making it more optimistic than a first correction that reuses only the source prefix.

| Runtime | Batch | Generation cold tok/s | Generation warm tok/s | Correction cold tok/s | Correction warm tok/s |
| --- | --- | --- | --- | --- | --- |
| ar | 1 | 140.45 | 155.24 | 129.05 | 151.34 |
| ar | 4 | 377.02 | 528.05 | 318.82 | 496.52 |
| ar | 8 | 456.26 | 910.08 | 446.73 | 848.58 |
| ar | 16 | 640.46 | 1415.60 | 559.65 | 1225.27 |
| ar | 32 | 842.72 | 2069.17 | 678.17 | 1722.64 |
| ar | 64 | 1029.11 | 2675.39 | 758.09 | 2085.00 |
| dflash4 | 1 | 111.93 | 115.62 | 146.17 | 167.80 |
| dflash4 | 4 | 206.12 | 405.17 | 350.50 | 545.94 |
| dflash4 | 8 | 410.46 | 766.65 | 472.66 | 924.39 |
| dflash4 | 16 | 602.99 | 1408.33 | 590.01 | 1640.53 |
| dflash4 | 32 | 878.31 | 1910.83 | 736.64 | 2672.51 |
| dflash4 | 64 | 1027.61 | 2680.43 | 808.83 | 2606.75 |
| dflash8 | 1 | 84.18 | 105.22 | 117.96 | 135.45 |
| dflash8 | 4 | 192.23 | 392.34 | 322.88 | 666.91 |
| dflash8 | 8 | 397.45 | 815.24 | 490.81 | 1118.75 |
| dflash8 | 16 | 613.72 | 1184.06 | 583.07 | 1373.44 |
| dflash8 | 32 | 815.36 | 1857.06 | 722.91 | 2255.22 |
| dflash8 | 64 | 881.95 | 897.28 | 756.34 | 761.44 |
| dflash15 | 1 | 102.16 | 101.35 | 153.36 | 188.18 |
| dflash15 | 4 | 186.86 | 278.71 | 337.47 | 649.77 |
| dflash15 | 8 | 332.18 | 572.96 | 453.01 | 937.57 |
| dflash15 | 16 | 608.67 | 1066.35 | 533.37 | 1622.47 |
| dflash15 | 32 | 768.45 | 1370.70 | 647.96 | 1995.75 |
| dflash15 | 64 | 754.43 | 770.68 | 706.05 | 701.89 |

`ar` means standalone Ornith target without speculative decoding; `dflash4/8/15` denote the configured number of speculative tokens. At matched correction batch 32, DFlash4 improves warm throughput by about **55%** over AR. Comparing each runtime's best warm correction batch gives about **28%**. At batch 1, DFlash4 cold generation is slower than AR (**112 vs 140 tokens/s**). DFlash therefore does not provide a universal speedup.

Larger batches are not always faster: DFlash8/15 at batch 64 show a pronounced warm correction slowdown. Prefix-cache pressure and speculative verification overhead are plausible contributors; metrics and cache counters are archived. Reported DFlash8 proposal acceptance is approximately **26–30%**; the shorter DFlash4 proposal achieves approximately **46.5%**. Acceptance alone is not end-to-end throughput. The initial selector chose the fastest **DFlash** candidate (8 versus 15); standalone AR had a higher initial combined probe score. DFlash4 was measured afterward and did not retroactively change the QA production configuration.

## 7. Complete-output request timings and token work

The following timers describe **individual API calls**, including retries and contention with concurrent requests. They are not end-to-end latency per paper. The generation and semantic-correction phases use full output budgets; their observed medians for Ornith are **100.78 s** and **94.10 s**, respectively. Adding phase means or medians does not produce measured wall time for a concurrently batched cohort.

| Model | Phase | Calls | Prompt tokens | Output tokens | Mean s | Median s | P95 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ornith_dflash | generation | 149 | 1209784 | 1558230 | 115.15 | 100.78 | 230.01 |
| ornith_dflash | raw_review | 142 | 1972561 | 131317 | 15.11 | 12.83 | 39.12 |
| ornith_dflash | semantic_correction | 90 | 1458865 | 684124 | 95.33 | 94.10 | 162.98 |
| ornith_dflash | field_repair | 75 | 531442 | 2800 | 1.55 | 1.23 | 4.23 |
| ornith_dflash | source_anchor_selection | 1631 | 30286017 | 107303 | 2.03 | 1.93 | 3.71 |
| ornith_dflash | corrected_review | 121 | 1452400 | 103453 | 12.00 | 9.79 | 34.31 |
| ornith_dflash | source_recovery_field_repair | 25 | 209253 | 1749 | 1.31 | 0.56 | 4.80 |
| ornith_dflash | source_recovery_corrected_review | 21 | 255686 | 20623 | 5.67 | 5.12 | 12.05 |
| ornith_dflash | source_recovery_single_anchor | 1307 | 8512652 | 11150 | 0.34 | 0.31 | 0.52 |
| ornith_dflash | source_recovery_generation | 15 | 115523 | 306250 | 93.89 | 103.05 | 168.90 |
| ornith_dflash | source_recovery_raw_review | 11 | 192464 | 13066 | 6.79 | 7.10 | 14.63 |
| ornith_dflash | source_recovery_semantic_correction | 10 | 185735 | 109278 | 51.06 | 44.42 | 97.19 |
| qwen38_fp8 | generation | 111 | 913754 | 361380 | 56.07 | 54.04 | 78.11 |
| qwen38_fp8 | raw_review | 103 | 942996 | 33044 | 6.21 | 2.09 | 19.11 |
| qwen38_fp8 | semantic_correction | 97 | 1093426 | 349012 | 61.76 | 60.86 | 88.27 |
| qwen38_fp8 | source_anchor_selection | 88 | 1228206 | 3721 | 1.90 | 1.61 | 4.01 |
| qwen38_fp8 | corrected_review | 101 | 926893 | 24781 | 4.75 | 2.04 | 18.47 |
| qwen38_fp8 | field_repair | 6 | 54547 | 1037 | 3.94 | 1.39 | 11.16 |
| qwen38_fp8 | source_recovery_field_repair | 3 | 24021 | 1389 | 6.25 | 5.75 | 7.46 |
| qwen38_fp8 | source_recovery_corrected_review | 2 | 18971 | 164 | 1.69 | 1.69 | 1.89 |
| qwen38_fp8 | source_recovery_generation | 1 | 6321 | 3713 | 46.25 | 46.25 | 46.25 |
| qwen38_fp8 | source_recovery_raw_review | 1 | 7531 | 82 | 1.48 | 1.48 | 1.48 |
| qwen38_fp8 | source_recovery_semantic_correction | 1 | 9551 | 4189 | 49.02 | 49.02 | 49.02 |
| qwen38_fp8 | source_recovery_single_anchor | 6 | 29068 | 71 | 0.33 | 0.31 | 0.46 |

Token counts include unsuccessful/repeated calls in the saved phase journals. Prompt tokens include repeated source prefixes; they do not measure uncached prefill work. The saved first-pass cohort timers are **643.04 s for Ornith on three GPUs** and **1,728.21 s for Qwen on one GPU**. The first pass produced only **90 raw / 84 corrected Ornith outputs** and **96 raw / 95 corrected Qwen outputs**. Completion counts were updated after recovery while those original timers remained unchanged: **neither timer is the elapsed time to finish all 97 papers**. GPU counts and output lengths also differ, so these timers are not a fair standalone speed ratio.

The QA checkpoint accumulates **265.36 s** of condition-scoring intervals, including superseded contexts. The sum for latest retained conditions is **265.36 s**. These exclude judge loading, checkpoint I/O between conditions and scheduler waits. Per-paper/condition times and format-attempt counts are in `qa_timings.csv`.

## 8. Scheduler allocation and GPU-hours

| Slurm job | Outcome | Allocated seconds | GPUs | GPU-hours |
| --- | --- | --- | --- | --- |
| 2160940 | COMPLETED | 2776 | 4 | 3.084444 |
| 2161270 | CANCELLED by 20488 | 223 | 4 | 0.247778 |
| 2162307 | COMPLETED | 617 | 4 | 0.685556 |
| 2162610 | COMPLETED | 255 | 4 | 0.283333 |
| 2162622 | FAILED | 54 | 4 | 0.060000 |

**Total: 4.361111 allocated GPU-hours** (all listed jobs, four GPUs per allocation). This is an allocation bill that includes startup and idle GPUs; it is not a per-model active compute estimate. Overlapping Slurm steps are counted only once within their allocation. The first recovery allocation was canceled after an observed long repetitive output. The final 54-second failed job was a redundant guarded recovery that refused to overwrite an already completed document and performed no model work. Its 0.06 GPU-hours are still included. A separately canceled queued duplicate had zero allocation time. No validated 60-million-paper cost projection or LoRA training-cost measurement follows from this small run.

## 9. Verification and preserved evidence

- **97/97 raw and 97/97 corrected summaries for each new model**: 388 narrative contexts.
- All **194 corrected** summaries pass schema/source-span checks: **14,078 Ornith and 8,600 Qwen** literal proof spans. Raw proof violations remain recorded.
- All **7,760 QA prompt hashes and parsed predictions** pass the final audit; source hashes, immutable questions and gold keys match the frozen bundle.
- Evaluation artifacts remain separate from the 1,000-paper training workspace. **Never use these 97 papers, their outputs, questions or answers as held-out LoRA training data.** Training-overlap screening/export guards are a separate pipeline.
- All final raw/corrected document records, actual model requests/responses, source-only findings/proposed corrections, failures and recovery histories are preserved in per-paper trace archives. QA responses and prior checkpoints, benchmark calls/metrics, server logs, code snapshots and model-file hashes are included.
- Thinking was disabled for this evaluation. Actual emitted reasoning fields are retained when present; no hidden reasoning trace is fabricated.

## 10. Conclusions and next comparison

1. **Use raw Ornith as the current QA baseline for this cohort.** It has the highest new-model summary score and avoids semantic-correction cost. This is a practical choice for question-answerability, not proof of universal factual superiority.
2. **Do not assume automatic full-summary correction improves QA.** Ornith shows a small uncertain decline and Qwen a small improvement. Targeted correction of independently verified errors is a candidate to test, not an established result.
3. **For this measured throughput workload, DFlash4 is the strongest correction candidate.** Use batch 32 for cached correction and batch 64 for cold generation as initial settings, then validate complete outputs on representative deployment lengths. Fully warmed 512-token rates must not be used as full-pipeline speed claims.
4. **Ornith provides a strong baseline without LoRA.** No trained-LoRA versus base comparison is reported here, so this does not demonstrate that LoRA is unnecessary. Future LoRA evaluation must preserve holdout exclusion and the same fixed QA protocol.
5. **Run a length-matched comparison and an independent factual audit before claiming model-level superiority.** Current results compare complete pipelines with different model sizes/precisions, output lengths, retry frequencies and self-audit availability. One synthetic convenience cohort and one stochastic evaluation run do not establish performance across scientific domains.

## Files and reproduction

| File | Contents |
| --- | --- |
| [scores.csv](scores.csv) | All eight scores, confidence intervals and invalid counts |
| [paper_scores.csv](paper_scores.csv) | All 776 paper/condition scores and source/context hashes |
| [paper_names.tsv](paper_names.tsv) | All 97 names, paper identifiers and DOIs |
| [throughput.csv](throughput.csv) | All 96 runtime/batch/cache/phase observations |
| [phase_timings.csv](phase_timings.csv) | Every saved generation/review/correction/repair phase aggregate |
| [qa_timings.csv](qa_timings.csv) | Latest condition intervals and format-attempt counts |
| [self_audit_scores.csv](self_audit_scores.csv) | All 388 same-model audit rows, including failures |
| [artifacts/report.json](artifacts/report.json) | Original complete numerical report, lengths and paired CIs |
| [artifacts/](artifacts/) | Full QA, audit findings, per-paper traces, benchmark traces, logs, verification and accounting |
| [provenance/](provenance/) | Pinned model-file fingerprints, configs, run protocol, primary/final source snapshots |
| [SHA256SUMS](SHA256SUMS) | Integrity checksums for every packaged file |

Verify the packaged results on CPU, without network or model inference, from the repository root:

```bash
python3 experiments/scientific_summaries/ornith_dflash_97/verify_results.py
```

To rebuild this package from the preserved JUPITER run directory:

```bash
python3 experiments/scientific_summaries/ornith_dflash_97/package_results.py \
  --run-dir /path/to/scientific-ornith-dflash-eval-97
```

Original HPC source snapshots contain machine-specific paths and are preserved as provenance; adapt paths and stage the pinned repository/model revisions before running inference elsewhere. Trace `.tar.gz` archives use relative paths. Model weights and caches are not committed. Third-party source data/models retain their upstream terms and attribution; this package does not change them.
