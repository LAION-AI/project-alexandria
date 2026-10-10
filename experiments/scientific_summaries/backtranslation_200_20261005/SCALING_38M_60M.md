# Speed-optimized Qwen-distilled rank-128 generation plus TranslateGemma repair

These are planning estimates on Jupiter GH200 GPUs, derived from complete measured outputs. The generator is **Gemma 4 12B IT with the Qwen3.8-27B-distilled rank-128 LoRA**. Thinking is disabled. The repair is one guarded EN→DE→EN round on selected narrative copy windows using TranslateGemma 4B.

**Later KU repair and summary advice:** [Repair recommendations](../backtranslation_repair_search_20261010/RECOMMENDATIONS.md) and [direct-Gemma cost scenarios](../backtranslation_repair_search_20261010/SCALING.md). The new targeted-edit measurements use KU residuals; their application to summaries is hypothetical. Existing one-round summary translation scores below remain the measured reference and have zero strict five-word passes.

## Planning GPU hours

**85% useful capacity** is assumed for every GPU stage. It is an allowance for scheduling and service overhead, not a statistically estimated uncertainty interval.

| Stage | 38 million summaries | 60 million summaries | 98 million combined |
| --- | ---: | ---: | ---: |
| Generation: merged FP8, no thinking | 69,820 | 110,241 | 180,061 |
| TranslateGemma 4B: one EN→DE→EN repair round | 2,143 | 3,384 | 5,526 |
| Bidirectional NLI meaning filter | 525 | 829 | 1,355 |
| **Optimized total** | 72,488 | 114,454 | 186,942 |
| Alternative total: reserve a GPU throughout the complete quality worker | 73,727 | 116,411 | 190,138 |

Critical numerical/formula signatures, candidate-copy checks and assembled-summary audits are also required. The optimized total assigns their CPU-only work to independent CPU workers; it does not count CPU/storage/I/O charges as GPU hours. NLI phase wall time, including its tokenization, is charged on one GPU. The alternative total conservatively reserves one GPU through the measured entire quality phase (CPU checks included); it replaces the NLI row rather than adding to it.

Guarded repair adds **3.82%** GPU work over generation under these assumptions. A rounded capacity budget is **75,000 GPU-hours for 38M** or **120,000 GPU-hours for 60M**. This is a provisional budget for a comparable corpus, not a guaranteed cost cap.

## Measured-rate totals before the capacity allowance

| Output count | Generation + translation + NLI GPU hours |
| --- | ---: |
| 38,000,000 | 61,615 |
| 60,000,000 | 97,286 |
| 98,000,000 | 158,901 |

## Measurements and calculation

- Full generation: **545.363982 s / 97 successful papers / one GH200**, including complete prefill, native tokenization and bounded format retries. Mean narrative length: **3096.74 native tokens**. Merged FP8 weights, BF16 KV cache, 64 concurrent requests, 16,384 batched tokens, CUDA graphs, FlashAttention 4, chunked prefill and prefix caching. FP8 KV was not used because the tested candidate failed.
- Complete translation: **34.511713 s / 200 summary versions / one GH200**, both directions, tokenization and complete outputs included. 6,379 selected narrative windows (**31.895 per version**). BF16 TranslateGemma 4B, vLLM greedy decoding, batch 512, language-only execution and FlashAttention 3. No additional quantization speedup is assumed.
- Bidirectional NLI: **8.459741 s / 200 versions / one GH200**, 12,758 ordered pairs, FP16 DeBERTa, length sorting and padded batch 32. Entire quality-worker time including CPU checks and overlap audit: **28.421104 s**.
- Formula per stage: `GPU hours = summaries × measured single-GPU seconds / measured summary count / 3600 / 0.85`. Optimized total sums generation, translation and NLI. All intermediate values and input hashes are in [SCALING_38M_60M.json](SCALING_38M_60M.json).

## Deployment assumptions

Keep models loaded, fill length-aware batches continuously, and send only flagged narrative windows to translation. Run CPU matching/signatures separately from the GPU services. Use all four GH200 GPUs per exclusive Jupiter node via independent replicas or balanced stage workers. Based on measured work, translation needs roughly one GPU per 33 generator GPUs; NLI roughly one per 133 generator GPUs. Integer allocation, idle periods and CPU/IO bottlenecks can change these ratios. Production scheduling and a multi-node end-to-end pipeline have not been benchmarked.

The 200 repair inputs are 97 FP8 versions, 97 BF16 versions and six live BF16 versions of the same 97 held-out papers. Repair throughput therefore uses a mixed no-thinking cohort, not 200 independent papers or a fresh FP8-only repair benchmark. Future document lengths, output lengths and copy-window density must be comparable for this extrapolation to hold.

Server startup/compilation should be amortized by long-running workers. The 1.54 GPU-hour development experiment included three arms, repeated checks, QA and failed starts; multiplying that total by output count would not describe this optimized single-arm service.

## Quality and copy constraint

### QA and compute together: each deployment on the same 97 papers

The following table aggregates the preserved TranslateGemma QA traces by
`original_condition`, with 970 immutable MCQs for each complete deployment.
The earlier 93.70% guarded QA figure pools 200 correlated FP8/BF16/live-BF16
versions; it is not the FP8-only score. No new model calls were made for this
aggregation. The model is Gemma 4 12B IT with the Qwen-distilled rank-128 LoRA,
thinking disabled, without an LLM critique or semantic-correction stage.

| Deployment | Pipeline | Correct / 970 | QA accuracy | Mean narrative copy coverage in runs ≥6 | Planning GPU-hours for 38M |
| --- | --- | ---: | ---: | ---: | ---: |
| Merged FP8 | Direct generation | 914 | 94.23% | 15.23% | 69,820 |
| Merged FP8 | Generation + guarded TranslateGemma EN-DE-EN | 905 | 93.30% | 9.69% | 72,488 |
| Merged BF16 | Direct generation | 911 | 93.92% | 15.74% | 95,495 |
| Merged BF16 | Generation + guarded TranslateGemma EN-DE-EN | 913 | 94.12% | 10.10% | 98,163 |

The BF16 combined estimate uses its measured complete-generation rate plus
the same mixed-cohort translation/NLI rates used in the existing FP8 scaling
table. Neither is a fresh deployment-specific production repair benchmark.
Both estimates assume 85% useful GPU capacity and a comparable corpus. The
FP8 combined budget rounds to 75,000 GPU-hours; the BF16 combined budget rounds
to 100,000. Numerical/formula and NLI acceptance checks select translated
passages; they do not invoke a separate semantic-correction generator.

All 97 guarded narratives in each deployment still exceed the strict five-word
copy limit. These costs are for one partial guarded paraphrasing round. The
QA changes are observed point estimates under the historical stochastic
answerer, not proof that translation improves or damages scientific accuracy.
Per-deployment counts and exact calculations are preserved in
[QA_AND_SCALING_38M.json](QA_AND_SCALING_38M.json), derived from
[the saved QA traces](evidence/translategemma/qa-results.json.gz) and
[overlap evidence](evidence/translategemma/copy_overlap.csv).

This is **one partial repair round**, not an estimate for enforcing strict five-word compliance. All 200 guarded summaries still contain a source-copy run of at least seven words; **0/200 pass the strict five-word narrative rule**. TranslateGemma lowers mean copied-word coverage from 15.40% to 9.85%, with pooled QA 93.70% versus 94.10% before repair. Those pooled QA scores refer to correlated mixed summary versions, not a new FP8-only QA evaluation.

Number/formula signature and meaning filters reject suspect candidates and retain the original text. Zero detected critical-signature drift in accepted summaries does not certify every possible mathematical expression or scientific claim. Additional rounds, fallback rewriting and guaranteed-compliance costs remain unmeasured and are excluded here.

Full production QA, training, source acquisition/OCR, CPU/storage/I/O billing and costs caused by a different document distribution are excluded. Findings: [FINDINGS.md](FINDINGS.md). Original generation benchmark: [FP8 RESULTS.md](../gemma_r128_fp8_97/evaluation/RESULTS.md).
