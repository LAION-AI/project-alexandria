# Gemma 4 12B IT rank 128: 38M / 60M paper GPU-hour estimate

English capacity planning estimate for the completed **Qwen-distilled rank-128** adapter. BF16 on JUPITER GH200, native autoregressive decoding, thinking enabled, source-only generation, no semantic correction. The future Ornith-distilled adapter has not been benchmarked.

| Papers | Measured GPU-hours, per attempted paper | Planning GPU-hours, 85% useful capacity | Planning GPU-hours, 85% capacity and observed successful yield |
| --- | ---: | ---: | ---: |
| 38,000,000 | 468,193 | 550,815 | 562,411 |
| 60,000,000 | 739,251 | 869,708 | 888,017 |
| 98,000,000 | 1,207,444 | 1,420,522 | 1,450,428 |

The 98M row is the sum of the two requested scenarios, if both corpora are processed. For budgeting successful outputs under these assumptions, use approximately **562k / 888k GPU-hours** for 38M / 60M papers. These are extrapolations, not a production measurement.

## Measured basis

- 97 papers: **4,302.44 seconds on one active GH200**, or 44.36 GPU-seconds per attempted source.
- 95 complete summaries, two generation failures; **132 returned generation calls**, including bounded format retries.
- 2,155,050 emitted completion tokens from all attempts, or **22,217 tokens per paper**, including thinking and JSON.
- Full-output completion throughput: **500.89 tokens/s/GPU**.
- Mean final narrative: **3,753 native tokens / 2,469 words**, averaged over all 97 slots; the two failed outputs contribute zeros.
- Matched QA: **897/970 = 92.47%**. This is answerability under the fixed QA student and synthetic MCQs, not independent human factual correctness.
- Source: [`evaluation/report.json`](evaluation/report.json) and [`evaluation/RESULTS.md`](evaluation/RESULTS.md).

## Calculation

```text
GPUh_attempted = papers * (4302.443625141983 / 97) / 3600
GPUh_planning = GPUh_attempted / 0.85
GPUh_yield_normalized = papers * (4302.443625141983 / 95) / 3600 / 0.85
```

## Practical limits

The 85% factor reserves capacity for production overhead and imperfect utilization; it is an explicit assumption. The benchmark already includes actual prefill and format-retry time, so these are not added again. Yield normalization expresses GPU work per returned successful summary at the observed yield. It assumes additional sources/retries retain that yield and does not guarantee recovery of the hardest failed papers.

Production must use all four GPUs per node with independent replicas. These GPU-hours assume comparable paper lengths, emitted reasoning/answer lengths and failure distribution. Model startup, final QA, training, semantic quality review/correction, PDF retrieval/OCR and CPU orchestration are excluded. The reported 92.47% QA score is for thinking-enabled inference; neither that score nor this throughput establishes results with thinking disabled. Earlier no-thinking Gemma estimates describe a different workload.

The machine-readable inputs, assumptions and exact values are in [`scaling_38m_60m_rank128.json`](scaling_38m_60m_rank128.json).
