# Generation and targeted-repair compute scenarios

All amounts are GH200 GPU-hours with **85% useful capacity**. Generator: Gemma-4-12B-it with the appropriate Qwen-distilled rank128 adapter. Summaries use merged FP8; the measured 500-word KU run uses live BF16 LoRA. Thinking is disabled in both. Chunk size limits source words per call, not output words or tokens.

| Pipeline / cost scenario | 38M papers: GPU-hours | 60M papers: GPU-hours | Basis |
| --- | ---: | ---: | --- |
| Summary generation only | 69,820 | 110,241 | Measured merged FP8 summary LoRA rank128, no thinking |
| 500-word Knowledge Unit generation only | 193,451 | 305,449 | Measured live BF16 KU LoRA rank128, no thinking |
| Gemma E4B repair only — optimistic proxy | 38,864 | 61,364 | Residual KU timing, not direct repair |
| Summary generation + Gemma repair — scenario | 108,683 | 171,605 | Summary generation plus hypothetical KU-derived repair proxy |
| 500-word KU generation + Gemma repair — scenario | 232,315 | 366,813 | KU generation plus residual-repair proxy |

## Interpretation

Generation uses complete measured outputs, including prefill and bounded format retries: 545.364 seconds / 97 summaries, and 311.558 seconds / 20 KU papers. It is a benchmark-derived extrapolation, not a production measurement. The future paper length and generated-output distribution must be comparable.

The Gemma-only proxy uses 62.591 seconds / 20 papers = 3.1296 GPU-seconds per 500-word KU paper. This is the sum of the two Gemma E4B stages with meaning filters, after the TranslateGemma pool and two Qwen editing stages. It is **not** a direct Gemma-only benchmark from raw KUs. Its use for prose summaries is entirely hypothetical. Actual direct-repair cost and QA remain unmeasured; the scenario is optimistic and is not a proven lower bound or a guarantee that these budgets suffice. Under this assumption the overhead is about 55.7% over summary generation and 20.1% over KU generation.

For 1000-word KUs, the two Gemma stages take 47.394 seconds / 20 papers, or 0.013165 warm GPU-hours. Their linear residual-repair extrapolations are 25,013 / 39,495 GPU-hours at measured rates for 38M / 60M papers, and 29,428 / 46,465 after the same 85% allowance. The corresponding 500-word amounts before the allowance are 33,034 / 52,159 GPU-hours. Do not compare those warm measured-rate figures directly with the planning table above.

The complete editing chain also includes Qwen work: 125.201 / 96.461 active single-GPU seconds per 20 papers for 500 / 1000-word KUs, including their NLI filters and intervening CPU work. This excludes translation. The two Gemma experimental allocations cost 0.725556 reserved GPU-hours across both profiles, including cold startup, QA and idle roles. All seven search allocations cost 2.590 GPU-hours, including two failed starts. These totals measure experiments; multiplying them by millions of papers would misprice a warm service.

## Cheaper, measured summary option: guarded TranslateGemma only

For the Qwen-distilled rank128 merged-FP8 summary deployment, generation plus one guarded round trip and bidirectional NLI costs **72,488 / 114,454 planning GPU-hours** for 38M / 60M summaries, an estimated 3.82% overhead over generation. On 97 papers / 970 MCQs, QA changes from 94.23% to 93.30% and mean copied-word coverage in runs >=6 drops from 15.23% to 9.69%. **0/97** narratives pass the strict five-word rule after that round. This option can be useful when partial overlap reduction is sufficient; it cannot be recommended as full compliance. Its repair rate comes from 200 correlated FP8/BF16 versions of 97 unique papers.

[Exact inputs and calculations](scaling.json). Earlier measured summary evidence: [QA and scaling](../backtranslation_200_20261005/SCALING_38M_60M.md). The KU generation timing is preserved in [the completed 500-word follow-up metrics](../gemma_ku_distillation_20261009/ku500_followup_20261010/metrics.json).
