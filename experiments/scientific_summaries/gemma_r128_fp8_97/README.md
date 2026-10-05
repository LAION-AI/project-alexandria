# Gemma rank-128 no-thinking: FP8 repeat

The first no-thinking experiment completed both BF16 conditions. FP8 was unmeasured because its generator port collided with the running QA server. This repeat uses disjoint ports and refuses to start on an occupied port.

- Reuse the three audited reference conditions without regenerating or rescoring them: thinking **897/970**, no-thinking live BF16 **906/970**, tuned merged BF16 **911/970**.
- Benchmark online `fp8_per_tensor` weights, with BF16 or optional FP8 KV cache, on the same 64 non-held-out optimization papers. Select concurrency 16/32/64 using capped 512-token probes before full held-out QA.
- Generate the same full 97 paper summaries with thinking disabled; evaluate all 970 immutable MCQs with the pinned BF16 Qwen2.5-7B judge.
- Run the mandatory source-copy n-gram audit, including narrative, explicit proof quotes, metadata, longest runs and overlap coverage. Five whitespace-delimited words are allowed, six borderline, seven or more flagged. Preserve punctuation-split diagnostics separately.
- The FP8 KV-cache candidate uses default unit scales without calibration. Inspect its full QA before a production recommendation. Source-copy constraints are evaluated independently of QA answerability.
- One four-GH200 Jupiter node; one-hour limit. Full-summary costs use complete generation with all format retries, never the capped probes. All idle allocation time is reported separately from active generator GPU-hours.

The original audited BF16 results are in [the no-thinking experiment](../gemma_r128_no_thinking_97/evaluation/RESULTS.md). **Completed job 2180360:** 914/970 = 94.23% QA, 97/97 generated papers, 545.364 seconds on one active GH200, 939.86 completion tokens/s/GPU. Selected configuration: `fp8_per_tensor` weights, BF16 KV, 16,384 batched tokens, concurrency 64. Full node allocation: 24 min 31 sec / 1.6344 allocated GPU-hours. All 97 narratives exceed the six-word overlap tolerance (maximum 32 words, mean coverage in runs ≥6: 15.23%). The production cost table counts finished outputs, not compliant paraphrases. [Systematic overlap findings](../copy_overlap_audit_20261005/README.md). Model weights and full sources are preserved outside GitHub.
