# Gemma 4 12B IT rank-128: no-thinking QA and inference optimization

Status: benchmark prepared; no new quality or throughput results are claimed until `evaluation/complete.json` is audited.

The user requested the previously trained Qwen-distilled rank-128 adapter with thinking disabled, a matched quality comparison, GPU-hour estimates for 38M and 60M papers, and reasonable inference optimization on Jupiter.

## Pinned reference

- Base model: `google/gemma-4-12B-it`, revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
- Adapter: the completed generation-only Qwen distillation on 865 training papers, rank 128, alpha 256, dropout 0.05, one epoch, learning rate 2e-5. No reviewer or corrector training.
- Thinking reference: **897/970 = 92.4742%** QA; 95/97 successful generation papers; 4,302.443625 seconds on one active GH200; 500.88977 emitted completion tokens/s/GPU, including thinking, JSON and retries. Final narrative averages 3,753.10 native tokens across all 97 slots.
- Frozen evaluation: 97 papers, 970 immutable MCQs, testset SHA256 `a0d5e5f99a0025c6cd8a5140a39a07a220ded6f37f04994500549886ffd83261` at project-alexandria commit `5aac4b5ba2a78b20637e8ab960fe79d01ecd769a`.
- QA model: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`, BF16; original historical QA prompt, temperature 0.5, top-p 0.95, max 100 output tokens, frequency/presence penalties 1.05. All failed summaries count as ten wrong answers.

## Comparison arms

| Arm | Precision and adapter | Thinking | Deployment selection |
| --- | --- | --- | --- |
| Cached audited reference | BF16 base plus live rank-128 LoRA | Enabled | Original concurrency 16 |
| Primary matched ablation | Same BF16 base plus same live rank-128 LoRA | Disabled | Same concurrency 16 |
| Optimized BF16 | Adapter merged into a separate BF16 checkpoint | Disabled | Non-held-out tuning |
| Optimized FP8, if supported | Online FP8 quantization of the merged checkpoint | Disabled | Non-held-out tuning, optional FP8 KV cache |

The primary ablation changes `enable_thinking` to false while preserving the generator system/user prompts, full untruncated source, 65,536 context limit, temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0, repetition penalty 1, deterministic per-paper/attempt seeds, JSON response mode, 24,576 output budget and three bounded format attempts. There is no semantic correction. Returned reasoning text and emitted reasoning-token counts must both be empty/zero.

## Optimization selection

64 training papers across ten scientific domains are selected deterministically using domain round-robin and document-ID hash order. None belongs to the held-out 97 papers. Held-out questions and gold answers are not used to select runtime configurations.

- Compare request concurrency **16, 32 and 64** with fixed 512-token partial probes.
- Compare maximum batched-token budgets **8,192 and 16,384** for merged BF16, and the supported FlashInfer alternative.
- Attempt online `fp8_per_tensor` quantization and a separate `fp8` KV-cache candidate. Unsupported configurations are recorded and skipped. The installed FlashAttention/FlashInfer backends reject `fp8_per_token_head`, so that format is excluded before allocating GPU time. The FP8 KV-cache candidate uses the runtime's default unit scales without calibration; its held-out QA must be inspected before any production recommendation.
- Select a deployment by probe throughput before running its full held-out generation and QA. Capped probe throughput is never extrapolated to full-summary costs.
- Merge into a separate model directory. Both original base weights and adapter remain available. Evaluate merging and quantization on all 97 papers because numerical changes can affect quality.

The existing inference runtime already uses **FlashAttention 4**, CUDA graphs, continuous batching, prefix caching and chunked prefill. Gemma has 512-dimensional global-attention heads, so an attention backend must support this shape; forcing a different FlashAttention version is not assumed to be an improvement. No shared software installation is upgraded.

## GPU-hour calculation

For `N` attempted papers:

`measured GPU-hours = N × complete-generation seconds / 97 / 3600`.

Planning for successful outputs uses `measured / 0.85 × 97 / successful_papers`. The 85% useful-capacity factor is an explicit assumption. Output-yield normalization assumes a comparable future paper mix; it cannot guarantee recovery of difficult failures. Costs include full outputs, prefill, tokenization and format retries. They exclude QA, training, startup, semantic correction, OCR and source acquisition. Production should run four independent replicas per Jupiter node because allocations are billed per complete four-GPU node.

The audited thinking reference gives approximately **562,411 GPU-hours for 38M** and **888,017 GPU-hours for 60M** successful outputs under these planning assumptions. No-thinking costs and quality are pending measured results.

### Preliminary token-proportional scenario (not a measured no-thinking result)

The existing thinking run emitted 2,155,050 completion tokens across 132 calls. The API reports 1,527,331 reasoning tokens and 627,719 other completion tokens. Removing the reported reasoning leaves **29.13%** of the current completion volume.

| Papers | Thinking planning GPU-hours, measured reference rate | No-thinking illustrative planning GPU-hours |
| --- | ---: | ---: |
| 38,000,000 | 562,411 | 163,818 |
| 60,000,000 | 888,017 | 258,660 |
| 98,000,000 | 1,450,428 | 422,478 |

This scenario simply multiplies the thinking estimate by 0.291278. It assumes unchanged answer volume, token throughput, retry behavior and output yield, while ignoring fixed prefill/tokenization costs. Disabling thinking and optimizing deployment can change all of these. This is neither a bound nor a guaranteed 3.43x speedup. No-thinking QA is unknown until the 970-question evaluation finishes. The complete-generation benchmark will replace this scenario with measured estimates.

QA measures answerability under a fixed model and synthetic MCQs, not independent human factuality. Narrative lengths, failure rates, paired paper-bootstrap confidence intervals and runtime are reported together.

## Reproduction and artifacts

The Slurm job uses one Jupiter Booster node / four GH200 GPUs, with a two-hour limit. GPU 0 runs the matched ablation; GPU 1 merges and benchmarks BF16 configurations; GPU 2 benchmarks FP8; GPU 3 runs the pinned QA model as summaries become ready. Model startup and optimization sweeps are recorded separately from full-summary throughput. The Ornith training/evaluation workflow is preserved.

`code/benchmark.py` performs held-out audits, generation, optimization and QA, then writes English `evaluation/RESULTS.md`, machine-readable scores and 38M/60M/98M scaling tables. `code/publish.py` copies final evidence into this repository and persists full request/response traces to Data1. Original source and model weights are not added to GitHub.

Official runtime references: [vLLM attention backends](https://docs.vllm.ai/en/stable/design/attention_backends/), [online quantization](https://docs.vllm.ai/en/stable/features/quantization/online/), [quantized KV cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/). The installed vLLM source and actual server logs determine supported configurations.
