# Qwen3.8 4B vs 9B parallel-KU evaluation

This release compares the requested `empero-ai/Qwen3.8-4B-Distill-GGUF` and
`empero-ai/Qwen3.8-9B-Distill-GGUF` extractors on all 196 long-paper evaluation documents. Both
sets of KUs were scored by the same fixed `Qwen/Qwen2.5-7B-Instruct` answerer and the same stored
Gemini-Pro question/answer pairs. The existing 27B result is included as a reference.

These are controlled local-model comparisons, not a byte-for-byte reproduction of the published
paper table. The fixed answerer is the Qwen2.5-7B benchmark used for the existing 27B release.

## Scores

Percentages are exact-match MCQ accuracy; invalid answers count as incorrect. Intervals are paired,
document-clustered percentile bootstrap intervals (10,000 resamples, seed `250219413`).

| Extractor | Domain | Papers | No context | Original | KU-only | KU − original |
|---|---|---:|---:|---:|---:|---:|
| 4B | Physics | 97 | 54.33% [50.62, 58.14] | 90.52% [88.66, 92.27] | 79.48% [76.70, 82.16] | −11.03 pp |
| 9B | Physics | 97 | 54.43% [50.62, 58.35] | 90.00% [88.14, 91.75] | 80.10% [77.32, 82.89] | −9.90 pp |
| 27B | Physics | 97 | 54.74% [51.03, 58.56] | 90.00% [88.25, 91.75] | 84.95% [82.58, 87.32] | −5.05 pp |
| 4B | Medical | 99 | 62.65% [59.03, 66.09] | 95.24% [93.83, 96.56] | 83.50% [80.61, 86.16] | −11.74 pp |
| 9B | Medical | 99 | 62.04% [58.54, 65.45] | 95.04% [93.62, 96.36] | 84.62% [81.62, 87.44] | −10.43 pp |
| 27B | Medical | 99 | 62.25% [58.62, 65.75] | 95.04% [93.52, 96.46] | 90.59% [88.27, 92.61] | −4.45 pp |

The paired 9B-minus-4B KU accuracy difference is +0.63 pp [−1.55, +2.89] for Physics and
+1.12 pp [−1.21, +3.44] for Medical. Thus 9B is directionally better than 4B, but the paired
intervals do not establish a statistically reliable improvement at this sample size. The 27B
extractor remains materially stronger on KU-only answering under this fixed judge.

## Reproduction configuration

- 4B revision: `391fc7d103e3942a408def3e4f51c2f85d464417`; Q4_K_M SHA-256:
  `dec96e8cf2e11b613bb46513dec485377f9ca5a351e71712ee0e244f287c6790`
- 9B revision: `760121cd70bb4c36b2b5ec58eb765e0df5987efe`; Q4_K_M SHA-256:
  `df13d66021cef676f82be74053220fd75af6bf2a6a7fb77f5222ab9e50744a7a`
- Runtime: llama.cpp commit `5d9e5ac`, one Q4_K_M server per RTX 3090, eight continuous-batching
  slots per server, 262,144-token server context, Q8 KV cache, thinking disabled.
- KU prompts: 500-word target chunks, up to 1,000 words of read-only context on each side, opening
  context from the first 350 words, temperature 0.2, 2,500-token output cap, and one 1,800-token
  document-level canonicalization pass.
- Judge: Qwen2.5-7B-Instruct BF16 through vLLM 0.27.1, tensor parallel over two RTX 3090s,
  temperature 0.5, top-p 0.95, 100-token cap, frequency/presence penalties 1.05, historical
  semicolon parser and prompt.

Extraction elapsed times recorded in the caches were 4.67 GPU-hours for 4B (both domains run on
separate GPUs) and 7.68 GPU-hours for 9B. The 9B Medical run encountered a small number of very
verbose/truncated generations; the hardened retry path and deterministic complete-object recovery
allowed it to finish without discarding any document. All final KU units have non-empty summaries,
named entities, and zero parser warnings.

## Files and integrity

The four `*-kus.json` files contain generated summaries/entities/relations plus non-reversible
source hashes and offsets, but no raw source passages or question text. The four `*-judge.json`
files contain only row IDs, gold/predicted answer letters, configuration, and scores. SHA-256
digests for every artifact are in [`SHA256SUMS`](SHA256SUMS). Regenerate the numeric table with:

```bash
python bootstrap_compare.py
```

The complete source datasets and licensing notes are in [`data/evaluation`](../../../data/evaluation)
and the architecture description is in [`docs/methodology.md`](../../../docs/methodology.md).
