# Full Qwen3.8 parallel-KU evaluation with fixed Qwen2.5 answering

This directory contains the complete 2026-08-21 evaluation requested after publication of
[Project Alexandria](https://arxiv.org/abs/2502.19413). Qwen3.8-27B generated independent,
contextualized Knowledge Units for every evaluable long paper; Qwen2.5-7B-Instruct then answered
the same stored Gemini-Pro-002 MCQs under no-context, original-text, and KU-only conditions.

This is a **new controlled benchmark, not an exact reproduction of the paper table**. Recovered
code indicates that the published table kept a Gemini Flash answerer fixed and used model names in
the rows for KU extraction. The Qwen2.5 answerer here was explicitly requested for a modern fixed
comparison. Its higher no-context Medical score is direct evidence that absolute comparison with
the published Qwen-extractor row would be misleading.

## Final results

Invalid outputs count as incorrect. Confidence intervals are paired, document-clustered bootstrap
percentile intervals (10,000 resamples, seed `250219413`), preserving within-paper question
dependence.

| Domain | Papers | MCQs | No context | Original text | Parallel KUs | KU − original |
|---|---:|---:|---:|---:|---:|---:|
| Physics | 97 | 970 | 54.74% [51.03, 58.56] | 90.00% [88.25, 91.75] | 84.95% [82.58, 87.32] | −5.05 pp [−7.01, −3.09] |
| Medical | 99 | 988 | 62.25% [58.62, 65.76] | 95.04% [93.52, 96.46] | 90.59% [88.27, 92.61] | −4.45 pp [−6.27, −2.83] |

There were four invalid Physics no-context responses and one invalid Medical no-context response
after five historical attempts. Original-text and KU-only responses were 100% parseable. The
Qwen2.5 pass took 862.3 seconds for Physics and 835.5 seconds for Medical while both jobs shared the
same two-GPU vLLM server; end-to-end judge wall time was therefore 14 minutes 22 seconds.

For orientation only, the paper's Qwen2.5 *extractor row*, evaluated with the historical fixed
answerer, reported Physics `52.23 / 89.69 / 79.04` and Medical `50.45 / 93.24 / 88.29`. These are
not like-for-like answerer scores and are not used in the confidence intervals above.

## Configuration

### KU extraction

- model: `Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound`, served as `qwen38`;
- runtime/hardware: vLLM, tensor parallel over 2 × RTX 3090, 32,768-token context;
- parallel targets: 500 words, plus up to 1,000 words before and after as read-only context;
- document opening: first 350 words supplied as abstract/opening context;
- extraction temperature 0.2, 2,500-token cap, thinking disabled;
- all target prompts flattened across each document batch for continuous batching;
- one batched document-level canonicalization pass using 1,800 output tokens;
- conservative canonical IDs and relationship rewriting after extraction.

The 97 Physics papers recorded 15,026.3 extraction seconds; the 99 Medical papers recorded
37,986.2 seconds. These phases ran sequentially on the two-GPU server: 14.73 wall-hours and 29.45
RTX-3090 GPU-hours total, or 0.150 extraction GPU-hour per evaluable paper.

### Fixed answerer

- model: `Qwen/Qwen2.5-7B-Instruct`, BF16, served as `qwen25`;
- runtime: vLLM 0.27.1, tensor parallel 2 × RTX 3090, max context 32,768;
- server: Triton attention, prefix caching, eager mode, model generation defaults disabled;
- recovered prompt and ASCII sanitizer; case-sensitive historical semicolon-split parser;
- temperature 0.5, top-p 0.95, 100-token cap, frequency/presence penalties 1.05;
- two concurrent domain jobs, each with client concurrency 5 and four-document checkpoints.

Code used by the final answer pass includes parser commit `c0860b0`. The final documentation and
release commit is newer; exact artifact digests are in `SHA256SUMS`.

Regenerate the reported table with `python bootstrap_summary.py > regenerated.csv`; the output is
byte-identical to `summary.csv`.

## Artifact boundary and overlap disclosure

The two `*-judge.json` files contain hashes, row indices, gold/predicted letters, configuration,
and scores; they contain neither source passages nor question wording. The two `*-kus.json` caches
contain generated KU summaries/entities/relations and non-reversible source provenance (hashes,
offsets, word counts, and MinHash), but no raw source or MCQ-question fields. A credential scan
found no API tokens, bearer tokens, private-key headers, or credential assignments.

“No raw source fields” does not mean “no verbatim overlap.” A case-folded exact screen using
`[a-z0-9@]+` tokens found at least one 12-token source match in 94/97 Physics documents and 95/99
Medical documents. The longest match inside one generated field was 37 Physics tokens and 86
Medical tokens; the median per-document maximum was 16 tokens in both domains. No fuzzy or semantic
matching was used. The source datasets were owner-authorized for repository redistribution, but
downstream users must still respect source-specific terms and review generated outputs.

## Integrity

- Dataset SHA-256 and ordered-selection digests match the released Parquets.
- All 196 cache document IDs exactly match the judge document IDs.
- All 1,958 `(document_id, question_index)` rows are unique.
- Gold and non-null predictions are restricted to `A`--`D`.
- Stored summaries were independently recomputed from prediction rows.
- The two malformed-parser and case-folding diagnostic partial runs remain local and are excluded.

See [the reproduction guide](../../../docs/reproducing-mcq.md),
[parallel architecture](../../../docs/methodology.md), and
[JUPITER scaling estimates](../../../docs/scaling-estimates.md).
