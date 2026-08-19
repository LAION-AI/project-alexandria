# Qwen3.8-27B MCQ pilot (2026-08-19)

This pilot validates the released datasets and new parallel/restartable evaluator before a
full 200-paper run. It uses the same stored questions and gold letters as the paper. KU extraction
and diagnostic answering both used the local quantized Qwen3.8-27B server; therefore these numbers
are **self-judge diagnostics**, not yet a fixed-judge reproduction of the published table.

## Configuration

- served model: `Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound` (`qwen38`);
- runtime: vLLM, tensor parallel over 2 × RTX 3090, 32,768-token context;
- KU temperature 0.2; thinking disabled;
- judge temperature 0 for deterministic pilot scoring;
- compact semicolon-only diagnostic judge prompt, 16-token answer cap, no penalties;
- fixed sample seed `250219413`;
- parallel long mode: 500-word targets, 1,000 words before/after, concurrency 8;
- conservative document-level entity resolution after extraction.

## Abstract pilot

Ten abstracts were sampled per domain. Most contribute three valid questions, so one error changes
a score by roughly 3.3 percentage points.

| Domain | Questions | No context | Original | KUs | Wall time |
|---|---:|---:|---:|---:|---:|
| Biology | 29 | 75.9% | 100.0% | 96.6% | 146.9 s |
| Computer science | 30 | 90.0% | 100.0% | 100.0% | 135.4 s |
| Mathematics | 30 | 80.0% | 96.7% | 96.7% | 123.1 s |
| Physics | 30 | 83.3% | 100.0% | 96.7% | 111.3 s |

The high no-context values are far above the paper's fixed-judge lower bounds. This is expected for
a much newer 27B self-judge and demonstrates why the original Gemini Flash 8B answerer must be
restored before claiming a like-for-like reproduction.

After this diagnostic began, the production reproduction runner was tightened to the exact
recovered historical prompt, 100-token cap, temperature 0.5, top-p 0.95, and 1.05 frequency and
presence penalties. The pilot is intentionally not relabeled as if it used those later settings.

## Long-paper pilot

Ten complete papers per domain were sampled; each contributes ten valid Gemini-Pro-002 MCQs.

| Domain | Papers | Questions | No context | Original | KUs | End-to-end time |
|---|---:|---:|---:|---:|---:|---:|
| Physics | 10 | 100 | 82.0% | 95.0% | 96.0% | 1,781.6 s (29:42) |
| Medical | 10 | 100 | 72.0% | 95.0% | 94.0% | 1,886.4 s (31:26) |

Two Physics and one Medical original-context outputs remained malformed after one retry and count
as incorrect. The one-point Physics KU/original inversion is within sampling noise at 100 questions
and does not imply that KUs contain more information than the source.

The two runs average 1,834 seconds per ten papers, or 3:03 per paper end to end. They used the older
serial document-resolver implementation; the repository now batches those independent resolver
calls. A conservative projection for all 200 long papers is therefore approximately 9–11 hours for
KU extraction, reconciliation, and local diagnostic scoring. A remote fixed judge adds its own
provider throughput and rate-limit cost. A 1,200-token compact KU cap may reduce this further, but
must pass a separate fidelity/coverage check before it replaces the 2,500-token baseline.

## Historical comparison boundary

The paper's Qwen 2.5 7B fixed-judge long scores were Physics `[52.23–89.69], KU 79.04` and Medical
`[50.45–93.24], KU 88.29`. Comparing those directly with the self-judge pilot would confound both
the extractor and answerer. A proper reproduction will hold the recovered
`gemini-1.5-flash-8b` judge fixed and first require its full-dataset no-context score to match the
historical lower bound within the stated tolerance.
