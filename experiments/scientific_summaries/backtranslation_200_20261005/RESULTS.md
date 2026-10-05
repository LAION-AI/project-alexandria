# Local back-translation benchmark: 200 scientific summaries

200 existing no-thinking summary versions of 97 held-out papers: 97 tuned merged FP8, 97 tuned merged BF16 and six live BF16 rank-128 Gemma 4 12B outputs. This is not 200 independent papers. All original and repaired outputs remain in the evaluation. Translation never receives questions, gold answers or reasoning traces.

Only narrative windows containing a source-copy run of at least six normalized words are translated English → German → English, once. Preceding context is included where available. Metadata and explicit proof quotes are retained verbatim and audited separately.

## Complete round-trip throughput

| Model / decoding | Chosen batch | Windows | Successful round trips | Seconds | Windows/s | Summary versions/hour | Native output tokens/s | Active translation GPU hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WindyTranslate EN→DE→EN / greedy | 512 | 6379 | 6339 | 73.45 | 86.85 | 9802.7 | 7149.4 | 0.02040 |
| WindyTranslate EN→DE→EN / beam 4 | 512 | 6379 | 6360 | 125.02 | 51.02 | 5759.0 | 4178.8 | 0.03473 |
| TranslateGemma 4B EN→DE→EN / greedy | 512 | 6379 | 6379 | 34.51 | 184.84 | 20862.5 | 16905.6 | 0.00959 |

Times include tokenization, both translation directions and complete outputs for all selected windows. They exclude model loading, compilation, the disjoint-training-paper batch sweep, quality checks and QA. Summary versions/hour depends on this cohort’s copy-window density; these are repairs, not newly generated summaries. Native token rates use different model tokenizers and are not a direct work-normalized comparison. One GPU per translating worker; two Windy arms run sequentially on their worker. TranslateGemma uses BF16, vLLM continuous batching, chunked prefill, prefix caching, language-only inference and default graph/attention selection. Windy uses FP16, SDPA, token-length sorting and dynamic batch padding. No source or candidate is silently truncated.

## Numbers, units and formulas before the acceptance filter

| Arm | Numeric windows | Suspect numeric changes | Suspect unit changes | Formula windows | Suspect formula/variable changes | Raw summaries with suspect changes | Guarded summaries with suspect changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WindyTranslate EN→DE→EN / greedy | 2961 | 471 | 222 | 2089 | 452 | 170 | 0 |
| WindyTranslate EN→DE→EN / beam 4 | 2979 | 399 | 196 | 2108 | 459 | 166 | 0 |
| TranslateGemma 4B EN→DE→EN / greedy | 2993 | 281 | 139 | 2125 | 279 | 168 | 0 |

Critical signatures preserve numeric signs, decimal/scientific values, associated units, superscripts/subscripts, Greek variable names, detected mathematical expressions and inequalities. Cosmetic spacing, trailing decimal zeros and supported Unicode/LaTeX variants are normalized. General algebraic equivalence, unit conversions, spelled-out numbers and all possible scientific notation are not fully parsed. A signature mismatch is a conservative review flag, not a human-labeled error. Zero suspect changes in guarded outputs is guaranteed by rejecting/rolling back mismatching candidates; it does not mean the translator itself preserved everything. Original scientific claims are not independently certified by this check.

## Accepted changes and meaning checks

| Arm | Selected windows | Accepted windows inserted | Changed summaries | Global rollbacks | NLI below cutoff | QC seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WindyTranslate EN→DE→EN / greedy | 6379 | 1987 | 199 | 0 | 1986 | 28.59 |
| WindyTranslate EN→DE→EN / beam 4 | 6379 | 1972 | 199 | 0 | 1777 | 29.51 |
| TranslateGemma 4B EN→DE→EN / greedy | 6379 | 2398 | 198 | 0 | 2250 | 30.03 |

Acceptance was fixed before QA: successful generation, unchanged critical signatures, bidirectional NLI entailment of at least 0.90, and no candidate-window source run above five words. Full assembled narratives are checked again. Rejected passages retain their original text and copy flags. NLI compares original and repaired summary passages; it is a heuristic, not a proof of scientific equivalence or source grounding. Overlength NLI pairs are rejected rather than truncated. The small positive/negative control set is diagnostic and not a scientific-domain accuracy estimate.

## Source-copy overlap

| Arm / output | Narrative ≤5 | Exactly 6 | ≥7 | Longest run | Mean coverage ≥6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WindyTranslate EN→DE→EN / greedy / original | 0 | 0 | 200 | 34 | 15.403% |
| WindyTranslate EN→DE→EN / greedy / raw_backtranslation | 0 | 0 | 200 | 22 | 6.478% |
| WindyTranslate EN→DE→EN / greedy / guarded_backtranslation | 0 | 0 | 200 | 34 | 11.156% |
| WindyTranslate EN→DE→EN / beam 4 / original | 0 | 0 | 200 | 34 | 15.403% |
| WindyTranslate EN→DE→EN / beam 4 / raw_backtranslation | 0 | 1 | 199 | 22 | 7.043% |
| WindyTranslate EN→DE→EN / beam 4 / guarded_backtranslation | 0 | 0 | 200 | 34 | 11.291% |
| TranslateGemma 4B EN→DE→EN / greedy / original | 0 | 0 | 200 | 34 | 15.403% |
| TranslateGemma 4B EN→DE→EN / greedy / raw_backtranslation | 0 | 2 | 198 | 25 | 4.339% |
| TranslateGemma 4B EN→DE→EN / greedy / guarded_backtranslation | 0 | 0 | 200 | 34 | 9.819% |

Audit 2.1 uses normalized whitespace-delimited words against each paper’s complete untruncated source. Five is allowed, six borderline, seven or more flagged. Expanded punctuation-token diagnostics, cumulative coverage, metadata and evidence-quote checks are retained in the evidence. No paper is removed for overlap.

## Fixed historical QA

| Arm | Original correct / 2,000 | Guarded correct / 2,000 | Original accuracy | Guarded accuracy | Difference (percentage points) | Paired 95% paper-cluster interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| WindyTranslate EN→DE→EN / greedy | 1882 | 1873 | 94.10% | 93.65% | -0.45 | [-0.85, -0.05] |
| WindyTranslate EN→DE→EN / beam 4 | 1882 | 1875 | 94.10% | 93.75% | -0.35 | [-0.71, +0.00] |
| TranslateGemma 4B EN→DE→EN / greedy | 1882 | 1874 | 94.10% | 93.70% | -0.40 | [-0.87, +0.05] |

The pinned 97-paper/970-MCQ test set, historical ASCII prompt and answer parser are unchanged. Each of the 200 versions is scored on its paper’s ten questions (2,000 slots per arm). Intervals resample 97 paper clusters, retaining all versions together. These are correlated versions, not 2,000 independent trials. Exact unchanged contexts reuse cached answers; changed contexts are answered by the same Qwen2.5-7B-Instruct judge with temperature 0.5, top-p 0.95, 100 output tokens, frequency/presence penalties 1.05 and up to four invalid-answer retries. Native vLLM batch scheduling differs from earlier HTTP concurrency four, and the stochastic judge adds sampling noise.

## Conclusions

TranslateGemma 4B EN→DE→EN / greedy inserted the most guarded repair windows (2398). Speed alone is insufficient: raw signature failures, acceptance coverage, residual whole-summary copying and QA must be considered together.
WindyTranslate EN→DE→EN / greedy leaves 200/200 narratives with runs of at least seven words; 0/200 meet the strict narrative five-word limit.
WindyTranslate EN→DE→EN / beam 4 leaves 200/200 narratives with runs of at least seven words; 0/200 meet the strict narrative five-word limit.
TranslateGemma 4B EN→DE→EN / greedy leaves 200/200 narratives with runs of at least seven words; 0/200 meet the strict narrative five-word limit.

A single round trip with conservative rejection should only be adopted if the measured reduction and useful acceptance rate justify its overhead. These results do not establish that repeatedly translating until zero overlap would preserve scientific meaning. The table describes this fixed experiment; it is not a quality guarantee or a production-scale cost claim.

## Provenance and evidence

Pinned model IDs/revisions: `model_pins.json`. Exact selection, acceptance rules and code hashes: `protocol.json`. All batch sweep measurements, raw German/English candidates, numerical/formula signatures, NLI scores, rejected candidates, original/repaired QA and source-copy evidence are retained without editing the published training datasets.

Durable evidence directory: `/e/data1/datasets/playground/mmlaion/schuhmann1/scientific-backtranslation-200-20261005`. `evidence_manifest.json` lists SHA256 and byte lengths. Model weights and credentials are not included in the evidence export. Allocated node GPU hours and failed preparation overhead are recorded separately from active translation time.
