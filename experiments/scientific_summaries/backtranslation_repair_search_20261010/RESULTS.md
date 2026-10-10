# Back-translation parameter search and targeted repair

## Result

Cached Gemma-4-12B-it rank-128 KU outputs are the input, with 500-word and 1000-word extraction profiles. TranslateGemma-4B performs the round trips; Gemma E4B is a subsequent local editor. Qwen2.5-7B remains the QA answerer. Twenty frozen papers; 200 MCQs per condition. Every condition retains all papers, original failures and question slots. No Knowledge Units were regenerated. Full source texts were used for every word-overlap audit.

The strict rule allows at most five consecutive normalized whitespace words. Exactly six counts as a strict failure, even though the older seven-word flag labels six borderline. Bibliographic metadata is reported separately using the pre-existing classification; the historical all-factual-string audit remains available.

| Condition | Chunk words | QA accuracy | Papers passing ≤5 | Papers violating >5 | Maximum copied run | Mean copied-word coverage ≥6 | Δ QA vs raw (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw KU | 500 | 88.0% | 0/20 | 100% | 35 | 10.889% | +0.0 pp [+0.0, +0.0] |
| Raw KU | 1000 | 84.5% | 0/20 | 100% | 37 | 9.810% | +0.0 pp [+0.0, +0.0] |
| bt | 500 | 88.0% | 0/20 | 100% | 35 | 7.474% | +0.0 pp [-1.5, +1.5] |
| bt | 1000 | 84.0% | 1/20 | 95% | 37 | 7.117% | -0.5 pp [-2.0, +1.0] |
| full_greedy | 500 | 88.0% | 0/20 | 100% | 35 | 5.625% | +0.0 pp [-2.0, +2.0] |
| full_greedy | 1000 | 83.5% | 1/20 | 95% | 37 | 5.538% | -1.0 pp [-3.0, +1.0] |
| sample035 | 500 | 89.0% | 0/20 | 100% | 35 | 5.129% | +1.0 pp [-1.0, +3.0] |
| sample035 | 1000 | 83.5% | 1/20 | 95% | 37 | 5.127% | -1.0 pp [-3.0, +1.0] |
| sample070 | 500 | 88.0% | 0/20 | 100% | 35 | 4.525% | +0.0 pp [-2.5, +2.0] |
| sample070 | 1000 | 82.5% | 1/20 | 95% | 37 | 4.222% | -2.0 pp [-4.0, +0.0] |
| chain100 | 500 | 87.0% | 0/20 | 100% | 35 | 4.663% | -1.0 pp [-3.0, +1.0] |
| chain100 | 1000 | 83.0% | 2/20 | 90% | 37 | 4.458% | -1.5 pp [-3.0, +0.0] |
| union | 500 | 87.0% | 0/20 | 100% | 35 | 3.927% | -1.0 pp [-2.5, +0.0] |
| union | 1000 | 83.5% | 2/20 | 90% | 37 | 3.772% | -1.0 pp [-3.0, +1.0] |
| hybrid | 500 | 88.0% | 1/20 | 95% | 17 | 1.623% | +0.0 pp [-1.5, +1.5] |
| hybrid | 1000 | 85.0% | 9/20 | 55% | 35 | 1.308% | +0.5 pp [+0.0, +1.5] |
| hybrid080 | 500 | 88.5% | 1/20 | 95% | 17 | 1.549% | +0.5 pp [-1.0, +2.0] |
| hybrid080 | 1000 | 85.0% | 9/20 | 55% | 35 | 1.308% | +0.5 pp [+0.0, +1.5] |
| hybrid070 | 500 | 87.5% | 1/20 | 95% | 17 | 1.498% | -0.5 pp [-1.5, +0.0] |
| hybrid070 | 1000 | 84.5% | 9/20 | 55% | 35 | 1.308% | +0.0 pp [+0.0, +0.0] |
| union_after_hybrid | 500 | 87.5% | 1/20 | 95% | 17 | 1.623% | -0.5 pp [-1.5, +0.0] |
| union_after_hybrid | 1000 | 85.0% | 9/20 | 55% | 35 | 1.308% | +0.5 pp [-1.0, +2.0] |
| localedit | 500 | 87.5% | 3/20 | 85% | 17 | 1.104% | -0.5 pp [-1.5, +0.0] |
| localedit | 1000 | 84.5% | 11/20 | 45% | 35 | 1.016% | +0.0 pp [-1.5, +1.5] |
| localedit080 | 500 | 87.5% | 4/20 | 80% | 17 | 1.047% | -0.5 pp [-1.5, +0.0] |
| localedit080 | 1000 | 85.0% | 11/20 | 45% | 35 | 1.016% | +0.5 pp [+0.0, +1.5] |
| localedit070 | 500 | 87.5% | 4/20 | 80% | 17 | 1.047% | -0.5 pp [-1.5, +0.0] |
| localedit070 | 1000 | 85.0% | 11/20 | 45% | 35 | 1.016% | +0.5 pp [+0.0, +1.5] |
| union_after_localedit | 500 | 88.0% | 3/20 | 85% | 17 | 1.104% | +0.0 pp [-1.5, +1.5] |
| union_after_localedit | 1000 | 85.0% | 11/20 | 45% | 35 | 1.016% | +0.5 pp [+0.0, +1.5] |
| gemmaedit | 500 | 87.5% | 10/20 | 50% | 17 | 0.669% | -0.5 pp [-1.5, +0.0] |
| gemmaedit | 1000 | 85.0% | 12/20 | 40% | 35 | 0.798% | +0.5 pp [+0.0, +1.5] |
| gemmaedit080 | 500 | 88.0% | 10/20 | 50% | 17 | 0.669% | +0.0 pp [-1.5, +1.5] |
| gemmaedit080 | 1000 | 85.0% | 12/20 | 40% | 35 | 0.729% | +0.5 pp [+0.0, +1.5] |
| gemmaedit070 | 500 | 88.0% | 10/20 | 50% | 17 | 0.669% | +0.0 pp [-1.5, +1.5] |
| gemmaedit070 | 1000 | 85.0% | 12/20 | 40% | 35 | 0.729% | +0.5 pp [+0.0, +1.5] |
| union_after_gemmaedit | 500 | 88.0% | 10/20 | 50% | 17 | 0.669% | +0.0 pp [-1.5, +1.5] |
| union_after_gemmaedit | 1000 | 85.0% | 12/20 | 40% | 35 | 0.798% | +0.5 pp [+0.0, +1.5] |
| iterative | 500 | 88.0% | 11/20 | 45% | 17 | 0.413% | +0.0 pp [-1.5, +1.5] |
| iterative | 1000 | 84.5% | 12/20 | 40% | 31 | 0.551% | +0.0 pp [-2.0, +2.0] |
| iterative080 | 500 | 87.5% | 11/20 | 45% | 17 | 0.413% | -0.5 pp [-1.5, +0.0] |
| iterative080 | 1000 | 85.0% | 12/20 | 40% | 31 | 0.551% | +0.5 pp [+0.0, +1.5] |
| iterative070 | 500 | 88.0% | 11/20 | 45% | 17 | 0.413% | +0.0 pp [-1.5, +1.5] |
| iterative070 | 1000 | 85.5% | 12/20 | 40% | 31 | 0.551% | +1.0 pp [+0.0, +2.5] |

## Conclusions

**No tested setting achieved zero strict five-word violations across all 20 papers.** The preferred conservative repair is the final iterative Gemma E4B stage with bidirectional NLI at least 0.90. It keeps critical signatures and QA while greatly reducing copied text, but it is not a compliant zero-overlap production setting.

| Conservative final pipeline | Raw QA | Final QA | Raw copied-word coverage >=6 | Final coverage >=6 | Papers still violating >5 |
|---|---:|---:|---:|---:|---:|
| 500-word KUs | 88.0% | 88.0% | 10.889% | 0.413% | 9/20 (45%) |
| 1000-word KUs | 84.5% | 84.5% | 9.810% | 0.551% | 8/20 (40%) |

The two coverage reductions are about 96.2% and 94.4% relative to raw KUs. Copied-word coverage and the fraction of papers with any violation are different quantities. The matched QA differences are 0.0 percentage points, with paper-bootstrap 95% intervals of [-1.5, +1.5] and [-2.0, +2.0] points. This does not establish quality equivalence on unseen papers.

Residual cases include mathematical notation, official names of classification systems and questionnaires, assay reagent quantities, and a long clinical respiratory inclusion criterion. These failures remain visible in every score and in the retained private residual audit. No metadata or formula exception was introduced to manufacture zero violations. Guard failure is a conservative rejection and is not always proof that a proposed rewrite was scientifically wrong.

Higher translation temperatures and different seeds alone did not solve the problem. The original repair also skipped long labels, aliases and targets. Full-field rewrites sometimes returned only a phrase; those candidates were rejected. Bounded local edits kept the rest of the field intact. Finally, retaining safe partial progress fixed the all-or-nothing loop, with every stage still checked against the immutable original field.

The stricter 0.90 semantic cutoff is preferred because the lower cutoffs produced no additional paper-level compliance in the final stage. Small accuracy differences among those variants are exploratory and may be sampling noise. The remaining cases need another repair approach or explicitly defined handling of mathematical notation and official names; simply repeating the same back-translation recipe cannot currently be advertised as a zero-violation solution.

## Experimental settings

- `bt`: earlier greedy TranslateGemma repair of descriptive prose only, reused unchanged.
- `full_greedy`: all factual text fields, greedy chained round trips, eight attempts.
- `sample035` / `sample070`: temperature 0.35 / 0.70, independent seeds and fresh original English input per attempt.
- `chain100`: temperature 1.0, varied seeds, only signature-safe intermediate English is chained.
- `union`: all safe BT candidates pooled; highest bidirectional entailment per original field; no QA used.
- `hybrid`: targeted Qwen2.5-7B full-field paraphrase after that pool. Minimum bidirectional NLI 0.90.
- `localedit`: bounded local substring edits after the guarded hybrid; untouched text is preserved automatically. Minimum bidirectional NLI 0.90.
- `gemmaedit`: pinned google/gemma-4-E4B-it performs bounded local edits after the guarded Qwen local-edit output; thinking disabled. QA remains Qwen2.5-7B.
- `iterative`: Gemma E4B keeps safe partial improvements when six-word copied coverage falls without increasing the longest run. Every iteration is checked against the immutable original field; numbers/formulas must remain unchanged. Remaining overlap is re-scanned before the next attempt.
- Suffix `080` / `070`: explicit lower semantic cutoffs, 0.80 / 0.70, on the same candidate bank; numeric/formula and source-overlap checks stay unchanged. These variants are less conservative.

Translation uses pinned google/translategemma-4b-it, English → German → English, BF16, batch 512, context 2048, output cap 768 per leg, prefix caching and chunked prefill. All eight temperature/seed attempts are QA blind. All semantic comparisons are against the saved original field. Numeric, unit, variable, operator and formula signatures are checked per field and across the final document. IDs, source fingerprints and graph cardinality stay fixed. Entity label replacements propagate consistently across aliases and relationships.

The local-edit jobs use the native vLLM sampler to avoid repeated FlashInfer sampling-kernel rebuilds; temperatures/top-p and QA protocol remain the same. Every job, including both failed launches, is included in allocation accounting.

## QA protocol and limits

Pinned Qwen/Qwen2.5-7B-Instruct, historical Alexandria MCQ prompts and parser, temperature 0.5, top-p 0.95, presence/frequency penalties 1.05, 100 output tokens, up to five parser attempts. No source or context was truncated. This is a parameter search on a repeatedly used 20-paper holdout, so the preferred setting still needs fresh validation. Entailment and critical-signature checks are useful filters and do not prove that every scientific fact survived.

## Compute

| Job | State | Seconds | Reserved GPU-hours |
|---|---|---:|---:|
| 2260526 | FAILED | 278 | 0.309 |
| 2260593 | COMPLETED | 536 | 0.596 |
| 2260730 | COMPLETED | 502 | 0.558 |
| 2260804 | FAILED | 57 | 0.063 |
| 2260826 | COMPLETED | 305 | 0.339 |
| 2260985 | COMPLETED | 365 | 0.406 |
| 2261045 | COMPLETED | 288 | 0.320 |

Total reserved compute: **2.590 GPU-hours**, including startup, compilation, QA, worker idle time, and both failed launches. Active translation/fallback/NLI timings and token counts remain in the per-worker performance JSON; these are not interchangeable with reserved GPU-hours.

## Audit continuity

The table separates embedded author/title/DOI/citation/reference attributes and exact document-title strings as bibliography using the classification already recorded before this search. No mathematical or official-name exception is added. The historical audit of all factual strings is also retained in metrics.json and copy_overlap.json. Under that historical audit, the final 500-word pipeline passes only 2/20 papers (maximum 23 copied words); the final 1000-word pipeline passes 4/20 (maximum 31). The 11/20 and 12/20 narrative-only passes must not be described as passes of every textual field.

## Public evidence

[All 44 conditions](metrics.json), [frozen protocol](protocol.json), [per-paper overlap](copy_overlap.json), [overlap CSV](copy_overlap.csv), [all 8,800 scored QA slots](qa_predictions.jsonl.gz), and [critical-check summary](critical_checks.json) are included. The QA export retains predictions, gold indices, source/context hashes and every original slot. It omits source bodies, prompts, model reasoning, and complete API responses. The residual passages and complete private traces remain in experiment storage.
