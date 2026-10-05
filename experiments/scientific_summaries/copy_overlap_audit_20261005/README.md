# Scientific summaries: source-copy findings

Completed on 2026-10-05. This audit implements the user’s limit: **at most five consecutive copied words; exactly six is borderline; seven or more is flagged**. The original full paper is the comparison source. QA scores remain unchanged.

## No-thinking summaries: Qwen-distilled Gemma 4 12B IT rank 128

| Deployment | Summaries | Narrative ≥7 | Longest copied run | Mean narrative coverage in runs ≥6 |
| --- | ---: | ---: | ---: | ---: |
| Live BF16 LoRA, concurrency 16 | 97 | 97 | 61 words | 14.91% |
| Merged BF16, concurrency 64 | 97 | 97 | 34 words | 15.74% |
| Merged FP8 weights, BF16 KV, concurrency 64 | 97 | 97 | 32 words | 15.23% |

**None of these 291 no-thinking narratives passes even the six-word tolerance.** No-thinking removes emitted reasoning; it does not enforce paraphrasing. The matches include technical phrases and formulas as well as long ordinary prose. This is a lexical-copy audit, not a plagiarism verdict.

The 61-word live-BF16 match occurs in a citation justification for `bethgelab-116`; the optimized BF16 run contains a 34-word narrative match. Evidence-quote failures are separately visible and cannot explain away these narrative results.

## Exact published distillation targets

Raw generator answers are the actual summary training targets. Corrected Qwen outputs are a separate view; reasoning traces are excluded from this prose audit.

| Collection/view | Summaries | Narrative ≥7 | Longest copied run | Mean coverage ≥6 | Evidence ≥7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen27 / raw generator training target | 865 | 865 | 40 words | 19.37% | 15 |
| Qwen27 / validated corrected final (separate view) | 865 | 865 | 42 words | 19.74% | 0 |
| Ornith9B / raw generator training target | 736 | 734 | 195 words | 16.32% | 687 |

Qwen’s existing correction/validation enforced short explicit grounding quotes. It did **not** impose the same limit on narrative sentences. Consequently all 865 corrected Qwen narratives remain flagged despite compliant proof quotes. Ornith’s frozen 736 collection retains its original validation failures; two narratives have a longest run of exactly six words and the other 734 exceed six. Both published datasets and all trained adapters are preserved.

## Other completed 97-paper comparisons

| Model/run | Stage | Summaries | Narrative ≥7 | Longest copied run | Mean coverage ≥6 |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma 4 E4B IT | raw | 97 | 84 | 35 | 7.22% |
| Gemma 4 E4B IT | corrected | 97 | 89 | 20 | 8.16% |
| Gemma 4 12B IT | raw | 97 | 97 | 34 | 17.72% |
| Gemma 4 12B IT | corrected | 97 | 97 | 34 | 18.15% |
| Ornith 1.5 9B + DFlash | raw | 97 | 97 | 246 | 32.25% |
| Ornith 1.5 9B + DFlash | corrected | 97 | 97 | 246 | 32.56% |
| Ornith 1.5 35B A3B + DFlash | raw | 97 | 96 | 124 | 23.30% |
| Ornith 1.5 35B A3B + DFlash | corrected | 97 | 96 | 124 | 23.94% |
| Qwen 3.8 27B FP8 | raw | 97 | 92 | 45 | 11.93% |
| Qwen 3.8 27B FP8 | corrected | 97 | 92 | 45 | 12.05% |

Correction does not consistently reduce narrative copying. The older native/quantized 9B and INT4 27B runs, recovery snapshots, failed drafts and returned generation/field-repair versions are listed separately in [legacy results](legacy_runs/RESULTS.md).

## Scope, counting and evidence

- Main historical scan: **2,097 full paper sources**, **31,488 JSON files**, **23,532 summary views** across 57 conditions. The main scan has 20,492 distinct source/summary pairs and zero read errors.
- Earlier repository runs: **1,133 additional condition-specific views** from 357 files, plus **9 early returned Qwen drafts**. Combined archive scans contain **24,674 condition-specific views**. This is not a count of distinct final papers; identical outputs can appear under separate model/stage/snapshot labels.
- The exact training-target scan adds a precise 865/736 target mapping, rather than treating all intermediate teacher attempts as training data. Final completed FP8 and no-thinking conditions are audited independently of the historical scan snapshot.
- The Ornith-distilled Gemma rank-64/rank-128 evaluation was still running at the historical snapshot. Its interim results are explicitly identified as a snapshot; the completion watcher audits all 97 final slots automatically and publishes their complete results.
- Deduplicate by condition, paper ID and canonical summary hash; preserve every location alias. Sources and output versions are not chosen by QA outcomes.
- Compare whitespace-delimited words after Unicode NFKC/case normalization and punctuation stripping within a word. Join PDF end-of-line hyphenation and remove soft hyphens. Preserve original word boundaries during Unicode compatibility expansion. Chemical terms, decimals and single mathematical expressions are not inflated into multiple words.
- Preserve a separate punctuation-split alphanumeric diagnostic. It is stricter for formulas and hyphenated terms and is not the primary five-word decision.
- Never concatenate separate structured statements to manufacture a long match. Longest runs include executive-summary prose and citation justifications. Narrative, explicit evidence and metadata have separate statistics; JSON also reports the strict all-fields pass count.
- Coverage is the union of summary-word positions covered by qualifying runs, divided by the narrative word count, averaged per summary. Multiple occurrences in the source do not count the same target word repeatedly. Runs of exactly six and runs of seven or more are also recorded separately.
- Titles, author names and bibliographic strings are visible in metadata diagnostics. They commonly require literal spelling, so they are not mixed into narrative coverage. This does not grant an unrecorded exception: strict all-fields flags include them.
- Exclude reasoning, prompts, source-anchor control replies, reviewer/QC verdicts and QA questions/gold from summary prose. Retain malformed returned summary text in diagnostic groups. Capped 512-token throughput probes are excluded because they were never delivered full summaries.
- No held-out paper is removed, no QA response is changed and no summary is silently regenerated or rewritten. Lexical overlap alone does not establish plagiarism or semantic quality.

## QA and production costs

The FP8 no-thinking deployment scores **914/970 = 94.23%** and takes **545.36 seconds** to generate all 97 summaries on one active GH200, including full prefill, tokenization and bounded format retries. It is about **1.37×** faster than tuned merged BF16. QA differences have paper-bootstrap intervals that include zero; superiority is not established. See [full scores, times and cost tables](../gemma_r128_fp8_97/evaluation/RESULTS.md).

Planning estimates at 85% useful capacity are about **69,820 GPU-hours for 38M**, **110,241 for 60M**, and **180,061 for 98M** FP8 outputs. **These count structurally finished outputs, not summaries that satisfy the new five-word paraphrasing limit.** With zero compliant no-thinking narratives in this test, the cost of compliant production summaries is not yet measured. Enforcing/revising the outputs and repeating the same frozen QA evaluation is needed before quoting that cost.

## Mandatory future checks

`evaluate.py`, all four active generation/LoRA evaluation workflows and the three earlier cohort evaluators now invoke the common audit. Returned attempt sidecars are also installed in the native generation helper. The in-flight jobs loaded older code, so the completion watcher applies version 2.1 before publication. The scoped [AGENTS.md](../AGENTS.md) records this requirement for subsequent work.

The checker writes `copy_overlap.json`, per-summary CSV/JSONL, longest matches with normalized source/summary offsets and a results section alongside unchanged QA. Eight focused regression cases verify the five/six/seven threshold, field separation, coverage, metadata, nested evidence, long-copy handling and Unicode boundary preservation.

## Files and durable archive

- [Main historical results](RESULTS.md), `report.json`, `per_summary.csv` and `longest_examples.jsonl.gz`.
- [Actual training targets](training_targets/RESULTS.md), their aggregate report, CSV and compact evidence.
- [Older/recovered runs](legacy_runs/RESULTS.md) and [early Qwen drafts](legacy_early_drafts/RESULTS.md).
- Full per-fragment evidence remains outside GitHub because the main JSONL is about 2.2 GB. Durable archive: `/e/data1/datasets/playground/mmlaion/schuhmann1/scientific-copy-overlap-audit-20261005`.
- `archive_checksums.json` lists SHA256 hashes and sizes of complete evidence artifacts; GitHub carries compact review evidence. The earlier punctuation-split-only draft is retained in the durable archive for provenance, not used as the primary word count.
- `legacy_evaluation_hooks/` contains updated runtime evaluator code and before/after hashes; the original past-run source-code archives remain intact.
