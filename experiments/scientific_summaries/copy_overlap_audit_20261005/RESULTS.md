# Scientific summaries: systematic source-copy audit

2097 paper sources; 23532 deduplicated summary views across 57 conditions; 20492 unique source/summary pairs. Intermediate returned attempts are separate from delivered final/raw/corrected views.

## Source-copy overlap audit

Exact normalized contiguous word overlap against each complete paper source. Five words are allowed, six are borderline, seven or more are flagged. Narrative prose, evidence quotes and bibliographic metadata are reported separately. Coverage counts each summary word once per threshold within each fragment. Separate statements are never concatenated to create matches.

| Condition | Audited outputs | Narrative ≤5 | Narrative exactly 6 | Narrative ≥7 | Longest narrative match | Mean narrative coverage ≥6 | Evidence ≥7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ornith9B distillation / final raw | 736 | 0 | 2 | 734 | 195 | 16.32% | 687 |
| Ornith9B distillation / returned generation attempts | 785 | 1 | 2 | 782 | 195 | 16.31% | 729 |
| Ornith9B distillation / returned generation attempts / unparsed text diagnostic | 111 | 12 | 0 | 99 | 122 | 30.46% | 0 |
| Qwen27 distillation / final corrected | 865 | 0 | 0 | 865 | 42 | 19.74% | 0 |
| Qwen27 distillation / returned correction attempts | 13907 | 0 | 0 | 13907 | 67 | 18.15% | 40 |
| Qwen27 distillation / returned correction attempts / unparsed text diagnostic | 17 | 0 | 0 | 17 | 49 | 17.26% | 0 |
| Qwen27 distillation / returned generation attempts | 3103 | 0 | 0 | 3103 | 67 | 18.24% | 56 |
| Qwen27 distillation / returned generation attempts / unparsed text diagnostic | 25 | 0 | 0 | 25 | 71 | 22.46% | 0 |
| ornith-distillation-736-20261004/gemma12_base / saved output | 97 | 0 | 2 | 95 | 50 | 22.29% | 8 |
| ornith-distillation-736-20261004/gemma12_r128 / returned attempts | 40 | 0 | 0 | 40 | 110 | 21.44% | 33 |
| ornith-distillation-736-20261004/gemma12_r128 / returned attempts / unparsed text diagnostic | 7 | 0 | 0 | 7 | 187 | 25.11% | 0 |
| ornith-distillation-736-20261004/gemma12_r128 / saved output | 40 | 0 | 0 | 40 | 110 | 21.44% | 33 |
| ornith-distillation-736-20261004/gemma12_r64 / returned attempts | 50 | 0 | 0 | 50 | 125 | 25.74% | 43 |
| ornith-distillation-736-20261004/gemma12_r64 / returned attempts / unparsed text diagnostic | 11 | 0 | 1 | 10 | 33 | 21.95% | 0 |
| ornith-distillation-736-20261004/gemma12_r64 / saved output | 46 | 0 | 0 | 46 | 125 | 25.96% | 40 |
| scientific-distillation-865-20261004/gemma12_base / returned attempts | 97 | 0 | 2 | 95 | 50 | 22.29% | 8 |
| scientific-distillation-865-20261004/gemma12_base / returned attempts / unparsed text diagnostic | 5 | 0 | 0 | 5 | 56 | 25.77% | 0 |
| scientific-distillation-865-20261004/gemma12_base / saved output | 97 | 0 | 2 | 95 | 50 | 22.29% | 8 |
| scientific-distillation-865-20261004/gemma12_r128 / returned attempts | 95 | 0 | 0 | 95 | 42 | 19.31% | 2 |
| scientific-distillation-865-20261004/gemma12_r128 / returned attempts / unparsed text diagnostic | 12 | 0 | 0 | 12 | 52 | 23.69% | 0 |
| scientific-distillation-865-20261004/gemma12_r128 / saved output | 95 | 0 | 0 | 95 | 42 | 19.31% | 2 |
| scientific-distillation-865-20261004/gemma12_r64 / returned attempts | 92 | 0 | 0 | 92 | 45 | 19.24% | 19 |
| scientific-distillation-865-20261004/gemma12_r64 / returned attempts / unparsed text diagnostic | 11 | 0 | 0 | 11 | 50 | 28.57% | 0 |
| scientific-distillation-865-20261004/gemma12_r64 / saved output | 90 | 0 | 0 | 90 | 45 | 19.22% | 19 |
| scientific-gemma-r128-fp8-20261005/no_thinking_matched / returned attempts | 97 | 0 | 0 | 97 | 61 | 14.91% | 34 |
| scientific-gemma-r128-fp8-20261005/no_thinking_matched / returned attempts / unparsed text diagnostic | 5 | 0 | 0 | 5 | 24 | 12.50% | 0 |
| scientific-gemma-r128-fp8-20261005/no_thinking_matched / saved output | 97 | 0 | 0 | 97 | 61 | 14.91% | 34 |
| scientific-gemma-r128-fp8-20261005/no_thinking_merged / returned attempts | 99 | 0 | 0 | 99 | 34 | 15.60% | 36 |
| scientific-gemma-r128-fp8-20261005/no_thinking_merged / returned attempts / unparsed text diagnostic | 2 | 0 | 0 | 2 | 50 | 18.42% | 0 |
| scientific-gemma-r128-fp8-20261005/no_thinking_merged / saved output | 97 | 0 | 0 | 97 | 34 | 15.74% | 36 |
| scientific-gemma-r128-fp8-20261005/thinking_reference / saved output | 95 | 0 | 0 | 95 | 42 | 19.31% | 2 |
| scientific-gemma-r128-no-thinking-20261004/no_thinking_matched / returned attempts | 97 | 0 | 0 | 97 | 61 | 14.91% | 34 |
| scientific-gemma-r128-no-thinking-20261004/no_thinking_matched / returned attempts / unparsed text diagnostic | 5 | 0 | 0 | 5 | 24 | 12.50% | 0 |
| scientific-gemma-r128-no-thinking-20261004/no_thinking_matched / saved output | 97 | 0 | 0 | 97 | 61 | 14.91% | 34 |
| scientific-gemma-r128-no-thinking-20261004/no_thinking_merged / returned attempts | 99 | 0 | 0 | 99 | 34 | 15.60% | 36 |
| scientific-gemma-r128-no-thinking-20261004/no_thinking_merged / returned attempts / unparsed text diagnostic | 2 | 0 | 0 | 2 | 50 | 18.42% | 0 |
| scientific-gemma-r128-no-thinking-20261004/no_thinking_merged / saved output | 97 | 0 | 0 | 97 | 34 | 15.74% | 36 |
| scientific-gemma-r128-no-thinking-20261004/thinking_reference / saved output | 95 | 0 | 0 | 95 | 42 | 19.31% | 2 |
| scientific-gemma4-eval-97/gemma4_12b / corrected | 97 | 0 | 0 | 97 | 34 | 18.15% | 0 |
| scientific-gemma4-eval-97/gemma4_12b / raw | 97 | 0 | 0 | 97 | 34 | 17.72% | 36 |
| scientific-gemma4-eval-97/gemma4_12b / returned attempts | 190 | 0 | 1 | 189 | 34 | 17.79% | 62 |
| scientific-gemma4-eval-97/gemma4_12b / returned attempts / unparsed text diagnostic | 10 | 0 | 0 | 10 | 65 | 17.04% | 0 |
| scientific-gemma4-eval-97/gemma4_e4b / corrected | 97 | 0 | 8 | 89 | 20 | 8.16% | 0 |
| scientific-gemma4-eval-97/gemma4_e4b / raw | 97 | 0 | 13 | 84 | 35 | 7.22% | 93 |
| scientific-gemma4-eval-97/gemma4_e4b / returned attempts | 212 | 0 | 23 | 189 | 35 | 7.48% | 203 |
| scientific-gemma4-eval-97/gemma4_e4b / returned attempts / unparsed text diagnostic | 63 | 0 | 3 | 60 | 84 | 24.14% | 0 |
| scientific-ornith-35b-dflash-eval-97/ornith35_dflash / corrected | 97 | 0 | 1 | 96 | 124 | 23.94% | 0 |
| scientific-ornith-35b-dflash-eval-97/ornith35_dflash / raw | 97 | 0 | 1 | 96 | 124 | 23.30% | 89 |
| scientific-ornith-35b-dflash-eval-97/ornith35_dflash / returned attempts | 220 | 0 | 3 | 217 | 124 | 23.41% | 184 |
| scientific-ornith-35b-dflash-eval-97/ornith35_dflash / returned attempts / unparsed text diagnostic | 14 | 0 | 0 | 14 | 124 | 41.28% | 0 |
| scientific-ornith-dflash-eval-97/ornith_dflash / corrected | 97 | 0 | 0 | 97 | 246 | 32.56% | 0 |
| scientific-ornith-dflash-eval-97/ornith_dflash / raw | 97 | 0 | 0 | 97 | 246 | 32.25% | 97 |
| scientific-ornith-dflash-eval-97/ornith_dflash / returned attempts | 181 | 0 | 0 | 181 | 246 | 32.43% | 181 |
| scientific-ornith-dflash-eval-97/ornith_dflash / returned attempts / unparsed text diagnostic | 54 | 0 | 0 | 54 | 156 | 52.61% | 0 |
| scientific-ornith-dflash-eval-97/qwen38_fp8 / corrected | 97 | 0 | 5 | 92 | 45 | 12.05% | 0 |
| scientific-ornith-dflash-eval-97/qwen38_fp8 / raw | 97 | 0 | 5 | 92 | 45 | 11.93% | 1 |
| scientific-ornith-dflash-eval-97/qwen38_fp8 / returned attempts | 200 | 0 | 10 | 190 | 47 | 12.46% | 4 |

Primary counts use whitespace-delimited words after Unicode normalization and punctuation stripping; mathematical expressions, decimal numbers and hyphenated terms are not split into artificial extra words. A separate punctuation-split diagnostic is retained in the JSONL. NFKC normalization, casefolding, soft-hyphen removal and PDF line-end word joining are used. Offsets refer to normalized text and word positions. Technical names and ordinary scientific phrases can match literally; lexical overlap alone does not establish plagiarism. All output flags remain visible and no paper is removed from QA. Reasoning traces, instructions and reviewer verdicts are not summary prose. Bibliographic names and titles are included in the metadata diagnostics, not silently mixed into paraphrase statistics.

## Scope and provenance

Preserved final, raw, corrected, recovery snapshots and returned generator/corrector attempts; capped throughput probes excluded; snapshots of running evaluations are identified by paths and scan time

Read errors: 0. JSONL evidence records include source/output hashes, paths, normalized offsets and longest matching excerpts.
