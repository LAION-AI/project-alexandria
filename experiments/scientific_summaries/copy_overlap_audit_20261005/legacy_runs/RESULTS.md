# Earlier scientific-summary runs: source-copy audit

97 paper sources; 1133 saved/returned summary views.

## Source-copy overlap audit

Exact normalized contiguous word overlap against each complete paper source. Five words are allowed, six are borderline, seven or more are flagged. Narrative prose, evidence quotes and bibliographic metadata are reported separately. Coverage counts each summary word once per threshold within each fragment. Separate statements are never concatenated to create matches.

| Condition | Audited outputs | Narrative ≤5 | Narrative exactly 6 | Narrative ≥7 | Longest narrative match | Mean narrative coverage ≥6 | Evidence ≥7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy ornith15_9b / failed draft | 6 | 0 | 0 | 6 | 32 | 31.36% | 5 |
| Legacy ornith15_9b / final | 97 | 0 | 0 | 97 | 69 | 32.34% | 0 |
| Legacy ornith15_9b / returned field_repair / unparsed text diagnostic | 16 | 0 | 0 | 16 | 32 | 44.62% | 0 |
| Legacy ornith15_9b / returned generation | 115 | 0 | 0 | 115 | 81 | 30.90% | 109 |
| Legacy ornith15_9b / returned generation / unparsed text diagnostic | 51 | 0 | 0 | 51 | 236 | 42.96% | 0 |
| Legacy ornith15_9b/v2_failed_archive / failed draft | 4 | 0 | 0 | 4 | 32 | 43.54% | 3 |
| Legacy ornith15_9b/v2_failed_archive / returned field_repair / unparsed text diagnostic | 16 | 0 | 0 | 16 | 32 | 44.62% | 0 |
| Legacy ornith15_9b/v2_failed_archive / returned generation | 2 | 0 | 0 | 2 | 32 | 43.09% | 2 |
| Legacy ornith15_9b_dflash20/baseline / final | 20 | 0 | 0 | 20 | 63 | 30.85% | 0 |
| Legacy ornith15_9b_dflash20/baseline / returned generation | 20 | 0 | 0 | 20 | 63 | 29.54% | 20 |
| Legacy ornith15_9b_dflash20/dflash / final | 20 | 0 | 1 | 19 | 65 | 35.23% | 0 |
| Legacy ornith15_9b_dflash20/dflash / returned generation | 20 | 0 | 1 | 19 | 65 | 33.52% | 20 |
| Legacy qwen27b / failed draft | 1 | 0 | 0 | 1 | 24 | 4.57% | 0 |
| Legacy qwen27b / final | 97 | 0 | 3 | 94 | 57 | 14.93% | 0 |
| Legacy qwen27b / returned field_repair | 104 | 3 | 11 | 90 | 57 | 14.92% | 0 |
| Legacy qwen27b / returned field_repair / unparsed text diagnostic | 4 | 0 | 0 | 4 | 8 | 6.83% | 0 |
| Legacy qwen27b / returned generation | 104 | 0 | 3 | 101 | 57 | 14.48% | 3 |
| Legacy qwen35_9b / failed draft | 8 | 0 | 0 | 8 | 31 | 33.80% | 1 |
| Legacy qwen35_9b / final | 97 | 2 | 2 | 93 | 41 | 24.55% | 0 |
| Legacy qwen35_9b / returned generation | 102 | 1 | 3 | 98 | 41 | 24.37% | 102 |
| Legacy qwen35_9b / returned generation / unparsed text diagnostic | 2 | 0 | 0 | 2 | 25 | 7.93% | 0 |
| Legacy qwen35_9b/pre_completion_recovery / failed draft | 8 | 0 | 0 | 8 | 31 | 33.80% | 1 |
| Legacy qwen35_9b/pre_completion_recovery / final | 94 | 2 | 2 | 90 | 41 | 24.67% | 0 |
| Legacy qwen35_9b/pre_completion_recovery / returned generation | 101 | 1 | 3 | 97 | 41 | 24.60% | 101 |
| Legacy qwen35_9b/pre_completion_recovery / returned generation / unparsed text diagnostic | 2 | 0 | 0 | 2 | 25 | 7.93% | 0 |
| Legacy qwen35_9b_v2_failed_20261002T093902 / failed draft | 2 | 0 | 0 | 2 | 24 | 36.50% | 2 |
| Legacy qwen35_9b_v2_failed_20261002T093902 / returned field_repair | 10 | 0 | 4 | 6 | 22 | 49.76% | 4 |
| Legacy qwen35_9b_v2_failed_20261002T093902 / returned field_repair / unparsed text diagnostic | 4 | 0 | 0 | 4 | 11 | 28.37% | 0 |
| Legacy qwen35_9b_v2_failed_20261002T093902 / returned generation | 2 | 0 | 0 | 2 | 24 | 22.76% | 2 |
| Legacy qwen35_9b_v3_failed_20261002T202208 / failed draft | 2 | 0 | 0 | 2 | 24 | 23.85% | 1 |
| Legacy qwen35_9b_v3_failed_20261002T202208 / returned generation | 2 | 0 | 0 | 2 | 24 | 22.76% | 2 |

Primary counts use whitespace-delimited words after Unicode normalization and punctuation stripping; mathematical expressions, decimal numbers and hyphenated terms are not split into artificial extra words. A separate punctuation-split diagnostic is retained in the JSONL. NFKC normalization, casefolding, soft-hyphen removal and PDF line-end word joining are used. Offsets refer to normalized text and word positions. Technical names and ordinary scientific phrases can match literally; lexical overlap alone does not establish plagiarism. All output flags remain visible and no paper is removed from QA. Reasoning traces, instructions and reviewer verdicts are not summary prose. Bibliographic names and titles are included in the metadata diagnostics, not silently mixed into paraphrase statistics.

Legacy repository final/recovered summaries, failed drafts and returned generator/field-repair text; source-anchor selections excluded as control messages
