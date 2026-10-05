# Earlier scientific-summary runs: source-copy audit

2 paper sources; 9 saved/returned summary views.

## Source-copy overlap audit

Exact normalized contiguous word overlap against each complete paper source. Five words are allowed, six are borderline, seven or more are flagged. Narrative prose, evidence quotes and bibliographic metadata are reported separately. Coverage counts each summary word once per threshold within each fragment. Separate statements are never concatenated to create matches.

| Condition | Audited outputs | Narrative ≤5 | Narrative exactly 6 | Narrative ≥7 | Longest narrative match | Mean narrative coverage ≥6 | Evidence ≥7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy Qwen27 early failed draft snapshot / returned full_summary_repair | 3 | 0 | 0 | 3 | 11 | 13.20% | 0 |
| Legacy Qwen27 early failed draft snapshot / returned generation | 6 | 0 | 0 | 6 | 14 | 15.44% | 0 |

Primary counts use whitespace-delimited words after Unicode normalization and punctuation stripping; mathematical expressions, decimal numbers and hyphenated terms are not split into artificial extra words. A separate punctuation-split diagnostic is retained in the JSONL. NFKC normalization, casefolding, soft-hyphen removal and PDF line-end word joining are used. Offsets refer to normalized text and word positions. Technical names and ordinary scientific phrases can match literally; lexical overlap alone does not establish plagiarism. All output flags remain visible and no paper is removed from QA. Reasoning traces, instructions and reviewer verdicts are not summary prose. Bibliographic names and titles are included in the metadata diagnostics, not silently mixed into paraphrase statistics.

Legacy repository final/recovered summaries, failed drafts and returned generator/field-repair text; source-anchor selections excluded as control messages
