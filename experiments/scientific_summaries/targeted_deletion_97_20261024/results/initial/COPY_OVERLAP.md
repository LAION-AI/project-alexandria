## Source-copy overlap audit

Exact normalized contiguous word overlap against each complete paper source. Five words are allowed, six are borderline, seven or more are flagged. Narrative prose, evidence quotes and bibliographic metadata are reported separately. Coverage counts each summary word once per threshold within each fragment. Separate statements are never concatenated to create matches.

| Condition | Audited outputs | Narrative ≤5 | Narrative exactly 6 | Narrative ≥7 | Longest narrative match | Mean narrative coverage ≥6 | Evidence ≥7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 97 | 0 | 0 | 97 | 25 | 9.04% | 34 |
| delete_fillers | 97 | 0 | 0 | 97 | 25 | 9.04% | 34 |
| delete_articles | 97 | 0 | 1 | 96 | 25 | 5.58% | 34 |
| delete_soft_modifiers | 97 | 0 | 1 | 96 | 25 | 5.56% | 34 |
| delete_pos_guarded | 97 | 3 | 3 | 91 | 21 | 3.51% | 34 |

Primary counts use whitespace-delimited words after Unicode normalization and punctuation stripping; mathematical expressions, decimal numbers and hyphenated terms are not split into artificial extra words. A separate punctuation-split diagnostic is retained in the JSONL. NFKC normalization, casefolding, soft-hyphen removal and PDF line-end word joining are used. Offsets refer to normalized text and word positions. Technical names and ordinary scientific phrases can match literally; lexical overlap alone does not establish plagiarism. All output flags remain visible and no paper is removed from QA. Reasoning traces, instructions and reviewer verdicts are not summary prose. Bibliographic names and titles are included in the metadata diagnostics, not silently mixed into paraphrase statistics.
