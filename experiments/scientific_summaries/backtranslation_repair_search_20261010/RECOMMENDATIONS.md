# Repair recommendations for summaries and Knowledge Units

## What is supported by measurements

The experimental input is the **Gemma-4-12B-it Qwen27 KU LoRA rank128**, used at 500 and 1000 source words per chunk. The same trained weights support both profiles. The targeted editor is the unadapted [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it), **BF16 with thinking disabled**. E4B names the model; this experiment does not benchmark 4-bit editor quantization. TranslateGemma-4B is a separate translation model.

The tested chain is saved KUs → guarded TranslateGemma candidates → Qwen2.5 full-field repair → Qwen2.5 local repair → Gemma E4B local repair → Gemma E4B iterative local repair. All stages select changes without evaluation questions or QA scores. Gemma is useful for repairing residual phrases, but its effect cannot be attributed to a standalone Gemma-only pipeline.

| Source chunk | Raw QA | Final conservative-chain QA | Raw copied-word coverage >=6 | Final coverage >=6 | Narrative-only five-word passes | Historical all-factual-field passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 words | 88.0% | 88.0% | 10.889% | 0.413% | 11/20 | 2/20 |
| 1000 words | 84.5% | 84.5% | 9.810% | 0.551% | 12/20 | 4/20 |

These are measured 20-paper / 200-question conditions for the 12B KU generator, not repair scores for the E4B KU adapter, summary adapters, or other corpora. Bibliography is separated using the pre-existing diagnostic; official names and formulas still count as narrative when present in factual fields. Maximum final narrative runs are 17 and 31 words. No tested setting eliminates all violations. QA differences versus raw are 0.0 percentage points with paired paper-bootstrap 95% intervals of [-1.5,+1.5] and [-2.0,+2.0].

## Proposed operational repair workflow

1. Generate the full summary or sequential KU graph, then scan every factual text field against its complete, untruncated original source. Report bibliography and explicit evidence separately; retain the historical audit. Treat six or more consecutive normalized whitespace words as a strict failure.
2. Select only overlapping text. Include enough surrounding sentence context to preserve meaning. A cheap guarded English → German → English round is an optional first attempt; accept only verified improvements. It reduces copying but is not a complete remedy in the measured cohorts.
3. Ask Gemma E4B for bounded local substitutions rather than a new full-paper generation. The tested format is `{"edits":[{"old":"exact substring","new":"equivalent wording"}]}`. Preserve unaffected text programmatically, validate the JSON, require the original substring to exist, and reject missing or unfinished output. Each old substring uses at most 16 words; at most 16 edits per request. Avoid arbitrary filler, punctuation-only tricks, or artificial hyphenation.
4. Check every candidate against the immutable pre-repair field: numbers, units, variables, operators, mathematical expressions, qualifiers and negations must survive. The measured filters use critical_values v1.1 and bidirectional DeBERTa entailment >=0.90; these are conservative filters, not proof of scientific identity. Propagate equivalent entity-name replacements consistently to aliases and relationship targets; preserve IDs, graph cardinality and source fingerprints, and reject name/key collisions.
5. Retain safe partial progress only when six-word copied coverage decreases without a longer maximum match. Rescan the current text before the next edit, but compare scientific meaning with the immutable original. Recheck the assembled graph or summary and roll back suspect changes.
6. Bound attempts and flag unresolved outputs explicitly. In the tested Gemma stages temperatures cycle through 0.35/0.65/0.95, top-p is 0.95, seeds are 2026101037 + 1009 × attempt, and each stage permits up to 20 attempts. Use length-aware batching, 128 sequences / 8192 batched tokens, context 4096 and output cap 768; do not truncate inputs. These are measured settings, not established optimal production settings.
7. Validate the chosen pipeline on new held-out papers before scaling. Freeze prompts, thresholds, seeds and retry budgets before QA; keep all paper/question slots and compare raw, BT-only, direct Gemma-only, and combined repair using the same fixed answerer. Direct Gemma-only from raw outputs is a proposed ablation, not a result supplied by this experiment.

### Starting prompt for local edits

```text
Edit scientific text without losing facts, qualifiers, negations, numbers,
units, variables or mathematical expressions. Return only JSON:
{"edits":[{"old":"verbatim substring from TEXT","new":"equivalent wording"}]}.
Propose up to 16 small meaningful substitutions, each old substring at most
16 whitespace words. Interrupt every supplied source-matching run of six
or more words. Preserve the full subject and every list item. Avoid filler,
punctuation-only changes and artificial hyphenation.
PROBLEMATIC PHRASES: <overlap-audit output>
FIELD PATH: <typed field path>
TEXT: <saved factual text, with adequate local context>
```

This is a prompt starting point extracted from the tested local-edit protocol. A production implementation still needs the automated guards, graph mapping, complete-source audit and unresolved-output handling above. The independently measured reference code is linked from the [earlier repair bundle](https://github.com/LAION-AI/project-alexandria/tree/findings/ornith-dflash-97-20261003/experiments/scientific_summaries/ku_backtranslation_20_20261010/code).

## Advice for summaries

Keep the summary generator separate from the optional wording-repair stage. The measured Qwen-distilled rank128 Gemma12 FP8 summary deployment reaches **94.23% raw QA**, and **93.30% after one guarded TranslateGemma round**, on 97 papers / 970 MCQs. Mean >=6-word coverage falls from **15.23% to 9.69%**, but **all 97** still violate the strict rule. BF16 reaches 93.92% raw / 94.12% after translation and also has zero strict passes. Small stochastic QA changes do not establish a benefit or harm.

Back-translation alone is a cheap option for partial wording-overlap reduction. It is **not supported as sufficient for strict five-word compliance**, including for summaries. A direct Gemma editor may be a useful alternative or fallback, but its complete-summary throughput, QA and strict-compliance rate have not been measured. The published Gemma-only million-paper costs are explicitly optimistic scenarios borrowed from residual KU timings.

The Qwen rank128 summary measurements must not be assigned to Qwen rank64, Ornith-distilled adapters, or the E4B KU generator. Their existing model cards retain their own evaluations. See [results](RESULTS.md), [all scores](metrics.json), [protocol](protocol.json), and [compute assumptions](SCALING.md).
