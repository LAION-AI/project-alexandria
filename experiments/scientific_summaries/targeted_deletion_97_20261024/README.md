# Cheap CPU edits for scientific summaries: results, failures and reproduction

Completed 24 October 2026. This is an additional experiment on **97 fixed papers
and 970 original MCQs**. It starts from saved summaries after one Gemma E2B repair
and the [frozen contextual CPU phrase bank](../contextual_procedural_repair_97_20261024/README.md).
No paper is regenerated, dropped or selected by QA score.

## Read the new views

- [Results and methods dashboard](index.html): a standalone English HTML report.
- [Before/after reader](examples.html): five complete papers across every treatment,
  plus labelled examples of successful edits and failures.
- [Machine-readable comparison](results/combined_comparison.json).
- [Full source-copy audit for the guarded number stage](results/numbers_guarded/COPY_OVERLAP.md).

Download/open the HTML files locally, or serve this directory with
`python -m http.server 8000`. They require no external JavaScript, APIs or fonts.
GitHub's normal HTML file view shows the source; it is not a separate live deployment.

## What the results say

| Treatment | Mean copied words ≥6 | Mean copied words ≥7 | QA correct | Major / unintelligible new windows | Unparsed judgments |
|---|---:|---:|---:|---:|---:|
| Before additional CPU edits | 9.04% | 5.95% | 915/970 | — | 0 |
| Filler phrases | 9.04% | 5.95% | 915/970 | — | 0 |
| Articles: original implementation | 5.58% | 3.43% | 913/970 | 9/1076 | 0 |
| Articles + eight selected adjectives | 5.56% | 3.42% | 914/970 | 11/1077 | 0 |
| Articles + broad adjectives/adverbs | 3.51% | 2.11% | 913/970 | 173/1514 | 0 |
| Articles with identifier/case protection | 5.94% | 3.67% | 913/970 | 0/996 | 0 |
| Adjectives/adverbs only | 5.70% | 3.61% | 909/970 | 243/1052 | 1 |
| Protected articles + percent notation | 5.78% | 3.55% | 911/970 | 0/63 | 0 |
| Numeric notation: original prototype | 4.65% | 2.70% | 914/970 | 4/392 | 2 |
| Protected articles + quantity-only numeric notation | 5.40% | 3.29% | 912/970 | 1/178 | 1 |

Protected article deletion reduces copying without a model call. Adding controlled
numeric notation gives **5.40%**
mean copied-word coverage and **912/970 QA**. It does not eliminate
all >5-word matches: **0/97**
summaries pass the strict whole-summary rule. Exactly six words remain a separately
reported borderline case; ≥7 is shown in its own column.

The separate **adjective/adverb-only** arm scores **909/970**, compared with
**915/970** before additional CPU edits. It has **243 major/unintelligible automatic
judgments among 1,052 changed windows**, plus 765 minor meaning changes and one
unparsed judgment. The earlier combined article+modifier arm scores **913/970**;
these are different treatments. Do not claim that deleting modifiers is safe
because the student still answers most questions correctly. The paired 95%
paper-bootstrap QA difference for modifiers only is
**-1.55 to +0.21 percentage points**,
so the six-question difference alone is not a firm population-level estimate.

### The metric is a word fraction, not a fraction of papers

For each summary, count narrative word positions covered by a consecutive match of
at least six normalized whitespace words against its own **complete** source.
Overlapping windows count a position once. Divide by all narrative words in that
summary, then average the 97 fractions. Explicit evidence quotes and bibliography
are audited separately. Case, compatibility Unicode and punctuation are normalized.
The punctuation-split diagnostic is separate; it does not replace the primary rule.

An isolated six-word match passes after any one-word deletion. The saved sanity
check confirms **6/6** possible deletions. No gapped matching is used. Other matches
in the same summary may still remain. A ten-word copied passage contributes ten
covered words, not just words six through ten. See complete CSV/JSONL evidence in
the bundles; all original QA slots are retained regardless of copying flags.

## What actually runs on the CPU

1. Index the complete source and locate six-word matches in narrative fields.
2. Propose a small edit only where it intersects an existing match.
3. Keep quotes, metadata, negation, modals and protected scientific information.
   Require both the absolute number of covered words and their fraction to fall.
4. Recheck the whole summary; roll back a document if its protections or longest
   source match fail. Keep every document in the evaluation.

The article guard preserves mid-sentence capitalized identifiers and requires a
following noun/adjective. This fixes the observed **An** molecule abbreviation
being mistagged as the article *an*. The original implementation and its nine
severe/unintelligible judgments from that paper remain in the evidence.

Filler-phrase deletion adds no edits in this experiment. The eight selected
evaluative adjectives add only two deleted words beyond the original article arm.
The broad JJ/RB arms are diagnostics, not recommended production settings. POS
tagging can be wrong, and scientifically essential descriptors can be adjectives.

## Numbers and percent notation

The first numeric prototype reaches **4.65%**
and **914/970 QA**, but it changes some technical labels and mathematical spellings.
Examples include DNA 3-prime/5-prime labels, a crystallographic index, and compound
identifiers. Decimal equality alone does not establish that a numeral is a quantity.

The guarded revision adds positive quantity-context checks and conservative
sentence-level formula protection. It preserves primed values, leading-zero
integers, slash/caret-adjacent notation, #IDs, explicit references, and known index
nouns. Only eligible prose quantities are changed. These are finite heuristics,
not complete mathematical parsing or named-entity recognition. A remaining
minor automatic flag concerns the instrument label LAXPC 20 becoming LAXPC
twenty; other flags concern numeric style or reviewer misreadings. All labels
remain visible.

- `25%` can become `twenty-five percent`.
- `five percent` can become `5%`.
- `25 participants` can become `twenty-five participants`.
- An identified formula, crystallographic index, reference ID or primed biological
  label stays unchanged in the guarded arm.

Each accepted number must round-trip with **Decimal value, sign and decimal-place
precision** unchanged. Supported cardinals are below 10^12, at most six decimal
places and at most twelve output words. No ordinals or scientific-notation
conversion. `%`, `percent` and `per cent` preserve the same unit. Exact edit
provenance proves nonnumeric text outside each accepted span is unchanged.

The previous digit-only signature cannot be applied literally to digit→word edits;
the new stage uses the explicit equivalence proof above plus formula/operator and
metadata protections. This distinction is recorded in the saved numeric audits.

**No denominator trick:** the guarded input has 191,702 narrative
words and 11,629 covered words; the output has
191,878 words and 10,528 covered words. With the
original per-paper denominators fixed, mean coverage is
**5.41%**, versus the reported
**5.40%**.

Changing numeric spelling reduces literal string overlap. It is **not independent
semantic paraphrasing**, and a lower overlap score alone does not certify quality.

## QA, meaning review and known limitations

Student: pinned **Qwen/Qwen2.5-7B-Instruct**. Ten original MCQs per paper, historical
prompts/parser, temperature 0.5, top-p 0.95, presence/frequency penalty 1.05, up to
100 output tokens and five format attempts. Question-specific paired seeds are
unchanged; identical evaluated contexts reuse the same answer. Model revisions
are in [model_pins.json](model_pins.json). No gold answer selects a rewrite.

Meaning reviewer: pinned **Qwen/Qwen3.8-27B-FP8**, greedy, thinking disabled. Every
new final sentence/window is reviewed; identical original/edited pairs reuse the
same raw judgment. NLTK 3.9.1 supplies POS and sentence boundaries on the CPU.

These are automatic labels, not human error rates. Controls undercalled removing
“randomized” and “may” as minor instead of major. One numeric judgment incorrectly
claimed **90→ninety** was 90→9; the exact numeric roundtrip proves value equality.
All raw labels, false positives and unparsed cases stay visible. Existing errors
in the preceding Gemma repair are not certified fixed by this additional audit.

The case/grammar and guarded-number revisions respond to observed meaning-audit
failures, not QA gold. This is a repeatedly used, **test-informed exploratory**
benchmark. Use the 865 training papers for future rule development and a fresh
held-out set before claiming generalization. Preserve the original 97 results.

## Compute and practical next steps

The new deployed stages use **0 additional GPU-hours**, assuming their CPU work
overlaps the existing GPU generation/repair. Protected articles plus
the guarded numeric stage extrapolate to roughly **1,682 CPU core-hours for
38M papers**. This is warm compute including source indexing, candidate search,
POS/guards and primary audits; worker startup is amortized and bulk I/O is outside
the measurement. Reuse indexes and batch POS work in production. Small differences
between warmed variant timings are not a speed ranking.

Generation plus the existing one-Gemma repair retains the preceding planning
estimate of **73,699 GH200 GPUh /38M**, with 85% useful capacity. CPU-only here
describes the **additional** stage. Offline QA/reviewer allocations are reported
separately in the saved allocation accounting, including whole-node reservations.

Promising **unmeasured next CPU experiments**:

1. Remove grammatically optional *that* only in specific reporting-verb frames;
   preserve the reporting verb and an explicit clause subject. Test narrow *of*
   removal in phrases such as “all of the samples”→“all the samples”.
2. Use a tiny exact unit map, such as *mg*↔*milligrams*, only next to quantities
   outside formulas. Do not apply unrestricted technical synonyms.
3. Protect paper-declared abbreviations and use domain-specific rule exclusions.
   Derive and validate new phrase rules on the training cohort before freezing.
4. Rank the unresolved sentences by copied-word reduction per safe edit. Cache
   source indexes; call a small model only for a limited high-yield remainder if
   CPU rules are insufficient. No GPU saving is claimed until measured.

## Reproduce saved results without GPUs

```bash
python replay.py --out /scratch/alexandria-deletion-evidence
```

Hash-checked bundles contain the original 97 full sources, testset, all repaired
summaries, edit traces, every QA response and raw judgment, complete primary and
punctuation-split overlap evidence, protocols and timing files. **13,580 raw QA
slots** across five saved experiments include repeated baselines; the combined
table has **10 distinct named conditions**. Replay verifies every source hash,
gold answer, evaluated context hash, QA count and mean covered-word fraction.
The small saved NLTK English tagger/Punkt assets are included with file hashes.

## Apply the measured CPU code to a new saved summary

```bash
python -m pip install -r requirements.txt

python repair_saved.py --summary gemma-summary.json --source full-paper.txt \
  --mode articles_guarded \
  --nltk-data /scratch/alexandria-deletion-evidence/common/nltk_data \
  --out articles.json

python repair_saved.py --summary articles.json --source full-paper.txt \
  --mode numbers_percent_guarded \
  --nltk-data /scratch/alexandria-deletion-evidence/common/nltk_data \
  --out articles-and-numbers.json
```

Other modes reproduce diagnostic treatments: `fillers`, `articles_legacy`,
`articles_soft_modifiers`, `articles_modifiers`, `modifiers_only`, `percent_only`,
and the initial `numbers_percent` prototype. **Do not treat these diagnostics as
equally safe alternatives.** Every output includes copying flags and rollback status.

Rebuild the readable pages from materialized evidence:

```bash
python build_views.py --evidence /scratch/alexandria-deletion-evidence
```

`historical_code/` preserves actual JUPITER scripts and launchers for every stage.
For fresh GPU QA/review, adapt their root/interpreter/model paths, install this
repository's package, restore the frozen inputs, and use the pinned models and
the preceding benchmark environment (torch 2.13.0+cu130, transformers 5.18.0,
vLLM 0.30.0). The portable CPU command requires only NLTK and the included assets.

Loose CSV sidecars use LF line endings for repository readability. Bundles retain
the original CSV bytes and are verified against the raw evidence manifest.
