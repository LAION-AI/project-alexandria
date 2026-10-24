# Follow-up: optional “that”, exact units and protected instrument IDs

Implemented and evaluated on **the same 97 saved summaries and 970 original MCQs**,
24 October 2026. Inputs, questions and gold answers are unchanged. All papers are
kept. Rules are frozen before QA; this remains a test-informed exploratory
extension of an already repeatedly used benchmark.

## Results

| Condition | Mean copied words ≥6 | Mean copied words ≥7 | QA correct | Major/unintelligible new windows |
|---|---:|---:|---:|---:|
| Previous guarded numeric pipeline | 5.4027% | 3.2910% | 912/970 | — |
| Added instrument/model identifier protection | 5.4052% | 3.2910% | 912/970 | 0/1 |
| ID protection + optional reporting-that | 5.4027% | 3.2910% | 912/970 | 0/2 |
| ID protection + exact unit aliases | 5.4022% | 3.2876% | 912/970 | 0/3 |
| ID protection + both grammar and units | 5.3997% | 3.2876% | 912/970 | 0/4 |

**The overlap gain is tiny:** 5.4027% → **5.3997%**,
only 0.0029 percentage points. This **does not reach 5%**.
All five arms score **912/970 (94.02%)**, with no invalid responses.
0/97 summaries pass the complete ≤5-word rule.
Copying is measured against each paper’s full source, in narrative fields; evidence
and bibliography remain separate. It is a copied-word fraction, not a fraction of
papers. Full [primary and punctuation-split audits](results/grammar_units/COPY_OVERLAP.md)
are retained.

## The three changes

1. **Instrument/model IDs:** recompute the numeric stage from the protected-article
   summaries. Keep numbers after uppercase acronyms, internal-capital model names
   and model-like alphanumeric labels. This keeps **LAXPC 20** as written. General
   lexical patterns are used; no list of test-paper IDs is hardcoded. Ordinary
   quantities such as “20 participants” remain eligible. Protecting a name can
   restore a literal match; correctness takes priority over reducing a string score.
2. **Optional reporting-clause “that”:** a finite reporting-verb list, followed by
   an explicit subject and a detected finite embedded clause. Keep demonstrative,
   relative, conditional and uncertain grammatical cases. Keep negation/modals.
   Only **one** edit passed all guards and reduced copying: “authors demonstrate
   that MIR196A2 is” → “authors demonstrate MIR196A2 is”.
3. **Exact unit aliases:** case-sensitive same-scale symbols/full names only next
   to supported quantities. **No unit conversion**. Preserve sign, precision,
   numbers, formulas and compound units; do not edit an isolated half of mg/mL.
   Only **two** edits were accepted: “five hundred ng” → “five hundred nanograms”
   and “three hundred mg” → “three hundred milligrams”.

Grammar/unit arms start after the new identifier-safe numeric stage. The old
numeric baseline is preserved, rather than inversely rewriting already formatted
IDs. This makes every numeric transformation reproducible from its input.
Each new grammar/unit edit must reduce both absolute copied words and six-gram
hits, as well as the fraction. Numbers/formulas/negation/modals and exact splice
provenance are checked; evidence and metadata are unchanged. There are no
whole-document rollbacks in these arms.

## Absolute copying, safety and QA

The previous numeric output has **10,528** covered narrative words out of
191,878. The ID fix restores six covered words: **10,534**. Grammar + units then
remove thirteen: **10,521** out of **191,877**,
for a net seven fewer covered words than the old baseline. With the old per-paper
word counts fixed as denominators the combined result is
**5.3997%**. This is not denominator padding.

The four distinct actually changed sentence pairs are automatically labelled
**equivalent** by the pinned 27B reviewer. Per-arm counts overlap because the ID
fix is shared. These labels are not a human safety guarantee; the reviewer’s
explanation for the “that” edit incorrectly says an article was removed even
though the equivalent label is plausible. Existing edits upstream are not
recertified by this small delta review.

QA uses the unchanged Qwen2.5-7B student, historical prompts/parser, paired
question seeds and every original slot. Only **40** new student calls were
needed; exact-context answers were reused. Paired whole-paper bootstrap score
differences are zero on this cohort. This does not establish equal quality on
new papers.

### Review bookkeeping correction, fully retained

The initial reviewer input incorrectly compared fifteen unchanged citation-label
and citation-justification fragments because their displayed paths are duplicated.
This created fifteen spurious major-change labels per arm, unrelated to visible
edits. The corrected producer aligns fragment **occurrence order** and asserts
path/category equality. Every review card must belong to a real changed fragment.

All initial wrong pairs, raw judgments, result aggregates, original code and
protocol are retained in the evidence bundle under
`grammar_units/outputs/diagnostics/review_alignment_initial/`.
The corrected review contains ten cards across the four arms, representing four
unique actual edits. All four already had exact cached judgments, so correction
required **zero additional GPU calls**. A saved SHA-256 proof confirms every
summary file, raw QA answer file and complete copy-audit file stayed byte-identical.
See [correction proof](results/grammar_units/review_alignment_correction.json).
The original pre-QA code hashes and the corrected review-metadata hashes are both
saved; no summary was regenerated or selected using QA.

## Cost for 38 million summaries

| Work | Warm CPU core-hours /38M | Additional deployed GPU-hours |
|---|---:|---:|
| Protected article stage, upstream | 1,038 | 0 |
| New numeric + ID + grammar + unit stage | 1,624 | 0 |
| These two CPU stages together | **2,662** | **0** |

These are short warm CPU measurements with indexing, candidate guards and primary
audits. They exclude bulk I/O and worker startup, assumed amortized over a large
run. CPU work must overlap GPU generation/repair for zero added GPU reservation
time. The small timing difference between individual arms is not a speed ranking.
The former generation + one Gemma repair planning estimate remains **73,699 GH200
GPU-hours /38M at 85% useful capacity**; these three CPU improvements do not change
that estimate.

The offline test job **2489400** reserved a complete **4-GH200 node for 305 seconds**:
**0.3389 GPU-hours**, including startup/idle/CPU work. It used two model lanes.
Initial reviewer calls include the fifteen mispaired comparisons; their cost is
not hidden. Offline test allocations are separate from the deployed CPU-only cost.

## Reproduction

```bash
python replay.py --out /scratch/alexandria-deletion-evidence
python -m pip install -r requirements.txt
python smoke_grammar_units.py \
  --nltk-data /scratch/alexandria-deletion-evidence/common/nltk_data

# Input is the saved summary AFTER the protected-article stage.
python repair_grammar_units.py --summary protected-articles.json \
  --source full-paper.txt --mode combined \
  --nltk-data /scratch/alexandria-deletion-evidence/common/nltk_data \
  --out identifier-grammar-units.json
```

Modes: `identifier_guard`, `optional_that`, `unit_aliases`, `combined`.
The code reruns the numeric stage with the ID guard before either extra rule.
Twenty-four small guard/provenance checks pass. Historical executables, frozen
inputs, all intermediate traces, reviewer judgments and original QA responses
are included. The CPU CLI is checked against saved outputs with a real edited
paper. [All results](results/grammar_units/results_and_scaling.json) and the
[before/after reader](examples.html) expose the remaining overlaps and examples.

## Conclusion

**Keep the identifier fix.** It prevents a real instrument-label formatting error.
The optional “that” and exact unit rules are cheap and narrowly applicable, but
contribute almost nothing to the remaining overlap on this cohort. Do not present
this as reaching the 5% goal or eliminating literal copying. More useful next CPU
rules need phrase-level development on training data and a fresh held-out check.
