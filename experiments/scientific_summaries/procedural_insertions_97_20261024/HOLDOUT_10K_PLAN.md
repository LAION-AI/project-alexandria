# Proposed independent 10,000-summary validation

**Status: proposed, not run.** No 10k results or runtime claims are being reported.
The existing 97 papers remain a regression set after repeated rule development.

## Freeze and sample before inspecting outcomes

1. Freeze the v6 code, rule catalog, tokenizer, overlap metric and model pins.
2. Select 10,000 previously unused papers with both a complete source and a saved
   structured summary. Exclude all development papers and exact/near duplicates.
   Use a recorded sampling seed, broad domain coverage and length strata.
3. Match the intended deployment mixture. Separate generator/upstream-stage strata
   if summaries come from different pipelines; do not apply the 4.8566% reference
   to an incompatible raw-summary baseline.
4. Tag formulas, numeric comparisons, clinical negation, acronym ambiguity and
   other challenge features before repair. Report both representative results and
   challenge slices with their actual denominators.

## What to check

- Run full-source primary and punctuation-split audits on **all 10,000** summaries,
  before and after repair. Keep every slot and every failure.
- Report fixed-original-denominator and pooled coverage, per-paper distributions,
  long-run lengths, strict-rule passes and changed-word counts. Bootstrap by paper.
- Check every edit's provenance, technical-term preservation, quantity/formula
  signatures, negation and modal/epistemic words. Retain whole-document rollbacks.
- Batch the pinned meaning reviewer over all unique changed sentence/window pairs.
  Inspect a random human sample plus a separately reported challenge sample; do
  not interpret automatic labels as human semantic error rates.
- Keep the original 970-question QA regression. Run paired QA on new papers that
  already have valid independent MCQs. A 10k summary audit does not automatically
  imply a 100k-question evaluation; generating/validating new MCQs is separate work.
- Preserve raw evidence and domain/length/case breakdowns. Reject catastrophic
  meaning changes even if QA remains unchanged, as the failed v5 example shows.

## Cost and decision

Use already generated summaries where suitable, avoiding generation expense.
The measured **additional v6 CPU stage alone** projects to roughly **13.2 CPU
core-minutes for 10k**, before bulk I/O and startup. Upstream stages, extra audits,
meaning-review inference, QA and human checks are additional work. Measure those
in batches before extrapolating total resources.

If the new cohort is inspected and rules are changed in response, it becomes a
development set. Retain its results and use a fresh cohort for the next independent
claim. Do not declare success solely because the aggregate mean is below 5%.
