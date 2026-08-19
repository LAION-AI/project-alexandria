# Results — *The Matrix*, 225 scenes

One run, one film, one extraction model, one student. Everything below is from
`results/matrix/`, reproducible from a checkout plus lawful access to the source.

---

## The artifact

| | |
|---|---|
| Scenes | **225** (224 mapped + `sc-000`, the pre-slugline opening) |
| Knowledge Units | 225, one per scene, exact tiling |
| Beats | **1,021** — the temporal spine |
| State changes | 323 |
| Canonical entities | 109 |
| Wall clock | **424 s** (~7 min) on 8×A100 |
| Stage 1 | 30 windows, 0 failures |
| Stage 2 | 29 seams, 0 failures — 3 alias merges, 25 context fixes |

Prefix caching did what it was measured to do: the shared 37k-token screenplay prefix
warmed to **0.8 s** on all eight endpoints, so every agent saw the whole film for
effectively nothing.

## Gate results

| Check | Status | Evidence |
|---|---|---|
| C1 coverage | **pass** | 225/225, no duplicates, no unknown ids |
| C2 verbatim overlap | **warn** | longest adjudicated run **7 words**, 0 at the 8-word fail bar |
| C3 fact fidelity | **warn** | **0 drifted numbers**; names 0.99, locations 1.00, numbers 0.82 |
| C4 temporal totality | **pass** | strict total order over 1,021 beats, chain unbroken, no cyclic causes |
| C5 referential integrity | **pass** | 0 dangling; 32% of entity records auto-declared and reported |
| C6 non-genericity | **pass** | median pairwise similarity **0.13** against a 0.45 bar |
| C7 canary | **pass** | emitted **0 units** from an empty target span |

All seven negative cases ran in the same execution; without that, `run_all` reports
`unverified` rather than `pass`.

### Independent leak audit

Re-implemented from scratch rather than reusing `checks.py`, so a bug in the gate could not
also hide the leak:

> **14,989 fields scanned. Longest verbatim run from the source: 7 words.** 41 fields carry
> runs of 6–7; none carries 8 or more.

The overlap repair earned this. Round by round the longest run went **18 → 8 → 7 → 7**
words; a single pass would have left the artifact blocked.

---

## Evaluation

100 questions over 25 scenes, five context arms, five samples each, options reshuffled per
sample. Student: Qwen3-4B-Instruct-2507. Non-leaky stratum, n=62.

| Arm | Context given | All | Non-leaky |
|---|---|---|---|
| `none` | nothing | 0.398 | *floor is chance, 0.25* |
| `full_text` | whole screenplay (37k tok) | 0.832 | 0.787 [0.69, 0.88] |
| `ku_chain` | whole KU chain (52k tok) | 0.766 | 0.729 [0.63, 0.82] |
| **`text_scene`** | **source text, questioned scene ±2** | **0.960** | **0.948** [0.91, 0.98] |
| **`ku_scene`** | **KUs, questioned scene ±2** | **0.882** | **0.871** [0.79, 0.95] |

`text_scene` is the control that makes the comparison fair, and it was added after a first
pass had already been written up. Without it, `ku_scene` was being compared against the whole
screenplay, which confounds two things at once — **representation** (units vs prose) and
**retrieval** (five scenes vs two hundred). Giving the source text the same retrieval
advantage separates them.

### The two effects, separated

**Retrieval is the larger effect, and it is representation-independent.**

| Comparison | Difference | 95% CI | p |
|---|---|---|---|
| `text_scene` − `full_text` | **+0.161** | [+0.08, +0.25] | <0.001 |
| `ku_scene` − `ku_chain` | **+0.142** | [+0.06, +0.23] | <0.001 |

Narrowing the context to the relevant five scenes is worth 14–16 points, and it is worth
about the same whether the context is source prose or Knowledge Units. Both whole-document
arms are diluted by the same mechanism.

**At matched retrieval, the units cost something.**

| Comparison | Difference | 95% CI | p |
|---|---|---|---|
| `ku_scene` − `text_scene` | **−0.077** | [−0.168, +0.003] | **0.059** |

This is the number that matters, and it is the honest cost of the transformation: converting
a scene to a Knowledge Unit loses roughly 8 points of answerable detail relative to reading
the scene itself. It sits just outside conventional significance at n=62, with the CI barely
touching zero — so the fair statement is *a consistent penalty of around 8 points that this
sample cannot quite separate from zero*, not "no difference".

**The units still carry most of it.** `ku_chain` − `none` is +0.629 (p<0.001), and `ku_scene`
at 0.871 sits far above the 0.25 chance floor. Whole-document, `ku_chain` − `full_text` is
−0.058 (p=0.145), not distinguishable.

### What changed when the control was added

The earlier write-up of this run led with "scene-local KUs are indistinguishable from handing
over the whole screenplay" (`ku_scene` − `full_text` = +0.084, p=0.122, still true above).
That is a true sentence and a misleading headline: it credits the units for a gain that came
from retrieval. With `text_scene` in place, the same data says something more useful and less
flattering — **retrieval is doing most of the work, and the units are slightly behind the
prose they replace.**

### Two caveats about the numbers above

**The `none` arm's non-leaky score is circular and is not reported.** The non-leaky stratum is
*defined* as the questions the `none` arm answered below 0.6, so its score on that stratum is
a selection artifact. The correct floor reference is chance, 0.25.

**38 of 100 questions are leaky** — answered without any context at ≥0.6. *The Matrix* is one
of the most widely described films in existence and the student has read about it. This is why
the headline numbers are the non-leaky ones, and why the calibration arm runs first.

---

## Closing the gap: KU generation variants

The 8-point matched-retrieval deficit is a defect with a cause, so the first step was to find
it rather than guess. Of the 9 non-leaky questions where `text_scene` beats `ku_scene` by 40
points or more, **5 are `state_change`** questions, and the failures share a shape:

| Scene | Words | What the source has | What the baseline unit recorded |
|---|---|---|---|
| `sc-150` | 35 | Tank speaks to Neo over the phone | 2 beats, **Tank absent entirely** — only Neo |
| `sc-125` | 31 | cover cracks → eyes appear → truck passes → cover opens → Trinity climbs out | 2 beats, physical detail merged away |

Both are ~30-word scenes, and the median scene in this screenplay is 45 words. The failure is
**over-compression of short scenes**: merged beats, dropped minor speakers, summarised physical
state. A short scene is not a simple scene, and the baseline prompt treated it as one.

### Four variants, run in parallel

| Variant | Change |
|---|---|
| `narrow` | 3 scenes per window instead of 12 |
| `detail` | prompt demanding one beat per distinct action, every speaker recorded, physical state captured |
| `both` | narrow + detail |
| `both_spk` | both, plus the scene map's own speaker list injected as a per-scene checklist |

### Results, with the control that matters

| Variant | Beats | **Longest overlap** | `ku_scene` | vs baseline | vs `text_scene` ceiling |
|---|---|---|---|---|---|
| baseline | 1,021 | 7 | 0.871 | — | −0.077 (p=**0.059**) |
| `narrow` | 1,126 | 6 | 0.903 | +0.032 (p=0.40) | −0.045 (p=0.15) |
| `detail` | 1,359 | 7 | 0.848 | −0.023 (p=0.60) | −0.100 (p=**0.014**) |
| **`both`** | 1,453 | 7 | **0.926** | +0.055 (p=0.15) | **−0.023 (p=0.47)** |
| **`both_spk`** | 1,472 | 7 | **0.926** | +0.055 (p=0.11) | **−0.023 (p=0.54)** |

**The overlap column is the point of the table.** A variant can always raise its score by
copying more of the source, which would destroy the artifact while appearing to improve it.
Every variant holds at 6–7 words with **zero fields at or above the 8-word bar** — identical to
baseline. The gains are not bought with source text.

**The gap is no longer detectable.** The baseline sat 7.7 points below reading the source
itself, marginally (p=0.059). `both` and `both_spk` sit 2.3 points below, nowhere near
significance (p≈0.5). At matched retrieval, units built this way are indistinguishable from the
screenplay for answering questions about it.

**Neither ingredient works alone.** `narrow` alone is +0.032 and not significant; `detail` alone
is *worse* than baseline and is the only variant significantly below the ceiling (p=0.014).
Granularity instructions given to a model reading 12 scenes at once produce longer records
without better ones — the attention has to be there first for the instruction to land. The
speaker checklist added nothing beyond `both` (identical 0.926), so the prompt was already
sufficient once the window was narrow.

### Mechanistic confirmation

The two diagnosed failures, under `both`:

- **`sc-150`** — 2 beats → 3. **Tank now appears** as the actor of his own speech beat and in
  `present`. The question asks whose voice Neo hears; it is now answerable.
- **`sc-125`** — 2 beats → 5, recovering cover cracks → eyes peek → truck passes → cover opens →
  Trinity climbs out. The question asks the cover's state; it is now recorded.

The fix reached the specific facts the diagnosis predicted, which is stronger evidence than the
aggregate movement alone.

### What it costs, and what it does not fix

| | baseline | `both` |
|---|---|---|
| Extraction wall clock | 424 s | ~2,070 s (**5×**) |
| Rendered chain | 52k tokens | 73k tokens (**+40%**) |
| `ku_chain` (whole-document) | 0.729 | 0.719 — **unchanged** (p=0.85) |

The improvement is **entirely in the retrieved condition**. Dumping the richer chain whole is if
anything slightly worse, because the extra detail also makes the haystack larger. This sharpens
rather than complicates the earlier finding: detail helps when it is retrieved and hurts when it
is dumped, so the granularity and the serving strategy have to be chosen together.

### The caveat this section needs

**Four variants were tried and the best was selected on the same 100 questions used to measure
it.** Uncorrected, `both` improves on baseline at p=0.15; with a Bonferroni correction for four
comparisons the bar is p<0.0125 and it does not clear it. The defensible claims are therefore
narrower than the table suggests:

1. The best variants are **not distinguishable from the source-text ceiling** (p≈0.5), where the
   baseline was marginally below it. This does not depend on selection — it is a comparison
   against a fixed reference.
2. The gains are **not from copying**, which the overlap column establishes directly.
3. The mechanism is confirmed on the specific diagnosed cases.

What is *not* established is that `both` beats the baseline by a specific margin. That needs a
confirmatory run on a freshly generated instrument, which has not been done.

---

## Honest limits

- **One film, one student, one extraction model.** Nothing here generalizes to other
  screenplays and should not be written up as if it does.
- **Question generator and KU extractor are the same model** (Qwen3.8-27B). They see
  different inputs and questions come from source text, so this is not circular — but a
  shared blind spot would be invisible to this design. A second generator would settle it.
- **The intended student would not load.** Gemma-4-E4B and E2B both fail on this
  vLLM/transformers build with `AmbiguousGlobalPerLayerAttributeError` — they are MatFormer
  models with per-layer configs. Qwen3-4B-Instruct-2507 was substituted as the nearest
  working 4B-class instruct model. It shares a family with the extractor, which is a real
  confound for the *absolute* scores; the arm *comparisons* are unaffected, since every arm
  uses the identical student and differs only in context.
- **The chain is larger than the source**: 52k tokens against 37k. The units add
  contextualization and structure per scene. No compression claim is available here.
- **MCQ measures recognition, not reconstruction.** That the temporal order is recoverable is
  established mechanically by C4, not by this instrument.
- **Variant selection is uncorrected.** The best of four variants was chosen on the same
  instrument used to score it; see the caveat above. A confirmatory run on fresh questions is
  the obvious next step and has not been run.
- **3 numbers are genuinely missing** from the units — an `M-16`, a `.45`, and the `4` in
  `GUARD #4`. Reported rather than tuned away.

## What the checks caught, and what caught the checks

The gate blocked the artifact on three separate runs, and was right each time. But of the
defects found across this work, **the checks were themselves wrong in six cases**:

| | The check believed | Actually |
|---|---|---|
| C2 | masking required-verbatim tokens was safe | exempting `the` from a date let a whole copied line pass the gate |
| C2 | its tokenizer was sound | `[a-z0-9']+` swallowed quote marks, so **any span in quotation marks was invisible** — quoted dialogue, the likeliest leak of all |
| C3 | 20 locations were missing | they were in `heading.location`, which the check never searched |
| C3 | 11 numbers were missing | it had split dates on hyphens and demanded the fragments |
| C3 | 17 numbers were missing | they were the script's own scene numbers and revision stamps |
| C3 | 2 numbers had drifted | single digits differ from each other in one position by definition |

Every one produced a confident number computed over the wrong thing, and none announced
itself: each was found by its own negative case, by reading violations, or — in the
tokenizer's case — by noticing a quotation in a worked example that the gate had passed. That
last one is worth stating plainly, because it is the closest this project came to shipping the
failure it was built to prevent: **the leak detector could not see quotation marks, which is
the exact form dialogue takes when it escapes.** The artifact survived the re-audit unchanged
at 7 words; the instrument did not, and was rebuilt.
