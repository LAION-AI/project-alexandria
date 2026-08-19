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

98 questions over 25 scenes, four context arms, five samples each, options reshuffled per
sample. Student: Qwen3-4B-Instruct-2507.

| Arm | All questions | Non-leaky (n=63) |
|---|---|---|
| `none` | 0.363 [0.29, 0.44] | — *(see below)* |
| `full_text` | 0.825 [0.76, 0.89] | **0.756** [0.65, 0.85] |
| `ku_chain` | 0.727 [0.65, 0.81] | **0.641** [0.53, 0.75] |
| `ku_scene` | 0.831 [0.76, 0.89] | **0.848** [0.77, 0.92] |

Paired bootstrap over questions, non-leaky stratum:

| Comparison | Difference | 95% CI | p | Verdict |
|---|---|---|---|---|
| `ku_chain` − `none` | **+0.565** | [+0.46, +0.67] | <0.001 | **significant** |
| `ku_scene` − `ku_chain` | **+0.206** | [+0.13, +0.30] | <0.001 | **significant** |
| `ku_chain` − `full_text` | **−0.114** | [−0.21, −0.02] | 0.020 | **significant** |
| `ku_scene` − `full_text` | +0.092 | [−0.03, +0.21] | 0.132 | not distinguishable |

### What this says

**The knowledge is in the units.** `ku_chain` beats `none` by 56 points, and `ku_scene` —
the questioned scene's units plus two either side, no source text at all — is
indistinguishable from handing the student the entire screenplay. Whatever the MCQ measures,
the units carry it.

**But the flat chain dilutes it.** The full 52k-token chain scores *significantly below* the
full screenplay, while the scene-local view scores 21 points above the same chain. The
content is present in both cases; only the amount of surrounding material differs. This is a
**retrieval problem, not a representation problem**, and it is precisely the distinction the
`ku_scene` diagnostic arm existed to draw.

The practical consequence: a KU chain should be served to a small model through retrieval,
not pasted whole. Reporting only `ku_chain` would have understated the artifact; reporting
only `ku_scene` would have overstated how it behaves when dumped in bulk.

### The instrument was rebuilt once, and it changed the answer

The first instrument contained 7 questions quoting 8+ consecutive words of source dialogue
in their answer options. The generation prompt forbids quoting; it did not enforce it. Those
questions were regenerated under a gate, and 2 that still quoted after three rounds were
dropped rather than shipped.

On the *leaky* instrument, `ku_chain` and `full_text` were indistinguishable (−0.012,
p=0.83). On the clean one, `ku_chain` is significantly worse (−0.114, p=0.020). **The quoted
questions were partly answerable by surface overlap and flattered the chain arm.** The
earlier, friendlier number is the one that was wrong.

### Two caveats about the numbers above

**The `none` arm's non-leaky score is circular and is not reported.** The non-leaky stratum
is *defined* as the questions the `none` arm answered below 0.6, so its score on that stratum
(0.076) is a selection artifact, not a floor. The correct floor reference is chance, 0.25.

**35 of 98 questions are leaky** — answered without any context at ≥0.6. *The Matrix* is one
of the most widely described films in existence and the student has read about it. This is
why the headline numbers are the non-leaky ones, and why the calibration arm ran first rather
than as a formality.

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
- **3 numbers are genuinely missing** from the units — an `M-16`, a `.45`, and the `4` in
  `GUARD #4`. Reported rather than tuned away.

## What the checks caught, and what caught the checks

The gate blocked the artifact on three separate runs, and was right each time. But of the
defects found across this work, **the checks were themselves wrong in four cases**:

| | The check believed | Actually |
|---|---|---|
| C2 | masking required-verbatim tokens was safe | exempting `the` from a date let a whole copied line pass the gate |
| C3 | 20 locations were missing | they were in `heading.location`, which the check never searched |
| C3 | 11 numbers were missing | it had split dates on hyphens and demanded the fragments |
| C3 | 17 numbers were missing | they were the script's own scene numbers and revision stamps |
| C3 | 2 numbers had drifted | single digits differ from each other in one position by definition |

Every one produced a confident number computed over the wrong thing. Four were found by
their own negative case or by reading the violations, none by the check passing quietly —
which is the only reason they are in this table rather than in the artifact.
