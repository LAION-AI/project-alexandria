# Procedural insertion repair: complete 97-paper study

The final **v6** system reaches **4.8566% mean narrative copied-word coverage
in contiguous source matches of at least six words**, with **913/970 QA**.
With the original per-paper denominators fixed, coverage is **4.8682%**.
This is an additional CPU stage after the existing Gemma repair and earlier
procedural stages. It does not regenerate papers or train another model.

- [English results dashboard](index.html), including every meaning flag and controls.
- [All comparison rows](results/comparison.json).
- [Final full scores and timings](results/v6/results_and_scaling.json).
- [Final full-source overlap report](results/v6/COPY_OVERLAP.md).
- [Actual failure regressions and corrections](results/failure_regressions.json).
- [Independent 10k validation plan](HOLDOUT_10K_PLAN.md), **proposed, not run**.

The HTML is standalone: download/open it or serve this directory with
`python -m http.server 8000`. GitHub's normal HTML file view displays source.

## Results and scope

| Frozen condition | Mean covered words ≥6 | Original words / original denominator | QA | Meaning labels on changed windows |
|---|---:|---:|---:|---|
| Before insertion system | 5.3997% | 5.3997% | 912/970 | — |
| v3 joint | 5.0540% | 5.0618% | 913/970 | 120 equivalent, 3 minor |
| v4 joint | 4.9145% | 4.9246% | 913/970 | 159 equivalent, 6 minor |
| v5 joint, **rejected** | 4.8475% | 4.8593% | 913/970 | 181 equivalent, 6 minor, **1 unintelligible** |
| v6 greedy | 4.8616% | 4.8730% | 913/970 | 180 equivalent, 4 minor |
| **v6 joint** | **4.8566%** | **4.8682%** | **913/970** | **180 equivalent, 5 minor** |
| v6 conservative | 5.0881% | 5.0936% | 914/970 | 103 equivalent, 5 minor, 1 unparsed |

The final joint version removes **114 additional original covered positions**
versus v4: **9,562 → 9,448**. It adds 421 words across all 97 summaries, against
191,877 baseline narrative words. The maximum narrative match falls from 25 to
20 words. **0/97** narratives eliminate every match longer than five words;
the under-5% result concerns the **mean word fraction**, not a fraction of papers
that satisfy the strict rule. Pooled final coverage is 4.9132%.

### What the rules do

The catalog contains 37 families; 15 fire in the final joint run. It uses optional
grammar markers, explicit purpose/process phrases, whole-unit explanations and
source-grounded initialism definitions. The joint planner can combine insertions
that do not help individually, but accepts only a final plan that decreases
covered original words, covered total words, matching six-grams and fraction.
A 3% whole-summary added-word cap applies. All original characters and normalized
words remain in order. This is **surface elaboration, not independent conceptual
paraphrasing**.

New protections preserve full multiword scientific terms such as “regions of
interest”. Unicode normalization handles PDF ligatures; mixed-case and plural
initialisms are recognized with conservative alignment. Unregistered mixed-case
material/product names require direct initials instead of internal-letter guesses.
Exact units, scientific notation and guarded ranges remain unconverted. Ambiguous
symbols require positive context. Provenance relatives reject preceding verbal
governors, modals and negation, with a lexical backstop for POS tagger mistakes.

Numerical, unit, Greek-variable, operator, acronym, negation, modal and epistemic
signatures must agree. Original detected formula, code and quote spans remain
contiguous. Metadata and explicit evidence are not edited. The full-source audit
retains primary whitespace-word counts and a separate punctuation-split diagnostic.

### Failed versions are retained

The first insertion prototype had a shared temperature/pressure scope problem;
the initial larger catalog split `mol%` incorrectly. Their traces and raw judgments
remain in the evidence, along with guarded revisions.

v5 rewrote “protects tumor cells from CTL killing” into an invalid provenance
clause. The automatic reviewer marked it unintelligible, although QA stayed at
913/970. v6 blocks this verb-complement error. It also blocks a truncated descriptor
being guessed as the definition of the material name `JigSAP`. The final output
keeps the clinical verb frame and may add only a source-attested CTL definition.
QA alone does not certify sentence-level meaning.

## QA, meaning review and limitations

The pinned Qwen2.5-7B student uses all original 970 MCQs, historical prompts/parser,
temperature 0.5, top-p 0.95, presence/frequency penalties 1.05 and paired seeds.
Only exact-context answers are reused. The final corrected run needs ten new
student answers. The pinned Qwen3.8-27B-FP8 reviewer uses greedy decoding with
thinking disabled; only exact original/edited pairs reuse judgments.
See [model revisions](model_pins.json).

The automatic reviewer labels 180/185 final joint windows equivalent, with five
minor flags and no major/unintelligible/unparsed labels. These are **not human
error rates**. The conservative arm has one unparsed judgment. Known reviewer
control failures remain visible. All 97 papers, QA failures and raw answers remain;
no paper, answer or summary is selected using a new QA score.

The 97 papers have been repeatedly used for development. This is exploratory;
small QA differences do not demonstrate a generalization improvement. The proposed
10k study must use a frozen version and a new source-complete cohort.

## Compute

The final new insertion stage takes 7.678 warm CPU core-seconds/97 papers,
linearly extrapolated to **835 CPU core-hours/38M** or **13.2 CPU core-minutes/10k**.
These estimates include source indexing and primary audits, exclude worker startup
and bulk I/O, and assume similar source lengths and copy density. They cover only
this additional stage; upstream generation and Gemma repair retain their GPU costs.

The rejected v5 and corrected v6 offline tests together reserve **0.8778 GH200
GPU-hours**, including full-node reservations, startup, idle GPUs, QA, review and
final audits. [Accounting](results/allocation_accounting.json) keeps both attempts.

## Reproduce without GPUs

```bash
python replay.py --out /scratch/insertion-evidence
python -m pip install -r requirements.txt
python code/smoke.py

# Input must be the saved summary at this stage's baseline, after the earlier
# Gemma/CPU stages. Source must be its complete paper.
python code/repair_saved.py --summary saved-baseline.json --source full-paper.txt \
  --mode system_joint --out repaired.json
```

The 174 targeted checks include joint zero-gain plans, units/formulas, terminology,
actual clinical complements and custom material-name regressions. The shared
common-input bundle lives in the preceding targeted-deletion study; the replay
manifest references it by relative path and verifies its hash. New bundles contain
all eight insertion runs, summaries, traces, every raw QA response and judgment,
complete primary/expanded copy evidence, exact historical code, protocols and
timings. The replay verifies source/gold/context/code hashes and saved scores;
it does not silently rerun model inference or a cohort audit on a login node.

The active CPU code is the exact frozen v6 implementation. Historical HPC launchers
record the original environment and paths; adapt account, paths and environment
for a fresh model rerun. Do not install NLTK with its unrelated optional training
corpora: the required English tagger and Punkt assets are included.
