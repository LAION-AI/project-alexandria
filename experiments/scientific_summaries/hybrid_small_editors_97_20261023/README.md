# Small-model hybrid wording repair: all 97 papers

Completed 23 October 2026; published 24 October 2026. This is a reproducible
comparison of conservative CPU phrase substitutions followed by
**Qwen3.5-0.8B** or **Gemma 4 E2B IT** on the same frozen 97 summaries.
The summaries were generated without thinking by the Qwen-teacher-distilled
Gemma 4 12B IT rank-128 LoRA, merged to FP8. The underlying papers, all 970
MCQs and gold answers remain fixed.

## Main results

| Condition | Mean copied-word coverage, ≥6-word matches | Source-match passages | QA correct / 970 | Warm repair seconds / 97 | Extra repair GPUh / 38M | Generation + repair GPUh / 38M |
|---|---:|---:|---:|---:|---:|---:|
| Unrepaired | 15.23% | 3,508 | 911 (93.92%) | — | 0 | 69,820 |
| Conservative CPU rules | 14.93% | 3,460 | 910 (93.81%) | CPU | 0 | 69,820 |
| CPU rules + Qwen 0.8B, one attempt | 10.61% | 2,585 | 910 (93.81%) | 41.70 | 5,339 | 75,158 |
| CPU rules + Qwen 0.8B, up to three attempts | 7.91% | 1,996 | 909 (93.71%) | 78.94 | 10,106 | 79,926 |
| CPU rules + Gemma E2B, one attempt | 9.07% | 2,295 | 916 (94.43%) | 30.30 | 3,880 | 73,699 |
| CPU rules + Gemma E2B, up to three attempts | 7.43% | 1,900 | 910 (93.81%) | 49.75 | 6,369 | 76,188 |

Coverage is the per-summary mean fraction of narrative words covered by source
matches of at least six normalized whitespace words. Explicit evidence quotes
and metadata are separately audited. Passage counts use longest-first,
nonoverlapping maximal target matches. Breaking a long passage may split it
into several shorter surviving passages, so counts and coverage differ.
**Every condition still fails complete five-word compliance on all 97 papers.**
No variant gets mean copied coverage below 5%.

Gemma with one attempt is the strongest measured cost/QA tradeoff, reducing
coverage about 40% relative to unrepaired summaries. Its +0.52 percentage-point
QA difference has a paired whole-paper bootstrap 95% interval of −0.21 to
+1.24 points, so improved QA is not established. Three Gemma attempts reduce
copying further but answer six fewer questions correctly.

## Meaning preservation: complete automatic review

An independent frozen 27B FP8 teacher reviewed **every final visible edited
window**, after document rollback. Identical A/B pairs reuse judgments. A window
can contain more than one sentence; these percentages are not sentence-level
human error estimates.

| Condition | Reviewed windows | Equivalent | Awkward only | Minor meaning change | Major change or unintelligible | Unparsed |
|---|---:|---:|---:|---:|---:|---:|
| Qwen, one attempt | 1,136 | 935 | 33 | 146 | 22 (1.94%) | 0 |
| Qwen, up to three | 1,679 | 1,289 | 40 | 289 | 60 (3.57%) | 1 |
| Gemma, one attempt | 1,450 | 1,245 | 7 | 160 | 36 (2.48%) | 2 |
| Gemma, up to three | 1,706 | 1,419 | 11 | 220 | 54 (3.17%) | 2 |

These errors remain in the released outputs and QA denominators. QA accuracy
does not certify faithful paraphrasing. The reviewer itself is imperfect:
it labelled statistical “significant → important” as minor although the control
expected major, and it can miss causal distinctions. All control responses are
included. Earlier CPU-rule judgments classified 104/104 changed units equivalent;
that is also automatic evidence, not a human safety guarantee.

A separate diagnostic found 21/22 orthographic-only Qwen edits in the one/three
attempt endpoints, versus zero for Gemma. Probable one-character spelling flags
were 15/20 for Qwen and 7/7 for Gemma. These are diagnostics, not post-QA filters.
The punctuation-split copying audit is retained separately because it changes
the token denominator and can split formulas into multiple components.

## Exact procedure and efficient execution

1. Apply the saved `function_prepositions` recipe: fixed discourse phrases,
   contractions and selected prepositional phrases. No general scientific
   noun/adjective substitutions, bare preposition swaps or broad `and` rule.
2. Find residual source-copy windows. Both editors receive the same 3,023
   selected windows; 2,954 require a first attempt. Neither editor sees QA gold.
3. Batch generation on one GH200 per editor: BF16, no thinking, up to 256
   sequences, 16,384 batched tokens, context 4,096, prefix caching and chunked
   prefill. Actual backends were FlashAttention 3 for Qwen and FlashAttention 4
   for Gemma. Pinning the runtime matters.
4. Attempt temperatures are 0.35 / 0.65 / 0.95, top-p 0.95, top-k 64 and at most
   384 output tokens. Decoder bans target bounded source six-word strings,
   not scientific nouns. Only unresolved windows retry. One- and three-attempt
   endpoints share their first round. The eight-request smoke outputs are reused.
5. Accept only finished prose within a 0.75–1.35 word-count ratio, preserving
   immutable-original number/unit/formula signatures, detected acronyms/chemical
   identifiers and negation count. Absolute copied-word coverage must decrease;
   the longest source match must not increase. Require bidirectional DeBERTa
   entailment ≥0.95. These guards demonstrably miss some scientific errors.
6. Reassemble structured fields, validate shape and protected evidence/metadata,
   and roll back a whole document when necessary. Gemma rolls back one document
   in each endpoint; no paper is dropped.
7. Answer all original questions with Qwen2.5-7B-Instruct using the historical
   context-before-question prompt and parser. Temperature 0.5, top-p 0.95,
   frequency/presence penalties 1.05, 100 output tokens, up to five format tries.
   Paired per-question seeds and exact-context answer reuse produce 5,280 unique
   calls for 5,820 slots. Invalid answers: zero. Bootstrap: 10,000 whole-paper
   resamples. Fresh raw/rules baselines avoid mixing old stochastic QA scores.

## Scaling and actual allocation

`GPUh = measured warm single-GPU worker seconds / 97 × papers / 3600 / 0.85`.
Generation uses the previous measured 545.363982 seconds per 97 complete
summaries, or 69,819.52 GH200 GPUh per 38M papers at 85% useful capacity.
For 60M, multiply the 38M figures by 60/38; exact values are in
[comparison_and_scaling.json](results/comparison_and_scaling.json).

Repair time includes prompt preparation, generation and candidate guards/NLI.
It excludes one-time loading, separate CPU final assembly and full-source audits,
benchmark QA and the 27B evaluation reviewer. Keep workers resident and overlap
CPU work in production. The first large batch still includes some sampling-kernel
shape compilation; the warm-loop estimate is conservative relative to a fully
warmed service. Extrapolation assumes comparable source and residual-window
distributions; it is not a guaranteed production cap.

The actual three allocations reserved **1.637 GPUh**, including startup and
evaluation: job 2468981 timed out after 668 seconds after completing all repairs
and QA; review-only job 2469526 timed out after 361 seconds during cold imports;
review-only job 2471215 completed in 444 seconds. All used four GH200 GPUs.
The final retry cached unchanged Python sources and package metadata in node
RAM, used in-process engines and checkpoints every 128 judgments, and reused all
completed outputs. Allocation accounting and per-stage timings are included.

## Reproduce saved scores without GPUs

From this directory, using Python 3.11+:

```bash
python bootstrap.py
python verify_results.py
python code/report.py
```

The hash-checked bundles contain the exact frozen inputs, all 582 saved summary
records, all QA predictions, automatic meaning judgments, candidate/rejection
traces, complete overlap evidence and repair checkpoints. `bootstrap.py` refuses
to overwrite a changed file. `evidence_manifest.json` records every archived
file hash. `historical_code/` preserves the original scripts byte-for-byte;
`code/` changes working-root/report/package-cache paths for portability.

For a new inference run use a separate clean working directory:

```bash
export ALEXANDRIA_HYBRID_ROOT=/path/to/new-hybrid-run
python bootstrap.py --inputs-only --download-models
cd "$ALEXANDRIA_HYBRID_ROOT"
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION /path/to/this/experiment/run.sbatch
```

Activate the matching CUDA 13 / Linux aarch64 environment first; recorded core
versions are in `requirements-recorded.txt`, exact model revisions in
`model_pins.json`. Download weights before allocating GPUs. For CPU baseline
reproduction, install `nltk` and fetch `punkt_tab` plus
`averaged_perceptron_tagger_eng` into `resources/nltk_data`, then run
`python /path/to/this/experiment/recompute_rules.py` on a compute allocation.
Baseline reconstruction is optional because frozen raw and CPU-rule outputs
are already provided. For a fresh full-source audit after extraction, run
`python recompute_overlap.py --out /path/to/fresh-audit` on compute. It refuses
to overwrite an existing audit directory. Never modify frozen published evidence.

GPU kernels, eager/compiled reviewer execution, scheduling and hardware can
produce small numerical differences despite fixed seeds. This repeatedly used
synthetic testset is an exploratory benchmark, not an untouched blind test.
Context-specific dictionary work derived from these 97 papers must be labelled
test-informed and validated separately before a generalization claim.
