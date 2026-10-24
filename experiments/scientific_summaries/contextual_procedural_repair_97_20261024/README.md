# Context-sensitive procedural repair after Gemma E2B

Two completed experiments, 24 October 2026. These reuse the same saved 97
summaries after conservative CPU rules and one Gemma 4 E2B IT repair round.
The [preceding small-editor benchmark](../hybrid_small_editors_97_20261023/README.md)
contains the generator, model pins, full97-paper QA protocol and original timings.

## Results: the additional gain is small

| Pipeline | Mean copied-word coverage ≥6 | Source-match passages | QA correct | Newly changed windows reviewed |
|---|---:|---:|---:|---:|
| Gemma one-round baseline | 9.0724% | 2,295 | 916/970 | 0 |
| Safe rules again | 9.0502% | 2,290 | 915/970 | 8 |
| First learned phrase bank | 9.0429% | 2,291 | 915/970 | 11 |
| Extended learned bank | 9.0380% | 2,292 | 915/970 | 15 |

The extended bank changes14 summaries and15 sentence/window units. All15 new
CPU changes were automatically judged equivalent. Existing Gemma-stage errors
remain a separate concern: its original review flagged36 major/unintelligible
windows among 1,450 edits. The new CPU audit does not certify those were fixed.

The extra bank reduces copying by only 0.0344 percentage points versus one-round
Gemma (about 0.38% relative). Every summary still has a source match longer than
five words. The one-question QA difference is exploratory, not evidence of a
reliable gain or loss. Candidate-rule derivation and validation never use QA gold.

## Why replace phrases instead of arbitrary words?

The existing guarded WordNet diagnostic produced 30 major/unintelligible cases
in 150 sampled changed windows; the naive diagnostic produced 42/150. Good or
merely awkward judgments were 50/150 and 53/150 respectively; remaining judgments
were minor meaning changes. Each diagnostic applies at most one greedy edit
within a selected copied window. These are deliberately targeted edit-level
samples, not error rates for all papers or all sentences.

Examples included incorrect senses such as acid→bitter, speech→address and
quenched→satisfied, as well as malformed inflections. A blacklist cannot supply a
universal correct replacement for every technical term. Luna reviewed training
failures and retained 15 concrete proposal/KEEP cases; these are explicitly marked
undeployed. No new97 output was silently modified after QA.

## Training-driven rule construction and two iterations

1. Scan all 865 released teacher-training papers. Verify none shares an ID or
   normalized source hash with the 97 testpapers. The prior dataset release also
   documents its original evaluation-exclusion audit.
2. Freeze 703 derivation and 162 validation papers using document-hash modulo 5.
   Mine concrete source-copy-breaking phrase alternatives and dictionary controls.
   Initial 33 candidates produced 860 occurrences. The second 50-candidate run
   produced1,056 occurrences. Both complete inventories are included.
3. Review at most 16 derivation and 8 validation exemplars per rule, one per paper.
   Sample 150 guarded and 150 naive dictionary controls by deterministic pair hash.
   First run: 503 cards; second: 638. Exact A/B judgments from the first are reused
   in the second, so only 135 new unique pairs need inference there.
4. Admit a rule only with at least 5 derivation documents and 3 validation documents,
   ≥95% and ≥90% equivalent/awkward judgments, respectively, and zero major,
   unintelligible or unparsed outcomes in either split. This is an empirical gate,
   not a statistical or human guarantee of 90% generalization.
5. Luna receives derivation failures only and may add contextual exclusions or
   disable a rule. The first phrase bank has no observed derivation failures;
   no refinements are invented. In the second run an `often` rule is conservatively
   disabled. Its judge incorrectly claimed `often` disappeared although the
   rewrite retains it; this inconsistent rationale is preserved, not treated as
   confirmed meaning loss. The accepted-bank outputs remain identical before
   and after that disable because the rule already failed the first gate.
6. Freeze the resulting bank before applying it to all 97 saved Gemma summaries.
   Reapply safe rules and test both banks as named conditions. Keep every paper,
   protected evidence/metadata, number/unit/formula, detected identifier,
   negation and modal count. Require absolute copying reduction, not denominator
   inflation. Whole-document rollback guards remain active.
7. Review every final visible new CPU edit and answer all 970 original questions
   per condition with the same paired-seed Qwen2.5-7B student. Reuse exact-context
   answers. Retain complete-source primary and punctuation-split overlap audits.

The first bank admits 6 new rules. The second admits 9: the same 6 plus three
positive copula/adverb orderings. New active changes on 97 are mostly causal
phrase alternations, safe contractions and four adverb moves. Static-property
possessives such as `the material's properties` lacked sufficient validation
occurrences in copied training passages and remain excluded. Frequent scientific
content and fixed technical phrases account for much of the remaining overlap.

Historical97 failures informed general safety blacklists and the repeatedly used
benchmark influenced research direction. Therefore these are test-informed,
exploratory results; the train/dev rule-selection split does not turn97 into a new
untouched blind evaluation.

## Compute

Generation plus one Gemma E2B repair retains the previous estimate of 73,699 GH200
GPUh per 38M summaries, including 85% useful GPU capacity. The new deployed step
uses CPU only. Measured extra core-time is about 570 core-hours per 38M; budget
roughly 600–700 for this compute portion, with bulk I/O outside the measurement.
Source indexes are shared/cached across conditions, so do not interpret small
CPU timing differences between banks as a speed ranking. CPU work must overlap
GPU generation/repair for zero additional GPU-hours in the production projection.

Offline discovery/review/QA allocations used four GH200s each: job 2487513 ran 578
seconds, job 2487740 ran 445 seconds, reserving 1.137 GPUh total. Models and source
artifacts were reused; Python sources/metadata were cached in node RAM, compiled
kernels reused on the second run, reviewers remained loaded across phases and
checkpointed judgments, and only unresolved/new pairs received model calls.

## Use the frozen CPU bank

No LLM, WordNet, NLI or GPU is required to deploy the saved finite rules:

```bash
python repair_saved.py --summary gemma-summary.json --source paper.txt \
  --rules rules/structural_deployed.json --out repaired.json
```

The summary input may be a structured 18-field object or a saved record with a
`summary` member. The source must be its complete paper text. Runtime code shares
the measured greedy rewrite/guard implementation and audits the final narrative;
protected quotes and metadata remain unchanged. Reuse source indexes and parallelize
independent papers in bulk production. This does not guarantee the five-word limit.

## Replay the saved experiments without GPUs

```bash
python replay.py --out /path/to/new-replay-directory
```

Hash-checked bundles contain exact raw inputs, training/development inventories,
reviewed edits and labels, learned banks, all repaired outputs, all 7,760 QA slots
across both four-condition experiments, copying evidence and timing markers.
`historical_code/` preserves the actual JUPITER scripts, including the Slurm launchers.
Full discovery reruns require adapting their root/interpreter paths and the same
pinned models/environment from the preceding benchmark. The865 training input is
`data/train.jsonl` from the [released teacher dataset](https://huggingface.co/datasets/ChristophSchuhmann/scientific-summary-distillation-Qwen3.8-27B-865-20261004),
revision `9f572fb367865b7eaf7dca37b5cb269558b84026`. The historical WordNet diagnostic
uses Open English WordNet 2025 plus nltk 3.9.1, wordfreq 3.1.1 and lemminflect 0.2.3;
production rule deployment and saved-evidence replay use the Python standard library.

All judgments remain available. No claim of zero meaning errors, human-reviewed
sentence accuracy, or elimination of all copying is made.

## Follow-up: deleting words and changing numeric notation

The [completed CPU-edit follow-up](../targeted_deletion_97_20261024/README.md) adds a readable dashboard, full-summary reader, separate adjective/adverb-only QA, article identifier protection, and two numeric-formatting prototypes. All preceding results are preserved.
