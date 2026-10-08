# Qwen 27B Knowledge Units with 1,000-word chunks

Requested experiment: extract Knowledge Units from all 97 frozen scientific
QA papers with Qwen3.8-27B, using 1,000-word source targets, then compare
answerability with the same fixed Qwen2.5-7B-Instruct protocol used for the
summary experiments. No training is performed on the evaluation papers.

## Method and relationship to the paper

The [Alexandria paper, arXiv v2](https://arxiv.org/html/2502.19413v2) describes
local entity/attribute/relation graphlets. Its long-paper experiment uses
200-word chunks and the preceding ten KUs as continuity context. This run is
an explicit larger-chunk extension, not a reproduction of those 200-word
results. It reuses the repository's few-shot sequential extraction prompt,
strict parser, KU schema, source fingerprints and factual-context serializer.

Targets are sentence-aware, nonoverlapping, **at most 1,000 whitespace-delimited
words**; the final target can be shorter. Every source word is covered exactly
once. Calls receive generated summaries/entity names from the preceding ten
KUs for naming continuity. No source opening abstract or neighboring source
window is added. The extractor sees source text and bibliographic metadata,
never MCQs, answer keys or evaluation predictions. Independent documents can
run concurrently; chunks within each document remain sequential. There is no
additional semantic-correction or global alias-resolution call.

Qwen3.8-27B-FP8 pin: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
Qwen2.5-7B-Instruct pin: `a09a35458c702b33eeacc393d103063234e8bc28`.
Thinking is disabled. Generation uses temperature 0.1, top-p 0.95, structured
JSON and fixed per-document/chunk seeds. The normal completion budget is 8,192
tokens; one predeclared format retry can use 16,384. Incomplete responses are
retained and rejected rather than salvaged as a complete KU.

## Fixed evaluation

All 97 papers and all 970 immutable MCQs remain in every condition:

1. No context.
2. Complete original source.
3. New 1,000-word KUs, serialized as factual context without fingerprints.
4. Previously generated direct Gemma rank-128 FP8 summaries.

All four conditions are freshly judged by Qwen2.5-7B-Instruct using the same
historical ASCII sanitizer, prompt, answer parser, temperature 0.5, top-p 0.95,
100 output tokens, frequency/presence penalties 1.05 and up to four
invalid-answer retries. Concurrency remains four on the judge. Judge inputs
are checked against a 32,768-token context limit, with no source/context
truncation. Failed extraction or context overflow retains wrong/invalid slots
and is explicitly reported. No extraction changes are made in response to QA.
Paired intervals resample 97 paper clusters, retaining all ten questions.

The frozen test set hash is
`a0d5e5f99a0025c6cd8a5140a39a07a220ded6f37f04994500549886ffd83261`.
The source-only input contains 341,673 words in total, 1,533–5,367 per paper.
Inputs, model revisions, repository snapshot and code hashes are recorded in
the run protocol. Raw source documents and source-containing request traces
remain in experiment storage outside GitHub.

## Source-copy and performance evidence

The complete untruncated-source n-gram audit is mandatory. KU summaries,
entity names and attribute/relation labels/values are checked as separate
factual fragments; titles are bibliographic metadata. No artificial matches
are created by concatenating separate graph fields. Five words are allowed,
six borderline and seven or more flagged. Preserve failed outputs and copy
flags alongside QA, and report 5/7/11-gram Jaccard diagnostics separately.
These diagnostics do not certify scientific correctness or legal status.

One Booster allocation uses three independent single-GH200 extractor servers
and one fixed judge server. Complete response tokens, bounded retries,
per-replica wall time, startup/server accounting, QA time and actual allocated
GPU-hours are recorded separately. Request latency sums are not GPU wall time.
The runtime produces atomic chunk, paper and QA checkpoints for resumption.

Scratch run:
`/e/fscratch/reformo/schuhmann1/scientific-qwen27-ku1000-97-20261008`.
Live status: `outputs/status.json`; per-paper extraction: `outputs/knowledge_units/`;
QA progress: `outputs/qa_progress.json`; final results: `outputs/RESULTS.md`.

Code: [common.py](common.py), [extract.py](extract.py),
[evaluate.py](evaluate.py), [report.py](report.py),
[orchestrate.py](orchestrate.py), [run.sbatch](run.sbatch).

## Completed results

The full 97-paper run completed successfully on 8 October 2026:
**391/391 chunks**, zero extraction failures and zero format retries.
[Results, comparisons, timings and conclusions](RESULTS.md) include:

| Context | Correct / 970 | Fresh QA accuracy |
| --- | ---: | ---: |
| No context | 603 | 62.16% |
| Complete original source | 941 | 97.01% |
| Qwen27 KUs, 1,000-word chunks | 879 | 90.62% |
| Direct Gemma rank-128 FP8 summaries | 912 | 94.02% |

The KU-minus-summary gap is **−3.40 percentage points** (95% paired
paper-bootstrap interval −5.26 to −1.55). The KUs improve answerability over
no context but do not outperform these summaries in this run. All four
conditions were freshly judged; the fixed stochastic answerer explains small
differences from earlier evaluations of the same summary texts.

Active extraction took **3.91 minutes** across three GPUs, or **0.1906 active
extractor GPU-hours**. The complete cold-start benchmark allocation took
**1.2067 reserved GPU-hours**, or **1.3022 including the first node's failed
filesystem-healthcheck allocation**. The complete-source overlap audit found
0/97 strict five-word passes; 96/97 contain a run of seven or more words,
maximum 24. Mean factual-fragment coverage in runs ≥6 words is 9.56%.

Machine-readable evidence: [report.json](report.json),
[QA predictions/responses](qa-results.json.gz),
[generated Knowledge Units](knowledge_units.jsonl.gz),
[copy audit](copy_overlap.json), [copy CSV](copy_overlap.csv),
[copy details](copy_overlap_details.jsonl.gz),
[performance](performance.json), [GPU accounting](allocation_accounting.json),
[runtime](runtime.json) and [artifact manifest](artifact_manifest.json).
Full source-containing request traces are preserved outside GitHub.
