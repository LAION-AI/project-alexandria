# Scientific summaries: frozen cohort and model comparisons

## Knowledge Units: chunk sizes and Gemma LoRA adapters

The [completed 500-word Gemma follow-up](gemma_ku_distillation_20261009/ku500_followup_20261010/README.md) adds all missing base/adapter conditions and freshly judges both chunk sizes on the same 20 papers.

The [English KU comparison](gemma_ku_distillation_20261009/README.md) brings together
Qwen 27B with 500-word and 1,000-word source chunks, Gemma E4B / 12B base models
and their rank-128 KU adapters, original-paper baselines and summary references.
It separates the disjoint 20-paper adapter evaluation from the original 97-paper
teacher benchmark, and includes confidence intervals, failed outputs, copy audits
and measured allocation costs. On the fresh papers, E4B improves from **65.50% to
81.00%** and 12B from **63.50% to 85.00%** with the 1,000-word KU adapters.

## Completed Gemma 4 E4B IT / 12B IT results

The [English Gemma findings/results](gemma4_97/README.md) cover both models on the same 97 papers / 970 MCQs, raw and corrected summaries, all 14 comparison scores, 48 batch throughput probes, paired confidence intervals, full traces and scheduler times. E4B scores **85.15% raw** / **84.64% corrected**; 12B scores **86.60% raw** / **86.80% corrected**. See the [shared GH200 comparison](ORNITH_DFLASH_GH200_RESULTS.md).

## Completed JUPITER Ornith + DFlash results (2026-10-03)

The [combined 9B / 35B-A3B GH200 comparison](ORNITH_DFLASH_GH200_RESULTS.md)
links both experiments. The [35B-A3B run](ornith35_dflash_97/README.md) uses the
same frozen test set and fixed student: **92.47% raw**, **92.68% corrected**,
with all 97 papers and all 9,700 combined QA answers validated. On the measured
35B workload, DFlash is slower than standalone autoregressive decoding.

The [complete English findings/results report](ornith_dflash_97/README.md) covers
all 97 papers and 970 MCQs, raw and corrected Ornith-1.5-9B BF16 + DFlash and
Qwen3.8-27B FP8 summaries, all eight QA conditions, paired confidence intervals,
standalone/DFlash throughput at every measured batch, complete-output request
times, self-audit failures, source recovery and scheduler GPU-hours. It includes
all per-paper traces, QA responses and checksums. Raw Ornith scores **93.81%**;
corrected Ornith **93.20%**. The small correction difference is statistically
uncertain, and Ornith summaries are longer than the new Qwen summaries.

See the [60-million-summary JUPITER GPU-hour estimate](ornith_dflash_97/SCALING_60M.md)
for an English table with and without correction, cache assumptions and scenario
ranges. These are planning estimates, not measurements at production scale.
This new GH200 BF16 experiment is separate from the earlier Q8 workstation
comparison and the 20-paper RTX3090 DFlash pilot described below.

This experiment compares existing `laion/Scientific-Summaries` summaries, Qwen3.8-27B
Knowledge Units, and new evidence-grounded summaries with the identical fixed
`Qwen/Qwen2.5-7B-Instruct` student. See the [Alexandria paper](https://arxiv.org/html/2502.19413v2).

Open [the English comparison dashboard](summary_comparison.html) locally. Pending
models are labelled pending, never zero. The previously completed four-condition
97-paper comparison remains in [report.partial.html](report.partial.html).

For an outsider-readable explanation of the complete summary workflow, open the
[standalone English methods and scaling report](summary_pipeline.html). It includes
the exact prompt inventory, source-check limitations, repair/recovery behavior,
fixed-student scoring, reproduction commands, model/runtime fingerprints, and
60-million-paper GH200 planning estimates for 27B and 9B FP8. These estimates are
explicitly not measured JUPITER performance. The companion `summary_pipeline.json`
contains prompt text, workload statistics, throughput assumptions, and input hashes.
Regenerate the report without inference using the standard-library-only command:

```bash
python experiments/scientific_summaries/pipeline_report.py
```

## Cohort and inputs

Download the [ready-to-use frozen test set and usage instructions](data/README.md).
`data/testset.json` contains exactly the 97 tested papers and 970 MCQs with the
**evaluated** option order; `data/testset_manifest.json` records IDs and input
hashes. Verify without GPU/network using `python experiments/scientific_summaries/export_testset.py --check`.

- `data/papers.json`: 100 frozen dataset records, source metadata, available paper texts,
  existing summaries, and content hashes. Fifty arXiv and fifty Bethgelab entries.
- `data/qa/`: ten source-only, Luna-authored four-choice questions per ready paper.
- The user approved proceeding with the existing **97** ready papers / **970** MCQs.
  Three entries without QA/KUs remain explicitly excluded in the comparison manifest.
- `pre_qwen_summary_results.json`: frozen no-context, original-text, existing-summary,
  and KU predictions. These controls are reused only after exact question/input-hash checks.
- Dataset license is declared CC-BY-4.0. Individual source identifiers remain saved.

"Paper text" means the text available in the dataset. It was not independently
checked against publisher PDFs. In particular, both initial arXiv diagnostic records
lack a bibliography in both their raw and sanitized fields. A conclusion marker and
a length threshold do **not** establish completeness. All conditions retain the same
frozen source text; missing sections are not silently fetched or substituted.

## Run order

**Fast Ornith run:** `python experiments/scientific_summaries/run_ornith9b.py`
uses two independent, pinned Q8_0 replicas (one per GPU), four 64k-context slots
per replica and eight concurrent paper requests. It must pass the two-paper smoke
test before scaling to the frozen 97-paper cohort. Original failed V2 outputs are
preserved under `summary_runs/ornith15_9b/v2_failed_archive/`; detailed initial
generations are recovered instead of shortened repair drafts. V3 shape/anchor
repairs are reused, with the Qwen completion's explicit bare-claim normalization:
narratives become unverified empty-quote entries, never automatically valid proof.
Initial prompt, full source text, 16k output budget and fixed-student decoding stay
unchanged. Context/batching/repair policy differ from earlier models and are saved
in the run config/snapshot. After all summaries validate, the owned replicas stop
and the same BF16 Qwen2.5-7B student evaluates only the new summaries, reusing
hash-checked frozen controls. The shared lock prevents overlapping queues.

**Final Ornith recovery:** `python experiments/scientific_summaries/finish_ornith9b.py`
preserves the original 94 completed representations and repairs only the three
missing records. Anchor decoding requires exactly the requested task IDs, with
only supplied candidate indices or `"drop"`; an invented `q8` can no longer block
the batch. Initial/field decoding requires the requested Schema-v4 keys, and bounds
grounded lists to 12 entries and claims to six to stop repetitive output loops.
These additional decoder constraints apply only to the recovery papers and are
disclosed under `completion_recovery` and `completion_schema_snapshot.json`.
The initial system prompt, 16k output limit, source text, scientific values and
strict final source validator are unchanged. The same fixed student warms up on
GPU 1 while recovery uses GPU 0; its evaluation starts automatically after 97/97
validate. Checkpointed recovery wall time is counted with one GPU, not two.

Full raw calls are preserved in `documents/` and `failure_journals/`, each referenced
and SHA-256-checked from the summary cache. The lean `summaries.json` retains every
attempt's metadata but avoids duplicating responses/tasks, because the original
monolithic Ornith cache exceeded GitHub's 100-MiB file limit. Model/source/student
contexts and proof spans are preserved. Publication verifies all referenced raw
journals and includes the decoder-schema snapshot; no raw responses are discarded.

1. Qwen3.8-27B MixedInt4 AutoRound: generate all 97 summaries, then fixed-student evaluation.
2. [Ornith-1.5-9B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF), Q8_0: same procedure.
3. [Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF), Q8_0: same procedure.

Pinned revisions and GGUF SHA-256 hashes are in `summary_comparison_manifest.json`.
Each model has separate `summaries.json`, `documents/`, `results.json`, prompt snapshot,
raw response journal and runtime log under `summary_runs/<model>/`. These checkpoints
cannot be confused with the earlier failed `qwen_summaries.json` experiment, which is
preserved for diagnosis.

```bash
# From the repository root, in the installed Alexandria Python environment:
python experiments/scientific_summaries/launch_summary_comparison.py
# Explicitly bypass another model's failed stage without changing the protocol:
python experiments/scientific_summaries/launch_summary_comparison.py --only-model qwen35_9b
# Render an audited progress/result dashboard without starting inference:
python experiments/scientific_summaries/summary_comparison_report.py
# Optional authorized GitHub publication after each audited model evaluation:
python experiments/scientific_summaries/publish_summary_results.py --watch
```

The launcher starts the queue in a new session with file-backed logs and no terminal
input, so closing the interactive terminal does not terminate generation. Use
`--publish` only when GitHub publication has been authorized. Queue and publisher
each hold a non-blocking filesystem lock: a second invocation exits without changing
the active run's status. Locks release automatically on exit; do not delete lock files.
The queue log is `summary_queue.log`, publication log `summary_publication.log`.
The foreground `run_summary_comparison.py` entry point remains available for debugging.
Both entry points accept repeated `--only-model` selections. The comparison manifest
retains all three model definitions and previous artifacts; completion status names
only the selected models. An unselected model is not represented as completed, and
the same two-paper smoke test, bounded repair protocol, and fixed-student evaluation
still apply. The publication watcher exits when the selected queue finishes.

If a previous queue exited but its Qwen server remains alive, the launcher accepts
`--reuse-qwen-server-pid PID`. This is explicit adoption, not GPU-wide termination:
the server must belong to the current user, lead its isolated process group, and
have the exact pinned vLLM module/model/revision/alias/localhost port. Linux process
starttime is retained to guard against PID reuse. The adopted server is stopped only
when its generation stage finishes (or fails); other GPU jobs are never targeted.

The queue waits for free GPUs, starts only its own local servers, and terminates only
its own process groups. The default paths match this workstation; the llama.cpp
binary is configurable with `--llama-server`. The 27B and judge environments/weights
can be configured using the existing `ALEXANDRIA_EXTRACTOR_*` / `ALEXANDRIA_JUDGE_*`
variables. No hosted API key is needed.

## Parallel generation and strict, bounded repair

### Versioned 9B restart (V3)

The final Qwen3.5-9B completion uses `finish_qwen9b.py`: it validates and preserves
all 94 already completed documents, backs up the checkpoint and prompt snapshot,
and processes only the three missing papers. Bare narrative strings inside claim
evidence/implications are rewrapped as entries with **unverified empty quotes**;
they cannot pass validation until the same model selects supporting source spans,
or explicitly drops unsupported statements. This is a logged shape repair, not
automatic evidence acceptance. The three remaining papers use a 65,536-token
context per slot (four slots, 262,144 total); the original full source and 16,000
output-token limit are preserved. Their context exception, policy, code checksum,
prior checkpoint checksum and affected IDs are recorded in `completion_recovery`.
The original 94 documents and their original V3 implementation hash are retained.
This heterogeneous completion policy must be disclosed with the resulting score.

Run `python experiments/scientific_summaries/finish_qwen9b.py` to resume this
workstation-specific completion. It holds the same queue lock, checkpoints each
finished document, and warms the unchanged BF16 fixed student on GPU 1/port 8011
while Qwen extracts on GPU 0/port 8010. The port can now be selected through
`ALEXANDRIA_JUDGE_PORT`; default remains 8010. Evaluation starts only when all 97
summaries validate. Only newly generated summary predictions are evaluated; frozen
controls, questions, option permutations and student decoding remain unchanged.

Citation repair examples use an empty string for the absent-bibliography case,
not a fictitious citation key that a small model could copy. Bibliography entries
are still allowed when actually present and must pass the unchanged validator.
Already repaired V3 drafts can resume without regenerating their narratives;
previous anchor selections, removals, transformations and raw calls are retained.
V2 recovery still starts from its original detailed generation, not a shortened
repair draft.

For an explicitly diagnosed implementation change, archive the failed run directory
and use `--only-model qwen35_9b --recover-failed-cache /absolute/path/to/archived/summaries.json`
with the detached launcher. Recovery accepts only failed-only V3 checkpoints with
identical model, pinned weights, source hashes, initial/repair system prompts and
generation settings. The new config records the archived checkpoint checksum,
previous implementation checksum and historical elapsed time; timing includes that
saved history. Ordinary resumes still reject implementation changes. Do not pass
an untrusted or actively changing checkpoint, or a completed/partly valid run.

The completed 27B run stays on V2 and is not regenerated. Restarted GGUF runs use
`--repair-protocol field_local_strict_grounding_v3` with the same initial system
prompt, source text, final validator, and fixed student. V3 does not spend five
4,096-token calls rewriting every failed field together. Instead it losslessly
rewraps only unambiguous `{narrative,evidence/source/quote}` entries, records those
shape edits, patches one malformed field at a time with an explicit shape example
(two attempts per field), and selects actual source quotes in batches of eight.
The anchor-call bound scales with the initial invalid-quote inventory (twice the
required batch count plus two, capped at 64), not a fixed six batches even when
more than 100 quotes need correction. Candidate source context is provided. No
quote is accepted merely because it was rewrapped; final literal alignment and
the <=5-word requirement remain unchanged. Unsupported drops stay auditable.

The original detailed generation is preferred over a shortened old repair draft.
Initial generation has at most two attempts with different recorded seeds. The
new templates, field examples and implementation hash are saved in each V3 prompt
snapshot/config. Checkpoints reject version/implementation changes. Failed V2
smoke artifacts are archived separately before a fresh V3 run. V2 mechanics below
describe the completed 27B run, not the restarted 9B policy; differences must be
disclosed in model comparisons and new cost estimates.

Each document is independent. Qwen27B has eight concurrent requests continuously
batched by vLLM across two RTX 3090s. Each 9B model has four llama.cpp slots on one
RTX 3090, with 32,768 tokens per slot and Q8_0 KV cache. The queue processes models
in the requested order, not simultaneously.

The effective user system prompt in `summary-systemprompt+.txt` is unchanged; only
its literal Python string-assignment wrapper is parsed without executing it. The
source-only user message explicitly explains missing bibliography/unreported-field
handling, consistent with the system prompt's empty-string rule. No QA, gold answers,
existing summaries or KUs enter generation.

Settings: temperature 0.2, top-p 0.95, thinking disabled, 16,000 output tokens,
32,768-token context. Requests use JSON-object constrained decoding. A conservative
parser also recognizes a single enclosing Markdown fence and logs its removal;
it never extracts arbitrary JSON from prose or accepts incomplete output.

The first Ornith GPU smoke test exposed a transport bug in the installed llama.cpp:
`response_format: {"type": "json_object"}` became an empty schema, which its template
code treated as no constraint. Repairs therefore emitted prose or exhausted their
token budgets. The resumed GGUF requests explicitly include the nonempty schema
`{"type": "object"}`; each response journal records the actual response-format payload.
The original failed journals are retained. Anchor candidate inventories now carry
explicit zero-based `index` values instead of relying on implicit list positions.
The already completed 27B predictions are not regenerated: its vLLM JSON constraint
was active. These transport/presentation fixes are disclosed, not a user-prompt or
fixed-student change. A resumed queue audits and reuses completed model evaluations.

All 19 fields are independently validated. If a field fails, **only that field** is
requested in a repair response, from the same model with the original source and a
separately saved evidence-repair system prompt. Untargeted fields cannot change.
Repairs have a 4,096-token cap and independently varied, recorded seeds. There are
at most six attempts in this primary stage. A bounded second stage (up to six calls)
handles persistent quote errors by lexically retrieving literal 3–5-word source
spans and asking the same model to choose a supporting span or explicitly drop the
unsupported entry. Retrieval is a proposal mechanism, **not** a declaration that a
malformed quote matched: only chosen actual source substrings enter the final ledger.
Unchanged narrative is preserved; removals, candidate inventories, selected indices,
and offsets are saved. Schema errors still use field-only patches. Each call handles
at most eight quote tasks, to bound inputs without truncating source text.

A failed whole-cohort pass is resumed once for missing papers; unresolved failures
stop the queue with a visible error, rather than being silently omitted or evaluated
as valid. Thus a document may have more than six journal records; the two bounded
stages and optional cohort retry are disclosed instead of misreporting them as one
generation. Initial generation uses the user prompt; the saved field-repair and
source-anchor-selection prompts are separate source-only instructions.

Every retained proof quote must be nonempty, at most five words, and located in the
source. Deterministic whitespace/NFKC-ligature alignment records the actual source
substring; there is no fuzzy evidence substitution. Unsupported entries may be
removed by the model and absent fields must be empty strings; every repair is saved.
Quoted text being present does not by itself establish that the narrative is true.
Citation placeholders, invented quotes and empty quote values are not accepted.
All inputs are preflighted with the serving tokenizer; no source text is truncated.
Finish reasons, token usage, raw output and prompt hashes are checkpointed.

## Evaluation and limitations

The student sees substantive narrative fields, including claims/takeaways, but not
proof quotes, metadata or citation rankings. Its exact input is saved as
`judge_context`. All models use the same paper IDs, questions and option permutations.
The fixed student is BF16 Qwen2.5-7B, temperature 0.5, top-p 0.95, 100 output tokens,
frequency/presence penalties 1.05, historical ASCII sanitizer and semicolon parser,
and up to five total formatting attempts. Final invalid answers count as incorrect.
`summary_runtime_fingerprints.json` records both GGUF checksums, the llama.cpp binary
hash/build version, local GPUs, and vLLM version. `gguf_smoke_results.json` records a
CPU-only tokenizer/JSON API transport test for both GGUFs, not a paper-quality score.

The dashboard audits evidence spans, source hashes, exact question/input hashes,
context budgets and unchanged controls before reporting scores. Confidence intervals
and cross-model differences use 10,000 paired document-cluster bootstrap draws,
keeping a paper's ten questions together, seed 250219413.

27B uses mixed-INT4/vLLM/TP2; 9B models use Q8_0/llama.cpp/one GPU. Runtime,
quantization, batching and KV cache therefore differ. Results measure the complete
representation pipelines, not a controlled weight-only ablation. Prompt word targets
are not guaranteed: median narrative length, repair rates and timing are disclosed.
Checkpointed active generation wall time includes repairs but excludes server
startup and idle gaps between runs. An interruption before the next completed
document checkpoint can also lose an uncheckpointed time interval. Multiplying
these saved intervals by the configured GPU count is a GPU-hours proxy, not a full
scheduler allocation bill and **not** measured GH200/Jupiter performance.

## Initial failure diagnosis

The old two-paper tests produced twelve outputs: nine were directly valid JSON,
three were valid JSON enclosed in Markdown fences. Sources and prompt hashes matched.
Examples of literal-proof mismatch included `r^-2` versus source `r−2`, and reconstructed
author accents versus PDF typography. The second paper also reproduced bibliography
template placeholders and unsupported ethics statements. The former repair loop
rewrote whole summaries and, after seeing code fences, lost the remaining grounding
errors; three retry responses were byte-identical. All generation HTTP requests
returned 200; there was no logged OOM. The pipeline itself shut down the server after
the failed smoke test. These journals remain available, rather than being erased.

## Source-copy audit (2026-10-05)

[Systematic English findings](copy_overlap_audit_20261005/README.md): more than 24,000 preserved summary versions audited, including the exact Qwen/Ornith training targets. All 291 no-thinking summaries exceed the six-word narrative overlap tolerance. The check is mandatory in future evaluations. [Completed FP8 quality, throughput and GPU-hour estimates](gemma_r128_fp8_97/evaluation/RESULTS.md) are reported alongside copy-compliance flags.
