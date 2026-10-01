# Scientific summaries: frozen cohort and three-model comparison

This experiment compares existing `laion/Scientific-Summaries` summaries, Qwen3.8-27B
Knowledge Units, and new evidence-grounded summaries with the identical fixed
`Qwen/Qwen2.5-7B-Instruct` student. See the [Alexandria paper](https://arxiv.org/html/2502.19413v2).

Open [the English comparison dashboard](summary_comparison.html) locally. Pending
models are labelled pending, never zero. The previously completed four-condition
97-paper comparison remains in [report.partial.html](report.partial.html).

## Cohort and inputs

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
python experiments/scientific_summaries/run_summary_comparison.py
# Render an audited progress/result dashboard without starting inference:
python experiments/scientific_summaries/summary_comparison_report.py
# Optional authorized GitHub publication after each audited model evaluation:
python experiments/scientific_summaries/publish_summary_results.py --watch
```

The queue waits for free GPUs, starts only its own local servers, and terminates only
its own process groups. The default paths match this workstation; the llama.cpp
binary is configurable with `--llama-server`. The 27B and judge environments/weights
can be configured using the existing `ALEXANDRIA_EXTRACTOR_*` / `ALEXANDRIA_JUDGE_*`
variables. No hosted API key is needed.

## Parallel generation and strict, bounded repair

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
Generation wall time includes repairs but excludes server startup. Allocated RTX
3090 GPU-hours are **not** measured GH200/Jupiter performance.

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
