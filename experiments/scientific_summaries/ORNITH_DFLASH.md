# Ornith DFlash: paired scientific-summary pilot

This experiment repeats **20 of the published 97 papers**, with their unchanged
200 multiple-choice questions and the same Qwen2.5-7B-Instruct BF16 fixed student.
The completed 97-paper Q8 Ornith experiment is preserved in
[summary_runs/ornith15_9b](summary_runs/ornith15_9b), including validated summaries,
raw inference/repair journals, failures, prompt snapshots and all QA responses.
See the [main comparison](summary_comparison.html) for those completed results.

## What DFlash changes

[Ornith-1.5-9B-DFlash](https://huggingface.co/ornith-ai/Ornith-1.5-9B-DFlash)
is a **draft checkpoint**, not a standalone summarizer. It proposes token blocks
which the Ornith-1.5-9B target verifies. The pilot compares that same target with
and without the draft. A speedup is measured, not assumed from marketing claims.

The local GPUs are two RTX3090s (24GB each); both new arms use BF16 and TP=2.
FP8 inference on GH200 is a separate deployment scenario, **not this benchmark**.
Target revision: `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`.
Draft revision: `21be3446a606afa67e20c33e68ca37396499248f`.
Runtime: vLLM 0.27.1, Transformers 5.15.0, PyTorch 2.13.0+cu129.
The model card recommends recent vLLM (tested there with 0.28.0).

## Frozen selection and matching conditions

Select ten eligible documents from each dataset source (`arxiv`, `bethgelab`) by
ascending SHA256 of `dflash20-v1:` plus the document ID. Interleave sources.
Selection does not consult accuracy, summary length or generation success.
The manifest records all 20 IDs and source hashes before any inference.

Both arms use the original effective summary system prompt, temperature 0.2,
top-p 0.95, disabled thinking, identical document-derived seeds, 16,000 output
tokens maximum, full untruncated sources, a 65,536-token context, concurrency 4,
prefix caching and eager execution. Both use the same exact 19-field decoder
schema (the existing missing-paper recovery schema), V3 field-local/source-anchor
repairs and unchanged final validator. This is a deliberate, documented schema
intervention for **both** arms; do not interpret changes from the historical Q8
experiment as caused solely by DFlash.

Startup and a small warmup are outside the measured paper interval. The main
speed metric includes initial generation **and** all source repairs. Prometheus
speculative-token counters verify that DFlash was actually exercised. More or
less generated text/retries can change end-to-end speed; raw usage is published
so output throughput can be inspected alongside wall time. Stochastic outputs
need not be byte-identical, even though verified speculative decoding is intended
to preserve the target distribution.

## Run and resume

From the repository root, with the package installed (`pip install -e .`):

```bash
python experiments/scientific_summaries/benchmark_ornith_dflash.py \
  --server-python /path/to/vllm-venv/bin/python \
  --cache-dir /path/with/at/least/25GB/free
```

Reserve **both GPUs** before launch. The runner never kills unrelated GPU jobs;
it only stops isolated process groups it starts itself. Each completed paper is
checkpointed and its full raw journal saved separately. An incomplete arm stops
without shrinking the denominator. Rerun the same command to resume that arm.
Changes to run identity or implementation are rejected rather than silently
mixing settings. Publication is opt-in: add `--publish` only if the configured
Git remote is the intended destination and authentication is already configured.
No token is accepted as a CLI argument, saved in artifacts or printed.

After both arms complete, the runner automatically starts the original BF16
Qwen2.5-7B fixed student and evaluates the generated summaries. Frozen original,
no-context, Gemini-summary and KU responses are reused unchanged and audited.
QA uses the exact original paper indices for answer-choice permutation; selecting
a subset does **not** renumber the questions. Paired confidence intervals resample
papers, not individual correlated questions. A 20-paper pilot has limited power
to establish small accuracy differences.

Outputs live in `summary_runs/ornith15_9b_dflash20/`: manifest, prompt snapshot,
per-arm summaries/raw journals, metrics snapshots, fixed controls, complete QA
responses, and an English `comparison.html` / machine-readable `comparison.json`.
These files are generated only as the pilot runs; this document is not a claim
that results already exist.

## Reproducibility links

- [Exact 97-paper test set and instructions](data/README.md)
- [Summary prompt](summary-systemprompt+.txt)
- [V3 repairs](summary_repair_v3.py), [decoder schema](finish_ornith9b.py)
- [Fixed student protocol](evaluate.py), [BF16 judge serving](serve_judge.sh)
- [Target model](https://huggingface.co/ornith-ai/Ornith-1.5-9B)
- [Draft model and serving instructions](https://huggingface.co/ornith-ai/Ornith-1.5-9B-DFlash)
- [Project repository](https://github.com/LAION-AI/project-alexandria)
- [Alexandria paper](https://arxiv.org/html/2502.19413v2)
