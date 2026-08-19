# Project Alexandria

> Toward reusable scientific knowledge without redistributing the expression of source texts.

Project Alexandria converts documents into **Knowledge Units (KUs)**: structured records of
entities, attributes, and relationships, accompanied by non-reversible source fingerprints. The
approach is introduced in [*Project Alexandria: Towards Freeing Scientific Knowledge from
Copyright Burdens via LLMs*](https://arxiv.org/abs/2502.19413) (arXiv:2502.19413v2).

This repository brings together the paper pipeline, results, and previously unpublished findings
from the original project report. It also adds a batched extraction path with document-level
entity resolution.

## What is here

- A polished **sequential baseline** matching the paper: sentence-aware 200-word chunks, the
  previous ten KUs as naming context, and a few-shot extraction prompt.
- A new **parallel pipeline**: independent 500-word targets, up to 1,000 words of context on each
  side, title/abstract context, vLLM batch generation, then one document-wide alias-resolution
  pass.
- Provider-neutral inference through any OpenAI-compatible endpoint; no HyperLab dependency and
  no hard-coded credentials.
- Reusable MCQ, n-gram overlap, embedding-control, quality-audit, and self-correction components.
- Published and unreleased result tables with explicit provenance.

## Key finding

Knowledge Units retained most task-relevant information in the published MCQ experiments: the
best abstract-level KU scores were around 93–95%, close to original-text upper bounds around
95–97%. A particularly instructive side experiment also showed why cosine similarity is not a
sufficient knowledge-retention measure. With BGE-M3, an original abstract was **more similar to a
word-scrambled version (0.8903)** than to its Knowledge Unit (0.8162), even though scrambling
destroys syntax and many relations. See [the embedding analysis](docs/results/embedding-similarity.md).

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Add `.[pdf]` for PDF input or `.[vllm]` for offline vLLM. No API key is needed for a local server.
For a protected compatible endpoint, set `ALEXANDRIA_API_KEY`; do not put keys in source files.
Add `.[eval]` to read the released MCQ Parquets and run the reproduction command.

## Sequential paper-style extraction

```bash
alexandria extract paper.txt \
  --mode sequential \
  --chunk-words 200 \
  --previous-units 10 \
  --backend openai \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen38 \
  --output outputs/paper.sequential.json
```

Short texts use exactly the same command; if the source is shorter than the chunk size, the
pipeline emits one KU.

## Parallel extraction and entity reconciliation

```bash
alexandria extract paper.txt \
  --mode parallel \
  --chunk-words 500 \
  --context-words 1000 \
  --concurrency 8 \
  --backend openai \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen38 \
  --output outputs/paper.parallel.json
```

Each target is extracted independently; neighboring text is marked as read-only disambiguation
context. One final model call sees the title, abstract, KU summaries, and extracted entity metadata,
producing conservative canonical IDs and alias lists such as
`alice_miller -> [Alice M., Alice Miller, Miller, Alice]`.

Entity resolution can be resumed or rerun without source text or re-extraction:

```bash
alexandria canonicalize outputs/paper.parallel.json \
  --base-url http://127.0.0.1:8000/v1 --model qwen38 \
  --max-tokens 4096 --output outputs/paper.parallel.canonical.json
```

For offline vLLM, replace `--backend openai` with `--backend vllm`. The requested GGUF route is
documented in [running Qwen3.8-27B](docs/running-qwen38.md).

## Documentation

- [Method and architecture](docs/methodology.md)
- [Published paper results](docs/results/paper.md)
- [Unpublished report results](docs/results/unpublished-report.md)
- [Embedding similarity caveat](docs/results/embedding-similarity.md)
- [Local Qwen3.8 validation and independent review](docs/results/qwen38-validation.md)
- [Qwen3.8 MCQ pilot](docs/results/qwen38-mcq-pilot.md)
- [Historical-code provenance](docs/provenance.md)
- [Reproducibility guide](docs/reproducibility.md)
- [Released evaluation datasets](data/evaluation/README.md)
- [MCQ reproduction and historical judge](docs/reproducing-mcq.md)

## Scope and legal caution

The paper offers a technical and legal position, not legal advice. Whether a workflow is lawful
depends on jurisdiction, access rights, purpose, source material, and operation. This code avoids
emitting source chunks and records only non-reversible fingerprints, but operators remain
responsible for lawful access, temporary-copy handling, output review, attribution, and deletion
policies.

The software in this repository is released under the [Apache License 2.0](LICENSE). Source
documents and third-party datasets/models retain their own terms and are not relicensed here.
