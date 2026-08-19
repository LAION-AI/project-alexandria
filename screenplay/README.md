# Screenplay Knowledge Units

Extends Project Alexandria from scientific prose to **narrative** source texts: a two-stage
parallel agent pipeline that converts a screenplay into one temporally ordered chain of
Knowledge Units, carrying enough factual and sequential content that a small model can answer
detailed questions about the film **without ever seeing the screenplay**.

Reference run: *The Matrix* (1998 numbered shooting script), 224 scenes, on 8×A100 with a
local Qwen3.8-27B.

## What is different from the base pipeline

The base Alexandria KU is built for scientific prose, where facts are order-free and chunks
are deliberately independent. Narrative inverts both: the order in which the audience learns
things *is* the content, and most of the text is dialogue. This adds three things and reuses
everything else:

- a **temporal spine** — beats ordered within scenes, giving a strict total order over the film
- **indirect-speech rendering** — dialogue in third person, enforced by a hard overlap gate
- **non-generic contextualization** — per scene: what state the story is in, what it sets up

`entities`, `relationships`, `SourceReference`, MinHash fingerprints, alias resolution, and
the MCQ and overlap experiments are reused unmodified from `src/project_alexandria/`.

## Design in one paragraph

The whole screenplay is 39,637 tokens and vLLM prefix caching makes a shared prefix cost
**15.8 s once, then 0.8 s** — so every agent sees the entire script for free. Windows tile on
**scene boundaries** rather than pages, which makes duplicate and missing scenes structurally
impossible instead of repaired afterwards; the longest scene (714 words) is comfortably inside
one window (~1,110 words), so no scene is ever split. Stage 1 extracts ~22 windows in
parallel. Stage 2 verifies the ~21 seams, seeing only KUs and never the source, so it can find
only defects a downstream reader could also see. Six checks then run, each with a negative case
that proves it fires.

## Documentation

| | |
|---|---|
| [Plan](docs/00-plan.md) | Rationale, measured facts, where this departs from the brief |
| [Architecture](docs/01-architecture.md) | Windowing, prompt layout, scheduling |
| [KU schema](docs/02-ku-schema.md) | The scene record and its temporal spine |
| [Checks](docs/03-checks.md) | Six checks and their negative cases |
| [Evaluation](docs/04-evaluation.md) | 100-question MCQ, four arms |
| [Provenance and scope](docs/05-provenance-and-scope.md) | Source by reference, and an honest limit |

## Source handling

The screenplay is read on-machine and never redistributed. Published artifacts contain
structure, offsets, digests, and a **link** to the public source — never its text. See
[provenance and scope](docs/05-provenance-and-scope.md), which also records a limit specific to
narrative that does not carry over from the paper's argument about scientific facts.

## Status

Planning complete. Implementation next.
