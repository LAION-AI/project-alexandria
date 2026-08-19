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
| [**Results**](docs/06-results.md) | **The Matrix run: artifact, gate, and five-arm evaluation** |
| [**Worked examples**](docs/07-worked-examples.md) | **Three scenes end to end: source, unit, questions** |

## Source handling

The screenplay is read on-machine and never redistributed. Published artifacts contain
structure, offsets, digests, and a **link** to the public source — never its text. See
[provenance and scope](docs/05-provenance-and-scope.md), which also records a limit specific to
narrative that does not carry over from the paper's argument about scientific facts.

## Results in brief

*The Matrix*, 225 scenes, extracted in **424 seconds** on 8×A100: 1,021 ordered beats, 323
state changes, 109 entities. An independent audit over 14,989 fields finds the **longest
verbatim run from the source is 7 words**.

On 100 questions with a small 4B student, five samples each, non-leaky stratum (n=62):

| Arm | Accuracy |
|---|---|
| no context | *floor is chance, 0.25* |
| whole screenplay | 0.787 |
| whole KU chain | 0.729 |
| **scene-local source text** | **0.948** |
| **scene-local KUs** | **0.871** |

Two effects, separated by the scene-local text control:

- **Retrieval dominates, and is representation-independent.** Narrowing to the relevant five
  scenes is worth +0.161 for source text and +0.142 for units, both p<0.001. Serve a KU chain
  through retrieval, never in bulk.
- **At matched retrieval the units cost about 8 points** (−0.077, p=0.059) — the honest price
  of the copyright-safe transformation, a consistent penalty this sample cannot quite separate
  from zero.

**Closing the gap.** The deficit traced to over-compression of short scenes — a 35-word scene
lost a speaker entirely, a 31-word scene lost the physical detail a question asked about.
Narrower windows (3 scenes, not 12) plus a granularity prompt take scene-local units to
**0.926**, no longer distinguishable from the source-text ceiling (p≈0.5), **with verbatim
overlap unchanged at 6–7 words** — the gains are not bought with source text. Neither change
works alone. Costs 5× the extraction time and does not help the whole-chain arm at all.

Full numbers, the selection caveat, and the mechanistic check in
[results](docs/06-results.md).

## Status

Implemented and run end to end. Artifact, instrument, protocol and evaluation in
[`results/matrix/`](results/matrix/).
