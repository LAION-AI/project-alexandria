# Screenplay Knowledge Units — plan

Extending Project Alexandria from scientific prose to **narrative** source texts, using a
two-stage parallel agent pipeline and a local 27B model on 8×A100.

Target artifact: one temporally ordered chain of Knowledge Units covering all 224 scenes of
*The Matrix* (1998 numbered shooting script), sufficient for a small student model to answer
detailed factual questions about the film **without ever seeing the screenplay text**.

---

## Why narrative needs a different KU

The existing Alexandria KU is built for scientific prose, where facts are *order-free*: that
photosynthesis releases oxygen is true regardless of which paragraph says so. The schema
therefore has no place to put time, and the pipeline is explicitly designed so that
"no target sees KUs from another target."

Narrative inverts both properties:

| | Scientific prose | Screenplay |
|---|---|---|
| Fact ordering | irrelevant | **is the content** — who knew what when is the plot |
| Chunk independence | a feature | loses causality across the seam |
| Expression to avoid | distinctive phrasing | **dialogue**, which is most of the text |
| Natural unit | ~200-word chunk | **the scene**, already delimited by the source |

So this work adds three things to the existing KU: a **temporal spine** (a total order over
beats), **indirect-speech rendering** of dialogue, and **non-generic contextualization**
(what state the story is in when the scene opens, what it sets up).

Everything else — `entities`, `attributes`, `relationships`, `SourceReference`, MinHash
fingerprints, document-level alias resolution, the n-gram overlap experiment — is reused
unchanged from `src/project_alexandria/`.

---

## Measured facts this plan is built on

Established before writing it, not assumed. All from the working copy at
`/home/deployer/laion/bookwriter/reconstruct/runs/matrix/`.

| Fact | Value | Consequence for the design |
|---|---|---|
| Screenplay length | 24,002 words / **39,637 tokens** | The **whole script fits in one 131k context** with 70% to spare |
| vLLM prefix caching | on, all 8 endpoints | Full-script prefix: **15.8 s cold → 0.8 s warm**, a 20× saving |
| Existing scene map | 224 scenes, char offsets | Chunk on **scene boundaries**, not pages |
| Scene map integrity | **224/224 anchors verify**, spans tile contiguously | Safe to slice — bug #6 of the storytree project does not recur here |
| Scene length | median 45 w, **max 714 w** | A scene can never straddle a 6-page (~1,100 w) window |
| Coverage | 98.1% (2,385-char head uncovered) | Title page + pre-slugline opening need explicit handling |
| Student model | `google/gemma-4-E4B-it`, **131k ctx**, cached locally | Can take the full screenplay *and* the full KU chain as context |

The two that changed the design are the last-but-one and the second. Because the longest
scene is 714 words and a window target is ~1,100, **the case the overlap scheme existed to
repair cannot occur.** And because the full script prefills once per endpoint and is then
free, there is no reason to withhold it from any agent.

---

## The design decision, and why it departs from the brief

The brief specifies 6-page windows with a 2-page overlap, and a second agent round that
removes duplicate KUs and repairs gaps at the seams.

**This plan tiles on scene boundaries instead, with no overlap in the target span.** Each of
the 224 scenes belongs to exactly one window. Duplicates and gaps then become *structurally
impossible* rather than *repaired after the fact*.

This is the storytree project's own most repeated finding, applied here:

> instructions repair local fields, structure repairs global consistency
> — `bookwriter/docs/00-HANDSHAKE.md`

The brief's open question — *"the question is if that ever occurs in a screenplay"* — now has
a measured answer for this text: the longest scene is 714 words, comfortably inside one
window, so no scene is ever split and the overlap is not needed to catch one.

**Stage 2 is kept, and still runs on every seam.** It stops being cleanup and becomes
verification: it checks that the temporal chain actually links across the boundary, that an
entity is named the same on both sides, and that each scene's "what came before / what this
sets up" is right when read against the neighbouring window. A pass that finds nothing is a
useful result here; a pass that silently had nothing to find would not be.

The page-aligned + overlap variant stays runnable via `--windowing pages --overlap-pages 2`,
so the two can be compared rather than argued about.

---

## Pipeline

```
                    script.normalized.txt  (never leaves the machine, never committed)
                              │
                    scene map (224 scenes, offsets verified against anchors)
                              │
        ┌─────────────────────┴─────────────────────┐
        │  STAGE 1 — extract, ~22 windows, parallel │
        │  each agent sees:                          │
        │    · the FULL screenplay      (cached prefix, ~free)
        │    · ±5 pages neighbour text  (read-only)
        │    · its 6-page target span   (shown a SECOND time)
        │  emits: one scene-KU per scene fully inside the target
        └─────────────────────┬─────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │  STAGE 2 — verify seams, ~21 pairs, parallel
        │  sees only the KUs either side of one seam │
        │  emits: link repairs, alias merges, order fixes
        └─────────────────────┬─────────────────────┘
                              │
              deterministic assembly + document-wide canonicalization
                              │
                    ku_chain.json  ← the publishable artifact
                              │
        ┌─────────────────────┴─────────────────────┐
        │  CHECKS (all six with negative test cases) │
        └─────────────────────┬─────────────────────┘
                              │
                    MCQ evaluation, 3 arms, Gemma-4-E4B
```

Stage 1 is ~22 windows over 8 endpoints ≈ 3 rounds. Stage 2 is ~21 independent pairs.
Neither stage has a barrier inside it.

---

## Documents

| | |
|---|---|
| [01-architecture.md](01-architecture.md) | Windowing, prompt construction, scheduling |
| [02-ku-schema.md](02-ku-schema.md) | The scene-KU record and its temporal spine |
| [03-checks.md](03-checks.md) | Six checks, each with the negative case that proves it fires |
| [04-evaluation.md](04-evaluation.md) | 100-question MCQ protocol, three arms |
| [05-provenance-and-scope.md](05-provenance-and-scope.md) | Source by reference, overlap gate, honest limits |

## Status

Planning complete. Implementation not yet started.
