# Architecture

## Windowing

Default `--windowing scenes`. Scenes are taken in source order and accumulated into a window
until adding the next one would exceed the target size; the next scene then starts a new
window. Boundaries therefore always fall *between* scenes.

```
target_words   = pages_target  × WORDS_PER_PAGE     # 6 × 185 ≈ 1,110
context_words  = pages_context × WORDS_PER_PAGE     # 5 × 185 ≈  925
```

`WORDS_PER_PAGE = 185`, derived from this document: 24,002 words over ~130 script pages.
It is a config value, not a constant, and is recorded in the run protocol.

Expected: ~22 windows over 224 scenes, ~10 scenes each.

**The uncovered head.** 2,385 characters precede the first slugline (title block, `FADE IN:`,
and the opening on-screen sequence). This is real content, not noise — it contains the first
Trinity/Cypher exchange. It is assigned to a synthetic scene `sc-000` with
`heading.kind = "PRE"` so it enters a window like any other. A check asserts the whole file is
attributed: no character of the source belongs to zero scenes.

**Alternative windowing.** `--windowing pages --overlap-pages 2` reproduces the page-aligned
scheme with overlap. A scene straddling a boundary is then assigned to the window containing
its *slugline*, and stage 2 gains a genuine dedup responsibility. Kept for comparison.

## What each stage-1 agent sees

Order matters, because it is what makes prefix caching work:

```
1. FULL SCREENPLAY            ← identical across every window ⇒ cached after the first call
2. DOCUMENT METADATA          ← title, credit, draft, date, source URL
3. SCENE INDEX                ← all 224 headings with ids, so ids are never invented
4. CONTEXT BEFORE (~5 pages)  ← read-only
5. CONTEXT AFTER  (~5 pages)  ← read-only
6. TARGET SPAN (~6 pages)     ← shown a SECOND time, verbatim
7. THE SCENE LIST TO EMIT     ← explicit scene ids, with a `const` binding in the schema
8. INSTRUCTIONS + FEW-SHOT
```

Regions 1–3 are byte-identical for every window in the run, so they prefill once per endpoint
(15.8 s) and are then free (0.8 s). Regions 4–8 are the only per-window cost.

Region 6 is the brief's deliberate repetition: the target text appears once inside the full
screenplay at step 1, once as the focused span at step 6. Repeating the span the model must
produce output for measurably concentrates attention on it.

Region 7 binds the output. The list of scene ids a window must emit is known before the call,
so it goes into the JSON schema as an `enum`/`const` rather than as a sentence in the prompt.
This is the storytree finding that structure fixes what instructions cannot: a prompt clause
asking for "one KU per scene" is advisory, a schema requiring exactly these ten keys is not.

## What each stage-2 agent sees

Deliberately **not** the screenplay. A seam agent receives only:

- the last ~3 scenes' KUs from window *i*
- the first ~3 scenes' KUs from window *i+1*
- the two windows' entity name lists

and returns a patch: entity alias merges, corrections to `preceded_by`/`followed_by`,
corrections to `context_before`/`context_after` that are wrong when read against the other
side, and (in `pages` windowing only) duplicate scene-KU removals.

Withholding the source is the point. A seam agent that could re-read the script would just
re-extract, and its agreement with stage 1 would measure nothing. Restricted to the KUs, it
can only find defects *visible in the KUs themselves* — which is exactly the property the
published artifact needs, since a downstream reader will also only have the KUs.

Seams are independent, so all ~21 run concurrently with no barrier.

## Scheduling

Eight endpoints, ports 8100–8107, one Qwen3.8-27B per A100. Dispatch is round-robin with a
per-endpoint worker pool; a window is retried on a different endpoint on failure.

Prefix caching is per-endpoint, so the first window sent to each of the eight pays the 15.8 s
prefill once. Total unavoidable prefill: 8 × 15.8 s ≈ 2 min, overlapped across endpoints,
so ~16 s of wall clock.

Concurrency caveat from the storytree work — *aggregate throughput is flat from 1 to 8
concurrent requests on a sparse MoE* — applies **within** one endpoint, not across the eight
independent ones. Per-endpoint concurrency is therefore kept low (2) and parallelism comes
from spreading windows across GPUs. The run protocol records per-endpoint timings so this is
checked rather than assumed.

## Structured output

Responses are constrained with a JSON schema via vLLM's guided decoding, passed through
`grammar_safe()`-style filtering for the keywords vLLM rejects, and then **validated in full
after the call**. The guarantee lives in the validator, not the grammar. Malformed responses
are retried once with the error text appended, then quarantined and reported — never silently
dropped, and never replaced by an empty unit.

> A guard that substitutes empty input for missing input is worse than no guard.

## Run protocol

Every run writes `protocol.json`: per-window and per-seam timings, endpoint assignment, token
counts, cache-hit evidence, retry and quarantine counts, every check's result including which
negative cases were exercised, and the resolved config. A run that cannot produce a complete
protocol is a failed run.
