# Checks

The storytree project that this pipeline borrows its scene map from accumulated **eight
measurement errors, and its own apparatus caught none of them**. Six were found by an outside
reader, two by recomputation. Every one produced a confident number computed over the wrong
thing, and not one failed loudly.

The rule earned from that record governs this document:

> **A check that has never been shown to fail is not a check.**

So every check below ships with a **negative case**: a deliberate corruption that it must
reject. The negative cases run in CI on synthetic fixtures, and the run protocol records which
were exercised. A check whose negative case has not run this release is reported as
`unverified`, not as `passed`.

---

## C1 — Coverage

Every scene in the map has **exactly one** KU; every KU names a scene in the map.

*Negative:* drop one KU; duplicate one KU; emit a KU for `sc-999`. All three must fail.

Also asserts total attribution: every character offset of the source belongs to exactly one
scene, including the 2,385-char head assigned to `sc-000`.

## C2 — Verbatim overlap **(hard gate)**

The check the artifact's legitimacy rests on. For every text-bearing field of every KU
(`content`, `context_before`, `context_after`, `style`, attribute and relationship values),
compute the longest contiguous word n-gram shared with the source scene, after case folding
and punctuation stripping.

| | |
|---|---|
| **Fail the run** | any single n-gram ≥ **8 words** |
| **Warn** | any ≥ 6 words, or 95th percentile ≥ 5 |
| **Report always** | full distribution, worst 20 offenders with scene ids, and the same statistic against the *whole screenplay* rather than just the local scene |

Proper nouns, slugline locations, and pure number strings are exempt from the n-gram count —
they are required to survive verbatim by C3 and would otherwise fight it. The exemption is
applied by masking those tokens before matching, and the masked-token count is reported, since
a large mask is itself a way to smuggle a phrase through.

This is a **gate, not a metric**: on failure the run stops and emits no `ku_chain.json`.

*Negative:* inject a verbatim 10-word line of dialogue into one KU; inject an 8-word action
line; inject a line split across two adjacent beats to test that matching is not per-field
naive. All must fail the run.

Implemented over `experiments/overlap.py`, extended from document-level to field-level.

## C3 — Factual fidelity of what must not be paraphrased

Extract from each source scene: all numbers, all dates/times, all proper nouns (from the
speaker list and slugline), all locations. Assert each appears, unaltered, in its scene's KU.

Reported as recall per category, per scene. Below 0.95 on numbers fails the run — a drifted
number is a destroyed fact, and this pipeline's entire value proposition is that facts
survive.

*Negative:* change a phone digit; change a floor number; rename a character in one KU;
substitute a synonym for a slugline location. All must fail.

C2 and C3 pull in opposite directions by construction. Running both is the point: C2 alone
would be satisfied by a vague summary, C3 alone by a copy.

## C4 — Temporal totality

`(scene_index, beat_order)` must be a strict total order: no duplicate pairs, `beat_order`
contiguous from 1 within each scene, `preceded_by`/`followed_by` forming a single unbroken
chain over all 224 scenes with the right two endpoints null.

`causes` must reference only beat orders present in the same scene, and must be acyclic.

*Negative:* duplicate a beat order; leave a gap at order 3; break the chain at one seam; make
beat 2 cause beat 4 which causes beat 2.

## C5 — Referential integrity

Every `entity_id` appearing in `present`, `referenced`, `actor`, `addressee`,
`state_changes[].entity`, or any `relationship.target_id` resolves to an entry in
`canonical_entities`.

*Negative:* a dangling `actor`; an alias that survived canonicalization unmerged; a
`target_id` pointing at an id that only existed pre-merge.

## C6 — Non-genericity

A model that has stopped reading emits the same sentence everywhere. For `context_before`,
`context_after`, and `style`, compute pairwise similarity across all 224 scenes and fail if
the median exceeds a threshold, or if any exact string repeats more than twice.

Also flags the known degenerate openings ("In this scene…", "The story continues…").

*Negative:* replace all `style` fields with one string; replace half with two alternating
strings.

## C7 — The canary

One window per run is dispatched with an **empty target span** and a scene list naming scenes
that are not in it. Anything it emits is recall from the full-screenplay prefix, not
extraction from the target.

This is the only check in the system that detects failure **by construction** rather than by
comparison, and in the storytree project it fired on its first execution. It is also the check
most likely to catch the specific risk this pipeline runs: that giving every agent the whole
screenplay lets it answer from the whole screenplay while appearing to read its window.

*Negative:* the canary is its own negative case — it must produce an empty KU list, and a run
where it produces content fails.

---

## The rule these are written against

Numbers 1 and 3 from the storytree record apply directly and were nearly violated in drafting
this file:

1. **Never grade against a list your own apparatus generated.** C1's scene list comes from the
   scene map, which is *not* produced by this pipeline and whose 224 anchors were
   independently verified against the source before any of this was designed. C3's numbers and
   proper nouns are extracted from the **source text** by a separate deterministic pass, never
   from the KUs.
2. **Validate the metric before trusting the result.** Each check's implementation and the
   schema it checks are reviewed together, and the negative cases exist so that "it passed" is
   evidence rather than absence of evidence.
3. **Presence is not integrity.** No check accepts "field is non-empty" as a pass. C6 exists
   precisely because C1-style presence checks would wave through 224 identical sentences.
