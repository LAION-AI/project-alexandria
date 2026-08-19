# CogniTino, adapted for the screenplay scene layer

## Source

> **CogniTino: Bridging Implicit and Explicit Knowledge through Semantic Knowledge Graphs.**
> Whitepaper, 22 pp. Copy held in this repository at
> [`reference/CogniTino-whitepaper.pdf`](../reference/CogniTino-whitepaper.pdf).

Everything in `src/screenplay_ku/cognitino/` derives from that document. This file records
what was taken unchanged, what was narrowed, and what was added — so a reader can tell the
paper's contribution from ours.

---

## Why the two projects compose at all

The join is cleaner than it had any right to be, and the reason is worth stating because it
was not designed for.

CogniTino divides Cognitive Objects into **Perception Objects** — *"data as it is directly
perceived, without added interpretation or inference"* — and **Abstraction Objects** —
*"higher-level interpretations, inferences, or mental constructs derived from processing and
analyzing Perception Objects."*

The Alexandria screenplay layer was built independently, to a constraint that had nothing to
do with CogniTino: record only what the screenplay *states*, and mark the boundary with a
`certainty` field whose `stated` value means the passage says it outright. That constraint
exists for a copyright reason, not a cognitive one.

But it produces exactly CogniTino's Perception Object. The KU beat is a direct observation
with contextual summaries either side and a source reference — which is the paper's Text
Chunk Cognitive Object, feature for feature. So:

| CogniTino | Here |
|---|---|
| Perception Object (Text Chunk) | a **KU beat**, addressed `sc-042#3` |
| Contextual Summaries (preceding/current/succeeding) | `context_before` / `content` / `context_after` |
| Source Reference | `source.start_char`, `sha256`, MinHash |
| Abstraction Object | the objects defined in `cognitino/schema.py` |

Nothing was retrofitted. The Alexandria layer stops precisely where CogniTino's abstraction
layer is defined to begin.

---

## Modules: what each one became

The paper specifies five modules. Four are implemented; the first was already done.

| # | Paper module | Here | Parallelism |
|---|---|---|---|
| 1 | Chunking and Contextualization | **already built** — the KU chain is the Perception layer | 30 windows |
| 2 | Abstraction Object Generation | `pipeline.draft_all` | 45 windows in parallel |
| 3 | Abstraction Object Researcher | `pipeline.research_all` | 45 windows × 2 rounds |
| 4 | Editor | `pipeline.canonicalize` | **sequential by necessity** |
| 5 | Semantic Connection | `pipeline.merge_tree` | hierarchical, parallel per level |

### Taken unchanged

- **The Perception/Abstraction split**, and the reason given for it: separation of data from
  interpretation, traceability, and flexibility under reinterpretation.
- **The traceability principle.** *"Every claim, belief, or inference should be linked to
  supporting evidence."* This is the single most important thing the paper contributes here.
- **The Abstraction Object taxonomy** — Hypothesis, Relationship, Entity, Concept, Event,
  Process — and their attribute sets: statement, supporting evidence, contradicting evidence,
  validation metrics, assumptions, limitations.
- **The Researcher's three steps**: evidence retrieval, contextual analysis (relevance,
  peripheral awareness, temporal consistency), and updating with supporting *and*
  contradicting evidence.
- **The four connection families** from the Semantic Connection Module — causal,
  hierarchical, temporal, thematic — explicitly *"not merely similarity-based associations"*.
- **The Editor's job**: synthesis, conflict resolution, and standardisation into consistent
  machine-readable form.

### Added

Three object types the paper does not enumerate, because its worked example (Bob's romantic
interest in Alice) needs only first-order inference, while dramatic analysis does not:

| Added type | Why |
|---|---|
| `mental_state` | What one character perceives, wants, fears, feels *now*, including the physiological and the spatial. The paper's Hypothesis type can express this but flattens it into a general claim about the world. |
| `theory_of_mind` | What A believes about B, recursively. `subject` is the believer, `about` is whom the belief concerns. This is the layer a screenwriter or actor actually works in, and the paper has no slot for it. |
| `authorial_intent` | What the writer is doing with this scene — setting up, withholding, mirroring, paying off. A narrative-specific object with no analogue in a general knowledge system. |
| `consequence` | What the scene makes possible or forecloses elsewhere. Distinct from the paper's Event type, which records what happened rather than what it licenses. |

Two enforcement mechanisms the paper states as principles but does not operationalise:

- **`grounded_in` is schema-bound to real beat ids.** The paper asserts traceability; here
  the JSON schema binds the field to an `enum` of beat references that actually exist in the
  window, and a validator rejects the response otherwise. This is the project's standing
  finding applied again: *instructions repair local fields, structure repairs global
  properties.* An ungrounded hypothesis about a mind is indistinguishable from a
  hallucination about a mind, so grounding cannot be advisory.
- **`falsifier` is mandatory.** Not in the paper. Every object must name the concrete thing
  that would defeat it. An inference nobody can imagine disconfirming is not a hypothesis,
  and requiring the field forces the distinction at write time rather than at review time.

### Narrowed

- **Modalities.** The paper handles image, audio and video Perception Objects. Only text
  exists here, so those types are dropped rather than stubbed.
- **Confidence.** The paper uses continuous scores (*"Confidence Level: 0.85"*). Here it is
  an ordinal band — `speculative` / `plausible` / `probable` / `near-certain`. A model
  emitting two decimal places about a fictional character's inner life is reporting precision
  it does not have, and the ordinal form makes the calibration check meaningful.
- **External tools.** The Researcher module's *"Databases: Accesses internal and external
  databases"* is deliberately not implemented. The evidence base is the screenplay and the
  Perception layer, and nothing else. An agent permitted to consult outside sources about
  *The Matrix* would import the film's reception rather than read the script, and every
  inference would be contaminated by material the pipeline cannot audit.

### Changed

**Module order: 5 before 4, not 4 before 5.** The paper runs Editor then Semantic Connection.
Here connection runs first, because merging *creates* objects (arcs) and supersedes others,
so canonicalising first would standardise names the merge then discards, and the editor would
have to run twice.

**The Semantic Connection Module becomes a hierarchical merge tree.** The paper describes
connection as a global operation over the graph. At 225 scenes that does not fit a context
window, so connection runs pairwise up a tree: windows of 5 scenes merge to 10, then 20, then
40. Level 1 sees full objects; higher levels see summaries only. The span each agent reasons
over doubles at every level while its resolution drops — a widening attention filter, which
is the only form of "global" available at this scale. The tree is capped rather than run to a
single root, and the remaining global consistency is left to the Editor.

**The Editor is sequential, and that is load-bearing.** Everything else here is parallelised;
this cannot be. The pass carries a running canonical map so batch *N* sees the decisions of
batches 1..*N*−1 and reuses their canonical forms. Parallel batches would each invent their
own, and the pass would produce exactly the inconsistency it exists to remove.

---

## Checks

The paper does not specify verification, and the Perception layer's checks do not transfer: a
Knowledge Unit can be wrong by contradicting the source, but an Abstraction Object is
*supposed* to say things the source does not. Seven checks were written for this layer, each
with a negative case in `tests/test_cognitino_checks.py`.

| | Catches |
|---|---|
| **G1** grounding | an inference with no pointer, or a pointer to a beat that does not exist |
| **G2** not-restatement | an "abstraction" that paraphrases the beat it cites — **the failure mode invisible to every other check**: grounded, well-formed, on-topic, and empty |
| **G3** calibration | uniform confidence, and `near-certain` second-order belief |
| **G4** contradiction sought | a researcher that only ever confirms |
| **G5** connectivity | objects with no links — a list rather than a graph |
| **G6** scene coverage | scenes the abstraction layer skipped |
| **G7** theory-of-mind present | the layer's reason to exist, missing |

G2 is the one that matters most and is the one the paper's framing would not have prompted.
The paper's concern is that abstractions be *grounded*; the risk in practice is that they be
grounded and vacuous.

As everywhere else in this repository, `run_ao_checks` reports `unverified` rather than
`pass` for any check whose negative case did not run in the same execution.

---

## Attribution

The architecture, the Perception/Abstraction distinction, the traceability principle, the
object taxonomy, the module decomposition, and the researcher's supporting/contradicting
evidence pattern are CogniTino's. The narrative-specific object types, the enforcement of
grounding at the schema level, the falsifier requirement, the hierarchical merge tree, and
the seven checks are this repository's additions, and are described above so the boundary is
legible.
