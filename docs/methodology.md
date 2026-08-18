# Method and architecture

## Knowledge Units

A Knowledge Unit is a local structured representation of a source excerpt. It records entities,
typed attributes, and directed relationships while asking the extraction model to omit expressive
wording. Each unit also carries chunk offsets, a SHA-256 digest, and a 16-value token MinHash. The
source excerpt itself is deliberately absent from the JSON artifact.
The abstract/opening context used for disambiguation is also omitted; only its SHA-256 is retained.

The new schema separates a human-readable entity `name` from its stable `entity_id`. This keeps
scientific notation readable while making aliases and cross-chunk references explicit.

## Two extraction modes

| Property | Sequential baseline | Parallel pipeline |
|---|---|---|
| Purpose | Reproduce the paper method | High-throughput document conversion |
| Default target | 200 words | 500 words |
| Chunk dependency | Previous 10 KUs | None during extraction |
| Source context | Target plus title/abstract | Target, 1,000 words before/after, title/abstract |
| Scheduling | One generation after another | One vLLM batch / continuously batched HTTP requests |
| Naming consistency | Encouraged in each next prompt | Reconciled once across the document |

```text
source document
      │
      ├── sequential ──► chunk 0 ─► chunk 1 + prior KUs ─► ...
      │
      └── parallel ────► [independent contextualized chunks] ─► batch generation
                                                              │
                                      all entity names ◄──────┘
                                              │
                                  document-level alias resolver
                                              │
                                 canonical IDs + rewritten links
```

## Parallel context boundaries

The prompt labels three regions: `CONTEXT BEFORE`, `TARGET TEXT`, and `CONTEXT AFTER`. Neighboring
text is available only to resolve local references; a fact is eligible for the output only when it
is asserted in the target. This is important because otherwise overlapping windows would duplicate
facts and make provenance ambiguous.

## Entity reconciliation

After all independent KUs exist, a single call receives the title, abstract, KU summaries, and
entity names/types/relation predicates grouped by chunk. It returns groups containing a
`canonical_name`, lower-snake-case `canonical_id`, and every alias. The deterministic
postprocessor:

1. validates and normalizes proposed IDs;
2. retains every source name even if the model omits it;
3. applies canonical IDs to entities;
4. rewrites relationship targets only when they exactly match a known alias;
5. rejects high-risk cross-category and generic-versus-specific scope merges; and
6. leaves literals and uncertain references untouched.

The resolver is intentionally conservative: merging related but distinct methods is usually more
damaging than retaining two aliases.

## Relation and naming conventions

- Entity display names use their most complete, conventional form in the document.
- Entity IDs use lower snake case and remain separate from display names.
- Predicates are concise lower-snake-case verbs (`proposed_by`, `measured_at`, `causes`).
- Quantities retain their units and qualifiers.
- Claims retain uncertainty (`suggests`, `associated_with`) rather than being strengthened.
- Attributes describe one entity; relationships connect an entity to another entity or literal.
- The extractor may paraphrase but must not supply background knowledge absent from the target.

## Relationship to the historical implementation

The original scripts emitted tagged Python-dictionary-like text, added a generated style analysis,
stored source MinHashes, and passed the preceding ten segments to the next call. The implementation
here retains the key experimental behavior but uses strict JSON, typed artifacts, deterministic
chunk metadata, atomic writes, retries, and interchangeable inference backends. Style analysis is
not generated because it is unnecessary for factual extraction and creates extra expressive data.
