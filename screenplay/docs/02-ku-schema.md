# The scene Knowledge Unit

Schema version `screenplay-1.0`. Extends the base Alexandria KU rather than replacing it:
`entities`, `attributes`, `relationships`, and `SourceReference` keep their existing shapes so
`canonicalize.py`, `fingerprints.py`, and `experiments/overlap.py` work unmodified.

## Document header

The artifact is self-describing: a reader who receives `ku_chain.json` alone can cite it,
locate the source, and know what was run. No consumer should ever need a second file to
interpret it.

```jsonc
{
  "schema_version": "screenplay-1.0",
  "document": {
    "title": "The Matrix",
    "work_type": "screenplay",
    "credited_as": "Larry and Andy Wachowski",   // the credit line as printed on the document
    "draft": "Numbered Shooting Script",
    "draft_date": "1998-03-29",
    "language": "en",
    "source_url": "<public URL of the script>",  // a link, never the text
    "source_sha256": "<digest of the normalized text>",
    "source_words": 24002,
    "scene_count": 224,
    "retrieved": "2026-08-19",
    "rights_note": "Structure derived under a text-and-data-mining exemption. No source text is redistributed."
  },
  "extraction": {
    "model": "qwen3.8-27b",
    "pipeline": "screenplay/two-stage",
    "windowing": "scenes",
    "pages_target": 6, "pages_context": 5, "words_per_page": 185,
    "stage1_windows": 22, "stage2_seams": 21,
    "run_id": "...", "completed": "..."
  },
  "canonical_entities": [ /* base Alexandria CanonicalEntity */ ],
  "knowledge_units": [ /* below */ ]
}
```

`credited_as` records the credit exactly as printed on the 1998 document, which is what
bibliographic provenance requires; it is a property of the artifact, not a claim about the
authors today.

## The unit

One per scene, in source order.

```jsonc
{
  "scene_id": "sc-042",
  "scene_index": 42,
  "window_index": 4,

  "heading": {
    "raw": "INT. NEBUCHADNEZZAR - CORE - NIGHT",
    "kind": "INT",                    // INT | EXT | INT/EXT | PRE
    "location": "NEBUCHADNEZZAR - CORE",
    "time_of_day": "NIGHT"
  },

  "preceded_by": "sc-041",
  "followed_by": "sc-043",

  "context_before": "Specific, non-generic: the state of the story as this scene opens — what
                     is unresolved, who believes what, what was just decided.",
  "context_after":  "What this scene sets up or makes possible.",
  "style":          "Specific, non-generic: how this scene is told — its staging, pacing,
                     point of view, and what the camera withholds.",

  "present": ["neo", "morpheus"],     // canonical entity_ids physically present
  "referenced": ["the_oracle"],       // discussed but not present

  "beats": [ /* the temporal spine — see below */ ],

  "entities": [ /* base Alexandria Entity: name, entity_id, type, attributes, relationships */ ],

  "source": {                          // by reference only; no text
    "scene_id": "sc-042",
    "start_char": 61204, "end_char": 62119,
    "word_count": 168,
    "sha256": "...",
    "sentence_minhash": [ /* 16 values */ ]
  },

  "extraction_warnings": []
}
```

`context_before`, `context_after`, and `style` are required to be **non-generic**: a check
rejects a run in which these fields are near-duplicates across scenes, because a model that
has stopped reading emits the same sentence everywhere. See [03-checks.md](03-checks.md).

## Beats — the temporal spine

A beat is one indivisible thing that happens. Beats carry the ordering the base schema has no
place for.

```jsonc
{
  "order": 3,                          // 1-based, contiguous within the scene
  "type": "speech",                    // action | speech | revelation | state_change | movement | perception
  "actor": "morpheus",
  "addressee": "neo",                  // null for non-directed beats
  "content": "Morpheus told Neo that the choice of pill was his alone to make.",
  "facts": {
    "quantities": [], "dates": [], "proper_nouns": ["Morpheus", "Neo"], "locations": []
  },
  "state_changes": [
    { "entity": "neo", "field": "knowledge.nature_of_reality", "from": "unaware", "to": "informed_of_choice" }
  ],
  "causes": [2],                       // beat orders in THIS scene that this beat follows from
  "certainty": "stated"                // stated | implied_by_action | ambiguous
}
```

### Global order

`(scene_index, beat_order)` is a **strict total order over the whole film**. Sorting every beat
in the chain by that pair reconstructs the presentation sequence exactly. This is the
requirement that the temporal order be recoverable, made mechanical and therefore checkable —
not a property the prose is trusted to convey.

Presentation order is not story order: the script's opening sequence is a phone call heard
before the events it concerns. `certainty` and `causes` capture local causality; the pair
above captures *the order in which the audience receives information*, which is what the brief
asks for.

### Indirect speech

Dialogue is rendered in third person, never quoted:

- not: `Morpheus says, "<line>"`
- but: `Morpheus told Neo that the choice was his alone.`

The instruction is necessary and not sufficient. Instructions repair local fields; the
verbatim-overlap gate in [03-checks.md](03-checks.md) is what actually enforces this, and it
is a hard gate — a run that fails it does not produce an artifact.

### What must survive verbatim

Paraphrase applies to *expression*. It must not touch:

- **numbers and quantities** — phone digits, dates, times, floor numbers, counts
- **proper nouns** — character, ship, program, and place names
- **locations** — as named in the slugline

These are checked mechanically against the source scene, not left to the prompt. A paraphrase
that drifts a number has destroyed the fact it existed to preserve.

## Relationship to storytree

The `state_changes` field is deliberately the same shape as the storytree entity-patch model,
so a scene KU can later be replayed as a JSON patch against an entity profile. That
integration is out of scope here; the field is shaped for it now so it does not need
reworking later.

This layer stays strictly at **what the script states**. Hypotheses, beliefs, and inferred
mental states are the next layer up and are explicitly not extracted here — `certainty`
exists to keep that boundary visible.
