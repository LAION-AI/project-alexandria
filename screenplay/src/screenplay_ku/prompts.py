"""Prompts for scene KU extraction and seam verification.

Prompt region order is load-bearing. Regions that are identical across every window come
first, so vLLM's prefix cache holds them: the full screenplay prefills once per endpoint
(15.8 s measured) and is then free (0.8 s). Per-window regions follow.

The few-shot examples use an invented scene rather than one from the source, so the model
is shown the *shape* of the task without being primed with the text it is about to read.
"""

from __future__ import annotations

import json
from typing import Dict, List, Sequence

from .scenes import Scene
from .windows import Window


SYSTEM_PROMPT = (
    "You build structured factual records from narrative source text. "
    "You record what a passage states — who is present, what occurs, in what order, and "
    "what changes as a result — as data, not as retold prose. "
    "You never add anything the passage does not state, and you never reuse its wording. "
    "You return only valid JSON."
)


_TASK = """\
Build one Knowledge Unit per scene listed in SCENES TO EMIT.

A Knowledge Unit is a STRUCTURED RECORD, not a retelling. You are populating fields of a
database: which entities are present, what discrete events occur, in what order, and what
state each event changes. The output should read like a case file, not like the scene.

## Beats — the temporal spine

Decompose each scene into ordered beats. A beat is one indivisible thing that happens.
Number them from 1 in the order the audience receives them. Sorting every beat in the film
by (scene, order) must reconstruct the sequence in which information was presented, so the
ordering is the primary thing you are recording.

`causes` lists earlier beat numbers **in the same scene** that a beat follows from. Leave
it empty when nothing in the scene causes it.

`certainty` separates what is stated from what you inferred:
  - `stated`            — the passage says it outright
  - `implied_by_action` — an action makes it unambiguous, though it is not said
  - `ambiguous`         — the passage deliberately withholds it
Use `stated` only when it truly is. This boundary is the point of the field.

## Recording speech

Record the PROPOSITIONAL CONTENT of what is communicated — who told what to whom, what was
asked, what was refused, what was revealed. Never the words used.

  wrong:  Ana said, "The shipment never reached the depot."
  wrong:  Ana said that the shipment never reached the depot.   <- still the original wording
  right:  Ana informed Boris that the delivery had failed to arrive at its destination.

The second is wrong because it keeps the source's phrasing while adding a frame. You want
the fact that was conveyed, expressed in your own vocabulary.

**Never reuse a distinctive run of words from the source.** If a phrase from the passage
appears in your output, you have recorded the expression instead of the fact. Restate it.

## What must survive exactly

Paraphrase applies to expression only. These must be preserved without alteration, and
listed in the beat's `facts`:

  - numbers and quantities: digits, counts, times, floor and room numbers, durations
  - dates
  - proper nouns: names of people, places, vehicles, organizations
  - locations as named in the scene heading

A restatement that drifts a number has destroyed the fact it existed to preserve.

## Contextualization — specific, never generic

  - `context_before`: the state of the story as this scene opens. What is unresolved, who
    believes what, what was just decided. Name the specific things.
  - `context_after`: what this scene makes possible or sets up.
  - `style`: how this scene is told — its staging, whose viewpoint it takes, what it
    withholds from the audience.

"In this scene, the story continues" is a failure. Every one of these must be true of THIS
scene and false of every other scene in the film.

## Entities

`entity_id` is lowercase snake_case and stable across the whole film (`agent_smith`, not
`smith_1`). Use ids from the ENTITY IDS ALREADY IN USE list whenever the same entity
appears; document-level aliases are reconciled later, but consistency here costs nothing.
`present` is who is physically in the scene, `referenced` is who is discussed but absent.
"""


_EXAMPLE = """\
EXAMPLE — for an invented scene, to show the shape only

SCENE sc-114  INT. HARBOUR OFFICE - NIGHT
(the passage: Ana confronts Boris about a missing container; he admits he re-routed it to
Pier 9 on the 14th because he was being watched; she takes his keycard and leaves)

{"knowledge_units":[{
 "scene_id":"sc-114",
 "context_before":"Ana has spent two days tracing a container that vanished from the manifest, and Boris is the last person who signed for it. She does not yet know he acted deliberately.",
 "context_after":"Ana now holds Boris's keycard and a destination, which is what lets her reach Pier 9 before the buyers do. Boris is left without access to the building.",
 "style":"A two-hander in a single cramped room, staged so the audience learns Boris is lying several beats before Ana does. The camera stays on his hands rather than his face.",
 "present":["ana_reyes","boris_kane"],
 "referenced":["the_buyers"],
 "beats":[
  {"order":1,"type":"action","actor":"ana_reyes","addressee":"boris_kane",
   "content":"Ana confronted Boris in the harbour office over a container absent from the manifest.",
   "facts":{"quantities":[],"dates":[],"proper_nouns":["Ana","Boris"],"locations":["HARBOUR OFFICE"]},
   "state_changes":[],"causes":[],"certainty":"stated"},
  {"order":2,"type":"speech","actor":"boris_kane","addressee":"ana_reyes",
   "content":"Boris initially denied any knowledge of where the container had gone.",
   "facts":{"quantities":[],"dates":[],"proper_nouns":["Boris"],"locations":[]},
   "state_changes":[],"causes":[1],"certainty":"stated"},
  {"order":3,"type":"revelation","actor":"boris_kane","addressee":"ana_reyes",
   "content":"Boris conceded that he had personally diverted the container to Pier 9 on the 14th, explaining that he believed he was under surveillance at the time.",
   "facts":{"quantities":[],"dates":["the 14th"],"proper_nouns":["Boris","Pier 9"],"locations":["Pier 9"]},
   "state_changes":[{"entity":"ana_reyes","field":"knowledge.container_location","from":"unknown","to":"pier_9"},
                    {"entity":"boris_kane","field":"status.complicity","from":"suspected","to":"admitted"}],
   "causes":[2],"certainty":"stated"},
  {"order":4,"type":"action","actor":"ana_reyes","addressee":"boris_kane",
   "content":"Ana took Boris's keycard from him and departed, leaving him in the office.",
   "facts":{"quantities":[],"dates":[],"proper_nouns":["Ana","Boris"],"locations":[]},
   "state_changes":[{"entity":"boris_kane","field":"possessions.keycard","from":"held","to":"lost"},
                    {"entity":"ana_reyes","field":"possessions.keycard","from":"none","to":"held"}],
   "causes":[3],"certainty":"stated"}],
 "entities":[
  {"name":"Ana Reyes","entity_id":"ana_reyes","type":"person","aliases":["Ana"],
   "attributes":{"role":"investigator","objective":"recover the container"},
   "relationships":[{"predicate":"confronts","target":"Boris Kane","target_id":"boris_kane"}]},
  {"name":"Boris Kane","entity_id":"boris_kane","type":"person","aliases":["Boris"],
   "attributes":{"role":"harbour clerk","motive":"believed he was being watched"},
   "relationships":[{"predicate":"diverted","target":"the container","target_id":"the_container"}]},
  {"name":"Pier 9","entity_id":"pier_9","type":"place","aliases":[],
   "attributes":{"significance":"where the container was re-routed"},"relationships":[]}]}]}

Note what the beats do NOT contain: no line of dialogue, and no phrase carried over from
the passage. Note what they preserve exactly: Pier 9, and the 14th.
"""


def extraction_prompt(
    window: Window,
    source: str,
    scenes_all: Sequence[Scene],
    document: Dict[str, object],
    scene_index_table: str,
    known_entity_ids: Sequence[str] = (),
) -> str:
    """Build a stage-1 prompt.

    Regions 1-3 are byte-identical across every window in the run and therefore cached.
    Region 6 repeats the target span already present inside region 1: showing the model the
    exact span it must produce output for concentrates attention on it.
    """
    scene_rows = [
        {
            "scene_id": scene.scene_id,
            "heading": scene.heading_raw,
            "kind": scene.kind,
            "location": scene.location,
            "time_of_day": scene.time_of_day,
            "words": scene.word_count,
        }
        for scene in window.scenes
    ]
    return """\
=== FULL SCREENPLAY (reference; read-only) ===
{full}

=== DOCUMENT ===
{document}

=== SCENE INDEX (authoritative ids — never invent one) ===
{index}

=== CONTEXT BEFORE (read-only; may disambiguate, may not be extracted from) ===
{before}

=== CONTEXT AFTER (read-only; may disambiguate, may not be extracted from) ===
{after}

=== TARGET SPAN (extract from this, and only this) ===
{target}

=== SCENES TO EMIT ({count}) ===
{scenes}

=== ENTITY IDS ALREADY IN USE ===
{known}

{task}

{example}

Return one JSON object with a "knowledge_units" array holding exactly {count} units, one
for each scene above, in that order. Every fact must be asserted in TARGET SPAN.
""".format(
        full=source,
        document=json.dumps(document, ensure_ascii=False, indent=1),
        index=scene_index_table,
        before=window.before_text(source) or "[start of screenplay]",
        after=window.after_text(source) or "[end of screenplay]",
        target=window.target_text(source) or "[EMPTY]",
        count=len(window.scenes),
        scenes=json.dumps(scene_rows, ensure_ascii=False, indent=1),
        known=", ".join(sorted(known_entity_ids)) or "[none yet]",
        task=_TASK,
        example=_EXAMPLE,
    )


def canary_prompt(*args, **kwargs) -> str:
    """Identical to an extraction prompt, but the target span is empty.

    The scene list still names real scenes, and the full screenplay is still present, so
    the model *could* answer from the surrounding context. Anything it emits is recall.
    """
    return extraction_prompt(*args, **kwargs) + """

=== NOTE ===
TARGET SPAN is empty. No scene content was supplied for extraction. The correct response is
an empty "knowledge_units" array. Do not reconstruct these scenes from the reference
screenplay or from context — the reference is present for disambiguation only, and content
recovered from it would not be an extraction of the target.
"""


def seam_prompt(left_units: List[Dict], right_units: List[Dict], left_ids, right_ids) -> str:
    """Stage 2. Deliberately excludes the source text.

    A seam agent that could re-read the screenplay would just re-extract, and its agreement
    with stage 1 would measure nothing. Restricted to the units, it can only find defects
    that are visible in the units themselves — which is the property the published artifact
    needs, because a downstream reader will also only have the units.
    """
    return """\
You are verifying the join between two independently extracted sections of one screenplay.
You do NOT have the screenplay. You have only the Knowledge Units either side of the seam,
and you must judge them as a reader with only these units would.

Find, and report only, defects visible in the units themselves:

1. `alias_merges` — the same entity given different ids on either side (`the_oracle` vs
   `oracle`). Merge only when the units make identity clear. Do not merge distinct entities
   that share words; when uncertain, leave them separate.
2. `context_fixes` — a `context_before` on the right that misdescribes the state the left
   side actually leaves the story in, or a `context_after` on the left that misdescribes
   what the right side actually does. Replace only when it is wrong, not merely terse.
3. `duplicate_scene_ids` — the same scene emitted on both sides.
4. `continuity_notes` — a beat's causes or state changes that contradict the other side.

Report nothing you cannot justify from these units. Empty arrays are the correct answer for
a clean seam, and are a useful result. Do not invent work.

=== LEFT SECTION, final scenes ({left_ids}) ===
{left}

=== RIGHT SECTION, opening scenes ({right_ids}) ===
{right}
""".format(
        left=json.dumps(left_units, ensure_ascii=False, indent=1),
        right=json.dumps(right_units, ensure_ascii=False, indent=1),
        left_ids=", ".join(left_ids),
        right_ids=", ".join(right_ids),
    )
