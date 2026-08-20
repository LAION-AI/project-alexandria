"""Prompts for the four CogniTino modules, specialised to the screenplay scene layer.

Region order follows the Alexandria extractor: the full screenplay comes first and is
byte-identical across every window, so it prefills once per endpoint and is then ~free. The
window's own scenes and Knowledge Units are then shown a **second** time, which is the same
deliberate repetition the extraction stage uses to concentrate attention on the span the
agent is accountable for.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence


SYSTEM = (
    "You infer the implicit content of dramatic scenes — what characters perceive, believe, "
    "want, and feel, what they believe about each other, and what the writer is doing — and "
    "you record it as evidence-linked structured objects. Every inference you make must point "
    "to the specific observation that licenses it. You never assert as observed what you have "
    "concluded. You return only valid JSON."
)


_DRAFT_TASK = """\
You are building the ABSTRACTION layer over scenes you own.

The observable layer already exists. The Knowledge Units below record what the screenplay
*states*: who was present, what happened, in what order, what changed. Those are
observations, and you must not repeat them. Your job is the layer above — everything a
screenwriter, director, or actor would work out from those observations but which the script
never says outright.

## What to produce

Objects of these types. Produce whatever mix the scenes actually support; do not pad.

  `mental_state`     — what one character perceives, thinks, wants, fears, or feels *now*.
                       Include the physiological when it bears on the mind: exhaustion,
                       pain, cold, adrenaline, hunger. Include their relation to the place
                       and the moment: whether they feel safe here, whether they are rushed.
  `theory_of_mind`   — what A believes about B. Nest it where the scene supports nesting:
                       what A believes B believes about A; what A believes B believes about C.
                       Set `subject` to the believer and `about` to whom the belief concerns.
  `hypothesis`       — an inferred fact about the world of the story that is not stated.
  `relationship`     — the state of a bond between two characters: trust, debt, suspicion,
                       obligation, and how this scene moves it.
  `entity_trait`     — a disposition of a character or object that this scene reveals but
                       does not state.
  `concept`          — an abstraction this scene instantiates, named so other scenes can
                       point at it.
  `process`          — a recurring pattern or procedure the scene displays.
  `authorial_intent` — what the writer appears to be doing here: what is being set up,
                       withheld, mirrored, or paid off, and why here rather than elsewhere.
  `consequence`      — what this scene makes possible or forecloses later.

## The rules that make this useful rather than decorative

**1. Ground everything.** `grounded_in` must list the beat references that license the
inference, using the exact `sc-XXX#N` ids given below. An ungrounded claim about a mind is
indistinguishable from an invention, and will be rejected.

**2. Infer, do not restate.** If the Knowledge Unit already says it, it is an observation and
does not belong here. "Neo asked what the Matrix is" is an observation. "Neo is willing to
appear ignorant in front of a man he wants to impress, which means he wants the answer more
than the standing" is an abstraction.

**3. Say what would prove you wrong.** `falsifier` must name the concrete thing that, if it
appeared in the screenplay, would defeat this object. If you cannot name one, the statement
is too vague to be worth recording.

**4. Calibrate honestly.** `speculative` / `plausible` / `probable` / `near-certain`. Most
theory-of-mind at second order is `plausible` at best. Reserve `near-certain` for inferences
a competent reader could not reasonably decline. Uniform high confidence is a failure.

**5. Make it this scene.** An object that would fit any thriller is worthless. Name the
specific pressures, the specific stakes, the specific people.

**6. Never quote the screenplay.** Not in `statement`, and especially not in `reasoning` —
a field asking why you believe something invites pasting the line that convinced you, and
that is how source text escapes into an artifact built to contain none. Describe what was
said or shown in your own words. If you need to point at a line, cite its beat reference
instead; that is what the reference is for.

**7. State assumptions.** `assumptions` lists what must hold for the inference to stand —
including cultural or genre assumptions you are importing.

---

## What a good scene record is judged on

These are the qualities that make this record useful to the layer above. Give the first two
the most attention; they are the ones most often done badly.

**Emotional intelligence — the hardest and the most valuable.** Is what people want, fear
and *conceal* read plausibly? The bar is not naming an emotion. Anyone can write "Neo is
afraid". The bar is reading a concealed motive or an unspoken pressure from behaviour, and
reading it briefly:

  weak   — "Trinity is tense during the escape."
  strong — "Trinity keeps working the trace after the line is compromised, which means she
            has decided the information is worth more than her own extraction, and she has
            not told anyone she made that decision."

For a scene with people in it, an empty inner life is a failure. For a scene with no people
in it, do not manufacture one. Where a character conceals something, say what they are
concealing *and from whom*.

**Change reality — are the changes real, and do they matter?** A change is a genuine state
transition the story uses later, not a restatement of the action.

  not a change — `door: closed -> open` (that is the action, restated)
  not a change — `before: "not explicitly stated"` (an unstated before is not a before)
  not a change — a state this scene does not touch
  a change    — `neo.trust_in_morpheus: provisional -> staked` — different after, and later
                 scenes depend on which side of it we are on

If you notice a real change the observation layer failed to record, put it in
`perception_patch` rather than smuggling it in as an inference.

**Completeness.** Could a reader rebuild what this scene *does for the story* from your
record alone? Every load-bearing change and participant should be present. A missing
participant or a missing turn is a hole.

**Specificity.** Could your record be pasted onto a different scene and still read as true?
If so it is worthless. Name the particular pressures, objects, and people of this scene.
Never write filler like "idle or observing".

**Fidelity.** Everything you assert must be true of *this* scene. Do not import a fact from a
neighbouring scene and present it as happening here. If your evidence comes from elsewhere,
say so in `reasoning` and keep `grounded_in` honest.

**Calibration — length and confidence proportionate to the scene.** This is where records
most often fail, in both directions.

  - A twelve-word establishing shot does not support four inferences about anyone's psyche.
    One object, or none, is the right answer. Writing a long analysis of a tiny scene is a
    failure even if every sentence is defensible.
  - A five-hundred-word scene built on a confrontation is under-served by two objects.
  - Confidence must track evidence. `near-certain` is for what a competent reader could not
    reasonably decline. Second-order belief — what A thinks B thinks — is rarely above
    `plausible`. If most of your objects carry the same confidence band, you are not
    calibrating, you are decorating.
"""


_PATCH_TASK = """\
## Repairing the observation layer

You are reading the scene text and the Knowledge Units together, which makes you the first
reader able to see where the extraction missed something. Use `perception_patch` to say so.

  `missing_state_changes` — a transition the script states outright that no beat records.
                            `stated_where` must say where in the scene text it is stated.
  `missing_beats`         — something that happens which no beat records at all.
                            `after_order` places it; use 0 to put it first.
  `wrong_state_changes`   — a recorded change that is a no-op, restates the action, or
                            changes a state this scene never touches.

**The observation layer records only what the script states.** A patch that adds an inference
would break that guarantee and will be rejected. If it needs interpreting, it is an
abstraction object, not a patch. Leave the arrays empty when the extraction is sound; an
empty patch is a normal and correct answer.
"""


_DRAFT_EXAMPLE = """\
EXAMPLE — invented scene, to show shape only

Observations available: `sc-114#1` Ana confronted Boris over a missing container;
`sc-114#2` Boris denied knowing where it went; `sc-114#3` Boris then admitted diverting it,
saying he believed he was being watched; `sc-114#4` Ana took his keycard and left.

{"abstraction_objects":[
 {"ao_id":"ao-114-01","type":"mental_state","scene_id":"sc-114",
  "subject":"boris_kane","about":null,
  "statement":"Boris arrives already resigned to being caught, and his denial is a formality he abandons at the first pressure rather than a plan he believes in.",
  "grounded_in":["sc-114#2","sc-114#3"],
  "reasoning":"A denial dropped after a single challenge, followed by an unprompted account of motive, reads as someone who had rehearsed being found out rather than someone improvising a defence.",
  "confidence":"probable",
  "assumptions":["A person committed to a lie sustains it past the first challenge."],
  "falsifier":"An earlier scene showing Boris successfully maintaining the same lie under sustained questioning."},
 {"ao_id":"ao-114-02","type":"theory_of_mind","scene_id":"sc-114",
  "subject":"boris_kane","about":"ana_reyes",
  "statement":"Boris believes Ana will treat surveillance as a mitigating explanation rather than an aggravating one, which is why he volunteers it instead of concealing it.",
  "grounded_in":["sc-114#3"],
  "reasoning":"He supplies the motive unasked. Volunteering a reason only makes sense if he expects the listener to weigh it in his favour.",
  "confidence":"plausible",
  "assumptions":["Boris is choosing what to disclose rather than confessing indiscriminately."],
  "falsifier":"Boris continuing to volunteer detail after Ana signals it is worsening his position."},
 {"ao_id":"ao-114-03","type":"authorial_intent","scene_id":"sc-114",
  "subject":"scene","about":null,
  "statement":"The scene withholds who was watching Boris, converting a resolved question about the container into an unresolved one about a third party, and moving the threat off screen.",
  "grounded_in":["sc-114#3","sc-114#4"],
  "reasoning":"The confession answers the question the scene opened with, and the answer introduces an agent never named. Ending on Ana's exit rather than on a reply leaves that agent unexamined.",
  "confidence":"probable",
  "assumptions":["The omission is deliberate rather than an artefact of drafting."],
  "falsifier":"The watcher being identified within the same sequence."},
 {"ao_id":"ao-114-04","type":"relationship","scene_id":"sc-114",
  "subject":"ana_reyes","about":"boris_kane",
  "statement":"The relation moves from colleagues with asymmetric suspicion to one of custody: Ana now holds both the evidence and his means of access, and Boris has nothing left to trade.",
  "grounded_in":["sc-114#1","sc-114#4"],
  "reasoning":"Taking the keycard converts a conversational advantage into a material one and removes his ability to act independently.",
  "confidence":"near-certain",
  "assumptions":["The keycard is his only access."],
  "falsifier":"Boris entering the building later without it."}]}

Note that none of these restate a beat. Each names something the beats imply.
"""


def draft_prompt(source: str, document: Dict[str, Any], scene_text: str,
                 units: Sequence[Dict[str, Any]], beat_refs: Sequence[str],
                 entity_ids: Sequence[str], allow_patch: bool = True) -> str:
    compact = [
        {
            "scene_id": u["scene_id"],
            "heading": (u.get("heading") or {}).get("raw"),
            "present": u.get("present"),
            "context_before": u.get("context_before"),
            "beats": [
                {"ref": "{}#{}".format(u["scene_id"], b.get("order")), "type": b.get("type"),
                 "actor": b.get("actor"), "addressee": b.get("addressee"),
                 "content": b.get("content"),
                 "state_changes": b.get("state_changes")}
                for b in u.get("beats") or []
            ],
        }
        for u in units
    ]
    return """\
=== FULL SCREENPLAY (reference; read-only) ===
{full}

=== DOCUMENT ===
{doc}

=== YOUR SCENES — VERBATIM TEXT (shown again, deliberately) ===
{text}

=== YOUR SCENES — OBSERVATION LAYER (Knowledge Units) ===
{units}

=== VALID BEAT REFERENCES FOR `grounded_in` ===
{refs}

=== ENTITY IDS IN USE ===
{ents}

{task}

{patch_task}

{example}

Return one JSON object with an "abstraction_objects" array covering every scene you own, and
a "perception_patch" object (empty arrays if the observation layer needs no repair).

Depth should follow what the scene supports, not a quota. A scene that turns on one character
misreading another deserves several nested theory-of-mind objects; a twelve-word establishing
shot deserves one or none. **Every scene with a person in it needs at least one reading of
that person's inner life** — silence there is not restraint, it is an omission.
""".format(
        full=source,
        doc=json.dumps(document, ensure_ascii=False, indent=1),
        text=scene_text,
        units=json.dumps(compact, ensure_ascii=False, indent=1),
        refs=", ".join(beat_refs),
        ents=", ".join(sorted(entity_ids)) or "[none]",
        task=_DRAFT_TASK,
        patch_task=_PATCH_TASK if allow_patch else "",
        example=_DRAFT_EXAMPLE,
    )


_RESEARCH_TASK = """\
You are the RESEARCHER for abstraction objects you previously drafted.

Your drafts were written from a single reading. Now test them. For each object, look through
the screenplay and the observation layer for evidence that **supports** it and, more
importantly, evidence that **contradicts** it. A researcher who only finds confirmation has
not researched.

Produce four things:

1. `evidence` — specific beat references that bear on an object, each marked `supports` or
   `contradicts`, with a reason. Search beyond the scenes you own: an inference about a
   character's belief is often confirmed or broken by something they do two scenes later.

2. `links` — relations *between* abstraction objects. This is where the graph forms:
   a mental state `caused_by` a relationship shift; a theory-of-mind object `contradicts`
   another; a consequence `enabled` by a hypothesis; two objects that are `instance_of` the
   same concept. Add every link you can justify.

3. `confidence_updates` — where the evidence moves an object up or down. **Downgrades are the
   valuable output here.** An object that survived a real search for contradiction has earned
   its confidence; one that was never tested has not.

4. `new_objects` — inferences that only became visible once you were looking across scenes:
   patterns, arcs, recurring processes, revised readings. Same grounding rules as before.

Do not restate observations. Do not invent beat references — use only the ids listed.
"""


def research_prompt(source: str, scene_text: str, units: Sequence[Dict[str, Any]],
                    drafts: Sequence[Dict[str, Any]], beat_refs: Sequence[str],
                    entity_ids: Sequence[str], round_index: int) -> str:
    compact_units = [
        {"scene_id": u["scene_id"],
         "beats": [{"ref": "{}#{}".format(u["scene_id"], b.get("order")),
                    "content": b.get("content")} for b in u.get("beats") or []]}
        for u in units
    ]
    compact_drafts = [
        {"ao_id": o["ao_id"], "type": o["type"], "scene_id": o["scene_id"],
         "subject": o.get("subject"), "about": o.get("about"),
         "statement": o["statement"], "confidence": o["confidence"],
         "grounded_in": o["grounded_in"], "falsifier": o.get("falsifier")}
        for o in drafts
    ]
    return """\
=== FULL SCREENPLAY (search this) ===
{full}

=== YOUR SCENES — VERBATIM TEXT ===
{text}

=== OBSERVATION LAYER FOR YOUR SCENES ===
{units}

=== YOUR ABSTRACTION OBJECTS (research round {rnd}) ===
{drafts}

=== VALID BEAT REFERENCES ===
{refs}

=== ENTITY IDS ===
{ents}

{task}
""".format(
        full=source, text=scene_text,
        units=json.dumps(compact_units, ensure_ascii=False),
        drafts=json.dumps(compact_drafts, ensure_ascii=False, indent=1),
        refs=", ".join(beat_refs), ents=", ".join(sorted(entity_ids)),
        rnd=round_index, task=_RESEARCH_TASK,
    )


_MERGE_TASK = """\
You are connecting two adjacent regions of one screenplay's abstraction layer.

Each region was analysed independently, so inferences that belong together are currently
strangers. You do **not** have the screenplay — you have the objects, and you must judge them
as a reader with only these objects would. That restriction is deliberate: re-reading the
source would make you re-derive rather than connect.

Produce:

1. `cross_links` — relations that span the boundary. The valuable ones are causal and
   temporal: a belief formed in the earlier region that a later object depends on, a
   consequence predicted earlier that a later object realises, a theory-of-mind object later
   contradicted by what the character then does.

2. `duplicates` — the same inference stated twice on either side of the boundary. Keep the
   better-grounded one. Merge only genuine duplicates; two objects about the same character
   at different moments are not duplicates, they are an arc.

3. `arcs` — sequences of two or more objects that trace one subject changing over time.
   This is what neither region could see alone, and it is the point of this pass.

Empty arrays are a legitimate answer for a boundary with nothing crossing it. Do not invent
connections to appear productive.
"""


def merge_prompt(left: Sequence[Dict[str, Any]], right: Sequence[Dict[str, Any]],
                 left_label: str, right_label: str, detail: bool = True) -> str:
    def render(objects):
        if detail:
            return [{"ao_id": o["ao_id"], "type": o["type"], "scene_id": o["scene_id"],
                     "subject": o.get("subject"), "about": o.get("about"),
                     "statement": o["statement"], "confidence": o["confidence"]}
                    for o in objects]
        # Higher merge levels see summaries only: the attention filter widens as the span
        # grows, which is the only way the top of the tree stays inside a context window.
        return [{"ao_id": o["ao_id"], "type": o["type"], "scene_id": o["scene_id"],
                 "subject": o.get("subject"), "statement": o["statement"][:170]}
                for o in objects]
    return """\
=== REGION A ({la}) ===
{a}

=== REGION B ({lb}) ===
{b}

{task}
""".format(la=left_label, lb=right_label,
           a=json.dumps(render(left), ensure_ascii=False, indent=1),
           b=json.dumps(render(right), ensure_ascii=False, indent=1),
           task=_MERGE_TASK)


_EDITOR_TASK = """\
You are standardising entity naming across one screenplay's knowledge graph.

Below is a batch of names as they currently appear, plus the mapping decisions already made
on earlier batches. Extend that mapping consistently: if an earlier batch settled on a
canonical form, reuse it exactly rather than inventing a competing one.

Group only spellings that denote the **same** entity — "Agent Smith", "Smith", "agent_smith".
Do not merge distinct entities that share words: a character and a place named after them are
two entities; a group and one of its members are two entities; a person and a program bearing
their likeness are two entities. When identity is uncertain, leave the name out of the map so
the deterministic fallback keeps them separate.

Canonical form is lowercase snake_case.
"""


def editor_prompt(names: Sequence[str], established: Dict[str, str]) -> str:
    return """\
=== MAPPING ALREADY ESTABLISHED (reuse these canonical forms) ===
{done}

=== NAMES IN THIS BATCH ===
{batch}

{task}
""".format(done=json.dumps(established, ensure_ascii=False, indent=1) or "{}",
           batch=json.dumps(sorted(names), ensure_ascii=False),
           task=_EDITOR_TASK)
