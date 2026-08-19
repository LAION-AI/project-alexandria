"""Abstraction Object schema, adapted from CogniTino for the screenplay scene layer.

CogniTino separates Cognitive Objects into **Perception Objects** (direct observation, no
interpretation) and **Abstraction Objects** (interpretation, inference, mental constructs),
and requires that every Abstraction Object carry pointers to the evidence supporting it.

The mapping used here:

    CogniTino Perception Object   ->  a Knowledge Unit beat  (already built, Alexandria layer)
    CogniTino Abstraction Object  ->  the objects defined in this file

That mapping is the whole reason the two projects compose. The Alexandria layer was built to
record *only what the screenplay states*, and its `certainty: "stated"` field exists to mark
that boundary — which makes its beats exactly the "direct observations, without added
interpretation" that CogniTino's Perception Objects are defined to be. Nothing had to be
retrofitted; the boundary was already drawn in the right place.

What is deliberately narrowed from the paper:

* Only the text modality exists here, so image/audio/video Perception Objects are dropped.
* Confidence is an ordinal band, not the paper's continuous 0.85-style score. A model that
  emits two decimal places of confidence about a fictional character's inner life is
  reporting precision it does not have.
* `grounded_in` is **mandatory and schema-bound to real beat ids**. The paper states the
  traceability principle; here it is enforced by an enum, because an ungrounded hypothesis
  about a mind is indistinguishable from a hallucination about a mind.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


# CogniTino's Abstraction Object taxonomy, restricted to the types that carry narrative
# meaning at the scene layer. Hypothesis / Relationship / Entity / Concept / Event / Process
# are the paper's; `mental_state` and `authorial_intent` are added — see NOTES.md.
AO_TYPES = [
    "hypothesis",        # CO-Hypo: an inferred belief about the world of the story
    "mental_state",      # added: perception / thought / emotion / value of one agent
    "theory_of_mind",    # added: what A believes about B (recursively)
    "relationship",      # CO-Rel: dynamic connection between entities
    "entity_trait",      # CO-Entity: inferred disposition, not stated
    "concept",           # CO-Concept: an abstraction grouping instances
    "process",           # CO-Process: a recurring pattern or procedure
    "authorial_intent",  # added: what the writer appears to be doing with this scene
    "consequence",       # added: what this scene sets up or makes possible elsewhere
]

CONFIDENCE = ["speculative", "plausible", "probable", "near-certain"]

# CogniTino's Semantic Connection Module lists causal, hierarchical, temporal and thematic
# relationships. These are those four families, made specific to narrative.
LINK_TYPES = [
    "supports", "contradicts",          # evidential
    "causes", "caused_by", "enables",   # causal
    "instance_of", "generalizes",       # hierarchical
    "precedes", "follows",              # temporal
    "concerns", "shares_theme_with",    # thematic
]


def _ao(beat_refs: Sequence[str], entity_ids: Sequence[str], ao_ids: Sequence[str] = ()) -> Dict[str, Any]:
    grounding = {
        "type": "array",
        "items": {"type": "string", "enum": list(beat_refs)},
        "minItems": 1,
    } if beat_refs else {"type": "array", "items": {"type": "string"}, "minItems": 1}

    subject = (
        {"type": "string", "enum": list(entity_ids)} if entity_ids else {"type": "string"}
    )
    return {
        "type": "object",
        "properties": {
            "ao_id": {"type": "string", "minLength": 3},
            "type": {"type": "string", "enum": AO_TYPES},
            "scene_id": {"type": "string"},
            "statement": {"type": "string", "minLength": 25},
            "subject": subject,
            "about": {"type": ["string", "null"]},
            # Mandatory, and bound to beats that actually exist. The paper asks for
            # traceability; the enum is what makes it true rather than requested.
            "grounded_in": grounding,
            "reasoning": {"type": "string", "minLength": 20},
            "confidence": {"type": "string", "enum": CONFIDENCE},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "falsifier": {"type": "string", "minLength": 10},
        },
        "required": ["ao_id", "type", "scene_id", "statement", "subject", "about",
                     "grounded_in", "reasoning", "confidence", "assumptions", "falsifier"],
        "additionalProperties": False,
    }


def draft_schema(beat_refs: Sequence[str], entity_ids: Sequence[str],
                 min_items: int = 6, max_items: int = 60) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "abstraction_objects": {
                "type": "array",
                "items": _ao(beat_refs, entity_ids),
                "minItems": min_items,
                "maxItems": max_items,
            }
        },
        "required": ["abstraction_objects"],
        "additionalProperties": False,
    }


def research_schema(beat_refs: Sequence[str], entity_ids: Sequence[str],
                    ao_ids: Sequence[str]) -> Dict[str, Any]:
    """The Researcher module's output: evidence found, links added, objects revised.

    Revision is a patch rather than a rewrite so the draft stays auditable — the paper's
    Step 3 is *updating* Abstraction Objects, and a full rewrite would hide what changed.
    """
    ao_ref = {"type": "string", "enum": list(ao_ids)} if ao_ids else {"type": "string"}
    beat_ref = {"type": "string", "enum": list(beat_refs)} if beat_refs else {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ao_id": ao_ref,
                        "beat_ref": beat_ref,
                        "stance": {"type": "string", "enum": ["supports", "contradicts"]},
                        "why": {"type": "string", "minLength": 15},
                    },
                    "required": ["ao_id", "beat_ref", "stance", "why"],
                    "additionalProperties": False,
                },
            },
            "links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from_ao": ao_ref,
                        "to_ao": ao_ref,
                        "link": {"type": "string", "enum": LINK_TYPES},
                        "why": {"type": "string", "minLength": 15},
                    },
                    "required": ["from_ao", "to_ao", "link", "why"],
                    "additionalProperties": False,
                },
            },
            "confidence_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ao_id": ao_ref,
                        "new_confidence": {"type": "string", "enum": CONFIDENCE},
                        "why": {"type": "string", "minLength": 15},
                    },
                    "required": ["ao_id", "new_confidence", "why"],
                    "additionalProperties": False,
                },
            },
            "new_objects": {
                "type": "array",
                "items": _ao(beat_refs, entity_ids),
                "maxItems": 20,
            },
        },
        "required": ["evidence", "links", "confidence_updates", "new_objects"],
        "additionalProperties": False,
    }


def merge_schema(ao_ids: Sequence[str]) -> Dict[str, Any]:
    """Cross-boundary connection. Sees objects from two adjacent regions, never the source."""
    ao_ref = {"type": "string", "enum": list(ao_ids)} if ao_ids else {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "cross_links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from_ao": ao_ref,
                        "to_ao": ao_ref,
                        "link": {"type": "string", "enum": LINK_TYPES},
                        "why": {"type": "string", "minLength": 15},
                    },
                    "required": ["from_ao", "to_ao", "link", "why"],
                    "additionalProperties": False,
                },
            },
            "duplicates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "keep": ao_ref,
                        "drop": ao_ref,
                        "why": {"type": "string", "minLength": 10},
                    },
                    "required": ["keep", "drop", "why"],
                    "additionalProperties": False,
                },
            },
            "arcs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 5},
                        "subject": {"type": "string"},
                        "stages": {"type": "array", "items": ao_ref, "minItems": 2},
                        "summary": {"type": "string", "minLength": 25},
                    },
                    "required": ["name", "subject", "stages", "summary"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["cross_links", "duplicates", "arcs"],
        "additionalProperties": False,
    }


def editor_schema(names: Sequence[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "canonical_map": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "canonical": {"type": "string", "minLength": 1},
                        "variants": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    },
                    "required": ["canonical", "variants"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["canonical_map"],
        "additionalProperties": False,
    }


def beat_refs_for(units: Sequence[Dict[str, Any]]) -> List[str]:
    """`sc-042#3` — the address of one Perception Object in CogniTino terms."""
    return [
        "{}#{}".format(unit["scene_id"], beat.get("order"))
        for unit in units
        for beat in unit.get("beats") or []
    ]


class AOValidationError(ValueError):
    pass


def validate_drafts(payload: Dict[str, Any], scene_ids: Sequence[str],
                    beat_refs: Sequence[str]) -> List[Dict[str, Any]]:
    objects = payload.get("abstraction_objects")
    if not isinstance(objects, list) or not objects:
        raise AOValidationError("no abstraction_objects returned")
    allowed_scenes, allowed_beats = set(scene_ids), set(beat_refs)
    seen = set()
    for obj in objects:
        aid = obj.get("ao_id")
        if not aid or aid in seen:
            raise AOValidationError("missing or duplicate ao_id: {!r}".format(aid))
        seen.add(aid)
        if obj.get("scene_id") not in allowed_scenes:
            raise AOValidationError("{}: scene_id {!r} outside this window".format(
                aid, obj.get("scene_id")))
        grounds = obj.get("grounded_in") or []
        if not grounds:
            raise AOValidationError("{}: no grounding".format(aid))
        bad = [g for g in grounds if g not in allowed_beats]
        if bad:
            raise AOValidationError("{}: grounded in non-existent beats {}".format(aid, bad[:4]))
        if obj.get("type") not in AO_TYPES:
            raise AOValidationError("{}: bad type {!r}".format(aid, obj.get("type")))
        if obj.get("confidence") not in CONFIDENCE:
            raise AOValidationError("{}: bad confidence {!r}".format(aid, obj.get("confidence")))
    return objects
