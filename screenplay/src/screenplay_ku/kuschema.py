"""JSON schema for scene Knowledge Units, plus a validator that runs after the call.

Two rules from the sibling project shape this file:

*Structure repairs what instructions cannot.* The set of scene ids a window must emit is
known before the call, so it is bound as an ``enum`` with ``minItems``/``maxItems`` rather
than asked for in a sentence. A prompt clause requesting "one unit per scene" is advisory;
a schema requiring exactly these keys is not.

*The guarantee lives in the validator, not the grammar.* vLLM rejects some keywords and
silently 500s on others, so the schema is filtered before it is sent and then validated in
full afterwards. Filtering must never be able to weaken the post-hoc check.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


BEAT_TYPES = ["action", "speech", "revelation", "state_change", "movement", "perception"]
CERTAINTY = ["stated", "implied_by_action", "ambiguous"]
ENTITY_TYPES = [
    "person", "organization", "place", "object", "program", "vehicle",
    "concept", "event", "faction", "other",
]

# vLLM's guided decoding backend rejects or mishandles these. Stripping them changes what
# the grammar enforces, never what the validator enforces.
_UNSUPPORTED = {"propertyNames", "patternProperties", "if", "then", "else", "not",
                "allOf", "oneOf", "dependentSchemas", "unevaluatedProperties"}


def _beat() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "order": {"type": "integer", "minimum": 1},
            "type": {"type": "string", "enum": BEAT_TYPES},
            "actor": {"type": "string"},
            "addressee": {"type": ["string", "null"]},
            "content": {"type": "string", "minLength": 10},
            "facts": {
                "type": "object",
                "properties": {
                    "quantities": {"type": "array", "items": {"type": "string"}},
                    "dates": {"type": "array", "items": {"type": "string"}},
                    "proper_nouns": {"type": "array", "items": {"type": "string"}},
                    "locations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["quantities", "dates", "proper_nouns", "locations"],
                "additionalProperties": False,
            },
            "state_changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string"},
                        "field": {"type": "string"},
                        "from": {"type": ["string", "null"]},
                        "to": {"type": "string"},
                    },
                    "required": ["entity", "field", "from", "to"],
                    "additionalProperties": False,
                },
            },
            "causes": {"type": "array", "items": {"type": "integer", "minimum": 1}},
            "certainty": {"type": "string", "enum": CERTAINTY},
        },
        "required": ["order", "type", "actor", "addressee", "content", "facts",
                     "state_changes", "causes", "certainty"],
        "additionalProperties": False,
    }


def _entity() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "entity_id": {"type": "string", "minLength": 1},
            "type": {"type": "string", "enum": ENTITY_TYPES},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "attributes": {"type": "object"},
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "predicate": {"type": "string"},
                        "target": {"type": "string"},
                        "target_id": {"type": ["string", "null"]},
                    },
                    "required": ["predicate", "target", "target_id"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["name", "entity_id", "type", "aliases", "attributes", "relationships"],
        "additionalProperties": False,
    }


def _unit(scene_ids: Sequence[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scene_id": {"type": "string", "enum": list(scene_ids)},
            "context_before": {"type": "string", "minLength": 20},
            "context_after": {"type": "string", "minLength": 20},
            "style": {"type": "string", "minLength": 20},
            "present": {"type": "array", "items": {"type": "string"}},
            "referenced": {"type": "array", "items": {"type": "string"}},
            "beats": {"type": "array", "items": _beat(), "minItems": 1},
            "entities": {"type": "array", "items": _entity()},
        },
        "required": ["scene_id", "context_before", "context_after", "style",
                     "present", "referenced", "beats", "entities"],
        "additionalProperties": False,
    }


def extraction_schema(scene_ids: Sequence[str]) -> Dict[str, Any]:
    """Bind the response to exactly the scenes this window owns."""
    count = len(scene_ids)
    return {
        "type": "object",
        "properties": {
            "knowledge_units": {
                "type": "array",
                "items": _unit(scene_ids),
                "minItems": count,
                "maxItems": count,
            }
        },
        "required": ["knowledge_units"],
        "additionalProperties": False,
    }


def canary_schema(scene_ids: Sequence[str]) -> Dict[str, Any]:
    """Same shape, but an empty list must be legal — that is the whole point of the canary."""
    return {
        "type": "object",
        "properties": {
            "knowledge_units": {"type": "array", "items": _unit(scene_ids), "maxItems": 0}
        },
        "required": ["knowledge_units"],
        "additionalProperties": False,
    }


def seam_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "alias_merges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "canonical_id": {"type": "string"},
                        "merge": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                    },
                    "required": ["canonical_id", "merge", "reason"],
                    "additionalProperties": False,
                },
            },
            "context_fixes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "scene_id": {"type": "string"},
                        "field": {"type": "string", "enum": ["context_before", "context_after"]},
                        "replacement": {"type": "string", "minLength": 20},
                        "reason": {"type": "string"},
                    },
                    "required": ["scene_id", "field", "replacement", "reason"],
                    "additionalProperties": False,
                },
            },
            "duplicate_scene_ids": {"type": "array", "items": {"type": "string"}},
            "continuity_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["alias_merges", "context_fixes", "duplicate_scene_ids", "continuity_notes"],
        "additionalProperties": False,
    }


def grammar_safe(schema: Any) -> Any:
    """Strip keywords vLLM's guided decoding cannot handle. Never called on the validator."""
    if isinstance(schema, dict):
        return {k: grammar_safe(v) for k, v in schema.items() if k not in _UNSUPPORTED}
    if isinstance(schema, list):
        return [grammar_safe(item) for item in schema]
    return schema


# ---------------------------------------------------------------- validation


class ValidationError(ValueError):
    pass


def validate_units(payload: Dict[str, Any], expected_scene_ids: Sequence[str]) -> List[Dict[str, Any]]:
    """Full post-hoc validation. Raises rather than repairing or substituting.

    A guard that replaces missing input with empty input converts a crash into a
    confident wrong answer, so every failure here is fatal to the window and the window is
    retried or quarantined — never silently emitted as an empty unit.
    """
    if not isinstance(payload, dict) or "knowledge_units" not in payload:
        raise ValidationError("response has no knowledge_units key")
    units = payload["knowledge_units"]
    if not isinstance(units, list):
        raise ValidationError("knowledge_units is not a list")

    expected = list(expected_scene_ids)
    got = [unit.get("scene_id") for unit in units if isinstance(unit, dict)]
    if sorted(got) != sorted(expected):
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        raise ValidationError(
            "scene id mismatch: missing={} unexpected={}".format(missing, extra)
        )
    if len(got) != len(set(got)):
        raise ValidationError("duplicate scene ids in one window response")

    for unit in units:
        scene_id = unit.get("scene_id")
        for key in ("context_before", "context_after", "style"):
            value = unit.get(key)
            if not isinstance(value, str) or len(value.strip()) < 20:
                raise ValidationError("{}: {} missing or too short".format(scene_id, key))
        beats = unit.get("beats")
        if not isinstance(beats, list) or not beats:
            raise ValidationError("{}: no beats".format(scene_id))
        orders = [beat.get("order") for beat in beats]
        if orders != list(range(1, len(beats) + 1)):
            raise ValidationError(
                "{}: beat order must be contiguous from 1, got {}".format(scene_id, orders)
            )
        for beat in beats:
            if beat.get("type") not in BEAT_TYPES:
                raise ValidationError("{}: bad beat type {!r}".format(scene_id, beat.get("type")))
            if beat.get("certainty") not in CERTAINTY:
                raise ValidationError("{}: bad certainty {!r}".format(scene_id, beat.get("certainty")))
            if not isinstance(beat.get("content"), str) or len(beat["content"].strip()) < 10:
                raise ValidationError("{}: beat {} has no content".format(scene_id, beat.get("order")))
            for cause in beat.get("causes") or []:
                if not isinstance(cause, int) or not (1 <= cause < beat.get("order", 0)):
                    raise ValidationError(
                        "{}: beat {} causes {!r}, which is not an earlier beat in the scene".format(
                            scene_id, beat.get("order"), cause
                        )
                    )
    return units
