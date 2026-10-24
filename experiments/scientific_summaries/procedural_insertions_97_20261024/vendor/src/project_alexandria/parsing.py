"""Parse and validate model JSON while retaining useful diagnostics."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from .schema import Entity


ALLOWED_ENTITY_TYPES = {
    "person",
    "organization",
    "method",
    "measurement",
    "concept",
    "event",
    "place",
    "other",
}


def normalize_entity_types(entities: List[Entity]) -> List[str]:
    """Apply the public type vocabulary and retain the model's original label."""
    warnings = []
    for entity in entities:
        if entity.entity_type not in ALLOWED_ENTITY_TYPES:
            original_type = entity.entity_type
            entity.entity_type = "other"
            entity.attributes.setdefault("model_entity_type", original_type)
            warnings.append(
                "normalized unsupported entity type {!r} for {!r}".format(
                    original_type, entity.name
                )
            )
    return warnings


def _recover_truncated_object(cleaned: str) -> Dict[str, Any]:
    """Recover complete top-level KU fields without inventing truncated content."""
    decoder = json.JSONDecoder()
    summary = ""
    summary_match = re.search(r'"context_summary"\s*:\s*', cleaned)
    if summary_match:
        try:
            value, _ = decoder.raw_decode(cleaned, summary_match.end())
            if isinstance(value, str):
                summary = value
        except json.JSONDecodeError:
            pass

    entities_match = re.search(r'"entities"\s*:\s*\[', cleaned)
    entities = []
    if entities_match:
        cursor = entities_match.end()
        while cursor < len(cleaned):
            while cursor < len(cleaned) and (cleaned[cursor].isspace() or cleaned[cursor] == ","):
                cursor += 1
            if cursor >= len(cleaned) or cleaned[cursor] == "]":
                break
            try:
                entity, cursor = decoder.raw_decode(cleaned, cursor)
            except json.JSONDecodeError:
                break
            if isinstance(entity, dict):
                entities.append(entity)
    if not entities:
        raise ValueError("truncated model response contains no complete entity objects")
    return {
        "context_summary": summary,
        "entities": entities,
        "_alexandria_recovered_truncated_json": True,
    }


def extract_json(text: str, allow_truncated: bool = False) -> Dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            if not allow_truncated:
                raise ValueError("model response contains no JSON object")
            value = _recover_truncated_object(cleaned)
        else:
            try:
                value = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                if not allow_truncated:
                    raise
                value = _recover_truncated_object(cleaned)
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def parse_unit_response(
    text: str, allow_truncated: bool = False
) -> Tuple[str, List[Entity], List[str]]:
    value = extract_json(text, allow_truncated=allow_truncated)
    warnings = []
    if value.get("_alexandria_recovered_truncated_json"):
        warnings.append("recovered complete entity objects from truncated JSON")
    entities_value = value.get("entities") or []
    if isinstance(entities_value, dict):
        entities_value = [dict(body, name=name) for name, body in entities_value.items()]
        warnings.append("converted legacy entity dictionary to entity list")
    entities = [Entity.from_dict(item) for item in entities_value if isinstance(item, dict)]
    entities = [entity for entity in entities if entity.name]
    warnings.extend(normalize_entity_types(entities))
    if not entities:
        warnings.append("model returned no named entities")
    return str(value.get("context_summary", "")).strip(), entities, warnings
