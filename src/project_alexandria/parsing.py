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


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response contains no JSON object")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def parse_unit_response(text: str) -> Tuple[str, List[Entity], List[str]]:
    value = extract_json(text)
    warnings = []
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
