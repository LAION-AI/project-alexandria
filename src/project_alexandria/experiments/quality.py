"""LLM-based KU audit and self-correction prompts."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..backends import GenerationBackend
from ..parsing import extract_json
from ..prompts import SYSTEM_PROMPT, quality_prompt


def evaluate_unit(
    backend: GenerationBackend, source_text: str, unit: Dict[str, Any]
) -> Dict[str, Any]:
    return extract_json(backend.generate(SYSTEM_PROMPT, quality_prompt(source_text, unit)))


def correction_prompt(source_text: str, unit: Dict[str, Any], audit: Dict[str, Any]) -> str:
    return """Correct the Knowledge Unit using only SOURCE and the AUDIT. Remove unsupported
claims, repair errors, and add important omitted facts. Preserve the Knowledge Unit JSON schema.
Return only JSON.

SOURCE:
{source}

DRAFT:
{unit}

AUDIT:
{audit}""".format(
        source=source_text,
        unit=json.dumps(unit, ensure_ascii=False),
        audit=json.dumps(audit, ensure_ascii=False),
    )
