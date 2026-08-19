"""Two-stage extraction: windows in parallel, then seam verification in parallel."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .client import EndpointPool, run_parallel
from .kuschema import (
    ValidationError,
    canary_schema,
    close_entity_references,
    extraction_schema,
    grammar_safe,
    seam_schema,
    validate_units,
)
from .prompts import SYSTEM_PROMPT, canary_prompt, extraction_prompt, seam_prompt
from .scenes import Scene, scene_index_listing
from .windows import Window


def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


class WindowFailure(RuntimeError):
    def __init__(self, window_index: int, scene_ids: Sequence[str], reason: str):
        super().__init__("window {} ({}): {}".format(window_index, ",".join(scene_ids), reason))
        self.window_index = window_index
        self.scene_ids = list(scene_ids)
        self.reason = reason


class Stage1:
    """Extract one KU per scene, one call per window."""

    def __init__(
        self,
        pool: EndpointPool,
        source: str,
        scenes: Sequence[Scene],
        document: Dict[str, Any],
        *,
        max_tokens: int = 16384,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.pool = pool
        self.source = source
        self.scenes = list(scenes)
        self.document = document
        self.max_tokens = max_tokens
        self.index_table = scene_index_listing(self.scenes)
        self.log_dir = log_dir
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)

    def _prompt(self, window: Window, known_ids: Sequence[str]) -> str:
        builder = canary_prompt if window.is_canary else extraction_prompt
        return builder(
            window, self.source, self.scenes, self.document, self.index_table, known_ids
        )

    def extract_window(self, window: Window, known_ids: Sequence[str] = ()) -> Dict[str, Any]:
        scene_ids = window.scene_ids
        schema = canary_schema(scene_ids) if window.is_canary else extraction_schema(scene_ids)
        prompt = self._prompt(window, known_ids)

        last_error: Optional[Exception] = None
        for attempt in range(2):
            suffix = "" if attempt == 0 else (
                "\n\nYour previous response was rejected: {}. Return complete, valid JSON "
                "with exactly the required scene ids and contiguous beat orders.".format(last_error)
            )
            result = self.pool.call(
                SYSTEM_PROMPT,
                prompt + suffix,
                schema=grammar_safe(schema),
                max_tokens=self.max_tokens,
            )
            try:
                payload = _parse_json(result.text)
                units = (
                    []
                    if window.is_canary
                    else validate_units(payload, scene_ids)
                )
                # Close references deterministically after validation, recording every
                # synthesized id, rather than rejecting the window for a repairable defect.
                closed = sum(len(close_entity_references(unit)) for unit in units)
                if window.is_canary:
                    emitted = payload.get("knowledge_units") or []
                    return {
                        "window_index": window.index,
                        "canary": True,
                        "emitted": len(emitted),
                        "units": emitted,
                        "usage": result.__dict__,
                    }
                return {
                    "window_index": window.index,
                    "canary": False,
                    "units": units,
                    "usage": {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "seconds": round(result.seconds, 2),
                        "port": result.port,
                        "attempts": result.attempts + attempt,
                        "auto_declared_entities": closed,
                    },
                }
            except (ValidationError, json.JSONDecodeError, KeyError, TypeError) as error:
                last_error = error
        raise WindowFailure(window.index, scene_ids, str(last_error))

    def run(self, windows: Sequence[Window], *, max_workers: int = 16, progress=None) -> Dict[str, Any]:
        started = time.time()
        results = run_parallel(
            list(windows),
            lambda window: self.extract_window(window),
            max_workers=max_workers,
            on_done=progress,
        )
        units: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        usage: List[Dict[str, Any]] = []
        for window, result in zip(windows, results):
            if isinstance(result, Exception):
                failures.append(
                    {
                        "window_index": window.index,
                        "scene_ids": window.scene_ids,
                        "error": str(result),
                    }
                )
                continue
            units.extend(result["units"])
            usage.append({"window_index": window.index, **result.get("usage", {})})
        return {
            "units": units,
            "failures": failures,
            "usage": usage,
            "seconds": round(time.time() - started, 1),
        }


class Stage2:
    """Verify each seam. Sees only units — never the source."""

    def __init__(self, pool: EndpointPool, *, flank: int = 3, max_tokens: int = 8192) -> None:
        self.pool = pool
        self.flank = flank
        self.max_tokens = max_tokens

    def verify_seam(self, left: List[Dict], right: List[Dict]) -> Dict[str, Any]:
        left_tail = left[-self.flank :]
        right_head = right[: self.flank]
        prompt = seam_prompt(
            left_tail,
            right_head,
            [unit["scene_id"] for unit in left_tail],
            [unit["scene_id"] for unit in right_head],
        )
        result = self.pool.call(
            SYSTEM_PROMPT, prompt, schema=grammar_safe(seam_schema()), max_tokens=self.max_tokens
        )
        payload = _parse_json(result.text)
        return {
            "alias_merges": payload.get("alias_merges") or [],
            "context_fixes": payload.get("context_fixes") or [],
            "duplicate_scene_ids": payload.get("duplicate_scene_ids") or [],
            "continuity_notes": payload.get("continuity_notes") or [],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "seconds": round(result.seconds, 2),
                "port": result.port,
            },
        }

    def run(
        self,
        units_by_window: Sequence[List[Dict]],
        *,
        max_workers: int = 16,
        progress=None,
    ) -> Dict[str, Any]:
        started = time.time()
        pairs = [
            (units_by_window[i], units_by_window[i + 1])
            for i in range(len(units_by_window) - 1)
            if units_by_window[i] and units_by_window[i + 1]
        ]
        results = run_parallel(
            pairs,
            lambda pair: self.verify_seam(pair[0], pair[1]),
            max_workers=max_workers,
            on_done=progress,
        )
        seams, failures = [], []
        for index, result in enumerate(results):
            if isinstance(result, Exception):
                failures.append({"seam_index": index, "error": str(result)})
            else:
                seams.append({"seam_index": index, **result})
        return {"seams": seams, "failures": failures, "seconds": round(time.time() - started, 1)}


_REPAIR_SYSTEM = (
    "You restate factual records in different words while preserving every fact exactly. "
    "You return only valid JSON."
)

_REPAIR_PROMPT = """\
The RECORD below reuses wording from the source document it was derived from. Restate it so
no distinctive phrasing from the source survives, while preserving every fact.

Rules:
  - Keep every number, date, name, and place EXACTLY as written. Change nothing factual.
  - Change the sentence structure and vocabulary around them.
  - Do not shorten it into a vaguer statement; a restatement that loses a fact is worse
    than the original problem.
  - A span of {length} consecutive words in this record matches the source. Rewrite so that
    no run of more than four ordinary words could match.

RECORD:
{record}

Return {{"restated": "..."}}
"""

_REPAIR_SCHEMA = {
    "type": "object",
    "properties": {"restated": {"type": "string", "minLength": 10}},
    "required": ["restated"],
    "additionalProperties": False,
}


def repair_overlap_fields(
    pool: EndpointPool,
    units: List[Dict[str, Any]],
    violations: Sequence[Dict[str, Any]],
    *,
    max_workers: int = 16,
) -> Dict[str, Any]:
    """Regenerate only the fields the overlap gate flagged.

    The repair model is given the unit's own sentence and the length of the matching span,
    never the source text. It therefore restates a record it already has rather than
    re-reading the original, which is the only way a repair here can be honest: an agent
    that could see the source would simply extract it again.

    This does not soften the gate. The gate re-runs afterwards on the repaired units and
    still blocks the artifact if anything remains over the threshold.
    """
    by_id = {unit["scene_id"]: unit for unit in units}
    targets = []
    for violation in violations:
        unit = by_id.get(violation.get("scene_id"))
        path = violation.get("path") or ""
        if unit is None:
            continue
        match = re.match(r"beats\[(\d+)\]\.content$", path)
        if match:
            order = int(match.group(1))
            beat = next((b for b in unit.get("beats") or [] if b.get("order") == order), None)
            if beat and isinstance(beat.get("content"), str):
                targets.append((unit, beat, "content", violation.get("length", 8)))
        elif path in ("context_before", "context_after", "style"):
            targets.append((unit, unit, path, violation.get("length", 8)))

    def work(target):
        holder, container, key, length = target
        result = pool.call(
            _REPAIR_SYSTEM,
            _REPAIR_PROMPT.format(length=length, record=container[key]),
            schema=_REPAIR_SCHEMA,
            max_tokens=1024,
            temperature=0.4,
        )
        payload = _parse_json(result.text)
        restated = (payload.get("restated") or "").strip()
        return (container, key, restated) if len(restated) >= 10 else None

    results = run_parallel(targets, work, max_workers=max_workers)
    applied = 0
    failed = 0
    for result in results:
        if isinstance(result, Exception) or result is None:
            failed += 1
            continue
        container, key, restated = result
        container[key] = restated
        applied += 1
    return {"targeted": len(targets), "restated": applied, "failed": failed}


def apply_seam_patches(
    units: List[Dict[str, Any]], seams: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Apply stage-2 findings deterministically, recording every change.

    Applied here rather than by a model so the artifact's provenance stays mechanical: the
    seam agent proposes, this function disposes, and the protocol records both.
    """
    by_id = {unit["scene_id"]: unit for unit in units}
    alias_map: Dict[str, str] = {}
    applied = {"alias_merges": 0, "context_fixes": 0, "duplicates_removed": 0, "rejected": []}

    for seam in seams:
        for merge in seam.get("alias_merges") or []:
            canonical = merge.get("canonical_id")
            if not canonical:
                continue
            for alias in merge.get("merge") or []:
                if alias and alias != canonical:
                    alias_map[alias] = canonical
                    applied["alias_merges"] += 1
        for fix in seam.get("context_fixes") or []:
            unit = by_id.get(fix.get("scene_id"))
            field = fix.get("field")
            replacement = fix.get("replacement")
            if unit and field in ("context_before", "context_after") and replacement:
                unit[field] = replacement
                applied["context_fixes"] += 1
            else:
                applied["rejected"].append({"kind": "context_fix", "detail": fix})

    # Resolve chains (a -> b -> c) before rewriting.
    def resolve(value: str) -> str:
        seen = set()
        while value in alias_map and value not in seen:
            seen.add(value)
            value = alias_map[value]
        return value

    for unit in units:
        unit["present"] = [resolve(x) for x in unit.get("present") or []]
        unit["referenced"] = [resolve(x) for x in unit.get("referenced") or []]
        for beat in unit.get("beats") or []:
            beat["actor"] = resolve(beat.get("actor") or "")
            if beat.get("addressee"):
                beat["addressee"] = resolve(beat["addressee"])
            for change in beat.get("state_changes") or []:
                change["entity"] = resolve(change.get("entity") or "")
        for entity in unit.get("entities") or []:
            entity["entity_id"] = resolve(entity.get("entity_id") or "")
            for relationship in entity.get("relationships") or []:
                if relationship.get("target_id"):
                    relationship["target_id"] = resolve(relationship["target_id"])

    return {"units": units, "applied": applied, "alias_map": alias_map}
