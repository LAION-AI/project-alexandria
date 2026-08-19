"""Render a KU chain as compact text for a reader model.

Raw JSON is the wrong shape to hand a student: punctuation and key names consume a large
share of the budget, and the reader has to parse structure it never asked for. This emits
the same content as an ordered digest.

Size matters here for a concrete reason. The KU chain must fit the student's context in one
piece, because an arm that silently truncates its context produces a confident wrong
comparison — the arm looks like it "had the KUs" while the tail was never shown.
``render_chain`` therefore reports its own token estimate and the caller asserts it.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence


def _fact_tail(beat: Dict[str, Any]) -> str:
    facts = beat.get("facts") or {}
    keep: List[str] = []
    for key in ("quantities", "dates"):
        keep.extend(str(value) for value in facts.get(key) or [])
    return "  [{}]".format("; ".join(keep)) if keep else ""


def render_unit(unit: Dict[str, Any], *, verbosity: str = "full") -> str:
    heading = unit.get("heading") or {}
    lines = [
        "## {} | {}".format(unit.get("scene_id"), heading.get("raw") or "(no heading)"),
    ]
    if verbosity != "minimal":
        if unit.get("context_before"):
            lines.append("Before: {}".format(unit["context_before"]))
    present = unit.get("present") or []
    if present:
        lines.append("Present: {}".format(", ".join(present)))

    for beat in unit.get("beats") or []:
        actor = beat.get("actor") or "?"
        addressee = beat.get("addressee")
        arrow = " -> {}".format(addressee) if addressee else ""
        lines.append(
            "{}. [{}] {}{}: {}{}".format(
                beat.get("order"), beat.get("type"), actor, arrow,
                beat.get("content"), _fact_tail(beat),
            )
        )
        for change in beat.get("state_changes") or []:
            lines.append(
                "     ~ {}.{}: {} -> {}".format(
                    change.get("entity"), change.get("field"),
                    change.get("from"), change.get("to"),
                )
            )
    if verbosity == "full" and unit.get("context_after"):
        lines.append("After: {}".format(unit["context_after"]))
    return "\n".join(lines)


def render_chain(
    artifact: Dict[str, Any],
    *,
    verbosity: str = "full",
    scene_ids: Optional[Sequence[str]] = None,
) -> str:
    document = artifact.get("document") or {}
    header = [
        "KNOWLEDGE UNITS — {} ({}, {})".format(
            document.get("title") or "untitled",
            document.get("draft") or "draft unknown",
            document.get("draft_date") or "date unknown",
        ),
        "Credited as: {}".format(document.get("credited_as") or "unknown"),
        "Scenes in order. Each scene lists its beats in the order the audience receives them.",
        "",
    ]
    wanted = set(scene_ids) if scene_ids is not None else None
    units = sorted(
        artifact.get("knowledge_units") or [], key=lambda unit: unit.get("scene_index", 0)
    )
    body = [
        render_unit(unit, verbosity=verbosity)
        for unit in units
        if wanted is None or unit.get("scene_id") in wanted
    ]
    return "\n".join(header) + "\n\n".join(body)


def estimate_tokens(text: str) -> int:
    """Rough but conservative: ~3.6 chars per token for English prose with markup."""
    return int(len(text) / 3.6)


def neighbourhood(
    artifact: Dict[str, Any], scene_id: str, radius: int = 2
) -> List[str]:
    units = sorted(
        artifact.get("knowledge_units") or [], key=lambda unit: unit.get("scene_index", 0)
    )
    ids = [unit.get("scene_id") for unit in units]
    if scene_id not in ids:
        return [scene_id]
    position = ids.index(scene_id)
    low = max(0, position - radius)
    return ids[low : position + radius + 1]
