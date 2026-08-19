"""The CogniTino modules, run over an Alexandria Knowledge Unit chain.

Stage map (paper module -> implementation):

    1. Chunking & Contextualization  -> already done: the KU chain is the Perception layer
    2. Abstraction Object Generation -> `draft_all`      (parallel, one agent per window)
    3. Abstraction Object Researcher -> `research_all`   (parallel, iterative per window)
    4. Editor                        -> `canonicalize`   (sequential, running alias map)
    5. Semantic Connection           -> `merge_tree`     (hierarchical, widening attention)

Modules 4 and 5 are run in the order 5-then-4, not 4-then-5. The paper's ordering assumes
the editor polishes objects before they are connected; here connection *creates* objects
(arcs) and merges others away, so canonicalising first would standardise names that the
merge then discards. Recorded in NOTES.md as a deliberate departure.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..client import EndpointPool, run_parallel
from ..kuschema import grammar_safe
from . import prompts
from .schema import (
    AOValidationError,
    beat_refs_for,
    draft_schema,
    editor_schema,
    merge_schema,
    research_schema,
    validate_drafts,
)


def _parse(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


class Window:
    """A contiguous run of scenes one abstraction agent owns."""

    def __init__(self, index: int, units: Sequence[Dict[str, Any]], scenes_by_id, source: str):
        self.index = index
        self.units = list(units)
        self.scene_ids = [u["scene_id"] for u in self.units]
        self.beat_refs = beat_refs_for(self.units)
        self.entity_ids = sorted({
            e.get("entity_id") for u in self.units for e in u.get("entities") or []
            if e.get("entity_id")
        })
        parts = []
        for sid in self.scene_ids:
            scene = scenes_by_id.get(sid)
            if scene is not None:
                parts.append(scene.text(source))
        self.text = "\n".join(parts)
        self.objects: List[Dict[str, Any]] = []

    @property
    def label(self) -> str:
        return "{}..{}".format(self.scene_ids[0], self.scene_ids[-1])


def build_windows(units: Sequence[Dict[str, Any]], scenes_by_id, source: str,
                  scenes_per_window: int = 5) -> List[Window]:
    ordered = sorted(units, key=lambda u: u.get("scene_index", 0))
    out = []
    for index, start in enumerate(range(0, len(ordered), scenes_per_window)):
        chunk = ordered[start:start + scenes_per_window]
        if chunk:
            out.append(Window(index, chunk, scenes_by_id, source))
    return out


# ------------------------------------------------------------------ module 2


def draft_all(pool: EndpointPool, windows: Sequence[Window], source: str,
              document: Dict[str, Any], *, workers: int = 14,
              max_tokens: int = 16384, progress=None) -> Dict[str, Any]:
    started = time.time()

    def work(window: Window):
        schema = draft_schema(window.beat_refs, window.entity_ids)
        prompt = prompts.draft_prompt(source, document, window.text, window.units,
                                      window.beat_refs, window.entity_ids)
        last = None
        for attempt in range(2):
            suffix = "" if attempt == 0 else (
                "\n\nYour previous response was rejected: {}. Ground every object in the "
                "listed beat references and keep scene ids inside your window.".format(last))
            result = pool.call(prompts.SYSTEM, prompt + suffix,
                               schema=grammar_safe(schema), max_tokens=max_tokens)
            try:
                payload = _parse(result.text)
                objects = validate_drafts(payload, window.scene_ids, window.beat_refs)
                # Namespace ids by window so the merge tree can never collide them.
                for position, obj in enumerate(objects):
                    obj["ao_id"] = "ao-w{:03d}-{:03d}".format(window.index, position)
                    obj["_window"] = window.index
                    obj["provenance"] = {"stage": "draft", "window": window.index}
                return {"window": window.index, "objects": objects,
                        "usage": {"prompt_tokens": result.prompt_tokens,
                                  "completion_tokens": result.completion_tokens,
                                  "seconds": round(result.seconds, 1), "port": result.port}}
            except (AOValidationError, json.JSONDecodeError, KeyError, TypeError) as error:
                last = error
        raise RuntimeError("window {} draft failed: {}".format(window.index, last))

    results = run_parallel(list(windows), work, max_workers=workers, on_done=progress)
    failures, usage = [], []
    for window, result in zip(windows, results):
        if isinstance(result, Exception):
            failures.append({"window": window.index, "error": str(result)})
            continue
        window.objects = result["objects"]
        usage.append({"window": window.index, **result["usage"]})
    return {"failures": failures, "usage": usage,
            "objects": sum(len(w.objects) for w in windows),
            "seconds": round(time.time() - started, 1)}


# ------------------------------------------------------------------ module 3


def research_all(pool: EndpointPool, windows: Sequence[Window], source: str, *,
                 rounds: int = 2, workers: int = 14, max_tokens: int = 12288,
                 progress=None) -> Dict[str, Any]:
    started = time.time()
    tally = {"evidence": 0, "links": 0, "downgrades": 0, "upgrades": 0, "new_objects": 0}
    failures: List[Dict[str, Any]] = []

    for round_index in range(1, rounds + 1):
        def work(window: Window, rnd=round_index):
            if not window.objects:
                return None
            ao_ids = [o["ao_id"] for o in window.objects]
            schema = research_schema(window.beat_refs, window.entity_ids, ao_ids)
            prompt = prompts.research_prompt(source, window.text, window.units,
                                             window.objects, window.beat_refs,
                                             window.entity_ids, rnd)
            result = pool.call(prompts.SYSTEM, prompt,
                               schema=grammar_safe(schema), max_tokens=max_tokens)
            return _parse(result.text)

        results = run_parallel(list(windows), work, max_workers=workers, on_done=progress)
        for window, payload in zip(windows, results):
            if isinstance(payload, Exception) or payload is None:
                if isinstance(payload, Exception):
                    failures.append({"window": window.index, "round": round_index,
                                     "error": str(payload)})
                continue
            by_id = {o["ao_id"]: o for o in window.objects}

            for item in payload.get("evidence") or []:
                target = by_id.get(item.get("ao_id"))
                if not target:
                    continue
                key = "supporting_evidence" if item.get("stance") == "supports" else "contradicting_evidence"
                target.setdefault(key, []).append(
                    {"beat_ref": item.get("beat_ref"), "why": item.get("why")})
                tally["evidence"] += 1

            for item in payload.get("links") or []:
                source_obj = by_id.get(item.get("from_ao"))
                if not source_obj or item.get("to_ao") not in by_id:
                    continue
                source_obj.setdefault("links", []).append(
                    {"to": item["to_ao"], "link": item["link"], "why": item.get("why")})
                tally["links"] += 1

            order = {"speculative": 0, "plausible": 1, "probable": 2, "near-certain": 3}
            for item in payload.get("confidence_updates") or []:
                target = by_id.get(item.get("ao_id"))
                new = item.get("new_confidence")
                if not target or new not in order:
                    continue
                if order[new] < order.get(target.get("confidence"), 1):
                    tally["downgrades"] += 1
                elif order[new] > order.get(target.get("confidence"), 1):
                    tally["upgrades"] += 1
                target.setdefault("confidence_history", []).append(
                    {"from": target.get("confidence"), "to": new, "why": item.get("why"),
                     "round": round_index})
                target["confidence"] = new

            for position, obj in enumerate(payload.get("new_objects") or []):
                if obj.get("scene_id") not in set(window.scene_ids):
                    continue
                grounds = obj.get("grounded_in") or []
                if not grounds or any(g not in set(window.beat_refs) for g in grounds):
                    continue
                obj["ao_id"] = "ao-w{:03d}-r{}-{:03d}".format(window.index, round_index, position)
                obj["_window"] = window.index
                obj["provenance"] = {"stage": "research", "round": round_index,
                                     "window": window.index}
                window.objects.append(obj)
                tally["new_objects"] += 1

    return {"rounds": rounds, "failures": failures, "tally": tally,
            "objects": sum(len(w.objects) for w in windows),
            "seconds": round(time.time() - started, 1)}


# ------------------------------------------------------------------ module 5


def merge_tree(pool: EndpointPool, windows: Sequence[Window], *, levels: int = 3,
               workers: int = 14, max_tokens: int = 8192, progress=None) -> Dict[str, Any]:
    """Pairwise hierarchical merge: 5 scenes -> 10 -> 20 -> 40.

    Level 1 sees full objects; higher levels see summaries. The span each agent reasons over
    doubles at every level while the resolution drops, which is what keeps the top of the
    tree inside a context window. Merging all the way to a single root would put every
    object in one call, so the tree is capped and the remaining global consistency is left to
    the editor.
    """
    started = time.time()
    groups: List[Tuple[str, List[Dict[str, Any]]]] = [
        (w.label, w.objects) for w in windows if w.objects
    ]
    all_objects = {o["ao_id"]: o for w in windows for o in w.objects}
    report = {"levels": [], "cross_links": 0, "duplicates": 0, "arcs": []}

    for level in range(1, levels + 1):
        pairs = [(groups[i], groups[i + 1]) for i in range(0, len(groups) - 1, 2)]
        if not pairs:
            break

        def work(pair, lvl=level):
            (llabel, left), (rlabel, right) = pair
            ids = [o["ao_id"] for o in left] + [o["ao_id"] for o in right]
            prompt = prompts.merge_prompt(left, right, llabel, rlabel, detail=(lvl == 1))
            result = pool.call(prompts.SYSTEM, prompt,
                               schema=grammar_safe(merge_schema(ids)), max_tokens=max_tokens)
            return _parse(result.text)

        results = run_parallel(pairs, work, max_workers=workers, on_done=progress)

        merged: List[Tuple[str, List[Dict[str, Any]]]] = []
        level_stats = {"level": level, "merges": len(pairs), "cross_links": 0,
                       "duplicates": 0, "arcs": 0, "failures": 0}
        for pair, payload in zip(pairs, results):
            (llabel, left), (rlabel, right) = pair
            combined = list(left) + list(right)
            label = "{}..{}".format(llabel.split("..")[0], rlabel.split("..")[-1])
            if isinstance(payload, Exception):
                level_stats["failures"] += 1
                merged.append((label, combined))
                continue

            for item in payload.get("cross_links") or []:
                node = all_objects.get(item.get("from_ao"))
                if node and item.get("to_ao") in all_objects:
                    node.setdefault("links", []).append(
                        {"to": item["to_ao"], "link": item["link"],
                         "why": item.get("why"), "via": "merge-L{}".format(level)})
                    level_stats["cross_links"] += 1

            dropped = set()
            for item in payload.get("duplicates") or []:
                keep, drop = item.get("keep"), item.get("drop")
                if keep in all_objects and drop in all_objects and keep != drop:
                    all_objects[drop]["superseded_by"] = keep
                    all_objects[keep].setdefault("absorbed", []).append(drop)
                    dropped.add(drop)
                    level_stats["duplicates"] += 1

            for arc in payload.get("arcs") or []:
                stages = [s for s in arc.get("stages") or [] if s in all_objects]
                if len(stages) >= 2:
                    report["arcs"].append({
                        "name": arc.get("name"), "subject": arc.get("subject"),
                        "stages": stages, "summary": arc.get("summary"),
                        "level": level, "span": label})
                    level_stats["arcs"] += 1

            merged.append((label, [o for o in combined if o["ao_id"] not in dropped]))

        if len(groups) % 2 == 1:
            merged.append(groups[-1])
        groups = merged
        report["cross_links"] += level_stats["cross_links"]
        report["duplicates"] += level_stats["duplicates"]
        report["levels"].append(level_stats)

    report["seconds"] = round(time.time() - started, 1)
    report["arc_count"] = len(report["arcs"])
    return report


# ------------------------------------------------------------------ module 4


def canonicalize(pool: EndpointPool, windows: Sequence[Window], units: Sequence[Dict[str, Any]],
                 *, batch: int = 40, max_tokens: int = 4096) -> Dict[str, Any]:
    """Sequential editor pass carrying a running map, exactly as specified.

    Deliberately not parallel. The point of the running map is that batch N sees the
    decisions of batches 1..N-1 and reuses their canonical forms; parallel batches would
    each invent their own and the pass would produce the inconsistency it exists to remove.
    """
    started = time.time()
    names = sorted({
        str(value)
        for w in windows for o in w.objects
        for value in (o.get("subject"), o.get("about")) if value
    } | {
        e.get("entity_id") for u in units for e in u.get("entities") or [] if e.get("entity_id")
    })

    established: Dict[str, str] = {}
    batches = 0
    for start in range(0, len(names), batch):
        chunk = names[start:start + batch]
        try:
            result = pool.call(
                prompts.SYSTEM, prompts.editor_prompt(chunk, established),
                schema=grammar_safe(editor_schema(chunk)), max_tokens=max_tokens)
            payload = _parse(result.text)
        except Exception:
            continue
        batches += 1
        for group in payload.get("canonical_map") or []:
            canonical = (group.get("canonical") or "").strip()
            if not canonical:
                continue
            for variant in group.get("variants") or []:
                variant = str(variant).strip()
                if variant and variant != canonical:
                    established[variant] = canonical

    def resolve(value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        seen = set()
        while value in established and value not in seen:
            seen.add(value)
            value = established[value]
        return value

    applied = 0
    for w in windows:
        for o in w.objects:
            for key in ("subject", "about"):
                new = resolve(o.get(key))
                if new != o.get(key):
                    o[key] = new
                    applied += 1
    return {"names": len(names), "batches": batches, "mappings": len(established),
            "renames_applied": applied, "alias_map": established,
            "seconds": round(time.time() - started, 1)}
