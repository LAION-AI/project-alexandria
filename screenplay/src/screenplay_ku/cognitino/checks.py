"""Checks for the abstraction layer.

The Perception layer's checks do not transfer. A Knowledge Unit can be wrong by
contradicting the source; an Abstraction Object cannot be checked that way, because it is
supposed to say things the source does not. Its failure modes are different:

* **Ungrounded** — an inference with no pointer to an observation. CogniTino's traceability
  principle exists for exactly this, and it is the difference between an inference and a
  hallucination.
* **Restatement** — an "abstraction" that merely paraphrases the beat it points at. This is
  the most likely silent failure: it looks like output, passes grounding, and adds nothing.
* **Uniform confidence** — every object marked `probable` means the confidence field is
  decorative and the calibration it claims does not exist.
* **No contradiction sought** — a researcher pass that only ever confirms has not searched.
* **Disconnection** — objects with no links are a list, not a graph.

Each check has a negative case in `tests/test_cognitino_checks.py`.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Sequence


_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def _tokens(text: str) -> List[str]:
    return _WORD.findall((text or "").casefold())


def _result(check_id: str, name: str, status: str, gate: bool,
            detail: Dict[str, Any], violations: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    return {"id": check_id, "name": name, "status": status, "gate": gate,
            "detail": detail, "violations": list(violations)[:25]}


def check_grounding(nodes: Sequence[Dict[str, Any]], units: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = {
        "{}#{}".format(u["scene_id"], b.get("order"))
        for u in units for b in u.get("beats") or []
    }
    total = dangling = ungrounded = 0
    violations = []
    for node in nodes:
        for obj in node["abstraction"]:
            total += 1
            grounds = obj.get("grounded_in") or []
            if not grounds:
                ungrounded += 1
                violations.append({"ao_id": obj["ao_id"], "kind": "ungrounded"})
                continue
            bad = [g for g in grounds if g not in valid]
            if bad:
                dangling += 1
                violations.append({"ao_id": obj["ao_id"], "kind": "dangling", "refs": bad[:3]})
    broken = ungrounded + dangling
    return _result("G1", "grounding", "fail" if broken else "pass", True,
                   {"objects": total, "ungrounded": ungrounded, "dangling": dangling,
                    "grounded_fraction": round(1 - broken / total, 4) if total else 1.0},
                   violations)


def check_not_restatement(nodes: Sequence[Dict[str, Any]], units: Sequence[Dict[str, Any]],
                          *, max_overlap: float = 0.6, max_share: float = 0.15) -> Dict[str, Any]:
    """An abstraction that paraphrases its own grounding beat has added nothing.

    Measured as token overlap between the object's statement and the beats it cites. This is
    the check most worth having, because a restatement is invisible to every other one: it is
    grounded, it is well-formed, it is on-topic, and it carries no new information.
    """
    beat_text = {
        "{}#{}".format(u["scene_id"], b.get("order")): b.get("content") or ""
        for u in units for b in u.get("beats") or []
    }
    total = flagged = 0
    violations = []
    for node in nodes:
        for obj in node["abstraction"]:
            total += 1
            statement = set(_tokens(obj.get("statement")))
            if not statement:
                continue
            worst = 0.0
            for ref in obj.get("grounded_in") or []:
                beat = set(_tokens(beat_text.get(ref, "")))
                if not beat:
                    continue
                worst = max(worst, len(statement & beat) / len(statement))
            if worst >= max_overlap:
                flagged += 1
                violations.append({"ao_id": obj["ao_id"], "overlap": round(worst, 3),
                                   "statement": obj.get("statement", "")[:90]})
    share = flagged / total if total else 0.0
    return _result("G2", "not_restatement",
                   "fail" if share > max_share else ("warn" if flagged else "pass"), True,
                   {"objects": total, "flagged": flagged, "share": round(share, 4),
                    "max_overlap": max_overlap, "max_share": max_share}, violations)


def check_calibration(nodes: Sequence[Dict[str, Any]], *, max_mode_share: float = 0.75) -> Dict[str, Any]:
    counts = Counter(o.get("confidence") for node in nodes for o in node["abstraction"])
    total = sum(counts.values())
    if not total:
        return _result("G3", "calibration", "fail", False, {"objects": 0})
    top, top_n = counts.most_common(1)[0]
    share = top_n / total
    # Second-order theory of mind asserted as near-certain is a calibration failure of its
    # own: the deeper the nesting, the weaker the warrant can possibly be.
    overconfident_tom = [
        o["ao_id"] for node in nodes for o in node["abstraction"]
        if o.get("type") == "theory_of_mind" and o.get("confidence") == "near-certain"
    ]
    return _result("G3", "calibration",
                   "fail" if share > max_mode_share else "pass", False,
                   {"distribution": dict(counts), "modal": top,
                    "modal_share": round(share, 4), "max_mode_share": max_mode_share,
                    "near_certain_theory_of_mind": len(overconfident_tom)},
                   [{"ao_id": a, "kind": "near-certain second-order belief"}
                    for a in overconfident_tom])


def check_contradiction_sought(nodes: Sequence[Dict[str, Any]],
                               *, min_ratio: float = 0.05) -> Dict[str, Any]:
    supporting = contradicting = 0
    for node in nodes:
        for obj in node["abstraction"]:
            supporting += len(obj.get("supporting_evidence") or [])
            contradicting += len(obj.get("contradicting_evidence") or [])
    total = supporting + contradicting
    ratio = contradicting / total if total else 0.0
    return _result("G4", "contradiction_sought",
                   "fail" if total == 0 else ("warn" if ratio < min_ratio else "pass"), False,
                   {"supporting": supporting, "contradicting": contradicting,
                    "contradiction_ratio": round(ratio, 4), "min_ratio": min_ratio,
                    "note": "a researcher that only ever confirms has not searched"})


def check_connectivity(nodes: Sequence[Dict[str, Any]], merge: Dict[str, Any],
                       *, min_linked_share: float = 0.25) -> Dict[str, Any]:
    objects = [o for node in nodes for o in node["abstraction"]]
    ids = {o["ao_id"] for o in objects}
    outgoing = Counter()
    incoming = Counter()
    dangling = []
    for obj in objects:
        for link in obj.get("links") or []:
            outgoing[obj["ao_id"]] += 1
            if link.get("to") in ids:
                incoming[link["to"]] += 1
            else:
                dangling.append({"ao_id": obj["ao_id"], "to": link.get("to")})
    linked = len({a for a in outgoing} | {a for a in incoming})
    share = linked / len(objects) if objects else 0.0
    return _result("G5", "connectivity",
                   "fail" if share < min_linked_share else "pass", False,
                   {"objects": len(objects), "linked_objects": linked,
                    "linked_share": round(share, 4), "total_links": sum(outgoing.values()),
                    "dangling_links": len(dangling),
                    "cross_window_links": merge.get("cross_links", 0),
                    "arcs": merge.get("arc_count", 0)}, dangling)


def check_coverage(nodes: Sequence[Dict[str, Any]], *, min_share: float = 0.9) -> Dict[str, Any]:
    covered = sum(1 for n in nodes if n["abstraction"])
    share = covered / len(nodes) if nodes else 0.0
    empty = [n["scene_id"] for n in nodes if not n["abstraction"]]
    return _result("G6", "scene_coverage",
                   "fail" if share < min_share else "pass", True,
                   {"scenes": len(nodes), "with_abstraction": covered,
                    "share": round(share, 4)},
                   [{"scene_id": s, "kind": "no abstraction objects"} for s in empty])


def check_tom_depth(nodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Theory of mind is the layer's reason to exist; measure whether it is actually there."""
    tom = [o for node in nodes for o in node["abstraction"] if o.get("type") == "theory_of_mind"]
    nested = [o for o in tom if o.get("about") and o.get("subject") != o.get("about")]
    scenes_with_tom = len({o["scene_id"] for o in tom})
    return _result("G7", "theory_of_mind_present",
                   "warn" if not tom else "pass", False,
                   {"theory_of_mind_objects": len(tom),
                    "with_distinct_target": len(nested),
                    "scenes_with_theory_of_mind": scenes_with_tom,
                    "scenes": len(nodes)})


def check_verbatim_overlap(nodes, source: str, *, fail_at: int = 8, warn_at: int = 6):
    """The check this layer was missing entirely, and the reason it was missing is instructive.

    The Perception layer has a hard verbatim gate (C2) because the whole artifact's premise is
    that no source expression is redistributed. Seven checks were written for the Abstraction
    layer and not one of them checked overlap — the property was enforced on layer 1 and
    assumed to survive composition. It did not: the first full run put 15-word verbatim runs
    into `reasoning` fields, because a field asking *why you believe this* invites quoting the
    line that made you believe it.

    A guarantee that holds for a component and is assumed for the composed system is not a
    guarantee. This is now a gate on the same terms as C2.
    """
    words = _tokens(source)
    index = {n: {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}
             for n in range(warn_at, 26)}
    worst = 0
    flagged = []
    for node in nodes:
        for obj in node["abstraction"]:
            for field in ("statement", "reasoning", "falsifier"):
                tokens = _tokens(obj.get(field))
                for n in range(min(25, len(tokens)), warn_at - 1, -1):
                    hit = next((tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)
                                if tuple(tokens[i:i + n]) in index[n]), None)
                    if hit:
                        worst = max(worst, n)
                        if n >= warn_at:
                            flagged.append({"ao_id": obj.get("ao_id"), "field": field,
                                            "length": n, "scene_id": node["scene_id"]})
                        break
    over = [f for f in flagged if f["length"] >= fail_at]
    result = _result("G8", "verbatim_overlap",
                   "fail" if over else ("warn" if flagged else "pass"), True,
                   {"longest_run": worst, "fail_at": fail_at, "warn_at": warn_at,
                    "fields_at_or_above_warn": len(flagged),
                    "fields_at_or_above_fail": len(over)},
                   sorted(flagged, key=lambda f: -f["length"]))
    # The display list is capped at 25 for readability. The repair pass must see all of
    # them: feeding it the truncated list made it fix 25 fields per round forever while the
    # reported count sat at exactly the cap, which looked like non-convergence and was
    # actually the check hiding its own tail from the thing meant to act on it.
    result["all_violations"] = sorted(flagged, key=lambda f: -f["length"])
    return result


def run_ao_checks(nodes: Sequence[Dict[str, Any]], units: Sequence[Dict[str, Any]],
                  merge: Dict[str, Any],
                  negative_cases_ran: Sequence[str] = (),
                  source: str = "") -> Dict[str, Any]:
    results = [
        check_grounding(nodes, units),
        check_not_restatement(nodes, units),
        check_calibration(nodes),
        check_contradiction_sought(nodes),
        check_connectivity(nodes, merge),
        check_coverage(nodes),
        check_tom_depth(nodes),
    ]
    if source:
        results.append(check_verbatim_overlap(nodes, source))
    verified = set(negative_cases_ran)
    for result in results:
        if result["status"] == "pass" and result["id"] not in verified:
            result["status"] = "unverified"
            result["detail"]["note"] = (
                "negative case did not run this release; a check never shown to fail is not a check")
    blocking = [r["id"] for r in results if r["gate"] and r["status"] == "fail"]
    return {"checks": results, "blocking": blocking, "gate_passed": not blocking,
            "negative_cases_ran": sorted(verified)}
