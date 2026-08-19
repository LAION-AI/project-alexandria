"""Checks C1-C7, each with a negative case in ``tests/test_checks_negative.py``.

The sibling project accumulated eight measurement errors and its own apparatus caught none
of them; six were found by an outside reader. Every one produced a confident number over
the wrong thing and none failed loudly. The rule earned from that record:

    A check that has never been shown to fail is not a check.

So no check here reports ``passed`` unless its negative case ran. ``run_all`` marks a check
``unverified`` otherwise, and ``unverified`` is not a pass.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .scenes import Scene


# --------------------------------------------------------------------- result


@dataclass
class CheckResult:
    check_id: str
    name: str
    status: str  # pass | fail | warn | unverified
    gate: bool
    detail: Dict[str, Any] = field(default_factory=dict)
    violations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        return self.gate and self.status == "fail"


_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> List[str]:
    return _WORD.findall(text.casefold())


def _text_fields(unit: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Every model-authored string in a unit, with a path for reporting."""
    out: List[Tuple[str, str]] = []
    for key in ("context_before", "context_after", "style"):
        if isinstance(unit.get(key), str):
            out.append((key, unit[key]))
    for beat in unit.get("beats") or []:
        order = beat.get("order")
        if isinstance(beat.get("content"), str):
            out.append(("beats[{}].content".format(order), beat["content"]))
        for change in beat.get("state_changes") or []:
            for sub in ("field", "from", "to"):
                if isinstance(change.get(sub), str):
                    out.append(("beats[{}].state_changes.{}".format(order, sub), change[sub]))
    for entity in unit.get("entities") or []:
        eid = entity.get("entity_id")
        for value in (entity.get("attributes") or {}).values():
            if isinstance(value, str):
                out.append(("entities[{}].attributes".format(eid), value))
        for relationship in entity.get("relationships") or []:
            if isinstance(relationship.get("target"), str):
                out.append(("entities[{}].relationships.target".format(eid), relationship["target"]))
    return out


# ------------------------------------------------------------------ C1 coverage


def check_coverage(units: Sequence[Dict], scenes: Sequence[Scene]) -> CheckResult:
    expected = [scene.scene_id for scene in scenes]
    got = [unit.get("scene_id") for unit in units]
    counts = Counter(got)
    missing = sorted(set(expected) - set(got))
    unknown = sorted(set(got) - set(expected))
    duplicated = sorted(sid for sid, n in counts.items() if n > 1)
    violations = (
        [{"kind": "missing", "scene_id": s} for s in missing]
        + [{"kind": "unknown", "scene_id": s} for s in unknown]
        + [{"kind": "duplicate", "scene_id": s, "count": counts[s]} for s in duplicated]
    )
    return CheckResult(
        "C1", "coverage", "fail" if violations else "pass", gate=True,
        detail={"expected": len(expected), "got": len(got),
                "missing": len(missing), "unknown": len(unknown), "duplicated": len(duplicated)},
        violations=violations[:50],
    )


# ---------------------------------------------------- C2 verbatim overlap (gate)


def _ngram_index(tokens: Sequence[str], low: int, high: int) -> Dict[int, Set[Tuple[str, ...]]]:
    index: Dict[int, Set[Tuple[str, ...]]] = {}
    for n in range(low, high + 1):
        index[n] = {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
    return index


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "as", "that", "this", "it", "its", "is", "was", "are", "were", "be",
    "been", "he", "she", "they", "them", "his", "her", "their", "him", "i", "you", "we",
    "not", "no", "so", "then", "there", "here", "up", "out", "into", "over", "about",
}


def _exempt_tokens(scene: Scene, unit: Dict[str, Any]) -> Set[str]:
    """Tokens C3 requires to survive verbatim, which must not by themselves prove copying.

    These are **not** masked out before matching. Masking was the first implementation and
    it was wrong: exempting the tokens of "the 14th" exempted the word "the", and a copied
    sentence containing any required fact then had no unmasked n-gram left to match on. The
    negative case ``test_c2_rejects_copied_dialogue`` caught it. A whole copied line was
    passing the gate that exists to stop exactly that.

    Instead the match runs unmasked and this set is used only to *adjudicate* it: see
    ``_novel_token_count``. A phrase built solely from required facts and stopwords is not
    copying; the same phrase plus three ordinary words is.
    """
    exempt: Set[str] = set()
    for speaker in scene.speakers:
        exempt.update(_tokens(speaker))
    exempt.update(_tokens(scene.location or ""))
    exempt.update(_tokens(scene.heading_raw or ""))
    for beat in unit.get("beats") or []:
        facts = beat.get("facts") or {}
        for key in ("quantities", "dates", "proper_nouns", "locations"):
            for value in facts.get(key) or []:
                exempt.update(_tokens(str(value)))
    return exempt


def _novel_token_count(gram: Sequence[str], exempt: Set[str]) -> int:
    """Tokens in a matched span that are neither required-verbatim facts nor stopwords.

    This is what separates "preserved a fact exactly, as required" from "reproduced the
    source's wording". Reported with every violation so the judgement is auditable.
    """
    return sum(1 for token in gram if token not in exempt and token not in _STOPWORDS)


def check_verbatim_overlap(
    units: Sequence[Dict],
    scenes_by_id: Dict[str, Scene],
    source: str,
    *,
    fail_at: int = 8,
    warn_at: int = 6,
    max_n: int = 25,
    min_novel_tokens: int = 3,
) -> CheckResult:
    """Longest contiguous word n-gram shared between a unit's prose and the source.

    A gate, not a metric: on failure the run emits no artifact. This is what actually
    enforces indirect speech — the prompt instruction alone is advisory.

    Matching is unmasked. A match is then *adjudicated*: it counts as copying only if it
    carries at least ``min_novel_tokens`` tokens that are neither required-verbatim facts
    (C3) nor stopwords. That keeps "Pier 9 on the 14th" — which C3 demands be preserved
    exactly — from registering as copied, without letting a full copied sentence hide
    behind the facts embedded in it.
    """
    document_tokens = _tokens(source)
    document_index = _ngram_index(document_tokens, warn_at, max_n)

    worst: List[Dict[str, Any]] = []
    longest_overall = 0
    longest_adjudicated = 0
    exempt_total = 0
    distribution: Counter = Counter()
    facts_only_matches = 0

    for unit in units:
        scene = scenes_by_id.get(unit.get("scene_id"))
        if scene is None:
            continue
        scene_tokens = _tokens(scene.text(source))
        scene_index = _ngram_index(scene_tokens, warn_at, max_n)
        exempt = _exempt_tokens(scene, unit)
        exempt_total += len(exempt)

        for path, value in _text_fields(unit):
            tokens = _tokens(value)
            best = 0
            best_gram: Tuple[str, ...] = ()
            scope = ""
            for n in range(min(max_n, len(tokens)), warn_at - 1, -1):
                if n <= best:
                    break
                for i in range(len(tokens) - n + 1):
                    gram = tuple(tokens[i : i + n])
                    if gram in scene_index.get(n, ()):
                        best, best_gram, scope = n, gram, "scene"
                        break
                    if gram in document_index.get(n, ()):
                        best, best_gram, scope = n, gram, "document"
                        break
                if best:
                    break
            if not best:
                continue
            longest_overall = max(longest_overall, best)
            novel = _novel_token_count(best_gram, exempt)
            if novel < min_novel_tokens:
                facts_only_matches += 1
                continue
            distribution[best] += 1
            longest_adjudicated = max(longest_adjudicated, best)
            worst.append({
                "scene_id": unit.get("scene_id"), "path": path,
                "length": best, "novel_tokens": novel, "scope": scope,
                "ngram": " ".join(best_gram),
            })

    worst.sort(key=lambda item: (-item["length"], -item["novel_tokens"]))
    failures = [item for item in worst if item["length"] >= fail_at]
    status = "fail" if failures else ("warn" if worst else "pass")
    return CheckResult(
        "C2", "verbatim_overlap", status, gate=True,
        detail={
            "fail_at": fail_at, "warn_at": warn_at, "min_novel_tokens": min_novel_tokens,
            "longest_ngram_raw": longest_overall,
            "longest_ngram_adjudicated": longest_adjudicated,
            "fields_at_or_above_warn": len(worst),
            "fields_at_or_above_fail": len(failures),
            "matches_dismissed_as_facts_only": facts_only_matches,
            "length_histogram": dict(sorted(distribution.items())),
            "mean_exempt_tokens_per_unit": round(exempt_total / max(1, len(units)), 1),
        },
        violations=worst[:20],
    )


# ------------------------------------------------------------- C3 fact fidelity


_NUMBER = re.compile(r"\b\d[\d,.:]*\b")

# Speaker cues that are roles, not names. A screenplay labels an unnamed character by
# function, and a unit that says "a police officer" rather than "COP" has recorded the
# fact correctly. Requiring these verbatim measured the check's own over-reach: on the
# first dry run every one of C3's name misses was one of these or a parse artifact.
_GENERIC_SPEAKER_ROLES = {
    "man", "woman", "boy", "girl", "cop", "big cop", "guard", "officer", "voice",
    "lieutenant", "pilot", "priestess", "operator", "agent", "guard #4", "fedex guy",
    "spoon boy", "monitor", "child", "driver", "soldier", "nurse", "doctor", "clerk",
}


def _is_requirable_name(speaker: str) -> bool:
    """A speaker cue counts as a proper name only if it plausibly is one.

    The scene map's speaker list is derived by a parser and contains residue such as
    ``CLICK.`` and ``MONITORS SNAP FLATLINE.``. Requiring those to appear in a unit would
    fail the run for the parser's mistakes rather than the model's.
    """
    cleaned = speaker.strip()
    if not cleaned or cleaned.endswith((".", "-", "--")):
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z .'#-]*", cleaned):
        return False
    if len(cleaned.split()) > 2:
        return False
    return cleaned.casefold() not in _GENERIC_SPEAKER_ROLES


# A composite numeric string: a date, phone number, time, or code. Matched whole, because
# splitting it produces fragments that are not independently meaningful facts.
_COMPOSITE_NUMBER = re.compile(r"\b\d[\d]*(?:[-/:.]\d+)+\b")


def _requirable_numbers(scene: Scene, source: str) -> Set[str]:
    """Numbers that are facts, excluding structural residue.

    The pre-slugline head carries the title block and scene numbering, whose digits are
    typography rather than content, and a scene's heading line can carry a scene number.

    Composite strings are kept whole. Splitting a date on its hyphens demanded three
    separate digit fragments and scored a unit as having lost the fact when it had merely
    written the date in another notation: 11 of 16 number misses on the first full run were
    this, and the check was reporting its own tokenizer rather than the model.
    """
    if scene.kind == "PRE":
        return set()
    text = scene.text(source)
    body = "\n".join(text.splitlines()[1:]) if "\n" in text else text
    masked = _mask_production_furniture(body, stamps=_revision_stamps(source))

    composites = set(_COMPOSITE_NUMBER.findall(masked))
    remainder = _COMPOSITE_NUMBER.sub(" ", masked)
    return composites | set(_NUMBER.findall(remainder))


_DATE_STAMP = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")

# Lines that are production furniture rather than content. Matched whole-line, because
# masking a fixed number of trailing characters cut through the middle of a scene number
# and left fragments: "119OMITTED 119" became "11", which the check then demanded as a
# fact in its own right. A partial mask is worse than none — it invents a requirement.
_FURNITURE_LINE = re.compile(
    r"""^\s*(?:
          \d+\s*OMITTED\s*\d*          # 119OMITTED 119
        | OMITTED\s*\d*
        | \(?\s*(?:CONTINUED|MORE)\s*\)?\s*\d*
        | [-\s]*Rev\.?\s.*             # - Rev. 3/9/98 125A.
        | \d+[A-Z]?\.?                 # a bare scene number
        )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _revision_stamps(source: str, min_occurrences: int = 2) -> Set[str]:
    """Date strings that recur across a shooting script are page furniture, not facts.

    A production draft stamps revision dates into its page headers. A date that is part of
    the story appears once, in dialogue or action; a date printed on every revised page
    appears many times in identical form.
    """
    counts = Counter(_DATE_STAMP.findall(source))
    return {value for value, count in counts.items() if count >= min_occurrences}


def _mask_production_furniture(body: str, stamps: Set[str]) -> str:
    """Drop the lines of a scene that are typography rather than content.

    Two kinds, both confirmed by position on a full run: scene numbers and OMITTED markers
    printed in a scene's margin, and page-break headers carrying a revision stamp.
    Requiring a unit to reproduce either scored the model for failing to record the page
    furniture of a 1998 shooting script.

    Whole lines are dropped rather than a fixed span of characters. The character version
    of this cut through "119OMITTED 119" and left "11", which the check then demanded as a
    fact — a mask that truncates does not remove a spurious requirement, it invents a new
    one that no faithful unit can satisfy.
    """
    kept = []
    for line in body.splitlines():
        if _FURNITURE_LINE.match(line):
            continue
        if any(stamp in line for stamp in stamps):
            continue
        kept.append(_strip_scene_number_wrapper(line))
    return "\n".join(kept)


_NUMBER_WRAPPED = re.compile(r"^\s*(\d+)([A-Z].*?)\1\s*$")


def _strip_scene_number_wrapper(line: str) -> str:
    """Remove a scene number printed either side of a slugline.

    A numbered shooting script prints the scene number in both margins, so a slugline
    arrives as ``219CLOSE ON COMPUTER SCREEN 219``. The number is typography; the text
    between is content and must survive.
    """
    match = _NUMBER_WRAPPED.match(line)
    return match.group(2) if match else line


def _number_satisfied(required: str, blob: str, blob_numbers: Set[str]) -> bool:
    """A composite is satisfied by itself or by its parts appearing together.

    ``2-19-98`` may legitimately be recorded as ``19 February 1998``. Requiring the exact
    punctuation would fail a faithful restatement, and requiring nothing would let the fact
    vanish, so the test is that every numeric component is present.
    """
    if required in blob_numbers:
        return True
    parts = [part for part in re.split(r"[-/:.]", required) if part]
    if len(parts) > 1:
        return all(part in blob_numbers for part in parts)
    return False


def check_fact_fidelity(
    units: Sequence[Dict],
    scenes_by_id: Dict[str, Scene],
    source: str,
    *,
    min_number_recall: float = 0.95,
) -> CheckResult:
    """Numbers, speaker names, and slugline locations from the source must reappear.

    Extracted from the **source text** by a deterministic pass, never from the units — a
    check that grades against a list its own apparatus generated measures compliance.

    What counts as *required* is narrowed deliberately: see ``_is_requirable_name`` and
    ``_requirable_numbers``. The first dry run failed this check 5 times and all 5 were the
    check's fault, which is the reason the narrowing exists and is documented here rather
    than tuned silently into a threshold.
    """
    number_hit = number_total = 0
    name_hit = name_total = 0
    location_hit = location_total = 0
    violations: List[Dict[str, Any]] = []

    for unit in units:
        scene = scenes_by_id.get(unit.get("scene_id"))
        if scene is None:
            continue
        # The heading is part of the unit, so a location preserved in `heading.location`
        # is preserved. Searching only the prose fields reported 20 locations as lost that
        # were sitting in the structured field the schema created for them — the check was
        # looking in the wrong place, not the model failing to record them.
        heading = unit.get("heading") or {}
        blob = " ".join(
            [value for _, value in _text_fields(unit)]
            + [str(heading.get("raw") or ""), str(heading.get("location") or "")]
        )
        blob_tokens = set(_tokens(blob))
        blob_numbers = set(_NUMBER.findall(blob))

        for number in _requirable_numbers(scene, source):
            number_total += 1
            if _number_satisfied(number, blob, blob_numbers):
                number_hit += 1
            else:
                violations.append({"kind": "number", "scene_id": scene.scene_id, "value": number})

        for speaker in set(scene.speakers):
            if not _is_requirable_name(speaker):
                continue
            head = _tokens(speaker)
            if not head:
                continue
            name_total += 1
            if any(token in blob_tokens for token in head):
                name_hit += 1
            else:
                violations.append({"kind": "name", "scene_id": scene.scene_id, "value": speaker})

        if scene.location:
            location_total += 1
            if any(token in blob_tokens for token in _tokens(scene.location)):
                location_hit += 1
            else:
                violations.append({"kind": "location", "scene_id": scene.scene_id,
                                   "value": scene.location})

    number_recall = number_hit / number_total if number_total else 1.0
    corrupted = _drifted_numbers(units, scenes_by_id, source)
    violations.extend(corrupted)

    # Two different failures, with different consequences, so they are gated differently.
    #
    # A *drifted* number is corruption: the unit asserts something the source contradicts,
    # and a reader has no way to detect it. That is what this gate exists to stop, and any
    # occurrence fails the run.
    #
    # An *omitted* number is incompleteness: the unit is silent, the reader is not misled,
    # and the fact is recoverable from the source by offset. It is reported as a rate and
    # warns, rather than blocking an otherwise sound artifact.
    #
    # Collapsing the two under one recall threshold made the gate hostage to whatever the
    # denominator happened to contain, which for three runs was the script's own page
    # furniture rather than any fact.
    status = "fail" if corrupted else ("warn" if number_recall < min_number_recall or violations else "pass")
    return CheckResult(
        "C3", "fact_fidelity", status, gate=True,
        detail={
            "numbers_drifted": len(corrupted),
            "number_recall": round(number_recall, 4), "numbers": number_total,
            "name_recall": round(name_hit / name_total, 4) if name_total else None,
            "names": name_total,
            "location_recall": round(location_hit / location_total, 4) if location_total else None,
            "locations": location_total,
            "min_number_recall": min_number_recall,
            "gate": "drift fails the run; omission warns",
        },
        violations=violations[:40],
    )


def _drifted_numbers(
    units: Sequence[Dict], scenes_by_id: Dict[str, Scene], source: str
) -> List[Dict[str, Any]]:
    """Numbers a unit asserts that its scene contradicts.

    Restricted to near variants of a real source number — same digit length, differing in
    one position. An unrelated number is usually the model writing a count in digits that
    the source spelled out, which is not corruption; a one-digit change to a number the
    scene actually contains is.
    """
    found: List[Dict[str, Any]] = []
    for unit in units:
        scene = scenes_by_id.get(unit.get("scene_id"))
        if scene is None:
            continue
        source_numbers = set(_NUMBER.findall(scene.text(source)))
        stated = set()
        for beat in unit.get("beats") or []:
            for value in (beat.get("facts") or {}).get("quantities") or []:
                stated.update(_NUMBER.findall(str(value)))
        for value in stated - source_numbers:
            # Single digits carry no evidence of drift: every digit differs from every
            # other in exactly one position, so the rule below would flag any lone number
            # the scene did not literally contain. On the first run this fired on "1 hour",
            # a correctly recorded quantity, because some unrelated digit sat in the scene.
            if len(value) < 2:
                continue
            for original in source_numbers:
                if len(original) == len(value) and sum(
                    1 for a, b in zip(original, value) if a != b
                ) == 1:
                    found.append({
                        "kind": "number_drift", "scene_id": scene.scene_id,
                        "value": value, "source_value": original,
                    })
                    break
    return found


# ---------------------------------------------------------- C4 temporal totality


def check_temporal_order(units: Sequence[Dict], scenes: Sequence[Scene]) -> CheckResult:
    order_by_scene = {scene.scene_id: scene.index for scene in scenes}
    violations: List[Dict[str, Any]] = []
    pairs: Set[Tuple[int, int]] = set()

    for unit in units:
        sid = unit.get("scene_id")
        sindex = order_by_scene.get(sid)
        beats = unit.get("beats") or []
        orders = [beat.get("order") for beat in beats]
        if orders != list(range(1, len(beats) + 1)):
            violations.append({"kind": "beat_order", "scene_id": sid, "orders": orders})
        for beat in beats:
            key = (sindex, beat.get("order"))
            if key in pairs:
                violations.append({"kind": "duplicate_pair", "scene_id": sid, "order": beat.get("order")})
            pairs.add(key)
            for cause in beat.get("causes") or []:
                if not isinstance(cause, int) or cause >= (beat.get("order") or 0) or cause < 1:
                    violations.append({"kind": "bad_cause", "scene_id": sid,
                                       "order": beat.get("order"), "cause": cause})
    total_beats = sum(len(unit.get("beats") or []) for unit in units)
    return CheckResult(
        "C4", "temporal_totality", "fail" if violations else "pass", gate=True,
        detail={"beats": total_beats, "distinct_order_keys": len(pairs),
                "total_order_is_strict": len(pairs) == total_beats},
        violations=violations[:40],
    )


# -------------------------------------------------------- C5 referential integrity


def check_referential_integrity(units: Sequence[Dict]) -> CheckResult:
    declared: Set[str] = set()
    for unit in units:
        for entity in unit.get("entities") or []:
            if entity.get("entity_id"):
                declared.add(entity["entity_id"])

    violations: List[Dict[str, Any]] = []
    referenced: Set[str] = set()

    def note(scene_id: str, path: str, value: Optional[str]) -> None:
        if not value:
            return
        referenced.add(value)
        if value not in declared:
            violations.append({"scene_id": scene_id, "path": path, "entity_id": value})

    for unit in units:
        sid = unit.get("scene_id")
        for value in unit.get("present") or []:
            note(sid, "present", value)
        for value in unit.get("referenced") or []:
            note(sid, "referenced", value)
        for beat in unit.get("beats") or []:
            note(sid, "beats[{}].actor".format(beat.get("order")), beat.get("actor"))
            note(sid, "beats[{}].addressee".format(beat.get("order")), beat.get("addressee"))
            for change in beat.get("state_changes") or []:
                note(sid, "beats[{}].state_changes.entity".format(beat.get("order")),
                     change.get("entity"))
        for entity in unit.get("entities") or []:
            for relationship in entity.get("relationships") or []:
                note(sid, "relationships.target_id", relationship.get("target_id"))

    # Reference closure synthesizes empty records for ids the model used but never
    # declared. That keeps the graph closed, but it must not be invisible: an artifact
    # where most entities were auto-declared has a real quality problem that a green
    # "referential integrity: pass" would hide.
    auto_declared = sum(
        1
        for unit in units
        for entity in unit.get("entities") or []
        if (entity.get("attributes") or {}).get("declared_by") == "reference_closure"
    )
    total_entity_records = sum(len(unit.get("entities") or []) for unit in units)
    auto_share = auto_declared / total_entity_records if total_entity_records else 0.0
    return CheckResult(
        "C5", "referential_integrity",
        "fail" if violations else ("warn" if auto_share > 0.5 else "pass"), gate=False,
        detail={"declared_entity_ids": len(declared), "referenced_entity_ids": len(referenced),
                "dangling": len(violations),
                "auto_declared_records": auto_declared,
                "entity_records": total_entity_records,
                "auto_declared_share": round(auto_share, 3)},
        violations=violations[:40],
    )


# ------------------------------------------------------------- C6 non-genericity


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def check_non_generic(units: Sequence[Dict], *, max_median_similarity: float = 0.45) -> CheckResult:
    """A model that has stopped reading emits the same sentence everywhere.

    Presence checks would wave that through, which is exactly why this exists.
    """
    detail: Dict[str, Any] = {}
    violations: List[Dict[str, Any]] = []
    worst_median = 0.0

    for key in ("context_before", "context_after", "style"):
        values = [(unit.get("scene_id"), unit.get(key) or "") for unit in units]
        exact = Counter(value for _, value in values if value)
        repeats = [{"kind": "exact_repeat", "field": key, "count": count, "text": text[:80]}
                   for text, count in exact.items() if count > 2]
        violations.extend(repeats)

        sets = [set(_tokens(value)) for _, value in values]
        sims = []
        step = max(1, len(sets) // 60)  # sample pairs; full N^2 is not needed for a median
        for i in range(0, len(sets), step):
            for j in range(i + 1, min(i + 12, len(sets))):
                sims.append(_jaccard(sets[i], sets[j]))
        sims.sort()
        median = sims[len(sims) // 2] if sims else 0.0
        worst_median = max(worst_median, median)
        detail[key] = {"median_pairwise_jaccard": round(median, 3),
                       "exact_repeats_over_2": len(repeats),
                       "distinct": len(exact), "units": len(values)}

    status = "fail" if (worst_median > max_median_similarity or violations) else "pass"
    return CheckResult(
        "C6", "non_genericity", status, gate=False,
        detail={**detail, "max_median_similarity": max_median_similarity,
                "worst_median": round(worst_median, 3)},
        violations=violations[:20],
    )


# ------------------------------------------------------------------- C7 canary


def check_canary(canary_result: Optional[Dict[str, Any]]) -> CheckResult:
    if canary_result is None:
        return CheckResult("C7", "canary", "unverified", gate=True,
                           detail={"reason": "canary was not dispatched"})
    emitted = int(canary_result.get("emitted", 0))
    return CheckResult(
        "C7", "canary", "pass" if emitted == 0 else "fail", gate=True,
        detail={"units_emitted": emitted,
                "meaning": "anything emitted is recall from context, not extraction"},
        violations=[{"kind": "recall", "count": emitted}] if emitted else [],
    )


# ---------------------------------------------------------------------- driver


def run_all(
    units: Sequence[Dict],
    scenes: Sequence[Scene],
    source: str,
    *,
    canary_result: Optional[Dict[str, Any]] = None,
    negative_cases_ran: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    scenes_by_id = {scene.scene_id: scene for scene in scenes}
    results = [
        check_coverage(units, scenes),
        check_verbatim_overlap(units, scenes_by_id, source),
        check_fact_fidelity(units, scenes_by_id, source),
        check_temporal_order(units, scenes),
        check_referential_integrity(units),
        check_non_generic(units),
        check_canary(canary_result),
    ]
    verified = set(negative_cases_ran or ())
    for result in results:
        if result.status == "pass" and result.check_id not in verified:
            result.status = "unverified"
            result.detail["note"] = (
                "negative case did not run this release; a check never shown to fail is not a check"
            )
    blocking = [result.check_id for result in results if result.blocking]
    return {
        "checks": [result.__dict__ for result in results],
        "blocking": blocking,
        "gate_passed": not blocking,
        "negative_cases_ran": sorted(verified),
    }
