"""Negative cases: every check must be shown to FAIL on data it should reject.

Eight measurement errors in the sibling project all passed silently on data they should
have rejected. A check that has only ever been seen to pass is indistinguishable from a
check that cannot fail. ``run_all`` refuses to report ``pass`` for any check whose id is
not listed in ``negative_cases_ran``, and this module is what earns that listing.

Run: python3 screenplay/tests/run_checks_tests.py   (no third-party deps)
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from screenplay_ku.checks import (  # noqa: E402
    check_canary,
    check_coverage,
    check_fact_fidelity,
    check_non_generic,
    check_referential_integrity,
    check_temporal_order,
    check_verbatim_overlap,
    run_all,
)
from screenplay_ku.scenes import Scene  # noqa: E402


# A tiny invented source, so the fixtures carry no third-party text.
SOURCE = (
    "INT. HARBOUR OFFICE - NIGHT\n"
    "Ana steps inside. The manifest is open on the desk.\n"
    "ANA\nContainer 47 never reached Pier 9 on the 14th.\n"
    "BORIS\nI moved it because they were watching me.\n"
    "She takes his keycard and goes.\n"
)

SCENE = Scene(
    scene_id="sc-001", index=1, heading_raw="INT. HARBOUR OFFICE - NIGHT", kind="INT",
    location="HARBOUR OFFICE", time_of_day="NIGHT",
    start_char=0, end_char=len(SOURCE), word_count=len(SOURCE.split()),
    speakers=["ANA", "BORIS"],
)
SCENES = [SCENE]
BY_ID = {"sc-001": SCENE}


def _unit():
    return {
        "scene_id": "sc-001",
        "context_before": "Ana has been tracing a container missing from the harbour manifest for two days.",
        "context_after": "Ana now holds the keycard and a destination, which lets her reach the pier first.",
        "style": "A cramped two-hander staged so the audience sees Boris lying before Ana does.",
        "present": ["ana_reyes", "boris_kane"],
        "referenced": [],
        "beats": [
            {
                "order": 1, "type": "action", "actor": "ana_reyes", "addressee": None,
                "content": "Ana entered the harbour office where the manifest lay open.",
                "facts": {"quantities": [], "dates": [], "proper_nouns": ["Ana"],
                          "locations": ["HARBOUR OFFICE"]},
                "state_changes": [], "causes": [], "certainty": "stated",
            },
            {
                "order": 2, "type": "speech", "actor": "ana_reyes", "addressee": "boris_kane",
                "content": "Ana stated that Container 47 had failed to arrive at Pier 9 on the 14th.",
                "facts": {"quantities": ["47"], "dates": ["the 14th"],
                          "proper_nouns": ["Ana", "Pier 9"], "locations": ["Pier 9"]},
                "state_changes": [], "causes": [1], "certainty": "stated",
            },
            {
                "order": 3, "type": "revelation", "actor": "boris_kane", "addressee": "ana_reyes",
                "content": "Boris admitted relocating the container, saying he believed he was under observation.",
                "facts": {"quantities": [], "dates": [], "proper_nouns": ["Boris"], "locations": []},
                "state_changes": [{"entity": "boris_kane", "field": "status.complicity",
                                   "from": "suspected", "to": "admitted"}],
                "causes": [2], "certainty": "stated",
            },
        ],
        "entities": [
            {"name": "Ana Reyes", "entity_id": "ana_reyes", "type": "person", "aliases": ["Ana"],
             "attributes": {"role": "investigator"}, "relationships": []},
            {"name": "Boris Kane", "entity_id": "boris_kane", "type": "person", "aliases": ["Boris"],
             "attributes": {"role": "clerk"}, "relationships": []},
        ],
    }


def _units():
    return [_unit()]


# ------------------------------------------------------------------ baseline


def test_clean_fixture_passes_every_check():
    """If the baseline does not pass, every negative case below proves nothing."""
    for result in (
        check_coverage(_units(), SCENES),
        check_verbatim_overlap(_units(), BY_ID, SOURCE),
        check_fact_fidelity(_units(), BY_ID, SOURCE),
        check_temporal_order(_units(), SCENES),
        check_referential_integrity(_units()),
        check_non_generic(_units()),
        check_canary({"emitted": 0}),
    ):
        assert result.status == "pass", (result.check_id, result.status, result.detail, result.violations)


# ----------------------------------------------------------------- C1 coverage


def test_c1_rejects_missing_unit():
    assert check_coverage([], SCENES).status == "fail"


def test_c1_rejects_duplicate_unit():
    assert check_coverage(_units() + _units(), SCENES).status == "fail"


def test_c1_rejects_unknown_scene():
    units = _units()
    units[0]["scene_id"] = "sc-999"
    assert check_coverage(units, SCENES).status == "fail"


# --------------------------------------------------------- C2 verbatim overlap


def test_c2_rejects_copied_dialogue():
    units = _units()
    units[0]["beats"][1]["content"] = (
        "Ana said Container 47 never reached Pier 9 on the 14th and that was that."
    )
    result = check_verbatim_overlap(units, BY_ID, SOURCE)
    assert result.status == "fail", result.detail


def test_c2_rejects_copied_action_line():
    units = _units()
    units[0]["style"] = "She takes his keycard and goes, which is how the scene ends."
    assert check_verbatim_overlap(units, BY_ID, SOURCE, fail_at=6).status == "fail"


def test_c2_exemption_cannot_hide_a_copied_span():
    """Proper nouns are exempt from counting, but must not mask the span around them."""
    units = _units()
    units[0]["beats"][0]["content"] = "I moved it because they were watching me right then."
    assert check_verbatim_overlap(units, BY_ID, SOURCE).status == "fail"


def test_c2_passes_genuine_paraphrase():
    assert check_verbatim_overlap(_units(), BY_ID, SOURCE).status == "pass"


# ------------------------------------------------------------ C3 fact fidelity


def test_c3_rejects_drifted_number():
    units = _units()
    units[0]["beats"][1]["content"] = units[0]["beats"][1]["content"].replace("47", "48")
    units[0]["beats"][1]["facts"]["quantities"] = ["48"]
    assert check_fact_fidelity(units, BY_ID, SOURCE).status == "fail"


def test_c3_rejects_dropped_character_name():
    units = _units()
    for beat in units[0]["beats"]:
        beat["content"] = beat["content"].replace("Boris", "the clerk")
    units[0]["context_before"] = units[0]["context_after"] = units[0]["style"] = "x" * 40
    assert check_fact_fidelity(units, BY_ID, SOURCE).status in ("fail", "warn")


# --------------------------------------------------------- C4 temporal order


def test_c4_rejects_duplicate_beat_order():
    units = _units()
    units[0]["beats"][2]["order"] = 2
    assert check_temporal_order(units, SCENES).status == "fail"


def test_c4_rejects_gap_in_beat_order():
    units = _units()
    units[0]["beats"][2]["order"] = 4
    assert check_temporal_order(units, SCENES).status == "fail"


def test_c4_rejects_forward_or_cyclic_cause():
    units = _units()
    units[0]["beats"][0]["causes"] = [3]
    assert check_temporal_order(units, SCENES).status == "fail"


# ---------------------------------------------------- C5 referential integrity


def test_c5_rejects_dangling_actor():
    units = _units()
    units[0]["beats"][0]["actor"] = "nobody_at_all"
    assert check_referential_integrity(units).status == "fail"


def test_c5_rejects_dangling_state_change_entity():
    units = _units()
    units[0]["beats"][2]["state_changes"][0]["entity"] = "ghost_id"
    assert check_referential_integrity(units).status == "fail"


# -------------------------------------------------------- C6 non-genericity


def _many(n, mutate=None):
    out = []
    for i in range(n):
        unit = copy.deepcopy(_unit())
        unit["scene_id"] = "sc-%03d" % (i + 1)
        if mutate:
            mutate(unit, i)
        else:
            # Genuinely varied vocabulary, not one template with a counter substituted.
            # A shared template scores high pairwise Jaccard and would make C6 fail here,
            # which is correct behaviour from the check and a bad fixture.
            unit["context_before"] = _VARIED_BEFORE[i % len(_VARIED_BEFORE)]
            unit["context_after"] = _VARIED_AFTER[i % len(_VARIED_AFTER)]
            unit["style"] = _VARIED_STYLE[i % len(_VARIED_STYLE)]
        out.append(unit)
    return out


_VARIED_BEFORE = [
    "Nobody aboard yet suspects the quartermaster falsified last winter's inventory.",
    "Rain has closed every northern road, stranding the delegation outside the city.",
    "A debt collector waits in the lobby, and the family cannot pay him.",
    "The radio signal died four hours ago and no relief has arrived.",
    "Two sisters have not spoken since their father changed his will.",
    "An unexploded shell sits under the schoolhouse, known only to the caretaker.",
    "The vote is tied, and one alderman has gone missing since dawn.",
    "Frost killed the seedlings, so the greenhouse is the last surviving crop.",
    "A forged passport is sewn into the lining of a coat nobody has claimed.",
    "The lighthouse keeper's replacement was due on Tuesday and never came.",
    "Someone emptied the safe during the funeral, and only six people had keys.",
    "A translator has been feeding both delegations different versions of the treaty.",
]

_VARIED_AFTER = [
    "The falsified ledger now circulates, and three officers can be blackmailed with it.",
    "With the roads shut, negotiations must happen in a barn nobody controls.",
    "Paying the collector empties the account meant for passage abroad.",
    "Silence on the radio forces the crew to decide without orders.",
    "The younger sister leaves, taking the only copy of the deed.",
    "Evacuating the school reveals what the caretaker has been hiding for years.",
    "The missing alderman's whereabouts become the only thing that matters.",
    "Whoever controls the greenhouse controls what the settlement eats.",
    "The unclaimed coat gives its finder a second identity.",
    "An unmanned light means the next ship in has no warning.",
    "Suspicion narrows to six keyholders, one of whom is the investigator.",
    "Both delegations sign documents that do not say the same thing.",
]

_VARIED_STYLE = [
    "Handheld and cramped, refusing any establishing shot of the room.",
    "Long static wide, letting the argument play out at a distance.",
    "Cut on glances rather than lines, so the audience tracks who avoids whom.",
    "Sound carries the scene; the camera stays on an unrelated object throughout.",
    "A single unbroken take that follows one character through four conversations.",
    "Intercut between two timelines without signalling which is which.",
    "Shot entirely in reflections, never showing a face directly.",
    "Near-silent, with dialogue withheld until the final beat.",
    "Overlapping speech and a moving camera, deliberately hard to follow.",
    "Locked-off frame, characters entering and leaving an empty composition.",
    "Extreme close-ups on hands, withholding the room until the last shot.",
    "Told from the eavesdropper's position, hearing only half of it.",
]


def test_c6_rejects_one_repeated_style_everywhere():
    def same(unit, i):
        unit["style"] = "In this scene, the story continues as before."
    assert check_non_generic(_many(30, same)).status == "fail"


def test_c6_rejects_two_alternating_strings():
    def alternate(unit, i):
        unit["context_before"] = ["The story continues onward from here.",
                                  "Events proceed as they were before."][i % 2]
    assert check_non_generic(_many(30, alternate)).status == "fail"


def test_c6_passes_varied_fields():
    # One unit per distinct string: the exact-repeat rule fires at >2 occurrences, so
    # cycling 12 strings over 30 units would trip it and prove nothing about similarity.
    assert check_non_generic(_many(len(_VARIED_STYLE))).status == "pass"


# ------------------------------------------------------------------ C7 canary


def test_c7_fails_when_canary_emits_anything():
    assert check_canary({"emitted": 1}).status == "fail"


def test_c7_unverified_when_canary_never_ran():
    assert check_canary(None).status == "unverified"


# ------------------------------------------------------- the gate on the gate


def test_run_all_refuses_to_pass_unverified_checks():
    """A check whose negative case did not run must not be reported as passing."""
    report = run_all(_units(), SCENES, SOURCE, canary_result={"emitted": 0})
    assert all(check["status"] != "pass" for check in report["checks"])
    assert all(check["status"] == "unverified" for check in report["checks"])


def test_run_all_reports_pass_once_negative_cases_are_declared():
    report = run_all(
        _units(), SCENES, SOURCE, canary_result={"emitted": 0},
        negative_cases_ran=["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
    )
    assert report["gate_passed"] is True
    assert all(check["status"] == "pass" for check in report["checks"])


def test_run_all_blocks_on_a_gate_failure():
    units = _units()
    units[0]["beats"][1]["content"] = "Ana said Container 47 never reached Pier 9 on the 14th."
    report = run_all(units, SCENES, SOURCE, canary_result={"emitted": 0},
                     negative_cases_ran=["C1", "C2", "C3", "C4", "C5", "C6", "C7"])
    assert report["gate_passed"] is False
    assert "C2" in report["blocking"]
