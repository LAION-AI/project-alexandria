"""Negative cases for the abstraction-layer checks.

Same rule as the Perception layer: a check that has never been shown to fail is not a check,
and `run_ao_checks` reports `unverified` rather than `pass` for any check whose negative case
did not run.

Run: python3 screenplay/tests/run_cognitino_tests.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from screenplay_ku.cognitino.checks import (  # noqa: E402
    check_calibration,
    check_connectivity,
    check_contradiction_sought,
    check_coverage,
    check_grounding,
    check_not_restatement,
    check_tom_depth,
    run_ao_checks,
)

UNITS = [
    {"scene_id": "sc-001", "scene_index": 1,
     "beats": [
         {"order": 1, "content": "Ana entered the harbour office where the manifest lay open."},
         {"order": 2, "content": "Boris denied knowing where the container had gone."},
         {"order": 3, "content": "Boris admitted diverting it, citing surveillance."},
     ],
     "entities": [{"entity_id": "ana_reyes"}, {"entity_id": "boris_kane"}]},
]

MERGE = {"cross_links": 3, "arc_count": 2}


def _obj(**over):
    base = {
        "ao_id": "ao-1", "type": "mental_state", "scene_id": "sc-001",
        "subject": "boris_kane", "about": None,
        "statement": "Boris arrives already resigned to exposure; the denial is a formality he drops at once.",
        "grounded_in": ["sc-001#2", "sc-001#3"],
        "reasoning": "A denial abandoned after one challenge reads as rehearsed capitulation.",
        "confidence": "probable", "assumptions": [], "falsifier": "Boris sustaining the lie elsewhere.",
        "supporting_evidence": [{"beat_ref": "sc-001#3", "why": "unprompted motive"}],
        "contradicting_evidence": [],
        "links": [{"to": "ao-2", "link": "causes", "why": "drives the concession"}],
    }
    base.update(over)
    return base


def _nodes(objects=None):
    objects = objects if objects is not None else [
        _obj(),
        # Carries contradicting evidence: the clean fixture must exercise the researcher
        # having actually looked for disconfirmation, or G4 correctly reports that it did not.
        _obj(ao_id="ao-2", type="theory_of_mind", subject="boris_kane", about="ana_reyes",
             statement="Boris believes Ana will treat surveillance as mitigating, so he volunteers it.",
             confidence="plausible", links=[],
             contradicting_evidence=[{"beat_ref": "sc-001#1", "why": "she arrives already accusing"}]),
    ]
    return [{"scene_id": "sc-001", "scene_index": 1, "heading": {}, "perception": UNITS[0],
             "abstraction": objects}]


ALL_IDS = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]


# ------------------------------------------------------------------ baseline


def test_clean_fixture_passes():
    for result in (check_grounding(_nodes(), UNITS), check_not_restatement(_nodes(), UNITS),
                   check_calibration(_nodes()), check_contradiction_sought(_nodes()),
                   check_connectivity(_nodes(), MERGE), check_coverage(_nodes()),
                   check_tom_depth(_nodes())):
        assert result["status"] == "pass", (result["id"], result["status"], result["detail"])


# ----------------------------------------------------------------- G1 grounding


def test_g1_rejects_ungrounded_object():
    assert check_grounding(_nodes([_obj(grounded_in=[])]), UNITS)["status"] == "fail"


def test_g1_rejects_grounding_in_a_nonexistent_beat():
    assert check_grounding(_nodes([_obj(grounded_in=["sc-001#99"])]), UNITS)["status"] == "fail"


def test_g1_rejects_grounding_in_another_scene():
    assert check_grounding(_nodes([_obj(grounded_in=["sc-777#1"])]), UNITS)["status"] == "fail"


# ------------------------------------------------------------- G2 restatement


def test_g2_rejects_a_paraphrase_of_its_own_grounding_beat():
    """The failure mode that is invisible to every other check: grounded, well-formed, empty."""
    restated = _obj(statement="Boris denied knowing where the container had gone at first.",
                    grounded_in=["sc-001#2"])
    assert check_not_restatement(_nodes([restated]), UNITS)["status"] == "fail"


def test_g2_passes_a_genuine_inference():
    assert check_not_restatement(_nodes(), UNITS)["status"] == "pass"


# ------------------------------------------------------------ G3 calibration


def test_g3_rejects_uniform_confidence():
    objects = [_obj(ao_id="ao-%d" % i, confidence="probable", links=[]) for i in range(10)]
    assert check_calibration(_nodes(objects))["status"] == "fail"


def test_g3_flags_near_certain_second_order_belief():
    objects = [_obj(ao_id="ao-%d" % i, confidence=c, links=[])
               for i, c in enumerate(["plausible", "probable", "speculative", "near-certain"])]
    objects.append(_obj(ao_id="ao-tom", type="theory_of_mind", about="ana_reyes",
                        confidence="near-certain", links=[]))
    result = check_calibration(_nodes(objects))
    assert result["detail"]["near_certain_theory_of_mind"] == 1


# --------------------------------------------------- G4 contradiction sought


def test_g4_warns_when_nothing_contradicting_was_ever_found():
    objects = [_obj(ao_id="ao-%d" % i, contradicting_evidence=[], links=[]) for i in range(6)]
    assert check_contradiction_sought(_nodes(objects))["status"] == "warn"


def test_g4_fails_when_no_evidence_at_all():
    objects = [_obj(supporting_evidence=[], contradicting_evidence=[], links=[])]
    assert check_contradiction_sought(_nodes(objects))["status"] == "fail"


# ----------------------------------------------------------- G5 connectivity


def test_g5_rejects_an_unlinked_pile():
    objects = [_obj(ao_id="ao-%d" % i, links=[]) for i in range(10)]
    assert check_connectivity(_nodes(objects), MERGE)["status"] == "fail"


def test_g5_reports_dangling_links():
    objects = [_obj(links=[{"to": "ao-does-not-exist", "link": "causes", "why": "x"}])]
    assert check_connectivity(_nodes(objects), MERGE)["detail"]["dangling_links"] == 1


# -------------------------------------------------------------- G6 coverage


def test_g6_rejects_scenes_with_no_abstraction():
    nodes = _nodes() + [{"scene_id": "sc-002", "scene_index": 2, "heading": {},
                         "perception": {}, "abstraction": []}]
    assert check_coverage(nodes)["status"] == "fail"


# ------------------------------------------------------------------- G7 ToM


def test_g7_warns_when_no_theory_of_mind_was_produced():
    objects = [_obj(ao_id="ao-%d" % i, type="mental_state") for i in range(4)]
    assert check_tom_depth(_nodes(objects))["status"] == "warn"


# ------------------------------------------------------- the gate on the gate


def test_run_all_refuses_to_pass_unverified_checks():
    report = run_ao_checks(_nodes(), UNITS, MERGE)
    assert all(c["status"] != "pass" for c in report["checks"])


def test_run_all_passes_once_negative_cases_declared():
    report = run_ao_checks(_nodes(), UNITS, MERGE, negative_cases_ran=ALL_IDS)
    assert report["gate_passed"] is True
    assert all(c["status"] == "pass" for c in report["checks"])


def test_run_all_blocks_on_a_gate_failure():
    report = run_ao_checks(_nodes([_obj(grounded_in=[])]), UNITS, MERGE,
                           negative_cases_ran=ALL_IDS)
    assert report["gate_passed"] is False
    assert "G1" in report["blocking"]
