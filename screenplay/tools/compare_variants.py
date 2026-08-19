#!/usr/bin/env python3
"""Score KU generation variants on the matched-retrieval arm, with the overlap control.

Only the `ku_scene` arm is run per variant: `text_scene`, `full_text` and `none` depend on
the source and the instrument, not on how the units were built, so they are fixed baselines
and re-running them would only add noise.

**Accuracy is reported next to verbatim overlap, always.** A variant that scores higher by
copying more of the source has not improved the artifact, it has broken it, and a table of
accuracies alone cannot tell the two apart.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from screenplay_ku.checks import check_verbatim_overlap  # noqa: E402
from screenplay_ku.client import EndpointPool  # noqa: E402
from screenplay_ku.evaluate import run_arm, summarise  # noqa: E402
from screenplay_ku.mcq import Question  # noqa: E402
from screenplay_ku.render import estimate_tokens, neighbourhood, render_chain  # noqa: E402
from screenplay_ku.scenes import load_scenes, load_source  # noqa: E402


def paired(cells_a, cells_b, question_ids, iterations=20000, seed=11):
    rng = random.Random(seed)
    diff = [
        cells_a[q]["correct"] / max(1, cells_a[q]["total"])
        - cells_b[q]["correct"] / max(1, cells_b[q]["total"])
        for q in question_ids
    ]
    observed = sum(diff) / len(diff)
    boots = []
    for _ in range(iterations):
        sample = [diff[rng.randrange(len(diff))] for _ in range(len(diff))]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    low, high = boots[int(0.025 * iterations)], boots[int(0.975 * iterations)]
    p = 2 * min(sum(1 for x in boots if x <= 0), sum(1 for x in boots if x >= 0)) / iterations
    return observed, low, high, p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", required=True, help="name=path/to/ku_chain.json,...")
    parser.add_argument("--baseline-eval", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--scene-map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--student-port", type=int, default=8107)
    parser.add_argument("--student-model", default="student-4b")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    payload = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = [Question.from_dict(item) for item in payload["questions"]]
    baseline = json.loads(Path(args.baseline_eval).read_text(encoding="utf-8"))
    leaky = set(baseline["leaky_question_ids"])
    non_leaky = [q.question_id for q in questions if q.question_id not in leaky]

    source = load_source(Path(args.source))
    scenes = load_scenes(Path(args.scene_map), source)
    scenes_by_id = {scene.scene_id: scene for scene in scenes}

    pool = EndpointPool([args.student_port], args.student_model,
                        temperature=0.7, max_tokens=16, timeout=600)

    rows = {}
    base_cells = {c["question_id"]: c for c in baseline["cells"]["ku_scene"]}
    text_cells = {c["question_id"]: c for c in baseline["cells"]["text_scene"]}

    for spec in args.variants.split(","):
        name, path = spec.split("=", 1)
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        units = artifact["knowledge_units"]

        overlap = check_verbatim_overlap(units, scenes_by_id, source)
        chain_tokens = estimate_tokens(render_chain(artifact))
        beats = sum(len(u.get("beats") or []) for u in units)

        print("\n=== {} ===".format(name))
        print("  units {} | beats {} | chain {} tok | longest overlap {} | fields>=8 {}".format(
            len(units), beats, chain_tokens,
            overlap.detail.get("longest_ngram_adjudicated"),
            overlap.detail.get("fields_at_or_above_fail")), flush=True)

        cells = run_arm(
            pool, questions, "ku_scene",
            lambda question, art=artifact: render_chain(
                art, scene_ids=neighbourhood(art, question.scene_id, radius=2)),
            samples=args.samples, workers=args.workers,
            progress=lambda d, t, r: print("    [{}/{}]".format(d, t), flush=True)
            if d % 40 == 0 or d == t else None,
        )
        summary = summarise(cells, leaky)
        by_id = {c.question_id: {"correct": c.correct, "total": c.total} for c in cells}
        rows[name] = {
            "units": len(units), "beats": beats, "chain_tokens": chain_tokens,
            "overlap_longest": overlap.detail.get("longest_ngram_adjudicated"),
            "overlap_fields_over_bar": overlap.detail.get("fields_at_or_above_fail"),
            "overlap_status": overlap.status,
            "accuracy_all": summary["all"]["accuracy"],
            "accuracy_non_leaky": summary["non_leaky"]["accuracy"],
            "ci95": summary["non_leaky"]["ci95"],
            "cells": by_id,
        }
        print("  ku_scene non-leaky: {:.3f}".format(summary["non_leaky"]["accuracy"] or 0))

    # Comparisons against the baseline units and against the source-text ceiling.
    for name, row in rows.items():
        cells = row["cells"]
        o, lo, hi, p = paired(cells, base_cells, non_leaky)
        row["vs_baseline_ku"] = {"diff": round(o, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                 "p": round(p, 4)}
        o, lo, hi, p = paired(cells, text_cells, non_leaky)
        row["vs_text_scene"] = {"diff": round(o, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                "p": round(p, 4)}

    report = {
        "baseline_ku_scene": baseline["arms"]["ku_scene"]["non_leaky"]["accuracy"],
        "ceiling_text_scene": baseline["arms"]["text_scene"]["non_leaky"]["accuracy"],
        "non_leaky_n": len(non_leaky),
        "variants": {k: {kk: vv for kk, vv in v.items() if kk != "cells"}
                     for k, v in rows.items()},
    }
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("\n{:<12} {:>7} {:>7} {:>9} {:>8} {:>20} {:>20}".format(
        "variant", "beats", "overlap", "acc", "vs base", "  vs baseline (p)", "  vs text ceiling (p)"))
    print("{:<12} {:>7} {:>7} {:>9.3f} {:>8}".format(
        "baseline", "1021", "7", report["baseline_ku_scene"] or 0, "-"))
    for name, row in rows.items():
        print("{:<12} {:>7} {:>7} {:>9.3f} {:>+8.3f} {:>20} {:>20}".format(
            name, row["beats"], row["overlap_longest"], row["accuracy_non_leaky"] or 0,
            row["vs_baseline_ku"]["diff"],
            "p={:.3f}".format(row["vs_baseline_ku"]["p"]),
            "{:+.3f} p={:.3f}".format(row["vs_text_scene"]["diff"], row["vs_text_scene"]["p"])))
    print("\nceiling (text_scene) = {:.3f}".format(report["ceiling_text_scene"] or 0))
    print("wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
