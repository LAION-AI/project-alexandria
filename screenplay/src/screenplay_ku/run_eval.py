"""Run the MCQ instrument across four context arms against a student model.

The no-context arm runs first and is treated as the instrument's calibration, not a
formality: the source is a widely described film, so any question the student answers
without context measures its memory rather than the artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .client import EndpointPool
from .evaluate import find_leaky, run_arm, summarise
from .mcq import Question
from .render import estimate_tokens, neighbourhood, render_chain
from .scenes import load_scenes, load_source


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate KU chain against context arms")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--ku-chain", required=True)
    parser.add_argument("--source", required=True, help="for the full_text arm only")
    parser.add_argument("--out", required=True)
    parser.add_argument("--student-port", type=int, default=8107)
    parser.add_argument("--student-model", default="gemma-4-e4b")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--context-budget-tokens", type=int, default=110000)
    parser.add_argument("--arms", default="none,full_text,text_scene,ku_chain,ku_scene")
    parser.add_argument("--scene-map", default="", help="required for the text_scene arm")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = [Question.from_dict(item) for item in payload["questions"]]
    artifact = json.loads(Path(args.ku_chain).read_text(encoding="utf-8"))
    source = load_source(Path(args.source))
    scenes_by_id = {}
    if args.scene_map:
        scenes_by_id = {scene.scene_id: scene for scene in
                        load_scenes(Path(args.scene_map), source)}

    chain_text = render_chain(artifact)
    budget_chars = args.context_budget_tokens * 4
    sizes = {
        "full_text_tokens": estimate_tokens(source),
        "ku_chain_tokens": estimate_tokens(chain_text),
        "budget_tokens": args.context_budget_tokens,
    }
    print("context sizes: {}".format(sizes))
    # Assert before spending anything. An arm that silently truncates looks like it had the
    # context while the tail was never shown, and produces a confident wrong comparison.
    for name, size in (("full_text", sizes["full_text_tokens"]),
                       ("ku_chain", sizes["ku_chain_tokens"])):
        if size > args.context_budget_tokens:
            print("{} arm needs {} tokens, budget is {} — refusing to run a truncated arm".format(
                name, size, args.context_budget_tokens))
            return 2

    pool = EndpointPool([args.student_port], args.student_model,
                        temperature=0.7, max_tokens=16, timeout=600)
    healthy = pool.health()
    if not all(ok for _, ok in healthy):
        print("student endpoint not reachable: {}".format(healthy))
        return 2

    def progress(done, total, result):
        if done % 20 == 0 or done == total:
            print("    [{}/{}]".format(done, total), flush=True)

    arms = args.arms.split(",")
    results: Dict[str, Any] = {}
    cells_by_arm: Dict[str, List] = {}

    def context_for_arm(arm: str):
        if arm == "none":
            return ""
        if arm == "full_text":
            return "SCREENPLAY:\n" + source
        if arm == "ku_chain":
            return chain_text
        if arm == "ku_scene":
            return lambda question: render_chain(
                artifact, scene_ids=neighbourhood(artifact, question.scene_id, radius=2)
            )
        if arm == "text_scene":
            # The honest control for ku_scene. Comparing scene-local KUs against the whole
            # screenplay confounds two things at once: representation (units vs prose) and
            # retrieval (five scenes vs two hundred). This arm gives the source text the
            # same retrieval advantage, so the remaining difference is representation alone.
            if not scenes_by_id:
                raise ValueError("--scene-map is required for the text_scene arm")

            def scene_local_text(question):
                wanted = neighbourhood(artifact, question.scene_id, radius=2)
                parts = []
                for scene_id in wanted:
                    scene = scenes_by_id.get(scene_id)
                    if scene is not None:
                        parts.append(scene.text(source))
                return "SCREENPLAY EXCERPT:\n" + "\n".join(parts)

            return scene_local_text
        raise ValueError("unknown arm: {}".format(arm))

    for arm in arms:
        print("\narm: {} ({} questions x {} samples)".format(arm, len(questions), args.samples))
        cells = run_arm(
            pool, questions, arm, context_for_arm(arm),
            samples=args.samples, workers=args.workers,
            max_context_chars=budget_chars, progress=progress,
        )
        cells_by_arm[arm] = cells
        print("  raw accuracy: {:.3f}".format(
            sum(cell.correct for cell in cells) / max(1, sum(cell.total for cell in cells))))

    leaky = find_leaky(cells_by_arm["none"]) if "none" in cells_by_arm else set()
    print("\nleaky questions (answered without context at >=0.6): {}/{}".format(
        len(leaky), len(questions)))

    for arm, cells in cells_by_arm.items():
        results[arm] = summarise(cells, leaky)

    report = {
        "instrument": payload["instrument"],
        "student": {"model": args.student_model, "port": args.student_port,
                    "samples_per_question": args.samples, "temperature": 0.7},
        "context_sizes": sizes,
        "leaky_question_ids": sorted(leaky),
        "leaky_fraction": round(len(leaky) / len(questions), 4) if questions else 0.0,
        "arms": results,
        "cells": {
            arm: [
                {"question_id": cell.question_id, "scene_id": cell.scene_id,
                 "correct": cell.correct, "total": cell.total,
                 "unparseable": cell.unparseable}
                for cell in cells
            ]
            for arm, cells in cells_by_arm.items()
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("\n{:<12} {:>18} {:>18} {:>8}".format("arm", "all", "non-leaky", "unparse"))
    for arm in arms:
        summary = results[arm]
        every, clean = summary["all"], summary["non_leaky"]
        print("{:<12} {:>8.3f} [{:.2f},{:.2f}] {:>8.3f} [{:.2f},{:.2f}] {:>8}".format(
            arm, every["accuracy"] or 0, every["ci95"][0], every["ci95"][1],
            clean["accuracy"] or 0, clean["ci95"][0], clean["ci95"][1],
            every["unparseable"]))
    print("\nwrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
