"""Generate the MCQ instrument from source scenes.

Questions come from the **source text**, never from the Knowledge Units. Generating them
from the units would grade the extractor against a list the extractor produced, which
measures compliance rather than correctness.

The generated questions are published; the scene excerpts they were written from are not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

from .client import EndpointPool, run_parallel
from .mcq import Question, generate_for_scene, sample_scenes
from .scenes import load_scenes, load_source


def _source_ngrams(source: str, low: int = 6, high: int = 20):
    from .checks import _tokens
    tokens = _tokens(source)
    return {n: {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}
            for n in range(low, high + 1)}


def _longest_source_run(text: str, index) -> int:
    from .checks import _tokens
    tokens = _tokens(text)
    for n in sorted(index, reverse=True):
        if len(tokens) < n:
            continue
        if any(tuple(tokens[i:i + n]) in index[n] for i in range(len(tokens) - n + 1)):
            return n
    return 0


def _question_leak(question, index) -> int:
    return max([_longest_source_run(question.question, index)]
               + [_longest_source_run(option, index) for option in question.options])


def _regenerate_leaking(pool, questions, scenes, source, per_scene, workers,
                        fail_at: int = 6, rounds: int = 3):
    """Regenerate any question that quotes the source, until none does.

    Applied per scene, because a scene's four questions are generated together. Questions
    that still leak after the last round are dropped and reported rather than shipped: a
    smaller honest instrument beats a complete one that redistributes dialogue.
    """
    index = _source_ngrams(source)
    by_id = {scene.scene_id: scene for scene in scenes}
    report = {"rounds": [], "dropped": []}

    for round_index in range(rounds):
        leaking = [q for q in questions if _question_leak(q, index) >= fail_at]
        if not leaking:
            break
        scene_ids = sorted({q.scene_id for q in leaking})
        print("  question leak repair {}/{}: {} question(s) across {} scene(s)".format(
            round_index + 1, rounds, len(leaking), len(scene_ids)), flush=True)
        regenerated = run_parallel(
            scene_ids,
            lambda sid: generate_for_scene(pool, by_id[sid], source, count=per_scene),
            max_workers=workers,
        )
        keep = [q for q in questions if q.scene_id not in set(scene_ids)]
        for sid, result in zip(scene_ids, regenerated):
            if isinstance(result, Exception):
                continue
            keep.extend(result)
        questions = keep
        report["rounds"].append({"round": round_index + 1, "leaking": len(leaking),
                                 "scenes_regenerated": len(scene_ids)})

    survivors = []
    for question in questions:
        leak = _question_leak(question, index)
        if leak >= fail_at:
            report["dropped"].append({"question_id": question.question_id, "run_words": leak})
        else:
            survivors.append(question)
    if report["dropped"]:
        print("  dropped {} question(s) that still quoted the source".format(len(report["dropped"])))
    return survivors, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the MCQ instrument")
    parser.add_argument("--source", required=True)
    parser.add_argument("--scene-map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenes", type=int, default=25)
    parser.add_argument("--per-scene", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--ports", default="8100-8107")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    source = load_source(Path(args.source))
    scenes = load_scenes(Path(args.scene_map), source)
    picked = sample_scenes(scenes, count=args.scenes, seed=args.seed)
    print("sampled {} scenes (seed {}): {}".format(
        len(picked), args.seed, ", ".join(scene.scene_id for scene in picked)))
    print("word counts: {}".format([scene.word_count for scene in picked]))

    low, high = (args.ports.split("-") + [None])[:2]
    ports = list(range(int(low), int(high) + 1)) if high else [int(low)]
    pool = EndpointPool(ports, args.model, temperature=0.6)

    def work(scene):
        return generate_for_scene(pool, scene, source, count=args.per_scene)

    results = run_parallel(
        list(picked), work, max_workers=args.workers,
        on_done=lambda done, total, result: print(
            "  [{}/{}] {}".format(done, total, "!" if isinstance(result, Exception) else "."),
            flush=True),
    )

    questions: List[Question] = []
    failures = []
    for scene, result in zip(picked, results):
        if isinstance(result, Exception):
            failures.append({"scene_id": scene.scene_id, "error": str(result)})
            continue
        questions.extend(result)

    # Gate the instrument exactly as the artifact is gated. The generation prompt asks for
    # no quoting and the first run produced seven questions carrying eight-word runs of
    # source dialogue in their answer options anyway — instructions do not enforce, and a
    # published instrument that quotes the screenplay leaks it just as surely as a KU would.
    questions, leak_report = _regenerate_leaking(
        pool, questions, picked, source, args.per_scene, args.workers
    )

    payload = {
        "instrument": {
            "scenes_sampled": len(picked),
            "scene_ids": [scene.scene_id for scene in picked],
            "per_scene_requested": args.per_scene,
            "questions": len(questions),
            "seed": args.seed,
            "generator_model": args.model,
            "generated_from": "source scene text (never from Knowledge Units)",
            "failures": failures,
            "leak_gate": leak_report,
            "leak_gate_note": "questions quoting >=6 consecutive source words are regenerated, then dropped",
        },
        "questions": [question.to_dict() for question in questions],
    }
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("\nwrote {} ({} questions from {} scenes, {} failures)".format(
        args.out, len(questions), len(picked), len(failures)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
