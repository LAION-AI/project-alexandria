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
        },
        "questions": [question.to_dict() for question in questions],
    }
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("\nwrote {} ({} questions from {} scenes, {} failures)".format(
        args.out, len(questions), len(picked), len(failures)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
