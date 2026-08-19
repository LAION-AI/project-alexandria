"""Run the CogniTino abstraction layer over an Alexandria KU chain."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..client import EndpointPool
from ..scenes import load_scenes, load_source
from . import pipeline
from .checks import run_ao_checks


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CogniTino abstraction layer")
    parser.add_argument("--ku-chain", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--scene-map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenes-per-window", type=int, default=5)
    parser.add_argument("--research-rounds", type=int, default=2)
    parser.add_argument("--merge-levels", type=int, default=3)
    parser.add_argument("--ports", default="8100-8106")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--limit-windows", type=int, default=0)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    artifact = json.loads(Path(args.ku_chain).read_text(encoding="utf-8"))
    units = artifact["knowledge_units"]
    document = artifact["document"]
    source = load_source(Path(args.source))
    scenes_by_id = {s.scene_id: s for s in load_scenes(Path(args.scene_map), source)}

    windows = pipeline.build_windows(units, scenes_by_id, source, args.scenes_per_window)
    if args.limit_windows:
        windows = windows[: args.limit_windows]
    print("units {} | windows {} ({} scenes each)".format(
        len(units), len(windows), args.scenes_per_window), flush=True)

    low, high = (args.ports.split("-") + [None])[:2]
    ports = list(range(int(low), int(high) + 1)) if high else [int(low)]
    pool = EndpointPool(ports, args.model, temperature=0.6, max_tokens=16384)
    unhealthy = [p for p, ok in pool.health() if not ok]
    if unhealthy:
        print("unhealthy endpoints: {}".format(unhealthy))
        return 2

    def progress(done, total, result):
        if done % 5 == 0 or done == total:
            print("    [{}/{}]{}".format(done, total,
                  " !" if isinstance(result, Exception) else ""), flush=True)

    print("\nmodule 2 — Abstraction Object Generation ({} windows)".format(len(windows)))
    draft = pipeline.draft_all(pool, windows, source, document,
                               workers=args.workers, progress=progress)
    print("  {} objects in {}s ({} failures)".format(
        draft["objects"], draft["seconds"], len(draft["failures"])), flush=True)

    print("\nmodule 3 — Abstraction Object Researcher ({} rounds)".format(args.research_rounds))
    research = pipeline.research_all(pool, windows, source, rounds=args.research_rounds,
                                     workers=args.workers, progress=progress)
    print("  {} in {}s | {}".format(research["objects"], research["seconds"],
                                    research["tally"]), flush=True)

    print("\nmodule 5 — Semantic Connection (merge tree, {} levels)".format(args.merge_levels))
    merge = pipeline.merge_tree(pool, windows, levels=args.merge_levels,
                                workers=args.workers, progress=progress)
    print("  {} cross-links, {} duplicates, {} arcs in {}s".format(
        merge["cross_links"], merge["duplicates"], merge["arc_count"],
        merge["seconds"]), flush=True)

    print("\nmodule 4 — Editor (sequential canonicalization)")
    editor = pipeline.canonicalize(pool, windows, units)
    print("  {} names, {} mappings, {} renames in {}s".format(
        editor["names"], editor["mappings"], editor["renames_applied"],
        editor["seconds"]), flush=True)

    # Assemble the scene graph: each scene node carries its Perception layer (the KU) and
    # its Abstraction layer, which is what the storytree rubric will be pointed at.
    objects = [o for w in windows for o in w.objects]
    live = [o for o in objects if not o.get("superseded_by")]
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for obj in live:
        by_scene.setdefault(obj["scene_id"], []).append(obj)

    # Verbatim-overlap repair, on the same terms as the Perception layer: the gate re-runs
    # after each round and still blocks, so this lowers the count without moving the bar.
    from .checks import check_verbatim_overlap
    repair_rounds = []
    for round_index in range(3):
        probe_nodes = [{"scene_id": sid, "abstraction": objs}
                       for sid, objs in by_scene.items()]
        pre = check_verbatim_overlap(probe_nodes, source)
        targets = [v for v in pre["all_violations"] if v["length"] >= 8] or \
                  ([] if pre["status"] == "pass" else pre["all_violations"])
        if not targets:
            break
        print("  overlap repair {}: {} field(s) at/over the bar, longest {}".format(
            round_index + 1, len(targets), pre["detail"]["longest_run"]), flush=True)
        outcome = pipeline.repair_overlap(pool, windows, targets,
                                          max_workers=args.workers)
        outcome["round"] = round_index + 1
        repair_rounds.append(outcome)
        print("    restated {}/{}".format(outcome["restated"], outcome["targeted"]), flush=True)

    # Grade only the scenes actually dispatched. On a full run this is every scene; on a
    # subset run it is the subset. Grading all 225 when 4 windows ran reports a confident
    # failure about work nobody requested — the same defect the KU layer's C1 had.
    dispatched = {sid for w in windows for sid in w.scene_ids}
    scene_nodes = []
    for unit in sorted(units, key=lambda u: u.get("scene_index", 0)):
        sid = unit["scene_id"]
        if sid not in dispatched:
            continue
        scene_nodes.append({
            "scene_id": sid,
            "scene_index": unit.get("scene_index"),
            "heading": unit.get("heading"),
            "perception": unit,
            "abstraction": by_scene.get(sid, []),
        })

    # Run the negative suite in this execution before grading, never a stored pass.
    verified = []
    runner = Path(__file__).resolve().parents[3] / "tests" / "run_cognitino_tests.py"
    if runner.exists():
        proc = subprocess.run([sys.executable, str(runner)], capture_output=True, text=True)
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("negative cases verified for:"):
                    ids = line.split(":", 1)[1].strip()
                    verified = [] if ids == "none" else [p.strip() for p in ids.split(",")]
        else:
            print("negative suite FAILED; no check will be reported as passing")
    print("  negative cases verified: {}".format(verified or "none"))

    checks = run_ao_checks(scene_nodes, units, merge, negative_cases_ran=verified,
                           source=source)
    graph = {
        "schema_version": "cognitino-screenplay-1.0",
        "document": document,
        "derived_from": {
            "ku_chain": str(Path(args.ku_chain).resolve().name),
            "ku_units": len(units),
        },
        "build": {
            "scenes_per_window": args.scenes_per_window,
            "windows": len(windows),
            "research_rounds": args.research_rounds,
            "merge_levels": args.merge_levels,
            "model": args.model,
        },
        "arcs": merge["arcs"],
        "alias_map": editor["alias_map"],
        "scene_nodes": scene_nodes,
    }
    # A failing gate must not produce an artifact. The Perception layer already works this
    # way; writing the graph before checking it meant a gate failure produced a file anyway,
    # which is the one behaviour a gate exists to prevent.
    if not checks["gate_passed"]:
        (out_dir / "protocol.json").write_text(json.dumps({"checks": checks}, indent=1),
                                               encoding="utf-8")
        print("\nGATE FAILED: {} — no artifact written".format(", ".join(checks["blocking"])))
        return 1
    (out_dir / "scene_graph.json").write_text(json.dumps(graph, indent=1), encoding="utf-8")

    protocol = {
        "seconds": round(time.time() - started, 1),
        "modules": {"generation": draft, "researcher": research,
                    "semantic_connection": {k: v for k, v in merge.items() if k != "arcs"},
                    "editor": {k: v for k, v in editor.items() if k != "alias_map"},
                    "overlap_repair": repair_rounds},
        "graph": {
            "scene_nodes": len(scene_nodes),
            "abstraction_objects": len(live),
            "superseded": len(objects) - len(live),
            "by_type": dict(Counter(o["type"] for o in live)),
            "by_confidence": dict(Counter(o["confidence"] for o in live)),
            "links": sum(len(o.get("links") or []) for o in live),
            "supporting_evidence": sum(len(o.get("supporting_evidence") or []) for o in live),
            "contradicting_evidence": sum(len(o.get("contradicting_evidence") or []) for o in live),
            "arcs": len(merge["arcs"]),
        },
        "checks": checks,
    }
    (out_dir / "protocol.json").write_text(json.dumps(protocol, indent=1), encoding="utf-8")

    print("\ngraph: {} scene nodes, {} abstraction objects".format(
        len(scene_nodes), len(live)))
    print("  by type: {}".format(protocol["graph"]["by_type"]))
    print("  by confidence: {}".format(protocol["graph"]["by_confidence"]))
    print("  links {} | support {} | contradict {} | arcs {}".format(
        protocol["graph"]["links"], protocol["graph"]["supporting_evidence"],
        protocol["graph"]["contradicting_evidence"], protocol["graph"]["arcs"]))
    print("\nchecks:")
    for check in checks["checks"]:
        print("  {} {:26s} {}".format(check["id"], check["name"], check["status"]))
    print("\nwrote {} in {:.0f}s".format(out_dir / "scene_graph.json", time.time() - started))
    return 0 if checks["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
