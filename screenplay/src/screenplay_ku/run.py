"""End-to-end run: window, extract, verify seams, check, emit.

The artifact is gated. If any gating check fails, ``ku_chain.json`` is not written and the
protocol records why. A run that cannot produce a complete protocol is a failed run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .checks import run_all
from .kuschema import close_entity_references
from .client import EndpointPool
from .fingerprint import minhash, sha256_text
from .pipeline import Stage1, Stage2, apply_seam_patches, repair_overlap_fields
from .prompts import SYSTEM_PROMPT
from .scenes import Scene, attribution_report, load_scenes, load_source, source_digest
from .windows import build_page_windows, build_windows, canary_window


def _negative_cases_ran(repo_root: Path) -> List[str]:
    """Run the negative suite and return the check ids it verified.

    Checks are reported ``unverified`` unless their negative case ran *in this execution*.
    Reading a stale pass from a file would reintroduce exactly the failure mode the suite
    exists to prevent.
    """
    runner = repo_root / "tests" / "run_checks_tests.py"
    if not runner.exists():
        return []
    result = subprocess.run(
        [sys.executable, str(runner)], capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print("negative suite FAILED; no check will be reported as passing")
        print(result.stdout[-2000:])
        return []
    for line in result.stdout.splitlines():
        if line.startswith("negative cases verified for:"):
            ids = line.split(":", 1)[1].strip()
            return [] if ids == "none" else [part.strip() for part in ids.split(",")]
    return []


def _canonical_entities(units: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    names: Dict[str, Counter] = defaultdict(Counter)
    aliases: Dict[str, set] = defaultdict(set)
    types: Dict[str, Counter] = defaultdict(Counter)
    for unit in units:
        for entity in unit.get("entities") or []:
            eid = entity.get("entity_id")
            if not eid:
                continue
            if entity.get("name"):
                names[eid][entity["name"]] += 1
            for alias in entity.get("aliases") or []:
                aliases[eid].add(alias)
            if entity.get("type"):
                types[eid][entity["type"]] += 1
    out = []
    for eid in sorted(names):
        canonical_name = names[eid].most_common(1)[0][0]
        out.append({
            "canonical_id": eid,
            "canonical_name": canonical_name,
            "entity_type": types[eid].most_common(1)[0][0] if types[eid] else "other",
            "aliases": sorted(aliases[eid] | set(names[eid]) - {canonical_name}),
            "scene_count": sum(
                1 for unit in units
                if eid in (unit.get("present") or []) or eid in (unit.get("referenced") or [])
            ),
        })
    return out


def _attach_source_refs(
    units: List[Dict[str, Any]], scenes_by_id: Dict[str, Scene], source: str
) -> None:
    """Reference only: offsets, a digest, and a MinHash sketch. Never the text."""
    for unit in units:
        scene = scenes_by_id.get(unit.get("scene_id"))
        if scene is None:
            continue
        text = scene.text(source)
        unit["scene_index"] = scene.index
        unit["heading"] = {
            "raw": scene.heading_raw, "kind": scene.kind,
            "location": scene.location, "time_of_day": scene.time_of_day,
        }
        unit["source"] = {
            "scene_id": scene.scene_id,
            "start_char": scene.start_char, "end_char": scene.end_char,
            "word_count": scene.word_count,
            "sha256": sha256_text(text),
            "sentence_minhash": minhash(text),
        }


def _redact_source_spans(report: Dict[str, Any]) -> None:
    """Strip matched source text from check violations before the protocol is written.

    The overlap check records the n-gram it matched, which is by definition a span of the
    source. That is useful while debugging on this machine and unacceptable in a published
    protocol, whose whole premise is that structure travels and text does not. The length
    and a short digest keep the finding auditable without carrying the words.
    """
    for check in report.get("checks") or []:
        for violation in check.get("violations") or []:
            span = violation.pop("ngram", None)
            if span:
                violation["ngram_sha256"] = sha256_text(span)[:16]
                violation["ngram_words"] = len(span.split())


def _link_chain(units: List[Dict[str, Any]]) -> None:
    units.sort(key=lambda unit: unit.get("scene_index", 0))
    for position, unit in enumerate(units):
        unit["preceded_by"] = units[position - 1]["scene_id"] if position else None
        unit["followed_by"] = (
            units[position + 1]["scene_id"] if position + 1 < len(units) else None
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract screenplay Knowledge Units")
    parser.add_argument("--source", required=True, help="normalized screenplay text")
    parser.add_argument("--scene-map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--credited-as", default="")
    parser.add_argument("--draft", default="")
    parser.add_argument("--draft-date", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--windowing", choices=["scenes", "pages"], default="scenes")
    parser.add_argument("--pages-target", type=int, default=6)
    parser.add_argument("--pages-context", type=int, default=5)
    parser.add_argument("--max-scenes", type=int, default=12)
    parser.add_argument("--ports", default="8100-8107")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit-windows", type=int, default=0, help="smoke-test subset")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--repair-rounds", type=int, default=3,
                        help="max overlap-repair passes; the gate re-runs after each")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    source = load_source(Path(args.source))
    scenes = load_scenes(Path(args.scene_map), source)
    digest = source_digest(source)
    attribution = attribution_report(scenes, source)
    print("scenes: {}  attribution: {:.4f}".format(len(scenes), attribution["coverage"]))

    document = {
        "title": args.title, "work_type": "screenplay",
        "credited_as": args.credited_as, "draft": args.draft, "draft_date": args.draft_date,
        "language": "en", "source_url": args.source_url, "source_sha256": digest,
        "source_words": len(source.split()), "scene_count": len(scenes),
        "rights_note": (
            "Structure derived under a text-and-data-mining research exemption. "
            "No source text is redistributed; scenes are referenced by offset and digest."
        ),
    }

    if args.windowing == "scenes":
        windows = build_windows(
            scenes, source, pages_target=args.pages_target,
            pages_context=args.pages_context, max_scenes=args.max_scenes,
        )
    else:
        windows = build_page_windows(
            scenes, source, pages_target=args.pages_target, pages_context=args.pages_context
        )
    if args.limit_windows:
        windows = windows[: args.limit_windows]
    print("windows: {} ({} windowing)".format(len(windows), args.windowing))

    low, high = (args.ports.split("-") + [None])[:2]
    ports = list(range(int(low), int(high) + 1)) if high else [int(low)]
    pool = EndpointPool(ports, args.model, temperature=args.temperature)
    unhealthy = [port for port, ok in pool.health() if not ok]
    if unhealthy:
        print("unhealthy endpoints: {}".format(unhealthy))
        return 2

    stage1 = Stage1(pool, source, scenes, document)

    print("warming shared prefix on {} endpoints...".format(len(ports)))
    warm = pool.warm(SYSTEM_PROMPT, "=== FULL SCREENPLAY ===\n" + source)
    print("  warm: {}".format(", ".join("{:.1f}s".format(value) for value in warm)))

    def progress(done: int, total: int, result: Any) -> None:
        mark = "!" if isinstance(result, Exception) else "."
        print("  [{}/{}] {}".format(done, total, mark), flush=True)

    print("stage 1: extracting {} windows".format(len(windows)))
    stage1_result = stage1.run(windows, max_workers=args.workers, progress=progress)
    units = stage1_result["units"]
    print("  {} units from {} windows in {}s ({} failures)".format(
        len(units), len(windows), stage1_result["seconds"], len(stage1_result["failures"])))

    print("canary...")
    try:
        canary = stage1.extract_window(canary_window(windows, source))
    except Exception as error:
        canary = {"emitted": -1, "error": str(error)}
    print("  canary emitted {} units".format(canary.get("emitted")))

    seam_result: Dict[str, Any] = {"seams": [], "failures": [], "seconds": 0.0}
    patch_report: Dict[str, Any] = {"applied": {}, "alias_map": {}}
    if not args.skip_stage2:
        by_window: List[List[Dict]] = []
        index = {unit["scene_id"]: unit for unit in units}
        for window in windows:
            group = [index[sid] for sid in window.scene_ids if sid in index]
            by_window.append(group)
        print("stage 2: verifying {} seams".format(max(0, len(by_window) - 1)))
        stage2 = Stage2(pool)
        seam_result = stage2.run(by_window, max_workers=args.workers, progress=progress)
        print("  {} seams in {}s ({} failures)".format(
            len(seam_result["seams"]), seam_result["seconds"], len(seam_result["failures"])))
        patch_report = apply_seam_patches(units, seam_result["seams"])
        # Alias resolution rewrites ids in beats and relationships. If a merge target was
        # only ever a relationship target and never a declared entity, the rewrite leaves a
        # dangling reference, so close again after patching rather than before only.
        reclosed = sum(len(close_entity_references(unit)) for unit in units)
        patch_report["applied"]["reclosed_entities"] = reclosed
        print("  applied: {}".format(patch_report["applied"]))

    scenes_by_id = {scene.scene_id: scene for scene in scenes}
    _attach_source_refs(units, scenes_by_id, source)
    _link_chain(units)

    # Dump before grading. A twelve-minute run whose checks fail should not have to be
    # repeated to re-measure a check; this makes check iteration free and keeps the
    # expensive part reproducible. Intermediate, gitignored, structure only.
    (out_dir / "units.raw.json").write_text(
        json.dumps({"units": units, "canary": canary}, indent=1), encoding="utf-8")

    print("running negative suite before grading...")
    verified = _negative_cases_ran(Path(__file__).resolve().parents[2])
    print("  verified: {}".format(verified or "none"))

    # Grade coverage against the scenes actually dispatched. On a full run this is every
    # scene; on a subset run it is the subset. Grading 225 scenes when 3 windows ran
    # reports a confident failure about work that was never requested.
    dispatched_ids = {sid for window in windows for sid in window.scene_ids}
    graded_scenes = [scene for scene in scenes if scene.scene_id in dispatched_ids]
    if len(graded_scenes) != len(scenes):
        print("  (subset run: grading {} of {} scenes)".format(len(graded_scenes), len(scenes)))

    # Pre-gate overlap sweep, then a targeted restatement of only what it flags. The gate
    # below re-runs on the repaired units and still blocks, so this reduces the number of
    # fields carrying source phrasing without weakening the threshold that decides.
    from .checks import check_verbatim_overlap

    scenes_for_overlap = {scene.scene_id: scene for scene in graded_scenes}
    repair_rounds: List[Dict[str, Any]] = []
    for round_index in range(args.repair_rounds):
        pre = check_verbatim_overlap(units, scenes_for_overlap, source)
        if not pre.violations:
            break
        print("overlap repair {}/{}: {} flagged field(s), longest {} tokens".format(
            round_index + 1, args.repair_rounds, len(pre.violations),
            pre.detail.get("longest_ngram_adjudicated")))
        outcome = repair_overlap_fields(pool, units, pre.violations, max_workers=args.workers)
        outcome["round"] = round_index + 1
        outcome["flagged_before"] = len(pre.violations)
        outcome["longest_before"] = pre.detail.get("longest_ngram_adjudicated")
        repair_rounds.append(outcome)
        print("  restated {}/{} ({} failed)".format(
            outcome["restated"], outcome["targeted"], outcome["failed"]))
    repair = {"rounds": repair_rounds,
              "total_restated": sum(item["restated"] for item in repair_rounds)}

    print("checks...")
    report = run_all(
        units, graded_scenes, source, canary_result=canary, negative_cases_ran=verified
    )
    for check in report["checks"]:
        print("  {} {:22s} {}".format(check["check_id"], check["name"], check["status"]))

    _redact_source_spans(report)

    protocol = {
        "run": {
            "started": started, "seconds": round(time.time() - started, 1),
            "model": args.model, "ports": ports, "windowing": args.windowing,
            "pages_target": args.pages_target, "pages_context": args.pages_context,
            "max_scenes": args.max_scenes, "temperature": args.temperature,
        },
        "document": document,
        "attribution": attribution,
        "windows": [
            {"index": w.index, "scenes": len(w.scenes), "words": w.word_count,
             "scene_ids": w.scene_ids}
            for w in windows
        ],
        "stage1": {"seconds": stage1_result["seconds"], "usage": stage1_result["usage"],
                   "failures": stage1_result["failures"]},
        "canary": {k: v for k, v in canary.items() if k != "units"},
        "stage2": {"seconds": seam_result["seconds"], "failures": seam_result["failures"],
                   "seams": seam_result["seams"], "applied": patch_report.get("applied"),
                   "alias_map": patch_report.get("alias_map")},
        "overlap_repair": repair,
        "checks": report,
    }
    (out_dir / "protocol.json").write_text(json.dumps(protocol, indent=1), encoding="utf-8")

    if not report["gate_passed"]:
        print("\nGATE FAILED: {} — no artifact written".format(", ".join(report["blocking"])))
        return 1

    artifact = {
        "schema_version": "screenplay-1.0",
        "document": document,
        "extraction": {
            "model": args.model, "pipeline": "screenplay/two-stage",
            "windowing": args.windowing, "pages_target": args.pages_target,
            "pages_context": args.pages_context, "stage1_windows": len(windows),
            "stage2_seams": len(seam_result["seams"]),
        },
        "canonical_entities": _canonical_entities(units),
        "knowledge_units": units,
    }
    (out_dir / "ku_chain.json").write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    print("\nwrote {} ({} units, {} entities) in {:.0f}s".format(
        out_dir / "ku_chain.json", len(units),
        len(artifact["canonical_entities"]), time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
