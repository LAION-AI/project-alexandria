#!/usr/bin/env python3
"""Build a worked-example document: source scene, its Knowledge Unit, and its questions.

This is the one place in the repository that reproduces source text, and it does so as a
short scholarly quotation for method illustration — three scenes of 225, under 2% of the
work. Everywhere else the source is referenced by offset and digest. The exception is
recorded in docs/05-provenance-and-scope.md rather than left implicit.

Usage:
  python3 tools/build_illustration.py --scenes sc-036,sc-011,sc-024 --out docs/07-worked-examples.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from screenplay_ku.scenes import load_scenes, load_source  # noqa: E402

LETTERS = "ABCD"


def fence(text: str, language: str = "text") -> str:
    return "```{}\n{}\n```".format(language, text.rstrip())


def render_beat(beat: dict) -> str:
    addressee = beat.get("addressee")
    arrow = " → `{}`".format(addressee) if addressee else ""
    lines = ["**{}.** `[{}]` `{}`{}  \n{}".format(
        beat.get("order"), beat.get("type"), beat.get("actor"), arrow, beat.get("content"))]
    facts = beat.get("facts") or {}
    kept = {key: value for key, value in facts.items() if value}
    if kept:
        lines.append("  \n*facts:* " + "; ".join(
            "{}: {}".format(key, ", ".join(str(item) for item in value))
            for key, value in kept.items()))
    for change in beat.get("state_changes") or []:
        lines.append("  \n*state:* `{}.{}`: {} → **{}**".format(
            change.get("entity"), change.get("field"),
            change.get("from"), change.get("to")))
    if beat.get("causes"):
        lines.append("  \n*follows from beat(s):* {}".format(
            ", ".join(str(item) for item in beat["causes"])))
    lines.append("  \n*certainty:* `{}`".format(beat.get("certainty")))
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--scene-map", required=True)
    parser.add_argument("--ku-chain", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = load_source(Path(args.source))
    scenes = {scene.scene_id: scene for scene in load_scenes(Path(args.scene_map), source)}
    artifact = json.loads(Path(args.ku_chain).read_text(encoding="utf-8"))
    units = {unit["scene_id"]: unit for unit in artifact["knowledge_units"]}
    payload = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions: dict = {}
    for item in payload["questions"]:
        questions.setdefault(item["scene_id"], []).append(item)

    wanted = args.scenes.split(",")
    document = artifact["document"]

    out = ["""# Worked examples

Three scenes end to end: the **source text**, the **Knowledge Unit** derived from it, and the
**questions** generated from it. Chosen to show different regimes — a dialogue-heavy exchange,
a dense action scene with many entities, and a scene so short it tests the method's floor.

> **On the source text below.** This is the only file in the repository that reproduces any
> of the screenplay. Three scenes of 225, under 2% of the work, quoted for method
> illustration so a reader can judge whether the units preserve what they claim to. Every
> other artifact references the source by offset and digest and contains no source prose.
> See [provenance and scope](05-provenance-and-scope.md).

**Source.** *{title}* ({draft}, {date}), credited to {credit}. Public copy at
<{url}>; sha256 `{digest}`.

---
""".format(title=document.get("title"), draft=document.get("draft"),
           date=document.get("draft_date"), credit=document.get("credited_as"),
           url=document.get("source_url"), digest=(document.get("source_sha256") or "")[:24])]

    for scene_id in wanted:
        scene = scenes.get(scene_id)
        unit = units.get(scene_id)
        if scene is None or unit is None:
            print("skipping unknown scene {}".format(scene_id))
            continue

        out.append("## {} — `{}`\n".format(scene.heading_raw, scene_id))
        out.append("*{} words, {:.0%} dialogue, {} beats, {} entities extracted.*\n".format(
            scene.word_count,
            0.0 if not scene.word_count else _dialogue_ratio(scene, source),
            len(unit.get("beats") or []), len(unit.get("entities") or [])))

        out.append("### 1. Source text\n")
        out.append(fence(scene.text(source)))

        out.append("\n### 2. Knowledge Unit\n")
        out.append("**Context before.** {}\n".format(unit.get("context_before")))
        out.append("**Style.** {}\n".format(unit.get("style")))
        out.append("**Present:** {}  \n**Referenced:** {}\n".format(
            ", ".join("`{}`".format(x) for x in unit.get("present") or []) or "—",
            ", ".join("`{}`".format(x) for x in unit.get("referenced") or []) or "—"))

        out.append("\n**Beats** — the temporal spine. Sorting every beat in the film by\n"
                   "`(scene_index, order)` reconstructs the order the audience receives\n"
                   "information.\n")
        for beat in unit.get("beats") or []:
            out.append("\n" + render_beat(beat) + "\n")

        entities = unit.get("entities") or []
        if entities:
            out.append("\n**Entities**\n")
            out.append("| id | name | type | attributes |")
            out.append("|---|---|---|---|")
            for entity in entities:
                attributes = entity.get("attributes") or {}
                shown = "; ".join("{}: {}".format(k, v) for k, v in attributes.items()
                                  if k != "declared_by") or "—"
                note = " *(auto-declared)*" if attributes.get("declared_by") else ""
                out.append("| `{}` | {}{} | {} | {} |".format(
                    entity.get("entity_id"), entity.get("name"), note,
                    entity.get("type"), shown[:120]))

        out.append("\n**Context after.** {}\n".format(unit.get("context_after")))
        out.append("\n*Source reference (no text stored):* chars {}–{}, sha256 `{}`\n".format(
            unit["source"]["start_char"], unit["source"]["end_char"],
            unit["source"]["sha256"][:16]))

        scene_questions = questions.get(scene_id) or []
        if scene_questions:
            out.append("\n### 3. Questions generated from this scene\n")
            out.append("Generated from the **source text**, never from the unit above.\n")
            for item in scene_questions:
                out.append("\n**{}** *(`{}`)*  \n{}\n".format(
                    item["question_id"], item["fact_type"], item["question"]))
                for index, option in enumerate(item["options"]):
                    marker = " ✅" if index == item["correct_index"] else ""
                    out.append("- **{}.** {}{}".format(LETTERS[index], option, marker))
        out.append("\n---\n")

    out.append("""## What these show

**Indirect speech.** No line of dialogue survives as dialogue. The units record what was
communicated — who told what to whom, what was refused, what was revealed — in different
words. The verbatim gate enforces this mechanically; across the whole artifact the longest
run shared with the source is 7 words.

**Facts survive exactly.** Names, numbers and locations pass through unchanged and are
listed per beat, because paraphrase applies to expression and not to fact.

**The floor is visible.** A 23-word scene yields one beat and one entity. The method does
not manufacture structure that the source does not contain, and a reader can see here
exactly how little a very short scene gives back.
""")

    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print("wrote {} ({} scenes)".format(args.out, len(wanted)))
    return 0


def _dialogue_ratio(scene, source: str) -> float:
    lines = [line.strip() for line in scene.text(source).splitlines() if line.strip()]
    if not lines:
        return 0.0
    speaking = sum(1 for line in lines if line.isupper() and len(line) < 40)
    return min(1.0, speaking * 2 / len(lines))


if __name__ == "__main__":
    raise SystemExit(main())
