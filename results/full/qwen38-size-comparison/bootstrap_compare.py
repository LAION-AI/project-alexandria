#!/usr/bin/env python3
"""Recompute clustered intervals and paired 4B/9B KU differences."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 250219413
RESAMPLES = 10_000
CONDITIONS = ("no_context", "original", "knowledge_units")


def interval(values):
    values = sorted(values)
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def artifact(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clustered_accuracy(rows, condition):
    by_document = defaultdict(list)
    for row in rows:
        by_document[row["document_id"]].append(row)
    document_ids = list(by_document)
    randomizer = random.Random(SEED)
    samples = []
    for _ in range(RESAMPLES):
        selected = [randomizer.choice(document_ids) for _ in document_ids]
        sample = [row for document_id in selected for row in by_document[document_id]]
        samples.append(
            sum(row["predictions"][condition] == row["gold"] for row in sample) / len(sample)
        )
    return interval(samples)


def summarize(path: Path, model: str, domain: str):
    data = artifact(path)
    rows = data["rows"]
    return {
        "model": model,
        "domain": domain,
        "papers": len({row["document_id"] for row in rows}),
        "questions": len(rows),
        "judge_elapsed_seconds": data["elapsed_seconds"],
        **{
            key: value
            for condition in CONDITIONS
            for key, value in (
                (f"{condition}_percent", 100 * data["summary"][condition]["accuracy"]),
                (f"{condition}_ci_low", 100 * clustered_accuracy(rows, condition)[0]),
                (f"{condition}_ci_high", 100 * clustered_accuracy(rows, condition)[1]),
                (f"{condition}_invalid", data["summary"][condition]["invalid"]),
            )
        },
    }


def paired_difference(path_a: Path, path_b: Path, domain: str):
    rows_a = {
        (row["document_id"], row["question_index"]): row
        for row in artifact(path_a)["rows"]
    }
    rows_b = {
        (row["document_id"], row["question_index"]): row
        for row in artifact(path_b)["rows"]
    }
    by_document = defaultdict(list)
    for key, row_a in rows_a.items():
        row_b = rows_b[key]
        by_document[key[0]].append(
            int(row_b["predictions"]["knowledge_units"] == row_b["gold"])
            - int(row_a["predictions"]["knowledge_units"] == row_a["gold"])
        )
    document_ids = list(by_document)
    randomizer = random.Random(SEED)
    samples = []
    for _ in range(RESAMPLES):
        selected = [randomizer.choice(document_ids) for _ in document_ids]
        values = [value for document_id in selected for value in by_document[document_id]]
        samples.append(sum(values) / len(values))
    low, high = interval(samples)
    return {
        "model": "9B-minus-4B-KU",
        "domain": domain,
        "papers": len(document_ids),
        "questions": len(rows_a),
        "knowledge_units_percent": 100 * sum(samples) / len(samples),
        "knowledge_units_ci_low": 100 * low,
        "knowledge_units_ci_high": 100 * high,
    }


def main():
    directory = Path(__file__).resolve().parent
    rows = []
    for model in ("4B", "9B", "27B"):
        source = directory if model != "27B" else directory.parent / "qwen38-qwen25"
        stem = "qwen38" if model == "27B" else f"qwen38-{model.lower()}"
        for domain in ("physics", "medical"):
            rows.append(
                summarize(
                    source / f"{domain}-{stem}-qwen25-judge.json"
                    if model != "27B"
                    else source / f"{domain}-qwen38-qwen25-judge.json",
                    model,
                    domain,
                )
            )
    for domain in ("physics", "medical"):
        rows.append(
            paired_difference(
                directory / f"{domain}-qwen38-4b-qwen25-judge.json",
                directory / f"{domain}-qwen38-9b-qwen25-judge.json",
                domain,
            )
        )
    fields = sorted({key for row in rows for key in row})
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
