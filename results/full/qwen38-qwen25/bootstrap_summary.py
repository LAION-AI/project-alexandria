#!/usr/bin/env python3
"""Regenerate summary.csv with paired document-cluster bootstrap intervals."""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


SEED = 250219413
RESAMPLES = 10_000
CONDITIONS = ("no_context", "original", "knowledge_units")
FIELDS = (
    "domain",
    "papers",
    "questions",
    "no_context_percent",
    "no_context_ci_low",
    "no_context_ci_high",
    "original_percent",
    "original_ci_low",
    "original_ci_high",
    "knowledge_units_percent",
    "knowledge_units_ci_low",
    "knowledge_units_ci_high",
    "ku_minus_original_pp",
    "ku_minus_original_ci_low",
    "ku_minus_original_ci_high",
    "invalid_no_context",
    "invalid_original",
    "invalid_knowledge_units",
    "judge_elapsed_seconds",
)


def percentile_interval(values):
    ordered = sorted(values)
    return ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered)) - 1]


def summarize(path: Path, domain: str):
    artifact = json.loads(path.read_text(encoding="utf-8"))
    rows = artifact["rows"]
    by_document = defaultdict(list)
    for row in rows:
        by_document[row["document_id"]].append(row)
    document_ids = list(by_document)
    samples = {condition: [] for condition in CONDITIONS}
    samples["ku_minus_original"] = []
    randomizer = random.Random(SEED)
    for _ in range(RESAMPLES):
        selected = [randomizer.choice(document_ids) for _ in document_ids]
        sample = [row for document_id in selected for row in by_document[document_id]]
        accuracy = {
            condition: sum(row["predictions"][condition] == row["gold"] for row in sample)
            / len(sample)
            for condition in CONDITIONS
        }
        for condition in CONDITIONS:
            samples[condition].append(accuracy[condition])
        samples["ku_minus_original"].append(
            accuracy["knowledge_units"] - accuracy["original"]
        )

    intervals = {
        name: tuple(100 * value for value in percentile_interval(values))
        for name, values in samples.items()
    }
    summary = artifact["summary"]
    points = {condition: 100 * summary[condition]["accuracy"] for condition in CONDITIONS}
    return {
        "domain": domain,
        "papers": len(document_ids),
        "questions": len(rows),
        "no_context_percent": points["no_context"],
        "no_context_ci_low": intervals["no_context"][0],
        "no_context_ci_high": intervals["no_context"][1],
        "original_percent": points["original"],
        "original_ci_low": intervals["original"][0],
        "original_ci_high": intervals["original"][1],
        "knowledge_units_percent": points["knowledge_units"],
        "knowledge_units_ci_low": intervals["knowledge_units"][0],
        "knowledge_units_ci_high": intervals["knowledge_units"][1],
        "ku_minus_original_pp": points["knowledge_units"] - points["original"],
        "ku_minus_original_ci_low": intervals["ku_minus_original"][0],
        "ku_minus_original_ci_high": intervals["ku_minus_original"][1],
        "invalid_no_context": summary["no_context"]["invalid"],
        "invalid_original": summary["original"]["invalid"],
        "invalid_knowledge_units": summary["knowledge_units"]["invalid"],
        "judge_elapsed_seconds": artifact["elapsed_seconds"],
    }


def main():
    directory = Path(__file__).resolve().parent
    results = [
        summarize(directory / "physics-qwen38-qwen25-judge.json", "physics"),
        summarize(directory / "medical-qwen38-qwen25-judge.json", "medical"),
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for result in results:
        writer.writerow(
            {
                key: "{:.3f}".format(value) if isinstance(value, float) else value
                for key, value in result.items()
            }
        )


if __name__ == "__main__":
    main()
