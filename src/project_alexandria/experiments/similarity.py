"""Embedding-control transformations from the unreleased report."""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


def scramble(text: str, group_size: int = 1, seed: int = 0) -> str:
    """Shuffle words, bigrams, or trigrams reproducibly."""
    if group_size < 1:
        raise ValueError("group_size must be positive")
    words = text.split()
    groups = [words[index : index + group_size] for index in range(0, len(words), group_size)]
    random.Random(seed).shuffle(groups)
    return " ".join(word for group in groups for word in group)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have equal dimensions")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
