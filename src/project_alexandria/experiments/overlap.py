"""Text-overlap measurements used by the paper."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple


def strip_non_source_context(text: str) -> str:
    text = re.sub(r"<style_analysis>.*?</style_analysis>", "", text, flags=re.DOTALL)
    return re.sub(r"<CONTEXT>.*?</CONTEXT>", "", text, flags=re.DOTALL)


def ngrams(text: str, n: int) -> Iterable[Tuple[str, ...]]:
    words = text.split()
    return (tuple(words[index : index + n]) for index in range(max(0, len(words) - n + 1)))


def ngram_jaccard(left: str, right: str, n: int) -> float:
    left_set, right_set = set(ngrams(left, n)), set(ngrams(right, n))
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def overlap_report(left: str, right: str) -> Dict[str, float]:
    return {"{}_gram_jaccard".format(n): ngram_jaccard(left, right, n) for n in (5, 7, 11)}
