"""Deterministic, sentence-aware word chunking with neighboring context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+(?=[A-Z0-9\"'(\[])" )


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    start_word: int
    end_word: int
    before: str = ""
    after: str = ""

    @property
    def word_count(self) -> int:
        return self.end_word - self.start_word


def _sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(normalized) if part.strip()]


def split_text(text: str, target_words: int = 500) -> List[TextChunk]:
    """Split near ``target_words`` without splitting ordinary sentences."""
    if target_words < 1:
        raise ValueError("target_words must be positive")
    sentences = _sentences(text)
    raw_chunks = []
    current = []
    current_count = 0
    for sentence in sentences:
        words = sentence.split()
        if len(words) > target_words:
            if current:
                raw_chunks.append(" ".join(current))
                current, current_count = [], 0
            for offset in range(0, len(words), target_words):
                raw_chunks.append(" ".join(words[offset : offset + target_words]))
            continue
        if current and current_count + len(words) > target_words:
            raw_chunks.append(" ".join(current))
            current, current_count = [], 0
        current.append(sentence)
        current_count += len(words)
    if current:
        raw_chunks.append(" ".join(current))

    chunks = []
    cursor = 0
    for index, chunk_text in enumerate(raw_chunks):
        count = len(chunk_text.split())
        chunks.append(TextChunk(index, chunk_text, cursor, cursor + count))
        cursor += count
    return chunks


def add_neighbor_context(chunks: List[TextChunk], context_words: int = 1000) -> List[TextChunk]:
    """Attach up to N source words before and after each target chunk."""
    if context_words < 0:
        raise ValueError("context_words cannot be negative")
    all_words = " ".join(chunk.text for chunk in chunks).split()
    contextualized = []
    for chunk in chunks:
        before = " ".join(all_words[max(0, chunk.start_word - context_words) : chunk.start_word])
        after = " ".join(all_words[chunk.end_word : chunk.end_word + context_words])
        contextualized.append(
            TextChunk(chunk.index, chunk.text, chunk.start_word, chunk.end_word, before, after)
        )
    return contextualized
