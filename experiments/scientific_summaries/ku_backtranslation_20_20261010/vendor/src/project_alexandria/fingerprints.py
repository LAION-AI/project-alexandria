"""Non-reversible source references for attribution and duplicate detection."""

from __future__ import annotations

import hashlib
import re
from typing import List


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sentence_minhash(text: str, permutations: int = 16) -> List[int]:
    """Return a deterministic token-set MinHash without storing source text."""
    tokens = {token.casefold() for token in re.findall(r"\w+", text, flags=re.UNICODE)}
    if not tokens:
        return []
    maximum = (1 << 64) - 1
    values = []
    for seed in range(permutations):
        smallest = maximum
        prefix = seed.to_bytes(4, "little")
        for token in tokens:
            digest = hashlib.blake2b(prefix + token.encode("utf-8"), digest_size=8).digest()
            smallest = min(smallest, int.from_bytes(digest, "little"))
        values.append(smallest)
    return values
