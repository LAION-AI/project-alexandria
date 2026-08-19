"""Group scenes into extraction windows.

Default windowing tiles on **scene boundaries**: each scene belongs to exactly one
window's target span, so duplicate and missing scenes are structurally impossible rather
than repaired by a later pass. In this corpus the longest scene is 714 words against a
~1,110-word window, so no scene is ever split and the overlap scheme the page variant
needs has nothing to catch.

``pages`` windowing is retained so the two can be compared rather than argued about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .scenes import Scene


WORDS_PER_PAGE = 185  # 24,002 words over ~130 script pages; a config value, not a constant


@dataclass(frozen=True)
class Window:
    index: int
    scenes: List[Scene]
    start_char: int
    end_char: int
    context_start: int
    context_end: int
    is_canary: bool = False

    @property
    def scene_ids(self) -> List[str]:
        return [scene.scene_id for scene in self.scenes]

    @property
    def word_count(self) -> int:
        return sum(scene.word_count for scene in self.scenes)

    def target_text(self, source: str) -> str:
        return "" if self.is_canary else source[self.start_char : self.end_char]

    def before_text(self, source: str) -> str:
        return source[self.context_start : self.start_char]

    def after_text(self, source: str) -> str:
        return source[self.end_char : self.context_end]


def build_windows(
    scenes: Sequence[Scene],
    source: str,
    *,
    pages_target: int = 6,
    pages_context: int = 5,
    words_per_page: int = WORDS_PER_PAGE,
    max_scenes: int = 12,
) -> List[Window]:
    """Accumulate whole scenes until adding the next would exceed the target size.

    ``max_scenes`` bounds how many KUs one call must emit. Word count alone does not:
    the median scene here is 45 words, so a window can hit its word target while holding
    31 scenes, and a 31-KU response risks running into ``max_tokens``. Truncation would
    drop trailing scenes *silently* — the coverage check would catch it, but only after a
    wasted run, and a check that fires late is worse than a bound that makes it impossible.
    """
    if not scenes:
        raise ValueError("no scenes to window")
    target_words = pages_target * words_per_page
    context_chars = pages_context * words_per_page * _chars_per_word(source, scenes)

    groups: List[List[Scene]] = []
    current: List[Scene] = []
    current_words = 0
    for scene in scenes:
        too_long = current and current_words + scene.word_count > target_words
        too_many = len(current) >= max_scenes
        if too_long or too_many:
            groups.append(current)
            current, current_words = [], 0
        current.append(scene)
        current_words += scene.word_count
    if current:
        groups.append(current)

    windows = []
    for index, group in enumerate(groups):
        start, end = group[0].start_char, group[-1].end_char
        windows.append(
            Window(
                index=index,
                scenes=list(group),
                start_char=start,
                end_char=end,
                context_start=max(0, start - int(context_chars)),
                context_end=min(len(source), end + int(context_chars)),
            )
        )
    return windows


def build_page_windows(
    scenes: Sequence[Scene],
    source: str,
    *,
    pages_target: int = 6,
    pages_context: int = 5,
    overlap_pages: int = 2,
    words_per_page: int = WORDS_PER_PAGE,
) -> List[Window]:
    """Page-aligned windows with overlap; a scene goes to the window holding its slugline.

    Targets overlap, so the same scene can be claimed by two windows. Stage 2 then has a
    real dedup responsibility. Kept for comparison against the scene-aligned default.
    """
    chars_per_word = _chars_per_word(source, scenes)
    step_chars = int((pages_target - overlap_pages) * words_per_page * chars_per_word)
    span_chars = int(pages_target * words_per_page * chars_per_word)
    context_chars = int(pages_context * words_per_page * chars_per_word)
    if step_chars <= 0:
        raise ValueError("overlap_pages must be smaller than pages_target")

    windows = []
    index = 0
    cursor = 0
    while cursor < len(source):
        end = min(len(source), cursor + span_chars)
        owned = [scene for scene in scenes if cursor <= scene.start_char < end]
        if owned:
            windows.append(
                Window(
                    index=index,
                    scenes=owned,
                    start_char=cursor,
                    end_char=end,
                    context_start=max(0, cursor - context_chars),
                    context_end=min(len(source), end + context_chars),
                )
            )
            index += 1
        if end >= len(source):
            break
        cursor += step_chars
    return windows


def canary_window(windows: Sequence[Window], source: str) -> Window:
    """A window with an empty target span and a scene list not inside it.

    Anything it emits is recall from the surrounding context rather than extraction from
    the target. This is the only check that detects failure by construction, and it is the
    one most likely to catch this pipeline's specific risk: every agent sees the whole
    screenplay, so an agent that has stopped reading its window can still sound right.
    """
    if len(windows) < 2:
        raise ValueError("need at least two windows to site a canary")
    donor = windows[len(windows) // 2]
    return Window(
        index=-1,
        scenes=list(donor.scenes),
        start_char=donor.start_char,
        end_char=donor.start_char,  # empty span
        context_start=donor.context_start,
        context_end=donor.context_end,
        is_canary=True,
    )


def seam_pairs(windows: Sequence[Window]) -> List[tuple]:
    """Consecutive window pairs. Seams are independent, so all of these run concurrently."""
    return [(windows[i], windows[i + 1]) for i in range(len(windows) - 1)]


def _chars_per_word(source: str, scenes: Sequence[Scene]) -> float:
    words = sum(scene.word_count for scene in scenes)
    return (len(source) / words) if words else 6.0
