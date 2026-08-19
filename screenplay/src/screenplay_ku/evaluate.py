"""Run the MCQ instrument across context arms and report with uncertainty.

Arms:
  none       nothing — parametric knowledge only. This is the instrument's calibration,
             not a formality: the source is a widely described film, and any question a
             model answers without context measures its memory, not the artifact.
  full_text  the complete screenplay. Upper bound.
  ku_chain   the complete Knowledge Unit chain, no source text. The claim under test.
  ku_scene   the questioned scene's units plus two either side. Separates whether an arm
             fails at representing the fact or at finding it.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .client import EndpointPool, run_parallel
from .mcq import LETTERS, Question, answer_question
from .render import estimate_tokens, neighbourhood, render_chain


@dataclass
class Cell:
    arm: str
    question_id: str
    scene_id: str
    correct: int
    total: int
    unparseable: int
    chosen_positions: List[int]


def _bootstrap_ci(
    per_question: Sequence[float], iterations: int = 2000, seed: int = 7
) -> Tuple[float, float]:
    """Bootstrap over questions, which is the unit of independence — not over samples."""
    if not per_question:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    size = len(per_question)
    for _ in range(iterations):
        draw = [per_question[rng.randrange(size)] for _ in range(size)]
        means.append(sum(draw) / size)
    means.sort()
    return (means[int(0.025 * iterations)], means[int(0.975 * iterations)])


def run_arm(
    pool: EndpointPool,
    questions: Sequence[Question],
    arm: str,
    context_for: Any,
    *,
    samples: int = 5,
    workers: int = 32,
    max_context_chars: int = 0,
    progress=None,
) -> List[Cell]:
    def work(question: Question) -> Cell:
        context = context_for(question) if callable(context_for) else context_for
        results = answer_question(
            pool, question, context, samples=samples,
            max_context_chars=max_context_chars,
        )
        correct = unparseable = 0
        positions: List[int] = []
        for letter, correct_position in results:
            if letter is None:
                unparseable += 1
                continue
            positions.append(LETTERS.index(letter))
            if LETTERS.index(letter) == correct_position:
                correct += 1
        return Cell(arm, question.question_id, question.scene_id, correct,
                    len(results) - unparseable, unparseable, positions)

    results = run_parallel(list(questions), work, max_workers=workers, on_done=progress)
    cells = []
    for question, result in zip(questions, results):
        if isinstance(result, Exception):
            cells.append(Cell(arm, question.question_id, question.scene_id, 0, 0, samples, []))
        else:
            cells.append(result)
    return cells


def summarise(cells: Sequence[Cell], leaky: Optional[set] = None) -> Dict[str, Any]:
    leaky = leaky or set()

    def stats(subset: Sequence[Cell]) -> Dict[str, Any]:
        answered = sum(cell.total for cell in subset)
        correct = sum(cell.correct for cell in subset)
        per_question = [
            cell.correct / cell.total for cell in subset if cell.total
        ]
        low, high = _bootstrap_ci(per_question)
        return {
            "questions": len(subset),
            "samples_answered": answered,
            "accuracy": round(correct / answered, 4) if answered else None,
            "ci95": [round(low, 4), round(high, 4)],
            "unparseable": sum(cell.unparseable for cell in subset),
        }

    positions = Counter()
    for cell in cells:
        positions.update(cell.chosen_positions)
    total_positions = sum(positions.values()) or 1
    return {
        "arm": cells[0].arm if cells else "",
        "all": stats(cells),
        "non_leaky": stats([cell for cell in cells if cell.question_id not in leaky]),
        "position_bias": {
            LETTERS[index]: round(positions.get(index, 0) / total_positions, 3)
            for index in range(4)
        },
    }


def find_leaky(none_cells: Sequence[Cell], *, threshold: float = 0.6) -> set:
    """Questions the student answers without context above ``threshold``.

    Chance is 0.25. A question well above it with no context is being answered from the
    model's prior knowledge of the film, so it cannot discriminate between context arms and
    is reported as a separate stratum rather than quietly averaged in.
    """
    return {
        cell.question_id
        for cell in none_cells
        if cell.total and (cell.correct / cell.total) >= threshold
    }
