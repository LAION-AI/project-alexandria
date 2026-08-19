"""MCQ instrument: build it from source scenes, then answer it under four context arms.

Questions are generated from the **source scene text**, never from the Knowledge Units.
Grading KUs against questions the KU apparatus produced would measure compliance rather
than correctness — the first of the eight measurement errors this project inherited.

Two additions to the published protocol, because without them the comparison is not
trustworthy at this scale:

* **n samples per question per arm**, not one greedy sample. Single-sample cells leave
  within-condition variance unmeasured, and it may exceed the differences being reported.
* **Option order shuffled per sample.** Small instruction-tuned models carry a substantial
  position prior; unshuffled, an arm can win on letter bias alone.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .client import EndpointPool
from .scenes import Scene


LETTERS = "ABCD"


# ------------------------------------------------------------------ sampling


def sample_scenes(scenes: Sequence[Scene], count: int = 25, seed: int = 20260819) -> List[Scene]:
    """Stratify by act position and by length, then freeze.

    The sample is drawn once and must not be re-picked: new numbers over a different sample
    are not comparable with earlier ones, which is the reason the sibling project's arms can
    be compared at all. Scenes under 20 words are excluded — four separable questions cannot
    be written about a dozen words, and including them would measure the instrument's floor
    rather than the artifact.
    """
    eligible = [scene for scene in scenes if scene.word_count >= 20]
    if not eligible:
        return []
    ordered = sorted(eligible, key=lambda scene: scene.index)
    thirds = [ordered[i::3] for i in range(3)]  # act position, interleaved to stay spread

    rng = random.Random(seed)
    # Nine strata: three act positions by three length terciles. Distribute `count` across
    # them and then top up from whatever is left, because integer division across nine
    # cells silently loses scenes — the first version of this asked for 25 and returned 18,
    # which would have made the instrument smaller than reported without saying so.
    strata: List[List[Scene]] = []
    for group in thirds:
        by_length = sorted(group, key=lambda scene: scene.word_count)
        strata.extend(tercile for tercile in (by_length[i::3] for i in range(3)) if tercile)

    picked: List[Scene] = []
    taken = {id(stratum): 0 for stratum in strata}
    while len(picked) < count:
        progressed = False
        for stratum in strata:
            if len(picked) >= count:
                break
            position = taken[id(stratum)]
            if position >= len(stratum):
                continue
            pool = [scene for scene in stratum if scene not in picked]
            if not pool:
                continue
            picked.append(rng.choice(pool))
            taken[id(stratum)] = position + 1
            progressed = True
        if not progressed:
            break  # every stratum exhausted; fewer eligible scenes than requested
    return sorted(picked, key=lambda scene: scene.index)


# ---------------------------------------------------------------- generation


_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "minLength": 15},
                    "options": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 4, "maxItems": 4,
                    },
                    "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
                    "fact_type": {
                        "type": "string",
                        "enum": ["who_did_what", "sequence", "quantity_or_name",
                                 "cause", "location", "state_change"],
                    },
                },
                "required": ["question", "options", "correct_index", "fact_type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


_GEN_PROMPT = """\
Write {count} multiple-choice questions testing detailed factual recall of ONE scene.

Each question must be answerable ONLY by someone who has read this specific scene closely.
Four options, exactly one correct.

The distractors are the hard part and the whole value of the instrument:

  - Each distractor must be plausible to someone who knows the film in general but has not
    read THIS scene. Draw them from things that happen elsewhere in the story, or from
    near-misses on the actual detail: a different character performing the action, a
    neighbouring location, an adjacent step in the sequence, a close but wrong number.
  - A distractor that is absurd, anachronistic, or eliminable by general knowledge is
    worthless — it turns a four-way question into a two-way one.
  - Do NOT quote the scene's wording in the question or in any option. Describe the fact.
  - Do not write questions answerable from the film's general reputation. If someone who
    had merely heard the film described could answer it, discard it and write another.

Spread the {count} questions across different `fact_type` values where the scene supports it.

SCENE HEADING: {heading}

SCENE TEXT:
{text}
"""


@dataclass
class Question:
    question_id: str
    scene_id: str
    scene_index: int
    question: str
    options: List[str]
    correct_index: int
    fact_type: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Question":
        return cls(**{key: value[key] for key in cls.__dataclass_fields__})


def generate_for_scene(
    pool: EndpointPool, scene: Scene, source: str, count: int = 4
) -> List[Question]:
    from .kuschema import grammar_safe

    result = pool.call(
        "You write difficult, fair multiple-choice questions. Return only valid JSON.",
        _GEN_PROMPT.format(count=count, heading=scene.heading_raw, text=scene.text(source)),
        schema=grammar_safe(_QUESTION_SCHEMA),
        max_tokens=4096,
        temperature=0.6,
    )
    text = result.text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    payload = json.loads(text)
    questions = []
    for position, item in enumerate(payload.get("questions") or []):
        options = item.get("options") or []
        if len(options) != 4 or not (0 <= item.get("correct_index", -1) <= 3):
            continue
        if len({option.strip().casefold() for option in options}) != 4:
            continue  # duplicate options make the question unanswerable
        questions.append(
            Question(
                question_id="{}-q{}".format(scene.scene_id, position + 1),
                scene_id=scene.scene_id,
                scene_index=scene.index,
                question=item["question"].strip(),
                options=[option.strip() for option in options],
                correct_index=int(item["correct_index"]),
                fact_type=item.get("fact_type", "who_did_what"),
            )
        )
    return questions


# ------------------------------------------------------------------ answering


_ANSWER_PROMPT = """\
Answer the multiple-choice question using CONTEXT when it is relevant.

Reply with exactly one capital letter surrounded by semicolons, like ;A; — nothing else.
If CONTEXT does not contain the answer, choose the most likely option anyway.

CONTEXT:
{context}

QUESTION:
{question}
{options}
"""


def render(question: Question, order: Sequence[int]) -> Tuple[str, int]:
    """Render options in ``order``; return the prompt block and the correct letter index."""
    lines = []
    correct_position = 0
    for position, original in enumerate(order):
        lines.append("{}) {}".format(LETTERS[position], question.options[original]))
        if original == question.correct_index:
            correct_position = position
    return "\n".join(lines), correct_position


def extract_choice(response: str) -> Optional[str]:
    marked = re.search(r";\s*([ABCD])\s*;", response.upper())
    if marked:
        return marked.group(1)
    bare = response.strip().upper()
    if bare in set(LETTERS):
        return bare
    loose = re.search(r"\b([ABCD])\b", response.upper())
    return loose.group(1) if loose else None


@dataclass
class ArmResult:
    arm: str
    samples: int
    correct: int = 0
    answered: int = 0
    unparseable: int = 0
    by_position: Dict[str, List[int]] = field(default_factory=dict)
    per_question: Dict[str, List[bool]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.answered if self.answered else 0.0


def answer_question(
    pool: EndpointPool,
    question: Question,
    context: str,
    *,
    samples: int = 5,
    seed: int = 0,
    max_context_chars: int = 0,
) -> List[Tuple[Optional[str], int]]:
    """Return ``(letter, correct_position)`` per sample, with options reshuffled each time."""
    rng = random.Random(hash((question.question_id, seed)) & 0xFFFFFFFF)
    out = []
    if max_context_chars and len(context) > max_context_chars:
        raise ValueError(
            "context of {} chars exceeds the {}-char budget; an arm that silently "
            "truncates produces a confident wrong comparison".format(
                len(context), max_context_chars
            )
        )
    for _ in range(samples):
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        block, correct_position = render(question, order)
        result = pool.call(
            "You answer multiple-choice questions precisely.",
            _ANSWER_PROMPT.format(
                context=context or "[no context provided]",
                question=question.question,
                options=block,
            ),
            max_tokens=12,
            temperature=0.7,
        )
        out.append((extract_choice(result.text), correct_position))
    return out
