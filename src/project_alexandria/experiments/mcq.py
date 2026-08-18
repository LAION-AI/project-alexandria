"""Prompt helpers for the paper's lower-bound/original/KU MCQ protocol."""

from __future__ import annotations

import re
from typing import Optional


def question_generation_prompt(text: str, count: int = 3) -> str:
    return """Create {count} difficult multiple-choice questions answerable only from SOURCE.
Each must test a verifiable fact, number, definition, method, or relation. Use four options,
mark exactly one correct answer, and return JSON with a questions list.

SOURCE:
{text}""".format(count=count, text=text)


def answer_prompt(question: str, context: str = "") -> str:
    return """Use CONTEXT when relevant. Answer with only A, B, C, or D.

CONTEXT:
{context}

QUESTION:
{question}""".format(context=context or "[no context]", question=question)


def extract_choice(response: str) -> Optional[str]:
    match = re.search(r"(?:^|\b)([ABCD])(?:\b|$)", response.upper())
    return match.group(1) if match else None
