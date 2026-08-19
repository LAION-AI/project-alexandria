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


def historical_answer_prompt(question: str, context: str = "") -> str:
    """Faithful copy of the prompt used for the published MCQ scores."""
    prompt = """Input: Context: This is a descriptive text or passage that provides background information necessary to understand the question. try your best to make good use of the following knowledge to answer the multiple choice question below if there is no useful information in the following context try to answer the question below as low as good as possible. if there is no useful information in this context to answer the question then just trust your gut feeling and your intuition.
      Questions: these are questions that need to be answered. The answer choice should be clearly marked (e.g., ';A;').

      Read the following context carefully:
      CONTEXT:
      {context}

      Based on the context, answer the following question by providing the capitalized letter corresponding to the most appropriate answer choice:

      {question}

      Example context:

      Today is Earth Day, a day dedicated to celebrating our planet and raising awareness about environmental issues.

      Examplequestion:

      A) Which of the following is NOT a renewable resource?
      B) Coal
      C) Solar energy
      D) Wind power

      Example output:

      ;B;
      Give your answer to the question. The answer choice should be clearly marked (e.g., ';A;').
      if there is no useful info in the CONTEXT guess the right answer. In any case output a valid answer according to the output format with a semicolons surrounding the capital letter.
      It is very important to me that you fulfill this task very accurately and intelligently.
      If you perform well, I will tip you 100 billion dollars.
      answer=""".format(context=context or "", question=question)
    return historical_sanitize(prompt)


def historical_sanitize(value: str) -> str:
    """Reproduce the legacy ASCII filter applied by the MCQ answering wrapper."""
    allowed_special_chars = r"\.,\-\+\\/\*%$!?[\]\(\){}"
    pattern = r"[^a-zA-Z0-9\s{}]".format(allowed_special_chars)
    return re.sub(pattern, "", value)


def extract_choice(response: str) -> Optional[str]:
    match = re.search(r"(?:^|\b)([ABCD])(?:\b|$)", response.upper())
    return match.group(1) if match else None


def extract_historical_choice(response: str) -> Optional[str]:
    """Accept the historical ``;A;`` shape or a bare single-letter response."""
    marked = re.search(r";\s*([ABCD])\s*;", response.upper())
    if marked:
        return marked.group(1)
    bare = response.strip().upper()
    return bare if bare in {"A", "B", "C", "D"} else None
