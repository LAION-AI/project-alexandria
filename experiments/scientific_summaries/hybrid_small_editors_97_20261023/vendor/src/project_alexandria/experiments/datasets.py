"""Readers for the released Project Alexandria MCQ Parquet datasets."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class EvaluationDocument:
    document_id: str
    source_index: int
    text: str
    questions: List[str]
    answers: List[str]


def _document_id(text_digest: str, source_index: int) -> str:
    """Identify a dataset row/group without conflating duplicate source texts."""
    return "{}:{}".format(text_digest, source_index)


def _valid_pairs(
    questions: Sequence[Any], answers: Sequence[Any]
) -> List[Tuple[str, str]]:
    pairs = []
    for question, answer in zip(questions, answers):
        choice = str(answer or "").strip().upper()
        if question and choice in {"A", "B", "C", "D"}:
            pairs.append((str(question), choice))
    return pairs


def load_evaluation_parquet(
    path: str, limit: int = 0, seed: int = 250219413, shuffle: bool = False
) -> List[EvaluationDocument]:
    """Load list-valued abstract rows or group expanded long-paper rows by source text."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("evaluation datasets require: pip install 'project-alexandria[eval]'") from error

    records: List[Dict[str, Any]] = parquet.read_table(path).to_pylist()
    documents = []
    if records and isinstance(records[0].get("question"), list):
        for index, record in enumerate(records):
            pairs = _valid_pairs(record.get("question") or [], record.get("answer") or [])
            if not record.get("text") or not pairs:
                continue
            digest = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
            documents.append(
                EvaluationDocument(
                    _document_id(digest, index),
                    index,
                    record["text"],
                    [pair[0] for pair in pairs],
                    [pair[1] for pair in pairs],
                )
            )
    else:
        grouped: Dict[str, EvaluationDocument] = {}
        for index, record in enumerate(records):
            text = record.get("text") or ""
            pairs = _valid_pairs([record.get("question")], [record.get("answer")])
            if not text or not pairs:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest not in grouped:
                grouped[digest] = EvaluationDocument(
                    _document_id(digest, index), index, text, [], []
                )
            grouped[digest].questions.append(pairs[0][0])
            grouped[digest].answers.append(pairs[0][1])
        documents = list(grouped.values())

    if shuffle:
        random.Random(seed).shuffle(documents)
    if limit > 0:
        documents = documents[:limit]
    return documents


def dataset_manifest(path: str, documents: Sequence[EvaluationDocument]) -> Dict[str, Any]:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    selection = hashlib.sha256()
    for document in documents:
        selection.update(document.document_id.encode("utf-8"))
        selection.update(b"\n")
    return {
        "filename": source.name,
        "sha256": digest.hexdigest(),
        "selected_documents": len(documents),
        "selected_questions": sum(len(document.questions) for document in documents),
        "ordered_selection_sha256": selection.hexdigest(),
    }
