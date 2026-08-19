"""Resumable reproduction of the paper's no-context/original/KU MCQ protocol."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..backends import GenerationBackend
from ..io import write_json_atomic
from ..pipeline import KnowledgeUnitPipeline
from ..schema import DocumentResult
from .datasets import EvaluationDocument, dataset_manifest
from .mcq import extract_historical_choice, historical_answer_prompt


JUDGE_SYSTEM_PROMPT = (
    "You are a very smart very intelligence assistant who is very helpful."
)


def knowledge_unit_context(result: DocumentResult) -> str:
    """Serialize only factual KU content, excluding fingerprints and source metadata."""
    units = []
    for unit in result.knowledge_units:
        units.append(
            {
                "context_summary": unit.context_summary,
                "entities": [
                    {
                        "name": entity.name,
                        "type": entity.entity_type,
                        "attributes": entity.attributes,
                        "relationships": [asdict(relation) for relation in entity.relationships],
                    }
                    for entity in unit.entities
                ],
            }
        )
    return json.dumps(units, ensure_ascii=False, separators=(",", ":"))


def _answer_batch(
    backend: GenerationBackend,
    documents: Sequence[EvaluationDocument],
    ku_results: Sequence[DocumentResult],
) -> List[Dict[str, Any]]:
    prompts = []
    slots = []
    prompt_by_slot = {}
    for document, ku_result in zip(documents, ku_results):
        contexts = {
            "no_context": "",
            "original": document.text,
            "knowledge_units": knowledge_unit_context(ku_result),
        }
        for question_index, question in enumerate(document.questions):
            for condition, context in contexts.items():
                slot = (document.document_id, question_index, condition)
                prompt = historical_answer_prompt(question, context)
                prompts.append(prompt)
                slots.append(slot)
                prompt_by_slot[slot] = prompt
    responses = backend.generate_batch(JUDGE_SYSTEM_PROMPT, prompts, max_tokens=100)
    if len(responses) != len(prompts):
        raise RuntimeError("judge returned a different number of responses than prompts")
    predictions = {}
    for slot, response in zip(slots, responses):
        choice = extract_historical_choice(response)
        predictions[slot] = choice
    retry_slots = [slot for slot, choice in predictions.items() if choice is None]
    for _ in range(4):
        if not retry_slots:
            break
        retry_responses = backend.generate_batch(
            JUDGE_SYSTEM_PROMPT,
            [prompt_by_slot[slot] for slot in retry_slots],
            max_tokens=100,
        )
        if len(retry_responses) != len(retry_slots):
            raise RuntimeError("judge returned a different number of retry responses than prompts")
        for slot, response in zip(retry_slots, retry_responses):
            predictions[slot] = extract_historical_choice(response)
        retry_slots = [slot for slot in retry_slots if predictions[slot] is None]
    rows = []
    for document in documents:
        for question_index, gold in enumerate(document.answers):
            rows.append(
                {
                    "document_id": document.document_id,
                    "source_index": document.source_index,
                    "question_index": question_index,
                    "gold": gold,
                    "predictions": {
                        condition: predictions.get(
                            (document.document_id, question_index, condition)
                        )
                        for condition in ("no_context", "original", "knowledge_units")
                    },
                }
            )
    return rows


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {}
    for condition in ("no_context", "original", "knowledge_units"):
        valid = [row for row in rows if row["predictions"].get(condition) is not None]
        correct = sum(row["predictions"][condition] == row["gold"] for row in valid)
        summary[condition] = {
            "correct": correct,
            "valid": len(valid),
            "total": len(rows),
            "accuracy": correct / len(rows) if rows else None,
            "valid_accuracy": correct / len(valid) if valid else None,
            "invalid": len(rows) - len(valid),
        }
    return summary


def extract_ku_cache(
    dataset_path: str,
    documents: Sequence[EvaluationDocument],
    pipeline: KnowledgeUnitPipeline,
    output_path: str,
    document_batch_size: int = 8,
) -> Dict[str, Any]:
    """Extract source-free KUs in resumable batches for a later fixed-judge pass."""
    if document_batch_size < 1:
        raise ValueError("document_batch_size must be positive")
    manifest = dataset_manifest(dataset_path, documents)
    output: Dict[str, Any] = {
        "schema_version": "1.0",
        "dataset": manifest,
        "extractor_model": pipeline.backend.model_name,
        "extraction_config": asdict(pipeline.config),
        "documents": [],
        "elapsed_seconds": 0.0,
    }
    destination = Path(output_path)
    if destination.exists():
        output = json.loads(destination.read_text(encoding="utf-8"))
        for field, expected in (
            ("dataset", manifest),
            ("extractor_model", pipeline.backend.model_name),
            ("extraction_config", asdict(pipeline.config)),
        ):
            if output.get(field) != expected:
                raise ValueError("KU cache has incompatible {}".format(field))
    completed = {item["document_id"] for item in output.get("documents", [])}
    remaining = [document for document in documents if document.document_id not in completed]
    started = time.time()
    for start in range(0, len(remaining), document_batch_size):
        batch = remaining[start : start + document_batch_size]
        results = pipeline.extract_many(
            [{"text": item.text, "title": "", "abstract": ""} for item in batch]
        )
        output["documents"].extend(
            {"document_id": document.document_id, "result": result.to_dict()}
            for document, result in zip(batch, results)
        )
        output["elapsed_seconds"] = float(output.get("elapsed_seconds", 0.0)) + (
            time.time() - started
        )
        started = time.time()
        write_json_atomic(output_path, output)
    if not remaining:
        write_json_atomic(output_path, output)
    return output


def judge_ku_cache(
    dataset_path: str,
    documents: Sequence[EvaluationDocument],
    ku_cache_path: str,
    judge_backend: GenerationBackend,
    output_path: str,
    document_batch_size: int = 8,
) -> Dict[str, Any]:
    """Evaluate an extracted KU cache with a fixed answerer, resumably."""
    if document_batch_size < 1:
        raise ValueError("document_batch_size must be positive")
    manifest = dataset_manifest(dataset_path, documents)
    cache = json.loads(Path(ku_cache_path).read_text(encoding="utf-8"))
    if cache.get("dataset") != manifest:
        raise ValueError("KU cache belongs to a different dataset selection")
    cached_results = {
        item["document_id"]: DocumentResult.from_dict(item["result"])
        for item in cache.get("documents", [])
    }
    missing = [item.document_id for item in documents if item.document_id not in cached_results]
    if missing:
        raise ValueError("KU cache is incomplete ({} documents missing)".format(len(missing)))

    output: Dict[str, Any] = {
        "schema_version": "1.1",
        "dataset": manifest,
        "extractor_model": cache["extractor_model"],
        "judge_model": judge_backend.model_name,
        "extraction_config": cache["extraction_config"],
        "sampling": {
            "judge_temperature": getattr(judge_backend, "temperature", None),
            "judge_max_tokens": 100,
            "judge_top_p": 0.95,
            "judge_frequency_penalty": getattr(judge_backend, "frequency_penalty", None),
            "judge_presence_penalty": getattr(judge_backend, "presence_penalty", None),
            "judge_prompt": "historical_semicolon_v1",
            "legacy_ascii_sanitizer": True,
            "document_batch_size": document_batch_size,
        },
        "rows": [],
        "summary": {},
        "elapsed_seconds": 0.0,
    }
    destination = Path(output_path)
    if destination.exists():
        loaded = json.loads(destination.read_text(encoding="utf-8"))
        for field in ("dataset", "extractor_model", "judge_model", "extraction_config"):
            if loaded.get(field) != output[field]:
                raise ValueError("judge checkpoint has incompatible {}".format(field))
        stable_sampling = {k: v for k, v in output["sampling"].items() if k != "document_batch_size"}
        stable_loaded = {
            k: v for k, v in loaded.get("sampling", {}).items() if k != "document_batch_size"
        }
        if stable_loaded != stable_sampling:
            raise ValueError("judge checkpoint has incompatible sampling settings")
        output = loaded

    completed = {row["document_id"] for row in output.get("rows", [])}
    remaining = [document for document in documents if document.document_id not in completed]
    started = time.time()
    for start in range(0, len(remaining), document_batch_size):
        batch = remaining[start : start + document_batch_size]
        output["rows"].extend(
            _answer_batch(
                judge_backend,
                batch,
                [cached_results[item.document_id] for item in batch],
            )
        )
        output["summary"] = summarize(output["rows"])
        output["elapsed_seconds"] = float(output.get("elapsed_seconds", 0.0)) + (
            time.time() - started
        )
        started = time.time()
        write_json_atomic(output_path, output)
    if not remaining:
        write_json_atomic(output_path, output)
    return output


def reproduce(
    dataset_path: str,
    documents: Sequence[EvaluationDocument],
    pipeline: KnowledgeUnitPipeline,
    judge_backend: GenerationBackend,
    output_path: str,
    document_batch_size: int = 8,
) -> Dict[str, Any]:
    """Evaluate in resumable document batches and atomically checkpoint after each batch."""
    if document_batch_size < 1:
        raise ValueError("document_batch_size must be positive")
    started = time.time()
    sampling = {
        "extractor_temperature": getattr(pipeline.backend, "temperature", None),
        "extractor_max_tokens": getattr(pipeline.backend, "max_tokens", None),
        "judge_temperature": getattr(judge_backend, "temperature", None),
        "judge_max_tokens": 100,
        "judge_top_p": 0.95,
        "judge_frequency_penalty": getattr(judge_backend, "frequency_penalty", None),
        "judge_presence_penalty": getattr(judge_backend, "presence_penalty", None),
        "judge_prompt": "historical_semicolon_v1",
        "legacy_ascii_sanitizer": True,
        "document_batch_size": document_batch_size,
    }
    output = {
        "schema_version": "1.1",
        "dataset": dataset_manifest(dataset_path, documents),
        "extractor_model": pipeline.backend.model_name,
        "judge_model": judge_backend.model_name,
        "extraction_config": asdict(pipeline.config),
        "sampling": sampling,
        "rows": [],
        "summary": {},
        "elapsed_seconds": 0.0,
    }
    destination = Path(output_path)
    if destination.exists():
        loaded = json.loads(destination.read_text(encoding="utf-8"))
        loaded_dataset = loaded.get("dataset", {})
        if loaded_dataset.get("sha256") != output["dataset"]["sha256"]:
            raise ValueError("resume artifact belongs to a different dataset")
        if (
            loaded_dataset.get("ordered_selection_sha256")
            != output["dataset"]["ordered_selection_sha256"]
        ):
            raise ValueError("resume artifact has a different ordered document selection")
        for field in ("extractor_model", "judge_model", "extraction_config"):
            if loaded.get(field) != output[field]:
                raise ValueError("resume artifact has incompatible {}".format(field))
        loaded_sampling = loaded.get("sampling")
        if loaded_sampling:
            stable_sampling = {k: v for k, v in sampling.items() if k != "document_batch_size"}
            stable_loaded = {
                k: v for k, v in loaded_sampling.items() if k != "document_batch_size"
            }
            if stable_loaded != stable_sampling:
                raise ValueError("resume artifact has incompatible sampling settings")
        output = loaded
        output["schema_version"] = "1.1"
        output["dataset"] = dataset_manifest(dataset_path, documents)
        output.setdefault("sampling", sampling)
        output["summary"] = summarize(output.get("rows", []))
    completed = {row["document_id"] for row in output.get("rows", [])}
    remaining = [document for document in documents if document.document_id not in completed]
    for start in range(0, len(remaining), document_batch_size):
        batch = remaining[start : start + document_batch_size]
        ku_results = pipeline.extract_many(
            [{"text": item.text, "title": "", "abstract": ""} for item in batch]
        )
        output["rows"].extend(_answer_batch(judge_backend, batch, ku_results))
        output["summary"] = summarize(output["rows"])
        output["elapsed_seconds"] = float(output.get("elapsed_seconds", 0.0)) + (
            time.time() - started
        )
        started = time.time()
        write_json_atomic(output_path, output)
    if not remaining:
        write_json_atomic(output_path, output)
    return output
