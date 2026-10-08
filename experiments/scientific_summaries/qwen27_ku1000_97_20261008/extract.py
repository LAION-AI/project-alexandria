"""Sequential 1,000-word targets; document parallelism without QA feedback."""
import concurrent.futures
from dataclasses import asdict
import os
import time

from common import ROOT, TracedBackend, digest, load, sources, write
from project_alexandria.chunking import split_text
from project_alexandria.pipeline import ExtractionConfig, KnowledgeUnitPipeline
from project_alexandria.prompts import SYSTEM_PROMPT, sequential_prompt
from project_alexandria.parsing import parse_unit_response
from project_alexandria.schema import DocumentResult, KnowledgeUnit

CONFIG = ExtractionConfig(mode="sequential", chunk_words=1000, context_words=0,
    previous_units=10, canonicalize=False, parse_retries=0)


def extract_document(paper, index, backend):
    docid = paper["document_id"]
    backend.local.document_id = docid
    directory = ROOT / "outputs/knowledge_units" / docid
    complete = directory / "document.json"
    if complete.exists():
        saved = load(complete)
        if saved["source_sha256"] != paper["fulltext_sha256"] or saved["config"] != asdict(CONFIG):
            raise ValueError("Incompatible extraction checkpoint")
        return saved
    chunks = split_text(paper["fulltext"], 1000)
    assert " ".join(c.text for c in chunks).split() == paper["fulltext"].split()
    assert all(0 < c.word_count <= 1000 for c in chunks)
    title = paper["source"].get("source_title") or ""
    pipeline = KnowledgeUnitPipeline(backend, CONFIG)
    units, failures, records = [], [], []
    started = time.monotonic()
    for chunk in chunks:
        backend.local.chunk_index = chunk.index
        backend.local.seed = 250219413 + index * 1000 + chunk.index
        checkpoint = directory / "chunks" / f"{chunk.index:04d}.json"
        if checkpoint.exists():
            record = load(checkpoint)
            if record["source_chunk_sha256"] != digest(chunk.text):
                raise ValueError("Chunk checkpoint source changed")
            if record["status"] == "generated":
                units.append(KnowledgeUnit.from_dict(record["unit"]))
            else:
                failures.append(chunk.index)
            records.append(record)
            continue
        # No automatic 350-word opening abstract and no neighboring source text:
        # each call receives <=1000 new source words plus generated naming context.
        prompt = sequential_prompt(chunk, units[-10:], title, "")
        record = dict(document_id=docid, chunk_index=chunk.index,
                      source_chunk_sha256=digest(chunk.text), word_count=chunk.word_count,
                      start_word=chunk.start_word, end_word=chunk.end_word, attempts=[])
        for attempt, budget in enumerate([8192, 16384]):
            request_prompt = prompt if attempt == 0 else prompt + (
                "\nReturn a complete compact JSON object. Preserve the stated facts; "
                "omit duplicate wording and commentary, and close every array and object.")
            try:
                response = backend.generate(SYSTEM_PROMPT, request_prompt, max_tokens=budget)
                request_record = dict(backend.local.last)
                record["attempts"].append(request_record)
                if request_record["finish_reason"] == "length":
                    raise ValueError("Incomplete output: generation token budget exhausted")
                # Strict parse before the repository attaches source fingerprints.
                # Never accept its optional truncated-response salvage path.
                parse_unit_response(response)
                unit = pipeline._unit(chunk, response, request_prompt)
                units.append(unit)
                record.update(status="generated", unit=unit.to_dict())
                break
            except Exception as error:
                record.setdefault("errors", []).append(repr(error))
        else:
            failures.append(chunk.index)
            record["status"] = "generation_failed"
        write(checkpoint, record)
        records.append(record)
    result = DocumentResult(schema_version="1.0", title=title, abstract_sha256="",
        mode="sequential", model="Qwen/Qwen3.8-27B-FP8", config=asdict(CONFIG),
        knowledge_units=units, canonical_entities=[])
    value = dict(document_id=docid, source_sha256=paper["fulltext_sha256"],
        config=asdict(CONFIG), result=result.to_dict(), status="generated" if not failures else "generation_failed",
        chunks_expected=len(chunks), chunks_generated=len(units), failed_chunks=failures,
        source_words=len(paper["fulltext"].split()),
        elapsed_seconds=time.monotonic()-started,
        prompt_tokens=sum(a.get("usage", {}).get("prompt_tokens", 0) for r in records for a in r["attempts"]),
        completion_tokens=sum(a.get("usage", {}).get("completion_tokens", 0) for r in records for a in r["attempts"]))
    write(complete, value)
    print("KU_DOCUMENT_COMPLETE", docid, value["status"], len(units), flush=True)
    return value


def run(endpoint, replica, replicas=3):
    papers = sources()
    backend = TracedBackend(endpoint, "qwen27-ku", "extraction", max_tokens=8192,
        temperature=.1, concurrency=16, context_limit=32768)
    start = time.monotonic()
    assigned = [(i, p) for i, p in enumerate(papers) if i % replicas == replica]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(extract_document, p, i, backend) for i, p in assigned]
        documents = [f.result() for f in futures]
    report = dict(replica=replica, documents=len(documents),
        successful_documents=sum(d["status"] == "generated" for d in documents),
        elapsed_seconds=time.monotonic()-start,
        completion_tokens=sum(d["completion_tokens"] for d in documents),
        prompt_tokens=sum(d["prompt_tokens"] for d in documents),
        newly_resumed_output_note="Token totals include saved attempts; phase elapsed may exclude prior allocations.")
    write(ROOT / f"outputs/extraction-replica-{replica}.json", report)
    return report
