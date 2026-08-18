"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .backends import OpenAICompatibleBackend, VLLMBackend
from .canonicalize import canonicalize_units
from .io import read_documents, write_json_atomic
from .pipeline import ExtractionConfig, KnowledgeUnitPipeline
from .schema import DocumentResult


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alexandria", description="Convert text or documents into Knowledge Units."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract", help="extract Knowledge Units")
    extract.add_argument("input", help="TXT, Markdown, PDF, JSON, JSONL, or CSV file")
    extract.add_argument("--output", "-o", required=True, help="output JSON path")
    extract.add_argument("--text-column", default="text", help="CSV/JSON source-text field")
    extract.add_argument("--offset", type=int, default=0, help="skip this many input records")
    extract.add_argument("--limit", type=int, default=None, help="process at most this many records")
    extract.add_argument("--mode", choices=("sequential", "parallel"), default="sequential")
    extract.add_argument("--chunk-words", type=int, default=None)
    extract.add_argument("--context-words", type=int, default=1000)
    extract.add_argument("--previous-units", type=int, default=10)
    extract.add_argument("--no-canonicalize", action="store_true")
    extract.add_argument("--backend", choices=("openai", "vllm"), default="openai")
    extract.add_argument("--model", required=True)
    extract.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    extract.add_argument("--api-key", default=None, help="prefer ALEXANDRIA_API_KEY")
    extract.add_argument("--concurrency", type=int, default=8)
    extract.add_argument("--max-tokens", type=int, default=4096)
    extract.add_argument("--canonicalization-max-tokens", type=int, default=4096)
    extract.add_argument("--temperature", type=float, default=0.1)
    extract.add_argument(
        "--thinking", action="store_true", help="enable model reasoning mode (off by default)"
    )
    extract.add_argument("--tokenizer", default=None, help="base tokenizer for offline GGUF")
    extract.add_argument("--tensor-parallel-size", type=int, default=1)
    extract.add_argument("--max-model-len", type=int, default=32768)

    canonicalize = subparsers.add_parser(
        "canonicalize", help="rerun document-level entity resolution on an artifact"
    )
    canonicalize.add_argument("input", help="Alexandria JSON artifact")
    canonicalize.add_argument("--output", "-o", required=True)
    canonicalize.add_argument("--model", required=True)
    canonicalize.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    canonicalize.add_argument("--api-key", default=None, help="prefer ALEXANDRIA_API_KEY")
    canonicalize.add_argument("--max-tokens", type=int, default=4096)
    canonicalize.add_argument("--temperature", type=float, default=0.1)
    canonicalize.add_argument("--thinking", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "canonicalize":
        with open(args.input, encoding="utf-8") as handle:
            document = DocumentResult.from_dict(json.load(handle))
        backend = OpenAICompatibleBackend(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            thinking=args.thinking,
        )
        units, entities = canonicalize_units(
            document.knowledge_units,
            backend,
            document.title,
            "",
            max_tokens=args.max_tokens,
        )
        document.knowledge_units = units
        document.canonical_entities = entities
        document.model = backend.model_name
        write_json_atomic(args.output, document.to_dict())
        return 0
    chunk_words = args.chunk_words
    if chunk_words is None:
        chunk_words = 200 if args.mode == "sequential" else 500
    config = ExtractionConfig(
        mode=args.mode,
        chunk_words=chunk_words,
        context_words=args.context_words,
        previous_units=args.previous_units,
        canonicalize=not args.no_canonicalize,
        canonicalization_max_tokens=args.canonicalization_max_tokens,
    )
    if args.backend == "vllm":
        backend = VLLMBackend(
            model=args.model,
            tokenizer=args.tokenizer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            thinking=args.thinking,
        )
    else:
        backend = OpenAICompatibleBackend(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            thinking=args.thinking,
        )
    pipeline = KnowledgeUnitPipeline(backend, config)
    documents = read_documents(args.input, args.text_column)
    documents = documents[args.offset :]
    if args.limit is not None:
        documents = documents[: args.limit]
    if not documents:
        raise ValueError("input selection contains no documents")
    results = []
    for index, document in enumerate(documents):
        print("extracting document {}/{}".format(index + 1, len(documents)), file=sys.stderr)
        result = pipeline.extract(
            document["text"], document["title"], document["abstract"], args.input
        )
        results.append(result.to_dict())
    write_json_atomic(args.output, results[0] if len(results) == 1 else results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
