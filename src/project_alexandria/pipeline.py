"""Sequential reproduction baseline and independent batched extraction pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

from .backends import GenerationBackend
from .canonicalize import canonical_id, canonicalize_many, canonicalize_units
from .chunking import TextChunk, add_neighbor_context, split_text
from .fingerprints import sentence_minhash, sha256_text
from .parsing import parse_unit_response
from .prompts import SYSTEM_PROMPT, parallel_prompt, sequential_prompt
from .schema import DocumentResult, KnowledgeUnit, SourceReference


@dataclass
class ExtractionConfig:
    mode: str = "sequential"
    chunk_words: int = 200
    context_words: int = 1000
    previous_units: int = 10
    canonicalize: bool = True
    minhash_permutations: int = 16
    canonicalization_max_tokens: int = 4096
    parse_retries: int = 1

    def __post_init__(self) -> None:
        if self.mode not in ("sequential", "parallel"):
            raise ValueError("mode must be 'sequential' or 'parallel'")
        if (
            self.chunk_words < 1
            or self.previous_units < 0
            or self.context_words < 0
            or self.canonicalization_max_tokens < 1
            or self.parse_retries < 0
        ):
            raise ValueError("chunk and context sizes must be non-negative")


class KnowledgeUnitPipeline:
    def __init__(self, backend: GenerationBackend, config: Optional[ExtractionConfig] = None):
        self.backend = backend
        self.config = config or ExtractionConfig()

    def _unit(self, chunk: TextChunk, response: str, prompt: str) -> KnowledgeUnit:
        last_error = None
        for attempt in range(self.config.parse_retries + 1):
            try:
                summary, entities, warnings = parse_unit_response(response)
                break
            except (ValueError, TypeError) as error:
                last_error = error
                if attempt >= self.config.parse_retries:
                    raise ValueError(
                        "chunk {} returned invalid structured output: {}".format(chunk.index, error)
                    )
                # A structurally valid answer can be cut off at the normal generation limit.
                # Repair only the failed chunk, with enough room to close a verbose JSON object,
                # instead of forcing the caller to regenerate every chunk in the document.
                normal_max_tokens = getattr(self.backend, "max_tokens", None)
                repair_max_tokens = (
                    min(normal_max_tokens * 2, 8192)
                    if isinstance(normal_max_tokens, int) and normal_max_tokens > 0
                    else None
                )
                if attempt == self.config.parse_retries - 1:
                    repair_instruction = (
                        "\n\nYour previous response was invalid or too long. Return one COMPLETE "
                        "JSON object only, using at most 12 entities. Keep the context summary under "
                        "120 words. Give each entity at most 4 short attributes and 4 relationships; "
                        "omit quotations, evidence passages, commentary, and duplicate facts. Use "
                        "compact one-line strings and close every array and object."
                    )
                else:
                    repair_instruction = (
                        "\n\nYour previous response was invalid. Return compact, complete JSON only; "
                        "omit no closing braces."
                    )
                response = self.backend.generate(
                    SYSTEM_PROMPT,
                    prompt + repair_instruction,
                    max_tokens=repair_max_tokens,
                )
        else:  # pragma: no cover - loop always breaks or raises
            raise RuntimeError(str(last_error))
        for entity in entities:
            if not entity.entity_id:
                entity.entity_id = canonical_id(entity.name)
        source = SourceReference(
            chunk_index=chunk.index,
            start_word=chunk.start_word,
            end_word=chunk.end_word,
            word_count=chunk.word_count,
            sha256=sha256_text(chunk.text),
            sentence_minhash=sentence_minhash(chunk.text, self.config.minhash_permutations),
        )
        return KnowledgeUnit(chunk.index, summary, entities, source, warnings)

    def extract(
        self, text: str, title: str = "", abstract: str = "", source_name: str = ""
    ) -> DocumentResult:
        del source_name  # reserved for a future provenance URI; source text is never emitted
        chunks = split_text(text, self.config.chunk_words)
        if not chunks:
            raise ValueError("input text is empty")
        units: List[KnowledgeUnit] = []
        prompt_abstract = abstract
        if not prompt_abstract and len(chunks) > 1:
            prompt_abstract = " ".join(text.split()[:350])
        if self.config.mode == "sequential":
            for chunk in chunks:
                context = units[-self.config.previous_units :] if self.config.previous_units else []
                response = self.backend.generate(
                    SYSTEM_PROMPT, sequential_prompt(chunk, context, title, prompt_abstract)
                )
                units.append(
                    self._unit(
                        chunk,
                        response,
                        sequential_prompt(chunk, context, title, prompt_abstract),
                    )
                )
        else:
            chunks = add_neighbor_context(chunks, self.config.context_words)
            prompts = [parallel_prompt(chunk, title, prompt_abstract) for chunk in chunks]
            responses = self.backend.generate_batch(SYSTEM_PROMPT, prompts)
            if len(responses) != len(chunks):
                raise RuntimeError("backend returned a different number of responses than prompts")
            units = [
                self._unit(chunk, response, prompt)
                for chunk, response, prompt in zip(chunks, responses, prompts)
            ]

        canonical_entities = []
        if self.config.canonicalize:
            units, canonical_entities = canonicalize_units(
                units,
                self.backend,
                title,
                prompt_abstract,
                max_tokens=self.config.canonicalization_max_tokens,
            )
        return DocumentResult(
            schema_version="1.0",
            title=title,
            abstract_sha256=sha256_text(prompt_abstract) if prompt_abstract else "",
            mode=self.config.mode,
            model=self.backend.model_name,
            config=asdict(self.config),
            knowledge_units=units,
            canonical_entities=canonical_entities,
        )

    def extract_many(self, documents: Sequence[Dict[str, str]]) -> List[DocumentResult]:
        """Extract several documents while sharing one continuous vLLM prompt batch.

        Parallel mode flattens every target chunk across the document batch. This keeps the
        server busy at document boundaries while preserving independent per-document context and
        canonicalization. Sequential mode intentionally retains its dependency chain.
        """
        if self.config.mode != "parallel":
            return [
                self.extract(
                    document["text"],
                    document.get("title", ""),
                    document.get("abstract", ""),
                )
                for document in documents
            ]
        prepared = []
        flat_prompts = []
        for document in documents:
            text = document["text"]
            chunks = split_text(text, self.config.chunk_words)
            if not chunks:
                raise ValueError("input text is empty")
            prompt_abstract = document.get("abstract", "")
            if not prompt_abstract and len(chunks) > 1:
                prompt_abstract = " ".join(text.split()[:350])
            chunks = add_neighbor_context(chunks, self.config.context_words)
            prompts = [
                parallel_prompt(chunk, document.get("title", ""), prompt_abstract)
                for chunk in chunks
            ]
            prepared.append((document, prompt_abstract, chunks, prompts))
            flat_prompts.extend(prompts)

        flat_responses = self.backend.generate_batch(SYSTEM_PROMPT, flat_prompts)
        if len(flat_responses) != len(flat_prompts):
            raise RuntimeError("backend returned a different number of responses than prompts")

        extracted = []
        cursor = 0
        for document, prompt_abstract, chunks, prompts in prepared:
            responses = flat_responses[cursor : cursor + len(chunks)]
            cursor += len(chunks)
            units = [
                self._unit(chunk, response, prompt)
                for chunk, response, prompt in zip(chunks, responses, prompts)
            ]
            extracted.append((document, prompt_abstract, units))

        if self.config.canonicalize:
            resolved = canonicalize_many(
                [item[2] for item in extracted],
                self.backend,
                [item[0].get("title", "") for item in extracted],
                [item[1] for item in extracted],
                max_tokens=self.config.canonicalization_max_tokens,
            )
        else:
            resolved = [(item[2], []) for item in extracted]

        results = []
        for (document, prompt_abstract, _), (units, canonical_entities) in zip(
            extracted, resolved
        ):
            results.append(
                DocumentResult(
                    schema_version="1.0",
                    title=document.get("title", ""),
                    abstract_sha256=(
                        sha256_text(prompt_abstract) if prompt_abstract else ""
                    ),
                    mode=self.config.mode,
                    model=self.backend.model_name,
                    config=asdict(self.config),
                    knowledge_units=units,
                    canonical_entities=canonical_entities,
                )
            )
        return results
