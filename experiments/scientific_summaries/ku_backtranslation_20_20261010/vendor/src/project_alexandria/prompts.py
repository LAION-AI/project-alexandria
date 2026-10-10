"""Prompts derived from the few-shot templates used in the Alexandria experiments."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Sequence

from .chunking import TextChunk
from .schema import KnowledgeUnit


SYSTEM_PROMPT = """You extract factual knowledge from source text with exceptional care.
Never add facts from memory. Do not copy expressive prose. Return only valid JSON."""


_EXAMPLES = r"""
EXAMPLE 1
INPUT: Javier Milei, an economist known for radical libertarian views, won the 2023
Argentine presidential election and unseated the long-dominant Peronist party.
OUTPUT:
{"context_summary":"Argentina's election produced a break with its established political order.",
 "entities":[
  {"name":"Javier Milei","type":"person","attributes":{"occupation":"economist","political_orientation":"radical libertarian"},
   "relationships":[{"predicate":"won","target":"2023 Argentine presidential election"},{"predicate":"unseated","target":"Peronist party"}]},
  {"name":"2023 Argentine presidential election","type":"event","attributes":{"year":2023},"relationships":[{"predicate":"winner","target":"Javier Milei"}]},
  {"name":"Peronist party","type":"organization","attributes":{"prior_status":"long-dominant"},"relationships":[{"predicate":"unseated_by","target":"Javier Milei"}]}
 ]}

EXAMPLE 2
INPUT: Most stars, including the Sun, lie on the main sequence and spend most of their
lifetimes fusing hydrogen into helium. Red giants have exhausted core hydrogen. White
dwarfs are dense remnants of low-mass stars that shed their outer layers.
OUTPUT:
{"context_summary":"The passage distinguishes stellar classes by lifecycle state.",
 "entities":[
  {"name":"main sequence stars","type":"stellar_class","attributes":{"lifetime_stage":"majority of stellar lifetime"},"relationships":[{"predicate":"fuse","target":"hydrogen into helium"},{"predicate":"includes","target":"Sun"}]},
  {"name":"red giants","type":"stellar_class","attributes":{"core_hydrogen":"exhausted"},"relationships":[]},
  {"name":"white dwarfs","type":"stellar_class","attributes":{"density":"dense"},"relationships":[{"predicate":"remnants_of","target":"low-mass stars"}]}
 ]}

EXAMPLE 3
INPUT: Photosynthesis converts light energy into chemical energy. Plants use carbon
dioxide and water to synthesize glucose and release oxygen.
OUTPUT:
{"context_summary":"Photosynthesis stores light energy through a chemical conversion.",
 "entities":[
  {"name":"photosynthesis","type":"biological_process","attributes":{},"relationships":[{"predicate":"converts","target":"light energy into chemical energy"},{"predicate":"uses","target":"carbon dioxide"},{"predicate":"uses","target":"water"},{"predicate":"produces","target":"glucose"},{"predicate":"releases","target":"oxygen"}]}
 ]}
"""


_SCHEMA = """Return exactly one JSON object with this shape:
{
  "context_summary": "at most three factual sentences in fresh wording",
  "entities": [
    {
      "name": "stable, descriptive display name",
      "type": "person|organization|method|measurement|concept|event|place|other",
      "attributes": {"descriptive_attribute": "value"},
      "relationships": [
        {"predicate": "concise_snake_case_relation", "target": "entity or literal value", "attributes": {}}
      ]
    }
  ]
}"""


def _header(title: str, abstract: str) -> str:
    pieces = []
    if title:
        pieces.append("DOCUMENT TITLE:\n" + title.strip())
    if abstract:
        pieces.append("DOCUMENT ABSTRACT:\n" + abstract.strip())
    return "\n\n".join(pieces) or "DOCUMENT METADATA: unavailable"


def sequential_prompt(
    chunk: TextChunk,
    previous_units: Sequence[KnowledgeUnit],
    title: str = "",
    abstract: str = "",
) -> str:
    previous = [
        {
            "context_summary": unit.context_summary,
            "entities": [entity.name for entity in unit.entities],
        }
        for unit in previous_units
    ]
    return """Create one Knowledge Unit for TARGET TEXT.

This is the faithful sequential method: PREVIOUS KNOWLEDGE UNITS are context for consistent
naming, but extract facts only from TARGET TEXT. Preserve all stated names, quantities,
definitions, methods, findings, qualifications, and causal/relational claims. Do not infer
unstated facts. Paraphrase rather than reproduce distinctive phrasing. Avoid duplicate facts.

{schema}

{examples}

{header}

PREVIOUS KNOWLEDGE UNITS (up to ten):
{previous}

TARGET TEXT:
{target}
""".format(
        schema=_SCHEMA,
        examples=_EXAMPLES,
        header=_header(title, abstract),
        previous=json.dumps(previous, ensure_ascii=False),
        target=chunk.text,
    )


def parallel_prompt(chunk: TextChunk, title: str = "", abstract: str = "") -> str:
    return """Create one independent Knowledge Unit for TARGET TEXT.

BEFORE and AFTER are read-only disambiguation context. They may clarify pronouns, abbreviations,
and local references, but every extracted fact must be asserted in TARGET TEXT. Use descriptive,
stable entity names. Do not depend on output from any other chunk; document-level entity IDs will
be reconciled in a later pass. Preserve all stated names, quantities, definitions, methods,
findings, qualifications, and causal/relational claims. Paraphrase distinctive wording.

Before returning, delete any claim supported only by BEFORE or AFTER. Never interpret masked
tokens such as @xmath4 as numbers or dimensions; preserve them as opaque symbols when essential.
Ignore figure-drawing commands, coordinates, and formatting residue. If TARGET TEXT is only an
incomplete caption or fragment, extract only facts explicitly complete within that fragment.

{schema}

{examples}

{header}

CONTEXT BEFORE:
{before}

TARGET TEXT:
{target}

CONTEXT AFTER:
{after}
""".format(
        schema=_SCHEMA,
        examples=_EXAMPLES,
        header=_header(title, abstract),
        before=chunk.before or "[none]",
        target=chunk.text,
        after=chunk.after or "[none]",
    )


def canonicalization_prompt(
    units: Sequence[KnowledgeUnit], title: str = "", abstract: str = ""
) -> str:
    names_by_chunk = [
        {
            "chunk_index": unit.chunk_index,
            "context_summary": unit.context_summary,
            "entities": [
                {
                    "name": entity.name,
                    "type": entity.entity_type,
                    "relation_predicates": [
                        relationship.predicate for relationship in entity.relationships
                    ],
                }
                for entity in unit.entities
            ],
        }
        for unit in units
    ]
    unique_names = sorted(
        {entity["name"] for item in names_by_chunk for entity in item["entities"]}
    )
    candidate_pairs = []
    for left_index, left in enumerate(unique_names):
        left_normalized = re.sub(r"\W+", " ", left.casefold()).strip()
        left_tokens = set(left_normalized.split())
        for right in unique_names[left_index + 1 :]:
            right_normalized = re.sub(r"\W+", " ", right.casefold()).strip()
            right_tokens = set(right_normalized.split())
            union = left_tokens | right_tokens
            token_overlap = len(left_tokens & right_tokens) / len(union) if union else 0.0
            string_score = SequenceMatcher(None, left_normalized, right_normalized).ratio()
            if string_score >= 0.72 or (min(len(left_tokens), len(right_tokens)) >= 2 and token_overlap >= 0.5):
                candidate_pairs.append([left, right])
    return """Resolve entity names across one document.

Group only names that clearly refer to the same entity in this document. Examples include
"Alice M.", "Alice Miller", and "Miller, Alice" when context supports identity. Do not merge
related but distinct concepts, methods, people, organizations, measurements, or variants.

Critical non-merge rules: an author is not an eponymous method or object; a method/approach is not
the object it constructs; a framework is not the theory it studies; a group choice is not an
approach; a basis is not one of its states/elements; and a broad class is not a narrower subclass.
Shared words and close lexical similarity are not enough. When identity is uncertain, omit the
group so the deterministic fallback keeps the entities separate.

Return a compact synonym dictionary. Include only nontrivial groups containing two or more
distinct spellings; omit singleton names. Each key is the most complete readable canonical name,
and each value contains every spelling from the input that denotes that same entity. Preserve
equations and scientific symbols. Return only JSON:
{{"alias_groups":{{"Alice Miller":["Alice M.","Alice Miller","Miller, Alice"]}}}}

Lexical candidate pairs are supplied only to focus review; merge a pair only when the chunk
summaries support identity. Also find aliases missed by the lexical candidates.

{header}

ENTITY NAMES, TYPES, AND RELATION ROLES BY CHUNK:
{names}

LEXICAL CANDIDATE PAIRS:
{candidates}
""".format(
        header=_header(title, abstract),
        names=json.dumps(names_by_chunk, ensure_ascii=False),
        candidates=json.dumps(candidate_pairs, ensure_ascii=False),
    )


def quality_prompt(source_text: str, unit: Dict[str, Any]) -> str:
    return """Audit a Knowledge Unit against its source. Score each criterion from 0 to 5:
factual_fidelity, coverage, relation_quality, naming_consistency, style_independence, and
schema_validity. List unsupported claims and important omissions. Return only JSON with
"scores", "unsupported_claims", "omissions", and "summary".

SOURCE:
{source}

KNOWLEDGE UNIT:
{unit}
""".format(source=source_text, unit=json.dumps(unit, ensure_ascii=False))
