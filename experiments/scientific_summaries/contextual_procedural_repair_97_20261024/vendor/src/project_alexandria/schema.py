"""Small dependency-free data model for Alexandria artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Relationship:
    predicate: str
    target: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    target_id: Optional[str] = None

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Relationship":
        return cls(
            predicate=str(value.get("predicate", "related_to")).strip(),
            target=str(value.get("target", "")).strip(),
            attributes=dict(value.get("attributes") or {}),
            target_id=value.get("target_id"),
        )


@dataclass
class Entity:
    name: str
    entity_id: str = ""
    entity_type: str = "concept"
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Entity":
        relationships = value.get("relationships") or value.get("relations") or []
        if isinstance(relationships, dict):
            relationships = [
                {"predicate": predicate, "target": target}
                for predicate, target in relationships.items()
            ]
        return cls(
            name=str(value.get("name", "")).strip(),
            entity_id=str(value.get("entity_id", "")).strip(),
            entity_type=str(value.get("type", value.get("entity_type", "concept"))).strip(),
            aliases=[str(alias) for alias in value.get("aliases") or []],
            attributes=dict(value.get("attributes") or {}),
            relationships=[Relationship.from_dict(item) for item in relationships],
        )


@dataclass
class SourceReference:
    chunk_index: int
    start_word: int
    end_word: int
    word_count: int
    sha256: str
    sentence_minhash: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SourceReference":
        return cls(
            chunk_index=int(value["chunk_index"]),
            start_word=int(value["start_word"]),
            end_word=int(value["end_word"]),
            word_count=int(value["word_count"]),
            sha256=str(value["sha256"]),
            sentence_minhash=[int(item) for item in value.get("sentence_minhash") or []],
        )


@dataclass
class KnowledgeUnit:
    chunk_index: int
    context_summary: str
    entities: List[Entity]
    source: SourceReference
    extraction_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "KnowledgeUnit":
        return cls(
            chunk_index=int(value["chunk_index"]),
            context_summary=str(value.get("context_summary") or ""),
            entities=[Entity.from_dict(item) for item in value.get("entities") or []],
            source=SourceReference.from_dict(value["source"]),
            extraction_warnings=[str(item) for item in value.get("extraction_warnings") or []],
        )


@dataclass
class CanonicalEntity:
    canonical_id: str
    canonical_name: str
    aliases: List[str]

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "CanonicalEntity":
        return cls(
            canonical_id=str(value["canonical_id"]),
            canonical_name=str(value["canonical_name"]),
            aliases=[str(item) for item in value.get("aliases") or []],
        )


@dataclass
class DocumentResult:
    schema_version: str
    title: str
    abstract_sha256: str
    mode: str
    model: str
    config: Dict[str, Any]
    knowledge_units: List[KnowledgeUnit]
    canonical_entities: List[CanonicalEntity] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "DocumentResult":
        return cls(
            schema_version=str(value.get("schema_version") or "1.0"),
            title=str(value.get("title") or ""),
            abstract_sha256=str(value.get("abstract_sha256") or ""),
            mode=str(value.get("mode") or "unknown"),
            model=str(value.get("model") or "unknown"),
            config=dict(value.get("config") or {}),
            knowledge_units=[KnowledgeUnit.from_dict(item) for item in value.get("knowledge_units") or []],
            canonical_entities=[CanonicalEntity.from_dict(item) for item in value.get("canonical_entities") or []],
        )
