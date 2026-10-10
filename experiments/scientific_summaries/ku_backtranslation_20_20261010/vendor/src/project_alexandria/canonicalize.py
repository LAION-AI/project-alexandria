"""Document-level entity resolution and deterministic identifier assignment."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from .backends import GenerationBackend
from .parsing import extract_json, normalize_entity_types
from .prompts import SYSTEM_PROMPT, canonicalization_prompt
from .schema import CanonicalEntity, KnowledgeUnit


def canonical_id(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
    if slug:
        return slug
    return "entity_" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]


def _fallback_groups(units: Sequence[KnowledgeUnit]) -> List[CanonicalEntity]:
    groups: Dict[str, List[str]] = {}
    for unit in units:
        for entity in unit.entities:
            key = re.sub(r"\W+", "", entity.name.casefold())
            groups.setdefault(key, [])
            if entity.name not in groups[key]:
                groups[key].append(entity.name)
    return [
        CanonicalEntity(canonical_id(aliases[0]), aliases[0], aliases)
        for aliases in groups.values()
    ]


def _semantic_class(name: str, observed_types: Sequence[str]) -> str:
    """Coarse guardrail against high-impact cross-category alias merges."""
    normalized = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
    tokens = set(normalized.split())
    suffix = normalized.split()[-1] if normalized else ""
    if "hilbert space" in normalized and "working space" in normalized:
        return "whole_lattice_working_space"
    if {"yang", "mills"}.issubset(tokens) and suffix in {"theory", "theories"}:
        return "yang_mills_theory"
    if suffix in {"approach", "method"}:
        return "procedure"
    if suffix in {"construction", "procedure", "process"}:
        return "process"
    if suffix == "framework":
        return "framework"
    if tokens & {"hamiltonian", "operator", "matrix"}:
        return "formal_object"
    if tokens & {"element", "elements", "state", "states"} and "basis" in tokens:
        return "basis_member"
    if suffix in {"basis", "bases"}:
        return "basis"
    if suffix in {"group", "groups"}:
        return "group"
    if suffix in {"theory", "theories"}:
        return "theory"
    if "person" in observed_types:
        return "person"
    return "unspecified"


def _guard_alias_groups(
    groups: Sequence[CanonicalEntity], units: Sequence[KnowledgeUnit]
) -> List[CanonicalEntity]:
    observed: Dict[str, List[str]] = {}
    for unit in units:
        for entity in unit.entities:
            observed.setdefault(entity.name.casefold(), []).append(entity.entity_type)
    guarded = []
    for group in groups:
        partitions: Dict[str, List[str]] = {}
        for alias in group.aliases:
            semantic_class = _semantic_class(alias, observed.get(alias.casefold(), []))
            partitions.setdefault(semantic_class, []).append(alias)
        for aliases in partitions.values():
            unique_aliases = list(dict.fromkeys(aliases))
            if len(unique_aliases) < 2:
                continue
            if group.canonical_name in unique_aliases:
                canonical_name = group.canonical_name
            else:
                canonical_name = max(unique_aliases, key=len)
            guarded.append(
                CanonicalEntity(canonical_id(canonical_name), canonical_name, unique_aliases)
            )
    return guarded


def canonicalize_units(
    units: Sequence[KnowledgeUnit],
    backend: GenerationBackend,
    title: str = "",
    abstract: str = "",
    max_tokens: int = 4096,
) -> Tuple[List[KnowledgeUnit], List[CanonicalEntity]]:
    """Ask one document-level agent to unify aliases, then apply the mapping in place."""
    for unit in units:
        warnings = normalize_entity_types(unit.entities)
        unit.extraction_warnings.extend(
            warning for warning in warnings if warning not in unit.extraction_warnings
        )
    names = [entity.name for unit in units for entity in unit.entities]
    if not names:
        return list(units), []
    response = backend.generate(
        SYSTEM_PROMPT,
        canonicalization_prompt(units, title, abstract),
        max_tokens=max_tokens,
    )
    return apply_canonicalization_response(units, response)


def apply_canonicalization_response(
    units: Sequence[KnowledgeUnit], response: str
) -> Tuple[List[KnowledgeUnit], List[CanonicalEntity]]:
    """Validate one resolver response and deterministically rewrite its document units."""
    try:
        value = extract_json(response)
        groups = []
        for canonical_name, aliases_value in (value.get("alias_groups") or {}).items():
            aliases = [str(alias).strip() for alias in aliases_value if str(alias).strip()]
            if aliases:
                groups.append(
                    CanonicalEntity(canonical_id(str(canonical_name)), str(canonical_name), aliases)
                )
        for item in value.get("canonical_entities") or []:
            aliases = [str(alias).strip() for alias in item.get("aliases") or [] if str(alias).strip()]
            if not aliases:
                continue
            canonical_name = str(item.get("canonical_name") or aliases[0]).strip()
            proposed_id = str(item.get("canonical_id") or canonical_id(canonical_name)).strip()
            groups.append(CanonicalEntity(canonical_id(proposed_id), canonical_name, aliases))
    except (ValueError, TypeError):
        groups = []

    groups = _guard_alias_groups(groups, units)

    mapped = {alias.casefold() for group in groups for alias in group.aliases}
    for fallback in _fallback_groups(units):
        missing = [alias for alias in fallback.aliases if alias.casefold() not in mapped]
        if missing:
            groups.append(CanonicalEntity(canonical_id(missing[0]), missing[0], missing))
            mapped.update(alias.casefold() for alias in missing)

    alias_lookup = {}
    for group in groups:
        for alias in group.aliases + [group.canonical_name]:
            alias_lookup[alias.casefold()] = group
    type_counts: Dict[str, Counter] = {}
    for unit in units:
        for entity in unit.entities:
            group = alias_lookup.get(entity.name.casefold())
            identifier = group.canonical_id if group else canonical_id(entity.name)
            type_counts.setdefault(identifier, Counter())[entity.entity_type] += 1
    stable_types = {
        identifier: counts.most_common(1)[0][0] for identifier, counts in type_counts.items()
    }
    for unit in units:
        for entity in unit.entities:
            group = alias_lookup.get(entity.name.casefold())
            if group:
                original = entity.name
                entity.name = group.canonical_name
                entity.entity_id = group.canonical_id
                entity.aliases = sorted(set(group.aliases + [original]))
            else:
                entity.entity_id = canonical_id(entity.name)
            entity.entity_type = stable_types[entity.entity_id]
            for relationship in entity.relationships:
                target_group = alias_lookup.get(relationship.target.casefold())
                if target_group:
                    relationship.target = target_group.canonical_name
                    relationship.target_id = target_group.canonical_id
    return list(units), groups


def canonicalize_many(
    documents: Sequence[Sequence[KnowledgeUnit]],
    backend: GenerationBackend,
    titles: Sequence[str],
    abstracts: Sequence[str],
    max_tokens: int = 4096,
) -> List[Tuple[List[KnowledgeUnit], List[CanonicalEntity]]]:
    """Continuously batch independent document-level resolver calls."""
    if not (len(documents) == len(titles) == len(abstracts)):
        raise ValueError("documents, titles, and abstracts must have matching lengths")
    prompts = []
    prompt_indices = []
    results: List[Optional[Tuple[List[KnowledgeUnit], List[CanonicalEntity]]]] = [
        None for _ in documents
    ]
    for index, (units, title, abstract) in enumerate(zip(documents, titles, abstracts)):
        for unit in units:
            warnings = normalize_entity_types(unit.entities)
            unit.extraction_warnings.extend(
                warning for warning in warnings if warning not in unit.extraction_warnings
            )
        if any(unit.entities for unit in units):
            prompts.append(canonicalization_prompt(units, title, abstract))
            prompt_indices.append(index)
        else:
            results[index] = (list(units), [])
    responses = backend.generate_batch(SYSTEM_PROMPT, prompts, max_tokens=max_tokens)
    if len(responses) != len(prompts):
        raise RuntimeError("backend returned a different number of resolver responses")
    for index, response in zip(prompt_indices, responses):
        results[index] = apply_canonicalization_response(documents[index], response)
    return [
        result if result is not None else (list(documents[i]), [])
        for i, result in enumerate(results)
    ]
