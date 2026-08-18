import json

from project_alexandria.canonicalize import canonicalize_units
from project_alexandria.schema import Entity, KnowledgeUnit, SourceReference


class AliasBackend:
    model_name = "alias-test"

    def generate(self, system, prompt, max_tokens=None):
        del system, prompt, max_tokens
        return json.dumps(
            {
                "alias_groups": {
                    "Kogut-Susskind Hamiltonian": [
                        "Kogut and Susskind",
                        "Kogut-Susskind Hamiltonian",
                    ],
                    "embedding approach": ["embedding approach", "embedding method"],
                    "larger Hilbert space": [
                        "enlarged Hilbert space",
                        "larger Hilbert space (working space)",
                    ],
                    "weakly coupled Yang-Mills theories": [
                        "weakly coupled Yang-Mills theories",
                        "weakly coupled gauge theories",
                    ],
                }
            }
        )


def _unit(*names_and_types):
    return KnowledgeUnit(
        chunk_index=0,
        context_summary="",
        entities=[Entity(name=name, entity_type=entity_type) for name, entity_type in names_and_types],
        source=SourceReference(0, 0, 1, 1, "hash"),
    )


def test_semantic_guard_rejects_person_object_merge_but_keeps_method_aliases():
    units = [
        _unit(
            ("Kogut and Susskind", "person"),
            ("Kogut-Susskind Hamiltonian", "other"),
            ("embedding approach", "method"),
            ("embedding method", "method"),
            ("enlarged Hilbert space", "mathematical structure"),
            ("larger Hilbert space (working space)", "state space"),
            ("weakly coupled Yang-Mills theories", "theory"),
            ("weakly coupled gauge theories", "model"),
        )
    ]
    resolved, groups = canonicalize_units(units, AliasBackend())
    by_name = {entity.name: entity.entity_id for entity in resolved[0].entities}
    assert by_name["Kogut and Susskind"] != by_name["Kogut-Susskind Hamiltonian"]
    embedding_ids = {
        entity.entity_id
        for entity in resolved[0].entities
        if entity.name == "embedding approach"
    }
    assert len(embedding_ids) == 1
    assert sum(entity.name == "embedding approach" for entity in resolved[0].entities) == 2
    assert any(len(group.aliases) == 2 for group in groups)
    assert by_name["enlarged Hilbert space"] != by_name["larger Hilbert space (working space)"]
    assert (
        by_name["weakly coupled Yang-Mills theories"]
        != by_name["weakly coupled gauge theories"]
    )
    assert all(entity.entity_type == "other" for entity in resolved[0].entities[4:])
