import pytest

from project_alexandria.parsing import parse_unit_response


def test_parses_fenced_json():
    summary, entities, warnings = parse_unit_response(
        '```json\n{"context_summary":"x","entities":[{"name":"Alice","relationships":[]}]}\n```'
    )
    assert summary == "x"
    assert entities[0].name == "Alice"
    assert not warnings


def test_converts_legacy_entity_dictionary():
    _, entities, warnings = parse_unit_response(
        '{"entities":{"Alice":{"attributes":{"role":"author"},"relations":{"wrote":"Paper"}}}}'
    )
    assert entities[0].relationships[0].predicate == "wrote"
    assert warnings


def test_normalizes_undeclared_entity_type():
    _, entities, warnings = parse_unit_response(
        '{"entities":[{"name":"Hamiltonian","type":"operator","relationships":[]}]}'
    )
    assert entities[0].entity_type == "other"
    assert entities[0].attributes["model_entity_type"] == "operator"
    assert warnings


def test_recovers_only_complete_entities_from_truncated_json_when_enabled():
    response = (
        '{"context_summary":"complete summary","entities":['
        '{"name":"Alice","type":"person","relationships":[]},'
        '{"name":"unfinished","attributes":{"long":"cut off'
    )
    with pytest.raises(ValueError):
        parse_unit_response(response)
    summary, entities, warnings = parse_unit_response(response, allow_truncated=True)
    assert summary == "complete summary"
    assert [entity.name for entity in entities] == ["Alice"]
    assert warnings == ["recovered complete entity objects from truncated JSON"]
