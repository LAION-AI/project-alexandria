import json

from project_alexandria.experiments.mcq import (
    extract_historical_choice,
    historical_answer_prompt,
)
from project_alexandria.experiments.reproduce import knowledge_unit_context, summarize
from project_alexandria.schema import (
    DocumentResult,
    Entity,
    KnowledgeUnit,
    SourceReference,
)


def test_historical_prompt_and_summary():
    prompt = historical_answer_prompt("A) α yes B) no")
    assert ";A;" not in prompt
    assert "α" not in prompt
    assert extract_historical_choice("; C ;") == "C"
    assert extract_historical_choice("I think answer A is likely") is None
    rows = [
        {"gold": "A", "predictions": {"no_context": "A", "original": "A", "knowledge_units": "B"}},
        {"gold": "B", "predictions": {"no_context": None, "original": "B", "knowledge_units": "B"}},
    ]
    result = summarize(rows)
    assert result["no_context"]["accuracy"] == 0.5
    assert result["no_context"]["valid_accuracy"] == 1.0
    assert result["no_context"]["invalid"] == 1
    assert result["knowledge_units"]["accuracy"] == 0.5


def test_ku_context_does_not_include_source_metadata():
    document = DocumentResult(
        "1.0",
        "",
        "abstract-hash",
        "parallel",
        "model",
        {},
        [
            KnowledgeUnit(
                0,
                "summary",
                [Entity("Alice")],
                SourceReference(0, 0, 3, 3, "source-hash"),
            )
        ],
    )
    value = json.loads(knowledge_unit_context(document))
    assert value[0]["entities"][0]["name"] == "Alice"
    assert "source-hash" not in knowledge_unit_context(document)
