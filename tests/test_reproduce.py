import json

from project_alexandria.backends import OpenAICompatibleBackend
from project_alexandria.experiments.datasets import (
    EvaluationDocument,
    _document_id,
    dataset_manifest,
)
from project_alexandria.experiments.mcq import (
    extract_historical_choice,
    historical_answer_prompt,
)
from project_alexandria.experiments.reproduce import (
    _extract_resilient,
    extract_ku_cache,
    judge_ku_cache,
    knowledge_unit_context,
    summarize,
)
from project_alexandria.pipeline import ExtractionConfig, KnowledgeUnitPipeline
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


def test_duplicate_text_rows_get_distinct_document_ids():
    digest = "a" * 64
    assert _document_id(digest, 7) != _document_id(digest, 8)


def test_selection_manifest_is_order_sensitive(tmp_path):
    dataset = tmp_path / "dataset.parquet"
    dataset.write_bytes(b"fixture")
    first = EvaluationDocument("hash:1", 1, "text", ["q"], ["A"])
    second = EvaluationDocument("hash:2", 2, "text", ["q"], ["A"])
    forward = dataset_manifest(str(dataset), [first, second])
    reverse = dataset_manifest(str(dataset), [second, first])
    assert forward["ordered_selection_sha256"] != reverse["ordered_selection_sha256"]


def test_explicit_empty_api_key_does_not_inherit_extractor_key(monkeypatch):
    monkeypatch.setenv("ALEXANDRIA_API_KEY", "extractor-only-secret")
    backend = OpenAICompatibleBackend(model="judge", api_key="")
    assert backend.api_key == ""


class _BatchBackend:
    def __init__(self, model_name, response):
        self.model_name = model_name
        self.response = response
        self.temperature = 0.0
        self.frequency_penalty = 1.05
        self.presence_penalty = 1.05

    def generate_batch(self, system, prompts, max_tokens=None):
        del system, max_tokens
        return [self.response for _ in prompts]


def test_two_stage_cache_and_judge(tmp_path):
    dataset = tmp_path / "dataset.parquet"
    dataset.write_bytes(b"fixture")
    documents = [EvaluationDocument("hash:0", 0, "Alice measured 3.", ["question"], ["A"])]
    extractor = _BatchBackend(
        "extractor",
        '{"context_summary":"measurement","entities":[]}',
    )
    pipeline = KnowledgeUnitPipeline(
        extractor,
        ExtractionConfig(mode="parallel", chunk_words=500, canonicalize=False),
    )
    cache_path = tmp_path / "kus.json"
    cache = extract_ku_cache(str(dataset), documents, pipeline, str(cache_path))
    assert len(cache["documents"]) == 1
    assert "Alice measured 3" not in cache_path.read_text()

    judge = _BatchBackend("judge", ";A;")
    output = judge_ku_cache(
        str(dataset), documents, str(cache_path), judge, str(tmp_path / "scores.json")
    )
    assert output["summary"]["knowledge_units"]["accuracy"] == 1.0


def test_resilient_extraction_isolates_a_failed_batch():
    class SplitPipeline:
        def extract_many(self, payloads):
            if len(payloads) > 1:
                raise ValueError("malformed chunk")
            return [DocumentResult("1.0", "", "", "parallel", "model", {}, [])]

    documents = [
        EvaluationDocument("hash:0", 0, "one", ["q"], ["A"]),
        EvaluationDocument("hash:1", 1, "two", ["q"], ["A"]),
    ]
    assert len(_extract_resilient(SplitPipeline(), documents)) == 2
