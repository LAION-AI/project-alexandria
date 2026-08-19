import json

from project_alexandria.pipeline import ExtractionConfig, KnowledgeUnitPipeline
from project_alexandria.schema import DocumentResult


class FakeBackend:
    model_name = "fake-model"

    def __init__(self):
        self.prompts = []
        self.batch_prompts = []

    def generate(self, system, prompt, max_tokens=None):
        del max_tokens
        self.prompts.append(prompt)
        if "LEXICAL CANDIDATE PAIRS" in prompt:
            return json.dumps(
                {
                    "canonical_entities": [
                        {
                            "canonical_id": "alice_miller",
                            "canonical_name": "Alice Miller",
                            "aliases": ["Alice M.", "Alice Miller"],
                        }
                    ]
                }
            )
        name = "Alice M." if len(self.prompts) == 1 else "Alice Miller"
        return json.dumps(
            {
                "context_summary": "Factual context.",
                "entities": [
                    {
                        "name": name,
                        "type": "person",
                        "attributes": {},
                        "relationships": [],
                    }
                ],
            }
        )

    def generate_batch(self, system, prompts, max_tokens=None):
        del max_tokens
        self.batch_prompts.extend(prompts)
        return [
            json.dumps(
                {
                    "context_summary": "Factual context.",
                    "entities": [
                        {"name": "Alice Miller", "attributes": {}, "relationships": []}
                    ],
                }
            )
            for _ in prompts
        ]


def test_sequential_passes_previous_units_and_canonicalizes():
    backend = FakeBackend()
    pipeline = KnowledgeUnitPipeline(
        backend, ExtractionConfig(mode="sequential", chunk_words=4, canonicalize=True)
    )
    result = pipeline.extract("Alice wrote a paper. Alice presented the paper.")
    assert len(result.knowledge_units) == 2
    assert "Alice M." in backend.prompts[1]
    assert result.knowledge_units[0].entities[0].entity_id == "alice_miller"


def test_parallel_uses_batch_and_neighbor_context():
    backend = FakeBackend()
    pipeline = KnowledgeUnitPipeline(
        backend,
        ExtractionConfig(mode="parallel", chunk_words=3, context_words=2, canonicalize=False),
    )
    result = pipeline.extract("A B C. D E F. G H I.")
    assert len(result.knowledge_units) == 3
    assert len(backend.batch_prompts) == 3
    assert "CONTEXT BEFORE:\nB C." in backend.batch_prompts[1]
    assert not backend.prompts


def test_document_result_round_trip():
    backend = FakeBackend()
    pipeline = KnowledgeUnitPipeline(
        backend, ExtractionConfig(mode="sequential", chunk_words=10, canonicalize=False)
    )
    original = pipeline.extract("Alice wrote a paper.")
    restored = DocumentResult.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()


def test_parallel_extract_many_batches_chunks_and_resolvers():
    backend = FakeBackend()
    pipeline = KnowledgeUnitPipeline(
        backend,
        ExtractionConfig(mode="parallel", chunk_words=3, context_words=1, canonicalize=True),
    )
    results = pipeline.extract_many(
        [
            {"text": "A B C. D E F.", "title": "one", "abstract": ""},
            {"text": "G H I. J K L.", "title": "two", "abstract": ""},
        ]
    )
    assert [len(result.knowledge_units) for result in results] == [2, 2]
    # Four extraction prompts and two independent document resolver prompts.
    assert len(backend.batch_prompts) == 6
    assert not backend.prompts
