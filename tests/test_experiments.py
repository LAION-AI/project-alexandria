from project_alexandria.experiments.mcq import extract_choice
from project_alexandria.experiments.overlap import ngram_jaccard
from project_alexandria.experiments.similarity import cosine_similarity, scramble


def test_scrambling_is_deterministic_and_grouped():
    source = "a b c d e f"
    assert scramble(source, 2, seed=7) == scramble(source, 2, seed=7)
    scrambled = scramble(source, 2, seed=7)
    assert "a b" in scrambled and "c d" in scrambled and "e f" in scrambled


def test_metrics_and_choice_parser():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert ngram_jaccard("a b c d e", "a b c d e", 5) == 1.0
    assert extract_choice("The answer is C.") == "C"
