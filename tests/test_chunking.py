from project_alexandria.chunking import add_neighbor_context, split_text


def test_sentence_aware_chunks_and_offsets():
    text = "One two three. Four five six. Seven eight nine."
    chunks = split_text(text, target_words=6)
    assert [chunk.word_count for chunk in chunks] == [6, 3]
    assert [(chunk.start_word, chunk.end_word) for chunk in chunks] == [(0, 6), (6, 9)]


def test_neighbor_context_excludes_target():
    chunks = add_neighbor_context(split_text("A B C. D E F. G H I.", 3), context_words=2)
    assert chunks[1].before == "B C."
    assert chunks[1].text == "D E F."
    assert chunks[1].after == "G H"


def test_sentence_boundaries_preserve_scientific_and_quoted_closers():
    text = 'Moment was measured (µB/f.u.) Values follow [9].) "The result held." Next sentence.'
    chunks = split_text(text, target_words=5)
    assert " ".join(chunk.text for chunk in chunks).split() == text.split()
    assert sum(chunk.word_count for chunk in chunks) == len(text.split())
    assert all(chunk.word_count <= 5 for chunk in chunks)
