"""Tests for the hashed embedder and cosine similarity."""

import math

from nanorag.embeddings import HashedNGramEmbedder, cosine, tokenize


def test_tokenize_is_lowercase_alnum():
    assert tokenize("The Quick-Brown_Fox, 42 times!") == ["the", "quick", "brown", "fox", "42", "times"]


def test_vectors_are_unit_norm():
    emb = HashedNGramEmbedder(dim=256)
    for text in ["hello world", "netrunning through the grid", "ICE breaker protocols"]:
        vec = emb.embed(text)
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-6


def test_embedding_is_deterministic():
    emb = HashedNGramEmbedder()
    assert emb.embed("night city data grid") == emb.embed("night city data grid")


def test_similar_texts_score_higher_than_unrelated():
    emb = HashedNGramEmbedder(dim=512)
    a = emb.embed("netrunners breach ICE with daemon viruses")
    b = emb.embed("netrunner breaches ICE using daemon virus")
    c = emb.embed("fresh sushi restaurants in the harbour district")
    assert cosine(a, b) > cosine(a, c)


def test_empty_text_is_zero_vector():
    vec = HashedNGramEmbedder().embed("   !!! ###   ")
    assert all(v == 0.0 for v in vec)


def test_cosine_bounds_and_errors():
    emb = HashedNGramEmbedder(dim=64)
    v = emb.embed("vector")
    assert -1.0 <= cosine(v, v) <= 1.0
    try:
        cosine(v, v[:-1])
        assert False, "expected ValueError"
    except ValueError:
        pass
