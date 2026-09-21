import pytest

from src.embeddings import Embedding


def test_embedding_shape_and_parameters():
    embedding = Embedding(vocab_size=5, embedding_dim=3, seed=1)
    assert len(embedding.forward(2)) == 3
    assert len(embedding.parameters()) == 15


def test_embedding_rejects_invalid_token():
    embedding = Embedding(vocab_size=3, embedding_dim=2)
    with pytest.raises(IndexError):
        embedding.forward(3)


def test_embedding_rejects_non_integer_token():
    embedding = Embedding(vocab_size=3, embedding_dim=2)
    with pytest.raises(TypeError):
        embedding.forward(1.5)
    with pytest.raises(TypeError):
        embedding.forward(True)
