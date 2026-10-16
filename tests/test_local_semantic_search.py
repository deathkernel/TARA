import math

import pytest

from src.local_semantic_search import LocalSemanticIndex, OllamaEmbedder


def test_ollama_embedder_batches_inputs_and_uses_local_embed_endpoint():
    calls = []

    def transport(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {"embeddings": [[1.0, 0.0] for _ in payload["input"]]}

    embedder = OllamaEmbedder(
        model="embeddinggemma",
        base_url="http://127.0.0.1:11434/",
        timeout=4,
        batch_size=2,
        transport=transport,
    )
    vectors = embedder.embed(["first", "second", "third"])

    assert vectors == [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    assert len(calls) == 2
    assert calls[0] == (
        "http://127.0.0.1:11434/api/embed",
        {"model": "embeddinggemma", "input": ["first", "second"], "truncate": False},
        4.0,
    )


@pytest.mark.parametrize("inputs", [[""], ["x" * 20]])
def test_embedder_rejects_empty_or_oversized_inputs(inputs):
    embedder = OllamaEmbedder(
        max_input_chars=10,
        transport=lambda *_: {"embeddings": [[1.0]]},
    )
    with pytest.raises(ValueError):
        embedder.embed(inputs)


def test_embedder_rejects_bad_response_count_dimensions_and_non_finite_values():
    wrong_count = OllamaEmbedder(transport=lambda *_: {"embeddings": []})
    with pytest.raises(RuntimeError, match="wrong number"):
        wrong_count.embed(["one"])

    bad_dimensions = OllamaEmbedder(
        batch_size=2,
        transport=lambda *_: {"embeddings": [[1.0], [1.0, 2.0]]},
    )
    with pytest.raises(RuntimeError, match="dimensions"):
        bad_dimensions.embed(["one", "two"])

    non_finite = OllamaEmbedder(transport=lambda *_: {"embeddings": [[math.inf]]})
    with pytest.raises(RuntimeError, match="finite"):
        non_finite.embed(["one"])


def test_embedder_rejects_invalid_constructor_limits():
    with pytest.raises(ValueError):
        OllamaEmbedder(batch_size=0)
    with pytest.raises(ValueError):
        OllamaEmbedder(timeout=0)


class FakeEmbedder:
    def __init__(self):
        self.fail = False
        self.calls = []

    def embed(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        if self.fail:
            raise RuntimeError("embedding unavailable")
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([
                float("cat" in lowered or "kitten" in lowered),
                float("space" in lowered or "planet" in lowered),
                1.0,
            ])
        return vectors


def test_index_search_ranks_semantically_similar_chunks_and_keeps_source():
    embedder = FakeEmbedder()
    index = LocalSemanticIndex(embedder, chunk_chars=120, overlap_chars=20)
    index.add_document("pets", "Cats are small pets. A kitten needs care.", source="pets.txt")
    index.add_document("astronomy", "Planets and space exploration are fascinating.", source="space.md")

    results = index.search("kitten and cat", top_k=2)

    assert results[0].document_id == "pets"
    assert results[0].source == "pets.txt"
    assert results[0].score > results[1].score
    assert index.document_count == 2
    assert index.chunk_count == 2


def test_index_is_in_memory_and_can_remove_or_clear_documents():
    index = LocalSemanticIndex(FakeEmbedder())
    index.add_document("one", "Cats are pets.")
    assert index.remove_document("one") is True
    assert index.remove_document("one") is False
    index.add_document("two", "Planets are in space.")
    index.clear()
    assert index.document_count == 0
    assert index.chunk_count == 0
    assert index.search("space") == []


def test_failed_document_replacement_keeps_previous_index_entry():
    embedder = FakeEmbedder()
    index = LocalSemanticIndex(embedder)
    index.add_document("doc", "Cats are wonderful.", source="old.txt")
    embedder.fail = True

    with pytest.raises(RuntimeError, match="unavailable"):
        index.add_document("doc", "Planets are wonderful.", source="new.txt")

    embedder.fail = False
    result = index.search("cats")[0]
    assert result.source == "old.txt"
    assert "Cats" in result.text
    assert index.document_count == 1
    assert index.chunk_count == 1


def test_index_enforces_document_chunk_and_query_limits():
    index = LocalSemanticIndex(FakeEmbedder(), max_documents=1, max_chunks=2, chunk_chars=100)
    index.add_document("one", "A short document.")
    with pytest.raises(ValueError, match="document limit"):
        index.add_document("two", "Another document.")
    with pytest.raises(ValueError, match="top_k"):
        index.search("cats", top_k=21)
    with pytest.raises(ValueError, match="query"):
        index.search("   ")


def test_index_rejects_mixed_dimensions_before_committing_new_document():
    class SwitchingEmbedder:
        def __init__(self):
            self.dimensions = 2

        def embed(self, texts):
            return [[1.0] * self.dimensions for _ in texts]

    embedder = SwitchingEmbedder()
    index = LocalSemanticIndex(embedder)
    index.add_document("first", "First document.")
    embedder.dimensions = 3

    with pytest.raises(RuntimeError, match="existing index"):
        index.add_document("second", "Second document.")

    assert index.document_count == 1
    assert index.chunk_count == 1
    assert index.search("query")[0].document_id == "first"


def test_replacing_only_document_can_change_embedding_dimensions():
    class SwitchingEmbedder:
        def __init__(self):
            self.dimensions = 2

        def embed(self, texts):
            return [[1.0] * self.dimensions for _ in texts]

    embedder = SwitchingEmbedder()
    index = LocalSemanticIndex(embedder)
    index.add_document("only", "Original document.")
    embedder.dimensions = 3

    assert index.add_document("only", "Replacement document.") == 1
    assert index.document_count == 1
    assert index.search("replacement")[0].document_id == "only"


def test_index_rejects_dimension_mismatch_during_search():
    class MismatchEmbedder:
        def __init__(self):
            self.count = 0

        def embed(self, texts):
            self.count += 1
            return [[1.0, 0.0]] if self.count == 1 else [[1.0, 0.0, 0.0]]

    index = LocalSemanticIndex(MismatchEmbedder())
    index.add_document("one", "Document content.")
    with pytest.raises(RuntimeError, match="dimensions"):
        index.search("query")


def test_batch_indexing_commits_all_documents_and_returns_chunk_counts():
    index = LocalSemanticIndex(FakeEmbedder())
    counts = index.add_documents([
        ("pets", "Cats are pets.", "pets.txt"),
        ("space", "Planets are in space."),
    ])
    assert counts == {"pets": 1, "space": 1}
    assert index.document_count == 2
    assert index.chunk_count == 2
    assert index.search("kitten")[0].document_id == "pets"


def test_batch_indexing_is_atomic_when_a_later_document_fails():
    embedder = FakeEmbedder()
    index = LocalSemanticIndex(embedder)
    index.add_document("existing", "Cats are pets.", source="old.txt")
    before_chunks = index.chunk_count
    original_embed = embedder.embed
    calls = {"count": 0}

    def fail_on_second_document(texts):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("second embedding failed")
        return original_embed(texts)

    embedder.embed = fail_on_second_document
    with pytest.raises(RuntimeError, match="second embedding"):
        index.add_documents([
            ("new", "Planets are in space."),
            ("existing", "Replacement document."),
        ])

    assert index.document_count == 1
    assert index.chunk_count == before_chunks
    embedder.embed = original_embed
    assert index.search("cats")[0].source == "old.txt"


def test_batch_indexing_rejects_duplicate_ids_and_bad_shapes_without_mutation():
    index = LocalSemanticIndex(FakeEmbedder())
    with pytest.raises(ValueError, match="duplicate document_id"):
        index.add_documents([("same", "Cats."), ("same", "Planets.")])
    with pytest.raises(ValueError, match="batch item"):
        index.add_documents([("valid", "Cats."), ("invalid",)])
    assert index.document_count == 0


def test_batch_indexing_replaces_existing_id_only_after_success():
    embedder = FakeEmbedder()
    index = LocalSemanticIndex(embedder)
    index.add_document("guide", "Cats are pets.", source="old.md")
    embedder.fail = True
    with pytest.raises(RuntimeError):
        index.add_documents([("guide", "Planets are in space.", "new.md"), ("other", "Other text.")])
    embedder.fail = False
    assert index.document_count == 1
    assert index.search("cats")[0].source == "old.md"


def test_retrieval_benchmark_reports_recall_mrr_ndcg_and_stable_fingerprint():
    from src.semantic_search_benchmark import RetrievalCase, evaluate_retrieval

    index = LocalSemanticIndex(FakeEmbedder())
    index.add_documents([
        ("pets", "Cats are small pets. A kitten needs care."),
        ("space", "Planets and space exploration are fascinating."),
    ])
    cases = [
        RetrievalCase("kitten and cat", ("pets",)),
        RetrievalCase("planet and space", ("space",)),
    ]
    first = evaluate_retrieval(index, cases, k=2)
    second = evaluate_retrieval(index, cases, k=2)

    assert first.case_count == 2
    assert first.recall_at_k == 1.0
    assert first.mean_reciprocal_rank == 1.0
    assert first.mean_ndcg_at_k == 1.0
    assert len(first.fingerprint) == 64
    assert first.fingerprint == second.fingerprint


def test_retrieval_benchmark_rejects_invalid_cases_and_k():
    from src.semantic_search_benchmark import RetrievalCase, evaluate_retrieval

    index = LocalSemanticIndex(FakeEmbedder())
    with pytest.raises(ValueError, match="at least one"):
        evaluate_retrieval(index, [])
    with pytest.raises(ValueError, match="k"):
        evaluate_retrieval(index, [RetrievalCase("query", ("doc",))], k=0)
    with pytest.raises(ValueError, match="relevant"):
        evaluate_retrieval(index, [RetrievalCase("query", ())])



def test_retrieval_benchmark_normalizes_expected_document_ids():
    from src.semantic_search_benchmark import RetrievalCase, evaluate_retrieval

    index = LocalSemanticIndex(FakeEmbedder())
    index.add_document("pets", "Cats are small pets. A kitten needs care.")

    result = evaluate_retrieval(
        index,
        [RetrievalCase("kitten and cat", (" pets ", "pets"))],
        k=1,
    )

    assert result.recall_at_k == 1.0
    assert result.mean_reciprocal_rank == 1.0
    assert result.mean_ndcg_at_k == 1.0


def test_retrieval_benchmark_rejects_whitespace_only_relevance_ids():
    from src.semantic_search_benchmark import RetrievalCase, evaluate_retrieval

    index = LocalSemanticIndex(FakeEmbedder())
    with pytest.raises(ValueError, match="relevant document IDs"):
        evaluate_retrieval(index, [RetrievalCase("query", ("   ",))])
