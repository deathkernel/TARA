"""Local, in-memory semantic document retrieval powered by Ollama embeddings.

No hosted embedding API, vector database, or on-disk index is used. Documents and
vectors stay in this Python process and disappear when the index is discarded.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EmbeddingTransport = Callable[[str, dict[str, Any], float], dict[str, Any]]


def _http_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(10_000_001)
    except HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama embedding API returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Could not reach Ollama embedding API at {url}: {exc}") from exc
    if len(body) > 10_000_000:
        raise RuntimeError("Ollama embedding response exceeded the 10 MB safety limit")
    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Ollama embedding API returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Ollama embedding API returned an unexpected response")
    if result.get("error"):
        raise RuntimeError(f"Ollama embedding error: {str(result['error'])[:300]}")
    return result


class OllamaEmbedder:
    """Generate vectors through Ollama's local /api/embed endpoint."""

    def __init__(
        self,
        *,
        model: str = "embeddinggemma",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
        batch_size: int = 16,
        max_input_chars: int = 8000,
        transport: EmbeddingTransport | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("embedding model must not be empty")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Ollama base URL must not be empty")
        if timeout <= 0 or batch_size < 1 or max_input_chars < 1:
            raise ValueError("timeout, batch_size, and max_input_chars must be positive")
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.batch_size = min(int(batch_size), 64)
        self.max_input_chars = int(max_input_chars)
        self._transport = transport or _http_json

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed texts in bounded batches and validate vector shape and values."""
        items = list(texts)
        if not items:
            return []
        if any(not isinstance(item, str) or not item.strip() for item in items):
            raise ValueError("every embedding input must be a non-empty string")
        if any(len(item) > self.max_input_chars for item in items):
            raise ValueError(f"embedding input exceeds {self.max_input_chars} characters")

        vectors: list[list[float]] = []
        expected_dimensions: int | None = None
        for offset in range(0, len(items), self.batch_size):
            batch = items[offset:offset + self.batch_size]
            result = self._transport(
                f"{self.base_url}/api/embed",
                {"model": self.model, "input": batch, "truncate": False},
                self.timeout,
            )
            raw_vectors = result.get("embeddings")
            if not isinstance(raw_vectors, list) or len(raw_vectors) != len(batch):
                raise RuntimeError("Ollama returned the wrong number of embedding vectors")
            for raw in raw_vectors:
                if not isinstance(raw, list) or not raw:
                    raise RuntimeError("Ollama returned an empty or invalid embedding vector")
                vector: list[float] = []
                for value in raw:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise RuntimeError("Embedding vectors must contain only finite numbers")
                    number = float(value)
                    if not math.isfinite(number):
                        raise RuntimeError("Embedding vectors must contain only finite numbers")
                    vector.append(number)
                if expected_dimensions is None:
                    expected_dimensions = len(vector)
                if len(vector) != expected_dimensions:
                    raise RuntimeError("Ollama returned inconsistent embedding dimensions")
                vectors.append(vector)
        return vectors


@dataclass(frozen=True)
class SemanticSearchResult:
    document_id: str
    source: str
    text: str
    score: float
    chunk_index: int


@dataclass(frozen=True)
class _Chunk:
    document_id: str
    source: str
    text: str
    vector: tuple[float, ...]
    chunk_index: int


def _cosine_similarity(left: tuple[float, ...] | list[float], right: tuple[float, ...] | list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must have the same non-zero dimensions")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _split_text(text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
    """Split text into bounded overlapping chunks, preferring nearby whitespace."""
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_chars, length)
        if end < length:
            boundary = text.rfind(" ", start + chunk_chars // 2, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        next_start = max(start + 1, end - overlap_chars)
        start = next_start
    return chunks


class LocalSemanticIndex:
    """Bounded in-memory document index; no disk writes or implicit persistence."""

    def __init__(
        self,
        embedder: OllamaEmbedder | Any | None = None,
        *,
        chunk_chars: int = 900,
        overlap_chars: int = 120,
        max_documents: int = 100,
        max_chunks: int = 2000,
        max_document_chars: int = 200_000,
    ) -> None:
        if chunk_chars < 100:
            raise ValueError("chunk_chars must be at least 100")
        if overlap_chars < 0 or overlap_chars >= chunk_chars:
            raise ValueError("overlap_chars must be between 0 and chunk_chars - 1")
        if max_documents < 1 or max_chunks < 1 or max_document_chars < 1:
            raise ValueError("index limits must be positive")
        self.embedder = embedder or OllamaEmbedder()
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.max_documents = max_documents
        self.max_chunks = max_chunks
        self.max_document_chars = max_document_chars
        self._documents: dict[str, tuple[_Chunk, ...]] = {}
        self._chunk_count = 0

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    def add_document(self, document_id: str, text: str, *, source: str = "") -> int:
        """Chunk and index one document. Existing IDs are replaced atomically."""
        if not isinstance(document_id, str) or not document_id.strip() or len(document_id) > 200:
            raise ValueError("document_id must contain 1-200 characters")
        document_id = document_id.strip()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("document text must not be empty")
        if len(text) > self.max_document_chars:
            raise ValueError(f"document exceeds {self.max_document_chars} characters")
        if not isinstance(source, str) or len(source) > 1000:
            raise ValueError("source must be a string of at most 1000 characters")
        pieces = _split_text(text.strip(), chunk_chars=self.chunk_chars, overlap_chars=self.overlap_chars)
        if not pieces:
            raise ValueError("document did not produce any searchable chunks")
        if len(pieces) > self.max_chunks:
            raise ValueError(f"document creates too many chunks (limit {self.max_chunks})")

        old = self._documents.get(document_id, ())
        if document_id not in self._documents and len(self._documents) >= self.max_documents:
            raise ValueError(f"document limit reached ({self.max_documents})")
        if self._chunk_count - len(old) + len(pieces) > self.max_chunks:
            raise ValueError(f"index chunk limit reached ({self.max_chunks})")

        vectors = self.embedder.embed(pieces)
        if len(vectors) != len(pieces):
            raise RuntimeError("embedder returned the wrong number of vectors")
        prepared: list[_Chunk] = []
        dimensions: int | None = None
        for index, (piece, raw_vector) in enumerate(zip(pieces, vectors)):
            if not isinstance(raw_vector, (list, tuple)) or not raw_vector:
                raise RuntimeError("embedder returned an empty or invalid vector")
            vector = tuple(float(value) for value in raw_vector)
            if any(not math.isfinite(value) for value in vector):
                raise RuntimeError("embedder returned a non-finite vector")
            if dimensions is None:
                dimensions = len(vector)
            if len(vector) != dimensions:
                raise RuntimeError("embedder returned inconsistent vector dimensions")
            prepared.append(_Chunk(document_id, source, piece, vector, index))

        # Commit only after every embedding validates; a failed replacement keeps old data.
        self._documents[document_id] = tuple(prepared)
        self._chunk_count = self._chunk_count - len(old) + len(prepared)
        return len(prepared)

    def search(self, query: str, *, top_k: int = 5, min_score: float = -1.0) -> list[SemanticSearchResult]:
        """Return best-matching chunks by cosine similarity, with stable tie ordering."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer from 1 to 20")
        if isinstance(min_score, bool) or not isinstance(min_score, (int, float)) or not -1 <= min_score <= 1:
            raise ValueError("min_score must be between -1 and 1")
        if not self._documents:
            return []
        vectors = self.embedder.embed([query.strip()])
        if len(vectors) != 1 or not vectors[0]:
            raise RuntimeError("embedder returned an invalid query vector")
        query_vector = tuple(float(value) for value in vectors[0])
        if any(not math.isfinite(value) for value in query_vector):
            raise RuntimeError("embedder returned a non-finite query vector")

        matches: list[SemanticSearchResult] = []
        for chunks in self._documents.values():
            for chunk in chunks:
                if len(chunk.vector) != len(query_vector):
                    raise RuntimeError("query and document embedding dimensions do not match")
                score = _cosine_similarity(query_vector, chunk.vector)
                if score >= min_score:
                    matches.append(SemanticSearchResult(
                        document_id=chunk.document_id,
                        source=chunk.source,
                        text=chunk.text,
                        score=score,
                        chunk_index=chunk.chunk_index,
                    ))
        matches.sort(key=lambda result: (-result.score, result.document_id, result.chunk_index))
        return matches[:top_k]

    def remove_document(self, document_id: str) -> bool:
        """Remove a document and its vectors from this process."""
        removed = self._documents.pop(document_id, None)
        if removed is None:
            return False
        self._chunk_count -= len(removed)
        return True

    def clear(self) -> None:
        """Discard all indexed documents and vectors from this process."""
        self._documents.clear()
        self._chunk_count = 0


__all__ = ["LocalSemanticIndex", "OllamaEmbedder", "SemanticSearchResult"]
