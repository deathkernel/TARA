"""Deterministic retrieval-quality metrics for local semantic search.

This evaluator scores a fixed set of queries against expected document IDs. It
does not train a model, call a hosted service, or write evaluation artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable

from .local_semantic_search import LocalSemanticIndex


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalEvaluation:
    case_count: int
    recall_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float
    k: int
    fingerprint: str


def evaluate_retrieval(
    index: LocalSemanticIndex,
    cases: Iterable[RetrievalCase],
    *,
    k: int = 5,
) -> RetrievalEvaluation:
    """Compute Recall@k, MRR, NDCG@k and a deterministic result fingerprint."""
    if type(k) is not int or not 1 <= k <= 20:
        raise ValueError("k must be an integer from 1 to 20")
    items = list(cases)
    if not items:
        raise ValueError("at least one retrieval case is required")
    for case in items:
        if not isinstance(case, RetrievalCase):
            raise ValueError("cases must contain RetrievalCase values")
        if not isinstance(case.query, str) or not case.query.strip():
            raise ValueError("each case query must be non-empty")
        if not case.relevant_document_ids:
            raise ValueError("each case must define at least one relevant document ID")
        if any(not isinstance(doc_id, str) or not doc_id for doc_id in case.relevant_document_ids):
            raise ValueError("relevant document IDs must be non-empty strings")

    recall_values: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    fingerprints: list[dict[str, object]] = []
    for case in items:
        relevant = set(case.relevant_document_ids)
        results = index.search(case.query, top_k=k)
        ranked_ids: list[str] = []
        for result in results:
            if result.document_id not in ranked_ids:
                ranked_ids.append(result.document_id)
        hits = [doc_id for doc_id in ranked_ids if doc_id in relevant]
        recall_values.append(len(hits) / len(relevant))
        reciprocal_ranks.append(
            1.0 / (next((rank for rank, doc_id in enumerate(ranked_ids, 1) if doc_id in relevant), math.inf))
        )
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, doc_id in enumerate(ranked_ids, 1)
            if doc_id in relevant
        )
        ideal_hits = min(len(relevant), k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcg_values.append(dcg / ideal_dcg if ideal_dcg else 0.0)
        fingerprints.append({
            "query": case.query,
            "relevant": sorted(relevant),
            "ranked": ranked_ids,
            "scores": [round(result.score, 8) for result in results],
        })

    digest = hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    count = len(items)
    return RetrievalEvaluation(
        case_count=count,
        recall_at_k=sum(recall_values) / count,
        mean_reciprocal_rank=sum(reciprocal_ranks) / count,
        mean_ndcg_at_k=sum(ndcg_values) / count,
        k=k,
        fingerprint=digest,
    )


__all__ = ["RetrievalCase", "RetrievalEvaluation", "evaluate_retrieval"]
