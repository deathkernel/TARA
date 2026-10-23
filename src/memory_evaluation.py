"""Small, transparent metrics for fixture-based lexical retrieval evaluation."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .memory import LongTermMemory


def evaluate_retrieval_cases(
    cases: Iterable[Mapping[str, Any]], *, k: int = 2
) -> dict[str, int | float]:
    """Evaluate labelled retrieval fixtures without hiding no-match cases.

    Ranking metrics are averaged only over cases with one or more expected
    relevant keys. Cases with no expected keys are reported separately via
    no_match_accuracy. A no-match case is correct only when retrieval returns
    no results. These metrics are intended for deterministic fixture checks,
    not as a semantic-retrieval benchmark.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    case_count = 0
    ranking_case_count = 0
    no_match_case_count = 0
    no_match_correct = 0
    precision_sum = 0.0
    recall_sum = 0.0
    reciprocal_rank_sum = 0.0
    hit_count = 0

    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("each retrieval case must be a mapping")
        query = case.get("query")
        memories = case.get("memories")
        expected = case.get("expected_keys")
        if not isinstance(query, str) or not isinstance(memories, list):
            raise ValueError("each case needs a string query and a memories list")
        if not isinstance(expected, list) or any(not isinstance(key, str) for key in expected):
            raise ValueError("expected_keys must be a list of strings")

        memory = LongTermMemory()
        for item in memories:
            if not isinstance(item, Mapping) or not isinstance(item.get("key"), str) or "value" not in item:
                raise ValueError("each memory needs a string key and a value")
            memory.remember(item["key"], item["value"])

        actual = memory.retrieve(query, limit=k)
        actual_keys = [item["key"] for item in actual]
        case_count += 1

        if not expected:
            no_match_case_count += 1
            if not actual_keys:
                no_match_correct += 1
            continue

        ranking_case_count += 1
        relevant = set(expected)
        hits = sum(1 for key in actual_keys if key in relevant)
        precision_sum += hits / k
        recall_sum += hits / len(relevant)
        if hits:
            hit_count += 1
        for rank, key in enumerate(actual_keys, start=1):
            if key in relevant:
                reciprocal_rank_sum += 1.0 / rank
                break

    denominator = ranking_case_count
    return {
        "case_count": case_count,
        "ranking_case_count": ranking_case_count,
        "no_match_case_count": no_match_case_count,
        "precision_at_k": precision_sum / denominator if denominator else 0.0,
        "recall_at_k": recall_sum / denominator if denominator else 0.0,
        "hit_rate_at_k": hit_count / denominator if denominator else 0.0,
        "mrr": reciprocal_rank_sum / denominator if denominator else 0.0,
        "no_match_accuracy": no_match_correct / no_match_case_count if no_match_case_count else 0.0,
    }


__all__ = ["evaluate_retrieval_cases"]
