import json
from pathlib import Path

import pytest

from src.memory import LongTermMemory


FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "memory_retrieval_cases.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("case", FIXTURES, ids=lambda case: case["name"])
def test_memory_retrieval_evaluation_cases(case):
    memory = LongTermMemory()
    for item in case["memories"]:
        memory.remember(item["key"], item["value"])

    actual = memory.retrieve(case["query"], limit=2)
    assert [item["key"] for item in actual] == case["expected_keys"]
    assert all(0.0 < item["score"] <= 1.0 for item in actual)



def test_retrieval_metrics_report_ranking_and_no_match_separately():
    from src.memory_evaluation import evaluate_retrieval_cases

    metrics = evaluate_retrieval_cases(FIXTURES, k=2)
    assert metrics["case_count"] == len(FIXTURES)
    assert metrics["ranking_case_count"] == len(FIXTURES) - 1
    assert metrics["no_match_case_count"] == 1
    assert metrics["precision_at_k"] == pytest.approx(7 / 12)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["hit_rate_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["no_match_accuracy"] == 1.0


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "2"])
def test_retrieval_metrics_reject_invalid_k(k):
    from src.memory_evaluation import evaluate_retrieval_cases

    with pytest.raises(ValueError, match="k must be a positive integer"):
        evaluate_retrieval_cases(FIXTURES, k=k)


def test_retrieval_metrics_handle_empty_fixture_collection():
    from src.memory_evaluation import evaluate_retrieval_cases

    metrics = evaluate_retrieval_cases([], k=2)
    assert metrics["case_count"] == 0
    assert metrics["ranking_case_count"] == 0
    assert metrics["no_match_case_count"] == 0
    assert metrics["precision_at_k"] == 0.0
    assert metrics["recall_at_k"] == 0.0
    assert metrics["hit_rate_at_k"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["no_match_accuracy"] == 0.0
