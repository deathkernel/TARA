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
