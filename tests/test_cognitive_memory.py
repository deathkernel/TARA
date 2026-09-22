import json

import pytest

from src.cognitive_memory import CognitiveMemory


def test_new_knowledge_shape_needs_no_schema_migration(tmp_path):
    path = tmp_path / "cognitive.sqlite"
    with CognitiveMemory(path) as memory:
        item_id = memory.remember(
            kind="algorithm_discovery_result",
            domain="graph_theory",
            content={
                "algorithm": "adaptive_cut_search",
                "steps": ["sample", "verify", "refine"],
                "future_field": {"new": True},
            },
            confidence=0.91,
            verification="verified",
            provenance={"source": "codeforces", "attempt": 7},
            attributes={"rating": 2400, "tags": ["graphs", "search"]},
        )

        record = memory.get(item_id)
        assert record["kind"] == "algorithm_discovery_result"
        assert record["content"]["future_field"]["new"] is True
        assert record["attributes"]["rating"] == 2400
        assert memory.search(kind="algorithm_discovery_result")[0]["id"] == item_id

    with CognitiveMemory(path) as reopened:
        assert reopened.count() == 1
        assert reopened.get(item_id)["verification"] == "verified"


def test_relationships_and_event_history(tmp_path):
    with CognitiveMemory(tmp_path / "memory.sqlite") as memory:
        problem = memory.remember(
            kind="problem",
            domain="algorithms",
            content={"title": "Dynamic graph task"},
        )
        attempt = memory.remember(
            kind="algorithm_attempt",
            domain="algorithms",
            content={"language": "rust", "status": "failed"},
        )
        memory.link(problem, "has_attempt", attempt, {"iteration": 1})

        related = memory.related(problem, relation="has_attempt")
        assert len(related) == 1
        assert related[0]["id"] == attempt
        assert related[0]["relationship_metadata"]["iteration"] == 1

        events = memory.events(problem)
        assert any(event["event_type"] == "relationship_added" for event in events)


def test_update_recomputes_fingerprint_and_preserves_flexible_content(tmp_path):
    with CognitiveMemory(tmp_path / "memory.sqlite") as memory:
        item_id = memory.remember(
            kind="unknown_future_type",
            content={"v1": 1},
            provenance={"source": "experiment"},
        )
        old = memory.get(item_id)["fingerprint"]
        memory.update(item_id, content={"v1": 1, "v2": [1, 2, 3]}, verification="verified")
        new = memory.get(item_id)
        assert new["content"]["v2"] == [1, 2, 3]
        assert new["verification"] == "verified"
        assert new["fingerprint"] != old


def test_validation_rejects_invalid_confidence_and_verification(tmp_path):
    with CognitiveMemory(tmp_path / "memory.sqlite") as memory:
        with pytest.raises(ValueError):
            memory.remember(kind="x", content={}, confidence=1.1)
        with pytest.raises(ValueError):
            memory.remember(kind="x", content={}, verification="guessed")
