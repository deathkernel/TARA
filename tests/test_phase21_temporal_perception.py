from src.perception import build_temporal_context, normalize_observation
from src.temporal_perception import ObservationNormalizer, TemporalPerception


def test_normalizer_produces_canonical_confidence_aware_observation():
    item = normalize_observation(
        {"text": "network online", "type": "system", "confidence": 0.91, "timestamp": 10.0},
        source="sensor",
    )
    assert item.source == "sensor"
    assert item.kind == "system"
    assert item.content == "network online"
    assert item.confidence == 0.91
    assert item.observation_id


def test_temporal_perception_orders_deduplicates_and_bounds_history():
    engine = TemporalPerception(max_history=3)
    engine.ingest("third", source="test", timestamp=30.0)
    engine.ingest("first", source="test", timestamp=10.0)
    engine.ingest("second", source="test", timestamp=20.0)
    engine.ingest("latest", source="test", timestamp=40.0)
    engine.ingest("latest", source="test", timestamp=40.0)

    ordered = engine.ordered()
    assert [item.content for item in ordered] == ["second", "third", "latest"]
    assert len(ordered) == 3


def test_temporal_context_tracks_relations_and_confidence_filter():
    engine = TemporalPerception(max_history=10)
    engine.ingest("weak", source="sensor", confidence=0.2, timestamp=1.0)
    engine.ingest("strong", source="sensor", confidence=0.95, timestamp=3.5)
    context = engine.context(limit=10, min_confidence=0.8)

    assert [item.content for item in context.observations] == ["strong"]
    assert context.relations == ()
    assert "confidence=0.95" in context.as_prompt_context()


def test_build_temporal_context_is_deterministic_for_existing_observations():
    normalizer = ObservationNormalizer()
    observations = [
        normalizer.normalize("later", source="a", timestamp=2.0),
        normalizer.normalize("earlier", source="a", timestamp=1.0),
    ]
    context = build_temporal_context(observations, limit=10)
    assert [item.content for item in context.observations] == ["earlier", "later"]
    assert len(context.relations) == 1
    assert context.relations[0].delta_seconds == 1.0
