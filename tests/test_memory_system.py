from src.tara_mind.memory.episodic import Episode, EpisodicMemory
from src.tara_mind.memory.semantic import SemanticMemory
from src.tara_mind.memory.schema import SchemaLearner
from src.tara_mind.memory.system import MemorySystem


def test_episodic_memory_preserves_context_and_retrieves_overlap():
    memory = EpisodicMemory(capacity=4)
    episode = Episode.create("e1", ["open", "read"], ["browser"], ["report"])
    memory.store(episode)
    assert memory.get("e1") == episode
    assert memory.retrieve(["browser"])[0].episode_id == "e1"


def test_semantic_memory_increases_confidence_with_repeated_evidence():
    memory = SemanticMemory()
    first = memory.learn("water", "liquid", 0.6, "e1")
    first_confidence = first.confidence
    second = memory.learn("water", "liquid", 0.6, "e2")
    assert second.confidence > first_confidence
    assert second.evidence_count == 2


def test_schema_needs_repeated_structure():
    learner = SchemaLearner(min_support=2)
    learner.observe("restaurant", ["arrive", "order", "eat"])
    assert learner.predict("restaurant") is None
    learner.observe("restaurant", ["arrive", "order", "eat"])
    schema = learner.predict("restaurant")
    assert schema is not None
    assert schema.support == 2


def test_consolidation_links_episodes_to_semantics_and_schema():
    memory = MemorySystem(schema_min_support=2)
    for index in range(2):
        memory.remember(Episode.create(
            f"e{index}",
            ["arrive", "order", "eat"],
            ["restaurant"],
            facts=["meal involves eating"],
            importance=0.8,
            novelty=0.2,
        ))
    report = memory.consolidate()
    assert report.episodes_considered == 2
    assert memory.semantic.get("meal involves eating") is not None
    assert memory.schemas.predict("restaurant") is not None
