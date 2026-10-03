from src.tara_mind.learning.continual import ContinualLearner
from src.tara_mind.memory.episodic import Episode


def test_experience_enters_memory_and_replay():
    learner = ContinualLearner()
    learner.learn_episode(Episode.create("e1", ["observe"], ["lab"], facts=["observation recorded"], prediction_error=1.2, novelty=0.8))
    assert len(learner.memory.episodic) == 1
    assert len(learner.replay) == 1


def test_consolidation_returns_report():
    learner = ContinualLearner()
    learner.learn_episode(Episode.create("e1", ["a"], ["x"], facts=["f"]))
    report = learner.consolidate()
    assert report.replayed == 1
    assert report.consolidation.episodes_considered == 1
