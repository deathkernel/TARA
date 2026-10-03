from src.tara_mind.cognition.neuromodulation import NeuromodulatoryController


def test_surprise_increases_learning_gain():
    controller = NeuromodulatoryController()
    calm = controller.update(0.0, 0.0, 0.0)
    surprise = controller.update(0.0, 0.0, 2.0)
    assert surprise.learning_gain > calm.learning_gain
    assert surprise.replay_gain > calm.replay_gain
