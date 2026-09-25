from src.tara_mind.cognition.world_model import WorldModel


def test_world_model_learns_repeated_transition():
    model = WorldModel()
    model.observe("A", "B")
    model.observe("A", "B")
    assert model.predict("A") == "B"


def test_unexpected_transition_has_higher_surprise():
    model = WorldModel()
    model.observe("A", "B")
    model.observe("A", "B")
    expected = model.observe("A", "B")
    unexpected = model.observe("A", "C")
    assert unexpected.surprise > expected.surprise


def test_observation_updates_current_state():
    model = WorldModel()
    update = model.observe("A", "B")
    assert update.observation.next_state == "B"
    assert model.current_state == "B"
