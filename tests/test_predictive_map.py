from src.tara_mind.cognition.predictive_map import PredictiveMap


def test_transition_learning_and_prediction():
    model = PredictiveMap()
    model.observe_transition("A", "B")
    model.observe_transition("A", "B")
    model.observe_transition("A", "C")
    assert model.predict_next("A") == "B"


def test_successor_features_include_future_state_mass():
    model = PredictiveMap(gamma=0.9)
    model.observe_transition("A", "B")
    model.observe_transition("B", "C")
    features = model.successor_features("A")
    assert features["B"] > 0.0
    assert features["C"] > 0.0
