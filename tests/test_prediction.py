from src.tara_mind.cognition.prediction import Prediction, PredictionEngine


def test_prediction_error_is_observed_minus_predicted():
    result = PredictionEngine().compare(Prediction("temperature", 20.0), 23.0)
    assert result.error == 3.0
    assert result.weighted_error == 3.0


def test_higher_precision_amplifies_error():
    engine = PredictionEngine()
    low = engine.compare(Prediction("x", 10.0, precision=1.0), 12.0)
    high = engine.compare(Prediction("x", 10.0, precision=2.0), 12.0)
    assert high.weighted_error == 2 * low.weighted_error


def test_prediction_update_moves_toward_observation():
    new = PredictionEngine.update(Prediction("x", 10.0), 20.0, learning_rate=0.25)
    assert new.predicted == 12.5
