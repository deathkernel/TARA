from src.tara_mind.cognition.active_experiment import ActiveExperimenter, ExperimentOption
from src.tara_mind.cognition.scientific_reasoning import Hypothesis


def test_experimenter_prefers_more_discriminative_option():
    hypothesis = Hypothesis("H", 0.5)
    experimenter = ActiveExperimenter()
    strong = ExperimentOption("strong", 0.9, 0.1)
    weak = ExperimentOption("weak", 0.6, 0.4)
    assert experimenter.choose(hypothesis, [weak, strong]).name == "strong"
