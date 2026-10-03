"""Active scientific-experiment selection for TARA Baby.

Grounding: scientific experimentation can be framed as selecting observations
that reduce uncertainty about competing hypotheses. Closed-loop neurophysiology
research also uses active learning to choose informative stimuli efficiently.
TARA implements a simple Bayesian information-gain selector for toy/educational
scientific reasoning; it is not a substitute for experimental design expertise.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .scientific_reasoning import Evidence, Hypothesis, BayesianReasoner


@dataclass(frozen=True)
class ExperimentOption:
    name: str
    evidence_if_true: float
    evidence_if_false: float
    cost: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment name must be non-empty")
        for value in (self.evidence_if_true, self.evidence_if_false):
            if not 0.0 < value <= 1.0:
                raise ValueError("evidence likelihoods must be in (0, 1]")
        if self.cost <= 0:
            raise ValueError("cost must be > 0")


class ActiveExperimenter:
    """Rank experiments by expected information gain per unit cost."""

    def __init__(self, reasoner: BayesianReasoner | None = None) -> None:
        self.reasoner = reasoner or BayesianReasoner()

    @staticmethod
    def _entropy(probability: float) -> float:
        if probability <= 0.0 or probability >= 1.0:
            return 0.0
        return -(probability * math.log2(probability) + (1.0 - probability) * math.log2(1.0 - probability))

    def expected_information_gain(self, hypothesis: Hypothesis, option: ExperimentOption) -> float:
        prior = hypothesis.prior
        p_positive = prior * option.evidence_if_true + (1.0 - prior) * option.evidence_if_false
        p_positive = min(1.0 - 1e-12, max(1e-12, p_positive))
        posterior_positive = self.reasoner.posterior(
            hypothesis, [Evidence(option.evidence_if_true, option.evidence_if_false)]
        )
        posterior_negative = self.reasoner.posterior(
            hypothesis, [Evidence(1.0 - option.evidence_if_true, 1.0 - option.evidence_if_false)]
        ) if option.evidence_if_true < 1.0 and option.evidence_if_false < 1.0 else prior
        prior_entropy = self._entropy(prior)
        expected_posterior_entropy = (
            p_positive * self._entropy(posterior_positive)
            + (1.0 - p_positive) * self._entropy(posterior_negative)
        )
        return max(0.0, prior_entropy - expected_posterior_entropy)

    def choose(self, hypothesis: Hypothesis, options: list[ExperimentOption]) -> ExperimentOption | None:
        if not options:
            return None
        ranked = sorted(
            options,
            key=lambda option: (
                self.expected_information_gain(hypothesis, option) / option.cost,
                option.name,
            ),
            reverse=True,
        )
        return ranked[0]
