"""Reproducible scientific experimentation for TARA.

The engine separates experiment design, execution, measurement, analysis and
conclusion. Execution is supplied by a caller so the core never assumes
permission to run arbitrary code or mutate external systems.

Statistics are intentionally dependency-free: repeated-trial summaries,
standard error, bootstrap confidence intervals, effect size and reproducibility
fingerprints are provided for bounded experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import math
import random
from statistics import mean, stdev
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ExperimentVariable:
    name: str
    values: tuple[Any, ...]
    role: str = "independent"


@dataclass(frozen=True)
class ExperimentCondition:
    parameters: tuple[tuple[str, Any], ...]
    condition_id: str


@dataclass(frozen=True)
class ExperimentDesign:
    hypothesis: str
    variables: tuple[ExperimentVariable, ...]
    conditions: tuple[ExperimentCondition, ...]
    control_condition: str | None
    repetitions: int
    seed: int


@dataclass(frozen=True)
class TrialResult:
    condition_id: str
    repetition: int
    seed: int
    measurements: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class MeasurementSummary:
    condition_id: str
    metric: str
    n: int
    mean: float
    stddev: float
    standard_error: float
    minimum: float
    maximum: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class EffectEstimate:
    metric: str
    control_condition: str
    treatment_condition: str
    mean_difference: float
    relative_change: float | None
    effect_size: float | None
    ci_low: float
    ci_high: float
    interpretation: str


@dataclass(frozen=True)
class ExperimentConclusion:
    status: str
    statement: str
    confidence: float


@dataclass(frozen=True)
class ExperimentReport:
    design: ExperimentDesign
    trials: tuple[TrialResult, ...]
    summaries: tuple[MeasurementSummary, ...]
    effects: tuple[EffectEstimate, ...]
    conclusions: tuple[ExperimentConclusion, ...]
    reproducibility_fingerprint: str


def _fingerprint(parts: Iterable[str]) -> str:
    return sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _jsonish(value: Any) -> str:
    if isinstance(value, Mapping):
        return "{" + ",".join(f"{k}={_jsonish(value[k])}" for k in sorted(value)) + "}"
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(_jsonish(item) for item in value) + "]"
    return repr(value)


def _bootstrap_ci(values: Sequence[float], *, seed: int, samples: int = 1000, confidence: float = 0.95) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    boot = []
    size = len(values)
    for _ in range(max(100, samples)):
        boot.append(mean(values[rng.randrange(size)] for _ in range(size)))
    boot.sort()
    alpha = (1.0 - confidence) / 2.0
    low = boot[max(0, int(alpha * len(boot)))]
    high = boot[min(len(boot) - 1, int((1.0 - alpha) * len(boot)))]
    return low, high


class ExperimentDesigner:
    def __init__(self, *, max_conditions: int = 64):
        if max_conditions <= 0:
            raise ValueError("max_conditions must be positive")
        self.max_conditions = max_conditions

    def design(
        self,
        hypothesis: str,
        variables: Sequence[ExperimentVariable],
        *,
        repetitions: int = 3,
        seed: int = 0,
        control: Mapping[str, Any] | None = None,
    ) -> ExperimentDesign:
        if not str(hypothesis).strip():
            raise ValueError("hypothesis must not be empty")
        if repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if not variables:
            raise ValueError("at least one variable is required")
        normalized = []
        for variable in variables:
            if not variable.name.strip() or not variable.values:
                raise ValueError("variables need a name and at least one value")
            normalized.append(variable)
        combinations = list(product(*(variable.values for variable in normalized)))
        if len(combinations) > self.max_conditions:
            raise ValueError("design exceeds max_conditions")
        conditions = []
        for values in combinations:
            parameters = tuple((variable.name, value) for variable, value in zip(normalized, values))
            condition_id = _fingerprint([hypothesis, _jsonish(parameters)])[:16]
            conditions.append(ExperimentCondition(parameters, condition_id))
        control_id = None
        if control is not None:
            control_tuple = tuple((variable.name, control[variable.name]) for variable in normalized if variable.name in control)
            matches = [condition for condition in conditions if dict(condition.parameters) == dict(control_tuple)]
            if not matches:
                raise ValueError("control configuration does not match design")
            control_id = matches[0].condition_id
        elif conditions:
            control_id = conditions[0].condition_id
        return ExperimentDesign(
            hypothesis=str(hypothesis),
            variables=tuple(normalized),
            conditions=tuple(conditions),
            control_condition=control_id,
            repetitions=repetitions,
            seed=seed,
        )


class ScientificExperimentEngine:
    def __init__(self, *, designer: ExperimentDesigner | None = None, bootstrap_samples: int = 1000):
        if bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be >= 100")
        self.designer = designer or ExperimentDesigner()
        self.bootstrap_samples = bootstrap_samples

    def run(
        self,
        design: ExperimentDesign,
        executor: Callable[[Mapping[str, Any], int], Mapping[str, float]],
    ) -> ExperimentReport:
        trials = []
        for condition_index, condition in enumerate(design.conditions):
            parameters = dict(condition.parameters)
            for repetition in range(design.repetitions):
                trial_seed = design.seed + condition_index * 100_000 + repetition
                raw = executor(parameters, trial_seed)
                if not isinstance(raw, Mapping) or not raw:
                    raise ValueError("executor must return a non-empty mapping")
                measurements = []
                for metric, value in sorted(raw.items()):
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(f"measurement {metric!r} is not finite")
                    measurements.append((str(metric), value))
                trials.append(TrialResult(condition.condition_id, repetition, trial_seed, tuple(measurements)))

        summaries = self._summaries(design, trials)
        effects = self._effects(design, trials)
        conclusions = self._conclusions(design, effects)
        fingerprint = _fingerprint([
            design.hypothesis,
            str(design.seed),
            str(design.repetitions),
            *(
                condition.condition_id + ":" + _jsonish(condition.parameters)
                for condition in design.conditions
            ),
            *(
                trial.condition_id + ":" + str(trial.repetition) + ":" + str(trial.seed) + ":" + _jsonish(trial.measurements)
                for trial in trials
            ),
        ])
        return ExperimentReport(
            design=design,
            trials=tuple(trials),
            summaries=tuple(summaries),
            effects=tuple(effects),
            conclusions=tuple(conclusions),
            reproducibility_fingerprint=fingerprint,
        )

    def _summaries(self, design: ExperimentDesign, trials: Sequence[TrialResult]) -> list[MeasurementSummary]:
        grouped: dict[tuple[str, str], list[float]] = {}
        for trial in trials:
            for metric, value in trial.measurements:
                grouped.setdefault((trial.condition_id, metric), []).append(value)
        summaries = []
        for (condition_id, metric), values in sorted(grouped.items()):
            avg = mean(values)
            sd = stdev(values) if len(values) > 1 else 0.0
            se = sd / math.sqrt(len(values)) if values else 0.0
            low, high = _bootstrap_ci(
                values,
                seed=design.seed ^ hash((condition_id, metric)) & 0xFFFFFFFF,
                samples=self.bootstrap_samples,
            )
            summaries.append(MeasurementSummary(condition_id, metric, len(values), avg, sd, se, min(values), max(values), low, high))
        return summaries

    def _effects(self, design: ExperimentDesign, trials: Sequence[TrialResult]) -> list[EffectEstimate]:
        if design.control_condition is None:
            return []
        control_trials = [trial for trial in trials if trial.condition_id == design.control_condition]
        if not control_trials:
            return []
        control_metrics = self._metric_values(control_trials)
        results = []
        for condition in design.conditions:
            if condition.condition_id == design.control_condition:
                continue
            treatment_metrics = self._metric_values([trial for trial in trials if trial.condition_id == condition.condition_id])
            for metric in sorted(set(control_metrics) & set(treatment_metrics)):
                control = control_metrics[metric]
                treatment = treatment_metrics[metric]
                difference = mean(treatment) - mean(control)
                relative = None if abs(mean(control)) < 1e-12 else difference / abs(mean(control))
                pooled = math.sqrt(
                    (max(0, len(control) - 1) * (stdev(control) if len(control) > 1 else 0.0) ** 2
                    + max(0, len(treatment) - 1) * (stdev(treatment) if len(treatment) > 1 else 0.0) ** 2)
                    / max(1, len(control) + len(treatment) - 2)
                )
                effect_size = difference / pooled if pooled > 1e-12 else None
                paired_count = min(len(control), len(treatment))
                paired = [treatment[i] - control[i] for i in range(paired_count)]
                ci_low, ci_high = _bootstrap_ci(
                    paired,
                    seed=design.seed ^ hash((condition.condition_id, metric, "effect")) & 0xFFFFFFFF,
                    samples=self.bootstrap_samples,
                )
                interpretation = (
                    "positive effect" if ci_low > 0 else
                    "negative effect" if ci_high < 0 else
                    "uncertain effect"
                )
                results.append(EffectEstimate(
                    metric, design.control_condition, condition.condition_id,
                    difference, relative, effect_size, ci_low, ci_high, interpretation,
                ))
        return results

    def _metric_values(self, trials: Sequence[TrialResult]) -> dict[str, list[float]]:
        grouped: dict[str, list[float]] = {}
        for trial in trials:
            for metric, value in trial.measurements:
                grouped.setdefault(metric, []).append(value)
        return grouped

    def _conclusions(self, design: ExperimentDesign, effects: Sequence[EffectEstimate]) -> list[ExperimentConclusion]:
        conclusions = []
        if not effects:
            return [ExperimentConclusion("insufficient-evidence", "No control-to-treatment effect could be estimated.", 0.0)]
        for effect in effects:
            width = abs(effect.ci_high - effect.ci_low)
            confidence = 0.95 if effect.ci_low > 0 or effect.ci_high < 0 else max(0.0, 0.5 - min(0.5, width))
            if effect.interpretation == "positive effect":
                statement = f"{effect.metric} shows a measurable positive difference for treatment {effect.treatment_condition} versus control {effect.control_condition}."
                status = "supported"
            elif effect.interpretation == "negative effect":
                statement = f"{effect.metric} shows a measurable negative difference for treatment {effect.treatment_condition} versus control {effect.control_condition}."
                status = "supported"
            else:
                statement = f"{effect.metric} remains statistically uncertain for treatment {effect.treatment_condition} versus control {effect.control_condition}."
                status = "inconclusive"
            conclusions.append(ExperimentConclusion(status, statement, confidence))
        return conclusions


class ReplicationChecker:
    """Compare independent reports for reproducibility."""

    def compare(self, first: ExperimentReport, second: ExperimentReport) -> tuple[bool, str]:
        same_design = first.design.hypothesis == second.design.hypothesis and first.design.repetitions == second.design.repetitions
        if not same_design:
            return False, "design mismatch"
        first_effects = {(e.metric, e.treatment_condition): e.interpretation for e in first.effects}
        second_effects = {(e.metric, e.treatment_condition): e.interpretation for e in second.effects}
        if not first_effects or not second_effects:
            return False, "no comparable effects"
        overlap = set(first_effects) & set(second_effects)
        if not overlap:
            return False, "no overlapping metrics"
        agreement = sum(first_effects[key] == second_effects[key] for key in overlap) / len(overlap)
        return agreement >= 0.8, f"effect-direction agreement={agreement:.2f}"
