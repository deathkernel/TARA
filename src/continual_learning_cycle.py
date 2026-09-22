"""Persistent continual-learning orchestration for TARA.

Phase 37.6 connects cognitive memory to reproducible training. The cycle
exports verified knowledge and failed attempts, converts them into deterministic
replay/targeted examples, builds a training corpus, and can run the existing
training pipeline. It records no claim of novelty: failures are supervision
signals and verified records are replay candidates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .cognitive_memory import CognitiveMemory
from .learning_bridge import LearningBridge
from .training_pipeline import TrainingConfig, TrainingPipeline, TrainingSummary


def _fingerprint_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _failure_text(item: dict[str, Any]) -> str:
    candidate = item.get("candidate", {})
    benchmark = item.get("benchmark", {})
    problem = candidate.get("problem", item.get("problem_id", "")) if isinstance(candidate, dict) else item.get("problem_id", "")
    source = candidate.get("source", "") if isinstance(candidate, dict) else str(candidate)
    failures = benchmark.get("failures", []) if isinstance(benchmark, dict) else []
    return (
        f"Problem: {problem}\n"
        f"Failed approach: {source}\n"
        f"Failure evidence: {json.dumps(failures, ensure_ascii=False, sort_keys=True)}"
    ).strip()


@dataclass(frozen=True)
class ContinualLearningReport:
    """Deterministic manifest for one persistent learning-cycle preparation."""

    base_records: int
    replay_records: int
    failure_records: int
    selected_replay: int
    selected_failures: int
    duplicates_removed: int
    dataset_fingerprint: str
    output_path: str
    replay_path: str
    failures_path: str


@dataclass(frozen=True)
class ContinualTrainingReport:
    """Training result plus the prepared continual-learning manifest."""

    cycle: ContinualLearningReport
    training: TrainingSummary
    regression_passed: bool | None
    regression_reason: str


class ContinualLearningCycle:
    """Prepare and optionally train from persistent cognitive memory."""

    def __init__(
        self,
        memory: CognitiveMemory,
        *,
        seed: int = 42,
        replay_ratio: float = 0.25,
        failure_ratio: float = 0.10,
        max_replay: int = 10000,
        max_failures: int = 10000,
    ) -> None:
        if not 0.0 <= replay_ratio <= 1.0:
            raise ValueError("replay_ratio must be in [0, 1]")
        if not 0.0 <= failure_ratio <= 1.0:
            raise ValueError("failure_ratio must be in [0, 1]")
        if max_replay <= 0 or max_failures <= 0:
            raise ValueError("memory limits must be positive")
        self.memory = memory
        self.bridge = LearningBridge(memory)
        self.seed = int(seed)
        self.replay_ratio = float(replay_ratio)
        self.failure_ratio = float(failure_ratio)
        self.max_replay = int(max_replay)
        self.max_failures = int(max_failures)

    @staticmethod
    def _read_base(path: str | Path) -> list[str]:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        values: list[str] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                if "problem" in item and "solution" in item:
                    text = f"Problem: {item['problem']}\nApproach: {item['solution']}\nTests: {item.get('tests', '')}".strip()
                else:
                    text = str(item.get("text", item.get("content", ""))).strip()
            else:
                raise ValueError("base dataset record must be a string or object")
            if text:
                values.append(text)
        if not values:
            raise ValueError("base dataset is empty")
        return values

    @staticmethod
    def _dedupe(values: list[str]) -> tuple[list[str], int]:
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            key = hashlib.sha256(value.encode("utf-8")).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(value)
        return unique, len(values) - len(unique)

    def _select(self, values: list[str], ratio: float, base_count: int) -> list[str]:
        if not values or ratio == 0.0:
            return []
        if ratio == 1.0:
            return list(values)
        count = max(1, round(base_count * ratio / (1.0 - ratio))) if base_count else len(values)
        # Stable ordering is preferable to stochastic replay for reproducible runs.
        return sorted(values, key=lambda value: hashlib.sha256((str(self.seed) + value).encode("utf-8")).hexdigest())[:count]

    def prepare(
        self,
        base_path: str | Path,
        output_path: str | Path,
        *,
        replay_path: str | Path,
        failures_path: str | Path,
    ) -> ContinualLearningReport:
        base = self._read_base(base_path)
        replay_destination = Path(replay_path)
        failures_destination = Path(failures_path)
        replay_destination.parent.mkdir(parents=True, exist_ok=True)
        failures_destination.parent.mkdir(parents=True, exist_ok=True)
        replay_count = self.bridge.export_verified_replay(replay_destination, limit=self.max_replay)
        failure_count = self.bridge.export_failures(failures_destination, limit=self.max_failures)

        replay_raw = []
        if replay_destination.exists():
            for line in replay_destination.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    replay_raw.append(json.dumps(json.loads(line), ensure_ascii=False, sort_keys=True))
        failure_raw = []
        if failures_destination.exists():
            for line in failures_destination.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    failure_raw.append(_failure_text(json.loads(line)))

        replay_texts = [json.loads(item).get("problem", "") + "\n" + json.loads(item).get("solution", "") for item in replay_raw]
        replay_texts = [text.strip() for text in replay_texts if text.strip()]
        failure_texts = [text for text in failure_raw if text]
        selected_replay = self._select(replay_texts, self.replay_ratio, len(base))
        selected_failures = self._select(failure_texts, self.failure_ratio, len(base))
        merged, duplicates = self._dedupe(base + selected_replay + selected_failures)

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for text in merged:
                handle.write(json.dumps({"text": text, "source": "persistent-continual-learning"}, ensure_ascii=False, sort_keys=True) + "\n")

        return ContinualLearningReport(
            base_records=len(base),
            replay_records=replay_count,
            failure_records=failure_count,
            selected_replay=len(selected_replay),
            selected_failures=len(selected_failures),
            duplicates_removed=duplicates,
            dataset_fingerprint=_fingerprint_lines(merged),
            output_path=str(destination),
            replay_path=str(replay_destination),
            failures_path=str(failures_destination),
        )

    @staticmethod
    def regression_check(previous_checkpoint: str | Path | None, training: TrainingSummary) -> tuple[bool | None, str]:
        """Check validation-loss regression against the previous same-dataset checkpoint.

        This is a conservative gate: it only compares checkpoints when the
        previous checkpoint contains a validation loss and the dataset
        fingerprint matches. A missing baseline is reported as not evaluated.
        """
        if previous_checkpoint is None:
            return None, "no previous checkpoint supplied"
        if training.validation_loss is None:
            return None, "training produced no validation loss"
        path = Path(previous_checkpoint)
        if not path.exists():
            return None, "previous checkpoint does not exist"
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("dataset_fingerprint") != training.dataset_fingerprint:
            return None, "previous checkpoint dataset fingerprint differs"
        previous_loss = payload.get("metrics", {}).get("validation_loss")
        if previous_loss is None:
            return None, "previous checkpoint has no validation loss"
        passed = float(training.validation_loss) <= float(previous_loss)
        reason = f"previous={float(previous_loss):.6f}, current={float(training.validation_loss):.6f}"
        return passed, reason

    def run(
        self,
        base_path: str | Path,
        output_path: str | Path,
        *,
        replay_path: str | Path,
        failures_path: str | Path,
        config: TrainingConfig,
        metrics_path: str | Path | None = None,
        resume: str | Path | None = None,
        previous_checkpoint: str | Path | None = None,
        device: str | None = None,
    ) -> ContinualTrainingReport:
        cycle = self.prepare(base_path, output_path.with_suffix(".prepared.jsonl") if isinstance(output_path, Path) else str(output_path) + ".prepared.jsonl", replay_path=replay_path, failures_path=failures_path)
        training = TrainingPipeline(config=config, device=device).train(cycle.output_path, output_path, resume=resume, metrics_path=metrics_path)
        passed, reason = self.regression_check(previous_checkpoint, training)
        return ContinualTrainingReport(cycle=cycle, training=training, regression_passed=passed, regression_reason=reason)
