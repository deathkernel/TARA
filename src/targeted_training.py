"""Build bounded, deterministic training examples from measured failures.

The builder turns benchmark evidence into synthetic supervised examples. These
examples are explicitly labeled as generated training data; they are not treated
as newly discovered facts. The source benchmark remains the evaluation authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TargetedExample:
    example_id: str
    prompt: str
    solution: str
    tests: tuple[str, ...]
    category: str
    source_case: str
    kind: str
    provenance: str = "synthetic-from-benchmark-failure"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class TargetedTrainingBuilder:
    """Create supervised examples focused on observed benchmark failures."""

    def __init__(self, *, variants_per_failure: int = 3, max_examples: int = 128):
        if variants_per_failure <= 0 or max_examples <= 0:
            raise ValueError("variants_per_failure and max_examples must be positive")
        self.variants_per_failure = variants_per_failure
        self.max_examples = max_examples

    @staticmethod
    def _variants(prompt: str, solution: str) -> tuple[tuple[str, str], ...]:
        return (
            (prompt, solution),
            (f"Answer directly: {prompt}", solution),
            (f"Give only the expected answer for: {prompt}", solution),
        )

    def build(self, diagnosis: dict[str, Any]) -> tuple[TargetedExample, ...]:
        examples: list[TargetedExample] = []
        for failure in diagnosis.get("failures", []):
            if not isinstance(failure, dict):
                continue
            prompt = str(failure.get("prompt", "")).strip()
            if not prompt:
                continue
            solution = str(failure.get("expected", "")).strip()
            if not solution:
                continue
            variants = self._variants(prompt, solution)[: self.variants_per_failure]
            for index, (variant_prompt, variant_solution) in enumerate(variants):
                payload = {
                    "prompt": variant_prompt,
                    "solution": variant_solution,
                    "case": failure.get("case_id"),
                    "index": index,
                }
                examples.append(TargetedExample(
                    example_id=_fingerprint(payload)[:16],
                    prompt=variant_prompt,
                    solution=variant_solution,
                    tests=(f"expected={variant_solution}",),
                    category=str(failure.get("category", "unknown")),
                    source_case=str(failure.get("case_id", "unknown")),
                    kind=str(failure.get("kind", "incorrect_answer")),
                ))
                if len(examples) >= self.max_examples:
                    return tuple(examples)
        return tuple(examples)


def load_diagnosis(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_targeted_dataset(examples: Iterable[TargetedExample], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(item.as_dict(), sort_keys=True) for item in examples]
    destination.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return destination


def build_targeted_dataset(
    diagnosis_path: str | Path,
    output_path: str | Path,
    *,
    variants_per_failure: int = 3,
    max_examples: int = 128,
) -> tuple[TargetedExample, ...]:
    diagnosis = load_diagnosis(diagnosis_path)
    examples = TargetedTrainingBuilder(
        variants_per_failure=variants_per_failure,
        max_examples=max_examples,
    ).build(diagnosis)
    write_targeted_dataset(examples, output_path)
    return examples
