"""Candidate generation and verification primitives for algorithm discovery.

This module deliberately separates *generation* from *verification*. A language
model may propose many candidates, but a candidate is accepted only when an
external verifier says it is correct. This makes the discovery loop testable and
prevents generated text from being treated as ground truth.
"""

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class AlgorithmCandidate:
    """One proposed algorithm and its verification metadata."""

    problem: str
    algorithm: str
    score: float | None = None
    verified: bool = False
    feedback: str = ""


@dataclass(frozen=True)
class DiscoveryResult:
    """Best verified candidate plus all attempted candidates."""

    best: AlgorithmCandidate | None
    attempts: tuple[AlgorithmCandidate, ...]


def rank_verified(candidates: Iterable[AlgorithmCandidate]) -> list[AlgorithmCandidate]:
    """Return verified candidates ordered by score, then by shorter code/text."""
    verified = [candidate for candidate in candidates if candidate.verified]
    return sorted(
        verified,
        key=lambda candidate: (
            float("inf") if candidate.score is None else candidate.score,
            len(candidate.algorithm),
        ),
    )


def discover_algorithms(
    problem: str,
    generate: Callable[[str, int], Iterable[str]],
    verify: Callable[[str, str], tuple[bool, float | None, str]],
    attempts: int = 8,
) -> DiscoveryResult:
    """Generate, verify, and rank candidate algorithms.

    ``generate`` can be backed by TARA's language model. ``verify`` should run
    deterministic tests/benchmarks in a sandbox owned by the caller. A model
    output is never considered correct merely because it was generated.
    Lower scores are preferred (for example runtime, memory, or a weighted
    objective).
    """
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be a non-empty string")
    if attempts <= 0:
        raise ValueError("attempts must be positive")

    generated = list(generate(problem, attempts))
    if not generated:
        return DiscoveryResult(best=None, attempts=())

    evaluated: list[AlgorithmCandidate] = []
    for algorithm in generated[:attempts]:
        if not isinstance(algorithm, str) or not algorithm.strip():
            continue
        verified, score, feedback = verify(problem, algorithm)
        evaluated.append(
            AlgorithmCandidate(
                problem=problem,
                algorithm=algorithm,
                score=score,
                verified=bool(verified),
                feedback=str(feedback),
            )
        )

    ranked = rank_verified(evaluated)
    return DiscoveryResult(
        best=ranked[0] if ranked else None,
        attempts=tuple(evaluated),
    )
