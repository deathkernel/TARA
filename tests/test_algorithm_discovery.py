from src.algorithm_discovery import (
    AlgorithmCandidate,
    discover_algorithms,
    rank_verified,
)


def test_rank_verified_prefers_lower_score():
    candidates = [
        AlgorithmCandidate("p", "slow", score=5, verified=True),
        AlgorithmCandidate("p", "fast", score=2, verified=True),
        AlgorithmCandidate("p", "bad", score=0, verified=False),
    ]
    ranked = rank_verified(candidates)
    assert [candidate.algorithm for candidate in ranked] == ["fast", "slow"]


def test_discovery_keeps_failed_feedback_and_best_verified():
    def generate(problem, attempts):
        assert problem == "sort"
        return ["wrong", "correct"]

    def verify(problem, algorithm):
        if algorithm == "correct":
            return True, 1.0, "all tests passed"
        return False, None, "counterexample: [2, 1]"

    result = discover_algorithms("sort", generate, verify, attempts=2)
    assert result.best.algorithm == "correct"
    assert result.attempts[0].verified is False
    assert "counterexample" in result.attempts[0].feedback


def test_discovery_returns_none_when_all_candidates_fail():
    result = discover_algorithms(
        "p",
        lambda problem, attempts: ["a", "b"],
        lambda problem, algorithm: (False, None, "failed"),
        attempts=2,
    )
    assert result.best is None
    assert len(result.attempts) == 2
