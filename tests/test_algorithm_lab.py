from src.algorithm_lab import CandidateDiversity, ProblemInterpreter
from src.polyglot.candidate import PolyglotCandidate


def test_problem_interpreter_extracts_optimization_target_and_constraints():
    model = ProblemInterpreter().interpret("Find a fast solution. Must use bounded memory.")
    assert model.optimization_target == "memory"
    assert model.constraints


def test_candidate_diversity_is_zero_for_identical_candidates():
    candidate = PolyglotCandidate("sort", "python", "def solve(x): return sorted(x)")
    assert CandidateDiversity().diversity(candidate, [candidate]) == 0.0
