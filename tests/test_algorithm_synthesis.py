from src.algorithm_synthesis import (
    AlgorithmArchive,
    AlgorithmDiscoveryEngine,
    AlgorithmProposal,
    VerificationReport,
    build_algorithm_prompt,
    frame_problem,
    proposal_from_dict,
)


def proposal(name: str, strategy: str = "greedy_with_invariant") -> AlgorithmProposal:
    return AlgorithmProposal(
        name=name,
        strategy=strategy,
        idea=f"idea-{name}",
        steps=("initialize", "iterate", "return"),
        invariant="the maintained state represents all processed input correctly",
        correctness_argument="the invariant holds initially and is preserved by each step",
        time_complexity="O(n)",
        space_complexity="O(1)",
    )


def test_frame_problem_extracts_constraints():
    frame = frame_problem("Find the answer. It must use at most O(n) memory.")
    assert frame.objective.startswith("Find")
    assert frame.constraints == ("It must use at most O(n) memory",)


def test_discovery_refines_with_counterexamples_and_selects_verified(tmp_path):
    calls = []
    candidate_bad = proposal("bad")
    candidate_good = proposal("good", "divide_and_conquer")

    def generate(frame, strategies, feedback, count):
        calls.append((strategies, feedback))
        return (candidate_bad, candidate_good)

    def verify(frame, candidate):
        if candidate.name == "bad":
            return VerificationReport(candidate, False, 9.0, "wrong invariant", ("counterexample: [2,1]",), 1, 2)
        return VerificationReport(candidate, True, 2.0, "verified", (), 2, 2)

    archive = AlgorithmArchive(tmp_path / "algorithms.jsonl")
    engine = AlgorithmDiscoveryEngine(generate, verify, archive=archive)
    result = engine.discover("Sort an array", rounds=2, candidates_per_round=2)

    assert result.best == candidate_good
    assert result.rounds == 2
    assert len(result.reports) == 4
    assert calls[1][1]
    assert len(archive.history()) == 2


def test_proposal_parser_requires_structured_reasoning():
    parsed = proposal_from_dict({
        "name": "scan",
        "strategy": "greedy",
        "idea": "single pass",
        "steps": ["start", "scan", "return"],
        "invariant": "prefix is processed",
        "correctness_argument": "invariant is preserved",
        "time_complexity": "O(n)",
        "space_complexity": "O(1)",
    })
    assert parsed.name == "scan"
    assert parsed.fingerprint


def test_prompt_requires_invariant_and_no_code():
    prompt = build_algorithm_prompt(frame_problem("Find max"), "greedy_with_invariant")
    assert "correctness_argument" in prompt
    assert "Do not output executable source code" in prompt
