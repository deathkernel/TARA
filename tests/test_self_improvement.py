from src.polyglot.candidate import PolyglotCandidate
from src.polyglot.self_improvement import SelfImprovementEngine


def test_engine_uses_benchmark_feedback_and_keeps_verified_candidate():
    calls = []

    def generator(problem, feedback_text, feedback, count):
        calls.append((problem, feedback_text, feedback, count))
        if not feedback:
            yield PolyglotCandidate(problem, "python", "print('wrong')\n")
        else:
            yield PolyglotCandidate(
                problem,
                "python",
                "data=list(map(int,input().split()))\n"
                "n=data[0]\n"
                "print(*sorted(data[1:1+n]))\n",
            )

    result = SelfImprovementEngine().improve("sorting", generator, rounds=2, candidates_per_round=1)
    assert result.best is not None
    assert not result.best.verified
    assert len(result.history) == 2
    assert len(calls) == 2
    assert calls[1][2]
