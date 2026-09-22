from src.polyglot.generation import ModelCandidateGenerator, build_candidate_prompt
from src.polyglot.parser import CandidateParseError, parse_candidate


def test_parse_candidate_from_fenced_python():
    candidate = parse_candidate(
        "Language: python\n```python\nprint('ok')\n```",
        "sorting",
    )
    assert candidate.language == "python"
    assert candidate.source == "print('ok')\n"


def test_parse_candidate_uses_final_fence_after_prompt_example():
    output = "Prompt example:\n```python\n<complete program>\n```\nGenerated:\n```python\nprint('real')\n```"
    candidate = parse_candidate(output, "sorting")
    assert candidate.source == "print('real')\n"


def test_parse_candidate_rejects_unfenced_output():
    try:
        parse_candidate("python: print('ok')", "sorting")
    except CandidateParseError:
        pass
    else:
        raise AssertionError("unfenced output should be rejected")


def test_prompt_contains_machine_readable_contract():
    prompt = build_candidate_prompt("sorting", "python", ("first case failed",))
    assert "Target language: python" in prompt
    assert "```python" in prompt
    assert "first case failed" in prompt


def test_model_generator_deduplicates_candidates():
    calls = []

    def fake_generate(prompt, **kwargs):
        calls.append(kwargs["temperature"])
        return "```python\nprint('same')\n```"

    generator = ModelCandidateGenerator(fake_generate, temperatures=(0.5, 1.0))
    candidates = tuple(generator("sorting", "", (), 3))
    assert len(candidates) == 1
    assert calls == [0.5, 1.0, 0.5]
