import json
from pathlib import Path

import pytest

from src.brain import TARABrain
from src.memory import LongTermMemory


CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "conversation_contract_cases.json").read_text(
        encoding="utf-8"
    )
)


class DeterministicProvider:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.answer


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_conversation_contract_fixture(case):
    memory = LongTermMemory()
    if case["memory"] is not None:
        memory.remember(case["memory"]["key"], case["memory"]["value"])
    provider = DeterministicProvider(case["answer"])
    brain = TARABrain(
        llm_provider=provider,
        memory=memory,
        language=case["language"],
        personality=case["personality"],
    )

    response = brain.respond(case["prompt"])

    assert response.text == case["expected_response"]
    assert brain.conversation_history == [tuple(item) for item in case["expected_history"]]
    assembled_prompt, kwargs = provider.calls[0]
    for expected in case["expected_prompt_contains"]:
        assert expected in assembled_prompt
    for expected in case["expected_system_contains"]:
        assert expected in kwargs["system"]
