from src.brain import TARABrain


class FakeProvider:
    model = "test-model"

    def __init__(self):
        self.system = ""
        self.prompts = []

    def complete(self, prompt, *, system=None, max_completion_tokens=32, temperature=0.6):
        self.system = system or ""
        self.prompts.append(prompt)
        return f"reply to {prompt}"


def make_brain():
    return TARABrain(llm_provider=FakeProvider())


def test_brain_connects_observation_memory_and_generation():
    brain = make_brain()
    response = brain.respond("tara", remember_key="prompt", max_new_tokens=3)
    assert response.text.startswith("reply to ")
    assert response.observations[-1] == "tara"
    assert response.recalled == ()


def test_brain_supports_goal_planning_and_verification():
    brain = make_brain()
    brain.set_goal("finish task", ("done",))
    brain.plan(["inspect", "finish"])
    assert brain.agent.next_task().task == "inspect"
    brain.verify_result("ok", "ok")
    assert brain.agent.next_task().task == "finish"


def test_brain_generation_uses_configured_inference_provider():
    brain = make_brain()
    output = brain.generate("tara", max_new_tokens=5, temperature=1.0)
    assert output == "reply to tara"


def test_brain_rejects_invalid_generation_arguments():
    brain = make_brain()
    for kwargs in ({"max_new_tokens": 0}, {"temperature": 0}):
        try:
            brain.generate("tara", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid generation argument should fail")


def test_reasoning_context_is_bounded_without_truncating_user_request(monkeypatch):
    brain = make_brain()
    monkeypatch.setenv("TARA_MAX_CONTEXT_CHARS", "100")

    context = brain.build_reasoning_context(
        "important user request",
        recalled=(),
    )

    assert len(context) <= 100
    assert "important user request" in context
    assert "trimmed for low-resource mode" in context

def test_brain_passes_emotional_guidance_to_language_provider():
    brain = make_brain()

    class FakeProvider:
        model = "test-model"

        def __init__(self):
            self.system = ""

        def complete(self, prompt, *, system=None, max_completion_tokens=32, temperature=0.6):
            self.system = system or ""
            return "I'm sorry you're feeling sad. Want to tell me what happened?"

    provider = FakeProvider()
    brain.llm_provider = provider
    response = brain.respond("I feel sad", max_new_tokens=32)

    assert response.emotion == "sadness"
    assert brain.last_emotion == "sadness"
    assert "Response style:" in provider.system
    assert "never claim TARA literally feels emotions" in provider.system


def test_brain_uses_opt_in_tool_completion_when_provider_supports_it(monkeypatch):
    class ToolProvider(FakeProvider):
        def __init__(self):
            super().__init__()
            self.tool_calls = []

        def complete_with_tools(self, prompt, *, system=None, max_completion_tokens=32,
                                temperature=0.6, public_tools=None):
            self.tool_calls.append({
                "prompt": prompt,
                "system": system,
                "max_completion_tokens": max_completion_tokens,
                "temperature": temperature,
                "public_tools": public_tools,
            })
            return "I checked the approved public source."

    provider = ToolProvider()
    monkeypatch.setenv("TARA_TOOL_CALLING_ENABLED", "true")
    brain = TARABrain(llm_provider=provider)

    response = brain.respond("What's the weather in Pune?", max_new_tokens=24)

    assert response.text == "I checked the approved public source."
    assert len(provider.tool_calls) == 1
    assert provider.tool_calls[0]["public_tools"] is brain.public_tools
    assert provider.prompts == []


def test_brain_keeps_plain_completion_when_tool_calling_is_disabled(monkeypatch):
    class ToolProvider(FakeProvider):
        def __init__(self):
            super().__init__()
            self.tool_calls = 0

        def complete_with_tools(self, *args, **kwargs):
            self.tool_calls += 1
            return "should not be called"

    provider = ToolProvider()
    monkeypatch.setenv("TARA_TOOL_CALLING_ENABLED", "false")
    brain = TARABrain(llm_provider=provider)

    response = brain.respond("Hello TARA")

    assert response.text.startswith("reply to ")
    assert provider.tool_calls == 0
    assert len(provider.prompts) == 1
