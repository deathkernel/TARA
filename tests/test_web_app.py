from src.web_app import ChatRequestError, answer_chat, build_conversation_prompt


def test_build_conversation_prompt_preserves_order_and_latest_user():
    prompt = build_conversation_prompt([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Help me"},
    ])
    assert "User: Hello" in prompt
    assert "TARA: Hi" in prompt
    assert prompt.endswith("TARA:")
    assert prompt.index("User: Hello") < prompt.index("TARA: Hi") < prompt.index("User: Help me")


def test_build_conversation_prompt_rejects_assistant_as_last_message():
    try:
        build_conversation_prompt([{"role": "assistant", "content": "hello"}])
    except ChatRequestError as exc:
        assert "last message" in str(exc)
    else:
        raise AssertionError("Expected assistant-only final message to be rejected")


def test_build_conversation_prompt_enforces_message_limit():
    try:
        build_conversation_prompt([{"role": "user", "content": "x" * 4001}])
    except ChatRequestError as exc:
        assert "4000" in str(exc)
    else:
        raise AssertionError("Expected oversized message to be rejected")


def test_answer_chat_returns_brain_response_text():
    class FakeResponse:
        text = "Hello from TARA"

    class FakeBrain:
        def respond(self, prompt, *, max_new_tokens, temperature):
            assert "User: Hello" in prompt
            assert max_new_tokens == 512
            return FakeResponse()

    assert answer_chat(FakeBrain(), {"messages": [{"role": "user", "content": "Hello"}]}) == {
        "reply": "Hello from TARA"
    }
