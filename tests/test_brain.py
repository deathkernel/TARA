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


def test_semantic_retrieval_is_opt_in_and_adds_untrusted_sources(monkeypatch):
    class Match:
        document_id = "guide"
        source = "guide.md"
        score = 0.91
        text = "Ollama runs locally."

    class FakeIndex:
        def __init__(self):
            self.queries = []

        def search(self, query, *, top_k=5, min_score=-1.0):
            self.queries.append((query, top_k))
            return [Match()]

    index = FakeIndex()
    brain = TARABrain(llm_provider=FakeProvider(), semantic_index=index)
    monkeypatch.setenv("TARA_SEMANTIC_RETRIEVAL_ENABLED", "false")

    without_docs = brain.build_reasoning_context("How does Ollama run?", recalled=())
    assert "Retrieved document excerpts" not in without_docs
    assert index.queries == []

    with_docs = brain.build_reasoning_context(
        "How does Ollama run?", recalled=(), include_documents=True, document_limit=2
    )
    assert "Ollama runs locally." in with_docs
    assert "guide.md" in with_docs
    assert "untrusted reference data" in with_docs
    assert index.queries == [("How does Ollama run?", 2)]


def test_semantic_retrieval_rejects_unbounded_document_limit():
    brain = make_brain()
    for limit in (0, 11, True):
        try:
            brain.build_reasoning_context("query", recalled=(), include_documents=True, document_limit=limit)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid document_limit should fail")


def test_brain_exposes_explicit_local_document_index_without_persisting():
    from src.local_semantic_search import LocalSemanticIndex

    class FakeIndex:
        def __init__(self):
            self.added = []
            self.cleared = False

        def add_document(self, document_id, text, *, source=""):
            self.added.append((document_id, text, source))
            return 1

        def search(self, query, *, top_k=5, min_score=-1.0):
            return []

        def remove_document(self, document_id):
            return document_id == "guide"

        def clear(self):
            self.cleared = True

    index = FakeIndex()
    brain = TARABrain(llm_provider=FakeProvider(), semantic_index=index)

    assert brain.index_document("guide", "Ollama is local.", source="guide.md") == 1
    assert index.added == [("guide", "Ollama is local.", "guide.md")]
    assert brain.search_documents("where is model?", top_k=2) == []
    assert brain.remove_indexed_document("guide") is True
    brain.clear_document_index()
    assert index.cleared is True


def test_brain_default_semantic_index_is_lazy_and_process_local():
    from src.local_semantic_search import LocalSemanticIndex

    brain = TARABrain(llm_provider=FakeProvider())
    assert isinstance(brain.semantic_index, LocalSemanticIndex)
    assert brain.semantic_index.document_count == 0
    assert brain.semantic_index.chunk_count == 0


def test_brain_exposes_atomic_bulk_document_indexing():
    class FakeIndex:
        def add_documents(self, documents):
            return {item[0]: 1 for item in documents}

    brain = TARABrain(llm_provider=FakeProvider(), semantic_index=FakeIndex())
    assert brain.index_documents([("guide", "Local docs.")]) == {"guide": 1}



def test_semantic_retrieval_failure_falls_back_to_other_chat_context(monkeypatch):
    class FailingIndex:
        def search(self, query, *, top_k=5, min_score=-1.0):
            raise RuntimeError("local Ollama embedding server is offline")

    provider = FakeProvider()
    brain = TARABrain(llm_provider=provider, semantic_index=FailingIndex())
    monkeypatch.setenv("TARA_SEMANTIC_RETRIEVAL_ENABLED", "true")

    context = brain.build_reasoning_context("How does TARA work?", recalled=())

    assert "User/task: How does TARA work?" in context
    assert "Semantic retrieval unavailable for this turn" in context
    assert "local Ollama embedding server is offline" not in context


def test_semantic_retrieval_fallback_does_not_disable_explicit_document_limit_validation():
    class FailingIndex:
        def search(self, query, *, top_k=5, min_score=-1.0):
            raise RuntimeError("must not be called")

    brain = TARABrain(llm_provider=FakeProvider(), semantic_index=FailingIndex())

    try:
        brain.build_reasoning_context("query", recalled=(), include_documents=True, document_limit=True)
    except ValueError as exc:
        assert "document_limit" in str(exc)
    else:
        raise AssertionError("invalid document_limit must still fail")



def test_respond_completes_when_opt_in_semantic_retrieval_is_unavailable(monkeypatch):
    class FailingIndex:
        def search(self, query, *, top_k=5, min_score=-1.0):
            raise RuntimeError("embedding service unavailable")

    provider = FakeProvider()
    brain = TARABrain(llm_provider=provider, semantic_index=FailingIndex())
    monkeypatch.setenv("TARA_SEMANTIC_RETRIEVAL_ENABLED", "true")

    response = brain.respond("Explain TARA briefly")

    assert response.text.startswith("reply to ")
    assert "Semantic retrieval unavailable for this turn" in provider.prompts[0]
    assert "embedding service unavailable" not in provider.prompts[0]


def test_respond_combines_opt_in_semantic_context_with_opt_in_tool_calling(monkeypatch):
    class Match:
        document_id = "guide"
        source = "guide.md"
        score = 0.88
        text = "TARA supports local semantic search."

    class FakeIndex:
        def search(self, query, *, top_k=5, min_score=-1.0):
            return [Match()]

    class ToolProvider(FakeProvider):
        def __init__(self):
            super().__init__()
            self.tool_calls = []

        def complete_with_tools(self, prompt, *, system=None, max_completion_tokens=32,
                                temperature=0.6, public_tools=None):
            self.tool_calls.append({"prompt": prompt, "system": system, "public_tools": public_tools})
            return "Answer generated with approved tools."

    provider = ToolProvider()
    brain = TARABrain(llm_provider=provider, semantic_index=FakeIndex())
    monkeypatch.setenv("TARA_SEMANTIC_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("TARA_TOOL_CALLING_ENABLED", "true")

    response = brain.respond("How does TARA search documents?")

    assert response.text == "Answer generated with approved tools."
    assert len(provider.tool_calls) == 1
    assert "TARA supports local semantic search." in provider.tool_calls[0]["prompt"]
    assert "untrusted reference data" in provider.tool_calls[0]["prompt"]
    assert provider.tool_calls[0]["public_tools"] is brain.public_tools
