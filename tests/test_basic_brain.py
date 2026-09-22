from src.basic_brain import BasicTARABrain, ToolRegistry


def test_full_basic_cycle_connects_all_core_layers():
    registry = ToolRegistry()
    registry.register("add", lambda a, b: a + b)
    brain = BasicTARABrain(tool_registry=registry)

    result = brain.run_cycle(
        "calculate 2 + 3",
        goal="calculate the result",
        subtasks=["parse input", "calculate", "verify"],
        tool="add",
        tool_kwargs={"a": 2, "b": 3},
        expected=5,
        verify=True,
    )

    assert result.percept.kind == "text"
    assert result.plan.steps[0].description == "parse input"
    assert result.tool_result is not None and result.tool_result.success
    assert result.verification is not None and result.verification.passed
    assert result.reflection is not None and result.reflection.success
    assert result.learned is True
    assert len(brain.learning.all()) == 1
    assert len(brain.long_term_memory.all()) == 1


def test_failed_verification_is_not_learned():
    brain = BasicTARABrain()
    result = brain.run_cycle(
        "bad result",
        goal="produce five",
        subtasks=["produce result"],
        expected=5,
        verify=True,
    )

    assert result.verification is not None
    assert result.verification.passed is False
    assert result.learned is False
    assert brain.learning.all() == ()


def test_hypothesis_experiment_and_knowledge_graph_are_connected_primitives():
    brain = BasicTARABrain()
    percept = brain.observe("sorting is slow")
    hypotheses = brain.hypotheses.propose("why is sorting slow?", [percept])
    assert len(hypotheses) == 2

    experiment = brain.experiments.run(hypotheses[0], lambda hypothesis: True)
    assert experiment.supported is True

    edge = brain.knowledge.add("sorting", "has_property", "runtime")
    assert edge in brain.knowledge.related("sorting")


def test_research_and_multimodal_boundaries_are_safe_by_default():
    brain = BasicTARABrain()
    assert brain.research.search("anything") == ()
    packed = brain.multimodal.pack(text=["hello"], images=["image"], audio=["audio"], files=["file"])
    assert packed.text == ("hello",)
    assert packed.images == ("image",)
    assert packed.audio == ("audio",)
    assert packed.files == ("file",)


def test_tool_registry_is_explicit_allow_list():
    brain = BasicTARABrain()
    assert brain.act("shell").success is False
    brain.tools.register("echo", lambda value: value)
    assert brain.act("echo", value="ok").output == "ok"
