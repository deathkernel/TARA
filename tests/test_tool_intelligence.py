from src.tool_intelligence import (
    CapabilityRegistry, ToolCapability, ToolIntelligence, ToolAttempt,
)


def test_contextual_selection_uses_capability_and_preconditions():
    intelligence = ToolIntelligence(CapabilityRegistry())
    intelligence.register(ToolCapability("slow", "slow browser", ("browse",), latency=5, reliability=.9))
    intelligence.register(ToolCapability("fast", "fast browser", ("browse",), latency=1, reliability=.95))
    result = intelligence.choose("find page", ["browse"], facts={})
    assert result.selected == "fast"


def test_preconditions_can_block_a_tool():
    intelligence = ToolIntelligence()
    intelligence.register(ToolCapability("write", "write file", ("write",), preconditions=("workspace_open",)))
    result = intelligence.choose("save", ["write"], facts={"workspace_open": False})
    assert result.selected is None


def test_failure_diagnosis_selects_declared_fallback():
    intelligence = ToolIntelligence()
    intelligence.register(ToolCapability("primary", "primary", ("read",), fallback_tools=("backup",)))
    intelligence.register(ToolCapability("backup", "backup", ("read",)))
    result = intelligence.recover(ToolAttempt("primary", False, "timeout"), "read", ["read"], facts={})
    assert result.category == "transient"
    assert result.fallback == "backup"
