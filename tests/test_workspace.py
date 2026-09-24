from src.tara_mind.cognition.global_workspace import GlobalWorkspace, WorkspaceItem


def test_workspace_is_bounded_and_keeps_high_priority_items():
    workspace = GlobalWorkspace(capacity=2)
    workspace.broadcast(WorkspaceItem("low", "a", 0.1))
    workspace.broadcast(WorkspaceItem("high", "b", 0.9))
    workspace.broadcast(WorkspaceItem("mid", "c", 0.5))
    assert len(workspace.items) == 2
    assert workspace.items[0].content == "high"
