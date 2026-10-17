"""Adapters for registering read-only GitHub actions with TARA."""
from __future__ import annotations

from typing import Any

from .client import GitHubClient


GITHUB_TOOL_NAMES = (
    "github.get_repository",
    "github.get_file",
    "github.list_issues",
    "github.list_pull_requests",
    "github.list_branches",
    "github.list_commits",
    "github.get_pull_request_diff",
)

_CAPABILITIES = {
    "github.get_repository": ("github", "read", "repository", "metadata"),
    "github.get_file": ("github", "read", "repository", "file"),
    "github.list_issues": ("github", "read", "issues"),
    "github.list_pull_requests": ("github", "read", "pull_requests"),
    "github.list_branches": ("github", "read", "branches"),
    "github.list_commits": ("github", "read", "commits"),
    "github.get_pull_request_diff": ("github", "read", "pull_requests", "diff"),
}


def register_github_tools(
    client: GitHubClient,
    *,
    application_tools: Any | None = None,
    controller: Any | None = None,
) -> tuple[str, ...]:
    """Register GitHub read actions with TARA's explicit tool interfaces.

    Both integrations are optional, but at least one must be supplied. These
    actions are GET-only and do not mutate repositories.
    """
    if application_tools is None and controller is None:
        raise ValueError("provide application_tools, controller, or both")

    actions = {
        "github.get_repository": client.get_repository,
        "github.get_file": client.get_file,
        "github.list_issues": client.list_issues,
        "github.list_pull_requests": client.list_pull_requests,
        "github.list_branches": client.list_branches,
        "github.list_commits": client.list_commits,
        "github.get_pull_request_diff": client.get_pull_request_diff,
    }
    for name, action in actions.items():
        if application_tools is not None:
            application_tools.register(name, action)
        if controller is not None:
            controller.register(name, action)
    return tuple(actions)


def connect_github_to_brain(
    brain: Any,
    client: GitHubClient,
    *,
    application_tools: Any | None = None,
    controller: Any | None = None,
) -> tuple[str, ...]:
    """Register GitHub actions and their capabilities with an existing TARABrain.

    This connects the GitHub handler to TARA's capability selector as well as
    its execution boundaries. Actual execution still goes through the host's
    explicit action executor/controller; the model cannot supply permission
    merely by mentioning a tool.
    """
    tool_intelligence = getattr(brain, "tools", None)
    if tool_intelligence is None or not callable(getattr(tool_intelligence, "register", None)):
        raise TypeError("brain must expose a register-capable tools component")

    names = register_github_tools(
        client,
        application_tools=application_tools,
        controller=controller,
    )

    # Import the TARA-specific type lazily so the standalone client remains
    # usable without importing the TARA application.
    from src.tool_intelligence import ToolCapability

    descriptions = {
        "github.get_repository": "Read public or authorized GitHub repository metadata",
        "github.get_file": "Read a UTF-8 text file from a GitHub repository",
        "github.list_issues": "List GitHub issues without pull requests",
        "github.list_pull_requests": "List GitHub pull requests",
        "github.list_branches": "List repository branches",
        "github.list_commits": "List repository commits",
        "github.get_pull_request_diff": "Read a pull request unified diff",
    }
    for name in names:
        tool_intelligence.register(
            ToolCapability(
                name=name,
                description=descriptions[name],
                capabilities=_CAPABILITIES[name],
                risk=0.0,
                latency=1.0,
                reliability=0.98,
                cost=0.1,
            )
        )
    return names
