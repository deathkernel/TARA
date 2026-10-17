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


def register_github_tools(
    client: GitHubClient,
    *,
    application_tools: Any | None = None,
    controller: Any | None = None,
) -> tuple[str, ...]:
    """Register GitHub read actions with existing TARA tool interfaces.

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
