"""Standalone GitHub client and TARA adapter."""

from .adapter import GITHUB_TOOL_NAMES, connect_github_to_brain, register_github_tools
from .client import GitHubAPIError, GitHubClient

__all__ = [
    "GITHUB_TOOL_NAMES",
    "GitHubAPIError",
    "GitHubClient",
    "connect_github_to_brain",
    "register_github_tools",
]
