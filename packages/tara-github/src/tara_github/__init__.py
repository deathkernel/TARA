"""Standalone GitHub client and TARA adapter."""

from .adapter import GITHUB_TOOL_NAMES, register_github_tools
from .client import GitHubAPIError, GitHubClient

__all__ = [
    "GITHUB_TOOL_NAMES",
    "GitHubAPIError",
    "GitHubClient",
    "register_github_tools",
]
