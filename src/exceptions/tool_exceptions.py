"""Exceptions raised while invoking or validating tools."""

from src.exceptions.base import TaraError


class ToolExecutionError(TaraError):
    """A tool failed while being validated, executed, or interpreted."""
