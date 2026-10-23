"""Stable interface expected from a TARA language provider.

Providers are synchronous and return plain text. They must enforce their own
network/request timeout and reject malformed provider responses with an exception.
TARA owns conversation context and memory; providers own model inference only.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LanguageProvider(Protocol):
    """Minimum provider contract consumed by :class:`TARABrain`."""

    @property
    def default_max_tokens(self) -> int:
        """Return the default completion-token cap for this provider profile."""
        ...

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_completion_tokens: int | None = None,
        temperature: float = 0.6,
    ) -> str:
        """Generate a single non-streaming response or raise an exception."""
        ...


__all__ = ["LanguageProvider"]
