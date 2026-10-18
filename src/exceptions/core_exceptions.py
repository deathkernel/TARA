"""Exceptions raised by TARA core lifecycle and orchestration."""

from src.exceptions.base import TaraError


class CoreError(TaraError):
    """Base class for core lifecycle and orchestration failures."""
