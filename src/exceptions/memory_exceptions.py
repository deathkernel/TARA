"""Exceptions raised by memory and retrieval components."""

from src.exceptions.base import TaraError


class MemorySubsystemError(TaraError):
    """Base class for memory storage and retrieval failures."""


class MemoryReadError(MemorySubsystemError):
    """Memory could not be retrieved or decoded."""


class MemoryWriteError(MemorySubsystemError):
    """Memory could not be persisted."""
