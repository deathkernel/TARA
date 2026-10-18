"""Exceptions raised by TARA's custom automatic differentiation code."""

from src.exceptions.base import TaraError


class AutogradError(TaraError):
    """Base class for automatic differentiation failures."""
