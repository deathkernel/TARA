"""Exceptions raised by TARA training and checkpoint workflows."""

from src.exceptions.base import TaraError


class TrainingError(TaraError):
    """Base class for training pipeline failures."""


class DatasetError(TrainingError):
    """Training data is missing, malformed, or unusable."""


class CheckpointError(TrainingError):
    """A training checkpoint cannot be read, validated, or restored."""
