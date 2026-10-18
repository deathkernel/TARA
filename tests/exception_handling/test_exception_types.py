"""Tests for the module-scoped exception taxonomy."""

from src.exceptions import (
    AutogradError, BrainError, CoreError, MemorySubsystemError,
    TrainingError, ToolExecutionError,
)
from src.exceptions.base import TaraError
from src.exceptions.memory_exceptions import MemoryReadError, MemoryWriteError
from src.exceptions.training_exceptions import CheckpointError, DatasetError


def test_module_errors_share_tara_base() -> None:
    for error_type in (AutogradError, BrainError, CoreError, MemorySubsystemError, TrainingError, ToolExecutionError):
        assert issubclass(error_type, TaraError)


def test_specific_errors_inherit_module_error() -> None:
    assert issubclass(DatasetError, TrainingError)
    assert issubclass(CheckpointError, TrainingError)
    assert issubclass(MemoryReadError, MemorySubsystemError)
    assert issubclass(MemoryWriteError, MemorySubsystemError)


def test_errors_preserve_normal_exception_message() -> None:
    error = CheckpointError("checkpoint is corrupt")
    assert str(error) == "checkpoint is corrupt"
