import pytest

from src.metrics import accuracy


def test_binary_accuracy():
    assert accuracy([0.1, 0.9, 0.8, 0.2], [0.0, 1.0, 1.0, 0.0]) == 1.0


def test_accuracy_rejects_empty_input():
    with pytest.raises(ValueError):
        accuracy([], [])
