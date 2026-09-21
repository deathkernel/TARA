import math

import pytest

from src.schedulers import CosineAnnealing


def test_cosine_schedule_starts_at_initial_lr_and_ends_at_min_lr():
    scheduler = CosineAnnealing(0.1, total_steps=10, min_lr=0.01)
    assert math.isclose(scheduler.get_lr(0), 0.1)
    assert math.isclose(scheduler.get_lr(10), 0.01)


def test_cosine_schedule_is_non_increasing():
    scheduler = CosineAnnealing(0.1, total_steps=10, min_lr=0.01)
    rates = [scheduler.get_lr(step) for step in range(11)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))


def test_cosine_schedule_clamps_after_total_steps():
    scheduler = CosineAnnealing(0.1, total_steps=10, min_lr=0.01)
    assert scheduler.get_lr(20) == scheduler.get_lr(10)


def test_cosine_schedule_validates_arguments():
    with pytest.raises(ValueError):
        CosineAnnealing(0.0, total_steps=10)
    with pytest.raises(ValueError):
        CosineAnnealing(0.1, total_steps=0)
    with pytest.raises(ValueError):
        CosineAnnealing(0.1, total_steps=10, min_lr=0.2)
    with pytest.raises(ValueError):
        CosineAnnealing(0.1, total_steps=10).get_lr(-1)
