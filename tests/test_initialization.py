import math
import random

import pytest

from src.initialization import xavier_uniform, xavier_uniform_limit
from src.mlp import MLP
from src.transformer import Linear


def test_xavier_uniform_limit_matches_formula():
    assert math.isclose(
        xavier_uniform_limit(2, 4),
        math.sqrt(6.0 / 6.0),
        rel_tol=1e-12,
    )


def test_xavier_uniform_is_deterministic_and_bounded():
    rng_a = random.Random(17)
    rng_b = random.Random(17)
    limit = xavier_uniform_limit(3, 5)
    values_a = [xavier_uniform(rng_a, 3, 5) for _ in range(20)]
    values_b = [xavier_uniform(rng_b, 3, 5) for _ in range(20)]
    assert values_a == values_b
    assert all(-limit <= value <= limit for value in values_a)


def test_xavier_uniform_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        xavier_uniform_limit(0, 4)
    with pytest.raises(ValueError):
        xavier_uniform_limit(4, -1)


def test_linear_uses_fan_in_and_fan_out_for_initialization():
    linear = Linear(3, 5, random.Random(9))
    limit = xavier_uniform_limit(3, 5)
    weights = [weight.data for row in linear.w for weight in row]
    assert all(-limit <= weight <= limit for weight in weights)


def test_mlp_xavier_initialization_is_deterministic():
    first = MLP(2, [3, 2, 1], seed=23)
    second = MLP(2, [3, 2, 1], seed=23)
    assert [p.data for p in first.parameters()] == [p.data for p in second.parameters()]
