"""Loss functions for TARA's scalar-autograd training core."""

from src.autograd import Value


class MSELoss:
    """Mean squared error for scalar predictions."""

    def __call__(self, prediction, target):
        prediction = prediction if isinstance(prediction, Value) else Value(prediction)
        return (prediction - target) ** 2


def mean(values):
    """Return the arithmetic mean of scalar Value objects."""
    values = list(values)
    if not values:
        raise ValueError("mean() requires at least one value")
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total / len(values)
