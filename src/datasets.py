"""Small deterministic datasets for TARA experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    """A minimal supervised dataset of feature vectors and scalar targets."""

    X: list
    Y: list

    def __post_init__(self):
        if len(self.X) != len(self.Y):
            raise ValueError("X and Y must contain the same number of samples")
        if not self.X:
            raise ValueError("dataset must contain at least one sample")

    def __len__(self):
        return len(self.X)


def xor_dataset():
    return Dataset(
        X=[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        Y=[0.0, 1.0, 1.0, 0.0],
    )


def linear_regression_dataset():
    """Deterministic y = 2x + 1 samples for a regression smoke test."""
    X = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
    Y = [-3.0, -1.0, 1.0, 3.0, 5.0]
    return Dataset(X=X, Y=Y)
