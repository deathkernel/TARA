from src.datasets import Dataset, linear_regression_dataset, xor_dataset


def test_dataset_validates_lengths():
    dataset = Dataset(X=[[1.0]], Y=[2.0])
    assert len(dataset) == 1


def test_dataset_rejects_mismatched_lengths():
    try:
        Dataset(X=[[1.0]], Y=[])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched X/Y should raise ValueError")


def test_xor_dataset():
    dataset = xor_dataset()
    assert len(dataset) == 4
    assert dataset.Y == [0.0, 1.0, 1.0, 0.0]


def test_linear_regression_dataset():
    dataset = linear_regression_dataset()
    assert len(dataset) == 5
    assert dataset.Y[0] == 2.0 * dataset.X[0][0] + 1.0
    assert dataset.Y[-1] == 2.0 * dataset.X[-1][0] + 1.0
