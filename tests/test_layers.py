import numpy as np
import pytest

from src.layers import DenseLayer


def test_dense_layer_rejects_non_positive_dimensions():
    with pytest.raises(ValueError):
        DenseLayer(0, 2)
    with pytest.raises(ValueError):
        DenseLayer(2, 0)
    with pytest.raises(ValueError):
        DenseLayer(-1, 2)


def test_dense_layer_forward_validates_batch_shape():
    layer = DenseLayer(3, 2)
    with pytest.raises(ValueError):
        layer.forward(np.ones(3))
    with pytest.raises(ValueError):
        layer.forward(np.ones((4, 2)))


def test_dense_layer_backward_validates_learning_rate_and_gradient_shape():
    layer = DenseLayer(3, 2)
    layer.forward(np.ones((4, 3)))

    with pytest.raises(ValueError):
        layer.backward(np.ones((4, 2)), 0.0)
    with pytest.raises(ValueError):
        layer.backward(np.ones((4, 2)), -0.1)
    with pytest.raises(ValueError):
        layer.backward(np.ones((4, 3)), 0.1)


def test_dense_layer_forward_backward_shapes():
    layer = DenseLayer(3, 2)
    output = layer.forward(np.ones((4, 3)))
    grad_input = layer.backward(np.ones((4, 2)), 0.01)
    assert output.shape == (4, 2)
    assert grad_input.shape == (4, 3)
