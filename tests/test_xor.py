from src.mlp import MLP
from src.training import SGD, train_step


X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]


def test_xor_training_converges():
    model = MLP(2, [3, 3, 1], seed=7)
    optimizer = SGD(model.parameters(), learning_rate=0.05)

    initial_loss = train_step(model, optimizer, X, Y)
    for _ in range(2999):
        train_step(model, optimizer, X, Y)

    final_predictions = [model.forward(x).data for x in X]
    final_loss = sum((prediction - target) ** 2 for prediction, target in zip(final_predictions, Y)) / len(Y)

    assert final_loss < initial_loss
    assert final_loss < 1e-6
    assert abs(final_predictions[0] - 0.0) < 1e-3
    assert abs(final_predictions[1] - 1.0) < 1e-3
    assert abs(final_predictions[2] - 1.0) < 1e-3
    assert abs(final_predictions[3] - 0.0) < 1e-3
