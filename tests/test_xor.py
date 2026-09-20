from src.mlp import MLP


def test_xor_training_reduces_loss():
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    Y = [0.0, 1.0, 1.0, 0.0]

    model = MLP(2, [3, 3, 1], seed=7)
    learning_rate = 0.05

    def loss_value():
        losses = []
        for x, y in zip(X, Y):
            prediction = model.forward(x)
            losses.append((prediction - y) ** 2)
        total = losses[0]
        for loss in losses[1:]:
            total = total + loss
        return (total / len(losses)).data

    initial_loss = loss_value()
    for _ in range(3000):
        model.zero_grad()
        losses = []
        for x, y in zip(X, Y):
            prediction = model.forward(x)
            losses.append((prediction - y) ** 2)
        total = losses[0]
        for loss in losses[1:]:
            total = total + loss
        total = total / len(losses)
        total.backward()
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad

    assert loss_value() < initial_loss
