"""Small XOR experiment for TARA's MLP."""

from src.mlp import MLP

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]

model = MLP(2, [3, 3, 1], seed=7)

for step in range(2000):
    model.zero_grad()
    loss = None
    for x, y in zip(X, Y):
        error = model.forward(x) - y
        sample_loss = error ** 2
        loss = sample_loss if loss is None else loss + sample_loss
    loss = loss / len(X)
    loss.backward()

    for parameter in model.parameters():
        parameter.data -= 0.05 * parameter.grad

    if step % 200 == 0:
        print(f"step={step:4d} loss={loss.data:.6f}")

print("predictions:")
for x, y in zip(X, Y):
    print(x, "target=", y, "prediction=", model.forward(x).data)
