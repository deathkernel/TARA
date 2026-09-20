"""Train TARA's scalar-autograd MLP on the XOR truth table."""

from src.mlp import MLP

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]

model = MLP(2, [3, 3, 1], seed=7)
learning_rate = 0.05
steps = 3000

for step in range(steps):
    model.zero_grad()
    losses = []
    for x, y in zip(X, Y):
        prediction = model.forward(x)
        losses.append((prediction - y) ** 2)

    total_loss = losses[0]
    for loss in losses[1:]:
        total_loss = total_loss + loss
    total_loss = total_loss / len(losses)
    total_loss.backward()

    for parameter in model.parameters():
        parameter.data -= learning_rate * parameter.grad

    if step % 300 == 0:
        print(f"step={step:4d} loss={total_loss.data:.6f}")

print("\nXOR predictions:")
for x, y in zip(X, Y):
    prediction = model.forward(x)
    print(f"{x} -> target={y:.0f}, prediction={prediction.data:.6f}")
