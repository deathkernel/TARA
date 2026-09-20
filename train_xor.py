"""Train TARA's autograd MLP on the XOR truth table."""

from src.mlp import MLP
from src.training import SGD, train_step


X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]

model = MLP(2, [3, 3, 1], seed=7)
optimizer = SGD(model.parameters(), learning_rate=0.05)
steps = 3000

for step in range(steps):
    loss = train_step(model, optimizer, X, Y)
    if step % 300 == 0 or step == steps - 1:
        print(f"step={step:4d} loss={loss:.10f}")

print("\nXOR predictions:")
for x, y in zip(X, Y):
    prediction = model.forward(x).data
    print(f"{x} -> target={y:.0f}, prediction={prediction:.10f}")
