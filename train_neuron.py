"""Minimal learning experiment: fit y = 2x + 1 with one ReLU neuron."""

from src.neuron import Neuron


X = [-1.0, 0.0, 1.0, 2.0, 3.0]
Y = [0.0, 1.0, 3.0, 5.0, 7.0]

neuron = Neuron(weight=0.0, bias=0.0)
learning_rate = 0.05

for step in range(200):
    losses = []

    for x, target in zip(X, Y):
        prediction = neuron.forward(x)
        loss = (prediction - target) ** 2
        losses.append(loss)

    total_loss = losses[0]
    for loss in losses[1:]:
        total_loss = total_loss + loss

    total_loss = total_loss * (1.0 / len(losses))
    total_loss.backward()

    for parameter in neuron.parameters():
        parameter.data -= learning_rate * parameter.grad
        parameter.grad = 0.0

    if step % 20 == 0 or step == 199:
        print(f"step={step:03d} loss={total_loss.data:.6f} w={neuron.w.data:.6f} b={neuron.b.data:.6f}")

print("\nPredictions:")
for x in X:
    print(f"x={x:>4.1f} -> {neuron.forward(x).data:.4f}")
