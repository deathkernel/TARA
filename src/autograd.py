"""TARA automatic differentiation core.

Research reference: Karpathy's micrograd (conceptual reference only).
TARA implements its own small scalar reverse-mode autodiff engine.
"""

import math


class Value:
    """A scalar value tracked in a computation graph."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("exponent must be int or float")
        out = Value(self.data**exponent, (self,), f"**{exponent}")
        def _backward():
            self.grad += exponent * self.data ** (exponent - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        """Exponential with its local derivative."""
        value = math.exp(self.data)
        out = Value(value, (self,), "exp")
        def _backward():
            self.grad += value * out.grad
        out._backward = _backward
        return out

    def log(self):
        """Natural logarithm with its local derivative."""
        if self.data <= 0.0:
            raise ValueError("log requires a positive value")
        out = Value(math.log(self.data), (self,), "log")
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "ReLU")
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        """Hyperbolic tangent with its exact local derivative."""
        value = math.tanh(self.data)
        out = Value(value, (self,), "tanh")
        def _backward():
            self.grad += (1.0 - value * value) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """Run reverse-mode autodiff from this scalar output."""
        # Use an iterative graph traversal to avoid Python recursion overhead
        # and recursion-depth failures on larger computation graphs.
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if node in visited and not expanded:
                continue
            if expanded:
                topo.append(node)
                continue
            visited.add(node)
            stack.append((node, True))
            for child in node._prev:
                if child not in visited:
                    stack.append((child, False))

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
