import numpy as np

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Random seed
np.random.seed(0)

# Architecture:
# 2 inputs -> 2 hidden neurons -> 1 output neuron

W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))

W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


learning_rate = 0.1

for epoch in range(10000):

    # ---- Forward pass ----

    z1 = X @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2
    y_pred = sigmoid(z2)

    # ---- Loss ----

    loss = np.mean((y - y_pred) ** 2)

    # ---- Backprop ----

    # Output layer
    d_output = 2 * (y_pred - y) * sigmoid_derivative(y_pred)

    dW2 = a1.T @ d_output
    db2 = np.sum(d_output, axis=0, keepdims=True)

    # Hidden layer
    d_hidden = (d_output @ W2.T) * sigmoid_derivative(a1)

    dW1 = X.T @ d_hidden
    db1 = np.sum(d_hidden, axis=0, keepdims=True)

    # ---- Gradient descent ----

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final predictions
print("\nPredictions:")
print(y_pred)