import numpy as np
import matplotlib.pyplot as plt
# Synthetic data
np.random.seed(42)
m = 100  # examples
X = np.random.randn(m, 1)        # 1 feature
y = (X.squeeze() > 0).astype(float)  # label: 1 if positive

def sigmoid(z):
    return 1 /(np.e ** (-z) + 1)

def compute_loss(y, y_hat): # cross entropy loss
    return (1/y_hat.shape[0]) * np.sum(-(y *np.log(y_hat) + (1-y) * np.log(1-y_hat)))

def compute_gradients(X, y, y_hat): # dL/dw & dL/db

    dw =(1/y_hat.shape[0]) * X.T @ (y_hat - y)
    db =(1/y_hat.shape[0]) * np.sum((y_hat - y))
    return (dw, db)

def train(X, y, alpha=0.01, epochs=1000):
    w = np.zeros(X.shape[1])
    b = 0.0
    for epoch in range(epochs):
        z = X @ w + b
        y_pred = sigmoid(z)
        #compute loss
        loss = compute_loss(y, y_pred)
        gradients = compute_gradients(X, y, y_pred)
        w -= alpha * gradients[0]
        b -= alpha * gradients[1]
    return w,b

w, b = train(X,y)

X_sorted = np.sort(X, axis=0)
z_sorted = X_sorted @ w + b
y_sorted = sigmoid(z_sorted)

plt.scatter(X, y, color='red')
plt.plot(X_sorted, y_sorted)
plt.show()