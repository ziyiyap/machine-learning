import numpy as np
import matplotlib.pyplot as plt

xlist = [n for n in range(1,11)]
X = np.array(xlist).reshape(len(xlist),1)
noise = np.random.normal(-0.1,0.1,size=X.shape)
y_actual = np.array([10 * n for n in xlist]).reshape(len(xlist),1) + noise
def add_ones(array):
    ones = np.ones((len(array),1))
    return np.hstack((ones,array))

X_ = add_ones(X)

def MSE(y_actual, y_pred):
    return np.sum((y_actual - y_pred) ** 2) / len(y_actual)

def compute_gradients(X, y_actual, y_pred):
    # w = w - alpha * dL/dw
    dw = (X.T @ (y_pred - y_actual)) /len(y_actual)

    return dw

def train(X, y_actual, alpha = 0.01, epochs = 1000):
    w = np.zeros((X.shape[1],1)) #columns (features)

    losses = []

    for epoch in range(epochs):
        y_pred = X @ w 
        loss = MSE(y_actual, y_pred)
        losses.append(loss)
        print(f"Epoch: {epoch} MSE: {loss} w = {w}")
        dw = compute_gradients(X, y_actual, y_pred)
        w -= alpha * dw
    return w, losses

w, losses = train(X_, y_actual)
y_pred = X_ @ w

weights = w.flatten().tolist()

w0, w1 = weights[0], weights[1]

plt.scatter(X, y_actual, color='red')
plt.plot(X, y_pred, color='blue')
plt.title(f"y = {w1:.2f}x + {w0:.2f}")
plt.show()
plt.close()