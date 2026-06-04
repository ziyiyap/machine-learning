import numpy as np
np.set_printoptions(precision=4, suppress=True)
X = np.array([
    # Fail (low across board)
    [30, 25, 20],
    [40, 35, 30],
    [20, 45, 25],
    [35, 30, 40],
    [25, 20, 35],

    # Pass (mid range)
    [60, 55, 50],
    [55, 60, 65],
    [65, 50, 60],
    [58, 62, 55],
    [70, 60, 50],

    # Distinction (high scores)
    [85, 90, 88],
    [92, 80, 85],
    [78, 95, 90],
    [88, 87, 92],
    [95, 93, 89]
])
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
y_actual = np.array([
    "Fail", "Fail", "Fail", "Fail", "Fail",
    "Pass", "Pass", "Pass", "Pass", "Pass",
    "Distinction", "Distinction", "Distinction", "Distinction", "Distinction"
])
classes = ["Fail","Pass", "Distinction"]

class_to_idx = {c: i for i, c in enumerate(classes)}

y_idx = np.array([
    class_to_idx[label] for label in y_actual
])

y_onehot = np.zeros((len(y_actual), len(classes)))
y_onehot[np.arange(len(y_actual)), y_idx] = 1

def cross_entropy_loss(y_actual, y_pred): #loss function
    return -(np.sum(y_actual * np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred)))

def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def compute_weights(X, y_actual, y_pred):
    dw = (X.T @ (y_pred - y_actual)) / X.shape[0]
    db = np.sum(y_pred - y_actual, axis=0, keepdims=True) / y_pred.shape[0]
    return dw, db

def train(X, y_actual, lr = 0.01, epochs = 1000):
    w1 = np.zeros(X.shape[1])
    w2 = np.zeros(X.shape[1])
    w3 = np.zeros(X.shape[1])
    b1 =  b2 =  b3 = 0.0
    for epoch in range(epochs):
        z1 = X @ w1 + b1
        z2 = X @ w2 + b2
        z3 = X @ w3 + b3

        Z = np.column_stack((z1, z2, z3))
        y_pred = softmax(Z)

        dw, db = compute_weights(X, y_actual, y_pred)
        w1 -= lr * dw[:, 0]
        w2 -= lr * dw[:, 1]
        w3 -= lr * dw[:, 2]

        b1 -= lr * db[0, 0]
        b2 -= lr * db[0, 1]
        b3 -= lr * db[0, 2]
    return w1, w2, w3, b1, b2, b3

def y_pred(X, w1, w2, w3, b1, b2, b3):
    z1 = X @ w1 + b1
    z2 = X @ w2 + b2
    z3 = X @ w3 + b3
    Z = np.column_stack((z1, z2, z3))
    return softmax(Z)

w1, w2, w3, b1, b2, b3 = train(X, y_onehot)


print(y_pred(X, w1, w2, w3, b1, b2, b3))
print()
print(y_onehot)

print(y_pred(np.array([1, -0.71, -0.71]), w1, w2, w3, b1, b2, b3))