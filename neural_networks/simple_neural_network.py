import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

# A neural network that learns y = x^2

X = np.linspace(-12, 12, num=60)
y_actual = X ** 2
indices = np.arange(len(X))

np.random.seed(42)
np.random.shuffle(indices)

X_shuffled, y_shuffled = X[indices].reshape(-1,1), y_actual[indices].reshape(-1,1)

relu = lambda Z : np.maximum(0,Z)

def backprop(X, Z, y_actual, y_pred, W_output):
    d_relu = (Z > 0).astype(float)
    dW_i = (-2 *  (((y_actual - y_pred) @ W_output.T).T * d_relu) @ X) 
    db_i = (-2 * np.sum(((y_actual - y_pred) @ W_output.T).T * d_relu, axis =1, keepdims=True)) 
    dW_o = (-2 * relu(Z) @ (y_actual - y_pred) )
    db_o = (-2 * np.sum(y_actual - y_pred, axis=0))
    return dW_i, db_i, dW_o, db_o

def SSE(y_actual, y_pred):
    return np.sum((y_actual - y_pred) ** 2, axis=0)

def train(X, y_actual, lr = 0.00001, epochs = 16000):

    hidden = 100

    W_input = np.random.normal(loc=0, scale= 1, size = (hidden,1))
    B_input = np.zeros((hidden,1))
    W_output = np.random.normal(loc=0, scale=1/np.sqrt(hidden), size = (hidden,1))
    b_output = 0.0
    ###
    # [[w1]
    #  [w2]
    #  ...
    #  [w8]]###
    for epoch in range(1, epochs+1):
        Z = W_input @ X.T  + B_input # transpose to allow matrix multiplication
        Z_act = relu(Z)

        y_pred = Z_act.T @ W_output  + b_output
        dW_i, db_i, dW_o, db_o = backprop(X,Z, y_actual, y_pred, W_output)

        W_input -= lr * dW_i
        B_input -= lr * db_i

        W_output -= lr * dW_o
        b_output -= lr * db_o

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} SSE: {SSE(y_actual, y_pred)}")

    return W_input, B_input, W_output, b_output

def predict(X, W_input, B_input, W_output, b_output):
    return (relu(W_input @ X.T + B_input)).T @ W_output  + b_output



W_input, B_input, W_output, b_output = train(X_shuffled, y_shuffled)

y_pred = predict(X.reshape(-1,1), W_input, B_input, W_output, b_output)

plt.scatter(X, y_actual, color='red', alpha=0.2)
plt.plot(X, y_pred)
plt.title("x²")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
plt.close()