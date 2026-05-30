import pandas as pd
import numpy as np
import os 
from pathlib import Path
np.set_printoptions(precision=7, suppress=True) #remove e-01..
BASE_DIR = Path(__file__).parent
os.chdir(BASE_DIR)

data = pd.read_csv(r"student_data.csv")
y = data["student_category"].to_numpy()
X = data.drop(columns=["student_category"]).to_numpy()

classes = ["Excellent Student", "Average Student", "Cooked Student"]
def normalize(arr):
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return (arr - mean) / std, mean, std

def pred_normalize(X, mean, std):
    return (X-mean) / std

X_normal, mean, std = normalize(X)

class_idx = {c: i for i, c in enumerate(classes)}
y_idx = np.array([
    class_idx[label] for label in y
    ])

idx_class = {v: k for k, v in class_idx.items()} #inverse of class_idx

y_actual = np.zeros((len(y), len(classes))) # (100, 3)

indices = np.arange(len(y)) #indices of the length of y (100)

y_actual[indices, y_idx] = 1

np.random.seed(42) 
np.random.shuffle(indices)

X_shuffled, y_shuffled = X[indices], y_actual[indices]

split = int(len(X_shuffled) * 0.8)
X_train_raw, X_test_raw = X_shuffled[:split], X_shuffled[split:]
y_train, y_test = y_shuffled[:split], y_shuffled[split:]

#normalize AFTER splitting to prevent data leakage
X_train, mean, std = normalize(X_train_raw)
X_test = pred_normalize(X_test_raw, mean, std)

def cross_entropy_loss(y_actual, y_pred): #loss function
    return -np.mean(np.sum(y_actual * np.log(y_pred + 1e-9), axis=1))

def softmax(Z):
    c = np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z - c)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def compute_weights(X, y_actual, y_pred):
    dW = (X.T @ (y_pred - y_actual)) / X.shape[0]
    db = np.sum((y_pred - y_actual), axis = 0, keepdims=True) / y_pred.shape[0]
    return dW, db

def train(X_train,y_actual, lr = 0.1, epochs = 1000):
    W = np.zeros((X_train.shape[1], y_actual.shape[1])) # (linear models, weights of linear models)
    b = np.zeros(y_actual.shape[1])
    ###
    # [[w11, w12, w13]  z1
    #  [w21, w22, w23]  z2
    #  [w31, w32, w33]] z3
    # ###
    for epoch in range(1, epochs+1):
        Z = X_train @ W + b # Z = {z1, z2, z3}
        y_pred = softmax(Z)
        dW, db = compute_weights(X_train, y_actual, y_pred)
        
        W -= lr * dW
        b -= lr * db.squeeze()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} Error: {round(cross_entropy_loss(y_actual, y_pred), 4)}")
    return W, b

def predict(X, W, b):
    Z = X @ W + b
    return softmax(Z)

def predict_class(X, W, b):
    Z = X @ W + b
    y_pred = softmax(Z)
    max_indices = np.argmax(y_pred, axis=1)
    pred_labels = np.array([idx_class[i] for i in max_indices])
    return pred_labels

W, b = train(X_train, y_train)


print(predict_class(np.array([pred_normalize([5.8, 0.99, 8], mean, std)]), W, b))
print(predict(np.array([pred_normalize([5.8, 0.99, 8], mean, std)]), W, b))