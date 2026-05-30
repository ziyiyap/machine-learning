import numpy as np
import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
os.chdir(BASE_DIR)

df = pd.read_csv(r"data.csv").drop(columns=['bmi', 'class_id'])

X = df.drop(columns=["class_label"]).to_numpy()
y = df["class_label"].to_numpy().reshape(-1,1)

indices = np.arange(y.shape[0])
classes = ["Underweight", "Normal", "Overweight"]

class_idx = {c: i for i, c in enumerate(classes)}
idx_class = {i:c for i,c in enumerate(classes)}

y_idx = np.array([class_idx[label] for label in y.flatten()])

y_actual = np.zeros((len(X), len(classes)))
y_actual[indices, y_idx] = 1
def normalization(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    return (X - X_mean) / X_std, X_mean, X_std

np.random.seed(42)
np.random.shuffle(indices)

split = int(len(y) * 0.8)

X_shuffled, y_shuffled = X[indices], y_actual[indices]

X_train, X_test = X_shuffled[:split], X_shuffled[split:]
y_train , y_test = y_shuffled[:split], y_shuffled[split:]

class NeuralNetwork:
    def __init__(self, i, o, hidden, lr = 0.001, epochs = 1000):
        self.input = i
        self.output = o
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
    def predict(self, X, W_input, b_input, W_output, b_output):
        Z_input = np.dot(W_input, X.T) + b_input

        atv = self.relu(Z_input)
        y_pred = np.dot(W_output, atv) + b_output

        y_prob = self.softmax(y_pred).T #transpose back to [prob prob prob]
        return y_prob
    
    def loss(self, y_actual, y_pred):
        return - np.sum(y_actual * np.log(y_pred), axis=0)

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, y_pred):
        c = np.max(y_pred, axis=0, keepdims=True)
        exp = np.exp(y_pred - c)
        return exp / np.sum(exp, axis=0, keepdims=True)
    
    def compute_weights(self, X, Z, y_actual, y_pred, W_output):
        #calc derivativev
        d_relu = (Z > 0).astype(float)
        dW_input = None
        db_input = None
        dW_o = None
        db_o = None
        return dW_input, db_input, dW_o, db_o

    def train(self, X, y_actual):
        W_input = np.random.normal(loc=0, scale = 1/np.sqrt(self.hidden), size=(self.hidden, self.input)) # number of inputs = 2, each row = for each node
        b_input =  np.zeros((self.hidden, 1))

        W_output = np.random.normal(loc=0, scale = 1/np.sqrt(self.hidden), size=(self.output, self.hidden))
        b_output = np.zeros((self.output, 1))
        ###
        # [[w11 w12]
        #  [w21 w22]
        #   ... 
        #  [wn1 wn2]]
        # Number of inputs = 2 
        # ###
        for epoch in range(self.epochs):
            Z_input = np.dot(W_input, X.T) + b_input

            atv = self.relu(Z_input)
            y_pred = np.dot(W_output, atv) + b_output

            y_prob = self.softmax(y_pred).T #transpose back to [prob prob prob]

            dW_input, db_input, dW_output, db_output = self.compute_weights(X, Z_input, y_actual, y_prob, W_output)

            W_input -= self.lr * dW_input
            b_input -= self.lr * db_input
            W_output -= self.lr * dW_output.T
            b_output -= self.lr * db_output.T

            if epoch % 100 == 0:
                print(f"Epoch: {epoch} Loss: {self.loss(y_actual, y_prob)}")



        return W_input, b_input, W_output, b_output

X_train_normalized, mean ,std = normalization(X_train)

NN = NeuralNetwork(2,3, 10)
W_input, b_input, W_output, b_output = NN.train(X_train_normalized, y_train)

y_pred = NN.predict(X_test, W_input, b_input, W_output, b_output)

print(y_pred)
print(y_test)