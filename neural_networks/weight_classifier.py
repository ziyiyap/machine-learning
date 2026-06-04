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
        index = np.argmax(y_actual, axis=1, keepdims=False)
        y_pred_one_hot = y_pred[np.arange(len(index)), index]
        return - np.sum(np.log(y_pred_one_hot), axis=0)

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, y_pred):
        c = np.max(y_pred, axis=1, keepdims=True)
        exp = np.exp(y_pred - c)
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def compute_weights(self, X, atv, Z, y_actual, y_pred, W_input, W_output):
        #calc derivative
        d_relu = (Z > 0).astype(float)
        dW_input = None
        db_input = np.sum(((y_pred-y_actual) @ W_output) * d_relu,axis=0,keepdims=True).T
        dW_o =  (y_pred-y_actual).T @ atv
        db_o = np.sum(y_pred- y_actual, axis=0, keepdims=True).T

        return dW_input, db_input, dW_o, db_o

    def train(self, X, y_actual):
        W_input = np.random.normal(loc=0, scale = 1/np.sqrt(self.hidden), size=(self.hidden, self.input)) # number of inputs = 2, each row = for each node
        b_input =  np.zeros((self.hidden, 1))

        W_output = np.random.normal(loc=0, scale = 1/np.sqrt(self.hidden), size=(self.output, self.hidden))
        b_output = np.zeros((self.output, 1))
        ### Input weights Ex: w12 means first input node to second hidden node
        # [[w11 w12]
        #  [w21 w22]
        #   ... 
        #  [wn1 wn2]]
        # Weights to sum to first hidden node == first column, Weights to sum to second hidden node== column 2, Number of inputs = 2 
        # ###

        ###
        # Output weights (i,j), where i = hidden node, j = output node
        # [[w(1,1), w(2,1)]
        #  [w(1,2), w(2,2)]
        #  [w(1,3), w(2,3)]]
        # ###
        for epoch in range(1,self.epochs+1):

            ###
            # Shape of X:
            # [[Height1 Weight1]
            #  [Height2 Weight2]
            #  [...     ...]
            #  [Height(n) Weight(n)]]###


            Z_input = X @ W_input.T + b_input.T
            ###
            # Shape of Z:
            # (i, j), where i = sample, j = hidden node
            #
            # [[z(1,1), z(1, 2)],
            #  [z(2,1), z(2,2)]]
            # ###
            atv = self.relu(Z_input)
            y_pred = atv @ W_output.T + b_output.T

            y_prob = self.softmax(y_pred)

            ###
            # Shape y_prob/y_pred:
            # [[O1, O2, O3]
            #  [O1, O2, O3]
            #  [ ... ]
            #  [O1, O2, O3]]
            # 
            # Each row == each sample, 3 Output nodes.
            # ###

            dW_input, db_input, dW_output, db_output = self.compute_weights(X, atv, Z_input, y_actual, y_prob, W_input, W_output)

            b_input -= self.lr * db_input
            b_output -= self.lr * db_output
            W_output -= self.lr * dW_output
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} Loss: {self.loss(y_actual, y_prob)}")




        return W_input, b_input, W_output, b_output

X_train_normalized, mean ,std = normalization(X_train)

NN = NeuralNetwork(2,3, 2)
W_input, b_input, W_output, b_output = NN.train(X_train_normalized, y_train)

y_pred = NN.predict(X_test, W_input, b_input, W_output, b_output)