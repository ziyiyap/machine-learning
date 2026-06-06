import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import set_config
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

set_config(transform_output="pandas")
mnist = fetch_openml("mnist_784", cache=True)

X, y = mnist.data, mnist.target

def normalize_mnist(X):
    return X/255.0

X_arr, y_arr = torch.from_numpy(X.to_numpy()), torch.from_numpy(y.to_numpy().astype("int")).reshape(-1,1)

# X_arr is in shape (70000, 784), 70000 samples and 784 columns (features)
# y_arr is in shape (70000, ), 70000 samples and a number from 0-9

y_actual = y_arr.flatten()


# shuffle & split train/test, 80/20

X_train, X_test, y_train, y_test = train_test_split(X_arr, y_actual, test_size=0.2, random_state=42)

class TorchNeuralNetwork(nn.Module):
    def __init__(self, lr = 0.01, epochs = 10):
        super().__init__()
        self.hidden = nn.Linear(28*28, 128)
        self.output = nn.Linear(128, 10)
        self.lr = lr
        self.epochs = epochs

    def forward(self, X):
        Z = self.hidden(X)
        atv = F.relu(Z)
        logit = self.output(atv)
        return logit
    
    def fit(self, X, y_actual):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        dataset = TensorDataset(X, y_actual) #combines the X and y into a tuple (image, label) (1 sample)
        loader = DataLoader(dataset, batch_size=64, shuffle=True) #multiple samples
        for epoch in range(self.epochs):

            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
                logits = self.forward(X_batch)
                optimizer.zero_grad()
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            print(f"Epoch: {epoch} Loss: {loss.item()}")

    def predict(self, X):
        with torch.no_grad():
            logits = self(X.to("cuda"))
            y_pred = torch.argmax(logits, dim=1)
            return y_pred

NN = TorchNeuralNetwork().to("cuda")
NN.fit(normalize_mnist(X_train), y_train)

y_pred = NN.predict(normalize_mnist(X_test))



# visualisation
for start in range(0,len(X_test[:50]),10):
    plt.figure(figsize=(15,4))
    for i in range(10):
        idx = start + i
        if idx >= len(X_test):
            break
        img = X_test[idx].reshape((28,28))
        plt.subplot(2, 5, i+1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"Predicted: {y_pred[idx]}")
    plt.show()
    plt.close()

canvas= np.zeros((28,28))
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

ax_predict = plt.axes([0.3, 0.05, 0.2, 0.075])
ax_clear   = plt.axes([0.55, 0.05, 0.2, 0.075])

btn_predict = Button(ax_predict, 'Predict')
btn_clear   = Button(ax_clear, 'Clear')

img = ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
drawing = [False]
last_pos = [None]
def on_press(event):
    drawing[0] = True
    if event.xdata is None:
        return
    y, x = int(round(event.ydata)), int(round(event.xdata))
    radius = 2
    for dy in range(-radius, radius):
        for dx in range(-radius, radius):
            nx, ny = x + dx, y + dy
            if 0 <= nx <= 28 and 0 <= ny <= 28:
                canvas[ny, nx] = 1
                img.set_data(canvas)
                fig.canvas.draw()

def on_move(event):
    if not drawing[0] or event.xdata is None:
        return
    x, y = int(round(event.xdata)), int(round(event.ydata))
    
    if last_pos[0] is not None:
        x0, y0 = last_pos[0]
        # draw all pixels between last and current
        steps = max(abs(x-x0), abs(y-y0)) + 1
        for t in np.linspace(0, 1, steps):
            ix = int(round(x0 + t*(x-x0)))
            iy = int(round(y0 + t*(y-y0)))
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    nx, ny = ix+dx, iy+dy
                    if 0 <= nx < 28 and 0 <= ny < 28:
                        canvas[ny, nx] = 1
    
    last_pos[0] = (x, y)
    img.set_data(canvas)
    fig.canvas.draw()

def on_release(event):
    drawing[0] = False
    last_pos[0] = None

def on_clear(event):
    canvas[:] = 0        # reset in place
    img.set_data(canvas)
    fig.canvas.draw()

def on_predict(event):
    flat = torch.from_numpy(canvas.flatten().reshape(1, 784)).float()
    result = NN.predict(flat)
    ax.set_title(f"Predicted: {result[0]}")
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_move)
fig.canvas.mpl_connect('button_release_event', on_release)

btn_predict.on_clicked(on_predict)
btn_clear.on_clicked(on_clear)

plt.axis("off")
plt.show()