import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.datasets import fetch_openml
from sklearn import set_config
set_config(transform_output="pandas")
mnist = fetch_openml("mnist_784", cache=True)

X, y = mnist.data, mnist.target

def normalize_mnist(X):
    return X/255.0

X_arr, y_arr = X.to_numpy(), y.to_numpy().reshape(-1,1) 

# X_arr is in shape (70000, 784), 70000 samples and 784 columns (features)
# y_arr is in shape (70000, ), 70000 samples and a number from 0-9

indices = np.arange(y_arr.shape[0]) # or we could use X_arr[0], its the same
# one-hot y:
classes = [n for n in range(10)]

y_idx = y_arr.flatten().astype("int")

y_actual = np.zeros((y.shape[0], len(classes)))
y_actual[indices, y_idx] = 1

# shuffle & split train/test, 80/20
np.random.seed(42) #set random seed
np.random.shuffle(indices)

split = int(y_arr.shape[0] * 0.8)

X_shuffled, y_shuffled = X_arr[indices], y_actual[indices]

X_train, X_test = X_shuffled[:split], X_shuffled[split:]
y_train, y_test = y_shuffled[:split], y_shuffled[split:]


class NeuralNetwork:
    def __init__(self, i, h, o, lr = 0.01, epochs = 10):
        self.input = i
        self.hidden = h
        self.output = o
        self.lr = lr
        self.epochs = epochs

    def loss(self, y_actual, y_pred):
        max_index = cp.argmax(y_actual, axis = 1, keepdims=False)
        y_pred_one_hot = y_pred[cp.arange(len(max_index)), max_index]
        return - cp.sum(cp.log(y_pred_one_hot), axis=0) / len(y_actual)

    def relu(self, Z):
        return cp.maximum(0, Z)
    
    def softmax(self, raw_logits):
        c = cp.max(raw_logits, axis=1, keepdims= True)
        exp = cp.exp(raw_logits - c)
        return exp / cp.sum(exp, axis=1, keepdims=True)
    
    def compute_weights(self, X, atv, Z, y_actual, y_pred, W_output):
        d_relu = (Z >0).astype(float)

        dW_input = cp.dot(X.T, (cp.dot((y_pred - y_actual), W_output.T) * d_relu))
        db_input = cp.sum(cp.dot((y_pred - y_actual), W_output.T) * d_relu, axis=0, keepdims=True)
        dW_output = cp.dot(atv.T, (y_pred-y_actual))
        db_output = cp.sum((y_pred - y_actual), axis=0, keepdims=True)
        return dW_input, db_input, dW_output, db_output

    def train(self, X, y_actual):
        W_input = cp.random.normal(loc=0, scale=1/cp.sqrt(self.input), size=(self.input, self.hidden))
        b_input = cp.zeros((1, self.hidden))

        W_output = cp.random.normal(loc=0, scale=1/cp.sqrt(self.hidden), size=(self.hidden, self.output))
        b_output = cp.zeros((1, self.output))

        X_batches, y_batches = cp.array_split(X, int(X.shape[0] / 64)), cp.array_split(y_actual, int(y_actual.shape[0] / 64))
        for epoch in range(self.epochs):
            for X_batch, y_batch in zip(X_batches, y_batches):
                #Forward pass

                Z = cp.dot(X_batch, W_input) + b_input  # Each row == 1 sample, [Z_node1, Z_node2]

                atv = self.relu(Z)

                raw_logits = cp.dot(atv, W_output) + b_output

                y_pred = self.softmax(raw_logits)

                #backprop

                dW_input, db_input, dW_output, db_output = self.compute_weights(X_batch, atv, Z, y_batch, y_pred, W_output)
                
                W_input -= self.lr * dW_input
                b_input -= self.lr * db_input
                W_output -= self.lr * dW_output
                b_output -= self.lr * db_output

            print(f"Epoch: {epoch} Loss: {self.loss(y_batch, y_pred)}")

        return cp.asnumpy(W_input), cp.asnumpy(b_input), cp.asnumpy(W_output), cp.asnumpy(b_output)
    
    def predict(self, X, W_input, b_input, W_output, b_output):
        Z = np.dot(X, W_input) + b_input
        atv = np.maximum(0,Z)
        raw_logits = np.dot(atv, W_output) + b_output
        y_pred = self.softmax(cp.asarray(raw_logits))

        max_indices = cp.argmax(y_pred, axis=1, keepdims=False)
        return cp.asnumpy(max_indices)
    
NN = NeuralNetwork(784, 128*2, 10)

# Use GPU for speed

X_train_gpu = cp.asarray(normalize_mnist(X_train))
y_train_gpu = cp.asarray(y_train)

W_input, b_input, W_output, b_output = NN.train(X_train_gpu, y_train_gpu)
y_pred = NN.predict(X_test, W_input, b_input, W_output, b_output)


y_test_labels = np.argmax(y_test, axis=1)  # convert one-hot back to digits
accuracy = np.mean(y_pred == y_test_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

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
    flat = canvas.flatten().reshape(1, 784)
    result = NN.predict(flat, W_input, b_input, W_output, b_output)
    ax.set_title(f"Predicted: {result[0]}")
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_move)
fig.canvas.mpl_connect('button_release_event', on_release)

btn_predict.on_clicked(on_predict)
btn_clear.on_clicked(on_clear)

plt.axis("off")
plt.show()