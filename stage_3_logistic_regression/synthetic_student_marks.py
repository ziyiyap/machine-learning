import matplotlib.pyplot as plt
import numpy as np

student_marks = [
    35, 42, 58, 61, 73, 29, 84, 77, 49, 90,
    55, 67, 38, 81, 46, 72, 64, 53, 88, 31,
    59, 62, 70, 45, 79, 68, 52, 41, 85, 60
]

pass_rate = [
    0, 0, 0, 1, 1, 0, 1, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 1, 0, 1, 0,
    0, 1, 1, 0, 1, 1, 0, 0, 1, 1
]

X = np.array(student_marks).reshape((len(student_marks),1))
X_mean = X.mean()
X_std = X.std()

X = (X - X_mean) / X_std
print(X)
y_actual = np.array(pass_rate).reshape((len(pass_rate),1))

index = np.arange(len(X))

#set random seed
np.random.seed(42)
np.random.shuffle(index)


X_shuffled, y_shuffled = X[index], y_actual[index]

split = int(len(X_shuffled) * 0.8)  # 80/20, training data / test data

X_train, y_train = X_shuffled[:split], y_shuffled[:split]
X_test, y_test = X_shuffled[split:], y_shuffled[split:]

def sigmoid(z):
    return 1/((np.e ** -z) + 1)

def cross_entropy_loss(y_actual, y_pred): #loss function
    return -(np.sum(y_actual * np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred)))

def compute_gradients(X, y_actual, y_pred): # X is feature matrix
    dw = (X.T @ (y_pred - y_actual)) / X.shape[0]
    db = np.sum((y_pred - y_actual)) / y_pred.shape[0]
    return dw, db

def train(X, y_actual, alpha = 0.1, epochs = 1000):
    w = np.zeros(X.shape[1]) # number of columns
    b = 0.0

    for _ in range(epochs):

        z = X @ w + b
        y_pred = sigmoid(z)
        dw, db = compute_gradients(X, y_actual, y_pred)

        w -= alpha * dw
        b -= alpha * db

    return w, b

def predict(x,w,b):
    z = x * w + b
    return f"Predict: {sigmoid(z)}"

w, b = train(X_train, y_train.squeeze())

X_sort = np.sort(X, axis=0)
y_sort = np.sort(y_actual, axis=0)

z = X_sort @ w + b
y_pred = sigmoid(z)

v = 40
print(predict((v - X_mean) / X_std, w, b))

plt.scatter(X_sort, y_sort, color='red')
plt.plot(X_sort, y_pred, color='blue')
plt.show()