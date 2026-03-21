import numpy as np
import matplotlib.pyplot as plt

###
# Simple linear regression model:
# y_hat = wx+b /wX
# 
# y_hat = predicted value
# w = weight/parameters/gradient
# b = bias/intercept
# x = input feature
# X = matrix of input features ###




X = np.array([1,2,3]).reshape(3,1)
y_actual = np.array([5,6,7]).reshape(3,1)
X_b = np.hstack((np.ones((3,1)),X))

theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_actual

b,w = theta.flatten()

linear_equation = w * X + b

def prediction(x):
    return f"Prediction: {round(w * x + b,2)}"

print(prediction(4))

plt.scatter(X,y_actual)
plt.plot(X,linear_equation.flatten())
plt.title(f"y = {w:.2f}x + {b:.2f}")
plt.show()
