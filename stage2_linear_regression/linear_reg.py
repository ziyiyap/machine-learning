

###
# Simple linear regression formula: y = wx + c, where w is the weight = gradient
# 
# Multiple linear regression formula:
# y = wx0 + w1x1 + ... + wnxn + c, where w is the weight != gradient, y is the predicted value.
# Alt: y= Wx+b, where W is (w0 w1 w2 ... wn) (matrix)
# Mean Squared Error
# = (1/n) · Σ(yᵢ - ŷᵢ)² , where n is the number of samples/data points, yᵢ is the actual value of (price),  ŷᵢ is the predicted (price). 
# 
# Why we square it:
# To get rid of negatives,
# To enlarge the loss, so the model MUST focus on minimizing the error.
# 
# Normal Equation
# w = (XT * X)^-1 * XT*y, where  XT is the transpose of X (rows become columns, columns become rows), 
# X is the design matrix, where each row is a data point, each column is a feature.
# ()^-1 is the inverse, 
# y is the column vector of actual outputs
# ###

#Cell 1 - Load Data
import numpy as np
from sklearn.datasets import fetch_california_housing
import pandas as pd

housing = fetch_california_housing()
housing.keys() 
###
# dict_keys(['data', 'target', 'frame', 'target_names', 'feature_names', 'DESCR'])
# data - feature matrix X
# target - target vector y
# feature names - feature col
# target names - names of the target variable
# DESCR - description
# frame - ready made dataframe###

df = pd.DataFrame(housing.data)
df.columns = housing.feature_names
df["Price"] = housing.target
print(df.head())

y = df["Price"]
X = df.drop(columns=["Price"])
X = np.column_stack((np.ones(len(X)), X))

#Linear Regression from scratch
index = np.arange(len(X)) #[    0     1     2 ... 20637 20638 20639]
index = np.random.permutation(index)
train_size = int(len(X) * 0.8)
train_indices = index[:train_size]
test_indices = index[train_size:]
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train


y_pred = X_test @ theta

mse = np.mean((y_test-y_pred)**2)
rmse = np.sqrt(mse)

#Linear Regression with sklearn
from sklearn.linear_model import LinearRegression

X_clean = housing.data

X_train_sk = X_clean[train_indices]
X_test_sk = X_clean[test_indices]

model = LinearRegression()
model.fit(X_train_sk, y_train)
print(model.coef_, model.intercept_)

y_pred_sk = model.predict(X_test_sk)

rmse_sk = np.sqrt(np.mean((y_test - y_pred_sk)**2))
print(rmse_sk)