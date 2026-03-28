###
# Linear Regression
# 
# Linear Regression is a good way to predict stuff, for example house price when given parameters. Eg. House Size
# 
# The general formula of linear regression:
# 
# ŷ = Xθ or, 
# 
# ŷ = θ₀ (bias) + θ₁x₁ + θ₂x₂ + ... + θdxd 
# 
# where ŷ (y-hat) is the predicted price (output),
# 
# θ (matrix) = weight/parameters the model learns
# X (matrix) = input features
#
# θ₀ = bias/intercept
# θ₁, θ₂, ...,  θd = weights/parameters
# x₁, x₂, ..., xd = input features
# ###

from sklearn.datasets._california_housing import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

housing = fetch_california_housing()

###
# MedInc - Median Income
# HouseAge - Median house age
# AveRooms - Average number of rooms per household
# AveBedrms - Average number of bedrooms
# Population - District population
# AveOccup - Average occupants per household
# Latitude - Geographic coordinate
# Longitude - Geographic coordinate
# ###

df = pd.DataFrame(housing.data,columns=housing.feature_names)
df['MedHouseVal'] = housing.target #Y value


def add_ones(*feature):
    ones = np.ones((feature[0].shape[0],1))
    return np.column_stack((ones,*feature))

#input features

medinc, houseage, averooms, avebedrms = df['MedInc'].to_numpy(),df['HouseAge'].to_numpy(), df['AveRooms'].to_numpy(), df['AveBedrms'].to_numpy()
latitude,longitude = df['Latitude'].to_numpy(), df["Longitude"].to_numpy()

X = add_ones(medinc,latitude,longitude)
X_sklearn = df[['MedInc','Latitude','Longitude']]

y_actual = df['MedHouseVal'].to_numpy()
y_actual = y_actual.reshape((len(y_actual),1))

y_sklearn = df['MedHouseVal']

# set seed 42

np.random.seed(42)

indices = np.arange(len(X))

np.random.shuffle(indices)

X_shuffled, y_shuffled = X[indices],y_actual[indices]

split = int(len(indices) * 0.8)

X_train, X_test = X_shuffled[:split], X_shuffled[split:]
y_train, y_test = y_shuffled[:split], y_shuffled[split:]


#sklearn train test split

Xsk_train =  X_sklearn.iloc[indices[:split]]
Xsk_test = X_sklearn.iloc[indices[split:]]
ysk_train = y_sklearn.iloc[indices[:split]]
ysk_test = y_sklearn.iloc[indices[split:]]


theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

y_pred = X_test @ theta

#Calculation

mean = np.mean(y_test)

SST = np.sum((y_test - mean) ** 2)
SSE = np.sum((y_test - y_pred) ** 2)

R_squared = 1 - (SSE/SST)

print(f"R² = {round(R_squared,5)}\nSST = {round(SST,5)}\nSSE = {round(SSE,5)}\n")


model = LinearRegression().fit(Xsk_train,ysk_train)

sk_r_squared = model.score(Xsk_test,ysk_test)

print(f"SKlearn's R² = {round(sk_r_squared,5)} ")


residual = y_test - y_pred

plt.scatter(y_test,residual,color='red',alpha=0.3)
plt.axhline(y=0,color='blue',linewidth=1)
plt.xlabel("Actual Price")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()