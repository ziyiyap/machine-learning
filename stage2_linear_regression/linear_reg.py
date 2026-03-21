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
import numpy as np

housing = fetch_california_housing()

print(housing.feature_names)