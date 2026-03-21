import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Manual Linear Regression (Normal Equation method)

data = pd.read_csv(r"stage2_linear_regression\hies_state.csv")

#Data cleaning

#Unwanted - date,state 

data = data.drop(columns=['date','state'])

#For now, we want to plot income_mean against expenditure_mean

income = data['income_mean']
income = income.to_numpy().reshape((len(income),1)) #convert to numpy array

expenditure = data['expenditure_mean']
expenditure = expenditure.to_numpy().reshape((len(expenditure),1)) #outputs, y

X = np.column_stack((np.ones((len(income),1)), income)) #add a column of ones, so intercept value does not get ignored.
y_actual = expenditure

###
# General Linear Regression Formula
# 
# y-hat = Xtheta
# 
# y-hat = predicted value 
# X = matrix of features (1 feature1 feature2 ..)
# theta = Normal Equation = computes the bestfit line
# ###


theta = np.linalg.inv(X.T @ X) @ X.T @ y_actual

y_pred = X @ theta

def predict(num):
    x = np.array([num])
    x_array = np.column_stack((np.ones((len(x),1)),x))
    return f"Predict: RM{round((x_array @ theta).flatten().tolist()[0],2)}"

print(predict(4885)) #Kelantan actual: 3305, predicted: 3316.89, approx RM11 diff
print(predict(8517)) #Johor actual: 5342, predicted 5333.84, approx RM 10+ diff, so one input feature isnt fully accurate.
print(predict(4885))#Predict: RM3316.89

plt.scatter(income,y_actual,color='red')
plt.plot(income,y_pred)
plt.xlabel("Income Means (RM)")
plt.ylabel("Expenditure Means (RM)")
plt.close()

#To make it more accurate, we must add more features.

income_median = data['income_median']
income_median = income_median.to_numpy().reshape((len(income_median),1))

gini = data['gini']
gini = gini.to_numpy().reshape((len(gini),1))

poverty = data['income_median']
poverty = poverty.to_numpy().reshape((len(poverty),1))

#add more features into X
#y_actual remains
X_features = np.column_stack((X,gini))


theta_feature = np.linalg.inv(X_features.T @ X_features) @ X_features.T @ y_actual

y_pred_feature = X_features @ theta_feature

def predict_gini(mean,gini):
    x1 = np.array([mean])
    x2 = np.array([gini])

    x1 = x1.reshape((len(x1),1))
    x2 = x2.reshape((len(x2),1))

    x_array = np.column_stack((np.ones((len(x1),1)),x1,x2))
    return f"Predict: RM{round((x_array @ theta_feature).flatten().tolist()[0],2)}"



print(predict_gini(4885,0.385)) #Predict: RM3383.33 #

#if we add poverty as a feature, the model will overfit and produces 'fake' data.
#Predict: RM8843.41

fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.scatter(income,gini,expenditure,color='red')

#create a grid

income_mean_range = np.linspace(income.min(),income.max(),20)
gini_range = np.linspace(gini.min(),gini.max(),20)

income_mean_grid, gini_grid = np.meshgrid(income_mean_range,gini_range)

z = theta_feature[0] + theta_feature[1]*income_mean_grid + theta_feature[2]*gini_grid

ax.plot_surface(income_mean_grid,gini_grid,z,alpha=0.7)
plt.show()