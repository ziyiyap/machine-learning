import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"https://storage.dosm.gov.my/hies/hies_district.csv")



#Data Cleaning

#Unwanted columns : date,state,district

data = data.drop(columns=['date','state','district'])

#Functions
convert_to_array = lambda *series: [col.to_numpy().reshape((len(col),1)) for col in series]

def add_columns_ones(*col):
    ones = np.ones((len(col[0]),1))
    return np.column_stack((ones,*col))

#Convert data columns into numpy array
income_mean, gini, expenditure_mean,poverty,income_median = convert_to_array(data['income_mean'],data['gini'],data['expenditure_mean'],data['poverty'],data['income_median'])


X = add_columns_ones(income_mean) #Adds two features: income_mean and gini

y_actual = expenditure_mean #What we want to predict (given actual values)


###
# General Linear Regression Formula:
# 
# y-hat = Xtheta
# 
# y-hat = predicted value
# X = input features (with column of ones in front)
# theta = Normal Equation, calculates the best fit line 
# 
# Alternatively,
# 
# y-hat = theta0 + x1theta1 + x2theta2 + ... + xdthetad
# ###


theta = np.linalg.inv(X.T @ X) @ X.T @y_actual

y_pred = X @ theta

#Predict values (2D)

def pred_2D(income_mean):
    return f"Predict 2D: RM{round((add_columns_ones(np.array([income_mean])) @ theta).flatten().tolist()[0],2)}"


#2D graph

plt.scatter(income_mean,y_actual,color='red')
plt.plot(income_mean,y_pred)
plt.xlabel("Income Mean (RM)")
plt.ylabel("Expenditure Mean (RM)")
plt.show()
plt.close()

#3D graph (gini)
X_3D = add_columns_ones(income_mean,gini)

theta_3D = np.linalg.inv(X_3D.T @ X_3D) @ X_3D.T @y_actual

y_3D = X_3D @ theta_3D

#Predict (3D)

def pred_3D(income_mean,gini):
    return f"Predict 3D: RM{round((add_columns_ones(np.array([income_mean]),np.array([gini])) @ theta_3D).flatten().tolist()[0],2)}"



fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.scatter(income_mean,gini,y_actual,color='red')
ax.set_xlabel("Income Mean (RM)")
ax.set_ylabel("Gini")
ax.set_zlabel("Expenditure Mean (RM)")

income_range, gini_range = [np.linspace(f.min(),f.max(),20) for f in [income_mean,gini]]

income_grid, gini_grid = np.meshgrid(income_range,gini_range)

z = theta_3D[0] + income_grid*theta_3D[1] + gini_grid * theta_3D[2]

ax.plot_surface(income_grid,gini_grid,z,alpha=0.7)


#4D + predict
X_4D = add_columns_ones(income_mean,gini,poverty)

theta_4D = np.linalg.inv(X_4D.T @ X_4D) @ X_4D.T @y_actual

y_4D = X_4D @ theta_4D

def pred_4D(income_mean,gini,poverty):
    return f"Predict 4D: RM{round((add_columns_ones(np.array([income_mean]),np.array([gini]),np.array([poverty])) @ theta_4D).flatten().tolist()[0],2)}"


#5D + predict
X_5D = add_columns_ones(income_mean,gini,poverty,income_median)

theta_5D = np.linalg.inv(X_5D.T @ X_5D) @ X_5D.T @y_actual

y_5D = X_5D @ theta_5D

def pred_5D(income_mean,gini,poverty,income_median):
    return f"Predict 5D: RM{round((add_columns_ones(np.array([income_mean]),np.array([gini]),np.array([poverty]),np.array([income_median])) @ theta_5D).flatten().tolist()[0],2)}"


im = 3477
imed = 2926
gin = 0.322
pov = 15.8
print(pred_2D(income_mean=im))
print(pred_3D(income_mean=im,gini=gin))
print(pred_4D(income_mean=im,gini=gin,poverty=pov))
print(pred_5D(income_mean=im,gini=gin,poverty=pov,income_median=imed))

plt.show()
plt.close()