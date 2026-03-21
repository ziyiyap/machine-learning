import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

income = [3, 5, 7, 2, 8, 6, 10, 12, 4, 9, 15, 11, 14, 1, 13, 20]
age    = [2, 4, 5, 1, 6, 3, 8, 10, 2, 7, 12, 9, 14, 1, 13, 20]
house_price = [4, 6, 8, 3, 9, 7, 11, 13, 5, 10, 15, 12, 16, 2, 14, 20]
#X

income_array = np.array(income).reshape(len(income),1)
age_array = np.array(age).reshape(len(age),1)
X = np.column_stack((income_array,age_array))

X_calc = np.column_stack((np.ones((len(income),1)), X))


y = np.array(house_price).reshape(len(house_price),1) #real price


theta = np.linalg.inv(X_calc.T @ X_calc) @ X_calc.T @ y #obtain weights
y_pred = X_calc @ theta
###
# 
# y-hat = Xtheta
# 
# theta = normal equation
# theta = (X.T @ X)^-1 @ X.T @ y###

#sorting

def pred(i,a):
    i,a = [i], [a]
    if len(i) != len(a):
        print("Length of income must be same as age")
        return
    else:
        x = np.column_stack((np.ones((len(np.array(i)),1)),np.array(i),np.array(a)))
        return f"Prediction: {(x @ theta).flatten()}"

print(pred(3,2))


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(income,age,house_price,color='red',label='Actual')

#create a grid over income and age ranges

income_range = np.linspace(income_array.min(),income_array.max(),20)
age_range = np.linspace(age_array.min(),age_array.max(),20)

income_grid, age_grid = np.meshgrid(income_range,age_range)

z = theta[0] + theta[1]*income_grid + theta[2]*age_grid

ax.plot_surface(income_grid,age_grid,z)

plt.show()