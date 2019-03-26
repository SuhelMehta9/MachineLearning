# -*- coding: utf-8 -*-
# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Getting the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Making linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Making polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Ploting the results
plt.scatter(X,y,color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Linear')
plt.show()

# Ploting polynomial values
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Polynomial')
plt.show()

x = np.ones((1,1),dtype = np.float32)
x[0][0] = 6.5
# Predicting the input with user input
print(lin_reg2.predict(poly_reg.fit_transform(x)))
