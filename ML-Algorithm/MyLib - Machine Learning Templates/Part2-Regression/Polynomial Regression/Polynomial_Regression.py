#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 20:00:46 2020

@author: ziyad
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() # Create an object of this class
lin_reg.fit(x, y)

# Training the polynomial regression model on the whole dataset
# y = b0 + b1*x1 + b2*x1^2 + ... + bn*x1^n
from sklearn.preprocessing import PolynomialFeatures
# Pre-process to create poly features e.g x ---> x x^2 for degree=2
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
# Perform linear regression on polynomial x
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(x_poly, y)

# Visualising Linear regression
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising Polynomial regression
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_poly.predict(x_poly), color = 'blue')
plt.title('Truth or Bluff ( Polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg_poly.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a result using linear regression
print(lin_reg.predict([[6.5]]))

# Predicting a result using polynomial regression
print(lin_reg_poly.predict(poly_reg.fit_transform([[6.5]])))

# Print linear coefficients
print(lin_reg.coef_)
print(lin_reg.intercept_)

# Print polynomial coefficients
print(lin_reg_poly.coef_)
print(lin_reg_poly.intercept_)
