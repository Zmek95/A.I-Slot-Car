#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:27:27 2020

@author: ziyad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Data preprocessing
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) # This will train our model from the data provided.

#Predict test set results
y_pred = regressor.predict(x_test)

#Visualisation
# Training set
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color ='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.xlabel('Salary')
plt.show()

# Test set
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color ='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.xlabel('Salary')
plt.show()

#Making a single prediction
print(regressor.predict([[12]])) # Predict the salary of an employee with 12 years of experience

#Getting the coefficients of the final linear regression equation
# y = b0 + b1*x1
print(regressor.coef_)
print(regressor.intercept_)
