# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:17:43 2019

@author: Lucifer
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:, 3])
onreHotEncoder  = OneHotEncoder(categorical_features=[3])
X = onreHotEncoder.fit_transform(X).toarray()

#Avoiding Dummy varable trap
X = X[:, 1:]

#Making test set and training set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting and training the regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, y_train)

# Predicting the model result
y_predict = regressor.predict(X_test)

#Building model with backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# making prediction on basis of backward elemination
regressor2 = LinearRegression()
X_train_OLS, X_test_OLS, y_train_OLS, y_test_OLS = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)
regressor2 = regressor2.fit(X_train_OLS, y_train_OLS)
y_Result = regressor2.predict(X_test_OLS)

print("Result for test set is: ")
print(y_Result)


