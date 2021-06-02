# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 19:20:22 2021

@author: Dhruv

Project: Building a system in Python that can predict whether an object is 
either Rock or Mine with SONAR Data and alert the submarines about danger around them.

"""

'''Import Libraries'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

'''Import Dataset'''
# need to specify header=None else first row will be taken as header
data = pd.read_csv(r'Resources/sonar_data.csv', header=None)

'''Exploratory data analysis'''
data.head()
data.tail()
data.info()
data.shape
data.describe()
data[60].value_counts() # to see training data balance by checking y variable

'''Prepare X and y'''
X = data.drop(columns=60, axis = 1)
y = data[60]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)

'''Model building'''
lr = LogisticRegression()
model = lr.fit(X_train, y_train)

'''Prediction'''
y_pred = model.predict(X_test)

'''Model Evaluation'''
# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)
print(f'Accuracy score on training data is : {round(training_data_accuracy,2)}')

## Note: Accuracy on training data is 84%

# Accuracy on testing data
testing_data_accuracy = accuracy_score(y_pred, y_test)
print(f'Accuracy score on training data is : {round(testing_data_accuracy,2)}')

## Note: Accuracy on testing data is 83%

'''Making a prediction system'''
input_data = (0.02,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.066,0.2273,0.31,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.555,0.6711,0.6415,0.7104,0.808,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.051,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.018,0.0084,0.009,0.0032)

# Change the input data into numpy array, also need to reshape the input
input_data_numpy = np.asarray(input_data).reshape(1,-1)

# Predict Rock or Mine based on input data
input_array_predict = model.predict(input_data_numpy)
print(f'Prediction is : {"Rock" if input_array_predict=="R" else "Mine"}')
# End