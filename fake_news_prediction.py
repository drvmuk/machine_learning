# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 12:01:26 2021

@author: Dhruv

Project: Fake news prediction(Kaggle Dataset)
Link: https://www.kaggle.com/c/fake-news/data?select=train.csv

ML Algorithm used: Logistics Regression(Binary prediction)

id: unique id for a news article
title: the title of a news article
author: author of the news article
text: the text of the article; could be incomplete
label: a label that marks the article as potentially unreliable
1: unreliable
0: reliable

"""
# Import Libraries
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter  import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download nltk stopwords
import nltk
nltk.download('stopwords')

# Read data
data = pd.read_csv(r'D:/Projects/Machine Learning Udemy/project_git/machine_learning/Resources/fake_news.csv/train.csv')

# EDA
data.shape
## Dataset balance check
data['label'].value_counts()
## Find out the null values in dataset
data.isnull().sum()
## Since the quantum of null data is insignificant/less, we can replace null value with ''
data = data.fillna('')

## Running prediction on text column takes lot of computation time, hence merging author and title for prediction
data['content'] = data['title'] + ' ' + data['author']

## Apply Stemming
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

data['content'] = data['content'].apply(stemming)

# Preparing data for modelling
X = data['content'].values
y = data['label'].values

# Converting textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
print(X)

## Split X and y to train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, stratify = y, random_state = 100)

# Training the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy score
## Model's performance on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)
print(round(training_data_accuracy,2))

## Model's performance on test data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, y_test)
print(round(testing_data_accuracy,2))

# Making a prediction system
new_data = X_test[0]
prediction = model.predict(new_data)

print(f'{"News is fake" if prediction[0] == 1 else "News is not fake"}')

# End