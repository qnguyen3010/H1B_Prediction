#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 10:54:39 2017

@author: AaronNguyen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
pre_dataset = pd.read_csv("h1b_kaggle.csv")
#print(pre_dataset.describe)
print(pre_dataset.count())

# preprocessing dataset
dataset = pre_dataset.drop(pre_dataset.columns[[9, 10]], axis=1).dropna()
#print(dataset.describe)
dataset.rename(columns={'Unnamed: 0':'ID'}, inplace=True)
print(dataset.count())

print(dataset.CASE_STATUS.unique())
print(dataset['CASE_STATUS'].value_counts())

# delete all dummy values in CASE_STATUS column
to_drop = ["INVALIDATED","REJECTED","PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED"]
final_dataset = dataset[~dataset['CASE_STATUS'].isin(to_drop)]
print(final_dataset['CASE_STATUS'].value_counts())
print(final_dataset.count())

# Seperate independant and dependant variables
X = final_dataset.iloc[:,5:8].values
y = final_dataset.iloc[:,1].values

# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
labelencoder_X1 = LabelEncoder()
X[:, 2] = labelencoder_X1.fit_transform(X[:, 2])

# one-hot encode the non-order categorical data
onehotencoder = OneHotEncoder(categorical_features = [0,2])
X = onehotencoder.fit_transform(X).toarray()

# label the outcome
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# check the encoding result for X and y
from scipy.stats import itemfreq
print(itemfreq(y))

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc)



