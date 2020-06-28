# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:16:22 2019

@author: saiki
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

dataset = pd.read_csv('heart.csv')
dataset2 = pd.read_csv('test2.csv')

dataset.info()

dataset2.info()

dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

dataset2 = pd.get_dummies(dataset2, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
dataset2[columns_to_scale] = standardScaler.fit_transform(dataset2[columns_to_scale])

y = dataset['target']
X = dataset.drop(['target'], axis = 1)

X_pred = dataset2.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

X_pred=standardScaler.fit_transform(X_pred)

k=8
knn_scores = []
knn_classifier = KNeighborsClassifier(n_neighbors = k)
knn_classifier.fit(X_train, y_train)
knn_scores.append(knn_classifier.score(X_test, y_test))

y_pred = knn_classifier.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
#dataset2.shape()

Y_pred = knn_classifier.predict(X_pred)

print(Y_pred)
    

dataset2['target'] = Y_pred
dataset2.to_csv('test2.csv',index=False)