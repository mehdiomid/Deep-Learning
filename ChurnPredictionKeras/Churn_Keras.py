# -*- coding: utf-8 -*-
"""
@author: Mehdi
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Initializing Neural Network
classifier = Sequential()

"""
output_dim: number of nodes you to add to the layer
init: initialization of Stochastic Gradient Decent. Need to assign weights to each mode. At the time of initialization, weights should be close to 0 and we randomly initialize weights using uniform function.
input_dim: For first layer, it is the number of input variables. Here input variables are 11.
In the second layer model automatically detects the number of input variable from the first hidden layer.

Activation Function: Here we are using rectifier(relu) function in our hidden layer and Sigmoid function in our output layer as we want binary result from output layer but if the number of categories in output layer is more than 2 then use SoftMax function.
"""

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

"""
Optimizer: The algorithm is Stochastic Gradient descent(SGD). Among several types of SGD algorithm the one which we will use is ‘Adam’. 
loss fuction: ‘binary_crossentropy’, if our dependent variable has more than 2 categories in output then use ‘categorical_crossentropy’.
metrics: accuracy 
"""
# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting our model 
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Creating the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

