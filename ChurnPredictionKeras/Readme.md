# Churn Prediction using Neural Networks

The goal is to predict customer churn for a bank. We want to predict which customera are going to leave out bank service.
Dataset is named churn_modelling.csv

I used Keras in Python, but Tensorflow, Theano libraries have to be already installed in Python.
 The reason is that Keras is built on top of Tensorflow and Theano so we need  these libraries to be 
running in back-end whenever you run the program in Keras./n

A sequential NN is used with a dense layers. Two hidden layyer were used as well as an output layer.
The number of nurons in the hidden layyers was 6. A uniform initialization and relu activation function were
set. The SGD optimizer was adam.

With 100 iterations and a batch size of 10: 

confusion matrix:
1534,   65],
[212,  189

Accuracy on test set= (1534+189)/(1534+65+189+212)=86.1% 
The percentage of true negative and true prediction

Precision = 189/(65+189)=74.5% 
Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. Here is the percentage of ones that we predicted ture churners are actually true churners.  

Recall =  189/(189+212)=47.1% 
Recall (Sensitivity) is the ratio of correctly predicted positive observations to the all observations in actual class - yes. Here is the percentage of the churners we could predict.




