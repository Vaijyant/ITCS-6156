# -*- coding: utf-8 -*-
"""
@author: Vaijyant Tomar
"""
########################## Question 3 #########################################
# Write your own python script for classification using logistic regression. 
# You may refer to relevant code posted on the UNCC Canvas page for our class. 
# Perform binary classification analysis of the virginica and versicolor 
# flowers using petal length and petal width in the iris data set using your 
# own logistic regression function. Use leave-one-out cross validation and get 
# the average error rate, i.e. you will perform logistic regression 100 times 
# total with each of the 100 flowers used once for testing.
#
# Compare classification results you have obtained using ANN with the results 
# you have obtained using logistic regression.
###############################################################################

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
data = iris.data[50:, 2:]  # we only take the first two features.
y = iris.target[50:] - 1


###############################################################################
# Scalling
###############################################################################
data_o = np.ones(100)
data_0 = (data[:, 0] - min(data[:, 0])) / (max(data[:, 0]) - min(data[:, 0]))
data_1 = (data[:, 1] - min(data[:, 1])) / (max(data[:, 1]) - min(data[:, 1]))

data_std = np.hstack((
        np.reshape(data_o, (100, 1)),
        np.reshape(data_0, (100, 1)), 
        np.reshape(data_1, (100, 1))
        ))

###############################################################################
J = np.zeros(100)
error_rate = dict.fromkeys((range(100)), 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

predict = np.zeros(100)
theta = np.random.random((3,1))

alpha = 0.1

cost0 = []
cost1 = []
error = 0
for iteration in range(100):
    
    for index in range(100):
        test_sample = data_std[index]
    
        
        train_sample = np.delete(data_std, index, 0)
        train_target = np.delete(y, index, 0)
              
        # tuning theta for current iteration
        for j in range(3):
            h_theta_x = sigmoid(theta.T.dot(train_sample.T))
            
            J_theta = np.sum(train_target * np.log(h_theta_x) + (1 - train_target) * np.log(1 - h_theta_x)) / -99
            J[iteration] = J[iteration] + J_theta
            
            # vectorized calculation for gradient decent
            theta[j] = theta[j]  - ( alpha / 99 ) * np.sum(((h_theta_x - train_target) * train_sample[:,j]))
        
        # class 0: x <= 0.5 and class 1: x>0.
        predict[index] = np.round(sigmoid(theta.T.dot(test_sample.T)))
        
        if predict[index] != y[index]:
            error += 1
            
    error_rate[iteration] =error
    
avg_error_rate = {k: v/10000.0 for k, v in error_rate.items()}        
print "Average Error rate: \n", avg_error_rate 


# Plotting J_theta vs number of iterations
plt.subplot(111)
plt.plot(range(iteration+1), J)
plt.xlabel('Number of iterations')
plt.ylabel('$J(\\theta)$')
plt.title('$J(\\theta)$ vs Iterations')
plt.show() 