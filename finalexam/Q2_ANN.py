# -*- coding: utf-8 -*-
"""
@author: Vaijyant Tomar
"""

########################## Question 2 #########################################
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

# =============================================================================

theta_1 = np.random.random((2,3))
theta_2 = np.random.random((1,3))
cost = np.zeros(100)

def sigmoid(x, deriv=False):
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

def prediction_y(a2,theta_2):
   return sigmoid(np.dot(theta_2, np.reshape(a2, (3,1)))).item()
#    if(p >= 0.5):
#        return 1
#    else:
#        return 0

prediction = np.zeros(100)

for iteration in range(100):
    predict = []
    print "Iteration: ", iteration
    #test_split
    test_sample = data_std[iteration]
    test_target = y[iteration]
        
    # train split
    train_sample = np.delete(data_std, iteration, 0)
    train_target = np.delete(y, iteration, 0)

    for sample_no in range(99):
        for l in range(1000):
     
            a1 = train_sample[sample_no]          # test sample is with bias term, appended d_o = 1
            a2 = sigmoid(np.dot(theta_1, a1.T))
            
            a2 = np.array([np.append(1, a2)])    # adding bias term
            a3 = sigmoid(np.dot(theta_2, a2.T))
            
            
            delta_3 = (a3 - train_target[sample_no])/99
            delta_2 = (np.dot(theta_2.T, delta_3.T) * a2 * (1-a2))/99
            #update weights
            theta_2 -= 1 * a2.T.dot(delta_3.T).T
            theta_1 -= 1 * a1.dot(delta_2)
        a1 = test_sample
        a2 = sigmoid(np.dot(theta_1, a1.T))
        a2 = np.array([np.append(1, a2)])
        prediction = prediction_y(a2,theta_2)
        #print prediction
        error = np.abs(test_target - round(prediction))
        predict.append(error)
    prediction = predict
      