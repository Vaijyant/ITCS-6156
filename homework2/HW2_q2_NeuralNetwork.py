# -*- coding: utf-8 -*-
"""
@author: Vaijyant Tomar
"""
###############################################################################
# Implement your own neural network for regression purpose in Python.
# Use the mean squared error as the cost function.
# The neural network has two input units (not counting the bias unit), one
# hidden layer with two units (not counting the bias unit), and two output
# units. Assume that there is only one sample being
# x =|0.05 |
#    |0.1  |  
# y =|0.01 |
#    |0.99 |
# Initialize values in Θ(1) and Θ(2) to be uniformly distributed random numbers
# between 0.0 and 1.0.
# Implement your algorithm based on the theory presented in class. Plot the
# total cost vs the iterations as well as every parameter θ vs the iterations.
###############################################################################

import numpy as np
import matplotlib.pyplot as plt


# objects for graph
objects1 = ('theta_1 (0, 0)', 'theta_1 (0, 1)', 'theta_1 (0, 2)',
            'theta_1 (1, 0)', 'theta_1 (1, 1)', 'theta_1 (1, 2)')
objects2 = ('theta_2 (0, 0)', 'theta_2 (0, 1)', 'theta_2 (0, 2)',
            'theta_2 (1, 0)', 'theta_2 (1, 1)', 'theta_2 (1, 2)')


X = np.array(([[0.05, 0.1]]), dtype=float)
y = np.array(([[0.01], 
               [0.99]]), dtype=float)

print "\n"

theta_1 = np.random.random((2,3))
theta_2 = np.random.random((2,3))

plt_theta1 = np.array(range(6))
plt_theta2 = np.array(range(6))
plt_theta1 = np.vstack((plt_theta1, np.reshape(theta_1, (1, 6))[0]))
plt_theta2 = np.vstack((plt_theta2, np.reshape(theta_2, (1, 6))[0]))

def sigmoid(x, deriv=False):
    if(deriv==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))


cost ={}
iterations = 1000
for j in range(iterations):
    a1 = np.array([np.append(1, X)])     # adding bias term
    a2 = sigmoid(np.dot(theta_1, a1.T))
    
    a2 = np.array([np.append(1, a2)])    # adding bias term
    a3 = sigmoid(np.dot(theta_2, a2.T))
    
    J_theta = np.sum(((y - a3)**2)) / 2
    cost[j] =  J_theta                   # for plotting
    
    delta_3 = a3 - y
    delta_2 = np.dot(theta_2.T, delta_3) * a2 * (1-a2)

    
    #update weights
    theta_2 -= 0.01 * a2.T.dot(delta_3.T).T
    theta_1 -= 0.01 * a1.dot(delta_2)
    
    plt_theta1 = np.vstack((plt_theta1, np.reshape(theta_1, (1, 6))[0]))
    plt_theta2 = np.vstack((plt_theta2, np.reshape(theta_2, (1, 6))[0]))
        

k = -1
for i in range(6):
    if i%3 == 0:
        k += 1
    row = k
    col = i%3
        
    label1 = "theta_1 ("+str(row)+", "+str(col)+") trend in " + str(j+1) + " iteration"

    ax = plt.subplot(111)
    plt.plot(range(iterations+1), plt_theta1[1:, i])
    plt.xlabel('Number of iterations')
    plt.ylabel('theta_1')
    plt.title(label1)
    plt.show() 



k = -1
for i in range(6):
    if i%3 == 0:
        k += 1
    row = k
    col = i%3
        
    label2 = "theta_2 ("+str(row)+", "+str(col)+") trend in " + str(j+1) + " iteration"

    ax = plt.subplot(111)
    plt.plot(range(iterations+1), plt_theta2[1:, i])
    plt.xlabel('Number of iterations')
    plt.ylabel('theta_2')
    plt.title(label2)
    plt.show()        
        
print("Output after training: ")
print(a3)


ax = plt.subplot(111)
plt.plot(cost.keys(), cost.values())
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Cost vs Number of iterations')
plt.show()


#Comment
# Increasing the number of iterations can increase the efficiency