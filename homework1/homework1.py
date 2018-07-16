# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 00:21:25 2017

@author: Vaijyant Tomar
"""

import os;
os.chdir(r'D:\OneDrive\My Documents\theInvestigations\UNCC\Semester 4\ML\Homework 1');

# Begin =======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


#1. loading data
dataread = pd.read_csv(r"linear_regression_test_data.csv") #if no headers header=None
data = dataread.values
data = data[:,range(1, 4)]
data = data.astype(float)
#the header of the columns are the features. If not, transpose the data as data = data.T

#2. Calulating sample_variance in a column
def sample_variance(feature):
    n = len(feature)
    feature_mean = sum(feature) / n
    var = sum((feature-feature_mean)**2) / (n - 1)
    return var

#3. mean vector
def mean_vector(data_values):
    mean_vector = data_values.mean(axis=0)
    return mean_vector

#4. mean center
def mean_center(data_values):
    return data_values - mean_vector(data_values)
    
#5. covariance matrix
def covariance_matrix(data):
    cov_mat = np.cov(data.astype(float), rowvar=False)
    #Does row represents a feature? False. Row is an observation.
    #Hence rowvar is False
    #bias by default is false, if bias is false
        #Default normalization (False) is by (N - 1).
        #If bias is True, then normalization is by N
    return cov_mat

#6. eigen values and eigen vectors
def eigen_value_vector(data):
    cov_mat = covariance_matrix(data)
    eig_value, eig_vector = np.linalg.eig(cov_mat)
    return eig_value, eig_vector

#7. PCA
def pca(data):
    mean_center_data = mean_center(data)
    eig_value, eig_vector = eigen_value_vector(data)
    eig_pairs = [(np.abs(eig_value[i]), eig_vector[:,i]) for i in range(len(data[0]))]
    
    sort_args = eig_value.argsort()[::-1]
    eig_value = eig_value[sort_args]
    eig_vector = eig_vector[:,sort_args]
    
    eig_pairs = [(np.abs(eig_value[i]), eig_vector[:,i]) for i in range(len(data[0]))]
    
    eig_vector_matrix = np.hstack((eig_pairs[i][1].reshape(data.shape[1],1)) for i in range(len(data[0])))
    
    pca_score = data.dot(eig_vector_matrix)
    
    pca_results = {'data': data,
                   'mean_centered_data': mean_center_data,
                   'PC_variance': eig_value,
                   'loadings': eig_vector,
                   'scores': pca_score}

    return pca_results

#8. displaying results
def displaying_pca_results(data, corr_logic=False):
    #Visualizing raw data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data[:,0], data[:,1], color='blue')
    ax.set_aspect('equal', 'box')
    fig.show()
    
    pca_result = pca(data)
    
    if not corr_logic:
        data = mean_center(data)
    else:
        pass
    
    #Explaining 1st PCA
    percentVarianceExplained = 100 * pca_result['PC_variance'][0] / sum(pca_result['PC_variance'])
    print "PC1 explains: " + str(round(percentVarianceExplained, 2)) + '% variance\n'
    
    #scree plot
    #A Scree Plot is a simple line segment plot that shows the fraction of
    #total variance in the data as explained or represented by each PC
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('scree plot')
    ax.scatter(range(len(pca_result['PC_variance'])), pca_result['PC_variance'], color='blue')
    fig.show()
    
    #scores
    #The principal component score is the length of the diameters of the ellipsoid.
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('scores plot')
    ax.scatter(pca_result['scores'][:,0], pca_result['scores'][:,1], color='blue')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.show()
    
    #loadings
    #The factor loadings, also called component loadings in PCA, are the
    #correlation coefficients between the variables (rows) and factors (columns)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('loadings plot')
    ax.scatter(pca_result['loadings'][:,0], pca_result['loadings'][:,1], color='blue')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.show()
    
    #raw data and perspective
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('raw data and PC axis')
    ax.scatter(data[0], data[1], color='blue')
    ax.plot([0, -20*pca_result['loadings'][0,0]], [0, -20*pca_result['loadings'][1,0]],
            color='green', linewidth=3)
    ax.plot([0, 20 * pca_result['loadings'][0, 1]], [0, 20 * pca_result['loadings'][1, 1]],
            color='green',linewidth=3)
    ax.set_aspect('equal', 'box')
    fig.show()
    
    
    # keep only the first dimension
    dataReconstructed = np.matmul(pca_result['scores'][:, 0].reshape((len(data), 1)), pca_result['loadings'][:,0].reshape((1,len(data[0]))))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('reconstructed data')
    ax.scatter(dataReconstructed[:, 0], dataReconstructed[:, 1], color='blue')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    fig.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('raw data and PC axis')
    ax.scatter(data[0], data[1], color='blue')
    k=3
    ax.plot([0, (-1)*k*pca_result['loadings'][0,0]], [0, (-1)*k*pca_result['loadings'][1,0]],
             color='green', linewidth=3)
    ax.plot([0, k * pca_result['loadings'][0, 1]], [0, k * pca_result['loadings'][1, 1]],
            color='green',linewidth=3)
    ax.set_aspect('equal', 'box')
    fig.show()

#===================== >> Modules End << ======================================
    
#=================== >> Homework 1 begins << ================================== 

#1) The data in linear_regression_test data.csv contains x, y, and
#   y−theoretical. Perform PCA on x and y. Plot y vs x, y-theoretical vs x, and the
#   PC1 axis in the same plot.

data_pca = data[:,range(0, 2)]

pca_result = pca(data_pca)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('y vs x, y-theoretical vs x')
ax.scatter(data[:, 0], data[:, 1], color='blue')
ax.scatter(data[:, 0], data[:, 2], color='red')
ax.plot([0, 5*pca_result['loadings'][0,0]], [0, 5*pca_result['loadings'][1,0]], color='green', linewidth=3)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.show()


#2) Perform linear regression on x and y with x being the independent
#   variable and y being the dependent variable. Plot the regression line in the same plot
#   as you obtained in (1). Compare the PC1 axis and the regression line obtained above.
#   Are they very different or very similar?

x = data_pca[:,0]
y = data_pca[:,1]
n = len(data)

b1_hat = (np.sum(x * y) - n * mean_vector(x) * mean_vector(y)) / (np.sum(x * x) - n * mean_vector(x)**2)
b0_hat = mean_vector(y)-b1_hat * mean_vector(x)
y_hat = b0_hat + b1_hat * x

y_hat= b0_hat + b1_hat * x



fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('y vs x, y-theoretical vs x')
ax.scatter(data[:, 0], data[:, 1], color='blue')
ax.scatter(data[:, 0], data[:, 2], color='red')
ax.plot([0, 5*pca_result['loadings'][0,0]], [0, 5*pca_result['loadings'][1,0]], color='green', linewidth=3)
plt.plot(x,y,'.')
plt.plot(x,y_hat)
ax.set_xlabel('x')
ax.set_ylabel('y/ytheoretical')
blue_dot = mlines.Line2D([], [], color='blue', marker='.', markersize=15, label='y vs x')
red_dot = mlines.Line2D([], [], color='red', marker='.', markersize=15, label='y-theoretical vs x')
green_line = mlines.Line2D([], [], color='green', markersize=15, label='PC 1')
orange_line = mlines.Line2D([], [], color='orange', markersize=15, label='Regression line')
plt.legend(handles=[blue_line, red_dot, green_line, orange_line])
fig.show()






    