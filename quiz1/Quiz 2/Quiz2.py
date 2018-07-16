# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:30:00 2017

@author: Vaijyant Tomar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1. loading data
dataread = pd.read_csv(r"C:\Users\vaijy\Desktop\SCLC_study_output_filtered.csv") #zif no headers header=None
data = dataread.values
data = data[:,range(1, 50)]
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
    #eig_pairs.sort(reverse=True) #sort in descending order
    
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

#===================== >> Quiz 2 Start << =====================================


#1. What is the total variance of the original variables and what is the total variance of the PCs. 
for i in range(0, len(data[0])):
    print("Variance is: ")
    print sample_variance(data[:,i])
    
np.var(data)

#2.	What is the covariance between the first PC and the second PC? 
pca_results = pca(data)
covariance_matrix(pca_results['PC_variance'])


#3.	Plot the scores plot using the first and second PCs. In your scatter plot,
#use red dots for the first 20 samples (rows 1 to 20) in the raw data table and
# use blue dots for the remaining 20 samples (rows 21 to 40
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('scores plot')
ax.scatter(pca_results['scores'][1,20:,0], pca_results['scores'][:,1], color='red')
ax.scatter(pca_results['scores'][21,40:,0], pca_results['scores'][:,1], color='blue')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
fig.show()


#4.	How many PCs do you need to keep in order to keep 75% of the variance in the data? 
percentVarianceExplained = 100 * pca_results['PC_variance'][0] / sum(pca_results['PC_variance'])
print "PC1 explains: " + str(round(percentVarianceExplained, 2)) + '% variance\n'

#5. Plot the loadings plot. Standardize each variable and perform PCA again
#and answer the following questions.
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('loadings plot')
ax.scatter(pca_results['loadings'][:,0], pca_results['loadings'][:,1], color='blue')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
fig.show()


#after standardizing
data_sd = data-mean_vector(data)
pca_results_sd = pca()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('loadings plot')
ax.scatter(pca_results_sd['loadings'][:,0], pca_results_sd['loadings'][:,1], color='blue')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
fig.show()


#6. 	What is the total variance of all of the PCs. Is it equal to the total
#variance of the original variables before standardization?
np.var(pca_results["PC_variance"])

for i in range(0, len(data_sd[0])):
    print("Variance is: ")
    print sample_variance(data_sd[:,i])

#Yes it is equal to the original variable