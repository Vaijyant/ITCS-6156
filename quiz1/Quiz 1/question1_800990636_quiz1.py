# -*- coding: utf-8 -*-
"""
Quiz 1
Created on Fri Sep 22 10:08:58 2017

@author: Vaijyant Tomar
"""
import numpy as np
import pandas as pd

#1. loading data
data = pd.read_csv(r"C:\users\vaijy\Desktop\dataset_1.csv")
tdata = data.values
tdata = tdata.T


#2. Calulating sample_variance
def sample_variance(x):
    n = len(x)
    x_mean = sum(x) / n
    var = sum((x-x_mean)**2) / (n - 1)
    return var

#3. mean vector
def mean_vector(mean_x, mean_y, mean_z):
    mean_vector = np.array([[mean_x],[mean_y],[mean_z]])
    return mean_vector

#4. mean centering
def mean_center(matrix, mean_vector):
    return matrix - mean_vector

#5. calculating covariance_matrix
def covariance_matrix(matrix):
    cov_mat = np.cov([matrix[0,:], matrix[1,:], matrix[2,:]])
    return cov_mat

#6. calculating eigen values and eigen vectors
def eigen_value_vector(matrix):
    cov_mat = covariance_matrix(matrix)
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    for i in range(len(eig_vec_cov)):   
        eigvec_cov = eig_vec_cov[:,i].reshape(1,3).T
        print('Eigenvector {}: \n{}'.format(i+1, eigvec_cov))
        print('Eigenvalue {} matrix: {}\n'.format(i+1, eig_val_cov[i]))
        
        
def pca(dataset, mean_x, mean_y, mean_z):
    
    #mean centering    
    data_mc=mean_center(dataset,  mean_vector(mean_x, mean_y, mean_z))
    covmatrix=np.cov(data_mc)
    eigen_value,eigen_vector  = np.linalg.eig(covmatrix)
    eig_pairs = [(np.abs(eigen_value[i]), eigen_vector[:,i]) for i in range(len(eigen_value))]
    eig_pairs.sort()
    eig_pairs.reverse()
    
    matrix_w = np.hstack((eig_pairs[i][1].reshape(dataset.shape[1],1)) for i in range(1000))    
    
                            
    print(matrix_w)                     
    y=data_mc.dot(matrix_w) 
    return y, matrix_w 

# ----------------------------MODULES END--------------------------------------

#==============================================================================
#calculate the variance of every variable in the data file
print("variance for x")
print(sample_variance(tdata[0,:]))


print("variance for y")
print(sample_variance(tdata[1,:]))

print("variance for z")
print(sample_variance(tdata[2,:]))

#==============================================================================
#calculate the covariance between x and y, and between y and z
print("covariance between x and y")
print(np.cov(tdata[0,:], tdata[1,:])[0][1])

print("covariance between x and y")
print(np.cov(tdata[1,:], tdata[2,:])[0][1])

#==============================================================================
#do PCA of all the data in the given data file using your own PCA module
y, matrix_w = pca(tdata, np.mean(tdata[0,:]), np.mean(tdata[1,:]), np.mean(tdata[2,:]))

print(y)
print(matrix_w)



