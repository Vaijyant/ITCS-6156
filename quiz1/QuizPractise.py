# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 20:20:49 2017

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

# -----------------------------------------------------------------------------
print(sample_variance(tdata[0,:]))
print(sample_variance(tdata[1,:]))
print(sample_variance(tdata[2,:]))

mean_vector = mean_vector(np.mean(tdata[:,0]), np.mean(tdata[:,1]), np.mean(tdata[:,2]))
print(covariance_matrix(tdata))
eigen_value_vector(tdata)

mean_center(tdata, mean_vector)
