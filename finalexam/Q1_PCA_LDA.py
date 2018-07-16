# -*- coding: utf-8 -*-
"""
@author: Vaijyant Tomar
"""

import os;
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Go to the file location
os.chdir(r"D:\OneDrive\My Documents\theInvestigations\UNCC\Semester 4\ML\Github ML-Fall2017\finalexam-vaijyant");

# loading data
dataread = pd.read_csv(r"dataset_1.csv", header=0)
data = dataread.values
data = data.astype(float)

########################## Question 1 #########################################
###############################################################################
# 1) Plot V2 vs V1. Do you see a clear separation of the raw data?
###############################################################################
# >>> Solution:<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# There is a pattern in data set. As evident from the plot (produced by
# following code) we can see data points arranged in two, somewhat, linear
# fashion.

# Plotting for (1)
ax = plt.subplot(111)
plt.scatter(x = data[:, 0],
            y = data[:, 1],
            marker = 'x',
            color = 'red',
            alpha = '0.75',
            label = "v2 vs v1")
            
plt.xlabel('V1')
plt.ylabel('V2')

leg = plt.legend(loc='upper left', fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.title('V2 vs V1')
plt.show()

###############################################################################
# 2) Apply your own PCA function to this dataset without scaling the two
#    variables.
#    Project the raw data onto your first principal component axis, i.e. the
#    PC1 axis. Do you still see a clear separation of the data in PC1, i.e. in
#    projections of your raw data on the PC1 axis?
###############################################################################
# >>> Solution:<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# As evident from the plot (produced by following code) the data points
# are arranged in two regions, top half and bottom half.
# ---------------------------[ PCA MODULE ]------------------------------------
## PCA Module Function 1. Mean Vetor 
def fn_mean_vector(data_values):
    mean_vector = data_values.mean(axis=0)
    return mean_vector

## PCA Module Function 2. Mean Centering
def mean_center(data_values):
    return data_values - fn_mean_vector(data_values)

## PCA Module Function 3. eigen values and eigen vectors
def eigen_value_vector(data):
    cov_mat = np.cov(data.astype(float), rowvar=False)
    eig_value, eig_vector = np.linalg.eig(cov_mat)
    return eig_value, eig_vector

## PCA Module Function 4. PCA function
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
# ---------------------------[ PCA MODULE ENDS]--------------------------------

data_pca =  data[:, range(0, 2)]
pca_result = pca(data_pca)
data_pca_projection = data_pca.dot(pca_result[ 'loadings'])

# Plotting
ax = plt.subplot(111)
plt.scatter(x = data_pca_projection[:, 0],
            y = data_pca_projection[:, 1],
            marker = 'o',
            color = 'red',
            alpha = '0.75',
            label = "v2 vs v1")            
plt.xlabel('PC1')
plt.ylabel('PC2')

leg = plt.legend(loc='upper left', fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.title('Data as seen on PC1 and PC2 space')
plt.show()

data_pca_projection_PC1 = pca_result['scores'][:, 0].reshape((60, 1)).dot(pca_result['loadings'][:, 0].reshape((1, 2)))

# Plotting
ax = plt.subplot(111)
plt.scatter(x = data_pca_projection_PC1[:, 0],
            y = np.zeros(60),
            marker = 'o',
            color = 'red',
            alpha = '0.75',
            label = "Projected data on PC1 axis")            
plt.xlabel('PC1')
plt.ylabel('')

leg = plt.legend(loc='upper left', fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.title('Data as seen on PC1 axis')
plt.show()

###############################################################################
# 3) Add the PC1 axis to the plot you obtained in (1)
###############################################################################
# >>> Solution:<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Plot
ax = plt.subplot(111)
plt.scatter(x = data[:, 0],
            y = data[:, 1],
            marker = 'x',
            color = 'red',
            alpha = '0.75',
            label = "v2 vs v1")
plt.plot([0, -50*pca_result['loadings'][0,0]],
         [0, -50*pca_result['loadings'][1,0]],
         color='blue',
         linewidth = 2,
         alpha = 0.6,
         label = "PC1")
            
plt.xlabel('V1')
plt.ylabel('V2')

leg = plt.legend(loc='upper left', fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.title('V2 vs V1 and Pricipal Component 1')
plt.show()

###############################################################################
# 4) Apply your own LDA function to this dataset and obtain W. The class
#    information of each data point is in the label column.
###############################################################################
# >>> Solution:<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ---------------------------[ LDA MODULE ]------------------------------------
no_of_classes = 2
no_of_features = 2

data_lda =  data[:, range(0, 2)]

#(i) Computing the d-dimensional mean vectors

mean_vector = []
mean_vector.append(np.mean(data_lda[range(30)], axis=0))
mean_vector.append(np.mean(data_lda[range(30, 60)], axis=0))


#(ii) Computing the Scatter Matrices: within (S_W) and between (S_B)

##(a) Within-class scatter
S_W = np.zeros((no_of_features, no_of_features))

class_sc_mat1 = np.zeros((no_of_features, no_of_features))
for row in data_lda[range(30)]:
    row, mean_vector[0] = row.reshape(no_of_features,1), mean_vector[0].reshape(no_of_features,1)
    class_sc_mat1 += (row-mean_vector[0]).dot((row-mean_vector[0]).T)


class_sc_mat2 = np.zeros((no_of_features, no_of_features))         
for row in data_lda[range(30,60)]:
    row, mean_vector[1] = row.reshape(no_of_features,1), mean_vector[1].reshape(no_of_features,1)
    class_sc_mat2 += (row-mean_vector[1]).dot((row-mean_vector[1]).T)
    
S_W += class_sc_mat1 + class_sc_mat2

##(b) Between-class scatter matrix
overall_mean = np.mean(data_lda, axis=0)
overall_mean = overall_mean.reshape(no_of_features,1)

S_B = np.zeros((no_of_features,no_of_features))

n1 = data_lda[range(30),:].shape[0]
mean_vec = mean_vector[0].reshape(no_of_features,1)
S_B = S_B + n1 * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

n2 = data_lda[range(30,60),:].shape[0]
mean_vec = mean_vector[1].reshape(no_of_features,1)
S_B = S_B + n2 * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)


#(iii)  Solving the generalized eigenvalue problem for the matrix S^−1_WS_B
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))


#(iv) Selecting linear discriminants for the new feature subspace
##(a)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

##(b) Choosing k eigenvectors with the largest eigenvalues
#W = eig_pairs[0][1].reshape(no_of_features,1)
W_p = np.hstack((eig_pairs[0][1].reshape(no_of_features,1), eig_pairs[1][1].reshape(no_of_features,1)))

# ---------------------------[ LDA MODULE END]---------------------------------
print "Matrix W: \n", W.real

###############################################################################
# 5) Project your raw data onto W. Do you see a clear separation of the data in
#    the projection onto W?
###############################################################################
# >>> Solution:<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# As evident from the plot (produced by following code) the data points
# are fairly grouped into two classes.

# Plot
data_lda_projection = data_lda.dot(W_p.real)
ax = plt.subplot(111)
plt.scatter(x = data_lda_projection[range(30), 0],
            y = np.zeros(30),
            marker = '1',
            s = 100,
            color = 'blue',
            alpha = '0.75',
            label = "Class 1")
plt.scatter(x = data_lda_projection[range(30, 60), 0],
            y = np.zeros(30),
            marker = '2',
            s = 100,
            color = 'red',
            alpha = '0.75',
            label = "Class 0")
 
plt.ylim( (-1, 1) )           
plt.xlabel('W_1')

leg = plt.legend(loc='upper left', fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.title('Data projection onto W1')
plt.show()

###############################################################################
#6) Add the W axis to your plot. At this point, your plot should contain the
#   raw data points, the PC1 axis you obtain from the PCA analysis, and the W
#   axis you obtain from the LDA analysis.
###############################################################################
# >>> Solution:<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#W_p  = W_p.real
#W_inv = np.linalg.inv(W_p)
#K = W_inv.dot(mean_vector[0] - mean_vector[1])
#center = np.sum(mean_vector, axis=0)/2

ax = plt.subplot(111)
plt.scatter(x = data[range(30), 0],
            y = data[range(30), 1],
            marker = '1',
            s = 100,
            color = 'blue',
            alpha = '0.75',
            label = "Class 1")
plt.scatter(x = data[range(30, 60), 0],
            y = data[range(30, 60), 1],
            marker = '2',
            s = 100,
            color = 'red',
            alpha = '0.75',
            label = "Class 0")
plt.plot([0, -50*pca_result['loadings'][0,0]],
         [0, -50*pca_result['loadings'][1,0]],
         color='orange',
         linewidth = 2,
         alpha = 0.6,
         label = "PC1")
# plt.plot([center[0], K[0]],
#         [center[1], K[1]],
#         color='green',
#         linewidth = 2,
#         alpha = 0.6,
#         label = "W_1")
#plt.plot([center[0]+20, K[0] - K[0]],
#         [center[1]+20, K[1] - K[0]],
#         color='magenta',
#         linewidth = 2,
#         alpha = 0.6,
#         label = "decision boundary")
plt.plot([0, -10*data_lda_projection[0,0], 5*data_lda_projection[0,0]],
         [0, -10*data_lda_projection[1,0], 5*data_lda_projection[1,0]],
         color='green',
         linewidth = 2,
         alpha = 0.6,
         label = "W_1")
            
plt.xlabel('V1')
plt.ylabel('V2')

leg = plt.legend(loc='upper left', fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.title('V2 vs V1 with PC1 and W1 axis')
plt.show()

###############################################################################
#7) Compute the variance of the projections onto PC1 and PC2 axes. What
#   is the relationship between these two variances and the eigenvalues of the
#   covariance matrix you use for computing PC1 and PC2 axes?
###############################################################################
# >>> Solution:<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#

np.var(data_pca_projection, axis=0) # array([ 158.35671177,    5.03986249])
np.var(pca_result['scores'][:,0])
np.var(pca_result['scores'][:,1])
pca_result['PC_variance']           # array([ 161.04072383,    5.12528389])


###############################################################################
#8) Compute the variance of the projections onto the W axis.
###############################################################################
# >>> Solution:<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
data_lda_projection_p = data_lda.dot(W_p)
np.var(data_lda_projection_p, axis=0) # array([  5.22507522,  74.91666667])

###############################################################################
#9) What message can you get from the above PCA and LDA analyses?
###############################################################################
# >>> Solution:<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# PCA attempts to maximize the variance
# LDA, on the other hand, maximizes the class separability