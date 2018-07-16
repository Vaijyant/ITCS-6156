# -*- coding: utf-8 -*-
"""
@author: Vaijyant Tomar
"""

import os;
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Go to the file location
os.chdir(r"D:\OneDrive\My Documents\theInvestigations\UNCC\Semester 4\ML\Github ML-Fall2017\homework2-vaijyant");

# loading data
dataread = pd.read_csv(r"SCLC_study_output_filtered_2.csv", header=0, index_col=0)
data = dataread.values
data = data.astype(float)



###############################################################################
#1) Implement your own LDA algorithm in Python based on the theory
#   presented in class.
###############################################################################
no_of_classes = 2
no_of_features = len(data[0])
#(i) Computing the d-dimensional mean vectors

mean_vector = []
mean_vector.append(np.mean(data[range(20)], axis=0))
mean_vector.append(np.mean(data[range(20,40)], axis=0))


#(ii) Computing the Scatter Matrices - within and between

##(a) Within-class scatter
S_W = np.zeros((no_of_features, no_of_features))                        # Initializing to zeros

class_sc_mat1 = np.zeros((no_of_features, no_of_features))              # scatter matrix for every class
for row in data[range(20)]:
    row, mean_vector[0] = row.reshape(no_of_features,1), mean_vector[0].reshape(no_of_features,1) # make column vectors
    class_sc_mat1 += (row-mean_vector[0]).dot((row-mean_vector[0]).T)


class_sc_mat2 = np.zeros((no_of_features, no_of_features))         
for row in data[range(20,40)]:
    row, mean_vector[1] = row.reshape(no_of_features,1), mean_vector[1].reshape(no_of_features,1) # make column vectors
    class_sc_mat2 += (row-mean_vector[1]).dot((row-mean_vector[1]).T)
    
S_W += class_sc_mat1 + class_sc_mat2                                    # sum class scatter matrices

##(b) Between-class scatter matrix
overall_mean = np.mean(data, axis=0)
overall_mean = overall_mean.reshape(no_of_features,1)

S_B = np.zeros((no_of_features,no_of_features))

n1 = data[range(20),:].shape[0]
mean_vec = mean_vector[0].reshape(no_of_features,1)
S_B = S_B + n1 * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

n2 = data[range(20,40),:].shape[0]
mean_vec = mean_vector[1].reshape(no_of_features,1)
S_B = S_B + n2 * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)


#(iii)  Solving the generalized eigenvalue problem for the matrix S^−1_WS_B
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))


#(iv) Selecting linear discriminants for the new feature subspace
##(a)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])
    
##(b) Choosing k eigenvectors with the largest eigenvalues
W = eig_pairs[0][1].reshape(no_of_features,1)
print('Matrix W:\n', W.real)

#(v) Transforming the samples onto the new subspace
data_lda = data.dot(W).real
#assert data_lda.shape == (40,1), "The matrix is not 40x2 dimensional."


# Plotting
ax = plt.subplot(111)
plt.scatter(x=data_lda[range(20)],
            y=np.zeros(20),#data_lda[range(20),1],
            marker='s',
            color='red',
            alpha='0.5',
            label='NSCLC')
plt.scatter(x=data_lda[range(20,40)],
            y=np.zeros(20),
            marker='o',
            color='green',
            alpha='0.5',
            label='SCLC')
plt.xlabel('LD1')
plt.ylabel('LD2')

leg = plt.legend(loc='upper center', fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.title('LDA using custom function')
plt.show()

###############################################################################
#2) Apply your own LDA algorithm to the cell line data in
#   SCLC study output filtered 2.csv
#   and compare your results with results from
#   sklearn.discriminant analysis.LinearDiscriminantAnalysis.
###############################################################################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

y=[0 for i in range(20)]            #"NSCLC"
y.extend([1 for i in range(20)])    #"SCLC"
y = np.array(y)

sklearn_lda = LDA(n_components=1)
data_lda_sklearn = sklearn_lda.fit_transform(data, y)

# Plotting
ax = plt.subplot(111)
plt.scatter(x=data_lda_sklearn[range(20)],
            y=np.zeros(20),
            marker='s',
            color='red',
            alpha='0.5',
            label='NSCLC')
plt.scatter(x=data_lda_sklearn[range(20,40)],
            y=np.zeros(20),
            marker='o',
            color='green',
            alpha='0.5',
            label='SCLC')
plt.xlabel('W_1')
plt.ylabel('')

leg = plt.legend(loc='upper center', fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.title('LDA using sklearn')
plt.show()

# Comments ====================================================================
# As evident from the from the graphs the both the ways produce thre required
# result. It is also interesting to note that, the graph produced using
# sklean libraray is scaled however it maintains the same structure of the
# graph obtained when the values were not scaled.