# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 00:53:58 2017

@author: Vaijyant Tomar
"""

from sklearn import datasets
diabetes = datasets.load_diabetes()
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.lines as mlines

x = diabetes.data[:,2]
x = x.reshape(-1,1)
y = diabetes['target']

random.seed(9001)
random_index = random.sample(range(0, len(x)-1), 20)

index_train = []
index_test = []

for i in range(0, len(x)):
    if i in random_index:
        index_test.append(True)
    else:
        index_test.append(False)

index_train = [not i for i in index_test]


# Split the data into training and  testing sets
x_train = x[index_train]
x_test = x[index_test]


# Split the targets training and  testing sets
y_train = diabetes.target[index_train]
y_test = diabetes.target[index_test]

# Doing Linear refression
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_predicted = regr.predict(x_test)

# Printing the values
print("Coefficients: ", regr.coef_[0])
print("Mean squared error: ", mean_squared_error(y_test, y_predicted))
print("Variance score: ", r2_score(y_test, y_predicted))



#target is response of interest, a quantitative measure of disease progression one year after baseline.
# Plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_test, y_test,  color='black')
ax.plot(x_test, y_predicted, color='orange', linewidth=2)
ax.set_xlabel('body mass index')
ax.set_ylabel('response of interest')
orange_line = mlines.Line2D([], [], color='orange', markersize=15, label='Regression line')
plt.legend(handles=[orange_line])
ax.show()