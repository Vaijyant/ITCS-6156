# -*- coding: utf-8 -*-
"""
Quiz 1
Created on Fri Sep 22 09:55:24 2017

@author: Vaijyant Tomar

"""
import numpy as np
from numpy import linalg as LA

a = np.array([[0, -1], [ 2, 3]], int)
lam, p = LA.eig(a)

lam; p
