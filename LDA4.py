# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 01:03:15 2020

@author: kevin
"""

import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### read data from pandas 


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names = names)

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4]. values

setosa = X[0:50,:]
versicolor = X[51:100,:]
virginica = X[101:150,:]

### finding the mean 

m_s = np.mean(setosa, axis = 0)
m_vc = np.mean(versicolor, axis = 0)
m_vg = np.mean(virginica, axis = 0)

m = (m_s + m_vc + m_vg)/3

### finding sb

sb_s = np.dot((m_s - m).T, (m_s - m))
sb_vc = np.dot((m_vc - m).T, (m_vc - m))
sb_vg = np.dot((m_vg - m).T, (m_vg - m))
sb = sb_s + sb_vc + sb_vg

### finding sw

cov_s = np.dot((setosa - m_s).T, (setosa - m_s))
cov_vc = np.dot((versicolor - m_vc).T, (versicolor - m_vc))
cov_vg = np.dot((virginica - m_vg).T, (virginica - m_vg))
sw = cov_s + cov_vc + cov_vg

### find eigenvalues

eigval, eigvec= np.eigval, eigvec = np.linalg.eig(np.dot(np.linalg.inv(sw),sb))

eigpairs = [(np.abs(eigval[i]), eigvec[:,i]) for i in range(len(eigval))]
eigpairs = sorted(eigpairs, key = lambda k: k[0], reverse = True)

wt = np.vstack((eigpairs[0][1], eigpairs[1][1]))
w = np.transpose(wt)

### transform all

X_transformed = np.dot(X,w)

plt.scatter(X_transformed[:,0], X_transformed[:,1])
plt.show()


### transform individually
setosa_t = np.dot(setosa, w)
versicolor_t = np.dot(versicolor, w)
virginica_t = np.dot(virginica,w)

plt.scatter(setosa_t[:,0],setosa_t[:,1])
plt.scatter(versicolor_t[:,0], versicolor_t[:,1])
plt.scatter(virginica_t[:,0], virginica_t[:,1])

### yay?




