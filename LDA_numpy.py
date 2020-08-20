# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 01:03:15 2020

@author: kevin
"""
#%%
import scipy as sp
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
### read data from pandas 
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
df = pd.read_csv("iris.csv", names = names)
#%%
bitmask = np.random.rand(len(df)) < 0.9
train = df[bitmask]
test = df[~bitmask]
#%%
class_labels = [x for x in np.unique(df["Class"])]
d = []
for x in class_labels:
    d += [np.array(df[df["Class"] == x])[:, :-1]]
d = np.array(d)
#%%
class FisherLinearDiscriminant:
    
    def __init__(self, n_dim, n_classes, params = None, g_params = None):
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.params = params #VECTORS REPRESENTING PLANES  TO PROJECT ONTO
        self.g_params = g_params #LIST OF GAUSSIAN MEAN AND STD FOR EACH CLASS POST-PROJECTION

    def get_mean(self, class_data): #class data should be array dim (n_samples x n_features x n_classes)
        means = np.mean(class_data, axis = 1)
        mu = np.mean(means, axis = 0)
        return means, mu
        
    def scatter_b(self, means, mu): #scatter taken with entire dataset
        for x in means:
            x.reshape(-1, 1)
        class_b = [np.dot((x - mu).T, (x - mu)) for x in means]
        return class_b, np.sum(class_b, axis = 0)

    def scatter_w(self, class_data, means): #scatter taken only within one class
        class_w = [np.dot((x - mx).T, (x - mx)) for x, mx in zip(class_data, means)]
        return class_w, np.sum(class_w, axis = 0)
        
    def set_eigvecs(self, scatter_b, scatter_w): #get directions to project onto, set attribute "params"
        scatter_b, scatter_w = scatter_b.astype(float), scatter_w.astype(float)
        eigval, eigvec = np.linalg.eig(np.dot(np.linalg.inv(scatter_w),scatter_b)) #solve generalized eigenvalue equation
        eigpairs = zip(eigval, eigvec)
        eigpairs = sorted(eigpairs, key = lambda k: np.abs(k[0]), reverse = True) #sort by eigenvalues in decreasing order
        w = np.array([eigpairs[i][1] for i in range(self.n_dim)])
        self.params = w
        
    def set_g_params(self, d):
        g_params = dict()
        for x in range(self.n_classes):
            proj = np.dot(np.array(d[x]), lda.params.T)
            mean, cov = np.mean(proj, axis = 0), np.cov(proj.astype(float), rowvar = False)
            g_params[x] = {"mean": mean, "cov": cov}
        self.g_params = g_params
        
    def predict(self, X):
        g = lambda x, y: multivariate_normal(x, y) #x and y are vectors!!
        transform = np.dot(X, self.params.T)
        g_vals = np.zeros((len(transform), self.n_classes))
        for x in range(self.n_classes):
                j, k = self.g_params[x]["mean"], self.g_params[x]["cov"]
                g_vals[:, x] = g(j, k).pdf(transform)  
        return np.argmax(g_vals, axis = 1)
#%%
lda = FisherLinearDiscriminant(2, 3) #create model
m, mu = lda.get_mean(d)  #get within-class means and overall mean (vectors of shape 4x1)
sb_c, sb, sw_c, sw = *lda.scatter_b(m, mu), *lda.scatter_w(d, m) #get class-wise and total between and within scatter
lda.set_eigvecs(sb, sw) #set model parameters (projection directions)   
lda.set_g_params(d)
lda.predict(np.array(test)[:, :-1])
#%%    



