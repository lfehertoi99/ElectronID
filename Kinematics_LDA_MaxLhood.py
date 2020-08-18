# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:36:16 2020

@author: lilif
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import scipy as sp
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import make_blobs
#%%
#GENERATING PARTICLES FROM MC GUNS WITH DIFFERENT (GAUSSIAN) PSEUDORAPIDITY AND TRANSVERSE MOMENTUM DISTRIBUTIONS
class BinaryParticleGun:
    
    def __init__(self, p_range = [[[175, 225], [1.2, 2.2]], [[1.7, 4], [0.25, 0.55]]], size = 1000, mass = 1):
        self.p_range = p_range #stores range of possible parameters
        self.size = size
        self.mass = mass
        
    def get_dist(self):
        normal = lambda x,y: norm(x,y).rvs(size = self.size)
        pt_mean, pt_std = np.random.uniform(*self.p_range[0][0]), np.random.uniform(*self.p_range[0][1])
        eta_mean, eta_std = np.random.uniform(*self.p_range[1][0]), np.random.uniform(*self.p_range[1][1])
        pt_dist, eta_dist = normal(pt_mean, pt_std), normal(eta_mean, eta_std)
        return [[pt_mean, eta_mean], [pt_dist, eta_dist]]

    def get_fmtm(self, pt, eta, phi_loc = 0, mass = 1): #all arguments should be vectors!
        phi = uniform(loc = phi_loc, scale = 2*np.pi).rvs(size = self.size)
        m = np.array([mass] * self.size)
        g_fmtm = np.column_stack((pt, eta, phi, m))
        return g_fmtm
    
    def get_samples(self):
        g1_params, g1_vars = self.get_dist()
        g2_params, g2_vars = self.get_dist()
        g1_data, g2_data = self.get_fmtm(*g1_vars), self.get_fmtm(*g2_vars)
        samples = np.concatenate((g1_data, g2_data))
        labels = np.concatenate((np.zeros(self.size), np.ones(self.size)))
        return np.column_stack((samples, labels))
        
    def split(self, data, ratio):
        bitmask = np.random.randn(len(data)) < ratio
        train = data[bitmask]
        test = data[~bitmask]
        return train, test
#%%
particlegun = BinaryParticleGun()
labeled_data = particlegun.get_samples()
train, test = particlegun.split(labeled_data, 0.75)
#%%
fig, ax = plt.subplots(figsize=(12,8))
for l,c,m in zip(np.unique(train[:, 4]),['r','g'],['o','x']): #"zip" creates iterator that pairs together values in respective tuples (sp.unique(y) contains class labels)
    ax.scatter(train[:, :2][train[:, 4]==l,0],
                train[:, :2][train[:, 4]==l,1],
                c=c, marker=m, label=l, s=10)
    
ax.set_title("Distribution of kinematic variables")
ax.set_xlabel("Transverse Momentum")
ax.set_ylabel("Pseudorapidity")
plt.legend()
plt.show()
#%%
#LDA for data in p_t - eta space, using sklearn
pt_eta_lda = LDA(n_components = 2) #OBJECT of class LDA with attribute n_components set to 2
pt_eta_lda.fit(train[:, :2], train[:, 4]) #finds best decision boundary based on maximizing inter-class and minimizing intra-class variance

train_accuracy = pt_eta_lda.score(train[:, :2], train[:, 4])
test_accuracy = pt_eta_lda.score(test[:, :2], test[:, 4])
print(train_accuracy, test_accuracy)
#%%
# Plot the hyperplanes and datapoints (including test data)
fig, ax = plt.subplots(figsize=(12,8))
for l,c,m in zip(np.unique(train[:, 4]),['r','g'],['o','x']): #"zip" creates iterator that pairs together values in respective tuples (sp.unique(y) contains class labels)
    ax.scatter(train[:, :2][train[:, 4]==l,0],
                train[:, :2][train[:, 4]==l,1],
                c=c, marker=m, label=l, s=10)

x1 = np.array([np.amin(train[:,0], axis=0), np.amax(train[:,0], axis=0)])

#lda.coef_ contains coefficients of decision boundary vectors
b, w1, w2 = pt_eta_lda.intercept_[0], pt_eta_lda.coef_[0][0], pt_eta_lda.coef_[0][1]
y1 = -(b+x1*w1)/w2    
ax.plot(x1,y1,c="k")

ax.set_xlabel("Transverse Momentum")
ax.set_ylabel("Pseudorapidity")
ax.legend()

plt.show()
#%%
#CREATE BINARY NAIVE BAYES CLASSIFIER: get best parameters for assumed Gaussian distribution using MLE
class ParticleNaiveBayes:
    
    def __init__(self, params = None):
        self.params = params
        
    #define MLE estimators
    def MLE(self, data):
        mean = np.mean(data)
        std = np.std(data, ddof = 1) #ddof = 1: use Bessel-corrected version of estimator to account for sampling error
        return [mean, std]
    
    #do MLE and get parameters for model boased on training data
    def get_params(self, train):
        train_g1, train_g2 = train[train[:, 4] == 0], train[train[:, 4] == 1] #training data contains pt data in column 1, eta data in column 2
        g1_pt_params, g2_pt_params = self.MLE(train_g1[:, 0]), self.MLE(train_g2[:, 0])
        g1_eta_params, g2_eta_params = self.MLE(train_g1[:, 1]), self.MLE(train_g2[:, 1])
        self.params = [[g1_pt_params, g2_pt_params], [g1_eta_params, g2_eta_params]]
        
    #2 get log-likelihood for each datapoint: p(features|parameters)
    def log_lhood(self, data, params):
        mean, std = params
        lhood_func = sp.stats.norm(loc = mean, scale = std)
        log_lhood_vec = sp.log(lhood_func.pdf(data))
        return log_lhood_vec #returns vector of same shape as data
    
    #get log-likelihood ratio of having come from gun 1 vs gun 2
    def lhood_ratio(self, data, var):
        g1, g2 = self.params[var]
        g1_lhood, g2_lhood = self.log_lhood(data, g1), self.log_lhood(data, g2)
        lhood_ratio = g1_lhood - g2_lhood #assumes equal distribution of data between classes
        return lhood_ratio #returns vector of same shape as data

    def predict(self, test):
        log_lhood_ratio_pt, log_lhood_ratio_eta = self.lhood_ratio(test[:, 0], 0), self.lhood_ratio(test[:, 1], 1)
        probabilities = log_lhood_ratio_pt  + log_lhood_ratio_eta #multiplication is addition in log space
        y_pred = [int(not(1 + np.sign(x))) for x in probabilities] #predictions for class labels (-ve: 1, +ve: 0)
        return probabilities, y_pred
#%%
classifier = ParticleNaiveBayes()
classifier.get_params(train)
probabilities, classes = classifier.predict(test)
accuracy = 1 - np.sum([(classes - test[:, 4])**2]) / len(test) #fraction of correctly classified points in test set
print(accuracy)
#%%
fig, ax = plt.subplots(figsize = (12, 8))

ax.scatter(test[:, 0], test[:, 1], c = probabilities, cmap="RdPu", s=8)

ax.set_title("Likelihood ratio of coming from particle gun 1")
ax.set_xlabel("Transverse momentum")
ax.set_ylabel("Pseudorapidity")

fig.colorbar(ax.scatter(test[:, 0], test[:, 1], c = probabilities, cmap="RdPu", s=8))

plt.show()
#%%        
#MAKE SCATTER PLOT TO VISUALIZE PERFORMANCE
fig, ax = plt.subplots(figsize=(12,8))

ax.scatter(test[:, 0], test[:, 1], c = classes, s = 8)
ax.scatter(train[:, 0], train[:, 1], c = train[:, 4], s = 8)

ax.set_title("Naive Bayes classifier performance")
ax.set_xlabel("Transverse momentum")
ax.set_ylabel("Pseudorapidity")  

plt.show()
#%%
