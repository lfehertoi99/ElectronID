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
from scipy.stats import poisson
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
#%%
#GENERATING PARTICLES FROM MC GUNS WITH DIFFERENT (GAUSSIAN) PSEUDORAPIDITY AND TRANSVERSE MOMENTUM DISTRIBUTIONS
def g_dist(pt, eta, phi, m): #all arguments should be vectors!
    g_fmtm = np.column_stack((pt, eta, phi, m))
    return g_fmtm
    
normal = lambda x,y: norm(x,y)
uniform_d = lambda x,y: uniform(loc=x, scale=y)

g1_eta = normal(2, 0.2).rvs(size=1000)
g2_eta = normal(3.5, 0.3).rvs(size=1000)


g1_pt = normal(200, 1).rvs(size=1000)
g2_pt = normal(203, 1.5).rvs(size=1000)

g1_phi = uniform_d(0, 2*np.pi).rvs(size=1000)
g2_phi = uniform_d(0, 2*np.pi).rvs(size=1000)

mass = np.ones(1000)
#%%
experiment1, experiment2 = g_dist(g1_pt, g1_eta, g1_phi, mass), g_dist(g2_pt, g2_eta, g2_phi, mass) #4-momentum vectors from guns
#CREATE PLOTS OF DATA IN p_t - eta space
fig, ax = plt.subplots(1, 1, figsize=(12,8))

ax.scatter(g1_pt, g1_eta, c = "r", label = "Gun 1", s = 10)
ax.scatter(g2_pt, g2_eta, c = "g", label = "Gun 2", s = 10)

ax.legend()
ax.set_xlabel("Transverse Momentum")
ax.set_ylabel("Pseudorapidity")

plt.show()
#%%
#CREATE CLASS LABELS FOR DATA
g1_labels = np.zeros(1000)
g2_labels = np.ones(1000)

pt_vals = np.concatenate((g1_pt, g2_pt))
eta_vals = np.concatenate((g1_eta, g2_eta))
labels = np.concatenate((g1_labels, g2_labels))

pt_eta = np.column_stack((pt_vals, eta_vals, labels))

#SPLIT DATA INTO TRAINING AND TEST
bitmask = np.random.randn(len(pt_eta)) < 0.7
train = pt_eta[bitmask]
test = pt_eta[~bitmask]
#%%
#TRIAL RUN OF LDA IN SKLEARN WITH RANDOMLY GENERATED DATA
X, y = make_blobs(n_samples=200, centers=3, n_features=2) #X contains a column vector for each feature, y contains integer class labels for each sample
lineardisc = LDA(n_components = 2) #OBJECT of class LDA with attribute n_components set to 2
lineartrans = lineardisc.fit(X[:150], y[:150]) #fit model to randomly generated blobs (75%)
accuracy = lineardisc.score(X[150:], y[150:]) #predict and assess accuracy (25%)
print(accuracy) #print accuracy
#%%
#LDA for data in p_t - eta space, using sklearn
pt_eta_lda = LDA(n_components = 2) #OBJECT of class LDA with attribute n_components set to 2
pt_eta_lda.fit(train[:, :2], train[:, 2]) #finds best decision boundary based on maximizing inter-class and minimizing intra-class variance

train_accuracy = pt_eta_lda.score(train[:, :2], train[:, 2])
test_accuracy = pt_eta_lda.score(test[:, :2], test[:, 2])
print(train_accuracy, test_accuracy)
#%%
# Plot the hyperplanes and datapoints (including test data)
fig, ax = plt.subplots(figsize=(12,8))
for l,c,m in zip(np.unique(train[:, 2]),['r','g'],['o','x']): #"zip" creates iterator that pairs together values in respective tuples (sp.unique(y) contains class labels)
    ax.scatter(pt_eta[:, :2][pt_eta[:, 2]==l,0],
                pt_eta[:, :2][pt_eta[:, 2]==l,1],
                c=c, marker=m, label=l, s=10)

x1 = np.array([np.amin(pt_eta[:,0], axis=0), np.amax(pt_eta[:,0], axis=0)])

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
        train_g1, train_g2 = train[train[:, 2] == 0], train[train[:, 2] == 1] #training data contains pt data in column 1, eta data in column 2
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
ax.scatter(train[:, 0], train[:, 1], c = train[:, 2], s = 8)

ax.set_title("Naive Bayes classifier performance")
ax.set_xlabel("Transverse momentum")
ax.set_ylabel("Pseudorapidity")  

plt.show()
#%%
