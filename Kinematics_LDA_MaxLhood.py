# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:36:16 2020

@author: lilif
"""
#%%
import pandas as pd
import scipy as sp
import random as rand
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import poisson
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import make_blobs
#%%
#Generating 4-momentum vectors with coordinates p_t, eta, phi, m
#calculate invariant mass for one particle
def inv_m(fmtm):
    energy = fmtm[3]
    mom_mag = fmtm[0]*sp.cosh(fmtm[1])
    inv = sp.sqrt(sp.absolute(energy**2 - mom_mag**2)) #both should have units of ev
    return inv
#calculate invariant mass for system of particles (argument should be vector)
def inv_m(FMTM):
    energy = FMTM[:,3]
    mom_mag = FMTM[:,0]*sp.cosh(FMTM[:,1])
    inv = sp.sqrt(sp.absolute(energy**2 - mom_mag**2))
    return inv
#%%
#Particle MC guns: phi uniformly distributed, different pt and eta distributions for each gun (both normal with different shape parameters)
    
normal = lambda x,y: norm(x,y)
uniform_d = lambda x,y: uniform(loc=x, scale=y)

g1_eta = normal(2, 0.2).rvs(size=1000)
g2_eta = normal(3.5, 0.3).rvs(size=1000)

g1_pt = normal(200, 1).rvs(size=1000)
g2_pt = normal(203, 1.5).rvs(size=1000)

g1_phi = uniform_d(0, 2*sp.pi).rvs(size=1000)
g2_phi = uniform_d(0, 2*sp.pi).rvs(size=1000)

mass = sp.ones(1000)
#%%
def g_dist(pt, eta, phi, m): #all arguments should be vectors!
    g_fmtm = sp.column_stack((pt, eta, phi, m))
    return g_fmtm
#%%
experiment1 = g_dist(g1_pt, g1_eta, g1_phi, mass)
experiment2 = g_dist(g2_pt, g2_eta, g2_phi, mass)
#%%
mass_1 = inv_m(experiment1)
mass_2 = inv_m(experiment2)
#%%
histogram = lambda x,y: sp.histogram(x, bins=y, density=True)
#Histogram of invariant masses from both guns
hist_1, bin_edges_1 = histogram(mass_1, 300)
hist_2, bin_edges_2 = histogram(mass_2, 300)
#%%
#PLOT distributions of invariant mass from both particle guns
fig, ax = plt.subplots(1, 1, sharex = True, sharey=True, figsize=(12,8))

ax.bar(bin_edges_1[1:], hist_1, width=5, label="Gun 1")
ax.bar(bin_edges_2[1:], hist_2, width=5, label="Gun 2")

plt.legend()
plt.show()
#%%
#TRIAL RUN OF LDA IN SKLEARN
#use sklearn create blobs to quickly generate data using make_blobs funciton
X, y = make_blobs(n_samples=100, centers=3, n_features=2) 
#X contains a column vector for each feature, y contains integer class labels for each sample
#%%
#plot blobs
def plot_clusters(X, y):
    df = pd.DataFrame(dict(x=X[:,0],y=X[:,1], label=y)) #can plot this way bc we have only 2 features, dict can be specified directly using key-value pairs
    colors = {0: "red", 1:"green", 2:"blue"} #dictionary relating class labels to plot colors

    fig, ax = plt.subplots(figsize=(12,8))

    grouped = df.groupby('label') #to make plot, we need to group datapoints according to their class (=label)
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key], s=10) #x and y are strings since they are the names of df columns
    
    plt.show()
#%%
lineardisc = LDA(n_components = 2) #OBJECT of class LDA with attribute n_components set to 2
lineartrans = lineardisc.fit(X, y)
#%%
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
pt_vals = sp.concatenate((g1_pt, g2_pt))
eta_vals = sp.concatenate((g1_eta, g2_eta))

X_pt_eta = sp.column_stack((pt_vals, eta_vals))

g1_labels = sp.zeros(1000)
g2_labels = sp.ones(1000)

y_pt_eta = sp.concatenate((g1_labels, g2_labels))
#%%
print(X_pt_eta)
print(y_pt_eta)
#%%
#LDA for data in p_t - eta space
pt_eta_lda = LDA(n_components = 2) #OBJECT of class LDA with attribute n_components set to 2
pt_eta_fit = pt_eta_lda.fit(X_pt_eta, y_pt_eta) #finds best decision boundary based on maximizing inter-class and minimizing intra-class variance
#%%
# Plot the hyperplanes
fig, ax = plt.subplots(figsize=(12,8))
for l,c,m in zip(sp.unique(y_pt_eta),['r','g'],['o','x']): #"zip" creates iterator that pairs together values in respective tuples (sp.unique(y) contains class labels)
    ax.scatter(X_pt_eta[y_pt_eta==l,0],
                X_pt_eta[y_pt_eta==l,1],
                c=c, marker=m, label=l, s=10)

x1 = sp.array([sp.amin(X_pt_eta[:,0], axis=0), sp.amax(X_pt_eta[:,0], axis=0)])

#lda.coef_ contains coefficients of decision boundary vectors
b, w1, w2 = pt_eta_lda.intercept_[0], pt_eta_lda.coef_[0][0], pt_eta_lda.coef_[0][1]
y1 = -(b+x1*w1)/w2    
ax.plot(x1,y1,c="k")

ax.set_xlabel("Transverse Momentum")
ax.set_ylabel("Pseudorapidity")
ax.legend()

plt.show()
#%%
#CREATE LIKELIHOOD-BASED ESTIMATOR: GET BEST ESTIMATE PDF FOR EACH DISTRIBUTION, look at likelihood ratios.
#1 get estimators from training sample
def n_params(data):
    
    n_mean = sp.mean(data)
    n_std = sp.std(data, ddof = 1) #ddof = 1: use Bessel-corrected version of estimator to account for sampling error
    
    return [n_mean, n_std]
#2 get log-likelihood for each datapoint, given estimates of function parameters
def n_log_lhood(data, mean, std):

    lhood_func = sp.stats.norm(loc = mean, scale = std)
    log_lhood_vec = sp.log(lhood_func.pdf(data))
    
    return log_lhood_vec
#get log-likelihood ratio of having come from gun 1 vs gun 2
def lhood_ratio(data, params):
    
    g1_m, g1_std, g2_m, g2_std = params
    
    g1_lhood = n_log_lhood(data, g1_m, g1_std)
    g2_lhood = n_log_lhood(data, g2_m, g2_std)
    
    lhood_ratio = g1_lhood - g2_lhood
    
    return lhood_ratio
#%%
g1_pt_params = n_params(g1_pt)
g2_pt_params = n_params(g2_pt)

g1_eta_params = n_params(g1_eta)
g2_eta_params = n_params(g2_eta)
#%%
#Draw new samples from MC particle guns for VALIDATAION of classifier

g1_eta_val = normal(2, 0.2).rvs(size=1000)
g2_eta_val = normal(3.5, 0.3).rvs(size=1000)

g1_pt_val = normal(200, 1).rvs(size=1000)
g2_pt_val = normal(203, 1.5).rvs(size=1000)

eta_val = sp.concatenate((g1_eta_val, g2_eta_val))
pt_val = sp.concatenate((g1_pt_val, g2_pt_val))

val_data = sp.column_stack((pt_val, eta_val))
#%%
log_lhood_ratio_pt = lhood_ratio(val_data[:,0], g1_pt_params + g2_pt_params)
log_lhood_ratio_eta = lhood_ratio(val_data[:,1], g1_eta_params + g2_eta_params)
#%%
tot_lhood_ratio = log_lhood_ratio_pt  + log_lhood_ratio_eta
#%%
fig, ax = plt.subplots(figsize = (12, 8))

ax.scatter(X_pt_eta[:, 0], X_pt_eta[:, 1], c=tot_lhood_ratio, cmap="RdPu", s=8)

ax.set_xlabel("Transverse momentum")
ax.set_ylabel("Pseudorapidity")

plt.show()
#%%
#CLASSIFY
def classify(data, tot_lhood_ratio):
    
    class_labels = sp.zeros(len(tot_lhood_ratio))
    
    for i in range(len(class_labels)):
        if tot_lhood_ratio[i] >= 0:
            class_labels[i] = 0
        if tot_lhood_ratio[i] < 0:
            class_labels[i] = 1
    
    labeled_data = sp.column_stack((data, class_labels))
    
    return labeled_data
#%%
lab_exp_data = classify(val_data, tot_lhood_ratio)
#%%        
#MAKE SCATTER PLOT TO VISUALIZE PERFORMANCE
fig, ax = plt.subplots(figsize=(12,8))

for l,c,m in zip(sp.unique(lab_exp_data[:,2]),['r','g'],['o','x']): #"zip" creates iterator that pairs together values in respective tuples (sp.unique(y) contains class labels)
    ax.scatter(lab_exp_data[:,0:2][lab_exp_data[:,2]==l,0],
                lab_exp_data[:,0:2][lab_exp_data[:,2]==l,1],
                c=c, marker=m, label=l, s=8)

ax.set_xlabel("Transverse momentum")
ax.set_ylabel("Pseudorapidity")  

plt.legend()
plt.show()
#%%
#unit-variance and zero-mean datasets
def standardize(array, axis=0):
    mean = sp.mean(array, axis)
    std = sp.std(array, axis)
    
    standard = (array - mean)/std
    
    return standard
#%%
def standardcopy(array, axis=0):
    copy = sp.copy(array)
    copy = standardize(array, axis)
    
    return copy
#%%
