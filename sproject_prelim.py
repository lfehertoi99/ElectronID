#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 02:44:23 2020

@author: AliAlAli
"""
import pandas as pd
import numpy as np
import random as rand
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import poisson
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import make_blobs
#%% generating random dists
c = 3*10**8
unit_mass = np.ones(1000) #array of 10000 ones
m_e = 9.11*10**(-31) # mass of electron in kg
m_p = 1.67*10**(-27)

#arrays of 1000
m_e_array = m_e*unit_mass
m_p_array = m_p*unit_mass

### random dists
# For random samples from N(\mu, \sigma^2), use:
#sigma * np.random.randn(...) + mu
pt_gauss1 = np.random.normal(200, 1, size=1000)
eta_gauss1 = np.random.normal(2, 0.2, size=1000)
phi_uniform= np.random.uniform(2*np.pi, size = 1000)

pt_gauss2 = np.random.normal(203, 1.5, size=1000)
eta_gauss2 = np.random.normal(3.5, 0.3, size=1000)
phi_uniform= np.random.uniform(2*np.pi, size = 1000)

pt_po = np.random.poisson(1000, 1000)
eta_po = np.random.poisson(1000, 1000)

#%% gauss MC particle gun

def gauss_gun(m, pt, eta, phi): #all args must be vectors
    gun_fmtm = np.column_stack((m, pt, eta, phi))
    return gun_fmtm

## gun 1
gauss_gun1 = gauss_gun(unit_mass, pt_gauss1, eta_gauss1, phi_uniform)   #containing 1000 4 momt vectors
                   
#gun 2
gauss_gun2 = gauss_gun(unit_mass, pt_gauss2, eta_gauss2, phi_uniform)   #containing 1000 4 momt vectors

def inv_m(fmtm): #must be a 4 vector (m, pt, eta, phi), and is in natural units
    energy = fmtm[:,0]
    mtm_mag = fmtm[:,1]*np.cosh(fmtm[:,2])
    inv = np.sqrt(np.abs(energy**2 - mtm_mag**2)) ### once you have proper data ADD SQRT (added abs to circumvent)
    return inv

invm_gun1 = inv_m(gauss_gun1)

invm_gun2 = inv_m(gauss_gun2)
    
                  
#%% plot gun histograms

plt.hist(invm_gun1, bins=300, alpha=0.5, label = "Gaussian Gun 1")
plt.hist(invm_gun2, bins=300, alpha=0.5, label = "Gaussian Gun 2")
plt.xlabel("Invariant Mass")
plt.ylabel("Number")
plt.legend(loc = "best")

#%% LDA

## our mean vectors are the means of the invm_gun distributions which give the means of the invariant mass dists

mean_gun1= np.mean(invm_gun1)
mean_gun2= np.mean(invm_gun2)

var_gun1 = np.var(invm_gun1)
var_gun2 = np.var(invm_gun2)

fisher_discriminant = (mean_gun1 - mean_gun2)**2 / (var_gun1 + var_gun2)

#%% use make_blobs to generate data

X, y = make_blobs(n_samples=100)
# X stores the data as a column vector and y stores class labels for each sample as an integer
# returns 2 columns, each col with vals for a class
# y values will all be 1 or 2 as n_features = 2 means there are 2 classes

#%% plot blobs

def plot_blobs(X,y):
    df = pd.DataFrame(dict(x=X[:,0],y=X[:,1], label=y)) #can use dict as have only 2 classes
    # label is y as y contains the classes
    colours = {0: "red", 1:"green", 2:"blue"} #dictionary relating class labels to plot colours
    fig, ax = plt.subplots(figsize=(12,8))

    grouped = df.groupby('label') #to make plot, we need to group datapoints according to their class (=label)
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colours[key], s=10) #x and y are strings since they are the names of df columns
    
    plt.show()


