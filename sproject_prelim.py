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
pt_gauss1 = np.random.normal(0.8, 0.3, size=1000)
eta_gauss1 = np.random.normal(1, 0.5, size=1000)
phi_uniform= np.random.uniform(2*np.pi, size = 1000)

pt_gauss2 = np.random.normal(0.7, 0.25, size=1000)
eta_gauss2 = np.random.normal(0.9, 0.1, size=1000)
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
    inv = energy**2 - mtm_mag**2 ### once you have proper data ADD SQRT
    return inv

invm_gun1 = inv_m(gauss_gun1)

invm_gun2 = inv_m(gauss_gun2)
    
                  
#%% plot gun histograms

plt.hist(invm_gun1, bins=40, alpha=0.5, label = "Gaussian Gun 1")
plt.hist(invm_gun2, bins=40, alpha=0.5, label = "Gaussian Gun 2")
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



