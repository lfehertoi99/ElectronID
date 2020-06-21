#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:55:18 2020

@author: AliAlAli
"""
import torch
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn import svm, datasets
import matplotlib.pyplot as plt
#%% getting actual data from lili
#from Electron_ID_data import file, output_data

#%%getting Kevin LDA data

from LDA_numpy import setosa, versicolor, setosa_t, versicolor_t, y

#%% skeleton for plotting ROC curve using Lili LDA

from Kinematics_LDA_MaxLhood import y_pt_eta, pt_eta_fit, pt_eta_lda ,X_pt_eta, y1
#X_pt_eta contains 2000 arrays of 2 values: pt and eta for each datapoint 
y_score_pt_eta = pt_eta_lda.decision_function(X_pt_eta) #input should have shape (n_samples, n_features)

n_classes = 2
n_samples = 2000
# Compute ROC curve and ROC area for each class

fpr, tpr, _ = roc_curve(y_pt_eta, y_score_pt_eta)
roc_auc = auc(fpr, tpr) #first arg is x coords, 2nd is y coords
#%%
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
