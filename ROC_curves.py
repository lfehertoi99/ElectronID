#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:55:18 2020

@author: AliAlAli
"""
import torch
from sklearn.metrics import roc_curve, roc_auc_score
#%% getting actual data from lili
from Electron_ID_data import file, output_data

#%%getting Kevin LDA data

from LDA_numpy import setosa, versicolor, setosa_t, versicolor_t

#%% test on Kevin's LDA

setosa_auc = roc_auc_score(setosa, setosa_t )
