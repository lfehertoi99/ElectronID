# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:57:27 2020

@author: lilif
"""
#%%
import scipy as sp
import random
import matplotlib.pyplot as plt
import torch
#%%
dtype = torch.float
device = torch.device("cpu")
#%%
input_data = torch.randn(10, 1)
output_data = torch.randn(5, 1)
#%%
threshold = 0.5
learning_rate = 0.000001
#%%
weights1 = torch.randn(10, 8)
weights2 = torch.randn(8, 5)
#%%
loss_data = []
#%%
loss = 100
while loss >= threshold:
    
        z1 = torch.mm(weights1.t(), input_data) #8 x 1
        a1 = torch.clamp(z1, 0) #8 x 1
        z2 = torch.mm(weights2.t(), a1) #5 x 1
        a2 = torch.clamp(z2, 0) #5 x 1
        
        loss = 1/len(output_data) * torch.sum((a2 - output_data)**2)
        loss_data.append(loss)
        
        grad_a2 = 2 * (a2 - output_data) #gradient w.r.t y_hat,dim n x 1
        grad_z2 = grad_a2 * torch.clamp(z2, 0)/z2 #gradient w.r.t z2, dim n x 1
        grad_w2 = torch.mm(a1, grad_z2.t()) #gradient w.r.t. w2, dim m x n
        grad_a1 = torch.mm(weights2, grad_z2) #m x 1
        grad_z1 = grad_a1 * torch.clamp(z1, 0)/z1 #m x 1
        grad_w1 = torch.mm(input_data, grad_z1.t()) #p x m
        
        weights1 -= learning_rate * grad_w1
        weights2 -= learning_rate * grad_w2
#%%
iterations = sp.arange(0, len(loss_data), 1)
#%%
plt.plot(iterations, loss_data)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.show()

