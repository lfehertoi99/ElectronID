# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:50:05 2020

@author: lilif
"""
#%%
import scipy as sp
import random
import matplotlib.pyplot as plt
#%%
input_data = sp.random.randn(10, 1)
output_data = sp.random.randn(5, 1)
#%%
threshold = 0.5
learning_rate = 0.000001
#%%
weights1 = sp.random.normal(size = (10, 8))
weights2 = sp.random.normal(size = (8, 5))
#%%
loss_data = []
#%%
loss = 100
while loss >= threshold:
    
        z1 = sp.dot(weights1.T, input_data) #8 x 1
        a1 = sp.maximum(0, z1) #8 x 1
        z2 = sp.dot(weights2.T, a1) #5 x 1
        a2 = sp.maximum(0, z2) #5 x 1
        
        loss = 1/len(output_data) * sp.sum((a2 - output_data)**2)
        loss_data.append(loss)
        
        grad_a2 = 2 * (a2 - output_data) #gradient w.r.t y_hat,dim n x 1
        grad_z2 = sp.multiply(grad_a2, sp.maximum(0, z2)/z2) #gradient w.r.t z2, dim n x 1
        grad_w2 = sp.dot(a1, grad_z2.T) #gradient w.r.t. w2, dim m x n
        grad_a1 = sp.dot(weights2, grad_z2) #m x 1
        grad_z1 = sp.multiply(grad_a1, sp.maximum(0, z1)/z1) #m x 1
        grad_w1 = sp.dot(input_data, grad_z1.T) #p x m
        
        weights1 -= learning_rate * grad_w1
        weights2 -= learning_rate * grad_w2
#%%
iterations = sp.arange(0, len(loss_data), 1)
#%%
plt.plot(iterations, loss_data)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.show()