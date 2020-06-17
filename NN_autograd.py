# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 22:52:36 2020

@author: lilif
"""
#%%
import torch
import random
import matplotlib.pyplot as plt
import scipy as sp
#%%
learning_rate = 10e-8
threshold = 10e-2
trials = 5
#%%
for k in range(trials):
    input_data = torch.randn(10, 1)
    output_data = torch.randn(5, 1)
    
    w1 = torch.randn(10, 8, requires_grad = True)
    w2 = torch.randn(8, 5, requires_grad = True)
    
    my_loss = 100
    loss_data = []
    iters = 0
    
    while (my_loss >= threshold) and (iters <= 1e4):
        
        y_hat = torch.mm(w2.t(), torch.clamp(torch.mm(w1.t(), input_data), 0))
        
        loss = (y_hat - output_data).pow(2).sum()
        my_loss = loss.item()
        #print(my_loss)
        loss_data.append(loss.item())
        
        loss.backward()
        
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
        
        iters+=1

    iterations = sp.arange(0, len(loss_data), 1)
    
    plt.figure(figsize = (12, 8))
    plt.plot(iterations, loss_data)
#%%
