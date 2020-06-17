# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 20:58:41 2020

@author: lilif
"""
#%%
import scipy as sp
import matplotlib.pyplot as plt
import random
#%%
domain = sp.arange(-5, 5, 0.0001)
max_index = len(domain)
#X, Y = sp.meshgrid(domain, domain)
#%%
def rosenbrock(x, y, a = 1, b = 100):
    return (a - x)**2 + b*(y - x**2)**2
#%%
def grad(x, y, a = 1, b = 100):
    x_grad = -2*(a - x) - 4*b*x*(y - x**2)
    y_grad = 2*b*(y - x**2)
    return sp.array([x_grad, y_grad])
#%%
learning_rate = 1e-4
threshold = 1e-3
max_iter = 1e3
#%%
pos = sp.random.choice(domain, size = 2)
#%%
val = rosenbrock(pos[0], pos[1])
iter = 0
vals = []
#%%
while iter <= max_iter:

    gradient = grad(pos[0], pos[1])
    
    pos -= sp.dot(learning_rate, gradient)
    val = rosenbrock(pos[0], pos[1])
    
    vals.append(val)
    iter +=1
#%%
iterations = sp.arange(0, len(vals), 1)
#%%
plt.figure(figsize = (12, 8))
plt.plot(iterations, vals)
plt.xlabel("Iterations")
plt.ylabel("Objective function value")
plt.show()
#%%
print(pos)