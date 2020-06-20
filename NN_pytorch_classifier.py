# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 22:50:46 2020

@author: lilif
"""
#%%
import torch
import torch.nn.functional as F
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#%%
n = 500
split = int(n/2)
maxiter = 3000
learning_rate = 0.01
D_in, H, D_out = 2, 3, 2
#%%
data, classes = sklearn.datasets.make_moons(n, noise = 0.2)
#%%
plt.title("Sklearn data")
plt.scatter(data[:, 0], data[:, 1], c = classes)
plt.show()
#%%
w_1 = torch.randn(D_in, H)
w_2 = torch.randn(H, D_out)

input_data = torch.from_numpy(data[:split, :])
output_data = torch.from_numpy(classes[:split])
#%%
validation_data = torch.from_numpy(data[split:, :])
validation_classes = torch.from_numpy(classes[split:])
#%%
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer
        #This applies Linear transformation to input data. 
        self.fc1 = torch.nn.Linear(D_in, H)
        
        #This applies linear transformation to produce output data
        self.fc2 = torch.nn.Linear(H, D_out)
        
    #This must be implemented
    def forward(self,x):
        #Output of the first layer
        x = self.fc1(x)
        #Activation function is Relu. Feel free to experiment with this
        x = torch.tanh(x)
        #This produces output
        x = self.fc2(x)
        return x
        
    #This function takes an input and predicts the class, (0 or 1)        
    def predict(self,x):
        #Apply softmax to output
        pred = F.softmax(self.forward(x), dim = 1)
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)
#%%
model = Net()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
#%%
loss_data = []
iters_arr = np.arange(0, maxiter, 1)
iters = 1
#%%
while iters <= maxiter:
    
    y_pred = model(input_data.float())
    loss = loss_fn(y_pred, output_data.long())
    loss_data.append(loss)
    
    optimizer.zero_grad()
    loss.backward(retain_graph = True)
    optimizer.step()
    
    iters += 1
#%%
print("Accuracy score on training set:", accuracy_score(model.predict(input_data.float()), output_data.long()))
print("Accuracy score on validation set:", accuracy_score(model.predict(validation_data.float()), output_data.long()))
#%%
plt.title("Training")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.plot(iters_arr, loss_data)
plt.show()
#%%
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.figure(figsize = (12, 5))
    plt.contourf(xx, yy, Z, cmap = "viridis")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="binary", s=4)
#%%
def predict(x):
 x = torch.from_numpy(x).type(torch.FloatTensor)
 ans = model.predict(x)
 return ans.numpy()
#%%
plot_decision_boundary(lambda x: predict(x), data, classes)
#%%