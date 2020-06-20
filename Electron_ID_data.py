# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 13:10:06 2020

@author: lilif
"""
#%%
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Reading ROOT files
#%%
#open directory
file = uproot.open(r"C:\Users\lilif\OneDrive\Documents\Physics\Computing\output_1.root")
file.keys()
#get tree
tree = file["ntuplizer"]
tree.keys()
#get branches with data in them
branch = tree["tree"]
branch.keys()
#%%
#make dfs from variables of interest. how can we add all efficiently?
label = branch.pandas.df(["is_e"])
eta = branch.pandas.df(["trk_pt"])
pt = branch.pandas.df(["trk_eta"])
#%%
#this will be the size of the training sample and validation sample
data_size = 5000
#%%
#get training data from dfs
label_arr = np.array(label[: 2 * data_size])
eta_arr = np.array(eta[: 2 * data_size])
pt_arr = np.array(pt[: 2 * data_size])
#%%
#function to find centre of data (i.e. 2d mean), does not necessarily correspond to a datapoint
def get_centre(eta_arr, pt_arr):
    eta_mean = np.mean(eta_arr)
    pt_mean = np.mean(pt_arr)
    return eta_mean, pt_mean
#%%
#get numpy array holding the distance from the centre of each datapoint
def get_dist(eta_arr, pt_arr):
    eta_mean, pt_mean = get_centre(eta_arr, pt_arr)
    dist =  (pt_arr - pt_mean)**2 + (eta_arr - eta_mean)**2
    return dist
#%%
#get distance column for training data
dist = get_dist(eta_arr, pt_arr)
#%%
#create dataframe including distances
data = pd.DataFrame({"Label" : label_arr.flatten(), "Eta" : eta_arr.flatten(), "Transverse momentum" : pt_arr.flatten(), "Euclidean distance" : dist.flatten()})
#%%
#remove outliers that are extremely far from centre of data
q = data["Euclidean distance"].quantile(.95)
removed_outliers = data[data["Euclidean distance"] < q]
#%%
#turn data into numpy arrays
new_eta = np.array(removed_outliers["Eta"])
new_pt = np.array(removed_outliers["Transverse momentum"])
new_labels = np.array(removed_outliers["Label"])
#%%
#only electrons
ele_data = removed_outliers[removed_outliers["Label"] == True]
ele_eta = np.array(ele_data["Eta"])
ele_pt = np.array(ele_data["Transverse momentum"])
#only background
non_ele_data = removed_outliers[removed_outliers["Label"] == False]
non_ele_eta = np.array(non_ele_data["Eta"])
non_ele_pt = np.array(non_ele_data["Transverse momentum"])
#%%
#plot kinematic distributions for only electrons and only background
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.set_title("Electrons")
ax1.scatter(ele_eta, ele_pt, s=1.5, c="r")
ax1.set_xlabel("Eta")
ax1.set_ylabel("Transverse momentum")

ax2.set_title("Background")
ax2.scatter(non_ele_eta, non_ele_pt, s=1.5, c="r")
ax2.set_xlabel("Eta")
ax2.set_ylabel("Transverse momentum")

plt.show()
#%%
#%%
#imports for torch
import torch
import torch.nn.functional as F
import sklearn
from sklearn import datasets
from sklearn.metrics import accuracy_score
#%%
#put data into torch tensors to pass into NN
training_coordinates = torch.from_numpy(np.column_stack((new_eta, new_pt)).astype(float)[:data_size])
training_labels = torch.from_numpy(new_labels.astype(float)[:data_size])

validation_coordinates = torch.from_numpy(np.column_stack((new_eta, new_pt)).astype(float)[data_size:])
validation_labels = torch.from_numpy(new_labels.astype(float)[data_size:])
#%%
#n = 500
#split = int(n/2)
maxiter = 3000 #max iterations for backprop
learning_rate = 0.01 #learning rate
D_in, H, D_out = 2, 3, 2 #number of neurons in each layer (1 hidden layer)
#%%
#data, classes = sklearn.datasets.make_moons(n, noise = 0.2)
#%%
#plt.title("Sklearn data")
#plt.scatter(data[:, 0], data[:, 1], c = classes)
#plt.show()
#%%
#randomly initialize weights
w_1 = torch.randn(D_in, H)
w_2 = torch.randn(H, D_out)
#%%
#get input data and ground truths in torch tensor

#input_data = torch.from_numpy(data[:split, :])
#output_data = torch.from_numpy(classes[:split])
input_data = training_coordinates
output_data = training_labels

#validation_data = torch.from_numpy(data[split:, :])
#validation_classes = torch.from_numpy(classes[split:])
validation_data = validation_coordinates
validation_classes = validation_labels
#%%
#define Net class
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
loss_fn = torch.nn.CrossEntropyLoss() #output is likelihood, so use CEL
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
#%%
loss_data = [] #array in which to store data
iters_arr = np.arange(0, maxiter, 1)
iters = 1
#%%
#optimization
while iters <= maxiter:
    
    y_pred = model(input_data.float()) #get predicted values
    loss = loss_fn(y_pred, output_data.long()) #calculate loss
    loss_data.append(loss) #store loss
    
    optimizer.zero_grad() #zero gradients
    loss.backward(retain_graph = True) #backprop
    optimizer.step() #update model parameters
    
    iters += 1
#%%
print("Accuracy score on training set:", accuracy_score(model.predict(input_data.float()), output_data.long()))
print("Accuracy score on validation set:", accuracy_score(model.predict(validation_data.float()), validation_classes.long()))
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
plot_decision_boundary(lambda x: predict(x), input_data, output_data)
#%%