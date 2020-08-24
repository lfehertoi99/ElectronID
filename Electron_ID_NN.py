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
import torch
import torch.nn.functional as F
from sklearn import datasets
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
#%%
torch.cuda.device(0)
#%%
#Reading ROOT files
#open directory
file = uproot.open(r"output_1.root")
file.keys()
#get tree
tree = file["ntuplizer"]
tree.keys()
#get branches with data in them
branch = tree["tree"]
branch.keys()
#%%
all_vars = branch.pandas.df(["gsf_bdtout1", "eid_rho", "eid_ele_pt", "eid_sc_eta", "eid_shape_full5x5_sigmaIetaIeta", 
                             "eid_shape_full5x5_sigmaIphiIphi", "eid_shape_full5x5_circularity", "eid_shape_full5x5_r9", 
                             "eid_sc_etaWidth", "eid_sc_phiWidth", "eid_shape_full5x5_HoverE", "eid_trk_nhits", "eid_trk_chi2red",
                             "eid_gsf_chi2red", "eid_brem_frac", "eid_gsf_nhits", "eid_match_SC_EoverP", "eid_match_eclu_EoverP",
                             "eid_match_SC_dEta", "eid_match_SC_dPhi", "eid_match_seed_dEta", "eid_sc_E", "eid_trk_p", "is_e"])
#filter out unwanted values
all_vars = all_vars.replace([np.inf, -np.inf, -10, -666, -999], np.nan)
all_vars = all_vars.dropna()
#%%
select = np.random.rand(len(all_vars)) < 0.02
all_vars = all_vars[select]
#%%
all_vars_arr = np.array(all_vars).reshape(-1, 24)
all_X, all_y = all_vars_arr[:, :-1], all_vars_arr[:, -1].astype(int)
#%%
sm = SMOTE()
X_res, y_res = sm.fit_resample(all_X, all_y)
#%%
bitmask = np.random.rand(len(X_res)) < 0.8
train_d, train_l = X_res[bitmask], y_res[bitmask]
test_d, test_l = X_res[~bitmask], y_res[~bitmask]
#%%
train_d, train_l = torch.from_numpy(train_d.astype(float)).cuda(), torch.from_numpy(train_l.astype(float)).cuda()
test_d, test_l = torch.from_numpy(test_d.astype(float)).cuda(), torch.from_numpy(test_l.astype(float)).cuda()
#%%
max_epochs = 500 #max iterations for backprop
learning_rate = 0.01 #learning rate
D_in, H, D_out = 23, 3, 2 #number of neurons in each layer (1 hidden layer)
#%%
#randomly initialize weights
w_1 = torch.randn(D_in, H).cuda()
w_2 = torch.randn(H, D_out).cuda()
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
#model parameters
model = Net().cuda()
loss_fn = torch.nn.CrossEntropyLoss() #output is likelihood, so use CEL
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
#%%
train_loss_data = [] #array to store training loss data
validation_loss_data = [] #array to store validation loss data
iters_arr = np.arange(0, max_epochs, 1)
iters = 1
#%%
#optimization
for epoch in range(max_epochs):
    #TRAINING
    model.train()
    y_pred = model(train_d.float()) #get predicted values
    loss = loss_fn(y_pred, train_l.long()) #calculate loss
    train_loss_data.append(loss) #store loss
    
    optimizer.zero_grad() #zero gradients
    loss.backward() #backprop
    optimizer.step() #update model parameters
    
    #VALIDATION
    model.eval()
    y_pred = model(test_d.float())
    loss = loss_fn(y_pred, test_l.long())
    validation_loss_data.append(loss)
#%%
#get accuracy scores (after training is complete) of 
print("Accuracy score on training set:", accuracy_score(model.predict(train_d.float()).cpu(), train_l.long().cpu()))
print("Accuracy score on validation set:", accuracy_score(model.predict(test_d.float()).cpu(), test_l.long().cpu()))
#%%
#plot training against no. of epochs
plt.title("Performance on training and validation data")
plt.plot(iters_arr, train_loss_data, label = "training")
plt.plot(iters_arr, validation_loss_data, label = "validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
