#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy               as np
import pandas              as pd
import h5py                as h5py
import matplotlib.pyplot   as plt
import torch
import torch.nn            as nn
import torch.nn.functional as F
import pickle

from matplotlib import cm
from matplotlib import colors
from matplotlib.image import NonUniformImage


import os.path
#from   os              import path
from   IPython.display import clear_output
from   numpy.random    import seed
from   sklearn.utils   import shuffle

np.set_printoptions(precision = 5)

devCPU = torch.device("cpu")
dev    = devCPU

from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torch.utils.data import DataLoader, Dataset


plt.rcParams["figure.figsize"] = (6,4)
plt.rcParams["figure.dpi"]     = 200


# In[5]:


file = '../../Data/N30_distEij_db_v1.hdf5'
state='/original/A/'
N='N30/'
category='homopolymer/'
fname='eij_mat'

Eij_homopolymer = np.array( pd.read_hdf(file, state+N+category+fname ) )

Eij_homopolymer = Eij_homopolymer[1:, :]

fname='rowij_mat'
rowij_homopolymer = np.array( pd.read_hdf(file, state+N+category+fname ) )

rowij_homopolymer = rowij_homopolymer [1:, :]

shuffleIdx = shuffle(np.arange(Eij_homopolymer.shape[0]))
Eij_homopolymer    = Eij_homopolymer[shuffleIdx]
rowij_homopolymer  = rowij_homopolymer[shuffleIdx]

Seq            = 10
Eij_Train_1    = Eij_homopolymer[:Seq, :]
rowij_Train_1  = rowij_homopolymer[:Seq,:]

print(Eij_Train_1 [:,:])
print(rowij_Train_1 [:,:])


Eij_Test_1    = Eij_homopolymer[Seq:, :]
rowij_Test_1  = rowij_homopolymer[Seq:, :]

print(Eij_Test_1 [:,:])

X1_Test_1 = torch.from_numpy(Eij_Test_1).float()
Y1_Test_1 = torch.from_numpy(rowij_Test_1).float()


# In[6]:


###############################################################

###file = 'N30_dist_eij_database.hdf5'
state='/original/A/'
N='N30/'
category='fullyrandom/'
fname='eij_mat'

Eij_fullyrandompolymer_A = np.array( pd.read_hdf(file, state+N+category+fname ) )

fname='rowij_mat'
rowij_fullyrandompolymer_A = np.array( pd.read_hdf(file, state+N+category+fname ) )

shuffleIdx = shuffle(np.arange(Eij_fullyrandompolymer_A.shape[0]))
Eij_fullyrandompolymer_A    = Eij_fullyrandompolymer_A[shuffleIdx]
rowij_fullyrandompolymer_A  = rowij_fullyrandompolymer_A[shuffleIdx]

Seq            = 50
Eij_Train_5    = Eij_fullyrandompolymer_A[:Seq, :]
rowij_Train_5  = rowij_fullyrandompolymer_A[:Seq,:]

print(Eij_Train_5 [:5,:10])
print(rowij_Train_5 [:,:])

Eij_Test_5    = Eij_fullyrandompolymer_A[Seq:, :]
rowij_Test_5  = rowij_fullyrandompolymer_A[Seq:, :]

print(Eij_Test_5 [:5,:10])

X1_Test_5 = torch.from_numpy(Eij_Test_5).float()
Y1_Test_5 = torch.from_numpy(rowij_Test_5).float()


# In[4]:


#file = 'N30_dist_eij_database.hdf5'
state='/original/A/'
N='N30/'
category='fourierpolymers/'
fname='eij_mat'

Eij_fourierpolymer_A = np.array( pd.read_hdf(file, state+N+category+fname ) )

fname='rowij_mat'
rowij_fourierpolymer_A = np.array( pd.read_hdf(file, state+N+category+fname ) )

shuffleIdx = shuffle(np.arange(Eij_fourierpolymer_A.shape[0]))
Eij_fourierpolymer_A    = Eij_fourierpolymer_A[shuffleIdx]
rowij_fourierpolymer_A  = rowij_fourierpolymer_A[shuffleIdx]

Seq            = 50
Eij_Train_6    = Eij_fourierpolymer_A[:Seq, :]
rowij_Train_6  = rowij_fourierpolymer_A[:Seq,:]

print(Eij_Train_6 [:5,:])
print(rowij_Train_6 [:,:])

Eij_Test_6    = Eij_fourierpolymer_A[Seq:, :]
rowij_Test_6  = rowij_fourierpolymer_A[Seq:, :]

print(Eij_Test_6 [:5,:])

X1_Test_6 = torch.from_numpy(Eij_Test_6).float()
Y1_Test_6 = torch.from_numpy(rowij_Test_6).float()


# In[7]:


###############################################################
file = '../../Data/N30_dist_eij_DB_FP.hdf5'
state='/original/NewData/'
N='N30/'
category='fourierpolymers_1/'

fname='eij_mat'
Eij_fourierpolymer_1 = np.array( pd.read_hdf(file, state+N+category+fname ) )

fname='rowij_mat'
rowij_fourierpolymer_1 = np.array( pd.read_hdf(file, state+N+category+fname ) )

shuffleIdx = shuffle(np.arange( Eij_fourierpolymer_1.shape[0] ))
Eij_fourierpolymer_1    = Eij_fourierpolymer_1[shuffleIdx]
rowij_fourierpolymer_1  = rowij_fourierpolymer_1[shuffleIdx]

Seq            = 50
Eij_Train_7    = Eij_fourierpolymer_1[:Seq, :]
rowij_Train_7  = rowij_fourierpolymer_1[:Seq,:]

print(Eij_Train_7 [:5,:])
print(rowij_Train_7 [:,:])

Eij_Test_7     = Eij_fourierpolymer_1[Seq:, :]
rowij_Test_7   = rowij_fourierpolymer_1[Seq:, :]

X1_Test_7 = torch.from_numpy(Eij_Test_7).float()
Y1_Test_7 = torch.from_numpy(rowij_Test_7).float()


# In[8]:


###############################################################
#file = 'N30_dist_eij_DB_FP.hdf5'
state='/original/NewData/'
N='N30/'
category='fourierpolymers_2/'

fname='eij_mat'
Eij_fourierpolymer_2 = np.array( pd.read_hdf(file, state+N+category+fname ) )

fname='rowij_mat'
rowij_fourierpolymer_2 = np.array( pd.read_hdf(file, state+N+category+fname ) )

shuffleIdx = shuffle(np.arange( Eij_fourierpolymer_2.shape[0] ))
Eij_fourierpolymer_2    = Eij_fourierpolymer_2[shuffleIdx]
rowij_fourierpolymer_2  = rowij_fourierpolymer_2[shuffleIdx]

#Seq            = 25
Eij_Train_8    = Eij_fourierpolymer_2[:Seq, :]
rowij_Train_8  = rowij_fourierpolymer_2[:Seq,:]

print(Eij_Train_8 [:5,:])
print(rowij_Train_8 [:,:])

Eij_Test_8     = Eij_fourierpolymer_2[Seq:, :]
rowij_Test_8   = rowij_fourierpolymer_2[Seq:, :]

X1_Test_8 = torch.from_numpy(Eij_Test_8).float()
Y1_Test_8 = torch.from_numpy(rowij_Test_8).float()


# In[9]:


###############################################################
#file = 'N30_dist_eij_DB_FP.hdf5'
state='/original/NewData/'
N='N30/'
category='fourierpolymers_3/'

fname='eij_mat'
Eij_fourierpolymer_3 = np.array( pd.read_hdf(file, state+N+category+fname ) )

fname='rowij_mat'
rowij_fourierpolymer_3 = np.array( pd.read_hdf(file, state+N+category+fname ) )

shuffleIdx = shuffle(np.arange( Eij_fourierpolymer_3.shape[0] ))
Eij_fourierpolymer_3    = Eij_fourierpolymer_3[shuffleIdx]
rowij_fourierpolymer_3  = rowij_fourierpolymer_3[shuffleIdx]

#Seq            = 25
Eij_Train_9    = Eij_fourierpolymer_3[:Seq, :]
rowij_Train_9  = rowij_fourierpolymer_3[:Seq,:]

print(Eij_Train_9 [:5,:])
print(rowij_Train_9 [:,:])

Eij_Test_9     = Eij_fourierpolymer_3[Seq:, :]
rowij_Test_9   = rowij_fourierpolymer_3[Seq:, :]

X1_Test_9 = torch.from_numpy(Eij_Test_9).float()
Y1_Test_9 = torch.from_numpy(rowij_Test_9).float()


# In[17]:


###############################################################
#file = 'N30_dist_eij_DB_FP.hdf5'
state='/original/NewData/'
N='N30/'
category='fourierpolymers_4/'

fname='eij_mat'
Eij_fourierpolymer_4 = np.array( pd.read_hdf(file, state+N+category+fname ) )

fname='rowij_mat'
rowij_fourierpolymer_4 = np.array( pd.read_hdf(file, state+N+category+fname ) )

shuffleIdx              = shuffle(np.arange( Eij_fourierpolymer_4.shape[0] ))
Eij_fourierpolymer_4    = Eij_fourierpolymer_4[shuffleIdx]
rowij_fourierpolymer_4  = rowij_fourierpolymer_4[shuffleIdx]

#Seq             = 25
Eij_Train_10    = Eij_fourierpolymer_4[:Seq, :]
rowij_Train_10  = rowij_fourierpolymer_4[:Seq,:]

print(Eij_Train_10 [:5,:])
print(rowij_Train_10 [:,:])

Eij_Test_10     = Eij_fourierpolymer_4[Seq:, :]
rowij_Test_10   = rowij_fourierpolymer_4[Seq:, :]

X1_Test_10      = torch.from_numpy(Eij_Test_10).float()
Y1_Test_10      = torch.from_numpy(rowij_Test_10).float()


# In[11]:


###############################################################
#file = 'N30_dist_eij_DB_FP.hdf5'
state='/original/NewData/'
N='N30/'
category='fourierpolymers_5/'

fname='eij_mat'
Eij_fourierpolymer_5 = np.array( pd.read_hdf(file, state+N+category+fname ) )

fname='rowij_mat'
rowij_fourierpolymer_5 = np.array( pd.read_hdf(file, state+N+category+fname ) )

shuffleIdx              = shuffle(np.arange( Eij_fourierpolymer_5.shape[0] ))
Eij_fourierpolymer_5    = Eij_fourierpolymer_5[shuffleIdx]
rowij_fourierpolymer_5  = rowij_fourierpolymer_5[shuffleIdx]

#Seq             = 25
Eij_Train_11    = Eij_fourierpolymer_5[:Seq, :]
rowij_Train_11  = rowij_fourierpolymer_5[:Seq,:]

print(Eij_Train_11 [:5,:])
print(rowij_Train_11 [:,:])

Eij_Test_11     = Eij_fourierpolymer_5[Seq:, :]
rowij_Test_11   = rowij_fourierpolymer_5[Seq:, :]

X1_Test_11      = torch.from_numpy(Eij_Test_11).float()
Y1_Test_11      = torch.from_numpy(rowij_Test_11).float()


# In[12]:


###############################################################
#file = 'N30_dist_eij_DB_FP.hdf5'
state='/original/NewData/'
N='N30/'
category='fourierpolymers_6/'

fname='eij_mat'
Eij_fourierpolymer_6 = np.array( pd.read_hdf(file, state+N+category+fname ) )

fname='rowij_mat'
rowij_fourierpolymer_6 = np.array( pd.read_hdf(file, state+N+category+fname ) )

shuffleIdx              = shuffle(np.arange( Eij_fourierpolymer_6.shape[0] ))
Eij_fourierpolymer_6    = Eij_fourierpolymer_6[shuffleIdx]
rowij_fourierpolymer_6  = rowij_fourierpolymer_6[shuffleIdx]

Seq             = 50
Eij_Train_12    = Eij_fourierpolymer_6[:Seq, :]
rowij_Train_12  = rowij_fourierpolymer_6[:Seq,:]

print(Eij_Train_12 [:5,:])
print(rowij_Train_12 [:,:])

Eij_Test_12     = Eij_fourierpolymer_6[Seq:, :]
rowij_Test_12   = rowij_fourierpolymer_6[Seq:, :]

X1_Test_12      = torch.from_numpy(Eij_Test_12).float()
Y1_Test_12      = torch.from_numpy(rowij_Test_12).float()


# In[13]:


###############################################################
#file = 'N30_dist_eij_DB_FP.hdf5'
state='/original/NewData/'
N='N30/'
category='fourierpolymers_7/'

fname='eij_mat'
Eij_fourierpolymer_7 = np.array( pd.read_hdf(file, state+N+category+fname ) )

fname='rowij_mat'
rowij_fourierpolymer_7 = np.array( pd.read_hdf(file, state+N+category+fname ) )

shuffleIdx              = shuffle(np.arange( Eij_fourierpolymer_7.shape[0] ))
Eij_fourierpolymer_7    = Eij_fourierpolymer_7[shuffleIdx]
rowij_fourierpolymer_7  = rowij_fourierpolymer_7[shuffleIdx]

Seq             = 50
Eij_Train_13    = Eij_fourierpolymer_7[:Seq, :]
rowij_Train_13  = rowij_fourierpolymer_7[:Seq,:]

print(Eij_Train_13 [:5,:])
print(rowij_Train_13 [:,:])

Eij_Test_13     = Eij_fourierpolymer_7[Seq:, :]
rowij_Test_13   = rowij_fourierpolymer_7[Seq:, :]

X1_Test_13      = torch.from_numpy(Eij_Test_13).float()
Y1_Test_13      = torch.from_numpy(rowij_Test_13).float()


# In[14]:


###############################################################
#file = 'N30_dist_eij_DB_FP.hdf5'
state='/original/NewData/'
N='N30/'
category='fourierpolymers_8/'

fname='eij_mat'
Eij_fourierpolymer_8 = np.array( pd.read_hdf(file, state+N+category+fname ) )

fname='rowij_mat'
rowij_fourierpolymer_8 = np.array( pd.read_hdf(file, state+N+category+fname ) )

shuffleIdx              = shuffle(np.arange( Eij_fourierpolymer_8.shape[0] ))
Eij_fourierpolymer_8    = Eij_fourierpolymer_8[shuffleIdx]
rowij_fourierpolymer_8  = rowij_fourierpolymer_8[shuffleIdx]

Seq             = 50
Eij_Train_14    = Eij_fourierpolymer_8[:Seq, :]
rowij_Train_14  = rowij_fourierpolymer_8[:Seq,:]

print(Eij_Train_14 [:5,:])
print(rowij_Train_14 [:,:])

Eij_Test_14     = Eij_fourierpolymer_8[Seq:, :]
rowij_Test_14   = rowij_fourierpolymer_8[Seq:, :]

X1_Test_14      = torch.from_numpy(Eij_Test_14).float()
Y1_Test_14      = torch.from_numpy(rowij_Test_14).float()


# In[15]:


###############################################################
#file = 'N30_dist_eij_DB_FP.hdf5'
state='/original/NewData/'
N='N30/'
category='fourierpolymers_9/'

fname='eij_mat'
Eij_fourierpolymer_9 = np.array( pd.read_hdf(file, state+N+category+fname ) )

fname='rowij_mat'
rowij_fourierpolymer_9 = np.array( pd.read_hdf(file, state+N+category+fname ) )

shuffleIdx              = shuffle(np.arange( Eij_fourierpolymer_9.shape[0] ))
Eij_fourierpolymer_9    = Eij_fourierpolymer_9[shuffleIdx]
rowij_fourierpolymer_9  = rowij_fourierpolymer_9[shuffleIdx]

Seq             = 50
Eij_Train_15    = Eij_fourierpolymer_9[:Seq, :]
rowij_Train_15  = rowij_fourierpolymer_9[:Seq,:]

print(Eij_Train_15 [:5,:])
print(rowij_Train_15 [:,:])

Eij_Test_15     = Eij_fourierpolymer_9[Seq:, :]
rowij_Test_15   = rowij_fourierpolymer_9[Seq:, :]

X1_Test_15      = torch.from_numpy(Eij_Test_15).float()
Y1_Test_15      = torch.from_numpy(rowij_Test_15).float()


# In[16]:


###############################################################
#file = 'N30_dist_eij_DB_FP.hdf5'
state='/original/NewData/'
N='N30/'
category='fourierpolymers_10/'

fname='eij_mat'
Eij_fourierpolymer_10 = np.array( pd.read_hdf(file, state+N+category+fname ) )

fname='rowij_mat'
rowij_fourierpolymer_10 = np.array( pd.read_hdf(file, state+N+category+fname ) )

shuffleIdx              = shuffle(np.arange( Eij_fourierpolymer_10.shape[0] ))
Eij_fourierpolymer_10    = Eij_fourierpolymer_10[shuffleIdx]
rowij_fourierpolymer_10  = rowij_fourierpolymer_10[shuffleIdx]

Seq             = 50
Eij_Train_16    = Eij_fourierpolymer_10[:Seq, :]
rowij_Train_16  = rowij_fourierpolymer_10[:Seq,:]

print(Eij_Train_16 [:5,:])
print(rowij_Train_16 [:,:])

Eij_Test_16     = Eij_fourierpolymer_10[Seq:, :]
rowij_Test_16   = rowij_fourierpolymer_10[Seq:, :]

X1_Test_16      = torch.from_numpy(Eij_Test_16).float()
Y1_Test_16      = torch.from_numpy(rowij_Test_16).float()


# In[18]:


Eij_Train = np.concatenate((Eij_Train_1, Eij_Train_5,   Eij_Train_6,
                            Eij_Train_7, Eij_Train_8,   Eij_Train_9, Eij_Train_10,
                            Eij_Train_11, Eij_Train_12, Eij_Train_13, Eij_Train_14,
                            Eij_Train_15, Eij_Train_16,), axis= 0)

rowij_Train = np.concatenate((rowij_Train_1, rowij_Train_5, rowij_Train_6,
                              rowij_Train_7, rowij_Train_8,
                              rowij_Train_9, rowij_Train_10,
                              rowij_Train_11, rowij_Train_12,
                              rowij_Train_13, rowij_Train_14,
                              rowij_Train_15, rowij_Train_16,), axis= 0)


Eij_Test = np.concatenate(( Eij_Test_1,  Eij_Test_5,  Eij_Test_6,
                            Eij_Test_7,  Eij_Test_8,
                            Eij_Test_9,  Eij_Test_10,
                            Eij_Test_11, Eij_Test_12,
                            Eij_Test_13, Eij_Test_14,
                            Eij_Test_15, Eij_Test_16, ), axis= 0)

rowij_Test = np.concatenate(( rowij_Test_1,
                              rowij_Test_5,  rowij_Test_6,
                              rowij_Test_7,  rowij_Test_8,
                              rowij_Test_9,  rowij_Test_10,
                              rowij_Test_11, rowij_Test_12,
                              rowij_Test_13, rowij_Test_14,
                              rowij_Test_15, rowij_Test_16,), axis= 0)


#Eij =  np.concatenate((Eij_homopolymer, Eij_GMRpolymer, 
#                       Eij_fullyrandompolymer, Eij_fourierpolymer), axis= 0)

#rowij = np.concatenate((rowij_homopolymer, rowij_GMRpolymer, 
#                        rowij_fullyrandompolymer, rowij_fourierpolymer ), axis= 0)

shuffleIdx = shuffle(np.arange(Eij_Train.shape[0]))
Eij_Train        = Eij_Train[shuffleIdx]
rowij_Train      = rowij_Train[shuffleIdx]

#shuffleIdx = shuffle(np.arange(Eij_Test.shape[0]))
#Eij_Test        = Eij_Test[shuffleIdx]
#rowij_Test      = rowij_Test[shuffleIdx]

#Seq          = 79
#Eij_Train    = Eij[:Seq, :]
#rowij_Train  = rowij[:Seq,:]

#Eij_Test    = Eij[Seq:, :]
#rowij_Test  = rowij[Seq:,:]

print(Eij_Train.shape)
print(rowij_Train.shape)

X1_Train = torch.from_numpy(Eij_Train).float()
Y1_Train = torch.from_numpy(rowij_Train).float()

print(Eij_Test.shape)
print(rowij_Test.shape)

X1_Test = torch.from_numpy(Eij_Test).float()
Y1_Test = torch.from_numpy(rowij_Test).float()


# In[19]:


class XY_Data(Dataset):
    def __init__(self, X, Y):
        #data loading
        self.input   = X
        self.output  = Y
        self.n_samples = X.shape[0]
        
    def __getitem__(self, index):
        # dataset[0]
        return self.input[index], self.output[index]
        
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
TrainDataset = XY_Data(X1_Train, Y1_Train)


# In[20]:


batch_size = 64

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=TrainDataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)


# In[28]:


# Hyper-parameters 
#input_size = 784 # 28x28
hidden_size   = 128
num_layers    = 3
# num_classes nothing but target Sequence size
num_classes   = Y1_Train.shape[1]
num_epochs    = 551
learning_rate = 0.0005
l2 = 1e-5
#input_size = 1
#sequence_length = X1_Train.shape[1]

input_size = 30
sequence_length = 30


device = dev


# In[29]:


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias=False, batch_first=True)
        ## x-> (batchsize, seq, input_size)
        #self.fc = nn.Linear(hidden_size, num_classes, bias=False)
        self.fc1 = nn.Linear(hidden_size,   hidden_size*4, bias=False)
        #self.fc2 = nn.Linear(hidden_size*4, hidden_size*4, bias=False)
        self.fc3 = nn.Linear(hidden_size*4, num_classes, bias=False)
        
         
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)
        #c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)
        
        out, h_n = self.gru(x, h0)
        ## out -> (batchsize, seq_length =1, hidden_size)
        ## h_n -> (num_layer, N, hidden_size)
        
        out = out[:, -1, :]
        ## out (batchsize, hidden_size)
        #out = self.fc(out)
        
        out = self.fc1(out)
        #out = self.fc2(out)
        out = self.fc3(out)
        
        return out


# In[30]:


model = RNN(input_size, hidden_size, num_layers, num_classes)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = l2)


# In[31]:


xTrain = X1_Train
yTrain = Y1_Train

mseHistory = list() # loss history


# In[32]:


for epoch in range(num_epochs):
    
    
    with torch.no_grad():
            if ( epoch % 10 == 0 ):
                
                xTrain   = xTrain.reshape(-1, sequence_length, input_size).to(device)
                yPred    = model(xTrain)
                M        = yTrain.shape[0]*yTrain.shape[1]
                mseTrain = (yPred - yTrain).pow(2).sum()/M
                
                xTest     = X1_Test_1
                yTest     = Y1_Test_1
                xTest     = xTest.reshape(-1, sequence_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_1 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                xTest     = X1_Test_5
                yTest     = Y1_Test_5
                xTest     = xTest.reshape(-1, sequence_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_5 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                xTest     = X1_Test_6
                yTest     = Y1_Test_6
                xTest     = xTest.reshape(-1, sequence_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_6 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                xTest     = X1_Test_7
                yTest     = Y1_Test_7
                xTest     = xTest.reshape(-1, sequence_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_7 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                xTest     = X1_Test_8
                yTest     = Y1_Test_8
                xTest     = xTest.reshape(-1, sequence_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_8 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                xTest     = X1_Test_9
                yTest     = Y1_Test_9
                xTest     = xTest.reshape(-1, sequence_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_9 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                xTest     = X1_Test_10
                yTest     = Y1_Test_10
                xTest     = xTest.reshape(-1, sequence_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_10 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                xTest     = X1_Test
                yTest     = Y1_Test
                xTest     = xTest.reshape(-1, sequence_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest = (yPred -  yTest).pow(2).sum()/M # mean square
                
                
                mseRecord = np.array ( ( epoch, float(mseTrain), float(mseTest_1),  
                                float(mseTest_5), float(mseTest_6), float(mseTest_7),
                                float(mseTest_8), float(mseTest_9), 
                                float(mseTest_10), float(mseTest) ) )
                
                print ( "rmse/kT ~", mseRecord[0], np.sqrt(mseRecord[1:] ))
                mseHistory.append(mseRecord)
                
                if ( epoch% 50 == 0 ):
                    
                    print(xTest.shape)
                    print(yTest.shape)
                    print(yPred.shape)
                    
                    xTest = xTest.reshape(-1, sequence_length*input_size)
                    yTest = yTest.reshape(-1, sequence_length*input_size)
                    yPred = yPred.reshape(-1, sequence_length*input_size)
                
                    np_yPred = yPred.cpu().detach().numpy()
                    yPred_DF =pd.DataFrame(np_yPred)
                
                    np_yTest = yTest.cpu().detach().numpy()
                    yTest_DF =pd.DataFrame(np_yTest)
                    
                    #xTest = xTest.squeeze(2)
                    
                    np_xTest = xTest.cpu().detach().numpy()
                    xTest_DF =pd.DataFrame(np_xTest)
                    
                    fname =  './GRU_N30_AB_data.hdf5'
                    path  =  '/N30/test/s50/'
                
                    yPred_DF.to_hdf(fname, path+'P'+str(epoch),mode='a')
                
                    yTest_DF.to_hdf(fname, path+'T'+str(epoch),mode='a')
                    
                    xTest_DF.to_hdf(fname, path+'eij'+str(epoch),mode='a')
                    
                
    for i, (images, labels) in enumerate(train_loader):  
        # origin shape: [N, seq_length=24]
        # resized: [N,seq_length=1,inputsize=24]
        images = images.reshape(-1, sequence_length, input_size)
        #print(images.shape)
        labels = labels.to(device)
        
        #images = images.double()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #if (i+1) % 100 == 0:
            #print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

DF_msehist =  pd.DataFrame(mseHistory)
DF_msehist.to_hdf('./GRU_N30_mse.hdf5','/N30/s50/test/msehist/',mode='a')


# In[34]:


norm = 1.

GRU_combine_AB = pd.read_hdf('./GRU_N30_mse.hdf5','/N30/s50/test/msehist/')

GRU_combine_AB.columns = ['epoch', 'train_mse', 't_1', 't_5', 't_6', 't_7', 't_8', 't_9',
                           't_10', 'test_mse']


#case_N30 = DF_msehist
case_N30 = GRU_combine_AB

case_N30.columns = ['epoch', 'train_mse', 't_1', 't_5', 't_6', 't_7', 't_8', 't_9',
                           't_10', 'test_mse']

plt.figure(1)
plt.xscale("log")
plt.ylim(0.0, 0.3) 
plt.ylabel("Test RMSE", fontsize = 12)
plt.xlabel("Epochs", fontsize = 12)

plt.plot( case_N30['epoch'], np.sqrt(case_N30['train_mse'])*norm, 'r.--',  ms = 4, lw=1., label="train")

plt.plot( case_N30['epoch'], np.sqrt(case_N30['test_mse'])*norm, '.--', color = '0.5',  ms = 4, lw=1., label="Test rmse")


#plt.title("Sequence length include", fontsize=12)
plt.grid()
plt.legend(loc='best',fontsize=12)
plt.savefig("./RMSE_combineAB_s50.png",dpi=300, )


# In[36]:


fname =  './GRU_N30_AB_data.hdf5'
path  =  '/N30/test/s50/'

Pred_500 = pd.read_hdf( fname, path+'P550')
Test_500 = pd.read_hdf( fname, path+'T550')

Pred_500 = Pred_500.to_numpy()
Test_500 = Test_500.to_numpy()

print(Pred_500.shape)
print(Test_500.shape)


# In[37]:


plt.rcParams["figure.figsize"] = (6,4)
plt.rcParams["figure.dpi"]     = 200

plt.figure(1)
    #plt.xscale("")
plt.ylabel("Predict")
plt.xlabel("Target")        
#plt.plot(Test_500[:,:], Pred_500[:,:], 'r.', ms = 1)




plt.plot(Test_500[:,:], Pred_500[:,:], 'b.', ms = 0.5,)
#plt.plot(Test_500[5,5], Pred_500[5,5], 'b.', ms = 1, label='GeometricMeanRandom B')




plt.plot(Test_500[:,:], Test_500[:,:], 'g.', ms = 0.75)

#plt.plot(Test_500[205:,:], Test_500[205:,:], 'g.', ms = 0.75)


plt.legend(loc='best')
#plt.title(str(Sequence_Bool[rowIndex][:]))
plt.grid()

plt.savefig("./complete_128x4_64.png",dpi=300, bbox_inches='tight' )


# In[74]:


np.sqrt(0.00045008587767370045)


# In[38]:


plt.hist2d(Test_500.reshape(605*900), Pred_500.reshape(605*900), bins=(500, 500), cmap=plt.cm.jet, norm = colors.LogNorm()  )
plt.colorbar()
plt.show
plt.savefig("./2D_Hist_s50.png",dpi=300, bbox_inches='tight' )


#plt.hist2d(x, y, bins=(300, 300), cmap=plt.cm.jet)
#plt.show()


# In[ ]:




