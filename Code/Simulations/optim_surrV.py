# import the required packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import math
import time
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import itertools

from tqdm import tqdm
import os, sys

pandas2ri.activate()
#size,setting,sims_No = 50000,1,100
size,setting,sims_No = int(sys.argv[2]), int(sys.argv[4]), int(sys.argv[6])
sys.stderr.write('\n'+str(size)+' '+str(setting)+' '+str(sims_No)+'\n')
# Import data
#sd=1
#df = pd.read_csv('~/Dropbox (HMS)/Documents/Research/RL/DTR/Surrogate_V4DTR/Results/df_size_'+str(size)+'_setting_'+str(setting)+'_seed_'+str(sd)+'_.csv')
################################################################################################################################
################################################################################################################################

##############################
###################################################################################################
################################################################################################################################
#df.head()
# Generate dataset using R:

#path = '~/Dropbox (HMS)/Documents/Research/RL/DTR/Surrogate_V4DTR/Code/Simulations/'
path = ''
def gen_data(size,setting,sd):
    r = ro.r
    # load the R functions
    r.source(path+"functions.R")
    # generate dataset
    df = r.gen_df(size,setting,sd)
    #pandas2ri.pandas(df)
    # turn it to Pandas df
    #df = com.load_data(df)
    return df

class net(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(net, self).__init__()
        self.fc1 = nn.Linear(inputSize, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, outputSize)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out    

class lm(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(lm, self).__init__()
        self.linear_reg = nn.Linear(inputSize,outputSize)
    def forward(self, x):
        out = self.linear_reg(x)
        return out    

def phi(x,phi_No):
    if phi_No == 1:
        return 1+x/torch.sqrt(1+x**2)
    elif phi_No == 2:
        return 1+x/(1+torch.abs(x))
    elif phi_No == 3:
        return 1+(2/math.pi)*torch.atan(math.pi*x/2)
    else:
        return 1+(2/math.pi)*torch.tanh(math.pi*x/2)

def my_loss(f1_hat,f2_hat,A1,A2,Y1,Y2,phi_No):
    loss = -torch.mean((Y1+Y2)*phi(f1_hat*A1,phi_No)*phi(f2_hat*A2,phi_No))
    #loss = torch.mean((output - target)**2)
    return loss

def simulations(size,setting,sims_No,phi_No,f_model):
    results = []
    #pandas2ri.deactivate()
    for sim in tqdm(range(sims_No)):
        df = gen_data(size,setting,sim)
        df, df_test = df.loc[df.index[:size//2],:], df.loc[df.index[size//2:],:]
        Y1,Y1_test = Variable(torch.from_numpy(df[['Y1']].to_numpy())).float(), Variable(torch.from_numpy(df_test[['Y1']].to_numpy())).float()
        Y2, Y2_test = Variable(torch.from_numpy(df[['Y2']].to_numpy())).float(), Variable(torch.from_numpy(df_test[['Y2']].to_numpy())).float()
        A1, A1_test = Variable(torch.from_numpy(df[['A1']].to_numpy())).float(), Variable(torch.from_numpy(df_test[['A1']].to_numpy())).float()
        A2, A2_test = Variable(torch.from_numpy(df[['A2']].to_numpy())).float(), Variable(torch.from_numpy(df_test[['A2']].to_numpy())).float()
        H1, H1_test = Variable(torch.from_numpy(df[['O1.1', 'O1.2', 'O1.3']].to_numpy())).float(), Variable(torch.from_numpy(df_test[['O1.1', 'O1.2', 'O1.3']].to_numpy())).float()
        if setting == 1:
            H2, H2_test = Variable(torch.from_numpy(df[['O1.1', 'O1.2', 'O1.3', 'Y1']].to_numpy())).float(), Variable(torch.from_numpy(df_test[['O1.1', 'O1.2', 'O1.3', 'Y1']].to_numpy())).float()
        else:
            H2, H2_test = Variable(torch.from_numpy(df[['O1.1', 'O1.2', 'O1.3', 'Y1', 'O2.1', 'O2.2']].to_numpy())).float(), Variable(torch.from_numpy(df_test[['O1.1', 'O1.2', 'O1.3', 'Y1', 'O2.1', 'O2.2']].to_numpy())).float()    
        # Generate loss function
        learningRate = 1 
        epochs = 10
        #
        if f_model == 'NN':
            f1 = net(H1.shape[1], 1)
            f2 = net(H2.shape[1], 1)
        else: # cubic splines
            f1 = lm(H1.shape[1], 1)
            f2 = lm(H2.shape[1], 1)            
        #
        ##### For GPU #######
        if torch.cuda.is_available():
            model.cuda()
        #
        #
        optimizer = torch.optim.SGD(itertools.chain(f1.parameters(), f2.parameters()), lr=learningRate, momentum=0.9)    
        #optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
        #
        start_time = time.clock()
        for epoch in range(epochs):
            # Converting inputs and labels to Variable
            if torch.cuda.is_available():
                inputs = Variable(torch.from_numpy(H1).cuda())
                labels = Variable(torch.from_numpy(y_train).cuda())
            else:
                inputs = H1#Variable(H1).float()
                labels = Y1#Variable(Y1).float()
            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()
            #
            # get output from the model, given the inputs
            f1_hat = f1(H1)
            f2_hat = f2(H2)
            #
            # get loss for the predicted output
            loss = my_loss(f1_hat,f2_hat,A1,A2,Y1,Y2,phi_No)#criterion(outputs, labels)
            # get gradients w.r.t to parameters
            loss.backward()
            #
            # update parameters
            optimizer.step()
            #
            #print('epoch {}, loss {}'.format(epoch, loss.item()))
        end_time = time.clock()
        #
        with torch.no_grad():
            d1_hat = np.sign(f1(H1_test).detach().numpy())
            d2_hat = np.sign(f2(H2_test).detach().numpy())
        #
        d1_err = np.mean(d1_hat.squeeze() != df_test['d1.star'].to_numpy())
        d2_err = np.mean(d2_hat.squeeze() != df_test['d2.star'].to_numpy())
        Rx_dict = {(1,1):['p1p1',0],(-1,1):['n1p1',1],(1,-1):['p1n1',2],(-1,-1):['n1n1',3]}
        Vs_np = df_test.loc[:,['p1p1','n1p1','p1n1','n1n1']].to_numpy()
        V_of_DTR_hat = np.mean([Vs_np[i,Rx_dict[(int(d1_hat[i]),int(d2_hat[i]))][1]] for i in range(len(Vs_np))])
        #
        V_of_DTR_star = np.mean(df_test.loc[:,['p1p1','n1p1','p1n1','n1n1']].max(1))
        #
        results.append([d1_err,d2_err,end_time - start_time,V_of_DTR_hat,V_of_DTR_star])
        #
    results = pd.DataFrame(results, columns = ['d1_err','d2_err','time','V_of_DTR_hat','V_of_DTR_star']) 
    #return results
    results.to_csv('../Results/'+'NN_size_'+str(size)+'_setting_'+str(setting)+'_sims_No_'+str(sims_No)+'.csv')
    sys.stderr.write(str(results.mean()))
