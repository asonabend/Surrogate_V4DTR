# import the required packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import math
import random
import time
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import itertools
import pywt
from tqdm import tqdm
import os, sys

pandas2ri.activate()
#sys.stderr.write('\n'+str(size)+' '+str(setting)+' '+str(sims_No)+'\n')
# Import data
#sd=1
#df = pd.read_csv('~/Dropbox (HMS)/Documents/Research/RL/DTR/Surrogate_V4DTR/Results/df_size_'+str(size)+'_setting_'+str(setting)+'_seed_'+str(sd)+'_.csv')
################################################################################################################################
################################################################################################################################

#import  lightgbm
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
        self.fc1 = nn.Linear(inputSize, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, outputSize)
        self.DO = nn.Dropout(p=0.5)
    def forward(self, x):
        x = F.relu(self.DO(self.fc1(x)))
        x = F.relu(self.DO(self.fc2(x)))
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

def batches(N, batch_size,seed):
    seq = [i for i in range(N)]
    random.seed(seed)
    if seed is not None:
        random.shuffle(seq)
    return (torch.tensor(seq[pos:pos + batch_size]) for pos in range(0, N, batch_size))

def pre_process(df):
    
##
Wavs, Wavs_test = pywt.dwtn(H1, 'db10',axes=[1]), pywt.dwtn(H1_test, 'db10',axes=[1])
features, features_test = np.hstack((Wavs['a'],Wavs['d'])), np.hstack((Wavs_test['a'],Wavs_test['d']))
##
def simulations(size,setting,sims_No,phi_No,f_model,learningRate = .1,epochs = 10):
    results = []
    for sim in tqdm(range(sims_No)):
        torch.manual_seed(sim)
        df = gen_data(size,setting,sim)
        df, df_test = df.loc[df.index[:size//2],:], df.loc[df.index[size//2:],:]
        Y1, Y1_test = Variable(torch.from_numpy(df[['Y1']].to_numpy())).float(), Variable(torch.from_numpy(df_test[['Y1']].to_numpy())).float()
        Y2, Y2_test = Variable(torch.from_numpy(df[['Y2']].to_numpy())).float(), Variable(torch.from_numpy(df_test[['Y2']].to_numpy())).float()
        A1, A1_test = Variable(torch.from_numpy(df[['A1']].to_numpy())).float(), Variable(torch.from_numpy(df_test[['A1']].to_numpy())).float()
        A2, A2_test = Variable(torch.from_numpy(df[['A2']].to_numpy())).float(), Variable(torch.from_numpy(df_test[['A2']].to_numpy())).float()
        H1, H1_test = Variable(torch.from_numpy(df[['O1.1', 'O1.2', 'O1.3']].to_numpy())).float(), Variable(torch.from_numpy(df_test[['O1.1', 'O1.2', 'O1.3']].to_numpy())).float()
        if setting == 1:
            H2, H2_test = Variable(torch.from_numpy(df[['O1.1', 'O1.2', 'O1.3', 'Y1']].to_numpy())).float(), Variable(torch.from_numpy(df_test[['O1.1', 'O1.2', 'O1.3', 'Y1']].to_numpy())).float()
        else:
            H2, H2_test = Variable(torch.from_numpy(df[['O1.1', 'O1.2', 'O1.3', 'Y1', 'O2.1', 'O2.2']].to_numpy())).float(), Variable(torch.from_numpy(df_test[['O1.1', 'O1.2', 'O1.3', 'Y1', 'O2.1', 'O2.2']].to_numpy())).float()            
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
            f1.cuda()
            f2.cuda()
        #
        #
        optimizer = torch.optim.SGD(itertools.chain(f1.parameters(), f2.parameters()), lr=learningRate, momentum=0.9)    
        #optimizer = torch.optim.RMSprop(itertools.chain(f1.parameters(), f2.parameters()), lr=learningRate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        
        #            
        start_time = time.clock()
        for epoch in range(epochs):
            for batch_indx in batches(N=H1.shape[0],batch_size=128,seed=epoch):
                H1_batch,H2_batch = torch.index_select(H1, 0, batch_indx),torch.index_select(H2, 0, batch_indx)
                Y1_batch,Y2_batch = torch.index_select(Y1, 0, batch_indx),torch.index_select(Y2, 0, batch_indx)
                A1_batch,A2_batch = torch.index_select(A1, 0, batch_indx),torch.index_select(A2, 0, batch_indx)
                '''
                # Converting inputs and labels to Variable
                if torch.cuda.is_available():
                    inputs = Variable(torch.from_numpy(H1_batch).cuda())
                    labels = Variable(torch.from_numpy(Y1_batch).cuda())
                else:
                    inputs = H1_batch#Variable(H1).float()
                    labels = Y1_batch#Variable(Y1).float()
                '''
                # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
                optimizer.zero_grad()
                #
                # get output from the model, given the inputs
                f1_hat = f1(H1_batch)
                f2_hat = f2(H2_batch)
                #
                # get loss for the predicted output
                loss = my_loss(f1_hat,f2_hat,A1_batch,A2_batch,Y1_batch,Y2_batch,phi_No)#criterion(outputs, labels)
                # get gradients w.r.t to parameters
                loss.backward()
                #
                # update parameters
                optimizer.step()
                #
                #print('epoch {}, loss {}'.format(epoch, loss.item()))
        end_time = time.clock()
        #
        d1_hat,d2_hat= np.array([]), np.array([])
        with torch.no_grad():
        #for batch_indx in batches(N=H1_test.shape[0],batch_size=128*4,seed=None):
            d1_hat = np.append(d1_hat,np.sign(f1(H1_test).detach().numpy()))
            d2_hat = np.append(d2_hat,np.sign(f2(H2_test).detach().numpy()))
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
    results.to_csv('../Results/'+'NN_size_'+str(size//2)+'_setting_'+str(setting)+'_sims_No_'+str(sims_No)+'.csv')
    sys.stderr.write(str(results.mean()))
    return results



##############################
########## SOWL ##############
##############################

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
def SOWL_simulations(size,setting,sims_No,gamma = 1):
    results = []
    for sim in tqdm(range(sims_No)):
        torch.manual_seed(sim)
        df = gen_data(size,setting,sim)
        df, df_test = df.loc[df.index[:size//2],:], df.loc[df.index[size//2:],:]
        Y1, Y1_test = df[['Y1']].to_numpy(), df_test[['Y1']].to_numpy()
        Y2, Y2_test = df[['Y2']].to_numpy(), df_test[['Y2']].to_numpy()
        A1, A1_test = df[['A1']].to_numpy(), df_test[['A1']].to_numpy()
        A2, A2_test = df[['A2']].to_numpy(), df_test[['A2']].to_numpy()
        H1, H1_test = df[['O1.1', 'O1.2', 'O1.3']].to_numpy(), df_test[['O1.1', 'O1.2', 'O1.3']].to_numpy()
        if setting == 1:
            H2, H2_test = df[['O1.1', 'O1.2', 'O1.3', 'Y1']].to_numpy(), df_test[['O1.1', 'O1.2', 'O1.3', 'Y1']].to_numpy()
        else:
            H2, H2_test = df[['O1.1', 'O1.2', 'O1.3', 'Y1', 'O2.1', 'O2.2']].to_numpy(), df_test[['O1.1', 'O1.2', 'O1.3', 'Y1', 'O2.1', 'O2.2']].to_numpy()
        #
        n = H2.shape[0]
        ##########
        ########## Objective function: Min (1/2)alpha^T * P * alpha + q^T alpha
        ##########
        ##### P matrix
        # Compute Kernel matrices
        K1 = rbf_kernel(H1) 
        K2 = rbf_kernel(H2)
        # Compute treatment matrices where A1A1t_(i,j) = AiAj
        A1A1t = A1.dot(A1.T)
        A2A2t = A2.dot(A2.T)
        # Elementwise (Kronecker) product:
        A1A1K1 = np.multiply(A1A1t,K1)
        A2A2K2 = np.multiply(A2A2t,K2)
        # Large weight matrix: for (alpha^T)*P*alpha
        P = np.block([
            [A1A1K1,                 np.zeros(A1A1K1.shape)],
            [np.zeros(A1A1K1.shape), A2A2K2               ]
            ])
        #Converting into cvxopt format
        P = cvxopt_matrix(P)
        ##### q vector
        # This vector is negative as we're minimizing in the cvxopt objective function
        q = cvxopt_matrix(-np.ones((2*n, 1)))
        ##########
        ########## Inequality constraint G * alpha <= h
        ##########
        ##### G matrix
        # G matrix such that G*alpha = (alpha1+alpha2,-alpha1,-alhpa2)
        G = np.block([[np.eye(n),np.eye(n)],
                    [-np.eye(n),np.zeros((n,n))],
                    [np.zeros((n,n)),-np.eye(n)]])
        #Converting into cvxopt format
        G = cvxopt_matrix(G)              
        # Weight vector (still need to adjust by propensity score)
        pi1,pi2 = .5, .5
        W = (Y1+Y2)/(pi1*pi2)
        ##### h vector
        # h vector such that G*alpha <= h where h = (gamma*W,0,0)
        h = cvxopt_matrix(np.vstack((gamma*W,np.zeros((2*n,1)))))
        ##########
        ########## Equality constraint A * alpha = b
        ##########
        ##### A matrix
        # A matrix such that A * alpha = (A1^T * alpha1, A2^T * alpha2)
        A = np.block([[A1.T,np.zeros((A1.T.shape))],
                    [np.zeros((A2.T.shape)),A2.T]])
        #Converting into cvxopt format
        A = cvxopt_matrix(A)              
        ##### b matrix
        b = cvxopt_matrix(np.zeros(2))
        #Setting solver parameters (change default to decrease tolerance) 
        cvxopt_solvers.options['show_progress'] = False
        cvxopt_solvers.options['abstol'] = 1e-10
        cvxopt_solvers.options['reltol'] = 1e-10
        cvxopt_solvers.options['feastol'] = 1e-10
        #Run solver
        start_time = time.clock()
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        end_time = time.clock()
        alphas = np.array(sol['x'])
        #==================Test===============================#
        # Select support vectors
        S = (alphas > 1e-4).flatten()
        alph_indx1 = S[:n]
        alph_indx2 = S[n:]
        # alpha_k*A_k, k=1,2 vectors
        A1alph1 = (A1*alphas[:n])[alph_indx1]
        A2alph2 = (A2*alphas[n:])[alph_indx2]
        # RBF kernel matrtices
        K1_test = rbf_kernel(H1,H1_test)[alph_indx1,:]
        K2_test = rbf_kernel(H2,H2_test)[alph_indx2,:]
        #
        d1_hat = np.sign(A1alph1.T.dot(K1_test)).T 
        d2_hat = np.sign(A2alph2.T.dot(K2_test)).T
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
    results = pd.DataFrame(results, columns = ['d1_err','d2_err','time','V_of_DTR_hat','V_of_DTR_star']) 
    results.to_csv('../Results/'+'SOWL_size_'+str(size//2)+'_setting_'+str(setting)+'_sims_No_'+str(sims_No)+'.csv')
    sys.stderr.write(str(results.mean()))
    return results

import lightgbm
def GBM_simulations(size,setting,sims_No,gamma = 1):
    results = []
    for sim in tqdm(range(sims_No)):
        torch.manual_seed(sim)
        df = gen_data(size,setting,sim)
        df, df_test = df.loc[df.index[:size//2],:], df.loc[df.index[size//2:],:]
        Y1, Y1_test = df[['Y1']].to_numpy(), df_test[['Y1']].to_numpy()
        Y2, Y2_test = df[['Y2']].to_numpy(), df_test[['Y2']].to_numpy()
        A1, A1_test = df[['A1']].to_numpy(), df_test[['A1']].to_numpy()
        A2, A2_test = df[['A2']].to_numpy(), df_test[['A2']].to_numpy()
        H1, H1_test = df[['O1.1', 'O1.2', 'O1.3']].to_numpy(), df_test[['O1.1', 'O1.2', 'O1.3']].to_numpy()
        if setting == 1:
            H2, H2_test = df[['O1.1', 'O1.2', 'O1.3', 'Y1']].to_numpy(), df_test[['O1.1', 'O1.2', 'O1.3', 'Y1']].to_numpy()
        else:
            H2, H2_test = df[['O1.1', 'O1.2', 'O1.3', 'Y1', 'O2.1', 'O2.2']].to_numpy(), df_test[['O1.1', 'O1.2', 'O1.3', 'Y1', 'O2.1', 'O2.2']].to_numpy()
        #
        n = H2.shape[0]
        ##########
        ##########
        ##########
        def phi_np1(x):
            return 1+x/np.sqrt(1+x**2)
        
        def phi_np(x,phi_No):
            if phi_No == 1:
                return 1+x/np.sqrt(1+x**2)
            elif phi_No == 2:
                return 1+x/(1+np.abs(x))
            elif phi_No == 3:
                return 1+(2/math.pi)*np.arctan(math.pi*x/2)
            else:
                return 1+(2/math.pi)*torch.tanh(math.pi*x/2)

        def my_loss(f1_hat,f2_hat,A1,A2,Y1,Y2,phi_No):
            loss = -np.mean((Y1+Y2)*phi_np1(f1_hat*A1)*phi_np1(f2_hat*A2))            
            #loss = -np.mean((Y1+Y2)*phi_np(f1_hat*A1,phi_No)*phi_np(f2_hat*A2,phi_No))
            return loss



        def custom_asymmetric_train(y_true, y_pred):
            residual = (y_true - y_pred).astype("float")
            grad = np.where(residual<0, -2*10.0*residual, -2*residual)
            hess = np.where(residual<0, 2*10.0, 2.0)
            return grad, hess
        # default lightgbm model with sklearn api
        gbm = lightgbm.LGBMRegressor() 
        # updating objective function to custom
        # default is "regression"
        # also adding metrics to check different scores
        gbm.set_params(**{'objective': my_loss}, metrics = ["mse", 'mae'])

        # fitting model 
        gbm.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=custom_asymmetric_valid,
            verbose=False,
        )
        ##########
        ##########
        ##########
        ########## Objective function: Min (1/2)alpha^T * P * alpha + q^T alpha
        ##########
        ##### P matrix
        # Compute Kernel matrices
        K1 = rbf_kernel(H1) 
        K2 = rbf_kernel(H2)
        # Compute treatment matrices where A1A1t_(i,j) = AiAj
        A1A1t = A1.dot(A1.T)
        A2A2t = A2.dot(A2.T)
        # Elementwise (Kronecker) product:
        A1A1K1 = np.multiply(A1A1t,K1)
        A2A2K2 = np.multiply(A2A2t,K2)
        # Large weight matrix: for (alpha^T)*P*alpha
        P = np.block([
            [A1A1K1,                 np.zeros(A1A1K1.shape)],
            [np.zeros(A1A1K1.shape), A2A2K2               ]
            ])
        #Converting into cvxopt format
        P = cvxopt_matrix(P)
        ##### q vector
        # This vector is negative as we're minimizing in the cvxopt objective function
        q = cvxopt_matrix(-np.ones((2*n, 1)))
        ##########
        ########## Inequality constraint G * alpha <= h
        ##########
        ##### G matrix
        # G matrix such that G*alpha = (alpha1+alpha2,-alpha1,-alhpa2)
        G = np.block([[np.eye(n),np.eye(n)],
                    [-np.eye(n),np.zeros((n,n))],
                    [np.zeros((n,n)),-np.eye(n)]])
        #Converting into cvxopt format
        G = cvxopt_matrix(G)              
        # Weight vector (still need to adjust by propensity score)
        pi1,pi2 = .5, .5
        W = (Y1+Y2)/(pi1*pi2)
        ##### h vector
        # h vector such that G*alpha <= h where h = (gamma*W,0,0)
        h = cvxopt_matrix(np.vstack((gamma*W,np.zeros((2*n,1)))))
        ##########
        ########## Equality constraint A * alpha = b
        ##########
        ##### A matrix
        # A matrix such that A * alpha = (A1^T * alpha1, A2^T * alpha2)
        A = np.block([[A1.T,np.zeros((A1.T.shape))],
                    [np.zeros((A2.T.shape)),A2.T]])
        #Converting into cvxopt format
        A = cvxopt_matrix(A)              
        ##### b matrix
        b = cvxopt_matrix(np.zeros(2))
        #Setting solver parameters (change default to decrease tolerance) 
        cvxopt_solvers.options['show_progress'] = False
        cvxopt_solvers.options['abstol'] = 1e-10
        cvxopt_solvers.options['reltol'] = 1e-10
        cvxopt_solvers.options['feastol'] = 1e-10
        #Run solver
        start_time = time.clock()
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        end_time = time.clock()
        alphas = np.array(sol['x'])
        #==================Test===============================#
        # Select support vectors
        S = (alphas > 1e-4).flatten()
        alph_indx1 = S[:n]
        alph_indx2 = S[n:]
        # alpha_k*A_k, k=1,2 vectors
        A1alph1 = (A1*alphas[:n])[alph_indx1]
        A2alph2 = (A2*alphas[n:])[alph_indx2]
        # RBF kernel matrtices
        K1_test = rbf_kernel(H1,H1_test)[alph_indx1,:]
        K2_test = rbf_kernel(H2,H2_test)[alph_indx2,:]
        #
        d1_hat = np.sign(A1alph1.T.dot(K1_test)).T 
        d2_hat = np.sign(A2alph2.T.dot(K2_test)).T
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
    results = pd.DataFrame(results, columns = ['d1_err','d2_err','time','V_of_DTR_hat','V_of_DTR_star']) 
    results.to_csv('../Results/'+'SOWL_size_'+str(size//2)+'_setting_'+str(setting)+'_sims_No_'+str(sims_No)+'.csv')
    sys.stderr.write(str(results.mean()))
    return results
####

####
from utils import *
import os, sys
size,setting,sims_No = 500,1,5
#size,setting,sims_No = 50000,1,100
size,sims_No,method = int(sys.argv[2]),  int(sys.argv[6]), str(sys.argv[8])
setting = int(sys.argv[4]) if len(sys.argv[4])==1 else str(sys.argv[4])


if method == 'NN':
    sys.stderr.write('\n Running NN method with ')
    sys.stderr.write('n = '+str(size)+', on setting '+str(setting)+', for '+str(sims_No)+' datasets\n') 
    _ = simulations(size=size,setting=setting,sims_No=sims_No,phi_No=1,f_model='NN',learningRate = .001,epochs = 20)
elif method == 'BE':
    sys.stderr.write('\n Running BE method with ')
    sys.stderr.write('n = '+str(size)+', on setting '+str(setting)+', for '+str(sims_No)+' datasets\n') 
    _ = simulations(size=size,setting=setting,sims_No=sims_No,phi_No=1,f_model='BE',learningRate = .1,epochs = 10)
elif method == 'SOWL':
    sys.stderr.write('\n Running SOWL method ')     
    sys.stderr.write('n = '+str(size)+', on setting '+str(setting)+', for '+str(sims_No)+' datasets\n') 
    _ = SOWL_simulations(size=size,setting=setting,sims_No=sims_No)
else:
    sys.stderr.write('\n No method provided, provide method ("NN" or "SOWL") \n')

#SOWL_simulations(size=1000,setting='disc',sims_No=10,gamma=1.5)
#simulations(size=1000,setting='disc',sims_No=10,phi_No=1,f_model='NN',learningRate = .1,epochs = 10)

###################
###################
###################
from GBM import *
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

# Definition of Hyper-Parameters
NUM_CLASSIFIERS = 5
MAX_DEPTH = 4
GRADIENT_BOOST_LEARNING_RATE = 0.1
MINIMIZER_LEARNING_RATE = 0.005
MINIMIZER_TRAINING_EPOCHS = 1000

# Read the training data
df = pd.read_csv("Titanic.csv", sep=",")
X = df.loc[:, ["Age", "Fare", "Pclass"]]
y = df.loc[:, "Survived"]
X = np.nan_to_num(X, 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
