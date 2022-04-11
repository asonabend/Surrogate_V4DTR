
'''
if method == 'NN': 
   sys.stderr.write('\n Running NN method with ')
   sys.stderr.write('n = '+str(size)+', on setting '+str(setting)+', for '+str(sims_No)+' datasets\n')  
   _ = simulations(size=size,setting=setting,sims_No=sims_No,phi_No=1,f_model='NN',learningRate = .1,epochs = 10)
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
'''
# import the required functions and packages:
from utils import *
import os, sys
if method == 'wavelets': 
    # Wavelets
    sys.stderr.write('\n\nWavelets:\n')
    sys.stderr.write('n = '+str(size)+', on setting '+str(setting)+', for '+str(sims_No)+' datasets\n')  
    _ = simulations(size=size,setting=setting,sims_No=sims_No,phi_No=1,f_model='wavelets',learningRate = .001,epochs =20,l1_reg=False)

    # Wavelets Lasso
    sys.stderr.write('\n\nWavelets Lasso:\n')
    sys.stderr.write('n = '+str(size)+', on setting '+str(setting)+', for '+str(sims_No)+' datasets\n')  
    _ = simulations(size=size,setting=setting,sims_No=sims_No,phi_No=1,f_model='wavelets',learningRate = .001,epochs =20,l1_reg=True,lamb=.01,alpha=1)#alpha = 1 is rigde, 0 is lasso

    # Neural network
    sys.stderr.write('\n\nNN:\n')
    sys.stderr.write('n = '+str(size)+', on setting '+str(setting)+', for '+str(sims_No)+' datasets\n')  
    _ = simulations(size=size,setting=setting,sims_No=sims_No,phi_No=1,f_model='NN',learningRate = .001,epochs =20)

    # Q-learning
    sys.stderr.write('\n\nQ-learning:\n')
    sys.stderr.write('n = '+str(size)+', on setting '+str(setting)+', for '+str(sims_No)+' datasets\n')  
    _ = simulations(size=size,setting=setting,sims_No=sims_No,phi_No=None,f_model='Qlearning',learningRate = .001,epochs =20)

    # linear function
    sys.stderr.write('\n\nlinear:\n')
    sys.stderr.write('n = '+str(size)+', on setting '+str(setting)+', for '+str(sims_No)+' datasets\n')  
    _ = simulations(size=size,setting=setting,sims_No=sims_No,phi_No=1,f_model='linear',learningRate = .001,epochs =20)

# SOWL
#sys.stderr.write('\n\nSOWL:\n')
#sys.stderr.write('n = '+str(size)+', on setting '+str(setting)+', for '+str(sims_No)+' datasets\n')  
#_ = simulations(size=size,setting=setting,sims_No=sims_No,phi_No=None,f_model='SOWL')


if method == 'NN': 
   sys.stderr.write('\n Running NN method with ')
   sys.stderr.write('n = '+str(size)+', on setting '+str(setting)+', for '+str(sims_No)+' datasets\n')  
   _ = simulations(size=size,setting=setting,sims_No=sims_No,phi_No=1,f_model='NN',learningRate = .1,epochs = 10)
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



# import the required functions and packages:
from utils import *
import os, sys
size,setting,sims_No = 250,'disc',500
# Deep Q-learning
print('\n\nDeep Q-learning NN:\n')
_ = simulations(size=size,setting=setting,sims_No=sims_No,f_model='DQlearning',learningRate = .001,epochs =20)
# Neural network
print('\n\nNN:\n')
_ = simulations(size=size,setting=setting,sims_No=sims_No,f_model='NN',learningRate = .001,epochs =20,phi_No=None,CV_K=5,cnvx_lss=None)
# Cubic splines
print('\n\nCubic splines:\n')
_ = simulations(size=size,setting=setting,sims_No=sims_No,f_model='splines',learningRate = .001,epochs =20,phi_No=None,CV_K=5)
# linear function
print('\n\nlinear:\n')
_ = simulations(size=size,setting=setting,sims_No=sims_No,f_model='linear',learningRate = .01,epochs =20,phi_No=None,CV_K=5)
# Wavelets
print('\n\nWavelets:\n')
_ = simulations(size=size,setting=setting,sims_No=sims_No,f_model='wavelets',learningRate = .01,epochs =20,l1_reg=False,phi_No=None,CV_K=5)
#  linear Q-learning
print('\n\nlinear Q-learning:\n')
_ = simulations(size=size,setting=setting,sims_No=sims_No,f_model='linQlearning',learningRate = .01,epochs =20)
# SOWL with linear fns
print('\n\nSOWL linear:\n')
_ = simulations(size=size,setting=setting,sims_No=sims_No,f_model='SOWLlinear')
# SOWL with RBF fns
print('\n\nSOWL RBF:\n')
_ = simulations(size=size,setting=setting,sims_No=sims_No,f_model='SOWLRBF')
# BOWL with linear fns
print('\n\nBOWL linear:\n')
_ = simulations(size=size,setting=setting,sims_No=sims_No,f_model='BOWLlinear')
# BOWL with RBF fns
print('\n\nBOWL RBF:\n')
_ = simulations(size=size,setting=setting,sims_No=sims_No,f_model='BOWLradial')

# Wavelets Lasso
print('\n\nWavelets Lasso:\n')
_ = simulations(size=size,setting=setting,sims_No=sims_No,phi_No=5,f_model='wavelets',learningRate = .01,epochs =20,l1_reg=True,lamb=.001,alpha=0)#alpha = 1 is rigde, 0 is lasso

#pd_results = pd.DataFrame(_, columns = ['d1_err','d2_err','time','V_of_DTR_hat','V_hat_of_DTR_hat','V_of_DTR_star','std_V_of_DTR_hat','std_V_of_DTR_star']) 

###########
###########
###########
'''
size,setting,sims_No = 50000,1,1
method='NN'
'''
from utils import *
import os, sys
size,sims_No,method = int(sys.argv[2]),  int(sys.argv[6]), str(sys.argv[8])
setting = int(sys.argv[4]) if len(sys.argv[4])==1 else str(sys.argv[4])
if method in ['linear','NN','wavelets','SOWL','DQlearning','linQlearning']:    
    learningRate = 0.01 if method in ['linear','linQlearning'] and setting != 1 else .001
    phi_No = 1 if method in ['linear','NN','wavelets'] else None
    learningRate, epochs = (learningRate,20) if method != 'SOWL' else (None,None)
    #if method == 'linear' and setting == 1:
    #    learningRate /=10
    if phi_No == 1:
        sys.stderr.write('\n Running '+'surrogate loss with '+method+' functions')
    else:
        sys.stderr.write('\n Running '+method+' benchmark')
    sys.stderr.write('\nn = '+str(size)+', on setting '+str(setting)+', for '+str(sims_No)+' datasets\n')     
    _ = simulations(size=size,setting=setting,sims_No=sims_No,phi_No=phi_No,f_model=method,learningRate = learningRate,epochs =epochs)
else:
    sys.stderr.write('Error: method not valid')


counts =  0
for method in ['linear','NN','splines','wavelets','BOWLlinear','BOWLradial','SOWLRBF','SOWLlinear','linQlearning','DQlearning']:
    for sample in [250,2500,5000]:
        phi_No,cnvx_lss = None,None
        for setting in ['1','2','3','4','5','disc','6']:#['disc','6']:#['10','11','12','13']+
            print('\nsbatch run_med_sims.sh '+str(sample)+' '+ setting +' 500 '+ method+' '+str(phi_No)+' '+str(cnvx_lss)+'\n\n\n\n\n\n\n')      
            counts += 1

counts
            for cnvx_lss in [1,2,3,4]:
            print('\nsbatch run_med_sims.sh '+str(sample)+' '+ setting +' 500 '+ method+' None')      
            if method in ['NN','splines','wavelets','linear']:
                for phi_No in [1,2,3,4,5]:
                    print('\nsbatch run_med_sims.sh '+str(sample)+' '+ setting +' 500 '+ method+' '+str(phi_No))      
                    counts += 1
            else:
                    print('\nsbatch run_med_sims.sh '+str(sample)+' '+ setting +' 500 '+ method+' None')      
                    counts += 1
                    
                
counts                
'''
method,sample = 'SOWL',500
for setting in ['disc','1','2','3','4']:
    print('\nsbatch run_short_sims.sh '+str(sample)+' '+ setting +' 500 '+ method)

for sample in [5000,50000]:
    for setting in ['disc','1','2','3','4']:
        print('\nsbatch run_med_sims.sh '+str(sample)+' '+ setting +' 500 '+ method)
'''

for method in ['NN']:
    for sample in [500,1000,2000,5000,10000,50000]:
        for setting in ['3','5']:
            print('\nsbatch run_med_sims.sh '+str(sample)+' '+ setting +' 500 '+ method)      
