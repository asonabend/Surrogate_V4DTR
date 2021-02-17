

rm(list=ls())
library(dplyr)
library(DynTxRegime)
#setwd('/Users/aaron/Dropbox (HMS)/Documents/Research/RL/DTR/Surrogate_V4DTR/Code/Simulations/')
source('functions.R')
write('args parameters', stderr())
args <- commandArgs(trailingOnly = TRUE)
size <- as.character(args[2]); write(size, stderr())
setting <- as.character(args[4]); write(setting, stderr())
sims_No <- as.integer(args[6]); write(setting, stderr())


################################################################
######################## Simulations ###########################
################################################################

#size=500
#setting <- 'disc'
if (setting!='disc') setting <- as.numeric(setting)
print(size)
print(setting)
#sim_res <- run_sims(size=as.numeric(size),setting='disc',sims_No=100)
#sim_res <- run_sims(size=as.numeric(size),setting=setting,sims_No=10)
sim_res <- run_bowl.Qlearn(size=as.numeric(size),setting=setting,sims_No=sims_No)

# Print tables for latex:
errs <- round(apply(sim_res$errs,2,mean,na.rm=T),2)
errs <- data.frame(cbind(',(',errs[grep('d1',names(errs))],',',errs[grep('d2',names(errs))],')'))
row.names(errs) <- gsub('d1.','',row.names(errs))
print(errs)

Vfuns <- round(apply(sim_res$V_fn,2,mean,na.rm=T),2)
Vfuns <- data.frame(Vfuns); rownames(Vfuns) <- gsub('Estimate.','',rownames(Vfuns))
print(Vfuns)

times <- round(apply(sim_res$time.taken,2,mean,na.rm=T),2)
times <- data.frame(times)
print(times)
size=2*50000;setting=1
df = gen_df(size,setting,sd=1)
write.csv(df,paste('~/Dropbox (HMS)/Documents/Research/RL/DTR/Surrogate_V4DTR/Results/df','size',size,'setting',setting,'seed',sd,'.csv',sep='_'))

#write.csv(df,paste('~/Dropbox (HMS)/Documents/Research/RL/DTR/Surrogate_V4DTR/Results/df','size',size,'setting',setting,'seed',sd,'.csv',sep='_'))


#sim_res <- run_sims(size=as.numeric(500),setting='disc',sims_No=500)

saveRDS(sim_res,paste('sigmoid_res',size,'setting',setting,'.RDS',sep='_'))

