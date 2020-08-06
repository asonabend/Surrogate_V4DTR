################################################################
########################## Functions ###########################
################################################################
library(splines)
library(dplyr)
expit <- function(x) exp(x)/(1+exp(x))
gen_data <- function(size,setting,seed){
  set.seed(seed)
  O1_dim <- 3
  #50-dimensional baseline covariatesX1,1, . . . , X1,50 are generated according to N(0, 1).
  O1 <- matrix(rnorm(size*O1_dim,0,1),nrow=size,O1_dim)
  
  #O1 <- cbind(rbinom(size,1,.2),rbinom(size,10,.4),rbinom(size,10,.8))
  
  # Treatments A1, A2 are randomly generated from {−1, 1} with equal probability 0.5.
  trt <- rbinom(2*size,1,.5); trt[trt==0] <- -1
  A1 <- trt[1:size]; A2 <- trt[(size+1):(2*size)]
  
  # The models for generating outcomes R1 and R2 vary under the different settings stated below:
  # Setting 1) 
  if(setting==1){
    c1 <- 3;c2 <- 15
    # Stage 1 outcome Y1 is generated according to: N(0.5*X_13*A1, 1), 
    Y1_mean <- function(O1,A1) .5*O1[,3]*A1 + c1
    Y1 <- Y1_mean(O1,A1) + rnorm(size,0,1)
    # and Stage 2 outcome Y2 is generated according to N((((X_11)^2 + (X_12)^2 − 0.2)(0.5 − (X_11)^2 − (X_12)^1) + Y1)*A2, 1).
    Y2_mean <- function(O1,Y1,A2) ((O1[,1]^2 + O1[,2]^2 - 0.2)*(0.5 - O1[,1]^2 - O1[,2]^2) + Y1)*A2 + c2
    Y2 <- Y2_mean(O1,Y1,A2) + rnorm(size,0,1)
    O2.1 <- O2.2 <- rep(NA,size)
    taus <- cbind(p1p1=Y1_mean(O1,A1=1)+Y2_mean(O1,Y1,A2=1),
                  n1p1=Y1_mean(O1,A1=-1)+Y2_mean(O1,Y1,A2=1),
                  p1n1=Y1_mean(O1,A1=1)+Y2_mean(O1,Y1,A2=-1),
                  n1n1=Y1_mean(O1,A1=-1)+Y2_mean(O1,Y1,A2=-1))
  }else if(setting==2){
    cnst <- 10
    # Setting 2) 
    # Stage 1 outcome Y1 is generated according to: N((1 +1.5X_13)*A1, 1);
    Y1_mean <- function(O1,A1) (1 +1.5*O1[,3])*A1 + cnst
    #Y1_mean <- function(O1,A1) O1[,3]*A1 + cnst
    Y1 <- Y1_mean(O1,A1) + rnorm(size,0,1)
    # two intermediate variables, O2,1 ∼ I {N(1.25*X_11*A1, 1) > 0}, and O_22 ∼ I {N(−1.75*O_12*A1, 1) > 0} are generated;
    O2.1 <- as.numeric(1.25*O1[,1]*A1+rnorm(size,0,1)>0); O2.2 <- as.numeric(-1.75*O1[,2]*A1+rnorm(size,0,1)>0)
    # then the Stage 2 outcome Y2 is generated according to N((0.5 + Y1 + 0.5*A1 +0.5*X_21 − 0.5*X2,2)*A2, 1).
    Y2_mean <- function(O1,Y1,A2) (0.5 + Y1 + 0.5*A1 +0.5*O2.1 - 0.5*O2.2)*A2 + cnst# previous one (Zhang et al)
    #Y2_mean <- function(O1,Y1,A2) -O1[,2]*A2 + cnst
    Y2 <- Y2_mean(O1,Y1,A2) + rnorm(size,0,1)
    taus <- cbind(p1p1=Y1_mean(O1,A1=1)+Y2_mean(O1,Y1,A2=1),
                  n1p1=Y1_mean(O1,A1=-1)+Y2_mean(O1,Y1,A2=1),
                  p1n1=Y1_mean(O1,A1=1)+Y2_mean(O1,Y1,A2=-1),
                  n1n1=Y1_mean(O1,A1=-1)+Y2_mean(O1,Y1,A2=-1))
    
  }
  d.argmax <- c('tau_+1+1','tau_-1+1','tau_+1-1','tau_-1-1')[apply(taus,1,which.max)]
  d.argmin <- c('tau_+1+1','tau_-1+1','tau_+1-1','tau_-1-1')[apply(taus,1,which.min)]
  df <- data.frame('O1'=O1,A1,Y1,O2.1,O2.2,A2,Y2,taus,d.argmax,d.argmin)
  df <- df %>% mutate(d1.star=if_else(substr(d.argmax,5,6)=='+1',1,-1),
                      d2.star=if_else(substr(d.argmax,7,8)=='+1',1,-1)) 
  
  return(df)
}

gen_disc_data <- function(size,seed,ret='all'){
  set.seed(seed)
  c1 <- 1; c2 <- 1
  # baseline covariates X1,1, . . . , X1,3 are generated according to Bern(p).
  O1.1 <- rbinom(size,1,.5); O1.1[O1.1==0] <- -1
  O1.2 <- rbinom(size,1,.5); O1.2[O1.2==0] <- -1
  O1.3 <- rbinom(size,1,.5); O1.2[O1.3==0] <- -1
  O1 <- cbind(O1.1,O1.2,O1.3)
  # Treatments A1, A2 are randomly generated from {−1, 1} with equal probability 0.5.
  trt <- rbinom(2*size,1,.5); trt[trt==0] <- -1
  A1 <- trt[1:size]; A2 <- trt[(size+1):(2*size)]
  
  # The model for generating outcomes R1 and R2 is defined under the setting stated below:
  
  # Setting 2) 
  # Stage 1 outcome Y1 is generated according to: N((1 +1.5X_13)*A1, 1);
  Y1_mean <- function(O1,A1) expit((O1[,3]-.5*O1[,1])*A1)#expit((1 +1.5*O1[,1]-2.5*O1[,2])*A1)#
  Y1 <- rbinom(size,1,Y1_mean(O1,A1))*c1
  # two intermediate variables, O2,1 ∼ I {N(1.25*X_11*A1, 1) > 0}, and O_22 ∼ I {N(−1.75*O_12*A1, 1) > 0} are generated;
  O2.1 <- as.numeric(1.25*O1[,1]*A1+rnorm(size,0,1)>0); O2.2 <- as.numeric(-1.75*O1[,2]*A1+rnorm(size,0,1)>0)
  # then the Stage 2 outcome Y2 is generated according  to N((0.5 + Y1 + 0.5*A1 +0.5*X_21 − 0.5*X2,2)*A2, 1).
  Y2_mean <- function(O1,O2.2,Y1,A1,A2) expit((.5*O1[,1]+O1[,2]-.2*O2.2+.5*A1+Y1)*A2)#expit((0.5 + Y1  -5*O2.1 - 4.5*O2.2)*A2) #+ 0.5*A1#
  Y2 <- rbinom(size,1,Y2_mean(O1,O2.2,Y1,A1,A2))*c2
  taus <- cbind(p1p1=Y1_mean(O1,A1=1)*c1+Y2_mean(O1,O2.2,Y1,A1=1,A2=1)*c2,
                n1p1=Y1_mean(O1,A1=-1)*c1+Y2_mean(O1,O2.2,Y1,A1=-1,A2=1)*c2,
                p1n1=Y1_mean(O1,A1=1)*c1+Y2_mean(O1,O2.2,Y1,A1=1,A2=-1)*c2,
                n1n1=Y1_mean(O1,A1=-1)*c1+Y2_mean(O1,O2.2,Y1,A1=-1,A2=-1)*c2)
  
  
  d.argmax <- c('tau_+1+1','tau_-1+1','tau_+1-1','tau_-1-1')[apply(taus,1,which.max)]
  d.argmin <- c('tau_+1+1','tau_-1+1','tau_+1-1','tau_-1-1')[apply(taus,1,which.min)]
  df <- data.frame(O1,A1,Y1,O2.1,O2.2,A2,Y2,taus,d.argmax,d.argmin)
  df <- df %>% mutate(d1.star=if_else(substr(d.argmax,5,6)=='+1',1,-1),
                      d2.star=if_else(substr(d.argmax,7,8)=='+1',1,-1)) 
  
  if (ret!='all') df <- taus
  return(df)
}

gen_toy_data <- function(size,seed,ret='all'){
  set.seed(seed)
  # Treatments A1, A2 are randomly generated from {−1, 1} with equal probability 0.5.
  trt <- rbinom(2*size,1,.5); trt[trt==0] <- -1
  A1 <- trt[1:size]; A2 <- trt[(size+1):(2*size)]
  
  # The model for generating outcomes R1 and R2 is defined under the setting stated below:
  
  # Setting 2) 
  
  Y1_mean <- function(A1,A2) 1
  Y1 <- 1
  # then the Stage 2 outcome Y2 is generated according  to N((0.5 + Y1 + 0.5*A1 +0.5*X_21 − 0.5*X2,2)*A2, 1).
  Y2_mean <- function(A1,A2) 4*(A1==1 & A2==1)+3*(A1==1 & A2==-1)+5*(A1==-1 & A2==1)+1*(A1==-1 & A2==-1)
  Y2 <- Y2_mean(A1,A2)
  taus <- cbind(p1p1=Y1_mean(A1=1,A2=1)+Y2_mean(A1=1,A2=1),
                n1p1=Y1_mean(A1=-1,A2=1)+Y2_mean(A1=-1,A2=1),
                p1n1=Y1_mean(A1=1,A2=-1)+Y2_mean(A1=1,A2=-1),
                n1n1=Y1_mean(A1=-1,A2=-1)+Y2_mean(A1=-1,A2=-1))
  
  
  d.argmax <- c('tau_+1+1','tau_-1+1','tau_+1-1','tau_-1-1')[apply(taus,1,which.max)]
  d.argmin <- c('tau_+1+1','tau_-1+1','tau_+1-1','tau_-1-1')[apply(taus,1,which.min)]
  df <- data.frame(A1,Y1,A2,Y2,taus,d.argmax,d.argmin)
  df <- df %>% mutate(d1.star=if_else(substr(d.argmax,5,6)=='+1',1,-1),
                      d2.star=if_else(substr(d.argmax,7,8)=='+1',1,-1)) 
  
  if (ret!='all') df <- taus
  return(df)
}

# sigmoid surrogate loss functions 
psi1 <- function(x,y) (1+x/(1+abs(x)))*(1+y/(1+abs(y)))
psi2 <- function(x,y) (1+(2/pi)*atan(pi*x/2))*(1+(2/pi)*atan(pi*y/2))
psi3 <- function(x,y) (1+x/sqrt(1+x^2))*(1+y/sqrt(1+y^2))
psi5 <- function(x,y) (1+tanh(x))*(1+tanh(y))
psi4 <- function(x,y) min(x-1,y-1,0)+1

#Cost Function (neg. Value fun.)
cost <- function(theta,dat,surr_fn,O1.vars,O2.bar.vars){
  # assign surrogate function to psi
  psi <- get(paste('psi',surr_fn,sep=''))
  theta1 <- theta[1:length(O1.vars)]
  theta2 <- theta[(length(O1.vars)+1):length(theta)]
  # stage 1&2 covariates
  O1 <- dat[,O1.vars]
  O2.bar <- dat[,O2.bar.vars]
  # Compute linear functions f(O,theta)
  dat$f1 <- crossprod(t(as.matrix(O1)),theta1)
  dat$f2 <- crossprod(t(as.matrix(O2.bar)),theta2)
  # Compute value function (missing IPWs)
  V <- with(dat,(Y1+Y2)*psi(A1*f1,A2*f2)) 
  cst <- -sum(V) #+ .5*sum(abs(c(theta1,theta2)))#+ .5*crossprod(c(theta1,theta2))
  return(cst)
}



## Function that fits to the data and predicts
fit.n.predict <- function(df.train,df.test,O1.vars,O2.bar.vars,surr_fn){
  #Intial theta
  initial_theta <- rep(0,length(c(O1.vars,O2.bar.vars)))
  
  #Cost at inital theta
  #cost(initial_theta,dat=df.train,surr_fn=1,O1.vars,O2.bar.vars)
  
  # Derive theta using gradient descent using optim function
  theta_optim <- optim(par=initial_theta,fn=cost,dat=df.train,surr_fn=surr_fn,O1.vars=O1.vars,O2.bar.vars=O2.bar.vars,method="SANN")
  
  #cost at optimal value of the theta
  #theta_optim$value
  
  #set theta
  theta.hat <- theta_optim$par
  
  theta1.hat <- theta.hat[1:length(O1.vars)]
  theta2.hat <- theta.hat[(length(O1.vars)+1):length(theta.hat)]
  # Compute linear functions f(O,theta)
  f1.hat <- crossprod(t(as.matrix(df.test[,O1.vars])),theta1.hat)
  f2.hat <- crossprod(t(as.matrix(df.test[,O2.bar.vars])),theta2.hat)
  #df.test <- data.frame(df.test)
  df.test[[paste('d1.hat.psi',surr_fn,sep='')]] <- as.numeric(sign(f1.hat))
  df.test[[paste('d2.hat.psi',surr_fn,sep='')]] <- as.numeric(sign(f2.hat))
  return(df.test)
}

fit.bowl <- function(df.train,df.test,O1.vars,O2.bar.vars,surr_fn){
  # Constant propensity model
  moPropen <- buildModelObj(model = ~1,
                            solver.method = 'glm',
                            solver.args = list('family'='binomial'),
                            predict.method = 'predict.glm',
                            predict.args = list(type='response'))
  
  # Second stage
  fitSS <- bowl(moPropen = moPropen, surrogate = surr_fn,kernel= 'linear',#lambdas = .5,
                data = df.train, reward = df.train$Y2, txName = 'A2.f',verbose=0,
                regime =as.formula(paste('~0+',paste(O2.bar.vars,collapse='+'))))#[1:5]
  # First stage
  fitFS <- bowl(moPropen = moPropen, surrogate=surr_fn,kernel= 'linear',
                data = df.train, reward = df.train$Y1, txName = 'A1.f',
                regime =as.formula(paste('~0+',paste(O1.vars,collapse='+'))),#[1:3]
                BOWLObj = fitSS,verbose=0)#, lambdas = c(0.5, 1.0), cvFolds = 4L)
  # Estimated value of the optimal treatment regime for training set
  #estimator(fitSS)
  # Estimated optimal treatment for new data
  df.test[[paste('d1.hat.bowl.',surr_fn,sep='')]] <- optTx(fitFS, df.test)[['optimalTx']]
  df.test[[paste('d2.hat.bowl.',surr_fn,sep='')]] <- optTx(fitSS, df.test)[['optimalTx']]
  return(df.test)
}

feat_transf <- function(df,setting){
  knots_No <- 2
  if(setting==1 |setting==2){
    features1 <- colnames(df)[grep('O1',colnames(df))]
    features2 <- colnames(df)[grep('O1',colnames(df))]
    
    # Design formula to include intercept, main effects, squares and pairwise interactions
    desing_mat_f1 <- as.formula(paste('~0+(',paste(features1,collapse='+'),')'))
    desing_mat_f2 <- as.formula(paste('~0+(',paste(features2,collapse='+'),')^3'))
    df.splines1 <- data.frame(model.matrix(desing_mat_f1,data=df))
    df.2 <- data.frame(model.matrix(desing_mat_f2,data=df))
    # Natural cubic splines:
    nms1 <- colnames(df.splines1)
    # compute basis for each relevant column
    df.splines1 <- as.data.frame(lapply(nms1, function(nm) {ns(df.splines1[,nm],df = knots_No)}))
    # name the basis
    colnames(df.splines1) <- paste('H1_',1:ncol(df.splines1),sep = '')
    # merge basis with the rest of the columns
    if(setting==1){
      O1.vars <- c(colnames(df.splines1),colnames(df.2),'int')
      O2.bar.vars <- c(O1.vars,'Y1')
    }else{
      O1.vars <- c(colnames(df.splines1),'int')
      O2.bar.vars <- c(O1.vars,colnames(df.2),'Y1','int')
    }
    df <- cbind(df,df.splines1,df.2,'int'=1); df <- df[,unique(colnames(df))]
  }else if (setting=='disc'){
    desing_mat_f <- as.formula(paste('~1+(O1.1+O1.2+O1.3+O2.2+Y1)^5'))
    features_X <- data.frame(model.matrix(desing_mat_f,data=df))
    df <- cbind(df,features_X,'int'=1); df <- df[,unique(colnames(df))]
    O1.vars <- c(colnames(features_X)[-c(grep('O1.2',colnames(features_X)),grep('O2.2',colnames(features_X)),grep('Y1',colnames(features_X)))])
    O2.bar.vars <- c(colnames(features_X)[-grep('O1.3',colnames(features_X))])
  }else if (setting=='toy'){
    O1.vars <- 'A1'
    
    O2.vars <- c('A1','A2')
  }
  return(list(df=df,O1.vars=O1.vars,O2.bar.vars=O2.bar.vars))
}
gen_df <- function(size,setting,sd) {
  # Generate Dataset
  if(setting==1 |setting==2){
    df <- gen_data(size,setting=setting,sd)
    }else{
    df <- gen_disc_data(size,sd,ret='all')
    }
  
  # Combute basis matrix for natural cubic splines
  df <- feat_transf(df,setting)[['df']]
  return(df)
  }
run_sims <- function(size,setting,sims_No){
  if(setting==1 |setting==2){df <- gen_data(10,2,1)}else{df <- gen_disc_data(10,1,ret='all')}
  df_ls <- feat_transf(df,setting)
  O1.vars <- df_ls[['O1.vars']]
  O2.bar.vars <- df_ls[['O2.bar.vars']]
  V_fn <- matrix(NA,sims_No,11); errs <- matrix(NA,sims_No,20); time.taken <- matrix(NA,sims_No,10)
  colnames(errs) <- c(paste(rep(paste('d',1:2,'.psi',sep=''),5),rep(c(1:5),each=2),sep=''),
                      paste(rep(paste('d',1:2,'.bowl.',sep=''),4),rep(c('hinge','exp','logit','huber'),each=2),sep=''),'d1.Q','d2.Q');
  
  colnames(V_fn) <- c('True',paste('Estimate.psi',c(1:5),sep=''),paste('Estimate.',c('hinge','exp','logit','huber'),sep=''),'Estimate.Q')
  colnames(time.taken) <- c(paste('psi',c(1:5),sep=''),paste('bowl.',c('hinge','exp','logit','huber'),sep=''),'Q')
  taus_errd1 <- taus_errd2 <- matrix(NA,0,4)
  sd <- sim <- 1
  # Will start loop here:
  while (sim <= sims_No){
    tryCatch({
      cat('Sim No: ',sim,'\n')
      if(setting==1 |setting==2){df <- gen_data(size,setting=setting,sd)}else{df <- gen_disc_data(size,sd,ret='all')}
      # apply(df.train[,c(10,11,12,13)],2,mean)
      df <- feat_transf(df,setting)[['df']]
      df <- df %>% mutate(A1.f = as.factor(A1),A2.f = as.factor(A2))
      df.train <- df[1:as.integer(size/2),]; df.test <- df[1:as.integer(size/2),]; df.test$int <- 1
      ##### fit conc. surrogates
      start.time <- Sys.time()
      df.test <- fit.n.predict(df.train,df.test,O1.vars,O2.bar.vars,surr_fn=1)
      end.time <- Sys.time()
      time.taken[sim,'psi1'] <- as.numeric(end.time - start.time, units = "secs")
      
      start.time <- Sys.time()
      df.test <- fit.n.predict(df.train,df.test,O1.vars,O2.bar.vars,surr_fn=2)
      end.time <- Sys.time()
      time.taken[sim,'psi2'] <- as.numeric(end.time - start.time, units = "secs")
      
      start.time <- Sys.time()
      df.test <- fit.n.predict(df.train,df.test,O1.vars,O2.bar.vars,surr_fn=3)
      end.time <- Sys.time()
      time.taken[sim,'psi3'] <- as.numeric(end.time - start.time, units = "secs")
      
      start.time <- Sys.time()
      df.test <- fit.n.predict(df.train,df.test,O1.vars,O2.bar.vars,surr_fn=4)
      end.time <- Sys.time()
      time.taken[sim,'psi4'] <- as.numeric(end.time - start.time, units = "secs")
      
      start.time <- Sys.time()
      df.test <- fit.n.predict(df.train,df.test,O1.vars,O2.bar.vars,surr_fn=5)
      end.time <- Sys.time()
      time.taken[sim,'psi5'] <- as.numeric(end.time - start.time, units = "secs")
      
      ##### Fit BOWL with surrogates
      start.time <- Sys.time()
      df.test <- fit.bowl(df.train,df.test,O1.vars,O2.bar.vars,surr_fn =  'hinge')#'exp')#
      end.time <- Sys.time()
      time.taken[sim,'bowl.hinge'] <- as.numeric(end.time - start.time, units = "secs")
      ##
      #colnames(df.test) <- gsub('exp','hinge',colnames(df.test))
      ##    
      d1.bowl.hinge.errs <- which(with(df.test,d1.hat.bowl.hinge!=d1.star))
      d2.bowl.hinge.errs <- which(with(df.test,d2.hat.bowl.hinge!=d2.star))
      
      start.time <- Sys.time()
      df.test <- fit.bowl(df.train,df.test,O1.vars,O2.bar.vars,surr_fn = 'exp')
      end.time <- Sys.time()
      time.taken[sim,'bowl.exp'] <- as.numeric(end.time - start.time, units = "secs")
      d1.bowl.exp.errs <- which(with(df.test,d1.hat.bowl.exp!=d1.star))
      d2.bowl.exp.errs <- which(with(df.test,d2.hat.bowl.exp!=d2.star))
      
      start.time <- Sys.time()
      df.test <- fit.bowl(df.train,df.test,O1.vars,O2.bar.vars,surr_fn = 'logit')
      end.time <- Sys.time()
      time.taken[sim,'bowl.logit'] <- as.numeric(end.time - start.time, units = "secs")
      d1.bowl.logit.errs <- which(with(df.test,d1.hat.bowl.logit!=d1.star))
      d2.bowl.logit.errs <- which(with(df.test,d2.hat.bowl.logit!=d2.star))
      
      start.time <- Sys.time()
      df.test <- fit.bowl(df.train,df.test,O1.vars,O2.bar.vars,surr_fn = 'huber')
      end.time <- Sys.time()
      time.taken[sim,'bowl.huber'] <- as.numeric(end.time - start.time, units = "secs")
      d1.bowl.huber.errs <- which(with(df.test,d1.hat.bowl.huber!=d1.star))
      d2.bowl.huber.errs <- which(with(df.test,d2.hat.bowl.huber!=d2.star))
      
      ##### Fit Q-learning
      start.time <- Sys.time()
      ### Second-Stage Analysis
      # outcome model
      if(setting=='disc'){
        moMain <- buildModelObj(model = ~1,
                                solver.method = 'lm')
        moCont <- buildModelObj(model =~O1.1+O1.2+O2.2+A1+Y1,
                                solver.method = 'lm')
      }else if(setting==1){
        moMain <- buildModelObj(model = ~O1.1+O1.2+O1.3+(O1.1+O1.2+O1.3)*A1+Y1,
                                solver.method = 'lm')
        
        moCont <- buildModelObj(model =~O1.1+O1.2+O1.3+(O1.1+O1.2+O1.3)*A1+Y1,
                                solver.method = 'lm')
      }else if(setting==2){
        moMain <- buildModelObj(model = ~O1.1+O1.2+O1.3+O2.1+O2.2+(O1.1+O1.2+O1.3+O2.1+O2.2)*A1+Y1,
                                solver.method = 'lm')
        
        moCont <- buildModelObj(model =~O1.1+O1.2+O1.3+O2.1+O2.2+(O1.1+O1.2+O1.3+O2.1+O2.2)*A1+Y1,
                                solver.method = 'lm')
      }
      
      # Second stage
      fitSS <- qLearn(moMain = moMain, moCont = moCont,
                      data = df.train, response = df.train$Y2, txName = 'A2')
      
      
      
      ### First-Stage Analysis Main Effects Term
      # main effects model
      if(setting=='disc'){
        moMain <- buildModelObj(model = ~1,
                                solver.method = 'lm')
        moCont <- buildModelObj(model =~O1.3+O1.1,
                                solver.method = 'lm')
      }else if(setting==1 | setting==2){
        moMain <- buildModelObj(model = ~O1.1+O1.2+O1.3,
                                solver.method = 'lm')
        moCont <- buildModelObj(model =~O1.1+O1.2+O1.3,
                                solver.method = 'lm')
      }
      fitFS <- qLearn(moMain = moMain, moCont = moCont,
                      data = df.train, response = fitSS, txName = 'A1')
      
      # Estimated value of the optimal treatment regime for training set
      
      # Estimated optimal treatment for new data
      df.test$d1.hat.Q <- optTx(fitFS, df.test)[['optimalTx']]
      df.test$d2.hat.Q <- optTx(fitSS, df.test)[['optimalTx']]
      end.time <- Sys.time()
      time.taken[sim,'Q'] <- as.numeric(end.time - start.time, units = "secs")
      
      d1.Q.errs <- which(with(df.test,d1.hat.Q!=d1.star))
      d2.Q.errs <- which(with(df.test,d2.hat.Q!=d2.star))
      #####
      
      d1.psi1.errs <- which(with(df.test,d1.hat.psi1!=d1.star))
      d2.psi1.errs <- which(with(df.test,d2.hat.psi1!=d2.star))
      d1.psi2.errs <- which(with(df.test,d1.hat.psi2!=d1.star))
      d2.psi2.errs <- which(with(df.test,d2.hat.psi2!=d2.star))
      d1.psi3.errs <- which(with(df.test,d1.hat.psi3!=d1.star))
      d2.psi3.errs <- which(with(df.test,d2.hat.psi3!=d2.star))
      d1.psi4.errs <- which(with(df.test,d1.hat.psi4!=d1.star))
      d2.psi4.errs <- which(with(df.test,d2.hat.psi4!=d2.star))
      d1.psi5.errs <- which(with(df.test,d1.hat.psi5!=d1.star))
      d2.psi5.errs <- which(with(df.test,d2.hat.psi5!=d2.star))
      # Storing taus for which decision rules are wrong
      taus_errd1 <- rbind(taus_errd1,df.test[d1.psi1.errs,c('p1p1','n1p1','p1n1','n1n1')])
      taus_errd2 <- rbind(taus_errd2,df.test[d2.psi1.errs,c('p1p1','n1p1','p1n1','n1n1')])
      # Computing the mean Value with the estimated regimes:
      df.test <- df.test %>% mutate(opt.V=case_when(d1.star==1 & d2.star==1~p1p1,d1.star==1 & d2.star==-1~p1n1,d1.star==-1 & d2.star==1~n1p1,T~n1n1),
                                    V.psi1=case_when(d1.hat.psi1==1 & d2.hat.psi1==1~p1p1,d1.hat.psi1==1 & d2.hat.psi1==-1~p1n1,d1.hat.psi1==-1 & d2.hat.psi1==1~n1p1,T~n1n1),
                                    V.psi2=case_when(d1.hat.psi2==1 & d2.hat.psi2==1~p1p1,d1.hat.psi2==1 & d2.hat.psi2==-1~p1n1,d1.hat.psi2==-1 & d2.hat.psi2==1~n1p1,T~n1n1),
                                    V.psi3=case_when(d1.hat.psi3==1 & d2.hat.psi3==1~p1p1,d1.hat.psi3==1 & d2.hat.psi3==-1~p1n1,d1.hat.psi3==-1 & d2.hat.psi3==1~n1p1,T~n1n1),
                                    V.psi4=case_when(d1.hat.psi4==1 & d2.hat.psi4==1~p1p1,d1.hat.psi4==1 & d2.hat.psi4==-1~p1n1,d1.hat.psi4==-1 & d2.hat.psi4==1~n1p1,T~n1n1),
                                    V.psi5=case_when(d1.hat.psi5==1 & d2.hat.psi5==1~p1p1,d1.hat.psi5==1 & d2.hat.psi5==-1~p1n1,d1.hat.psi5==-1 & d2.hat.psi5==1~n1p1,T~n1n1),
                                    # Q learning and BOWL
                                    V.bowl.hinge=case_when(d1.hat.bowl.hinge==1 & d2.hat.bowl.hinge==1~p1p1,d1.hat.bowl.hinge==1 & d2.hat.bowl.hinge==-1~p1n1,d1.hat.bowl.hinge==-1 & d2.hat.bowl.hinge==1~n1p1,T~n1n1),
                                    V.bowl.exp=case_when(d1.hat.bowl.exp==1 & d2.hat.bowl.exp==1~p1p1,d1.hat.bowl.exp==1 & d2.hat.bowl.exp==-1~p1n1,d1.hat.bowl.exp==-1 & d2.hat.bowl.exp==1~n1p1,T~n1n1),
                                    V.bowl.logit=case_when(d1.hat.bowl.logit==1 & d2.hat.bowl.logit==1~p1p1,d1.hat.bowl.logit==1 & d2.hat.bowl.logit==-1~p1n1,d1.hat.bowl.logit==-1 & d2.hat.bowl.logit==1~n1p1,T~n1n1),
                                    V.bowl.huber=case_when(d1.hat.bowl.huber==1 & d2.hat.bowl.huber==1~p1p1,d1.hat.bowl.huber==1 & d2.hat.bowl.huber==-1~p1n1,d1.hat.bowl.huber==-1 & d2.hat.bowl.huber==1~n1p1,T~n1n1),
                                    
                                    V.Qlearn=case_when(d1.hat.Q==1 & d2.hat.Q==1~p1p1,d1.hat.Q==1 & d2.hat.Q==-1~p1n1,d1.hat.Q==-1 & d2.hat.Q==1~n1p1,T~n1n1))
      V_fn[sim,] <- c(mean(df.test$opt.V),mean(df.test$V.psi1),mean(df.test$V.psi2),mean(df.test$V.psi3),mean(df.test$V.psi4),mean(df.test$V.psi5),
                      mean(df.test$V.bowl.hinge),mean(df.test$V.bowl.exp),mean(df.test$V.bowl.logit),mean(df.test$V.bowl.huber),mean(df.test$V.Qlearn))
      
      errs[sim,c('d1.psi1','d2.psi1')] <- c(length(d1.psi1.errs),length(d2.psi1.errs))/nrow(df.test)
      errs[sim,c('d1.psi2','d2.psi2')] <- c(length(d1.psi2.errs),length(d2.psi2.errs))/nrow(df.test)
      errs[sim,c('d1.psi3','d2.psi3')] <- c(length(d1.psi3.errs),length(d2.psi3.errs))/nrow(df.test)
      errs[sim,c('d1.psi4','d2.psi4')] <- c(length(d1.psi4.errs),length(d2.psi4.errs))/nrow(df.test)
      errs[sim,c('d1.psi5','d2.psi5')] <- c(length(d1.psi5.errs),length(d2.psi5.errs))/nrow(df.test)
      errs[sim,c('d1.bowl.hinge','d2.bowl.hinge')] <- c(length(d1.bowl.hinge.errs),length(d2.bowl.hinge.errs))/nrow(df.test)
      errs[sim,c('d1.bowl.exp','d2.bowl.exp')] <- c(length(d1.bowl.exp.errs),length(d2.bowl.exp.errs))/nrow(df.test)
      errs[sim,c('d1.bowl.logit','d2.bowl.logit')] <- c(length(d1.bowl.logit.errs),length(d2.bowl.logit.errs))/nrow(df.test)
      errs[sim,c('d1.bowl.huber','d2.bowl.huber')] <- c(length(d1.bowl.huber.errs),length(d2.bowl.huber.errs))/nrow(df.test)
      errs[sim,c('d1.Q','d2.Q')] <- c(length(d1.Q.errs),length(d2.Q.errs))/nrow(df.test)
      cat('sim: ',sim,', n: ','setting: ',setting,size,'\n')
      print(round(apply(errs,2,mean,na.rm=T),2))
      print(round(apply(V_fn,2,mean,na.rm=T),2))
      print(round(apply(time.taken,2,mean,na.rm=T),2))
      
      sim <- sim + 1
    }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})
    sd <- sd + 1
  }
  return(list(errs=errs,V_fn=V_fn,taus_errd1=taus_errd1,taus_errd2=taus_errd2,tot.trials=sd,time.taken=time.taken))
}


#####


