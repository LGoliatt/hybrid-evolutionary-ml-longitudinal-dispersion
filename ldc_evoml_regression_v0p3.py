#!/usr/bin/python
# -*- coding: utf-8 -*-    
import numpy as np
import pandas as pd
import pygmo as pg
import re, os, sys, getopt
from scipy import stats
import pylab as pl
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_predict, 
                                     TimeSeriesSplit, cross_val_score, 
                                     LeaveOneOut, KFold, StratifiedKFold,
                                     cross_val_predict,train_test_split)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, MaxAbsScaler, Normalizer, StandardScaler, MaxAbsScaler, FunctionTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)




from read_data import *

program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)


if len(arguments)>0:
    if arguments[0]=='-r':
        run0 = int(arguments[1])
        n_runs = run0+1
else:
    run0, n_runs = 0,10


#%%----------------------------------------------------------------------------   
def accuracy_log(y_true, y_pred):
    y_true=np.abs(np.array(y_true))
    y_pred=np.abs(np.array(y_pred))
    return (np.abs(np.log10(y_true/y_pred))<0.3).sum()/len(y_true)*100

def rms(y_true, y_pred):
    y_true=np.abs(np.array(y_true))
    y_pred=np.abs(np.array(y_pred))
    return ( (np.log10(y_pred/y_true)**2).sum()/len(y_true) )**0.5

def model_base_evaluation(x, data_args, estimator_args,
                          normalizer_args, transformer_args):
    
  (X_train_, y_train, X_test_, y_test, flag, task,  n_splits, 
     random_seed, scoring, target, 
     n_samples_train, n_samples_test, n_features)   = data_args
  (normalizer_type,)                                = normalizer_args
  (transformer_type, n_components, kernel_type)     = transformer_args
  (clf_name, n_decision_variables, p, clf)          = estimator_args

  #
  # normalizer
  #
  normalizer={ 
              'None'            : FunctionTransformer(),
              'MinMax'          : MinMaxScaler(), 
              'MaxAbs'          : MaxAbsScaler(), 
              'Standard'        : StandardScaler(),
              'Log'             : FunctionTransformer(np.log1p),
              'Quantile Norm.'  : QuantileTransformer(n_quantiles=n_features,  output_distribution='normal'),
              'Quantile Unif.'  : QuantileTransformer(n_quantiles=n_features,  output_distribution='uniform'),
              'Poly'            : PolynomialFeatures(),
             }
  
  normalizer_dict={0:'None', 1:'MinMax', 2:'MaxAbs', 3:'Standard', 4:'Log', 5:'Quantile Norm.', 6:'Quantile Unif.', 7:'Poly',}
  n=normalizer_dict[normalizer_type]
  #
  # transformer
  #
  kernel={0:"linear", 1:"poly", 2:"rbf", 3:"sigmoid", 4:"cosine", }
  
  if transformer_type=='Identity':
      n_components=None
      
  if transformer_type=='KPCA':
      k = kernel[kernel_type]
  else:
      k=None
      
  transformer={
               'Identity'   : FunctionTransformer(),
               'PCA'        : KernelPCA(kernel='linear', n_components=n_components, random_state=random_seed),
               'KPCA'       : KernelPCA(kernel=k       , n_components=n_components, random_state=random_seed),
              }
  transformer_dict={0:'Identity', 1:'PCA', 2:'KPCA',}
  t=transformer_dict[transformer_type]
  #
  # estimator pipeline
  #
  model=Pipeline([ 
          ('normalizer', normalizer[n]), 
          ('tranformer', transformer[t]),
          ('estimator' , clf),
          ]);

  if len(x)<=n_decision_variables:
      clfnme=clf_name
      ft = np.array([1 for i in range(n_features)])
      ft = np.where(ft>0.5)[0]
  else:
      clfnme=clf_name+'-FS'
      ft = np.array([1 if k>0.5 else 0 for k in x[n_decision_variables::]])
      ft = np.where(ft>0.5)[0]
  
  #print(len(x), n_features, n_decision_variables, ft)
  if task=='regression':
      cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed),)
      #cv=LeaveOneOut()
  elif task=='forecast':
      cv=TimeSeriesSplit(n_splits=n_splits,)
  else:
      sys.exit('Cross-validation does not defined for estimator '+clf_name)
      
  ##--
  #y_scaler = MaxAbsScaler()
  #y_scaler.fit(y_train.reshape(-1,1))
  ##--  
  
  if flag=='eval':
    try:
        #r=cross_val_score(model,X_train[:,ft], y_train, cv=cv, n_jobs=1, scoring=scoring)
        #r=np.abs(r).mean()
        
        #y_p  = cross_val_predict(model,X_train[:,ft], y_scaler.transform(y_train.reshape(-1,1)).ravel(), cv=cv, n_jobs=1)
        #r = rms(y_scaler.inverse_transform(y_p.reshape(-1,1)).ravel(), y_train)
                
        y_p  = cross_val_predict(model,X_train[:,ft], y_train, cv=cv, n_jobs=1)
        #r = -accuracy_log(y_p, y_train)      
        #r = mean_squared_error(y_p, y_train)#**.5
        #r = -r2_score(y_p, y_train)
        #r = mean_squared_error(y_p, y_train)**.5#/rms(y_p, y_train)/r2_score(y_p, y_train)**2
        r = rms(y_p, y_train)
    except:
        r=1e12 
    
    #print(r,'\t',p, )#'\t', ft)  
    return r
  elif flag=='run':
    model.fit(X_train[:,ft], y_train)
    #model.fit(X_train[:,ft], y_scaler.transform(y_train.reshape(-1,1)).ravel())
    if task=='regression':
        y_p  = cross_val_predict(model,X_train[:,ft], y_train, cv=cv, n_jobs=1)
        #y_p  = cross_val_predict(model,X_train[:,ft], y_scaler.transform(y_train.reshape(-1,1)).ravel(), cv=cv, n_jobs=1)
    else:
        #y_p=np.array([None for i in range(len(y_train))])
        y_p=model.predict(X_train[:,ft])#y_train
        
    if n_samples_test>0:
        y_t  = model.predict(X_test[:,ft])
    else:
        y_t=np.array([None for i in range(len(y_test))])
        
    
    return {
            'Y_TRAIN_TRUE':y_train, 
            'Y_TRAIN_PRED':y_p, 
            #'Y_TRAIN_PRED': y_scaler.inverse_transform(y_p.reshape(-1,1)).ravel(),
            'Y_TEST_TRUE':y_test, 
            'Y_TEST_PRED':y_t,             
            #'Y_TEST_PRED':y_scaler.inverse_transform(y_t.reshape(-1,1)).ravel(),
            'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':clfnme,
            'SCALES_PARAMS':{'scaler':n},
            'TRANSF_PARAMS':{'tranformer':t, 'kernel':k, 'n_components':n_components},
            #'ESTIMATOR':clf, 
            'ACTIVE_VAR':ft, 'SCALER':n,
            'SEED':random_seed, 'N_SPLITS':n_splits,
            #'ACTIVE_FEATURES':ft,
            'OUTPUT':target
            }
  else:
      sys.exit('Model evaluation doe not performed for estimator '+clf_name)
      
#------------------------------------------------------------------------------
def fun_svr_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='SVR' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 7
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  clf = SVR(kernel='rbf', max_iter=10000)
  kernel = {
            0:'rbf', 
            1:'sigmoid', 
            2:'chi2',
            3:'laplacian', 
            4:'poly', 
            5:'linear', 
            }  
  
  _gamma = int(x[4]*1000)/1000.
  p={
     'gamma'        :'scale' if _gamma<=0 else _gamma, 
     'C'            : x[5],  
     'epsilon'      : int(x[6]*1000)/1000., 
     #'kernel'      : kernel[0],
     #'tol'         : 1e-6,
     #'max_iter'    : 10000,
     #'shrinking'   : False,
     }

  clf.set_params(**p)
  estimator_args=(clf_name, n_decision_variables, p, clf, )
  
  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#%%----------------------------------------------------------------------------     
def fun_gpr_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='GPR' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 9
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  k1 = x[8]#int(x[8]*10000)/10000.
  clf = GaussianProcessRegressor(random_state=int(random_seed), optimizer=None)
  kernel = {
        0: k1**2 * Matern(length_scale=int(x[5]*1000)/1000., length_scale_bounds=(1e-1, 100.0), nu=int(x[6]*1000)/1000.),
        1: k1**2 * RationalQuadratic(length_scale=int(x[5]*1000)/1000., length_scale_bounds=(1e-1, 100.0), alpha=int(x[6]*1000)/1000.),
        3: k1**2 * RBF(length_scale=int(x[5]*1000)/1000.),
        }
  
  p={'kernel': kernel[int(x[4]+0.995)], 'alpha':x[7]}#int(x[7]*1000)/1000.}

  clf.set_params(**p)
  p['k1']=k1
  estimator_args=(clf_name, n_decision_variables, p, clf, )
  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#------------------------------------------------------------------------------   
def RMSE(y, y_pred):
    y, y_pred = np.array(y).ravel(), np.array(y_pred).ravel()
    error = y -  y_pred    
    return np.sqrt(np.mean(np.power(error, 2)))



import pygmo as pg
class evoML:
    def __init__(self, args, fun, lb, ub):
         self.args = args
         self.obj = fun
         self.lb, self.ub= lb, ub
         
    def fitness(self, x):     
        self.res=self.obj(x,*self.args)
        return [self.res]
    
    def get_bounds(self):
         return (self.lb, self.ub)  
     
    def get_name(self):
         return "evoML"
        
#%%----------------------------------------------------------------------------   

basename='evo_ml_'

datasets = [
            read_data_ldc_tayfur( case = 0 ),            
            #read_data_ldc_tayfur( case = 1 ),
            #read_data_ldc_tayfur( case = 2 ),
            #read_data_ldc_tayfur( case = 3 ),
            #read_data_ldc_tayfur( case = 4 ),
            #read_data_ldc_tayfur( case = 5 ),
            #read_data_ldc_tayfur( case = 6 ),
            #read_data_ldc_tayfur( case = 7 ),
            #read_data_ldc_tayfur( case = 8 ),
            ]
     
#%%----------------------------------------------------------------------------   

pop_size    = 30
max_iter    = 50
n_splits    = 5
scoring     = 'neg_root_mean_squared_error'
for run in range(run0, n_runs):
    random_seed=run+100
    
    for dataset in datasets:#[:1]:
        dr=dataset['name'].replace(' ','_').replace("'","").lower()
        path='./pkl_'+dr+'/'
        os.system('mkdir  '+path)

        for (target,y_train,y_test) in zip(dataset['target_names'], dataset['y_train'], dataset['y_test']):                        
            dataset_name, X_train, X_test   = dataset['name'], dataset['X_train'], dataset['X_test']
            n_samples_train, n_features     = dataset['n_samples'], dataset['n_features']
            task, normalize                 = dataset['task'], dataset['normalize']
            n_samples_test                  = len(y_test)
            np.random.seed(random_seed)

            s=''+'\n'
            s+='='*80+'\n'
            s+='Dataset                    : '+dataset_name+' -- '+target+'\n'
            s+='Number of training samples : '+str(n_samples_train) +'\n'
            s+='Number of testing  samples : '+str(n_samples_test) +'\n'
            s+='Number of features         : '+str(n_features)+'\n'
            s+='Normalization              : '+str(normalize)+'\n'
            s+='Task                       : '+str(dataset['task'])+'\n'
            s+='Reference                  : '+str(dataset['reference'])+'\n'
            s+='='*80
            s+='\n'            
            
            print(s)
            #------------------------------------------------------------------
            lb_svr = [2,    0, 0., 0,    -1e-1, 1e+1, 1e-1,           ] + [0.0]*n_features
            ub_svr = [2,    0, 1., 4,     1e+1, 1e+4,    4,           ] + [1.0]*n_features
            #------------------------------------------------------------------         
            lb_gpr = [2,    0, 0., 0,     0   , 1e-3, 1e-3,  0.0,  0.0] + [0.0]*n_features
            ub_gpr = [2,    0, 1., 4,     0   , 1e+1,    4,  0.1, 10.0] + [1.0]*n_features
            #------------------------------------------------------------------         
            args = (X_train, y_train, X_test, y_test, 'eval', task,  n_splits, 
                    int(random_seed), scoring, target, 
                    n_samples_train, n_samples_test, n_features)
            #------------------------------------------------------------------         
            optimizers=[             
                ('SVR'  , lb_svr, ub_svr, fun_svr_fs, args, random_seed,),    # OK
                ('GPR'  , lb_gpr, ub_gpr, fun_gpr_fs, args, random_seed,),    # OK            
                ]
            #------------------------------------------------------------------         
            for (clf_name, lb, ub, fun, args, random_seed) in optimizers:
                np.random.seed(random_seed)
                list_results=[]
                #--------------------------------------------------------------
                s=''
                s='-'*80+'\n'
                s+='Estimator                  : '+clf_name+'\n'
                s+='Function                   : '+str(fun)+'\n'
                s+='Run                        : '+str(run)+'\n'
                s+='Random seed                : '+str(random_seed)+'\n'
                
                algo = pg.algorithm(pg.cmaes(gen = max_iter, force_bounds = True, seed=random_seed))
                
                s+='Optimizer                  : '+algo.get_name()+'\n'                
                s+='-'*80+'\n'
                print(s)
                algo.set_verbosity(1)
                prob = pg.problem(evoML(args, fun, lb, ub))
                pop = pg.population(prob,pop_size, seed=random_seed)
                pop = algo.evolve(pop)
                xopt = pop.champion_x
                args1 = (X_train, y_train, X_test, y_test, 'run', task,  n_splits, 
                    int(random_seed), scoring,  target,
                    n_samples_train, n_samples_test, n_features)
                
                sim = fun(xopt, *args1)
                print(xopt, '\n\n', sim)
                sim['ALGO'] = algo.get_name()
                sim['OUTPUT'] = sim['TARGET'] = target

                sim['ACTIVE_VAR_NAMES']=dataset['feature_names'][sim['ACTIVE_VAR']]


                pl.figure()
                pl.plot(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_TRUE'].ravel(), 'r-', 
                            sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel(), 'b.' )
                r2=r2_score(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel())
                r=stats.pearsonr(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel())[0]
                rmse=RMSE(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel())  
                pl.ylabel(dataset_name)
                pl.title(sim['EST_NAME']+': (Training) R$^2$='+str('%1.3f' % r2)+'\t RMSE='+str('%1.3f' % rmse))                
                pl.show()
               
                if n_samples_test > 0:    
                    pl.figure()
                    pl.plot(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_TRUE'].ravel(), 'r-', 
                            sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel(), 'b.' )
                    r2=r2_score(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())
                    r=stats.pearsonr(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())[0]
                    rmse=RMSE(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())                
                    acc=accuracy_log(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())                
                    pl.ylabel(dataset_name)
                    pl.title(sim['EST_NAME']+': (Testing) R$^2$='+str('%1.3f' % r2)+'\t RMSE='+str('%1.3f' % rmse))
                    pl.show()                    
                                                                           
                
                sim['RUN']=run;
                sim['DATASET_NAME']=dataset_name; 
                list_results.append(sim) 
        
                data    = pd.DataFrame(list_results)
                ds_name = dataset_name.replace('/','_').replace("'","").lower()
                tg_name = target.replace('/','_').replace("'","").lower()
                algo    = sim['ALGO'].split(':')[0] 
                pk=(path+#'_'+
                    basename+'_'+
                    '_run_'+str("{:02d}".format(run))+'_'+
                    ("%15s"%ds_name         ).rjust(15).replace(' ','_')+
                    ("%9s"%sim['EST_NAME']  ).rjust( 9).replace(' ','_')+
                    ("%10s"%algo            ).rjust(10).replace(' ','_')+
                    ("%15s"%tg_name         ).rjust(25).replace(' ','_')+
                    '.pkl') 
                pk=pk.replace(' ','_').replace("'","").lower()
                pk=pk.replace('(','_').replace(")","_").lower()
                pk=pk.replace('[','_').replace("]","_").lower()
                pk=pk.replace('-','_').replace("$","").lower()
                #print(pk)
                data.to_pickle(pk)
                
##%%----------------------------------------------------------------------------
                
