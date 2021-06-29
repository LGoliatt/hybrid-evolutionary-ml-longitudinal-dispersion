#!/usr/bin/python
# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn import metrics  
import os

from sklearn.decomposition import KernelPCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import (MinMaxScaler, PolynomialFeatures, MaxAbsScaler, 
                                   Normalizer, StandardScaler, MaxAbsScaler, 
                                   FunctionTransformer, PowerTransformer, 
                                   QuantileTransformer)
from sklearn.svm import SVR
from xgboost import XGBRegressor
from util.RBFNN import RBFNNRegressor as RBFNN

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel,WhiteKernel, 
                                              ExpSineSquared)

from sklearn.model_selection import GridSearchCV
from read_data import *
#%%

def rms(y_true, y_pred):
    y_true=np.abs(np.array(y_true))
    y_pred=np.abs(np.array(y_pred))
    return ( (np.log10(y_pred/y_true)**2).sum()/len(y_true) )**0.5

def accuracy_log(y_true, y_pred):
    y_true=np.abs(np.array(y_true))
    y_pred=np.abs(np.array(y_pred))
    return (np.abs(np.log10(y_true/y_pred))<0.3).sum()/len(y_true)*100

def _logical(x1, x2, x3, x4):
    return np.where(x1 > x2, x3, x4)

#%%
pd.options.display.float_format = '{:.3f}'.format
datasets = [
            #read_data_ldc_toprak(),
            #read_data_ldc_etemad(),
            read_data_ldc_tayfur( case = 0 ),
            read_data_ldc_tayfur( case = 1 ),
            read_data_ldc_tayfur( case = 2 ),
            read_data_ldc_tayfur( case = 3 ),
            read_data_ldc_tayfur( case = 4 ),
            read_data_ldc_tayfur( case = 5 ),
            read_data_ldc_tayfur( case = 6 ),
            read_data_ldc_tayfur( case = 7 ),
           ]

random_seed=0
for dataset in datasets:
    task             = dataset['task'            ]
    dataset_name     = dataset['name'            ]
    feature_names    = dataset['feature_names'   ]
    target_names     = dataset['target_names'    ]
    n_samples        = dataset['n_samples'       ]
    n_features       = dataset['n_features'      ]
    X_train          = dataset['X_train'         ]
    X_test           = dataset['X_test'          ]
    y_train          = dataset['y_train'         ]
    y_test           = dataset['y_test'          ]
    targets          = dataset['targets'         ]
    true_labels      = dataset['true_labels'     ]
    predicted_labels = dataset['predicted_labels']
    descriptions     = dataset['descriptions'    ]
    items            = dataset['items'           ]
    reference        = dataset['reference'       ]
    normalize        = dataset['normalize'       ]
    n_samples_train  = len(y_train)
    
    #%%
    dr='gpr_'+dataset_name.replace(' ','_').replace("'","").lower()
    path='./pkl_'+dr+'/'
    os.system('mkdir  '+path)
          
    for n, target in enumerate(target_names):
        y_train = dataset['y_train'][n]#.reshape(-1,1)
        y_test  = dataset['y_test' ][n]#.reshape(-1,1)
        n_samples_test                  = len(y_test)
    
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
        # 
        kernels = [
                    #('RBF              ', 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),                                                          ),
                    #('RationalQuadratic', 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),                                                                   ),
                    #('ExpSineSquared   ', 1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,length_scale_bounds=(0.1, 10.0),periodicity_bounds=(1.0, 10.0)), ),
                    #('DotProduct       ', 1.0 * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),                          ),
                    ('Matern           ', 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e+3),nu=3.1),                  ),
                    #('ExpSineSquared   ', ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1),                                           ),
                  ]


        for k in range(10):

            scalers = [
                        MinMaxScaler(),
                        PowerTransformer(method='yeo-johnson',),
                        QuantileTransformer(n_quantiles=30, random_state=k),#n_quantiles=X_train.shape[0], random_state=k, output_distribution='normal')
                        FunctionTransformer(),
                        FunctionTransformer(np.exp),
                    ]
            
            
            scaler_grid         = {'output_distribution':['normal','uniform'], }#'n_quantiles':[X_train.shape[0],]}
            transformer_grid    = {}
            kern = [ 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e+3),nu=i) for i in np.linspace(0.1,3.,30)]
            #kern+= [ 1.0 * RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-2, 1e+3),alpha=i) for i in np.linspace(0.1,3.,30)]
            estimator_grid      = {'kernel': kern, }
            
            
            ##transformer = KernelPCA(kernel='linear', n_components=n_components)
            transformer = FunctionTransformer()
            ##estimator   = ExtraTreesRegressor(n_estimators=1500)
            ##estimator   = SVR(C=1e3)
            ##estimator   = XGBRegressor(n_estimators=60, learning_rate=0.2, objective='reg:squarederror', max_depth=20, random_state=k)
            estimator   = GaussianProcessRegressor(kernel='linear', random_state=k)

            clf          = Pipeline([
                                    ('scaler'       , scalers[2]), 
                                    ('transformer'  , transformer),
                                    ('estimator'    , estimator)
                                ])            
            
            clf_steps           = [ clf.steps[i][0] for i in range(len(clf.steps)) ] 
        
            param_grid={}
            for a, b in zip(clf_steps, (scaler_grid, transformer_grid, estimator_grid) ):
                for b1 in b.keys():
                    param_grid[a+'__'+b1] = b[b1]
            
            cv = KFold(n_splits=3, shuffle=True,    random_state=k)
            model=GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, verbose=0, scoring='r2') 
            model.fit(X_train, y_train.squeeze())
            
            print(model.best_params_)
            
            y_pred      = model.predict(X_test)
            rmse        = metrics.mean_squared_error(y_test, y_pred)**.5
            r2          = metrics.r2_score(y_test, y_pred)
            rmsl        = rms(y_test, y_pred)
            acc         = accuracy_log(y_test, y_pred)
            r           = sp.stats.pearsonr(y_test.ravel(), y_pred.ravel())[0] 
            print(
                    "%15s | %4d | %10.2f %10.2f \t %10.2f < %10.2f  %10.2f  %10.2f" % 
                    (dataset_name,  X_test.shape[1],rmse, acc, rmse*rmsl/(min(r2**2,1)), rmsl, r2, r)
                )

            pl.figure(figsize=(16,4));
            s = [i for i in range(len(y_test))] #y_test.argsort()
            #pl.plot([a for a in y_train]+[None for a in y_test]); 
            pl.plot([None for a in y_train]+[a for a in y_test[s]], 'r-.o', label='Real data');
            pl.plot([None for a in y_train]+[a for a in y_pred[s]], 'b-.o', label='Predicted');
            pl.legend(); 
            pl.title(dataset_name+' -- '+target+'\nRMSE = '+str(rmse)+', '+'R$^2$ = '+str(r2)+', '+'ACC = '+str(acc))            
            pl.show()

            pl.figure(figsize=(6,6)); 
            pl.plot(y_test, y_pred, 'ro', y_test, y_test, 'k-')
            pl.title('RMSE = '+str(rmse)+'\n'+'R$^2$ = '+str(r2)+'\n'+'ACC = '+str(acc))
            pl.tight_layout()
            pl.show()




        #for k,kernel in kernels:
            #for n_components in range(1,X_train.shape[1]+1):
                ##scaler      = MinMaxScaler()
                ##scaler      = PowerTransformer(method='yeo-johnson',)
                #scaler      = QuantileTransformer(n_quantiles=X_train.shape[0], random_state=0, output_distribution='normal')
                ##scaler      = FunctionTransformer()
                ##scaler      = FunctionTransformer(np.exp)
                ##transformer = KernelPCA(kernel='linear', n_components=n_components)
                #transformer = FunctionTransformer()
                ##estimator   = ExtraTreesRegressor(n_estimators=1500)
                ##estimator   = SVR(C=1e3)
                ##estimator   = XGBRegressor(n_estimators=60, learning_rate=0.2, objective='reg:squarederror', max_depth=20, random_state=6)
                #estimator   = GaussianProcessRegressor(kernel=kernel, normalize_y=False)
                
                #clf          = Pipeline([
                                        #('scaler'       , scaler), 
                                        #('transformer'  , transformer),
                                        #('estimator'    , estimator)
                                    #])            
                #clf.fit(X_train, y_train.squeeze())

                
                #y_pred      = clf.predict(X_test)
                #rmse    = metrics.mean_squared_error(y_test, y_pred)**.5
                #r2      = metrics.r2_score(y_test, y_pred)
                #rmsl    = rms(y_test, y_pred)
                #acc     = accuracy_log(y_test, y_pred)
                #r       = sp.stats.pearsonr(y_test.ravel(), y_pred.ravel())[0] 
                #print(
                        #"%15s %15s | %4d || %4d | %10.2f %10.2f \t %10.2f < %10.2f  %10.2f  %10.2f" % 
                        #(dataset_name, k, n_components, X_test.shape[1],rmse, acc, rmse*rmsl/(min(r2**2,1)), rmsl, r2, r)
                    #)

                ##pl.figure(figsize=(16,4));
                ##s=y_test.argsort()
                ###pl.plot([a for a in y_train]+[None for a in y_test]); 
                ##pl.plot([None for a in y_train]+[a for a in y_test[s]], 'r-.o', label='Real data');
                ##pl.plot([None for a in y_train]+[a for a in y_pred[s]], 'b-.o', label='Predicted');
                ##pl.legend(); 
                ##pl.title(k+'\n'+dataset_name+' -- '+target+'\nRMSE = '+str(rmse)+', '+'R$^2$ = '+str(r2)+', '+'R = '+str(r))            
                ##pl.show()

                ##pl.figure(figsize=(6,6)); 
                ##pl.plot(y_test, y_pred, 'ro', y_test, y_test, 'k-')
                ##pl.title(k+'\nRMSE = '+str(rmse)+'\n'+'R$^2$ = '+str(r2)+'\n'+'R = '+str(r))
                ##pl.tight_layout()
                ##pl.show()

#%%-----------------------------------------------------------------------------
