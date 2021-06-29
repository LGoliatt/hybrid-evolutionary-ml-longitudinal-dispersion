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
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, MaxAbsScaler, Normalizer, StandardScaler, MaxAbsScaler, FunctionTransformer, PowerTransformer, QuantileTransformer
from sklearn.svm import SVR
from xgboost import XGBRegressor
from util.RBFNN import RBFNNRegressor as RBFNN

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
    dr='kpca_'+dataset_name.replace(' ','_').replace("'","").lower()
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
        f, (ax0, ax1) = pl.subplots(1, 2)

        ax0.hist(y_train, bins=100, **{'density': True})
        ax0.set_xlim([0, 2000])
        ax0.set_ylabel('Probability')
        ax0.set_xlabel('Target')
        ax0.set_title('Target distribution')

        y_train_trans = np.log1p(y_train)
        ax1.hist(y_train_trans, bins=100, **{'density': True})
        ax1.set_ylabel('Probability')
        ax1.set_xlabel('Target')
        ax1.set_title('Transformed target distribution')

        f.suptitle("Synthetic data", y=0.035)
        f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        #               
        kernel = ["linear" , "poly" , "rbf" , "sigmoid" , "cosine" ]
        print('\n')
        for kernel in kernel:
            #print('\n')
            for n_components in range(1,X_train.shape[1]+1):
                
                #scaler      = FunctionTransformer(np.exp)
                #scaler      = MinMaxScaler()
                #scaler      = PowerTransformer(method='yeo-johnson',)
                scaler      = QuantileTransformer(n_quantiles=49, random_state=0, output_distribution='normal')
                #scaler      = FunctionTransformer()
                transformer = KernelPCA(kernel=kernel, n_components=n_components)
                #transformer = FunctionTransformer()
                #estimator   = ExtraTreesRegressor(n_estimators=1500)
                estimator   = SVR(C=1e3)
                #estimator   = XGBRegressor(n_estimators=60, learning_rate=0.2, objective='reg:squarederror', max_depth=20, random_state=6)
                #estimator   = RBFNN(func='rbf')
                clf          = Pipeline([
                                        ('scaler',scaler), 
                                        ('kpca', transformer),
                                        ('estimator', estimator)
                                    ])            
                clf.fit(X_train, y_train.squeeze())

                
                y_pred      = clf.predict(X_test)
                rmse    = metrics.mean_squared_error(y_test, y_pred)**.5
                r2      = metrics.r2_score(y_test, y_pred)
                rmsl    = rms(y_test, y_pred)
                acc     = accuracy_log(y_test, y_pred)
                r       = sp.stats.pearsonr(y_test.ravel(), y_pred.ravel())[0] 
                print(
                        "%15s %8s | %4d || %4d | %10.2f %10.2f \t %10.2f < %10.2f  %10.2f  %10.2f" % 
                        (dataset_name, kernel, n_components, X_test.shape[1],rmse, acc, rmse*rmsl/(min(r2**2,1)), rmsl, r2, r)
                    )

                #pl.figure(figsize=(16,4)); 
                ##pl.plot([a for a in y_train]+[None for a in y_test]); 
                #pl.plot([None for a in y_train]+[a for a in y_test], 'r-.o', label='Real data');
                #pl.plot([None for a in y_train]+[a for a in y_pred], 'b-.o', label='Predicted');
                #pl.legend(); 
                #pl.title(kernel+'\n'+dataset_name+' -- '+target+'\nRMSE = '+str(rmse)+', '+'R$^2$ = '+str(r2)+', '+'R = '+str(r))            
                #pl.show()

                #pl.figure(figsize=(6,6)); 
                #pl.plot(y_test, y_pred, 'ro', y_test, y_test, 'k-')
                #pl.title(kernel+'\nRMSE = '+str(rmse)+'\n'+'R$^2$ = '+str(r2)+'\n'+'R = '+str(r))
                #pl.tight_layout()
                #pl.show()

#%%-----------------------------------------------------------------------------
