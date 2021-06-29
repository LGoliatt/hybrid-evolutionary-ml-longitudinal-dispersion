#!/usr/bin/python
# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn import metrics  
import os

from gplearn.genetic import SymbolicRegressor
import graphviz 
from IPython.display import Image
from gplearn.functions import make_function
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.preprocessing import MaxAbsScaler   

from read_data import *
#%%
def _logical(x1, x2, x3, x4):
    return np.where(x1 > x2, x3, x4)


class SahayRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, 
                 random_state=None
                 ):
        pass
    
    def predict(self, X_test, verbose=0):
        return None

#%%
pd.options.display.float_format = '{:.3f}'.format
datasets = [
            #read_data_ldc_toprak(),
            #read_data_ldc_etemad(),
            #read_data_ldc_noori2017a(),
    
            #read_data_ldc_tayfur( case = 0 ),
            #read_data_ldc_tayfur( case = 1 ),
            #read_data_ldc_tayfur( case = 2 ),
            #read_data_ldc_tayfur( case = 3 ),
            #read_data_ldc_tayfur( case = 4 ),
            #read_data_ldc_tayfur( case = 5 ),
            #read_data_ldc_tayfur( case = 6 ),
            #read_data_ldc_tayfur( case = 7 ),
            
            read_data_ldc_tayfur( case = 11 ),
           ]

def _sigmoid(x):
    return 1/(1+np.exp(-x))
    
IH = np.matrix([
    [  4.6537,-0.55669,0.40181,  6.0446,  6.0268,], 
    [ -2.4918, -3.6388,-4.4239,  3.9328,  3.2533,],  
    [  3.2301,  5.6027, 4.4843,   2.559,  1.0168,], 
    [  3.4226,   6.632,-2.2349,-0.92931, 11.0702,],  
    [ 13.5052,  4.1295, 1.5028, -1.5409, -9.1873,], 
    [-10.6682,  3.0621,-4.1874,  1.7116, -1.3672,],  
    [  7.6101, -1.2886, 2.2302,  1.2365,  2.5023,], 
    [-11.8728, -5.7405,-4.6032, 10.6936, -6.0229,],  
 ])

HO = np.array([
     2.2156, 1.9546,
 -1.1937, 8.379, -12.7936, -11.8344, 3.6668, -13.9641
    ])


bHL = np.array([
      -10.5285, 7.1862, -10.9455, 24.9784, -2.0838,
 0.55192, 7.821, 9.6898
    ])

bOL = np.array([
        -1.0271
    ])


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
    dr='sahay_'+dataset_name.replace(' ','_').replace("'","").lower()
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

        X=X_test
        X = MaxAbsScaler().fit(X_test).transform(X_test)
        o=np.zeros((X.shape[0],1))
        for k in range(X.shape[0]):
            z=X[k]
            a=np.zeros((IH.shape[0],1))
            for j in range(IH.shape[0]):
                aux=0
                for i in range(IH.shape[1]):
                    aux+= IH[j,i]*z[i]
                
                a[j] = bHL[j] + aux
             
            y=_sigmoid(a)                 
            o[k]=_sigmoid( np.dot(y.ravel(), HO)+ bOL[0] ) 
            #y=a                 
            #o[k]=( np.dot(y.ravel(), HO)+ bOL[0] ) 
                    
             
        #clf=est_gp
        ##%%
        #y_pred = clf.predict(X_test)
        
        y_pred = o*max(y_test)
        import seaborn as sns; sns.regplot(x=y_test, y=y_pred.ravel())

        rmse, r2 = metrics.mean_squared_error(y_test, y_pred)**.5, metrics.r2_score(y_test, y_pred)
        r=sp.stats.pearsonr(y_test.ravel(), y_pred.ravel())[0] 
        print(rmse, r2,r)

        pl.figure(figsize=(16,4)); 
        #pl.plot([a for a in y_train]+[None for a in y_test]); 
        pl.plot([None for a in y_train]+[a for a in y_test], 'r-.o', label='Real data');
        pl.plot([None for a in y_train]+[a for a in y_pred], 'b-.o', label='Predicted');
        pl.legend(); pl.title(dataset_name+' -- '+target+'\nRMSE = '+str(rmse)+', '+'R$^2$ = '+str(r2)+', '+'R = '+str(r))
        pl.show()

        pl.figure(figsize=(6,6)); 
        pl.plot(y_test, y_pred, 'ro', y_test, y_test, 'k-')
        pl.title('RMSE = '+str(rmse)+'\n'+'R$^2$ = '+str(r2)+'\n'+'R = '+str(r))
        pl.show()

#%%-----------------------------------------------------------------------------
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
reg = MLPRegressor(
                    hidden_layer_sizes=(8),
                    activation='logistic',
                    validation_fraction=0.1,
                    #solver='lbfgs',
                    learning_rate='constant',
                    learning_rate_init=0.02,
                    max_iter=10000,
                    alpha=0,
                    random_state=None,     
                    )
#
#from util.MLP import  MLPRegressor
#reg = MLPRegressor(
#                    n_hidden=(8),            
#                    )


model = Pipeline(steps=[('normalizer',MaxAbsScaler()), ('estimator',reg)])

y_scaler = MaxAbsScaler()
y_scaler.fit(y_train.reshape(-1,1))

model.fit(X_train, y_scaler.transform(y_train.reshape(-1,1)).ravel())

y_pred =  y_scaler.fit(y_test.reshape(-1,1)).inverse_transform(model.predict(X_test).reshape(-1,1))

#--
import seaborn as sns; sns.regplot(x=y_test, y=y_pred.ravel())

rmse, r2 = metrics.mean_squared_error(y_test, y_pred)**.5, metrics.r2_score(y_test, y_pred)
r=sp.stats.pearsonr(y_test.ravel(), y_pred.ravel())[0] 
print(rmse, r2,r)

pl.figure(figsize=(16,4)); 
#pl.plot([a for a in y_train]+[None for a in y_test]); 
pl.plot([None for a in y_train]+[a for a in y_test], 'r-.o', label='Real data');
pl.plot([None for a in y_train]+[a for a in y_pred], 'b-.o', label='Predicted');
pl.legend(); pl.title(dataset_name+' -- '+target+'\nRMSE = '+str(rmse)+', '+'R$^2$ = '+str(r2)+', '+'R = '+str(r))
pl.show()

pl.figure(figsize=(6,6)); 
pl.plot(y_test, y_pred, 'ro', y_test, y_test, 'k-')
pl.title('RMSE = '+str(rmse)+'\n'+'R$^2$ = '+str(r2)+'\n'+'R = '+str(r))
pl.show()
#--
#%%
#model.named_steps['estimator'].coefs_[0]=IH.T
#model.named_steps['estimator'].coefs_[1]= HO.reshape(-1,1)
#model.named_steps['estimator'].intercepts_[0]= bHL
#model.named_steps['estimator'].intercepts_[1]= bOL
#
#
#clf.coefs_[0]=IH.T
#clf.coefs_[1]= HO.reshape(-1,1)
#clf.intercepts_[0]= bHL
#clf.intercepts_[1]= bOL

#%%
from ffnet import ffnet, mlgraph, tmlgraph, imlgraph

X,y = X_train, y_train

n_samples, n_features = X.shape
try:
  n_outputs=y.shape[1]
except:
  n_outputs=1

n_hidden=[8]
par=np.r_[n_features, n_hidden,n_outputs]
conec = mlgraph(par, biases=True)
net = ffnet(conec)
net.train_rprop(X_train, y_train, disp=True)

y_pred =  net.call(X_test)

#--
import seaborn as sns; sns.regplot(x=y_test, y=y_pred.ravel())

rmse, r2 = metrics.mean_squared_error(y_test, y_pred)**.5, metrics.r2_score(y_test, y_pred)
r=sp.stats.pearsonr(y_test.ravel(), y_pred.ravel())[0] 
print(rmse, r2,r)

pl.figure(figsize=(16,4)); 
#pl.plot([a for a in y_train]+[None for a in y_test]); 
pl.plot([None for a in y_train]+[a for a in y_test], 'r-.o', label='Real data');
pl.plot([None for a in y_train]+[a for a in y_pred], 'b-.o', label='Predicted');
pl.legend(); pl.title(dataset_name+' -- '+target+'\nRMSE = '+str(rmse)+', '+'R$^2$ = '+str(r2)+', '+'R = '+str(r))
pl.show()

#pl.figure(figsize=(6,6)); 
#pl.plot(y_test, y_pred, 'ro', y_test, y_test, 'k-')
#pl.title('RMSE = '+str(rmse)+'\n'+'R$^2$ = '+str(r2)+'\n'+'R = '+str(r))
#pl.show()
#--
#%%-----------------------------------------------------------------------------
