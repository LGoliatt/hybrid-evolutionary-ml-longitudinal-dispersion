#!/usr/bin/python
# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn import metrics  
import os

from sklearn.ensemble import  RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance

from read_data import *
#%%

pd.options.display.float_format = '{:.3f}'.format
datasets = [
            read_data_ldc_tayfur(),
           ]


random_seed=0

models = [ 
            ('RF', RandomForestRegressor(n_estimators=2500, random_state=random_seed)), 
            ('ET', ExtraTreesRegressor(n_estimators=2500, random_state=random_seed)), 
            ('GB', GradientBoostingRegressor(n_estimators=2500, random_state=random_seed)),
            ('XGB', XGBRegressor(n_estimators=2500, random_state=random_seed)),
         ]

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
    dr='tpot_'+dataset_name.replace(' ','_').replace("'","").lower()
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
        
        for model_name, model in models:
            model.fit(X_train, y_train)
            print(model_name+" train accuracy: %0.3f" % model.score(X_train, y_train))
            print(model_name+" test  accuracy: %0.3f" % model.score(X_test, y_test))
            importances = model.feature_importances_            
            #std = np.std([tree.feature_importances_ for tree in np.ravel(model.estimators_)],axis=0)
            indices = np.argsort(importances)[::-1]
            
            print("Feature ranking for "+model_name)
            for f in range(X_train.shape[1]):
                print("%d. feature %d  -- %10s -- (%f)" % (f + 1, indices[f], feature_names[f], importances[indices[f]]))
                
            sorted_idx = importances.argsort()            
            y_ticks = np.arange(0, len(feature_names))
            fig, ax = pl.subplots()
            ax.barh(y_ticks, importances[sorted_idx])
            ax.set_yticklabels(feature_names[sorted_idx])
            ax.set_yticks(y_ticks)
            ax.set_title(model_name)
            fig.tight_layout()
            pl.show()