#!/usr/bin/python
# -*- coding: utf-8 -*
"""
- Estimation of natural streams longitudinal dispersion coefficient using 
   hybrid evolutionary machine learning model
   
- Source file: read_data.py
- Author: Leonardo Goliatt (leonardo.goliatt@ufjf.edu.br)
"""   
import pandas as pd
import numpy as np
#-------------------------------------------------------------------------------
def read_data_ldc_tayfur(filename='./data/data_ldc_vijay/vijay_ldc_paper_data.csv', 
                         case=0,
                         feature_selection=False,
                         ):
#%%    
    df=pd.read_csv(filename,  delimiter=';', index_col='Training')
    df.drop(labels=['Stream'], axis=1, inplace=True)
    col_names = ['$B$', '$H$', '$U$', '$u^*$', '$Q$', '$U/u^*$', '$\\beta$','$\\sigma$', '$K_x$']
    target_names  = ['$K_x$']
    df.columns    = col_names
    
    if   case == 0:
        feature_names = ['$B$', '$H$', '$U$', '$u^*$',   '$Q$', '$U/u^*$', '$\\beta$', '$\\sigma$',]
    elif case == 1:
        feature_names = ['$B$', '$H$', '$U$',                                                      ]
    elif case == 2:
        feature_names = [                                '$Q$',                                    ]
    elif case == 3:
        feature_names = [             '$U$',                                                       ]
    elif case == 4: 
        feature_names = [             '$U$',                               '$\\beta$',             ]
    elif case == 5:
        feature_names = [             '$U$',                               '$\\beta$', '$\\sigma$',]
    elif case == 6:
        feature_names = [                                       '$U/u^*$',                         ]
    elif case == 7:
        feature_names = [                                       '$U/u^*$', '$\\beta$', '$\\sigma$',]
    elif   case == 8:
         feature_names = ['$B$',      '$U$',                                           '$\\sigma$',]
    elif   case == 9:
         feature_names = ['$B$',      '$U$',             '$Q$',                        '$\\sigma$',]
    elif   case == 10:
         feature_names = ['$B$',      '$U$',                               '$\\beta$', '$\\sigma$',]
    elif   case == 11:
         feature_names = ['$B$', '$H$', '$U$', '$u^*$', '$\\sigma$',]
    else:
        sys.exit('Case not found')
        
    df_train =    df[df.index=='*']    
    df_test  =    df[df.index=='**']    
    df_train.describe().T.to_latex('/tmp/train.tex')
    df_test.describe().T.to_latex('/tmp/test.tex')
    
    X_train = df[feature_names][df.index=='*'].values
    X_test  = df[feature_names][df.index=='**'].values
    y_train = df[df.index=='*'][target_names].values
    y_test  = df[df.index=='**'][target_names].values
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'             : 'regression',
      'name'             : 'LDC case '+str(case),
      'feature_names'    : np.array(feature_names),
      'target_names'     : target_names,
      'n_samples'        : n_samples, 
      'n_features'       : n_features,
      'X_train'          : X_train,
      'X_test'           : X_test,
      'y_train'          : y_train.T,
      'y_test'           : y_test.T,      
      'targets'          : target_names,
      'true_labels'      : None,
      'predicted_labels' : None,
      'descriptions'     : 'None',
      'items'            : None,
      'reference'        : "https://doi.org/10.1061/(ASCE)0733-9429(2005)131:11(991)",      
      'normalize'        : 'MinMax',
      'date_range'       : None,
      'feature_selection': feature_selection,
      }
#%%
    return dataset  
