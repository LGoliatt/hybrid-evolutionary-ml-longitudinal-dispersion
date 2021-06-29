#!/usr/bin/python
# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from scipy import stats
from sklearn import metrics  
import seaborn as sns
import os

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression,SelectFromModel
from sklearn.feature_selection import RFECV

from itertools import  combinations
from math import factorial
    
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami

from read_data import *
#%%
pl.rc('text', usetex=True)
pl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
pl.rc('font', family='serif',  serif='Times')
sns.set_palette("Set1", )
sns.set_context("paper", font_scale=1.8, )

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)
    
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

import scipy.stats as st
def get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto",
                  "genextreme", "uniform"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]

#%%
basename='eml____'

pd.options.display.float_format = '{:.3f}'.format
datasets = [
            read_data_ldc_tayfur( case = 0 ),
            #read_data_cahora_bassa_sequence(look_back=1, look_forward=1, kind='ml', unit='month'),            
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
    dr='data_analysis_'+dataset_name.replace(' ','_').replace("'","").lower()
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
               
        #%%
        D=pd.DataFrame()
        for X,y in [(X_train, y_train), (X_test, y_test)]:
            A = pd.DataFrame(data=X, columns=feature_names)
            A[target]=y
            print(A.describe())
            D = pd.concat([D,A])
            
        print(D.describe().T.to_latex(index=False))
        #%%
        D=pd.DataFrame()
        for X,y in [(X_train, y_train), (X_test, y_test)]:
            A = pd.DataFrame(data=X, columns=feature_names)
            for c in A:
                bd=get_best_distribution(A[c].values)
                print()
                
                pl.figure()
                sns.distplot(A[c], label=c,)
                pl.plot()
                
            A[target]=y
                        
        #%%
        D = pd.DataFrame(data=X_train, columns=feature_names) 
        D[target] = y_train
        
        corr=D.corr(method='pearson',) # 'kendall', 'spearman'
        corr.drop(labels=[target], axis=0, inplace=True)
        corr[target].plot(kind='bar')
        pl.show()
        
        pl.figure(figsize=(8,7))
        sns.heatmap(corr.drop([target], axis=1),
            vmin=-1,
            cmap='coolwarm',
            annot=True);
        
        pl.figure(figsize=(2,8))
        corr=corr.sort_values(by=[target], axis=0, ascending=False)
        sns.heatmap(corr[[target]],
            vmin=-1,
            cmap='coolwarm',
            annot=True);
                    
        pl.figure(figsize=(8,.8))
        corr=corr.sort_values(by=[target], axis=0, ascending=False)
        g = sns.heatmap(corr[[target]].T,
            vmin=-1, cbar=False,
            cmap='coolwarm',
            annot=True);
        
        fn = basename+'300dpi_correlation'+'_target_'+'__'+target+'.png'
        fn = re.sub('\^','', re.sub('\$','',fn))
        fn = re.sub('\(','', re.sub('\)','',fn)) 
        fn = re.sub(' ','_', re.sub('\/','',fn))
        fn = re.sub('\\\\','', re.sub('^','',fn))
        fn = re.sub('-','_', re.sub('\/','',fn)).lower()
        fn = fn.lower()
        print(fn)
        pl.savefig(fn, transparent=True, optimize=True,
                   bbox_inches='tight', 
                   dpi=300)
                        
        pl.show()
        #%%
        comb=[ (k,len([i for i in combinations(feature_names,k)])) for k in range(1,n_features+1)]
        cols=['$r$', '$C('+str(len(feature_names))+',r)$']
        C=pd.DataFrame(data=comb, 
                        columns=cols)
        C.plot(kind='bar')
        g = sns.factorplot(x=cols[0], y=cols[1], data=C, kind='bar')
        #%%
        target_var='$Q$'    ; order=2
        target_var='$K_x$'  ; order=1
        for f in feature_names:    
            ratio=np.array([1,1.618])*2.5
            pl.figure(figsize=ratio)
            slope, intercept, r_value, p_value, std_err = stats.linregress(D[f], D[target_var])
            g=sns.regplot(x=f, y=target_var, data=D, order=order,)
                          #line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})            
            #g.legend()

            fn = basename+'300dpi_scatter'+'_target_'+f+'_'+target_var+'.png'
            fn = re.sub('\^','', re.sub('\$','',fn))
            fn = re.sub('\(','', re.sub('\)','',fn)) 
            fn = re.sub(' ','_', re.sub('\/','',fn))
            fn = re.sub('\\\\','', re.sub('^','',fn))
            fn = re.sub('\*','s', re.sub('^','',fn))
            fn = re.sub('-','_', re.sub('\/','',fn)).lower()
            fn = fn.lower()
            print(fn)
            pl.savefig(fn, transparent=True, optimize=True,
                       bbox_inches='tight', 
                       dpi=300)            
            pl.show()
        #%%
        model  = RandomForestRegressor()
        model.fit(X_train, y_train)# Mostrando importância de cada feature
        model.feature_importances_
        importances = pd.Series(data=model.feature_importances_, index=feature_names)
        importances.sort_values(ascending=False, inplace=True)
        sns.barplot(x=importances, y=importances.index, orient='h').set_title('Feature Importance')
        pl.show()
        #%%
        model  = XGBRegressor()
        model.fit(X_train, y_train)# Mostrando importância de cada feature
        model.feature_importances_
        importances = pd.Series(data=model.feature_importances_, index=feature_names)
        importances.sort_values(ascending=False, inplace=True)
        sns.barplot(x=importances, y=importances.index, orient='h').set_title('Feature Importance')
        pl.show()
        #%%
        problem = {
          'num_vars': n_features,
          'names': feature_names,
          'bounds': list(zip(X_train.min(axis=0), X_train.max(axis=0))),
        }
        param_values = saltelli.sample(problem, 1000)
        Y = Ishigami.evaluate(param_values)
        Si = sobol.analyze(problem, Y, print_to_console=True)

        #%%
        for k in range(1,n_features):
            fs=SelectKBest(f_regression, k=k).fit(X_train, y_train)
            print(k,feature_names[fs.get_support()])
            
        #%%
        models=[
                SVR(kernel='linear', C=1e0),
                SVR(kernel='linear', C=1e1),
                SVR(kernel='linear', C=1e2),
                XGBRegressor(),
                RandomForestRegressor(),
                ]
        for reg in models: 
            reg.fit(X_train, y_train)
            fs = SelectFromModel(estimator=reg).fit(X_train, y_train)
            print('\n',reg.__str__(),'\n',feature_names[fs.get_support()])

#%%-----------------------------------------------------------------------------
