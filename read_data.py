#!/usr/bin/python
# -*- coding: utf-8 -*
import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")


from sklearn.decomposition import PCA, KernelPCA
from scipy.spatial import distance
from sklearn.cluster import * #DBSCAN,KMeans,MeanShift,Ward,AffinityPropagation,SpectralClustering
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import kneighbors_graph
#from sklearn.mixture import GMM, DPGMM
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from xlrd import open_workbook,cellname,XL_CELL_TEXT
#import openpyxl as px
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  TimeSeriesSplit

#from mvpa2.suite import SimpleSOMMapper # http://www.pymvpa.org/examples/som.html
#from sklearn.grid_search import GridSearchCV
from sklearn.metrics import euclidean_distances

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

import numpy as np
import pylab as pl
import matplotlib.pyplot as pl
import pandas as pd
import os
import sys
import re
from scipy import stats
#from sklearn import cross_validation
from sklearn import preprocessing

#from sklearn.cross_validation import cross_val_score, ShuffleSplit, LeaveOneOut, LeavePOut, KFold
#-------------------------------------------------------------------------------
def mean_std_percentual_error(y_true, y_pred):    
  y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
  s=np.mean(np.abs(y_pred - y_true)/np.abs(y_true))*100, np.std(np.abs(y_pred - y_true)/np.abs(y_true))*100
  return s
#-------------------------------------------------------------------------------
def max_div_cv(n):
  l=[]
  for i in range (1, n): 
    if (n%i == 0  and i<8): 
      l.append(i)	
      
  return max(4,max(l))
#-------------------------------------------------------------------------------
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[np.int(window_len/2-1):-np.int(window_len/2)]
#-------------------------------------------------------------------------------
def read_data_ldc_tayfur(filename='./data/data_ldc_vijay/tayfur_2005.csv', case=0):
#%%    
    #filename='./data/data_ldc_vijay/tayfur_2005.csv'
    filename='./data/data_ldc_vijay/vijay_ldc_paper_data.csv'
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
      'task'            : 'regression',
      'name'            : 'LDC case '+str(case),
      'feature_names'   : np.array(feature_names),
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.T,
      'y_test'          : y_test.T,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "t.ly/jJP6J",      
      'normalize'       : 'MinMax',
      'date_range'      : None
      }
#%%
    return dataset  
#%%
def read_data_ldc_vijay(filename='./data/data_ldc_vijay/vijay_ldc_paper_data.csv'):
#%%    
    #filename='./data/data_ldc_vijay/vijay_ldc_paper_data.csv'
    df=pd.read_csv(filename,  delimiter=';', index_col='Training')
    target_names=['Kx (m2 /s)']
    feature_names = df.columns.drop(target_names)
    
    X_train = df[feature_names][df.index==1].values
    X_test  = df[feature_names][df.index==0].values
    y_train = df[df.index==1][target_names].values
    y_test  = df[df.index==0][target_names].values
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'            : 'regression',
      'name'            : 'Longitudinal Dispersion Coefficient',
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.T,
      'y_test'          : y_test.T,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "t.ly/jJP6J",      
      'normalize'       : 'MinMax',
      'date_range'      : None
      }
    return dataset  
#%%
def read_data_ldc_etemad(filename='./data/data_ldc_vijay/etemad_2012.csv'):
#%%    
    #
    filename='./data/data_ldc_vijay/etemad_2012.csv'
    df=pd.read_csv(filename,  delimiter=';', index_col='No')
    target_names=['Kx(m2∕s)']
    df.drop(['Stream', 'sigma'], axis=1, inplace=True)
    df['W(m)']/df['H(m)']
    df['U/U*'] =df['U(m∕s)']/df['U*(m∕s)']
    df['Kx/U*H'] =df['Kx(m2∕s)']/df['U*(m∕s)']/df['H(m)']
    feature_names = df.columns.drop(target_names)
    
    X_train = df[feature_names][df.index<=119]#.values
    X_test  = df[feature_names][df.index> 119]#.values
    y_train = df[df.index<=119][target_names]#.values
    y_test  = df[df.index> 119][target_names]#.values
    
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'            : 'regression',
      'name'            : 'LDC 149',
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train.values,
      'X_test'          : X_test.values,
      'y_train'         : y_train.T.values,
      'y_test'          : y_test.T.values,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "t.ly/jJP6J",      
      'normalize'       : 'MinMax',
      'date_range'      : None
      }
#%%    
def read_data_ldc_noori2017a(filename='./data/data_ldc_vijay/noori2017a.csv'):
#%%    
    #
    filename='./data/data_ldc_vijay/noori2017a.csv'
    df=pd.read_csv(filename,  delimiter=';', index_col='Number')
    target_names=['Kx']
    df.drop(['Stream',], axis=1, inplace=True, errors='ignore')
    #df['W(m)']/df['H(m)']
    #df['U/U*'] =df['U(m∕s)']/df['U*(m∕s)']
    #df['Kx/U*H'] =df['Kx(m2∕s)']/df['U*(m∕s)']/df['H(m)']
    feature_names = df.columns.drop(target_names)
    
    n = 70
    X_train = df[feature_names][df.index<=n]#.values
    X_test  = df[feature_names][df.index> n]#.values
    y_train = df[df.index<=n][target_names]#.values
    y_test  = df[df.index> n][target_names]#.values
    
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'            : 'regression',
      'name'            : 'LDC Noori 2017a',
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train.values,
      'X_test'          : X_test.values,
      'y_train'         : y_train.T.values,
      'y_test'          : y_test.T.values,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "https://tinyurl.com/tz976ts",      
      'normalize'       : 'MinMax',
      'date_range'      : None
      }
#%%    
    return dataset  
#%%
def read_data_ldc_toprak(filename='./data/data_ldc_vijay/toprak_2004.csv'):
#%%    
    #filename='./data/data_ldc_vijay/toprak_2004.csv'
    df=pd.read_csv(filename, delimiter=';', index_col='Dataset') 
    target_names=[ 'D1(m2/s)']
    df.drop(labels=['No', 'Source', 'Channel',], axis=1, inplace=True)
    feature_names = df.columns.drop(target_names)
    
    X_train = df[feature_names][df.index=='C'].values
    X_test  = df[feature_names][df.index=='CC'].values
    y_train = df[df.index=='C'][target_names].values
    y_test  = df[df.index=='CC'][target_names].values
    n_samples, n_features = X_train.shape
    dataset=  {
      'task'            : 'regression',
      'name'            : 'LDC Toprak 2008',
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.T,
      'y_test'          : y_test.T,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "www.doi.org/10.1002/hyp.7012",      
      'normalize'       : 'MinMax',
      'date_range'      : None
      }
    #%%
    return dataset  
#%% 
#%%-----------------------------------------------------------------------------
if __name__ == "__main__":
    datasets = [
#                read_data_energy_appliances(),
                 read_data_ldc_tayfur(),
#                #read_data_efficiency(),
#                read_data_burkina_faso_boromo(),
#                read_data_burkina_faso_dori(),
#                read_data_burkina_faso_gaoua(),
#                read_data_burkina_faso_po(),
#                read_data_burkina_faso_bobo_dioulasso(),
#                read_data_burkina_faso_bur_dedougou(),
#                read_data_burkina_faso_fada_ngourma(),
#                read_data_burkina_faso_ouahigouy(),
#                read_data_b2w(),
#                read_data_qsar_aquatic(),
                 #read_data_cahora_bassa(),
                 #read_data_iraq_monthly(),
#                read_data_cahora_bassa_sequence(look_back=21, look_forward=7, kind='ml', unit='day'),                
#                read_data_cergy(),
#                read_data_bogas(),
#                read_data_dutos_csv(),
#                read_data_yeh(),
#                read_data_lim(),
#                read_data_siddique(),
#                read_data_pala(),
#                read_data_bituminous_marshall(),
#                read_data_slump(),
#                read_data_shamiri(),
#                read_data_borgomano(),
#                read_data_xie_dgf(),
#                read_data_xie_hgf(),
#                read_data_nguyen_01(),
#                read_data_nguyen_02(),
#                read_data_tahiri(),
            ]
    for D in datasets:
        print('='*80+'\n'+D['name']+'\n'+'='*80)
        print(D['reference'])
        print( D['y_train'])
        print('\n')
#%%-----------------------------------------------------------------------------
#
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn import datasets, preprocessing
#from sklearn.model_selection import train_test_split
#from neupy import algorithms, layers
#from sklearn import metrics
#
#from gplearn.genetic import SymbolicRegressor
#import graphviz 
#dataset = read_data_iraq_monthly()
#dataset = read_data_cahora_bassa_monthly()
##dataset = read_data_burkina_faso_boromo()
#
#
#(task, name, feature_names, target_names, n_samples, n_features, 
#     X_train, X_test, y_train, y_test, targets, true_labels, 
#     predicted_labels, descriptions, items, reference, 
#     normalize) = dataset.values()
#
#y_train = np.array([[i] for i in y_train.ravel()])
#y_test  = np.array([[i] for i in y_test.ravel()])
#
##optimizer = algorithms.Hessian(
##        network=[
##            layers.Input(X_train.shape[1]),
##            layers.Relu(50),
##            layers.Relu(10),
##            layers.Linear(y_train.shape[1]),
##        ],
##        verbose=True,
##    )
##
##optimizer.train(X_train, y_train, X_test, y_test, epochs=10)
##y_predict = optimizer.predict(X_test)
##y_test = target_scaler.inverse_transform(y_test.reshape((-1, 1)))
##y_predict = target_scaler.inverse_transform(y_predict).T.round(1)
##%%
#
#for std in np.linspace(45,70,20):
#    estimator = algorithms.GRNN(std=std, verbose=False)
#    estimator.train(X_train, y_train)
#    y_predict = estimator.predict(X_test)
#    print(metrics.r2_score(y_test, y_predict))
#
#pl.figure(figsize=(16,4)); 
#pl.plot([a for a in y_train]+[None for a in y_test]); 
#pl.plot([None for a in y_train]+[a for a in y_test]); 
#pl.plot([None for a in y_train]+[a for a in y_predict]);
#pl.show()


#%%-----------------------------------------------------------------------------

