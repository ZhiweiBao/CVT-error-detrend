# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:06:39 2019

@author: dell
"""

import data_import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis

from sklearn import svm

import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller #ADF检验
from statsmodels.stats.diagnostic import acorr_ljungbox #白噪声检验
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf #画图定阶
from statsmodels.tsa.arima_model import ARIMA #模型
from statsmodels.tsa.arima_model import ARMA #模型
from statsmodels.stats.stattools import durbin_watson #DW检验
from statsmodels.graphics.api import qqplot #qq图





#dataset = data_import.load_dataset()
#X = data_import.load_dataset(flag = False)[range(0,3600,15), 0:3]
#
#
##X = StandardScaler().fit_transform(X)
#
##pca = PCA(n_components=2, whiten=True)
##X_r = pca.fit(X).transform(X)
#
#transformer = FactorAnalysis(n_components=1)
#X_fa = transformer.fit_transform(X)[:, 0]
#
#
#index = np.arange(240)
#timeseries = pd.DataFrame({'values':X_fa})
#timeseries_train = timeseries[0:220]
#timeseries_test = timeseries[220:240]

###------------------平稳性与非白噪声-----------------------------
# ---------------- diff ----------------------
def difference(timeseries):
    diff1 = timeseries.diff(1).dropna()
    diff2 = diff1.diff(1).dropna()
    
    timeseries.plot(color='black', title='orgin')
    diff1.plot(color='r', title='diff1')
    diff2.plot(color='b', title='diff2')
    
    return diff1, diff2

#---------------------------------------------

# ---------------- rolling ----------------------

#rolmean = timeseries_train.rolling(window=4,center = False).mean()
#rolstd = timeseries_train.rolling(window=4,center = False).std()

#rolmean.plot(color = 'yellow',title='Rolling Mean',figsize=(10,4))
#
#rolstd.plot(color = 'blue',title='Rolling Std',figsize=(10,4))

#---------------------------------------------

# ---------------- ADF ----------------------

#x = np.array(diff1['values'])
#adftest = adfuller(x, autolag='AIC')

#---------------------------------------------

# ---------------- acorr_ljungbox ----------------------

#p_value = acorr_ljungbox(timeseries_train, lags=1)

#---------------------------------------------
###------------------平稳性与非白噪声-----------------------------

###------------------时间序列定阶-----------------------------
#-------------------ACF/PACF------------------------
def plot_acf_pacf(timeseries):
    plot_acf(timeseries,lags=40) #延迟数
    plot_pacf(timeseries,lags=40)
    plt.show()

#---------------------------------------------------

#-------------------信息准则定阶------------------------
def order_select(timeseries):
    AIC = sm.tsa.arma_order_select_ic(timeseries, max_ar=5, max_ma=4,ic='aic')['aic_min_order']
    
    BIC = sm.tsa.arma_order_select_ic(timeseries,max_ar=5, max_ma=4,ic='bic')['bic_min_order']
    
    HQIC = sm.tsa.arma_order_select_ic(timeseries,max_ar=5, max_ma=4,ic='hqic')['hqic_min_order']
    
    print('the AIC is{},\nthe BIC is{}\n the HQIC is{}'.format(AIC,BIC,HQIC))
#---------------------------------------------------
###------------------时间序列定阶-----------------------------


###------------------构建模型与预测-----------------------------
#-------------------ARIMA------------------------
def model_fit(timeseries, opt_order):
    #arima_model = ARIMA(timeseries_train,order =(4,1,1)) #ARIMA模型
    arima_model = ARIMA(timeseries[:int(len(timeseries)*0.9)], order=opt_order)
    #arima_model = ARIMA(timeseries_train,order =(2,1,1))
    model = arima_model.fit()
    
    delta = model.fittedvalues
#    score = 1 - delta.var()/train[1:].var()
    
    pred = model.predict(start=10, end=len(timeseries), dynamic=False)
    #pred[0] = 0
    #pred[1] = 0
    #pred_restore = pred + timeseries_train['values']
    plt.figure()
    plt.plot(pred, color='red')
    plt.plot(timeseries)
    
    #plt.title('RSS: %.4f'% sum((pred_restore-timeseries_train['values'])**2))
    
    plt.show()
    return delta, pred
    
#---------------------------------------------


#-------------------模型检验------------------------
#resid = result.resid
#plt.figure(figsize=(12,8))
#qqplot(resid,line='q',fit=True)
#print('D-W检验值为{}'.format(durbin_watson(resid.values)))
#------------------------------------------------

###------------------构建模型与预测-----------------------------
