# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 09:20:49 2020

@author: dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

source_data_1 = pd.read_excel("E:/Workspaces/Python3/CVTAnomalyDetection/data/Table3_data_15min.xlsx")
source_data_2 = pd.read_excel("E:/Workspaces/Python3/CVTAnomalyDetection/data/Table4_data_15min.xlsx")
source_data = pd.concat([source_data_1, source_data_2], axis=1)

#source_data = pd.read_excel("E:/Workspaces/Python3/CVTAnomalyDetection/data/zhenwu_15min.xlsx")


#timeseries = pd.date_range(start = '2019-03-20 00:00:00', end = '2019-06-22 23:59:59', freq='15min')
#timeseries = pd.date_range(start = '2017-11-29 00:00:00', end = '2018-01-27 23:59:59', freq='15min')
#timeseries = pd.date_range(start = '2018-01-22 00:00:00', end = '2018-03-22 23:59:59', freq='15min')
timeseries = pd.date_range(start = '2018-02-01 00:00:00', end = '2018-03-19 23:59:59', freq='15min')

#dataset = source_data.loc[timeseries]
#dataset = pd.concat([dataset.iloc[:,:3], dataset.iloc[:,6:7], dataset.iloc[:,4:6]], axis=1)
#dataset_1 = dataset.iloc[:,:3]
#dataset_2 = pd.concat([dataset.iloc[:,6:7], dataset.iloc[:,4:6]], axis=1)
#dataset_1_error = dataset.iloc[:,:3]
#dataset_2_error = pd.concat([dataset.iloc[:,6:7], dataset.iloc[:,4:6]], axis=1)
#--------------数据清洗1------------------------
#dataset = source_data.loc[timeseries]
##dataset = dataset.iloc[:,-6:]
#dataset_1 = dataset.iloc[:,:3]
#dataset_2 = dataset.iloc[:,3:]
#re_error = (dataset_1.values - dataset_2.values)/(dataset_1.values + dataset_2.values)*2*100
#dataset.iloc[re_error[:,0] < -.012] = np.nan
#dataset.iloc[re_error[:,0] > .025] = np.nan
#dataset.iloc[re_error[:,1] < -0.156] = np.nan
#dataset.iloc[re_error[:,1] > -0.139] = np.nan
#dataset.iloc[[3978,4633,5035,5038]] = np.nan
#dataset=dataset.interpolate(method='linear')

#--------------数据清洗2------------------------
dataset = source_data.loc[timeseries]
##dataset = dataset.iloc[:,-6:]
dataset_1 = dataset.iloc[:,:3]
dataset_2 = dataset.iloc[:,3:]
re_error = (dataset_1.values - dataset_2.values)/(dataset_1.values + dataset_2.values)*2*100
dataset.iloc[re_error[:,0] < -.012] = np.nan
dataset.iloc[re_error[:,0] > .025] = np.nan
dataset.iloc[re_error[:,1] < -0.153] = np.nan
dataset.iloc[re_error[:,1] > -0.131] = np.nan
#dataset.iloc[[682,2172,2174,3590]] = np.nan
dataset.iloc[[1212,1214,2630]] = np.nan
dataset=dataset.interpolate(method='linear')



#-------------加误差-----------------------------------
dataset_error = dataset.loc[timeseries]

step_error_point = 3500
step_error_array = [1 for _ in range(step_error_point)]
step_error_array.extend([(1+0e-3) for _ in range(len(dataset_error)-step_error_point)])

gradual_error_point = 3500
gradual_error_array = [1 for _ in range(gradual_error_point)]
gradual_error_array.extend([(1+1e-6*i) for i in range(len(dataset_error)-gradual_error_point)])

phase_fault = 1
dataset_error.iloc[:,phase_fault] = dataset_error.iloc[:,phase_fault]*step_error_array

dataset_1 = dataset.iloc[:,:3]
dataset_2 = dataset.iloc[:,3:]
dataset_1_error = dataset_error.iloc[:,:3]
dataset_2_error = dataset_error.iloc[:,3:]

#------------------数据划分------------------------
train_data_1 =  dataset_1_error.iloc[:2000,:]
train_data_2 =  dataset_2_error.iloc[:2000,:]
test_data_1 =  dataset_1_error.iloc[2000:,:]
test_data_2 =  dataset_2_error.iloc[2000:,:]
#----------------标准化----------------------
scaler_1 = StandardScaler()
scaler_1.fit(train_data_1)
train_data_1 = scaler_1.transform(train_data_1)
test_data_1 = scaler_1.transform(test_data_1)

scaler_2 = StandardScaler()
scaler_2.fit(train_data_2)
train_data_2 = scaler_2.transform(train_data_2)
test_data_2 = scaler_2.transform(test_data_2)
#
#train_data[:, 0] = train_data[:, 0]*5
#test_data[:, 0] = test_data[:, 0]*5

#len_window = 672


#----------------平滑----------------------
def movingaverage(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')

def smooth(a,WSZ):
    # a:原始数据，NumPy 1-D array containing the data to be smoothed
    # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化 
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

#----------------计算统计量----------------------
#train_data = dataset_error.iloc[:2000,:]
#test_data = dataset_error.iloc[2000:,:]
#
#transformer = PCA(n_components=1, svd_solver = 'full')
##pca = PCA(n_components='mle', svd_solver = 'full')
##transformer = KernelPCA(n_components=1, kernel='linear', fit_inverse_transform=True)
#transformer.fit(train_data)
#
#PCs = transformer.transform(train_data)
#PCs_test = transformer.transform(test_data)
#
#train_data_est = transformer.inverse_transform(PCs)
#test_data_est = transformer.inverse_transform(PCs_test)
#
#res = train_data - train_data_est
#res_test = test_data - test_data_est
#
#
#res_1 = res.iloc[:,:3]
#res_2 = res.iloc[:,3:]
#res_test_1 = res_test.iloc[:,:3]
#res_test_2 = res_test.iloc[:,3:]
#
#fig = plt.figure()
#plot_data_1 = movingaverage(np.append(res_1.iloc[:,0], res_test_1.iloc[:,0]), 96)
#plot_data_2 = movingaverage(np.append(res_2.iloc[:,0], res_test_2.iloc[:,0]), 96)
#plot_data_3 = movingaverage(np.append(res_1.iloc[:,1], res_test_1.iloc[:,1]), 96)
#plot_data_4 = movingaverage(np.append(res_2.iloc[:,1], res_test_2.iloc[:,1]), 96)
#plot_data_5 = movingaverage(np.append(res_1.iloc[:,2], res_test_1.iloc[:,2]), 96)
#plot_data_6 = movingaverage(np.append(res_2.iloc[:,2], res_test_2.iloc[:,2]), 96)
#plot_X = list(range(1, len(plot_data_1)+1))
#ax = fig.add_subplot(311) 
#ax.plot(plot_X, plot_data_1)
#ax.plot(plot_X, plot_data_2)
#ax.hlines(0, 1, len(plot_data_1)+1, colors = "black", linestyles = "dashed")
#ax = fig.add_subplot(312) 
#ax.plot(plot_X, plot_data_3)
#ax.plot(plot_X, plot_data_4)
#ax.hlines(0, 1, len(plot_data_1)+1, colors = "black", linestyles = "dashed")
#ax = fig.add_subplot(313) 
#ax.plot(plot_X, plot_data_5)
#ax.plot(plot_X, plot_data_6)
#ax.hlines(0, 1, len(plot_data_1)+1, colors = "black", linestyles = "dashed")








transformer_1 = PCA(n_components=1, svd_solver = 'full')
transformer_1.fit(train_data_1)

PCs = transformer_1.transform(train_data_1)
PCs_test = transformer_1.transform(test_data_1)

train_data_est = transformer_1.inverse_transform(PCs)
test_data_est = transformer_1.inverse_transform(PCs_test)

res_1 = train_data_1 - train_data_est
res_test_1 = test_data_1 - test_data_est

Q_1 = np.array(res_1.dot(res_1.transpose())).diagonal()
Q_test_1 = np.array(res_test_1.dot(res_test_1.transpose())).diagonal()

fdr_1 = transformer_1.transform(res_1)
fdr_test_1 = transformer_1.transform(res_test_1)

#
#
#fig = plt.figure()
#plot_data = res_1
#plot_data_2 = res_test_1
#
#ax = fig.add_subplot(111, projection='3d') 
##ax.scatter(plot_data[:,0], plot_data[:,1], plot_data[:,2])
#ax.scatter(plot_data[:,0], plot_data[:,1], plot_data[:,2], color="b")
#ax.scatter(plot_data_2[:1000,0], plot_data_2[:1000,1], plot_data_2[:1000,2], color="b")
#ax.scatter(plot_data_2[1000:2000,0], plot_data_2[1000:2000,1], plot_data_2[1000:2000,2], color="r")


transformer_2 = PCA(n_components=1, svd_solver = 'full')
transformer_2.fit(train_data_2)

PCs = transformer_2.transform(train_data_2)
PCs_test = transformer_2.transform(test_data_2)

train_data_est = transformer_2.inverse_transform(PCs)
test_data_est = transformer_2.inverse_transform(PCs_test)

res_2 = train_data_2 - train_data_est
res_test_2 = test_data_2 - test_data_est

Q_2 = np.array(res_2.dot(res_2.transpose())).diagonal()
Q_test_2 = np.array(res_test_2.dot(res_test_2.transpose())).diagonal()





fdr_2 = transformer_2.transform(res_2)
fdr_test_2 = transformer_2.transform(res_test_2)


Q_1_per = np.percentile(Q_1, [25,50,75])
q1, q3 = Q_1_per[0], Q_1_per[2]
Q_1_low = q1 - 1.5*(q3 - q1)
Q_1_up = q3 + 1.5*(q3 - q1)



Q_2_per = np.percentile(Q_2, [25,50,75])
q1, q3 = Q_2_per[0], Q_2_per[2]
Q_2_low = q1 - 1.5*(q3 - q1)
Q_2_up = q3 + 1.5*(q3 - q1)



#Q_1_ma = movingaverage(np.append(Q_1, Q_test_1), 96)
#Q_2_ma = movingaverage(np.append(Q_2, Q_test_2), 96)
#-----------------anomaly detection------------------------
fig = plt.figure()
plot_data_1 = movingaverage(np.append(Q_1, Q_test_1), 96)
plot_data_2 = movingaverage(np.append(Q_2, Q_test_2), 96)
plot_X = list(range(1, len(plot_data_1)+1))

#-----------------subplot1---------------------
ax = fig.add_subplot(211) 
l1, = ax.plot(plot_X, plot_data_1, color='black', linewidth=1.0)
cl1 = ax.hlines(Q_sigma_1, 1, len(plot_X)+1, linestyles = "dashed", color='red', linewidth=1.0)
ax.vlines(2000, -1, 1, linestyles = "dotted", linewidth=1.0)

ax.set_xlim(0,4512)
ax.set_ylim(0, 0.175)

#ax.legend(handles = [l1, cl1], labels = ['Q statistic', 'control limit'], loc = 'best')

#ax.set_xlabel("Sample Number")
ax.set_ylabel("Q statistic of 1st group")

#-----------------subplot2---------------------
ax = fig.add_subplot(212) 
l2, = ax.plot(plot_X, plot_data_2, color='black', linewidth=1.0)
cl2 = ax.hlines(Q_sigma_2, 1, len(plot_X)+1, linestyles = "dashed", color='red', linewidth=1.0)
ax.vlines(2000, -1, 1, linestyles = "dotted", linewidth=1.0)

ax.set_xlim(0,4512)
ax.set_ylim(0, 0.175)

#ax.legend(handles = [l2, cl2], labels = ['Q statistic', 'control limit'], loc = 'best')

ax.set_xlabel("Sample Number")
ax.set_ylabel("Q statistic of 2nd group")


#-----------------fault location------------------------
fig = plt.figure()
plot_data_1 = movingaverage(np.append(Q_1, Q_test_1), 96)
plot_data_2 = movingaverage(np.append(Q_2, Q_test_2), 96)
plot_X = list(range(1, len(plot_data_1)+1))

ax = fig.add_subplot(111) 
l1, = ax.plot(plot_X, plot_data_1, linewidth=1.0)
l2, = ax.plot(plot_X, plot_data_2, linewidth=1.0)

ax.set_xlim(3000,4512)
ax.set_ylim(0, 0.175)

ax.legend(handles = [l1, l2], labels = ['1st group', '2nd group'], loc = 'best')

ax.set_xlabel("Sample Number")
ax.set_ylabel("Q statistic")






#---------------计算统计量及阈值----------------------
Q_diff = Q_1 - Q_2
Q_test_diff = Q_test_1 - Q_test_2

Q_diff_median = np.median(Q_diff)
Q_test_diff_median = np.median(Q_test_diff[1000:])

Q_diff_per = np.percentile(Q_diff, [25,50,75])
q1, q3 = Q_diff_per[0], Q_diff_per[2]
Q_diff_low = q1 - 3*(q3 - q1) 
Q_diff_up = q3 + 3*(q3 - q1) 


#---------------计算贡献值-------------------------

A1_value = np.mean(res_test_1[:, 0]**2)
B1_value = np.mean(res_test_1[:, 1]**2)
C1_value = np.mean(res_test_1[:, 2]**2)

A2_value = np.mean(res_test_2[:, 0]**2)
B2_value = np.mean(res_test_2[:, 1]**2)
C2_value = np.mean(res_test_2[:, 2]**2)




fig = plt.figure()
plot_data = np.append(Q_diff, Q_test_diff)
plot_X = list(range(1, len(plot_data)+1))
ax = fig.add_subplot(111) 
ax.plot(plot_X, plot_data)
ax.hlines(Q_diff_low, 1, len(plot_data)+1, colors = "c", linestyles = "dashed")
ax.hlines(Q_diff_up, 1, len(plot_data)+1, colors = "c", linestyles = "dashed")
ax.hlines(Q_diff_median, 1, len(plot_data)+1, colors = "b", linestyles = "dashed")
ax.hlines(Q_test_diff_median, 1, len(plot_data)+1, colors = "r", linestyles = "dashed")


fig = plt.figure()
X = ['A1','B1','C1', 'A2','B2','C2']
Y = [A1_value, B1_value, C1_value, A2_value, B2_value, C2_value]  
plt.bar(X,Y,0.4)

fig = plt.figure()
plot_data = Q_test_diff
ax = fig.add_subplot(111) 
ax.hist(plot_data,100)








Q_mean = np.mean(Q)
Q_mean_test = np.mean(Q_test)
Q_mean_diff_05 = Q_mean_test - Q_mean

P = transformer.components_.transpose()
#pca.n_components_
#pca.explained_variance_ratio_
#pca.singular_values_

#train_data.iloc[:] = train_data.iloc[:] - np.mean(train_data.values, axis=0)

#U, D, Vt = np.linalg.svd(train_data)
#V = Vt.transpose()
train_data = train_data_2
V, D, Vt = np.linalg.svd(np.transpose(train_data).dot(train_data)/(len(train_data)-1))
Pe = V[:, 1:]

theta1 = np.sum(D[1:]**1)
theta2 = np.sum(D[1:]**2)
theta3 = np.sum(D[1:]**3)
h0 = 1 - 2*theta1*theta3/(3*theta2**2)
C_alpha = 2.33
Q_sigma_2 = theta1 * (C_alpha * h0 * np.sqrt(2*theta2) / theta1 + 1 + theta2 * h0 * (h0-1) / theta1**2) ** (1/h0)



X = dataset_1.iloc[3500:,:].values
e = dataset_1_error.iloc[3500:,:].values - X

mat1 = X.dot(Pe).dot(Pe.transpose()).dot(e.transpose()).diagonal()
mat2 = e.dot(Pe).dot(Pe.transpose()).dot(X.transpose())

Q_1 = np.array(res_1.dot(res_1.transpose())).diagonal()
Q_1_svd = train_data_1.dot(Pe).dot(Pe.transpose()).dot(train_data_1.transpose())
Q_1_norm=(np.linalg.norm(train_data_1[0,:].dot(Pe), ord=2, keepdims=True))**2



fig = plt.figure()
plot_data = np.append(Q_2,Q_test_2)
plot_X = list(range(1, len(plot_data)+1))
ax = fig.add_subplot(111) 
ax.scatter(plot_X, plot_data, s=1)
ax.hlines(Q_sigma, 1, len(plot_data)+1, colors = "c", linestyles = "dashed")


fig = plt.figure()
plot_data = np.vstack((res, res_test))
plot_X = list(range(1, len(plot_data)+1))
ax = fig.add_subplot(311) 
ax.plot(plot_X, plot_data[:,0])
ax = fig.add_subplot(312) 
ax.plot(plot_X, plot_data[:,1])
ax = fig.add_subplot(313) 
ax.plot(plot_X, plot_data[:,2])


fig = plt.figure()
plot_data = res_test**2
plot_data_1 = Q_test
ax = fig.add_subplot(411) 
ax.hist(plot_data[:,0],100)
ax = fig.add_subplot(412) 
ax.hist(plot_data[:,1],100)
ax = fig.add_subplot(413) 
ax.hist(plot_data[:,2],100)
ax = fig.add_subplot(414) 
ax.hist(Q,100)




