import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis

source_data_1 = pd.read_excel("E:/Workspaces/Python3/CVTAnomalyDetection/data/Table3_data_15min.xlsx", index_col=[0])
source_data_2 = pd.read_excel("E:/Workspaces/Python3/CVTAnomalyDetection/data/Table4_data_15min.xlsx", index_col=[0])
source_data = pd.concat([source_data_1, source_data_2], axis=1)

#source_data = pd.read_excel("E:/Workspaces/Python3/CVTAnomalyDetection/data/zhenwu_15min.xlsx")


#timeseries = pd.date_range(start = '2019-03-20 00:00:00', end = '2019-06-22 23:59:59', freq='15min')
#timeseries = pd.date_range(start = '2017-11-29 00:00:00', end = '2018-01-27 23:59:59', freq='15min')
#timeseries = pd.date_range(start = '2018-01-22 00:00:00', end = '2018-03-22 23:59:59', freq='15min')
timeseries = pd.date_range(start = '2018-02-01 00:00:00', end = '2018-03-19 23:59:59', freq='15min')


#dataset = source_data.loc[timeseries]
#dataset = pd.concat([dataset.iloc[:,:3], dataset.iloc[:,6:7], dataset.iloc[:,4:6]], axis=1)
#



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

#------------------误差判定------------------------
re_error = (dataset_1_error.values - dataset_2_error.values)/(dataset_1_error.values + dataset_2_error.values)*100

re_error_train = re_error[:2000]
re_error_test = re_error[2000:]

ex_error_low = []
ex_error_up = []
error_A_per = np.percentile(re_error_train[:, 0], [25,50,75])
q1, q3 = error_A_per[0], error_A_per[2]
ex_error_low.append(q1 - 1.5*(q3 - q1))
ex_error_up.append(q3 + 1.5*(q3 - q1)) 

error_B_per = np.percentile(re_error_train[:, 1], [25,50,75])
q1, q3 = error_B_per[0], error_B_per[2]
ex_error_low.append(q1 - 1.5*(q3 - q1))
ex_error_up.append(q3 + 1.5*(q3 - q1)) 

error_C_per = np.percentile(re_error_train[:, 2], [25,50,75])
q1, q3 = error_C_per[0], error_C_per[2]
ex_error_low.append(q1 - 1.5*(q3 - q1))
ex_error_up.append(q3 + 1.5*(q3 - q1)) 

a1 = np.array(ex_error_up) - np.array(ex_error_low)





re_error_train = re_error[:2000, :]
re_error_test = re_error[2000:, :]


error_A_low = q1 - 1.5*(q3 - q1) 
error_A_up = q3 + 1.5*(q3 - q1) 

 
error_B_low = q1 - 1.5*(q3 - q1) 
error_B_up = q3 + 1.5*(q3 - q1) 

 
error_C_low = q1 - 1.5*(q3 - q1) 
error_C_up = q3 + 1.5*(q3 - q1) 


re_error_mean = np.mean(re_error_train, axis=0)
re_error_std = np.std(re_error_train, axis=0)    
re_error_low = re_error_mean - 3*re_error_std
re_error_up = re_error_mean + 3*re_error_std




fig = plt.figure()
plot_data = re_error
plot_X = list(range(1,len(plot_data)+1))
plot_error_low = ex_error_low
plot_error_up = ex_error_up
ax = fig.add_subplot(311)
ax.plot(plot_X, smooth(plot_data[:,0], 97), color="black", linewidth=1.0)
ax.hlines(plot_error_low[0], 1, len(plot_data)+1, linestyles = "dashed", linewidth=1.0, color="red")
ax.hlines(plot_error_up[0], 1, len(plot_data)+1, linestyles = "dashed", linewidth=1.0, color="red")
ax.vlines(2000, -1, 1, linestyles = "dotted", linewidth=1.0)
ax.set_xlim(0, len(plot_X))
ax.set_ylim(-0.10, 0.10)
ax.set_ylabel("εA/%")

ax = fig.add_subplot(312)
ax.plot(plot_X, smooth(plot_data[:,1], 97), color="black", linewidth=1.0)
ax.hlines(plot_error_low[1], 1, len(plot_data)+1, colors = "c", linestyles = "dashed", linewidth=1.0, color="red")
ax.hlines(plot_error_up[1], 1, len(plot_data)+1, colors = "c", linestyles = "dashed", linewidth=1.0, color="red")
ax.vlines(2000, -1, 1, linestyles = "dotted", linewidth=1.0)
ax.set_xlim(0, len(plot_X))
ax.set_ylim(-0.125, -0.025)
ax.set_ylabel("εB/%")

ax = fig.add_subplot(313)
ax.plot(plot_X, smooth(plot_data[:,2], 97), color="black", linewidth=1.0)
ax.hlines(plot_error_low[2], 1, len(plot_data)+1, colors = "c", linestyles = "dashed", linewidth=1.0, color="red")
ax.hlines(plot_error_up[2], 1, len(plot_data)+1, colors = "c", linestyles = "dashed", linewidth=1.0, color="red")
ax.vlines(2000, -1, 1, linestyles = "dotted", linewidth=1.0)
ax.set_xlim(0, len(plot_X))
ax.set_ylim(0.05, 0.15)
ax.set_ylabel("εC/%")
ax.set_xlabel("Sample Number")



fig = plt.figure()
plot_data = re_error
plot_X = list(range(1,len(plot_data)+1))
ax = fig.add_subplot(311)
ax.hist(plot_data[:,0],100)
ax = fig.add_subplot(312)
ax.hist(plot_data[:,1],100)
ax = fig.add_subplot(313)
ax.hist(plot_data[:,2],100)

#pt = PowerTransformer()
#re_error = pt.fit_transform(re_error)


#re_err_mean = np.mean(re_error, axis=0)
#re_err_std =np.std(re_error, axis=0)
#
#re_error[re_error > re_err_mean + 3*re_err_std] = np.nan
#re_error[re_error < re_err_mean - 3*re_err_std] = np.nan
#
#imputer = KNNImputer(n_neighbors=2, weights="uniform")
#re_error = imputer.fit_transform(re_error)


data = dataset_1_error
U_mean = np.mean(data.values, axis=1, keepdims=True)
degree_unbalance_1 = np.max(np.abs(data.values - U_mean), axis=1, keepdims=True)/U_mean

data = dataset_2_error
U_mean = np.mean(data.values, axis=1, keepdims=True)
degree_unbalance_2 = np.max(np.abs(data.values - U_mean), axis=1, keepdims=True)/U_mean

dataset_1 = dataset_1 - np.mean(dataset_1, axis=0)
dataset_2 = dataset_2 - np.mean(dataset_2, axis=0)

dataset = dataset - np.mean(dataset, axis=0)
dataset.corr()

dataset_1.corr()
dataset_2.corr()

data_A = dataset_1



fig = plt.figure()
plot_data_1 = degree_unbalance_1
plot_data_2 = degree_unbalance_2
plot_X = list(range(1,len(plot_data_1)+1))
ax = fig.add_subplot(211)
ax.plot(plot_X, plot_data_1)
ax = fig.add_subplot(212)
ax.plot(plot_X, plot_data_2)




transformer = FactorAnalysis(n_components=1, random_state=0)
X_transformed = transformer.fit_transform(data)
conpon = transformer.components_
fea_mean = transformer.mean_
factor1_A1 = X_transformed[:,0] * conpon[0][0] 
factor1_B1 = X_transformed[:,0] * conpon[0][1] 
factor1_C1 = X_transformed[:,0] * conpon[0][2] 
factor1_A2 = X_transformed[:,0] * conpon[0][3] 
factor1_B2 = X_transformed[:,0] * conpon[0][4] 
factor1_C2 = X_transformed[:,0] * conpon[0][5] 

#factor2_A1 = X_transformed[:,1] * conpon[1][0] 
#factor2_B1 = X_transformed[:,1] * conpon[1][1] 
#factor2_C1 = X_transformed[:,1] * conpon[1][2] 
#factor2_A2 = X_transformed[:,1] * conpon[1][3] 
#factor2_B2 = X_transformed[:,1] * conpon[1][4] 
#factor2_C2 = X_transformed[:,1] * conpon[1][5] 

res_A1 = data.values[:,0] - factor1_A1 - np.mean(fea_mean)
res_B1 = data.values[:,1] - factor1_B1 - np.mean(fea_mean)
res_C1 = data.values[:,2] - factor1_C1 - np.mean(fea_mean)
res_A2 = data.values[:,3] - factor1_A2 - np.mean(fea_mean)
res_B2 = data.values[:,4] - factor1_B2 - np.mean(fea_mean)
res_C2 = data.values[:,5] - factor1_C2 - np.mean(fea_mean)

factor1 = np.array(np.transpose([factor1_A1, factor1_B1, factor1_C1, factor1_A2, factor1_B2, factor1_C2]))
#factor2 = np.array(np.transpose([factor2_A1, factor2_B1, factor2_C1, factor2_A2, factor2_B2, factor2_C2]))
res = np.array(np.transpose([res_A1, res_B1, res_C1, res_A2, res_B2, res_C2]))

fig = plt.figure()
plot_data = res
plot_X = list(range(1,len(plot_data)+1))
for i in range(6):
    ax = fig.add_subplot(6,1,i+1)
    ax.plot(plot_X, plot_data[:,i])

