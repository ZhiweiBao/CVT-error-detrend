import numpy as np
import pandas as pd
from scipy import stats
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.rcParams['font.sans-serif'] = ['SimHei']
#matplotlib.rcParams['axes.unicode_minus']=False
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose



source_data_1 = pd.read_excel("./data/Table3_data_15min.xlsx", index_col=[0])
source_data_2 = pd.read_excel("./data/Table4_data_15min.xlsx", index_col=[0])
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

dataset_1 = dataset.iloc[:,:3]
dataset_2 = dataset.iloc[:,3:]


#------------------数据划分------------------------
#train_data_1 =  dataset_1.iloc
#train_data_2 =  dataset_2.iloc
#

#------------------误差判定------------------------


re_error = (dataset_1.values - dataset_2.values)/(dataset_1.values + dataset_2.values)*2*100

#re_error_pd = pd.DataFrame(re_error, index=timeseries, columns=['A','B','C'])
#re_error_pd.to_csv('re_error.csv')
#
#re_error_train = re_error[:2000]
#re_error_test = re_error[2000:]
#
#ex_error_low = []
#ex_error_up = []
#error_A_per = np.percentile(re_error_train[:, 0], [25,50,75])
#q1, q3 = error_A_per[0], error_A_per[2]
#ex_error_low.append(q1 - 1.5*(q3 - q1))
#ex_error_up.append(q3 + 1.5*(q3 - q1)) 
#
#error_B_per = np.percentile(re_error_train[:, 1], [25,50,75])
#q1, q3 = error_B_per[0], error_B_per[2]
#ex_error_low.append(q1 - 1.5*(q3 - q1))
#ex_error_up.append(q3 + 1.5*(q3 - q1)) 
#
#error_C_per = np.percentile(re_error_train[:, 2], [25,50,75])
#q1, q3 = error_C_per[0], error_C_per[2]
#ex_error_low.append(q1 - 1.5*(q3 - q1))
#ex_error_up.append(q3 + 1.5*(q3 - q1)) 
#
#a1 = np.array(ex_error_up) - np.array(ex_error_low)



re_error_train = re_error[:2000, :]
re_error_test = re_error[2000:, :]

#re_error_train_yj_A, lambda_A = stats.yeojohnson(re_error_train[:,0])
#re_error_train_yj_B, lambda_B = stats.yeojohnson(re_error_train[:,1])
#re_error_train_yj_C, lambda_C = stats.yeojohnson(re_error_train[:,2])

decomposition = seasonal_decompose(re_error_train, freq=96)  #timeseries时间序列数据
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
seasadj = re_error_train - seasonal

error_A_per = np.percentile(seasadj[:, 0], [25,50,75])
q1, q3 = error_A_per[0], error_A_per[2]
ex_error_low = q1 - 3*(q3 - q1)
ex_error_up = q3 + 3*(q3 - q1)

cl_A_low = np.array([])
cl_A_up = np.array([])
for i in range(47):
    cl_A_low = np.append(cl_A_low, ex_error_low + seasonal[:96, 0])
    cl_A_up = np.append(cl_A_up, ex_error_up + seasonal[:96, 0])

error_B_per = np.percentile(seasadj[:, 1], [25,50,75])
q1, q3 = error_B_per[0], error_B_per[2]
ex_error_low = q1 - 3*(q3 - q1)
ex_error_up = q3 + 3*(q3 - q1)

cl_B_low = np.array([])
cl_B_up = np.array([])
for i in range(47):
    cl_B_low = np.append(cl_B_low, ex_error_low + seasonal[:96, 1])
    cl_B_up = np.append(cl_B_up, ex_error_up + seasonal[:96, 1])

error_C_per = np.percentile(seasadj[:, 2], [25,50,75])
q1, q3 = error_C_per[0], error_C_per[2]
ex_error_low = q1 - 3*(q3 - q1)
ex_error_up = q3 + 3*(q3 - q1)

cl_C_low = np.array([])
cl_C_up = np.array([])
for i in range(47):
    cl_C_low = np.append(cl_C_low, ex_error_low + seasonal[:96, 2])
    cl_C_up = np.append(cl_C_up, ex_error_up + seasonal[:96, 2])


#---------------------------------------------------------
fig = plt.figure()
plot_data = re_error_train
plot_X = list(range(1,len(plot_data)+1))
ax = fig.add_subplot(511)
ax.plot(plot_X, plot_data[:,2], color="black", linewidth=1.0)
ax.plot(plot_X, seasadj[:,2], linewidth=1.0)
ax.set_xlim(0, len(plot_X))
#ax.set_ylim(-0.10, 0.10)
ax.set_ylabel("εA/%")


plot_data = trend
ax = fig.add_subplot(512)
ax.plot(plot_X, plot_data[:,2], color="black", linewidth=1.0)
ax.set_xlim(0, len(plot_X))
#ax.set_ylim(-0.25, -0.05)
ax.set_ylabel("εB/%")

plot_data = seasonal
ax = fig.add_subplot(513)
ax.plot(plot_X, plot_data[:,2], color="black", linewidth=1.0)
ax.set_xlim(0, len(plot_X))
#ax.set_ylim(-0.25, -0.05)
ax.set_ylabel("εB/%")


plot_data = residual
ax = fig.add_subplot(514)
ax.plot(plot_X, plot_data[:,2], color="black", linewidth=1.0)
ax.set_xlim(0, len(plot_X))
#ax.set_ylim(-0.25, -0.05)
ax.set_ylabel("εB/%")

plot_data = smooth(re_error_train[:,2], 97)
ax = fig.add_subplot(515)
ax.plot(plot_X, plot_data, color="black", linewidth=1.0)
ax.set_xlim(0, len(plot_X))
#ax.set_ylim(-0.25, -0.05)
ax.set_ylabel("εB/%")



#---------------------------------------------------------
fig = plt.figure()
plot_data = re_error_train
plot_X = list(range(1,len(plot_data)+1))
ax = fig.add_subplot(311)
ax.plot(plot_X, plot_data[:,0], color="black", linewidth=1.0)
#ax.set_xlim(0, len(plot_X))
#ax.set_ylim(-0.10, 0.10)
#ax.set_ylabel("εA/%")
ax = fig.add_subplot(312)
ax.plot(plot_X, plot_data[:,1], color="black", linewidth=1.0)
#ax.set_xlim(0, len(plot_X))
#ax.set_ylim(-0.25, -0.05)
#ax.set_ylabel("εB/%")
ax = fig.add_subplot(313)
ax.plot(plot_X, plot_data[:,2], color="black", linewidth=1.0)
#ax.set_xlim(0, len(plot_X))
#ax.set_ylim(-0.25, -0.05)
#ax.set_ylabel("εB/%")


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



fig = plt.figure()
plot_data = np.concatenate((re_error_train, re_error_test), axis=0)
plot_X = list(range(1,len(plot_data)+1))

ax = fig.add_subplot(311)
l1, = ax.plot(plot_X, smooth(plot_data[:,0], 7), color="black", linewidth=2.0)
l2, = ax.plot(plot_X, cl_A_low, linewidth=1.5, color="gray", linestyle = "dotted")
ax.plot(plot_X, cl_A_up, linewidth=1.5, color="gray", linestyle = "dotted")
#ax.hlines(ex_error_low, 1, len(plot_data)+1, linestyles = "dotted", linewidth=1.0)
#ax.hlines(ex_error_up, 1, len(plot_data)+1, linestyles = "dotted", linewidth=1.0)
ax.vlines(2000, -1, 1, linestyles = "dotted", linewidth=1.0)
ax.set_xlim(0, len(plot_X))
ax.set_ylim(-0.05, 0.15)
ax.set_ylabel("$ε_A/\%$")
ax.legend(handles = [l1, l2], labels = ['$ε_A$', '$CL_A$'], loc = 'upper left', prop={'size':8})

ax = fig.add_subplot(312)
l1, = ax.plot(plot_X, smooth(plot_data[:,1],7), color="black", linewidth=2.0)
l2, = ax.plot(plot_X, cl_B_low, linewidth=1.5, color="gray", linestyle = "dotted")
ax.plot(plot_X, cl_B_up, linewidth=1.5, color="gray", linestyle = "dotted")
#ax.hlines(ex_error_low, 1, len(plot_data)+1, linestyles = "dotted", linewidth=1.0)
#ax.hlines(ex_error_up, 1, len(plot_data)+1, linestyles = "dotted", linewidth=1.0)
ax.vlines(2000, -1, 1, linestyles = "dotted", linewidth=1.0)
ax.set_xlim(0, len(plot_X))
ax.set_ylim(-0.20, -0.00)
ax.set_ylabel("$ε_B/\%$")
ax.legend(handles = [l1, l2], labels = ['$ε_B$', '$CL_B$'], loc = 'upper left', prop={'size':8})

ax = fig.add_subplot(313)
l1, = ax.plot(plot_X, smooth(plot_data[:,2], 7), color="black", linewidth=2.0)
l2, = ax.plot(plot_X, cl_C_low, linewidth=1.5, color="gray", linestyle = "dotted")
ax.plot(plot_X, cl_C_up, linewidth=1.5, color="gray", linestyle = "dotted")
#ax.hlines(ex_error_low, 1, len(plot_data)+1, linestyles = "dotted", linewidth=1.0)
#ax.hlines(ex_error_up, 1, len(plot_data)+1, linestyles = "dotted", linewidth=1.0)
ax.vlines(2000, -1, 1, linestyles = "dotted", linewidth=1.0)
ax.set_xlim(0, len(plot_X))
ax.set_ylim(0.10, 0.30)
ax.set_ylabel("$ε_C/\%$")
ax.set_xlabel("Sample Number")
ax.legend(handles = [l1, l2], labels = ['$ε_C$', '$CL_C$'], loc = 'upper left', prop={'size':8})




fig = plt.figure()
plot_data = np.concatenate((re_error_train, re_error_test), axis=0)[2000:,:]
plot_X = list(range(2000,len(plot_data)+2000))

ax = fig.add_subplot(311)
l1, = ax.plot(plot_X, smooth(plot_data[:,0], 7), color="black", linewidth=2.0)
l2, = ax.plot(plot_X, cl_A_low[2000:], linewidth=1.5, color="gray", linestyle = "dotted")
ax.plot(plot_X, cl_A_up[2000:], linewidth=1.5, color="gray", linestyle = "dotted")
#ax.hlines(ex_error_low, 1, len(plot_data)+1, linestyles = "dotted", linewidth=1.0)
#ax.hlines(ex_error_up, 1, len(plot_data)+1, linestyles = "dotted", linewidth=1.0)
#ax.vlines(2000, -1, 1, linestyles = "dotted", linewidth=1.0)
ax.set_xlim(plot_X[0], plot_X[-1])
ax.set_ylim(-0.05, 0.15)
ax.set_ylabel("$ε_A^*/\%$")
ax.legend(handles = [l1, l2], labels = ['$ε_A^*$', '$CL_A$'], loc = 'upper left', prop={'size':8})

ax = fig.add_subplot(312)
l1, = ax.plot(plot_X, smooth(plot_data[:,1],7), color="black", linewidth=2.0)
l2, = ax.plot(plot_X, cl_B_low[2000:], linewidth=1.5, color="gray", linestyle = "dotted")
ax.plot(plot_X, cl_B_up[2000:], linewidth=1.5, color="gray", linestyle = "dotted")
#ax.hlines(ex_error_low, 1, len(plot_data)+1, linestyles = "dotted", linewidth=1.0)
#ax.hlines(ex_error_up, 1, len(plot_data)+1, linestyles = "dotted", linewidth=1.0)
#ax.vlines(2000, -1, 1, linestyles = "dotted", linewidth=1.0)
ax.set_xlim(plot_X[0], plot_X[-1])
ax.set_ylim(-0.20, 0)
ax.set_ylabel("$ε_B^*/\%$")
ax.legend(handles = [l1, l2], labels = ['$ε_B^*$', '$CL_B$'], loc = 'upper left', prop={'size':8})

ax = fig.add_subplot(313)
l1, = ax.plot(plot_X, smooth(plot_data[:,2], 7), color="black", linewidth=2.0)
l2, = ax.plot(plot_X, cl_C_low[2000:], linewidth=1.5, color="gray", linestyle = "dotted")
ax.plot(plot_X, cl_C_up[2000:], linewidth=1.5, color="gray", linestyle = "dotted")
#ax.hlines(ex_error_low, 1, len(plot_data)+1, linestyles = "dotted", linewidth=1.0)
#ax.hlines(ex_error_up, 1, len(plot_data)+1, linestyles = "dotted", linewidth=1.0)
#ax.vlines(2000, -1, 1, linestyles = "dotted", linewidth=1.0)
ax.set_xlim(plot_X[0], plot_X[-1])
ax.set_ylim(0.10, 0.30)
ax.set_ylabel("$ε_C^*/\%$")
ax.set_xlabel("Sample Number")
ax.legend(handles = [l1, l2], labels = ['$ε_C^*$', '$CL_C$'], loc = 'upper left', prop={'size':8})





