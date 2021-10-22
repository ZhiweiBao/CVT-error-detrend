import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()
from datetime import datetime, timedelta
import scipy
from scipy.stats import kstest, ks_2samp, normaltest

from pyhht.emd import EMD
from pyhht.visualization import plot_imfs


from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import QuantileTransformer

from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis
from sklearn.neighbors import KernelDensity
from sklearn.covariance import EllipticEnvelope


source_data_1 = pd.read_excel("E:/Workspaces/Python3/CVTAnomalyDetection/data/Table3_data_15min.xlsx")
source_data_2 = pd.read_excel("E:/Workspaces/Python3/CVTAnomalyDetection/data/Table4_data_15min.xlsx")
#source_data = pd.read_excel("E:/Workspaces/Python3/CVTAnomalyDetection/data/zhenwu_15min_mean.xlsx")

source_data = pd.concat([source_data_1, source_data_2], axis=1)
#dataset = pd.DataFrame(dataset.values[:,1:4], index=dataset.iloc[:,0], columns=['A', 'B', 'C'])
#timeseries = pd.date_range(start = '2019-03-20 00:00:00', end = '2019-06-22 23:59:59', freq='15min')
timeseries = pd.date_range(start = '2017-11-29 00:00:00', end = '2018-01-27 23:59:59', freq='15min')

#--------------数据清洗------------------------
dataset = source_data.loc[timeseries]
dataset_1 = dataset.iloc[:,:3]
dataset_2 = dataset.iloc[:,3:]
re_error = (dataset_1.values - dataset_2.values)/(dataset_1.values + dataset_2.values)*2*100
dataset.iloc[re_error[:,0] < -.012] = np.nan
dataset.iloc[re_error[:,0] > .025] = np.nan
dataset.iloc[re_error[:,1] < -0.156] = np.nan
dataset.iloc[re_error[:,1] > -0.139] = np.nan
dataset.iloc[[3978,4633,5035,5038]] = np.nan
dataset=dataset.interpolate(method='linear')

#-------------加误差-----------------------------------
dataset_error = dataset.loc[timeseries]

step_error_point = 4000
step_error_array = [1 for _ in range(step_error_point)]
step_error_array.extend([(1-1e-3) for _ in range(len(dataset_error)-step_error_point)])

gradual_error_point = 4000
gradual_error_array = [1 for _ in range(gradual_error_point)]
gradual_error_array.extend([(1-1e-6*i) for i in range(len(dataset_error)-gradual_error_point)])

phase_fault = 2
dataset_error.iloc[:,phase_fault] = dataset_error.iloc[:,phase_fault]*step_error_array

dataset_1 = dataset.iloc[:,:3]
dataset_2 = dataset.iloc[:,3:]
dataset_1_error = dataset_error.iloc[:,:3]
dataset_2_error = dataset_error.iloc[:,3:]

#------------------数据划分------------------------
train_data_1 =  dataset_1_error.iloc[:3000,:]
train_data_2 =  dataset_2_error.iloc[:3000,:]
test_data_1 =  dataset_1_error.iloc[3000:,:]
test_data_2 =  dataset_2_error.iloc[3000:,:]


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

#----------------计算统计量----------------------





#fig = plt.figure()
#
#plot_data = dataset_1
#plot_X = list(range(1,len(plot_data)+1))
#ax = fig.add_subplot(111)
#ax.plot(plot_X, plot_data.iloc[:,0])
#
#ax.plot(plot_X, plot_data.iloc[:,1]) 
#
#ax.plot(plot_X, plot_data.iloc[:,2]) 
#
#ax.set_xlabel("Point")
#
#
fig = plt.figure()

plot_data = dataset_1_error
plot_X = list(range(1,len(plot_data)+1))
ax = fig.add_subplot(311)
ax.plot(plot_X, plot_data.iloc[:,0])
ax.set_ylabel("Phase A/kV")
ax = fig.add_subplot(312)
ax.plot(plot_X, plot_data.iloc[:,1]) 
ax.set_ylabel("Phase B/kV")
ax = fig.add_subplot(313)
ax.plot(plot_X, plot_data.iloc[:,2]) 
ax.set_ylabel("Phase C/kV")
ax.set_xlabel("Point")



fig = plt.figure()

plot_data = dataset_1_error
plot_X = list(range(1,len(plot_data)+1))
ax = fig.add_subplot(311)
ax.hist(plot_data.iloc[:,0], 100)
ax.set_ylabel("Phase A/kV")
ax = fig.add_subplot(312)
ax.hist(plot_data.iloc[:,1], 100)
ax.set_ylabel("Phase B/kV")
ax = fig.add_subplot(313)
ax.hist(plot_data.iloc[:,2], 100)
ax.set_ylabel("Phase C/kV")
ax.set_xlabel("Point")



len_fa_step = 96
len_fa_window = 672
len_lag = timedelta(weeks=0)

transformer = FactorAnalysis(n_components=1, random_state=0)
transform_data = dataset_1_error
transf_ini_data = transform_data.iloc[:len_fa_window, :]
#transf_ini_data = transform_data
voltage_fluc = pd.DataFrame(transformer.fit_transform(transf_ini_data), index=transf_ini_data.index, columns=['factor1'])
conpon = transformer.components_
fea_mean = transformer.mean_
res_A = transf_ini_data.values[:,0] - np.mean(fea_mean) - voltage_fluc.values[:,0] * conpon[0][0] 
res_B = transf_ini_data.values[:,1] - np.mean(fea_mean) - voltage_fluc.values[:,0] * conpon[0][1]
res_C = transf_ini_data.values[:,2] - np.mean(fea_mean) - voltage_fluc.values[:,0] * conpon[0][2]
residual =  pd.DataFrame(np.transpose(np.array([res_A, res_B, res_C])), index=transf_ini_data.index, columns=['res_A', 'res_B', 'res_C'])


res_diff_AB = res_A - res_B
res_diff_BC = res_B - res_C
res_diff_CA = res_C - res_A

res_diff = pd.DataFrame(np.transpose(np.array([res_diff_AB, res_diff_BC, res_diff_CA])), index=transf_ini_data.index, columns=['res_AB', 'res_BC', 'res_CA'])

res = residual



#
#fig = plt.figure()
#plot_data = residual
#
#
#ax = fig.add_subplot(311)
#ax.plot(plot_data.index, plot_data.iloc[:,0])
#    
#ax = fig.add_subplot(312)
#ax.plot(plot_data.index, plot_data.iloc[:,1])
#    
#ax = fig.add_subplot(313)
#ax.plot(plot_data.index, plot_data.iloc[:,2])
#
#
#fig = plt.figure()
#plot_data = residual.values
#ax = fig.add_subplot(111, projection='3d') 
#ax.scatter(plot_data[:,0], plot_data[:,1], plot_data[:,2])
#
#fig = plt.figure()
#plot_data = voltage_fluc
#ax = fig.add_subplot(111)
#ax.plot(plot_data.index, plot_data.iloc[:,0])




train_data = res.values

kmeans_train = KMeans(n_clusters=1, random_state=0).fit(train_data)
center_train = kmeans_train.cluster_centers_[0]
    
train_diff =  train_data-center_train
distance = np.array(np.sqrt(np.sum(train_diff**2, axis = 1)))
distances = pd.DataFrame(distance, index=transf_ini_data.index)
    
qt = QuantileTransformer(output_distribution='normal')
qt.fit(distances)
distance_qt = qt.transform(distances)
    
mu = np.mean(distance_qt)
sigma = np.std(distance_qt)
limit_qt = mu + 3*sigma
    
limit = qt.inverse_transform(np.array([limit_qt]).reshape(1,-1))[0][0]

limits_date = pd.date_range(start = transf_ini_data.index[0], end = transf_ini_data.index[-1]+timedelta(days=1)+len_lag, freq='15min')    
limits = pd.DataFrame(np.array([limit for _ in range(len(limits_date))]), index=limits_date)


#res_mean = np.mean(residual)
#res_std = np.std(residual)
#res_up = pd.DataFrame(np.array([res_mean + 3*res_std for i in range(2)]), index = [residual.index[0], residual.index[-1] + timedelta(days=1)], columns=['res_A_up', 'res_B_up', 'res_C_up'])
#res_low = pd.DataFrame(np.array([res_mean - 3*res_std for i in range(2)]), index = [residual.index[0], residual.index[-1] + timedelta(days=1)], columns=['res_A_low', 'res_B_low', 'res_C_low'])

fig = plt.figure()
plot_data = fdr.values
plot_X = list(range(1,len(plot_data)+1))
ax = fig.add_subplot(311)
ax.plot(plot_X, plot_data[:,0], color="black")
ax.set_ylim(-0.02, 0.02)
ax.set_ylabel("Phase A/kV")
#ax = fig.add_subplot(322)
#ax.hist(plot_data.iloc[:,0], bins=50)
#
ax = fig.add_subplot(312)
ax.plot(plot_X, plot_data[:,1], color="black")
ax.set_ylim(-0.02, 0.02)
ax.set_ylabel("Phase B/kV")
#ax = fig.add_subplot(324)
#ax.hist(plot_data.iloc[:,1], bins=50)
#
ax = fig.add_subplot(313)
ax.plot(plot_X, plot_data[:,2], color="black")
ax.set_ylim(-0.02, 0.02)
ax.set_ylabel("Phase C/kV")
ax.set_xlabel("Point")
#ax = fig.add_subplot(326)
#ax.hist(plot_data.iloc[:,2], bins=50)



error_index = []
error_rates = []

for i in range(len_fa_step, len(transform_data)-len_fa_window+len_fa_step, len_fa_step):
    if len(error_index) == 0:
        transf_roll_data = transform_data.iloc[:i+len_fa_window, :]
    else:
        transf_roll_data = transform_data.iloc[:i+len_fa_window, :]
        transf_roll_data = transf_roll_data.drop(error_index)
    time_index = transf_roll_data.index
    voltage_fluc = pd.DataFrame(transformer.fit_transform(transf_roll_data), index=time_index, columns=['voltage_fluc'])
    conpon = transformer.components_[0]
    fea_mean = transformer.mean_
    res_A = transf_roll_data.values[:,0] - np.mean(fea_mean) - voltage_fluc.values[:,0] * conpon[0]
    res_B = transf_roll_data.values[:,1] - np.mean(fea_mean) - voltage_fluc.values[:,0] * conpon[1]
    res_C = transf_roll_data.values[:,2] - np.mean(fea_mean) - voltage_fluc.values[:,0] * conpon[2]
    residual =  pd.DataFrame(np.transpose(np.array([res_A, res_B, res_C])), index=time_index, columns=['res_A', 'res_B', 'res_C'])
    
#    res_diff_AB = res_A - res_B
#    res_diff_BC = res_B - res_C
#    res_diff_CA = res_C - res_A
#    res_diff = pd.DataFrame(np.transpose(np.array([res_diff_AB, res_diff_BC, res_diff_CA])), index=time_index, columns=['res_AB', 'res_BC', 'res_CA'])   
    
    res = res.append(residual.iloc[-len_fa_step:,:])
   
    if len(error_index) == 0:
        train_data = res.iloc[:-len_fa_step,:]
        test_data = res.iloc[-len_fa_step:,:]
    else:
        train_data = res.drop(error_index).iloc[:-len_fa_step,:]
        test_data = res.drop(error_index).iloc[-len_fa_step:,:]
    
    kmeans_train = KMeans(n_clusters=1, random_state=0).fit(train_data)
    center_train = kmeans_train.cluster_centers_[0]
    
    test_diff =  test_data-center_train
    distance = np.array(np.sqrt(np.sum(test_diff**2, axis = 1)))
    distances = distances.append(pd.DataFrame(distance, index=transf_roll_data.index[-len_fa_step:]))
    
    error_rate = len(np.where(distance > limits.loc[time_index[-len_fa_step]].values[0])[0])/len(distance)*100
    error_rates.append(error_rate)
    if error_rate > 5:
        if len(error_index) == 0:
            error_index = transf_roll_data.index[-len_fa_step:]
        else:
            error_index = error_index.append(transf_roll_data.index[-len_fa_step:])
    
    
    if len(error_index) == 0:
        limit_train = distances
    else:
        limit_train = distances.drop(error_index)
    
    
    qt.fit(limit_train)
    distance_qt = qt.transform(limit_train)
    
    mu = np.mean(distance_qt)
    sigma = np.std(distance_qt)
    limit_qt = mu + 3*sigma
    
    limit = qt.inverse_transform(np.array([limit_qt]).reshape(1,-1))[0][0]
    limits_date = pd.date_range(start = transf_roll_data.index[-1]+timedelta(minutes=15)+len_lag, end = transf_roll_data.index[-1]+timedelta(days=1)+len_lag, freq='15min')    

    limits = limits.append(pd.DataFrame(np.array([limit for _ in range(len_fa_step)]), index=limits_date))

    

    
fig = plt.figure()
plot_data = distances
plot_data_2 = limits
ax = fig.add_subplot(111)
ax.scatter(list(range(len(plot_data.index))), plot_data, s=1)
ax.plot(list(range(len(plot_data.index))), plot_data_2.iloc[:len(plot_data),:], linestyle='dashed')
ax.set_xlabel("Point")
ax.set_ylabel("Distance")
#ax.set_xlim(2000,3000)

fig = plt.figure()
plot_data = distance_qt
ax = fig.add_subplot(111)
ax.hist(plot_data,100)
ax.set_xlabel("Distance")
ax.set_ylabel("Proportion")
    

i = len_fa_step * 0
train_data = res.values[:i+len_fa_window]
test_data = res.values[i+len_fa_window:i+96+len_fa_window]

cov = EllipticEnvelope(contamination=0.003,random_state=0).fit(train_data)
shape = cov.support_

test_result = cov.predict(test_data)

fig = plt.figure()
plot_data = train_data
plot_data_2 = test_data

ax = fig.add_subplot(111, projection='3d') 
#ax.scatter(plot_data[:,0], plot_data[:,1], plot_data[:,2])
ax.scatter(plot_data[np.where(shape==True),0], plot_data[np.where(shape==True),1], plot_data[np.where(shape==True),2])
#ax.scatter(plot_data[np.where(shape==False),0], plot_data[np.where(shape==False),1], plot_data[np.where(shape==False),2], color='r')
ax.scatter(plot_data_2[np.where(test_result==1),0], plot_data_2[np.where(test_result==1),1], plot_data_2[np.where(test_result==1),2], color='green')
ax.scatter(plot_data_2[np.where(test_result==-1),0], plot_data_2[np.where(test_result==-1),1], plot_data_2[np.where(test_result==-1),2], color='orange')
ax.set_title("Test Error Rate = %.2f%%" % (len(np.where(test_result==-1)[0])/len(test_result)*100))







fig = plt.figure()
plot_data = res
plot_data_up = res_up.iloc[:-2,:]
plot_data_low = res_low.iloc[:-2,:]
plot_data_mean = (plot_data_up.values + plot_data_low.values)/2

ax = fig.add_subplot(311)
ax.plot(plot_data.index, plot_data.iloc[:,0])
ax.plot(plot_data_up.index, plot_data_up.iloc[:,0])
ax.plot(plot_data_low.index, plot_data_low.iloc[:,0])
ax.plot(plot_data_low.index, plot_data_mean[:,0])
    
ax = fig.add_subplot(312)
ax.plot(plot_data.index, plot_data.iloc[:,1])
ax.plot(plot_data_up.index, plot_data_up.iloc[:,1])
ax.plot(plot_data_low.index, plot_data_low.iloc[:,1])
ax.plot(plot_data_low.index, plot_data_mean[:,1])
    
ax = fig.add_subplot(313)
ax.plot(plot_data.index, plot_data.iloc[:,2])
ax.plot(plot_data_up.index, plot_data_up.iloc[:,2])
ax.plot(plot_data_low.index, plot_data_low.iloc[:,2])
ax.plot(plot_data_low.index, plot_data_mean[:,2])



fig = plt.figure()
plot_data = res.values
error_point = step_error_point
end_point = error_point + 96*1


ax = fig.add_subplot(111, projection='3d') 
ax.scatter(plot_data[:error_point,0], plot_data[:error_point,1], plot_data[:error_point,2])
ax.scatter(plot_data[error_point:end_point,0], plot_data[error_point:end_point,1], plot_data[error_point:end_point,2], color='r')


fig = plt.figure()
plot_data = res.values
error_point = step_error_point
end_point = error_point + 96*3

ax = fig.add_subplot(111, projection='3d') 
ax.scatter(plot_data[np.where(shape==True),0], plot_data[np.where(shape==True),1], plot_data[np.where(shape==True),2])
ax.scatter(plot_data[np.where(shape==False),0], plot_data[np.where(shape==False),1], plot_data[np.where(shape==False),2], color='r')








fig = plt.figure()
plot_data = residual.values
ax = fig.add_subplot(111, projection='3d') 
ax.scatter(plot_data[:,0], plot_data[:,1], plot_data[:,2])

transformer = FactorAnalysis(n_components=1, random_state=0)

eigenvectors = pd.DataFrame()
for i in range(0, len(dataset)-len_fa_window, len_fa_step):
    train_point = i
    time_index = dataset_2.index[i+len_fa_window]
    transform_data = dataset_2.iloc[train_point:train_point+len_fa_window,:]
#    transform_data = dataset
    voltage_fluc = pd.DataFrame(transformer.fit_transform(transform_data), index=transform_data.index, columns=['voltage_fluc'])
    conpon = transformer.components_[0]
    fea_mean = transformer.mean_
    res_A = transform_data.values[:,0] - fea_mean[0] - voltage_fluc.values[:,0] * conpon[0]
    res_B = transform_data.values[:,1] - fea_mean[1] - voltage_fluc.values[:,0] * conpon[1]
    res_C = transform_data.values[:,2] - fea_mean[2] - voltage_fluc.values[:,0] * conpon[2]
    residual_2 =  pd.DataFrame(np.transpose(np.array([res_A, res_B, res_C])), index=transform_data.index, columns=['res_A', 'res_B', 'res_C'])
     






    