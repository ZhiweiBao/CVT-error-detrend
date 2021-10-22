# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:06:45 2019

@author: dell
"""

import data_import

import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis


dataset = data_import.load_dataset(flag = False)

X = data_import.load_dataset(flag = False)[:, 0:3]
#X = StandardScaler().fit_transform(X)

amp_A = dataset[:, 0]
amp_B = dataset[:, 1]
amp_C = dataset[:, 2]

angle_A = dataset[:, 3] * (math.pi) / (60*180)
angle_B = dataset[:, 4] * (math.pi) / (60*180)
angle_C = dataset[:, 5] * (math.pi) / (60*180)

#u_A = np.complex(amp_A * np.cos(angle_A), amp_A * np.sin(angle_A))

u_A = amp_A * np.exp(angle_A * 1j)
u_B = amp_B * np.exp(angle_B * 1j)
u_C = amp_C * np.exp(angle_C * 1j)
i
u = abs(u_A + u_B + u_C)
angle = np.angle(u_A + u_B + u_C)

transformer = FactorAnalysis(n_components=1)
X_transformed = transformer.fit_transform(X)
score = transformer.score_samples(X)
cov = transformer.get_covariance()
precision = transformer.get_precision()
s = transformer.score(X)
components = transformer.components_
noise_variance = transformer.noise_variance_



fig = plt.figure()

for i in range(3):
    ax = fig.add_subplot(3, 1, i+1)
    
    x = list(range(1,3601))
    ax.plot(x, dataset[:, i], linewidth=1)
    
    
    if i == 0:
        ax.set_title(u'Three-phase voltage amplitude of CVT')
        ax.set_ylabel(u'A')
    elif i == 1:
        ax.set_ylabel(u'B')
    elif i == 2:
        ax.set_ylabel(u'C')
        ax.set_xlabel(u'Time')
        
        
        
#    ax.hist(dataset[:, i], 100)
        
    #ax.set_title('PCA outlier %s, outliers_fraction = %.2f' % (str(i+1),outliers_fraction))
    #ax.set_ylabel('Y')
    #ax.set_xlabel('X')
    ax.set_xlim(0,3601)
    #ax.set_ylim(-8,8)

ax = fig.add_subplot(2, 1, 2)
    
x = list(range(1,3601))
ax.plot(x, score, linewidth=1)
#ax.hist(score)
#    ax.hist(dataset[:, i], 100)
        
    #ax.set_title('PCA outlier %s, outliers_fraction = %.2f' % (str(i+1),outliers_fraction))
    #ax.set_ylabel('Y')
    #ax.set_xlabel('X')
ax.set_xlim(0,3601)



plt.show()
