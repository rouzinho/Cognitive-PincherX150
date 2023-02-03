#! /usr/bin/python3
import numpy as np
import random
import math
import time
from sklearn.preprocessing import MinMaxScaler

w = [0.5,0,0.5]

n_x = np.array(w[0:2])
n_x = n_x.reshape(-1,1)
n_p = np.array(w[2])
n_p = n_p.reshape(-1,1)
scaler_xy = MinMaxScaler(feature_range=(-0.4, 0.4))
scaler_p = MinMaxScaler(feature_range=(0, 1.6))
xy_minmax = np.array([0, 1])
p_minmax = np.array([0, 1])
scaler_xy.fit(xy_minmax[:, np.newaxis])
scaler_p.fit(p_minmax[:, np.newaxis])
n_x = scaler_xy.transform(n_x)
n_x = n_x.reshape(1,-1)
n_x = n_x.flatten()
n_p = scaler_p.transform(n_p)
n_p = n_p.reshape(1,-1)
n_p = n_p.flatten()
scaled_w = [n_x[0],n_x[1],n_p[0]]

print(scaled_w)
#data = []
#for u,v in zip(n_x,n_y):
#   tup = [u,v]
#   data.append(copy.deepcopy(tup))