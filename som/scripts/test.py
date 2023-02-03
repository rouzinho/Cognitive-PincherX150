#! /usr/bin/python3
import numpy as np
import random
import math
import time
from sklearn.preprocessing import MinMaxScaler

w = [0.1,0.0,0.8]

def col_to_w():
   n_x = np.array(w[0])
   n_x = n_x.reshape(-1,1)
   n_y = np.array(w[1])
   n_y = n_x.reshape(-1,1)
   n_p = np.array(w[2])
   n_p = n_p.reshape(-1,1)
   scaler_x = MinMaxScaler(feature_range=(0.05, 0.4))
   scaler_y = MinMaxScaler(feature_range=(-0.4, 0.4))
   scaler_p = MinMaxScaler(feature_range=(0, 1.6))
   x_minmax = np.array([0, 1])
   y_minmax = np.array([0, 1])
   p_minmax = np.array([0, 1])
   scaler_x.fit(x_minmax[:, np.newaxis])
   scaler_y.fit(y_minmax[:, np.newaxis])
   scaler_p.fit(p_minmax[:, np.newaxis])
   n_x = scaler_x.transform(n_x)
   n_x = n_x.reshape(1,-1)
   n_x = n_x.flatten()
   n_y = scaler_y.transform(n_y)
   n_y = n_y.reshape(1,-1)
   n_y = n_y.flatten()
   n_p = scaler_p.transform(n_p)
   n_p = n_p.reshape(1,-1)
   n_p = n_p.flatten()
   scaled_w = [n_x[0],n_y[0],n_p[0]]

   return scaled_w

def w_to_c():
   n_x = np.array(w[0])
   n_x = n_x.reshape(-1,1)
   n_y = np.array(w[1])
   n_y = n_x.reshape(-1,1)
   n_p = np.array(w[2])
   n_p = n_p.reshape(-1,1)
   scaler_x = MinMaxScaler()
   scaler_y = MinMaxScaler()
   scaler_p = MinMaxScaler()
   x_minmax = np.array([0.05, 0.4])
   y_minmax = np.array([-0.4, 0.4])
   p_minmax = np.array([0, 1.6])
   scaler_x.fit(x_minmax[:, np.newaxis])
   scaler_y.fit(y_minmax[:, np.newaxis])
   scaler_p.fit(p_minmax[:, np.newaxis])
   n_x = scaler_x.transform(n_x)
   n_x = n_x.reshape(1,-1)
   n_x = n_x.flatten()
   n_y = scaler_y.transform(n_y)
   n_y = n_y.reshape(1,-1)
   n_y = n_y.flatten()
   n_p = scaler_p.transform(n_p)
   n_p = n_p.reshape(1,-1)
   n_p = n_p.flatten()
   scaled_w = [n_x[0],n_y[0],n_p[0]]
   
   return scaled_w

t = col_to_w()
p = w_to_c()

print(t)
print(p)



#data = []
#for u,v in zip(n_x,n_y):
#   tup = [u,v]
#   data.append(copy.deepcopy(tup))