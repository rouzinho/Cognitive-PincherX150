#! /usr/bin/python3
import numpy as np
import random
import math
import time
from sklearn.preprocessing import MinMaxScaler

w = [0.5,0.5,0.5]
c = [0.27,-0.106,0.033]

def pose_to_color(w):
   n_x = np.array(w[0])
   n_x = n_x.reshape(-1,1)
   n_y = np.array(w[1])
   n_y = n_y.reshape(-1,1)
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
    
def color_to_pose(c):
   n_x = np.array(c[0])
   n_x = n_x.reshape(-1,1)
   n_y = np.array(c[1])
   n_y = n_y.reshape(-1,1)
   n_p = np.array(c[2])
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


p = color_to_pose(w)
j = pose_to_color(c)


print(p)
print(j)


#data = []
#for u,v in zip(n_x,n_y):
#   tup = [u,v]
#   data.append(copy.deepcopy(tup))