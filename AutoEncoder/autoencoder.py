#!/usr/bin/env python3
import re
import sys
import copy
from turtle import right
import matplotlib.pyplot as plt

#from torch._C import T
import rospy
#import rosbag
#from std_msgs.msg import Float64
import time
#import moveit_msgs.msg
import geometry_msgs.msg
#import roslib
#import rospy
import numpy as np
import cv2
from math import pi
import math
#from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
import os.path
from os import listdir
from os import path
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter1d
#from visdom import Visdom
import random
import glob

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :128, :128]

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            #128 * 128 * 1
            nn.Conv2d(1, 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #64 * 64 * 16
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            #32*32 * 32
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(4*4*64,512),
            nn.ReLU(True),
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,128),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(64,32),
            nn.ReLU(True),
            nn.Linear(32,16),
            nn.Tanh(),
            nn.Linear(16,8),
        )
        self.decoder = torch.nn.Sequential(
            nn.Linear(8,16),
            nn.Tanh(),
            nn.Linear(16,32),
            nn.ReLU(True),
            nn.Linear(32,64),
            nn.ReLU(True),
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            nn.Linear(512,4*4*64),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 4, 4)),
            #8 * 8 * 64
            nn.ConvTranspose2d(64, 32, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            #16*16*128
            nn.ConvTranspose2d(16, 8, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #32 * 32 * 64
            nn.ConvTranspose2d(8, 4, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            #64 * 64 * 32
            nn.ConvTranspose2d(4, 1, 3, stride=2,padding=1,output_padding=1),
            #Trim(),
            nn.Sigmoid(),
        )
        self.training = []
        self.test = []

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def forward_latent(self, x):
        decoded = self.decoder(x)
        return decoded

class CompressState(object):
    def __init__(self):
        self.ae = AutoEncoder()
        rospy.init_node('autoencoder', anonymous=True)
        rospy.Subscriber("/autoencoder/action", Pose, self.callbackAction)
        self.memory = []

    def callbackAction(self,msg):
        tmp = []
        tmp.append(msg.position.x)
        tmp.append(msg.position.y)
        tmp.append(msg.position.z)
        t = self.getTensor(tmp)
        dec = self.decodeLatent(t)
        res = dec.cpu().detach().numpy()
        norm = self.reScaleTensor(res)
        print(norm)

    def printModel(self):
        print(self.ae)

    def scaleData(self,dat):
        #scale origine point position
        n_x = np.array(dat[0])
        n_x = n_x.reshape(-1,1)
        n_y = np.array(dat[1])
        n_y = n_y.reshape(-1,1)
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        x_minmax = np.array([0.1, 0.38])
        y_minmax = np.array([-0.25, 0.25])
        scaler_x.fit(x_minmax[:, np.newaxis])
        scaler_y.fit(y_minmax[:, np.newaxis])
        n_x = scaler_x.transform(n_x)
        n_x = n_x.reshape(1,-1)
        n_x = n_x.flatten()
        n_y = scaler_y.transform(n_y)
        n_y = n_y.reshape(1,-1)
        n_y = n_y.flatten()
        #scale vector displacement
        v_x = np.array(dat[2])
        v_x = v_x.reshape(-1,1)
        v_y = np.array(dat[3])
        v_y = v_y.reshape(-1,1)
        scaler_vec = MinMaxScaler()
        vec_minmax = np.array([-0.2, 0.2])
        scaler_vec.fit(vec_minmax[:, np.newaxis])
        v_x = scaler_vec.transform(v_x)
        v_x = v_x.reshape(1,-1)
        v_x = v_x.flatten()
        v_y = scaler_vec.transform(v_y)
        v_y = v_y.reshape(1,-1)
        v_y = v_y.flatten()
        #scale pitch and roll
        p = np.array(dat[4])
        p = p.reshape(-1,1)
        r = np.array(dat[5])
        r = r.reshape(-1,1)
        scaler_pitch = MinMaxScaler()
        scaler_roll = MinMaxScaler()
        p_minmax = np.array([0, 1.5])
        r_minmax = np.array([-0.5, 0.5])
        scaler_pitch.fit(p_minmax[:, np.newaxis])
        scaler_roll.fit(r_minmax[:, np.newaxis])
        p = scaler_pitch.transform(p)
        p = p.reshape(1,-1)
        p = p.flatten()
        r = scaler_roll.transform(r)
        r = r.reshape(1,-1)
        r = r.flatten()
        data = []
        data.append(n_x[0])
        data.append(n_y[0])
        data.append(v_x[0])
        data.append(v_y[0])
        data.append(p[0])
        data.append(r[0])

        return data

    def reScaleTensor(self,data):
        #scale origine point position
        n_x = np.array(data[0])
        n_x = n_x.reshape(-1,1)
        n_y = np.array(data[1])
        n_y = n_y.reshape(-1,1)
        scaler_x = MinMaxScaler(feature_range=[0.1,0.38])
        scaler_y = MinMaxScaler(feature_range=[-0.25,0.25])
        x_minmax = np.array([0, 1])
        y_minmax = np.array([0, 1])
        scaler_x.fit(x_minmax[:, np.newaxis])
        scaler_y.fit(y_minmax[:, np.newaxis])
        n_x = scaler_x.transform(n_x)
        n_x = n_x.reshape(1,-1)
        n_x = n_x.flatten()
        n_y = scaler_y.transform(n_y)
        n_y = n_y.reshape(1,-1)
        n_y = n_y.flatten()
        #scale vector displacement
        v_x = np.array(data[2])
        v_x = v_x.reshape(-1,1)
        v_y = np.array(data[3])
        v_y = v_y.reshape(-1,1)
        scaler_vec = MinMaxScaler(feature_range=[-0.2,0.2])
        vec_minmax = np.array([0, 1])
        scaler_vec.fit(vec_minmax[:, np.newaxis])
        v_x = scaler_vec.transform(v_x)
        v_x = v_x.reshape(1,-1)
        v_x = v_x.flatten()
        v_y = scaler_vec.transform(v_y)
        v_y = v_y.reshape(1,-1)
        v_y = v_y.flatten()
        #scale pitch and roll
        p = np.array(data[4])
        p = p.reshape(-1,1)
        r = np.array(data[5])
        r = r.reshape(-1,1)
        scaler_pitch = MinMaxScaler(feature_range=[0,1.5])
        scaler_roll = MinMaxScaler(feature_range=[-0.5,0.5])
        p_minmax = np.array([0, 1])
        r_minmax = np.array([0, 1])
        scaler_pitch.fit(p_minmax[:, np.newaxis])
        scaler_roll.fit(r_minmax[:, np.newaxis])
        p = scaler_pitch.transform(p)
        p = p.reshape(1,-1)
        p = p.flatten()
        r = scaler_roll.transform(r)
        r = r.reshape(1,-1)
        r = r.flatten()
        data = []
        data.append(n_x[0])
        data.append(n_y[0])
        data.append(v_x[0])
        data.append(v_y[0])
        data.append(p[0])
        data.append(r[0])

        return data

    def buildDataSet(self):
        datas = []
        for i in range(0,100):
            for j in range(0,100):
                if i > 0 or j > 0:
                    t = torch.tensor([[i/100,j/100]])
                    datas.append(t)
        random.shuffle(datas)
        self.test = datas[-1000:]
        self.training = datas[:-1000]
        #self.training = datas

    def addTensorToMemory(self,data):
        t = torch.tensor(data,dtype=torch.float)
        self.memory.append(t)

    def getTensor(self,data):
        t = torch.tensor(data,dtype=torch.float)
        return t

    #takes object location and motor command as input and produces the expected future object location as output
    def trainModel(self):
        current_cost = 0
        current_test = 0
        last_cost = 15
        learning_rate = 1e-1
        epochs = 2000
        data_input = []

        self.ae.to(device)
        criterion = torch.nn.BCELoss()
        #optimizer = torch.optim.Adam(self.ae.parameters(),lr=learning_rate)
        optimizer = torch.optim.Adadelta(self.ae.parameters(),lr=learning_rate)
        #get inputs and targets

        img = cv2.imread('/home/altair/interbotix_ws/src/depth_perception/states/state_0.jpg')
        t = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        res = np.float32(t)
        res = res*(1/255.0)
        res = res[np.newaxis,...]
        print(res.shape)
        sample = torch.from_numpy(res)
        target = sample
        target = target.long()
        sample = sample.unsqueeze(0)
        sample = sample.to(device)
        for i in range(0,epochs):
            self.ae.train()
            current_cost = 0
            optimizer.zero_grad()
            enc, dec = self.ae(sample)
            cost = criterion(dec,sample)
            cost.backward()
            optimizer.step()
            current_cost = current_cost + cost.item()
            print("Epoch: {}/{}...".format(i, epochs),
                                "Cross Entropy Training : ",current_cost)
            """random.shuffle(self.training)
            for j in range(0,len(self.training)):
                self.ae.train()
                sample = self.training[j]
                enc, dec = self.ae(sample)
                cost = criterion(dec,sample)
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                current_cost = current_cost + cost.item()
            print("Epoch: {}/{}...".format(i, epochs),
                                "MSE Training : ",current_cost)
            current_test = 0
            if i % 10 == 0:
                for k in range(0,len(self.test)):
                    self.ae.eval()
                    sample = self.test[k]
                    enc, dec = self.ae(sample)
                    cost = criterion(dec,sample)
                    #optimizer.zero_grad()
                    #cost.backward()
                    #optimizer.step()
                    current_test = current_test + cost.item()
                print("Epoch: {}/{}...".format(i, epochs),
                                    "MSE Test: ",current_test)"""
            
        name = "autoenc.pt"

        torch.save(self.ae, name)

    def loadModel(self):
        self.ae = torch.load("autoenc.pt")

    def getRepresentation(self,input):
        self.ae.eval()
        enc, dec = self.ae(input)
        return enc, dec

    def decodeLatent(self,input):
        self.ae.eval()
        dec = self.ae.forward_latent(input)
        return dec


if __name__ == "__main__":
    torch.manual_seed(58)
    cs = CompressState()
    cs.trainModel()
    #cs.loadModel()
    img = cv2.imread('/home/altair/interbotix_ws/src/depth_perception/states/state_0.jpg')
    t = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    res = np.float32(t)
    res = res*(1/255.0)
    res = res[np.newaxis,...]
    
    sample = torch.from_numpy(res)
    target = sample
    target = target.long()
    sample = sample.unsqueeze(0)
    sample = sample.to(device)
    enc, dec = cs.getRepresentation(sample)
    dec = torch.squeeze(dec)
    
    test = dec.cpu().detach().numpy()
    plt.imshow(test)
    plt.show()
