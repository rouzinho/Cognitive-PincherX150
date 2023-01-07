#!/usr/bin/env python3
import re
import sys
import copy
from turtle import right

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
    device = torch.device("cpu")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(4, 2),
            torch.nn.Sigmoid(),
            #torch.nn.Linear(3, 2),
            #torch.nn.Sigmoid(),
            torch.nn.Linear(2, 1)
            #torch.nn.Tanh()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(1, 2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(2, 4),
            torch.nn.Sigmoid(),
            #torch.nn.Linear(4, 3),
            #torch.nn.Sigmoid(),
            torch.nn.Linear(4, 2),
            torch.nn.Sigmoid()
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