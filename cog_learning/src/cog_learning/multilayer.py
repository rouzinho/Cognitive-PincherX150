#!/usr/bin/env python3
import torch;
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import math
from os import path
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 100
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

is_cuda = torch.cuda.is_available()
#device = torch.device("cpu")
torch.manual_seed(3407)

if not is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

class MultiLayerEncoder(nn.Module):
    def __init__(self,input_layer,output_layer):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_layer, output_layer),
            nn.Tanh()
            #nn.Linear(output_layer, output_layer),
            #nn.Tanh()
            #nn.Linear(middle_layer1, output_layer),
            ##nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)
    
class MultiLayerDecoder(nn.Module):
    def __init__(self,input_layer,middle_layer1,middle_layer2,output_layer):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_layer, middle_layer1),
            nn.Tanh(),
            nn.Linear(middle_layer1, middle_layer2),
            nn.Tanh(),
            nn.Linear(middle_layer2, output_layer),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.layers(x)
    
class MultiLayerP(nn.Module):
    def __init__(self,input_layer,middle_layer,output_layer):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_layer, middle_layer),
            nn.Tanh(),
            nn.Linear(middle_layer, middle_layer),
            nn.Tanh(),
            nn.Linear(middle_layer, output_layer),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.layers(x)
    
class MultiLayerPredictor(nn.Module):
    def __init__(self,input_layer,middle_layer,output_layer):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_layer, middle_layer),
            nn.Tanh(),
            nn.Linear(middle_layer, output_layer),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)