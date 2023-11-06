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

if not is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

class VariationalEncoder(nn.Module):
   def __init__(self, latent_dims):
      super(VariationalEncoder, self).__init__()
      self.linear1 = nn.Linear(5, 3)
      self.linear2 = nn.Linear(3, latent_dims)
      self.linear3 = nn.Linear(3, latent_dims)

      self.N = torch.distributions.Normal(0, 1)
      #self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
      #self.N.scale = self.N.scale.cuda()
      self.kl = 0

   def forward(self, x):
      #x = torch.flatten(x, start_dim=1)
      x = F.tanh(self.linear1(x))
      mu =  self.linear2(x)
      sigma = torch.exp(self.linear3(x))
      z = mu + sigma*self.N.sample(mu.shape)
      #self.kl = torch.mean(sigma**2 + mu**2 - torch.log(sigma) - 1/2)
      self.kl = -0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp())
      return z

class Decoder(nn.Module):
   def __init__(self, latent_dims):
      super(Decoder, self).__init__()
      self.linear1 = nn.Linear(latent_dims, 3)
      self.linear2 = nn.Linear(3, 5)

   def forward(self, z):
      z = F.tanh(self.linear1(z)) #F.relu
      z = torch.sigmoid(self.linear2(z))
      return z
      #return z.reshape((-1, 1, 28, 28))

class VariationalAutoencoder(nn.Module):
   def __init__(self, latent_dims):
      super(VariationalAutoencoder, self).__init__()
      self.encoder = VariationalEncoder(latent_dims)
      self.decoder = Decoder(latent_dims)

   def forward(self, x):
      z = self.encoder(x)
      return self.decoder(z)


class Habituation(object):
   def __init__(self):
      self.vae = VariationalAutoencoder(2)

   def train(self, data, epochs=6000):
      kl_weight = 0.8
      opt = torch.optim.Adam(self.vae.parameters())
      train_loss = 0.0
      stop = False
      i = 0
      nb_data = len(data)
      #for epoch in range(epochs):
      while not stop:
         for sample, y in data:
            s = sample.to(device) # GPU
            opt.zero_grad()
            x_hat = self.vae(s)
            #loss = ((s - x_hat)**2).sum() + autoencoder.encoder.kl
            loss_mse = ((s - x_hat)**2).sum() 
            # KL divergence between encoder distrib. and N(0,1) distrib. 
            loss_kl = self.vae.encoder.kl 
            # Get weighted loss
            loss = (loss_mse * (1 - kl_weight) + loss_kl * kl_weight)
            train_loss += loss.item()
            print("loss reconstruct : ",loss_mse)
            print("loss KL : ",loss_kl)
            loss.backward()
            opt.step()
            i += 1
            if nb_data > 1:
               if loss_mse < 9e-5 and loss_kl < 9e-5:
                  stop = True
                  print("epochs : ",i)
            else:
               if i > 10000:
                  stop = True

   def plot_latent(self, data, num_batches=100):
      colors = []
      for i in range(0,10):
         for sample, col in data:
               z = self.vae.encoder(sample)
               z = z.to('cpu').detach().numpy()
               #print(z)
               plt.scatter(z[0], z[1], c=col, cmap='tab10')

if __name__ == "__main__":
    torch.manual_seed(58)
    latent_dims = 2
    data = []
    #data = torch.utils.data.DataLoader(
    #     torchvision.datasets.MNIST('./data',
    #            transform=torchvision.transforms.ToTensor(),
    #            download=True),
    #     batch_size=128,
    #     shuffle=True)
    goal = [0.1,0.1,0.5,0.2,0.3]
    sec_goal = [0.4,0.4,0.1,0.6,0.1]
    third_goal = [0.2,0.34,0.12,0.43,0.8]
    test_goal = [0.1,0.1,0.5,0.3,0.3]
    tensor_goal = torch.tensor(goal,dtype=torch.float)
    tensor_secgoal = torch.tensor(sec_goal,dtype=torch.float)
    tensor_rdgoal = torch.tensor(third_goal,dtype=torch.float)
    tensor_test = torch.tensor(test_goal,dtype=torch.float)
    c_r = "red"
    c_b = "blue"
    c_g = "green"
    c_y = "yellow"
    data.append([tensor_goal,c_r])
    vae = VariationalAutoencoder(latent_dims).to(device) # GPU
    data.append([tensor_secgoal,c_b])
    data.append([tensor_rdgoal,c_g])
    vae = train(vae, data)
    res = vae.forward(tensor_goal)
    res2 = vae.forward(tensor_secgoal)
    res3 = vae.forward(tensor_rdgoal)
    print("RECONSTRUCTION ",res)
    print("RECONSTRUCTION 2",res2)
    print("RECONSTRUCTION 3",res3)
    #data.append([tensor_test,c_y])
    plot_latent(vae, data)
    plt.show()