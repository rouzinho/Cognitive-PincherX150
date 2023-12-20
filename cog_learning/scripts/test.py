#!/usr/bin/env python3
import torch;
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from torch.autograd import Variable
import numpy as np
import random
import math
from os import path
import os
from os import listdir
from os.path import isfile, join
from torch.distributions.normal import Normal
from sklearn.preprocessing import MinMaxScaler
import copy
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

class Sampling(nn.Module):
   def forward(self, z_mean, z_log_var):
      # get the shape of the tensor for the mean and log variance
      #batch, dim = z_mean.shape
      # generate a normal random tensor (epsilon) with the same shape as z_mean
      # this tensor will be used for reparameterization trick
      #print(z_mean.shape)
      #epsilon = Normal(0, 1).sample(z_mean.shape).to(z_mean.device)
      # apply the reparameterization trick to generate the samples in the
      # latent space
      #return z_mean + torch.exp(0.5 * z_log_var) * epsilon
      vector_size = z_log_var.size()
      eps = Variable(torch.FloatTensor(vector_size).normal_())
      std = z_log_var.mul(0.5).exp_()
      return eps.mul(std).add_(z_mean)

class VariationalEncoder(nn.Module):
   def __init__(self, latent_dims):
      super(VariationalEncoder, self).__init__()
      self.linear1 = nn.Linear(8, 6)
      self.linear2 = nn.Linear(6, 4)
      self.linear3 = nn.Linear(4, 3)
      self.linear4 = nn.Linear(3, latent_dims)
      self.linear5 = nn.Linear(3, latent_dims)
      #self.N = torch.distributions.Normal(0, 1)
      #self.kl = 0
      self.sampling = Sampling()

   def forward(self, x):
      #x = torch.flatten(x, start_dim=1)
      x = torch.relu(self.linear1(x))
      x = torch.relu(self.linear2(x))
      x = torch.relu(self.linear3(x))
      z_mean = self.linear4(x)
      z_log_var = self.linear5(x)
      #mu =  torch.tanh(self.linear2(x))
      #sigma = torch.exp(self.linear3(x))
      #z = mu + sigma*self.N.sample(mu.shape)
      #self.kl = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
      #return z
      z = self.sampling(z_mean, z_log_var)
      #epsilon = torch.randn_like(z_mean)
      #z_reparametrized = z_mean + z_log_var*epsilon
      return z_mean, z_log_var, z

class Decoder(nn.Module):
   def __init__(self, latent_dims):
      super(Decoder, self).__init__()
      self.linear1 = nn.Linear(latent_dims, 3)
      self.linear2 = nn.Linear(3, 4)
      self.linear3 = nn.Linear(4, 6)
      self.linear4 = nn.Linear(6, 8)

   def forward(self, z):
      z = torch.relu(self.linear1(z)) #F.relu
      z = torch.relu(self.linear2(z))
      z = torch.relu(self.linear3(z))
      z = torch.sigmoid(self.linear4(z))
      #z = torch.tanh(self.linear4(z))
      return z
      #return z.reshape((-1, 1, 28, 28))

class VariationalAutoencoder(nn.Module):
   def __init__(self, latent_dims):
      super(VariationalAutoencoder, self).__init__()
      self.encoder = VariationalEncoder(latent_dims)
      self.decoder = Decoder(latent_dims)

   def forward(self, x):
      #z = self.encoder(x)
      #return self.decoder(z)
      z_mean, z_log_var, z = self.encoder(x)
      # pass the latent vector through the decoder to get the reconstructed
      # image
      reconstruction = self.decoder(z)
      # return the mean, log variance and the reconstructed image
      return z_mean, z_log_var, reconstruction

class Habituation(object):
   def __init__(self):
      self.vae = VariationalAutoencoder(2)

   def vae_gaussian_kl_loss(self, mu, logvar):
      # see Appendix B from VAE paper:
      # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
      # https://arxiv.org/abs/1312.6114
      #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
      #return KLD.mean()
      #return KLD
      KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
      KLD = torch.sum(KLD_element).mul_(-0.5)
      return KLD

   def reconstruction_loss(self, x_reconstructed, x):
      bce_loss = nn.BCELoss()
      return bce_loss(x_reconstructed, x)

   def vae_loss(self, y_pred, y_true):
      mu, logvar, recon_x = y_pred
      self.recon_loss = self.reconstruction_loss(recon_x, y_true)
      self.kld_loss = self.vae_gaussian_kl_loss(mu, logvar)
      return 500 * self.recon_loss + self.kld_loss
      #return (self.recon_loss + self.kld_loss) / self.nb_data

   def train(self, data, epochs=6000):
      kl_weight = 0.8
      #opt = torch.optim.Adam(list(self.vae.encoder.parameters()) + list(self.vae.decoder.parameters()), lr=0.001)
      opt = torch.optim.Adam(self.vae.parameters(), lr=0.007)
      train_loss = 0.0
      last_loss = 10
      stop = False
      i = 0
      self.nb_data = len(data)
      
      #for epoch in range(epochs):
      while not stop:
         #random.shuffle(data)
         for sample, y in data:
            self.vae.train()
            s = sample.to(device) # GPU
            opt.zero_grad()
            #x_hat = self.vae(s)
            #      loss = ((s - x_hat)**2).sum() + autoencoder.encoder.kl
            #loss_mse = ((s - x_hat)**2).sum() 
            # KL divergence between encoder distrib. and N(0,1) distrib. 
            #loss_kl = self.vae.encoder.kl 
            # Get weighted loss
            #loss = (loss_mse * (1 - kl_weight) + loss_kl * kl_weight)
            #train_loss += loss.item()
            #print("loss reconstruct : ",loss_mse)
            #print("loss KL : ",loss_kl)
            #print("loss total : ",loss)
            
            pred = self.vae(s)
            loss = self.vae_loss(pred, s)
            print("loss reconstruct : ",self.recon_loss)
            #print("loss KL : ",self.kld_loss)
            #print("loss total : ",loss)
            loss.backward()
            opt.step()
            i += 1
            if self.kld_loss < 0.005 and self.recon_loss < 0.0001:
               stop = True
            #last_loss = loss_mse
            #if nb_data > 1:
            #   if loss_mse < 9e-5 and loss_kl < 9e-5:
            #      stop = True
            #      print("epochs : ",i)
            #else:
            #   if i > 10000:
            #      stop = True

   def plot_latent(self, dataset, num_batches=100):
      colors = []
      for i in range(0,1):
         for sample, col in dataset:
               self.vae.eval()
               z, z_log, recon = self.vae.encoder(sample)
               print(col,z)
               z = z.to('cpu').detach().numpy()
               plt.scatter(z[0], z[1], c=col, cmap='tab10')

def scaleDatasPosition(ee, min_, max_):
    n_x = np.array(ee)
    n_x = n_x.reshape(-1,1)
    scaler_x = MinMaxScaler(feature_range=(-1,1))
    x_minmax = np.array([min_, max_])
    scaler_x.fit(x_minmax[:, np.newaxis])
    n_x = scaler_x.transform(n_x)
    n_x = n_x.reshape(1,-1)
    n_x = n_x.flatten()
    data = []
        
    return n_x[0]

def UnScaleDatasFromTensor(tensor_):
    tmp = tensor_.tolist()
    n_x = np.array(tmp[0])
    n_x = n_x.reshape(-1,1)
    n_y = np.array(tmp[1])
    n_y = n_y.reshape(-1,1)
    scaler_x = MinMaxScaler(feature_range=(0.18,0.67))
    scaler_y = MinMaxScaler(feature_range=(-0.35,0.35))
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
    data = []
    for u,v in zip(n_x,n_y):
        tup = [u,v]
        data.append(copy.deepcopy(tup))
        
    return data[0]

def scale_data(data, min_, max_):
   n_x = np.array(data)
   n_x = n_x.reshape(-1,1)
   scaler_x = MinMaxScaler(feature_range=(min_,max_))
   x_minmax = np.array([-1, 1])
   scaler_x.fit(x_minmax[:, np.newaxis])
   n_x = scaler_x.transform(n_x)
   n_x = n_x.reshape(1,-1)
   n_x = n_x.flatten()
            
   return n_x[0]

if __name__ == "__main__":
   max_ = 0.45
   min_ = -0.45
   t = scale_data(0.5,min_,max_)
   print(t)
   #d = scaleDatasPosition(x,0.18,0.67)
   #print(d)
   

   """"torch.manual_seed(58)
   latent_dims = 2
   data = []
   #data = torch.utils.data.DataLoader(
   #     torchvision.datasets.MNIST('./data',
   #            transform=torchvision.transforms.ToTensor(),
   #            download=True),
   #     batch_size=128,
   #     shuffle=True)
   goal = [0.1,0.1,0.5,0.2,0.3,0.25,0.65,0.9]
   sec_goal = [0.4,0.4,0.1,0.6,0.1,0.6,0.1,0.45]
   third_goal = [0.2,0.34,0.12,0.43,0.8,0.85,0.45,0.76]
   test_goal = [0.6,0.7,0.4,0.15,0.67,0.77,0.8,0.35]
   test_goal2 = [0.8,0.6,0.1,0.3,0.5,0.45,0.9,0.1]
   sim1 = [0.1,0.1,0.5,0.2,0.3,0.25,0.4,0.9]
   sim2 = [0.4,0.4,0.1,0.3,0.1,0.6,0.1,0.45]
   sim3 = [0.2,0.34,0.12,0.43,0.8,0.85,0.45,0.9]
   sim4 = [0.6,0.7,0.9,0.15,0.67,0.77,0.8,0.35]
   sim5 = [0.8,0.6,0.1,0.3,0.5,0.15,0.9,0.1]
   tensor_goal = torch.tensor(goal,dtype=torch.float)
   tensor_secgoal = torch.tensor(sec_goal,dtype=torch.float)
   tensor_rdgoal = torch.tensor(third_goal,dtype=torch.float)
   tensor_test = torch.tensor(test_goal,dtype=torch.float)
   tensor_test2 = torch.tensor(test_goal2,dtype=torch.float)
   ts1 = torch.tensor(sim1,dtype=torch.float)
   ts2 = torch.tensor(sim2,dtype=torch.float)
   ts3 = torch.tensor(sim3,dtype=torch.float)
   ts4 = torch.tensor(sim4,dtype=torch.float)
   ts5 = torch.tensor(sim5,dtype=torch.float)
   c_r = "red"
   c_b = "blue"
   c_g = "green"
   c_y = "yellow"
   c_p = "pink"
   c_c = "cyan"
   c_o = "orange"
   c_pu = "purple"
   c_br = "brown"
   c_gr = "gray"
   data.append([tensor_goal,c_r])
   data.append([tensor_secgoal,c_b])
   data.append([tensor_rdgoal,c_g])
   data.append([tensor_test,c_y])
   data.append([tensor_test2,c_p])
   data.append([ts1,c_c])
   data.append([ts2,c_o])
   data.append([ts3,c_pu])
   data.append([ts4,c_br])
   data.append([ts5,c_gr])
   habit = Habituation()
   habit.train(data)
   res = habit.vae.forward(tensor_goal)
   print("original : ",tensor_goal)
   print("reconstructed : ",res)
    #res2 = vae.forward(tensor_secgoal)
    #res3 = vae.forward(tensor_rdgoal)
    #print("RECONSTRUCTION ",res)
    #print("RECONSTRUCTION 2",res2)
    #print("RECONSTRUCTION 3",res3)
    #data.append([tensor_test,c_y])
   habit.plot_latent(data)
   plt.show()"""