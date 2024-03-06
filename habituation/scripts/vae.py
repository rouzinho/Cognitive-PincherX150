#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Int16
from std_msgs.msg import Bool
import torch;
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import random
import math
from os import path
import os
from os import listdir
from os.path import isfile, join
from torch.distributions.normal import Normal
from torch.autograd import Variable
from motion.msg import DmpOutcome
from motion.msg import Dmp
from detector.msg import Outcome
from habituation.msg import LatentPos
from sklearn.preprocessing import MinMaxScaler
from cog_learning.msg import LatentGoalDnf
from cog_learning.msg import LatentNNDNF
from cog_learning.msg import Goal
from cog_learning.msg import LatentGoalNN
from cluster_message.msg import SampleExplore
import copy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

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
   def __init__(self, input_dim, middle_dim, latent_dims):
      super(VariationalEncoder, self).__init__()
      self.linear1 = nn.Linear(input_dim, middle_dim)
      #self.linear2 = nn.Linear(7, 5)
      #self.linear3 = nn.Linear(5, 3)
      self.linear4 = nn.Linear(middle_dim, latent_dims)
      self.linear5 = nn.Linear(middle_dim, latent_dims)
      #self.N = torch.distributions.Normal(0, 1)
      #self.kl = 0
      self.sampling = Sampling()

   def forward(self, x):
      #x = torch.flatten(x, start_dim=1)
      x = torch.tanh(self.linear1(x)) #relu
      #x = torch.tanh(self.linear2(x))
      #x = torch.tanh(self.linear3(x)) #relu
      z_mean = self.linear4(x)
      #z_log_var = self.linear5(x)
      z_log_var = torch.exp(self.linear5(x))
      #z = mu + sigma*self.N.sample(mu.shape)
      #self.kl = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
      #return z
      #z_reparametrized = self.sampling(z_mean, z_log_var)
      epsilon = torch.randn_like(z_mean)
      ##z_reparametrized = z_mean + z_log_var*epsilon
      self.N = torch.distributions.Normal(0, 1)
      z_reparametrized = z_mean + z_log_var*self.N.sample(z_mean.shape)
      return z_mean, z_log_var, z_reparametrized

class Decoder(nn.Module):
   def __init__(self, latent_dims, middle_dim, output_dim):
      super(Decoder, self).__init__()
      self.linear1 = nn.Linear(latent_dims, middle_dim)
      #self.linear2 = nn.Linear(3, 5)
      #self.linear3 = nn.Linear(5, 7)
      self.linear4 = nn.Linear(middle_dim, output_dim)

   def forward(self, z):
      z = torch.tanh(self.linear1(z)) #F.relu
      z = self.linear4(z)

      return z

class VariationalAutoencoder(nn.Module):
   def __init__(self, input_dim, middle_dim, latent_dims):
      super(VariationalAutoencoder, self).__init__()
      self.encoder = VariationalEncoder(input_dim, middle_dim, latent_dims)
      self.decoder = Decoder(latent_dims, middle_dim, input_dim)

   def forward(self, x):
      #z = self.encoder(x)
      #return self.decoder(z)
      z_mean, z_log_var, z = self.encoder(x)
      # pass the latent vector through the decoder to get the reconstructed
      # image
      reconstruction = self.decoder(z)
      # return the mean, log variance and the reconstructed image
      return z_mean, z_log_var, reconstruction
   
   def save(self, name):
      torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            }, name)
      
   def load(self, name):
      checkpoint = torch.load(name)
      self.encoder.load_state_dict(checkpoint['encoder'])
      self.decoder.load_state_dict(checkpoint['decoder'])
      

class VariationalAE(object):
   def __init__(self,id_object,input_dim,middle_dim,latent_dim):
      self.input_dim = input_dim
      self.middle_dim = middle_dim
      self.latent_dim = latent_dim
      self.vae = VariationalAutoencoder(self.input_dim,self.middle_dim,self.latent_dim)
      self.memory = []
      self.mt_field = np.zeros((100,100,1), np.float32)
      self.id = id_object
      self.list_latent = []
      self.list_latent_scaled = []
      self.list_latent_action = []
      self.scale_factor = 30
      self.tmp_list = []
      self.bound_x = 0
      self.bound_y = 0
      self.max_bound_x = 0
      self.max_bound_y = 0
      self.min_latent_x = -1
      self.max_latent_x = 1
      self.min_latent_y = -1
      self.max_latent_y = 1

   def fill_latent(self):
      self.list_latent = []
      for sample in self.memory:
         self.vae.eval()
         z, z_log, recon = self.vae.encoder(sample)
         z = z.to('cpu').detach().numpy()
         self.list_latent.append(z)
      print("latent space : ",len(self.list_latent))

   def set_latent_dnf(self, exploration):
      ext_x, ext_y = self.get_latent_extremes(self.list_latent)
      self.list_latent_scaled = []
      if exploration == "static":
         self.bound_x = 100
         self.bound_y = 100
         if len(self.list_latent) >= 1:
            for i in self.list_latent:
               self.min_latent_x = -1
               self.max_latent_x = 1
               self.min_latent_y = -1
               self.max_latent_y = 1
               if (ext_x[0] < -1 or ext_x[1] > 1) or (ext_y[0] < -1 or ext_y[1] > 1):
                  self.min_latent_x = ext_x[0]
                  self.max_latent_x = ext_x[1]
                  self.min_latent_y = ext_y[0]
                  self.max_latent_y = ext_y[1]
               x = self.scale_latent_to_dnf_static(i[0],self.min_latent_x,self.max_latent_x)
               y = self.scale_latent_to_dnf_static(i[1],self.min_latent_y,self.max_latent_y)
               self.list_latent_scaled.append([round(x),round(y)])
         else:
            self.list_latent_scaled.append([50,50])
      else:
         dist_x = abs(ext_x[0]) + abs(ext_x[1])
         dist_y = abs(ext_y[0]) + abs(ext_y[1])
         self.max_bound_x = (dist_x * self.scale_factor)
         self.max_bound_y = (dist_y * self.scale_factor)
         #control on bounding
         if self.max_bound_x < 1:
            self.max_bound_x = 1
         if self.max_bound_y < 1:
            self.max_bound_y = 1 
         #padding to avoid having extreme values on the edge of DNF
         padding_x = round(self.max_bound_x * 0.1)
         padding_y = round(self.max_bound_y * 0.1)
         self.bound_x = round(self.max_bound_x)
         self.bound_y = round(self.max_bound_y)
         if len(self.list_latent) >= 1:
            for i in self.list_latent:
               x = self.scale_latent_to_dnf_dynamic(i[0],ext_x[0],ext_x[1],padding_x,self.bound_x-padding_x)
               y = self.scale_latent_to_dnf_dynamic(i[1],ext_y[0],ext_y[1],padding_y,self.bound_y-padding_y)
               self.list_latent_scaled.append([round(x),round(y)])
         else:
            self.list_latent_scaled.append([5,5])
            self.bound_x = round(10)
            self.bound_y = round(10)

   def get_latent_dnf_split(self, sample):
      msg_display_one = LatentPos()
      msg_display_minus = LatentPos()
      new_latent_single = LatentNNDNF()
      new_latent_single.max_x = self.bound_x
      new_latent_single.max_y = self.bound_y
      new_latent_minus_one = LatentNNDNF()
      new_latent_minus_one.max_x = self.bound_x
      new_latent_minus_one.max_y = self.bound_y
      self.vae.eval()
      z, z_log, recon = self.vae.encoder(sample)
      z = z.to('cpu').detach().numpy()
      for (i,j) in zip(self.list_latent,self.list_latent_scaled):
         if abs(abs(i[0]) - abs(z[0])) < 0.01 and abs(abs(i[1]) - abs(z[1])) < 0.01:
            print("found latent value")
            g = Goal()
            g.x = j[0]
            g.y = j[1]
            g.value = 1.0
            new_latent_single.list_latent.append(g)
            msg_display_one.x.append(i[0])
            msg_display_one.y.append(i[1])
         else:
            print("latent not found")
            g = Goal()
            g.x = j[0]
            g.y = j[1]
            g.value = 1.0
            new_latent_minus_one.list_latent.append(g)
            msg_display_minus.x.append(i[0])
            msg_display_minus.y.append(i[1])
      
      return new_latent_single, new_latent_minus_one, msg_display_one, msg_display_minus
            
   def set_dnf_to_latent(self, peak, exploration):
      ext_x, ext_y = self.get_latent_extremes(self.list_latent)
      latent_value = []
      #print("ext_x :",ext_x)
      #print("ext_y :",ext_y)
      if exploration == "static":
         self.max_bound_x = 100
         self.max_bound_y = 100
         if len(self.list_latent) >= 1:
            #more generic scaling
            x = self.scale_latent_to_dnf_dynamic(peak[0],10,90,ext_x[0],ext_x[1])
            y = self.scale_latent_to_dnf_dynamic(peak[1],10,90,ext_y[0],ext_y[1])
            latent_value.append(x)
            latent_value.append(y)
      else:
         #padding to avoid having extreme values on the edge of DNF
         padding_x = round(self.max_bound_x * 0.1)
         padding_y = round(self.max_bound_y * 0.1)
         if len(self.list_latent) >= 1:
            x = self.scale_latent_to_dnf_dynamic(peak[0],padding_x,self.bound_x-padding_x,ext_x[0],ext_x[1])
            y = self.scale_latent_to_dnf_dynamic(peak[1],padding_y,self.bound_y-padding_y,ext_y[0],ext_y[1])
            latent_value.append(x)
            latent_value.append(y)
      #print("Latent value : ",latent_value)
      #print("LATENT TESTED bound x ", self.bound_x, " bound y ", self.bound_y," max_bound_x ", self.max_bound_x, " max_bound_y ", self.max_bound_y)
      return latent_value

   #create eval latent value by including it in the latent space so it can expand this one
   def set_eval_to_latent_dnf(self, z, exploration):
      new_latent = LatentNNDNF()
      eval_value = Goal()
      list_eval = copy.deepcopy(self.list_latent)
      list_eval.append(z)
      ext_x, ext_y = self.get_latent_extremes(list_eval)
      if exploration == "static":
         new_latent.max_x = 100
         new_latent.max_y = 100
         min_x = -1
         max_x = 1
         min_y = -1
         max_y = 1
         if (ext_x[0] < -1 or ext_x[1] > 1) or (ext_y[0] < -1 or ext_y[1] > 1):
            min_x = ext_x[0]
            max_x = ext_x[1]
            min_y = ext_y[0]
            max_y = ext_y[1]
         x = self.scale_latent_to_dnf_static(z[0],min_x,max_x)
         y = self.scale_latent_to_dnf_static(z[1],min_y,max_y)
         eval_value.x = round(x)
         eval_value.y = round(y)
         eval_value.value = 1.0
      else:
         dist_x = abs(ext_x[0]) + abs(ext_x[1])
         dist_y = abs(ext_y[0]) + abs(ext_y[1])
         max_bound_x = (dist_x * self.scale_factor)
         max_bound_y = (dist_y * self.scale_factor) 
         #padding to avoid having extreme values on the edge of DNF
         padding_x = round(max_bound_x * 0.1)
         padding_y = round(max_bound_y * 0.1)
         max_bound_x = round(max_bound_x)
         max_bound_y = round(max_bound_y)
         x = self.scale_latent_to_dnf_dynamic(z[0],ext_x[0],ext_x[1],padding_x,max_bound_x-padding_x)
         #print("data",z[1])
         #print("min y ",ext_y[0])
         #print("max y ",ext_y[1])
         #print("padding min ",padding_y)
         #print("padding max ",max_bound_y-padding_y)
         y = self.scale_latent_to_dnf_dynamic(z[1],ext_y[0],ext_y[1],padding_y,max_bound_y-padding_y)
         eval_value.x = round(x)
         eval_value.y = round(y)
         eval_value.value = 1.0
         new_latent.max_x = max_bound_x
         new_latent.max_y = max_bound_y
      new_latent.list_latent.append(eval_value)
      #print("Eval DNF : ",new_latent)

      return new_latent
   
   #get the DNF value of eval value without integrating it in the latent space size
   def get_eval_latent_to_dnf(self, z, exploration):
      new_latent = LatentNNDNF()
      eval_value = Goal()
      list_eval = copy.deepcopy(self.list_latent)
      #list_eval.append(z)
      ext_x, ext_y = self.get_latent_extremes(list_eval)
      #print("extremes x : ",ext_x)
      #print("extremes y : ",ext_y)
      #print("z : ",z)
      if exploration == "static":
         new_latent.max_x = 100
         new_latent.max_y = 100
         x = self.scale_latent_to_dnf_static(z[0],self.min_latent_x,self.max_latent_x)
         y = self.scale_latent_to_dnf_static(z[1],self.min_latent_y,self.max_latent_y)
         eval_value.x = round(x)
         eval_value.y = round(y)
         eval_value.value = 1.0
      else:
         padding_x = round(self.max_bound_x * 0.1)
         padding_y = round(self.max_bound_y * 0.1)
         max_bound_x = round(self.max_bound_x)
         max_bound_y = round(self.max_bound_y)
         x = self.scale_latent_to_dnf_dynamic(z[0],ext_x[0],ext_x[1],padding_x,max_bound_x-padding_x)
         y = self.scale_latent_to_dnf_dynamic(z[1],ext_y[0],ext_y[1],padding_y,max_bound_y-padding_y)
         #print("data",z[1])
         #print(" x : ",x)
         #print(" y : ",y)
         #print("padding min ",padding_y)
         #print("padding max ",self.max_bound_y-padding_y)
         #y = self.scale_latent_to_dnf_dynamic(z[1],ext_y[0],ext_y[1],padding_y,self.max_bound_y-padding_y)
         eval_value.x = round(x)
         eval_value.y = round(y)
         eval_value.value = 1.0
         #print("bound x : ", self.bound_x)
         #print("bound y : ", self.bound_y)
         #print("max bound x : ", self.max_bound_x)
         #print("max bound y : ", self.max_bound_y)
         new_latent.max_x = self.bound_x
         new_latent.max_y = self.bound_y
      new_latent.list_latent.append(eval_value)

      return new_latent

   def get_latent_space(self):
      cp_val = copy.deepcopy(self.list_latent)
      return cp_val

   def get_latent_space_dnf(self):
      cp_val = copy.deepcopy(self.list_latent_scaled)
      return cp_val
   
   def get_bound_x(self):
      return self.bound_x
   
   def get_bound_y(self):
      return self.bound_y
   
   def get_max_bound_x(self):
      return self.max_bound_x
   
   def get_max_bound_y(self):
      return self.max_bound_y
   
   def get_id(self):
      return self.id
   
   def set_mt_field(self, img):
      self.mt_field = img

   def get_mt_field(self):
      return self.mt_field

   #get min max of x and y in latent space, used for scaling
   def get_latent_extremes(self, l_lat):
      best_min_x = 1
      best_max_x = -1
      ind_min_x = 0
      ind_max_x = 0
      best_min_y = 0
      best_max_y = 0
      ind_min_y = 0
      ind_max_y = 0
      for i in range(0,len(l_lat)):
         if l_lat[i][0] < best_min_x:
            ind_min_x = i
            best_min_x = l_lat[i][0]
         if l_lat[i][0] > best_max_x:
            ind_max_x = i
            best_max_x = l_lat[i][0]
         if l_lat[i][1] < best_min_y:
            ind_min_y = i
            best_min_y = l_lat[i][1]
         if l_lat[i][1] > best_max_y:
            ind_max_y = i
            best_max_y = l_lat[i][1]

      if len(l_lat) > 0:
         min_x = l_lat[ind_min_x][0]
         max_x = l_lat[ind_max_x][0]
         min_y = l_lat[ind_min_y][1]
         max_y = l_lat[ind_max_y][1]
         ext_x = [min_x,max_x]
         ext_y = [min_y,max_y]
      else:
         ext_x = []
         ext_y = []

      return ext_x, ext_y

   def scale_latent_to_dnf_static(self, data, min_, max_):
      n_x = np.array(data)
      n_x = n_x.reshape(-1,1)
      scaler_x = MinMaxScaler(feature_range=(10,90))
      x_minmax = np.array([min_, max_])
      scaler_x.fit(x_minmax[:, np.newaxis])
      n_x = scaler_x.transform(n_x)
      n_x = n_x.reshape(1,-1)
      n_x = n_x.flatten()
            
      return n_x[0]
   
   def scale_latent_to_dnf_dynamic(self, data, min_v, max_v, min_b, max_b):
      n_x = np.array(data)
      n_x = n_x.reshape(-1,1)
      scaler_x = MinMaxScaler(feature_range=(min_b,max_b))
      x_minmax = np.array([min_v, max_v])
      scaler_x.fit(x_minmax[:, np.newaxis])
      n_x = scaler_x.transform(n_x)
      n_x = n_x.reshape(1,-1)
      n_x = n_x.flatten()
            
      return n_x[0]

   def add_to_memory(self, data):
      self.memory.append(data)

   def vae_gaussian_kl_loss(self, mu, logvar):
      KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
      return KLD.mean()
      #return KLD

   def reconstruction_loss(self, x_reconstructed, x):
      mse_loss = nn.MSELoss()
      return mse_loss(x_reconstructed, x)

   def vae_loss(self, y_pred, y_true):
      mu, logvar, recon_x = y_pred
      self.recon_loss = self.reconstruction_loss(recon_x, y_true)
      self.kld_loss = self.vae_gaussian_kl_loss(mu, logvar)
      return 200 * self.recon_loss + self.kld_loss

   def train(self):
      kl_weight = 0.8
      opt = torch.optim.Adam(self.vae.parameters(), lr=0.01)
      train_loss = 0.0
      last_loss = 10
      stop = False
      i = 0
      min_err = 1.0
      err_rec = 1.0
      mem = copy.deepcopy(self.memory)
      while not stop:
         random.shuffle(mem)
         for sample in mem:
            self.vae.train()
            s = sample.to(device) # GPU
            opt.zero_grad()
            pred = self.vae(s)
            loss = self.vae_loss(pred, s)
            err_rec = self.recon_loss.item()
            #print("loss reconstruct : ",self.recon_loss.item())
            #print("loss KL : ",self.kld_loss.item())
            #print("loss total : ",loss)
            loss.backward()
            opt.step()
            i += 1
            if err_rec < min_err:
               min_err = err_rec
               #print("min reconstructed : ",min_err)
               #print("loss KL : ",self.kld_loss.item())
            if self.kld_loss < 0.04 and self.recon_loss < 0.0005:
               stop = True
               break

   def get_sample_latent(self, sample):
      self.vae.eval()
      z, z_log, recon = self.vae.encoder(sample)
      z = z.to('cpu').detach().numpy()

      return z
   
   def reconstruct_latent(self, sample):
      self.vae.eval()
      sample = sample.to(device)
      output = self.vae.decoder(sample)
      out = output.to('cpu').detach().numpy()

      return out

   #send latent space for display and keep it in memory
   def plot_latent(self):
      msg_latent = LatentPos()
      for i in self.list_latent:
         msg_latent.x.append(i[0])
         msg_latent.y.append(i[1])
            
      return msg_latent
   
   def test_reconstruct(self):
      self.vae.eval()
      res = self.vae(self.memory[0][0])
      print("original : ", self.memory[0][0])
      print("reconstruct : ",res[2])
      print(self.memory)
      for j in range(0,4):
         for i in self.memory:
            z, z_log, recon = self.vae.encoder(i[0])
            print(i[1],z)

   def reset_model(self):
      self.vae = VariationalAutoencoder(self.input_dim,self.middle_dim,self.latent_dim)

   def get_memory_size(self):
      return len(self.memory)
   
   def get_memory(self):
      return self.memory
   
   def remove_last_sample(self):
      self.memory.pop()

   def remove_last_latent_dnf(self):
      self.list_latent_scaled.pop()

   def remove_last_latent(self):
      self.list_latent.pop()
   
   def saveNN(self, name_folder, id_obj, model_name):
      name_dir = name_folder + str(id_obj) 
      n = name_dir + "/" + model_name + "_vae.pt"
      path = os.path.join(name_folder, str(id_obj)) 
      access = 0o755
      if os.path.isdir(path):
         if os.path.isfile(n):
            os.remove(n)
         self.vae.save(n)
      else:
         os.makedirs(path,access)  
         self.vae.save(n)

   def load_nn(self, name_folder, id_obj, model_name):
      name_dir = name_folder + str(id_obj) + "/" + model_name + "_vae.pt"
      self.vae.load(name_dir)

   def save_memory(self, name_folder, id_object, model_name):
      path = os.path.join(name_folder, str(id_object)) 
      n_mem = name_folder + str(id_object) + "/" + model_name + "_memory_samples.pkl"
      n_latent = name_folder + str(id_object) + "/" + model_name + "_latent_space.pkl"
      n_latent_scaled = name_folder + str(id_object) + "/" + model_name + "_latent_space_scaled.pkl"
      n_latent_bounds = name_folder + str(id_object) + "/" + model_name + "_bounds.pkl"
      n_latent_max_bounds = name_folder + str(id_object) + "/" + model_name + "_max_bounds.pkl"
      n_mtlatent = name_folder + str(id_object) + "/" + model_name + "_latent_space.npy"
      if os.path.isdir(path):
         if os.path.isfile(n_mem):
            os.remove(n_mem)
            os.remove(n_latent)
            os.remove(n_latent_scaled)
            os.remove(n_latent_bounds)
            os.remove(n_latent_max_bounds)
            os.remove(n_mtlatent)
         np.save(n_mtlatent,self.mt_field)
         filehandler = open(n_mem, 'wb')
         pickle.dump(self.memory, filehandler)
         filehandler_l = open(n_latent, 'wb')
         pickle.dump(self.list_latent, filehandler_l)
         filehandler_ls = open(n_latent_scaled, 'wb')
         pickle.dump(self.list_latent_scaled, filehandler_ls)
         filehandler_b = open(n_latent_bounds, 'wb')
         t = [self.bound_x,self.bound_y]
         pickle.dump(t, filehandler_b)
         filehandler_mb = open(n_latent_max_bounds, 'wb')
         mb = [self.max_bound_x,self.max_bound_y,self.bound_x,self.bound_y]
         pickle.dump(mb, filehandler_mb)

   def load_memory(self, name_folder, model_name):
      n_mem = name_folder + model_name + "_memory_samples.pkl"
      n_l = name_folder + model_name + "_latent_space.pkl"
      n_ls = name_folder + model_name + "_latent_space_scaled.pkl"
      n_b = name_folder + model_name + "_bounds.pkl"
      n_mb = name_folder + model_name + "_max_bounds.pkl"
      n_mtlatent = name_folder + model_name + "_latent_space.npy"
      self.mt_field = np.load(n_mtlatent)
      filehandler = open(n_mem, 'rb') 
      mem = pickle.load(filehandler)
      self.memory = mem
      filehandler_l = open(n_l, 'rb') 
      nl = pickle.load(filehandler_l)
      self.list_latent = nl
      filehandler_ls = open(n_ls, 'rb') 
      nls = pickle.load(filehandler_ls)
      self.list_latent_scaled = nls
      filehandler_b = open(n_b, 'rb') 
      nb = pickle.load(filehandler_b)
      t = nb
      self.bound_x = t[0]
      self.bound_y = t[1]
      filehandler_mb = open(n_mb, 'rb') 
      mb = pickle.load(filehandler_mb)
      tmb = mb
      self.max_bound_x = tmb[0]
      self.max_bound_y = tmb[1]
      self.bound_x = tmb[2]
      self.bound_y = tmb[3]


class Habituation(object):
   def __init__(self):
      rospy.init_node('habituation', anonymous=True)
      self.bridge = CvBridge()
      self.id_defined = False
      self.index_vae = -1
      self.id_object = 0
      self.prev_id_object = -1
      self.count_color = 0
      self.incoming_dmp = False
      self.incoming_outcome = False
      self.dmp = Dmp()
      self.outcome = Outcome()
      self.dmp_exploit = Dmp()
      self.outcome_exploit = Outcome()
      self.habit = []
      self.vae_action = []
      self.max_pitch = 1.5
      self.min_vx = -0.2
      self.max_vx = 0.2
      self.min_vy = -0.2
      self.max_vy = 0.2
      self.min_vpitch = -1.2
      self.max_vpitch = 1.2
      self.min_roll = -1.5
      self.max_roll = 1.5
      self.min_grasp = 0
      self.max_grasp = 1
      self.min_angle = -180
      self.max_angle = 180
      self.busy = False
      self.tot = 0
      self.colors = []
      #self.colors.append("orange")
      self.colors.append("red")
      #self.colors.append("purple")
      self.colors.append("blue")
      #self.colors.append("green")
      #self.colors.append("yellow")
      #self.colors.append("pink")
      #self.colors.append("cyan")
      #self.colors.append("brown")
      #self.colors.append("gray")
      self.time = 0
      self.first = True
      self.pub_latent_space_display_out = rospy.Publisher("/display/latent_space_out", LatentPos, queue_size=1, latch=True)
      self.pub_latent_space_display_act = rospy.Publisher("/display/latent_space_act", LatentPos, queue_size=1, latch=True)
      self.pub_ready = rospy.Publisher("/habituation/ready", Bool, queue_size=1, latch=True)
      self.pub_latent_space_dnf_out = rospy.Publisher("/habituation/outcome/latent_space_dnf", LatentNNDNF, queue_size=1, latch=True)
      self.pub_latent_space_dnf_act = rospy.Publisher("/habituation/action/latent_space_dnf", LatentNNDNF, queue_size=1, latch=True)
      self.pub_test_latent = rospy.Publisher("/display/latent_test", LatentPos, queue_size=1, latch=True)
      self.pub_eval_latent = rospy.Publisher("/habituation/evaluation", LatentNNDNF, queue_size=1, latch=True)
      self.pub_eval_perception = rospy.Publisher("/habituation/goal_perception", LatentNNDNF, queue_size=1, latch=True)
      self.pub_perception = rospy.Publisher("/habituation/test_perception", LatentNNDNF, queue_size=1, latch=True)
      self.pub_field = rospy.Publisher("/habituation/cedar/mt",Image, queue_size=1, latch=True)
      self.pub_field_action = rospy.Publisher("/habituation/cedar/mt_action",Image, queue_size=1, latch=True)
      self.pub_direct = rospy.Publisher("/motion_pincher/dmp_direct_exploration",Dmp, queue_size=1, latch=True)
      self.pub_busy = rospy.Publisher("/cluster_msg/busy",Bool, queue_size=1, latch=True)
      self.exploration_mode = rospy.get_param("exploration")
      self.folder_habituation = rospy.get_param("habituation_folder")
      rospy.Subscriber("/habituation/mt", Image, self.field_callback)
      rospy.Subscriber("/cog_learning/id_object", Int16, self.callback_id)
      rospy.Subscriber("/cluster_msg/sample_explore", SampleExplore, self.callback_sample_explore)
      rospy.Subscriber("/habituation/input_latent", LatentGoalDnf, self.callback_input_latent)
      rospy.Subscriber("/habituation/eval_perception", DmpOutcome, self.callback_eval)
      rospy.Subscriber("/cluster_msg/perception", DmpOutcome, self.callback_perception)
      rospy.Subscriber("/habituation/same_perception", Float64, self.callback_same_perception)
      self.load = rospy.get_param("load_vae")
      if(self.load):
         self.load_nn()

   def field_callback(self,msg):
      try:
         # Convert your ROS Image message to OpenCV2
         cv2_img = self.bridge.imgmsg_to_cv2(msg, "32FC1")
         if self.id_defined:
            self.habit[self.index_vae].set_mt_field(cv2_img)
      except CvBridgeError as e:
         print(e)

   #scale inputs from real values to [-1,1]
   def scale_data(self, data, min_, max_):
      n_x = np.array(data)
      n_x = n_x.reshape(-1,1)
      scaler_x = MinMaxScaler(feature_range=(-1,1))
      x_minmax = np.array([min_, max_])
      scaler_x.fit(x_minmax[:, np.newaxis])
      n_x = scaler_x.transform(n_x)
      n_x = n_x.reshape(1,-1)
      n_x = n_x.flatten()
            
      return n_x[0]
   
   #scale output of decoder VAE to real values
   def scale_data_to_real(self, data, min_, max_):
      n_x = np.array(data)
      n_x = n_x.reshape(-1,1)
      scaler_x = MinMaxScaler(feature_range=(min_,max_))
      x_minmax = np.array([-1, 1])
      scaler_x.fit(x_minmax[:, np.newaxis])
      n_x = scaler_x.transform(n_x)
      n_x = n_x.reshape(1,-1)
      n_x = n_x.flatten()
            
      return n_x[0]
   
   def callback_same_perception(self, msg):
      #receiving high value once in a while from cedar even if it's 0
      if msg.data > 0.9:
         self.tot += 1
      else:
         self.tot = 0
         self.busy = False
      if self.tot > 10 and not self.busy:
         self.busy = True
         print("SAME PERCEPTION")
         b = Bool()
         b.data = True
         self.pub_busy.publish(b)
         self.habit[self.index_vae].remove_last_sample()
         self.habit[self.index_vae].remove_last_latent()
         self.habit[self.index_vae].remove_last_latent_dnf()
         self.send_latent_space_outcome()
         #rospy.sleep(10.0)
         b.data = False
         self.pub_busy.publish(b)
   
   def send_latent_space_outcome(self):
      ls = self.habit[self.index_vae].get_latent_space_dnf()
      print("sending latent with ",len(ls))
      msg_latent = LatentNNDNF()
      msg_latent.max_x = self.habit[self.index_vae].get_bound_x()
      msg_latent.max_y = self.habit[self.index_vae].get_bound_y()
      for i in ls:
         lg = Goal() 
         lg.x = i[0]
         lg.y = i[1]
         lg.value = 1.0
         msg_latent.list_latent.append(lg)
      #print("Latent space DNF : ",msg_latent)
      self.pub_latent_space_dnf_out.publish(msg_latent)

   def send_latent_space_outcome_minus(self,msg):
      self.pub_latent_space_dnf_out.publish(msg)

   def send_latent_space_action(self):
      ls_a = self.vae_action[self.index_vae].get_latent_space_dnf()
      msg_latent = LatentNNDNF()
      msg_latent.max_x = self.vae_action[self.index_vae].get_bound_x()
      msg_latent.max_y = self.vae_action[self.index_vae].get_bound_y()
      for i in ls_a:
         lg = Goal() 
         lg.x = i[0]
         lg.y = i[1]
         lg.value = 1.0
         msg_latent.list_latent.append(lg)
      #print("Latent space DNF : ",msg_latent)
      self.pub_latent_space_dnf_act.publish(msg_latent)

   def send_latent_test(self, v):
      tmp = LatentGoalNN()
      tmp.latent_x = v[0]
      tmp.latent_y = v[1]
      self.pub_test_latent.publish(tmp)

   def send_eval_latent(self, msg):
      self.pub_eval_latent.publish(msg)

   def send_eval_perception(self, msg):
      self.pub_eval_perception.publish(msg)

   def send_perception(self, msg):
      self.pub_perception.publish(msg)

   def send_mt_field(self):
      img_field = self.habit[self.index_vae].get_mt_field()
      img_field_action = self.vae_action[self.index_vae].get_mt_field()
      img_msg = self.bridge.cv2_to_imgmsg(img_field, encoding="passthrough")
      act_msg = self.bridge.cv2_to_imgmsg(img_field_action, encoding="passthrough")
      self.pub_field.publish(img_msg)
      self.pub_field_action.publish(act_msg)

   def callback_sample_explore(self, msg):
      if self.first:
         self.time = rospy.get_time()
         self.first = False
      self.dmp.v_x = self.scale_data(msg.v_x,self.min_vx,self.max_vx)
      self.dmp.v_y = self.scale_data(msg.v_y,self.min_vy,self.max_vy)
      self.dmp.v_pitch = self.scale_data(msg.v_pitch,self.min_vpitch,self.max_vpitch)
      self.dmp.roll = self.scale_data(msg.roll,self.min_roll,self.max_roll)
      self.dmp.grasp = self.scale_data(msg.grasp,self.min_grasp,self.max_grasp)
      self.outcome.x = self.scale_data(msg.outcome_x, self.min_vx, self.max_vx)
      self.outcome.y = self.scale_data(msg.outcome_y, self.min_vy, self.max_vy)
      self.outcome.angle = self.scale_data(msg.outcome_angle, self.min_angle, self.max_angle)
      self.outcome.touch = self.scale_data(msg.outcome_touch, self.min_grasp, self.max_grasp)
      sample_outcome = [self.outcome.x,self.outcome.y,self.outcome.angle,self.outcome.touch]
      sample_action = [self.dmp.v_x,self.dmp.v_y,self.dmp.v_pitch,self.dmp.roll,self.dmp.grasp]
      tensor_outcome = torch.tensor(sample_outcome,dtype=torch.float)
      tensor_action = torch.tensor(sample_action,dtype=torch.float)
      self.add_to_memory(tensor_outcome,tensor_action)
      self.learn_new_latent()
      self.habit[self.index_vae].fill_latent()
      self.vae_action[self.index_vae].fill_latent()
      self.habit[self.index_vae].set_latent_dnf(self.exploration_mode)
      self.vae_action[self.index_vae].set_latent_dnf(self.exploration_mode)
      self.send_latent_space_action()
      #for display
      msg_act = self.vae_action[self.index_vae].plot_latent()
      self.pub_latent_space_display_act.publish(msg_act)
      if self.habit[self.index_vae].get_memory_size() > 0:
         lat_one, lat_minus, dis_one, dis_minus = self.habit[self.index_vae].get_latent_dnf_split(tensor_outcome)
         self.send_latent_space_outcome_minus(lat_minus)
         self.send_eval_latent(lat_one)
         #for display
         self.pub_latent_space_display_out.publish(dis_minus)
         self.pub_test_latent.publish(dis_one)
         rospy.sleep(4.0)
         self.send_latent_space_outcome()
         l = LatentNNDNF()
         self.send_eval_latent(l)
         #for display
         msg_out = self.habit[self.index_vae].plot_latent()
         self.pub_latent_space_display_out.publish(msg_out)
      self.pub_ready.publish(True)
      t = self.time - rospy.get_time()
      #rospy.sleep(5.0)
      #print("Time elapsed : ",t)
      

   def callback_input_latent(self, msg):
      print("got latent value for direct exploration : ",msg)
      x_dnf = msg.latent_x
      y_dnf = msg.latent_y      
      latent_value = self.habit[self.index_vae].set_dnf_to_latent([x_dnf,y_dnf],self.exploration_mode)
      print("latent : ",latent_value)
      t_latent = torch.tensor(latent_value,dtype=torch.float)
      output = self.habit[self.index_vae].reconstruct_latent(t_latent)
      dmp = Dmp()
      outcome = Outcome()
      dmp.v_x = self.scale_data_to_real(output[4],self.min_vx,self.max_vx)
      dmp.v_y = self.scale_data_to_real(output[5],self.min_vy,self.max_vy)
      dmp.v_pitch = self.scale_data_to_real(output[6],self.min_vpitch,self.max_vpitch)
      dmp.roll = self.scale_data_to_real(output[7],self.min_roll,self.max_roll)
      dmp.grasp = self.scale_data_to_real(output[8],self.min_grasp,self.max_grasp)
      outcome.x = self.scale_data_to_real(output[0], self.min_vx, self.max_vx)
      outcome.y = self.scale_data_to_real(output[1], self.min_vy, self.max_vy)
      outcome.angle = self.scale_data_to_real(output[2], self.min_angle, self.max_angle)
      outcome.touch = self.scale_data_to_real(output[3], self.min_grasp, self.max_grasp)
      self.pub_direct.publish(dmp)
      #print("DMP : ",dmp)
      #print("Outcome : ",outcome)

   def callback_eval(self,msg):
      dmp_out = DmpOutcome()
      dmp_out.v_x = self.scale_data(msg.v_x,self.min_vx,self.max_vx)
      dmp_out.v_y = self.scale_data(msg.v_y,self.min_vy,self.max_vy)
      dmp_out.v_pitch = self.scale_data(msg.v_pitch,self.min_vpitch,self.max_vpitch)
      dmp_out.roll = self.scale_data(msg.roll,self.min_roll,self.max_roll)
      dmp_out.grasp = self.scale_data(msg.grasp,self.min_grasp,self.max_grasp)
      dmp_out.x = self.scale_data(msg.x, self.min_vx, self.max_vx)
      dmp_out.y = self.scale_data(msg.y, self.min_vy, self.max_vy)
      dmp_out.angle = self.scale_data(msg.angle, self.min_angle, self.max_angle)
      dmp_out.touch = self.scale_data(msg.touch, self.min_grasp, self.max_grasp)
      sample = [dmp_out.x,dmp_out.y,dmp_out.angle,dmp_out.touch,dmp_out.v_x,dmp_out.v_y,dmp_out.v_pitch,dmp_out.roll,dmp_out.grasp]
      tensor_sample = torch.tensor(sample,dtype=torch.float)
      t = self.habit[self.index_vae].get_memory()
      print("memory : ",t)
      print("tensor sample : ",tensor_sample)
      z = self.habit[self.index_vae].get_sample_latent(tensor_sample)
      print("TESTING new sample...")
      msg = self.habit[self.index_vae].get_eval_latent_to_dnf(z,self.exploration_mode)
      print(msg)
      self.send_eval_perception(msg)
      #self.send_latent_test(z)
      #rospy.sleep(1.0)
      #l = LatentNNDNF()
      #self.send_eval_perception(l)

   def callback_id(self, msg):
      if self.prev_id_object != self.id_object and msg.data != -1:
         self.id_object = msg.data
         found = False
         for i in range(0,len(self.habit)):
            tmp = self.habit[i].get_id()
            if tmp == self.id_object:
                  self.index_vae = i
                  found = True
                  print("found MT")
                  self.send_mt_field()
         if not found:
            tmp_habbit = VariationalAE(self.id_object,4,3,2)
            self.habit.append(tmp_habbit)
            tmp_action = VariationalAE(self.id_object,5,4,2)
            self.vae_action.append(tmp_action)
            self.index_vae = len(self.habit) - 1
            print("Creation new VAE : ",self.id_object)
            blank_mt = np.zeros((100,100,1), np.float32)
            self.habit[self.index_vae].set_mt_field(blank_mt)
            self.vae_action[self.index_vae].set_mt_field(blank_mt)
            self.send_mt_field()
         self.prev_id_object = self.id_object
         self.id_defined = True

   def callback_perception(self, msg):
      dmp_out = DmpOutcome()
      dmp_out.v_x = self.scale_data(msg.v_x,self.min_vx,self.max_vx)
      dmp_out.v_y = self.scale_data(msg.v_y,self.min_vy,self.max_vy)
      dmp_out.v_pitch = self.scale_data(msg.v_pitch,self.min_vpitch,self.max_vpitch)
      dmp_out.roll = self.scale_data(msg.roll,self.min_roll,self.max_roll)
      dmp_out.grasp = self.scale_data(msg.grasp,self.min_grasp,self.max_grasp)
      dmp_out.x = self.scale_data(msg.x, self.min_vx, self.max_vx)
      dmp_out.y = self.scale_data(msg.y, self.min_vy, self.max_vy)
      dmp_out.angle = self.scale_data(msg.angle, self.min_angle, self.max_angle)
      dmp_out.touch = self.scale_data(msg.touch, self.min_grasp, self.max_grasp)
      sample = [dmp_out.x,dmp_out.y,dmp_out.angle,dmp_out.touch,dmp_out.v_x,dmp_out.v_y,dmp_out.v_pitch,dmp_out.roll,dmp_out.grasp]
      tensor_sample = torch.tensor(sample,dtype=torch.float)
      t = self.habit[self.index_vae].get_memory()
      #print("memory : ",t)
      #print("tensor sample : ",tensor_sample)
      z = self.habit[self.index_vae].get_sample_latent(tensor_sample)
      #print("TESTING new sample...")
      msg = self.habit[self.index_vae].get_eval_latent_to_dnf(z,self.exploration_mode)
      #print(msg)
      self.send_perception(msg)
      #self.send_latent_test(z)
      rospy.sleep(1.0)
      l = LatentNNDNF()
      self.send_perception(l)

   def learn_new_latent(self):
      print("TRAINING VAE...")
      torch.manual_seed(24)
      self.habit[self.index_vae].reset_model()
      self.vae_action[self.index_vae].reset_model()
      self.time = rospy.get_time()
      self.habit[self.index_vae].train()
      t = self.time - rospy.get_time()
      print("Training time OUTCOME: ",t)
      self.time = rospy.get_time()
      self.vae_action[self.index_vae].train()
      t = self.time - rospy.get_time()
      print("Training time ACTION: ",t)
      self.save_nn()
      self.save_memory()
      self.incoming_dmp = False
      self.incoming_outcome = False
      print("finished training VAE")
      

   def add_to_memory(self, sample_out, sample_act):
      self.habit[self.index_vae].add_to_memory(sample_out)
      self.vae_action[self.index_vae].add_to_memory(sample_act)
      self.outcome = Outcome()
      self.dmp = Dmp()
      self.count_color += 1

   def test_reconstruct(self):
      self.habit[self.index_vae].test_reconstruct()

   def save_nn(self):
      self.habit[self.index_vae].saveNN(self.folder_habituation, self.id_object,"outcome")
      self.vae_action[self.index_vae].saveNN(self.folder_habituation, self.id_object,"action")

   def save_memory(self):
      self.habit[self.index_vae].save_memory(self.folder_habituation, self.id_object, "outcome")
      self.vae_action[self.index_vae].save_memory(self.folder_habituation, self.id_object, "action")

   def load_nn(self):
      list_dir = os.listdir(self.folder_habituation)
      for i in range(0,len(list_dir)):
         tmp_habit = VariationalAE(i,4,3,2)
         tmp_act = VariationalAE(i,5,4,2)
         n_f = self.folder_habituation + str(i) + "/"
         tmp_habit.load_nn(self.folder_habituation,i,"oucome")
         tmp_habit.load_memory(n_f,"outcome")
         tmp_act.load_nn(self.folder_habituation,i,"action")
         tmp_act.load_memory(n_f,"action")
         self.habit.append(tmp_habit)
         self.vae_action.append(tmp_act)
      for i in self.habit:
         print("VAE : ",i.get_id())
         print("memory : ",len(i.memory))
         print("latent space : ",i.get_latent_space())
         print("latent space scaled : ",i.get_latent_space_dnf())


if __name__ == "__main__":
   habituation = Habituation()
   rospy.spin()