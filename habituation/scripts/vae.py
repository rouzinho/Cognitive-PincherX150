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
import shutil
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
import csv

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
      torch.manual_seed(3407)
      self.vae = VariationalAutoencoder(self.input_dim,self.middle_dim,self.latent_dim)
      self.memory = []
      self.mt_field = np.zeros((100,100,1), np.float32)
      self.id = id_object
      self.list_latent = []
      self.list_latent_scaled = []
      #self.scale_factor = rospy.get_param("scale_factor")
      self.scale_factor = 100
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
      #print("latent space : ",len(self.list_latent))

   def set_latent_dnf(self, exploration):
      ext_x, ext_y = self.get_latent_extremes(self.list_latent)
      self.list_latent_scaled = []
      if exploration == "static":
         self.bound_x = 100
         self.bound_y = 100
         if len(self.list_latent) > 1:
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
         dist_x = abs(ext_x[0] - ext_x[1])
         dist_y = abs(ext_y[0] - ext_y[1])
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
         if len(self.list_latent) > 1:
            for i in self.list_latent:
               x = self.scale_latent_to_dnf_dynamic(i[0],ext_x[0],ext_x[1],padding_x,self.bound_x-padding_x)
               y = self.scale_latent_to_dnf_dynamic(i[1],ext_y[0],ext_y[1],padding_y,self.bound_y-padding_y)
               self.list_latent_scaled.append([round(x),round(y)])
         else:
            self.list_latent_scaled.append([5,5])
            self.bound_x = round(10)
            self.bound_y = round(10)

   def get_value_dnf(self,z,exploration):
      ext_x, ext_y = self.get_latent_extremes(self.list_latent)
      x = 0
      y = 0
      if exploration == "static":
         min_latent_x = -1
         max_latent_x = 1
         min_latent_y = -1
         max_latent_y = 1
         if (ext_x[0] < -1 or ext_x[1] > 1) or (ext_y[0] < -1 or ext_y[1] > 1):
            min_latent_x = ext_x[0]
            max_latent_x = ext_x[1]
            min_latent_y = ext_y[0]
            max_latent_y = ext_y[1]
         x = self.scale_latent_to_dnf_static(z[0],min_latent_x,max_latent_x)
         y = self.scale_latent_to_dnf_static(z[1],min_latent_y,max_latent_y)
      else:
         padding_x = round(self.max_bound_x * 0.1)
         padding_y = round(self.max_bound_y * 0.1)
         x = self.scale_latent_to_dnf_dynamic(z[0],ext_x[0],ext_x[1],padding_x,self.bound_x-padding_x)
         y = self.scale_latent_to_dnf_dynamic(z[1],ext_y[0],ext_y[1],padding_y,self.bound_y-padding_y)

      return [x,y]

   def get_latent_dnf_split(self, sample):
      msg_display_one = LatentPos()
      msg_display_minus = LatentPos()
      new_latent_single = LatentNNDNF()
      new_latent_single.max_x = self.bound_x
      new_latent_single.max_y = self.bound_y
      new_latent_minus_one = LatentNNDNF()
      new_latent_minus_one.max_x = self.bound_x
      new_latent_minus_one.max_y = self.bound_y
      found = False
      self.vae.eval()
      z, z_log, recon = self.vae.encoder(sample)
      z = z.to('cpu').detach().numpy()
      #print("latent : ",self.list_latent)
      #print("DNF : ",self.list_latent_scaled)
      for (i,j) in zip(self.list_latent,self.list_latent_scaled):
         if abs(abs(i[0]) - abs(z[0])) < 0.01 and abs(abs(i[1]) - abs(z[1])) < 0.01 and not found:
            #print("found latent value")
            g = Goal()
            g.x = j[0]
            g.y = j[1]
            g.value = 1.0
            new_latent_single.list_latent.append(g)
            msg_display_one.x.append(i[0])
            msg_display_one.y.append(i[1])
            found = True
         else:
            #print("latent not found")
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
   
   def get_eval(self,peak):
      l = LatentNNDNF()
      l.max_x = self.bound_x
      l.max_y = self.bound_y
      g = Goal()
      g.x = peak[0]
      g.y = peak[1]
      g.value = 1.0
      l.list_latent.append(g)

      return l
   
   def set_scale_factor(self,val):
      self.scale_factor = val

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
   
   def search_dnf_value(self, sample):
      z = self.forward_encoder_outcome(sample)
      min_dist = 2
      best = -1
      j = 0
      for i in self.list_latent:
         dist = math.sqrt(pow(z[0]-i[0],2)+pow(z[1]-i[1],2))
         if dist <= min_dist:
            best = j
            min_dist = dist
         j += 1
      peak = self.list_latent_scaled[best]

      return peak
      
   def forward_encoder_outcome(self,sample):
      self.vae.eval()
      z, z_log, recon = self.vae.encoder(sample)
      z = z.to('cpu').detach().numpy()

      return z

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

   #for data analysis
   def merge_samples(self,data1, data2):
      self.memory = copy.deepcopy(data1) + copy.deepcopy(data2)
      print(self.memory)

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
      self.vae.train()
      kl_weight = 0.8
      opt = torch.optim.Adam(self.vae.parameters(), lr=0.001) #0.01
      train_loss = 0.0
      last_loss = 10
      stop = False
      i = 0
      min_err = 1.0
      err_rec = 1.0
      err_kld = 5.0
      min_kld = 5.0
      total_rec = 50
      total_kld = 50
      mem = copy.deepcopy(self.memory)
      while not stop:
         #random.shuffle(mem)
         for sample in mem:
            s = sample.to(device) # GPU
            opt.zero_grad()
            pred = self.vae(s)
            loss = self.vae_loss(pred, s)
            #err_kld = self.kld_loss.item()
            #err_rec = self.recon_loss.item()
            #total_rec += self.recon_loss.item()
            #total_kld += self.kld_loss.item()
            #self.kld_loss = self.kld_loss.item()
            #self.recon_loss = self.recon_loss.item()
            #print("loss reconstruct : ",self.recon_loss.item())
            #print("loss KL : ",self.kld_loss.item())
            #print("loss total : ",loss)
            loss.backward()
            opt.step()
            #if err_rec < min_err and err_kld < min_kld:
            #   min_err = err_rec
            #   min_kld = err_kld
            #   print("min reconstructed : ",min_err)
            #   print("loss KL : ",min_kld)
            #if min_kld < 0.04 and min_err < 0.001: #0.0005
            #   print("training... i : ",i)
            #   print("min reconstructed : ",min_err)
            #   print("loss KL : ",self.kld_loss.item())
            if self.kld_loss < 0.05 and self.recon_loss < 0.0005:
               stop = True
         i += 1
         #print("loss rec : ",total_rec)
         #print("loss kld : ",total_kld)
         #if total_kld < 7 and total_rec < 5.7:
         #   print("loss rec : ",total_rec)
         #   print("loss kld : ",total_kld)
         #   stop = True
         total_rec = 0
         total_kld = 0
         #if i > 20000:
         #   stop = True
         if i % 10000 == 0:
            print("Step training...")

   def get_list_latent(self, list_sample):
      self.vae.eval()
      x = None
      y = None
      j = 0
      for i in list_sample:
         z, z_log, recon = self.vae.encoder(i)
         z = z.to('cpu').detach().numpy()
         if j == 0:
            x = np.array([z[0]])
            y = np.array([z[1]])
         else:
            x = np.append(x,[z[0]])
            y = np.append(y,[z[1]])
         j+=1

      return x, y
   
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
         #print("write MT")
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
      self.min_vx = -0.18
      self.max_vx = 0.18
      self.min_vy = -0.18
      self.max_vy = 0.18
      self.min_vpitch = 0.1
      self.max_vpitch = 1.5
      self.min_roll = -1.5
      self.max_roll = 1.5
      self.min_grasp = 0
      self.max_grasp = 1
      self.min_angle = -180
      self.max_angle = 180
      self.busy_out = False
      self.busy_act = False
      self.total_out = 0
      self.total_act = 0
      self.total_rnd = 0
      self.total_direct = 0
      self.total_exp = 0
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
      self.rnd_exploration = False
      self.direct_exploration = False
      self.exploitation = False
      self.current_exploration = SampleExplore()
      self.img_outcome = np.zeros((100,100,1), np.float32)
      self.img_action = np.zeros((100,100,1), np.float32)
      self.time = 0
      self.first = True
      self.lock = False
      self.change = False
      self.goal_perception = Goal()
      self.new_perception = Goal()
      self.pub_latent_space_display_out = rospy.Publisher("/display/latent_space_out", LatentPos, queue_size=1, latch=True)
      self.pub_latent_space_display_act = rospy.Publisher("/display/latent_space_act", LatentPos, queue_size=1, latch=True)
      self.pub_ready = rospy.Publisher("/habituation/ready", Bool, queue_size=1, latch=True)
      self.pub_latent_space_dnf_out = rospy.Publisher("/habituation/outcome/latent_space_dnf", LatentNNDNF, queue_size=1, latch=True)
      self.pub_latent_space_dnf_act = rospy.Publisher("/habituation/action/latent_space_dnf", LatentNNDNF, queue_size=1, latch=True)
      self.pub_test_latent = rospy.Publisher("/display/latent_test", LatentPos, queue_size=1, latch=True)
      self.pub_eval_outcome = rospy.Publisher("/habituation/outcome/evaluation", LatentNNDNF, queue_size=1, latch=True)
      self.pub_eval_action = rospy.Publisher("/habituation/action/evaluation", LatentNNDNF, queue_size=1, latch=True)
      self.pub_eval_perception = rospy.Publisher("/habituation/existing_goal_perception", LatentNNDNF, queue_size=1, latch=True)
      self.pub_perception = rospy.Publisher("/habituation/perception_new_goal", LatentNNDNF, queue_size=1, latch=True)
      self.pub_field = rospy.Publisher("/habituation/cedar/mt_outcome",Image, queue_size=1, latch=True)
      self.pub_field_action = rospy.Publisher("/habituation/cedar/mt_action",Image, queue_size=1, latch=True)
      self.pub_direct = rospy.Publisher("/motion_pincher/dmp_direct_exploration",Dmp, queue_size=1, latch=True)
      self.pub_busy_out = rospy.Publisher("/cluster_msg/vae/busy_out",Bool, queue_size=1, latch=True)
      self.pub_busy_act = rospy.Publisher("/cluster_msg/vae/busy_act",Bool, queue_size=1, latch=True)
      self.pub_reward = rospy.Publisher("/cog_learning/action_reward",Float64, queue_size=1, latch=True)
      self.exploration_mode = rospy.get_param("exploration")
      self.folder_habituation = rospy.get_param("habituation_folder")
      self.folder_exploration = "/home/altair/PhD/Codes/Experiment-IMVAE/datas/production/"
      rospy.Subscriber("/habituation/outcome/mt", Image, self.field_callback)
      rospy.Subscriber("/habituation/action/mt", Image, self.field_action_callback)
      rospy.Subscriber("/cog_learning/id_object", Int16, self.callback_id)
      rospy.Subscriber("/cluster_msg/sample_explore", SampleExplore, self.callback_sample_explore)
      rospy.Subscriber("/habituation/input_latent", Goal, self.callback_input_latent)
      rospy.Subscriber("/habituation/existing_perception", Outcome, self.callback_eval)
      rospy.Subscriber("/habituation/new_perception", Outcome, self.callback_perception)
      #rospy.Subscriber("/cog_learning/outcome/not_learning", Float64, self.callback_same_perception)
      #rospy.Subscriber("/cog_learning/action/not_learning", Float64, self.callback_same_action)
      rospy.Subscriber("/cog_learning/rnd_exploration", Float64, self.callback_rnd_exploration)
      rospy.Subscriber("/cog_learning/direct_exploration", Float64, self.callback_direct_exploration)
      rospy.Subscriber("/cog_learning/exploitation", Float64, self.callback_exploitation)
      rospy.Subscriber("/recording/exploration", Bool, self.callback_recording)
      rospy.Subscriber("/habituation/save_vae_out", Bool, self.callback_save_outcome)
      rospy.Subscriber("/habituation/save_vae_action", Bool, self.callback_save_action)
      self.load = rospy.get_param("load_vae")
      self.sf = rospy.get_param("scale_factor")

      if(self.load):
         self.load_nn_action()
         self.load_nn_outcome()
         
      else:
         self.rm_samples()
         self.create_exploration_data()

   def field_callback(self,msg):
      try:
         # Convert your ROS Image message to OpenCV2
         self.img_outcome = self.bridge.imgmsg_to_cv2(msg, "32FC1")
      except CvBridgeError as e:
         print(e)

   def field_action_callback(self,msg):
      try:
         # Convert your ROS Image message to OpenCV2
         self.img_action = self.bridge.imgmsg_to_cv2(msg, "32FC1")
      except CvBridgeError as e:
         print(e)

   def callback_rnd_exploration(self, msg):
      if msg.data > 0.9:
         self.total_rnd += 1
      else:
         self.total_rnd = 0
         self.rnd_exploration = False
      if self.total_rnd > 10:
         self.rnd_exploration = True
         if self.change:
            l = LatentNNDNF()
            self.send_eval_perception(l)
            self.change = False

   def callback_direct_exploration(self, msg):
      if msg.data > 0.9:
         self.total_direct += 1
      else:
         self.total_direct = 0
         self.direct_exploration = False
      if self.total_direct > 10:
         self.direct_exploration = True
         if self.change:
            l = LatentNNDNF()
            self.send_eval_perception(l)
            self.change = False

   def callback_exploitation(self, msg):
      if msg.data > 0.9:
         self.total_exp += 1
      else:
         self.total_exp = 0
         self.exploitation = False
      if self.total_exp > 10:
         self.exploitation = True
         if not self.change:
            self.change = True

   def rm_samples(self):
      n_rec = "/home/altair/PhD/Codes/Experiment-IMVAE/datas/production/records/"
      n_hab = "/home/altair/PhD/Codes/Experiment-IMVAE/datas/production/habituation/0/"
      n_nnga = "/home/altair/PhD/Codes/Experiment-IMVAE/datas/production/nn_ga/0/"
      n_exp = self.folder_exploration + "exploration_data.csv"
      if os.path.isfile(n_rec+"peaks.pkl"):
         os.remove(n_rec+"peaks.pkl")
      if os.path.isfile(n_rec+"time.pkl"):
         os.remove(n_rec+"time.pkl")
      if os.path.isfile(n_rec+"inv_peaks.pkl"):
         os.remove(n_rec+"inv_peaks.pkl")
      if os.path.isdir(n_rec+"-1/"):
         shutil.rmtree(n_rec+"-1/")
      if os.path.isdir(n_rec+"0/"):
         shutil.rmtree(n_rec+"0/")
      if os.path.isdir(n_hab):
         shutil.rmtree(n_hab)
      if os.path.isdir(n_nnga):
         shutil.rmtree(n_nnga)
      if os.path.isfile(n_exp):
         os.remove(n_exp)

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
      if msg.data > 0.9 and (self.rnd_exploration or self.direct_exploration):
         self.total_out += 1
      else:
         self.total_out = 0
         self.busy_out = False
      if self.total_out > 10 and not self.busy_out:
         self.busy_out = True
         print("SAME PERCEPTION")
         b = Bool()
         b.data = True
         self.pub_busy_out.publish(b)
         self.habit[self.index_vae].remove_last_sample()
         self.learn_new_latent_outcome()
         self.habit[self.index_vae].fill_latent()
         self.habit[self.index_vae].set_latent_dnf(self.exploration_mode)
         self.send_latent_space_outcome()
         #rospy.sleep(10.0)
         b.data = False
         self.pub_busy_out.publish(b)

   def callback_save_outcome(self, msg):
      if msg.data:
         print("SAVING VAE OUT...")
         self.save_nn_outcome()
         self.save_memory_outcome()
      else:
         print("KNOWN PERCEPTION, RELOADING VAE OUT...")
         self.busy_out = True
         b = Bool()
         b.data = True
         self.pub_busy_out.publish(b)
         self.load_nn_outcome()
         self.habit[self.index_vae].fill_latent()
         self.habit[self.index_vae].set_latent_dnf(self.exploration_mode)
         self.send_latent_space_outcome()
         b.data = False
         self.pub_busy_out.publish(b)
         self.busy_out = False

   def callback_same_action(self, msg):
      #receiving high value once in a while from cedar even if it's 0
      if msg.data > 0.9 and (self.rnd_exploration or self.direct_exploration):
         self.total_act += 1
      else:
         self.total_act = 0
         self.busy_act = False
      if self.total_act > 10 and not self.busy_act:
         self.busy_act = True
         print("SAME ACTION")
         b = Bool()
         b.data = True
         self.pub_busy_act.publish(b)
         self.vae_action[self.index_vae].remove_last_sample()
         self.learn_new_latent_action()
         self.vae_action[self.index_vae].fill_latent()
         self.vae_action[self.index_vae].set_latent_dnf(self.exploration_mode)
         self.send_latent_space_action()
         #rospy.sleep(10.0)
         b.data = False
         self.pub_busy_act.publish(b)

   def callback_save_action(self, msg):
      if msg.data:
         print("SAVING VAE ACTION...")
         self.save_nn_action()
         self.save_memory_action()
      else:
         print("KNOWN ACTION, RELOADING VAE ACTION...")
         self.busy_act = True
         b = Bool()
         b.data = True
         self.pub_busy_act.publish(b)
         self.load_nn_action()
         self.vae_action[self.index_vae].fill_latent()
         self.vae_action[self.index_vae].set_latent_dnf(self.exploration_mode)
         self.send_latent_space_action()
         b.data = False
         self.pub_busy_act.publish(b)
         self.busy_act = False

   def callback_recording(self, msg):
      if msg.data:
         self.write_exploration_data()

   def write_exploration_data(self):
        name_f = self.folder_exploration + "exploration_data.csv"
        data_exp = [self.current_exploration.outcome_x,self.current_exploration.outcome_y,self.current_exploration.outcome_angle,
                    self.current_exploration.outcome_touch,self.current_exploration.v_x,self.current_exploration.v_y,
                    self.current_exploration.v_pitch,self.current_exploration.roll,self.current_exploration.grasp,
                    self.current_exploration.rnd_exploration,self.current_exploration.direct_exploration]
        with open(name_f, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data_exp)

   def create_exploration_data(self):
      name_f = self.folder_exploration + "exploration_data.csv"
      line = ["out_x","out_y","out_angle","out_touch","vx","vy","vpitch","roll","grasp","rnd","direct"]
      with open(name_f, 'w', newline='') as csvfile:
         writer = csv.writer(csvfile)
         writer.writerow(line)
   
   def send_latent_space_outcome(self):
      ls = self.habit[self.index_vae].get_latent_space_dnf()
      msg_latent = LatentNNDNF()
      msg_latent.max_x = self.habit[self.index_vae].get_bound_x()
      msg_latent.max_y = self.habit[self.index_vae].get_bound_y()
      for i in ls:
         lg = Goal() 
         lg.x = i[0]
         lg.y = i[1]
         lg.value = 1.0
         msg_latent.list_latent.append(lg)
      #print("Latent space outcome : ",msg_latent.list_latent)
      self.pub_latent_space_dnf_out.publish(msg_latent)

   def send_latent_space_outcome_minus(self,msg):
      self.pub_latent_space_dnf_out.publish(msg)

   def send_latent_space_action_minus(self,msg):
      self.pub_latent_space_dnf_act.publish(msg)

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

   def send_eval_outcome(self, msg):
      self.pub_eval_outcome.publish(msg)

   def send_eval_action(self, msg):
      self.pub_eval_action.publish(msg)

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
      self.fill_current_exploration_data(msg)
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
      self.learn_new_latent_outcome()
      self.learn_new_latent_action()
      #for display
      msg_act = self.vae_action[self.index_vae].plot_latent()
      self.pub_latent_space_display_act.publish(msg_act)
      if self.habit[self.index_vae].get_memory_size() > 1:
         #for perception
         lat_one, lat_minus, dis_one, dis_minus = self.habit[self.index_vae].get_latent_dnf_split(tensor_outcome)
         self.send_latent_space_outcome_minus(lat_minus)
         self.send_eval_outcome(lat_one)
         #for actions
         lat_one_act, lat_minus_act, dis_one_act, dis_minus_act = self.vae_action[self.index_vae].get_latent_dnf_split(tensor_action)
         #print("eval action : ",lat_one_act)
         self.send_latent_space_action_minus(lat_minus_act)
         self.send_eval_action(lat_one_act)
         #for display
         self.pub_latent_space_display_out.publish(dis_minus)
         self.pub_test_latent.publish(dis_one)
         rospy.sleep(2.0)
         #lock updates if same outcome or same action was detected
         #send full latent space and empty the evaluation
         l = LatentNNDNF()
         self.send_eval_outcome(l)
         self.send_eval_action(l)
         rospy.sleep(0.5)
         if not self.busy_out:
            self.send_latent_space_outcome()
         if not self.busy_act:
            self.send_latent_space_action()
         rospy.sleep(1.0) 
         #self.save_nn()
         #self.save_memory()
         #for display
         msg_out = self.habit[self.index_vae].plot_latent()
         self.pub_latent_space_display_out.publish(msg_out)
      else:
         #For the first value to trigger the learning through DFT
         empty_latent = LatentNNDNF()
         empty_latent.max_x = self.habit[self.index_vae].get_bound_x()
         empty_latent.max_y = self.habit[self.index_vae].get_bound_y()
         new_latent_single = LatentNNDNF()
         new_latent_single.max_x = self.habit[self.index_vae].get_bound_x()
         new_latent_single.max_y = self.habit[self.index_vae].get_bound_y()
         t = self.habit[self.index_vae].get_latent_space_dnf()
         g = Goal()
         g.x = t[0][0]
         g.y = t[0][1]
         g.value = 1.0
         new_latent_single.list_latent.append(g)
         self.send_latent_space_outcome_minus(empty_latent)
         self.send_eval_outcome(new_latent_single)
         self.send_latent_space_action_minus(empty_latent)
         self.send_eval_action(new_latent_single)
         rospy.sleep(2.0)
         l = LatentNNDNF()
         self.send_eval_outcome(l)
         self.send_eval_action(l)
         rospy.sleep(0.5)
         if not self.busy_out:
            self.send_latent_space_outcome()
            self.save_nn_outcome()
            self.save_nn_action()
            self.save_memory_outcome()
            self.save_memory_action()
         if not self.busy_act:
            self.send_latent_space_action()
            self.save_nn_outcome()
            self.save_nn_action()
            self.save_memory_outcome()
            self.save_memory_action()
      self.pub_ready.publish(True)
      t = self.time - rospy.get_time()
      #rospy.sleep(5.0)
      #print("Time elapsed : ",t)

   def fill_current_exploration_data(self,msg):
      self.current_exploration.outcome_x = msg.outcome_x
      self.current_exploration.outcome_y = msg.outcome_y
      self.current_exploration.outcome_angle = msg.outcome_angle
      self.current_exploration.outcome_touch = msg.outcome_touch
      self.current_exploration.v_x = msg.v_x
      self.current_exploration.v_y = msg.v_y
      self.current_exploration.v_pitch = msg.v_pitch
      self.current_exploration.roll = msg.roll
      self.current_exploration.grasp = msg.grasp
      self.current_exploration.rnd_exploration = msg.rnd_exploration
      self.current_exploration.direct_exploration = msg.direct_exploration

   def callback_input_latent(self, msg):
      print("got latent value for direct exploration : ",msg)
      x_dnf = msg.x
      y_dnf = msg.y      
      latent_value = self.vae_action[self.index_vae].set_dnf_to_latent([x_dnf,y_dnf],self.exploration_mode)
      print("latent : ",latent_value)
      t_latent = torch.tensor(latent_value,dtype=torch.float)
      output = self.vae_action[self.index_vae].reconstruct_latent(t_latent)
      dmp = Dmp()
      dmp.v_x = self.scale_data_to_real(output[0],self.min_vx,self.max_vx)
      dmp.v_y = self.scale_data_to_real(output[1],self.min_vy,self.max_vy)
      dmp.v_pitch = self.scale_data_to_real(output[2],self.min_vpitch,self.max_vpitch)
      dmp.roll = self.scale_data_to_real(output[3],self.min_roll,self.max_roll)
      dmp.grasp = self.scale_data_to_real(output[4],self.min_grasp,self.max_grasp)
      print(dmp)
      self.pub_direct.publish(dmp)
      #print("DMP : ",dmp)
      #print("Outcome : ",outcome)

   #send the expected goal perception
   def callback_eval(self,msg):
      x = self.scale_data(msg.x, self.min_vx, self.max_vx)
      y = self.scale_data(msg.y, self.min_vy, self.max_vy)
      angle = self.scale_data(msg.angle, self.min_angle, self.max_angle)
      touch = self.scale_data(msg.touch, self.min_grasp, self.max_grasp)
      sample = [x,y,angle,touch]
      #tmp = [msg.x,msg.y,msg.angle,msg.touch]
      #print("received raw sample : ",tmp)
      #print("received sample : ",sample)
      tensor_sample = torch.tensor(sample,dtype=torch.float)
      peak = self.habit[self.index_vae].search_dnf_value(tensor_sample)
      #print("peak found ",peak)
      #print("peak reconstructed : ",rec)
      self.goal_perception.x = peak[0]
      self.goal_perception.y = peak[1]
      msg = self.habit[self.index_vae].get_eval(peak)
      self.send_eval_perception(msg)

   #send the fresh perception
   def callback_perception(self, msg):
      #print("got perception : ")
      x = self.scale_data(msg.x, self.min_vx, self.max_vx)
      y = self.scale_data(msg.y, self.min_vy, self.max_vy)
      angle = self.scale_data(msg.angle, self.min_angle, self.max_angle)
      touch = self.scale_data(msg.touch, self.min_grasp, self.max_grasp)
      sample = [x,y,angle,touch]
      tensor_sample = torch.tensor(sample,dtype=torch.float)
      z = self.habit[self.index_vae].forward_encoder_outcome(tensor_sample)
      p = self.habit[self.index_vae].get_value_dnf(z,self.exploration_mode)
      peak = [round(p[0]),round(p[1])]
      self.new_perception.x = peak[0]
      self.new_perception.y = peak[1]
      msg = self.habit[self.index_vae].get_eval(peak)
      #print("got perception : ",msg)
      self.send_distance()
      self.send_perception(msg)
      rospy.sleep(2.0)
      l = LatentNNDNF()
      self.send_perception(l)
      
   def send_distance(self):
      dist = math.sqrt(pow((self.goal_perception.x/100) - (self.new_perception.x/100),2) + pow((self.goal_perception.y/100) - (self.new_perception.y/100),2))
      r = 1 - dist
      f = Float64()
      f.data = r
      self.pub_reward.publish(f) 

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
                  #self.send_mt_field()
                  self.habit[self.index_vae].set_latent_dnf(self.exploration_mode)
                  self.send_latent_space_outcome()
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
            self.habit[self.index_vae].set_scale_factor(self.sf)
            self.vae_action[self.index_vae].set_scale_factor(self.sf)
            self.send_mt_field()
         self.prev_id_object = self.id_object
         self.id_defined = True

   def learn_new_latent_outcome(self):
      print("training VAE outcome...")
      torch.manual_seed(3407)
      self.habit[self.index_vae].reset_model()
      self.time = rospy.get_time()
      self.habit[self.index_vae].train()
      t = self.time - rospy.get_time()
      print("Training time OUTCOME: ",t)
      print("finished training VAE OUT")
      self.habit[self.index_vae].fill_latent()
      self.habit[self.index_vae].set_latent_dnf(self.exploration_mode)
      #self.save_nn()
      #self.save_memory()
      self.incoming_dmp = False
      self.incoming_outcome = False
      

   def learn_new_latent_action(self):
      print("training VAE action...")
      torch.manual_seed(3407)
      self.vae_action[self.index_vae].reset_model()
      self.time = rospy.get_time()
      self.vae_action[self.index_vae].train()
      t = self.time - rospy.get_time()
      print("Training time ACTION: ",t)
      print("finished training VAE ACTION")
      self.vae_action[self.index_vae].fill_latent()
      self.vae_action[self.index_vae].set_latent_dnf(self.exploration_mode)
      #self.save_nn()
      #self.save_memory()
      self.incoming_dmp = False
      self.incoming_outcome = False
      

   def add_to_memory(self, sample_out, sample_act):
      self.habit[self.index_vae].add_to_memory(sample_out)
      self.vae_action[self.index_vae].add_to_memory(sample_act)
      self.outcome = Outcome()
      self.dmp = Dmp()
      self.count_color += 1

   def test_reconstruct(self):
      self.habit[self.index_vae].test_reconstruct()

   def save_nn_outcome(self):
      self.habit[self.index_vae].saveNN(self.folder_habituation, self.id_object,"outcome")

   def save_nn_action(self):
      self.vae_action[self.index_vae].saveNN(self.folder_habituation, self.id_object,"action")

   def save_memory_outcome(self):
      self.habit[self.index_vae].set_mt_field(self.img_outcome)
      self.habit[self.index_vae].save_memory(self.folder_habituation, self.id_object, "outcome")

   def save_memory_action(self):
      self.vae_action[self.index_vae].set_mt_field(self.img_action)
      self.vae_action[self.index_vae].save_memory(self.folder_habituation, self.id_object, "action")

   def load_nn_outcome(self):
      list_dir = os.listdir(self.folder_habituation)
      self.habit = []
      for i in range(0,len(list_dir)):
         tmp_habit = VariationalAE(i,4,3,2)
         #tmp_act = VariationalAE(i,5,4,2)
         n_f = self.folder_habituation + str(i) + "/"
         tmp_habit.load_nn(self.folder_habituation,i,"outcome")
         tmp_habit.load_memory(n_f,"outcome")
         tmp_habit.set_scale_factor(self.sf)
         #tmp_act.load_nn(self.folder_habituation,i,"action")
         #tmp_act.load_memory(n_f,"action")
         self.habit.append(tmp_habit)
         #self.vae_action.append(tmp_act)
         
      for i in self.habit:
         print("VAE : ",i.get_id())
         #print("memory : ",len(i.memory))
         print("latent space : ",i.get_latent_space())
         print("latent space scaled : ",i.get_latent_space_dnf())

   def load_nn_action(self):
      list_dir = os.listdir(self.folder_habituation)
      self.vae_action = []
      for i in range(0,len(list_dir)):
         #tmp_habit = VariationalAE(i,4,3,2)
         tmp_act = VariationalAE(i,5,4,2)
         n_f = self.folder_habituation + str(i) + "/"
         #tmp_habit.load_nn(self.folder_habituation,i,"outcome")
         #tmp_habit.load_memory(n_f,"outcome")
         tmp_act.load_nn(self.folder_habituation,i,"action")
         tmp_act.load_memory(n_f,"action")
         #self.habit.append(tmp_habit)
         self.vae_action.append(tmp_act)
      #for i in self.habit:
      #   print("VAE : ",i.get_id())
      #   print("memory : ",len(i.memory))
      #   print("latent space : ",i.get_latent_space())
      #   print("latent space scaled : ",i.get_latent_space_dnf())


if __name__ == "__main__":
   habituation = Habituation()
   rospy.spin()