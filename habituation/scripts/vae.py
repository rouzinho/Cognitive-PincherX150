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
from motion.msg import DmpAction
from motion.msg import Dmp
from detector.msg import Outcome
from habituation.msg import LatentPos
from sklearn.preprocessing import MinMaxScaler
from cog_learning.msg import LatentGoalDnf
from cog_learning.msg import LatentDNF
from cog_learning.msg import LatentGoalNN

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
      self.linear1 = nn.Linear(9, 6)
      #self.linear2 = nn.Linear(7, 5)
      #self.linear3 = nn.Linear(5, 3)
      self.linear4 = nn.Linear(6, latent_dims)
      self.linear5 = nn.Linear(6, latent_dims)
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
   def __init__(self, latent_dims):
      super(Decoder, self).__init__()
      self.linear1 = nn.Linear(latent_dims, 6)
      #self.linear2 = nn.Linear(3, 5)
      #self.linear3 = nn.Linear(5, 7)
      self.linear4 = nn.Linear(6, 9)

   def forward(self, z):
      z = torch.tanh(self.linear1(z)) #F.relu
      z = self.linear4(z)

      return z

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

class VariationalAE(object):
   def __init__(self,id_object):
      self.vae = VariationalAutoencoder(2)
      self.memory = []
      self.id = id_object
      self.list_latent = []
      self.list_latent_scaled = []
      self.scale_factor = 40
      self.tmp_list = []
      self.bound_x = 0
      self.bound_y = 0

   def set_latent_dnf(self,exploration):
      ext_x, ext_y = self.get_latent_extremes(self.list_latent)
      self.list_latent_scaled = []
      if exploration == "fixed":
         self.bound_x = 100
         self.bound_y = 100
         if len(self.list_latent) > 1:
            for i in self.list_latent:
               x = self.scale_latent_to_dnf_static(i[0],ext_x[0],ext_x[1])
               y = self.scale_latent_to_dnf_static(i[1],ext_y[0],ext_y[1])
               self.list_latent_scaled.append([round(x),round(y)])
         else:
            self.list_latent_scaled.append([50,50])
      else:
         dist_x = abs(ext_x[0]) + abs(ext_x[1])
         dist_y = abs(ext_y[0]) + abs(ext_y[1])
         max_bound_x = (dist_x * self.scale_factor)
         max_bound_y = (dist_y * self.scale_factor) 
         #padding to avoid having extreme values on the edge of DNF
         padding_x = round(max_bound_x * 0.1)
         padding_y = round(max_bound_y * 0.1)
         self.bound_x = round(max_bound_x)
         self.bound_y = round(max_bound_y)
         print("max bound X ",self.bound_x)
         print("max bound Y ",self.bound_y)
         if len(self.list_latent) > 1:
            for i in self.list_latent:
               x = self.scale_latent_to_dnf_dynamic(i[0],ext_x[0],ext_x[1],padding_x,self.bound_x-padding_x)
               y = self.scale_latent_to_dnf_dynamic(i[1],ext_y[0],ext_y[1],padding_y,self.bound_y-padding_y)
               self.list_latent_scaled.append([round(x),round(y)])
         else:
            self.list_latent_scaled.append([5,5])
            self.bound_x = round(10)
            self.bound_y = round(10)

   def set_eval_to_latent_dnf(self, z, exploration):
      eval_value = LatentGoalDnf()
      list_eval = self.list_latent
      list_eval.append(z)
      ext_x, ext_y = self.get_latent_extremes(list_eval)
      if exploration == "fixed":
         x = self.scale_latent_to_dnf_static(z[0],ext_x[0],ext_x[1])
         y = self.scale_latent_to_dnf_static(z[1],ext_y[0],ext_y[1])
         eval_value.latent_x = round(x)
         eval_value.latent_y = round(y)
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
         y = self.scale_latent_to_dnf_dynamic(z[1],ext_y[0],ext_y[1],padding_y,max_bound_y-padding_y)
         eval_value.latent_x = round(x)
         eval_value.latent_y = round(y)

      return eval_value


   def get_latent_space_dnf(self):
      return self.list_latent_scaled
   
   def get_bound_x(self):
      return self.bound_x
   
   def get_bound_y(self):
      return self.bound_y

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

      if len(self.l_lat) > 0:
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
      while not stop:
         random.shuffle(self.memory)
         for sample, y in self.memory:
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
      print("END TRAINING")

   def evaluate_new_sample(self, sample):
      self.vae.eval()
      z, z_log, recon = self.vae.encoder(sample)
      z = z.to('cpu').detach().numpy()

      return z

   #send latent space for display and keep it in memory
   def plot_latent(self, num_batches=100):
      msg_latent = LatentPos()
      self.list_latent = []
      self.tmp_list = []
      for i in range(0,1):
         for sample, col in self.memory:
            self.vae.eval()
            z, z_log, recon = self.vae.encoder(sample)
            z = z.to('cpu').detach().numpy()
            msg_latent.x.append(z[0])
            msg_latent.y.append(z[1])
            t = [z[0],z[1],col]
            self.tmp_list.append(t)
            msg_latent.colors.append(col)
            self.list_latent.append(z)
            #plt.scatter(z[0], z[1], c=col, cmap='tab10')
            
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
      self.vae = VariationalAutoencoder(2)

   def get_memory_size(self):
      return len(self.memory)


class Habituation(object):
   def __init__(self):
      rospy.init_node('habituation', anonymous=True)
      rospy.Subscriber("/habituation/dmp", Dmp, self.callback_dmp)
      rospy.Subscriber("/outcome_detector/outcome", Outcome, self.callback_outcome)
      rospy.Subscriber("/habituation/id_object", Int16, self.callback_id)
      self.pub_latent_space_display = rospy.Publisher("/display/latent_space", LatentPos, queue_size=1, latch=True)
      self.pub_ready = rospy.Publisher("/habituation/ready", Bool, queue_size=1, latch=True)
      self.pub_latent_space_dnf = rospy.Publisher("/habituation/latent_space_dnf", LatentDNF, queue_size=1, latch=True)
      self.pub_test_latent = rospy.Publisher("/display/latent_test", LatentGoalNN, queue_size=1, latch=True)
      self.exploration_mode = rospy.get_param("exploration")
      self.id_vae = -1
      self.count_color = 0
      self.incoming_dmp = False
      self.incoming_outcome = False
      self.dmp = Dmp()
      self.outcome = Outcome()
      self.objects_vae = []
      self.habit = VariationalAE(1)
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
      self.colors = []
      self.colors.append("orange")
      self.colors.append("red")
      self.colors.append("purple")
      self.colors.append("blue")
      self.colors.append("green")
      self.colors.append("yellow")
      self.colors.append("pink")
      self.colors.append("cyan")
      self.colors.append("brown")
      self.colors.append("gray")
      self.time = 0
      self.first = True

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
   
   def send_latent_space(self):
      ls = self.habit.get_latent_space_dnf()
      msg_latent = LatentDNF()
      msg_latent.max_x = self.habit.get_bound_x()
      msg_latent.max_y = self.habit.get_bound_y()
      for i in ls:
         lg = LatentGoalDnf() 
         lg.latent_x = i[0]
         lg.latent_y = i[1]
         msg_latent.list_latent.append(lg)
      self.pub_latent_space_dnf.publish(msg_latent)

   def send_latent_test(self, v):
      tmp = LatentGoalNN()
      tmp.latent_x = v[0]
      tmp.latent_y = v[1]
      self.pub_test_latent.publish(tmp)

   def callback_dmp(self, msg):
      print("got DMP")
      if self.first:
         self.time = rospy.get_time()
         self.first = False
      self.dmp.v_x = self.scale_data(msg.v_x,self.min_vx,self.max_vx)
      self.dmp.v_y = self.scale_data(msg.v_y,self.min_vy,self.max_vy)
      self.dmp.v_pitch = self.scale_data(msg.v_pitch,self.min_vpitch,self.max_vpitch)
      self.dmp.roll = self.scale_data(msg.roll,self.min_roll,self.max_roll)
      self.dmp.grasp = self.scale_data(msg.grasp,self.min_grasp,self.max_grasp)
      self.incoming_dmp = True
      if(self.incoming_dmp and self.incoming_outcome):
         sample = [self.outcome.x,self.outcome.y,self.outcome.angle,self.outcome.touch,self.dmp.v_x,self.dmp.v_y,self.dmp.v_pitch,self.dmp.roll,self.dmp.grasp]
         tensor_sample = torch.tensor(sample,dtype=torch.float)
         if self.habit.get_memory_size() > 0:
            print("Testing new sample...")
            z = self.habit.evaluate_new_sample(tensor_sample)
            self.send_latent_test(z)
            rospy.sleep(10.0)
         self.learn_new_latent(tensor_sample)
         t = self.time - rospy.get_time()
         rospy.sleep(10.0)
         print("Time elapsed : ",t)

   def callback_outcome(self, msg):
      print("got outcome")
      if self.first:
         self.time = rospy.get_time()
         self.first = False
      self.outcome.x = self.scale_data(msg.x, self.min_vx, self.max_vx)
      self.outcome.y = self.scale_data(msg.y, self.min_vy, self.max_vy)
      self.outcome.angle = self.scale_data(msg.angle, self.min_angle, self.max_angle)
      self.outcome.touch = self.scale_data(msg.touch, self.min_grasp, self.max_grasp)
      self.incoming_outcome = True
      if(self.incoming_dmp and self.incoming_outcome):
         sample = [self.outcome.x,self.outcome.y,self.outcome.angle,self.outcome.touch,self.dmp.v_x,self.dmp.v_y,self.dmp.v_pitch,self.dmp.roll,self.dmp.grasp]
         tensor_sample = torch.tensor(sample,dtype=torch.float)
         if self.habit.get_memory_size() > 0:
            print("Testing new sample...")
            z = self.habit.evaluate_new_sample(tensor_sample)
            self.send_latent_test(z)
            rospy.sleep(10.0)
         self.learn_new_latent(tensor_sample)
         t = self.time - rospy.get_time()
         rospy.sleep(10.0)
         print("Time elapsed : ",t)

   def learn_new_latent(self, sample):
      torch.manual_seed(24)
      self.habit.reset_model()
      self.add_to_memory(sample)
      self.habit.train()
      msg = self.habit.plot_latent()
      self.pub_latent_space_display.publish(msg)
      self.habit.set_latent_dnf(self.exploration_mode)
      self.send_latent_space()
      self.incoming_dmp = False
      self.incoming_outcome = False
      self.pub_ready.publish(True)
      #self.test_reconstruct()

   def callback_id(self, msg):
      self.id_vae = msg.data

   def add_to_memory(self, sample):
      self.habit.add_to_memory([sample,self.colors[self.count_color]])
      self.outcome = Outcome()
      self.dmp = Dmp()
      self.count_color += 1

   def test_reconstruct(self):
      self.habit.test_reconstruct()


if __name__ == "__main__":
   habituation = Habituation()
   rospy.spin()