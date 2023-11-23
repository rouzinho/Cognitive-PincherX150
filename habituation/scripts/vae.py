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
from detector.msg import Outcome
from habituation.msg import LatentPos
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
      self.linear1 = nn.Linear(9, 7)
      self.linear2 = nn.Linear(7, 5)
      self.linear3 = nn.Linear(5, 3)
      self.linear4 = nn.Linear(3, latent_dims)
      self.linear5 = nn.Linear(3, latent_dims)
      #self.N = torch.distributions.Normal(0, 1)
      #self.kl = 0
      self.sampling = Sampling()

   def forward(self, x):
      #x = torch.flatten(x, start_dim=1)
      x = torch.relu(self.linear1(x))
      x = torch.tanh(self.linear2(x))
      x = torch.relu(self.linear3(x))
      z_mean = self.linear4(x)
      #z_log_var = self.linear5(x)
      #mu =  torch.tanh(self.linear2(x))
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
      self.linear1 = nn.Linear(latent_dims, 3)
      self.linear2 = nn.Linear(3, 5)
      self.linear3 = nn.Linear(5, 7)
      self.linear4 = nn.Linear(7, 9)

   def forward(self, z):
      z = torch.tanh(self.linear1(z)) #F.relu
      z = torch.tanh(self.linear2(z))
      z = torch.tanh(self.linear3(z))
      z = self.linear4(z)
      #z = torch.sigmoid(self.linear4(z))
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
      return 70 * self.recon_loss + self.kld_loss
      #return self.recon_loss + 2 * self.kld_loss

   def train(self, epochs=6000):
      kl_weight = 0.8
      #opt = torch.optim.Adam(list(self.vae.encoder.parameters()) + list(self.vae.decoder.parameters()), lr=0.001)
      opt = torch.optim.Adam(self.vae.parameters(), lr=0.01)
      train_loss = 0.0
      last_loss = 10
      stop = False
      i = 0
      min_err = 1.0
      err_rec = 1.0
      #for epoch in range(epochs):
      while not stop:
         random.shuffle(self.memory)
         for sample, y in self.memory:
            self.vae.train()
            s = sample.to(device) # GPU
            opt.zero_grad()
            pred = self.vae(s)
            loss = self.vae_loss(pred, s)
            err_rec = self.recon_loss.item()
            print("loss reconstruct : ",self.recon_loss.item())
            print("loss KL : ",self.kld_loss.item())
            #print("loss total : ",loss)
            loss.backward()
            opt.step()
            i += 1
            if err_rec < min_err:
               min_err = err_rec
               print("min reconstructed : ",min_err)
               print("loss KL : ",self.kld_loss.item())
            if self.kld_loss < 0.01 and self.recon_loss < 0.001:
               stop = True
               break
            #last_loss = loss_mse
            #if nb_data > 1:
            #   if loss_mse < 9e-5 and loss_kl < 9e-5:
            #      stop = True
            #      print("epochs : ",i)
            #else:
            #   if i > 10000:
            #      stop = True
      print("END TRAINING")

   def plot_latent(self, num_batches=100):
      msg_latent = LatentPos()
      for i in range(0,1):
         for sample, col in self.memory:
            self.vae.eval()
            z, z_log, recon = self.vae.encoder(sample)
            #print(col,z)
            z = z.to('cpu').detach().numpy()
            msg_latent.x.append(z[0])
            msg_latent.y.append(z[1])
            msg_latent.colors.append(col)
            #plt.scatter(z[0], z[1], c=col, cmap='tab10')
            
      return msg_latent
   
   def test_reconstruct(self):
      self.vae.eval()
      res = self.vae(self.memory[0][0])
      print("original : ", self.memory[0][0])
      print("reconstruct : ",res[2])
      print(self.memory)


class Habituation(object):
   def __init__(self):
      rospy.init_node('habituation', anonymous=True)
      rospy.Subscriber("/motion_pincher/dmp_param", Dmp, self.callback_dmp)
      rospy.Subscriber("/outcome_detector/outcome", Outcome, self.callback_outcome)
      rospy.Subscriber("/habituation/id_object", Int16, self.callback_id)
      self.pub_latent_space = rospy.Publisher("/display/latent_space", LatentPos, queue_size=1, latch=True)
      self.pub_ready = rospy.Publisher("/habituation/ready", Bool, queue_size=1, latch=True)
      self.id_vae = -1
      self.count_color = 0
      self.incoming_dmp = False
      self.incoming_outcome = False
      self.dmp = DmpAction()
      self.outcome = Outcome()
      self.objects_vae = []
      self.habit = VariationalAE(1)
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

   def callback_dmp(self, msg):
      print("got DMP")
      self.dmp.v_x = msg.v_x
      self.dmp.v_y = msg.v_y
      self.dmp.v_pitch = msg.v_pitch
      self.dmp.roll = msg.roll
      self.dmp.grasp = msg.grasp
      self.incoming_dmp = True
      if(self.incoming_dmp and self.incoming_outcome):
         self.add_to_memory()
         self.habit.train()
         msg = self.habit.plot_latent()
         self.pub_latent_space.publish(msg)
         self.incoming_dmp = False
         self.incoming_outcome = False
         self.pub_ready.publish(True)
         self.test_reconstruct()

   def callback_outcome(self, msg):
      print("got outcome")
      self.outcome.x = msg.x
      self.outcome.y = msg.y
      self.outcome.roll = msg.roll
      self.outcome.touch = msg.touch
      self.incoming_outcome = True
      if(self.incoming_dmp and self.incoming_outcome):
         self.add_to_memory()
         self.habit.train()
         msg = self.habit.plot_latent()
         self.pub_latent_space.publish(msg)
         self.incoming_dmp = False
         self.incoming_outcome = False
         self.pub_ready.publish(True)
         self.test_reconstruct()

   def callback_id(self, msg):
      self.id_vae = msg.data

   def add_to_memory(self):
      sample = [self.outcome.x,self.outcome.y,self.outcome.roll,self.outcome.touch,self.dmp.v_x,self.dmp.v_y,self.dmp.v_pitch,self.dmp.roll,self.dmp.grasp]
      tensor_sample = torch.tensor(sample,dtype=torch.float)
      self.habit.add_to_memory([tensor_sample,self.colors[self.count_color]])
      self.outcome = Outcome()
      self.dmp = Dmp()
      self.count_color += 1

   def test_reconstruct(self):
      self.habit.test_reconstruct()


if __name__ == "__main__":
   torch.manual_seed(58)
   habituation = Habituation()
   rospy.spin()
   """latent_dims = 2
   data = []
   #data = torch.utils.data.DataLoader(
   #     torchvision.datasets.MNIST('./data',
   #            transform=torchvision.transforms.ToTensor(),
   #            download=True),
   #     batch_size=128,
   #     shuffle=True)
   goal = [0.1,0.1,0.5,0.2,0.3,0.25,0.65,0.9,0.0]
   sec_goal = [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.0]
   third_goal = [0.2,0.34,0.12,0.43,0.8,0.85,0.45,0.76,0.0]
   test_goal = [0.6,0.7,0.4,0.15,0.67,0.77,0.8,0.35,1.0]
   test_goal2 = [0.8,0.6,0.1,0.3,0.5,0.45,0.9,0.1,0.0]
   sim1 = [0.1,0.1,0.5,0.2,0.3,0.25,0.4,0.9,0.0]
   sim2 = [0.4,0.4,0.1,0.3,0.1,0.6,0.1,0.45,0.0]
   sim3 = [0.2,0.34,0.12,0.43,0.8,0.85,0.45,0.9,0.0]
   sim4 = [0.6,0.7,0.9,0.15,0.67,0.77,0.8,0.35,1.0]
   sim5 = [0.8,0.6,0.1,0.3,0.5,0.15,0.9,0.1,0.0]
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
   #habit = VariationalAE(1)
   #habit.train(data)
   data.append([ts1,c_c])
   data.append([ts2,c_o])
   data.append([ts3,c_pu])
   data.append([ts4,c_br])
   data.append([ts5,c_gr])
   habit = VariationalAE(1)
   habit.train(data)
   res = habit.vae.forward(tensor_goal)
   print("original : ",tensor_goal)
   print("reconstructed : ",res[2])
   #data.append([ts1,c_c])
    #res2 = vae.forward(tensor_secgoal)
    #res3 = vae.forward(tensor_rdgoal)
    #print("RECONSTRUCTION ",res)
    #print("RECONSTRUCTION 2",res2)
    #print("RECONSTRUCTION 3",res3)
    #data.append([tensor_test,c_y])
   habit.plot_latent(data)
   plt.show()"""