#! /usr/bin/python3
import numpy as np
import csv
import torch;
import torch.nn as nn
import torch.utils
import torch.distributions
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 100
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
from vae import Sampling,VariationalEncoder,Decoder,VariationalAutoencoder,VariationalAE

min_vx = -0.22
max_vx = 0.22
min_vy = -0.22
max_vy = 0.22
min_vpitch = 0.1
max_vpitch = 1.5
min_roll = -1.5
max_roll = 1.5
min_grasp = 0
max_grasp = 1
min_angle = -180
max_angle = 180


def generate_rnd_samples(folder,run,number):
   rnd_act = []
   rnd_out = []
   count = 0
   for i in range(0,number):
      name = folder + run + "/" + str(i) + "/exploration_data.csv"
      name_rnd_act = folder + run + "/rnd_act.pkl"
      name_rnd_out = folder + run + "/rnd_out.pkl"
      with open(name, "r") as file:
         j = 0
         csvreader = csv.reader(file)
         for row in csvreader:
            if j > 0:
               if float(row[9]) > 0.5:
                  s_out = [float(row[0]),float(row[1]),float(row[2]),float(row[3])]
                  s_act = [float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8])]
                  #out = torch.tensor(s_out,dtype=torch.float)
                  #act = torch.tensor(s_act,dtype=torch.float)
                  rnd_act.append(s_act)
                  rnd_out.append(s_out)
                  count+=1
            j+=1
   filehandler = open(name_rnd_out, 'wb')
   pickle.dump(rnd_out, filehandler)
   filehandler = open(name_rnd_act, 'wb')
   pickle.dump(rnd_act, filehandler)
   print("rnd : ",count)

def generate_direct_samples(folder,run,number):
   rnd_act = []
   rnd_out = []
   count = 0
   for i in range(0,number):
      name = folder + run + "/" + str(i) + "/exploration_data.csv"
      name_rnd_act = folder + run + "/direct_act.pkl"
      name_rnd_out = folder + run + "/direct_out.pkl"
      with open(name, "r") as file:
         j = 0
         csvreader = csv.reader(file)
         for row in csvreader:
            if j > 0:
               if float(row[10]) > 0.5:
                  s_out = [float(row[0]),float(row[1]),float(row[2]),float(row[3])]
                  s_act = [float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8])]
                  #out = torch.tensor(s_out,dtype=torch.float)
                  #act = torch.tensor(s_act,dtype=torch.float)
                  rnd_act.append(s_act)
                  rnd_out.append(s_out)
                  count+=1
            j+=1
   filehandler = open(name_rnd_out, 'wb')
   pickle.dump(rnd_out, filehandler)
   filehandler = open(name_rnd_act, 'wb')
   pickle.dump(rnd_act, filehandler)
   print("direct : ",count)

def open_sample(name):
   filehandler = open(name, 'rb') 
   mem = pickle.load(filehandler)
   
   return mem

def get_sum_exploration_rnd(folder,run,number):
   rnd_tot = None
   direct_tot = None
   rnd = None
   direct = None
   for i in range(0,number):
      name = folder + run + "/" + str(i) + "/exploration_data.csv"
      name_rnd_act = folder + run + "/direct_act.csv"
      name_rnd_out = folder + run + "/direct_out.csv"
      with open(name, "r") as file:
         j = 0
         rnd = None
         direct = None
         csvreader = csv.reader(file)
         for row in csvreader:
            if j == 1:
               rnd = np.array([float(row[9])])
               direct = np.array([float(row[10])])
            if j > 1:
               rnd = np.append(rnd,[float(row[9])])
               direct = np.append(direct,[float(row[10])])
            j+=1
      if i == 0:
         rnd_tot = rnd
         direct_tot = direct
      else:
         rnd_tot = np.vstack((rnd_tot,rnd))
         direct_tot = np.vstack((direct_tot,direct))
   
   return rnd_tot, direct_tot

def display_exploration(folder,run,number):
   random, direct = get_sum_exploration_rnd(folder,run,number)
   r = random.mean(axis=0)
   d = direct.mean(axis=0)
   fig, ax = plt.subplots(figsize=(12, 8))
   x = np.arange(8)
   ax.plot(x, r, label="random exploration")
   ax.plot(x, d, label="direct exploration")
   ax.set_xlabel('Number of stimuli')  # Add an x-label to the axes.
   ax.set_ylabel('Exploration level')  # Add a y-label to the axes.
   #ax.set_title("Learning Progress")  # Add a title to the axes.
   ax.legend();  # Add a legend.
   plt.ylim(0, None)
   plt.show()

def scale_data(data, min_, max_):
   n_x = np.array(data)
   n_x = n_x.reshape(-1,1)
   scaler_x = MinMaxScaler(feature_range=(-1,1))
   x_minmax = np.array([min_, max_])
   scaler_x.fit(x_minmax[:, np.newaxis])
   n_x = scaler_x.transform(n_x)
   n_x = n_x.reshape(1,-1)
   n_x = n_x.flatten()
         
   return n_x[0]

def scale_action(data):
   vx = scale_data(data[0],min_vx,max_vx)
   vy = scale_data(data[1],min_vy,max_vy)
   vp = scale_data(data[2],min_vpitch,max_vpitch)
   r = scale_data(data[3],min_roll,max_roll)
   g = scale_data(data[4],min_grasp,max_grasp)
   d = [vx,vy,vp,r,g]

   return d

def scale_out(data):
   outx = scale_data(data[0],min_vx,max_vx)
   outy = scale_data(data[1],min_vy,max_vy)
   outa = scale_data(data[2],min_angle,max_angle)
   outt = scale_data(data[3],min_grasp,max_grasp)
   d = [outx,outy,outa,outt]

   return d

def scale_all_action(data):
   data_scaled = []
   for i in data:
      d = scale_action(i)
      data_scaled.append(d)
   
   return data_scaled

def scale_all_outcome(data):
   data_scaled = []
   for i in data:
      d = scale_out(i)
      data_scaled.append(d)
   
   return data_scaled

def get_samples_tensor(data):
   tensors = []
   for i in data:
      t = torch.tensor(i,dtype=torch.float)
      tensors.append(t)

   return tensors



def generate_latent_action(folder,run):
   vae_action = VariationalAE(0,5,4,2)
   n_rnd = folder + run  + "/rnd_act.pkl"
   n_direct = folder + run + "/direct_act.pkl"
   #open numpy real datas
   rnd_act_real = open_sample(n_rnd)
   direct_act_real = open_sample(n_direct)
   #scale datas
   rnd_act_scaled = scale_all_action(rnd_act_real)
   direct_act_scaled = scale_all_action(direct_act_real)
   #make them tensors
   t_rnd_act = get_samples_tensor(rnd_act_scaled)
   t_direct_act = get_samples_tensor(direct_act_scaled)
   #send into vae memory
   vae_action.merge_samples(t_rnd_act,t_direct_act)
   vae_action.train()
   rnd_x, rnd_y = vae_action.get_list_latent(t_rnd_act)
   dir_x, dir_y = vae_action.get_list_latent(t_direct_act)
   n_x_r = folder + run  + "/rnd_act_lx.pkl"
   n_y_r = folder + run  + "/rnd_act_ly.pkl"
   n_x_d = folder + run  + "/dir_act_lx.pkl"
   n_y_d = folder + run  + "/dir_act_ly.pkl"
   filehandler = open(n_x_r, 'wb')
   pickle.dump(rnd_x, filehandler)
   filehandler = open(n_y_r, 'wb')
   pickle.dump(rnd_y, filehandler)
   filehandler = open(n_x_d, 'wb')
   pickle.dump(dir_x, filehandler)
   filehandler = open(n_y_d, 'wb')
   pickle.dump(dir_y, filehandler)

def open_latent_action(folder,run):
   nx_rnd = folder + run + "/rnd_act_lx.pkl"
   ny_rnd = folder + run + "/rnd_act_ly.pkl"
   nx_dir = folder + run + "/dir_act_lx.pkl"
   ny_dir = folder + run + "/dir_act_ly.pkl"
   filehandler = open(nx_rnd, 'rb') 
   x_rnd = pickle.load(filehandler)
   filehandler = open(ny_rnd, 'rb') 
   y_rnd = pickle.load(filehandler)
   filehandler = open(nx_dir, 'rb') 
   x_dir = pickle.load(filehandler)
   filehandler = open(ny_dir, 'rb') 
   y_dir = pickle.load(filehandler)

   return x_rnd, y_rnd, x_dir, y_dir

def plot_latent_space(folder,run):
   fig, ax = plt.subplots()
   colors_dir = ["red"]
   colors_rnd = ["blue"]
   x_rnd, y_rnd, x_dir, y_dir = open_latent_action(folder,run)
   ax.scatter(x_rnd, y_rnd, c='red', label="random")
   ax.scatter(x_dir, y_dir , c='blue', label="direct")
   ax.legend()
   ax.set_xlim((-1.5,1.5))
   ax.set_ylim((-1.5,1.5))
   plt.show()


if __name__ == '__main__':
   folder = "/home/altair/PhD/Codes/Experiment-IMVAE/datas/analysis/cube/exploration/"
   run = "35"
   number = 14
   #display_exploration(folder,run,number)
   #generate_rnd_samples(folder,run,number)
   #generate_direct_samples(folder,run,number)
   #generate_latent_action(folder,run)
   plot_latent_space(folder,run)