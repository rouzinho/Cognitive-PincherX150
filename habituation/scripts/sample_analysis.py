#! /usr/bin/python3
import numpy as np
import csv
import torch;
import torch.nn as nn
import torch.utils
import torch.distributions
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 100
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
from vae import Sampling,VariationalEncoder,Decoder,VariationalAutoencoder,VariationalAE

def generate_rnd_samples(folder,run,number):
   rnd_act = []
   rnd_out = []
   count = 0
   for i in range(0,number):
      name = folder + run + "/" + str(i) + "/exploration_data.csv"
      name_rnd_act = folder + run + "/rnd_act.csv"
      name_rnd_out = folder + run + "/rnd_out.csv"
      with open(name, "r") as file:
         j = 0
         csvreader = csv.reader(file)
         for row in csvreader:
            if j > 0:
               if float(row[9]) > 0.5:
                  s_out = [float(row[0]),float(row[1]),float(row[2]),float(row[3])]
                  s_act = [float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8])]
                  out = torch.tensor(s_out,dtype=torch.float)
                  act = torch.tensor(s_act,dtype=torch.float)
                  rnd_act.append(act)
                  rnd_out.append(out)
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
      name_rnd_act = folder + run + "/direct_act.csv"
      name_rnd_out = folder + run + "/direct_out.csv"
      with open(name, "r") as file:
         j = 0
         csvreader = csv.reader(file)
         for row in csvreader:
            if j > 0:
               if float(row[10]) > 0.5:
                  s_out = [float(row[0]),float(row[1]),float(row[2]),float(row[3])]
                  s_act = [float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8])]
                  out = torch.tensor(s_out,dtype=torch.float)
                  act = torch.tensor(s_act,dtype=torch.float)
                  rnd_act.append(act)
                  rnd_out.append(out)
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

if __name__ == '__main__':
   folder = "/home/altair/PhD/Codes/Experiment-IMVAE/datas/analysis/cube/exploration/"
   run = "100"
   number = 15
   display_exploration(folder,run,number)
   #generate_rnd_samples(folder,run,number)
   #generate_direct_samples(folder,run,number)