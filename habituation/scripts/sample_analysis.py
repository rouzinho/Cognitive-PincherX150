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

if __name__ == '__main__':
   folder = "/home/altair/PhD/Codes/Experiment-IMVAE/datas/analysis/cube/exploration/"
   run = "100"
   number = 15
   generate_rnd_samples(folder,run,number)
   generate_direct_samples(folder,run,number)