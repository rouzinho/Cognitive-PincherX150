#!/usr/bin/env python3
from cog_learning.multilayer import *
from cog_learning.hebb_server import *
from cog_learning.skill import *
from detector.msg import Outcome
from motion.msg import DmpAction
from sklearn.preprocessing import MinMaxScaler
import copy

class NNGoalAction(object):
    def __init__(self):
        self.encoder = MultiLayerGA(9,6,4,2)
        self.encoder.to(device)
        self.decoder = MultiLayerGA(2,4,6,9)
        self.decoder.to(device)
        self.memory_size = 20
        self.memory = []
        self.hebbian = HebbServer()
        self.skills = []
        self.min_x = 0.18
        self.max_x = 0.67
        self.min_y = -0.35
        self.max_y = 0.35
        self.min_pitch = 0
        self.max_pitch = 1.5
        self.min_vx = -0.2
        self.max_vx = 0.2
        self.min_vy = -0.2
        self.max_vy = 0.2
        self.min_vpitch = -1.2
        self.max_vpitch = 1.2
        self.min_roll = -1.5
        self.max_roll = 1.5
        self.min_angle = -180
        self.max_angle = 180
        

    def create_skill(self):
        new_skill = Skill()
        self.skills.append(new_skill)
        ind = len(self.skills) - 1
        return ind
    
    def scale_data(self, data, min_, max_):
        n_x = np.array(data)
        n_x = n_x.reshape(-1,1)
        scaler_x = MinMaxScaler()
        x_minmax = np.array([min_, max_])
        scaler_x.fit(x_minmax[:, np.newaxis])
        n_x = scaler_x.transform(n_x)
        n_x = n_x.reshape(1,-1)
        n_x = n_x.flatten()
            
        return n_x[0]
    
    def scale_samples(self, outcome, dmp):
        #scale sample betwen [0-1] for learning
        new_outcome = Outcome()
        new_outcome.state_x = self.scale_data(outcome.state_x, self.min_x, self.max_x)
        new_outcome.state_y = self.scale_data(outcome.state_y, self.min_y, self.max_y)
        new_outcome.state_angle = self.scale_data(outcome.state_angle, self.min_angle, self.max_angle)
        new_outcome.x = self.scale_data(outcome.x, self.min_vx, self.max_vx)
        new_outcome.y = self.scale_data(outcome.y, self.min_vy, self.max_vy)
        new_outcome.roll = self.scale_data(outcome.roll, self.min_angle, self.max_angle)
        new_outcome.touch = outcome.touch
        new_dmp = DmpAction()
        new_dmp.v_x = self.scale_data(dmp.v_x, self.min_vx, self.max_vx)
        new_dmp.v_y = self.scale_data(dmp.v_y, self.min_vy, self.max_vy)
        new_dmp.v_pitch = self.scale_data(dmp.v_pitch, self.min_vpitch, self.max_vpitch)
        new_dmp.roll = self.scale_data(dmp.roll, self.min_roll, self.max_roll)
        new_dmp.grasp = dmp.grasp
        new_dmp.lpos_x = self.scale_data(dmp.lpos_x, self.min_x, self.max_x)
        new_dmp.lpos_y = self.scale_data(dmp.lpos_y, self.min_y, self.max_y)
        new_dmp.lpos_pitch = self.scale_data(dmp.lpos_pitch, self.min_pitch, self.max_pitch)

        return new_outcome, new_dmp


    #bootstrap learning when we discover first skill during exploration
    def bootstrap_learning(self, outcome_, dmp_):
        outcome, dmp = self.scale_samples(outcome_, dmp_)
        tmp_sample = [outcome.x,outcome.y,outcome.roll,outcome.touch,dmp.v_x,dmp.v_y,dmp.v_pitch,dmp.roll,dmp.grasp]
        tensor_sample_go = torch.tensor(tmp_sample,dtype=torch.float)
        sample_inp_fwd = [outcome.state_x,outcome.state_y,outcome.state_angle,dmp.lpos_x,dmp.lpos_y,dmp.lpos_pitch]
        sample_out_fwd = [outcome.x,outcome.y,outcome.roll,outcome.touch]
        sample_inp_inv = [outcome.state_x,outcome.state_y,outcome.state_angle,outcome.x,outcome.y,outcome.roll,outcome.touch]
        sample_out_inv = [dmp.lpos_x,dmp.lpos_y,dmp.lpos_pitch]
        t_inp_fwd = torch.tensor(sample_inp_fwd,dtype=torch.float)
        t_out_fwd = torch.tensor(sample_out_fwd,dtype=torch.float)
        t_inp_inv = torch.tensor(sample_inp_inv,dtype=torch.float)
        t_out_inv = torch.tensor(sample_out_inv,dtype=torch.float)
        self.add_to_memory(tensor_sample_go)
        ind_skill = self.create_skill()
        sample = []
        sample.append(t_out_fwd)
        sample.append(t_out_inv)
        sample.append(t_inp_fwd)
        sample.append(t_inp_inv)
        self.skills[ind_skill].add_to_memory(sample)
        t_inputs = self.encoder(tensor_sample_go)
        inputs = t_inputs.numpy()
        self.hebbian.hebbianLearning(inputs,ind_skill)
        self.trainDecoder()

    def trainDecoder(self):
        current_cost = 0
        last_cost = 15
        learning_rate = 5e-3
        epochs = 150

        #self.inverse_model.to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.decoder.parameters(),lr=learning_rate)        
        current_cost = 0
        for i in range(0,1):
            self.decoder.train()
            optimizer.zero_grad()
            sample = self.memory[-1]
            sample = sample.to(device)
            inputs = self.encoder(sample)
            inputs = inputs.to(device)
            targets = sample
            targets = targets.to(device)
            outputs = self.decoder(inputs)
            cost = criterion(outputs,targets)
            cost.backward()
            optimizer.step()
            #current_cost = current_cost + cost.item()
        for i in range(0,epochs):
            for j in range(0,len(self.memory)):
                self.decoder.train()
                optimizer.zero_grad()
                sample = self.memory[j]
                sample = sample.to(device)
                inputs = self.encoder(sample)
                inputs = inputs.to(device)
                targets = sample
                targets = targets.to(device)
                outputs = self.decoder(inputs)
                cost = criterion(outputs,targets)
                cost.backward()
                optimizer.step()
                current_cost = cost.item()
            #print("Epoch: {}/{}...".format(i, epochs),"MSE : ",current_cost)

    def forward_encoder(self, data):
        data = data.to(device)
        output = self.encoder(data)

        return output
    
    def forward_decoder(self, data):
        data = data.to(device)
        output = self.decoder(data)

        return output

    def add_to_memory(self, data):
        self.memory.append(data)