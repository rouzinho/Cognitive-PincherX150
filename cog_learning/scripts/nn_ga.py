#!/usr/bin/env python3
from multilayer import *
from hebb_server import *
from skill import *

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

    def create_skill(self):
        new_skill = Skill()
        self.skills.append(new_skill)
        ind = len(self.skills) - 1
        return ind
    
    def hebbian_learning(self):
        pass

    def bootstrap_learning(self, outcome, dmp):
        tmp_sample = [outcome.x,outcome.y,outcome.roll,outcome.touch,dmp.v_x,dmp.v_y,dmp.v_pitch,dmp.roll,dmp.grasp]
        tensor_sample_go = torch.tensor(tmp_sample,dtype=torch.float)
        sample_inp_fwd = [outcome.state_x,outcome.state_y,outcome.state_roll,dmp.lpos_x,dmp.lpos_y,dmp.lpos_pitch]
        sample_out_fwd = [outcome.x,outcome.y,outcome.roll,outcome.touch]
        sample_inp_inv = [outcome.state_x,outcome.state_y,outcome.state_roll,outcome.x,outcome.y,outcome.roll,outcome.touch]
        sample_out_inv = [dmp.lpos_x,dmp.lpos_y,dmp.lpos_pitch]
        t_inp_fwd = torch.tensor(sample_inp_fwd,dtype=torch.float)
        t_out_fwd = torch.tensor(sample_out_fwd,dtype=torch.float)
        t_inp_inv = torch.tensor(sample_inp_inv,dtype=torch.float)
        t_out_inv = torch.tensor(sample_out_inv,dtype=torch.float)
        self.add_to_memory(tensor_sample_go)
        ind_skill = self.create_skill()
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