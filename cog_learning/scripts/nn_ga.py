#!/usr/bin/env python3
from multilayer import *


class NNGoalAction(object):
    def __init__(self):
        self.encoder = MultiLayer(8,6,4,2)
        self.encoder.to(device)
        self.decoder = MultiLayer(2,4,6,8)
        self.decoder.to(device)
        self.memory_size = 20
        self.memory = []

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