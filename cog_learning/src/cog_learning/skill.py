#!/usr/bin/env python3
from cog_learning.multilayer import *


class Skill(object):
    def __init__(self):
        #7 3
        #I:state x_object, y_object, angle_object, outcome vx_object, vy_object, vangle_object, touch_object O: lposx, lposy, lpospitch, ind x, ind y
        self.inverse_model = MultiLayerP(7,6,5)
        self.inverse_model.to(device)
        #6 4
        #I: x_object, y_object, angle_object, lposx, lposy, lpospitch, ind x, ind y O: vx_object, vy_object, vangle_object, touch_object
        self.forward_model = MultiLayerP(8,6,4)
        self.forward_model.to(device)
        self.memory_size = 30
        self.memory = []
        self.name_skill = ""
        torch.manual_seed(58)

    def set_name(self, data):
        self.name_skill = str(data[0]) + "_" + str(data[1]) 

    def get_name(self):
        return self.name_skill

    def save_memory(self, pwd):
        n = pwd + self.name_skill + "_memory.pkl"
        exist = path.exists(n)
        if exist:
            os.remove(n)
        filehandler = open(n, 'wb')
        pickle.dump(self.memory, filehandler)

    def load_memory(self, pwd):
        filehandler_l = open(pwd, 'rb') 
        nl = pickle.load(filehandler_l)
        self.memory = nl

    def save_fwd_nn(self,pwd):
        n = pwd + self.name_skill + "_forward.pt"
        exist = path.exists(n)
        if exist:
            os.remove(n)
        torch.save({'forward': self.forward_model.state_dict()}, n)

    def save_inv_nn(self,pwd):
        n = pwd + self.name_skill + "_inverse.pt"
        exist = path.exists(n)
        if exist:
            os.remove(n)
        torch.save({'inverse': self.inverse_model.state_dict()}, n)

    def load_fwd_nn(self,pwd):
        checkpoint = torch.load(pwd)
        self.forward_model.load_state_dict(checkpoint['forward'])

    def load_inv_nn(self,pwd):
        checkpoint = torch.load(pwd)
        self.inverse_model.load_state_dict(checkpoint['inverse'])

    def add_to_memory(self,sample):
        self.memory.append(sample)
        s = len(self.memory)
        if s > self.memory_size:
            self.memory.pop(0)

    def get_memory(self):
        return self.memory
    
    def print_memory(self):
        print("size memory : ",len(self.memory))
        print("memory skills : ",self.memory)

    def train_inverse_model(self):
        current_cost = 0
        last_cost = 15
        learning_rate = 5e-3
        epochs = 2

        #self.inverse_model.to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.inverse_model.parameters(),lr=learning_rate)        
        current_cost = 0
        for i in range(0,1):
            self.inverse_model.train()
            optimizer.zero_grad()
            sample = self.memory[-1]
            inputs = sample[3]
            targets = sample[1]
            #print("input inverse : ",inputs)
            #print("output inverse : ",targets)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = self.inverse_model(inputs)
            cost = criterion(outputs,targets)
            cost.backward()
            optimizer.step()
            #current_cost = current_cost + cost.item()
        for i in range(0,epochs):
            #print("Inverse model")
            for j in range(0,len(self.memory)):
                self.inverse_model.train()
                optimizer.zero_grad()
                sample = self.memory[j]
                inputs = sample[3]
                targets = sample[1]
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = self.inverse_model(inputs)
                cost = criterion(outputs,targets)
                cost.backward()
                optimizer.step()
                current_cost = current_cost + cost.item()
            #print("Epoch: {}/{}...".format(i, epochs),"MSE : ",current_cost)
            current_cost = 0
            #if current_cost > last_cost:
            #    break
            #last_cost = current_cost
            #current_cost = 0

    #takes object location and motor command as input and produces the expected future object location as output
    def train_forward_model(self):
        current_cost = 0
        last_cost = 15
        learning_rate = 5e-3
        epochs = 2
        data_input = []
        self.forward_model.to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.forward_model.parameters(),lr=learning_rate)
        current_cost = 0
        for i in range(0,1):
            self.forward_model.train()
            optimizer.zero_grad()
            sample = self.memory[-1]
            inputs = sample[2]
            targets = sample[0]
            #print("input forward : ",inputs)
            #print("output forward : ",targets)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = self.forward_model(inputs)
            cost = criterion(outputs,targets)
            cost.backward()
            optimizer.step()
            #current_cost = current_cost + cost.item()
        for i in range(0,epochs):
            #print("Forward model")
            for j in range(0,len(self.memory)):
                self.forward_model.train()
                optimizer.zero_grad()
                sample = self.memory[j]
                inputs = sample[2]
                targets = sample[0]
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = self.forward_model(inputs)
                cost = criterion(outputs,targets)
                cost.backward()
                optimizer.step()
                current_cost = current_cost + cost.item()
            #print("Epoch: {}/{}...".format(i, epochs),"MSE : ",current_cost)
            current_cost = 0
            #if current_cost > last_cost:
            #    break
            #last_cost = current_cost
            #current_cost = 0
            
    
    
    #compute the error between the prediction and the actual data
    def getErrorPrediction(self,prediction,actual):
        error = math.sqrt((actual[0][0]-prediction[0][0])**2 + (actual[0][1]-prediction[0][1])**2)

        return error

    def getErrorForward(self,prediction,actual):
        error = math.sqrt((actual[0][0]-prediction[0][0])**2 + (actual[0][1]-prediction[0][1])**2)

        return error

    #return the error between prediction and actual motor command for the inverse model
    #parameters are tensors
    def predictInverseModel(self,inputs,targets):
        self.inverse_model.eval()
        inputs = inputs.to(device)
        targets = targets.to(device)
        out = self.inverse_model(inputs)
        mse_loss = nn.MSELoss()
        error = mse_loss(out, targets)
        #error = self.getErrorPrediction(out,targets)

        return error

    def predictIVModel(self,inputs):
        self.inverse_model.eval()
        inputs = inputs.to(device)
        out = self.inverse_model(inputs)

        return out

    def predictForwardModel(self,inputs,targets):
        self.forward_model.eval()
        inputs = inputs.to(device)
        targets = targets.to(device)
        out = self.forward_model(inputs)
        mse_loss = nn.MSELoss()
        error = mse_loss(out, targets)
        #error = self.getErrorPrediction(out,targets)

        return error