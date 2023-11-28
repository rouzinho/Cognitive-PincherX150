#!/usr/bin/env python3
from cog_learning.multilayer import *


class Skill(object):
    def __init__(self):
        #7 3
        #I:x_object, y_object, angle_object, vx_object, vy_object, vangle_object, touch_object O: lposx, lposy, lpospitch
        self.inverse_model = MultiLayerP(7,5,3)
        self.inverse_model.to(device)
        #6 4
        #I: x_object, y_object, angle_object, lposx, lposy, lpospitch O: vx_object, vy_object, vangle_object, touch_object
        self.forward_model = MultiLayerP(6,5,4)
        self.forward_model.to(device)
        self.memory_size = 30
        self.memory = []

    def add_to_memory(self,sample):
        self.memory.append(sample)
        s = len(self.memory)
        if s > self.memory_size:
            self.memory.pop(0)

    def getMemory(self):
        return self.memory

    def trainInverseModel(self):
        current_cost = 0
        last_cost = 15
        learning_rate = 5e-3
        epochs = 1

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
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = self.inverse_model(inputs)
            cost = criterion(outputs,targets)
            cost.backward()
            optimizer.step()
            #current_cost = current_cost + cost.item()
        for i in range(0,epochs):
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
                #current_cost = current_cost + cost.item()
            #print("Epoch: {}/{}...".format(i, epochs),
                                #"MSE : ",current_cost)

            #if current_cost > last_cost:
            #    break
            #last_cost = current_cost
            #current_cost = 0

    #takes object location and motor command as input and produces the expected future object location as output
    def trainForwardModel(self):
        current_cost = 0
        last_cost = 15
        learning_rate = 5e-3
        epochs = 1
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
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = self.forward_model(inputs)
            cost = criterion(outputs,targets)
            cost.backward()
            optimizer.step()
            current_cost = current_cost + cost.item()
        for i in range(0,epochs):
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
                #current_cost = current_cost + cost.item()
            #print("Epoch: {}/{}...".format(i, epochs),
                                #"MSE : ",current_cost)

            #if current_cost > last_cost:
            #    break
            #last_cost = current_cost
            #current_cost = 0
            
        

    def saveNN(self):
        name_inv = "/home/altair/PhD/Codes/ExperimentIM-LCNE/datas/complete/inverse_"+str(int(self.model_object.object))+"_"+str(int(self.model_object.goal))+".pt"
        torch.save(self.inverse_model, name_inv)
        name_fwd = "/home/altair/PhD/Codes/ExperimentIM-LCNE/datas/complete/forward_"+str(int(self.model_object.object))+"_"+str(int(self.model_object.goal))+".pt"
        torch.save(self.forward_model, name_fwd)
    
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

    def saveMemory(self):
        #name = "/home/altair/PhD/catkin_noetic/rosbags/experiment/datas/goal_"+str(int(self.model_object.object))+"_"+str(int(self.model_object.goal))+".pkl"
        name = "/home/altair/PhD/Codes/ExperimentIM-LCNE/datas/complete/goal_"+str(int(self.model_object.object))+"_"+str(int(self.model_object.goal))+".pkl"
        exist = path.exists(name)
        if exist:
            os.remove(name)
        filehandler = open(name, 'wb')
        pickle.dump(self.memory, filehandler)
        #with open(name, 'wb') as outp:
        #    pickle.dump(self.memory, outp, pickle.HIGHEST_PROTOCOL)

    def retrieveMemory(self):
        #name = "/home/altair/PhD/catkin_noetic/rosbags/experiment/datas/goal_"+str(int(self.model_object.object))+"_"+str(int(self.model_object.goal))+".pkl"
        #name = "/home/altair/PhD/Codes/ExperimentIM-LCNE/datas/neural_memory/goal_"+str(int(self.model_object.object))+"_"+str(int(self.model_object.goal))+".pkl"
        name = "/home/altair/PhD/Codes/ExperimentIM-LCNE/datas/complete/goal_"+str(int(self.model_object.object))+"_"+str(int(self.model_object.goal))+".pkl"
        #with open(name, 'rb') as inp:
        #    mem = pickle.load(inp)
        filehandler = open(name, 'rb') 
        mem = pickle.load(filehandler)
        self.memory = mem

