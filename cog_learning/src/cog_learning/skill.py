#!/usr/bin/env python3
from cog_learning.multilayer import *
import random
import copy

class Skill(object):
    def __init__(self):
        torch.manual_seed(1024) #1024
        #7 4
        #I:state x_object, y_object, angle_object O: fposx, fposy, ind x, ind y
        self.inverse_model = MultiLayerP(3,5,4)
        self.inverse_model.to(device)
        #7 4
        #F: state x_object, y_object, angle_object, fposx, fposy, ind x, ind y O: vx_object, vy_object, vangle_object, touch_object
        self.forward_model = MultiLayerP(7,6,4)
        self.forward_model.to(device)
        #I: state x_object, y_object, angle_object, ind x, ind y O: probability
        self.r_predictor = MultiLayerPredictor(5,7,1)
        self.r_predictor.to(device)
        self.error_fwd = 1.0
        self.error_inv = 1.0
        self.memory_size = 30
        self.memory = []
        self.memory_pred = []
        self.name_skill = ""

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

    def save_memory_pred(self, pwd):
        n = pwd + self.name_skill + "_memory_pred.pkl"
        exist = path.exists(n)
        if exist:
            os.remove(n)
        filehandler = open(n, 'wb')
        pickle.dump(self.memory_pred, filehandler)

    def load_memory(self, pwd):
        filehandler_l = open(pwd, 'rb') 
        nl = pickle.load(filehandler_l)
        self.memory = nl

    def load_memory_pred(self, pwd):
        filehandler_l = open(pwd, 'rb') 
        nl = pickle.load(filehandler_l)
        self.memory_pred = nl

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

    def save_pred_nn(self,pwd):
        n = pwd + self.name_skill + "_predictor.pt"
        exist = path.exists(n)
        if exist:
            os.remove(n)
        torch.save({'predictor': self.r_predictor.state_dict()}, n)

    def load_fwd_nn(self,pwd):
        checkpoint = torch.load(pwd)
        self.forward_model.load_state_dict(checkpoint['forward'])

    def load_inv_nn(self,pwd):
        checkpoint = torch.load(pwd)
        self.inverse_model.load_state_dict(checkpoint['inverse'])

    def load_pred_nn(self,pwd):
        checkpoint = torch.load(pwd)
        self.r_predictor.load_state_dict(checkpoint['predictor'])

    def add_to_memory(self,sample):
        self.memory.append(sample)
        s = len(self.memory)
        if s > self.memory_size:
            self.memory.pop(0)

    def add_to_pred_memory(self,sample):
        self.memory_pred.append(sample)

    def get_memory(self):
        return self.memory
    
    def get_inverse_error(self):
        return self.error_inv
    
    def print_memory(self):
        print("size memory : ",len(self.memory))
        print("memory skills : ",self.memory)

    def train_inverse_model(self):
        print("train inverse...")
        current_cost = 0
        last_cost = 15
        learning_rate = 5e-3
        epochs = 10

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
        print("end training inverse")

    #takes object location and motor command as input and produces the expected future object location as output
    def train_forward_model(self):
        print("train forward...")
        current_cost = 0
        last_cost = 15
        learning_rate = 5e-3
        epochs = 10
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
        print("end training forward")
            #if current_cost > last_cost:
            #    break
            #last_cost = current_cost
            #current_cost = 0
            
    #takes object state and action (ind x,y from nnga) and predict reward
    def train_predictor(self):
        print("train predictor...")
        current_cost = 0
        last_cost = 15
        learning_rate = 1e-3
        epochs = 10000
        data_input = []
        stop = False
        self.r_predictor.to(device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.r_predictor.parameters(),lr=learning_rate)
        current_cost = 0
        mem = copy.deepcopy(self.memory_pred)
        i = 0
        while not stop:
            random.shuffle(mem)
            for j in range(0,len(mem)):
                self.r_predictor.train()
                optimizer.zero_grad()
                sample = mem[j]
                inputs = sample[0]
                targets = sample[1]
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = self.r_predictor(inputs)
                cost = criterion(outputs,targets)
                cost.backward()
                optimizer.step()
                current_cost = current_cost + cost.item()
            #print("Epoch: {}/{}...".format(i, epochs),"BCE : ",current_cost)
            i+=1
            if current_cost < 0.01:
                stop = True
            current_cost = 0
        print("end training predictor")
    
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
        fin_error = math.erf(error.item())
        self.error_inv = fin_error
        #error = self.getErrorPrediction(out,targets)

        return fin_error
    
    def predict_inverse(self,inputs):
        self.inverse_model.eval()
        inputs = inputs.to(device)
        out = self.inverse_model(inputs)
        res = out.detach().numpy()

        return res

    def predictForwardModel(self,inputs,targets):
        self.forward_model.eval()
        inputs = inputs.to(device)
        targets = targets.to(device)
        out = self.forward_model(inputs)
        mse_loss = nn.MSELoss()
        error = mse_loss(out, targets)
        fin_error = math.erf(error.item())
        self.error_fwd = fin_error
        #print("Error before erf : ",error.item())
        #print("Error after : ",fin_error)
        #error = self.getErrorPrediction(out,targets)

        return fin_error
    
    def forward_r_predictor(self,inputs):
        self.r_predictor.eval()
        inputs = inputs.to(device)
        out = self.r_predictor(inputs)
        res = out.detach().numpy()

        return res
    
if __name__ == "__main__":
    s = Skill()
    inp = [-0.2143,0.0448,0.5272,0.0,0.48]
    out = [1.0]
    t_in = torch.tensor(inp,dtype=torch.float)
    t_out = torch.tensor(out,dtype=torch.float)
    samp = []
    samp.append(t_in)
    samp.append(t_out)
    s.add_to_pred_memory(samp)
    s.train_predictor()
