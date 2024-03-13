#! /usr/bin/python3
from numpy import save
# load numpy array from npy file
from numpy import load
from os.path import exists
import os
import numpy as np
import pickle

class HebbServer(object):
    def __init__(self):
        super(HebbServer, self).__init__() 
        self.weights = np.zeros((100,100,40))
        self.weights_action = np.zeros((100,100,100,100))
        self.weights_init = False
        self.activation = 0
        self.init_size = True
        self.weights_name = "/home/altair/PhD/Codes/ExperimentIM-LCNE/datas/complete/hebbian_weights.npy"
        self.name_dmps = "/home/altair/PhD/Codes/ExperimentIM-LCNE/datas/complete/dmps.pkl"
        #self.loading = rospy.get_param("/load_datas")
        self.reset = False

    def callbackReset(self,msg):
        tmp = msg.data
        if tmp > 0.5:
            self.reset = True
        else:
            self.reset = False

    def init_weights(self):
        self.weights = np.zeros((100,100,40))
    
    def getReset(self):
        return self.reset

    def printWeights(self):
        print(self.weights)

    def hebbianLearning(self, point_ga, ind_fwdinv): #lupdate weights when there is a incoming field with reward
        #one shot learning, setting weights directly to one for faster processing
        # learninf a small patch, easier to retrieve with DNF
        for i in range(point_ga[0]-1,point_ga[0]+2):
            for j in range(point_ga[1]-1,point_ga[1]+2):
                self.weights[i,j,ind_fwdinv] = 1
        

    def hebbianActivation(self, point_ga):
        ind = -1
        for i in range(0,self.weights.shape[2]):
            if self.weights[point_ga[0],point_ga[1],i] == 1:
                ind = i

        return ind
    
    def hebbianLearningAction(self, point_ga, point_act): #lupdate weights when there is a incoming field with reward
        #one shot learning, setting weights directly to one for faster processing
        # learninf a small patch, easier to retrieve with DNF
        for i in range(point_ga[0]-1,point_ga[0]+2):
            for j in range(point_ga[1]-1,point_ga[1]+2):
                self.weights_action[i,j,point_act[0],point_act[1]] = 1
        

    def hebbianActivationAction(self, point_ga):
        l_ind = []
        for i in range(0,self.weights_action.shape[2]):
            for j in range(0,self.weights_action.shape[3]):
                if self.weights_action[point_ga[0],point_ga[1],i,j] == 1:
                    l_ind.append([i,j])

        return l_ind
    
    def get_all(self):
        l_ind = []
        for i in range(0,self.weights_action.shape[0]):
            for j in range(0,self.weights_action.shape[1]):
                for k in range(0,self.weights_action.shape[2]):
                    for l in range(0,self.weights_action.shape[3]):
                        if self.weights_action[i,j,k,l] == 1:
                            l_ind.append([i,j,k,l])
        return l_ind

    def saveWeights(self, name_weights):
        save(name_weights,self.weights)
        
    def loadWeights(self, name_weights):
        self.weights = load(name_weights)

    def saveWeightsAction(self, name_weights):
        save(name_weights,self.weights_action)
        
    def loadWeightsAction(self, name_weights):
        self.weights_action = load(name_weights)

if __name__ == "__main__":
    hebb_srv = HebbServer()
    hebb_srv.loadWeightsAction("/home/altair/PhD/Codes/Experiment-IMVAE/datas/production/nn_ga/0/hebbian_weights_action.npy")
    ga = [90,50]
    pa = [10,20]
    p2 = [35,45]
    #hebb_srv.hebbianLearningAction(ga,pa)
    #hebb_srv.hebbianLearningAction(ga,p2)
    #hebb_srv.hebbianLearningAction([50,50],9)
    #ind = hebb_srv.hebbianActivation(ga)
    #print(ind)
    #hebb_srv.saveWeightsAction("/home/altair/interbotix_ws/src/cog_learning/src/cog_learning/weights.npy")
    #ind = hebb_srv.hebbianActivationAction([41,41])
    #hebb_srv
    #print("res : ",ind)
    

