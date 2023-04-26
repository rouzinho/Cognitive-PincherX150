#! /usr/bin/python3
import numpy as np
import rospy
from geometry_msgs.msg import Pose
import random
import math
import time
import matplotlib.pyplot as plt
from matplotlib import animation
import os.path
from scipy.cluster.hierarchy import dendrogram, linkage
import geometry_msgs.msg

class DSom(object):
    def __init__(self,num_features,s,ep):
        super(DSom, self).__init__()
        rospy.init_node('som', anonymous=True)
        rospy.Subscriber("/dsom/node_value", Pose, self.callbackNode)
        self.pub_node = rospy.Publisher('/som/action', Pose, queue_size=1)
        self.num_features = num_features
        self.size = s
        self.shape = (s,s,num_features)
        self.grid = np.ones(self.shape)*0.5
        self.epoch = ep
        self.learning_rate = 0.7
        self.elasticity = 0.25
        self.current_time = 0
        self.neighbour_rad = -1.0
        self.influence = 0
        self.current_time = 0
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-1, s), ylim=(-1, s))
        self.map = np.random.random((s, s, num_features))
        self.im = plt.imshow(self.grid,interpolation='none')
        self.cluster_map = np.zeros((self.size,self.size,1))
        self.distance_map = np.zeros((self.size,self.size))
        self.j = 7
        self.worst = 0
        self.max = 0

    def callbackNode(self,msg):
        tmp = self.getWeightsNode(int(msg.position.x),int(msg.position.y))
        data = geometry_msgs.msg.Pose()
        data.position.x = tmp[0,0]
        data.position.y = tmp[0,1]
        data.position.z = tmp[0,2]
        self.pub_node.publish(data)

    def init_grid(self,mode):
        if mode == "fixed":
            self.grid = np.ones(self.shape)*0.5

        # Regular grid initialization
        if mode == "regular":
            self.grid = np.zeros(self.shape)
            for i in range(self.shape[0]):
                self.grid[i,:,0] = np.linspace(0,1,self.shape[1])
                self.grid[:,i,1] = np.linspace(0,1,self.shape[1])
                
        # Random initialization
        if mode == "random":
            self.grid = np.random.random(self.shape)

    def initNodeColor(self,number):
        weights = np.zeros((1,self.num_features))
        if number == 0:
            weights[0,0] = 0
            weights[0,1] = 0
            weights[0,2] = 0
        if number == 1:
            weights[0,0] = 1
            weights[0,1] = 0
            weights[0,2] = 0
        if number == 2:
            weights[0,0] = 0
            weights[0,1] = 1
            weights[0,2] = 0
        if number == 3:
            weights[0,0] = 0
            weights[0,1] = 0
            weights[0,2] = 1
        if number == 4:
            weights[0,0] = 1
            weights[0,1] = 1
            weights[0,2] = 0
        if number == 5:
            weights[0,0] = 0
            weights[0,1] = 1
            weights[0,2] = 1
        if number == 6:
            weights[0,0] = 1
            weights[0,1] = 0
            weights[0,2] = 1
        if number == 7:
            weights[0,0] = 1
            weights[0,1] = 1
            weights[0,2] = 1

        return weights

    def printSOM(self):
        print(self.grid)

    def printDistance(self):
        print(self.distance_map)

    def getBMU(self,sample):
        self.distance_map = np.zeros((self.size,self.size))
        self.distance_map = ((self.grid-sample)**2).sum(axis=-1)
        winner = np.unravel_index(np.argmin(self.distance_map), self.distance_map.shape)

        return winner

    def Gaussian(self,shape,center,sigma):
        def g(x):

            return np.exp(-x**2/sigma**2)
        return self.fromdistance(g,shape,center)

    def fromdistance(self,fn, shape, center=None, dtype=float):
        def distance(*args):
            d = 0
            for i in range(len(shape)):
                d += ((args[i]-center[i])/float(max(1,shape[i]-1)))**2
    #            d += ((args[i]-center[i])/float(shape[i]))**2
            return np.sqrt(d)/np.sqrt(len(shape))
        if center == None:
            center = np.array(list(shape))//2
        return fn(np.fromfunction(distance,shape,dtype=dtype))

    def defineClusters(self):
        nb_cluster = 0
        for i in range(0,self.cluster_map.shape[0]):
            for j in range(0,self.cluster_map.shape[1]):
                if self.cluster_map[i,j] == 0:
                    nb_cluster += 1
                    base_weights = self.network[i][j].getWeights()
                    base_weights = base_weights[0]
                    #print(base_weights)
                    base_node = Node(3)
                    base_node.setWeights(base_weights)
                    for k in range(0,self.size):
                        for l in range(0,self.size):
                            sample_weights = self.network[k][l].getWeights()
                            d = base_node.getDistance(sample_weights)
                            #print(d)
                            if d < 0.5:
                                self.cluster_map[k,l] = nb_cluster
                        #print("")
                #print(j)
                #return
                    

    def printClusters(self):
        for i in range(0,self.cluster_map.shape[0]):
            for j in range(0,self.cluster_map.shape[1]):
                print(self.cluster_map[i,j],end = "")
            print("")

    def trainSOMColor(self,color):
        self.current_time = 0
        while self.current_time < self.epoch:
            sample = self.initNodeColor(color)
            #print(sample)
            winner = self.getBMU(sample)
            self.max = max(self.distance_map.max(), self.max)
            d = np.sqrt(self.distance_map/self.max)
            sigma = self.elasticity*d[winner]
            #print(sigma)
            G = self.Gaussian(self.distance_map.shape, winner, sigma)
            G = np.nan_to_num(G)
            delta = (self.grid - sample)
            for i in range(self.grid.shape[-1]):
                self.grid[...,i] -= self.learning_rate*d*G*delta[...,i]
            self.current_time = self.current_time + 1
            
            #print(self.current_time)
        if color == 4:
            self.im.set_array(self.grid)

    def init(self):
        self.im.set_data(self.grid)
        return [self.im]

    def animateSOMDynamic(self,k):
        inside = False
        if k < self.epoch:
            inside = True
            sample = self.initNodeColor(self.j)
            winner = self.getBMU(sample)
            self.max = max(self.distance_map.max(), self.max)
            d = np.sqrt(self.distance_map/self.max)
            sigma = self.elasticity*d[winner]
            G = self.Gaussian(self.distance_map.shape, winner, sigma)
            G = np.nan_to_num(G)
            delta = (self.grid - sample)
            for i in range(self.grid.shape[-1]):
                self.grid[...,i] -= self.learning_rate*d*G*delta[...,i]
            self.im.set_array(self.grid)
        
        if self.j == 7 and not inside:
            self.j = 0
        if self.j < 7 and not inside:
            self.j += 1
        return [self.im]

    def saveSOM(self,name):
        if os.path.exists(name):
            os.remove(name)
        tmp = self.getNumpySOM()
        np.save(name,tmp)
    
    def loadSOM(self,name):
        if os.path.exists(name):
            dat = np.load(name)
            self.setNumpySOM(dat)
            t = self.getNumpySOM()
            self.im.set_array(t)
        else:
            print("file doesn't exist")

    #build datas for pitch roll and grasp
    def build_dataset(self):
        file = open("dataset.txt","w+")
        for i in range(5,100,5):
            for j in range(5,100,5):
                for k in range(0,2):
                    dat = str(i/100) +" "+ str(j/100)+" "+ str(k)
                    #tmp = str(dat)
                    file.write(dat)
                    file.write("\n")
        file.close()

    def load_dataset(self):
        datas = []
        f_open = open("dataset.txt","r")
        for line in f_open:
            arr = line.split()
            tmp = [float(arr[0]),float(arr[1]),float(arr[2])]
            datas.append(tmp)
        f_open.close()
        random.shuffle(datas)

        return datas



if __name__ == "__main__":
    #name = "som.npy"
    som = DSom(3,100,500)

    #som.build_dataset()
    #som.load_dataset()
    som.init_grid("fixed")
    #som.printSOM()
    
    #som.saveSOM("init")
    #som.loadSOM("init.npy")
    #som.trainSOMColor(1)
    #som.trainSOMColor(2)
    #som.trainSOMColor(3)
    #som.trainSOMColor(4)
    #som.trainSOMColor(5)
    #som.trainSOMColor(6)
    #som.trainSOMColor(7)
    #som.defineClusters()
    #som.printClusters()
    #som.saveSOM("simple_50_som")
    #som.loadSOM("trained_dataset_5.npy")
    anim = animation.FuncAnimation(som.fig, som.animateSOMDynamic, init_func=som.init,frames=501, interval=1, blit=True)
    #som.getOneDimensionalData()
    #som.getCluster()
    plt.show()
    while not rospy.is_shutdown():
        rospy.spin()
