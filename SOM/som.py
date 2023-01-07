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

class Node(object):
    def __init__(self,num_features):
        super(Node, self).__init__()
        self.x = 0
        self.y = 0
        self.num_features = num_features
        self.weights = np.zeros((1,num_features))

    def initNodeRnd(self):
        for i in range(0,self.weights.shape[1]):
            self.weights[0,i] = random.random()

    def initNodeValues(self,data):
        self.weights[0,0] = data[0]
        self.weights[0,1] = data[1]
        self.weights[0,2] = data[2]

    def initNodeColor(self,number):
        if number == 0:
            self.weights[0,0] = 0
            self.weights[0,1] = 0
            self.weights[0,2] = 0
        if number == 1:
            self.weights[0,0] = 1
            self.weights[0,1] = 0
            self.weights[0,2] = 0
        if number == 2:
            self.weights[0,0] = 0
            self.weights[0,1] = 1
            self.weights[0,2] = 0
        if number == 3:
            self.weights[0,0] = 0
            self.weights[0,1] = 0
            self.weights[0,2] = 1
        if number == 4:
            self.weights[0,0] = 1
            self.weights[0,1] = 1
            self.weights[0,2] = 0
        if number == 5:
            self.weights[0,0] = 0
            self.weights[0,1] = 1
            self.weights[0,2] = 1
        if number == 6:
            self.weights[0,0] = 1
            self.weights[0,1] = 0
            self.weights[0,2] = 1
        if number == 7:
            self.weights[0,0] = 1
            self.weights[0,1] = 1
            self.weights[0,2] = 1

    def getXofLattice(self):
        return self.x

    def getYofLattice(self):
        return self.y

    def getDistance(self,input):
        dist = 0
        for i in range(0,self.weights.shape[1]):
            dist += (input[0,i] - self.weights[0,i]) ** 2

        return math.sqrt(dist)

    def getWeights(self):
        return self.weights

    def setWeights(self,datas):
        self.weights[0,0] = datas[0]
        self.weights[0,1] = datas[1]
        self.weights[0,2] = datas[2]

    def initNodeCoor(self,i,j):
        self.x = i
        self.y = j

    def adjustWeights(self,node,lr,infl):
        n = Node(self.num_features)
        n = node
        tmp = n.getWeights()
        for i in range(0,self.weights.shape[1]):
            self.weights[0,i] += lr * (tmp[0,i] - self.weights[0,i]) * infl


    def printNode(self):
        print(self.weights,end="")


class Som(object):
    def __init__(self,num_features,s,ep):
        super(Som, self).__init__()
        rospy.init_node('som', anonymous=True)
        rospy.Subscriber("/som/node_value", Pose, self.callbackNode)
        self.pub_node = rospy.Publisher('/som/action', Pose, queue_size=1)
        self.num_features = num_features
        self.size = s
        self.epoch = ep
        self.map_radius = self.size/2
        self.lamda = self.epoch/math.log(self.map_radius) 
        self.learning_rate = 0.1
        self.current_time = 0
        self.network = [] 
        self.bmu = Node(num_features)
        self.neighbour_rad = -1.0
        self.influence = 0
        self.current_time = 0
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-1, s), ylim=(-1, s))
        self.map = np.random.random((s, s, num_features))
        self.im = plt.imshow(self.map,interpolation='none')
        self.cluster_map = np.zeros((self.size,self.size,1))

    def callbackNode(self,msg):
        tmp = self.getWeightsNode(int(msg.position.x),int(msg.position.y))
        data = geometry_msgs.msg.Pose()
        data.position.x = tmp[0,0]
        data.position.y = tmp[0,1]
        data.position.z = tmp[0,2]
        self.pub_node.publish(data)


    def init_network(self):
        for i in range(self.size):
            tmp_l = []
            for j in range(self.size):
                tmp = Node(self.num_features)
                tmp.initNodeCoor(i,j)
                tmp.initNodeRnd()
                tmp_l.append(tmp)
            self.network.append(tmp_l)

    def printSOM(self):
        for i in range(self.size):
            for j in range(self.size):
                self.network[i][j].printNode()
            print("")

    def getNumpySOM(self):
        tmp = np.zeros((self.size,self.size,self.num_features))
        for i in range(0,tmp.shape[0]):
            for j in range(0,tmp.shape[1]):
                tmp[i,j] = self.network[i][j].getWeights()

        return tmp

    def setNumpySOM(self,datas):
        for i in range(0,datas.shape[0]):
            for j in range(0,datas.shape[1]):
                self.network[i][j].setWeights(datas[i,j])


    def getBestMatchUnit(self,node):
        best = 1000
        last = 1000
        tmp_i = -1
        tmp_j = -1
        for i in range(0,self.size):
            for j in range(0,self.size):
                last = self.network[i][j].getDistance(node.getWeights())
                if last < best:
                    best = last
                    tmp_i = i
                    tmp_j = j
        self.bmu = self.network[tmp_i][tmp_j]

        return self.bmu

    def getWeightsNode(self,x,y):
        return self.network[x][y].getWeights()

    def neighbourRadius(self,iter_count):
        self.neighbour_rad = self.map_radius * math.exp(-iter_count/self.lamda)

    def calculateNewWeights(self,node):
        for i in range(0,self.size):
            for j in range(0,self.size):
                dist_node = ((self.bmu.getXofLattice()-self.network[i][j].getXofLattice())**2 \
                    + (self.bmu.getYofLattice()-self.network[i][j].getYofLattice())**2)
                widthsq = self.neighbour_rad **2
                if dist_node < (self.neighbour_rad**2):
                    self.influence = math.exp(-dist_node / (2*widthsq))
                    self.network[i][j].adjustWeights(node,self.learning_rate,self.influence)

    def reduceLR(self,iter_count):
        self.learning_rate = 0.1 * math.exp(-iter_count/self.lamda)

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


    def trainSOMDataset(self):
        dat = self.load_dataset()
        s_dat = len(dat)
        j = 0
        while self.current_time < self.epoch:
            n = Node(self.num_features)
            if j < s_dat:
                n.initNodeValues(dat[j])
                j += 1
            if j >= s_dat:
                j = 0
            tmp = self.getBestMatchUnit(n)
            self.calculateNewWeights(n)
            self.neighbourRadius(self.current_time)
            self.reduceLR(self.current_time)
            self.current_time = self.current_time + 1
            a = self.getNumpySOM()
            self.im.set_array(a)

    def trainSOMColor(self):
        j = 1
        while self.current_time < self.epoch:
            n = Node(self.num_features)
            n.initNodeColor(j)
            tmp = self.getBestMatchUnit(n)
            self.calculateNewWeights(n)
            self.neighbourRadius(self.current_time)
            self.reduceLR(self.current_time)
            self.current_time = self.current_time + 1
            a = self.getNumpySOM()
            self.im.set_array(a)
            j += 1
            if j == 6:
                j = 1

    def init(self):
        self.im.set_data(self.getNumpySOM())
        return [self.im]

    def animateSOM(self,i):
        #a = self.im.get_array()
        dat = self.load_dataset()
        s_dat = len(dat)
        j = 0
        if i < self.epoch:
            n = Node(self.num_features)
            if j < s_dat:
                n.initNodeValues(dat[j])
                j += 1
            if j >= s_dat:
                j = 0
            self.getBestMatchUnit(n)
            self.calculateNewWeights(n)
            self.neighbourRadius(i)
            self.reduceLR(i)
            print(self.neighbour_rad)
        a = self.getNumpySOM()
        self.im.set_array(a)
        #print(i)
        return [self.im]

    def getOneDimensionalData(self):
        tot = np.array([])
        for i in range(0,self.size):
            for j in range(0,self.size):
                if i == 0 and j == 0:
                    tmp = self.network[i][j].getWeights()
                    tot = tmp
                else:
                    tmp = self.network[i][j].getWeights()
                    tot = np.vstack((tot,tmp))
        return tot

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
    som = Som(3,20,100)
    #som.build_dataset()
    #som.load_dataset()
    som.init_network()
    #som.loadSOM("simple_50_som.npy")

    som.trainSOMColor()
    som.defineClusters()
    som.printClusters()
    #som.saveSOM("simple_50_som")
    #som.loadSOM("trained_dataset_5.npy")
    #anim = animation.FuncAnimation(som.fig, som.animateSOM, init_func=som.init,frames=1000, interval=1, blit=True)
    #som.getOneDimensionalData()
    #som.getCluster()
    plt.show()
    while not rospy.is_shutdown():
        rospy.spin()
