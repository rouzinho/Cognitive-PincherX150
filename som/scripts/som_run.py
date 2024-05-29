#! /usr/bin/python3
import numpy as np
import rospy
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
import random
import math
import time
import matplotlib.pyplot as plt
from matplotlib import animation
import os.path
from scipy.cluster.hierarchy import dendrogram, linkage
import geometry_msgs.msg
from som.msg import GripperOrientation
from som.msg import VectorAction
from som.msg import ListPeaks
from som.msg import ListPose
from som.srv import *
from motion.msg import Dmp
from sklearn.preprocessing import MinMaxScaler


class Node(object):
    def __init__(self,num_features):
        super(Node, self).__init__()
        self.x = 0
        self.y = 0
        self.num_features = num_features
        self.weights = np.zeros((1,num_features))
        self.list_pitch = [0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6]
        self.list_roll = [-1.5,-1.2,-0.9,-0.6,-0.3,0,0.3,0.6,0.9,1.2,1.5]
        self.list_touch = [0,1]

    def initNodeRnd(self):
        for i in range(0,self.weights.shape[1]):
            self.weights[0,i] = random.random()

    def initNodeRndPose(self):
        self.weights[0,0] = random.uniform(0.05,0.4)
        self.weights[0,1] = random.uniform(-0.4,0.4)
        self.weights[0,2] = random.choice(self.list_pitch)

    def initNodeRndAction(self):
        self.weights[0,0] = random.uniform(-0.15,0.15)
        self.weights[0,1] = random.uniform(-0.15,0.15)
        self.weights[0,2] = random.choice(self.list_roll)
        self.weights[0,3] = random.choice(self.list_touch)

    def initNodeStaticPose(self):
        self.weights[0,0] = 0.5
        self.weights[0,1] = 0.5
        self.weights[0,2] = 0.5

    def initNodeStaticAction(self):
        self.weights[0,0] = 0.5
        self.weights[0,1] = 0.5
        self.weights[0,2] = 0.5
        self.weights[0,3] = 0.5

    def initNodeValues(self,data):
        self.weights[0,0] = data[0]
        self.weights[0,1] = data[1]
        self.weights[0,2] = data[2]

    def initNodeData(self,data):
        for i in range(0,len(data)):
            self.weights[0,i] = data[i]

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
        for i in range(0,len(datas)):
            self.weights[0,i] = datas[i]

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
    def __init__(self,name,num_features,s,ep,mode):
        super(Som, self).__init__()
        #rospy.init_node("som", anonymous=True)
        n_sub = name + "node_coord"
        rospy.Service('set_pose', GetPoses, self.setup_poses)
        rospy.Subscriber(n_sub, Point, self.callbackNode)
        rospy.Subscriber('/cog_learning/exploitation', Float64, self.callback_exploitation)
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
        self.mode = mode
        self.exploit = False
        self.list_coords = []
        if self.mode == "motion":
            self.pub_node = rospy.Publisher('/motion_pincher/vector_action', VectorAction, queue_size=1)
        else:
            n_bmu = name + "node_value/bmu"
            ni_peaks = name + "input_list_peaks"
            no_peaks = name + "output_list_peaks"
            in_path = name + "dmp_path"
            rospy.Subscriber(n_bmu, GripperOrientation, self.callback_bmu)
            rospy.Subscriber(ni_peaks, ListPeaks, self.callback_list_peaks)
            #rospy.Subscriber(in_path, ListPose, self.callback_list_pose)
            self.pub_bmu = rospy.Publisher('/motion_pincher/gripper_orientation/bmu_last_pose', GripperOrientation, queue_size=1)
            self.pub_node = rospy.Publisher('/motion_pincher/gripper_orientation/first_pose', GripperOrientation, queue_size=1)
            self.pub_peaks = rospy.Publisher(no_peaks, ListPeaks, queue_size=1)

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-1, s), ylim=(-1, s))
        self.map = np.random.random((s, s, num_features))
        self.im = plt.imshow(self.map,interpolation='none')
        self.cluster_map = np.zeros((self.size,self.size,1))

    def setup_poses(self,req):
        list_c = self.get_list_peaks(req.pitch)
        #print("coords : ",list_c)
        l = ListPeaks()
        for i in list_c:
            p = Point()
            p.x = i[0]
            p.y = i[1] 
            l.list_peaks.append(p)
        self.pub_peaks.publish(l)
        res = GetPosesResponse()
        if len(list_c) > 0:
            res.success = True
        else:
            res.success = False
        
        return res

    def callback_exploitation(self,msg):
        if msg.data > 0.5:
            self.exploit = True
        else:
            self.exploit = False

    def callback_bmu(self,msg):
        data = [msg.x,msg.y,msg.pitch]
        n = Node(self.num_features)
        n.setWeights(data)
        bmu = self.get_bmu(n)
        dat_bmu = bmu.getWeights()
        go = GripperOrientation()
        go.x = dat_bmu[0,0]
        go.y = dat_bmu[0,1]
        go.pitch = dat_bmu[0,2]
        self.pub_bmu.publish(go)

    def callbackNode(self,msg):
        tmp = self.get_weights_node(int(msg.x),int(msg.y))
        #print("node index x :",int(msg.x))
        #print("node index y :",int(msg.y))
        #print("node values : ",tmp)
        if self.mode == "motion":
            va = VectorAction()
            va.x = tmp[0,0]
            va.y = tmp[0,1]
            va.roll = tmp[0,2]
            va.grasp = tmp[0,3]
            self.pub_node.publish(va)
        else:
            go = GripperOrientation()
            go.x = tmp[0,0]
            go.y = tmp[0,1]
            go.pitch = tmp[0,2]
            self.pub_node.publish(go)

    def callback_list_peaks(self,msg):
        #print(msg)
        self.list_peaks(msg)
        #print("list peaks : ",self.list_coords)
        #print(l_peaks)
        l = ListPeaks()
        for i in self.list_coords:
            p = Point()
            p.x = i[0]
            p.y = i[1] 
            l.list_peaks.append(p)
        #print(l.list_peaks)
        self.pub_peaks.publish(l)

    def list_peaks(self,data):
        self.list_coords = []
        for sample in data.list_peaks:
            for i in range(0,self.size):
                for j in range(0,self.size):
                    val = self.network[i][j].getWeights()
                    #print("val ",val)
                    dist = math.sqrt(pow(val[0,0] - sample.x,2)+pow(val[0,1] - sample.y,2))
                    if dist < 0.01:
                        #print("val ",val[0,2])
                        coords = [i,j]
                        res = self.check_list(self.list_coords,coords)
                        if not res:
                            self.list_coords.append(coords)
    
    def get_list_peaks(self,pitch):
        list_good_coords = []
        #print("list peaks : ",self.list_coords)
        for sample in self.list_coords:
            val = self.network[sample[0]][sample[1]].getWeights()
            dist = abs(val[0,2]-pitch)
            if dist <= 0.2:
                #print("val ",val[0,2])
                res = self.check_list(list_good_coords,sample)
                if not res:
                    list_good_coords.append(sample)
                    go = GripperOrientation()
                    
                        
        return list_good_coords
    
    def check_list(self,l,sample):
        inside = False
        for i in l:
            if i[0] == sample[0] and i[1] == sample[1]:
                inside = True
                #print(inside)

        return inside


    def init_network(self):
        for i in range(self.size):
            tmp_l = []
            for j in range(self.size):
                tmp = Node(self.num_features)
                tmp.initNodeCoor(i,j)
                tmp.initNodeRnd()
                tmp_l.append(tmp)
            self.network.append(tmp_l)

    def init_network_som_pose(self):
        for i in range(self.size):
            tmp_l = []
            for j in range(self.size):
                tmp = Node(self.num_features)
                tmp.initNodeCoor(i,j)
                #tmp.initNodeRndPose()
                tmp.initNodeStaticPose()
                tmp_l.append(tmp)
            self.network.append(tmp_l)

    def init_network_som_action(self):
        for i in range(self.size):
            tmp_l = []
            for j in range(self.size):
                tmp = Node(self.num_features)
                tmp.initNodeCoor(i,j)
                #tmp.initNodeRndAction()
                tmp.initNodeStaticAction()
                tmp_l.append(tmp)
            self.network.append(tmp_l)

    def print_som(self):
        for i in range(self.size):
            for j in range(self.size):
                self.network[i][j].printNode()
            print("")

    def get_numpy_som(self):
        tmp = np.zeros((self.size,self.size,self.num_features))
        for i in range(0,tmp.shape[0]):
            for j in range(0,tmp.shape[1]):
                tmp[i,j] = self.network[i][j].getWeights()

        return tmp
    
    def get_pose_numpy_som(self):
        tmp = np.zeros((self.size,self.size,self.num_features))
        for i in range(0,tmp.shape[0]):
            for j in range(0,tmp.shape[1]):
                w = self.network[i][j].getWeights()
                scaled = self.pose_to_color(w[0])
                tmp[i,j] = scaled

        return tmp
    
    def get_action_numpy_som(self):
        tmp = np.zeros((self.size,self.size,self.num_features))
        for i in range(0,tmp.shape[0]):
            for j in range(0,tmp.shape[1]):
                w = self.network[i][j].getWeights()
                scaled = self.action_to_color(w[0])
                #for display
                #if scaled[3] < 0.5:
                #    scaled[3] = 0.3
                tmp[i,j] = scaled

        return tmp
    
    def pose_to_color(self,w):
        n_x = np.array(w[0])
        n_x = n_x.reshape(-1,1)
        n_y = np.array(w[1])
        n_y = n_y.reshape(-1,1)
        n_p = np.array(w[2])
        n_p = n_p.reshape(-1,1)
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_p = MinMaxScaler()
        x_minmax = np.array([0.18, 0.45])
        y_minmax = np.array([-0.35, 0.32])
        p_minmax = np.array([0, 1.6])
        scaler_x.fit(x_minmax[:, np.newaxis])
        scaler_y.fit(y_minmax[:, np.newaxis])
        scaler_p.fit(p_minmax[:, np.newaxis])
        n_x = scaler_x.transform(n_x)
        n_x = n_x.reshape(1,-1)
        n_x = n_x.flatten()
        n_y = scaler_y.transform(n_y)
        n_y = n_y.reshape(1,-1)
        n_y = n_y.flatten()
        n_p = scaler_p.transform(n_p)
        n_p = n_p.reshape(1,-1)
        n_p = n_p.flatten()
        scaled_w = [n_x[0],n_y[0],n_p[0]]

        return scaled_w
    
    def color_to_pose(self,c):
        n_x = np.array(c[0])
        n_x = n_x.reshape(-1,1)
        n_y = np.array(c[1])
        n_y = n_y.reshape(-1,1)
        n_p = np.array(c[2])
        n_p = n_p.reshape(-1,1)
        scaler_x = MinMaxScaler(feature_range=(0.18, 0.45))
        scaler_y = MinMaxScaler(feature_range=(-0.35, 0.32))
        scaler_p = MinMaxScaler(feature_range=(0, 1.6))
        x_minmax = np.array([0, 1])
        y_minmax = np.array([0, 1])
        p_minmax = np.array([0, 1])
        scaler_x.fit(x_minmax[:, np.newaxis])
        scaler_y.fit(y_minmax[:, np.newaxis])
        scaler_p.fit(p_minmax[:, np.newaxis])
        n_x = scaler_x.transform(n_x)
        n_x = n_x.reshape(1,-1)
        n_x = n_x.flatten()
        n_y = scaler_y.transform(n_y)
        n_y = n_y.reshape(1,-1)
        n_y = n_y.flatten()
        n_p = scaler_p.transform(n_p)
        n_p = n_p.reshape(1,-1)
        n_p = n_p.flatten()
        scaled_w = [n_x[0],n_y[0],n_p[0]]

        return scaled_w
    
    def action_to_color(self,w):
        n_x = np.array(w[0])
        n_x = n_x.reshape(-1,1)
        n_y = np.array(w[1])
        n_y = n_y.reshape(-1,1)
        n_r = np.array(w[2])
        n_r = n_r.reshape(-1,1)
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_r = MinMaxScaler()
        x_minmax = np.array([-0.15, 0.15])
        y_minmax = np.array([-0.15, 0.15])
        r_minmax = np.array([-1.5, 1.5])
        scaler_x.fit(x_minmax[:, np.newaxis])
        scaler_y.fit(y_minmax[:, np.newaxis])
        scaler_r.fit(r_minmax[:, np.newaxis])
        n_x = scaler_x.transform(n_x)
        n_x = n_x.reshape(1,-1)
        n_x = n_x.flatten()
        n_y = scaler_y.transform(n_y)
        n_y = n_y.reshape(1,-1)
        n_y = n_y.flatten()
        n_r = scaler_r.transform(n_r)
        n_r = n_r.reshape(1,-1)
        n_r = n_r.flatten()
        scaled_w = [n_x[0],n_y[0],n_r[0],w[3]]

        return scaled_w
    
    def color_to_action(self,c):
        n_x = np.array(c[0])
        n_x = n_x.reshape(-1,1)
        n_y = np.array(c[1])
        n_y = n_y.reshape(-1,1)
        n_r = np.array(c[2])
        n_r = n_r.reshape(-1,1)
        scaler_x = MinMaxScaler(feature_range=(-0.15, 0.15))
        scaler_y = MinMaxScaler(feature_range=(-0.15, 0.15))
        scaler_r = MinMaxScaler(feature_range=(-1.5, 1.5))
        x_minmax = np.array([0, 1])
        y_minmax = np.array([0, 1])
        r_minmax = np.array([0, 1])
        scaler_x.fit(x_minmax[:, np.newaxis])
        scaler_y.fit(y_minmax[:, np.newaxis])
        scaler_r.fit(r_minmax[:, np.newaxis])
        n_x = scaler_x.transform(n_x)
        n_x = n_x.reshape(1,-1)
        n_x = n_x.flatten()
        n_y = scaler_y.transform(n_y)
        n_y = n_y.reshape(1,-1)
        n_y = n_y.flatten()
        n_r = scaler_r.transform(n_r)
        n_r = n_r.reshape(1,-1)
        n_r = n_r.flatten()
        scaled_w = [n_x[0],n_y[0],n_r[0],c[3]]

        return scaled_w

    def set_pose_numpy_som(self,datas):
        for i in range(0,datas.shape[0]):
            for j in range(0,datas.shape[1]):
                tmp = datas[i,j]
                w = self.color_to_pose(tmp)
                self.network[i][j].setWeights(w)

    def set_action_numpy_som(self,datas):
        for i in range(0,datas.shape[0]):
            for j in range(0,datas.shape[1]):
                tmp = datas[i,j]
                w = self.color_to_action(tmp)
                self.network[i][j].setWeights(w)

    def set_numpy_som(self,datas):
        for i in range(0,datas.shape[0]):
            for j in range(0,datas.shape[1]):
                self.network[i][j].setWeights(datas[i,j])


    def get_bmu(self,node):
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

    def get_weights_node(self,x,y):
        return self.network[x][y].getWeights()

    def neighbour_radius(self,iter_count):
        self.neighbour_rad = self.map_radius * math.exp(-iter_count/self.lamda)

    def compute_new_weights(self,node):
        for i in range(0,self.size):
            for j in range(0,self.size):
                dist_node = ((self.bmu.getXofLattice()-self.network[i][j].getXofLattice())**2 \
                    + (self.bmu.getYofLattice()-self.network[i][j].getYofLattice())**2)
                widthsq = self.neighbour_rad **2
                if dist_node < (self.neighbour_rad**2):
                    self.influence = math.exp(-dist_node / (2*widthsq))
                    self.network[i][j].adjustWeights(node,self.learning_rate,self.influence)

    def reduce_lr(self,iter_count):
        self.learning_rate = 0.1 * math.exp(-iter_count/self.lamda)

    def define_clusters(self):
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
                    

    def print_clusters(self):
        for i in range(0,self.cluster_map.shape[0]):
            for j in range(0,self.cluster_map.shape[1]):
                print(self.cluster_map[i,j],end = "")
            print("")


    def train_som_dataset_pose(self,name_ds):
        dat = self.load_dataset_pose(name_ds)
        s_dat = len(dat)
        j = 0
        while self.current_time < self.epoch:
            n = Node(self.num_features)
            if j < s_dat:
                n.initNodeData(dat[j])
                j += 1
            if j >= s_dat:
                j = 0
            tmp = self.get_bmu(n)
            self.compute_new_weights(n)
            self.neighbour_radius(self.current_time)
            self.reduce_lr(self.current_time)
            self.current_time = self.current_time + 1
            print(self.current_time)
        a = self.get_pose_numpy_som()
        self.im.set_array(a)

    def train_som_dataset_outcome(self,name_ds):
        dat = self.load_dataset_outcome(name_ds)
        s_dat = len(dat)
        j = 0
        while self.current_time < self.epoch:
            n = Node(self.num_features)
            if j < s_dat:
                n.initNodeData(dat[j])
                j += 1
            if j >= s_dat:
                j = 0
            tmp = self.get_bmu(n)
            self.compute_new_weights(n)
            self.neighbour_radius(self.current_time)
            self.reduce_lr(self.current_time)
            self.current_time = self.current_time + 1
            print(self.current_time)
        a = self.get_numpy_som()
        self.im.set_array(a)
        
    def arrange2D(self):
        x = []
        y = []
        for i in range(0,self.size):
            for j in range(0,self.size):
                tmp = self.network[i][j].getWeights()
                tmp = tmp[0]
                x.append(tmp[0])
                y.append(tmp[1])
        return x, y

    def init(self):
        self.im.set_data(self.get_numpy_som())
        return [self.im]

    def get_one_dimensional_data(self):
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

    def save_som_pose(self,name):
        if os.path.exists(name):
            os.remove(name)
        tmp = self.get_pose_numpy_som()
        np.save(name,tmp)

    def save_som_outcome(self,name):
        if os.path.exists(name):
            os.remove(name)
        tmp = self.get_numpy_som()
        np.save(name,tmp)
    
    def load_som(self,name,mode):
        if os.path.exists(name):
            dat = np.load(name)
            if mode == "position":
                self.set_pose_numpy_som(dat)
                t = self.get_pose_numpy_som()
                self.im.set_array(t)
            else:
                self.set_action_numpy_som(dat)
                t = self.get_action_numpy_som()
                self.im.set_array(t)
                
        else:
            print("file doesn't exist")

    #build datas for pitch roll and grasp
    def build_dataset_pose(self):
        file = open("/home/altair/interbotix_ws/src/som/dataset/dataset_pose.txt","w")
        for i in range(5,100,5):
            for j in range(5,100,5):
                dat = str(i/100) +" "+ str(j/100)
                file.write(dat)
                file.write("\n")
        file.close()

    def build_dataset_outcome(self):
        file = open("/home/altair/interbotix_ws/src/som/dataset/dataset_outcome.txt","w")
        for i in range(5,100,5):
            for j in range(5,100,5):
                for k in range(5,100,5):
                    for l in range(0,2):
                        dat = str(i/100) +" "+ str(j/100) + " " + str(k/100) + " " + str(l) + "\n"
                        file.write(dat)
        file.close()

    def build_dataset_motion(self):
        file = open("/home/altair/interbotix_ws/src/som/dataset/dataset_motion.txt","w")
        x = 0
        y = 0
        r = 0
        t = 0
        for i in range(-15,16,1):
            for j in range(-15,16,1):
                for k in range(-15,16,3):
                    for l in range(0,2):
                        if i != 0 or j != 0:
                            if i == 0:
                                x = 0
                            else:
                                x = i/100
                            if j == 0:
                                y = 0
                            else:
                                y = j/100
                            r = k/10
                            t = l
                            dat = str(x) +" "+ str(y) + " " + str(r) + " " + str(t) + "\n"
                            file.write(dat)
        file.close()

    def load_dataset_pose(self,name):
        datas = []
        f_open = open(name,"r")
        for line in f_open:
            arr = line.split()
            tmp = [float(arr[0]),float(arr[1]),float(arr[2])]
            datas.append(tmp)
        f_open.close()
        random.shuffle(datas)

        return datas

    def load_dataset_outcome(self,name):
        datas = []
        f_open = open(name,"r")
        for line in f_open:
            arr = line.split()
            tmp = [float(arr[0]),float(arr[1]),float(arr[2]),float(arr[3])]
            datas.append(tmp)
        f_open.close()
        random.shuffle(datas)

        return datas



if __name__ == "__main__":
    rospy.init_node("som")
    name_dataset = ""  
    ns = rospy.get_namespace()
    name_init = ns + "som/"
    training = rospy.get_param(name_init+"train_som")
    data_set = rospy.get_param(name_init+"dataset")
    ep = rospy.get_param(name_init+"epochs")
    model_name = rospy.get_param(name_init+"model")
    size_map = rospy.get_param(name_init+"size")
    num_feat = rospy.get_param(name_init+"num_feat")
    if data_set == "motion":
        name_dataset = "/home/altair/interbotix_ws/src/som/dataset/dataset_motion.txt"  
    if data_set == "position":
        name_dataset = "/home/altair/interbotix_ws/src/som/dataset/dataset_positions.txt"
    if data_set == "outcome":
        name_dataset = "/home/altair/interbotix_ws/src/som/dataset/dataset_outcome.txt"
    som = Som(name_init,num_feat,size_map,ep,data_set)
    
    #srv_path = rospy.Service('get_path', GetPath, som.optimal_path)
    #som.load_som("simple_50_som.npy")
    if training == True and data_set == "motion":
        som.init_network_som_action()
        #som.build_dataset_motion()
        som.train_som_dataset_outcome(name_dataset)
        som.save_som_outcome("/home/altair/interbotix_ws/src/som/models/model_actions.npy")
    if training == True and data_set == "position":
        som.init_network_som_pose()
        #som.build_dataset_pose()
        som.train_som_dataset_pose(name_dataset)
        som.save_som_pose("/home/altair/interbotix_ws/src/som/models/model_pose_opt2.npy")
        #som.print_som()
    if training == True and data_set == "outcome":
        som.init_network_som_action()
        #som.build_dataset_outcome()
        som.train_som_dataset_outcome(name_dataset)
        som.save_som_outcome("/home/altair/interbotix_ws/src/som/models/model_outcome.npy")
        #som.print_som()
    if training == False and data_set == "motion":
        som.init_network_som_action()
        som.load_som(model_name,data_set)
        #som.print_som()
    if training == False and data_set == "outcome":
        som.init_network_som_action()
        som.load_som(model_name,data_set)
    if training == False and data_set == "position":
        som.init_network_som_pose()
        som.load_som(model_name,data_set)
    print("SOM READY")
    #plt.show()
    #while not rospy.is_shutdown():
    #    pass
    rospy.spin()
