#! /usr/bin/python3
import numpy as np
import rospy
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
import random
import math
import time
import matplotlib.pyplot as plt
from matplotlib import animation
import os.path
from scipy.cluster.hierarchy import dendrogram, linkage
import geometry_msgs.msg
from motion.msg import GripperOrientation
from motion.msg import VectorAction
from som.msg import ListPeaks
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
        rospy.init_node("som", anonymous=True)
        n_sub = name + "node_coord"
        rospy.Subscriber(n_sub, Point, self.callbackNode)
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
        if self.mode == "motion":
            self.pub_node = rospy.Publisher('/motion_pincher/vector_action', VectorAction, queue_size=1)
        else:
            n_bmu = name + "node_values"
            ni_peaks = name + "input_list_peaks"
            no_peaks = name + "output_list_peaks"
            #rospy.Subscriber(n_bmu, Point, self.callback_bmus)
            rospy.Subscriber(ni_peaks, ListPeaks, self.callback_list_peaks)
            self.pub_node = rospy.Publisher('/motion_pincher/gripper_orientation', GripperOrientation, queue_size=1)
            self.pub_peaks = rospy.Publisher(no_peaks, ListPeaks, queue_size=1)

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-1, s), ylim=(-1, s))
        self.map = np.random.random((s, s, num_features))
        self.im = plt.imshow(self.map,interpolation='none')
        self.cluster_map = np.zeros((self.size,self.size,1))

    def callbackNode(self,msg):
        tmp = self.get_weights_node(int(msg.x),int(msg.y))
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
        l_peaks = self.list_peaks(msg)
        l = ListPeaks()
        for i in l_peaks:
            p = Point()
            p.x = i[0]
            p.y = i[1] 
            l.list_peaks.append(p)
        self.pub_peaks.publish(l)

    def list_peaks(self,data):
        list_coords = []
        for sample in data.list_peaks:
            for i in range(0,self.size):
                for j in range(0,self.size):
                    val = self.network[i][j].getWeights()
                    if abs(val[0,0] - sample.x) < 0.005 and abs(val[0,1] - sample.y) < 0.005:
                        coords = [i,j]
                        list_coords.append(coords)
        
        return list_coords

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
                tmp.initNodeRndPose()
                tmp_l.append(tmp)
            self.network.append(tmp_l)

    def init_network_som_action(self):
        for i in range(self.size):
            tmp_l = []
            for j in range(self.size):
                tmp = Node(self.num_features)
                tmp.initNodeCoor(i,j)
                tmp.initNodeRndAction()
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
        x_minmax = np.array([0.05, 0.4])
        y_minmax = np.array([-0.4, 0.4])
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
        scaler_x = MinMaxScaler(feature_range=(0.05, 0.4))
        scaler_y = MinMaxScaler(feature_range=(-0.4, 0.4))
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

    def train_som_dataset_motion(self,name_ds):
        dat = self.load_dataset_motion(name_ds)
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
        a = self.get_action_numpy_som()
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

    def save_som_action(self,name):
        if os.path.exists(name):
            os.remove(name)
        tmp = self.get_action_numpy_som()
        np.save(name,tmp)
    
    def load_som(self,name,mode):
        if os.path.exists(name):
            dat = np.load(name)
            if mode == "pose":
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

    def load_dataset_motion(self,name):
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
    if data_set == "pose":
        name_dataset = "/home/altair/interbotix_ws/src/som/dataset/dataset_pose.txt"
    som = Som(name_init,num_feat,size_map,ep,data_set)
    #som.load_som("simple_50_som.npy")
    if training == True and data_set == "motion":
        som.init_network_som_action()
        #som.build_dataset_motion()
        som.train_som_dataset_motion(name_dataset)
        som.save_som_action("/home/altair/interbotix_ws/src/som/models/model_actions.npy")
    if training == True and data_set == "pose":
        som.init_network_som_pose()
        #som.build_dataset_pose()
        som.train_som_dataset_pose(name_dataset)
        som.save_som_pose("/home/altair/interbotix_ws/src/som/models/model_pose.npy")
        #som.print_som()
    if training == False and data_set == "pose":
        som.init_network_som_pose()
        som.load_som(model_name,data_set)
        #som.print_som()
    if training == False and data_set == "motion":
        som.init_network_som_action()
        som.load_som(model_name,data_set)
        #som.print_som()
    plt.show()
    while not rospy.is_shutdown():
        pass
    rospy.spin()
