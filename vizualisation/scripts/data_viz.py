#!/usr/bin/env python3
from ossaudiodev import control_labels
import kivy
#kivy.require('2.0.0') # replace with your current kivy version !
import rospy
import copy
# ROS Image message
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from std_msgs.msg import Int16
from motion.msg import Dmp
from detector.msg import Outcome
import geometry_msgs.msg
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
#from hebbian_dmp.msg import ObjectGoal
from geometry_msgs.msg import Pose
#from perception.msg import ListGoals
from cog_learning.msg import Goal
import cv2
from csv import writer
import numpy as np
import math
from kivy.app import App
from vizualisation.circular_progress_bar import *
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.properties import ListProperty, StringProperty
from kivy.uix import image
import pickle
from os.path import exists
import os
from os import listdir
import shutil
import http.client, urllib
from playsound import playsound
from kivy.uix.widget import Widget

_DEFAULT_BACKGROUND_ACTIVE = (255/255, 245/255, 231/255, 0.0)
_BLACK = (0/255, 0/255, 0/255, 0.8)
_DEFAULT_BACKGROUND_NONACTIVE = (255/255, 245/255, 231/255, 1.0)
_DEFAULT_WIDGET_ACTIVE = (1, 1, 1, 0.7)
_DEFAULT_WIDGET_NONACTIVE = (1, 1, 1, 0)
_OBJECT_BLUE = (0.110, 0.427, 0.815, 0.1)
_GREEN_LIGHT = (0.180, 0.690, 0.525, 0.9)
_RED_LIGHT = (0.721, 0.250, 0.368, 1)
_GRAY_LIGHT = (0.26,0.26,0.26,0.3)
_ACTIVE_LIGHT = (48/255,84/255,150/255,1)
_YELLOW_LIGHT = (255/255,185/255,97/255,0.8)


Window.clearcolor = (255/255, 245/255, 231/255, 0)
Window.size = (1400, 800)

class DataRecorder(object):
    def __init__(self, name_topic_error, name_data, rec_inv,time_step):
        super(DataRecorder, self).__init__()
        #rospy.init_node('DataRecorder')   
        self.bridge = CvBridge()
        self.cv2_img = None
        self.name_data = name_data
        self.rec_inv = rec_inv
        self.time_step = time_step
        self.current_field = np.zeros((100,100))
        self.dim_field = [0,0]
        self.init_size = True
        self.size = 10
        self.peaks = []
        self.list_peaks = []
        self.list_tmp_peaks = []
        self.list_inv = []
        self.list_lp = []
        self.prev_id = -1 
        self.id = -1
        self.objects = []
        self.id_defined = False
        self.data_folder = rospy.get_param("data_folder")
        self.sub = rospy.Subscriber(name_topic_error, Image, self.field_callback)
        self.sub_og = rospy.Subscriber("/intrinsic/new_goal", Goal, self.callbackOG)
        self.sub_og = rospy.Subscriber("/intrinsic/inverse_error", Goal, self.callback_inv)
        self.sub_id = rospy.Subscriber("/cog_learning/id_object", Int16, self.callback_object)
        self.sub_time = rospy.Subscriber("/data_recorder/time", Float64, self.callback_time)


    def field_callback(self,msg):
        try:
            # Convert your ROS Image message to OpenCV2
            #print("got field")
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            self.current_field = np.asarray(cv2_img)
        except CvBridgeError as e:
            print(e)

    def callbackOG(self,msg):
        p = [int(msg.x),int(msg.y),0.0]
        if not self.isInlist(p):
            self.list_lp.append(p)
            self.list_peaks.append(p)
            
        #print(self.list_peaks)
            
    def callback_inv(self,msg):
        if self.rec_inv:
            p = [int(msg.x),int(msg.y),msg.value]
            self.isIn_inv_list(p)
            self.list_inv.append(p)

    def callback_object(self,msg):
        self.id = msg.data
        path = os.path.join(self.data_folder, str(self.id))
        found = os.path.isdir(path)
        if not found:
            os.mkdir(path)
        self.id_defined = True

    def callback_time(self,msg):
        time = round(msg.data)
        if not self.time_step:
            if time % 10 == 0:
                self.getValuePeaks()
                self.writeDatas(time)
                if self.rec_inv:
                    self.write_inverse_datas(time)
        else:
            if time % 12 == 0:
                self.getValuePeaks()
                self.writeDatas(time)

    def getValuePeaks(self):
        for i in range(0,len(self.list_peaks)):
            val = self.current_field[self.list_peaks[i][1],self.list_peaks[i][0]]
            self.list_peaks[i][2] = val
        #print("error ",self.list_peaks)

    
    def get_errors(self):
        #print("error ",self.list_peaks)
        return copy.deepcopy(self.list_peaks)

    def isInlist(self,peak):
        inList = False
        for i in range(0,len(self.list_peaks)):
            if self.list_peaks[i][0] <= peak[0] + 2 and self.list_peaks[i][0] >= peak[0] - 2:
                if self.list_peaks[i][1] <= peak[1] + 2 and self.list_peaks[i][1] >= peak[1] - 2:
                    inList = True

        return inList
    
    def isIn_inv_list(self,peak):
        for i in range(0,len(self.list_inv)):
            if self.list_inv[i][0] <= peak[0] + 2 and self.list_inv[i][0] >= peak[0] - 2:
                if self.list_inv[i][1] <= peak[1] + 2 and self.list_inv[i][1] >= peak[1] - 2:
                    self.list_inv.pop(i)

    def setList(self, list_p):
        self.list_tmp_peaks = []
        self.list_tmp_peaks = list_p

    def resetList(self):
        self.list_peaks = []
        
    def resetCurrentField(self):
        self.current_field = np.zeros((100,100))

    def writeDatas(self,time):
        name = self.data_folder + str(self.id) + "/"
        if self.id_defined:
            for i in range(0,len(self.list_peaks)):
                n = name + str(self.list_peaks[i][0])+"_"+str(self.list_peaks[i][1])+"_" + self.name_data + ".csv"
                with open(n, 'a', newline='') as f_object:
                    writer_object = writer(f_object)
                    row = [time,self.list_peaks[i][2]]
                    writer_object.writerow(row)
                    f_object.close()

    def write_inverse_datas(self,time):
        name = self.data_folder + str(self.id) + "/"
        if self.id_defined:
            for i in range(0,len(self.list_inv)):
                n = name + str(self.list_inv[i][0])+"_"+str(self.list_inv[i][1])+"_INV.csv"
                with open(n, 'a', newline='') as f_object:
                    writer_object = writer(f_object)
                    row = [time,self.list_inv[i][2]]
                    writer_object.writerow(row)
                    f_object.close()

    def save_values(self,name_peaks):
        p = exists(name_peaks)
        if p:
            os.remove(name_peaks)
        filehandler = open(name_peaks, 'wb')
        pickle.dump(self.list_peaks, filehandler)

    def save_inverse_values(self,name_peaks):
        p = exists(name_peaks)
        if p:
            os.remove(name_peaks)
        filehandler = open(name_peaks, 'wb')
        pickle.dump(self.list_inv, filehandler)

    def load_values(self,name_peaks):
        print("loading Peaks")
        filehandler = open(name_peaks, 'rb') 
        self.list_peaks = pickle.load(filehandler)
        print(self.list_peaks)

    def load_inverse_values(self,name_peaks):
        print("loading Peaks")
        filehandler = open(name_peaks, 'rb') 
        self.list_inv = pickle.load(filehandler)
        #print(self.list_peaks)

class ControlArch(object):
    def __init__(self):
        super(ControlArch, self).__init__()
        self.pub_reset_memory = rospy.Publisher("/architecture/save_memory",Float64,queue_size=10)
        self.pub_reset_sim = rospy.Publisher("/architecture/reset_sim",Float64,queue_size=10)
        self.pub_unstuck_sim = rospy.Publisher("/architecture/unstuck",Float64,queue_size=10)
        self.pub_start_expl = rospy.Publisher("/architecture/exp",Bool,queue_size=10)
        rospy.Subscriber("/motion_panda/ee_moving", Bool, self.callbackMotion)
        self.is_moving = False
        self.start_expl = False

    def callbackMotion(self,msg):
        self.is_moving = msg.data

    def getIsMoving(self):
        return self.is_moving

    def publishUnstuck(self,val):
        uns = Float64()
        uns.data = val
        self.pub_unstuck_sim.publish(uns)

    def publishExpl(self,val):
        d = Bool()
        d.data = val
        self.pub_start_expl.publish(d)

    def saveMemory(self,val):
        res = Float64()
        res.data = val
        self.pub_reset_memory.publish(res)

    def resetArchitecture(self,val):
        res = Float64()
        res.data = val
        self.pub_reset_sim.publish(res)

class DataNodeRecorder(object):
    def __init__(self, name_topic, name_mode):
        super(DataNodeRecorder, self).__init__()
        self.node = 0.0
        self.name_mode = name_mode 
        self.prev_id = -1 
        self.id = -1
        self.objects = []
        self.id_defined = False
        self.data_folder = rospy.get_param("data_folder")
        self.sub = rospy.Subscriber(name_topic, Float64, self.node_callback)
        self.sub_id = rospy.Subscriber("/cog_learning/id_object", Int16, self.callback_object)
        self.sub_time = rospy.Subscriber("/data_recorder/time", Float64, self.callback_time)

    def node_callback(self, msg):
        self.node = msg.data

    def callback_object(self,msg):
        self.id = msg.data
        path = os.path.join(self.data_folder, str(self.id))
        found = os.path.isdir(path)
        if not found:
            os.mkdir(path)
        self.id_defined = True

    def callback_time(self,msg):
        self.writeValue(msg.data)

    def getNode(self):
        return self.node

    def writeValue(self,time):
        if self.id_defined:
            n = self.data_folder + str(self.id) + "/" + self.name_mode+".csv"
            with open(n, 'a', newline='') as f_object:
                writer_object = writer(f_object)
                row = [time,self.node]
                writer_object.writerow(row)
                f_object.close()

class VisualDatas(App):

    explore_rnd = ListProperty([0.26, 0.26, 0.26, 0.3])
    explore_direct = ListProperty([0.26, 0.26, 0.26, 0.3])
    exploit = ListProperty([0.26, 0.26, 0.26, 0.3])
    #learning_dmp = ListProperty([0.26, 0.26, 0.26, 0.3])
    #retrieve_dmp = ListProperty([0.26, 0.26, 0.26, 0.3])
    start_record = ListProperty([48/255,84/255,150/255,1])
    stop_record = ListProperty([0.26, 0.26, 0.26, 0.3])
    name_record = StringProperty('Start')
    v_x = StringProperty('0')
    v_y = StringProperty('0')
    v_pitch = StringProperty('0')
    roll = StringProperty('0')
    grasp = StringProperty('0')
    out_x = StringProperty('0')
    out_y = StringProperty('0')
    out_angle = StringProperty('0')
    out_touch = StringProperty('0')
    mt_error = StringProperty('/home/altair/PhD/Codes/Experiment-IMVAE/datas/production/datas/blank.jpg')
    mt_vae = StringProperty('/home/altair/PhD/Codes/Experiment-IMVAE/datas/production/datas/blank.jpg')
    mt_lp = StringProperty('/home/altair/PhD/Codes/Experiment-IMVAE/datas/production/datas/blank.jpg')
    mt_inhib = StringProperty('/home/altair/PhD/Codes/Experiment-IMVAE/datas/production/datas/blank.jpg')

    def __init__(self, **kwargs):
        super(VisualDatas, self).__init__(**kwargs)
        self.record = False
        self.time = 0
        self.steps = 0
        rospy.init_node('DataRecorder')
        rate = rospy.Rate(50)
        self.data_folder = rospy.get_param("data_folder")
        self.error = DataRecorder("/cog_learning/mt_error","ERROR",True,False)
        self.lp = DataRecorder("/cog_learning/mt_lp","LP",False,False)
        self.persist = DataRecorder("/cog_learning/persistence","persist",False,True)
        self.node_rnd_explore = DataNodeRecorder("/data_recorder/node_rnd_explore","rnd_exploration")
        self.node_direct_explore = DataNodeRecorder("/data_recorder/node_direct_explore","direct_exploration")
        self.node_exploit = DataNodeRecorder("/data_recorder/node_exploit","exploit")
        #self.node_learning_dmp = DataNodeRecorder("/data_recorder/hebbian","hebbian")
        self.pub_time = rospy.Publisher("/data_recorder/time",Float64,queue_size=1)
        self.pub_pause_dft = rospy.Publisher("/cluster_msg/pause_dft",Bool,queue_size=1)
        self.pub_pause_perception = rospy.Publisher("/cluster_msg/pause_perception",Bool,queue_size=1)
        #self.pub_signal = rospy.Publisher("/data_recorder/signal",Bool,queue_size=1)
        #self.dmp = DmpListener()
        #self.control_arch = ControlArch()
        self.current_dmp = 0
        self.first = True
        self.n_exploit = True
        self.launch = True
        self.name_time = self.data_folder + "time.pkl"
        self.name_peaks = self.data_folder + "peaks.pkl"
        self.name_inv_peaks = self.data_folder + "inv_peaks.pkl"
        self.working_dir = "/home/altair/PhD/Codes/Experiment-IMVAE/datas/production/"
        self.dmp_dir = "/home/altair/PhD/Codes/catkin_noetic/rosbags/experiment/dmp/"
        self.mode_record = "Start"
        self.state_experiment = "running"
        self.index = 1
        self.name_object = "ball"
        self.experiment = "mid_lp"
        self.rec = True
        self.rst = False
        self.last_time = 0
        self.keep_exploring = False
        self.keep_exploit = False
        self.too_many = False
        self.cv2_img = None
        self.cv2_mt = None
        self.cv2_lp = None
        self.cv2_inhib = None
        self.bridge = CvBridge()
        #self.coded_skills = [[67, 79], [38, 72], [42, 85], [67, 51], [56, 73]] 
        self.explicit_skills = ["up","right","grasp","down","left"]
        #self.explicit_skills = ["up","right","grasp"]
        rospy.Subscriber("/cog_learning/mt_error", Image, self.error_callback)
        rospy.Subscriber("/habituation/outcome/mt", Image, self.vae_callback)
        rospy.Subscriber("/cog_learning/mt_lp", Image, self.lp_callback)
        rospy.Subscriber("/habituation/action/mt", Image, self.vae_act_callback)
        rospy.Subscriber("/display/dmp", Dmp, self.action_callback)
        rospy.Subscriber("/outcome_detector/outcome", Outcome, self.outcome_callback)
        rospy.Subscriber("/cluster_msg/pause_experiment", Bool, self.pause_callback)
        self.action_text = ""
        self.count_img = 0

    def error_callback(self,msg):
        upscale = (200, 200)
        if self.count_img > 10:
            try:
                name = self.data_folder + "error.jpg"
                self.cv2_img = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                resized_up = cv2.resize(self.cv2_img, upscale, interpolation= cv2.INTER_LINEAR)
                img = resized_up.astype("float32")*255
                cv2.imwrite(name, img)
            except CvBridgeError as e:
                print(e)
            self.count_img = 0
        self.count_img += 1

    def vae_callback(self,msg):
        upscale = (200, 200)
        if self.count_img > 10:
            try:
                name = self.data_folder + "vae_out.jpg"
                self.cv2_mt = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                resized_up = cv2.resize(self.cv2_mt, upscale, interpolation= cv2.INTER_LINEAR)
                img = resized_up.astype("float32")*255
                cv2.imwrite(name, img)
            except CvBridgeError as e:
                print(e)

    def lp_callback(self,msg):
        upscale = (200, 200)
        if self.count_img > 10:
            try:
                name = self.data_folder + "lp.jpg"
                self.cv2_lp = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                resized_up = cv2.resize(self.cv2_lp, upscale, interpolation= cv2.INTER_LINEAR)
                img = resized_up.astype("float32")*255
                cv2.imwrite(name, img)
            except CvBridgeError as e:
                print(e)

    def vae_act_callback(self,msg):
        upscale = (200, 200)
        if self.count_img > 10:
            try:
                name = self.data_folder + "vae_act.jpg"
                self.cv2_inhib = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                resized_up = cv2.resize(self.cv2_inhib, upscale, interpolation= cv2.INTER_LINEAR)
                img = resized_up.astype("float32")*255
                cv2.imwrite(name, img)
            except CvBridgeError as e:
                print(e)

    def action_callback(self,msg):
        #print("KIVY got action !")
        self.v_x = "v_x : " + str(round(msg.v_x,2))
        self.v_y = "v_y : " + str(round(msg.v_y,2))
        self.v_pitch = "v_pitch : " + str(round(msg.v_pitch,2))
        self.roll = "roll : " + str(round(msg.roll,2))
        self.grasp = "grasp : " + str(msg.grasp)

    def outcome_callback(self,msg):
        print("got KIVY outcome")
        self.out_x = "out_x : " + str(round(msg.x,2))
        self.out_y = "out_y : " + str(round(msg.y,2))
        self.out_angle = "out_angle : " + str(round(msg.angle,2))
        self.out_touch = "out_touch : " + str(round(msg.touch,1))

    def pause_callback(self,msg):
        if msg.data == True and self.name_record != "Resume":
            print("pause")
            self.name_record = "Resume"
            self.mode_record = "Resume"
            self.start_record = _GREEN_LIGHT
            self.record = False
            r = Bool()
            r.data = True
            self.pub_pause_dft.publish(r)
            self.pub_pause_perception.publish(r)
            self.saveTime()
            self.error.save_values(self.name_peaks)
            self.error.save_inverse_values(self.name_inv_peaks)
            self.lp.save_values(self.name_peaks)
            self.record = False
        if msg.data == False and self.name_record != "Pause":
            print("resume")
            self.name_record = "Pause"
            self.mode_record = "Pause"
            self.start_record = _RED_LIGHT
            self.record = True
            r = Bool()
            r.data = False
            self.pub_pause_dft.publish(r)
            self.pub_pause_perception.publish(r)

    def saveTime(self):
        p = exists(self.name_time)
        if p:
            os.remove(self.name_time)
        filehandler = open(self.name_time, 'wb')
        pickle.dump(self.time, filehandler)

    def loadTime(self):
        filehandler = open(self.name_time, 'rb') 
        self.time = pickle.load(filehandler)
        print(self.time)

    def notify(self,notif):
        if notif == True:
            playsound('/home/altair/PhD/Codes/ExperimentIM-LCNE/happy.wav')
        else:
            playsound('/home/altair/PhD/Codes/ExperimentIM-LCNE/wrong.wav')

    def getListTmpFiles(self,name):
        list_files = []
        for dir in os.listdir(name):
            list_files.append(dir)

        return list_files

    def removeFiles(self):
        ftr = self.getListTmpFiles(self.working_dir)
        for i in ftr:
            file = self.working_dir + i
            os.remove(file)
        #remove dmp files
        dmptr = self.getListTmpFiles(self.dmp_dir)
        for i in dmptr:
            file = self.dmp_dir + i
            os.remove(file)

    def startExploration(self):
        self.control_arch.publishExpl(True)
        self.control_arch.resetArchitecture(0.0)
        rospy.sleep(1)
        self.control_arch.publishExpl(False)


    def copy_all_datas(self,name_object,type_exp,index):
        list_files = self.getListTmpFiles(self.working_dir)
        for i in list_files:
            source = self.working_dir + i
            new_dest = "/home/altair/PhD/Codes/ExperimentIM-LCNE/datas/" + name_object + "/" + type_exp + "/" + str(index) + "/"
            newPath = shutil.copy(source, new_dest)

    def copy_datas_explore(self,name_object,type_exp,index):
        list_files = self.getListTmpFiles(self.working_dir)
        for i in list_files:
            source = self.working_dir + i
            new_dest = "/home/altair/PhD/Codes/ExperimentIM-LCNE/datas/" + name_object + "/" + type_exp + "/" + str(index) + "/mid/"
            newPath = shutil.copy(source, new_dest)
        list_dmp = self.getListTmpFiles(self.dmp_dir)
        for i in list_dmp:
            src_dmp = self.dmp_dir + i
            dest_dmp = "/home/altair/PhD/Codes/ExperimentIM-LCNE/datas/" + name_object + "/" + type_exp + "/" + str(index) + "/dmp/"
            newPath = shutil.copy(src_dmp, dest_dmp)

    def set_record(self):
        change = True
        if self.mode_record == "Resume" and change:
            self.name_record = "Pause"
            self.mode_record = "Pause"
            self.start_record = _RED_LIGHT
            self.record = True
            r = Bool()
            r.data = False
            self.first = False
            self.pub_pause_dft.publish(r)
            self.pub_pause_perception.publish(r)
            change = False
        if self.mode_record == "Pause" and change:
            self.name_record = "Resume"
            self.mode_record = "Resume"
            self.start_record = _GREEN_LIGHT
            r = Bool()
            r.data = True
            self.pub_pause_dft.publish(r)
            self.pub_pause_perception.publish(r)
            self.saveTime()
            self.error.save_values(self.name_peaks)
            self.error.save_inverse_values(self.name_inv_peaks)
            self.lp.save_values(self.name_peaks)
            self.record = False
            change = False
        if self.mode_record == "Start" and change:
            print("start")
            self.name_record = "Pause"
            self.mode_record = "Pause"
            self.start_record = _RED_LIGHT
            self.record = True
            self.first = False
            change = False
        

    def load_datas(self):
        self.loadTime()
        self.error.load_values(self.name_peaks)
        self.error.load_inverse_values(self.name_inv_peaks)
        self.lp.load_values(self.name_peaks)
        self.persist.load_values(self.name_peaks)

    def update_image(self,dt):
        self.mt_error = "/home/altair/PhD/Codes/Experiment-IMVAE/datas/production/records/error.jpg"
        self.mt_lp = "/home/altair/PhD/Codes/Experiment-IMVAE/datas/production/records/lp.jpg"
        self.mt_vae = "/home/altair/PhD/Codes/Experiment-IMVAE/datas/production/records/vae_out.jpg"
        self.mt_inhib = "/home/altair/PhD/Codes/Experiment-IMVAE/datas/production/records/vae_act.jpg"
        self.root.children[1].children[2].children[0].reload() #error
        self.root.children[1].children[1].children[0].reload() #lp
        self.root.children[1].children[3].children[0].reload() #vae_out
        self.root.children[1].children[4].children[0].reload() #vae_act
        #print(self.root.children[1].children[4].children[0].source)

    def update_events(self, dt):
        if self.node_rnd_explore.getNode() > 0.8:
            self.explore_rnd = _GREEN_LIGHT
            self.retrieve_dmp = _GRAY_LIGHT
            self.n_exploit = True
            if self.keep_exploit == True:
                self.keep_exploring = True
        else:
            self.explore_rnd = _GRAY_LIGHT
        if self.node_direct_explore.getNode() > 0.8:
            self.explore_direct = _GREEN_LIGHT
            self.retrieve_dmp = _GRAY_LIGHT
            self.n_exploit = True
            if self.keep_exploit == True:
                self.keep_exploring = True
        else:
            self.explore_direct = _GRAY_LIGHT
        if self.node_exploit.getNode() > 0.8:
            self.exploit = _GREEN_LIGHT
            self.retrieve_dmp = _YELLOW_LIGHT
            self.keep_exploit = True
            """if self.n_exploit == True:
                self.saveTime()
                self.error.save_values(self.name_peaks)
                self.copy_datas_explore(self.name_object,self.experiment,self.index)
                self.n_exploit = False"""
        else:
            self.exploit = _GRAY_LIGHT
            self.learning_dmp = _GRAY_LIGHT


    # Update the progress with datas coming from neural fields
    def update_gauges(self, dt):
        list_error_fwd = self.error.get_errors()
        list_lp = self.lp.get_errors()
        size_l = len(list_error_fwd)

        if self.record == True:
            if self.first == True:
                self.time = 0
                self.first = False
            if self.steps == 10:
                t = Float64()
                t.data = self.time
                self.pub_time.publish(t)
                self.steps = 0
            self.steps += 1
        for i in range(0,size_l):
            err = math.ceil(list_error_fwd[i][2]*100)/100
            lp = math.ceil(list_lp[i][2]*100)/100
            if err > 1.0:
                err = 1.0
            if lp > 1.0:
                lp = 1.0
            self.root.children[0].children[i].value_normalized_error = err
            self.root.children[0].children[i].value_normalized = lp
            self.root.children[0].children[i].value_error_string = str(round(err*100))
            self.root.children[0].children[i].value_lp = str(round(lp*100))
            self.root.children[0].children[i].value_normalized_goal = str(int(list_error_fwd[i][0])) + "_" + str(int(list_error_fwd[i][1]))
            #self.root.children[0].children[i].value_normalized_goal = self.explicit_skills[i]
            if list_error_fwd[i][0] < 30:
                self.root.children[0].children[i].value_object = 0
            if list_error_fwd[i][0] > 30 and list_error_fwd[i][0] < 60:
                self.root.children[0].children[i].value_object = 1
            if list_error_fwd[i][0] > 60:
                self.root.children[0].children[i].value_object = 2
            self.root.children[0].children[i].background_non_active = _DEFAULT_BACKGROUND_ACTIVE
            self.root.children[0].children[i].background_active = _DEFAULT_WIDGET_NONACTIVE
        if self.rst == True:
            for j in range(0,2):
                for i in range(0,len(self.root.children[0].children)):
                    self.root.children[0].children[i].value_normalized_error = 0
                    self.root.children[0].children[i].value_normalized = 0
                    self.root.children[0].children[i].value_error_string = 0
                    self.root.children[0].children[i].value_lp = 0
                    self.root.children[0].children[i].value_normalized_goal = str(int(0))
                    self.root.children[0].children[i].background_non_active = _DEFAULT_BACKGROUND_NONACTIVE
                    self.root.children[0].children[i].background_active = _DEFAULT_BACKGROUND_NONACTIVE
                    #self.root.children[0].children[i].background_colour = _DEFAULT_BACKGROUND_NONACTIVE
            #self.startExploration()
            self.rst = False
        if self.record == True:
            self.time += 0.1

    # Simple layout for easy example
    def build(self):
        #Clock.schedule_interval(lambda dt: img.reload(), 0.2)
        self.container = Builder.load_string('''
#:import Label kivy.core.text.Label           
#:set _label Label(text="ERROR {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_lp Label(text="LP {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_goal Label(text="Goal : {}", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _labelone Label(text="ERROR {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_lpone Label(text="LP {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_goalone Label(text="Goal : {}", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _labeltwo Label(text="ERROR {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_lptwo Label(text="LP {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_goaltwo Label(text="Goal : {}", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _labelthree Label(text="ERROR {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_lpthree Label(text="LP {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_goalthree Label(text="Goal : {}", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _labelfour Label(text="ERROR {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_lpfour Label(text="LP {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_goalfour Label(text="Goal : {}", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _labelfive Label(text="ERROR {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_lpfive Label(text="LP {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_goalfive Label(text="Goal : {}", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _labelsix Label(text="ERROR {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_lpsix Label(text="LP {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_goalsix Label(text="Goal : {}", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _labelseven Label(text="ERROR {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_lpseven Label(text="LP {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_goalseven Label(text="Goal : {}", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _labeleight Label(text="ERROR {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_lpeight Label(text="LP {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_goaleight Label(text="Goal : {}", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _labelnine Label(text="ERROR {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_lpnine Label(text="LP {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_goalnine Label(text="Goal : {}", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _labelten Label(text="ERROR {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_lpten Label(text="LP {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_goalten Label(text="Goal : {}", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _labeleleven Label(text="ERROR {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_lpeleven Label(text="LP {}%", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
#:set _label_goaleleven Label(text="Goal : {}", font_size=16, color=(0.933,0.902,0.807,1), halign="center")
BoxLayout:
    id: bl
    orientation: 'horizontal'
    BoxLayout:
        orientation: 'vertical'
        size: 410, 800
        size_hint: (None,None)
        BoxLayout:
            orientation: 'horizontal'
            size: 400, 55
            size_hint: (None,None)
            padding: 5
            spacing: 10
            Label:
                text_size: self.size
                size: self.texture_size
                halign: 'center'
                valign: 'middle'
                font_size: 18
                text: "random"
                canvas.before:
                    Color:
                        rgba: app.explore_rnd
                    RoundedRectangle:
                        size: self.size
                        pos: self.pos
                        radius: [15]
            Label:
                text_size: self.size
                size: self.texture_size
                halign: 'center'
                valign: 'middle'
                font_size: 18
                text: "direct"
                canvas.before:
                    Color:
                        rgba: app.explore_direct
                    RoundedRectangle:
                        size: self.size
                        pos: self.pos
                        radius: [15]
            Label:
                text_size: self.size
                size: self.texture_size
                halign: 'center'
                valign: 'middle'
                font_size: 18
                text: "exploit"
                canvas.before:
                    Color:
                        rgba: app.exploit
                    RoundedRectangle:
                        size: self.size
                        pos: self.pos
                        radius: [15]
        BoxLayout:
            orientation: 'vertical'
            size: 410, 170
            size_hint: (None,None)
            padding: 0
            spacing: 0
            BoxLayout:
                orientation: 'horizontal'
                size: 410, 25
                pos: 0, 400
                size_hint: (None,None)
                padding: 0
                spacing: 0
                Label:
                    bold: True
                    text_size: self.size
                    size: self.texture_size
                    halign: 'center'
                    valign: 'middle'
                    font_size: 18
                    text: app.v_x
                    color: 0, 0, 0, 0.8
                Label:
                    bold: True
                    text_size: self.size
                    size: self.texture_size
                    halign: 'center'
                    valign: 'middle'
                    #pos: 0, -100
                    font_size: 18
                    text: app.out_x
                    color: 0, 0, 0, 0.8
            BoxLayout:
                orientation: 'horizontal'
                size: 410, 25
                pos: 0, 400
                size_hint: (None,None)
                padding: 0
                spacing: 0
                Label:
                    bold: True
                    text_size: self.size
                    size: self.texture_size
                    halign: 'center'
                    valign: 'middle'
                    font_size: 18
                    text: app.v_y
                    color: 0, 0, 0, 0.8
                Label:
                    bold: True
                    text_size: self.size
                    size: self.texture_size
                    halign: 'center'
                    valign: 'middle'
                    #pos: 0, -100
                    font_size: 18
                    text: app.out_y
                    color: 0, 0, 0, 0.8
            BoxLayout:
                orientation: 'horizontal'
                size: 410, 25
                pos: 0, 400
                size_hint: (None,None)
                padding: 0
                spacing: 0
                Label:
                    bold: True
                    text_size: self.size
                    size: self.texture_size
                    halign: 'center'
                    valign: 'middle'
                    font_size: 18
                    text: app.v_pitch
                    color: 0, 0, 0, 0.8
                Label:
                    bold: True
                    text_size: self.size
                    size: self.texture_size
                    halign: 'center'
                    valign: 'middle'
                    #pos: 0, -100
                    font_size: 18
                    text: app.out_angle
                    color: 0, 0, 0, 0.8
            BoxLayout:
                orientation: 'horizontal'
                size: 410, 25
                pos: 0, 400
                size_hint: (None,None)
                padding: 0
                spacing: 0
                Label:
                    bold: True
                    text_size: self.size
                    size: self.texture_size
                    halign: 'center'
                    valign: 'middle'
                    font_size: 18
                    text: app.roll
                    color: 0, 0, 0, 0.8
                Label:
                    bold: True
                    text_size: self.size
                    size: self.texture_size
                    halign: 'center'
                    valign: 'middle'
                    #pos: 0, -100
                    font_size: 18
                    text: app.out_touch
                    color: 0, 0, 0, 0.8
            BoxLayout:
                orientation: 'horizontal'
                size: 410, 25
                pos: 0, 400
                size_hint: (None,None)
                padding: 0
                spacing: 0
                Label:
                    bold: True
                    text_size: self.size
                    size: self.texture_size
                    halign: 'center'
                    valign: 'middle'
                    font_size: 18
                    text: app.grasp
                    color: 0, 0, 0, 0.8
                Label:
                    bold: True
                    text_size: self.size
                    size: self.texture_size
                    halign: 'center'
                    valign: 'middle'
                    #pos: 0, -100
                    font_size: 18
                    text: ""
                    color: 0, 0, 0, 0.8                           
            BoxLayout:
                orientation: 'horizontal'
                size: 400, 45
                pos: 0, 400
                size_hint: (None,None)
                padding: 10
                spacing: 15
                Label:
                    text_size: self.size
                    size: self.texture_size
                    halign: 'center'
                    valign: 'bottom'
                    font_size: 18
                    text: "VAE action"
                    color: 0, 0, 0, 0.8
                Label:
                    text_size: self.size
                    size: self.texture_size
                    halign: 'center'
                    valign: 'middle'
                    #pos: 0, -100
                    font_size: 18
                    text: "VAE outcome"
                    color: 0, 0, 0, 0.8   
        FloatLayout:
            pos: 100,100
            Image:
                size_hint: None, None
                size: 200, 200
                pos: 0, 350
                source: app.mt_inhib
        FloatLayout:
            pos: 100,100
            Image:
                size_hint: None, None
                size: 200, 200
                pos: 210, 350
                source: app.mt_vae
        FloatLayout:
            pos: 100,100
            Image:
                size_hint: None, None
                size: 200, 200
                pos: 0, 130
                source: app.mt_error           
        FloatLayout:
            pos: 100,100
            Image:
                size_hint: None, None
                size: 200, 200
                pos: 210, 130
                source: app.mt_lp 
        BoxLayout:
            orientation: 'vertical'
            size: 420, 250
            size_hint: (None,None)
            BoxLayout:
                orientation: 'horizontal'
                size: 420, 70
                size_hint: (None,None)
                padding: 10
                spacing: 10
                Label:
                    text_size: self.size
                    size: 50, 50
                    halign: 'center'
                    valign: 'middle'
                    pos: 0, 50
                    font_size: 22
                    text: "Errors"
                    color: 0, 0, 0, 0.8
                Label:
                    text_size: self.size
                    size: 50, 50
                    halign: 'center'
                    valign: 'middle'
                    pos: 280, 50
                    font_size: 22
                    text: "Learning Progress"
                    color: 0, 0, 0, 0.8
            BoxLayout:
                orientation: 'horizontal'
                size: 410, 70
                size_hint: (None,None)
                padding: 10
                spacing: 15
                Button:
                    #text: "Start record"
                    text: app.name_record
                    font_size: 24
                    size: 50, 50
                    pos: 10, 10
                    background_color: app.start_record
                    #background_normal: ''
                    on_press: app.set_record()
                Button:
                    text: "Load Datas"
                    font_size: 24
                    size: 50, 50
                    pos: 10, 10
                    background_color: app.stop_record
                    #background_normal: ''
                    on_press: app.load_datas()
    GridLayout:
        id: gl
        cols: 4
        rows: 3
        CircularProgressBar:
            pos: 100, 100
            thickness: 5
            widget_size: 150
            label_error_string: _label
            label_lp: _label_lp
            label_goal: _label_goal
        CircularProgressBar:
            pos: 100, 100
            thickness: 5
            widget_size: 150
            label_error_string: _labelone
            label_lp: _label_lpone
            label_goal: _label_goalone
        CircularProgressBar:
            pos: 100, 100
            thickness: 5
            widget_size: 150
            label_error_string: _labeltwo
            label_lp: _label_lptwo
            label_goal: _label_goaltwo
        CircularProgressBar:
            pos: 100, 100
            thickness: 5
            widget_size: 150
            label_error_string: _labelthree
            label_lp: _label_lpthree
            label_goal: _label_goalthree
        CircularProgressBar:
            pos: 100, 100
            thickness: 5
            widget_size: 150
            label_error_string: _labelfour
            label_lp: _label_lpfour
            label_goal: _label_goalfour
        CircularProgressBar:
            pos: 100, 100
            thickness: 5
            widget_size: 150
            label_error_string: _labelfive
            label_lp: _label_lpfive
            label_goal: _label_goalfive
        CircularProgressBar:
            pos: 100, 100
            thickness: 5
            widget_size: 150
            label_error_string: _labelsix
            label_lp: _label_lpsix
            label_goal: _label_goalsix
        CircularProgressBar:
            pos: 100, 100
            thickness: 5
            widget_size: 150
            label_error_string: _labelseven
            label_lp: _label_lpseven
            label_goal: _label_goalseven
        CircularProgressBar:
            pos: 100, 100
            thickness: 5
            widget_size: 150
            label_error_string: _labeleight
            label_lp: _label_lpeight
            label_goal: _label_goaleight
        CircularProgressBar:
            pos: 100, 100
            thickness: 5
            widget_size: 150
            label_error_string: _labelnine
            label_lp: _label_lpnine
            label_goal: _label_goalnine
        CircularProgressBar:
            pos: 100, 100
            thickness: 5
            widget_size: 150
            label_error_string: _labelten
            label_lp: _label_lpten
            label_goal: _label_goalten
        CircularProgressBar:
            pos: 100, 100
            thickness: 5
            widget_size: 150
            label_error_string: _labeleleven
            label_lp: _label_lpeleven
            label_goal: _label_goaleleven''')
        # Animate the progress bar
        Clock.schedule_interval(self.update_gauges, 0.1)
        Clock.schedule_interval(self.update_events, 0.1)
        Clock.schedule_interval(self.update_image, 0.5)
        return self.container


if __name__ == '__main__':
    #rospy.init_node('DataRecorder')
    #rate = rospy.Rate(50)
    VisualDatas().run()
    
