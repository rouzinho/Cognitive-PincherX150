#!/usr/bin/env python3


from os import name

import sys
import copy
import rospy
import roslib;
roslib.load_manifest('dmp')
from dmp.srv import *
from dmp.msg import *
import rosbag
import numpy as np
from math import pi
import math
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
import glob
import os.path
import random
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from interbotix_xs_msgs.msg import *
from std_msgs.msg import Time
from std_msgs.msg import Header
from std_msgs.msg import Duration
from std_msgs.msg import UInt16
from std_msgs.msg import Int16
from std_msgs.msg import String
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from som.msg import GripperOrientation
from motion.msg import Dmp
from motion.msg import Action
from som.msg import ListPose
from som.srv import *
import os
import os.path
from os import path
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import interbotix_common_modules.angle_manipulation as ang
from pathlib import Path
from tf.transformations import *
from cluster_message.srv import *

#Set a DMP as active for planning
def makeSetActiveRequest(dmp_list):
    try:
        sad = rospy.ServiceProxy('set_active_dmp', SetActiveDMP)
        sad(dmp_list)
    except rospy.ServiceException:
        print("Service call failed: %s")


#Generate a plan from a DMP
def makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh, 
                    seg_length, tau, dt, integrate_iter):
    print("Starting DMP planning...")
    rospy.wait_for_service('get_dmp_plan')
    try:
        gdp = rospy.ServiceProxy('get_dmp_plan', GetDMPPlan)
        resp = gdp(x_0, x_dot_0, t_0, goal, goal_thresh, 
                   seg_length, tau, dt, integrate_iter)
    except rospy.ServiceException:
        print("Service call failed: %s")
    print("DMP planning done")   
            
    return resp;


class Motion(object):
  def __init__(self):
    super(Motion, self).__init__()
    #rospy.init_node('motion', anonymous=True)
    #rate = rospy.Rate(100)
    self.threshold_touch_min = 0.02
    self.threshold_touch_max = 0.0278
    self.record = False
    self.dmp_folder = rospy.get_param("dmp_folder")
    self.current_folder = ""
    self.gripper_state = 0.0
    self.js = JointState()
    self.js_positions = []
    self.touch_value = False
    self.last_pose = GripperOrientation()
    self.poses = []
    self.possible_grasp = [0.0,1.0]
    self.possible_roll = [-1.5,-0.7,0,0,0.7,1.5]
    self.move = False
    self.name_state = ""
    self.path = []
    self.name_ee = self.dmp_folder + "js.bag"
    self.dims = 5
    self.dt = 1.0
    self.K_gain = 100              
    self.D_gain = 2.0 * np.sqrt(self.K_gain)      
    self.num_bases = 5
    self.single_msg = True
    self.rnd_explore = True
    self.direct_explore = False
    self.exploit = False
    self.dmp_exploit = Dmp()
    self.dmp_explore = Dmp()
    self.dmp_direct_explore = Dmp()
    self.dmp_name = ""
    self.dmp_found = False
    self.goal_dmp = False
    self.pose_ee = Pose()
    self.bool_dmp_plan = False
    self.path = ListPose()
    self.prop = JointState()
    self.ready = False
    self.ready_depth = False
    self.ready_outcome = False
    self.got_action = False
    self.recording_dmp = False
    self.last_time = 0
    self.prev_id_object = -1
    self.id_object = 0
    self.count_touch = 0
    self.bot = InterbotixManipulatorXS("px150", "arm", "gripper")
    self.pub_gripper = rospy.Publisher("/px150/commands/joint_single", JointSingleCommand, queue_size=1, latch=True)
    self.pub_touch = rospy.Publisher("/motion_pincher/touch", Bool, queue_size=1, latch=True)
    self.pub_bmu = rospy.Publisher("/som_pose/som/node_value/bmu", GripperOrientation, queue_size=1, latch=True)
    self.pub_path = rospy.Publisher("/som_pose/som/dmp_path", ListPose, queue_size=1, latch=True)
    self.pub_activate_perception = rospy.Publisher("/depth_perception/activate", Bool, queue_size=1, latch=True)
    self.pub_signal_action = rospy.Publisher("/motion_pincher/signal_action", Bool, queue_size=1, latch=True)
    self.pub_display_fpose = rospy.Publisher("/display/first_pose", GripperOrientation, queue_size=1, latch=True)
    self.pub_display_lpose = rospy.Publisher("/display/last_pose", GripperOrientation, queue_size=1, latch=True)
    self.pub_action_sample = rospy.Publisher("/motion_pincher/action_sample", Action, queue_size=1, latch=True)
    self.pub_dmp_action = rospy.Publisher("/motion_pincher/dmp", Dmp, queue_size=1, latch=True)
    self.pub_trigger_state = rospy.Publisher("/outcome_detector/trigger_state", Bool, queue_size=1, latch=True)
    self.pub_inhib = rospy.Publisher("/motion_pincher/inhibition", Float64, queue_size=1, latch=True)
    rospy.Subscriber('/px150/joint_states', JointState, self.callback_joint_states)
    rospy.Subscriber('/proprioception/joint_states', JointState, self.callback_proprioception)
    #rospy.Subscriber('/motion_pincher/go_to_pose', PoseRPY, self.callback_pose)
    rospy.Subscriber('/motion_pincher/gripper_orientation/first_pose', GripperOrientation, self.callback_first_pose)
    rospy.Subscriber('/cluster_msg/new_state', Bool, self.callback_new_state)
    rospy.Subscriber('/cluster_msg/retry', Bool, self.callback_retry)
    rospy.Subscriber('/cog_learning/rnd_exploration', Float64, self.callback_rnd_exploration)
    rospy.Subscriber('/cog_learning/direct_exploration', Float64, self.callback_direct_exploration)
    rospy.Subscriber('/cog_learning/exploitation', Float64, self.callback_exploitation)
    rospy.Subscriber('/motion_pincher/dmp_direct_exploration', Dmp, self.callback_dmp_direct_exploration)
    rospy.Subscriber('/motion_pincher/retrieve_dmp', Dmp, self.callback_dmp)
    #rospy.Subscriber('/cluster_msg/signal', Bool, self.callback_ready)
    rospy.Subscriber("/cog_learning/id_object", Int16, self.callback_id)
    rospy.Subscriber("/cluster_msg/pause", Float64, self.callback_pause)

  def transform_dmp_cam_rob(self, dmp_):
    rospy.wait_for_service('transform_dmp_cam_rob')
    try:
        transform_dmp = rospy.ServiceProxy('transform_dmp_cam_rob', tfCamRob)
        resp1 = transform_dmp(dmp_)
        return resp1.dmp_robot
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

  def transform_dmp_rob_cam(self, dmp_):
    rospy.wait_for_service('transform_dmp_cam_rob')
    try:
        transform_dmp = rospy.ServiceProxy('transform_dmp_rob_cam', tfRobCam)
        resp1 = transform_dmp(dmp_)
        return resp1.dmp_cam
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

  #def callback_pose(self,msg):
  #  self.init_position()
  #  self.bot.arm.set_ee_pose_components(x=msg.x, y=msg.y, z=msg.z, roll=msg.r, pitch=msg.p)
  #  self.init_position()
  #  self.sleep_pose()

  def callback_first_pose(self,msg):
    print("got pose : ",msg)
    tmp = GripperOrientation()
    tmp.x = msg.x
    tmp.y = msg.y
    tmp.pitch = msg.pitch
    self.ready = False
    if self.rnd_explore and len(self.poses) < 2:
      self.poses.append(tmp)
    if self.rnd_explore and len(self.poses) == 1:
      self.send_inhibition(1.0)
    if self.rnd_explore and len(self.poses) == 2:
      self.ready = True
    if self.direct_explore and len(self.poses) == 0:
      self.poses.append(tmp)
      self.ready = True
    print("poses cb : ",len(self.poses))

  def callback_joint_states(self,msg):
    self.js = msg
    self.js_positions = msg.position
    self.gripper_state = msg.position[6]
    if self.gripper_state < self.threshold_touch_max and self.gripper_state > self.threshold_touch_min:
      self.count_touch = self.count_touch +1
      #print(self.gripper_state)
      print("touch : ",self.count_touch)
    else:
      self.count_touch = 0
    if(self.count_touch > 450):
      self.touch_value = True
    #self.count_touch = 0
    #self.pub_touch.publish(t)
    #print(self.touch_value)

  def callback_proprioception(self,msg):
    self.prop = msg
    if self.record == True:
      self.write_joints_bag(self.name_ee,self.prop)

  def callback_dmp(self,msg):
    self.dmp_found, self.dmp_name = self.find_dmp(msg)
    if self.dmp_found:
      print("found DMP : ",self.dmp_name)
    else:
      print("DMP not found")

  def callback_new_state(self,msg):
    if msg.data == True:
      self.poses  = []
      print("ACTION successful")
      if self.recording_dmp and (self.rnd_explore or self.direct_explore):
        self.make_dmp()
        self.delete_js_bag()
        self.recording_dmp = False
      #self.send_inhibition(True)

  def callback_retry(self,msg):
    print("Retry with another pose")
    if msg.data == True and self.direct_explore:
      self.poses = []

  def callback_rnd_exploration(self,msg):
    if msg.data > 0.5:
      self.rnd_explore = True
    else:
      self.rnd_explore = False

  def callback_direct_exploration(self,msg):
    if msg.data > 0.5:
      self.direct_explore = True
    else:
      self.direct_explore = False

  def callback_exploitation(self,msg):
    if msg.data > 0.5:
      self.exploit = True
    else:
      self.exploit = False

  def callback_ready(self,msg):
    self.ready = msg.data

  def get_ready(self):
    return self.ready
  
  def set_seady(self,val):
    self.ready = val

  def callback_id(self,msg):
    self.id_object = msg.data
    if self.prev_id_object != self.id_object:
      path = os.path.join(self.dmp_folder, str(self.id_object))
      access = 0o755
      if not os.path.isdir(path):
        os.makedirs(path,access)
      self.current_folder = self.dmp_folder + str(self.id_object) + "/"
      self.prev_id_object = self.id_object

  def callback_dmp_direct_exploration(self,msg):
    self.dmp_direct_explore.v_x = msg.v_x
    self.dmp_direct_explore.v_y = msg.v_y
    self.dmp_direct_explore.v_pitch = msg.v_pitch
    self.dmp_direct_explore.roll = msg.roll
    self.dmp_direct_explore.grasp = msg.grasp
    print("direct explore : ",self.dmp_direct_explore)

  def callback_pause(self, msg):
    self.poses = []
  
  def get_number_pose(self):
    return len(self.poses)

  def send_state(self,val):
    tmp = Bool()
    tmp.data = val
    self.pub_trigger_state.publish(tmp)

  def send_signal_action(self):
    if self.ready:
      print("sending signal...")
      msg = Bool()
      msg.data = False
      self.pub_signal_action.publish(msg)
      rospy.sleep(1.0)
      msg.data = True
      self.pub_signal_action.publish(msg)
      while not self.got_action:
        pass
      self.got_action = False
      print("got action !")
      self.last_time = rospy.get_time()
    elapsed = rospy.get_time() - self.last_time
    if elapsed > 35:
      pass
      #print("still waiting for perception...")
    
  def send_inhibition(self, val):
    inh = Float64()
    inh.data = val
    self.pub_inhib.publish(inh)
    rospy.sleep(0.3)
    inh.data = 0.0
    self.pub_inhib.publish(inh)

  #remove temporary js path bag file created to generate DMP
  def delete_js_bag(self):
    if os.path.exists(self.name_ee):
      os.remove(self.name_ee)
    else:
      print("JS Bag file doesn't exist") 

  def makeLFDRequest(self,traj):
    demotraj = DMPTraj()        
    for i in range(len(traj)):
      pt = DMPPoint();
      pt.positions = traj[i]
      demotraj.points.append(pt)
      demotraj.times.append(self.dt*i)
                
    k_gains = [self.K_gain]*self.dims
    d_gains = [self.D_gain]*self.dims
    print(demotraj)
            
    print ("Starting LfD...")
    rospy.wait_for_service('learn_dmp_from_demo')
    print("test")
    try:
      lfd = rospy.ServiceProxy('learn_dmp_from_demo', LearnDMPFromDemo)
      resp = lfd(demotraj, k_gains, d_gains, self.num_bases)
    except rospy.ServiceException:
      print("Service call failed: %s")
      print("LfD done")    
                
    return resp;

  #write js positions path in file
  def write_joints_bag(self,n,pos):
    exist = path.exists(n)
    opening = ""
    if(exist == True):
      opening = "a"
    else:
      opening = "w"
    bag = rosbag.Bag(n, opening)
    try:
      print("writing js")
      bag.write("js",pos)
    except:
      print("error recording js")
    bag.close()

  #get the js path as an array to generae DMP
  def form_data_joint_states(self):
    name = self.name_ee
    tot = []
    bag = rosbag.Bag(name)
    for topic, msg, t in bag.read_messages(topics=['js']):
      j = []
      for i in range(0,5):
        j.append(msg.position[i])
      tot.append(copy.deepcopy(j))
    bag.close() 

    return tot

  #write the DMP in a bag
  def write_dmp_bag(self,data,n):
    name = n
    exist = path.exists(name)
    opening = "w"
    bag = rosbag.Bag(name, opening)
    try:
      bag.write("dmp_pos",data)
    finally:
      bag.close()

  #read DMP
  def get_dmp(self,name):
    bag = rosbag.Bag(name)
    for topic, msg, t in bag.read_messages(topics=['dmp_pos']):
      resp = msg
    bag.close()

    return resp
  
  #if it's exploring
  def get_rnd_explore(self):
    return self.rnd_explore
  
  def get_direct_explore(self):
    return self.direct_explore
  
  #if it's exploiting ->learning a skill
  def get_exploit(self):
    return self.exploit

  #naming the DMP
  def name_dmp(self):
    nx = "x"+str(round(self.dmp_explore.v_x,2))
    ny = "y"+str(round(self.dmp_explore.v_y,2))
    np = "p"+str(round(self.dmp_explore.v_pitch,1))
    nr = "r"+str(round(self.dmp_explore.roll,1))
    gr = "g"+str(round(self.dmp_explore.grasp,0))
    name = self.current_folder + nx + ny + np + nr + gr + "end.bag"

    return name
    
  #find the dmp file
  def find_dmp(self,dmp):
    found = False
    right_file = ""
    for file in os.listdir(self.current_folder):
        p_x = file.find('x')
        p_y = file.find('y')
        p_p = file.find('p')
        p_g = file.find('g')
        p_r = file.find('r')
        p_end = file.find('end')
        x = file[p_x+1:p_y]
        y = file[p_y+1:p_p]
        p = file[p_p+1:p_r]
        r = file[p_r+1:p_g]
        g = file[p_g+1:p_end]
        x = float(x)
        y = float(y)
        r = float(r)
        g = float(g)
        if dmp.v_x - x < 0.05 and dmp.v_y - y < 0.05 and dmp.roll - r < 0.05 and dmp.grasp - g < 0.05:
            self.dmp_exploit.v_x = x
            self.dmp_exploit.v_y = y
            self.dmp_exploit.v_pitch = p
            self.dmp_exploit.roll = r
            self.dmp_exploit.grasp = g
            found = True
            right_file = file
    right_file = self.current_folder + right_file
    return found, right_file

  def make_dmp(self):
    traj = self.form_data_joint_states()
    resp = self.makeLFDRequest(traj)
    n = self.name_dmp()
    self.write_dmp_bag(resp,n)

  def play_motion_dmp(self):
    self.pub_dmp_action.publish(self.dmp_exploit)
    tmp = self.js_positions
    curr = []
    for i in range(0,5):
      curr.append(tmp[i])
    resp = self.get_dmp(self.dmp_name)
    #print("Get DMP :")
    #print(resp)
    makeSetActiveRequest(resp.dmp_list)
    goal, found = self.pose_to_joints(self.last_pose.x,self.last_pose.y,0.06,self.dmp_exploit.roll,self.last_pose.pitch) 
    print("goal : ",goal)
    print("cartesian goal : ",self.last_pose)
    if found:
      print("found goal to reach")
    goal_thresh = [0.1,0.1,0.1,0.1,0.1]
    x_0 = curr         #Plan starting at a different point than demo 
    x_dot_0 = [0.0,0.0,0.0,0.0,0.0]   
    t_0 = 0                
    seg_length = -1          #Plan until convergence to goal
    tau = resp.tau / 2       # /4 is good enough
    dt = 1.0 
    integrate_iter = 5       #dt is rather large, so this is > 1  
    planned_dmp = makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter)
    j = 0
    path = self.shrink_dmp(planned_dmp.plan.points,goal)
    print("path length ", len(planned_dmp.plan.points))
    print("path length shrinked", len(path))
    for i in path:
      self.bot.arm.set_joint_positions(i,moving_time=0.1,accel_time=0.02) #moving time 0.4 accel 0.1
      print(j)
      j+=1
    sample = Action()
    sample.lpos_x = self.last_pose.x
    sample.lpos_y = self.last_pose.y
    sample.lpos_pitch = self.last_pose.pitch
    self.pub_action_sample.publish(sample)

  #shrink the dmp so the JS positions that are too close are skipped
  def shrink_dmp(self,data,desired):
    tot = 0
    path = []
    for i in data:
      tot = 0
      for j in range(0,len(desired)):
        diff = abs(abs(i.positions[j]) - abs(desired[j]))
        tot = tot + diff
      if tot > 0.01:
        path.append(i.positions)
    return path
  
  def execute_dmp(self):
    if self.dmp_found and self.goal_dmp:
      self.init_position()     
      if self.dmp_exploit.grasp > 0.5:
        self.open_gripper()
      else:
        self.close_gripper()
      self.play_motion_dmp()
      self.close_gripper()
      self.sleep_pose()
      self.goal_dmp = False

  #convert 3D EE position to JS through IK
  def pose_to_joints(self,x,y,z,r,p):
    T_sd = np.identity(4)
    yaw = math.atan2(y,x)
    T_sd[:3,:3] = ang.eulerAnglesToRotationMatrix([r, p, yaw])
    T_sd[:3, 3] = [x, y, z]
    joints, found = self.bot.arm.set_ee_pose_matrix(T_sd, custom_guess=None, execute=False, moving_time=None, accel_time=None, blocking=True)
    return joints, found

  def open_gripper(self):
    self.bot.gripper.open(2.0)

  def close_gripper(self):
    jsc = JointSingleCommand()
    jsc.name = "gripper"
    jsc.cmd = -250.0
    self.pub_gripper.publish(jsc)
    while not self.touch_value and self.gripper_state > 0.017:
      #print(self.gripper_state)
      pass
    jsc.cmd = 0
    self.pub_gripper.publish(jsc)
    if self.touch_value:
      print("object grasped !")
      
  #execute the action
  def execute_rnd_exploration(self,record_dmp):
    self.ready = False
    self.send_state(True)
    print("RANDOM EXPLORATION")
    r = random.choice(self.possible_roll)
    g = random.choice(self.possible_grasp)
    #r = 0
    #g = 1
    self.dmp_explore.v_x = self.poses[1].x - self.poses[0].x
    self.dmp_explore.v_y = self.poses[1].y - self.poses[0].y
    self.dmp_explore.v_pitch = self.poses[1].pitch - self.poses[0].pitch
    self.dmp_explore.roll = r
    self.dmp_explore.grasp = g
    self.dmp_explore.fpos_x = self.poses[0].x
    self.dmp_explore.fpos_y = self.poses[0].y
    msg = self.transform_dmp_cam_rob(self.dmp_explore)
    self.pub_dmp_action.publish(msg)
    self.bot.gripper.set_pressure(1.0)
    #rospy.sleep(3.0)
    self.init_position()     
    if g > 0.5:
      self.bot.gripper.open()
      print("OPEN GRIPPER")
    else:
      self.bot.gripper.close()
      print("CLOSE GRIPPER")
    rospy.sleep(2.0)
    self.record = record_dmp
    self.recording_dmp = record_dmp
    print("nb pose : ",len(self.poses))
    self.bot.arm.set_ee_pose_components(x=self.poses[0].x, y=self.poses[0].y, z=0.06, roll=r, pitch=self.poses[0].pitch)
    self.bot.arm.set_ee_pose_components(x=self.poses[1].x, y=self.poses[1].y, z=0.06, roll=r, pitch=self.poses[1].pitch)
    self.record = False
    self.bot.gripper.close()
    #rospy.sleep(3.0)
    self.init_position()  
    self.sleep_pose()
    #send touch value
    t = Bool()
    t.data = self.touch_value
    self.pub_touch.publish(t)
    print("ACTION DONE")
    self.last_time = rospy.get_time()
    sample = Action()
    sample.lpos_x = self.poses[1].x
    sample.lpos_y = self.poses[1].y
    sample.lpos_pitch = self.poses[1].pitch
    self.pub_action_sample.publish(sample)
    self.poses.pop()
    self.bool_last_p = False
    self.ready_depth = False
    self.ready_outcome = False
    self.touch_value = False
    b = Bool()
    b.data = True
    self.pub_activate_perception.publish(b)
    self.bot.gripper.open()
    #self.ready = True

  def execute_direct_exploration(self,record_dmp):
    self.ready = False
    self.send_state(True)
    print("DIRECT EXPLORATION")
    self.dmp_direct_explore.fpos_x = self.poses[0].x
    self.dmp_direct_explore.fpos_y = self.poses[0].y
    msg = self.transform_dmp_rob_cam(self.dmp_direct_explore)
    self.pub_dmp_action.publish(msg)
    self.bot.gripper.set_pressure(0.8)
    #rospy.sleep(3.0)
    self.init_position()     
    if self.dmp_direct_explore.grasp > 0.5:
      self.bot.gripper.open()
    else:
      self.bot.gripper.close()
    self.record = record_dmp
    self.recording_dmp = record_dmp
    lpos_x = self.poses[0].x + self.dmp_direct_explore.v_x
    lpos_y = self.poses[0].y + self.dmp_direct_explore.v_y
    lpos_p = self.poses[0].pitch + self.dmp_direct_explore.v_pitch
    self.bot.arm.set_ee_pose_components(x=self.poses[0].x, y=self.poses[0].y, z=0.06, roll=self.dmp_direct_explore.roll, pitch=self.poses[0].pitch)
    self.bot.arm.set_ee_pose_components(x=lpos_x, y=lpos_y, z=0.06, roll=self.dmp_direct_explore.roll, pitch=lpos_p)
    self.record = False
    self.bot.gripper.close()
    #rospy.sleep(2.0)
    self.init_position()  
    self.sleep_pose()
    #send touch value
    t = Bool()
    t.data = self.touch_value
    self.pub_touch.publish(t)
    print("ACTION DONE")
    self.last_time = rospy.get_time()
    sample = Action()
    sample.lpos_x = lpos_x
    sample.lpos_y = lpos_y
    sample.lpos_pitch = lpos_p
    self.pub_action_sample.publish(sample)
    self.poses.pop()
    self.ready_depth = False
    self.ready_outcome = False
    self.touch_value = False
    b = Bool()
    b.data = True
    self.pub_activate_perception.publish(b)
    self.bot.gripper.open()
    #self.ready = True

  def execute_exploitation(self):


  def run_possibilities(self):
    name_dataset = "/home/altair/interbotix_ws/src/motion/dataset/home_positions.txt"
    exist = path.exists(name_dataset)
    self.init_position()
    x = 0
    y = 0
    p = 0
    r = 0
    for i in range(10,20):
      for j in range(-1,2):
        for k in range(0,18,2):
          for l in range(0,50):
            if i == 0:
              x = 0
            else:
              x = i/100
            if j == 0:
              y = 0
            else:
              y = j/100
            if k == 0:
              p = 0
            else:
              p = k/10
            if l == 0:
              z = 0
            else:
              z = l/100
            r = 0.0
            data = str(x) + " " + str(y) + " " + str(z) + " " + str(p) + "\n"
            joints, f = self.pose_to_joints(x,y,z,r,p)
            if f == True:
              print("Done ",data)
              with open(name_dataset, "a") as f:
                f.write(data)
              f.close()

  def test_interface(self):
    self.bot.gripper.open()
    self.bot.gripper.set_pressure(1.0)
    self.bot.gripper.close()
    #self.close_gripper()
    rospy.sleep(2.0)
    self.bot.gripper.open()

  def init_position(self):
    #self.bot.arm.go_to_home_pose()
    self.bot.arm.set_ee_pose_components(x=0.15, y=0, z=0.25, roll=0, pitch=0.0)
    #rospy.sleep(1.0)

  def sleep_pose(self):
    self.bot.arm.go_to_sleep_pose(moving_time=2.0,accel_time=0.3)

  def check_pos(self):
    #self.bot.arm.set_ee_pose_components(x=0.45, y=0.0, z=0.03, roll=0,pitch=0.0)
    j, f = self.pose_to_joints(0.05,0.0,0.02,0,1.5)
    print(f)
    #self.init_position()
    #self.bot.arm.set_ee_pose_components(x=0.25, y=0.01, z=0.03, roll=0, pitch=1.0)
    ##j, f = self.pose_to_joints(0.25,0,0.03,0,1.5)
    #print(f)
    #self.init_position()

if __name__ == '__main__':
  motion_pincher = Motion()
  first = True
  sent_inh = False
  rospy.sleep(3.0)
  #motion_pincher.init_position()
  while not rospy.is_shutdown():
    if motion_pincher.get_rnd_explore() and motion_pincher.get_number_pose() == 2 and motion_pincher.get_ready():
        print("EXECUTE rnd action")
        motion_pincher.execute_rnd_exploration(False)
    if motion_pincher.get_direct_explore() and motion_pincher.get_number_pose() == 1 and motion_pincher.get_ready():
      motion_pincher.execute_direct_exploration(False)
      #motion_pincher.send_signal_action()
    if motion_pincher.get_exploit():
      motion_pincher.execute_dmp()
  #if first:
  #  motion_pincher.test_interface()
    #motion_pincher.run_possibilities()
  #  print("DONE !")  
  #  first = False

  rospy.spin()