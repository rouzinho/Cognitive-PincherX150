#!/usr/bin/env python3


from os import name

import sys
import copy
import rospy
import geometry_msgs.msg
import roslib;
roslib.load_manifest('dmp')
from dmp.srv import *
from dmp.msg import *
import rosbag
import numpy as np
from math import pi
import math
from std_msgs.msg import String
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
import glob
import os.path
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from interbotix_xs_msgs.msg import *
from std_msgs.msg import Time
from std_msgs.msg import Header
from std_msgs.msg import Duration
from std_msgs.msg import UInt16
from sensor_msgs.msg import JointState
from motion.msg import PoseRPY
from motion.msg import GripperOrientation
from motion.msg import VectorAction
import os.path
from os import path
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import interbotix_common_modules.angle_manipulation as ang
from pathlib import Path

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
        gdp = rospy.ServiceProxy('get_dmp_plan', get_dmpPlan)
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
    self.bot = InterbotixManipulatorXS("px150", "arm", "gripper")
    self.pub_gripper = rospy.Publisher("/px150/commands/joint_single", JointSingleCommand, queue_size=1, latch=True)
    self.pub_touch = rospy.Publisher("/outcome_detector/touch", Bool, queue_size=1, latch=True)
    rospy.Subscriber('/px150/joint_states', JointState, self.callback_joint_states)
    rospy.Subscriber('/proprioception/joint_states', JointState, self.callback_proprioception)
    rospy.Subscriber('/motion_pincher/go_to_pose', PoseRPY, self.callback_pose)
    rospy.Subscriber('/motion_pincher/start_position', Pose, self.callback_xy)
    rospy.Subscriber('/motion_pincher/gripper_orientation', GripperOrientation, self.callback_gripper)
    rospy.Subscriber('/motion_pincher/vector_action', VectorAction, self.callback_vector_action)
    rospy.Subscriber('/motion_pincher/touch_pressure', UInt16, self.callback_pressure)
    rospy.Subscriber('/depth_perception/new_state', Bool, self.callback_new_state)
    rospy.Subscriber('/depth_perception/retry', Bool, self.callback_retry)

    self.gripper_state = 0.0
    self.js = JointState()
    self.js_positions = []
    self.stop_pressing = False
    self.init_pose = geometry_msgs.msg.Pose()
    self.gripper_orientation = GripperOrientation()
    self.action = VectorAction()
    self.bool_init_p = False
    self.bool_grip_or = False
    self.bool_act = False
    self.move = False
    self.move_dmp = False
    self.activate_dmp = False
    self.path = []
    self.name_ee = "/home/altair/interbotix_ws/rosbags/forward.bag"
    self.name_dmp = "/home/altair/interbotix_ws/rosbags/forward_dmp.bag"
    self.record = False
    self.dims = 5
    self.dt = 1.0
    self.K_gain = 100              
    self.D_gain = 2.0 * np.sqrt(self.K_gain)      
    self.num_bases = 4

  def callback_pose(self,msg):
    self.bot.arm.set_ee_pose_components(x=msg.x, y=msg.y, z=msg.z, roll=msg.r, pitch=msg.p)
    self.init_position()

  def callback_xy(self,msg):
    if self.bool_init_p == False:
      self.init_pose.position.x = msg.position.x
      self.init_pose.position.y = msg.position.y
      self.init_pose.position.z = msg.position.z
      self.bool_init_p = True

  def callback_gripper(self,msg):
    if self.bool_grip_or == False:
      self.gripper_orientation.roll = msg.roll
      self.gripper_orientation.pitch = msg.pitch
      self.bool_grip_or = True

  def callback_vector_action(self,msg):
    if self.bool_act == False:
      self.action.x = msg.x
      self.action.y = msg.y
      self.action.z = 0.0
      self.action.grasp = msg.grasp
      self.bool_act = True

  def callback_pressure(self,msg):
    if msg.data < 200:
      self.stop_pressing = True
    else:
      self.stop_pressing = False
    t = Bool()
    t.data = self.stop_pressing
    self.pub_touch.publish(t)

  def callback_joint_states(self,msg):
    self.js = msg
    self.js_positions = msg.position
    self.gripper_state = msg.position[6]

  def callback_proprioception(self,msg):
    if self.record == True:
      self.write_joints_bag(self.name_ee,msg)

  def callback_new_state(self,msg):
    if msg.data == True:
      self.bool_init_p = False
      self.bool_grip_or = False
      self.bool_act = False
      self.update_offline_dataset(True)

  def callback_retry(self,msg):
    if msg.data == True:
      self.bool_init_p = False
      self.update_offline_dataset(False)

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

  def write_joints_bag(self,n,pos):
    name = n
    exist = path.exists(name)
    opening = ""
    if(exist == True):
      opening = "a"
    else:
      opening = "w"
    bag = rosbag.Bag(name, opening)
    try:
      bag.write("js",pos)
    finally:
      bag.close()

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
    opening = ""
    if(exist == False):
      opening = "w"
    else:
      opening = "a"
    bag = rosbag.Bag(name, opening)
    try:
      bag.write("dmp_pos",data)
    finally:
      bag.close()

  def read_dmp_bag(self,name):
    bag = rosbag.Bag(self.name_dmp)
    for topic, msg, t in bag.read_messages(topics=['dmp_pos']):
      print(msg)
    bag.close()

  def get_dmp(self,name):
    bag = rosbag.Bag(name)
    for topic, msg, t in bag.read_messages(topics=['dmp_pos']):
      resp = msg
    bag.close()

    return resp

  def update_offline_dataset(self,status):
    name_dataset_states = "/home/altair/interbotix_ws/src/depth_perception/states/"
    paths = sorted(Path(name_dataset_states).iterdir(), key=os.path.getmtime)
    name_state = str(paths[len(paths)-1])
    name_dataset = "/home/altair/interbotix_ws/src/motion/dataset/datas.txt"
    exist = path.exists(name_dataset)
    opening = ""
    if(exist == False):
      opening = "w"
    else:
      opening = "a"
    data = str(self.action.x) + " " + str(self.action.y) + " " + str(self.action.grasp) + " " + str(self.gripper_orientation.pitch) + " " + str(self.gripper_orientation.roll) + " " +name_state + " " + str(status) + "\n"
    with open(name_dataset, opening) as f:
        f.write(data)
    f.close()


  def name_dmp(action):
    name = "/home/altair/interbotix_ws/src/motion/dmp/"
    nx = ""
    ny = ""
    gr = ""
    nx = "x"+str(action.x)
    ny = "y"+str(action.y)
    gr = "g"+str(action.grasp)
    name = name + nx + ny + gr + "r.bag"

    return name
    
  def find_dmp(action):
    name_dir = "/home/altair/interbotix_ws/src/motion/dmp/"
    found = False
    right_file = ""
    for file in os.listdir(name_dir):
        p_x = file.find('x')
        p_y = file.find('y')
        p_g = file.find('g')
        p_r = file.find('r')
        x = file[p_x+1:p_y]
        y = file[p_y+1:p_g]
        g = file[p_g+1:p_r]
        x = float(x)
        y = float(y)
        g = float(g)
        if action.x - x < 0.05 and action.y - y < 0.05 and action.grasp - g < 0.05:
            found = True
            right_file = file
    
    return found, right_file

  def makeDMP(self,name_dmp):
    traj = self.form_data_joint_states()
    resp = self.makeLFDRequest(traj)
    print(resp)
    self.write_dmp_bag(resp,name_dmp)

  def play_motion_dmp(self,name_dmp,goal):
    tmp = self.js_positions
    print(tmp)
    curr = []
    for i in range(0,5):
      curr.append(tmp[i])
    resp = self.get_dmp(name_dmp)
    #print("Get DMP :")
    #print(resp)
    makeSetActiveRequest(resp.dmp_list)
    #goal = [0.3709951601922512, 0.08797463793307543, 0.7709327504038811, -0.07890698444098235, 1.505357144537811e-06]
    #goal = [-0.3137078918516636, 0.2984987303111935, 0.4029244794423692, 0.07907685257397824, -1.1780148867046465e-06]
    goal_thresh = [0.1]
    x_0 = curr         #Plan starting at a different point than demo 
    x_dot_0 = [0.0,0.0,0.0,0.0,0.0]   
    t_0 = 0                
    seg_length = -1          #Plan until convergence to goal
    tau = resp.tau / 1.5       # /4 is good enough
    dt = 1.0 
    integrate_iter = 5       #dt is rather large, so this is > 1  
    planned_dmp = makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter)
    
    j = 0
    path = self.shrink_dmp(planned_dmp.plan.points,goal)
    print("path length", len(path))
    for i in path:
      self.bot.arm.set_joint_positions(i,moving_time=0.4,accel_time=0.1)
      print(j)
      j+=1


  def shrink_dmp(self,data,desired):
    tot = 0
    path = []
    for i in data:
      tot = 0
      for j in range(0,len(desired)):
        diff = abs(abs(i.positions[j]) - abs(desired[j]))
        tot = tot + diff
      if tot > 0.04:
        path.append(i.positions)
    return path

  def get_move(self):
    return self.move

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
    jsc.cmd = -50.0
    self.pub_gripper.publish(jsc)
    while not self.stop_pressing and self.gripper_state > 0.020:
      pass
    jsc.cmd = 0
    self.pub_gripper.publish(jsc)

  def execute_action(self,record):
    if self.bool_act == True  and self.bool_grip_or == True and self.bool_init_p == True:
      print("in the loop")
      self.init_position()
      u = self.init_pose.position.x + self.action.x
      v = self.init_pose.position.y + self.action.y
      if self.action.grasp > 0.5:
        self.open_gripper()
      else:
        self.close_gripper()
      self.bot.arm.set_ee_pose_components(x=self.init_pose.position.x, y=self.init_pose.position.y, z=self.init_pose.position.z, roll=self.gripper_orientation.roll, pitch=self.gripper_orientation.pitch)
      self.bot.arm.set_ee_pose_components(x=u, y=v, z=self.init_pose.position.z, roll=self.gripper_orientation.roll, pitch=self.gripper_orientation.pitch)
      self.close_gripper()
      self.sleep_pose()
      self.bool_init_p = False

  def run_possibilities(self):
    name_dataset = "/home/altair/interbotix_ws/src/motion/dataset/data_short.txt"
    exist = path.exists(name_dataset)
    self.init_position()
    x = 0
    y = 0
    p = 0
    r = 0
    for i in range(5,40):
      for j in range(-40,40):
        for k in range(0,18,2):
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
          z = 0.02
          r = 0.0
          data = str(x) + " " + str(y) + " " + str(p) + "\n"
          joints, f = self.pose_to_joints(x,y,z,r,p)
          if f == True:
            print("Done ",data)
            with open(name_dataset, "a") as f:
              f.write(data)
            f.close()
          

  def test_interface(self):
    self.bot.arm.go_to_home_pose()
    #self.write_joints_bag(self.name_ee,self.js)
    #rospy.sleep(1.0)
    self.record = True
    #self.write_joints_bag(self.name_ee,self.js)
    self.bot.arm.set_ee_pose_components(x=0.15, y=0.0, z=0.02, roll=0.0, pitch=0.8)
    #self.write_joints_bag(self.name_ee,self.js)
    #rospy.sleep(0.5)
    #elf.record = True
    #print(self.js_positions)
    #self.write_joints_bag(self.name_ee,self.js)
    #print("done first")
    #self.joints_pos()
    #print("done second in joints space")
    self.bot.arm.set_ee_pose_components(x=0.3, y=0.0, z=0.02, roll=0.0, pitch=0.8)
    self.record = False
    #self.write_joints_bag(self.name_ee,self.js)
    #ospy.sleep(0.5)
    #self.record = False
    #print(self.js)
    #bot.arm.set_single_joint_position("waist", -np.pi/4.0)
    #print("sleep pose")
    self.bot.arm.go_to_home_pose()
    self.bot.arm.set_ee_pose_components(x=0.3, y=-0.1, z=0.02, roll=0.0, pitch=0.8)
    print(self.js)
    self.bot.arm.go_to_home_pose()
    #print("went to sleep")

  def init_position(self):
    self.bot.arm.go_to_home_pose()
    rospy.sleep(1.0)

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
  record = False
  #first = True
  rospy.sleep(2.0)
  #motion_planning.open_gripper()
  #motion_planning.close_gripper()
  #motion_planning.pose_to_joints(0.3,-0.1,0.02,0.0,0.8)  

  while not rospy.is_shutdown():
    if first == True:
      #motion_pincher.open_gripper()
      #motion_pincher.close_gripper()
      #motion_pincher.execute_action(record)
      #motion_pincher.check_pos()
      motion_pincher.run_possibilities()
      print("finished")
      first = False
  rospy.spin()