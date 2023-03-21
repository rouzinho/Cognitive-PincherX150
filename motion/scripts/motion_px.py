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
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from som.msg import PoseRPY
from som.msg import GripperOrientation
from som.msg import VectorAction
from motion.msg import Dmp
from som.msg import ListPose
from som.srv import *
import os
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
        gdp = rospy.ServiceProxy('get_dmp_plan', GetDMPPlan)
        resp = gdp(x_0, x_dot_0, t_0, goal, goal_thresh, 
                   seg_length, tau, dt, integrate_iter)
    except rospy.ServiceException:
        print("Service call failed: %s")
    print("DMP planning done")   
            
    return resp;

def return_bmu(sample):
  #rospy.wait_for_service('get_bmu')
  bm = rospy.ServiceProxy('som_pose/get_bmu', GetBMU)
  try:
      req = GetBMURequest()
      req.sample = sample
      resp = bm(req)
      return resp
  except rospy.ServiceException as e:
      print("Service call failed: %s"%e)


class Motion(object):
  def __init__(self):
    super(Motion, self).__init__()
    #rospy.init_node('motion', anonymous=True)
    #rate = rospy.Rate(100)
    self.bot = InterbotixManipulatorXS("px150", "arm", "gripper")
    self.pub_gripper = rospy.Publisher("/px150/commands/joint_single", JointSingleCommand, queue_size=1, latch=True)
    self.pub_touch = rospy.Publisher("/outcome_detector/touch", Bool, queue_size=1, latch=True)
    self.pub_bmu = rospy.Publisher("/som_pose/som/node_value/bmu", GripperOrientation, queue_size=1, latch=True)
    self.pub_path = rospy.Publisher("/som_pose/som/dmp_path", ListPose, queue_size=1, latch=True)
    self.pub_new_state = rospy.Publisher("/depth_perception/activate", Bool, queue_size=1, latch=True)
    self.pub_outcome = rospy.Publisher("/outcome_detector/activate", Bool, queue_size=1, latch=True)
    rospy.Subscriber('/px150/joint_states', JointState, self.callback_joint_states)
    rospy.Subscriber('/proprioception/joint_states', JointState, self.callback_proprioception)
    rospy.Subscriber('/motion_pincher/go_to_pose', PoseRPY, self.callback_pose)
    rospy.Subscriber('/motion_pincher/gripper_orientation/first_pose', GripperOrientation, self.callback_first_pose)
    rospy.Subscriber('/motion_pincher/vector_action', VectorAction, self.callback_vector_action)
    rospy.Subscriber('/motion_pincher/touch_pressure', UInt16, self.callback_pressure)
    rospy.Subscriber('/depth_perception/new_state', Bool, self.callback_new_state)
    rospy.Subscriber('/depth_perception/retry', Bool, self.callback_retry)
    rospy.Subscriber('/motion_pincher/exploration', Bool, self.callback_exploration)
    rospy.Subscriber('/motion_pincher/exploitation', Bool, self.callback_exploitation)
    rospy.Subscriber('/motion_pincher/dmp', Dmp, self.callback_dmp)
    rospy.Subscriber('/depth_perception/name_state', String, self.callback_name_state)

    self.gripper_state = 0.0
    self.js = JointState()
    self.js_positions = []
    self.stop_pressing = False
    self.init_pose = geometry_msgs.msg.Pose()
    self.first_pose = GripperOrientation()
    self.last_pose = GripperOrientation()
    self.action = VectorAction()
    self.bool_init_p = False
    self.bool_last_p = False
    self.bool_act = False
    self.possible_grasp = [0.0,1.0]
    self.possible_roll = [-1.5,-1.2,-0.9,-0.6,-0.3,0,0.3,0.6,0.9,1.2,1.5]
    self.move = False
    self.move_dmp = False
    self.activate_dmp = False
    self.name_state = ""
    self.path = []
    self.name_ee = "/home/altair/interbotix_ws/src/motion/dmp/js.bag"
    self.record = False
    self.dims = 5
    self.dt = 1.0
    self.K_gain = 100              
    self.D_gain = 2.0 * np.sqrt(self.K_gain)      
    self.num_bases = 5
    self.single_msg = True
    self.explore = False
    self.exploit = False
    self.dmp = Dmp()
    self.dmp_name = ""
    self.dmp_found = False
    self.goal_dmp = False
    self.pose_ee = Pose()
    self.bool_dmp_plan = False
    self.path = ListPose()
    self.prop = JointState()

  def callback_pose(self,msg):
    self.init_position()
    self.bot.arm.set_ee_pose_components(x=msg.x, y=msg.y, z=msg.z, roll=msg.r, pitch=msg.p)
    self.init_position()
    self.sleep_pose()

  def callback_first_pose(self,msg):
    if self.bool_init_p == False and self.explore:
      self.first_pose.x = msg.x
      self.first_pose.y = msg.y
      self.first_pose.pitch = msg.pitch
      self.bool_init_p = True
      print("first pose : ",self.first_pose)
    elif self.bool_init_p == True and self.explore:
      self.last_pose.x = msg.x
      self.last_pose.y = msg.y
      self.last_pose.pitch = msg.pitch
      self.bool_last_p = True
    if self.exploit:
      self.last_pose.x = msg.x
      self.last_pose.y = msg.y
      self.last_pose.pitch = msg.pitch
      self.goal_dmp = True
      self.bool_last_p = True

  def callback_vector_action(self,msg):
    if self.bool_act == False:
      self.action.x = msg.x
      self.action.y = msg.y
      self.action.z = 0.0
      self.action.roll = msg.roll
      self.action.grasp = msg.grasp
      self.bool_act = True
      print("action : ",self.action)

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
    self.prop = msg
    if self.record == True:
      self.write_joints_bag(self.name_ee,self.prop)

  def callback_dmp(self,msg):
    self.dmp.x = msg.x
    self.dmp.y = msg.y
    self.dmp.roll = msg.roll
    self.dmp.pitch = msg.pitch
    self.dmp.grasp = msg.grasp
    self.dmp_found, self.dmp_name = self.find_dmp(msg)
    if self.dmp_found:
      print("found DMP : ",self.dmp_name)
    else:
      print("DMP not found")

  def callback_new_state(self,msg):
    if msg.data == True:
      self.bool_init_p = False
      self.bool_last_p = False
      self.bool_act = False
      self.make_dmp()
      self.update_offline_dataset(True)
      self.delete_js_bag()

  def callback_retry(self,msg):
    if msg.data == True:
      self.bool_last_p = False
      self.update_offline_dataset(False)

  def callback_exploration(self,msg):
    self.explore = msg.data
    self.bool_act = False
    self.bool_init_p = False
    self.bool_last_p = False

  def callback_exploitation(self,msg):
    self.exploit = msg.data
    self.bool_act = False
    self.bool_init_p = False
    self.bool_last_p = False

  def callback_name_state(self,msg):
    self.name_state = msg.data

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
  def get_explore(self):
    return self.explore
  
  #if it's exploiting ->learning a skill
  def get_exploit(self):
    return self.exploit

  #update dataset 
  def update_offline_dataset(self,status):
    name_dataset = "/home/altair/interbotix_ws/src/motion/dataset/datas.txt"
    exist = path.exists(name_dataset)
    opening = ""
    if(exist == False):
      opening = "w"
    else:
      opening = "a"
    data = str(self.last_pose.x) + " " + str(self.last_pose.y) + " " + str(self.last_pose.pitch) + " " + str(self.action.x) + " " + str(self.action.y) + " " + str(self.action.roll) + " " + str(self.action.grasp) + " " +self.name_state + " " + str(status) + "\n"
    with open(name_dataset, opening) as f:
        f.write(data)
    f.close()

  #naming the DMP
  def name_dmp(self):
    name = "/home/altair/interbotix_ws/src/motion/dmp/"
    nx = "x"+str(round(self.action.x,2))
    ny = "y"+str(round(self.action.y,2))
    nr = "r"+str(round(self.action.roll,1))
    gr = "g"+str(round(self.action.grasp,0))
    p = "p"+str(round(self.last_pose.pitch,1))
    name = name + nx + ny + nr + gr + p + "end.bag"

    return name
    
  #find the dmp file
  def find_dmp(self,dmp):
    name_dir = "/home/altair/interbotix_ws/src/motion/dmp/"
    found = False
    right_file = ""
    for file in os.listdir(name_dir):
        p_x = file.find('x')
        p_y = file.find('y')
        p_g = file.find('g')
        p_r = file.find('r')
        p_p = file.find('p')
        p_end = file.find('end')
        x = file[p_x+1:p_y]
        y = file[p_y+1:p_r]
        r = file[p_r+1:p_g]
        g = file[p_g+1:p_p]
        p = file[p_p+1:p_end]
        x = float(x)
        y = float(y)
        r = float(r)
        g = float(g)
        p = float(p)
        if dmp.x - x < 0.05 and dmp.y - y < 0.05 and dmp.roll - r < 0.05 and dmp.grasp - g < 0.05 and dmp.pitch - p < 0.05:
            found = True
            right_file = file
    right_file = name_dir + right_file
    return found, right_file

  def make_dmp(self):
    traj = self.form_data_joint_states()
    resp = self.makeLFDRequest(traj)
    n = self.name_dmp()
    self.write_dmp_bag(resp,n)

  def play_motion_dmp(self):
    tmp = self.js_positions
    curr = []
    for i in range(0,5):
      curr.append(tmp[i])
    resp = self.get_dmp(self.dmp_name)
    #print("Get DMP :")
    #print(resp)
    makeSetActiveRequest(resp.dmp_list)
    goal, found = self.pose_to_joints(self.last_pose.x,self.last_pose.y,0.03,self.dmp.roll,self.dmp.pitch) 
    print("goal : ",goal)
    print("cartesian goal : ",self.last_pose)
    if found:
      print("found goal to reach")
    goal_thresh = [0.01,0.01,0.01,0.01,0.01]
    x_0 = curr         #Plan starting at a different point than demo 
    x_dot_0 = [0.0,0.0,0.0,0.0,0.0]   
    t_0 = 0                
    seg_length = -1          #Plan until convergence to goal
    tau = resp.tau /2       # /4 is good enough
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
      if self.dmp.grasp > 0.5:
        self.open_gripper()
      else:
        self.close_gripper()
      self.play_motion_dmp()
      self.close_gripper()
      self.sleep_pose()
      self.goal_dmp = False
      self.bool_last_p = False

  def get_move(self):
    return self.move

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
    jsc.cmd = -50.0
    self.pub_gripper.publish(jsc)
    while not self.stop_pressing and self.gripper_state > 0.020:
      pass
    jsc.cmd = 0
    self.pub_gripper.publish(jsc)

  #gather pose and action before moving on
  def define_action(self):
    if self.bool_init_p and self.bool_last_p:
      go = GripperOrientation()
      go.x = self.first_pose.x + self.action.x
      go.y = self.first_pose.y + self.action.y
      go.pitch = self.first_pose.pitch
      resp = return_bmu(go)
      self.last_pose.x = resp.bmu.x
      self.last_pose.y = resp.bmu.y
      self.last_pose.pitch = resp.bmu.pitch
      print("last pose : ",self.last_pose)
      self.bool_last_p = True
      
  #execute the action
  def execute_action(self,record_dmp):
    if self.bool_init_p and self.bool_last_p:
      print("in the loop")
      r = random.choice(self.possible_roll)
      g = random.choice(self.possible_grasp)
      print("roll ",r)
      print("grasp ",g)
      self.init_position()     
      if g > 0.5:
        self.open_gripper()
      else:
        self.close_gripper()
      self.record = record_dmp
      self.bot.arm.set_ee_pose_components(x=self.first_pose.x, y=self.first_pose.y, z=0.03, roll=r, pitch=self.first_pose.pitch)
      self.bot.arm.set_ee_pose_components(x=self.last_pose.x, y=self.last_pose.y, z=0.03, roll=r, pitch=self.last_pose.pitch)
      self.record = False
      self.close_gripper()
      self.sleep_pose()
      self.bool_last_p = False
      b = Bool()
      b.data = True
      self.pub_new_state.publish(b)
      self.pub_outcome.publish(b)
      #self.bool_init_p = False
      #self.bool_act = False

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
    #self.record = True
    #self.write_joints_bag(self.name_ee,self.js)
    self.bot.arm.set_ee_pose_components(x=0.35, y=0.03, z=0.03, roll=0.0, pitch=0.6)
    rospy.sleep(4)
    self.bot.arm.set_ee_pose_components(x=0.3, y=0.1, z=0.03, roll=0.0, pitch=0.6)
    rospy.sleep(4)
    #self.write_joints_bag(self.name_ee,self.js)
    #rospy.sleep(0.5)
    #elf.record = True
    #print(self.js_positions)
    #self.write_joints_bag(self.name_ee,self.js)
    #print("done first")
    #self.joints_pos()
    #print("done second in joints space")
    #self.bot.arm.set_ee_pose_components(x=0.3, y=0.0, z=0.02, roll=0.0, pitch=0.8)
    #self.record = False
    #self.write_joints_bag(self.name_ee,self.js)
    #ospy.sleep(0.5)
    #self.record = False
    #print(self.js)
    #bot.arm.set_single_joint_position("waist", -np.pi/4.0)
    #print("sleep pose")
    #self.bot.arm.go_to_home_pose()
    #self.bot.arm.set_ee_pose_components(x=0.3, y=-0.1, z=0.02, roll=0.0, pitch=0.8)
    #print(self.js)
    #self.bot.arm.go_to_home_pose()
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
  #first = True
  #record = False
  #go = GripperOrientation()
  #ac = VectorAction()
  #go.pitch = 0.6
  #ac.x = -0.01
  #ac.y = 0.09
  #ac.roll = 0.0
  #ac.grasp = 0.0
  #first = True
  rospy.sleep(2.0)
  #motion_planning.open_gripper()
  #motion_planning.close_gripper()
  #motion_planning.pose_to_joints(0.3,-0.1,0.02,0.0,0.8)  

  while not rospy.is_shutdown():
    if motion_pincher.get_explore():
      motion_pincher.execute_action(False)
    if motion_pincher.get_exploit():
      motion_pincher.execute_dmp()
    # if first:
    #   motion_pincher.test_interface()
    #   first = False

  rospy.spin()