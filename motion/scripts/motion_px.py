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
#from trajectory_msgs.msg import JointSingleCommand
from interbotix_xs_msgs.msg import *
from std_msgs.msg import Time
from std_msgs.msg import Header
from std_msgs.msg import Duration
from std_msgs.msg import UInt16
from sensor_msgs.msg import JointState
import os.path
from os import path
from interbotix_xs_modules.arm import InterbotixManipulatorXS
#import PyKDL as kdl
#import kdl_parser_py.urdf as kdl_parser
import interbotix_common_modules.angle_manipulation as ang

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
    self.bot = InterbotixManipulatorXS("px150", "arm", "gripper")
    self.pub = rospy.Publisher('/px150/gripper_controller/command', JointTrajectory, queue_size=1,latch=True)
    self.pub_traj = rospy.Publisher("/px150/commands/joint_trajectory", JointTrajectoryCommand, queue_size=1, latch=True)
    self.pub_group = rospy.Publisher("/px150/commands/joint_group", JointGroupCommand, queue_size=1, latch=True)
    self.pub_gripper = rospy.Publisher("/px150/commands/joint_single", JointSingleCommand, queue_size=1, latch=True)
    rospy.Subscriber('/px150/joint_states', JointState, self.joint_states)
    rospy.Subscriber('/motion/joint_states', JointState, self.late_joint_states)
    rospy.Subscriber('/motion_pincher/path_ee', Pose, self.callback_path)
    rospy.Subscriber('/motion_pincher/proprioception', Pose, self.callback_proprioception)
    rospy.Subscriber('/pressure', UInt16, self.get_pressure)
    self.gripper_state = 0.0
    self.js = JointState()
    self.js_positions = []
    self.stop = False
    ## Get the name of the robot - this will be used to properly define the end-effector link when adding a box
    #self.baselink = rospy.get_param("px150/baselink")
    #self.endlink = rospy.get_param("px150/ee_arm_link")
    #print(self.baselink)
    self.pose_goal = geometry_msgs.msg.Pose()
    self.init_pose = geometry_msgs.msg.Pose()
    self.pose_home = geometry_msgs.msg.Pose()
    self.current_pose = geometry_msgs.msg.Pose()
    self.ee_pose = geometry_msgs.msg.Pose()
    self.move = False
    self.move_dmp = False
    self.home = False
    self.activate_dmp = False
    self.goal_dmp = geometry_msgs.msg.Pose()
    self.count = 0
    self.path = []
    self.ee_is_on_path = False
    self.name_ee = "/home/altair/interbotix_ws/rosbags/forward.bag"
    self.name_dmp = "/home/altair/interbotix_ws/rosbags/forward_dmp.bag"
    self.record = False
    self.dims = 5
    self.dt = 1.0
    self.K_gain = 100              
    self.D_gain = 2.0 * np.sqrt(self.K_gain)      
    self.num_bases = 4
    self.stop = False

  def callback_path(self,data):
    print("GOT POSE")
    print(self.count)
    if self.count == 1:
      self.pose_goal.position.x = data.position.x
      self.pose_goal.position.y = data.position.y
      self.pose_goal.position.z = data.position.z
      #self.pose_goal.orientation.x = data.orientation.x
      #self.pose_goal.orientation.y = data.orientation.y
      #self.pose_goal.orientation.z = data.orientation.z
      #self.pose_goal.orientation.w = data.orientation.w
      self.count += 1
      self.path.append(copy.deepcopy(self.pose_goal))
    if self.count == 0:
      #first approach to not tackle to object
      self.path = []
      #but first set the first point as start state
      self.pose_goal.position.x = self.ee_pose.position.x
      self.pose_goal.position.y = self.ee_pose.position.y
      self.pose_goal.position.z = self.ee_pose.position.z
      #self.pose_goal.orientation.x = self.ee_pose.orientation.x
      #self.pose_goal.orientation.y = self.ee_pose.orientation.y
      #self.pose_goal.orientation.z = self.ee_pose.orientation.z
      #self.pose_goal.orientation.w = self.ee_pose.orientation.w
      self.path.append(copy.deepcopy(self.pose_goal))
      #then move on to the approach
      self.pose_goal.position.x = data.position.x
      self.pose_goal.position.y = data.position.y
      self.pose_goal.position.z = 0.1
      #self.pose_goal.orientation.x = data.orientation.x
      #self.pose_goal.orientation.y = data.orientation.y
      #self.pose_goal.orientation.z = data.orientation.z
      #self.pose_goal.orientation.w = data.orientation.w
      self.path.append(copy.deepcopy(self.pose_goal))
      #going down to the desired pose
      self.pose_goal.position.z = 0.03
      self.path.append(copy.deepcopy(self.pose_goal))
      self.count += 1
    if self.count == 2:
      print("DECIDE TO MOVE")
      self.count = 0
      self.move = True

  def get_pressure(self,msg):
    if msg.data < 200:
      self.stop = True
    else:
      self.stop = False

  def joint_states(self,msg):
    self.js = msg
    self.js_positions = msg.position
    self.gripper_state = msg.position[6]

  def late_joint_states(self,msg):
    if self.record == True:
      self.writeBagEE(self.name_ee,msg)

  def callback_proprioception(self,msg):
    self.ee_pose.position.x = msg.position.x
    self.ee_pose.position.y = msg.position.y
    self.ee_pose.position.z = msg.position.z
    self.ee_pose.orientation.x = msg.orientation.x
    self.ee_pose.orientation.y = msg.orientation.y
    self.ee_pose.orientation.z = msg.orientation.z
    self.ee_pose.orientation.w = msg.orientation.w
    #if self.move == True:
      #self.writeBagEE(self.name_ee,msg)
      #print(self.ee_pose)

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

  def writeBagEE(self,n,pos):
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

  def formDatasJS(self):
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
  def writeDMPBag(self,data,n):
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

  def readDMPBag(self,name):
    bag = rosbag.Bag(self.name_dmp)
    for topic, msg, t in bag.read_messages(topics=['dmp_pos']):
      print(msg)
    bag.close()

  def getDMP(self,name):
    bag = rosbag.Bag(name)
    for topic, msg, t in bag.read_messages(topics=['dmp_pos']):
      resp = msg
    bag.close()

    return resp

  def makeDMP(self):
    traj = self.formDatasJS()
    resp = self.makeLFDRequest(traj)
    print(resp)
    self.writeDMPBag(resp,self.name_dmp)

  def playMotionDMP(self):
    tmp = self.js_positions
    print(tmp)
    curr = []
    for i in range(0,5):
      curr.append(tmp[i])
    resp = self.getDMP(self.name_dmp)
    #print("Get DMP :")
    #print(resp)
    makeSetActiveRequest(resp.dmp_list)
    #goal = [0.3709951601922512, 0.08797463793307543, 0.7709327504038811, -0.07890698444098235, 1.505357144537811e-06]
    goal = [-0.3137078918516636, 0.2984987303111935, 0.4029244794423692, 0.07907685257397824, -1.1780148867046465e-06]
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
    path = self.shrinkDMP(planned_dmp.plan.points,goal)
    print("path length", len(path))
    for i in path:
      self.bot.arm.set_joint_positions(i,moving_time=0.4,accel_time=0.1)
      print(j)
      j+=1


  def shrinkDMP(self,data,desired):
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

  def poseToJoints(self,x,y,z,r,p):
    T_sd = np.identity(4)
    yaw = math.atan2(y,x)
    T_sd[:3,:3] = ang.eulerAnglesToRotationMatrix([r, p, yaw])
    T_sd[:3, 3] = [x, y, z]
    joints, found = self.bot.arm.set_ee_pose_matrix(T_sd, custom_guess=None, execute=False, moving_time=None, accel_time=None, blocking=True)
    return joints, found

  def openGripper(self):
    self.bot.gripper.open(2.0)

  def close_gripper(self):
    jsc = JointSingleCommand()
    jsc.name = "gripper"
    jsc.cmd = -50.0
    self.pub_gripper.publish(jsc)
    while not self.stop and self.gripper_state > 0.015:
      pass
    jsc.cmd = 0
    self.pub_gripper.publish(jsc)



  def test_interface(self):
    self.bot.arm.go_to_home_pose()
    #self.writeBagEE(self.name_ee,self.js)
    #rospy.sleep(1.0)
    self.record = True
    #self.writeBagEE(self.name_ee,self.js)
    self.bot.arm.set_ee_pose_components(x=0.15, y=0.0, z=0.02, roll=0.0, pitch=0.8)
    #self.writeBagEE(self.name_ee,self.js)
    #rospy.sleep(0.5)
    #elf.record = True
    #print(self.js_positions)
    #self.writeBagEE(self.name_ee,self.js)
    #print("done first")
    #self.joints_pos()
    #print("done second in joints space")
    self.bot.arm.set_ee_pose_components(x=0.3, y=0.0, z=0.02, roll=0.0, pitch=0.8)
    self.record = False
    #self.writeBagEE(self.name_ee,self.js)
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

if __name__ == '__main__':
  motion_planning = Motion()
  first = True
  rospy.sleep(2.0)
  #motion_planning.openGripper()
  #motion_planning.close_gripper()
  #motion_planning.poseToJoints(0.3,-0.1,0.02,0.0,0.8)  

  while not rospy.is_shutdown():
    if first:
      #motion_planning.test_interface()
      #motion_planning.makeDMP()
      motion_planning.init_position()
      motion_planning.playMotionDMP()
      motion_planning.sleep_pose()
      #motion_planning.makeDMP()
      #motion_planning.test_interface()
      #motion_planning.init_position()
      #motion_planning.reproduce_group()
      #motion_planning.init_position()
      print("slept")
      first = False
    rospy.spin()