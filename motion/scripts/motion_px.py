#!/usr/bin/env python3


from os import name

import sys
import copy
import rospy
import roslib;
import rosbag
import numpy as np
from math import pi
import math
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
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
from som.msg import ListPeaks
import os
import os.path
from os import path
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import interbotix_common_modules.angle_manipulation as ang
from pathlib import Path
from tf.transformations import *
from cluster_message.srv import *
from cog_learning.msg import DmpDnf
from cog_learning.msg import ActionDmpDnf
from cog_learning.msg import LatentGoalDnf
from cog_learning.msg import SamplePred
from cog_learning.srv import *
from detector.srv import *
from detector.msg import State
from detector.msg import Outcome
from std_srvs.srv import Empty

class Motion(object):
  def __init__(self):
    super(Motion, self).__init__()
    #rospy.init_node('motion', anonymous=True)
    #rate = rospy.Rate(100)
    self.threshold_touch_min = 0.02
    self.threshold_touch_max = 0.0278
    self.record = False
    self.current_folder = ""
    self.gripper_state = 0.0
    self.js = JointState()
    self.js_positions = []
    self.touch_value = False
    self.last_pose = GripperOrientation()
    self.poses = []
    self.possible_grasp = [0.0,1.0]
    self.possible_roll = [-1.5,-0.7,0,0,0.7,1.5]
    self.possible_pitch = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6]
    self.possible_action = []
    self.move = False
    self.l_touch = 0
    self.name_state = ""
    self.path = []
    self.dims = 5
    self.dt = 1.0
    self.K_gain = 100              
    self.D_gain = 2.0 * np.sqrt(self.K_gain)      
    self.num_bases = 5
    self.single_msg = True
    self.rnd_explore = False
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
    self.change_action = True
    self.choice = 0
    self.count_init = 0
    self.b_init = False
    self.list_peaks = []
    self.ready_init = False
    self.go = False
    self.count_end = 0
    self.b_end = False
    self.state_object = State()
    self.new_state = False
    self.emer_pose = GripperOrientation()
    self.outcome = Outcome()
    self.choose_pred = True
    self.pause_process = False
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
    self.pub_dmp_candidate = rospy.Publisher("/cluster_msg/dmp_candidate", Dmp, queue_size=1, latch=True)
    self.pub_display_action = rospy.Publisher("/display/dmp", Dmp, queue_size=1, latch=True)
    self.pub_dnf_action = rospy.Publisher("/motion_pincher/dmp_dnf", LatentGoalDnf, queue_size=1, latch=True)
    self.pub_trigger_state = rospy.Publisher("/outcome_detector/trigger_state", Bool, queue_size=1, latch=True)
    self.pub_inhib = rospy.Publisher("/motion_pincher/inhibition", Float64, queue_size=1, latch=True)
    self.pub_init = rospy.Publisher("/motion_pincher/initiate_action", Float64, queue_size=1, latch=True)
    rospy.Subscriber('/px150/joint_states', JointState, self.callback_joint_states)
    rospy.Subscriber('/motion_pincher/gripper_orientation/first_pose', GripperOrientation, self.callback_first_pose)
    rospy.Subscriber('/cluster_msg/new_state', Bool, self.callback_new_state)
    rospy.Subscriber('/cog_learning/rnd_exploration', Float64, self.callback_rnd_exploration)
    rospy.Subscriber('/cog_learning/direct_exploration', Float64, self.callback_direct_exploration)
    rospy.Subscriber('/cog_learning/exploitation', Float64, self.callback_exploitation)
    rospy.Subscriber('/motion_pincher/dmp_direct_exploration', Dmp, self.callback_dmp_direct_exploration)
    rospy.Subscriber('/motion_pincher/retrieve_dmp', Dmp, self.callback_dmp)
    rospy.Subscriber('/cluster_msg/signal', Float64, self.callback_end)
    rospy.Subscriber("/cluster_msg/pause", Float64, self.callback_pause)
    rospy.Subscriber("/motion_pincher/activate_actions", ActionDmpDnf, self.callback_actions)
    rospy.Subscriber("/motion_pincher/change_action", Bool, self.callback_change)
    rospy.Subscriber("/motion_pincher/list_candidates", ListPose, self.callback_list_pose)
    rospy.Subscriber("/motion_pincher/ready_init", Float64, self.callback_ready_init)
    rospy.Subscriber("/motion_pincher/bool_init", Bool, self.callback_bool_init)
    rospy.Subscriber("/cluster_msg/pause_dft", Bool, self.callback_pause_process)

  def transform_dmp_cam_rob(self, dmp_):
    rospy.wait_for_service('transform_dmp_cam_rob')
    try:
        transform_dmp = rospy.ServiceProxy('transform_dmp_cam_rob', tfCamRob)
        resp1 = transform_dmp(dmp_)
        return resp1.dmp_robot
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

  def transform_dmp_rob_cam(self, dmp_):
    rospy.wait_for_service('transform_dmp_rob_cam')
    try:
        transform_dmp = rospy.ServiceProxy('transform_dmp_rob_cam', tfRobCam)
        resp1 = transform_dmp(dmp_)
        return resp1.dmp_cam
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

  def get_object_state(self,req):
    rospy.wait_for_service('get_object_state')
    try:
        action_predictions = rospy.ServiceProxy('get_object_state', GetState)
        resp1 = action_predictions(req)
        return resp1.state
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

  def get_action_prediction(self, act):
    rospy.wait_for_service('predict_action')
    try:
        action_predictions = rospy.ServiceProxy('predict_action', PredAction)
        resp1 = action_predictions(act)
        return resp1.outputs
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

  def callback_ready_init(self,msg):
    if msg.data > 0.9:
      self.count_init += 1
    else:
      self.count_init = 0
      self.b_init = False
    if self.count_init > 10 and not self.b_init:
      self.b_init = True
      self.count_init = 0
      if self.rnd_explore or self.direct_explore:
        print("LAUNCHING EXPLORATION")
        self.send_init(1.0)
      if self.exploit:
        print("LAUNCHING EXPLOITATION")
        self.init_exploitation()

  def callback_bool_init(self,msg):
    if msg.data == True:
      if self.rnd_explore or self.direct_explore:
        self.send_init(1.0)
      if self.exploit:
        print("LAUNCHING EXPLOITATION")
        self.init_exploitation()

  def callback_first_pose(self,msg):
    #print("got pose : ",msg)
    self.send_init(0.0)
    tmp = GripperOrientation()
    tmp.x = msg.x
    tmp.y = msg.y
    tmp.pitch = msg.pitch
    #print("tmp",tmp)
    self.ready = False
    if self.rnd_explore and len(self.poses) == 1:
      self.poses.append(tmp)
      self.go = True
    if self.rnd_explore and len(self.poses) < 2:
      self.poses.append(tmp)
      rospy.sleep(1.0)
      self.send_init(1.0)
    if self.direct_explore and len(self.poses) == 0:
      self.poses.append(tmp)
      self.go = True
    if self.exploit and len(self.poses) == 0:
      self.poses.append(tmp)
      self.go = True
    #print("poses cb : ",self.poses)

  def callback_joint_states(self,msg):
    self.js = msg
    self.js_positions = msg.position
    self.gripper_state = msg.position[6]
    if self.gripper_state < self.threshold_touch_max and self.gripper_state > self.threshold_touch_min:
      self.count_touch = self.count_touch +1
      #print(self.gripper_state)
      #print("touch : ",self.count_touch)
    else:
      self.count_touch = 0
    if(self.count_touch > 450):
      self.l_touch = self.count_touch
      #self.touch_value = True
    #self.count_touch = 0
    #self.pub_touch.publish(t)
    #print(self.touch_value)

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
      #if self.recording_dmp and (self.rnd_explore or self.direct_explore):
      ##  self.make_dmp()
      #  self.delete_js_bag()
      #  self.recording_dmp = False
      #self.send_inhibition(True)

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

  def callback_end(self,msg):
    if msg.data > 0.5:
      self.count_end += 1
    else:
      self.count_end = 0
      self.b_end = False
    if self.count_end > 10 and not self.b_end:
      self.b_end = True
      self.count_end = 0
      if self.rnd_explore or self.direct_explore:
        print("Init new action...")
        self.send_init(1.0)
      if self.exploit:
        self.init_exploitation()
    
  #
  def callback_change(self,msg):
    if msg.data == True:
      self.change_action = True

  def callback_pause_process(self,msg):
    self.pause_process = msg.data

  def get_ready(self):
    return self.ready
  
  def set_seady(self,val):
    self.ready = val

  def get_go(self):
    return self.go
  
  def set_got(self,val):
    self.go = val

  def callback_dmp_direct_exploration(self,msg):
    self.dmp_direct_explore.v_x = msg.v_x
    self.dmp_direct_explore.v_y = msg.v_y
    self.dmp_direct_explore.v_pitch = msg.v_pitch
    self.dmp_direct_explore.roll = msg.roll
    self.dmp_direct_explore.grasp = msg.grasp
    print("direct explore : ",self.dmp_direct_explore)

  def callback_pause(self, msg):
    self.poses = []

  def callback_actions(self,msg):
    self.possible_action = []
    for i in msg.list_action:
      tmp_a = [i.v_x,i.v_y,i.v_pitch,i.roll,i.grasp,i.dnf_x,i.dnf_y]
      self.possible_action.append(tmp_a)
      #print("callback : ",self.possible_action)

  def callback_list_pose(self,msg):
    self.list_peaks = []
    for i in msg.list_peaks:
      self.list_peaks.append(i)

  def get_number_pose(self):
    return len(self.poses)

  def send_state(self,val):
    tmp = Bool()
    tmp.data = val
    self.pub_trigger_state.publish(tmp)

  def send_init(self,val):
    d = Float64()
    d.data = val
    self.pub_init.publish(d)

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
  
  #if it's exploring
  def get_rnd_explore(self):
    return self.rnd_explore
  
  def get_direct_explore(self):
    return self.direct_explore
  
  #if it's exploiting ->learning a skill
  def get_exploit(self):
    return self.exploit
  
  def apply_pause(self):
    while self.pause_process:
      pass

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

  def find_best_pose(self,x,y,z,r,p):
    i = 0
    pair = True
    found = False
    pose = False
    while not found:
      if pair:
        p_i = i
      else:
        p_i = -i
      j, found = self.pose_to_joints(x,y,z,r,p+p_i)
      if found:
        pose = True
      if not pair:
        pair = True
        i += 0.1
      else:
        pair = False
      if i > 1.6:
        found = True
        pose = False
      

    return p+p_i, pose
      
  #execute the action
  def execute_rnd_exploration(self):
    self.go = False
    self.send_state(True)
    print("RANDOM EXPLORATION")
    correct_motion = False
    attempts = 0
    p_first = 0
    p_last = 0
    r_ = 0
    g_ = 0
    p_ = 0
    z_ = 0.06
    while not correct_motion and attempts < 3:
      r_ = random.choice(self.possible_roll)
      g_ = random.choice(self.possible_grasp)
      p_ = random.choice(self.possible_pitch)
      #r = 0
      #g = 0
      z_ = 0.06
      p_first, found_1 = self.find_best_pose(self.poses[0].x,self.poses[0].y,z_,r_,p_)
      if found_1:
        p_last, found_2 = self.find_best_pose(self.poses[1].x,self.poses[1].y,z_,r_,p_first)
      if found_1 and found_2:
        m_first = f"first pose : x {self.poses[0].x}, y {self.poses[0].y}, pitch {p_first}, roll {r_}"
        message = f"Second pose : x {self.poses[1].x}, y {self.poses[1].y}, pitch {p_last}"
        print(m_first)
        print(message)
        correct_motion = True
      attempts += 1
    if correct_motion:
      self.dmp_explore.v_x = self.poses[1].x - self.poses[0].x
      self.dmp_explore.v_y = self.poses[1].y - self.poses[0].y
      self.dmp_explore.v_pitch = p_first
      self.dmp_explore.roll = r_
      self.dmp_explore.grasp = g_
      self.dmp_explore.fpos_x = self.poses[0].x
      self.dmp_explore.fpos_y = self.poses[0].y
      msg = self.transform_dmp_cam_rob(self.dmp_explore)
      print("DMP : ",msg)
      #display on interface
      self.pub_display_action.publish(msg)
      self.pub_dmp_action.publish(msg)
      self.bot.gripper.set_pressure(1.0)
      #rospy.sleep(3.0)
      self.init_position()     
      if g_ > 0.5:
        self.bot.gripper.open()
        print("OPEN GRIPPER")
        z_ = 0.05
      else:
        self.bot.gripper.close()
        print("CLOSE GRIPPER")
      self.bot.arm.set_ee_pose_components(x=self.poses[0].x, y=self.poses[0].y, z=z_, roll=r_, pitch=p_first)
      self.bot.arm.set_ee_pose_components(x=self.poses[1].x, y=self.poses[1].y, z=z_, roll=r_, pitch=p_last)
      self.record = False
      self.bot.gripper.close()
      #rospy.sleep(3.0)
      self.init_position()  
      self.sleep_pose()
      #send touch value
      t = Bool()
      t.data = self.touch_value
      print("touch value : ",self.touch_value)
      self.pub_touch.publish(t)
      print("ACTION DONE")
      self.last_time = rospy.get_time()
      sample = Action()
      sample.fpos_x = self.poses[0].x
      sample.fpos_y = self.poses[0].y
      sample.fpos_pitch = p_first
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
    else:
      print("no valid motion found")
      self.poses.pop()

  def execute_direct_exploration(self):
    self.go = False
    self.send_state(True)
    self.bound_exploration()
    print("DIRECT EXPLORATION")
    #print(self.poses)
    #display on interface
    self.pub_display_action.publish(self.dmp_direct_explore)
    self.dmp_direct_explore.fpos_x = self.poses[0].x
    self.dmp_direct_explore.fpos_y = self.poses[0].y
    msg = self.transform_dmp_rob_cam(self.dmp_direct_explore)
    #print(msg)
    self.pub_dmp_action.publish(msg)
    self.bot.gripper.set_pressure(1.0)
    z_ = 0.06
    #rospy.sleep(3.0)
    self.init_position()     
    if self.dmp_direct_explore.grasp > 0.5:
      self.bot.gripper.open()
      z_ = 0.05
    else:
      self.bot.gripper.close()
    lpos_x = self.poses[0].x + msg.v_x
    lpos_y = self.poses[0].y + msg.v_y
    p_first, found_1 = self.find_best_pose(self.poses[0].x,self.poses[0].y,z_,self.dmp_direct_explore.roll,self.dmp_direct_explore.v_pitch)
    if found_1:
      p_last, found_2 = self.find_best_pose(lpos_x,lpos_y,z_,self.dmp_exploit.roll,p_first)
    if found_1 and found_2:
      m_first = f"first pose : x {self.poses[0].x}, y {self.poses[0].y}, pitch {p_first}, roll {self.dmp_direct_explore.roll}"
      message = f"Second pose : x {lpos_x}, y {lpos_y}, pitch {p_last}"
      print(m_first)
      print(message)
      self.bot.arm.set_ee_pose_components(x=self.poses[0].x, y=self.poses[0].y, z=z_, roll=self.dmp_direct_explore.roll, pitch=p_first)
      self.bot.arm.set_ee_pose_components(x=lpos_x, y=lpos_y, z=z_, roll=self.dmp_direct_explore.roll, pitch=p_last)
      self.record = False
      self.bot.gripper.close()
      #rospy.sleep(2.0)
      self.init_position()  
      self.sleep_pose()
      self.bot.gripper.open()
      #send touch value
      t = Bool()
      t.data = self.touch_value
      self.pub_touch.publish(t)
      print("ACTION DONE")
      self.last_time = rospy.get_time()
      sample = Action()
      sample.fpos_x = self.poses[0].x
      sample.fpos_y = self.poses[0].y
      sample.fpos_pitch = p_first
      self.pub_action_sample.publish(sample)
      self.poses.pop()
      self.ready_depth = False
      self.ready_outcome = False
      self.touch_value = False
      b = Bool()
      b.data = True
      self.pub_activate_perception.publish(b)
    else:
      print("no valid pose found !")
      self.poses.pop()
      #self.bot.gripper.open()
      
  def bound_exploration(self):
    if self.dmp_direct_explore.roll > 1.5:
      self.dmp_direct_explore.roll = 1.5
    if self.dmp_direct_explore.roll < -1.5:
      self.dmp_direct_explore.roll = -1.5
    if self.dmp_direct_explore.v_pitch > 1.6:
      self.dmp_direct_explore.v_pitch = 1.6
    if self.dmp_direct_explore.v_pitch < 0.2:
      self.dmp_direct_explore.v_pitch = 0.2

  #init old
  def init_exploitation(self):
    if self.change_action:
      s = len(self.possible_action)
      if s > 1:
        #change this here
        l_pred = []
        for i in range(0,s):
          st = self.get_object_state()
          sample = SamplePred()
          sample.state_x = st.state_x
          sample.state_y = st.state_y
          sample.state_angle = st.state_angle
          sample.dnf_x = self.possible_action[i][5]
          sample.dnf_y = self.possible_action[i][6]
          l_pred.append(sample)
        res_outputs = self.get_action_prediction(l_pred)
        min_p = 0
        ind = 0
        for i in range(0,len(res_outputs)):
          if res_outputs[i] > min_p:
            ind = i
            min_p = res_outputs[i]
        self.choice = ind
        self.change_action = False
    dmp_choice = self.possible_action[self.choice]
    #suc = self.get_correct_pose(dmp_choice[2])
    self.dmp_exploit = Dmp()
    self.dmp_exploit.v_x = dmp_choice[0]
    self.dmp_exploit.v_y = dmp_choice[1]
    self.dmp_exploit.v_pitch = dmp_choice[2]
    self.dmp_exploit.roll = dmp_choice[3]
    self.dmp_exploit.grasp = dmp_choice[4]
    #print("choosing DMP : ",self.dmp_exploit)
    #self.pub_dmp_candidate.publish(dmp_exploit)
    #rospy.sleep(3.0)
    print("Init ACTION !")
    self.send_init(1.0)

  def execute_exploitation(self):
    self.go = False
    self.send_state(True)
    print("DIRECT EXPLOITATION")
    #display on the interface
    self.pub_display_action.publish(self.dmp_exploit)
    #include first pose
    self.dmp_exploit.fpos_x = self.poses[0].x
    self.dmp_exploit.fpos_y = self.poses[0].y
    msg = self.transform_dmp_rob_cam(self.dmp_exploit)
    lat_action = LatentGoalDnf()
    #print("choice : ",self.choice)
    #print("possible action : ",self.possible_action)
    lat_action.latent_x = self.possible_action[self.choice][5]
    lat_action.latent_y = self.possible_action[self.choice][6]
    self.pub_dnf_action.publish(lat_action)
    self.emer_pose.x = self.poses[0].x
    self.emer_pose.y = self.poses[0].y
    self.bot.gripper.set_pressure(1.0)
    z_ = 0.06
    self.init_position()     
    if self.dmp_exploit.grasp > 0.5:
      self.bot.gripper.open()
      z_ = 0.05
    else:
      self.bot.gripper.close()
    if len(self.poses) == 0:
      self.poses.append(self.emer_pose)
    fpos_x = self.poses[0].x
    fpos_y = self.poses[0].y
    lpos_x = fpos_x + msg.v_x
    lpos_y = fpos_y + msg.v_y
    p_first, found_1 = self.find_best_pose(self.poses[0].x,self.poses[0].y,z_,self.dmp_exploit.roll,self.dmp_exploit.v_pitch)
    if found_1:
      p_last, found_2 = self.find_best_pose(lpos_x,lpos_y,z_,self.dmp_exploit.roll,p_first)
    if found_1 and found_2:
      m_first = f"first pose : x {self.poses[0].x}, y {self.poses[0].y}, pitch {p_first}, roll {self.dmp_exploit.roll}"
      message = f"Second pose : x {lpos_x}, y {lpos_y}, pitch {p_last}"
      print(m_first)
      print(message)
      self.bot.arm.set_ee_pose_components(x=self.poses[0].x, y=self.poses[0].y, z=z_, roll=self.dmp_exploit.roll, pitch=p_first)
      self.bot.arm.set_ee_pose_components(x=lpos_x, y=lpos_y, z=z_, roll=self.dmp_exploit.roll, pitch=p_last)
      self.record = False
      self.bot.gripper.close()
      self.init_position()  
      self.sleep_pose()
      t = Bool()
      t.data = self.touch_value
      #print("touch value : ",self.touch_value)
      self.pub_touch.publish(t)
      if self.dmp_exploit.grasp > 0.5:
        self.bot.gripper.open()
      print("ACTION DONE")
      self.l_touch = 0
      self.last_time = rospy.get_time()
      sample = Action()
      sample.fpos_x = fpos_x
      sample.fpos_y = fpos_y
      sample.fpos_pitch = p_first
      self.pub_action_sample.publish(sample)
      self.poses.pop()
      self.ready_depth = False
      self.ready_outcome = False
      self.touch_value = False
      self.count_touch = 0
      b = Bool()
      b.data = True
      self.pub_activate_perception.publish(b)
      #self.bot.gripper.open()
    else:
      print("no valid pose found !")
      self.poses.pop()
      #self.bot.gripper.open()
      self.init_exploitation()
    
    #send touch value
    

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
    #self.bot.gripper.open()
    self.bot.gripper.set_pressure(1.0)
    self.bot.gripper.close()
    #self.close_gripper()
    #rospy.sleep(2.0)
    #self.bot.gripper.open()

  def test_position(self):
    self.init_position()
    self.bot.arm.set_ee_pose_components(x=0.25, y=0, z=0.06, roll=1.4, pitch=1.1)
    self.bot.arm.set_ee_pose_components(x=0.33, y=0, z=0.06, roll=1.4, pitch=1.1)
    self.init_position()
    self.sleep_pose()

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
  #motion_pincher.test_interface()
  #motion_pincher.test_position()

  #motion_pincher.init_position()
  while not rospy.is_shutdown():
    if motion_pincher.get_rnd_explore() and motion_pincher.get_number_pose() == 2 and motion_pincher.get_go():
        print("EXECUTE rnd action")
        motion_pincher.execute_rnd_exploration()
    if motion_pincher.get_direct_explore() and motion_pincher.get_number_pose() == 1 and motion_pincher.get_go():
      print("EXECUTE Direct action")
      motion_pincher.execute_direct_exploration()
    if motion_pincher.get_exploit() and motion_pincher.get_number_pose() == 1 and motion_pincher.get_go():
      motion_pincher.execute_exploitation()
  #if first:
  #  motion_pincher.test_interface()
    #motion_pincher.run_possibilities()
  #  print("DONE !")  
  #  first = False

  rospy.spin()