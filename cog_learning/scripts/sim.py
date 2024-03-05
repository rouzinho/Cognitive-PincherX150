#!/usr/bin/env python3
import torch;
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import math
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 100
from cog_learning.nn_ga_old import *
from cog_learning.msg import LatentGoalDnf
from cog_learning.msg import LatentGoalNN
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Int16
from std_msgs.msg import Bool
from motion.msg import Dmp
from motion.msg import Action
from motion.msg import DmpAction
from motion.msg import DmpOutcome
from detector.msg import Outcome
from detector.msg import State
from cog_learning.msg import Goal

class Test(object):
   def __init__(self):
      rospy.init_node('simulation', anonymous=True)
      self.pub_action_sample = rospy.Publisher('/motion_pincher/action_sample', Action, latch=True, queue_size=1)
      self.pub_state = rospy.Publisher('/cog_learning/object_state', State, latch=True, queue_size=1)
      self.pub_dmp_outcome = rospy.Publisher('/cog_learning/dmp_outcome', DmpOutcome, latch=True, queue_size=1)
      self.pub_latent_dnf = rospy.Publisher('/cog_learning/latent_space', LatentGoalDnf, latch=True, queue_size=1)
      self.pub_id = rospy.Publisher('/cog_learning/id_object', Int16, latch=True, queue_size=1)
      self.current_goal = LatentGoalDnf()
      rospy.Subscriber("/intrinsic/new_goal", Goal, self.callback_goal)
      rospy.Subscriber("/motion_pincher/activate_dmp", Dmp, self.callback_dmp)
      rospy.Subscriber("/cog_learning/ready", Bool, self.callback_ready)

   def callback_goal(self, msg):
      if msg.value > 0.5:
         print("Got latent value : ",msg)
         self.current_goal.latent_x = msg.x
         self.current_goal.latent_y = msg.y
      
   
   def callback_dmp(self, msg):
      print("Got DMP activation : ", msg)

   def callback_ready(self, msg):
      if msg.data == True:
         self.send_latent()

   def send_action(self, msg):
      self.pub_action_sample.publish(msg)

   def send_state(self, msg):
      self.pub_state.publish(msg)

   def send_dmp_outcome(self, msg):
      self.pub_dmp_outcome.publish(msg)

   def send_latent(self, val):
      self.pub_latent_dnf.publish(val)

   def send_id(self, val):
      self.pub_id.publish(val)

if __name__ == "__main__":
   sim = Test()
   rospy.sleep(2)
   t = Int16()
   t.data = 34
   sim.send_id(t)
   s = State()
   s.state_x = 0.2
   s.state_y = 0.0
   s.state_angle = 45
   a = Action()
   a.lpos_x = 0.3
   a.lpos_y = 0.0
   a.lpos_pitch = 1.0
   act = DmpOutcome()
   act.v_x = 0.1
   act.v_y = 0
   act.v_pitch = 0.2
   act.grasp = 0
   act.roll = 0.1
   act.angle = 10
   act.x = 0.1
   act.y = 0.0
   act.touch = 0
   sim.send_action(a)
   rospy.sleep(0.5)
   sim.send_state(s)
   rospy.sleep(0.5)
   sim.send_dmp_outcome(act)
   print("sent 1 sample, sleeping 15s")
   rospy.sleep(15.0)
   s = State()
   s.state_x = 0.3
   s.state_y = -0.1
   s.state_angle = 100
   a = Action()
   a.lpos_x = 0.3
   a.lpos_y = -0.2
   a.lpos_pitch = 1.0
   act = DmpOutcome()
   act.v_x = 0.0
   act.v_y = -0.1
   act.v_pitch = 0.0
   act.grasp = 1
   act.roll = 0.1
   act.angle = 10
   act.x = 0.0
   act.y = -0.1
   act.touch = 1
   sim.send_action(a)
   rospy.sleep(0.5)
   sim.send_state(s)
   rospy.sleep(0.5)
   sim.send_dmp_outcome(act)
   print("sent 2nd sample, sleep 15s")
   rospy.sleep(15.0)
   lat = LatentGoalDnf()
   lat.latent_x = 70
   lat.latent_y = 47
   sim.send_latent(lat)
   print("sent 1st latent")
   rospy.sleep(2.0)
   lat.latent_x = 54
   lat.latent_y = 69
   sim.send_latent(lat)
   rospy.spin()