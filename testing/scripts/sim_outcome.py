#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Int16
from std_msgs.msg import Bool
from motion.msg import Dmp
from detector.msg import Outcome
from detector.msg import State
from motion.msg import Action
from som.msg import GripperOrientation
from cog_learning.msg import LatentGoalDnf
from cog_learning.msg import Goal

class Testing(object):
   def __init__(self):
      rospy.init_node('test_vae', anonymous=True)
      self.pub_perception = rospy.Publisher("/habituation/perception_new_goal", Outcome, queue_size=1, latch=True)
      self.tot_learning = 0
      self.tot_not_learning = 0
      self.tot_signal = 0
      self.busy_exploit = False
      self.busy_not_exploit = False
      self.new_sample = False
      self.outcome
      rospy.Subscriber("/cog_learning/exploitation/learning", Float64, self.callback_learning)
      rospy.Subscriber("/cog_learning/exploitation/not_learning", Float64, self.callback_not_learning)

   def callback_learning(self,msg):
      if msg.data > 0.9:
         self.tot_learning += 1
         self.tot_signal = 0
      else:
         self.tot_learning = 0
         self.busy_exploit = False
      if self.tot_learning > 10 and not self.busy_exploit:
         self.busy_exploit = True
         print("LEARNING...")

         self.new_sample = True

   def callback_not_learning(self,msg):
      if msg.data > 0.9:
         self.tot_not_learning += 1
         self.tot_signal = 0
      else:
         self.tot_not_learning = 0
         self.busy_not_exploit = False
      if self.tot_not_learning > 15 and not self.busy_not_exploit and not self.send_signal:
         self.busy_not_exploit = True
         print("LEARNING NOTHING")

         self.new_sample = True
      if self.tot_learning == 0 and self.tot_not_learning == 0:
         self.tot_signal += 1
         if self.tot_signal > 100:
            self.new_sample = False
            self.tot_signal = 0

   def wait_for_ready(self):
      while not self.new_sample:
         pass


if __name__ == "__main__":
   test = Testing()
   rospy.sleep(0.5)
   for x in range(0,0.15,0.01):
      for y in range(-0.1,0.1,0.01):
         for angle in range(-180,180,1):

         

   rospy.spin()