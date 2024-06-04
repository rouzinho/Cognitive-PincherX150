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
      self.pub_perception = rospy.Publisher("/sim/perception", Goal, queue_size=1, latch=True)
      self.pub_error = rospy.Publisher("/sim/error", Goal, queue_size=1, latch=True)
      self.pub_value1 = rospy.Publisher("/value1", Goal, queue_size=1, latch=True)
      self.pub_value2 = rospy.Publisher("/value2", Goal, queue_size=1, latch=True)
      self.pub_new = rospy.Publisher("/new", Goal, queue_size=1, latch=True)
      self.pub_boost = rospy.Publisher("/boost", Float64, queue_size=1, latch=True)
      self.pub_eoa = rospy.Publisher("/sim/eoa", Float64, queue_size=1, latch=True)
      self.pub_time = rospy.Publisher("/sim/time", Float64, queue_size=1, latch=True)
      #self.pub_boost = rospy.Publisher("/sim/boost_lc", Float64, queue_size=1, latch=True)
      self.stop = 0.0
      self.count_stop = 0
      self.wait = False
      rospy.Subscriber("/sim/stop", Float64, self.callback_stop)

   def t(self):
      self.send_boost(1.0)
      rospy.sleep(0.5)
      #self.send_error_timing(87,0.9)
      self.send_error_timing(37,0.6)
      self.send_perception(24,0.0)
      """print("trial 1")
      self.send_trials()
      self.send_error_timing(87,0.9)
      print("trial 2")
      self.send_trials()
      self.send_error_timing(87,0.7)
      print("trial 3")
      self.send_trials()
      self.send_error_timing(87,0.4)
      print("trial 4")
      self.send_trials()
      self.send_error_timing(87,0.2)
      print("trial 5")
      self.send_trials()
      self.send_error_timing(87,0.05)
      print("trial 6")
      self.send_trials()
      self.send_error_timing(87,0.02)
      print("trial 7")
      self.send_trials()
      self.send_error_timing(87,0.02)
      print("trial 8")
      self.send_trials()
      self.send_error_timing(87,0.02)
      print("trial 9")
      self.send_trials()
      self.send_error_timing(87,0.02)
      print("trial 10")
      self.send_trials()
      self.send_error_timing(87,0.02)
      print("trial 11")
      self.send_trials()
      self.send_error_timing(87,0.02)"""
      print("trial 12")
      self.send_trials()
      self.send_error_timing(37,0.6)
      print("trial 13")
      self.send_trials()
      self.send_error_timing(37,0.4)
      print("trial 14")
      self.send_trials()
      self.send_error_timing(37,0.1)
      print("trial 15")
      self.send_trials()
      self.send_error_timing(37,0.05)
      print("trial 16")
      self.send_trials()
      self.send_error_timing(37,0.02)
      print("trial 17")
      self.send_trials()
      self.send_error_timing(37,0.02)
      print("trial 18")
      self.send_trials()
      self.send_error_timing(37,0.02)
      print("trial 19")
      self.send_trials()
      self.send_error_timing(37,0.02)
      print("trial 20")
      self.send_trials()
      self.send_error_timing(37,0.02)
      print("trial 21")
      self.send_trials()
      self.send_error_timing(37,0.02)

   def callback_stop(self,msg):
      if msg.data < 0.7:
         self.count_stop += 1
      else:
         self.count_stop = 0
         self.wait = False
      if self.count_stop > 5:
         #print("waiting...")
         self.wait = True

   def send_perception(self,x,v):
      g = Goal()
      g.x = x
      g.value = v
      self.pub_perception.publish(g)

   def send_error(self,x,v):
      g = Goal()
      g.x = x
      g.value = v
      self.pub_error.publish(g)

   def send_value1(self,x,v):
      g = Goal()
      g.x = x
      g.value = v
      self.pub_value1.publish(g)

   def send_value2(self,x,v):
      g = Goal()
      g.x = x
      g.value = v
      self.pub_value2.publish(g)
   
   def send_new(self,x,v):
      g = Goal()
      g.x = x
      g.value = v
      self.pub_new.publish(g)

   def send_boost(self,v):
      f = Float64()
      f.data = v
      self.pub_boost.publish(f)

   def send_time(self,v):
      f = Float64()
      f.data = v
      self.pub_time.publish(f)

   def send_eoa(self,v):
      f = Float64()
      f.data = v
      self.pub_eoa.publish(f)

   def check_node(self):
      while(self.wait):
         pass

   def send_trials(self):
      self.check_node()
      self.send_eoa(1.0)
      rospy.sleep(1.0)
      self.send_eoa(0.0)
      rospy.sleep(1.0)
      self.check_node()
      self.send_eoa(1.0)
      rospy.sleep(1.0)
      self.send_eoa(0.0)
      rospy.sleep(1.0)
      self.check_node()
      self.send_eoa(1.0)
      rospy.sleep(1.0)
      self.send_eoa(0.0)
      self.check_node()
      rospy.sleep(1.0)


   def send_error_timing(self,p,val):
      self.send_error(p,val)
      self.send_time(1.0)
      rospy.sleep(1.0)
      self.send_error(p,0.0)
      self.send_time(0.0)
      self.send_eoa(1.0)
      rospy.sleep(1.0)
      self.send_eoa(0.0)
      rospy.sleep(1.0)

   def send_habit(self):
      self.send_boost(1.0)
      rospy.sleep(1.0)
      self.send_boost(0.0)
      rospy.sleep(1.0)

   def simulation(self):
      self.send_value1(24,1.0)
      self.send_value2(80,1.0)
      rospy.sleep(1.0)
      self.send_habit()
      self.send_habit()
      self.send_habit()
      self.send_habit()
      #self.send_new(24,1.0)
      #rospy.sleep(1.0)
      #self.send_new(24,0.0)
      #rospy.sleep(1.0)
      self.send_habit()
      self.send_habit()
      self.send_habit()
      self.send_habit()
      rospy.sleep(2.0)
      self.send_value2(80,1.5)
      #self.send_habit()

      
      print("END")







if __name__ == "__main__":
   test = Testing()
   rospy.sleep(0.5)
   test.simulation()

   rospy.spin()