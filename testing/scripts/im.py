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
      rospy.init_node('test_im', anonymous=True)
      self.pub_update_lp = rospy.Publisher('/intrinsic/goal_error', Goal, latch=True, queue_size=1)
      self.pub_timer = rospy.Publisher('/intrinsic/updating_lp', Float64, latch=True, queue_size=1)
      self.pub_end = rospy.Publisher('/intrinsic/end_action', Bool, queue_size=1)
      self.x = rospy.get_param("x")
      self.y = rospy.get_param("y")
      self.v = rospy.get_param("v")

   def send_timing(self,value):
      v = Float64()
      v.data = value
      self.pub_timer.publish(v)

   def end_action(self,status):
      v = Bool()
      v.data = status
      self.pub_end.publish(v)
   
   def send_value(self):
      g = Goal()
      g.x = self.x
      g.y = self.y
      g.value = self.v
      self.pub_update_lp.publish(g)
      self.send_timing(1.0)
      rospy.sleep(1.0)
      g.value = 0
      self.pub_update_lp.publish(g)
      self.send_timing(0)
      self.end_action(True)
      rospy.sleep(1)
      self.end_action(False)


if __name__ == "__main__":
   rospy.sleep(0.5)
   test = Testing()
   test.send_value()
   rospy.spin()