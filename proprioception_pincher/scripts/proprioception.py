#!/usr/bin/env python3
from os import name
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import roslib
import rosbag
import numpy as np
from math import pi
from std_msgs.msg import String
from std_msgs.msg import Bool
from moveit_commander.conversions import pose_to_list
from geometry_msgs.msg import Pose
import glob
import os.path
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import JointState
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import interbotix_common_modules.angle_manipulation as ang

class Proprioception(object):
  def __init__(self):
    super(Proprioception, self).__init__()
    #moveit_commander.roscpp_initialize(sys.argv)
    #rospy.init_node('proprioception', anonymous=True)
    self.bot = InterbotixManipulatorXS("px150", "arm", "gripper")
    self.pub = rospy.Publisher('/proprioception/joint_states', JointState, queue_size=10)
    self.pub_ee = rospy.Publisher('/proprioception/ee_pose', Pose, queue_size=1)
    rospy.Subscriber('/px150/joint_states', JointState, self.joint_states)
    self.js = JointState()

  def joint_states(self,msg):
    self.js = msg

  def publishJS(self):
    self.pub.publish(self.js)

  def publishEE(self):
    p = Pose()
    t = self.bot.arm.get_ee_pose()
    p.position.x = round(t[0,3],2)
    p.position.y = round(t[1,3],2)
    p.position.z = round(t[2,3],2)
    self.pub_ee.publish(p)

if __name__ == '__main__':
  try:
    prop = Proprioception()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        prop.publishJS()
        rate.sleep()

  except rospy.ROSInterruptException:
    print("ROS INTERRUPTION")
  except KeyboardInterrupt:
    print("KEYBOARD STOP")