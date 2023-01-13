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

def all_close(goal, actual, tolerance):
  """
  Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
  @param: goal       A list of floats, a Pose or a PoseStamped
  @param: actual     A list of floats, a Pose or a PoseStamped
  @param: tolerance  A float
  @returns: bool
  """
  all_equal = True
  if type(goal) is list:
    for index in range(len(goal)):
      if abs(actual[index] - goal[index]) > tolerance:
        return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
    return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:
    return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

  return True

class Proprioception(object):
  def __init__(self):
    super(Proprioception, self).__init__()
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('proprioception', anonymous=True)
    self.pub = rospy.Publisher('/motion_pincher/proprioception', Pose, queue_size=10)

    self.robot_model = rospy.get_param("~robot_model")
    self.robot_name = rospy.get_namespace().strip("/")
    self.ee_link_offset = rospy.get_param("~ee_link_offset")
    self.joint_goal = rospy.get_param("~joint_goal")
    pose_goal_raw = rospy.get_param("~pose_goal")
    self.robot = moveit_commander.RobotCommander()
    self.scene = moveit_commander.PlanningSceneInterface()
    group_name = "interbotix_arm"
    self.group = moveit_commander.MoveGroupCommander(group_name)
    self.display_trajectory_publisher = rospy.Publisher("move_group/display_planned_path",
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)

    self.planning_frame = self.group.get_planning_frame()
    self.eef_link = self.group.get_end_effector_link()
    self.group_names = self.robot.get_group_names()
    self.current_pose = geometry_msgs.msg.Pose()

  def publishEEPose(self):
      wpose = self.group.get_current_pose().pose
      self.pub.publish(wpose)

if __name__ == '__main__':
  try:
    prop = Proprioception()
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        prop.publishEEPose()
        rate.sleep()

  except rospy.ROSInterruptException:
    print("ROS INTERRUPTION")
  except KeyboardInterrupt:
    print("KEYBOARD STOP")