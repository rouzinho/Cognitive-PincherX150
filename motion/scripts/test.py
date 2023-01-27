#!/usr/bin/env python3
import roslib 
import rospy 
import numpy as np
from motion.msg import PoseRPY
from motion.msg import GripperOrientation
from motion.msg import VectorAction
from geometry_msgs.msg import Pose

pub_p = rospy.Publisher("/motion_pincher/start_position", Pose, queue_size=1, latch=True)
pub_o = rospy.Publisher("/motion_pincher/gripper_orientation", GripperOrientation, queue_size=1, latch=True)
pub_a = rospy.Publisher("/motion_pincher/vector_action", VectorAction, queue_size=1, latch=True)

def send_position():
    p = Pose()
    p.position.x = 0.2
    p.position.y = 0.0
    p.position.z = 0.05
    pub_p.publish(p)

def send_orientation():
    o = GripperOrientation()
    o.roll = 0.0
    o.pitch = 1.0
    o.grasp = 0.0
    pub_o.publish(o)

def send_action():
    a = VectorAction()
    a.x = 0.1
    a.y = 0.1
    a.z = 0.0
    pub_a.publish(a)



if __name__ == '__main__':
    rospy.init_node('test')
    first = True
    if first == True:
        send_position()
        send_orientation()
        send_action()
        first = False
    #rospy.sleep(100000)
    rospy.spin()



#if first:
      #motion_planning.test_interface()
      #motion_planning.makeDMP()
      #motion_planning.init_position()
      #motion_planning.play_motion_dmp()
     # motion_planning.sleep_pose()
      #motion_planning.makeDMP()
      #motion_planning.test_interface()
      #motion_planning.init_position()
      #motion_planning.reproduce_group()
      #motion_planning.init_position()
      #print("slept")
      #first = False