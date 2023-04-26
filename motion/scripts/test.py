#!/usr/bin/env python3
import roslib 
import rospy 
import numpy as np
import rosbag
from motion.msg import PoseRPY
from motion.msg import GripperOrientation
from motion.msg import VectorAction
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
import os
from os import path
from pathlib import Path

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

def name_dmp(action):
    name = "/home/altair/interbotix_ws/src/motion/dmp/"
    nx = ""
    ny = ""
    gr = ""
    nx = "x"+str(action.x)
    ny = "y"+str(action.y)
    gr = "g"+str(action.grasp)
    name = name + nx + ny + gr + "r.bag"

    return name
    
def find_dmp(action):
    name_dir = "/home/altair/interbotix_ws/src/motion/dmp/"
    found = False
    right_file = ""
    for file in os.listdir(name_dir):
        p_x = file.find('x')
        p_y = file.find('y')
        p_g = file.find('g')
        p_r = file.find('r')
        x = file[p_x+1:p_y]
        y = file[p_y+1:p_g]
        g = file[p_g+1:p_r]
        x = float(x)
        y = float(y)
        g = float(g)
        if action.x - x < 0.05 and action.y - y < 0.05 and action.grasp - g < 0.05:
            found = True
            right_file = file
    
    return found, right_file


def update_offline_dataset(status):
    name_dataset_states = "/home/altair/interbotix_ws/src/depth_perception/states/"
    a = VectorAction()
    p = GripperOrientation()
    a.x = 1.0
    a.y = 0.8
    a.grasp = 1.0
    p.pitch = 1.0
    p.roll = 0.0
    paths = sorted(Path(name_dataset_states).iterdir(), key=os.path.getmtime)
    name_state = str(paths[len(paths)-1])
    name_dataset = "/home/altair/interbotix_ws/src/motion/dataset/datas.txt"
    exist = path.exists(name_dataset)
    opening = ""
    if(exist == False):
      opening = "w"
    else:
      opening = "a"
    data = str(a.x) + " " + str(a.y) + " " + str(a.grasp) + " " + str(p.pitch) + " " + str(p.roll) + " " + name_state + " " + str(status) + "\n"
    with open(name_dataset, opening) as f:
        f.write(data)
    f.close()


if __name__ == '__main__':
    rospy.init_node('test')
    # first = True
    # a = VectorAction()
    # a.x = 0.3456
    # a.y = 0.3746
    # a.z = 0.0
    # a.grasp = 0.89

    # if first == True:
    #     update_offline_dataset(True)
    #     #send_position()
    #     #send_orientation()
    #     #send_action()
    #     #name_dmp()
    #     first = False
    # #rospy.sleep(100000)
    t = 180
    r = t%90
    print(r)
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