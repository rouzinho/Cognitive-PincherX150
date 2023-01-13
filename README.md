# Cognitive-PincherX150

Development of a cognitive architecture for the trossen robotics robot PX150. The idea is for the robot to continuously and autonomously learn new skills
through Dynamic Neural Curiosity.

work in progress...

The packages are currently implemented and others will follow in due time. A quick description :

## AutoEncoder
Package implementing a convolutional autoencoder that takes as input a 128*128 one channel depth image.
The code is done with Pytorch and use ros, and the ROS structure package will be implemented later.

## DSOM
Python package implementing the DSOM algorithm, which is basically online learning with SOM according to an elasticity parameter.
The code also use ROS but the package structure will be implemented later.

## SOM
Python SOM implementation with generation of dataset for robotics application.

## depth_perception
Proper ROS C++ implementation of an object depth map. The inputs are PointClouds coming from an Azure kinect and filtered by the perception module 
provided by Interbotix. The code use the pointclouds to generte a depth image of an object in front of the camera. The image is then used as input 
for the Convolutional AutoEncoder.
The code uses the PCL Library.

## detector
Proper ROS C++ object rotation/translation detector. The code captures 2 different point clouds from the azure kinect camera and perform  an ICP 
algorithm (Iterative Closest Point). The result provide the rotation between the original and the final capture. The ICP isn't providing correct
results for translation, so a simple translation is computed between initial and final pose.

## Motion
Proper python ROS package to easily control the PincherX150. It can go to custom poses and use the Dynamic Motion Primitives package
from https://wiki.ros.org/dmp. This means the robot can record the end effector trajectory and generate a DMP based on joint angles for later
use (e.g repeat the same motion with a different end pose).

## proprioception pincher
python ROS package used to record End Effector position. The package publishes the EE position ata chosen rate, which is convenient for later use
with Dynamic Motion Primitives (e.g the more points recorded means the more waypoints generated for the trajectory). So lower rate is much better. 

