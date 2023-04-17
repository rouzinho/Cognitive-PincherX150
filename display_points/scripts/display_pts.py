#! /usr/bin/python3
import geometry_msgs.msg
from som.msg import ListPeaks
import rospy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation
from som.msg import GripperOrientation

class DisplayPts(object):
   def __init__(self):
      super(DisplayPts, self).__init__()
      self.sub_object = rospy.Subscriber('/display/object', ListPeaks, self.callback_object)
      self.sub_gauss = rospy.Subscriber('/display/gauss', ListPeaks, self.callback_gauss)
      self.sub_fp = rospy.Subscriber('/display/first_pose', GripperOrientation, self.callback_first_pose)
      self.sub_lp = rospy.Subscriber('/display/last_pose', GripperOrientation, self.callback_last_pose)
      self.x_object = np.array([0.3])
      self.y_object = np.array([0.0])
      self.x_gauss = np.array([0.3])
      self.y_gauss = np.array([0.3])
      self.x_fpose = np.array([0.0])
      self.y_fpose = np.array([-0.4])
      self.x_lpose = np.array([0.0])
      self.y_lpose = np.array([-0.4])
      self.fig, self.ax = plt.subplots()
      #self.fig = plt.figure()
      #self.ax = self.fig.add_subplot(111)
      self.ani = matplotlib.animation.FuncAnimation(self.fig, self.update, interval=5, init_func=self.setup_plot, blit=True)
      #plt.xlim(0,0.5)
      #plt.ylim(-0.5,0.5)

   def callback_object(self,msg):
      x = []
      y = []
      for sample in msg.list_peaks:
         x.append(sample.x)
         y.append(sample.y)
      self.x_object = np.array(x)
      self.y_object = np.array(y)

   def callback_gauss(self,msg):
      x = []
      y = []
      for sample in msg.list_peaks:
         x.append(sample.x)
         y.append(sample.y)
      self.x_gauss = np.array(x)
      self.y_gauss = np.array(y)

   def callback_first_pose(self,msg):
      self.x_fpose = np.array([msg.x])
      self.y_fpose = np.array([msg.y])

   def callback_last_pose(self,msg):
      self.x_lpose = np.array([msg.x])
      self.y_lpose = np.array([msg.y])

   def setup_plot(self):
      self.scat_o = self.ax.scatter(self.x_object, self.y_object, c='red')
      self.scat_g = self.ax.scatter(self.x_gauss, self.y_gauss, c='blue')
      self.scat_f = self.ax.scatter(self.x_fpose, self.y_fpose, c='forestgreen')
      self.scat_l = self.ax.scatter(self.x_lpose, self.y_lpose, c='lime')
      self.ax.axis([0, 0.6, -0.4, 0.4])
      # For FuncAnimation's sake, we need to return the artist we'll be using
      # Note that it expects a sequence of artists, thus the trailing comma.
      return self.scat_o, self.scat_g, self.scat_f, self.scat_l

   def update(self, i):
      # Set x and y data...
      X = np.c_[self.x_object, self.y_object]
      X_b = np.c_[self.x_gauss, self.y_gauss]
      X_f = np.c_[self.x_fpose, self.y_fpose]
      X_l = np.c_[self.x_lpose, self.y_lpose]
      self.scat_o.set_offsets(X)
      self.scat_g.set_offsets(X_b)
      self.scat_f.set_offsets(X_f)
      self.scat_l.set_offsets(X_l)
      # Set sizes...
      #self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
      # Set colors..
      #self.scat.set_array(data[:, 3])
      # We need to return the updated artist for FuncAnimation to draw..
      # Note that it expects a sequence of artists, thus the trailing comma.
      return self.scat_o, self.scat_g, self.scat_f, self.scat_l

if __name__ == '__main__':
   rospy.init_node("display")
   d = DisplayPts()
   plt.show()
   rospy.spin()

