#! /usr/bin/python3
import geometry_msgs.msg
from som.msg import ListPeaks
import rospy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation
from som.msg import GripperOrientation
from habituation.msg import LatentPos

class DisplayPts(object):
   def __init__(self):
      super(DisplayPts, self).__init__()
      self.sub_object = rospy.Subscriber('/display/object', ListPeaks, self.callback_object)
      self.sub_gauss = rospy.Subscriber('/display/gauss', ListPeaks, self.callback_gauss)
      self.sub_fp = rospy.Subscriber('/display/first_pose', GripperOrientation, self.callback_first_pose)
      self.sub_lp = rospy.Subscriber('/display/last_pose', GripperOrientation, self.callback_last_pose)
      self.sub_latent = rospy.Subscriber('/display/latent_space', LatentPos, self.callback_latent)
      self.x_object = np.array([0.3])
      self.y_object = np.array([0.0])
      self.x_gauss = np.array([0.3])
      self.y_gauss = np.array([0.3])
      self.x_fpose = np.array([0.0])
      self.y_fpose = np.array([-0.4])
      self.x_lpose = np.array([0.0])
      self.y_lpose = np.array([-0.4])
      self.latent_x = np.array([0.0])
      self.latent_y = np.array([0.0])
      self.fig, self.ax = plt.subplots()
      self.colors = ["red"]
      #self.fig = plt.figure()
      #self.ax = self.fig.add_subplot(111)
      #self.ani = matplotlib.animation.FuncAnimation(self.fig, self.update_pos, interval=5, init_func=self.setup_plot_pos, blit=True)
      self.ani = matplotlib.animation.FuncAnimation(self.fig, self.update_latent, interval=5, init_func=self.setup_plot_latent, blit=True)
      #plt.xlim(0,0.5)
      #plt.ylim(-0.5,0.5)

   def init_datas(self):
      x = []
      y = []
      x.append(0.2)
      x.append(-0.1)
      y.append(-1.0)
      y.append(-2.0)
      self.latent_x = np.array(x)
      self.latent_y = np.array(y)

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

   def callback_latent(self, msg):
      self.latent_x = np.array(msg.x)
      self.latent_y = np.array(msg.y)
      self.colors = msg.colors
      print(self.colors)

   def setup_plot_pos(self):
      self.scat_o = self.ax.scatter(self.y_object, self.x_object, c='red')
      self.scat_g = self.ax.scatter(self.y_gauss, self.x_gauss, c='blue')
      self.scat_f = self.ax.scatter(self.y_fpose, self.x_fpose, c='forestgreen')
      self.scat_l = self.ax.scatter(self.y_lpose, self.x_lpose, c='lime')
      self.ax.axis([-0.4, 0.4, 0.0, 0.8])
      # For FuncAnimation's sake, we need to return the artist we'll be using
      # Note that it expects a sequence of artists, thus the trailing comma.
      return self.scat_o, self.scat_g, self.scat_f, self.scat_l
   
   def setup_plot_latent(self):
      self.scat_o = self.ax.scatter(self.latent_y, self.latent_x,s=100, c=self.colors,cmap="jet", edgecolor="k")
      self.ax.axis([-2, 2, -2.0, 2.0])
      # For FuncAnimation's sake, we need to return the artist we'll be using
      # Note that it expects a sequence of artists, thus the trailing comma.
      return self.scat_o,

   def update_pos(self, i):
      # Set x and y data...
      X = np.c_[self.y_object, self.x_object]
      X_b = np.c_[self.y_gauss, self.x_gauss]
      X_f = np.c_[self.y_fpose, self.x_fpose]
      X_l = np.c_[self.y_lpose, self.x_lpose]
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
   
   def update_latent(self, i):
      # Set x and y data...
      X = np.c_[self.latent_y, self.latent_x]
      self.scat_o.set_offsets(X)
      #self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
      # Set colors..
      #self.scat_o.set_array(self.colors)

      return self.scat_o,

if __name__ == '__main__':
   rospy.init_node("display")
   d = DisplayPts()
   plt.show()
   rospy.spin()

