#! /usr/bin/python3
import rospy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation
from display_points.msg import Activations

class DisplayPts(object):
   def __init__(self):
      super(DisplayPts, self).__init__()
      self.sub_gauss1 = rospy.Subscriber('/sim_dnf/gauss1', Activations, self.callback_gauss1)
      self.sub_gauss2 = rospy.Subscriber('/sim_dnf/gauss2', Activations, self.callback_gauss2)
      self.sub_dnf = rospy.Subscriber('/sim_dnf/2d', Activations, self.callback_dnf)
      self.shape1 = rospy.get_param("size_gauss1")
      self.shape2 = rospy.get_param("size_gauss2")
      self.x_gauss1 = np.arange(self.shape1)
      self.y_gauss1 = np.zeros(self.shape1)
      self.x_gauss2 = np.arange(self.shape2)
      self.y_gauss2 = np.zeros(self.shape2)
      self.shape_grid = (self.shape1,self.shape2)
      self.grid = np.zeros(self.shape_grid)
      self.fig, (self.ax0, self.ax1, self.ax2) = plt.subplots(3,1,gridspec_kw={'height_ratios': [20, 2,2]})
      #self.im = self.ax0.imshow(self.grid,interpolation='nearest',aspect=1.0)
      self.im = self.ax0.matshow(self.grid)
      self.colors = ["red"]
      self.colors_act = ["blue"]
      #self.fig = plt.figure()
      #self.ax = self.fig.add_subplot(111)
      #self.ani = matplotlib.animation.FuncAnimation(self.fig, self.update_pos, interval=5, init_func=self.setup_plot_pos, blit=True)
      self.ani = matplotlib.animation.FuncAnimation(self.fig, self.update_gauss, interval=5, init_func=self.setup_plot_gauss, blit=True)
      #plt.xlim(0,0.5)
      #plt.ylim(-0.5,0.5)

   def callback_gauss1(self,msg):
      self.x_gauss1 = np.arange(self.shape1)
      self.y_gauss1 = np.array(msg.activation)

   def callback_gauss2(self,msg):
      self.x_gauss2 = np.arange(self.shape2)
      self.y_gauss2 = np.array(msg.activation)

   def callback_dnf(self,msg):
      tmp = np.array(msg.activation)
      tmp = tmp.reshape((self.shape1,-1))
      self.grid = tmp
      #self.im.set_data(self.grid)
   
   def setup_plot_gauss(self):
      #self.scat_g1 = self.ax.scatter(self.latent_x, self.latent_y,s=100, c=self.colors,cmap="jet", edgecolor="k")
      #self.scat_a = self.ax.scatter(self.latent_x_act, self.latent_y_act,s=100, c=self.colors_act,cmap="jet", edgecolor="k")
      #self.scat_l = self.ax.scatter(self.x_test, self.y_test, c='lime')
      self.im.set_data(self.grid)
      self.g1, = self.ax1.plot(self.x_gauss1,self.y_gauss1, color='red')
      self.g2, = self.ax2.plot(self.x_gauss2,self.y_gauss2, color='blue')
      self.ax0.axis([0, self.shape1-1, 0.0, self.shape2-1])
      self.ax1.axis([0, self.shape1-1, 0.0, 1.5])
      self.ax2.axis([0, self.shape2-1, 0.0, 1.5])
      # For FuncAnimation's sake, we need to return the artist we'll be using
      # Note that it expects a sequence of artists, thus the trailing comma.
      return self.im, self.g1, self.g2,

   def update_gauss(self, i):
      self.im.set_data(self.grid)
      self.g1.set_ydata(self.y_gauss1)
      self.g2.set_ydata(self.y_gauss2)

      return self.im, self.g1, self.g2,

if __name__ == '__main__':
   rospy.init_node("display")
   d = DisplayPts()
   plt.show()
   rospy.spin()

