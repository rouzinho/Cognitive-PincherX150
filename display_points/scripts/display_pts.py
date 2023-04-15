
import geometry_msgs.msg
from som.msg import ListPeaks
import rospy
import matplotlib.pyplot as plt
import numpy as np

class DisplayPts(object):
   def __init__(self):
      super(DisplayPts, self).__init__()
      

   def display(self):
      plt.rcParams["figure.figsize"] = [7.50, 3.50]
      plt.rcParams["figure.autolayout"] = True
      N = 5
      x = np.random.rand(N)
      y = np.random.rand(N)
      plt.plot(x, y, 'r*')
      for xy in zip(x, y):
         plt.annotate('(%.2f, %.2f)' % xy, xy=xy)
      plt.show()

if __name__ == '__main__':
   d = DisplayPts()
   d.display()
   rospy.spin()