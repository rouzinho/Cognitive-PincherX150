#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Int16
from std_msgs.msg import Bool
from motion.msg import Dmp
from detector.msg import Outcome

class Testing(object):
   def __init__(self):
      rospy.init_node('test_vae', anonymous=True)
      self.pub_dmp = rospy.Publisher("/habituation/dmp", Dmp, queue_size=1, latch=True)
      self.pub_outcome = rospy.Publisher("/outcome_detector/outcome", Outcome, queue_size=1, latch=True)
      rospy.Subscriber("/habituation/ready", Bool, self.callback_ready)
      self.ready = True

   def publish_outcome(self,data):
      tmp = Outcome()
      tmp.x = data[0]
      tmp.y = data[1]
      tmp.angle = data[2]
      tmp.touch = data[3]
      self.pub_outcome.publish(tmp)

   def publish_dmp(self,data):
      tmp = Dmp()
      tmp.v_x = data[0]
      tmp.v_y = data[1]
      tmp.v_pitch = data[2]
      tmp.roll = data[3]
      tmp.grasp = data[4]
      self.pub_dmp.publish(tmp)

   def callback_ready(self,msg):
      self.ready = msg.data

   def get_ready(self):
      return self.ready
   
   def set_ready(self, val):
      self.ready = val



if __name__ == "__main__":
   test = Testing()
   data = []
   outcome1 = [0.1,0.1,40.0,0.0]
   dmp1 = [0.1,0.1,1.0,0.5,0.0]
   outcome2 = [-0.1,-0.1,10.0,0.0]
   dmp2 = [-0.1,-0.1,0.2,0.1,0.0]
   outcome3 = [0.1,0.0,100.0,0.0]
   dmp3 = [0.1,0.0,0.3,-0.5,0.0]
   outcome4 = [0.0,0.0,0.0,1.0]
   dmp4 = [0.15,0.0,0.6,0.0,1.0]
   outcome5 = [-0.1,0.1,10.0,0.0]
   dmp5 = [-0.1,0.1,-0.2,0.0,0.0]
   sim1_out = [0.1,0.1,30.0,0.0]
   sim1_dmp = [0.1,0.11,1.0,0.6,0.0]
   sim2_out = [0.4,0.4,0.1,0.0]
   sim2_dmp = [0.1,0.6,0.1,0.45,0.0]
   sim3_out = [0.2,0.34,0.12,0.0]
   sim3_dmp = [0.8,0.85,0.45,0.9,0.0]
   sim4_out = [0.6,0.7,0.9,1.0]
   sim4_dmp = [0.67,0.77,0.8,0.35,1.0]
   sim5_out = [0.8,0.6,0.1,0.0]
   sim5_dmp = [0.5,0.15,0.9,0.1,0.0]
   data.append([outcome1,dmp1])
   data.append([outcome2,dmp2])
   #data.append([sim1_out,sim1_dmp])
   data.append([outcome4,dmp4])
   data.append([outcome3,dmp3])
   data.append([outcome5,dmp5])
   #data.append([sim1_out,sim1_dmp])
   #data.append([sim2_out,sim2_dmp])
   #data.append([sim3_out,sim3_dmp])
   #data.append([sim4_out,sim4_dmp])
   #data.append([sim5_out,sim5_dmp])

   i = 0
   seconds = 0
   while not rospy.is_shutdown():
      if(test.get_ready() and i < 5):
         test.publish_dmp(data[i][1])
         rospy.sleep(0.5)
         test.publish_outcome(data[i][0])
         test.set_ready(False)
         i += 1

   rospy.spin()