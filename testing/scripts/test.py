#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Int16
from std_msgs.msg import Bool
from motion.msg import Dmp
from detector.msg import Outcome
from detector.msg import State
from motion.msg import Action

class Testing(object):
   def __init__(self):
      rospy.init_node('test_vae', anonymous=True)
      self.pub_dmp = rospy.Publisher("/motion_pincher/dmp", Dmp, queue_size=1, latch=True)
      self.pub_outcome = rospy.Publisher("/outcome_detector/outcome", Outcome, queue_size=1, latch=True)
      self.pub_id = rospy.Publisher("/cog_learning/id_object", Int16, queue_size=1, latch=True)
      self.pub_action_sample = rospy.Publisher('/motion_pincher/action_sample', Action, latch=True, queue_size=1)
      self.pub_state = rospy.Publisher('/outcome_detector/state', State, latch=True, queue_size=1)
      self.pub_explore = rospy.Publisher('/cog_learning/exploration', Bool, latch=True, queue_size=1)
      self.pub_exploit = rospy.Publisher('/cog_learning/exploitation', Bool, latch=True, queue_size=1)
      rospy.Subscriber("/cog_learning/ready", Bool, self.callback_ready)
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

   def publish_action(self,data):
      tmp = Action()
      tmp.lpos_x = data[0]
      tmp.lpos_y = data[1]
      tmp.lpos_pitch = data[2]
      self.pub_action_sample.publish(tmp)

   def publish_state(self,data):
      tmp = State()
      tmp.state_x = data[0]
      tmp.state_y = data[1]
      tmp.state_angle = data[2]
      self.pub_state.publish(tmp)

   def callback_ready(self,msg):
      self.ready = msg.data

   def get_ready(self):
      return self.ready
   
   def set_ready(self, val):
      self.ready = val

   def send_id(self, n):
      tmp = Int16()
      tmp.data = n
      self.pub_id.publish(tmp)

   def pub_exploration(self):
      tmp = Bool()
      tmp.data = True
      self.pub_explore.publish(tmp)

   def pub_exploitation(self):
      tmp = Bool()
      tmp.data = True
      self.pub_exploit.publish(tmp)



if __name__ == "__main__":
   test = Testing()
   test.send_id(0)
   data = []
   actions = []
   states = []
   outcome1 = [0.1,0.1,40.0,0.0]
   dmp1 = [0.1,0.1,1.0,0.5,0.0]
   action1 = [0.18,0.1,1.0]
   state1 = [0.2,0.0,20.0]
   outcome2 = [-0.1,-0.1,10.0,0.0]
   dmp2 = [-0.1,-0.1,0.2,0.1,0.0]
   action2 = [0.2,0.2,1.2]
   state2 = [0.3,0.1,60.0]
   outcome3 = [0.1,0.0,100.0,0.0]
   dmp3 = [0.1,0.0,0.3,-0.5,0.0]
   action3 = [0.19,0.1,0.5]
   state3 = [0.3,-0.1,100.0]
   outcome4 = [0.0,0.0,0.0,1.0]
   dmp4 = [0.15,0.0,0.6,0.0,1.0]
   action4 = [0.5,-0.1,1.2]
   state4 = [0.3,-0.2,80.0]
   outcome5 = [-0.1,0.1,10.0,0.0]
   dmp5 = [-0.1,0.1,-0.2,0.0,0.0]
   action5 = [0.35,-0.1,1.0]
   state5 = [0.2,0.3,110.0]
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
   actions.append(action1)
   actions.append(action2)
   actions.append(action3)
   actions.append(action4)
   actions.append(action5)
   states.append(state1)
   states.append(state2)
   states.append(state3)
   states.append(state4)
   states.append(state5)

   i = 0
   seconds = 0
   explore = True
   while not rospy.is_shutdown():
      if explore:
         test.pub_exploration()
         if(test.get_ready() and i < 5):
            #if i == 1:
            #   test.send_id(1)
            #   rospy.sleep(0.5)
            test.publish_state(states[i])
            test.publish_dmp(data[i][1])
            test.publish_action(actions[i])
            rospy.sleep(0.5)
            test.publish_outcome(data[i][0])
            test.set_ready(False)
            i += 1
      else:
         test.pub_exploitation()
         test.send_id(0)
         rospy.sleep(0.5)
         if(test.get_ready() and i < 5):
            test.publish_state(states[i])
            #test.publish_dmp(data[i][1])
            test.publish_action(actions[i])
            rospy.sleep(0.5)
            test.publish_outcome(data[i][0])
            test.set_ready(False)
            i += 1

   rospy.spin()