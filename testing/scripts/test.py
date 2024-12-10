#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Int16
from std_msgs.msg import Bool
from motion.msg import Dmp
from detector.msg import Outcome
from detector.msg import State
from motion.msg import Action
from som.msg import GripperOrientation
from cog_learning.msg import LatentGoalDnf

class Testing(object):
   def __init__(self):
      rospy.init_node('test_vae', anonymous=True)
      self.pub_dmp = rospy.Publisher("/motion_pincher/dmp", Dmp, queue_size=1, latch=True)
      self.pub_retrieve_dmp = rospy.Publisher("/motion_pincher/retrieve_dmp", Dmp, queue_size=1, latch=True)
      self.pub_outcome = rospy.Publisher("/outcome_detector/outcome", Outcome, queue_size=1, latch=True)
      self.pub_id = rospy.Publisher("/cog_learning/id_object", Int16, queue_size=1, latch=True)
      self.pub_action_sample = rospy.Publisher('/motion_pincher/action_sample', Action, latch=True, queue_size=1)
      self.pub_state = rospy.Publisher('/outcome_detector/state', State, latch=True, queue_size=1)
      self.pub_explore = rospy.Publisher('/cog_learning/rnd_exploration', Float64, latch=True, queue_size=1)
      self.pub_exploit = rospy.Publisher('/cog_learning/exploitation', Float64, latch=True, queue_size=1)
      self.pub_new_state = rospy.Publisher('/depth_perception/new_state', Bool, latch=True, queue_size=1)
      self.pub_action_dnf = rospy.Publisher('/motion_pincher/dmp_dnf', LatentGoalDnf, latch=True, queue_size=1)
      self.pub_lpos = rospy.Publisher('/motion_pincher/gripper_orientation/first_pose', GripperOrientation, latch=True, queue_size=1)
      rospy.Subscriber("/test/ready", Bool, self.callback_ready)
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
      tmp.fpos_x = data[5]
      tmp.fpos_y = data[6]
      self.pub_dmp.publish(tmp)

   def publish_dnf_action(self,data):
      l = LatentGoalDnf()
      l.latent_x = data[0]
      l.latent_y = data[1]
      self.pub_action_dnf.publish(l)

   def publish_action(self,data):
      tmp = Action()
      tmp.fpos_x = data[0]
      tmp.fpos_y = data[1]
      tmp.fpos_pitch = data[2]
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
   
   def send_new_state(self):
      b = Bool()
      b.data = True
      self.pub_new_state.publish(b)
   
   def set_ready(self, val):
      self.ready = val

   def send_id(self, n):
      tmp = Int16()
      tmp.data = n
      self.pub_id.publish(tmp)

   def pub_exploration(self):
      tmp = Float64()
      tmp.data = 1.0
      self.pub_explore.publish(tmp)

   def pub_exploitation(self):
      tmp = Float64()
      tmp.data = 1.0
      self.pub_exploit.publish(tmp)

   def retrieve_dmp(self):
      tmp = Dmp()
      tmp.v_x = 0.1
      tmp.v_y = 0
      tmp.v_pitch = 0
      tmp.roll = -0.3
      tmp.grasp = 1.0
      tmp.fpos_x = 0
      tmp.fpos_y = 0
      self.pub_retrieve_dmp.publish(tmp)

   def last_pos(self,data):
      tmp = GripperOrientation()
      tmp.x = data[0]
      tmp.y = data[1]
      tmp.z = data[2]
      tmp.pitch = data[3]
      self.pub_lpos.publish(tmp)

if __name__ == "__main__":
   test = Testing()
   rospy.sleep(0.5)
   test.send_id(0) 
   #test.pub_exploitation()
   """rospy.sleep(0.5)
   test.retrieve_dmp()
   rospy.sleep(0.5)
   #d = [0.2,0.0,0.03,1.2]
   #test.last_pos(d)
   d = [0.25,-0.2,0.06,0.8]
   rospy.sleep(0.5)
   test.last_pos(d)


   """
   data = []
   actions = []
   states = []
   dnf = []
   #up - modified fpos
   outcome1 = [0.08,0.0,-14.0,0.0]
   dmp1 = [0.08,0.0,1.0,1.5,0.0,0.26,0.0]
   action1 = [0.26,0.0,1.0]
   state1 = [0.29,0.0,94.9]
   outcomeup1 = [0.06,0.02,-14.0,0.0]
   dmpup1 = [0.06,0.02,1.0,1.5,0.0,0.26,0.0]
   actionup1 = [0.26,0.0,1.0]
   stateup1 = [0.29,0.0,94.9]
   outcomeup2 = [0.06,-0.02,-14.0,0.0]
   dmpup2 = [0.06,-0.02,1.0,1.5,0.0,0.26,0.0]
   actionup2 = [0.26,0.0,1.0]
   stateup2 = [0.29,0.0,94.9]
   #down
   outcome2 = [-0.08,0.00,13.0,0.0]
   dmp2 = [-0.08,0.0,1.1,1.5,0.0,0.34,0.0]
   action2 = [0.26,0.0,1.0]
   state2 = [0.29,0.0,99.4]
   #left
   outcome3 = [0.0,0.06,4.0,0.0]
   dmp3 = [0.0,0.08,1.0,0.0,0.0,0.29,-0.04]
   action3 = [0.29,0.04,1.0]
   state3 = [0.3,0.0,95.5]
   #right
   outcome4 = [0.01,-0.08,-20.0,0.0]
   dmp4 = [-0.015,-0.09,1.0,0.0,0.0,0.29,0.05]
   action4 = [0.29,0.06,1.0]
   state4 = [0.29,0.0,95.5]
   #grasp
   outcome5 = [0.0,0.0,0.0,1.0]
   dmp5 = [0.08,0.0,0.15,0.0,1.0,0.34,0.0]
   action5 = [0.26,0.0,0.5]
   state5 = [0.29,0.0,95.7]
   
   outcome6 = [0.0,0.0,0.0,1.0]
   dmp6 = [0,-0.1,0.0,0.0,0.0,1.0,0.19,0.23]
   action6 = [0.2,0.2,1.0]
   state6 = [0.2,0.2,20.0]

   outcome3_1 = [0.0,0.0,25.0,0.0]
   dmp3_1 = [-0.1,0.0,0.3,-0.5,0.0,0.1,-0.1]
   action3_1 = [0.2,0.2,1.0]
   state3_1 = [0.2,0.2,20.0]

   outcome3_2 = [0.0,0.0,45.0,0.0]
   dmp3_2 = [0.0,0.1,0.1,-0.5,0.0,0.1,-0.1]
   action3_2 = [0.2,0.2,1.0]
   state3_2 = [0.2,0.2,20.0]

   outcome3_3 = [0.0,0.0,45.0,0.0]
   dmp3_3 = [0.0,-0.1,0.1,-0.5,0.0,0.1,-0.1]
   action3_3 = [0.2,0.2,1.0]
   state3_3 = [0.2,0.2,20.0]


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
   #data.append([outcome4,dmp4])
   #data.append([outcome5,dmp5])
   #comment here simmilar
   data.append([outcomeup1,dmpup1])
   data.append([outcomeup2,dmpup2])

   data.append([outcome2,dmp2])
   #data.append([sim1_out,sim1_dmp])
   data.append([outcome3,dmp3])
   
   
   data.append([outcome6,dmp6])
   data.append([outcome3_1,dmp3_1])
   #data.append([outcome3_2,dmp3_2])
   #data.append([outcome3_3,dmp3_3])
   #data.append([sim1_out,sim1_dmp])
   #data.append([sim2_out,sim2_dmp])
   #data.append([sim3_out,sim3_dmp])
   #data.append([sim4_out,sim4_dmp])
   #data.append([sim5_out,sim5_dmp])
   actions.append(action1)
   actions.append(action4)
   actions.append(action5)
   
   actions.append(action2)
   actions.append(action3)
   
   
   actions.append(action6)
   actions.append(action3_1)
   actions.append(action3_2)
   actions.append(action3_3)
   states.append(state1)
   states.append(state4)
   states.append(state5)
   states.append(state2)
   states.append(state3)
   
   
   states.append(state6)
   states.append(state3_1)
   states.append(state3_2)
   states.append(state3_3)
   dnf.append([21,10])
   dnf.append([14,42])
   dnf.append([41,41])
   dnf.append([41,41])
   dnf.append([21,10])
   dnf.append([41,41])
   dnf.append([76,44])
   #i = rospy.get_param("i")
   i = 0
   seconds = 0
   explore = True
   first = True
   #skills = ["up","down","left","right","grasp"]
   skills = ["up","right","grasp"]
   #[67, 79, 0.9], [38, 72, 0.9], [42, 85, 0.9], [67, 51, 0.9], [56, 73, 0.9] order of skills
   while not rospy.is_shutdown():
      test.pub_exploration()
      #test.pub_exploitation()
      if explore:
         if(test.get_ready() and i < 3):
            print(i)
            #if i == 2:
            #   print("change object")
            #   rospy.sleep(1.5)
            #   test.send_id(1)
            #   rospy.sleep(5.5)
            test.send_new_state()
            test.publish_state(states[i])
            test.publish_dmp(data[i][1])
            test.publish_action(actions[i])
            rospy.sleep(0.5)
            test.publish_outcome(data[i][0])
            test.set_ready(False)
            first = False
            i += 1
            rospy.sleep(4.5)
      else:
         #test.pub_exploitation()
         test.send_id(0)
         rospy.sleep(0.5)
         if(test.get_ready() and i < 5):
            print("sending...",skills[i])
            rospy.sleep(1.5)
            test.send_new_state()
            test.publish_state(states[i])
            test.publish_dmp(data[i][1])
            test.publish_dnf_action(dnf[i])
            test.publish_action(actions[i])
            rospy.sleep(0.5)
            test.publish_outcome(data[i][0])
            test.set_ready(False)
            i += 1
      #if i == 4:
      #   i = 0

   rospy.spin()