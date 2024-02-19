#include <ros/ros.h>
// PCL specific includes

#include <iostream>
#include <string>
#include <std_msgs/Header.h>
#include <ros/header.h>
#include <std_msgs/Bool.h>
#include <stdio.h>
#include "cluster_message/ClusterMessage.h"
#include "cluster_message/OutcomeAE.h"
#include "cluster_message/SampleExplore.h"
#include "cluster_message/SampleExploit.h"
#include "cluster_message/State.h"
#include "motion/Dmp.h"
#include "motion/DmpOutcome.h"
#include "detector/Outcome.h"
#include "detector/State.h"
#include "motion/Action.h"
#include <geometry_msgs/Point.h>
#include "motion/TwoPoses.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_eigen/tf2_eigen.h>

class ClusterMessage
{
  private:
   ros::NodeHandle nh_;
   ros::Subscriber sub_dmp;
   ros::Subscriber sub_outcome;
   ros::Subscriber sub_explore;
   ros::Subscriber sub_exploit;
   ros::Subscriber sub_state;
   ros::Subscriber sub_sample;
   ros::Subscriber sub_ready_habituation;
   ros::Subscriber sub_ready_nnga;
   ros::Subscriber sub_ready_depth;
   ros::Subscriber sub_ready_outcome;
   ros::Publisher pub_datas_explore;
   ros::Publisher pub_datas_exploit;
   ros::Publisher pub_ready_sensor;
   ros::Publisher pub_ready;
   detector::Outcome outcome;
   detector::State state;
   motion::Action sample;
   motion::Dmp dmp;
   bool dmp_b;
   bool out;
   bool explore;
   bool exploit;
   bool state_b;
   bool sample_b;
   bool ready_h;
   bool ready_nn;
   bool ready_depth;
   bool ready_outcome;

  public:
   ClusterMessage()
   {
      sub_dmp = nh_.subscribe("/motion_pincher/dmp", 10, &ClusterMessage::CallbackDMP,this);
      sub_outcome = nh_.subscribe("/outcome_detector/outcome", 10, &ClusterMessage::CallbackOutcome,this);
      sub_explore = nh_.subscribe("/cog_learning/exploration", 10, &ClusterMessage::CallbackExplore,this);
      sub_exploit = nh_.subscribe("/cog_learning/exploitation", 10, &ClusterMessage::CallbackExploit,this);
      sub_state = nh_.subscribe("/outcome_detector/state", 10, &ClusterMessage::CallbackState,this);
      sub_sample = nh_.subscribe("/motion_pincher/action_sample", 10, &ClusterMessage::CallbackSample,this);
      sub_ready_habituation = nh_.subscribe("/habituation/ready", 10, &ClusterMessage::CallbackReadyHabit,this);
      sub_ready_nnga = nh_.subscribe("/cog_learning/ready", 10, &ClusterMessage::CallbackReadyNN,this);
      sub_ready_depth = nh_.subscribe("/depth_perception/ready", 10, &ClusterMessage::CallbackReadyDepth,this);
      sub_ready_outcome = nh_.subscribe("/outcome_detector/ready", 10, &ClusterMessage::CallbackReadyOutcome,this);
      pub_datas_explore = nh_.advertise<cluster_message::SampleExplore>("/cluster_msg/sample_explore",1);
      pub_datas_exploit = nh_.advertise<cluster_message::SampleExploit>("/cluster_msg/sample_exploit",1);
      //pub_datas_exploit = nh_.advertise<cluster_message::SampleExploit>("/cluster_msg/sample_exploit",1);
      pub_ready = nh_.advertise<std_msgs::Bool>("/motion_pincher/ready",1);
      pub_ready_sensor = nh_.advertise<std_msgs::Bool>("/cluster_msg/sensor_ready",1);
      explore = false;
      exploit = false;
   }
   ~ClusterMessage()
   {
   }

   void CallbackDMP(const motion::Dmp::ConstPtr& msg)
   {
      tf2::Quaternion q_orig(0,0,0,1);
      tf2::Quaternion q_rot;
      tf2::Quaternion q_vector;
      geometry_msgs::Point new_vec;
      geometry_msgs::Point vec_ori;
      geometry_msgs::Point res;
      geometry_msgs::PoseStamped first_pose;
      first_pose.pose.position.x = msg->fpos_x;
      first_pose.pose.position.y = msg->fpos_y;
      vec_ori.x = msg->fpos_x;
      vec_ori.y = msg->fpos_y;
      vec_ori.x = msg->fpos_x + vec_ori.x;
      vec_ori.y = msg->fpos_y + vec_ori.y;
      float dot_prod = (vec_ori.x*0.1) + (vec_ori.y*0);
      float det = (vec_ori.x*0) + (vec_ori.y*0.1);
      float ang = atan2(det,dot_prod);
      q_rot.setRPY(0,0,ang);
      q_vector = q_rot*q_orig;
      q_vector.normalize();
      res = findVectorTransform(first_pose,msg->v_x,msg->v_y,q_vector);
      dmp.v_x = res.x;
      dmp.v_y = res.y;
      dmp.v_pitch = msg->v_pitch;
      dmp.grasp = msg->grasp;
      dmp.roll = msg->roll;  
      dmp_b = true;
      //std::cout<<"cluster : got DMP\n";
   }

   void CallbackOutcome(const detector::Outcome::ConstPtr& msg)
   {
      outcome.x = msg->x;
      outcome.y = msg->y;
      outcome.angle = msg->angle;
      outcome.touch = msg->touch;
      if(explore == true && dmp_b == true && state_b == true && sample_b == true)
      {
         cluster_message::SampleExplore s;
         s.state_x = state.state_x;
         s.state_y = state.state_y;
         s.state_angle = state.state_angle;
         s.v_x = dmp.v_x;
         s.v_y = dmp.v_y;
         s.v_pitch = dmp.v_pitch;
         s.roll = dmp.roll;
         s.grasp = dmp.grasp;
         s.lpos_x = sample.lpos_x;
         s.lpos_y = sample.lpos_y;
         s.lpos_pitch = sample.lpos_pitch;
         s.outcome_x = outcome.x;
         s.outcome_y = outcome.y;
         s.outcome_angle = outcome.angle;
         s.outcome_touch = outcome.touch;
         pub_datas_explore.publish(s);
         dmp_b = false;
         state_b = false;
         sample_b = false;
         //std::cout<<"cluster : got sample explore\n";
      }
      if(exploit == true && state_b == true && sample_b == true)
      {
         cluster_message::SampleExploit s;
         s.state_x = state.state_x;
         s.state_y = state.state_y;
         s.state_angle = state.state_angle;
         s.lpos_x = sample.lpos_x;
         s.lpos_y = sample.lpos_y;
         s.lpos_pitch = sample.lpos_pitch;
         s.outcome_x = outcome.x;
         s.outcome_y = outcome.y;
         s.outcome_angle = outcome.angle;
         s.outcome_touch = outcome.touch;
         pub_datas_exploit.publish(s);
         state_b = false;
         sample_b = false;
      }

   }

   void CallbackState(const detector::State::ConstPtr& msg)
   {
      state.state_angle = msg->state_angle;
      state.state_x = msg->state_x;
      state.state_y = msg->state_y;
      state_b = true;
      //std::cout<<"cluster : got state\n";
   }

   void CallbackSample(const motion::Action::ConstPtr& msg)
   {
      sample.lpos_x = msg->lpos_x;
      sample.lpos_y = msg->lpos_y;
      sample.lpos_pitch = msg->lpos_pitch;
      sample_b = true;
      //std::cout<<"cluster : got sample\n";
   }

   void CallbackExplore(const std_msgs::Bool::ConstPtr& msg)
   {
      explore = msg->data;
      //std::cout<<"cluster : got explore\n";
   }

   void CallbackExploit(const std_msgs::Bool::ConstPtr& msg)
   {
      exploit = msg->data;
   }

   void CallbackReadyHabit(const std_msgs::Bool::ConstPtr& msg)
   {
      ready_h = msg->data;
      if(ready_h && ready_nn)
      {
         std_msgs::Bool tmp;
         tmp.data = true;
         pub_ready.publish(tmp);
         ready_h = false;
         ready_nn = false;
      }
   }

   void CallbackReadyNN(const std_msgs::Bool::ConstPtr& msg)
   {
      ready_nn = msg->data;
      if(ready_h && ready_nn)
      {
         std_msgs::Bool tmp;
         tmp.data = true;
         pub_ready.publish(tmp);
         ready_h = false;
         ready_nn = false;
      }
   }

   void CallbackReadyDepth(const std_msgs::Bool::ConstPtr& msg)
   {
      ready_depth = msg->data;
      if(ready_depth && ready_outcome)
      {
         std_msgs::Bool tmp;
         tmp.data = true;
         pub_ready_sensor.publish(tmp);
         ready_depth = false;
         ready_outcome = false;
         std::cout<<"Cluster msg READY by depth\n";
      }
   }

   void CallbackReadyOutcome(const std_msgs::Bool::ConstPtr& msg)
   {
      ready_outcome = msg->data;
      if(ready_depth && ready_outcome)
      {
         std_msgs::Bool tmp;
         tmp.data = true;
         pub_ready_sensor.publish(tmp);
         ready_depth = false;
         ready_outcome = false;
         std::cout<<"Cluster msg READY by outcome\n";
      }
   }

   geometry_msgs::Point findVectorTransform(geometry_msgs::PoseStamped first_pose, float tx, float ty, tf2::Quaternion q_vector)
   {
      geometry_msgs::Point p;
      geometry_msgs::Point p_robot;
      tf2::Vector3 vec(tx,ty,0);
      tf2::Vector3 v_new = tf2::quatRotate(q_vector.inverse(),vec);
      p_robot.x = first_pose.pose.position.x + v_new.getX();
      p_robot.y = first_pose.pose.position.y + v_new.getY();
      p_robot.z = 0;
      p.x = p_robot.x - first_pose.pose.position.x;
      p.y = p_robot.y - first_pose.pose.position.y;

      return p;
   }
};
    

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cluster_msgs");
  ClusterMessage ci;
  ros::spin();

  return 0;
}