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
   ros::Publisher pub_datas_explore;
   ros::Publisher pub_datas_exploit;
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

  public:
   ClusterMessage()
   {
      sub_dmp = nh_.subscribe("/motion_pincher/dmp", 10, &ClusterMessage::CallbackDMP,this);
      sub_outcome = nh_.subscribe("/outcome_detector/outcome", 10, &ClusterMessage::CallbackOutcome,this);
      sub_explore = nh_.subscribe("/cog_learning/exploration", 10, &ClusterMessage::CallbackExplore,this);
      sub_exploit = nh_.subscribe("/cog_learning/exploitation", 10, &ClusterMessage::CallbackExploit,this);
      sub_state = nh_.subscribe("/outcome_detector/state", 10, &ClusterMessage::CallbackState,this);
      sub_sample = nh_.subscribe("/motion_pincher/action_sample", 10, &ClusterMessage::CallbackSample,this);
      pub_datas_explore = nh_.advertise<cluster_message::SampleExplore>("/cog_learning/sample_explore",1);
      pub_datas_exploit = nh_.advertise<cluster_message::SampleExploit>("/cog_learning/sample_exploit",1);
      explore = false;
      exploit = false;
   }
   ~ClusterMessage()
   {
   }

   void CallbackDMP(const motion::Dmp::ConstPtr& msg)
   {
      dmp.v_x = msg->v_x;
      dmp.v_y = msg->v_y;
      dmp.v_pitch = msg->v_pitch;
      dmp.grasp = msg->grasp;
      dmp.roll = msg->roll;  
      dmp_b = true;
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
   }

   void CallbackSample(const motion::Action::ConstPtr& msg)
   {
      sample.lpos_x = msg->lpos_x;
      sample.lpos_y = msg->lpos_y;
      sample.lpos_pitch = msg->lpos_pitch;
      sample_b = true;
   }

   void CallbackExplore(const std_msgs::Bool::ConstPtr& msg)
   {
      explore = msg->data;
   }

   void CallbackExploit(const std_msgs::Bool::ConstPtr& msg)
   {
      exploit = msg->data;
   }
};
    

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cluster_msgs");
  ClusterMessage ci;
  ros::spin();

  return 0;
}