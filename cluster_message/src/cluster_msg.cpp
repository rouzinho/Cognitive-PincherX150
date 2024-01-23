#include <ros/ros.h>
// PCL specific includes

#include <iostream>
#include <string>
#include <std_msgs/Header.h>
#include <ros/header.h>
#include <std_msgs/Bool.h>
#include <stdio.h>
#include "cluster_message/ClusterMessage.h"
#include "cluster_message/Action.h"
#include "cluster_message/OutcomeAE.h"
#include "cluster_message/State.h"
#include "motion/Dmp.h"
#include "motion/DmpOutcome.h"
#include "detector/Outcome.h"


class ClusterMessage
{
  private:
   ros::NodeHandle nh_;
   ros::Subscriber sub_dmp;
   ros::Subscriber sub_outcome;
   ros::Publisher pub_datas;
   detector::Outcome outcome;
   motion::Dmp dmp;
   bool action;
   bool out;

  public:
   ClusterMessage()
   {
      sub_dmp = nh_.subscribe("/motion_pincher/dmp", 10, &ClusterMessage::CallbackDMP,this);
      sub_outcome = nh_.subscribe("/outcome_detector/outcome", 10, &ClusterMessage::CallbackOutcome,this);
      pub_datas = nh_.advertise<motion::DmpOutcome>("/cog_learning/dmp_outcome",1);
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
      action = true;
      if(action == true && out == true)
      {
         motion::DmpOutcome dmpout;
         dmpout.v_x = dmp.v_x;
         dmpout.v_y = dmp.v_y;
         dmpout.v_pitch = dmp.v_pitch;
         dmpout.grasp = dmp.grasp;
         dmpout.roll = dmp.roll;
         dmpout.x = outcome.x;
         dmpout.y = outcome.y;
         dmpout.angle = outcome.angle;
         dmpout.touch = outcome.touch;
         action = false;
         out = false;
         pub_datas.publish(dmpout);
      }
   }

   void CallbackOutcome(const detector::Outcome::ConstPtr& msg)
   {
      outcome.x = msg->x;
      outcome.y = msg->y;
      outcome.angle = msg->angle;
      outcome.touch = msg->touch;
      out = true;
      if(action == true && out == true)
      {
         motion::DmpOutcome dmpout;
         dmpout.v_x = dmp.v_x;
         dmpout.v_y = dmp.v_y;
         dmpout.v_pitch = dmp.v_pitch;
         dmpout.grasp = dmp.grasp;
         dmpout.roll = dmp.roll;
         dmpout.x = outcome.x;
         dmpout.y = outcome.y;
         dmpout.angle = outcome.angle;
         dmpout.touch = outcome.touch;
         action = false;
         out = false;
         pub_datas.publish(dmpout);
      }
   }

};
    

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cluster_msgs");
  ClusterMessage ci;
  ros::spin();

  return 0;
}