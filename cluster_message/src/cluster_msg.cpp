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
#include "cluster_message/Outcome.h"
#include "cluster_message/State.h"


class ClusterMessage
{
  private:
   ros::NodeHandle nh_;
   ros::Subscriber sub_state;
   ros::Subscriber sub_action;
   ros::Subscriber sub_outcome;
   ros::Publisher pub_datas;

  public:
   ClusterMessage()
   {
      sub_state = nh_.subscribe("/perception/state", 1, &ClusterMessage::stateCallback,this);
      sub_action = nh_.subscribe("/pc_filter/pointcloud/filtered", 1, &ClusterMessage::stateCallback,this);
      sub_outcome = nh_.subscribe("/pc_filter/pointcloud/filtered", 1, &ClusterMessage::stateCallback,this);
      pub_datas = nh_.advertise<cluster_message::ClusterMessage>("/vae/inputs",1);
   }
   ~ClusterMessage()
   {
   }

   void stateCallback(const cluster_message::StateConstPtr& msg)
   {

   }

   void actionCallback(const cluster_message::ActionConstPtr& msg)
   {

   }

   void outcomeCallback(const cluster_message::OutcomeConstPtr& msg)
   {

   }
};
    

int main(int argc, char** argv)
{
  ros::init(argc, argv, "depth_perceptions");
  ClusterMessage ci;
  ros::spin();

  return 0;
}