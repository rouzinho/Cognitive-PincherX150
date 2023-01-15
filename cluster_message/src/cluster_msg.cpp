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
   std::vector<float> v_state;
   std::vector<float> v_action;
   std::vector<float> v_outcome;
   std::vector<float> v_cluster;
   bool action;
   bool outcome;

  public:
   ClusterMessage()
   {
      sub_state = nh_.subscribe("/perception/state", 1, &ClusterMessage::stateCallback,this);
      sub_action = nh_.subscribe("/pc_filter/pointcloud/filtered", 1, &ClusterMessage::actionCallback,this);
      sub_outcome = nh_.subscribe("/pc_filter/pointcloud/filtered", 1, &ClusterMessage::outcomeCallback,this);
      pub_datas = nh_.advertise<cluster_message::ClusterMessage>("/vae/inputs",1);
      v_state.resize(0);
      v_action.resize(0);
      v_outcome.resize(0);
      v_cluster.resize(0);
   }
   ~ClusterMessage()
   {
   }

   void stateCallback(const cluster_message::StateConstPtr& msg)
   {
      v_state.resize(0);
      for(int i = 0; i < msg->state.size(); i++)
      {
         v_state.push_back(msg->state[i]);
      }

      if(action && outcome)
      {
         int size_cl = v_state.size() + v_action.size() + v_outcome.size();
         cluster_message::ClusterMessage cl_input;
         cl_input.input.resize(0);
         for(int i = 0; i < v_state.size(); i++)
         {
            cl_input.input.push_back(v_state[i]);
         }
         for(int i = 0; i < v_action.size(); i++)
         {
            cl_input.input.push_back(v_action[i]);
         }
         for(int i = 0; i < v_outcome.size(); i++)
         {
            cl_input.input.push_back(v_outcome[i]);
         }
         pub_datas.publish(cl_input);
         action = false;
         outcome = false;
      }

   }

   void actionCallback(const cluster_message::ActionConstPtr& msg)
   {
      action = true;
      v_action.resize(0);
      for(int i = 0; i < msg->action.size(); i++)
      {
         v_action.push_back(msg->action[i]);
      }
   }

   void outcomeCallback(const cluster_message::OutcomeConstPtr& msg)
   {
      outcome = true;
      v_outcome.resize(0);
      for(int i = 0; i < msg->outcome.size(); i++)
      {
         v_outcome.push_back(msg->outcome[i]);
      }
   }
};
    

int main(int argc, char** argv)
{
  ros::init(argc, argv, "depth_perceptions");
  ClusterMessage ci;
  ros::spin();

  return 0;
}