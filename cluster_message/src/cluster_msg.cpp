#include <ros/ros.h>
// PCL specific includes

#include <iostream>
#include <string>
#include <std_msgs/Header.h>
#include <ros/header.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
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
#include "cluster_message/tfCamRob.h"
#include "cluster_message/tfRobCam.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_eigen/tf2_eigen.h>

class ClusterMessage
{
  private:
   ros::NodeHandle nh_;
   ros::Subscriber sub_dmp;
   ros::Subscriber sub_outcome;
   ros::Subscriber sub_rnd_explore;
   ros::Subscriber sub_direct_explore;
   ros::Subscriber sub_exploit;
   ros::Subscriber sub_state;
   ros::Subscriber sub_sample;
   ros::Subscriber sub_ready_habituation;
   ros::Subscriber sub_ready_nnga;
   ros::Subscriber sub_ready_depth;
   ros::Subscriber sub_ready_outcome;
   ros::Subscriber sub_new;
   ros::Subscriber sub_retry;
   ros::Subscriber sub_validate;
   ros::Subscriber sub_invalidate;
   ros::Subscriber sub_pause;
   ros::Subscriber sub_touch;
   ros::Subscriber sub_busy_out;
   ros::Subscriber sub_busy_act;
   ros::Subscriber sub_busy_nn_out;
   ros::Subscriber sub_busy_nn_act;
   ros::Publisher pub_datas_explore;
   ros::Publisher pub_datas_exploit;
   ros::Publisher pub_new_state;
   ros::Publisher pub_retry;
   ros::Publisher pub_signal;
   ros::Publisher pub_dmp_outcome;
   ros::Publisher pub_pause;
   ros::Publisher pub_ready;
   ros::ServiceServer service;
   ros::ServiceServer service_;
   detector::Outcome outcome;
   detector::State state;
   motion::Action sample;
   motion::Dmp dmp;
   motion::Dmp dmp_eval;
   bool dmp_b;
   bool outcome_b;
   double rnd_explore;
   double direct_explore;
   double exploit;
   bool state_b;
   bool sample_b;
   bool ready_habbit;
   bool ready_nn;
   bool ready_outcome;
   bool new_state;
   bool retry;
   bool send_sample;
   bool send_perception;
   double valid_perception;
   double invalid_perception;
   bool init_valid;
   bool init_invalid;
   bool touch;
   bool busy_vae_out;
   bool busy_vae_act;
   bool busy_nn_out;
   bool busy_nn_act;


  public:
   ClusterMessage()
   {
      sub_dmp = nh_.subscribe("/motion_pincher/dmp", 1, &ClusterMessage::CallbackDMP,this);
      sub_outcome = nh_.subscribe("/outcome_detector/outcome", 1, &ClusterMessage::CallbackOutcome,this);
      sub_rnd_explore = nh_.subscribe("/cog_learning/rnd_exploration", 1, &ClusterMessage::CallbackRndExplore,this);
      sub_direct_explore = nh_.subscribe("/cog_learning/direct_exploration", 1, &ClusterMessage::CallbackDirectExplore,this);
      sub_exploit = nh_.subscribe("/cog_learning/exploitation", 1, &ClusterMessage::CallbackExploit,this);
      sub_state = nh_.subscribe("/outcome_detector/state", 1, &ClusterMessage::CallbackState,this);
      sub_sample = nh_.subscribe("/motion_pincher/action_sample", 1, &ClusterMessage::CallbackSample,this);
      sub_ready_habituation = nh_.subscribe("/habituation/ready", 1, &ClusterMessage::CallbackReadyHabit,this);
      sub_ready_nnga = nh_.subscribe("/cog_learning/ready", 1, &ClusterMessage::CallbackReadyNN,this);
      sub_new = nh_.subscribe("/depth_perception/new_state", 1, &ClusterMessage::CallbackNewState,this);
      sub_retry = nh_.subscribe("/depth_perception/retry", 1, &ClusterMessage::CallbackRetry,this);
      sub_validate = nh_.subscribe("/habituation/valid_perception", 1, &ClusterMessage::CallbackValidate,this);
      sub_invalidate = nh_.subscribe("/habituation/invalid_perception", 1, &ClusterMessage::CallbackInvalidate,this);
      sub_pause = nh_.subscribe("/cluster_msg/pause_experiment", 1, &ClusterMessage::CallbackPause,this);
      sub_touch = nh_.subscribe("/motion_pincher/touch", 1, &ClusterMessage::CallbackTouch,this);
      sub_busy_out = nh_.subscribe("/cluster_msg/vae/busy_out", 1, &ClusterMessage::CallbackBusyVAEout,this);
      sub_busy_act = nh_.subscribe("/cluster_msg/vae/busy_act", 1, &ClusterMessage::CallbackBusyVAEact,this);
      sub_busy_nn_out = nh_.subscribe("/cluster_msg/nnga/busy_out", 1, &ClusterMessage::CallbackBusyNNout,this);
      sub_busy_nn_act = nh_.subscribe("/cluster_msg/nnga/busy_act", 1, &ClusterMessage::CallbackBusyNNact,this);
      pub_datas_explore = nh_.advertise<cluster_message::SampleExplore>("/cluster_msg/sample_explore",1);
      pub_datas_exploit = nh_.advertise<cluster_message::SampleExploit>("/cluster_msg/sample_exploit",1);
      pub_new_state = nh_.advertise<std_msgs::Bool>("/cluster_msg/new_state",1);
      pub_signal = nh_.advertise<std_msgs::Float64>("/cluster_msg/signal",1);
      pub_retry = nh_.advertise<std_msgs::Bool>("/cluster_msg/retry",1);
      pub_pause = nh_.advertise<std_msgs::Float64>("/cluster_msg/pause",1);
      pub_dmp_outcome = nh_.advertise<motion::DmpOutcome>("/cluster_msg/perception",1);
      pub_ready = nh_.advertise<std_msgs::Bool>("/test/ready",1);
      service = nh_.advertiseService("transform_dmp_cam_rob",&ClusterMessage::transformCamRob,this);
      service_ = nh_.advertiseService("transform_dmp_rob_cam",&ClusterMessage::transformRobCam,this);
      rnd_explore = 0.0;
      direct_explore = 0.0;
      exploit = 0.0;
      send_sample = false;
      outcome_b = false;
      ready_habbit = false;
      ready_nn = false;
   }
   ~ClusterMessage()
   {
   }

   bool transformCamRob(cluster_message::tfCamRob::Request  &req, cluster_message::tfCamRob::Response &res)
   {
      tf2::Quaternion q_orig(0,0,0,1);
      tf2::Quaternion q_rot;
      tf2::Quaternion q_vector;
      geometry_msgs::Point new_vec;
      geometry_msgs::Point vec_ori;
      geometry_msgs::Point result;
      geometry_msgs::PoseStamped first_pose;
      first_pose.pose.position.x = req.dmp_cam.fpos_x;
      first_pose.pose.position.y = req.dmp_cam.fpos_y;
      vec_ori.x = req.dmp_cam.fpos_x;
      vec_ori.y = req.dmp_cam.fpos_y;
      vec_ori.x = req.dmp_cam.fpos_x + vec_ori.x;
      vec_ori.y = req.dmp_cam.fpos_y + vec_ori.y;
      float dot_prod = (vec_ori.x*0.1) + (vec_ori.y*0);
      float det = (vec_ori.x*0) + (vec_ori.y*0.1);
      float ang = atan2(det,dot_prod);
      q_rot.setRPY(0,0,ang);
      q_vector = q_rot*q_orig;
      q_vector.normalize();
      result = findVectorTransform(first_pose,req.dmp_cam.v_x,req.dmp_cam.v_y,q_vector);
      res.dmp_robot.v_x = result.x;
      res.dmp_robot.v_y = result.y;
      res.dmp_robot.v_pitch = req.dmp_cam.v_pitch;
      res.dmp_robot.grasp = req.dmp_cam.grasp;
      res.dmp_robot.roll = req.dmp_cam.roll;
      res.dmp_robot.fpos_x = req.dmp_cam.fpos_x;
      res.dmp_robot.fpos_y = req.dmp_cam.fpos_y;
      
      return true;
   }

   bool transformRobCam(cluster_message::tfRobCam::Request  &req, cluster_message::tfRobCam::Response &res)
   {
      tf2::Quaternion q_orig(0,0,0,1);
      tf2::Quaternion q_rot;
      tf2::Quaternion q_vector;
      geometry_msgs::Point new_vec;
      geometry_msgs::Point vec_ori;
      geometry_msgs::Point result;
      geometry_msgs::PoseStamped first_pose;
      first_pose.pose.position.x = req.dmp_robot.fpos_x;
      first_pose.pose.position.y = req.dmp_robot.fpos_y;
      vec_ori.x = req.dmp_robot.fpos_x;
      vec_ori.y = req.dmp_robot.fpos_y;
      vec_ori.x = req.dmp_robot.fpos_x + vec_ori.x;
      vec_ori.y = req.dmp_robot.fpos_y + vec_ori.y;
      float dot_prod = (vec_ori.x*0.1) + (vec_ori.y*0);
      float det = (vec_ori.x*0) + (vec_ori.y*0.1);
      float ang = atan2(det,dot_prod);
      q_rot.setRPY(0,0,ang);
      q_vector = q_rot*q_orig;
      q_vector.normalize();
      result = findVectorTransformCam(first_pose,req.dmp_robot.v_x,req.dmp_robot.v_y,q_vector);
      res.dmp_cam.v_x = result.x;
      res.dmp_cam.v_y = result.y;
      res.dmp_cam.v_pitch = req.dmp_robot.v_pitch;
      res.dmp_cam.grasp = req.dmp_robot.grasp;
      res.dmp_cam.roll = req.dmp_robot.roll;
      
      return true;
   }

   void CallbackDMP(const motion::Dmp::ConstPtr& msg)
   {
      dmp.fpos_x = msg->fpos_x;
      dmp.fpos_y = msg->fpos_y;
      dmp.v_x = msg->v_x;
      dmp.v_y = msg->v_y;
      dmp.v_pitch = msg->v_pitch;
      dmp.grasp = msg->grasp;
      dmp.roll = msg->roll;
      dmp_b = true;
      std::cout<<"cluster : got DMP\n";
   }

   void CallbackOutcome(const detector::Outcome::ConstPtr& msg)
   {
      outcome.x = msg->x;
      outcome.y = msg->y;
      outcome.angle = msg->angle;
      outcome.touch = msg->touch;
      outcome_b = true;
      std::cout<<"cluster : got outcome\n";
   }

   void CallbackState(const detector::State::ConstPtr& msg)
   {
      state.state_angle = msg->state_angle;
      state.state_x = msg->state_x;
      state.state_y = msg->state_y;
      state_b = true;
      std::cout<<"cluster : got state\n";
   }

   void CallbackSample(const motion::Action::ConstPtr& msg)
   {
      sample.lpos_x = msg->lpos_x;
      sample.lpos_y = msg->lpos_y;
      sample.lpos_pitch = msg->lpos_pitch;
      sample_b = true;
      std::cout<<"cluster : got sample\n";
   }

   void CallbackRndExplore(const std_msgs::Float64::ConstPtr& msg)
   {
      rnd_explore = msg->data;
      if(rnd_explore > 0.5 && new_state && outcome_b && dmp_b && sample_b && state_b && !busy_nn_out && !busy_nn_act && !busy_vae_out && !busy_vae_act)
      {
         std::cout<<"Cluster_msg : RANDOM exploration, sending datas to models...\n";
         //ros::Duration(4.5).sleep();
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
         send_sample = true;
         outcome_b = false;
         dmp_b = false;
         state_b = false;
         sample_b = false;
      }
      if(rnd_explore > 0.5 && ready_habbit && ready_nn)
      {
         std::cout<<"Cluster_msg : RANDOM exploration DONE\n";
         //ros::Duration(3.5).sleep();
         std_msgs::Bool b;
         b.data = true;
         pub_new_state.publish(b);
         new_state = false;
         ready_habbit = false;
         ready_nn = false;
         pub_ready.publish(b);
         /*
         if(!touch)
         {
            std_msgs::Float64 f;
            f.data = 1.0;
            pub_signal.publish(f);
            ros::Duration(2.5).sleep();
            f.data = 0.0;
            pub_signal.publish(f);
         } */
      }
      if(rnd_explore > 0.5 && retry)
      {
         std::cout<<"Cluster_msg : RANDOM exploration Retry\n";
         std_msgs::Bool b;
         b.data = true;
         new_state = false;
         ready_habbit = false;
         ready_nn = false;
         outcome_b = false;
         dmp_b = false;
         state_b = false;
         sample_b = false;
         retry = false;
         std_msgs::Float64 f;
         f.data = 1.0;
         pub_signal.publish(f);
         ros::Duration(0.5).sleep();
         f.data = 0.0;
         pub_signal.publish(f);
      }
   }

   void CallbackDirectExplore(const std_msgs::Float64::ConstPtr& msg)
   {
      direct_explore = msg->data;
      if(direct_explore > 0.5 && new_state && outcome_b && dmp_b && sample_b && state_b && !busy_nn_out && !busy_nn_act && !busy_vae_out && !busy_vae_act)
      {
         std::cout<<"Cluster_msg : DIRECT exploration, sending datas to models...\n";
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
         outcome_b = false;
         dmp_b = false;
         state_b = false;
         sample_b = false;
      }
      if(direct_explore > 0.5 && new_state && ready_habbit && ready_nn)
      {
         std::cout<<"Cluster_msg : DIRECT exploration DONE\n";
         std_msgs::Bool b;
         b.data = true;
         pub_new_state.publish(b);
         new_state = false;
         ready_habbit = false;
         ready_nn = false;
         send_sample = false;
         if(!touch)
         {
            std_msgs::Float64 f;
            f.data = 1.0;
            pub_signal.publish(f);
            ros::Duration(2.5).sleep();
            f.data = 0.0;
            pub_signal.publish(f);
         }  
      }
      if(direct_explore > 0.5 && retry)
      {
         new_state = false;
         ready_habbit = false;
         ready_nn = false;
         send_sample = false;
         outcome_b = false;
         dmp_b = false;
         state_b = false;
         sample_b = false;
         retry = false;
         std_msgs::Float64 f;
         f.data = 1.0;
         pub_signal.publish(f);
         ros::Duration(2.5).sleep();
         f.data = 0.0;
         pub_signal.publish(f);
      }
   }

   void CallbackExploit(const std_msgs::Float64::ConstPtr& msg)
   {
      exploit = msg->data;
      if(exploit > 0.5 && new_state && outcome_b && !send_perception)
      {
         motion::DmpOutcome tmp;
         tmp.v_x = dmp.v_x;
         tmp.v_y = dmp.v_y;
         tmp.v_pitch = dmp.v_pitch;
         tmp.grasp = dmp.grasp;
         tmp.roll = dmp.roll;
         tmp.x = outcome.x;
         tmp.y = outcome.y;
         tmp.angle = outcome.angle;
         tmp.touch = outcome.touch;
         pub_dmp_outcome.publish(tmp);
         send_perception = true;
         outcome_b = false;
      }
      if(exploit > 0.5 && new_state && valid_perception > 0.5 && !send_sample)
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
         send_sample = true;
      }
      if(exploit > 0.5 && new_state && valid_perception > 0.5 && ready_nn)
      {
         ready_nn = false;
         send_sample = false;
         new_state = false;
         valid_perception = 0.0;
         std_msgs::Float64 f;
         f.data = 1.0;
         pub_signal.publish(f);
         ros::Duration(0.5).sleep();
         f.data = 0.0;
         pub_signal.publish(f);
         init_valid = false;
      }
      if(exploit > 0.5 && new_state && invalid_perception > 0.5)
      {
         new_state = false;
         invalid_perception = 0.0;
         std_msgs::Float64 f;
         f.data = 1.0;
         pub_signal.publish(f);
         ros::Duration(0.5).sleep();
         f.data = 0.0;
         pub_signal.publish(f);
         init_invalid = false;
      }
      if(exploit > 0.5 && retry)
      {
         retry = false;
         state_b = false;
         sample_b = false;
         outcome_b = false;
         dmp_b = false;
         std_msgs::Float64 f;
         f.data = 1.0;
         pub_signal.publish(f);
         ros::Duration(0.5).sleep();
         f.data = 0.0;
         pub_signal.publish(f);
      }
   }

   void CallbackNewState(const std_msgs::Bool::ConstPtr& msg)
   {
      new_state = msg->data;
      if(new_state)
      {
         std::cout<<"cluster_msg: got new state\n";
      }
   }

   void CallbackRetry(const std_msgs::Bool::ConstPtr& msg)
   {
      retry = msg->data;
   }

   void CallbackReadyHabit(const std_msgs::Bool::ConstPtr& msg)
   {
      ready_habbit = msg->data;
      if(ready_habbit)
      {
         std::cout<<"cluster_msg: got ready_habit\n";
      }
   }

   void CallbackReadyNN(const std_msgs::Bool::ConstPtr& msg)
   {
      ready_nn = msg->data;
      if(ready_nn)
      {
         std::cout<<"cluster_msg: got ready_nnga\n";
      }
   }

   void CallbackValidate(const std_msgs::Float64::ConstPtr& msg)
   {
      if(msg->data > 0.5 && !init_valid)
      {
         valid_perception = msg->data;
         init_invalid = true;
      }
   }

   void CallbackInvalidate(const std_msgs::Float64::ConstPtr& msg)
   {
      if(msg->data > 0.5 && !init_invalid)
      {
         invalid_perception = msg->data;
         init_invalid = true;
      }
   }

   void CallbackPause(const std_msgs::Bool::ConstPtr& msg)
   {
      if(msg->data == true)
      {
         std_msgs::Float64 tmp;
         tmp.data = 1.0;
         pub_pause.publish(tmp);
      }
      else
      {
         std_msgs::Float64 tmp;
         tmp.data = 0.0;
         pub_pause.publish(tmp);
      }
   }

   void CallbackTouch(const std_msgs::Bool::ConstPtr& msg)
   {
      touch = msg->data;
   }

   void CallbackBusyVAEout(const std_msgs::Bool::ConstPtr& msg)
   {
      busy_vae_out = msg->data;
   }

   void CallbackBusyVAEact(const std_msgs::Bool::ConstPtr& msg)
   {
      busy_vae_act = msg->data;
   }

   void CallbackBusyNNout(const std_msgs::Bool::ConstPtr& msg)
   {
      busy_nn_out = msg->data;
   }

   void CallbackBusyNNact(const std_msgs::Bool::ConstPtr& msg)
   {
      busy_nn_act = msg->data;
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

   geometry_msgs::Point findVectorTransformCam(geometry_msgs::PoseStamped first_pose, float tx, float ty, tf2::Quaternion q_vector)
   {
      geometry_msgs::Point p;
      geometry_msgs::Point p_robot;
      tf2::Vector3 vec(tx,ty,0);
      tf2::Vector3 v_new = tf2::quatRotate(q_vector,vec);
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