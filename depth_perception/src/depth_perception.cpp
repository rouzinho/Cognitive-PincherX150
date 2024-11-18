#include <ros/ros.h>
// PCL specific includes
#include <tf2_ros/buffer.h>
#include <tf2/transform_datatypes.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>
#include <pcl_ros/transforms.h>
#include <iostream>
#include <fstream>
#include <string>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/TransformStamped.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
//#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>
#include <ros/header.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <std_msgs/Float64.h>
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <future>
#include <thread>
#include <chrono>

using namespace std;
static const std::string OPENCV_WINDOW = "Image window";
static const std::string OPENCV_WINDOW_BIS = "Image";
typedef pcl::PointXYZRGB PointT;

static bool getAnswer()
{    
  char ch = getchar();
  return true;
}

class DepthImage
{
  private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    ros::Subscriber sub_point_cloud;
    ros::Subscriber sub_point_cloud_object;
    ros::Subscriber sub_activate;
    ros::Subscriber sub_pause;
    ros::Subscriber sub_touch;
    geometry_msgs::TransformStamped transformStamped;
    image_transport::Publisher pub_state;
    ros::Publisher pub_retry;
    ros::Publisher pub_pause;
    ros::Publisher pub_pause_perception;
    ros::Publisher pub_pause_dft;
    ros::Publisher pub_new_state;
    ros::Publisher pub_activate_detector;
    ros::Publisher pub_activate_depth;
    ros::Publisher pub_reset_detector;
    ros::Publisher pub_name_state;
    ros::Publisher pub_success;
    ros::Publisher pub_ready;
    ros::Publisher pub_grasping;
    ros::Publisher pub_center;
    bool tf_in;
    tf2_ros::TransformListener tfListener;
    tf2_ros::Buffer tfBuffer;
    Eigen::Matrix4d robot_frame;
    std_msgs::Header header;
    float crop_min_x;
    float crop_min_y;
    float crop_max_x;
    float crop_max_y;
    float crop_max_z;
    float crop_min_z;
    double ax;
    double bx;
    double ay;
    double by;
    double az;
    double bz;
    float threshold_depth;
    bool first;
    cv::Mat cv_image;
    cv::Mat res;
    cv::Mat res_b;
    cv::Mat fil;
    cv::Mat fil_b;
    cv::Mat cv_nf;
    int count;
    int threshold;
    int size_neural_field;
    bool start;
    int threshold_change;
    cv::Mat final_image;
    cv::Mat mask;
    cv::Mat tmp_mask;
    bool out_boundary;
    bool init_values;
    float gain;
    int s_x;
    int s_y;
    int s_reduce_w;
    int s_reduce_h;
    bool first_gen;
    int count_init;
    bool lock_callback;
    int count_lock;
    bool begin_count;
    bool init_params;
    cv::Mat display;
    bool pause;
    bool touch;
    bool grasping;
    bool setup_grasping;

  public:
    DepthImage():
    tfListener(tfBuffer),
    it_(nh_)
    {
      sub_point_cloud = nh_.subscribe("/pc_filter/pointcloud/filtered", 1, &DepthImage::pointCloudCb,this);
      sub_point_cloud_object = nh_.subscribe("/pc_filter/pointcloud/objects", 1, &DepthImage::pointCloudObjectCb,this);
      sub_activate = nh_.subscribe("/depth_perception/activate", 1, &DepthImage::activateCb,this);
      sub_touch = nh_.subscribe("/motion_pincher/touch", 1, &DepthImage::callbackTouch,this);
      sub_pause = nh_.subscribe("/cluster_msg/pause_perception", 1, &DepthImage::activatePause,this);
      pub_pause = nh_.advertise<std_msgs::Bool>("/cluster_msg/pause_experiment",1);
      pub_pause_dft = nh_.advertise<std_msgs::Bool>("/cluster_msg/pause_dft",1);
      pub_pause_perception = nh_.advertise<std_msgs::Bool>("/cluster_msg/pause_perception",1);
      pub_state = it_.advertise("/depth_perception/dnf_state", 1);
      pub_retry = nh_.advertise<std_msgs::Bool>("/depth_perception/retry",1);
      pub_new_state = nh_.advertise<std_msgs::Bool>("/depth_perception/new_state",1);
      pub_activate_detector = nh_.advertise<std_msgs::Bool>("/outcome_detector/activate",1);
      pub_activate_depth = nh_.advertise<std_msgs::Bool>("/depth_perception/activate",1);
      pub_reset_detector = nh_.advertise<std_msgs::Bool>("/outcome_detector/reset",1);
      pub_name_state = nh_.advertise<std_msgs::String>("/depth_perception/name_state",1);
      pub_success = nh_.advertise<std_msgs::Bool>("/depth_perception/sample_success",1);
      pub_ready = nh_.advertise<std_msgs::Bool>("/depth_perception/ready",1);
      pub_grasping = nh_.advertise<std_msgs::Bool>("/outcome_detector/grasping",1);
      pub_center = nh_.advertise<geometry_msgs::Point>("/depth_perception/center",1);
      tf_in = false;
      size_neural_field = 100;
      threshold_depth = 0.05;
      first = true;
      s_x = 620;  //620
      s_y = 840;  //840
      s_reduce_w = 180; //there's a crop in s_x
      s_reduce_h = 100;
      display = cv::Mat(s_x, s_y, CV_32F,cv::Scalar(std::numeric_limits<float>::min()));
      ros::param::get("init_params", init_params);
      ros::param::get("crop_min_x", crop_min_x);
      ros::param::get("crop_max_x", crop_max_x);
      ros::param::get("crop_min_y", crop_min_y);
      ros::param::get("crop_max_y", crop_max_y);
      ros::param::get("crop_min_z", crop_min_z);
      ros::param::get("crop_max_z", crop_max_z);
      ros::param::get("ax", ax);
      ros::param::get("bx", bx);
      ros::param::get("ay", ay);
      ros::param::get("by", by);
      ros::param::get("az", az);
      ros::param::get("bz", bz);
      cv_image = cv::Mat(s_x, s_y, CV_32F,cv::Scalar(std::numeric_limits<float>::min()));
      fil = cv::Mat(s_x, s_y, CV_8U,cv::Scalar(std::numeric_limits<float>::min()));
      fil_b = cv::Mat(s_x, s_y, CV_8U,cv::Scalar(std::numeric_limits<float>::min()));
      final_image = cv::Mat(s_reduce_h, s_reduce_w, CV_8U,cv::Scalar(std::numeric_limits<float>::min()));
      //cv_nf = cv::Mat(50, 50, CV_32FC1,cv::Scalar(std::numeric_limits<float>::min()));
      //cv_image = cv::Mat(1024, 1024, CV_32FC1,cv::Scalar(0));
      cv::Mat m = cv::imread("/home/altair/interbotix_ws/src/depth_perception/mask/new_mask.jpg");
      cv::cvtColor(m,tmp_mask,cv::COLOR_RGB2GRAY);
      cv::resize(tmp_mask, mask, cv::Size(s_reduce_w, s_reduce_h), cv::INTER_LANCZOS4);
      rmStates();
      count = 0;
      threshold = 15;
      start = true;
      threshold_change = 40;
      out_boundary = false;
      init_values = false;
      first_gen = true;
      count_init = 0;
      lock_callback = false;
      count_lock = 0;
      grasping = false;
      setup_grasping = false;
    }
    ~DepthImage()
    {
    }

    void pointCloudCb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped(new pcl::PointCloud<pcl::PointXYZ>);
      if(!tf_in)
      {
        listenTransform();
      }
      //std::cout<<cloud_msg->header.frame_id<<"/n";
      
      pcl::PCLPointCloud2 pcl_pc2;
      pcl_conversions::toPCL(*cloud_msg, pcl_pc2);
      pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
      pcl::transformPointCloud(*temp_cloud,*cloud_transformed,robot_frame);

      //print4x4Matrix(robot_frame);
      if(!init_params)
      {
        if(count_init < 30)
        {
          getExtremeValues(cloud_transformed);
          count_init++;
        }
        else
        {
          init_values = true;
          //genDepthFromPcl(cloud_transformed);
        }
      }
      
      if(!lock_callback)
      {
        count_lock++;
        //std::cout<<"count lock: "<<count_lock<<"\n";
      }
      if(count_lock > 30 && !lock_callback && !pause)
      {
        resetDepth();
        lock_callback = true;
        count_lock = 0; 
      }

    }

    void pointCloudObjectCb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
      lock_callback = true;
      count_lock = 0;
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped(new pcl::PointCloud<pcl::PointXYZ>);
      header = cloud_msg->header;
      if(!tf_in)
      {
        listenTransform();
      }
      //std::cout<<cloud_msg->header.frame_id<<"/n";
      
      pcl::PCLPointCloud2 pcl_pc2;
      pcl_conversions::toPCL(*cloud_msg, pcl_pc2);
      pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
      pcl::transformPointCloud(*temp_cloud,*cloud_transformed,robot_frame);

      //print4x4Matrix(robot_frame);
      //getExtremeValues(cloud_transformed);
      //if(init_params)
      if(!grasping)
      {
        genDepthFromPcl(cloud_transformed);
      }
      //std::string name_state = "/home/altair/interbotix_ws/src/depth_perception/states/state_48.jpg";
      //cv::Mat img1 = imread(name_state, cv::IMREAD_COLOR);
      //stateChanged(img1,48);
      lock_callback = false;
      begin_count = false;
    }

    void activateCb(const std_msgs::BoolConstPtr& msg)
    {
      begin_count = true;
      grasping = false;
      first_gen = true;
      if(msg->data == true)
      {
        start = true;
        count = 0;
        //s_y = static_cast<int>(720 * gain);
        cv_image = cv::Mat(s_x, s_y, CV_32F,cv::Scalar(std::numeric_limits<float>::min()));
        fil = cv::Mat(s_x, s_y, CV_8U,cv::Scalar(std::numeric_limits<float>::min()));
        fil_b = cv::Mat(s_x, s_y, CV_8U,cv::Scalar(std::numeric_limits<float>::min()));
        final_image = cv::Mat(s_reduce_h, s_reduce_w, CV_8U,cv::Scalar(std::numeric_limits<float>::min()));
      }
      else
      {
        start = false;
      }
    }

    void callbackTouch(const std_msgs::BoolConstPtr& msg)
    {
      touch = msg->data;
    }

    void activatePause(const std_msgs::BoolConstPtr& msg)
    {
      if(msg->data == true)
      {
        std::cout<<"Pause called, resetting states...\n";
        pub_pause.publish(msg);
        pub_pause_dft.publish(msg);
        setup_grasping = false;
        grasping = true;
        pause = true;
        first = true;
        rmStates();
        pub_reset_detector.publish(msg);
      }
      if(msg->data == false)
      {
        std::cout<<"resuming experiment\n";
        std_msgs::Bool b;
        b.data = true;
        pub_new_state.publish(b);
        pub_activate_depth.publish(b);
        ros::Duration(0.5).sleep();
        pub_activate_detector.publish(b);
        b.data = false;
        pause = false;
        pub_pause.publish(b);
        pub_pause_dft.publish(b);
      }
    }

    void resetDepth()
    {
      if(!setup_grasping && !grasping)
      {
        std::cout<<"Emergency Grasping detected, waiting for object to be back\n";
        grasping = true;
        std_msgs::Bool msg;
        msg.data = true;
        pub_grasping.publish(msg);
        pub_pause_perception.publish(msg);
        start = false;
        rmStates();
      }
      
    }

    void listenTransform()
    {
      if(!tf_in)
      {
        try
        {
          transformStamped = tfBuffer.lookupTransform("px150/base_link", "rgb_camera_link",ros::Time(0));
        } 
        catch (tf2::TransformException &ex) 
        {
          ROS_WARN("%s", ex.what());
          ros::Duration(1.0).sleep();
        }
        tf_in = true;
        Eigen::Isometry3d mat = tf2::transformToEigen(transformStamped);
        robot_frame = mat.matrix();
      }
    }

    void getExtremeValues(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
      float min_x = 1000;
      float max_x = -1000;
      float min_y = 1000;
      float max_y = 0;
      float min_z = 1000;
      float max_z = 0;
      float px;
      float py;
      float pz;
      std::vector<float> min_max_values;
      for (int i=0; i< cloud->points.size();i++)
      {
        if (cloud->points[i].z == cloud->points[i].z)
        {
            px = cloud->points[i].x * 1000.0*-1;// *1000.0;
            py = cloud->points[i].y * 1000.0*-1;// *1000.0*-1; //revert image because it's upside down for display
            pz = cloud->points[i].z * 1000.0;
            //std::cout<<px<<"\n";
            if(px < min_x)
            {
              min_x = px;
            }
            if(px > max_x)
            {
              max_x = px;
            }
            if(py < min_y)
            {
              min_y = py;
            }
            if(py > max_y)
            {
              max_y = py;
            }
            if(pz < min_z)
            {
              min_z = pz;
            }
            if(pz > max_z)
            {
              max_z = pz;
            }
        }
      }
      if(min_x < crop_min_x)
      {
        crop_min_x = min_x;
      }
      if(min_y < crop_min_y)
      {
        crop_min_y = min_y;
      }
      if(max_x > crop_max_x)
      {
        crop_max_x = max_x;
      }
      if(max_y > crop_max_y)
      {
        crop_max_y = max_y;
      }
      if(min_z < crop_min_z)
      {
        crop_min_z = min_z;
      }
      if(max_z > crop_max_z)
      {
        crop_max_z = max_z;
      }
      float length_x;
      float length_y;
      if(min_x < 0 && max_x < 0)
      {
        length_x = std::abs(min_x) - std::abs(max_x);
      }
      if(min_y < 0 && max_y > 0)
      {
        length_y = std::abs(min_y) + max_y;
      }
      gain = length_y / length_x;
      
      //s_y = static_cast<int>(720 * gain);
      //cv_image = cv::Mat(720, s_y, CV_32F,cv::Scalar(std::numeric_limits<float>::min()));
      std::cout<<"crop min x : "<<crop_min_x<<"\n";
      std::cout<<"crop max x : "<<crop_max_x<<"\n";
      std::cout<<"crop min y : "<<crop_min_y<<"\n";
      std::cout<<"crop max y : "<<crop_max_y<<"\n";
      std::cout<<"crop min z : "<<crop_min_x<<"\n";
      std::cout<<"crop max z : "<<crop_max_z<<"\n";
      std::cout<<"gain : "<<gain<<"\n";
    }

    void genDepthFromPcl(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
      if(first_gen)
      {
        cv_image = cv::Mat(s_x, s_y, CV_32F,cv::Scalar(std::numeric_limits<float>::min()));
        first_gen = false;
      }
      cv::Mat cv_nf = cv::Mat(s_reduce_h, s_reduce_w, CV_32FC1,cv::Scalar(std::numeric_limits<float>::min()));
      if(!first_gen)
      {
        //crop image to fit robot space to DNF
        //IMPORTANT for calibration
        //dim cv_image
        cv::Rect crop_img(0,160,s_y-40,s_x-200);
        cv::Mat rot;
        const float bad_point = std::numeric_limits<float>::quiet_NaN();
        int pixel_pos_x;
        int pixel_pos_y;
        float pixel_pos_z;
        float px;
        float py;
        float pz;
        if(!init_params)
        {
          ax = (static_cast<double>(s_x))/(std::abs(crop_min_x)-std::abs(crop_max_x)); //1024 image width
          bx = 0 - (ax*crop_min_x);
          ay = (static_cast<double>(s_y))/(crop_max_y-crop_min_y); //1024 image height
          by = 0 - (ay*crop_min_y);
          az = (static_cast<double>(1000))/(crop_max_z-crop_min_z);
          bz = 0 - (az*crop_min_z);
          std::cout<<"ax : "<<ax<<"\n";
          std::cout<<"bx : "<<bx<<"\n";
          std::cout<<"ay : "<<ay<<"\n";
          std::cout<<"by : "<<by<<"\n";
          std::cout<<"az : "<<az<<"\n";
          std::cout<<"bz : "<<bz<<"\n";
        }
        if(start == true)
        {
          if(count < threshold)
          {
            for (int i=0; i< cloud->points.size();i++)
            {
              px = cloud->points[i].x * 1000.0*-1;
              py = cloud->points[i].y * 1000.0*-1;//revert image because it's upside down for display
              pz = cloud->points[i].z * 1000.0;
              pixel_pos_x = (int) (ax * px + bx);
              pixel_pos_y = (int) (ay * py + by);
              pixel_pos_z = (az * pz + bz);
              pixel_pos_z = pixel_pos_z/1000.0;
              if(px > crop_max_x || px < crop_min_x || py > crop_max_y || py < crop_min_y)
              {
                cloud->points[i].z = bad_point;
              }
              if (cloud->points[i].z == cloud->points[i].z)
              {
                if((pixel_pos_x > 0 && pixel_pos_x < s_x) && (pixel_pos_y > 0 && pixel_pos_y < s_y))
                {
                  cv_image.at<float>(pixel_pos_x,pixel_pos_y) = pixel_pos_z;
                }
              }
            }
            display = cv::Mat(s_x, s_y, CV_32F,cv::Scalar(std::numeric_limits<float>::min()));
            display = cv_image(crop_img);
          }
          else
          {
            //ros::Duration(10.5).sleep();
            cv::Mat fil_nf;// = cv::Mat(1024, 1024, CV_8U,cv::Scalar(std::numeric_limits<float>::min()));
            cv::Mat r_nf;
            cv::Mat padded;
            cv::Mat resized;
            cv::Mat filtered;
            cv::Mat filtered_b;
            cv::Mat resized_b;
            bool change;
            //one for dnf, one bigger for changes detection
            cv::resize(display, resized, cv::Size(s_reduce_w, s_reduce_h), cv::INTER_LANCZOS4);
            cv::resize(cv_image, resized_b, cv::Size(s_reduce_w, s_reduce_h), cv::INTER_LANCZOS4);
            cv::cvtColor(resized,res,cv::COLOR_GRAY2RGB);
            cv::cvtColor(resized_b,res_b,cv::COLOR_GRAY2RGB);
            res.convertTo(res, CV_8U, 255.0);
            res_b.convertTo(res_b, CV_8U, 255.0);
            cv::medianBlur(res,fil,(3,3)); //9 9
            cv::medianBlur(res_b,fil_b,(3,3)); //9 9
            fil = enhanceDepth(fil,0.01);
            fil_b = enhanceDepth(fil_b,0.01);
            cv::Mat tmp = fil.clone();
            findCenter(tmp);
            //cv::circle(fil, p, 5, cv::Scalar(0,0,255), -1);

            //get filtered image
            //for dnf
            cv::cvtColor(fil,r_nf,cv::COLOR_RGB2GRAY);
            r_nf.convertTo(cv_nf, CV_32FC1, 1/255.0);
            sensor_msgs::ImagePtr dobject_nf = cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_32FC1, cv_nf).toImageMsg();
            pub_state.publish(dobject_nf);
            //std::cout<<"rows : "<<filtered_b.rows<<"\n";
            //std::cout<<"cols : "<<filtered_b.cols<<"\n";
            final_image = fil_b.clone();
            //for bigger image
            //cv::resize(filtered_b, final_image, cv::Size(s_reduce_w, s_reduce_h), cv::INTER_LANCZOS4);
            
            //ros::Duration(5.5).sleep();
            int c = getFilesCount();
            if(first == false)
            {
              //check if object isn't out of robot's reach
              bool border = false;//detectBoundaries(final_image);
              if(!border)
              {
                change = stateChanged(final_image,c);
                if(change)
                {
                  std::cout<<"DEPTH : changes !\n";
                  std::string s = std::to_string(c);
                  std::string name_state = "/home/altair/interbotix_ws/src/depth_perception/states/state_"+s+".jpg";
                  cv::imwrite(name_state, final_image);
                  std_msgs::String msg_state;
                  msg_state.data = name_state;
                  //pub_name_state.publish(msg_state);
                  //std::cout<<"RESET DNF\n";
                  std_msgs::Bool msg;
                  msg.data = true;
                  pub_new_state.publish(msg);
                  ros::Duration(0.5).sleep();
                  //pub_reset_detector.publish(msg);
                  //ros::Duration(0.5).sleep();
                  pub_activate_detector.publish(msg);
                }
                else
                {
                  if(touch)
                  {
                    std::cout<<"DEPTH : grasping detected. changes\n";
                    std::string s = std::to_string(c);
                    std::string name_state = "/home/altair/interbotix_ws/src/depth_perception/states/state_"+s+".jpg";
                    cv::imwrite(name_state, final_image);
                    std_msgs::String msg_state;
                    msg_state.data = name_state;
                    touch = false;
                    //pub_name_state.publish(msg_state);
                    std_msgs::Bool msg;
                    msg.data = true;
                    pub_grasping.publish(msg);
                    setup_grasping = true;
                    pub_pause.publish(msg);
                    pub_pause_dft.publish(msg);
                    //pub_activate_depth.publish(msg);
                  }
                  else
                  {
                    std::cout<<"DEPTH : no changes\n";
                    std::string s = std::to_string(c);
                    std::string name_state = "/home/altair/interbotix_ws/src/depth_perception/states/state_"+s+".jpg";
                    cv::imwrite(name_state, final_image);
                    std_msgs::Bool msg_f;
                    msg_f.data = false;
                    //pub_success.publish(msg_f);
                    std_msgs::Bool msg;
                    msg.data = true;
                    pub_retry.publish(msg);
                    ros::Duration(0.5).sleep();
                    pub_reset_detector.publish(msg);
                    pub_activate_detector.publish(msg);
                    grasping = true;
                    //pub_ready.publish(msg);
                  }
                }
              }
              else
              {
                std::cout<<"out of border\n";
                first = true;
                out_boundary = true;
                std_msgs::Bool msg;
                msg.data = true;
                //pub_success.publish(msg);
                //pub_activate_detector.publish(msg);
                pub_reset_detector.publish(msg);
                pub_activate_depth.publish(msg);
              }
              start = false;
            }
            else
            {
              std::cout<<"first time\n";
              rmStates();
              int c = getFilesCount();
              std::string name_state;
              if(c > 0)
              {
                std::string s = std::to_string(c);
                name_state = "/home/altair/interbotix_ws/src/depth_perception/states/state_"+s+".jpg";
              }
              else
              {
                name_state = "/home/altair/interbotix_ws/src/depth_perception/states/state_0.jpg";
              }
              cv::imwrite(name_state, final_image);
              std_msgs::Bool msg;
              msg.data = true;
              //pub_reset_detector.publish(msg);
              //pub_activate_detector.publish(msg);
              //pub_new_state.publish(msg);
              /*if(out_boundary)
              {
                pub_activate_detector.publish(msg);
                out_boundary = false;
              }*/

              //ros::Duration(1.5).sleep();
              msg.data = false;
              //pub_new_state.publish(msg);
              first = false;
              start = false;
              msg.data = true;
              pub_ready.publish(msg);
            }
          }
        }
        count++;
      
      }
      //cv::imshow(OPENCV_WINDOW,fil);
      //cv::waitKey(1);
    }

    void findCenter(cv::Mat img)
    {
      cv::Mat thr; 
      cv::Mat gray;
      // convert image to grayscale
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY );
      // convert grayscale to binary image
      cv::threshold(gray, thr, 100,255, cv::THRESH_BINARY);
      // find moments of the image
      cv::Moments m = cv::moments(thr,true);
      cv::Point p(m.m10/m.m00, m.m01/m.m00);
      // publish center
      geometry_msgs::Point c;
      c.x = static_cast<double>(p.x);
      c.y = static_cast<double>(p.y);
      pub_center.publish(c);
      
    }

    cv::Mat enhanceDepth(cv::Mat img, float thr)
    {
      cv::Mat tmp;
      cv::Mat d;
      cv::Mat color_img;
      cv::cvtColor(img,tmp,cv::COLOR_RGB2GRAY);
      tmp.convertTo(d, CV_32FC1, 1/255.0);
      for(int i = 0; i < d.rows; i++)
      {
          for(int j = 0; j < d.cols; j++)
          {
            float pix = d.at<float>(i,j);
            if(pix > thr)
            {
                d.at<float>(i,j) = pix * 100;
            }
            else
            {
                d.at<float>(i,j) = 0.0;
            }
          }
      }
      cv::cvtColor(d,color_img,cv::COLOR_GRAY2RGB);
      color_img.convertTo(color_img, CV_8U, 255.0);
      
      return color_img;
    }

    cv::Mat filterDepthSample(cv::Mat img_depth, cv::Mat img_color)
    {
      cv::Mat mask_filter = cv::Mat(s_reduce_h, s_reduce_w, CV_8U,cv::Scalar(std::numeric_limits<float>::min()));
      cv::Mat filtered_img;
      cv::Point tl = detectObject(img_depth);
      cv::circle(mask_filter, tl,14, cv::Scalar(255, 255, 255), -1);
      img_color.copyTo(filtered_img,mask_filter);

      return filtered_img;
    }

    void rmStates()
    {
      int c = getFilesCount();
      c++;
      std::string n = "/home/altair/interbotix_ws/src/depth_perception/states/";
      for(int i = 0; i< c;i++)
      {
        std::string s = n + "state_" + to_string(i) + ".jpg";
        const char * f = s.c_str();
        std::remove(f);
      }
    }

    bool detectBoundaries(cv::Mat img)
    {
      cv::Mat gray_test;
      cv::Mat depth;
      cv::Mat detect;
      cv::Mat resized;
      bool tmp = false;
      cv::cvtColor(img,gray_test,cv::COLOR_RGB2GRAY);
      gray_test.convertTo(depth, CV_32F, 1/255.0);
      depth.copyTo(detect,mask);
      tmp = detectCluster(detect);
      if(tmp)
      {
        //send outcome before reset
        std_msgs::Bool msg;
        msg.data = true;
        //pub_success.publish(msg);
        pub_activate_detector.publish(msg);
        bool answer = false;
        std::chrono::seconds duration(3);
        std::future<bool> future = std::async(getAnswer);
        while(!answer)
        {
          if (future.wait_for(duration) == std::future_status::ready)
          {
            answer = future.get();
          }
          std::cout<<"waiting for user input... \n";
          system("aplay /home/altair/interbotix_ws/bip.wav");
        }
        std::cout<<"reset state \n";
       }
       return tmp;
    }

    string type2str(int type) {
      string r;

      uchar depth = type & CV_MAT_DEPTH_MASK;
      uchar chans = 1 + (type >> CV_CN_SHIFT);

      switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
      }

      r += "C";
      r += (chans+'0');

      return r;
    }

    cv::Point detectObject(cv::Mat img)
    {
      cv::Point tl;
      int sum = 0;
      int best = 0;
      bool found = false;
      for(int i = 0; i < img.rows; i++)
      {
        for(int j = 0; j < img.cols; j++)
        {
          int k = 0;
          int l = 0;
          float t = img.at<float>(i,j); 
          //std::cout<<t<<" ";
          if(img.at<float>(i,j) > 0.01)
          {
            sum = 0;
            while(k < 6 && sum < 32 && !found)
            {
              while(l < 6 && sum < 32 && !found)
              {
                if(img.at<float>(i+k,j+l) > 0.01)
                {
                  
                  sum++;
                  
                }
                l++;
              }
              l = 0;
              k++;
            }
            //std::cout<<sum<<"\n";
          }
          if(sum >= 32 && !found)
          {
            tl.x = j+4;
            tl.y = i+4;   
            found = true;
          }
        }
        //std::cout<<"\n";
      }

      return tl;
    }

    bool detectCluster(cv::Mat img)
    {
      bool result = false;
      int sum = 0;
      int i = 0;
      int j = 0;
      while(i < img.rows && !result)
      {
        while(j < img.cols && !result)
        {
          float t = img.at<float>(i,j);
          int k = 0;
          int l = 0;
          if(img.at<float>(i,j) > threshold_depth)
          {
            sum = 0;
            while(k < 4 && sum < 15)
            {
              while(l < 4 && sum < 15)
              {
                if(img.at<float>(i+k,j+l) > threshold_depth)
                {
                  sum++;
                }
                l++;
              }
              l = 0;
              k++;
            }
          }
          if(sum >= 12)
          {
            result = true;
            
          }
          j++;
        }
        j = 0;
        i++;
      }
      return result;
    }

    bool stateChanged(cv::Mat img2, int file_nb)
    {
      cv::Mat im2_gray;
      cv::Mat im1_gray;
      bool suc = false;
      file_nb = file_nb - 1;
      std::string s = std::to_string(file_nb);
      std::string name_state = "/home/altair/interbotix_ws/src/depth_perception/states/state_"+s+".jpg";
      cv::Mat img1 = imread(name_state, cv::IMREAD_COLOR);
      cv::cvtColor(img1,im1_gray,cv::COLOR_RGB2GRAY);
      cv::cvtColor(img2,im2_gray,cv::COLOR_RGB2GRAY);
      cv::Mat tmp;
      cv::subtract(im2_gray,im1_gray,tmp);
      int tot = 0;
      for(int i = 0; i < tmp.rows; i++)
      {
        for(int j = 0; j < tmp.cols; j++)
        {
          int pix = static_cast<int>(tmp.at<uchar>(i,j));
          //std::cout<<"pix value : "<<pix<<"\n";
          if(pix > 12)
          {
            tot = tot + 1;
          }
        }
      }
      std::cout<<"total change : "<<tot<<"\n";
      if(tot > threshold_change)
      {
        suc = true;
      }
      //cv::imwrite("/home/altair/interbotix_ws/src/depth_perception/states/sub.jpg", res);
      return suc;
    }

    int getFilesCount()
    {
      DIR *dp;
      int i = 0;
      struct dirent *ep;     
      dp = opendir ("/home/altair/interbotix_ws/src/depth_perception/states/");

      if (dp != NULL)
      {
        while (ep = readdir (dp))
          i++;

        (void) closedir (dp);
      }
      else
      {
        std::cout<<"no directory to save state\n";
      }
      i = i - 2;
      //std::cout<<"number of files : "<<i;

      return i;
    }

    void print4x4Matrix (const Eigen::Matrix4d & matrix)
    {
      printf ("Rotation matrix : \n");
      printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
      printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
      printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
      printf ("Translation vector :\n");
      printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
    }


};


int main(int argc, char** argv)
{
  ros::init(argc, argv, "depth_perceptions");
  DepthImage di;
  ros::spin();

  return 0;
}