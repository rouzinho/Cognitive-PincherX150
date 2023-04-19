#include <ros/ros.h>
// PCL specific includes
#include <tf2_ros/buffer.h>
#include <tf2/transform_datatypes.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>
#include <sensor_msgs/PointCloud2.h>
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
    geometry_msgs::TransformStamped transformStamped;
    image_transport::Publisher pub_state;
    ros::Publisher pub_retry;
    ros::Publisher pub_new_state;
    ros::Publisher pub_activate_detector;
    ros::Publisher pub_reset;
    ros::Publisher pub_reset_detector;
    ros::Publisher pub_name_state;
    ros::Publisher pub_success;
    ros::Publisher pub_ready;
    ros::Publisher pub_ready_robot;
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
    float threshold_depth;
    bool first;
    cv::Mat cv_image;
    cv::Mat res;
    cv::Mat fil;
    cv::Mat cv_nf;
    int count;
    int threshold;
    int size_neural_field;
    bool start;
    int threshold_change;
    cv::Mat final_image;
    cv::Mat mask;
    cv::Mat open_state;
    cv::Mat tmp_mask;
    bool reactivate;
    bool init_values;
    float gain;
    int s_y;
    bool first_gen;

  public:
    DepthImage():
    tfListener(tfBuffer),
    it_(nh_)
    {
      sub_point_cloud = nh_.subscribe("/pc_filter/pointcloud/filtered", 1, &DepthImage::pointCloudCb,this);
      sub_point_cloud_object = nh_.subscribe("/pc_filter/pointcloud/objects", 1, &DepthImage::pointCloudObjectCb,this);
      sub_activate = nh_.subscribe("/depth_perception/activate", 1, &DepthImage::activateCb,this);
      pub_state = it_.advertise("/depth_perception/state", 1);
      pub_retry = nh_.advertise<std_msgs::Bool>("/depth_perception/retry",1);
      pub_new_state = nh_.advertise<std_msgs::Bool>("/depth_perception/new_state",1);
      pub_activate_detector = nh_.advertise<std_msgs::Bool>("/outcome_detector/activate",1);
      pub_reset = nh_.advertise<std_msgs::Bool>("/depth_perception/activate",1);
      pub_reset_detector = nh_.advertise<std_msgs::Bool>("/outcome_detector/reset",1);
      pub_name_state = nh_.advertise<std_msgs::String>("/depth_perception/name_state",1);
      pub_success = nh_.advertise<std_msgs::Bool>("/depth_perception/sample_success",1);
      pub_ready = nh_.advertise<std_msgs::Bool>("/depth_perception/ready",1);
      pub_ready_robot = nh_.advertise<std_msgs::Bool>("/motion_pincher/ready",1);
      tf_in = false;
      crop_max_x = -157.78;
      crop_max_y = 463.131;
      crop_min_x = -593.047;
      crop_min_y = -430.127;
      crop_min_z = -35.7793;
      crop_max_z = 0;
      size_neural_field = 100;
      threshold_depth = 0.05;
      first = true;
      s_y = 1480;
      cv_image = cv::Mat(720, s_y, CV_32F,cv::Scalar(std::numeric_limits<float>::min()));
      fil = cv::Mat(720, s_y, CV_8U,cv::Scalar(std::numeric_limits<float>::min()));
      final_image = cv::Mat(128, 128, CV_8U,cv::Scalar(std::numeric_limits<float>::min()));
      //cv_nf = cv::Mat(50, 50, CV_32FC1,cv::Scalar(std::numeric_limits<float>::min()));
      //cv_image = cv::Mat(1024, 1024, CV_32FC1,cv::Scalar(0));
      cv::Mat m = cv::imread("/home/altair/interbotix_ws/src/depth_perception/mask/mask_border.jpg");
      cv::cvtColor(m,tmp_mask,cv::COLOR_RGB2GRAY);
      cv::resize(tmp_mask, mask, cv::Size(128, 128), cv::INTER_LANCZOS4);
      open_state = cv::imread("/home/altair/interbotix_ws/src/depth_perception/states/state_0.jpg");
      count = 0;
      threshold = 40;
      start = true;
      threshold_change = 10;
      reactivate = false;
      init_values = false;
      first_gen = true;
    }
    ~DepthImage()
    {
    }

    void pointCloudCb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
      if(!init_values)
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
        getExtremeValues(cloud_transformed);
        //genDepthFromPcl(cloud_transformed);
        init_values = true;
      }

    }

    void pointCloudObjectCb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
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
      genDepthFromPcl(cloud_transformed);
      //std::string name_state = "/home/altair/interbotix_ws/src/depth_perception/states/state_48.jpg";
      //cv::Mat img1 = imread(name_state, cv::IMREAD_COLOR);
      //stateChanged(img1,48);
    }

    void activateCb(const std_msgs::BoolConstPtr& msg)
    {
      if(msg->data == true)
      {
        start = true;
        count = 0;
        //s_y = static_cast<int>(720 * gain);
        cv_image = cv::Mat(720, s_y, CV_32F,cv::Scalar(std::numeric_limits<float>::min()));
        fil = cv::Mat(720, s_y, CV_8U,cv::Scalar(std::numeric_limits<float>::min()));
        final_image = cv::Mat(128, 128, CV_8U,cv::Scalar(std::numeric_limits<float>::min()));
      }
      else
      {
        start = false;
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
            pz = cloud->points[i].z *1000.0;
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
      crop_min_x = min_x;
      crop_min_y = min_y;
      crop_max_x = max_x;
      crop_max_y = max_y;
      crop_min_z = min_z;
      crop_max_z = max_z;
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
      
      s_y = static_cast<int>(720 * gain);
      cv_image = cv::Mat(720, s_y, CV_32F,cv::Scalar(std::numeric_limits<float>::min()));
      std::cout<<"max x : "<<max_x<<"\n";
      std::cout<<"min x : "<<min_x<<"\n";
      std::cout<<"max y : "<<max_y<<"\n";
      std::cout<<"min y : "<<min_y<<"\n";
      std::cout<<"max z : "<<max_z<<"\n";
      std::cout<<"min z : "<<min_z<<"\n";
      std::cout<<"sy : "<<s_y<<"\n";
    }

    cv::Mat fillDepthMapBlanks(cv::Mat img)
    {
      int k;
      int l;
      bool found = false;
      bool black = false;
      bool again = true;
      float angle = 0;
      bool rot = true;
      while(again == true)
      {
        for(int i = 0; i < img.rows;i++)
        {
          for(int j = 0; j < img.cols;j++)
          {
            if(i > 10 && j > 10)
            {
              if(img.at<float>(i,j) > 0 && img.at<float>(i,j) < 2)
              {
                if(img.at<float>(i,j+1) > 0)
                {
                  if(img.at<float>(i,j+2) > 0 && img.at<float>(i,j+2) < 2)
                  {
                    k = i;
                    l = j+1;
                    black = true;
                    
                  }
                }

              }
              if(img.at<float>(i,j) > 0 && img.at<float>(i,j) < 2 && black == false)
              {
                if(img.at<float>(i+1,j) > 0)
                {
                  if(img.at<float>(i+2,j) > 0 && img.at<float>(i+2,j) < 2)
                  {
                    k = i+1;
                    l = j;
                    black = true;
                    
                  }
                }
              }
              if(img.at<float>(i,j) > 0 && img.at<float>(i,j) < 2 && black == false)
              {
                if(img.at<float>(i+1,j+1) > 0)
                {
                  if(img.at<float>(i+2,j+2) > 0 && img.at<float>(i+2,j+2) < 2)
                  {
                    k = i+1;
                    l = j+1;
                    black = true;
                    
                  }
                }
              }
              
              
              if(black == true)
              {
                //std::cout<<" k : "<<i<<" l : "<<j<<"\n";
                if(img.at<float>(k,l-1) > 0 && img.at<float>(k,l+1) > 0)
                {
                  float av = (img.at<float>(k,l-1) + img.at<float>(k,l+1)) / 2;
                  img.at<float>(k,l) = av;
                  found = true;
                }
                if(img.at<float>(k-1,l) > 0 && img.at<float>(k+1,l) > 0 && found == false)
                {
                  float av = (img.at<float>(k-1,l) + img.at<float>(k+1,l)) / 2;
                  img.at<float>(k,l) = av;
                  found = true;
                }
                if(img.at<float>(k-1,l-1) > 0 && img.at<float>(k+1,l+1) > 0 && found == false)
                {
                  float av = (img.at<float>(k-1,l-1) + img.at<float>(k+1,l+1)) / 2;
                  img.at<float>(k,l) = av;
                  found = true;
                }
                if(img.at<float>(k-1,l+1) > 0 && img.at<float>(k+1,l-1) > 0 && found == false)
                {
                  float av = (img.at<float>(k-1,l+1) + img.at<float>(k+1,l-1)) / 2;
                  img.at<float>(k,l) = av;
                  found = true;
                }
                found = false;
                black = false;
              }
            }
          }
        }
        if(angle <= 350)
        {
          angle = angle + 90;
          cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
        }
        else
        {
          again = false;
          //cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
        }
      }
      return img;
    }

    void genDepthFromPcl(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
      if(first_gen)
      {
        cv_image = cv::Mat(720, s_y, CV_32F,cv::Scalar(std::numeric_limits<float>::min()));
        first_gen = false;
      }
      if(!first_gen)
      {
        std::cout<<"in !\n";
        cv::Mat rot;
        const float bad_point = std::numeric_limits<float>::quiet_NaN();
        int pixel_pos_x;
        int pixel_pos_y;
        float pixel_pos_z;
        float px;
        float py;
        float pz;
        float test;
        /*double ax = (static_cast<double>(720))/(crop_max_x-crop_min_x); //1024 image width
        double bx = 0 - (ax*crop_min_x);
        double ay = (static_cast<double>)(720)/(crop_max_y-crop_min_y); //1024 image height
        double by = 0 - (ay*crop_min_y);*/
        double ax = (static_cast<double>(720))/(std::abs(crop_min_x)-std::abs(crop_max_x)); //1024 image width
        double bx = 0 - (ax*crop_min_x);
        double ay = (static_cast<double>(s_y))/(crop_max_y-crop_min_y); //1024 image height
        double by = 0 - (ay*crop_min_y);
        double az = (static_cast<double>(1000))/(crop_max_z-crop_min_z);
        double bz = 0 - (az*crop_min_z);
        if(start == true)
        {
          if(count < threshold)
          {
            for (int i=0; i< cloud->points.size();i++)
            {
              px = cloud->points[i].x * 1000.0*-1;
              py = cloud->points[i].y * 1000.0*-1;//revert image because it's upside down for display
              pz = cloud->points[i].z * 1000.0;
              std::cout<<pz<<"\n";
              pixel_pos_x = (int) (ax * px + bx);
              pixel_pos_y = (int) (ay * py + by);
              pixel_pos_z = (az * pz + bz);
              pixel_pos_z = pixel_pos_z/1000.0;
              if(px > crop_max_x || px < crop_min_x || py > crop_max_y || py < crop_min_y)
              {
                cloud->points[i].z = bad_point;
                cv_image.at<float>(pixel_pos_x,pixel_pos_y) = 0.0;
              }
              if (cloud->points[i].z == cloud->points[i].z)
              {
                {
                  cv_image.at<float>(pixel_pos_x,pixel_pos_y) = pixel_pos_z;
                }    
              }
            }
          }
          cv::cvtColor(cv_image,res,cv::COLOR_GRAY2RGB);
          res.convertTo(res, CV_8U, 255.0);
          cv::imwrite("/home/altair/interbotix_ws/src/depth_perception/states/calibration_mask.jpg", res);
          /*else
          {
            cv::Mat cv_nf;// = cv::Mat(100, 100, CV_32FC1,cv::Scalar(std::numeric_limits<float>::min()));
            cv::Mat fil_nf;// = cv::Mat(1024, 1024, CV_8U,cv::Scalar(std::numeric_limits<float>::min()));
            cv::Mat r_nf;
            bool change;
            cv_image = fillDepthMapBlanks(cv_image);
            cv::rotate(cv_image, rot, cv::ROTATE_90_COUNTERCLOCKWISE);
            //convert to gray
            cv::cvtColor(rot,res,cv::COLOR_GRAY2RGB);
            res.convertTo(res, CV_8U, 255.0);
            cv::medianBlur(res,fil,(9,9));
            //for dnf
            cv::cvtColor(fil,r_nf,cv::COLOR_RGB2GRAY);
            cv::resize(r_nf, fil_nf, cv::Size(200, 200), cv::INTER_LANCZOS4);
            fil_nf.convertTo(cv_nf, CV_32FC1, 1/255.0);
            //cv::imwrite("/home/altair/interbotix_ws/src/depth_perception/states/dnf.jpg", fil_nf);
            cv::resize(fil, final_image, cv::Size(128, 128), cv::INTER_LANCZOS4);
            sensor_msgs::ImagePtr dobject_nf = cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_32FC1, cv_nf).toImageMsg();
            pub_state.publish(dobject_nf);
            ros::Duration(3.5).sleep();
            int c = getFilesCount();
            if(first == false)
            {
              //check if object isn't out of robot's reach
              bool border = detectBoundaries(final_image);
              if(!border)
              {
                change = stateChanged(final_image,c);
                if(change)
                {
                  std::cout<<"changes !\n";
                  std::string s = std::to_string(c);
                  std::string name_state = "/home/altair/interbotix_ws/src/depth_perception/states/state_"+s+".jpg";
                  cv::imwrite(name_state, final_image);
                  std_msgs::String msg_state;
                  msg_state.data = name_state;
                  pub_name_state.publish(msg_state);
                  std::cout<<"RESET DNF\n";
                  std_msgs::Bool msg;
                  msg.data = true;
                  pub_success.publish(msg);
                  pub_new_state.publish(msg);
                  ros::Duration(4.5).sleep();
                  msg.data = false;
                  pub_new_state.publish(msg);
                  reactivate = false;
                  msg.data = true;
                  pub_ready.publish(msg);
                }
                else
                {
                  std::cout<<"no changes\n";
                  std::string s = std::to_string(c);
                  std::string name_state = "/home/altair/interbotix_ws/src/depth_perception/states/state_"+s+".jpg";
                  cv::imwrite(name_state, final_image);
                  std_msgs::Bool msg_f;
                  msg_f.data = false;
                  pub_success.publish(msg_f);
                  std_msgs::Bool msg;
                  msg.data = true;
                  pub_retry.publish(msg);
                  //ros::Duration(0.5).sleep();
                  //msg.data = false;
                  //pub_retry.publish(msg);
                  msg.data = true;
                  pub_ready.publish(msg);
                }
                std_msgs::Bool msg;
                msg.data = true;
                pub_activate_detector.publish(msg);
              }
              else
              {
                first = true;
                std_msgs::Bool msg;
                msg.data = true;
                //pub_success.publish(msg);
                //pub_activate_detector.publish(msg);
                pub_reset_detector.publish(msg);
                pub_reset.publish(msg);
              }
              start = false;
            }
            else
            {
              std::cout<<"first time\n";
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
              pub_new_state.publish(msg);
              pub_activate_detector.publish(msg);
              ros::Duration(1.5).sleep();
              msg.data = false;
              pub_new_state.publish(msg);
              first = false;
              start = false;
              msg.data = true;
              pub_ready.publish(msg);
            }
          }*/
        count++;
      }
      cv::imshow(OPENCV_WINDOW, cv_image);
      cv::waitKey(1);
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
        pub_success.publish(msg);
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