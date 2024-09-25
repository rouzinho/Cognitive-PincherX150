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
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/Marker.h>
#include <fstream>
#include <string>

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
    geometry_msgs::TransformStamped transformStamped;
    image_transport::Publisher pub_depth;
    bool tf_in;
    tf2_ros::TransformListener tfListener;
    tf2_ros::Buffer tfBuffer;
    Eigen::Matrix4d robot_frame;
    std_msgs::Header header;
    geometry_msgs::PoseStamped pose_object;
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
    cv::Mat cv_image;
    bool init_values;
    int count_init;
    float gain;
    int s_x;
    int s_y;
    bool init_params;
    int count;
    cv::Mat display;
    cv::Mat fil;
    bool saved;

  public:
    DepthImage():
    tfListener(tfBuffer),
    it_(nh_)
    {
      sub_point_cloud = nh_.subscribe("/pc_filter/pointcloud/filtered", 1, &DepthImage::pointCloudCb,this);
      sub_point_cloud_object = nh_.subscribe("/pc_filter/pointcloud/objects", 1, &DepthImage::pointCloudObjectCb,this);
      pub_depth = it_.advertise("/depth_perception/depth", 1);
      tf_in = false;
      count_init = 0;
      s_x = 620;  //620
      s_y = 840;  //840
      count = 0;
      saved = false;
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
      init_values = false;
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
          std::cout<<"crop min x : "<<crop_min_x<<"\n";
          std::cout<<"crop max x : "<<crop_max_x<<"\n";
          std::cout<<"crop min y : "<<crop_min_y<<"\n";
          std::cout<<"crop max y : "<<crop_max_y<<"\n";
          std::cout<<"crop min z : "<<crop_min_x<<"\n";
          std::cout<<"crop max z : "<<crop_max_z<<"\n";
          std::cout<<"ax : "<<ax<<"\n";
          std::cout<<"bx : "<<bx<<"\n";
          std::cout<<"ay : "<<ay<<"\n";
          std::cout<<"by : "<<by<<"\n";
          std::cout<<"az : "<<az<<"\n";
          std::cout<<"bz : "<<bz<<"\n";
          init_params = true;
        }
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
      
      pcl::PCLPointCloud2 pcl_pc2;
      pcl_conversions::toPCL(*cloud_msg, pcl_pc2);
      pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
      pcl::transformPointCloud(*temp_cloud,*cloud_transformed,robot_frame);

      genDepthFromPcl(cloud_transformed);
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
            px = cloud->points[i].x * 1000.0*-1;// 
            py = cloud->points[i].y * 1000.0*-1;//revert image because it's upside down for display
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
    }

    void writePoint(cv::Point p)
    {
      ofstream myfile ("/home/altair/interbotix_ws/depth_points.txt");
      string x = to_string(p.x);
      string y = to_string(p.y);
      myfile<<x<<" "<<y<<"\n";
      myfile.close();
    }

    void genDepthFromPcl(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
      cv_image = cv::Mat(s_x, s_y, CV_32F,cv::Scalar(std::numeric_limits<float>::min()));
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
      }
      //if(count < 60)
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
        //display = cv::Mat(s_x, s_y, CV_32FC1,cv::Scalar(std::numeric_limits<float>::min()));
        //display = cv_image(crop_img); 
        //count++;
      }
      //else
      {
        cv::Mat res;
        cv::Mat resized;
        cv::Mat fil_b;
        cv::Point p;
        cv::Point m;
        //cv::resize(display, resized, cv::Size(180, 100), cv::INTER_LANCZOS4);
        //cv::cvtColor(cv_image,res,cv::COLOR_GRAY2RGB);
        //res.convertTo(res, CV_8U, 255.0);
        //cv::medianBlur(res,fil,(5,5));
        //m = findCenter(fil);
        //sensor_msgs::ImagePtr d_object = cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_32FC1, fil_b).toImageMsg();
        //pub_depth.publish(d_object);
        //cv::circle(fil, m, 3, cv::Scalar(0,0,255  ), -1);
      }

      cv::imshow(OPENCV_WINDOW,cv_image);
      cv::waitKey(1);
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
                d.at<float>(i,j) = 1.0;//pix * 100;
            }
            else
            {
                d.at<float>(i,j) = 0.0;
            }
            //std::cout<<d.at<float>(i,j);
          }
          //std::cout<<"\n";
      }
      //cv::cvtColor(d,color_img,cv::COLOR_GRAY2RGB);
      //color_img.convertTo(color_img, CV_8U, 255.0);
      
      return d;
    }

    cv::Point findCenter(cv::Mat img)
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
      
      return p;
    }

};


int main(int argc, char** argv)
{
  ros::init(argc, argv, "depth_calibration");
  DepthImage di;
  ros::spin();

  return 0;
}