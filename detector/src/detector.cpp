#include <open3d/Open3D.h>
#include <open3d/pipelines/registration/GlobalOptimization.h>
#include <open3d/pipelines/registration/PoseGraph.h>
#include <open3d/pipelines/registration/FastGlobalRegistration.h>
#include <open3d/pipelines/registration/ColoredICP.h>
#include <Eigen/Dense>
// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
// open3d_conversions
#include "open3d_conversions/open3d_conversions.h"
#include <tf2_ros/buffer.h>
#include <tf2/transform_datatypes.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>
#include <pcl_ros/transforms.h>
#include <math.h>
#include <std_msgs/Bool.h>
#include "detector/Outcome.h"

using namespace message_filters;
using namespace std;

class Detector
{
    private:
        ros::NodeHandle nh_;
        ros::Subscriber sub_ori;
        ros::Subscriber sub_fin;
        ros::Publisher pub_tf;
        ros::Publisher pub_outcome;
        ros::Subscriber sub_activate;
        ros::Subscriber sub_object;
        ros::Subscriber sub_touch;
        open3d::geometry::PointCloud cloud_origin;
        open3d::geometry::PointCloud cloud_backup;
        open3d::geometry::PointCloud cloud_final;
        geometry_msgs::TransformStamped transformStamped;
        geometry_msgs::PoseStamped pose_object;
        geometry_msgs::PoseStamped first_pose;
        geometry_msgs::PoseStamped second_pose;
        bool tf_in;
        tf2_ros::TransformListener tfListener;
        tf2_ros::Buffer tfBuffer;
        Eigen::Matrix4d robot_frame;
        int count;
        bool first;
        bool second;
        bool ori_one;
        bool fin_one;
        float roll;
        float pitch;
        float yaw;
        open3d::pipelines::registration::RegistrationResult icp_coarse;
        open3d::pipelines::registration::RegistrationResult icp_fine;
        double rmse;
        Eigen::Matrix4d_u transform;
        sensor_msgs::PointCloud2 msg_open;
        sensor_msgs::PointCloud2 cloud_tf;
        sensor_msgs::PointCloud2 cloud_pcl_backup;
        bool activate;
        bool touch;

    public:

    Detector():
    tfListener(tfBuffer)
    {
        sub_ori = nh_.subscribe("/pc_filter/pointcloud/objects", 1, &Detector::callbackPointCloud, this);
        sub_activate = nh_.subscribe("/outcome_detector/activate", 1, &Detector::activateCallback,this);
        sub_object = nh_.subscribe("/pc_filter/markers/objects", 1, &Detector::objectCallback,this);
        sub_touch = nh_.subscribe("/outcome_detector/touch", 1, &Detector::touchCallback,this);
        pub_tf = nh_.advertise<sensor_msgs::PointCloud2>("/outcome_detector/cloud_icp",1);
        pub_outcome = nh_.advertise<detector::Outcome>("/outcome_detector/outcome",1);
        pose_object.pose.position.x = 0.0;
        pose_object.pose.position.y = 0.0;
        pose_object.pose.position.z = 0.0;
        pose_object.pose.orientation.x = 0.0;
        pose_object.pose.orientation.y = 0.0;
        pose_object.pose.orientation.z = 0.0;
        pose_object.pose.orientation.w = 1.0;
        cloud_origin.Clear();
        count = 0;
        first = true;
        second = false;
        rmse = 1.0;
        transform = Eigen::Matrix4d_u(4,4);
        tf_in = false;
        ori_one = false;
        fin_one = false;
        activate = false;
        touch = false;
    }

    virtual ~Detector()
    {
    }

    void callbackPointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_ori)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        if(!tf_in)
        {
            listenTransform();
        }
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*cloud_ori, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
        pcl::transformPointCloud(*temp_cloud,*cloud_transformed,robot_frame);
        pcl::toROSMsg(*cloud_transformed,cloud_tf);
        sensor_msgs::PointCloud2ConstPtr cl(new sensor_msgs::PointCloud2(cloud_tf));
        cloud_backup.Clear();
        open3d_conversions::rosToOpen3d(cl, cloud_backup);
        if(count == 1 && activate == true)
        {
            cout<<"Recording first pointcloud\n";
            cloud_origin.Clear();
            cout<<"size cloud : "<<cloud_transformed->size()<<"\n";
            open3d_conversions::rosToOpen3d(cl, cloud_origin);
            first_pose = pose_object;
            activate = false;
        }
        if(count == 2 && activate == true)
        {
            cout<<"Recording second pointcloud\n";
            cloud_final.Clear();
            cout<<"size cloud : "<<cloud_transformed->size()<<"\n";
            open3d_conversions::rosToOpen3d(cl, cloud_final);
            second_pose = pose_object;
            activate = false;
            count = 0;
            Eigen::Vector3f r = performICP(cloud_origin,cloud_final);
            setOutcome(first_pose,second_pose,r);
        }
        pub_tf.publish(msg_open);
    }

    void activateCallback(const std_msgs::BoolConstPtr& msg)
    {
        if(msg->data == true)
        {
            count++;
            activate = true;
        }
        if(msg->data == true && touch == true)
        {
            count = 0;
            activate = false;
            detector::Outcome res;
            res.x = 0.0 - first_pose.pose.position.x;
            res.y = 0.0 - first_pose.pose.position.y;
            res.roll = 0.0;
            res.touch = 1.0;
            pub_outcome.publish(res);
        }
    }

    void touchCallback(const std_msgs::BoolConstPtr& msg)
    {
        touch = msg->data;
    }

    void objectCallback(const visualization_msgs::MarkerConstPtr& msg)
    {
        geometry_msgs::PoseStamped tmp;
        tmp.header = msg->header;
        tmp.pose.position.x = msg->pose.position.x;
        tmp.pose.position.y = msg->pose.position.y;
        tmp.pose.position.z = msg->pose.position.z;
        tmp.pose.orientation.x = 0.0;
        tmp.pose.orientation.y = 0.0;
        tmp.pose.orientation.z = 0.0;
        tmp.pose.orientation.w = 1.0;
        if(tf_in)
        {
            tf2::doTransform(tmp,pose_object,transformStamped);
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

    void setOutcome(geometry_msgs::PoseStamped first, geometry_msgs::PoseStamped second, Eigen::Vector3f rot)
    {
        detector::Outcome res;
        float t_x = second.pose.position.x - first.pose.position.x;
        float t_y = second.pose.position.y - first.pose.position.y;
        res.x = t_x;
        res.y = t_y;
        res.roll = rot[2];
        if(touch == true)
        {
            res.touch = 1.0;
        }
        else
        {
            res.touch = 0.0;
        }
        writeDataset(res);
        pub_outcome.publish(res);
    }

    Eigen::Vector3f performICP(open3d::geometry::PointCloud cloud_ori, open3d::geometry::PointCloud cloud_fin)
    {
        int stop = false;
        
        std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr = std::make_shared<open3d::geometry::PointCloud>(cloud_ori);
        std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr_fin = std::make_shared<open3d::geometry::PointCloud>(cloud_fin);
        std::shared_ptr<open3d::geometry::PointCloud> cloud_fixed = cloud_ptr;
        bool t = true;
        rmse = 1.0;
        double th = 0.3;
        double th_fine = 0.06;
        double th_ultra = 0.04;
        Eigen::Matrix3d d;
        Eigen::Vector3d ea;
        cloud_ptr->EstimateNormals();
        cloud_ptr_fin->EstimateNormals();
        Eigen::Vector3f rot(0,0,0);
        
        while(rmse > 0.006)
        {
            if(first)
            {
                icp_coarse = open3d::pipelines::registration::RegistrationICP(*cloud_ptr,*cloud_ptr_fin,th,Eigen::Matrix4d::Identity(),open3d::pipelines::registration::TransformationEstimationPointToPoint(false),open3d::pipelines::registration::ICPConvergenceCriteria(1e-6,1e-6,200));
                first = false;
                second = true;
                std::cout<<"FIRST : "<<icp_fine.fitness_<<" inlier RMSE : "<<icp_fine.inlier_rmse_<<" correspondence set size : "<<icp_fine.correspondence_set_.size()<<std::endl;
            }
            else
            {
                if(second)
                {
                    icp_fine = open3d::pipelines::registration::RegistrationICP(*cloud_ptr,*cloud_ptr_fin,th_fine,icp_coarse.transformation_,open3d::pipelines::registration::TransformationEstimationPointToPoint(),open3d::pipelines::registration::ICPConvergenceCriteria(1e-6,1e-6,200));
                    second = false;
                    std::cout<<"SECOND : "<<icp_fine.fitness_<<" inlier RMSE : "<<icp_fine.inlier_rmse_<<" correspondence set size : "<<icp_fine.correspondence_set_.size()<<std::endl;

                }
                else
                {
                    icp_fine = open3d::pipelines::registration::RegistrationICP(*cloud_ptr,*cloud_ptr_fin,th_ultra,icp_fine.transformation_,open3d::pipelines::registration::TransformationEstimationPointToPoint(),open3d::pipelines::registration::ICPConvergenceCriteria(1e-6,1e-6,200));
                    std::cout<<"FINE    fitness : "<<icp_fine.fitness_<<" inlier RMSE : "<<icp_fine.inlier_rmse_<<" correspondence set size : "<<icp_fine.correspondence_set_.size()<<std::endl;
                    rmse = icp_fine.inlier_rmse_;
                    transform = icp_coarse.transformation_ * icp_fine.transformation_;
                }
                print4x4Matrix(icp_fine.transformation_);
                
                d.array() = 0.0;
                d(0,0) = icp_fine.transformation_(0,0);
                d(0,1) = icp_fine.transformation_(0,1);
                d(0,2) = icp_fine.transformation_(0,2);
                d(1,0) = icp_fine.transformation_(1,0);
                d(1,1) = icp_fine.transformation_(1,1);
                d(1,2) = icp_fine.transformation_(1,2);
                d(2,0) = icp_fine.transformation_(2,0);
                d(2,1) = icp_fine.transformation_(2,1);
                d(2,2) = icp_fine.transformation_(2,2);
                ea = d.eulerAngles(2, 1, 0); 
                roll = (ea[0] * 180) / M_PI ;
                pitch = (ea[1] * 180) / M_PI;
                yaw = (ea[2] * 180) / M_PI;
                if(roll > 180)
                {
                    cout<<"roll over limit !\n";
                    ros::Duration(1.0).sleep();
                    cloud_ptr_fin = std::make_shared<open3d::geometry::PointCloud>(cloud_backup);
                    first = true;
                    second = true;
                    rmse = 1.0;
                    cout<<"starting over !\n";
                }
                else
                {
                    if(roll > 170 && roll < 180)
                    {
                        roll = 0.0;
                    }
                    if(roll > 95)
                    {
                        roll = 180 - roll;
                    }
                }
            }
            Eigen::Vector3d r(yaw,pitch,roll);
            rot[0] = yaw;
            rot[1] = pitch;
            rot[2] = roll;
            //cout << "to Euler angles:" << endl;
            //cout << ea << endl << endl;
            cout <<"roll : "<<roll<<"\n";
            cout <<"rmse : "<<rmse<<"\n";
        }
        std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr_final = std::make_shared<open3d::geometry::PointCloud>(cloud_ptr->Transform(icp_fine.transformation_));
        open3d_conversions::open3dToRos(*cloud_ptr_final,msg_open,"px150/base_link");
        //cout<<"icp coarse \n";
        //print4x4Matrix(icp_coarse.transformation_);
        //cout<<"icp fine \n";
        return rot;
    }

    void writeDataset(detector::Outcome sample)
    {
      std::ofstream ofile;
      ofile.open("/home/altair/interbotix_ws/src/detector/dataset/samples.txt", std::ios::app);
      ofile << sample.x <<" "<<sample.y<<" "<<sample.roll<<" "<<sample.touch<<std::endl;
      ofile.close();
    }

    /*void readDataset()
    {
      std::string line;
      std::ifstream icpfile("/home/altair/odin/src/pcl_fusion/calibration/params.txt");
      if(icpfile.is_open())
      {
        getline(icpfile,line);
        ax = std::stod(line); 
        getline(icpfile,line);
        bx = std::stod(line);
        getline(icpfile,line);
        ay = std::stod(line);
        getline(icpfile,line);
        by = std::stod(line);
        getline(icpfile,line);
        az = std::stod(line);
        getline(icpfile,line);
        bz = std::stod(line);
        icpfile.close();
      }
    }*/

    void print4x4Matrix (const Eigen::Matrix4d_u & matrix)
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
    
    ros::init(argc, argv, "open3d");
    Detector detector;
    ros::spin();

    return 0;
}