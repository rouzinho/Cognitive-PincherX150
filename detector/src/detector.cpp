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
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>
#include <pcl_ros/transforms.h>
#include <math.h>
#include <cmath>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <std_msgs/Int16.h>
#include <std_msgs/Float32.h>
#include "detector/Outcome.h"
#include "detector/State.h"
#include "cog_learning/ListObject.h"
#include <opencv2/aruco.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <depth_interface/ElementUI.h>
#include <depth_interface/InterfacePOI.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/Marker.h>
#include "detector/GetState.h"

using namespace message_filters;
using namespace std;
static const std::string OPENCV_WINDOW = "Image window";

class Detector
{
    private:
        ros::NodeHandle nh_;
        image_transport::ImageTransport it_;
        image_transport::Subscriber img_sub;
        ros::Subscriber sub_ori;
        ros::Subscriber sub_fin;
        ros::Publisher pub_tf;
        ros::Publisher pub_outcome;
        ros::Publisher pub_aruco;
        ros::Publisher pub_ready;
        ros::Publisher pub_state_object;
        ros::Publisher pub_id_object;
        ros::Publisher pub_new_state;
        ros::Subscriber sub_activate;
        ros::Subscriber sub_trigger_state;
        ros::Subscriber sub_object;
        ros::Subscriber sub_touch;
        ros::Subscriber sub_angle;
        ros::Subscriber sub_first_time;
        ros::Subscriber sub_grasping;
        ros::Subscriber sub_monitored;
        ros::ServiceServer service;
        open3d::geometry::PointCloud cloud_origin;
        open3d::geometry::PointCloud cloud_backup;
        open3d::geometry::PointCloud cloud_final;
        geometry_msgs::TransformStamped transformStamped;
        geometry_msgs::PoseStamped pose_object;
        geometry_msgs::PoseStamped first_pose;
        geometry_msgs::PoseStamped second_pose;
        cv_bridge::CvImagePtr cv_ptr;
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
        bool activate_angle;
        bool touch;
        bool mode;
        int monitored_object;
        float first_angle;
        float second_angle;
        float object_state_angle;
        bool first_time;
        bool success_sample;
        tf2::Quaternion q_vector;
        detector::State state_object;
        int id_object;
        int prev_id_object;

    public:

    Detector():
    tfListener(tfBuffer),
    it_(nh_)
    {
        //sub_ori = nh_.subscribe("/pc_filter/pointcloud/objects", 1, &Detector::callbackPointCloud, this);
        sub_activate = nh_.subscribe("/outcome_detector/activate", 1, &Detector::activateCallback,this);
        sub_trigger_state = nh_.subscribe("/outcome_detector/trigger_state", 1, &Detector::activateTrigger,this);
        sub_object = nh_.subscribe("/pc_filter/markers/objects", 1, &Detector::objectCallback,this);
        sub_touch = nh_.subscribe("/motion_pincher/touch", 1, &Detector::touchCallback,this);
        sub_angle = nh_.subscribe("/depth_interface/aruco_angle", 1, &Detector::AngleCallback,this);
        sub_first_time = nh_.subscribe("/outcome_detector/reset", 1, &Detector::ResetCallback,this);
        sub_grasping = nh_.subscribe("/outcome_detector/grasping", 1, &Detector::GraspingCallback,this);
        sub_monitored = nh_.subscribe("/outcome_detector/monitored_object", 1, &Detector::intCallback,this);
        img_sub = it_.subscribe("/rgb/image_raw", 1,&Detector::RgbCallback, this);
        service = nh_.advertiseService("get_object_state",&Detector::getState,this);
        pub_tf = nh_.advertise<sensor_msgs::PointCloud2>("/outcome_detector/cloud_icp",1);
        pub_outcome = nh_.advertise<detector::Outcome>("/outcome_detector/outcome",1);
        pub_aruco = nh_.advertise<depth_interface::InterfacePOI>("/outcome_detector/aruco_corners",1);
        pub_ready = nh_.advertise<std_msgs::Bool>("/outcome_detector/ready",1);
        pub_state_object = nh_.advertise<detector::State>("/outcome_detector/state",1);
        pub_id_object = nh_.advertise<cog_learning::ListObject>("/outcome_detector/id_object",1);
        pub_new_state = nh_.advertise<std_msgs::Bool>("/depth_perception/new_state",1);
        pose_object.pose.position.x = 0.0;
        pose_object.pose.position.y = 0.0;
        pose_object.pose.position.z = 0.0;
        pose_object.pose.orientation.x = 0.0;
        pose_object.pose.orientation.y = 0.0;
        pose_object.pose.orientation.z = 0.0;
        pose_object.pose.orientation.w = 1.0;
        cloud_origin.Clear();
        id_object = -1;
        prev_id_object = -2;
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
        mode = false;
        first_angle = 0.0;
        second_angle = 0.0;
        first_time = true;
        cv::namedWindow(OPENCV_WINDOW,cv::WINDOW_NORMAL);
    }

    virtual ~Detector()
    {
        cv::destroyWindow(OPENCV_WINDOW);
    }

    bool getState(detector::GetState::Request &req, detector::GetState::Response &res)
    {
        res.state.state_x = pose_object.pose.position.x;
        res.state.state_y = pose_object.pose.position.y;
        res.state.state_angle = object_state_angle;
        
        return true;
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
            setOutcomeICP(first_pose,second_pose,r);
        }
        pub_tf.publish(msg_open);
    }

    void activateCallback(const std_msgs::BoolConstPtr& msg)
    {
        if(msg->data == true)
        {
            activate_angle = true;
        }
    }

    void activateTrigger(const std_msgs::BoolConstPtr& msg)
    {
        if(msg->data)
        {
            state_object.state_x = pose_object.pose.position.x;
            state_object.state_y = pose_object.pose.position.y;
            state_object.state_angle = object_state_angle;
            pub_state_object.publish(state_object);
        }
    }

    void touchCallback(const std_msgs::BoolConstPtr& msg)
    {
        touch = msg->data;
    }

    void intCallback(const std_msgs::Int16ConstPtr& msg)
    {
        monitored_object = msg->data;
    }

    void AngleCallback(const std_msgs::Float32ConstPtr& msg)
    {
        object_state_angle = msg->data;
        if(first_time && activate_angle)
        {
            cout<<"Recording first angle\n";
            std::cout<<object_state_angle<<"\n";
            first_angle = msg->data;
            first_pose = pose_object;
            activate_angle = false;
            first_time = false;
            std_msgs::Bool tmp;
            tmp.data = true;
            //std::cout<<"sending activation\n";
            pub_ready.publish(tmp);
        }
        if(!first_time && activate_angle)
        {
            cout<<"Recording second angle\n";
            std::cout<<object_state_angle<<"\n";
            second_angle = msg->data;
            second_pose = pose_object;
            setOutcomeAngle(first_pose,second_pose,first_angle,second_angle);
            activate_angle = false;
            first_angle = second_angle;
            first_pose = second_pose;
        }
    }

    void ResetCallback(const std_msgs::BoolConstPtr& msg)
    {
        first_time = msg->data;
        count = 0;
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
        else
        {
            listenTransform();
        }
        //state_object.state_x = pose_object.pose.position.x;
        //state_object.state_y = pose_object.pose.position.y;
        //state_object.state_angle = object_state_angle;
        //pub_state_object.publish(state_object);
        tf2::Quaternion q_orig(0,0,0,1);
        tf2::Quaternion q_rot;
        geometry_msgs::Point new_vec;
        geometry_msgs::Point vec_ori;
        vec_ori.x = pose_object.pose.position.x;
        vec_ori.y = pose_object.pose.position.y;
        vec_ori.x = pose_object.pose.position.x + vec_ori.x;
        vec_ori.y = pose_object.pose.position.y + vec_ori.y;
        float dot_prod = (vec_ori.x*0.1) + (vec_ori.y*0);
        float det = (vec_ori.x*0) + (vec_ori.y*0.1);
        float ang = atan2(det,dot_prod);

        q_rot.setRPY(0,0,ang);
        q_vector = q_rot*q_orig;
        q_vector.normalize();
        //tf2::convert(q_vector, p.pose.orientation);
    }

    //sending 2D corners to be transform in 3D space
    void RgbCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_ptr = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::BGR8);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners, rejectedCandidates;
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();;
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        cv::aruco::detectMarkers(cv_ptr->image, dictionary, corners, ids);
        //cv::aruco::drawDetectedMarkers(cv_ptr->image, corners, ids);
        depth_interface::InterfacePOI pts;
        cog_learning::ListObject lo;
        if(ids.size() == 1)
        {
            //id_object = ids[0];
            lo.list_object.push_back(ids[0]);
            for(int i = 0; i < ids.size();  i++)
            {
                if(corners[ids[i]].size() == 4)
                {
                    for(int j = 0; j < corners[ids[i]].size(); j++)
                    {
                        depth_interface::ElementUI pt;
                        pt.elem.x = corners[ids[i]][j].x;
                        pt.elem.y = corners[ids[i]][j].y;
                        pts.poi.push_back(pt);
                    }
                    pub_aruco.publish(pts);
                }
            }
            
        }
        if(ids.size() > 1)
        {
            id_object = monitored_object;
            for(int i = 0; i < ids.size(); i++)
            {
                lo.list_object.push_back(ids[i]);
                if(ids[0] == monitored_object)
                {
                    if(corners[ids[i]].size() == 4)
                    {
                        for(int j = 0; j < corners[ids[i]].size(); j++)
                        {
                            depth_interface::ElementUI pt;
                            pt.elem.x = corners[ids[i]][j].x;
                            pt.elem.y = corners[ids[i]][j].y;
                            pts.poi.push_back(pt);
                        }
                        pub_aruco.publish(pts);
                    }
                }
            }
        }
        if(ids.size() == 0)
        {
            id_object = -1;
        }
        //cv::imshow("out", cv_ptr->image);
        //cv::waitKey(1);
        pub_id_object.publish(lo);
    }

    void GraspingCallback(const std_msgs::BoolConstPtr& msg)
    {
        detector::Outcome tmp;
        tmp.x = 0;
        tmp.y = 0;
        tmp.angle = 0;
        tmp.touch = 1.0;
        pub_outcome.publish(tmp);
        activate_angle = false;
        std_msgs::Bool b;
        b.data = true;
        pub_new_state.publish(b);
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

    void setOutcomeICP(geometry_msgs::PoseStamped first, geometry_msgs::PoseStamped second, Eigen::Vector3f rot)
    {
        detector::Outcome res;
        float t_x = second.pose.position.x - first.pose.position.x;
        float t_y = second.pose.position.y - first.pose.position.y;
        res.x = t_x;
        res.y = t_y;
        res.angle = rot[2];
        if(touch == true)
        {
            res.touch = 1.0;
        }
        else
        {
            res.touch = 0.0;
        }
        //writeDataset(res);
        pub_outcome.publish(res);
    }

    void setOutcomeAngle(geometry_msgs::PoseStamped first, geometry_msgs::PoseStamped second, float first_ang, float sec_ang)
    {
        detector::Outcome res;
        float t_x = second.pose.position.x - first.pose.position.x;
        float t_y = second.pose.position.y - first.pose.position.y;
        float diff;
        if(touch == true)
        {
            res.touch = 1.0;
        }
        else
        {
            res.touch = 0.0;
        }
        first_ang = first_ang + 180.0;
        sec_ang = sec_ang + 180.0;
        diff = sec_ang - first_ang;
        if(std::abs(diff) > 260)
        {
            if(first_ang > sec_ang)
            {
                float tmp = 360 - first_ang;
                diff = tmp + sec_ang;
            }
            else
            {
                float tmp = 360 - sec_ang;
                diff = - (tmp + first_ang);
            }
        }
        geometry_msgs::Point vec_ref_object = findVectorTransform(first, t_x,t_y);
        res.x = vec_ref_object.x;
        res.y = vec_ref_object.y;
        //std::cout<<"first angle : "<<first_ang<<"\n";
        //std::cout<<"second angle : "<<sec_ang<<"\n";
        std::cout<<"Angle difference : "<<diff<<"\n";
        std::cout<<"vector x : "<<res.x<<"\n";
        std::cout<<"vector y : "<<res.y<<"\n";
        res.angle = diff;
        std_msgs::Bool tmp;
        tmp.data = true;
        pub_outcome.publish(res);
        pub_ready.publish(tmp);
    }

    geometry_msgs::Point findVectorTransform(geometry_msgs::PoseStamped first_pose, float tx, float ty)
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