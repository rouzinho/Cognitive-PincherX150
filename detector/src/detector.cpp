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

using namespace message_filters;

class Detector
{
    private:
        ros::NodeHandle nh_;
        ros::Subscriber sub_ori;
        ros::Subscriber sub_fin;
        ros::Publisher pub_tf;
        open3d::geometry::PointCloud cloud_origin;
        open3d::geometry::PointCloud cloud_final;
        open3d::visualization::Visualizer vis;
        int count;
        bool first;
        bool second;
        open3d::pipelines::registration::RegistrationResult icp_coarse;
        open3d::pipelines::registration::RegistrationResult icp_fine;
        double rmse;
        //message_filters::Subscriber<sensor_msgs::PointCloud2> sub_ori;
        //message_filters::Subscriber<sensor_msgs::PointCloud2> sub_fin;
        
        //typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy;
        //typedef Synchronizer<MySyncPolicy> Sync;
        //boost::shared_ptr<Sync> sync;

    public:
        

    Detector()
    {
        sub_ori = nh_.subscribe("/original", 1, &Detector::callbackOriginal, this);
        sub_fin = nh_.subscribe("/final", 1, &Detector::callbackFinal, this);
        pub_tf = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_icp",1);
        count = 0;
        first = true;
        second = false;
        rmse = 1.0;
        //vis.CreateVisualizerWindow("Open3D");
        //vis.Run();
        //sub_ori.subscribe(nh_, "/original", 1);
        //sub_fin.subscribe(nh_, "/final", 1);
        //sync.reset(new Sync(MySyncPolicy(10), sub_ori, sub_fin));      
        //sync->registerCallback(boost::bind(&Detector::callbackICP, this, _1, _2));
    }

    virtual ~Detector()
    {
    }

    void callbackOriginal(const sensor_msgs::PointCloud2ConstPtr& cloud_ori)
    {
        //std::cout<<"received original\n";
        cloud_origin.Clear();
        open3d_conversions::rosToOpen3d(cloud_ori, cloud_origin);
        //ROS_INFO("Recieved pointcloud with sequence number: %d", cloud_data->header.seq);
        // Do something with the Open3D pointcloud
    }

    void callbackFinal(const sensor_msgs::PointCloud2ConstPtr& cloud_fin)
    {
        //std::cout<<cloud_fin->header.frame_id<<"\n";
        //open3d::geometry::PointCloud cloud_final;
        cloud_final.Clear();
        open3d_conversions::rosToOpen3d(cloud_fin, cloud_final);
        if(!cloud_origin.IsEmpty())
        {
            performICP(cloud_origin,cloud_final);
        }
    }

    void performICP(open3d::geometry::PointCloud cloud_ori, open3d::geometry::PointCloud cloud_fin)
    {
        
        int stop = false;
        std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr = std::make_shared<open3d::geometry::PointCloud>(cloud_ori);
        std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr_fin = std::make_shared<open3d::geometry::PointCloud>(cloud_fin);
        std::shared_ptr<open3d::geometry::PointCloud> cloud_fixed = cloud_ptr;
        bool t = true;

        double th = 0.3;
        double th_fine = 0.07;
        cloud_ptr->EstimateNormals();
        cloud_ptr_fin->EstimateNormals();
        if(rmse > 0.0031)
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
                }
                else
                {
                    icp_fine = open3d::pipelines::registration::RegistrationICP(*cloud_ptr,*cloud_ptr_fin,th_fine,icp_fine.transformation_,open3d::pipelines::registration::TransformationEstimationPointToPoint(),open3d::pipelines::registration::ICPConvergenceCriteria(1e-6,1e-6,200));
                    std::cout<<"FINE    fitness : "<<icp_fine.fitness_<<" inlier RMSE : "<<icp_fine.inlier_rmse_<<" correspondence set size : "<<icp_fine.correspondence_set_.size()<<std::endl;
                    rmse = icp_fine.inlier_rmse_;
                }
                
            }
        }
        
        std::shared_ptr<open3d::geometry::PointCloud> cloud_ptr_final = std::make_shared<open3d::geometry::PointCloud>(cloud_ptr->Transform(icp_fine.transformation_));
        sensor_msgs::PointCloud2 msg;
        open3d_conversions::open3dToRos(*cloud_ptr_final,msg,"camera_depth_optical_frame");
        pub_tf.publish(msg);
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