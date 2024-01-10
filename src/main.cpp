#include <fstream>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"

#include "visual_odometry/visual_odometry.hpp"
#include "visual_odometry/rviz_visualization_node.hpp"

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    cv::Size img_size;
    
    // camera intrinsic matrix obtained from KITTI's calibration.txt file
    cv::Matx33d K(
        7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 
        0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 
        0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00
    );

    // path to video file
    cv::VideoCapture cap("/home/dev_ws/visual_odometry/data/video_2.mp4");
    if (!cap.isOpened())
    {
        std::cout << "Error opening video.\n";
        return -1;
    }
    // get fps
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    visual_odometry::VisualOdometry VO(K);
    if (!VO.setTruePoses("/home/dev_ws/visual_odometry/data/00.txt"))
    {
        std::cout << "Error setting true pose.\n";
        return -1;
    }

    std::ofstream error_log("error_log.txt");
    if (!error_log.is_open())
    {
        std::cout << "Error opening error log file.\n";
        return -1;
    }

    auto node = std::make_shared<rclcpp::Node>("rviz_pub");
    RvizVisualizationNode rviz_node(node);
    rviz_node.printOdomAll(VO.true_poses); // print ground truth

    cv::Mat img;
    while (rclcpp::ok())
    {
        bool ret = cap.read(img);

        if (!ret)
        {
            fprintf(stderr, "Failed to read video.\n");
            break;   
        }

        VO.process_img(img);

        // visualize key points
        for (int i=0; i<VO.frame.filt_kps.size(); ++i)
        {
            cv::circle(img, VO.frame.filt_kps[i], 3, cv::Scalar(255, 0.0, 0.0));
        }

        int curr_pose_idx = VO.estimated_poses.size() - 1;

        // log error
        error_log << VO.true_poses[curr_pose_idx] - VO.estimated_poses[curr_pose_idx] << std::endl;
        
        // print odometry
        if (curr_pose_idx <= 0)
        {
            continue;
        }
        int prev_pose_idx = curr_pose_idx - 1;
        rviz_node.printOdom(VO.estimated_poses[curr_pose_idx], VO.estimated_poses[prev_pose_idx]);

        cv::imshow("Front-facing camera", img);
        if (cv::waitKey(fps) == 'q')
        {
            break;
        }
        
    }
    error_log.close();
    cap.release();
    cv::destroyAllWindows();
    rclcpp::shutdown();
    return 0;
}