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

    bool debug = false;
    if (argc > 1 && strcmp("debug", argv[1]))
    {
        debug = true;
    }

    // get ground truth
    std::string ground_truth_file("/home/dev_ws/visual_odometry/data/00.txt");
    std::ifstream f(ground_truth_file);

    if (!f.is_open())
    {
        std::cout << "Could not open file: " << ground_truth_file << std::endl; 
        return -1;
    }

    std::vector<Pose3d> true_poses;
    true_poses.emplace_back(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    std::string line;
    while (std::getline(f, line))
    {
        cv::Matx44d T = cv::Matx44d::zeros();
        std::stringstream ss(line);
        ss >> T(0, 0);
        ss >> T(0, 1);
        ss >> T(0, 2);
        ss >> T(0, 3);
        ss >> T(1, 0);
        ss >> T(1, 1);
        ss >> T(1, 2);
        ss >> T(1, 3);
        ss >> T(2, 0);
        ss >> T(2, 1);
        ss >> T(2, 2);
        ss >> T(2, 3);
        T(3, 3) = 1.0;

        true_poses.push_back(getPoseFromTransformationMatrix(T));
    }

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
        fprintf(stderr, "Error opening video.\n");
        return -1;
    }
    // get fps
    double fps = cap.get(cv::CAP_PROP_FPS);

    // get image size
    cv::Mat img(img_size, CV_64FC3);
    bool first_ret = cap.read(img);
    if (!first_ret)
    {
        std::cout << "Failed to retrieve first image.\n";
        return -1;
    }
    img_size = img.size();
    visual_odometry::VisualOdometry VO(K, img_size);

    // set first true translation (for scale)
    VO.setFirstTrueTranslation(cv::Vec3d(true_poses[1](0), true_poses[1](1), true_poses[1](2)));

    auto node = std::make_shared<rclcpp::Node>("rviz_pub");
    RvizVisualizationNode rviz_node(node);
    rviz_node.printOdomAll(true_poses); // print ground truth

    cv::Point3d prev_pose;
    cv::Mat prev_img(img_size, CV_64FC3); // for debug
    cv::Mat two_imgs(img_size*2, CV_64FC3);
    bool first_img = true;
    while (rclcpp::ok())
    {
        bool ret = cap.read(img);

        if (!ret)
        {
            fprintf(stderr, "Failed to read video.\n");
            break;   
        }

        if (!debug)
        {
            VO.process_img(img);

            // visualize key points
            for (int i=0; i<VO.prev_frame.points.size(); ++i)
            {
                cv::circle(img, VO.prev_frame.points[i].filt_kp, 3, cv::Scalar(255, 0.0, 0.0));
            }
            
            // print triangulated points
            rviz_node.printTriangulatedPts(VO.prev_frame.points, VO.prev_points, VO.corr_idx, debug);
            // print odometry
            int curr_pose_idx = VO.poses.size() - 1;
            int prev_pose_idx = curr_pose_idx - 1;
            rviz_node.printOdom(VO.poses[curr_pose_idx], VO.poses[prev_pose_idx]);

            cv::imshow("Front-facing camera", img);
            if (cv::waitKey(fps) == 'q')
            {
                break;
            }
        }
        else
        {
            VO.process_img(img);

            if (!first_img)
            {
                // print triangulated points
                rviz_node.printTriangulatedPts(VO.prev_frame.points, VO.prev_points, VO.corr_idx, debug);
                // print odometry
                int curr_pose_idx = VO.poses.size() - 1;
                int prev_pose_idx = curr_pose_idx - 1;
                rviz_node.printOdom(VO.poses[curr_pose_idx], VO.poses[prev_pose_idx]);

                // visualize filtered key points
                for (int i=0; i<VO.frame.points.size(); ++i)
                {
                    cv::circle(img, VO.frame.points[i].filt_kp, 3, cv::Scalar(0.0, 0.0, 255));
                }
                for (int i=0; i<VO.prev_frame.points.size(); ++i)
                {
                    cv::circle(prev_img, VO.prev_frame.points[i].filt_kp, 3, cv::Scalar(0.0, 0.0, 255));
                }

                // highlight points that appears in consecutive feature matching
                for (int i=0; i<VO.corr_idx.size(); ++i)
                {
                    cv::circle(img, VO.prev_points[VO.corr_idx[i].second].filt_kp, 5, cv::Scalar(0.0, 255, 0.0));
                    cv::circle(prev_img, VO.prev_frame.points[VO.corr_idx[i].first].filt_kp, 5, cv::Scalar(0.0, 255, 0.0));
                }

                cv::vconcat(img, prev_img, two_imgs);

                // draw correspondence lines
                for (int i=0; i<VO.corr_idx.size(); ++i)
                {
                    cv::line(two_imgs, 
                        VO.prev_points[VO.corr_idx[i].second].filt_kp, 
                        VO.prev_frame.points[VO.corr_idx[i].first].filt_kp + cv::Point2d(0.0, img_size.height),
                        cv::Scalar(0.0, 0.0, 255));
                }

                cv::imshow("Debug view", two_imgs);
                if (cv::waitKey(0) == 'q')
                {
                    break;
                }
            }
            prev_img = img.clone();
            first_img = false;
        }
        
    }
    cap.release();
    cv::destroyAllWindows();
    rclcpp::shutdown();
    return 0;
}