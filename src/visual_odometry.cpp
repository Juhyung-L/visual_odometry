#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/color_rgba.hpp"

#include <random>
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <algorithm>

#include "visual_odometry/util.hpp"

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dis(0.0, 1.0);

namespace visual_odometry
{

struct Point
{
    int kps_idx;
    cv::Point3d tri_kp;
    cv::Point2d filt_kp;
    cv::Vec3b color; // stored in BGR order
    Point(int kps_idx, cv::Point3d tri_kp, cv::Point2d filt_kp, cv::Vec3b color): kps_idx(kps_idx), tri_kp(tri_kp), filt_kp(filt_kp), color(color)
    {}
};

// class for storing all the information related to an image frame
class Frame
{
public:
    Frame()
    {

    }

    void clear()
    {
        kps.clear();
        points.clear();
    }

    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    cv::Matx44d T = cv::Matx44d::eye(); // identity transformation matrix
    std::vector<Point> points;
private:
};

class VisualOdometry
{
public:
    VisualOdometry(cv::Matx33d K, cv::Size img_size): K(K), img_size(img_size)
    {
        sift = cv::SIFT::create(5000);
        fbm = cv::FlannBasedMatcher::create();
        bf = cv::BFMatcher::create();
        poses.push_back(cv::Vec4d(0.0, 0.0, 0.0, 1.0));
    }

    void process_img(cv::Mat& img)
    {
        prev_frame = frame; // store current frame information
        frame.clear();
        corr_idx.clear();
        prev_points.clear();

        // detect features
        cv::Mat gray_img;
        cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2d> feats;

        // using sift feature extractor
        sift->detectAndCompute(gray_img, cv::Mat(), frame.kps, frame.desc);
        
        // match
        std::vector<std::vector<cv::DMatch>> matches;
        std::vector<cv::Point2d> matched_kps, prev_matched_kps;
        std::vector<cv::Point2d> filt_kps, prev_filt_kps;
        std::vector<cv::Point3d> tri_kps;
        std::vector<int> matched_idx, filt_idx, prev_matched_idx, prev_filt_idx;
        cv::Matx33d E, R;
        cv::Vec3d t;
        std::vector<uchar> inlier_mask;
        cv::Matx34d P, prev_P;

        if (!prev_frame.desc.empty())
        {
            fbm->knnMatch(frame.desc, prev_frame.desc, matches, 3);
            // Lowe's ratio test to remove bad matches
            for (int i=0; i<matches.size(); ++i)
            {
                if (matches[i][0].distance < 0.8*matches[i][1].distance) // a good best match has significantly lower distance than second-best match
                {
                    matched_kps.push_back(frame.kps[matches[i][0].queryIdx].pt);
                    matched_idx.push_back(matches[i][0].queryIdx);
                    
                    prev_matched_kps.push_back(prev_frame.kps[matches[i][0].trainIdx].pt);
                    prev_matched_idx.push_back(matches[i][0].trainIdx);
                }
            }
            assert(matched_kps.size() == prev_matched_kps.size()); // make sure number of matched key points are the same

            int inlier_count;
            E = cv::findEssentialMat(prev_matched_kps, matched_kps, K, cv::FM_RANSAC, 0.9999, 3, 1000, inlier_mask);
            inlier_count = cv::recoverPose(E, prev_matched_kps, matched_kps, K, R, t, inlier_mask);
            // because recoverPose() uses left-handed frame (?)
            R = R.t();
            t *= -1.0;
            
            for (int i=0; i<inlier_mask.size(); ++i)
            {
                if (inlier_mask[i])
                {
                    filt_kps.push_back(matched_kps[i]);
                    filt_idx.push_back(matched_idx[i]);

                    prev_filt_kps.push_back(prev_matched_kps[i]);
                    prev_filt_idx.push_back(prev_matched_idx[i]);
                }
            }

            if (inlier_count < min_recover_pose_inlier_count)
            {
                std::cout << "recoverPose inlier count below threshold. Skipping this frame.\n";
                return;
            }

            // get new transformation matix
            cv::Matx44d T = prev_frame.T * cv::Matx44d(R(0,0), R(0,1), R(0,2), t(0),
                                                       R(1,0), R(1,1), R(1,2), t(1),
                                                       R(2,0), R(2,1), R(2,2), t(2),
                                                       0.0,    0.0,    0.0,    1.0);
            
            // get projection matrices
            P = getProjectionMat(T, K);
            prev_P = getProjectionMat(prev_frame.T, K);

            // get 3D points of filt kps by triangulation (frame -> world transform)
            cv::Mat tri_kps_hom;
            cv::triangulatePoints(prev_P, P, prev_filt_kps, filt_kps, tri_kps_hom);
            cv::convertPointsFromHomogeneous(tri_kps_hom.t(), tri_kps);

            // print rotation and translation values
            // cv::Vec3d angles = rot2euler(R);
            // std::cout << "Rotation:\n" 
            //           << "x: " << angles[0] << std::endl
            //           << "y: " << angles[2] << std::endl
            //           << "z: " << angles[1] << std::endl;
            // std::cout << "Translation:\n" << t << std::endl;

            for (int i=0; i<tri_kps.size(); ++i)
            {
                frame.points.emplace_back(filt_idx[i], tri_kps[i], filt_kps[i], img.at<cv::Vec3b>(filt_kps[i]));
                prev_points.emplace_back(prev_filt_idx[i], tri_kps[i], prev_filt_kps[i], img.at<cv::Vec3b>(prev_filt_kps[i]));
            }
            
            std::sort(frame.points.begin(), frame.points.end(), 
                [](const Point& a, const Point& b)
                {return a.kps_idx < b.kps_idx;}
            );
            std::sort(prev_points.begin(), prev_points.end(), 
                [](const Point& a, const Point& b)
                {return a.kps_idx < b.kps_idx;}
            );

            if (!prev_frame.points.empty())
            {
                // find correspondence
                auto prev_it = prev_frame.points.begin();
                auto it = prev_points.begin();

                while (prev_it != prev_frame.points.end() &&
                       it != prev_points.end())
                {
                    if (it->kps_idx > prev_it->kps_idx)
                    {
                        ++prev_it;
                    }
                    else if (it->kps_idx < prev_it->kps_idx)
                    {
                        ++it;
                    }
                    else if (it->kps_idx == prev_it->kps_idx) // found corresponding point
                    {
                        int prev_idx = std::distance(prev_frame.points.begin(), prev_it);
                        int idx = std::distance(prev_points.begin(), it);
                        corr_idx.push_back(std::make_pair(prev_idx, idx));
                        ++prev_it;
                        ++it;
                    }
                }
            }

            // find correction for translation vector
            std::vector<std::pair<double, cv::Vec3d>> norm_vec_pairs(corr_idx.size());
            cv::Matx33d global_R = T.get_minor<3, 3>(0, 0);
            for (int i=0; i<corr_idx.size(); ++i)
            {
                cv::Vec3d diff = prev_frame.points[corr_idx[i].first].tri_kp - prev_points[corr_idx[i].second].tri_kp;
                diff = global_R.t() * diff; // perspective transform
                norm_vec_pairs.push_back(std::make_pair(cv::norm(diff), diff));
            }
            // find the median
            std::sort(norm_vec_pairs.begin(), norm_vec_pairs.end(),
                [](const std::pair<double, cv::Vec3d>& a, const std::pair<double, cv::Vec3d>& b)
                {return a.first < b.first;}
            );

            // calculate t_correction
            int n = norm_vec_pairs.size();
            cv::Vec3d t_correction;
            if (n > min_corr_idx_count)
            {
                if (n % 2 == 0) // even
                {
                    t_correction = (norm_vec_pairs[n/2-1].second + norm_vec_pairs[n/2].second) / 2;
                }
                else // odd
                {
                    t_correction = norm_vec_pairs[n/2].second;
                }
            }
            else
            {
                std::cout << "corr_idx count below threshold. Skipping this frame.\n";
                return;

            }            
            // std::cout << "Calculated t:" << t << std::endl;
            // std::cout << "Correction t:" << t_correction << std::endl;

            // correct the translation and update the transformation matrix
            t += t_correction;
            frame.T = prev_frame.T * cv::Matx44d(R(0,0), R(0,1), R(0,2), t(0),
                                                 R(1,0), R(1,1), R(1,2), t(1),
                                                 R(2,0), R(2,1), R(2,2), t(2),
                                                 0.0,    0.0,    0.0,    1.0);

            // update pose
            cv::Vec4d pose = frame.T * cv::Vec4d(0.0, 0.0, 0.0, 1.0);
            poses.push_back(pose);
        }
    }

    cv::Matx34d getProjectionMat(const cv::Matx44d& T, const cv::Matx33d& K)
    {
        cv::Matx33d R = T.get_minor<3, 3>(0, 0);
        cv::Matx31d t = T.get_minor<3, 1>(0, 3);

        cv::Matx34d P;
        cv::hconcat(R.t(), -R.t()*t, P);
        return  K*P;
    }

    std::vector<cv::Vec4d> poses;
    Frame frame, prev_frame;
    std::vector<Point> prev_points;
    std::vector<std::pair<int, int>> corr_idx;

private:
    std::shared_ptr<cv::SIFT> sift;
    std::shared_ptr<cv::FlannBasedMatcher> fbm;
    std::shared_ptr<cv::BFMatcher> bf;
    cv::Matx33d K;
    cv::Size img_size;

    // constants
    const int min_recover_pose_inlier_count = 150;
    const int min_corr_idx_count = 150;
};
}

void printTriangulatedPts(
    const std::vector<visual_odometry::Point>& prev_points,
    const std::vector<visual_odometry::Point>& points,
    const std::vector<std::pair<int, int>> corr_idx,
    const rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr& pub, const rclcpp::Node::SharedPtr& node,
    bool debug=false)
{
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map";
    m.header.stamp = node->now();
    m.lifetime = rclcpp::Duration::from_seconds(0);
    m.frame_locked = false;
    m.id = node->now().nanoseconds();
    m.type = visualization_msgs::msg::Marker::POINTS;
    m.color.a = 1.0;
    m.scale.x = 0.1;
    m.scale.y = 0.1;
    m.action = visualization_msgs::msg::Marker::ADD;

    geometry_msgs::msg::Point p;
    std_msgs::msg::ColorRGBA c;
    c.a = 1.0;
    for (int i=0; i<prev_points.size(); ++i)
    {
        p.x = prev_points[i].tri_kp.x;
        p.y = prev_points[i].tri_kp.y;
        p.z = prev_points[i].tri_kp.z;
        m.points.push_back(p);

        c.b = (double)prev_points[i].color[0] / 255.0;
        c.g = (double)prev_points[i].color[1] / 255.0;
        c.r = (double)prev_points[i].color[2] / 255.0;
        m.colors.push_back(c);
    }
    pub->publish(m);
    
    if (debug)
    {
        m.points.clear();
        m.header.frame_id = "map";
        m.header.stamp = node->now();
        m.lifetime = rclcpp::Duration::from_seconds(0);
        m.frame_locked = false;
        m.id = node->now().nanoseconds();
        m.type = visualization_msgs::msg::Marker::POINTS;
        m.color.a = 1.0;
        m.color.r = dis(gen);
        m.color.g = dis(gen);
        m.color.b = dis(gen);
        m.scale.x = 0.1;
        m.scale.y = 0.1;
        m.action = visualization_msgs::msg::Marker::ADD;
        for (int i=0; i<points.size(); ++i)
        {
            p.x = points[i].tri_kp.x;
            p.y = points[i].tri_kp.y;
            p.z = points[i].tri_kp.z;
            m.points.push_back(p);
        }
        pub->publish(m);

        m.points.clear();
        m.header.frame_id = "map";
        m.header.stamp = node->now();
        m.lifetime = rclcpp::Duration::from_seconds(0);
        m.frame_locked = false;
        m.id = node->now().nanoseconds();
        m.type = visualization_msgs::msg::Marker::LINE_LIST;
        m.scale.x = 0.01;
        m.color.a = 1.0;
        m.color.r = 0.0;
        m.color.g = 1.0;
        m.color.b = 0.0;
        m.action = visualization_msgs::msg::Marker::ADD;
        for(int i=0; i<corr_idx.size(); ++i)
        {
            p.x = prev_points[corr_idx[i].first].tri_kp.x;
            p.y = prev_points[corr_idx[i].first].tri_kp.y;
            p.z = prev_points[corr_idx[i].first].tri_kp.z;
            m.points.push_back(p);

            p.x = points[corr_idx[i].second].tri_kp.x;
            p.y = points[corr_idx[i].second].tri_kp.y;
            p.z = points[corr_idx[i].second].tri_kp.z;
            m.points.push_back(p);
        }
        pub->publish(m);
    }
}

void printOdom(const cv::Vec4d& pose, const cv::Vec4d& prev_pose, 
    const rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr& pub, const rclcpp::Node::SharedPtr& node)
{
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map";
    m.header.stamp = node->now();
    m.lifetime = rclcpp::Duration::from_seconds(0);
    m.frame_locked = false;
    m.id = node->now().nanoseconds();
    m.type = visualization_msgs::msg::Marker::LINE_STRIP;
    m.scale.x = 0.1;
    // make sure alpha and color is set or else the points will be invisible
    m.color.a = 1.0;
    m.color.r = 1.0;
    m.color.g = 0.0;
    m.color.b = 0.0;
    m.action = visualization_msgs::msg::Marker::ADD;

    geometry_msgs::msg::Point p;
    p.x = pose(0);
    p.y = pose(1);
    p.z = pose(2);
    m.points.push_back(p);

    p.x = prev_pose(0);
    p.y = prev_pose(1);
    p.z = prev_pose(2);
    m.points.push_back(p);

    pub->publish(m);
}

int main(int argc, char* argv[])
{

    bool debug = false;
    if (argc > 1 && strcmp("debug", argv[1]))
    {
        debug = true;
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

    auto node = std::make_shared<rclcpp::Node>("rviz_pub");
    auto odom_pub = node->create_publisher<visualization_msgs::msg::Marker>("visual_odom", 10);
    auto tri_kps_pub = node->create_publisher<visualization_msgs::msg::Marker>("triangulated_pts", 10);

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
            printTriangulatedPts(VO.prev_frame.points, VO.prev_points, VO.corr_idx, tri_kps_pub, node, debug);
            // print odometry
            int curr_pose_idx = VO.poses.size() - 1;
            int prev_pose_idx = curr_pose_idx - 1;
            printOdom(VO.poses[curr_pose_idx], VO.poses[prev_pose_idx], odom_pub, node);

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
                printTriangulatedPts(VO.prev_frame.points, VO.prev_points, VO.corr_idx, tri_kps_pub, node, debug);
                // print odometry
                int curr_pose_idx = VO.poses.size() - 1;
                int prev_pose_idx = curr_pose_idx - 1;
                printOdom(VO.poses[curr_pose_idx], VO.poses[prev_pose_idx], odom_pub, node);

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