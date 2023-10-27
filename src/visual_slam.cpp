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

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dis(0.0, 1.0);

struct helper_kp
{
    size_t kps_idx;
    cv::Point3d tri_kp;
    cv::Point2d filt_kp;
    helper_kp(size_t kps_idx, cv::Point3d tri_kp, cv::Point2d filt_kp): kps_idx(kps_idx), tri_kp(tri_kp), filt_kp(filt_kp)
    {}
};

// class for storing all the information related to an image frame
class FrameInfo
{
public:
    FrameInfo()
    {

    }

    void clearInfo()
    {
        kps.clear();
        tri_kps.clear();
        helper_kps.clear();
    }

    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    cv::Matx44d T = cv::Matx44d::eye(); // identity transformation matrix
    std::vector<cv::Point3d> tri_kps;
    std::vector<helper_kp> helper_kps;
private:
};

class OrbSLAM
{
public:
    OrbSLAM(cv::Matx33d K, cv::Size img_size): K(K), img_size(img_size)
    {
        orb = cv::ORB::create(3000); // 3000 max features
        bf = cv::BFMatcher::create(cv::NORM_HAMMING2);
        poses.push_back(cv::Point3d(0.0, 0.0, 0.0));
    }

    void process_frame(cv::Mat& frame)
    {
        prev_fi = fi; // store current frame information
        fi.clearInfo();
        corr_idx.clear();
        prev_helper_kps.clear();

        // detect features
        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2d> feats;
        cv::goodFeaturesToTrack(
            gray_frame,
            feats,
            3000,
            0.01,
            3
        );

        // extract key points
        fi.kps.reserve(feats.size());
        for (cv::Point2d& feat : feats)
        {
            fi.kps.emplace_back(feat, 20); // key point diameter is 20
        }
        // compute descriptors for the key points
        orb->compute(frame, fi.kps, fi.desc);

        // match
        std::vector<std::vector<cv::DMatch>> matches;
        std::vector<cv::Point2d> matched_kps, prev_matched_kps;
        std::vector<cv::Point2d> filt_kps, prev_filt_kps;
        std::vector<size_t> matched_idx, filt_idx, prev_matched_idx, prev_filt_idx;
        // std::vector<helper_kp> prev_helper_kps;
        cv::Matx33d E, R;
        cv::Vec3d t;
        std::vector<uchar> inlier_mask;

        if (!prev_fi.desc.empty())
        {
            bf->knnMatch(fi.desc, prev_fi.desc, matches, 3);
            // Lowe's ratio test to remove bad matches
            for (size_t i=0; i<matches.size(); ++i)
            {
                if (matches[i][0].distance < 0.8*matches[i][1].distance)
                {
                    matched_kps.push_back(fi.kps[matches[i][0].queryIdx].pt);
                    matched_idx.push_back(matches[i][0].queryIdx);
                    
                    prev_matched_kps.push_back(prev_fi.kps[matches[i][0].trainIdx].pt);
                    prev_matched_idx.push_back(matches[i][0].trainIdx);
                }
            }
            assert(matched_kps.size() == prev_matched_kps.size()); // make sure number of matched key points are the same

            int inlier_count;
            E = cv::findEssentialMat(matched_kps, prev_matched_kps, K, cv::FM_RANSAC, 0.99, 3, 1000, inlier_mask);
            inlier_count = cv::recoverPose(E, matched_kps, prev_matched_kps, K, R, t, inlier_mask);            
            // make filt_kps, prev_filt_kps, filt_prev_kps_idx
            for (size_t i=0; i<inlier_mask.size(); ++i)
            {
                if (inlier_mask[i])
                {
                    filt_kps.push_back(matched_kps[i]);
                    filt_idx.push_back(matched_idx[i]);

                    prev_filt_kps.push_back(prev_matched_kps[i]);
                    prev_filt_idx.push_back(prev_matched_idx[i]);
                }
            }

            if (inlier_count < min_inlier_count)
            {
                std::cout << "Inlier count below threshold, skipping this frame...\n";
                return;
            }

            // update transformation matix
            fi.T = prev_fi.T * cv::Matx44d(R(0,0), R(0,1), R(0,2), t(0),
                                           R(1,0), R(1,1), R(1,2), t(1),
                                           R(2,0), R(2,1), R(2,2), t(2),
                                           0.0,    0.0,    0.0,    1.0);
            
            cv::Matx44d T, prev_T;
            T = fi.T;
            prev_T = prev_fi.T;

            // get projection matrices
            cv::Matx34d proj_mat, prev_proj_mat;
            proj_mat = K * cv::Matx34d(T(0,0), T(0,1), T(0,2), T(0,3),
                                       T(1,0), T(1,1), T(1,2), T(1,3),
                                       T(2,0), T(2,1), T(2,2), T(2,3));
            prev_proj_mat = K * cv::Matx34d(prev_T(0,0), prev_T(0,1), prev_T(0,2), prev_T(0,3),
                                            prev_T(1,0), prev_T(1,1), prev_T(1,2), prev_T(1,3),
                                            prev_T(2,0), prev_T(2,1), prev_T(2,2), prev_T(2,3));

            // proj_mat = getProjectionMat(fi.T, K);
            // prev_proj_mat = getProjectionMat(prev_fi.T, K);

            // get 3D points of filt kps by triangulation (frame -> world transform)
            cv::Mat tri_kps_hom(4, filt_kps.size(), CV_64F);
            cv::triangulatePoints(prev_proj_mat, proj_mat, prev_filt_kps, filt_kps, tri_kps_hom);
            cv::convertPointsFromHomogeneous(tri_kps_hom.t(), fi.tri_kps);
            
            // make pair
            for (size_t i=0; i<fi.tri_kps.size(); ++i)
            {
                fi.helper_kps.emplace_back(filt_idx[i], fi.tri_kps[i], filt_kps[i]);
                prev_helper_kps.emplace_back(prev_filt_idx[i], fi.tri_kps[i], filt_kps[i]);
            }
            
            // sort pairs
            std::sort(fi.helper_kps.begin(), fi.helper_kps.end(), 
                [](const helper_kp& a, const helper_kp& b)
                {return a.kps_idx < b.kps_idx;}
            );
            std::sort(prev_helper_kps.begin(), prev_helper_kps.end(), 
                [](const helper_kp& a, const helper_kp& b)
                {return a.kps_idx < b.kps_idx;}
            );

            // corr_idx maps index of prev_fi.idx_filt_pairs to fi.idx_filt_pairs
            // that are the same point in 3D space
            // std::vector<std::pair<size_t, size_t>> corr_idx;
            if (!prev_fi.helper_kps.empty())
            {
                // find correspondence
                auto prev_it = prev_fi.helper_kps.begin();
                auto it = prev_helper_kps.begin();

                while (prev_it != prev_fi.helper_kps.end() &&
                       it != prev_helper_kps.end())
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
                        size_t prev_idx = std::distance(prev_fi.helper_kps.begin(), prev_it);
                        size_t idx = std::distance(prev_helper_kps.begin(), it);
                        corr_idx.push_back(std::make_pair(prev_idx, idx));
                        ++prev_it;
                        ++it;
                    }
                }
            }

            // find the scale for translation vector
            cv::Point3d avg_diff;
            for (size_t i=0; i<corr_idx.size(); ++i)
            {
                cv::Point3d diff = prev_fi.helper_kps[corr_idx[i].first].tri_kp - prev_helper_kps[corr_idx[i].second].tri_kp;
                avg_diff += diff;
                // std::cout << diff << std::endl;
            }
            avg_diff /= (double)corr_idx.size();
            std::cout << "Calculated translation vector:\n" << t << std::endl;
            std::cout << "Avg_diff:\n" << avg_diff << std::endl;

            ;
        }
    }

    // ?
    void applyTransformation(cv::Point3d& pose, cv::Matx33d& R, cv::Matx31d& t)
    {
        cv::Matx31d mat_pose(
            pose.x, pose.y, pose.z
        );
        mat_pose = R * mat_pose;
        mat_pose += t;
        pose.x = mat_pose(0,0);
        pose.y = mat_pose(1,0);
        pose.z = mat_pose(2,0);
    }

    // ?
    cv::Matx34d getProjectionMat(const cv::Matx44d& T, const cv::Matx33d& K)
    {
        cv::Matx33d R = T.get_minor<3, 3>(0, 0);
        cv::Matx31d t = T.get_minor<3, 1>(0, 3);

        cv::Matx34d P;

        P.get_minor<3, 3>(0, 0) = R.t();
        P.get_minor<3, 1>(0, 3) = -R.t()*t;
        return  K*P;
    }

    std::vector<cv::Point3d> poses;
    FrameInfo fi, prev_fi;
    std::vector<helper_kp> prev_helper_kps;
    std::vector<std::pair<size_t, size_t>> corr_idx;

private:
    std::shared_ptr<cv::ORB> orb;
    std::shared_ptr<cv::BFMatcher> bf;
    cv::Matx33d K;
    cv::Size img_size;
    int min_inlier_count = 25;
};

void printTriangulatedPts(
    const std::vector<helper_kp>& prev_helper_kps,
    const std::vector<helper_kp>& helper_kps,
    const std::vector<std::pair<size_t, size_t>> corr_idx,
    const rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr& pub, const rclcpp::Node::SharedPtr& node)
{
    visualization_msgs::msg::Marker m;
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
    m.scale.x = 1.0;
    m.scale.y = 1.0;
    m.action = visualization_msgs::msg::Marker::ADD;

    geometry_msgs::msg::Point p;
    for (size_t i=0; i<prev_helper_kps.size(); ++i)
    {
        p.x = prev_helper_kps[i].tri_kp.x;
        p.y = prev_helper_kps[i].tri_kp.y;
        p.z = prev_helper_kps[i].tri_kp.z;
        m.points.push_back(p);
    }
    pub->publish(m);
    
    // m.points.clear();
    // m.color.a = 1.0;
    // m.color.r = dis(gen);
    // m.color.g = dis(gen);
    // m.color.b = dis(gen);
    // for (size_t i=0; i<helper_kps.size(); ++i)
    // {
    //     p.x = helper_kps[i].tri_kp.x;
    //     p.y = helper_kps[i].tri_kp.y;
    //     p.z = helper_kps[i].tri_kp.z;
    //     m.points.push_back(p);
    // }
    // pub->publish(m);

    // m.points.clear();
    // m.type = visualization_msgs::msg::Marker::LINE_STRIP;
    // m.scale.x = 1.0;
    // m.color.a = 1.0;
    // m.color.r = 0.0;
    // m.color.g = 1.0;
    // m.color.b = 0.0;
    // for(size_t i=0; i<corr_idx.size(); ++i)
    // {
    //     p.x = prev_helper_kps[corr_idx[i].first].tri_kp.x;
    //     p.y = prev_helper_kps[corr_idx[i].first].tri_kp.y;
    //     p.z = prev_helper_kps[corr_idx[i].first].tri_kp.z;
    //     m.points.push_back(p);

    //     p.x = prev_helper_kps[corr_idx[i].second].tri_kp.x;
    //     p.y = prev_helper_kps[corr_idx[i].second].tri_kp.y;
    //     p.z = prev_helper_kps[corr_idx[i].second].tri_kp.z;
    //     m.points.push_back(p);
    // }
    // pub->publish(m);
}

void printOdom(const cv::Point3d& pose, const cv::Point3d& prev_pose, 
    const rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr& pub, const rclcpp::Node::SharedPtr& node)
{
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map";
    m.header.stamp = node->now();
    m.lifetime = rclcpp::Duration::from_seconds(0);
    m.frame_locked = false;
    m.id = node->now().nanoseconds();
    m.type = visualization_msgs::msg::Marker::LINE_STRIP;
    m.scale.x = 1.0;
    // make sure alpha and color is set or else the points will be invisible
    m.color.a = 1.0;
    m.color.r = 1.0;
    m.color.g = 0.0;
    m.color.b = 0.0;
    m.action = visualization_msgs::msg::Marker::ADD;

    geometry_msgs::msg::Point p;
    p.x = pose.x;
    p.y = pose.y;
    p.z = pose.z;
    m.points.push_back(p);

    p.x = prev_pose.x;
    p.y = prev_pose.y;
    p.z = prev_pose.z;
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
    
    // set camera intrinsic matrix
    cv::Matx33d K(
        7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,
        0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02,
        0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00
    );

    cv::VideoCapture cap("/home/dev_ws/visual_slam/data/video_0.mp4");
    if (!cap.isOpened())
    {
        fprintf(stderr, "Error opening video.\n");
        return -1;
    }
    // get fps
    double fps = cap.get(cv::CAP_PROP_FPS);

    // get image size
    cv::Mat frame(img_size, CV_64FC3);
    bool first_ret = cap.read(frame);
    if (!first_ret)
    {
        std::cout << "Failed to retrieve first frame.\n";
        return -1;
    }
    img_size = frame.size();
    OrbSLAM orb(K, img_size);

    auto node = std::make_shared<rclcpp::Node>("rviz_pub");
    auto odom_pub = node->create_publisher<visualization_msgs::msg::Marker>("visual_odom", 10);
    auto tri_kps_pub = node->create_publisher<visualization_msgs::msg::Marker>("triangulated_pts", 10);

    cv::Point3d prev_pose;
    cv::Mat prev_frame(img_size, CV_64FC3); // for debug
    cv::Mat two_frames(img_size*2, CV_64FC3);
    bool first_frame = true;
    while (rclcpp::ok())
    {
        bool ret = cap.read(frame);

        if (!ret)
        {
            fprintf(stderr, "Failed to read frame.\n");
            break;   
        }

        if (!debug)
        {
            // process frame
            orb.process_frame(frame);

            // visualize key points
            // for (size_t i=0; i<orb.prev_filt_kps.size(); ++i)
            // {
            //     cv::circle(frame, orb.filt_kps[i], 3, cv::Scalar(255, 0.0, 0.0));
            //     cv::line(frame, orb.filt_kps[i], orb.prev_filt_kps[i], cv::Scalar(0.0, 255, 0.0));
            // }
            
            // // visualize odomtery
            // if (orb.poses.size() >= 2)
            // {
            //     printOdom(orb.poses.back(), prev_pose, odom_pub, node);
            // }
            // prev_pose = orb.poses.back();

            cv::imshow("frame", frame);
            if (cv::waitKey(fps) == 'q')
            {
                break;
            }
        }
        else
        {
            // show previous frame on top of current frame
            orb.process_frame(frame);

            // visualize common kps
            if (!first_frame)
            {      
                // visualize key points
                // for (size_t i=0; i<orb.filt_kps.size(); ++i)
                // {
                //     cv::circle(frame, orb.filt_kps[i], 3, cv::Scalar(255, 0.0, 0.0));
                // }
                // for (size_t i=0; i<orb.prev_fi.filt_kps.size(); ++i)
                // {
                //     cv::circle(prev_frame, orb.prev_fi.filt_kps[i], 3, cv::Scalar(255, 0.0, 0.0));
                // }
                printTriangulatedPts(orb.prev_fi.helper_kps, orb.prev_helper_kps, orb.corr_idx, tri_kps_pub, node);

                for (size_t i=0; i<orb.fi.helper_kps.size(); ++i)
                {
                    cv::circle(frame, orb.fi.helper_kps[i].filt_kp, 3, cv::Scalar(0.0, 0.0, 255));
                }
                for (size_t i=0; i<orb.prev_fi.helper_kps.size(); ++i)
                {
                    cv::circle(prev_frame, orb.prev_fi.helper_kps[i].filt_kp, 3, cv::Scalar(0.0, 0.0, 255));
                }
                for (size_t i=0; i<orb.corr_idx.size(); ++i)
                {
                    cv::circle(frame, orb.prev_helper_kps[orb.corr_idx[i].second].filt_kp, 5, cv::Scalar(0.0, 255, 0.0));
                    cv::circle(prev_frame, orb.prev_fi.helper_kps[orb.corr_idx[i].first].filt_kp, 5, cv::Scalar(0.0, 255, 0.0));
                }

                cv::vconcat(frame, prev_frame, two_frames);

                for (size_t i=0; i<orb.corr_idx.size(); ++i)
                {
                    cv::line(two_frames, 
                        orb.prev_helper_kps[orb.corr_idx[i].second].filt_kp, 
                        orb.prev_fi.helper_kps[orb.corr_idx[i].first].filt_kp + cv::Point2d(0.0, img_size.height),
                        cv::Scalar(0.0, 0.0, 255));
                }

                cv::imshow("two_frames", two_frames);
                if (cv::waitKey(0) == 'q')
                {
                    break;
                }
            }
            prev_frame = frame.clone();
            first_frame = false;
        }
        
    }
    cap.release();
    cv::destroyAllWindows();
    rclcpp::shutdown();
    return 0;
}