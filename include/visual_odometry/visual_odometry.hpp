#ifndef VISUAL_ODOMETRY_HPP_
#define VISUAL_ODOMETRY_HPP_

#include <vector>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <Eigen/Dense>

using Pose3d = cv::Vec<double, 7>;

inline Pose3d getPoseFromTransformationMatrix(cv::Matx44d& T)
{
    Pose3d pose;
    pose(0) = T(0, 3);
    pose(1) = T(1, 3);
    pose(2) = T(2, 3);

    Eigen::Matrix3d R;
    for (int i=0; i<3; ++i)
    {
        for (int j=0; j<3; ++j)
        {
            R(i, j) = T(i, j);
        }
    }
    Eigen::Quaterniond quat(R);
    
    pose(3) = quat.x();
    pose(4) = quat.y();
    pose(5) = quat.z();
    pose(6) = quat.w();

    return pose;
}

namespace visual_odometry
{
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
        filt_kps.clear();
    }

    std::vector<cv::KeyPoint> kps;
    std::vector<cv::Point2d> filt_kps;
    cv::Mat desc;
    cv::Matx44d T = cv::Matx44d::eye(); // identity transformation matrix
};

class VisualOdometry
{
public:
    VisualOdometry(cv::Matx33d K);
    bool setTruePoses(const std::string& true_pose_file);
    void process_img(cv::Mat& img);

    std::vector<Pose3d> estimated_poses;
    std::vector<Pose3d> true_poses;
    Frame frame, prev_frame;

private:
    std::shared_ptr<cv::SIFT> sift;
    std::shared_ptr<cv::FlannBasedMatcher> fbm;
    cv::Matx33d K;
    int true_pose_idx = 0;

    // constants
    const int min_recover_pose_inlier_count = 150;
};
}

#endif