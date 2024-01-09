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
struct Point
{
    int kps_idx;
    cv::Point3d tri_kp;
    cv::Point2d filt_kp;
    cv::Vec3b color; // stored in BGR order
    Point(int kps_idx, cv::Point2d filt_kp, cv::Point3d tri_kp, cv::Vec3b color): kps_idx(kps_idx), filt_kp(filt_kp), tri_kp(tri_kp), color(color)
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
};

class VisualOdometry
{
public:
    VisualOdometry(cv::Matx33d K, cv::Size img_size);
    void process_img(cv::Mat& img);
    cv::Matx34d getProjectionMat(const cv::Matx44d& T, const cv::Matx33d& K);
    void setFirstTrueTranslation(const cv::Vec3d& t);

    std::vector<Pose3d> poses;
    Frame frame, prev_frame;
    std::vector<Point> prev_points;
    std::vector<std::pair<int, int>> corr_idx;

private:
    std::shared_ptr<cv::SIFT> sift;
    std::shared_ptr<cv::FlannBasedMatcher> fbm;
    std::shared_ptr<cv::BFMatcher> bf;
    cv::Matx33d K;
    cv::Size img_size;
    cv::Vec3d first_true_translation;
    bool first_translation = true;

    // constants
    const int min_recover_pose_inlier_count = 150;
    const int min_corr_idx_count = 150;
};
}

#endif