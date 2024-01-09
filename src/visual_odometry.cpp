#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/color_rgba.hpp"

#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <algorithm>

#include "visual_odometry/visual_odometry.hpp"

namespace visual_odometry
{
VisualOdometry::VisualOdometry(cv::Matx33d K, cv::Size img_size): K(K), img_size(img_size)
{
    sift = cv::SIFT::create(5000);
    fbm = cv::FlannBasedMatcher::create();
    bf = cv::BFMatcher::create();
    poses.push_back(Pose3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0));
}

void VisualOdometry::setFirstTrueTranslation(const cv::Vec3d& t)
{
    first_true_translation = t;
}

void VisualOdometry::process_img(cv::Mat& img)
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

    if (prev_frame.desc.empty())
    {
        return;
    }

    fbm->knnMatch(frame.desc, prev_frame.desc, matches, 3);
    // Lowe's ratio test to remove bad matches
    for (int i=0; i<matches.size(); ++i)
    {
        if (matches[i][0].distance < 0.8*matches[i][1].distance) // a good best match has significantly lower distance than second-best match
        {
            matched_kps.emplace_back(frame.kps[matches[i][0].queryIdx].pt);
            matched_idx.push_back(matches[i][0].queryIdx);
            
            prev_matched_kps.emplace_back(prev_frame.kps[matches[i][0].trainIdx].pt);
            prev_matched_idx.push_back(matches[i][0].trainIdx);
        }
    }
    assert(matched_kps.size() == prev_matched_kps.size()); // make sure number of matched key points are the same

    int inlier_count;
    E = cv::findEssentialMat(prev_matched_kps, matched_kps, K, cv::FM_RANSAC, 0.9999, 3, 1000, inlier_mask);
    inlier_count = cv::recoverPose(E, prev_matched_kps, matched_kps, K, R, t, inlier_mask); // recover pose gives unit translation vector (due to ambiguity)

    // because recoverPose() uses left-handed frame (?)
    R = R.t();
    t *= -1.0;
    if (first_translation)
    {
        double estimated_dist = cv::norm(t);
        double actual_norm = cv::norm(first_true_translation);

        t *= actual_norm / estimated_dist; // multiply by scale
        first_translation = false;
    }
    
    for (int i=0; i<inlier_mask.size(); ++i)
    {
        if (inlier_mask[i])
        {
            filt_kps.emplace_back(matched_kps[i]);
            filt_idx.push_back(matched_idx[i]);

            prev_filt_kps.emplace_back(prev_matched_kps[i]);
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

    for (int i=0; i<tri_kps.size(); ++i)
    {
        frame.points.emplace_back(filt_idx[i], filt_kps[i], tri_kps[i], img.at<cv::Vec3b>(filt_kps[i]));
        prev_points.emplace_back(prev_filt_idx[i], prev_filt_kps[i], tri_kps[i], img.at<cv::Vec3b>(prev_filt_kps[i]));
    }
    
    // sort the points in increasing kps_idx
    std::sort(frame.points.begin(), frame.points.end(), 
        [](const Point& a, const Point& b)
        {return a.kps_idx < b.kps_idx;}
    );
    std::sort(prev_points.begin(), prev_points.end(), 
        [](const Point& a, const Point& b)
        {return a.kps_idx < b.kps_idx;}
    );

    if (prev_frame.points.empty())
    {
        return;
    }

    // find correspondence
    auto prev_it = prev_frame.points.begin();
    auto it = prev_points.begin();

    while (prev_it != prev_frame.points.end() && it != prev_points.end())
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

    std::vector<cv::Point3d> src_pc, tgt_pc;
    for (int i=0; i<corr_idx.size(); ++i)
    {
        src_pc.emplace_back(prev_frame.points[corr_idx[i].first].tri_kp);
        tgt_pc.emplace_back(prev_points[corr_idx[i].second].tri_kp);
    }

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

    // correct the translation and update the transformation matrix
    t += t_correction;
    frame.T = prev_frame.T * cv::Matx44d(R(0,0), R(0,1), R(0,2), t(0),
                                         R(1,0), R(1,1), R(1,2), t(1),
                                         R(2,0), R(2,1), R(2,2), t(2),
                                         0.0,    0.0,    0.0,    1.0);

    // update pose
    poses.push_back(getPoseFromTransformationMatrix(frame.T));

}

cv::Matx34d VisualOdometry::getProjectionMat(const cv::Matx44d& T, const cv::Matx33d& K)
{
    cv::Matx33d R = T.get_minor<3, 3>(0, 0);
    cv::Matx31d t = T.get_minor<3, 1>(0, 3);

    cv::Matx34d P;
    cv::hconcat(R.t(), -R.t()*t, P);
    return  K*P;
}
}