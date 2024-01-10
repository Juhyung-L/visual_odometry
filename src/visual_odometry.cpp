#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "visual_odometry/visual_odometry.hpp"

namespace visual_odometry
{
VisualOdometry::VisualOdometry(cv::Matx33d K): K(K)
{
    sift = cv::SIFT::create(5000);
    fbm = cv::FlannBasedMatcher::create();
    estimated_poses.push_back(Pose3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)); // initial pose
}

bool VisualOdometry::setTruePoses(const std::string& true_pose_file)
{
    std::ifstream f(true_pose_file);
    if (!f.is_open())
    {
        return false;
    }
   
    true_poses.emplace_back(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0); // initial pose

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
    return true;
}

void VisualOdometry::process_img(cv::Mat& img)
{
    prev_frame = frame; // store current frame information
    frame.clear();
    ++pose_idx;

    // detect features
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2d> feats;

    // using sift feature extractor
    sift->detectAndCompute(gray_img, cv::Mat(), frame.kps, frame.desc);
    
    // match
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::Point2d> matched_kps, prev_matched_kps;
    cv::Matx33d E, R;
    cv::Vec3d t;
    std::vector<uchar> inlier_mask;

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
            prev_matched_kps.emplace_back(prev_frame.kps[matches[i][0].trainIdx].pt);
        }
    }
    assert(matched_kps.size() == prev_matched_kps.size()); // make sure number of matched key points are the same

    int inlier_count;
    E = cv::findEssentialMat(prev_matched_kps, matched_kps, K, cv::FM_RANSAC, 0.9999, 3, 1000, inlier_mask);
    inlier_count = cv::recoverPose(E, prev_matched_kps, matched_kps, K, R, t, inlier_mask); // recover pose gives unit translation vector (due to ambiguity)

    // because recoverPose() uses left-handed frame (?)
    R = R.t();
    t *= -1.0;

    for (int i=0; i<inlier_mask.size(); ++i)
    {
        if (inlier_mask[i])
        {
            frame.filt_kps.push_back(matched_kps[i]);
        }
    }

    // apply the translation only if the inlier_count is greater than the threshold
    if (inlier_count < min_recover_pose_inlier_count)
    {
        estimated_poses.push_back(getPoseFromTransformationMatrix(frame.T)); // append current pose without applying the transform
        return;
    }

    // get new transformation matix
    frame.T = prev_frame.T * cv::Matx44d(R(0,0), R(0,1), R(0,2), t(0),
                                         R(1,0), R(1,1), R(1,2), t(1),
                                         R(2,0), R(2,1), R(2,2), t(2),
                                         0.0,    0.0,    0.0,    1.0);

    // get the scale of translation from ground truth
    cv::Vec3d true_translation(true_poses[pose_idx](0), true_poses[pose_idx](1), true_poses[pose_idx](2));
    double estimated_dist = cv::norm(cv::Vec3d(frame.T(0, 3), frame.T(1, 3), frame.T(2, 3)));
    double true_dist = cv::norm(true_translation);
    double correction_factor = true_dist / estimated_dist;

    frame.T(0, 3) *= correction_factor;
    frame.T(1, 3) *= correction_factor;
    frame.T(2, 3) *= correction_factor;

    estimated_poses.push_back(getPoseFromTransformationMatrix(frame.T));

    // print accumulated error
    std::cout << "Error in position: \t" << "Error in orientation: "
              << "\nx: " << true_poses[pose_idx](0) - estimated_poses[pose_idx](0) << "\t\t" << "x: " << true_poses[pose_idx](3) - estimated_poses[pose_idx](3)
              << "\ny: " << true_poses[pose_idx](1) - estimated_poses[pose_idx](1) << "\t\t" << "y: " << true_poses[pose_idx](4) - estimated_poses[pose_idx](4)
              << "\nz: " << true_poses[pose_idx](2) - estimated_poses[pose_idx](2) << "\t\t" << "z: " << true_poses[pose_idx](5) - estimated_poses[pose_idx](5)
              << "\n\t\t\tw: " << true_poses[pose_idx](6) - estimated_poses[pose_idx](6)
              << std::endl;

}
}