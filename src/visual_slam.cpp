#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/color_rgba.hpp"

class OrbSLAM
{
public:
    OrbSLAM(cv::Matx33d K): K(K)
    {
        orb = cv::ORB::create(3000); // 3000 max features
        bf = cv::BFMatcher::create(cv::NORM_HAMMING2);
        poses.push_back(cv::Point3d(0.0, 0.0, 0.0));
    }

    void extract(cv::Mat& frame)
    {
        // clear key points
        filt_kps.clear();
        prev_filt_kps.clear();

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

        std::vector<cv::KeyPoint> kps;
        cv::Mat desc;
        // extract key points
        kps.reserve(feats.size());
        for (cv::Point2d& feat : feats)
        {
            kps.emplace_back(feat, 20); // key point diameter is 20
        }
        // compute descriptors for the key points
        orb->compute(frame, kps, desc);

        // match
        std::vector<std::vector<cv::DMatch>> matches;
        std::vector<cv::Point2d> matched_kps, prev_matched_kps;
        std::vector<uchar> inliers;
        cv::Matx33d E, R;
        cv::Matx31d t;
        if (!prev_desc.empty())
        {
            bf->knnMatch(desc, prev_desc, matches, 3);
            // Lowe's ratio test to remove bad matches
            for (size_t i=0; i<matches.size(); ++i)
            {
                if (matches[i][0].distance < 0.8*matches[i][1].distance)
                {
                    matched_kps.push_back(kps[matches[i][0].queryIdx].pt);
                    prev_matched_kps.push_back(prev_kps[matches[i][0].trainIdx].pt);
                }
            }

            // filter (using fundamental matrix)
            cv::findFundamentalMat(matched_kps, prev_matched_kps, cv::FM_RANSAC, 3, 0.99, 500, inliers);
            for (size_t i=0; i<matched_kps.size(); ++i)
            {
                if (inliers[i])
                {
                    filt_kps.push_back(matched_kps[i]);
                    prev_filt_kps.push_back(prev_matched_kps[i]);
                }
            }

            E = cv::findEssentialMat(filt_kps, prev_filt_kps, K, cv::FM_RANSAC, 0.99, 3, 500);
            cv::recoverPose(E, filt_kps, prev_filt_kps, K, R, t);

            cv::Point3d pose = poses.back();
            applyTransformation(pose, R, t);
            poses.push_back(pose);
        }
        prev_kps = kps; // store current key points
        prev_desc = desc; // store current key point descriptions
    }

    void process_frame(cv::Mat& frame)
    {
        extract(frame);
        for (size_t i=0; i<filt_kps.size(); ++i)
        {
            cv::circle(frame, filt_kps[i], 5, cv::Scalar(0.0, 0.0, 255));
            cv::line(frame, filt_kps[i], prev_filt_kps[i], cv::Scalar(255, 0.0, 0.0), 1);
        }
    }

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

    std::vector<cv::Point3d> poses;

private:
    std::shared_ptr<cv::ORB> orb;
    std::shared_ptr<cv::BFMatcher> bf;
    cv::Mat prev_desc;
    std::vector<cv::KeyPoint> prev_kps;
    std::vector<cv::Point2d> filt_kps, prev_filt_kps;
    cv::Matx33d K;
};

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
    std_msgs::msg::ColorRGBA c;
    c.a = 1.0;
    c.r = 1.0;
    c.g = 0.0;
    c.b = 0.0;
    m.color = c;
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
    rclcpp::init(argc, argv);
    cv::Size img_size;
    
    // set camera intrinsic matrix
    cv::Matx33d K(
        7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,
        0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02,
        0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00
    );
    OrbSLAM orb(K);

    cv::VideoCapture cap("/home/dev_ws/visual_slam/data/video_0.mp4");
    if (!cap.isOpened())
    {
        fprintf(stderr, "Error opening video.\n");
        return -1;
    }
    // get fps
    double fps = cap.get(cv::CAP_PROP_FPS);

    // get image size
    cv::Mat frame;
    bool first_ret = cap.read(frame);
    if (!first_ret)
    {
        std::cout << "Failed to retrieve first frame.\n";
        return -1;
    }
    img_size = frame.size();

    auto node = std::make_shared<rclcpp::Node>("rviz_pub");
    auto odom_pub = node->create_publisher<visualization_msgs::msg::Marker>("visual_odom", 10);

    cv::Point3d prev_pose;
    while (true)
    {
        bool ret = cap.read(frame);

        if (!ret)
        {
            fprintf(stderr, "Failed to read frame.\n");
            break;   
        }
        cv::resize(frame, frame, img_size, 1.0, 1.0, cv::INTER_LINEAR);

        // process frame
        orb.process_frame(frame);
        if (orb.poses.size() >= 2)
        {
            printOdom(orb.poses.back(), prev_pose, odom_pub, node);
        }
        prev_pose = orb.poses.back();

        cv::imshow("frame", frame);
        if (cv::waitKey(fps) == 'q')
        {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    rclcpp::shutdown();
    return 0;
}