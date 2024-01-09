#ifndef RVIZ_VISUALIZATION_NODE_HPP_
#define RVIZ_VISUALIZATION_NODE_HPP_

#include <random>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>

#include <Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "geometry_msgs/msg/point.hpp"

#include "visual_odometry/visual_odometry.hpp"

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dis(0.0, 1.0);

class RvizVisualizationNode
{
public:
    RvizVisualizationNode(rclcpp::Node::SharedPtr& node)
    : node_(node)
    {
        odom_pub_ = node->create_publisher<visualization_msgs::msg::Marker>("visual_odom", 10);
        tri_kps_pub_ = node->create_publisher<visualization_msgs::msg::Marker>("triangulated_pts", 10);
    }

    void printTriangulatedPts(
        const std::vector<visual_odometry::Point>& prev_points,
        const std::vector<visual_odometry::Point>& points,
        const std::vector<std::pair<int, int>> corr_idx,
        bool debug=false) const
    {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";
        m.header.stamp = node_->now();
        m.lifetime = rclcpp::Duration::from_seconds(0);
        m.frame_locked = false;
        m.id = node_->now().nanoseconds();
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
        tri_kps_pub_->publish(m);
        
        if (debug)
        {
            m.points.clear();
            m.header.frame_id = "map";
            m.header.stamp = node_->now();
            m.lifetime = rclcpp::Duration::from_seconds(0);
            m.frame_locked = false;
            m.id = node_->now().nanoseconds();
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
            tri_kps_pub_->publish(m);

            m.points.clear();
            m.header.frame_id = "map";
            m.header.stamp = node_->now();
            m.lifetime = rclcpp::Duration::from_seconds(0);
            m.frame_locked = false;
            m.id = node_->now().nanoseconds();
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
            tri_kps_pub_->publish(m);
        }
    }

    void printOdom(const Pose3d& pose, const Pose3d& prev_pose) const
    {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";
        m.header.stamp = node_->now();
        m.lifetime = rclcpp::Duration::from_seconds(0);
        m.frame_locked = false;
        m.id = node_->now().nanoseconds();
        m.type = visualization_msgs::msg::Marker::LINE_STRIP;
        m.scale.x = 1.0;
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

        odom_pub_->publish(m);
    }

    void printOdomAll(const std::vector<Pose3d>& poses) const
    {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";
        m.header.stamp = node_->now();
        m.lifetime = rclcpp::Duration::from_seconds(0);
        m.frame_locked = false;
        m.id = node_->now().nanoseconds();
        m.type = visualization_msgs::msg::Marker::LINE_LIST;
        m.scale.x = 1.0;
        // make sure alpha and color is set or else the points will be invisible
        m.color.a = 1.0;
        m.color.r = 0.0;
        m.color.g = 0.0;
        m.color.b = 1.0;
        m.action = visualization_msgs::msg::Marker::ADD;

        geometry_msgs::msg::Point p;
        for (auto& pose : poses)
        {
            p.x = pose(0);
            p.y = pose(1);
            p.z = pose(2);
            m.points.push_back(p);
        }
        odom_pub_->publish(m);
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr odom_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr tri_kps_pub_;
};

#endif