#ifndef RVIZ_VISUALIZATION_NODE_HPP_
#define RVIZ_VISUALIZATION_NODE_HPP_

#include <vector>
#include <fstream>
#include <string>
#include <sstream>

#include <Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "geometry_msgs/msg/point.hpp"

#include "visual_odometry/visual_odometry.hpp"

class RvizVisualizationNode
{
public:
    RvizVisualizationNode(rclcpp::Node::SharedPtr& node)
    : node_(node)
    {
        odom_pub_ = node->create_publisher<visualization_msgs::msg::Marker>("visual_odom", 10);
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
        m.scale.x = 2.0;
        // make sure alpha and color is set or else the points will be invisible
        m.color.a = 1.0;
        m.color.r = 1.0; // red
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
        m.scale.x = 2.0;
        // make sure alpha and color is set or else the points will be invisible
        m.color.a = 1.0;
        m.color.r = 0.0;
        m.color.g = 1.0; // green
        m.color.b = 0.0;
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
};

#endif