#!/usr/bin/env python3
# coding:utf-8

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Int32MultiArray
import tf.transformations as tf_trans
import math
import copy

class TicTacToeDrawer:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('tic_tac_toe_drawer', anonymous=True)

        # 订阅井字棋位置和标记
        self.sub = rospy.Subscriber("/position_and_sign", Int32MultiArray, self.position_callback)
        
        # 发布目标姿态
        self.target_pose_pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=1)

        self.home_pose = self.create_pose(0.1782323271036148, 0, 0.11, -1.1807824762399832e-07, 0.999998927116394, 8.342405100059125e-10, 0.001613360014744103)
        self.height = 0.075
        self.circle_or_cross = 0 #0:circle, 1:cross

        # 九个格子的中心坐标和姿态（请根据实际情况修改这些值）
        self.grid_centers = {
            1: self.create_pose(0.3, 0.10, self.height, self.home_pose.pose.orientation.x, self.home_pose.pose.orientation.y, self.home_pose.pose.orientation.z, self.home_pose.pose.orientation.w),  # 第一行第一列
            2: self.create_pose(0.3, 0.0, self.height, self.home_pose.pose.orientation.x, self.home_pose.pose.orientation.y, self.home_pose.pose.orientation.z, self.home_pose.pose.orientation.w),  # 第一行第二列
            3: self.create_pose(0.3, -0.10, self.height, self.home_pose.pose.orientation.x, self.home_pose.pose.orientation.y, self.home_pose.pose.orientation.z, self.home_pose.pose.orientation.w), # 第一行第三列
            4: self.create_pose(0.22, 0.10, self.height, self.home_pose.pose.orientation.x, self.home_pose.pose.orientation.y, self.home_pose.pose.orientation.z, self.home_pose.pose.orientation.w),  # 第二行第一列
            5: self.create_pose(0.22, 0.0, self.height, self.home_pose.pose.orientation.x, self.home_pose.pose.orientation.y, self.home_pose.pose.orientation.z, self.home_pose.pose.orientation.w),  # 第二行第二列
            6: self.create_pose(0.22, -0.10, self.height, self.home_pose.pose.orientation.x, self.home_pose.pose.orientation.y, self.home_pose.pose.orientation.z, self.home_pose.pose.orientation.w), # 第二行第三列
            7: self.create_pose(0.14, 0.10, self.height, self.home_pose.pose.orientation.x, self.home_pose.pose.orientation.y, self.home_pose.pose.orientation.z, self.home_pose.pose.orientation.w), # 第三行第一列
            8: self.create_pose(0.14, 0.0, self.height, self.home_pose.pose.orientation.x, self.home_pose.pose.orientation.y, self.home_pose.pose.orientation.z, self.home_pose.pose.orientation.w), # 第三行第二列
            9: self.create_pose(0.14, -0.10, self.height, -self.home_pose.pose.orientation.x, self.home_pose.pose.orientation.y, self.home_pose.pose.orientation.z, self.home_pose.pose.orientation.w) # 第三行第三列
        }
        
        # 圆形和叉形的轨迹点
        self.circle_trajectory = self.generate_circle_trajectory(0.015, 256)  # 半径0.05m，16个点
        self.cross_trajectory = self.generate_cross_trajectory(0.02)  # 边长0.05m
        
        rospy.loginfo("Tic Tac Toe drawer node started. Waiting for position and sign messages...")

    def create_pose(self, x, y, z, qx, qy, qz, qw):
        """创建姿态消息"""
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def generate_circle_trajectory(self, radius, num_points):
        """生成圆形轨迹"""
        trajectory = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = 0.0
            trajectory.append(self.create_pose(x, y, z, self.home_pose.pose.orientation.x, self.home_pose.pose.orientation.y, self.home_pose.pose.orientation.z, self.home_pose.pose.orientation.w))

        return trajectory

    def generate_cross_trajectory(self, size):
        """生成叉形轨迹（带抬笔移动）"""
        half_size = size / 2
        return [
            # 第一条对角线：左下到右上
            self.create_pose(-half_size, -half_size, 0.0, self.home_pose.pose.orientation.x, self.home_pose.pose.orientation.y, self.home_pose.pose.orientation.z, self.home_pose.pose.orientation.w),
            self.create_pose(half_size, half_size, 0.0, self.home_pose.pose.orientation.x, self.home_pose.pose.orientation.y, self.home_pose.pose.orientation.z, self.home_pose.pose.orientation.w),
            
            # 抬笔移动到第二条线的起点
            self.create_pose(-half_size, half_size, 0.03, self.home_pose.pose.orientation.x, self.home_pose.pose.orientation.y, self.home_pose.pose.orientation.z, self.home_pose.pose.orientation.w),  # Z稍微抬高
            
            # 第二条对角线：左上到右下
            self.create_pose(-half_size, half_size, 0.0, self.home_pose.pose.orientation.x, self.home_pose.pose.orientation.y, self.home_pose.pose.orientation.z, self.home_pose.pose.orientation.w),
            self.create_pose(half_size, -half_size, 0.0, self.home_pose.pose.orientation.x, self.home_pose.pose.orientation.y, self.home_pose.pose.orientation.z, self.home_pose.pose.orientation.w)
        ]

    def position_callback(self, msg):
        """位置和标记回调函数"""
        if len(msg.data) < 2:
            rospy.logwarn("Invalid message format. Expected [position, sign]")
            return
        
        position = msg.data[0]
        sign = msg.data[1]  # 0: 圆形, 1: 叉形
        
        if position not in self.grid_centers:
            rospy.logwarn(f"Invalid position: {position}. Must be between 1 and 9.")
            return
        center_pose = self.grid_centers[position]
        
        if sign == 0:  # 圆形
            self.circle_or_cross = 0
            trajectory = self.circle_trajectory
            rospy.loginfo(f"Drawing circle at position {position}")
        elif sign == 1:  # 叉形
            self.circle_or_cross = 1
            trajectory = self.cross_trajectory
            rospy.loginfo(f"Drawing cross at position {position}")
        else:
            rospy.logwarn(f"Invalid sign: {sign}. Must be 0 (circle) or 1 (cross).")
            return
        
        # 发布轨迹点
        self.publish_trajectory(center_pose, trajectory)

    def publish_trajectory(self, center_pose, trajectory):
        """发布轨迹点"""
        # 创建第一个点的高位版本，而不是直接修改原始轨迹点
        high_start_pose = copy.deepcopy(trajectory[0])
        high_start_pose.pose.position.z += 0.03
        
        # 发布高位起始点
        target_pose = PoseStamped()
        target_pose.header.frame_id = center_pose.header.frame_id
        target_pose.header.stamp = rospy.Time.now()
        
        # 设置位置（相对于中心点的偏移）
        target_pose.pose.position.x = center_pose.pose.position.x + high_start_pose.pose.position.x
        target_pose.pose.position.y = center_pose.pose.position.y + high_start_pose.pose.position.y
        target_pose.pose.position.z = center_pose.pose.position.z + high_start_pose.pose.position.z
        
        target_pose.pose.orientation = self.home_pose.pose.orientation
        
        self.target_pose_pub.publish(target_pose)

        rospy.sleep(1)

        for i in range(30):
            target_pose.pose.position.z -= 0.001
            self.target_pose_pub.publish(target_pose)
            rospy.sleep(0.01)
        rospy.loginfo("Moving to high starting position")
        rospy.sleep(1)

        if self.circle_or_cross == 0:
            for pose in trajectory:
                # 创建相对于中心点的姿态
                target_pose = PoseStamped()
                target_pose.header.frame_id = center_pose.header.frame_id
                target_pose.header.stamp = rospy.Time.now()
                
                # 设置位置（相对于中心点的偏移）
                target_pose.pose.position.x = center_pose.pose.position.x + pose.pose.position.x
                target_pose.pose.position.y = center_pose.pose.position.y + pose.pose.position.y
                target_pose.pose.position.z = center_pose.pose.position.z + pose.pose.position.z
                
                target_pose.pose.orientation = self.home_pose.pose.orientation
                
                self.target_pose_pub.publish(target_pose)
                
                rospy.sleep(0.05)
            
            self.target_pose_pub.publish(self.home_pose)

        elif self.circle_or_cross == 1:
            for pose in trajectory:
                # 创建相对于中心点的姿态
                target_pose = PoseStamped()
                target_pose.header.frame_id = center_pose.header.frame_id
                target_pose.header.stamp = rospy.Time.now()
                
                # 设置位置（相对于中心点的偏移）
                target_pose.pose.position.x = center_pose.pose.position.x + pose.pose.position.x
                target_pose.pose.position.y = center_pose.pose.position.y + pose.pose.position.y
                target_pose.pose.position.z = center_pose.pose.position.z + pose.pose.position.z
                
                target_pose.pose.orientation = self.home_pose.pose.orientation
                
                self.target_pose_pub.publish(target_pose)
                
                rospy.sleep(1)
            
            self.target_pose_pub.publish(self.home_pose)

        rospy.loginfo("Trajectory completed")

if __name__ == '__main__':
    try:
        drawer = TicTacToeDrawer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass