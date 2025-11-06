#ifndef AABB_GRASP_PLANNER_HPP
#define AABB_GRASP_PLANNER_HPP

#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <visualization_msgs/Marker.h>
#include <ros/ros.h>
#include <algorithm>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <vector>
#include <algorithm>
#include <cmath>

namespace grasp_planner {

class EfficientOutlierRemoval {
public:
    EfficientOutlierRemoval() : mean_k_(50), stddev_thresh_(1.5) {}
    
    void setMeanK(int k) { mean_k_ = k; }
    void setStddevMulThresh(float thresh) { stddev_thresh_ = thresh; }
    
    void filter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input, 
                pcl::PointCloud<pcl::PointXYZ>::Ptr& output) {
        // 1. 构建KD树用于快速邻域搜索
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(input);
        
        // 2. 预分配内存
        std::vector<float> avg_distances(input->size());
        std::vector<int> indices(mean_k_);
        std::vector<float> distances(mean_k_);
        
        // 3. 第一遍：计算每个点的平均邻域距离
        #pragma omp parallel for
        for (size_t i = 0; i < input->size(); ++i) {
            if (kdtree.nearestKSearch(input->points[i], mean_k_, indices, distances) > 0) {
                float sum = 0.0f;
                for (float d : distances) sum += std::sqrt(d);
                avg_distances[i] = sum / mean_k_;
            }
        }
        
        // 4. 计算全局统计量
        float mean = 0.0f, stddev = 0.0f;
        for (float d : avg_distances) mean += d;
        mean /= avg_distances.size();
        
        for (float d : avg_distances) stddev += (d - mean) * (d - mean);
        stddev = std::sqrt(stddev / avg_distances.size());
        
        float threshold = mean + stddev_thresh_ * stddev;
        
        // 5. 第二遍：过滤离群点
        output->reserve(input->size());
        for (size_t i = 0; i < input->size(); ++i) {
            if (avg_distances[i] <= threshold) {
                output->push_back(input->points[i]);
            }
        }
    }

private:
    int mean_k_;
    float stddev_thresh_;
};

class AABBGraspPlanner {
public:
    /**
     * @brief 构造函数，接收外部 NodeHandle
     * @param nh 外部传入的 NodeHandle
     */
    explicit AABBGraspPlanner(ros::NodeHandle& nh) : nh_(nh) {
        aabb_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("aabb_marker", 1);
        grasp_pose_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("grasp_pose_marker", 1);
        grasp_pose_posestamp_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("grasp_pose_posestamp", 1);
    }
    
    /**
     * @brief 计算点云的 AABB 盒并生成抓取姿态（带可视化）
     * @param cloud 输入点云（必须是单个物体的点云）
     * @param gripper_max_opening 机械爪的最大张开距离（单位：米）
     * @param frame_id 点云的坐标系（默认："camera_color_optical_frame"）
     * @param visualize 是否可视化 AABB 盒和抓取姿态（默认：true）
     * @return 抓取姿态（PoseStamped），如果不可抓取则返回空的 PoseStamped
     */
    geometry_msgs::PoseStamped computeGraspPose(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
        float gripper_max_opening,
        const std::string &frame_id = "camera_color_optical_frame",
        bool visualize = true
    ) {
        geometry_msgs::PoseStamped grasp_pose;
        
        // 检查点云是否为空
        if (cloud->empty()) {
            ROS_WARN("Input cloud is empty!");
            return grasp_pose;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud_copy(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);//滤波后的结果

        pcl::copyPointCloud(*cloud, *src_cloud_copy);
        
        EfficientOutlierRemoval filter;
        filter.setMeanK(50);
        filter.setStddevMulThresh(1.5);
        filter.filter(src_cloud_copy, cloud_filtered);

        // 计算点云的中心点
        Eigen::Vector4f pcaCentroid;
        pcl::compute3DCentroid(*cloud_filtered, pcaCentroid);
        
        // 计算协方差矩阵
        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrixNormalized(*cloud_filtered, pcaCentroid, covariance);
        
        // 计算特征值和特征向量
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
        Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
        Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
        
        // 确保特征向量构成右手坐标系
        eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
        eigenVectorsPCA.col(1) = eigenVectorsPCA.col(2).cross(eigenVectorsPCA.col(0));
        eigenVectorsPCA.col(0) = eigenVectorsPCA.col(1).cross(eigenVectorsPCA.col(2));
        
        // 归一化特征向量
        eigenVectorsPCA.col(0).normalize();
        eigenVectorsPCA.col(1).normalize();
        eigenVectorsPCA.col(2).normalize();
        
        // 按特征值降序排列特征向量
        std::vector<int> indices(3);
        indices[0] = 0;
        indices[1] = 1;
        indices[2] = 2;
        
        // 对特征值和对应的特征向量索引进行排序
        for (int i = 0; i < 2; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                if (eigenValuesPCA(indices[j]) > eigenValuesPCA(indices[i])) {
                    std::swap(indices[i], indices[j]);
                }
            }
        }
        
        // 构建排序后的特征向量矩阵
        Eigen::Matrix3f sortedEigenVectors;
        for (int i = 0; i < 3; ++i) {
            sortedEigenVectors.col(i) = eigenVectorsPCA.col(indices[i]);
        }
        
        // 确保是右手坐标系（行列式为正）
        if (sortedEigenVectors.determinant() < 0) {
            sortedEigenVectors.col(2) = -sortedEigenVectors.col(2);
        }
        
        eigenVectorsPCA = sortedEigenVectors;
        
        // 构建变换矩阵
        Eigen::Matrix4f tm = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f tm_inv = Eigen::Matrix4f::Identity();
        tm.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
        tm.block<3, 1>(0, 3) = -1.0f * (eigenVectorsPCA.transpose()) * (pcaCentroid.head<3>());
        tm_inv = tm.inverse();
        
        // 将点云变换到主方向坐标系
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*cloud_filtered, *transformedCloud, tm);
        
        // 计算变换后点云的AABB盒
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*transformedCloud, min_pt, max_pt);
        
        // 计算AABB盒尺寸
        const float aabb_length_x = max_pt.x - min_pt.x;
        const float aabb_width_y = max_pt.y - min_pt.y;
        const float aabb_height_z = max_pt.z - min_pt.z;
        
        // 输出调试信息
        ROS_INFO("AABB dimensions: X=%.3f, Y=%.3f, Z=%.3f", aabb_length_x, aabb_width_y, aabb_height_z);
        ROS_INFO("Eigenvalues: %.3f, %.3f, %.3f", eigenValuesPCA(0), eigenValuesPCA(1), eigenValuesPCA(2));
        
        // 找到最短的边作为夹持方向
        float min_dimension = std::min({aabb_length_x, aabb_width_y, aabb_height_z});
        
        // 如果最短边超过机械爪的极限张开距离，则无法抓取
        if (min_dimension > gripper_max_opening) {
            ROS_WARN("Object is too wide to grasp (min_dimension = %f, gripper_max_opening = %f)",
                     min_dimension, gripper_max_opening);
            return grasp_pose;
        }
        
        // 确定哪个维度应该作为夹持方向（Z轴）
        int grasp_axis = -1;
        if (min_dimension == aabb_length_x) {
            grasp_axis = 0; // X轴是最短边
        } else if (min_dimension == aabb_width_y) {
            grasp_axis = 1; // Y轴是最短边
        } else {
            grasp_axis = 2; // Z轴是最短边
        }
        
        // 计算AABB盒中心在变换后坐标系中的位置
        Eigen::Vector3f aabb_center_local(
            (min_pt.x + max_pt.x) / 2.0f,
            (min_pt.y + max_pt.y) / 2.0f,
            (min_pt.z + max_pt.z) / 2.0f
        );
        
        // 将AABB盒中心转换回原始坐标系
        Eigen::Vector3f aabb_center_global = tm_inv.block<3, 3>(0, 0) * aabb_center_local + tm_inv.block<3, 1>(0, 3);
        
        // 设置抓取姿态的 header
        grasp_pose.header.frame_id = frame_id;
        grasp_pose.header.stamp = ros::Time::now();
        
        // 设置抓取位置（AABB 盒中心）
        grasp_pose.pose.position.x = aabb_center_global[0];
        grasp_pose.pose.position.y = aabb_center_global[1];
        grasp_pose.pose.position.z = aabb_center_global[2];
        
        // 设置抓取方向（基于主方向）
        Eigen::Matrix3f rotation_matrix = tm_inv.block<3, 3>(0, 0);
        
        // 根据夹持方向调整坐标系
        // 默认假设机械爪的夹持方向是Z轴
        if (grasp_axis == 0) {
            // 如果X轴应该作为夹持方向，调整坐标系
            // 新的Z轴 = 原来的X轴（夹持方向）
            // 新的X轴 = 原来的Y轴
            // 新的Y轴 = 原来的Z轴
            Eigen::Matrix3f adjusted_rotation;
            adjusted_rotation.col(0) = rotation_matrix.col(1); // X = 原来的Y
            adjusted_rotation.col(1) = rotation_matrix.col(2); // Y = 原来的Z
            adjusted_rotation.col(2) = rotation_matrix.col(0); // Z = 原来的X（夹持方向）
            rotation_matrix = adjusted_rotation;
        } else if (grasp_axis == 1) {
            // 如果Y轴应该作为夹持方向，调整坐标系
            // 新的Z轴 = 原来的Y轴（夹持方向）
            // 新的X轴 = 原来的Z轴
            // 新的Y轴 = 原来的X轴
            Eigen::Matrix3f adjusted_rotation;
            adjusted_rotation.col(0) = rotation_matrix.col(2); // X = 原来的Z
            adjusted_rotation.col(1) = rotation_matrix.col(0); // Y = 原来的X
            adjusted_rotation.col(2) = rotation_matrix.col(1); // Z = 原来的Y（夹持方向）
            rotation_matrix = adjusted_rotation;
        }
        // 如果grasp_axis == 2，不需要调整（Z轴已经是夹持方向）
        
        // 确保Z轴朝向远离原点的方向
        Eigen::Vector3f z_axis = rotation_matrix.col(2);
        Eigen::Vector3f position_vector = aabb_center_global;
        
        // 计算Z轴与位置向量的点积
        float dot_product = z_axis.dot(position_vector.normalized());
        
        // 如果点积为负，说明Z轴朝向原点方向，需要翻转
        if (dot_product < 0) {
            ROS_INFO("Flipping Z axis to point away from origin (dot product: %.3f)", dot_product);
            rotation_matrix.col(2) = -rotation_matrix.col(2);
            // 为了保持右手坐标系，需要翻转一个其他轴
            rotation_matrix.col(0) = -rotation_matrix.col(0);
        }
        
        // 验证旋转矩阵是正交的右手坐标系
        float determinant = rotation_matrix.determinant();
        if (std::abs(determinant - 1.0f) > 0.1f) {
            ROS_WARN("Rotation matrix determinant is not 1.0 (%.3f), correcting...", determinant);
            // 使用QR分解重新正交化
            Eigen::HouseholderQR<Eigen::Matrix3f> qr(rotation_matrix);
            rotation_matrix = qr.householderQ();
            // 确保是右手系
            if (rotation_matrix.determinant() < 0) {
                rotation_matrix.col(2) = -rotation_matrix.col(2);
            }
        }
        
        Eigen::Quaternionf quat(rotation_matrix);
        quat.normalize();
        
        grasp_pose.pose.orientation.x = quat.x();
        grasp_pose.pose.orientation.y = quat.y();
        grasp_pose.pose.orientation.z = quat.z();
        grasp_pose.pose.orientation.w = quat.w();
        
        // 可视化 AABB 盒和抓取姿态
        if (visualize) {
            visualizeAABB(min_pt, max_pt, tm_inv, frame_id);
            visualizeGraspPose(grasp_pose);
        }
        
        grasp_pose_posestamp_pub_.publish(grasp_pose);
        return grasp_pose;
    }

private:
    ros::NodeHandle& nh_;
    ros::Publisher aabb_marker_pub_;
    ros::Publisher grasp_pose_marker_pub_;
    ros::Publisher grasp_pose_posestamp_pub_;

    /**
     * @brief 可视化 AABB 盒（在 RViz 中显示）
     * @param min_pt 变换后坐标系中的AABB盒最小点
     * @param max_pt 变换后坐标系中的AABB盒最大点
     * @param tm_inv 逆变换矩阵
     * @param frame_id 坐标系
     */
    void visualizeAABB(
        const pcl::PointXYZ &min_pt,
        const pcl::PointXYZ &max_pt,
        const Eigen::Matrix4f &tm_inv,
        const std::string &frame_id
    ) {
        visualization_msgs::Marker aabb_marker;
        aabb_marker.header.frame_id = frame_id;
        aabb_marker.header.stamp = ros::Time::now();
        aabb_marker.ns = "aabb_box";
        aabb_marker.id = 0;
        aabb_marker.type = visualization_msgs::Marker::CUBE;
        aabb_marker.action = visualization_msgs::Marker::ADD;
        
        // 计算AABB盒中心在变换后坐标系中的位置
        Eigen::Vector3f aabb_center_local(
            (min_pt.x + max_pt.x) / 2.0f,
            (min_pt.y + max_pt.y) / 2.0f,
            (min_pt.z + max_pt.z) / 2.0f
        );
        
        // 将AABB盒中心转换回原始坐标系
        Eigen::Vector3f aabb_center_global = tm_inv.block<3, 3>(0, 0) * aabb_center_local + tm_inv.block<3, 1>(0, 3);
        
        // 设置 AABB 盒的位置
        aabb_marker.pose.position.x = aabb_center_global[0];
        aabb_marker.pose.position.y = aabb_center_global[1];
        aabb_marker.pose.position.z = aabb_center_global[2];
        
        // 设置 AABB 盒的方向（基于主方向）
        Eigen::Matrix3f rotation_matrix = tm_inv.block<3, 3>(0, 0);
        Eigen::Quaternionf quat(rotation_matrix);
        aabb_marker.pose.orientation.x = quat.x();
        aabb_marker.pose.orientation.y = quat.y();
        aabb_marker.pose.orientation.z = quat.z();
        aabb_marker.pose.orientation.w = quat.w();
        
        // 设置 AABB 盒的尺寸
        aabb_marker.scale.x = max_pt.x - min_pt.x;
        aabb_marker.scale.y = max_pt.y - min_pt.y;
        aabb_marker.scale.z = max_pt.z - min_pt.z;
        
        // 设置 AABB 盒的颜色（红色半透明）
        aabb_marker.color.r = 1.0f;
        aabb_marker.color.g = 0.0f;
        aabb_marker.color.b = 0.0f;
        aabb_marker.color.a = 0.5f;
        
        aabb_marker.lifetime = ros::Duration(5.0); // 显示 5 秒
        aabb_marker_pub_.publish(aabb_marker);
    }

    /**
     * @brief 可视化抓取姿态（在 RViz 中显示）
     * @param grasp_pose 抓取姿态
     */
    void visualizeGraspPose(const geometry_msgs::PoseStamped &grasp_pose) {
        visualization_msgs::Marker axes_marker;
        axes_marker.header = grasp_pose.header;
        axes_marker.ns = "grasp_axes";
        axes_marker.id = 2;
        axes_marker.type = visualization_msgs::Marker::LINE_LIST;
        axes_marker.action = visualization_msgs::Marker::ADD;
        axes_marker.pose = grasp_pose.pose;

        // 设置轴的粗细
        axes_marker.scale.x = 0.005; // 线宽

        // 每条轴的长度
        double axis_length = 0.1;

        // 定义三个轴的起点和终点（局部坐标系下）
        geometry_msgs::Point x_start, x_end, y_start, y_end, z_start, z_end;
        x_start.x = 0; x_start.y = 0; x_start.z = 0;
        x_end.x = axis_length; x_end.y = 0; x_end.z = 0;

        y_start.x = 0; y_start.y = 0; y_start.z = 0;
        y_end.x = 0; y_end.y = axis_length; y_end.z = 0;

        z_start.x = 0; z_start.y = 0; z_start.z = 0;
        z_end.x = 0; z_end.y = 0; z_end.z = axis_length;

        // 将点添加到 marker 中
        axes_marker.points.push_back(x_start);
        axes_marker.points.push_back(x_end);
        axes_marker.points.push_back(y_start);
        axes_marker.points.push_back(y_end);
        axes_marker.points.push_back(z_start);
        axes_marker.points.push_back(z_end);

        // 设置颜色（X:红色, Y:绿色, Z:蓝色）
        std_msgs::ColorRGBA x_color, y_color, z_color;
        x_color.r = 1.0; x_color.g = 0.0; x_color.b = 0.0; x_color.a = 1.0;
        y_color.r = 0.0; y_color.g = 1.0; y_color.b = 0.0; y_color.a = 1.0;
        z_color.r = 0.0; z_color.g = 0.0; z_color.b = 1.0; z_color.a = 1.0;

        axes_marker.colors.push_back(x_color);
        axes_marker.colors.push_back(x_color);
        axes_marker.colors.push_back(y_color);
        axes_marker.colors.push_back(y_color);
        axes_marker.colors.push_back(z_color);
        axes_marker.colors.push_back(z_color);

        axes_marker.lifetime = ros::Duration(5.0);
        grasp_pose_marker_pub_.publish(axes_marker);
    }
};

} // namespace grasp_planner

#endif // AABB_GRASP_PLANNER_HPP
