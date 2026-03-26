import argparse
import json
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import time

import cv2
import numpy as np

# ROS2 TF & 图像（可选：未安装 ROS2 时仅跳过 TF/相机）
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import TransformStamped, PoseStamped
    from tf2_ros import TransformBroadcaster
    from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
    from std_msgs.msg import Header
    from visualization_msgs.msg import Marker, MarkerArray
    from builtin_interfaces.msg import Duration as BuiltinDuration
    from cv_bridge import CvBridge

    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False
    rclpy = None
    Node = None
    TransformStamped = None
    TransformBroadcaster = None
    Image = None
    CameraInfo = None
    CvBridge = None
import open3d as o3d
import pyrealsense2 as rs
import torch
from PIL import Image as PILImage

# ------------------- 机械臂相关配置（与 heihei.py 一致）-------------------
ROBOT_TYPE = "piper"
CAN_CHANNEL = "can0"
TCP_OFFSET = [0.0, 0.0, 0.10, 0.0, 0.0, 0.0]  # TCP Z 轴 +14cm
SPEED_PERCENT = 30
MOTION_TIMEOUT = 15.0
POSE_SAVE_FILE = "/home/agilex/.nanobot/workspace/skills/grab_skill/sam3/recorded_j6_pose.json"  # J6 法兰位姿保存路径（与 heihei 共用）
GRIPPER_MAX_WIDTH = 0.1  # 夹爪最大开口 (m)
GRIPPER_MIN_WIDTH = 0.0  # 夹爪最小开口 (m)
GRIPPER_FORCE = 2.0  # 夹爪夹持力 (N)

# 使用 RealSense ROS2 驱动时，深度图单位通常为毫米（uint16）
DEPTH_SCALE_ROS = 0.001  # 每个深度单位对应的米数（mm -> m）

# J6 到相机坐标系的变换 [x, y, z, qx, qy, qz, qw]（与 j6_pose_tf_node.py 一致）
# 即 T_j6_to_camera，用于 T_result = T_j6 @ T_j6_to_camera（相机在基座系下）
J6_TO_CAMERA = [
    -0.06657143797304754,
    -0.007181201910632633,
    0.033259474804825814,
    -0.1665998530,
    0.1479507149,
    -0.6490963521,
    0.7273437981,
]

# ROS TF 坐标系名称
TF_FRAME_BASE = "base_link"
TF_FRAME_J6_FLANGE = "piper_j6_flange"
TF_FRAME_CAMERA = "camera_link"
TF_FRAME_GRASP_TARGET = "grasp_target"

# RealSense 自身 TF：camera_link -> camera_color_optical_frame
# 来源：ros2 run tf2_ros tf2_echo camera_link camera_color_optical_frame
T_CAMLINK_TO_COLOR_OPTICAL = [
    0.0,
    0.015,
    0.0,
    0.5,
    -0.5,
    0.501,
    -0.5,
]

# RealSense 光学坐标系 frame 名称（与 /camera/camera/aligned_depth_to_color/image_raw 一致）
CAMERA_OPTICAL_FRAME = "camera_color_optical_frame"

# RealSense ROS2 相机节点：程序内自动拉起/退出
CAMERA_WS_PATH = "/home/agilex/ros_workspace/camera_ws"
CAMERA_SETUP_SCRIPT = os.path.join(CAMERA_WS_PATH, "install", "setup.sh")
CAMERA_LAUNCH_CMD = "ros2 launch realsense2_camera rs_align_depth_launch.py"
CAMERA_LAUNCH_SHELL_CMD = (
    f"source {CAMERA_SETUP_SCRIPT} && {CAMERA_LAUNCH_CMD}"
)


def make_pointcloud2(points_xyz, colors_rgb, frame_id, stamp):
    """
    将 (N,3) XYZ 和 (N,3) RGB 数组转换为 PointCloud2（XYZRGB），发布在给定坐标系下。
    """
    if points_xyz.size == 0:
        return None
    pts = np.asarray(points_xyz, dtype=np.float32)
    cols = np.clip(np.asarray(colors_rgb, dtype=np.float32) * 255.0, 0, 255).astype(
        np.uint8
    )
    if pts.shape[0] != cols.shape[0]:
        n = min(pts.shape[0], cols.shape[0])
        pts = pts[:n]
        cols = cols[:n]

    # 按 PCL/ROS 约定打包为 x,y,z,rgb（rgb 为 float32，内部存放 uint32 的 BGR）
    r = cols[:, 0].astype(np.uint32)
    g = cols[:, 1].astype(np.uint32)
    b = cols[:, 2].astype(np.uint32)
    rgb_uint32 = (r << 16) | (g << 8) | b
    rgb_float = rgb_uint32.view(np.float32)

    # 组装结构化数组：x,y,z,rgb
    cloud_arr = np.zeros(
        pts.shape[0],
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("rgb", np.float32),
        ],
    )
    cloud_arr["x"] = pts[:, 0]
    cloud_arr["y"] = pts[:, 1]
    cloud_arr["z"] = pts[:, 2]
    cloud_arr["rgb"] = rgb_float

    msg = PointCloud2()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = cloud_arr.shape[0]
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16  # 3*4 (xyz) + 4 (rgb)
    msg.row_step = msg.point_step * cloud_arr.shape[0]
    msg.is_dense = True
    msg.data = cloud_arr.tobytes()
    return msg


def start_camera_launch():
    """
    在子进程中拉起 RealSense ROS2 节点：source camera_ws/setup.sh && ros2 launch ...
    返回 subprocess.Popen 实例，退出时需调用 stop_camera_launch(proc)。
    """
    if not os.path.isfile(CAMERA_SETUP_SCRIPT):
        print(f"[Camera] 未找到 setup 脚本: {CAMERA_SETUP_SCRIPT}，请检查路径")
        return None
    print("[Camera] 正在拉起 RealSense ROS2 节点...")
    try:
        proc = subprocess.Popen(
            CAMERA_LAUNCH_SHELL_CMD,
            shell=True,
            executable="/bin/bash",
            cwd=CAMERA_WS_PATH,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        print("[Camera] RealSense launch 已启动，等待数秒使节点就绪...")
        time.sleep(5)
        if proc.poll() is not None:
            err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            print(f"[Camera] launch 进程已异常退出: {err}")
            return None
        return proc
    except Exception as e:
        print(f"[Camera] 启动失败: {e}")
        return None


def stop_camera_launch(proc, timeout=10):
    """终止由 start_camera_launch 拉起的进程（及其进程组）。"""
    if proc is None:
        return
    try:
        if proc.poll() is None:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            proc.wait(timeout=timeout)
    except ProcessLookupError:
        pass
    except Exception as e:
        print(f"[Camera] 关闭 launch 时出错: {e}")
        try:
            proc.kill()
            proc.wait(timeout=3)
        except Exception:
            pass


# 无保存文件时的 fallback：相机在基座系下的齐次矩阵（仅当未按 D 记录过时使用）
T_BASE_CAM_FALLBACK = np.array(
    [
        [-0.043365, -0.977782, 0.205091, 0.196176],
        [-0.989188, 0.013236, -0.146053, 0.238188],
        [0.140093, -0.209207, -0.967784, 0.219417],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def euler_to_rotation_matrix(roll, pitch, yaw):
    """欧拉角 Z-Y-X (rad) -> 3x3 旋转矩阵"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return R


def rotation_matrix_to_euler(R):
    """3x3 旋转矩阵 -> Z-Y-X 欧拉角 (roll, pitch, yaw) rad"""
    sy = -R[2, 0]
    pitch = np.arcsin(np.clip(sy, -1.0, 1.0))
    cp = np.cos(pitch)
    if np.abs(cp) > 1e-6:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw = 0.0
        roll = np.arctan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


def pose_to_homogeneous_matrix(pose):
    """[x,y,z,roll,pitch,yaw] -> 4x4 齐次矩阵"""
    x, y, z = pose[0], pose[1], pose[2]
    roll, pitch, yaw = pose[3], pose[4], pose[5]
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def homogeneous_to_pose(T):
    """4x4 齐次矩阵 -> [x,y,z,roll,pitch,yaw] (m, rad)"""
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    R = T[:3, :3]
    roll, pitch, yaw = rotation_matrix_to_euler(R)
    return [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)]


def quat_pose_to_homogeneous_matrix(pose7):
    """[x, y, z, qx, qy, qz, qw] -> 4x4 齐次矩阵"""
    x, y, z = pose7[0], pose7[1], pose7[2]
    qx, qy, qz, qw = pose7[3], pose7[4], pose7[5], pose7[6]
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """四元数 (x, y, z, w) -> 3x3 旋转矩阵"""
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float32,
    )


def rotation_matrix_to_quaternion(R):
    """3x3 旋转矩阵 -> 四元数 (qx, qy, qz, qw)"""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return (float(qx), float(qy), float(qz), float(qw))


def load_T_result_from_saved_j6():
    """
    从 POSE_SAVE_FILE 加载 J6 法兰位姿，与 J6_TO_CAMERA 相乘得到 T_result（相机在基座系下）。
    若文件不存在或解析失败则返回 None，调用方用 T_BASE_CAM_FALLBACK。
    """
    if not os.path.exists(POSE_SAVE_FILE):
        return None
    try:
        with open(POSE_SAVE_FILE, "r", encoding="utf-8") as f:
            recorded_j6_pose = json.load(f)
        T_j6 = pose_to_homogeneous_matrix(recorded_j6_pose)
        T_j6_to_camera = quat_pose_to_homogeneous_matrix(J6_TO_CAMERA)
        T_result = np.matmul(T_j6, T_j6_to_camera).astype(np.float32)
        return T_result
    except Exception as e:
        print(f"[T_result] 加载 J6 位姿失败: {e}，将使用 fallback 矩阵")
        return None


def init_robot_and_gripper():
    """
    初始化机械臂与夹爪（与 heihei 一致）。失败时返回 (None, None)，主程序可仅做视觉。
    """
    try:
        from pyAgxArm import create_agx_arm_config, AgxArmFactory

        cfg = create_agx_arm_config(
            robot=ROBOT_TYPE,
            comm="can",
            channel=CAN_CHANNEL,
            bitrate=1000000,
            auto_connect=True,
        )
        robot = AgxArmFactory.create_arm(cfg)
        end_effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        robot.connect(start_read_thread=True)
        time.sleep(0.5)
        if not robot.is_ok():
            raise ConnectionError("机械臂通信异常")
        if not end_effector.is_ok():
            raise ConnectionError("夹爪通信异常")
        robot.set_tcp_offset(TCP_OFFSET)
        # 使能
        for _ in range(1000):
            if robot.enable(joint_index=255):
                break
            time.sleep(0.01)
        else:
            raise TimeoutError("关节使能超时")
        robot.set_speed_percent(SPEED_PERCENT)
        robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.P)
        print("[Robot] 机械臂与夹爪初始化完成")
        return robot, end_effector
    except Exception as e:
        print(f"[Robot] 初始化失败（将仅做视觉，不控制机械臂）: {e}")
        return None, None


def compute_grasp_from_pca_aabb(pts_base, gripper_max_opening=0.1):
    """
    参考 demo.hpp / AABBGraspPlanner：3D PCA → 主轴系 AABB → 最短边作为夹持方向(Z) → 调整坐标系 →
    Z 轴朝“远离原点”的方向（与 C++ 实现一致）。
    输入点云应为滤波后、基座系下的点云；返回 (center, R_grasp) 或 None。
    """
    pts = np.asarray(pts_base, dtype=np.float64)
    if pts.shape[0] < 4:
        return None
    # 1) 中心点（对应 pcl::compute3DCentroid）
    pca_centroid = np.mean(pts, axis=0)
    # 2) 协方差矩阵（PCL 归一化：除以 n-1）
    centered = pts - pca_centroid
    covariance = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    # 3) 特征值、特征向量（对应 Eigen::SelfAdjointEigenSolver）
    eigen_values, eigen_vectors = np.linalg.eigh(covariance)
    # 4) 确保特征向量构成右手坐标系（与 demo 完全一致）
    ev = eigen_vectors
    ev = ev.copy()
    ev[:, 2] = np.cross(ev[:, 0], ev[:, 1])
    ev[:, 1] = np.cross(ev[:, 2], ev[:, 0])
    ev[:, 0] = np.cross(ev[:, 1], ev[:, 2])
    for i in range(3):
        n = np.linalg.norm(ev[:, i])
        if n > 1e-10:
            ev[:, i] = ev[:, i] / n
    # 5) 按特征值降序排列特征向量（demo: indices 使 eigenValuesPCA(indices[j]) > eigenValuesPCA(indices[i]) 则 swap）
    indices = np.argsort(eigen_values)[::-1]
    sorted_eigen_vectors = ev[:, indices].copy()
    # 6) 确保右手系（行列式为正）
    if np.linalg.det(sorted_eigen_vectors) < 0:
        sorted_eigen_vectors[:, 2] = -sorted_eigen_vectors[:, 2]
    # 7) 变换矩阵：tm 的 3x3 = R^T，平移 = -R^T @ centroid；local = R^T @ (p - centroid)
    R = sorted_eigen_vectors
    local_pts = (R.T @ (pts - pca_centroid).T).T
    # 8) 变换后点云的 AABB（对应 getMinMax3D）
    min_pt = np.min(local_pts, axis=0)
    max_pt = np.max(local_pts, axis=0)
    aabb_length_x = float(max_pt[0] - min_pt[0])
    aabb_width_y = float(max_pt[1] - min_pt[1])
    aabb_height_z = float(max_pt[2] - min_pt[2])
    # 9) 最短边作为夹持方向
    min_dimension = min(aabb_length_x, aabb_width_y, aabb_height_z)
    if min_dimension > gripper_max_opening:
        return None
    if min_dimension == aabb_length_x:
        grasp_axis = 0
    elif min_dimension == aabb_width_y:
        grasp_axis = 1
    else:
        grasp_axis = 2
    # 10) AABB 中心转回世界系（tm_inv 的 3x3 = R，平移 = centroid）
    aabb_center_local = (min_pt + max_pt) * 0.5
    aabb_center_global = R @ aabb_center_local + pca_centroid
    center = np.array(aabb_center_global, dtype=np.float32)
    # 11) 抓取方向 = tm_inv 的 3x3 = R（列为主方向）
    rotation_matrix = R.copy()
    # 12) 根据夹持方向调整坐标系：机械爪夹持方向为 Z（与 demo 一致）
    if grasp_axis == 0:
        # 新 Z = 原 X（夹持），新 X = 原 Y，新 Y = 原 Z
        adjusted = np.column_stack([
            rotation_matrix[:, 1],
            rotation_matrix[:, 2],
            rotation_matrix[:, 0],
        ])
        rotation_matrix = adjusted
    elif grasp_axis == 1:
        # 新 Z = 原 Y（夹持），新 X = 原 Z，新 Y = 原 X
        adjusted = np.column_stack([
            rotation_matrix[:, 2],
            rotation_matrix[:, 0],
            rotation_matrix[:, 1],
        ])
        rotation_matrix = adjusted
    # grasp_axis == 2 不调整
    # 13) 确保 Z 轴“远离原点”方向（与 demo.hpp 一致）
    z_axis = rotation_matrix[:, 2]
    position_vector = center.astype(np.float64)
    norm_pos = np.linalg.norm(position_vector)
    if norm_pos > 1e-8:
        dot_product = float(z_axis.dot(position_vector / norm_pos))
        if dot_product < 0.0:
            rotation_matrix[:, 2] = -rotation_matrix[:, 2]
            rotation_matrix[:, 0] = -rotation_matrix[:, 0]
    # 14) 正交化（demo 中 determinant 偏离 1 时用 QR 修正）
    det = np.linalg.det(rotation_matrix)
    if abs(det - 1.0) > 0.1:
        Q, _ = np.linalg.qr(rotation_matrix)
        rotation_matrix = Q.copy()
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 2] = -rotation_matrix[:, 2]
    R_grasp = np.array(rotation_matrix, dtype=np.float32)
    return center, R_grasp


def align_grasp_x_toward_base(R_grasp, T_cam_to_base):
    """
    调整抓取姿态旋转矩阵，使 x 轴朝向 base_link 一侧。
    T_cam_to_base: 4x4 齐次矩阵，将点从 camera_color_optical_frame 变换到 base_link。
    若当前 x 轴背向 base，则翻转 x、y 以保持右手系。
    """
    T_base_to_cam = np.linalg.inv(np.asarray(T_cam_to_base, dtype=np.float64))
    base_origin_in_cam = T_base_to_cam[:3, 3]
    n = np.linalg.norm(base_origin_in_cam)
    if n < 1e-6:
        return np.asarray(R_grasp, dtype=np.float32)
    dir_to_base = base_origin_in_cam / n
    R = np.asarray(R_grasp, dtype=np.float64).copy()
    if np.dot(R[:, 0], dir_to_base) < 0.0:
        R[:, 0] = -R[:, 0]
        R[:, 1] = -R[:, 1]  # 保持右手系
    return R.astype(np.float32)


def save_pose_to_file(j6_pose):
    """将 J6 法兰位姿保存到 POSE_SAVE_FILE"""
    try:
        with open(POSE_SAVE_FILE, "w", encoding="utf-8") as f:
            json.dump(j6_pose, f, indent=4)
        print(f"[Robot] J6 法兰位姿已保存: {POSE_SAVE_FILE}")
    except Exception as e:
        print(f"[Robot] 保存位姿失败: {e}")


def safe_shutdown_robot(robot, end_effector):
    """安全失能机械臂与夹爪"""
    if robot is not None:
        try:
            robot.disable(joint_index=255)
            print("[Robot] 机械臂已失能")
        except Exception:
            pass
    if end_effector is not None:
        try:
            end_effector.disable_gripper()
            print("[Robot] 夹爪已失能")
        except Exception:
            pass

def sam_results_to_masklet_outputs(results, img_h, img_w):
    """
    将 SAM3 的结果格式转换为 visualization_utils.render_masklet_frame 所需格式。
    results 需要包含:
      - "scores": list[Tensor]
      - "boxes":  list[Tensor]，XYXY 像素坐标
      - "masks":  list[Tensor]，形状 [1, H, W]
    """
    outputs = {
        "out_boxes_xywh": [],
        "out_probs": [],
        "out_obj_ids": [],
        "out_binary_masks": [],
    }

    num_objs = len(results.get("scores", []))
    for i in range(num_objs):
        score = results["scores"][i].item()
        box_xyxy = results["boxes"][i].cpu().numpy()
        x1, y1, x2, y2 = box_xyxy

        # 归一化 XYWH
        x = x1 / img_w
        y = y1 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        mask_tensor = results["masks"][i].squeeze(0).cpu()
        mask_np = mask_tensor.numpy()
        # 如果是概率图，简单阈值为 0.5
        if mask_np.dtype != np.bool_:
            mask_np = mask_np > 0.5

        outputs["out_boxes_xywh"].append([x, y, w, h])
        outputs["out_probs"].append(score)
        outputs["out_obj_ids"].append(i)
        outputs["out_binary_masks"].append(mask_np.astype(np.uint8))

    return outputs


def sam_worker(
    input_q: mp.Queue,
    output_q: mp.Queue,
    prompt_q: mp.Queue,
    initial_prompt: str = "person",
):
    """
    子进程：专门跑 SAM3 推理（在 GPU 上），避免与 RealSense 的 C++/CUDA 冲突。
    """
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.visualization_utils import render_masklet_frame

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM Worker] 使用设备: {device}")

    checkpoint_path = (
        "/home/agilex/.cache/modelscope/hub/models/facebook/sam3/sam3.pt"
    )
    print(f"[SAM Worker] 加载 SAM3 模型: {checkpoint_path}")
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
    )
    processor = Sam3Processor(model, confidence_threshold=0.5)
    print("[SAM Worker] 模型加载完成，开始等待图像帧...")

    current_prompt = initial_prompt
    print(f"[SAM Worker] 初始文本提示: '{current_prompt}'")

    with torch.no_grad():
        while True:
            frame_rgb = input_q.get()
            if frame_rgb is None:
                print("[SAM Worker] 收到退出信号，结束进程。")
                break

            if not isinstance(frame_rgb, np.ndarray):
                continue

            # 无阻塞检查是否有新的 prompt
            try:
                while True:
                    new_prompt = prompt_q.get_nowait()
                    if isinstance(new_prompt, str) and new_prompt.strip():
                        current_prompt = new_prompt.strip()
                        print(f"[SAM Worker] 更新文本提示: '{current_prompt}'")
            except queue.Empty:
                pass

            h, w = frame_rgb.shape[:2]
            pil_image = PILImage.fromarray(frame_rgb)

            try:
                state = processor.set_image(pil_image)
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(state=state, prompt=current_prompt)

                sam_outputs = sam_results_to_masklet_outputs(
                    state, img_h=h, img_w=w
                )
                overlay_rgb = render_masklet_frame(
                    frame_rgb, sam_outputs, frame_idx=None, alpha=0.5
                )

                # 为每个目标分别保留掩码，便于单独生成点云和 AABB/OBB
                per_object_masks = []
                for m in sam_outputs["out_binary_masks"]:
                    if m.shape[0] != h or m.shape[1] != w:
                        m = cv2.resize(
                            m.astype(np.uint8),
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    per_object_masks.append(m.astype(np.uint8))

                # 队列满就丢弃旧结果，只保留最新的
                try:
                    while True:
                        output_q.get_nowait()
                except queue.Empty:
                    pass
                output_q.put_nowait(
                    {
                        "overlay": overlay_rgb,
                        "masks": per_object_masks,
                        "ids": sam_outputs["out_obj_ids"],
                    }
                )
            except Exception as e:
                print(f"[SAM Worker] 推理异常: {e}")


class RealSenseAlignAdvanced:
    def __init__(
        self,
        text_prompt: str,
        sam_input_q: mp.Queue,
        sam_output_q: mp.Queue,
        prompt_q: mp.Queue,
        robot=None,
        end_effector=None,
        node=None,
        tf_broadcaster=None,
        auto_mode=False,
    ):
        # 机械臂（可选）：None 时仅视觉，不下发抓取
        self.robot = robot
        self.end_effector = end_effector
        # ROS2 TF 发布（可选）
        self._node = node
        self._tf_broadcaster = tf_broadcaster
        self._T_j6_to_camera = quat_pose_to_homogeneous_matrix(J6_TO_CAMERA)
        # RealSense TF：camera_link -> camera_color_optical_frame 及其逆
        self._T_camlink_to_optical = quat_pose_to_homogeneous_matrix(
            T_CAMLINK_TO_COLOR_OPTICAL
        )
        self._T_optical_to_camlink = np.linalg.inv(self._T_camlink_to_optical)
        # 相机在基座系下：优先用 J6×J6_TO_CAMERA，无文件则用 fallback
        self.T_result = load_T_result_from_saved_j6()
        if self.T_result is None:
            self.T_result = T_BASE_CAM_FALLBACK.copy()
            print("[T_result] 使用 fallback 矩阵，建议在 heihei 中按 D 记录位姿并保存")
        else:
            print("[T_result] 已从 J6 位姿文件加载并计算 T_result")
        # 按 p 计算出的最佳抓取姿态（基座系 4x4），按 g 时下发
        self.best_grasp_T = None
        # 主臂/普通模式及记录的 J6 位姿（A/D/S/X 与 heihei 一致）
        self.is_master_mode = False
        self.recorded_j6_pose = None
        if os.path.exists(POSE_SAVE_FILE):
            try:
                with open(POSE_SAVE_FILE, "r", encoding="utf-8") as f:
                    self.recorded_j6_pose = json.load(f)
            except Exception:
                pass

        # 深度裁剪参数（用于去背景）
        self.depth_clipping_distance = 1.0  # 默认裁剪距离（米）
        self.running = False
        # 自动化模式（--auto）：状态机 self._auto_state，时间戳 self._auto_state_ts
        self.auto_mode = bool(auto_mode)
        self._auto_state = 0  # 0=等SAM就绪 1=X 2=q 3=p 4=delay2s 5=g 6=e 7=x 8=s 9=退出
        self._auto_state_ts = 0.0
        self._auto_p_retries = 0

        # 点云与抓取可视化发布（基于 /camera/camera/aligned_depth_to_color/image_raw，坐标系为 camera_color_optical_frame）
        if _ROS_AVAILABLE and self._node is not None:
            self._pcd_pub = self._node.create_publisher(
                PointCloud2, "sam3_grasp_cloud", 1
            )
            self._aabb_marker_pub = self._node.create_publisher(
                Marker, "sam3_aabb_marker", 1
            )
            self._grasp_axes_pub = self._node.create_publisher(
                Marker, "sam3_grasp_axes", 1
            )
            self._grasp_pose_pub = self._node.create_publisher(
                PoseStamped, "sam3_grasp_pose", 1
            )
            self._marker_array_pub = self._node.create_publisher(
                MarkerArray, "sam3_aabb_marker_array", 1
            )
        else:
            self._pcd_pub = None
            self._marker_array_pub = None

        # OpenCV窗口配置
        self.window_name = "RealSense + SAM3 Realtime"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        # 创建滑块控制裁剪距离（0~6米，步长0.01）
        cv2.createTrackbar(
            "Depth Clip (m)", self.window_name, 100, 600, self.on_trackbar_change
        )

        # SAM3 推理相关（通过子进程通信）
        self.text_prompt = text_prompt
        self.sam_input_q = sam_input_q
        self.sam_output_q = sam_output_q
        self.prompt_q = prompt_q
        self.last_overlay_bgr = None
        # SAM3 掩码与 ID（按目标分别保存，便于单独点云/AABB 处理）
        self.last_masks = None  # list[np.ndarray]
        self.last_obj_ids = None  # list[int]

        # 相机内参（首次帧时初始化，用于点云反投影）
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        # 帧计数器（仅用于调试打印，不影响功能）
        self.frame_counter = 0

        # ROS 相机订阅（使用 RealSense 对齐后的深度图与对应的相机内参）
        self.bridge = CvBridge() if _ROS_AVAILABLE and self._node is not None else None
        self._latest_color = None  # BGR8, np.ndarray[h,w,3]
        self._latest_depth = None  # uint16, 对齐到彩色，np.ndarray[h,w]
        self._have_cam_info = False

        if _ROS_AVAILABLE and self._node is not None and self.bridge is not None:
            # 颜色图
            self._color_sub = self._node.create_subscription(
                Image,
                "/camera/camera/color/image_raw",
                self._color_cb,
                10,
            )
            # 已对齐到彩色的深度图
            self._depth_sub = self._node.create_subscription(
                Image,
                "/camera/camera/aligned_depth_to_color/image_raw",
                self._depth_cb,
                10,
            )
            # 对齐深度图的相机内参（/camera/camera/aligned_depth_to_color/camera_info）
            self._caminfo_sub = self._node.create_subscription(
                CameraInfo,
                "/camera/camera/aligned_depth_to_color/camera_info",
                self._caminfo_cb,
                10,
            )

    def _color_cb(self, msg: Image):
        if self.bridge is None:
            return
        try:
            # RealSense ROS 默认编码为 rgb8/bgr8，这里统一转成 BGR，便于 OpenCV 使用
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._latest_color = cv_image
        except Exception as e:
            print(f"[ROS Camera] 颜色图转换失败: {e}")

    def _depth_cb(self, msg: Image):
        if self.bridge is None:
            return
        try:
            # 深度图通常为 16UC1（单位 mm），保留为 uint16，在主循环中乘以 DEPTH_SCALE_ROS 得到米
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self._latest_depth = depth
        except Exception as e:
            print(f"[ROS Camera] 深度图转换失败: {e}")

    def _caminfo_cb(self, msg: CameraInfo):
        # 只在首次收到时初始化内参
        if self._have_cam_info:
            return
        # msg.k 是长度为 9 的数组，不能直接在 if 中作为布尔判定
        if len(msg.k) == 9:
            self.fx = float(msg.k[0])
            self.fy = float(msg.k[4])
            self.cx = float(msg.k[2])
            self.cy = float(msg.k[5])
            self._have_cam_info = True
            print(
                f"[Intrinsics] fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f} (from ROS CameraInfo)"
            )

    def _publish_tf(self):
        """发布参与坐标变换的 ROS TF：base_link -> piper_j6_flange -> camera，以及 base_link -> grasp_target（若有）."""
        if not _ROS_AVAILABLE or self._node is None or self._tf_broadcaster is None:
            return
        stamp = self._node.get_clock().now().to_msg()
        # 当前 J6 位姿：有机械臂则实时读取，否则用记录的位姿
        j6_pose = None
        if self.robot is not None and not self.is_master_mode:
            fp = self.robot.get_flange_pose()
            if fp is not None and len(fp.msg) >= 6:
                j6_pose = list(fp.msg)
        if j6_pose is None and self.recorded_j6_pose is not None:
            j6_pose = self.recorded_j6_pose
        if j6_pose is not None:
            t_j6 = TransformStamped()
            t_j6.header.stamp = stamp
            t_j6.header.frame_id = TF_FRAME_BASE
            t_j6.child_frame_id = TF_FRAME_J6_FLANGE
            t_j6.transform.translation.x = float(j6_pose[0])
            t_j6.transform.translation.y = float(j6_pose[1])
            t_j6.transform.translation.z = float(j6_pose[2])
            R = euler_to_rotation_matrix(j6_pose[3], j6_pose[4], j6_pose[5])
            qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
            t_j6.transform.rotation.x = qx
            t_j6.transform.rotation.y = qy
            t_j6.transform.rotation.z = qz
            t_j6.transform.rotation.w = qw
            self._tf_broadcaster.sendTransform(t_j6)
        # 固定：piper_j6_flange -> camera_link
        # 已标定外参为：piper_j6_flange -> camera_color_optical_frame（J6_TO_CAMERA）
        # RealSense 自身发布：camera_link -> camera_color_optical_frame（T_CAMLINK_TO_COLOR_OPTICAL）
        # 这里发布的是：piper_j6_flange -> camera_link = (piper_j6_flange -> optical) * (optical -> camera_link)
        Tc = self._T_j6_to_camera @ self._T_optical_to_camlink
        tc = TransformStamped()
        tc.header.stamp = stamp
        tc.header.frame_id = TF_FRAME_J6_FLANGE
        tc.child_frame_id = TF_FRAME_CAMERA
        tc.transform.translation.x = float(Tc[0, 3])
        tc.transform.translation.y = float(Tc[1, 3])
        tc.transform.translation.z = float(Tc[2, 3])
        qx, qy, qz, qw = rotation_matrix_to_quaternion(Tc[:3, :3])
        tc.transform.rotation.x = qx
        tc.transform.rotation.y = qy
        tc.transform.rotation.z = qz
        tc.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(tc)
        # 若有最佳抓取姿态：base_link -> grasp_target
        if self.best_grasp_T is not None:
            tg = TransformStamped()
            tg.header.stamp = stamp
            tg.header.frame_id = TF_FRAME_BASE
            tg.child_frame_id = TF_FRAME_GRASP_TARGET
            tg.transform.translation.x = float(self.best_grasp_T[0, 3])
            tg.transform.translation.y = float(self.best_grasp_T[1, 3])
            tg.transform.translation.z = float(self.best_grasp_T[2, 3])
            qx, qy, qz, qw = rotation_matrix_to_quaternion(self.best_grasp_T[:3, :3])
            tg.transform.rotation.x = qx
            tg.transform.rotation.y = qy
            tg.transform.rotation.z = qz
            tg.transform.rotation.w = qw
            self._tf_broadcaster.sendTransform(tg)

    def _publish_aabb_marker(self, min_pt, max_pt, R_obb, center_obb, frame_id):
        """在 RViz 中可视化 AABB/OBB 盒（用 CUBE marker 表示）。"""
        if not _ROS_AVAILABLE or self._node is None or self._aabb_marker_pub is None:
            return
        stamp = self._node.get_clock().now().to_msg()
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = "sam3_aabb"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = float(center_obb[0])
        marker.pose.position.y = float(center_obb[1])
        marker.pose.position.z = float(center_obb[2])
        qx, qy, qz, qw = rotation_matrix_to_quaternion(R_obb)
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw

        marker.scale.x = float(max_pt[0] - min_pt[0])
        marker.scale.y = float(max_pt[1] - min_pt[1])
        marker.scale.z = float(max_pt[2] - min_pt[2])
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.4
        marker.lifetime = BuiltinDuration(sec=0, nanosec=0)  # 0 = 常驻不自动删除
        self._aabb_marker_pub.publish(marker)

    def _make_aabb_marker(self, min_pt, max_pt, R_obb, center_obb, frame_id, marker_id=0):
        """构造 AABB CUBE Marker（用于加入 MarkerArray），不发布。"""
        if not _ROS_AVAILABLE or self._node is None:
            return None
        stamp = self._node.get_clock().now().to_msg()
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = "sam3_aabb"
        marker.id = int(marker_id)
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(center_obb[0])
        marker.pose.position.y = float(center_obb[1])
        marker.pose.position.z = float(center_obb[2])
        qx, qy, qz, qw = rotation_matrix_to_quaternion(R_obb)
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        marker.scale.x = float(max_pt[0] - min_pt[0])
        marker.scale.y = float(max_pt[1] - min_pt[1])
        marker.scale.z = float(max_pt[2] - min_pt[2])
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.4
        marker.lifetime = BuiltinDuration(sec=0, nanosec=0)
        return marker

    def _publish_grasp_pose_markers(self, center, R_grasp, frame_id):
        """发布抓取姿态：PoseStamped + 轴线 Marker，供 RViz 可视化。"""
        if not _ROS_AVAILABLE or self._node is None:
            return
        stamp = self._node.get_clock().now().to_msg()

        # PoseStamped
        if self._grasp_pose_pub is not None:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = stamp
            pose_msg.header.frame_id = frame_id
            pose_msg.pose.position.x = float(center[0])
            pose_msg.pose.position.y = float(center[1])
            pose_msg.pose.position.z = float(center[2])
            qx, qy, qz, qw = rotation_matrix_to_quaternion(R_grasp)
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            self._grasp_pose_pub.publish(pose_msg)

        # 轴线 Marker（LINE_LIST）
        if self._grasp_axes_pub is not None:
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = frame_id
            marker.ns = "sam3_grasp_axes"
            marker.id = 0
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD

            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])
            qx, qy, qz, qw = rotation_matrix_to_quaternion(R_grasp)
            marker.pose.orientation.x = qx
            marker.pose.orientation.y = qy
            marker.pose.orientation.z = qz
            marker.pose.orientation.w = qw

            marker.scale.x = 0.005  # 线宽
            axis_len = 0.1

            # 定义局部坐标下的三条轴
            from geometry_msgs.msg import Point as GeoPoint  # local alias

            def p(x, y, z):
                pt = GeoPoint()
                pt.x = float(x)
                pt.y = float(y)
                pt.z = float(z)
                return pt

            x_start, x_end = p(0, 0, 0), p(axis_len, 0, 0)
            y_start, y_end = p(0, 0, 0), p(0, axis_len, 0)
            z_start, z_end = p(0, 0, 0), p(0, 0, axis_len)

            marker.points = [x_start, x_end, y_start, y_end, z_start, z_end]

            from std_msgs.msg import ColorRGBA

            xr = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            yr = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            zr = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
            marker.colors = [xr, xr, yr, yr, zr, zr]
            marker.lifetime = BuiltinDuration(sec=0, nanosec=0)  # 常驻
            self._grasp_axes_pub.publish(marker)

    def _make_grasp_axes_marker(self, center, R_grasp, frame_id, marker_id=0):
        """构造抓取轴线 LINE_LIST Marker（用于加入 MarkerArray），不发布。"""
        if not _ROS_AVAILABLE or self._node is None:
            return None
        from geometry_msgs.msg import Point as GeoPoint
        from std_msgs.msg import ColorRGBA
        stamp = self._node.get_clock().now().to_msg()
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = "sam3_grasp_axes"
        marker.id = int(marker_id)
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2])
        qx, qy, qz, qw = rotation_matrix_to_quaternion(R_grasp)
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        marker.scale.x = 0.005
        axis_len = 0.1
        def p(x, y, z):
            pt = GeoPoint()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = float(z)
            return pt
        marker.points = [p(0, 0, 0), p(axis_len, 0, 0), p(0, 0, 0), p(0, axis_len, 0), p(0, 0, 0), p(0, 0, axis_len)]
        marker.colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0), ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
        ]
        marker.lifetime = BuiltinDuration(sec=0, nanosec=0)
        return marker

    def remove_background(self, color_image, depth_image_m, clipping_dist):
        """基于深度距离移除背景，输入为 numpy 数组形式的彩色图和以米为单位的深度图。"""
        if color_image is None or depth_image_m is None:
            return None, None

        original_color_image = color_image.copy()
        bg_removed_color_image = color_image.copy()

        pixels_distance = depth_image_m
        background_mask = (pixels_distance <= 0) | (pixels_distance > clipping_dist)
        bg_removed_color_image[background_mask] = [0x99, 0x99, 0x99]

        return bg_removed_color_image, original_color_image

    def on_trackbar_change(self, value):
        """滑块回调函数，转换为米（value/100）"""
        self.depth_clipping_distance = value / 100.0

    def _enable_robot_joints(self):
        """使能所有关节（带超时）"""
        if self.robot is None:
            return False
        start_t = time.monotonic()
        while time.monotonic() - start_t < 10.0:
            if self.robot.enable(joint_index=255):
                return True
            time.sleep(0.01)
        return False

    def _switch_to_master_mode(self):
        """A：切换到主臂零力拖动模式"""
        if self.robot is None:
            print("[Robot] 未连接机械臂")
            return
        if self.is_master_mode:
            print("[Robot] 已处于主臂模式")
            return
        self.robot.disable(joint_index=255)
        time.sleep(0.2)
        self.robot.set_master_mode()
        time.sleep(1)
        self.is_master_mode = True
        print("[Robot] 已切换到主臂模式（零力拖动）")

    def _switch_to_normal_mode_and_record(self):
        """D：切换到普通模式并记录当前位姿，更新 T_result 与 recorded_j6_pose"""
        if self.robot is None:
            print("[Robot] 未连接机械臂")
            return
        if self.is_master_mode:
            self.robot.reset()
            time.sleep(0.5)
            self.robot.set_slave_mode()
            time.sleep(0.5)
            if not self._enable_robot_joints():
                print("[Robot] 使能失败")
                return
            self.robot.set_speed_percent(SPEED_PERCENT)
            self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.P)
            self.is_master_mode = False
            print("[Robot] 已切换到普通模式")
        my_flange_pose_msg = self.robot.get_flange_pose()
        print("my_flange_pose_msg:", my_flange_pose_msg)
        self.recorded_j6_pose = my_flange_pose_msg.msg
        save_pose_to_file(self.recorded_j6_pose)
        print("[Robot] 位姿已记录并保存，J6:", [round(p, 4) for p in self.recorded_j6_pose])
        T_j6 = pose_to_homogeneous_matrix(self.recorded_j6_pose)
        T_j6_to_camera = quat_pose_to_homogeneous_matrix(J6_TO_CAMERA)
        self.T_result = np.matmul(T_j6, T_j6_to_camera).astype(np.float32)
        print("[Robot] T_result 已更新（相机在基座系下）")

    def _move_to_home(self):
        """S：机械臂回零"""
        if self.robot is None:
            print("[Robot] 未连接机械臂")
            return
        if self.is_master_mode:
            print("[Robot] 主臂模式下无法回零，请先按 D 切换普通模式")
            return
        print("[Robot] 回零中...")
        self.robot.move_j([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        start_t = time.monotonic()
        while time.monotonic() - start_t < MOTION_TIMEOUT:
            st = self.robot.get_arm_status()
            if st and st.msg.motion_status == 0:
                print("[Robot] 回零完成")
                return
            time.sleep(0.1)
        print("[Robot] 回零超时")

    def _restore_recorded_pose(self):
        """X：复现记录的 J6 法兰位姿"""
        if self.robot is None:
            print("[Robot] 未连接机械臂")
            return
        if self.is_master_mode:
            print("[Robot] 主臂模式下无法复现，请先按 D 切换普通模式")
            return
        if self.recorded_j6_pose is None:
            print("[Robot] 未记录位姿，请先按 D 记录")
            return
        print("[Robot] 复现 J6 位姿...")
        self.robot.move_p(self.recorded_j6_pose)
        start_t = time.monotonic()
        while time.monotonic() - start_t < MOTION_TIMEOUT:
            st = self.robot.get_arm_status()
            if st and st.msg.motion_status == 0:
                print("[Robot] 位姿复现完成")
                return
            time.sleep(0.1)
        print("[Robot] 复现超时")

    def _gripper_max(self):
        """Q：夹爪开到最大"""
        if self.end_effector is None:
            print("[Robot] 未连接夹爪")
            return
        try:
            self.end_effector.move_gripper(width=GRIPPER_MAX_WIDTH, force=GRIPPER_FORCE)
            time.sleep(1.0)
            print(f"[Robot] 夹爪已开到最大 ({GRIPPER_MAX_WIDTH*100}cm)")
        except Exception as e:
            print(f"[Robot] 夹爪控制失败: {e}")

    def _gripper_min(self):
        """E：夹爪闭合到最小"""
        if self.end_effector is None:
            print("[Robot] 未连接夹爪")
            return
        try:
            self.end_effector.move_gripper(width=GRIPPER_MIN_WIDTH, force=GRIPPER_FORCE)
            time.sleep(1.0)
            print("[Robot] 夹爪已闭合")
        except Exception as e:
            print(f"[Robot] 夹爪控制失败: {e}")

    def _compute_grasp_from_current_frame(self, original_color_image):
        """
        基于当前帧（original_color_image + self._latest_depth + self.last_masks）计算点云与抓取姿态。
        设置 self.best_grasp_T，发布 AABB/抓取 Marker；返回 (是否检测到有效目标, all_pts, all_cols)。
        """
        if (
            self._latest_depth is None
            or self.last_masks is None
            or self.last_obj_ids is None
            or self.fx is None
            or self.fy is None
        ):
            return False, [], []
        depth_raw = self._latest_depth.copy()
        h_d, w_d = depth_raw.shape
        depth_m = depth_raw.astype(np.float32) * float(DEPTH_SCALE_ROS)
        color_rgb = cv2.cvtColor(original_color_image, cv2.COLOR_BGR2RGB)
        best_grasp_score = -1.0
        best_grasp_T_cam = None
        all_pts = []
        all_cols = []
        aabb_markers = []
        grasp_markers = []
        for idx, mask_arr in enumerate(self.last_masks):
            if idx >= len(self.last_obj_ids):
                break
            mask = mask_arr
            if mask.shape != depth_raw.shape:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (w_d, h_d),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = mask.astype(bool)
            valid = (depth_m > 0.0) & mask
            ys, xs = np.where(valid)
            if ys.size == 0:
                continue
            z_cam = depth_m[ys, xs]
            x_cam = (
                (xs.astype(np.float32) - float(self.cx)) * z_cam / float(self.fx)
            )
            y_cam = (
                (ys.astype(np.float32) - float(self.cy)) * z_cam / float(self.fy)
            )
            pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
            cols = color_rgb[ys, xs].astype(np.float32) / 255.0
            pcd_raw = o3d.geometry.PointCloud()
            pcd_raw.points = o3d.utility.Vector3dVector(pts_cam)
            pcd_raw.colors = o3d.utility.Vector3dVector(cols)
            pcd_down = pcd_raw.voxel_down_sample(voxel_size=0.005)
            if len(pcd_down.points) > 0:
                pcd_denoised, _ = pcd_down.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=2.0
                )
            else:
                pcd_denoised = pcd_raw
            pcd_show = pcd_denoised if len(pcd_denoised.points) > 0 else pcd_raw
            all_pts.append(np.asarray(pcd_show.points, dtype=np.float32))
            all_cols.append(np.asarray(pcd_show.colors, dtype=np.float32))
            obb = pcd_show.get_oriented_bounding_box()
            center_obb = obb.center
            R_obb = obb.R
            pts_for_grasp = np.asarray(pcd_show.points, dtype=np.float64)
            result = compute_grasp_from_pca_aabb(
                pts_for_grasp, gripper_max_opening=float(GRIPPER_MAX_WIDTH)
            )
            if result is None:
                continue
            center, R_grasp = result
            R_grasp = align_grasp_x_toward_base(R_grasp, self.T_result)
            T_grasp_cam = np.eye(4, dtype=np.float32)
            T_grasp_cam[:3, :3] = R_grasp
            T_grasp_cam[:3, 3] = center
            dist = np.linalg.norm(center)
            score = 1.0 / (1.0 + dist)
            if score > best_grasp_score:
                best_grasp_score = score
                best_grasp_T_cam = T_grasp_cam
            min_b = np.asarray(obb.get_min_bound(), dtype=np.float64)
            max_b = np.asarray(obb.get_max_bound(), dtype=np.float64)
            try:
                self._publish_aabb_marker(
                    min_b, max_b, R_obb, center_obb, CAMERA_OPTICAL_FRAME
                )
                self._publish_grasp_pose_markers(
                    center, R_grasp, CAMERA_OPTICAL_FRAME
                )
                if _ROS_AVAILABLE and getattr(self, "_marker_array_pub", None):
                    m_aabb = self._make_aabb_marker(
                        min_b, max_b, R_obb, center_obb,
                        CAMERA_OPTICAL_FRAME, len(aabb_markers)
                    )
                    m_grasp = self._make_grasp_axes_marker(
                        center, R_grasp, CAMERA_OPTICAL_FRAME,
                        len(aabb_markers) + len(grasp_markers)
                    )
                    if m_aabb is not None:
                        aabb_markers.append(m_aabb)
                    if m_grasp is not None:
                        grasp_markers.append(m_grasp)
            except Exception:
                pass
        if _ROS_AVAILABLE and getattr(self, "_marker_array_pub", None) and (aabb_markers or grasp_markers):
            arr = MarkerArray()
            arr.markers = aabb_markers + grasp_markers
            self._marker_array_pub.publish(arr)
        if best_grasp_T_cam is not None:
            T_grasp_cam_h = np.eye(4, dtype=np.float32)
            T_grasp_cam_h[:3, :3] = best_grasp_T_cam[:3, :3]
            T_grasp_cam_h[:3, 3] = best_grasp_T_cam[:3, 3]
            T_grasp_base = (self.T_result @ T_grasp_cam_h).astype(np.float32)
            self.best_grasp_T = T_grasp_base
            print(
                "[GRASP] 选中的最佳抓取姿态（基座坐标系下 4x4 齐次矩阵）:\n",
                self.best_grasp_T,
            )
        else:
            self.best_grasp_T = None
        return self.best_grasp_T is not None, all_pts, all_cols

    def _do_grasp_move(self):
        """执行 g 功能：下发当前最佳抓取姿态并等待机械臂到达。"""
        if self.robot is None:
            print("[GRASP] 未连接机械臂，无法下发抓取姿态")
            return
        if self.best_grasp_T is None:
            print("[GRASP] 无抓取姿态，请先执行 p 计算")
            return
        T_tcp = self.best_grasp_T
        T_tcp_to_flange = np.eye(4, dtype=np.float32)
        T_tcp_to_flange[2, 3] = -float(TCP_OFFSET[2])
        T_flange = (T_tcp @ T_tcp_to_flange).astype(np.float32)
        pose_flange = homogeneous_to_pose(T_flange)
        print("[GRASP] 下发抓取姿态（法兰位姿）:", [round(p, 4) for p in pose_flange])
        try:
            self.robot.move_p(pose_flange)
            start_t = time.monotonic()
            while True:
                arm_status = self.robot.get_arm_status()
                if arm_status and arm_status.msg.motion_status == 0:
                    print("[GRASP] 机械臂已到达目标位姿")
                    break
                if time.monotonic() - start_t > MOTION_TIMEOUT:
                    print("[GRASP] 运动超时")
                    break
                time.sleep(0.1)
        except Exception as e:
            print(f"[GRASP] 下发失败: {e}")

    def run(self):
        """主运行函数：RealSense 采集 + SAM3 实时推理"""
        try:
            self.running = True

            with torch.no_grad():
                while self.running:
                    self.frame_counter += 1
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    # 有机械臂且非主臂模式时，用当前 J6 位姿更新 T_result（相机在基座系下）
                    if self.robot is not None and not self.is_master_mode:
                        fp = self.robot.get_flange_pose()
                        if fp is not None and len(fp.msg) >= 6:
                            T_j6 = pose_to_homogeneous_matrix(fp.msg)
                            self.T_result = (T_j6 @ self._T_j6_to_camera).astype(np.float32)
                    if _ROS_AVAILABLE and self._node is not None:
                        rclpy.spin_once(self._node, timeout_sec=0)
                    # 有机械臂且非主臂模式时，用当前 J6 位姿更新 T_result（相机在基座系下）
                    if self.robot is not None and not self.is_master_mode:
                        fp = self.robot.get_flange_pose()
                        if fp is not None and len(fp.msg) >= 6:
                            T_j6 = pose_to_homogeneous_matrix(fp.msg)
                            self.T_result = (T_j6 @ self._T_j6_to_camera).astype(np.float32)
                    self._publish_tf()

                    # 等待 ROS 相机话题（颜色图 + 对齐深度图）准备就绪
                    if self._latest_color is None or self._latest_depth is None:
                        continue
                    color_image = self._latest_color.copy()
                    depth_raw = self._latest_depth.copy()

                    # 相机内参应从 CameraInfo 话题获取，若尚未获取则暂不处理
                    if (
                        self.fx is None
                        or self.fy is None
                        or self.cx is None
                        or self.cy is None
                    ):
                        continue

                    # depth_raw: uint16, 单位 mm -> 转为 m
                    depth_m = depth_raw.astype(np.float32) * float(DEPTH_SCALE_ROS)

                    color_image_bg_removed, original_color_image = self.remove_background(
                        color_image,
                        depth_m,
                        self.depth_clipping_distance,
                    )

                    # 可视化深度伪彩色图
                    depth_norm = depth_m / max(self.depth_clipping_distance, 1e-3)
                    depth_norm = np.clip(depth_norm, 0.0, 1.0)
                    depth_colormap = cv2.applyColorMap(
                        (depth_norm * 255.0).astype(np.uint8), cv2.COLORMAP_JET
                    )

                    h, w = color_image_bg_removed.shape[:2]
                    pip_w, pip_h = int(w / 5), int(h / 5)
                    margin = int(max(w, h) / 25)

                    # ===== SAM3 实时推理（通过子进程）=====
                    rgb_for_sam = cv2.cvtColor(original_color_image, cv2.COLOR_BGR2RGB)

                    # 往子进程发送当前帧（队列满则跳过，避免阻塞）
                    if not self.sam_input_q.full():
                        try:
                            self.sam_input_q.put_nowait(rgb_for_sam)
                        except queue.Full:
                            pass

                    # 尝试从子进程获取最新的分割结果与掩码（如无新结果则继续使用上一帧）
                    try:
                        while True:
                            data = self.sam_output_q.get_nowait()
                            self.last_overlay_bgr = cv2.cvtColor(
                                data["overlay"], cv2.COLOR_RGB2BGR
                            )
                            self.last_masks = data.get("masks", None)
                            self.last_obj_ids = data.get("ids", None)
                    except queue.Empty:
                        pass

                    # ===== 组合深度/RGB 画中画 =====
                    # 主画面：去背景后的彩色图
                    final_display_image = color_image_bg_removed.copy()

                    # 深度图画中画（右上角）
                    depth_pip = cv2.resize(depth_colormap, (pip_w, pip_h))
                    depth_pip_x = w - pip_w - margin
                    depth_pip_y = margin
                    final_display_image[
                        depth_pip_y : depth_pip_y + pip_h,
                        depth_pip_x : depth_pip_x + pip_w,
                    ] = depth_pip

                    # 原始 RGB 画中画（右下角）
                    rgb_pip = cv2.resize(original_color_image, (pip_w, pip_h))
                    rgb_pip_x = w - pip_w - margin
                    rgb_pip_y = h - pip_h - margin
                    final_display_image[
                        rgb_pip_y : rgb_pip_y + pip_h,
                        rgb_pip_x : rgb_pip_x + pip_w,
                    ] = rgb_pip

                    # SAM3 分割结果画中画（左上角），如果已有结果
                    if self.last_overlay_bgr is not None:
                        sam_pip = cv2.resize(self.last_overlay_bgr, (pip_w, pip_h))
                        sam_pip_x = margin
                        sam_pip_y = margin
                        final_display_image[
                            sam_pip_y : sam_pip_y + pip_h,
                            sam_pip_x : sam_pip_x + pip_w,
                        ] = sam_pip

                    # 文字提示
                    cv2.putText(
                        final_display_image,
                        f"Depth Clip: {self.depth_clipping_distance:.2f}m",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        final_display_image,
                        f"SAM3 Text Prompt: '{self.text_prompt}'",
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        final_display_image,
                        "Depth",
                        (depth_pip_x + 5, depth_pip_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        final_display_image,
                        "RGB",
                        (rgb_pip_x + 5, rgb_pip_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    if self.last_overlay_bgr is not None:
                        cv2.putText(
                            final_display_image,
                            "SAM3",
                            (margin + 5, margin + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                    cv2.imshow(self.window_name, final_display_image)

                    # ---------- 自动化模式（--auto）状态机 ----------
                    if self.auto_mode:
                        now = time.monotonic()
                        sam_ready = (
                            self.last_masks is not None
                            and self.last_obj_ids is not None
                            and self.fx is not None
                            and self._latest_depth is not None
                        )
                        if self._auto_state == 0:
                            if sam_ready:
                                print("[Auto] SAM3 已就绪，执行 X（复现位姿）")
                                self._restore_recorded_pose()
                                self._auto_state_ts = now
                                self._auto_state = 1
                        elif self._auto_state == 1:
                            if now - self._auto_state_ts >= 2.0:
                                print("[Auto] 执行 q（夹爪开）")
                                self._gripper_max()
                                self._auto_state_ts = now
                                self._auto_state = 2
                        elif self._auto_state == 2:
                            if now - self._auto_state_ts >= 1.0:
                                self._auto_p_retries = 0
                                self._auto_state = 3
                        elif self._auto_state == 3:
                            found, all_pts, all_cols = self._compute_grasp_from_current_frame(
                                original_color_image
                            )
                            if found:
                                if (
                                    self._pcd_pub is not None
                                    and len(all_pts) > 0
                                    and len(all_cols) > 0
                                ):
                                    pts_cat = np.concatenate(all_pts, axis=0)
                                    cols_cat = np.concatenate(all_cols, axis=0)
                                    stamp = self._node.get_clock().now().to_msg()
                                    msg = make_pointcloud2(
                                        pts_cat, cols_cat, CAMERA_OPTICAL_FRAME, stamp
                                    )
                                    if msg is not None:
                                        self._pcd_pub.publish(msg)
                                print("[Auto] p 检测到有效目标，延迟 2s 后执行 g")
                                self._auto_state_ts = now
                                self._auto_state = 4
                            else:
                                self._auto_p_retries += 1
                                if self._auto_p_retries >= 10:
                                    print("[Auto] p 连续 10 次无有效目标，关闭程序")
                                    self.running = False
                                    break
                        elif self._auto_state == 4:
                            if now - self._auto_state_ts >= 2.0:
                                print("[Auto] 执行 g（下发抓取）")
                                self._do_grasp_move()
                                self._auto_state_ts = now
                                self._auto_state = 5
                        elif self._auto_state == 5:
                            if now - self._auto_state_ts >= 2.0:
                                print("[Auto] 执行 e（夹爪合）")
                                self._gripper_min()
                                self._auto_state_ts = now
                                self._auto_state = 6
                        elif self._auto_state == 6:
                            if now - self._auto_state_ts >= 1.0:
                                print("[Auto] 执行 x（复现位姿）")
                                self._restore_recorded_pose()
                                self._auto_state_ts = now
                                self._auto_state = 7
                        elif self._auto_state == 7:
                            if now - self._auto_state_ts >= 2.0:
                                print("[Auto] 执行 s（回零）")
                                self._move_to_home()
                                self._auto_state_ts = now
                                self._auto_state = 8
                        elif self._auto_state == 8:
                            if now - self._auto_state_ts >= 2.0:
                                print("[Auto] 流程结束，退出程序")
                                self.running = False
                                break
                    key = cv2.waitKey(1)
                    if key & 0xFF == 27:  # ESC 退出
                        break
                    key_lower = (key & 0xFF) | 0x20 if key >= 0 else -1  # 小写便于不区分大小写
                    if key_lower == ord("a"):
                        self._switch_to_master_mode()
                    elif key_lower == ord("d"):
                        self._switch_to_normal_mode_and_record()
                    elif key_lower == ord("s"):
                        self._move_to_home()
                    elif key_lower == ord("x"):
                        self._restore_recorded_pose()
                    elif key_lower == ord("q"):
                        self._gripper_max()
                    elif key_lower == ord("e"):
                        self._gripper_min()
                    # 按 'p' 键：基于当前帧计算点云 + AABB + 抓取姿态，并通过 ROS (RViz) 可视化
                    if (
                        key & 0xFF == ord("p")
                        and self.last_masks is not None
                        and self.last_obj_ids is not None
                        and self.fx is not None
                        and self.fy is not None
                    ):
                        if self._latest_depth is None:
                            print("[PCD] 当前无有效深度帧，无法生成点云")
                            continue
                        found, all_pts, all_cols = self._compute_grasp_from_current_frame(
                            original_color_image
                        )
                        if (
                            self._pcd_pub is not None
                            and len(all_pts) > 0
                            and len(all_cols) > 0
                        ):
                            pts_cat = np.concatenate(all_pts, axis=0)
                            cols_cat = np.concatenate(all_cols, axis=0)
                            stamp = self._node.get_clock().now().to_msg()
                            msg = make_pointcloud2(
                                pts_cat, cols_cat, CAMERA_OPTICAL_FRAME, stamp
                            )
                            if msg is not None:
                                self._pcd_pub.publish(msg)
                    # 按 't' 键：在终端输入新的文本提示词
                    if key & 0xFF == ord("t"):
                        try:
                            new_prompt = input(
                                "\n请输入新的 SAM3 文本提示（回车确认，留空则忽略）："
                            ).strip()
                        except EOFError:
                            new_prompt = ""
                        if new_prompt:
                            self.text_prompt = new_prompt
                            # 将新提示传递给 SAM 子进程
                            try:
                                self.prompt_q.put_nowait(new_prompt)
                            except queue.Full:
                                # 丢弃旧的，只保留最新
                                try:
                                    while True:
                                        self.prompt_q.get_nowait()
                                except queue.Empty:
                                    pass
                                self.prompt_q.put_nowait(new_prompt)
                            print(f"[Main] 已更新文本提示: '{self.text_prompt}'")

                    # 按 'g' 键：下发当前选中的最佳抓取姿态，控制机械臂 TCP 到达
                    if key & 0xFF == ord("g"):
                        self._do_grasp_move()

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """停止并释放资源"""
        self.running = False
        try:
            if hasattr(self, "pipeline") and self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("RealSense pipeline stopped, resources released")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RealSense + SAM3 实时分割（SAM 在子进程 GPU 上运行），可选机械臂控制"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="person",
        help="SAM3 文本提示，例如 'person', 'hand', 'chair' 等",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="自动化运行：X→q→p(最多10次)→g→e→x→s 后退出",
    )
    parser.add_argument(
        "--no-camera-launch",
        action="store_true",
        help="不自动拉起 RealSense ROS 节点（已手动启动时使用）",
    )
    args = parser.parse_args()

    # RealSense ROS2 节点：程序内拉起，退出时关闭（可用 --no-camera-launch 跳过）
    camera_proc_ref = [None]  # 用列表以便 signal 回调和 finally 共享

    def _camera_exit_handler(signum, frame):
        stop_camera_launch(camera_proc_ref[0])
        camera_proc_ref[0] = None
        sys.exit(128 + (signum if signum < 128 else 0))

    if _ROS_AVAILABLE and not args.no_camera_launch:
        camera_proc_ref[0] = start_camera_launch()
        if camera_proc_ref[0] is not None:
            signal.signal(signal.SIGINT, _camera_exit_handler)
            signal.signal(signal.SIGTERM, _camera_exit_handler)

    node = None
    if _ROS_AVAILABLE:
        rclpy.init(args=None)
        node = Node("realsense_sam")
        tf_broadcaster = TransformBroadcaster(node)
    else:
        tf_broadcaster = None
        print("[TF] 未检测到 ROS2，将不发布 TF")

    mp.set_start_method("spawn", force=True)
    sam_input_q: mp.Queue = mp.Queue(maxsize=2)
    sam_output_q: mp.Queue = mp.Queue(maxsize=2)
    prompt_q: mp.Queue = mp.Queue(maxsize=4)

    sam_proc = mp.Process(
        target=sam_worker,
        args=(sam_input_q, sam_output_q, prompt_q, args.prompt),
        daemon=True,
    )
    sam_proc.start()

    robot, end_effector = init_robot_and_gripper()

    try:
        rs_align = RealSenseAlignAdvanced(
            text_prompt=args.prompt,
            sam_input_q=sam_input_q,
            sam_output_q=sam_output_q,
            prompt_q=prompt_q,
            robot=robot,
            end_effector=end_effector,
            node=node,
            tf_broadcaster=tf_broadcaster,
            auto_mode=args.auto,
        )
        if args.auto:
            print("[Auto] 自动化流程已启用：等待 SAM3 就绪后执行 X→q→p(最多10次)→g→e→x→s→退出")
        else:
            print(
                "按键: A=主臂零力  D=普通模式+记录位姿  S=回零  X=复现位姿  Q=夹爪开  E=夹爪合  "
                "p=点云/抓取  t=改提示词  g=下发抓取  Esc=退出"
            )
        rs_align.run()
    finally:
        stop_camera_launch(camera_proc_ref[0])
        camera_proc_ref[0] = None
        safe_shutdown_robot(robot, end_effector)
        if node is not None:
            node.destroy_node()
        if _ROS_AVAILABLE:
            rclpy.shutdown()
        try:
            sam_input_q.put_nowait(None)
        except queue.Full:
            pass
        sam_proc.join(timeout=5.0)
