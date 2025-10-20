#!/usr/bin/env python3
# coding:utf-8

import rospy
import sys
import select
import tty
import termios
import time
import math
from enum import Enum
from collections import deque

# ROS消息导入
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from piper_msgs.msg import PosCmd
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped, Point, PoseStamped, Quaternion
import tf2_geometry_msgs
from tf2_geometry_msgs import do_transform_pose
from std_msgs.msg import Bool, Float64, Int32
import tf.transformations as tf_trans

class RobotState(Enum):
    IDLE = 0
    SEARCHING = 1
    APPROACHING = 2
    GRASPING = 3
    RETREATING = 4
    ERROR = 5

class IKStatusManager:
    """逆解状态管理器"""
    def __init__(self):
        self.ik_success = False
        self.ik_status_received = False
        self.ik_sub = rospy.Subscriber("/ik_status", Bool, self.ik_status_callback)
        
    def ik_status_callback(self, msg):
        """逆解状态回调"""
        self.ik_success = msg.data
        self.ik_status_received = True
        rospy.logdebug(f"IK status: {self.ik_success}")

class PoseAdjuster:
    """姿态调整器"""
    def __init__(self):
        self.max_pitch_adjustment = math.pi / 2  # 最大90度调整
        self.max_roll_adjustment = math.pi / 4   # 最大45度roll调整
        self.pitch_step = math.pi / 18 / 2  # 5度步长
        self.roll_step = math.pi / 18 / 2   # 5度步长
        self.adjustment_sequence = []  # 调整序列 (pitch, roll)
        
    def generate_adjustment_sequence(self):
        """生成调整序列:使用螺旋式渐进搜索，保持yaw不变"""
        self.adjustment_sequence = []
        
        # 首先尝试原始姿态
        self.adjustment_sequence.append((0, 0))
        
        # 螺旋式渐进搜索参数
        spiral_step = math.pi / 18  # 10度步长
        max_layers = int(max(self.max_pitch_adjustment, self.max_roll_adjustment) / spiral_step)
        
        # 生成螺旋序列
        for layer in range(1, max_layers + 1):
            current_radius = layer * spiral_step
            
            # 每层采样点数随半径增加
            points_in_layer = max(8, int(8 * layer))
            
            for i in range(points_in_layer):
                angle = 2 * math.pi * i / points_in_layer
                
                # 椭圆螺旋，pitch范围大，roll范围小
                pitch_adj = current_radius * math.sin(angle) * (self.max_pitch_adjustment / (self.max_pitch_adjustment + self.max_roll_adjustment))
                roll_adj = current_radius * math.cos(angle) * (self.max_roll_adjustment / (self.max_pitch_adjustment + self.max_roll_adjustment))
                
                # 限制在最大范围内
                pitch_adj = max(min(pitch_adj, self.max_pitch_adjustment), -self.max_pitch_adjustment)
                roll_adj = max(min(roll_adj, self.max_roll_adjustment), -self.max_roll_adjustment)
                
                # 过滤掉不合理的极端组合
                if self._is_valid_combination(pitch_adj, roll_adj):
                    self.adjustment_sequence.append((pitch_adj, roll_adj))
        
        rospy.loginfo(f"Generated spiral adjustment sequence with {len(self.adjustment_sequence)} poses")
        return self.adjustment_sequence
    
    def _is_valid_combination(self, pitch, roll):
        """检查是否为有效组合（过滤掉不合理的组合）"""
        # 避免极端组合，比如大pitch+大roll
        if abs(pitch) > math.pi/4 and abs(roll) > math.pi/6:
            return False
        # 避免过小的调整（与原点太近的重复点）
        if abs(pitch) < 0.01 and abs(roll) < 0.01:
            return False
        return True
    
    def get_adjustment_value(self, attempt_count):
        """根据尝试次数获取调整值"""
        if attempt_count < len(self.adjustment_sequence):
            return self.adjustment_sequence[attempt_count]
        
        return None  # 超出调整范围
    
    def adjust_pose_orientation(self, pose, pitch_adjustment, roll_adjustment):
        """调整姿态的俯仰角和滚转角，保持yaw不变"""
        # 获取原始四元数
        orig_quat = [
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w
        ]
        
        # 转换为欧拉角
        orig_euler = tf_trans.euler_from_quaternion(orig_quat)
        
        # 调整俯仰角和滚转角，yaw保持不变
        new_euler = [
            orig_euler[0] + roll_adjustment,    # roll
            orig_euler[1] + pitch_adjustment,   # pitch
            orig_euler[2]                       # yaw保持不变
        ]
        
        # 转换回四元数
        new_quat = tf_trans.quaternion_from_euler(*new_euler)
        
        # 创建新的姿态
        adjusted_pose = PoseStamped()
        adjusted_pose.header = pose.header
        adjusted_pose.pose.position = pose.pose.position
        adjusted_pose.pose.orientation.x = new_quat[0]
        adjusted_pose.pose.orientation.y = new_quat[1]
        adjusted_pose.pose.orientation.z = new_quat[2]
        adjusted_pose.pose.orientation.w = new_quat[3]
        
        return adjusted_pose

class ActionSequence:
    """动作序列管理类"""
    def __init__(self):
        self.sequence = deque()
        self.current_action = None
        self.last_action_time = 0
        self.action_delay = 1.0  # 默认动作延迟2秒
        
    def add_action(self, action_name, action_func, delay=None, *args, **kwargs):
        """添加动作到序列"""
        action_delay = delay if delay is not None else self.action_delay
        self.sequence.append((action_name, action_func, action_delay, args, kwargs))
        
    def execute_next(self):
        """执行下一个动作"""
        if self.sequence and (time.time() - self.last_action_time > self.action_delay or self.last_action_time == 0):
            action_name, action_func, delay, args, kwargs = self.sequence.popleft()
            rospy.loginfo(f"Executing action: {action_name}")
            
            # 执行动作函数
            result = action_func(*args, **kwargs)
            
            # 记录执行时间并设置延迟
            self.last_action_time = time.time()
            self.action_delay = delay
            
            rospy.loginfo(f"Action '{action_name}' completed. Waiting {delay:.1f} seconds...")
            return result
            
        return None
        
    def clear(self):
        """清空动作序列"""
        self.sequence.clear()
        self.current_action = None
        self.last_action_time = 0
        self.action_delay = 1.0
        
    def is_empty(self):
        """检查序列是否为空"""
        return len(self.sequence) == 0

class PoseTransformer:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('piper_pose_transformer', anonymous=True)

        # 初始化tf2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 订阅和发布
        self.sub = rospy.Subscriber("/grasp_pose_posestamp", PoseStamped, self.pose_callback)
        self.target_pose_sub = rospy.Subscriber("/target_pose", PoseStamped, self.target_pose_callback)
        self.target_pose_pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=10)
        self.gripper_cmd_pub = rospy.Publisher('/gripper_cmd_topic', Float64, queue_size=1)
        self.outer_cmd_sub = rospy.Subscriber("/vision_task", Int32, self.tast_callback)
        self.task_reslut_pub = rospy.Publisher('/vision_result', Int32, queue_size=1)

        # 逆解状态管理
        self.ik_manager = IKStatusManager()
        self.pose_adjuster = PoseAdjuster()
        
        self.rate = rospy.Rate(30)
        
        # 状态管理
        self.original_pose = None
        self.transformed_pose = None
        self.current_target_pose = None
        self.original_target_pose = None  # 保存原始目标姿态
        self.continuous_publishing = False
        self.gripper_cmd = 0.0
        self.robot_state = RobotState.IDLE
        self.waiting_for_ik = False
        self.ik_check_start_time = 0
        self.adjustment_attempts = 0
        self.max_adjustment_attempts = len(self.pose_adjuster.generate_adjustment_sequence())
        self.ik_success_pose = None  # 保存成功的姿态
        self.motion_complete_time = 0  # 运动完成时间
        self.motion_in_progress = False  # 运动进行中标志

        # 外部请求管理
        self.task_cmd = 0
        self.task_reslut = 0
        
        # 动作序列管理
        self.action_sequence = ActionSequence()
        
        # 预定义路径点
        self.via_point = self.create_via_pose(
            position=[0.300, 0.0, 0.360],
            orientation=[0.007, 0.915, 0.009, 0.403]
        )

        self.target_pose_current = PoseStamped()
        self.via_pose_list = []
        
        # 终端设置
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        rospy.loginfo("Pose transformer node started. Waiting for PoseStamped messages...")

    def create_via_pose(self, position, orientation, frame_id="base_link"):
        """创建路径点姿势"""
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        pose.pose.orientation.x = orientation[0]
        pose.pose.orientation.y = orientation[1]
        pose.pose.orientation.z = orientation[2]
        pose.pose.orientation.w = orientation[3]
        return pose

    def transform_pose(self, pose_msg, target_frame='base_link'):
        """坐标变换"""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                pose_msg.header.frame_id,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            
            transformed_pose = do_transform_pose(pose_msg, transform)
            transformed_pose.header.frame_id = target_frame
            transformed_pose.header.stamp = rospy.Time.now()
            
            return transformed_pose
            
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("坐标变换失败: %s", str(e))
            return None

    def pose_callback(self, msg):
        """PoseStamped消息回调"""
        self.original_pose = msg
        self.transformed_pose = self.transform_pose(msg, "base_link")
        
        if self.transformed_pose:
            rospy.loginfo("Pose transformed successfully")
            self.print_pose_info("Original", self.original_pose)
            self.print_pose_info("Transformed", self.transformed_pose)

    def target_pose_callback(self, msg):
        """TargetPose消息回调"""
        self.target_pose_current = msg

    def tast_callback(self, msg):
        """外部请求task_cmd消息回调"""
        self.task_cmd = msg


    def print_pose_info(self, label, pose):
        """打印姿势信息"""
        if pose:
            pos = pose.pose.position
            orient = pose.pose.orientation
            # 转换为欧拉角显示
            euler = tf_trans.euler_from_quaternion([
                orient.x, orient.y, orient.z, orient.w
            ])
            print(f"{label} pose - Frame: {pose.header.frame_id}, "
                  f"Position: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}), "
                  f"Orientation (RPY): ({np.degrees(euler[0]):.1f}°, "
                  f"{np.degrees(euler[1]):.1f}°, {np.degrees(euler[2]):.1f}°)")

    def publish_pose_with_ik_check(self, pose=None):
        """发布姿势并检查逆解状态"""
        target_pose = pose if pose else self.transformed_pose
        
        if not target_pose:
            rospy.logwarn("No pose available to publish")
            return False
        
        # 如果已经有运动在进行中，不发布新的目标
        if self.motion_in_progress:
            rospy.logwarn("Motion in progress, skipping new target")
            return False
        
        # 保存原始目标姿态
        self.original_target_pose = target_pose
        self.pose_adjuster.generate_adjustment_sequence()
        
        # 重置调整状态
        self.reset_ik_check_state()
        
        # 发布初始姿态（尝试0：原始姿态）
        self.publish_adjusted_pose(0)
        
        return True

    def reset_ik_check_state(self):
        """重置逆解检查状态"""
        self.ik_manager.ik_status_received = False
        self.ik_manager.ik_success = False
        self.waiting_for_ik = True
        self.ik_check_start_time = time.time()
        self.adjustment_attempts = 0
        self.ik_success_pose = None
        self.motion_in_progress = False

    def publish_adjusted_pose(self, attempt_count):
        """发布调整后的姿态"""
        adjustment_values = self.pose_adjuster.get_adjustment_value(attempt_count)
        
        if adjustment_values is None:
            rospy.logerr("No more adjustment values available")
            self.waiting_for_ik = False
            return False
        
        pitch_adjustment, roll_adjustment = adjustment_values
        
        # 调整姿态
        adjusted_pose = self.pose_adjuster.adjust_pose_orientation(
            self.original_target_pose, pitch_adjustment, roll_adjustment
        )
        
        # 发布调整后的姿态
        adjusted_pose.header.stamp = rospy.Time.now()
        self.target_pose_pub.publish(adjusted_pose)
        self.current_target_pose = adjusted_pose
        
        # 打印调整信息
        euler = tf_trans.euler_from_quaternion([
            adjusted_pose.pose.orientation.x,
            adjusted_pose.pose.orientation.y,
            adjusted_pose.pose.orientation.z,
            adjusted_pose.pose.orientation.w
        ])
        
        if attempt_count == 0:
            rospy.loginfo(f"Attempt {attempt_count}: Original pose, "
                         f"roll = {np.degrees(euler[0]):.1f}°, "
                         f"pitch = {np.degrees(euler[1]):.1f}°")
        else:
            rospy.loginfo(f"Attempt {attempt_count}: "
                         f"Adjusted roll = {np.degrees(roll_adjustment):.1f}°, "
                         f"pitch = {np.degrees(pitch_adjustment):.1f}°, "
                         f"total roll = {np.degrees(euler[0]):.1f}°, "
                         f"total pitch = {np.degrees(euler[1]):.1f}°")
        
        return True

    def check_ik_status(self):
        """检查逆解状态并处理"""
        if not self.waiting_for_ik:
            return None
        
        current_time = time.time()
        
        # 检查是否收到逆解状态
        if self.ik_manager.ik_status_received:
            if self.ik_manager.ik_success:
                # 逆解成功，设置运动进行中标志
                rospy.loginfo("IK solution found successfully! Motion started...")
                self.ik_success_pose = self.current_target_pose
                self.waiting_for_ik = False
                self.motion_in_progress = True
                self.motion_complete_time = current_time + 1.5  # 假设运动需要5秒完成
                return True
            else:
                # 逆解失败，等待1秒后尝试下一个调整
                if current_time - self.ik_check_start_time > 0.01:
                    self.adjustment_attempts += 1
                    
                    if self.adjustment_attempts >= self.max_adjustment_attempts:
                        rospy.logerr("Maximum adjustment attempts reached, giving up")
                        self.waiting_for_ik = False
                        return False
                    
                    # 尝试下一个调整值
                    rospy.loginfo(f"Waiting 1 second before next adjustment...")
                    self.ik_manager.ik_status_received = False
                    self.ik_check_start_time = time.time()
                    
                    # 发布下一个调整姿态
                    return self.publish_adjusted_pose(self.adjustment_attempts)
        
        return None

    def check_motion_completion(self):
        """检查运动是否完成"""
        if self.motion_in_progress and time.time() >= self.motion_complete_time:
            rospy.loginfo("Motion completed successfully!")
            self.motion_in_progress = False
            return True
        return False

    def control_gripper(self, position):
        """控制夹爪"""
        self.gripper_cmd = position
        self.gripper_cmd_pub.publish(Float64(self.gripper_cmd))
        rospy.loginfo(f"Gripper set to: {position}")
        return True

    def wait(self, duration=1.0):
        """等待指定时间"""
        rospy.loginfo(f"Waiting for {duration} seconds")
        rospy.sleep(duration)
        return True

    def execute_grasp_sequence(self):
        """执行抓取序列"""
        self.action_sequence.clear()
        
        # 构建抓取动作序列
        self.action_sequence.add_action("Open Gripper", self.control_gripper, 0.1, 0.07)
        self.action_sequence.add_action("Move to Target", self.publish_pose_with_ik_check, 1.0)
        self.action_sequence.add_action("Close Gripper", self.control_gripper, 0.5, 0.0)
        self.action_sequence.add_action("Move to Via Point", self.publish_pose_with_ik_check, 1.5, self.via_point)
        self.action_sequence.add_action("Open Gripper", self.control_gripper, 0.1, 0.07)
        
        self.robot_state = RobotState.GRASPING
        rospy.loginfo("Grasp sequence started with delays")
        self.task_cmd = 0
        self.task_reslut = 2

    def record_search_route(self):
        """记录搜索路径"""
        self.via_pose_list.append(self.target_pose_current)
        rospy.loginfo("Recording search route, now is %d point", len(self.via_pose_list)+1)

    def search_mode(self):
        """执行搜索模式"""
        self.action_sequence.clear()
        self.action_sequence.add_action("Close Gripper", self.control_gripper, 0.5, 0.0)

        if (len(self.via_pose_list)>0):
            for i in range(0, len(self.via_pose_list)):
                self.action_sequence.add_action("Move to Via Point", self.publish_pose_with_ik_check, 1.0, self.via_pose_list[i])
        
        self.robot_state = RobotState.GRASPING
        rospy.loginfo("Searching...")

    def get_key(self):
        """获取键盘输入"""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None

    def print_instructions(self):
        """打印操作指令"""
        instructions = [
            "\nControl commands:",
            "s: 执行抓取序列 (包含延迟)",
            "p: 切换连续发布模式",
            "b: 移动到安全位置并打开夹爪",
            "t: 切换夹爪状态",
            "c: 清除当前动作序列",
            "i: 显示当前信息和指令"
        ]
        
        print("\n".join(instructions))
        
        if self.transformed_pose:
            self.print_pose_info("Current Target", self.transformed_pose)

    def handle_key_input(self, key):
        """处理键盘输入"""
        key_actions = {
            's': self.execute_grasp_sequence,
            'p': lambda: setattr(self, 'continuous_publishing', not self.continuous_publishing),
            'b': lambda: [self.control_gripper(0.07), self.publish_pose_with_ik_check(self.via_point)],
            't': lambda: self.control_gripper(0.0 if self.gripper_cmd >= 0.06 else 0.07),
            'c': self.action_sequence.clear,
            'i': self.print_instructions,
            'r': self.record_search_route,
            'q': self.search_mode
        }
        
        if key in key_actions:
            key_actions[key]()
            
            if key == 'p':
                state = "Started" if self.continuous_publishing else "Stopped"
                rospy.loginfo(f"{state} continuous publishing")
            elif key == 'c':
                rospy.loginfo("Action sequence cleared")
        
        if self.task_cmd:
            self.execute_grasp_sequence()

    def run(self):
        """主循环"""
        tty.setcbreak(sys.stdin.fileno())
        self.print_instructions()
        
        try:
            while not rospy.is_shutdown():
                # 处理键盘输入
                if key := self.get_key() or self.task_cmd:
                    self.handle_key_input(key)
                
                # 检查运动是否完成
                if self.check_motion_completion():
                    # 运动完成，等待2秒后继续
                    rospy.loginfo("Motion completed! Waiting 2 seconds before continuing...")
                    rospy.sleep(1.0)
                    # 这里可以添加状态转换逻辑
                
                # 检查逆解状态（只有在没有运动进行时才检查）
                if not self.motion_in_progress:
                    ik_result = self.check_ik_status()
                
                # 执行动作序列（只有在没有运动进行时才执行）
                if not self.action_sequence.is_empty() and not self.waiting_for_ik and not self.motion_in_progress:
                    self.action_sequence.execute_next()
                
                # 连续发布模式（只有在没有运动进行时才发布）
                if (self.continuous_publishing and self.transformed_pose and 
                    not self.waiting_for_ik and not self.motion_in_progress):
                    pass
                
                self.rate.sleep()
                
        finally:
            # 恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

if __name__ == '__main__':
    try:
        transformer = PoseTransformer()
        transformer.run()
    except rospy.ROSInterruptException:
        pass