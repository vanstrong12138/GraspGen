#!/usr/bin/env python3
# coding=utf-8

import rospy
import mujoco
import mujoco.viewer
import os
import time
import numpy as np
import subprocess
from std_msgs.msg import Header
from sensor_msgs.msg import JointState

class Mujoco_Model():
    def __init__(self):
        rospy.init_node("mujoco_piper_with_gripper_controller_node", anonymous=True)
        rospy.Subscriber("/mujoco_joint_states_ctrl", JointState, self.joint_state_callback)

        self.joint_targets = {}
        self.joint_state_pub = rospy.Publisher("/mujoco_joint_states_pub", JointState, queue_size=1, tcp_nodelay=True)

        # 加载模型
        package_path = subprocess.check_output('rospack find piper_description', shell=True).strip().decode('utf-8')
        model_path = os.path.join(package_path, 'mujoco_model', 'piper_description.xml')
        model_path = os.path.abspath(model_path)
        
        print("The model path is: ", model_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化查看器
        self.viewer = None
        self._init_viewer()

    def _init_viewer(self):
        """初始化 MuJoCo 查看器"""
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # 设置默认相机视角
            self.viewer.cam.distance = 2.0
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -20
        except Exception as e:
            rospy.logwarn(f"Could not initialize viewer: {e}")
            self.viewer = None

    def joint_state_callback(self, msg):
        """ROS 关节状态回调"""
        for i, name in enumerate(msg.name):
            self.joint_targets[name] = msg.position[i]

    def pos_ctrl(self, joint_name, target_angle):
        """控制 MuJoCo 关节角度"""
        try:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                rospy.logwarn(f"Joint {joint_name} not found in Mujoco model.")
                return

            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if actuator_id == -1:
                rospy.logwarn(f"Actuator for joint {joint_name} not found")
                return

            self.data.ctrl[actuator_id] = target_angle
        except Exception as e:
            rospy.logerr(f"Error controlling joint {joint_name}: {e}")

    def publish_joint_states(self):
        """发布当前 MuJoCo 关节状态"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        
        # 获取所有关节名称和位置
        joint_names = []
        joint_positions = []
        for i in range(self.model.njnt):
            joint_id = i
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name:
                joint_names.append(joint_name)
                joint_positions.append(self.data.qpos[self.model.jnt_qposadr[joint_id]])
        
        msg.name = joint_names
        msg.position = joint_positions
        self.joint_state_pub.publish(msg)

    def control_loop(self):
        """MuJoCo 控制主循环"""
        rate = rospy.Rate(100)  # 100Hz 控制循环
        tolerance = 0.05  # 角度误差容忍度

        while not rospy.is_shutdown():
            all_reached = True

            # 处理关节控制
            for joint, target_angle in self.joint_targets.items():
                # 处理 joint8 等于 joint7 的负数
                if joint == "joint8" and "joint7" in self.joint_targets:
                    target_angle = -self.joint_targets["joint7"]

                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                if joint_id != -1:
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    current_angle = self.data.qpos[qpos_adr]

                    if abs(current_angle - target_angle) > tolerance:
                        all_reached = False

                    self.pos_ctrl(joint, target_angle)

            # 步进模拟
            mujoco.mj_step(self.model, self.data)
            
            # 更新查看器
            if self.viewer is not None:
                try:
                    self.viewer.sync()
                except:
                    self.viewer = None  # 查看器已关闭

            self.publish_joint_states()
            rate.sleep()

def main():
    test = Mujoco_Model()
    test.control_loop()

if __name__ == "__main__":  
    main()
