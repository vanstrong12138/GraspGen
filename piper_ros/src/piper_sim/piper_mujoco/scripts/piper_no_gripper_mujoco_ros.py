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
        rospy.init_node("mujoco_piper_no_gripper_controller_node", anonymous=True)
        rospy.Subscriber("/mujoco_joint_states_ctrl", JointState, self.joint_state_callback)

        self.joint_targets = {}
        self.joint_state_pub = rospy.Publisher("/mujoco_joint_states_pub", JointState, 
                                             queue_size=1, tcp_nodelay=True)

        # 加载模型
        package_path = subprocess.check_output('rospack find piper_description', 
                                             shell=True).strip().decode('utf-8')
        model_path = os.path.join(package_path, 'mujoco_model', 
                                 'piper_no_gripper_description.xml')
        model_path = os.path.abspath(model_path)
        
        rospy.loginfo(f"The model path is: {model_path}")

        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化查看器
        self.viewer = None
        self._init_viewer()

        # 预计算关节映射
        self._setup_joint_mappings()

    def _init_viewer(self):
        """初始化MuJoCo查看器"""
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # 设置相机参数
            self.viewer.cam.distance = 2.5
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -20
        except Exception as e:
            rospy.logwarn(f"Could not initialize viewer: {e}")

    def _setup_joint_mappings(self):
        """建立关节名称到ID的映射关系"""
        self.joint_name_to_id = {}
        self.actuator_name_to_id = {}
        
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                self.joint_name_to_id[joint_name] = i
                
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name:
                self.actuator_name_to_id[actuator_name] = i

    def joint_state_callback(self, msg):
        """ROS关节状态回调"""
        for i, name in enumerate(msg.name):
            self.joint_targets[name] = msg.position[i]

    def pos_ctrl(self, joint_name, target_angle):
        """控制MuJoCo关节角度"""
        try:
            # 获取关节ID
            joint_id = self.joint_name_to_id.get(joint_name, -1)
            if joint_id == -1:
                rospy.logwarn_once(f"Joint {joint_name} not found in Mujoco model")
                return

            # 获取执行器ID
            actuator_id = self.actuator_name_to_id.get(joint_name, -1)
            if actuator_id == -1:
                rospy.logwarn_once(f"No actuator found for joint {joint_name}")
                return

            # 设置控制信号
            self.data.ctrl[actuator_id] = target_angle

        except Exception as e:
            rospy.logerr(f"Error controlling joint {joint_name}: {e}")

    def publish_joint_states(self):
        """发布当前MuJoCo关节状态"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        
        # 收集所有关节状态
        joint_names = []
        joint_positions = []
        
        for joint_name, joint_id in self.joint_name_to_id.items():
            qpos_adr = self.model.jnt_qposadr[joint_id]
            joint_names.append(joint_name)
            joint_positions.append(self.data.qpos[qpos_adr])
        
        msg.name = joint_names
        msg.position = joint_positions
        self.joint_state_pub.publish(msg)

    def control_loop(self):
        """MuJoCo控制主循环"""
        rate = rospy.Rate(100)  # 100Hz控制频率
        tolerance = 0.05  # 角度容差

        while not rospy.is_shutdown():
            all_reached = True

            # 处理每个目标关节
            for joint, target_angle in self.joint_targets.items():
                if joint in self.joint_name_to_id:
                    joint_id = self.joint_name_to_id[joint]
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    current_angle = self.data.qpos[qpos_adr]

                    # 检查是否达到目标
                    if abs(current_angle - target_angle) > tolerance:
                        all_reached = False

                    # 控制关节
                    self.pos_ctrl(joint, target_angle)

            # 步进模拟
            mujoco.mj_step(self.model, self.data)
            
            # 更新查看器
            if self.viewer is not None:
                try:
                    self.viewer.sync()
                except:
                    self.viewer = None  # 查看器关闭时重置

            # 发布关节状态
            self.publish_joint_states()
            
            # 保持循环频率
            rate.sleep()

def main():
    try:
        test = Mujoco_Model()
        test.control_loop()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")

if __name__ == "__main__":  
    main()
