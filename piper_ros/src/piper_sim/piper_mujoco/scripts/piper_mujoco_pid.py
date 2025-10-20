import math
import time
import mujoco
import mujoco.viewer
import numpy as np
import os

# 加载模型
current_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_path, '..', '..', 'piper_description', 'mujoco_model', 'piper_description.xml')
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 定义PID控制器类
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_error = 0
        self.prev_error = 0
        self.dt = 0.01  # 控制周期

    def calculate(self, target, current):
        error = target - current
        self.integral_error += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = (self.Kp * error) + (self.Ki * self.integral_error) + (self.Kd * derivative)
        self.prev_error = error
        return output

# 定义六自由度机械臂角度控制类
class ArmControl:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                         for name in self.joint_names]
        self.pid_controllers = [PIDController(1.1, 0, 0) for _ in range(6)]
        self.target_angles = np.zeros(6)

    def set_target_angles(self, target_angles):
        self.target_angles = target_angles

    def get_current_angles(self):
        return [self.data.qpos[id] for id in self.joint_ids]

    def control(self):
        current_angles = self.get_current_angles()
        for i in range(6):
            control_signal = self.pid_controllers[i].calculate(
                self.target_angles[i], current_angles[i])
            self.data.ctrl[i] = control_signal

# 主程序
arm_control = ArmControl(model, data)

# 使用 mujoco.viewer 的上下文管理器
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 设置相机参数
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -20

    count = 0
    while viewer.is_running():
        # 更新目标角度
        if 0 <= count < 300:
            target_angles = [0.5, 0.5, -0.5, 0.5, -0.5, 0.5]
        elif 300 <= count < 600:
            target_angles = [-0.5, -0.5, 0.5, -0.5, 0.5, -0.5]
        else:
            count = 0
            continue

        arm_control.set_target_angles(target_angles)
        arm_control.control()

        # 步进模拟
        mujoco.mj_step(model, data)

        # 同步视图
        viewer.sync()
        
        # 控制循环频率
        time.sleep(0.01)
        count += 1
