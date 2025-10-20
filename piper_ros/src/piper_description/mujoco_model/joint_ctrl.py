import mujoco
import numpy as np
import time
import mujoco.viewer
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

class DoubleLoopPIDController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Define controlled joints (excluding gripper joints 7/8)
        self.controlled_joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        
        # Pre-compute joint IDs for faster access
        self.joint_ids = {j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) 
                         for j in self.controlled_joints}
        
        # Get joint limits from model
        self.joint_limits = {j: (model.jnt_range[self.joint_ids[j]][0], 
                                model.jnt_range[self.joint_ids[j]][1]) 
                           for j in self.controlled_joints}
        
        # PID parameters for double-loop control (position and velocity only)
        self.pid_params = {
            'joint1': {
                'position': {'kp': 1.0, 'ki': 0, 'kd': 0.0},
                'velocity': {'kp': 1.0, 'ki': 0, 'kd': 0.0}
            },
            'joint2': {
                'position': {'kp': 3000, 'ki': 0, 'kd': 0},
                'velocity': {'kp': 3000.0, 'ki': 0, 'kd': 0.0}
            },
            'joint3': {
                'position': {'kp': 3000.0, 'ki': 0.1, 'kd': 0.0},
                'velocity': {'kp': 3000, 'ki': 0, 'kd': 0.0}
            },
            'joint4': {
                'position': {'kp': 1.0, 'ki': 0, 'kd': 0.0},
                'velocity': {'kp': 1.0, 'ki': 0, 'kd': 1.0}
            },
            'joint5': {
                'position': {'kp': 100.0, 'ki': 0, 'kd': 1},
                'velocity': {'kp': 100.0, 'ki': 0, 'kd': 0.0}
            },
            'joint6': {
                'position': {'kp': 1.0, 'ki': 0, 'kd': 0.0},
                'velocity': {'kp': 1.0, 'ki': 0, 'kd': 0.0}
            }
        }
        
        # Initialize PID controllers for position and velocity loops
        self.position_pids = {
            j: PIDController(
                self.pid_params[j]['position']['kp'],
                self.pid_params[j]['position']['ki'],
                self.pid_params[j]['position']['kd']
            ) for j in self.controlled_joints
        }
        
        self.velocity_pids = {
            j: PIDController(
                self.pid_params[j]['velocity']['kp'],
                self.pid_params[j]['velocity']['ki'],
                self.pid_params[j]['velocity']['kd']
            ) for j in self.controlled_joints
        }
        
        # Data logging for visualization
        self.position_history = defaultdict(list)
        self.velocity_history = defaultdict(list)
        self.torque_history = defaultdict(list)
        self.target_position_history = defaultdict(list)
        self.target_velocity_history = defaultdict(list)
        self.time_history = []
        self.start_time = time.time()
        self.log_counter = 0
        self.log_interval = 5  # Log every 5 steps to reduce data volume
    
    def compute_torques(self, target_positions):
        """Compute torque for all joints using double-loop PID control (position and velocity)"""
        torques = {}
        
        for joint in self.controlled_joints:
            # Get current joint state
            current_pos = self.data.qpos[self.model.jnt_qposadr[self.joint_ids[joint]]]
            current_vel = self.data.qvel[self.model.jnt_dofadr[self.joint_ids[joint]]]
            
            # Position loop: target position to target velocity
            pos_error = target_positions[joint] - current_pos
            target_velocity = self.position_pids[joint].update(pos_error, self.model.opt.timestep)
            
            # Velocity loop: target velocity to torque command
            vel_error = target_velocity - current_vel
            torque = self.velocity_pids[joint].update(vel_error, self.model.opt.timestep)
            
            # Apply torque limits (safety)
            max_torque = 100.0  # Adjust based on your robot's capabilities
            torque = np.clip(torque, -max_torque, max_torque)
            torques[joint] = torque
            
            # Log data for visualization
            if self.log_counter % self.log_interval == 0:
                current_time = time.time() - self.start_time
                self.position_history[joint].append(current_pos)
                self.velocity_history[joint].append(current_vel)
                self.torque_history[joint].append(torque)
                self.target_position_history[joint].append(target_positions[joint])
                self.target_velocity_history[joint].append(target_velocity)
                if joint == self.controlled_joints[0]:
                    self.time_history.append(current_time)
        
        self.log_counter += 1
        return torques
    
    def set_joint_torques(self, target_positions):
        """Set target positions for all controlled joints using double-loop control"""
        torques = self.compute_torques(target_positions)
        
        # Apply torques directly to the joints
        for joint, torque in torques.items():
            self.data.ctrl[self.joint_ids[joint]] = torque
    
    def plot_joint_data(self):
        """Create plots showing control performance for each joint"""
        plt.figure(figsize=(18, 15))
        
        # Find the minimum length among all arrays to ensure they match
        min_length = len(self.time_history)
        for joint in self.controlled_joints:
            min_length = min(min_length, len(self.position_history[joint]))
            min_length = min(min_length, len(self.velocity_history[joint]))
            min_length = min(min_length, len(self.torque_history[joint]))
            min_length = min(min_length, len(self.target_position_history[joint]))
            min_length = min(min_length, len(self.target_velocity_history[joint]))
        
        # Trim all arrays to the minimum length
        time_data = self.time_history[:min_length]
        
        for i, joint in enumerate(self.controlled_joints, 1):
            # Position plot
            plt.subplot(len(self.controlled_joints), 3, 3*i-2)
            plt.plot(time_data, self.target_position_history[joint][:min_length], 'r--', label='Target Pos')
            plt.plot(time_data, self.position_history[joint][:min_length], 'b-', label='Actual Pos')
            plt.title(f'{joint} Position')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (rad)')
            plt.legend()
            plt.grid(True)
            
            # Velocity plot
            plt.subplot(len(self.controlled_joints), 3, 3*i-1)
            plt.plot(time_data, self.target_velocity_history[joint][:min_length], 'r--', label='Target Vel')
            plt.plot(time_data, self.velocity_history[joint][:min_length], 'g-', label='Actual Vel')
            plt.title(f'{joint} Velocity')
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity (rad/s)')
            plt.legend()
            plt.grid(True)
            
            # Torque plot
            plt.subplot(len(self.controlled_joints), 3, 3*i)
            plt.plot(time_data, self.torque_history[joint][:min_length], 'm-', label='Torque')
            plt.title(f'{joint} Torque')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque (Nm)')
            plt.legend()
            plt.grid(True)
            
            # Calculate and display performance metrics
            pos_errors = (np.array(self.target_position_history[joint][:min_length]) - 
                        np.array(self.position_history[joint][:min_length]))
            pos_rmse = np.sqrt(np.mean(pos_errors**2))
            pos_max_error = np.max(np.abs(pos_errors))
            
            vel_errors = (np.array(self.target_velocity_history[joint][:min_length]) - 
                        np.array(self.velocity_history[joint][:min_length]))
            vel_rmse = np.sqrt(np.mean(vel_errors**2))
            vel_max_error = np.max(np.abs(vel_errors))
            
            plt.text(0.05, 0.85, 
                    f"Pos RMSE: {pos_rmse:.4f}\nPos Max Err: {pos_max_error:.4f}\n"
                    f"Vel RMSE: {vel_rmse:.4f}\nVel Max Err: {vel_max_error:.4f}", 
                    transform=plt.gca().transAxes, 
                    bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.show()

def create_control_window(joint_limits):
    """Create OpenCV control window with sliders"""
    cv2.namedWindow("Joint Control Panel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Joint Control Panel", 800, 600)
    
    for i, (joint, (min_val, max_val)) in enumerate(joint_limits.items()):
        min_deg = int(np.degrees(min_val))
        max_deg = int(np.degrees(max_val))
        default_pos = int((0 - min_val) / (max_val - min_val) * 100) if min_val < 0 < max_val else 50
        
        cv2.createTrackbar(
            f"{joint} ({min_deg}° to {max_deg}°)", 
            "Joint Control Panel", 
            default_pos, 100, lambda x: None
        )
    
    cv2.createTrackbar("Speed Factor", "Joint Control Panel", 50, 100, lambda x: None)
    cv2.createTrackbar("Gripper Position", "Joint Control Panel", 0, 100, lambda x: None)
    cv2.createButton("Show Plots", lambda x: None)
    cv2.createButton("Reset Plots", lambda x: None)

def get_target_positions(joint_limits):
    """Read target positions from sliders"""
    targets = {}
    speed_factor = cv2.getTrackbarPos("Speed Factor", "Joint Control Panel") / 50.0
    
    for joint, (min_val, max_val) in joint_limits.items():
        slider_pos = cv2.getTrackbarPos(
            f"{joint} ({int(np.degrees(min_val))}° to {int(np.degrees(max_val))}°)", 
            "Joint Control Panel"
        )
        target = min_val + (max_val - min_val) * (slider_pos / 100.0)
        targets[joint] = target
    
    gripper_pos = cv2.getTrackbarPos("Gripper Position", "Joint Control Panel") / 100.0
    targets['joint7'] = gripper_pos * 0.035
    targets['joint8'] = -gripper_pos * 0.035
    
    return targets

def simulate():
    # Load model and data
    model = mujoco.MjModel.from_xml_path("piper_description.xml")
    data = mujoco.MjData(model)
    
    # Initialize double-loop controller
    controller = DoubleLoopPIDController(model, data)
    
    # Create control window
    create_control_window(controller.joint_limits)
    
    # Initialize viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5
        viewer.cam.lookat = np.array([0, 0, 0.5])
        
        last_time = time.time()
        show_plots = False
        
        while viewer.is_running():
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                show_plots = True
            
            # Get target positions
            targets = get_target_positions(controller.joint_limits)
            
            # Update controller (double-loop control)
            controller.set_joint_torques(targets)
            
            # Apply gripper control directly (position control)
            data.ctrl[model.actuator("joint7_actuator").id] = targets['joint7']
            data.ctrl[model.actuator("joint8_actuator").id] = targets['joint8']
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer
            viewer.sync()
            
            # Show plots if requested
            if show_plots:
                controller.plot_joint_data()
                show_plots = False
            
            # Maintain real-time simulation
            elapsed = time.time() - last_time
            sleep_time = max(0, model.opt.timestep - elapsed)
            time.sleep(sleep_time)
            last_time = time.time()
    
    # Cleanup and show final plots
    controller.plot_joint_data()
    cv2.destroyAllWindows()

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.integral_limit = 10.0  # Anti-windup limit

    def update(self, error, dt):
        # Update integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        # Calculate derivative term
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        # Compute PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Update previous error
        self.prev_error = error
        
        return output

if __name__ == "__main__":
    simulate()