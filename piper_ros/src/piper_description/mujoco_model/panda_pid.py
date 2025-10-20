import mujoco
import numpy as np
import mujoco.viewer
import matplotlib.pyplot as plt
import time
import math

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class Test:
    def __init__(self, path):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.path = path
        self.positions = []
        self.torques = []
        
        # Initialize PID controller
        kp = 10.0
        ki = 1.0
        kd = 5.0
        self.pid = PIDController(kp, ki, kd)
        self.target_position = 3.14 / 4
        self.dt = self.model.opt.timestep

    def run_simulation(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Set camera parameters
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -30
            viewer.cam.distance = 3.0

            while viewer.is_running():
                step_start = time.time()
                
                # Get current joint position
                current_position = self.data.qpos[0]
                
                # Calculate error
                error = self.target_position - current_position
                
                # Use PID controller to compute torque
                torque = self.pid.update(error, self.dt)
                print(f"error: {error}, Current Position: {current_position}, Target Position: {self.target_position}, Torque: {torque}")
                
                # Set joint torque
                self.data.ctrl[0] = torque
                
                # Record data
                self.positions.append(current_position)
                self.torques.append(torque)
                
                # Step the simulation
                mujoco.mj_step(self.model, self.data)
                
                # Sync viewer with simulation
                viewer.sync()
                
                # Check if target is reached
                if math.fabs(error) < 0.001:
                    self.plot_results()
                    self.target_position = 0.0
                
                # Add small delay to control simulation speed
                time.sleep(max(0, self.dt - (time.time() - step_start)))

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.positions, label='Joint Position')
        plt.axhline(y=self.target_position, color='r', linestyle='--', label='Target Position')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.torques, label='Torque')
        plt.xlabel('Time Step')
        plt.ylabel('Torque')
        plt.legend()

        plt.show()

if __name__ == "__main__":
    test = Test("/home/khalillee/sim1_ws/src/piper_ros/src/piper_description/mujoco_model/scene.xml")
    test.run_simulation()