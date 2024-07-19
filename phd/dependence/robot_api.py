import numpy as np
import rclpy
import transforms3d
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from tm_msgs.srv import SetEvent, SetPositions, SendScript
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from PyQt5.QtCore import QThread, QTimer
import pyvista as pv
import time
import math


class ROSNodeThread(QThread):
    def __init__(self, node):
        super().__init__()
        self.node = node

    def run(self):
        rclpy.spin(self.node)


class RobotController(Node):
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        super().__init__('robot_controller')

        self.ros_thread = ROSNodeThread(self)
        self.ros_thread.start()

        self.client = self.create_client(SetPositions, "/set_positions")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for SetPositions service to become available...')

        self.send_script_client = self.create_client(SendScript, 'send_script')

        self.current_positions = None
        self.subscription = self.create_subscription(JointState, '/joint_states', self.joint_position_callback, 10)

        self.current_tool_pose = None
        self.tool_pose_subscription = self.create_subscription(PoseStamped, '/tool_pose', self.tool_pose_callback, 10)

        self.my_service_client = self.create_client(SetEvent, "/set_event")
        while not self.my_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for SetEvent to become available...')

    def send_and_process_request(self, positions):
        # Convert positions from radians to a script command
        command = self.convert_positions_to_script(positions)
        self.send_request(command)
        # time.sleep(0.1)
        # command = 'StopAndClearBuffer()'
        # self.send_request(command)

    def send_request(self, command):
        request = SendScript.Request()
        request.id = 'damn'  # <-- not important can self-define
        request.script = command
        self.future = self.send_script_client.call_async(request)

    def convert_positions_to_script(self, positions):
        # Convert radians to degrees and format as a script command
        positions_deg = [math.degrees(pos) for pos in positions]
        position_str = ','.join(map(str, positions_deg))
        return f"PTP(\"JPP\",{position_str},100,0,100,true)"

    def enable_joint_velocity_mode(self,):
        print("Enabling joint velocity mode")
        return "ContinueVJog()"

    def stop_joint_velocity_mode(self):
        print("Stopping joint velocity mode")
        return "StopContinueVmode()"

    def set_joint_velocity(self, velocity):
        print(f"Setting joint velocity to {velocity}")
        velocity_str = ','.join(map(str, velocity))
        return f"SetContinueVJog({velocity_str})"

    def enable_end_effector_velocity_mode(self):
        print("Enabling end effector velocity mode")
        return "ContinueVLine()"

    def stop_end_effector_velocity_mode(self):
        print("Stopping end effector velocity mode")
        return "StopContinueVmode()"

    def set_end_effector_velocity(self, velocity):
        print(f"Setting end effector velocity to {velocity}")
        velocity_str = ','.join(map(str, velocity))
        return f"SetContinueVLine({velocity_str})"

    def stop_end_effector_velocity(self):
        print("Stopping end effector velocity")
        return "StopContinueVmode()"

    def send_positions_joint_angle(self, positions):
        req = SetPositions.Request()
        req.motion_type = 1
        req.positions = positions
        req.velocity = 3.14
        req.acc_time = 0.0
        req.blend_percentage = 100
        req.fine_goal = False
        self.client.call_async(req)

    def send_positions_tool_position(self, positions, quaternion, velocity=0.25, acc_time=0.0, blend_percentage=100, fine_goal=False):
        request = SetPositions.Request()
        euler = transforms3d.euler.quat2euler(quaternion, axes='sxyz')
        full_position = positions + list(euler)
        request.motion_type = 2
        request.positions = full_position
        request.velocity = float(velocity)
        request.acc_time = float(acc_time)
        request.blend_percentage = int(blend_percentage)
        request.fine_goal = fine_goal
        self.client.call_async(request)

    def joint_position_callback(self, msg):
        self.current_positions = msg.position

    def get_current_positions(self):
        if self.current_positions is not None:
            return self.current_positions
        else:
            print("No position data available.")
            return None

    def tool_pose_callback(self, msg):
        self.current_tool_pose = msg.pose

    def get_current_tool_position(self):
        if self.current_tool_pose is not None:
            position = self.current_tool_pose.position
            quaternion = self.current_tool_pose.orientation
            position_str = f"{position.x:.3f}, {position.y:.3f}, {position.z:.3f}"
            quaternion_str = f"{quaternion.w:.3f}, {quaternion.x:.3f}, {quaternion.y:.3f}, {quaternion.z:.3f}"
            return position_str, quaternion_str
        else:
            print("No tool pose data available.")
            return None, None

    def turn_left(self, velocity):
        print("Turning left")
        self.send_request("ContinueVJog()")
        velocity_str = ','.join(map(str, velocity))
        self.send_request(f"SetContinueVJog({velocity_str})")

    def turn_right(self, velocity):
        print("Turning right")
        self.send_request("ContinueVJog()")
        velocity_str = ','.join(map(str, velocity))
        self.send_request(f"SetContinueVJog({velocity_str})")








