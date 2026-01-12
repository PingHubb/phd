import numpy as np
import subprocess
import transforms3d
import math

# Attempt to import ROS2 and Vision dependencies
try:
    import rclpy
    from rclpy.node import Node
    from tm_msgs.srv import SetEvent, SetPositions, SendScript
    from sensor_msgs.msg import JointState, Image  # Added Image
    from geometry_msgs.msg import PoseStamped
    from cv_bridge import CvBridge  # Added for ROS->OpenCV conversion
    import cv2  # Added for image handling
    from PyQt5.QtCore import QThread
    ROSPY_AVAILABLE = True
except ImportError as e:
    print(f"[Import Error] Missing dependency: {e}")
    ROSPY_AVAILABLE = False


class ROSNodeThread(QThread if ROSPY_AVAILABLE else object):
    """Spins the ROS node in its own thread."""
    def __init__(self, node):
        if ROSPY_AVAILABLE:
            super().__init__()
            self.node = node

    def run(self):
        if ROSPY_AVAILABLE:
            rclpy.spin(self.node)


class RobotController(Node if ROSPY_AVAILABLE else object):
    """
    RobotController wraps ROS2 service calls for motion, script commands, AND Vision.
    """
    def __init__(self):
        # 1) Auto-detect ROS2 + required service
        self.use_ros = False
        if ROSPY_AVAILABLE:
            try:
                services = subprocess.check_output(
                    ["ros2", "service", "list"], stderr=subprocess.DEVNULL
                ).decode()
                if "/set_positions" in services:
                    self.use_ros = True
            except Exception:
                self.use_ros = False

        if not self.use_ros:
            print("[RobotController] OFFLINE MODE – skipping ROS2 calls")
            self.service_ok = False
            self.script_ok = False
            self.event_ok = False
            self.current_positions = None
            self.current_tool_pose = None
            self.latest_image = None
            return

        # 2) Initialize ROS2 node
        if not rclpy.ok():
            rclpy.init()
        super().__init__('robot_controller')

        # Spin in background
        self.ros_thread = ROSNodeThread(self)
        self.ros_thread.start()

        # --- VISION SETUP ---
        # self.bridge = CvBridge()
        # self.latest_image = None
        # # Subscribe to the image topic (default is 'techman_image')
        # self.create_subscription(Image, 'techman_image', self._image_cb, 10)
        # self.get_logger().info("Subscribed to /techman_image")

        # --- SERVICES SETUP ---
        # SetPositions service
        self.client = self.create_client(SetPositions, "/set_positions")
        self.service_ok = self.client.wait_for_service(timeout_sec=0.5)
        if not self.service_ok:
            self.get_logger().warning("SetPositions unavailable")

        # SendScript service
        self.send_script_client = self.create_client(SendScript, "send_script")
        self.script_ok = self.send_script_client.wait_for_service(timeout_sec=0.5)
        if not self.script_ok:
            self.get_logger().warning("SendScript unavailable")

        # SetEvent service
        self.event_client = self.create_client(SetEvent, "/set_event")
        self.event_ok = self.event_client.wait_for_service(timeout_sec=0.5)
        if not self.event_ok:
            self.get_logger().warning("SetEvent unavailable")

        # --- MOTION SUBSCRIBERS ---
        self.current_positions = None
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.current_tool_pose = None
        self.create_subscription(PoseStamped, '/tool_pose', self._tool_cb, 10)

    # ==========================
    #      VISION METHODS
    # ==========================

    #     self._vision_job = "ros_cam"  # must match your TMflow Vision job name
    #     self._vision_busy = False
    #     self._seq = 0
    #     self.create_timer(1.0/1.0, self._vision_tick)  # 2 Hz; tune as needed
    #
    # def _vision_tick(self):
    #     if not self.script_ok or self._vision_busy:
    #         return
    #
    #     self._seq += 1
    #     req = SendScript.Request()
    #     req.id = f"vision{self._seq}"  # unique, alphanumeric
    #     req.script = f'Vision_DoJob("{self._vision_job}")'
    #     print(self._seq)
    #
    #     self._vision_busy = True
    #     fut = self.send_script_client.call_async(req)
    #     fut.add_done_callback(lambda _f: setattr(self, "_vision_busy", False))
    #
    # def _image_cb(self, msg):
    #     try:
    #         if msg.encoding == "8UC3":
    #             row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    #             img = row[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    #             self.latest_image = img.copy()
    #         else:
    #             self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to decode image: {e}")
    #
    #
    # def trigger_camera(self, job_name="ros_cam"):
    #     """
    #     Triggers the robot to run a specific vision job.
    #     Ensure 'job_name' exists in TM Flow -> Vision -> Job.
    #     """
    #     if not (self.use_ros and self.script_ok):
    #         print("Cannot trigger camera: ROS not connected")
    #         return
    #
    #     command = f'Vision_DoJob("{job_name}")'
    #     self.get_logger().info(f"Triggering Vision Job: {job_name}")
    #     self.send_request(command)
    #
    # def get_latest_image(self):
    #     """Returns the last captured OpenCV image (numpy array) or None."""
    #     return self.latest_image

    # ==========================
    #      MOTION COMMANDS
    # ==========================
    def send_positions_joint_angle(self, positions):
        if not (self.use_ros and self.service_ok): return
        req = SetPositions.Request()
        req.motion_type = 1
        req.positions = positions
        req.velocity = 3.14
        req.acc_time = 0.0
        req.blend_percentage = 100
        req.fine_goal = False
        self.client.call_async(req)

    def send_positions_tool_position(
        self, positions, quaternion,
        velocity=3.14, acc_time=0.0, blend_percentage=100, fine_goal=False
    ):
        if not (self.use_ros and self.service_ok): return
        euler = transforms3d.euler.quat2euler(quaternion, axes='sxyz')
        full = positions + list(euler)
        req = SetPositions.Request()
        req.motion_type = 2
        req.positions = full
        req.velocity = float(velocity)
        req.acc_time = float(acc_time)
        req.blend_percentage = int(blend_percentage)
        req.fine_goal = fine_goal
        self.client.call_async(req)

    # ---- Script Commands ----
    def convert_positions_to_script(self, positions):
        degs = [math.degrees(p) for p in positions]
        s = ','.join(f"{d:.2f}" for d in degs)
        return f"PTP(\"JPP\",{s},100,0,100,true)"

    def send_request(self, command: str):
        if not (self.use_ros and self.script_ok): return
        req = SendScript.Request()
        req.id = 'tm_script_cmd'
        req.script = command
        self.send_script_client.call_async(req)

    # ---- Velocity Mode Helpers ----
    def enable_joint_velocity_mode(self):       return "ContinueVJog()"
    def stop_joint_velocity_mode(self):         return "StopContinueVmode()"
    def set_joint_velocity(self, v):            return f"SetContinueVJog({','.join(map(str, v))})"
    def enable_end_effector_velocity_mode(self):    return "ContinueVLine(20000,100000)"
    def suspend_end_effector_velocity_mode(self):   return "SuspendContinueVmode()"
    def stop_end_effector_velocity_mode(self):      return "StopContinueVmode()"
    def set_end_effector_velocity(self, v):         return f"SetContinueVLine({','.join(map(str, v))})"
    def stop_and_clear_buffer(self):               return "StopAndClearBuffer()"

    # ---- Callbacks & getters ----
    def _joint_cb(self, msg):
        self.current_positions = msg.position

    def get_current_positions(self):
        return self.current_positions

    def _tool_cb(self, msg):
        self.current_tool_pose = msg.pose

    def get_current_tool_position(self):
        if not self.current_tool_pose:
            return None, None
        p = self.current_tool_pose.position
        o = self.current_tool_pose.orientation
        return (p.x, p.y, p.z), (o.w, o.x, o.y, o.z)

    def set_end_effector_velocity_pre_joint_frame(
            self, v_lin, v_rot=(0.0, 0.0, 0.0), joint: int = 6, axes_map: dict = None,
    ):
        if axes_map is None:
            axes_map = {6: 'z', 5: 'y', 4: 'z', 3: 'y', 2: 'z', 1: 'z'}
        pos_quat = self.get_current_tool_position()
        if not pos_quat:
            return self.set_end_effector_velocity(list(v_lin) + list(v_rot))
        _, quat = pos_quat
        R = transforms3d.quaternions.quat2mat(quat)
        if joint != 6:
            joints = self.get_current_positions()
            if not joints or len(joints) < 6:
                return self.set_end_effector_velocity(list(v_lin) + list(v_rot))
            for j in range(6, joint, -1):
                axis = axes_map.get(j, 'z').lower()
                q = float(joints[j - 1])
                c, s = math.cos(-q), math.sin(-q)
                if axis == 'x': R = R.dot(np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float))
                elif axis == 'y': R = R.dot(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float))
                else: R = R.dot(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float))
        v_lin_world = R.dot(np.asarray(v_lin, dtype=float))
        v_rot_world = R.dot(np.asarray(v_rot, dtype=float))
        return self.set_end_effector_velocity(list(v_lin_world) + list(v_rot_world))

    def set_end_effector_velocity_in_frame(self, v_lin, v_rot=(0.0, 0.0, 0.0), frame="tool"):
        f = (frame or "tool").strip().lower()
        aliases = {"tcp": "joint6", "tool": "joint6", "world": "base", "j1": "base", "joint1": "base", "j6": "joint6"}
        f = aliases.get(f, f)
        if f == "base": return self.set_end_effector_velocity(list(v_lin) + list(v_rot))
        if f.startswith("joint"):
            n = int(f[5:])
            if not (1 <= n <= 6): raise ValueError("jointN must be between 1 and 6")
            if n == 1: return self.set_end_effector_velocity(list(v_lin) + list(v_rot))
            return self.set_end_effector_velocity_pre_joint_frame(v_lin, v_rot, joint=n)
        raise ValueError(f"Unknown frame '{frame}'")