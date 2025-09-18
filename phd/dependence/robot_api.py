import numpy as np
import subprocess
import transforms3d
import math
# Attempt to import ROS2 dependencies; fallback if unavailable
try:
    import rclpy
    from rclpy.node import Node
    from tm_msgs.srv import SetEvent, SetPositions, SendScript
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from PyQt5.QtCore import QThread
    ROSPY_AVAILABLE = True
except ImportError:
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
    RobotController wraps ROS2 service calls for motion and script commands.
    It auto-detects ROS availability at startup: if "/set_positions" is not found,
    it runs in offline mode and skips all ROS interactions.
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
            # placeholders so attribute lookups won't fail
            self.service_ok = False
            self.script_ok = False
            self.event_ok = False
            self.current_positions = None
            self.current_tool_pose = None
            return

        # 2) Initialize ROS2 node
        if not rclpy.ok():
            rclpy.init()
        super().__init__('robot_controller')

        # Spin in background
        self.ros_thread = ROSNodeThread(self)
        self.ros_thread.start()

        # SetPositions service
        self.client = self.create_client(SetPositions, "/set_positions")
        self.service_ok = self.client.wait_for_service(timeout_sec=0.5)
        if not self.service_ok:
            self.get_logger().warning(
                "SetPositions unavailable – motion commands will be skipped"
            )

        # SendScript service
        self.send_script_client = self.create_client(SendScript, "send_script")
        self.script_ok = self.send_script_client.wait_for_service(timeout_sec=0.5)
        if not self.script_ok:
            self.get_logger().warning(
                "SendScript unavailable – script commands will be skipped"
            )

        # SetEvent service
        self.event_client = self.create_client(SetEvent, "/set_event")
        self.event_ok = self.event_client.wait_for_service(timeout_sec=0.5)
        if not self.event_ok:
            self.get_logger().warning(
                "SetEvent unavailable – event commands will be skipped"
            )

        # Subscriptions for feedback
        self.current_positions = None
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.current_tool_pose = None
        self.create_subscription(PoseStamped, '/tool_pose', self._tool_cb, 10)

    # ---- Motion Commands (guarded) ----
    def send_positions_joint_angle(self, positions):
        """Send a joint-space PTP via SetPositions service."""
        if not (self.use_ros and self.service_ok):
            return
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
        """Send a tool-space PTP via SetPositions service."""
        if not (self.use_ros and self.service_ok):
            return
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
        """Format joint angles (radians) to a PTP script command."""
        degs = [math.degrees(p) for p in positions]
        s = ','.join(f"{d:.2f}" for d in degs)
        return f"PTP(\"JPP\",{s},100,0,100,true)"

    def send_request(self, command: str):
        """Send an arbitrary script command via SendScript service."""
        if not (self.use_ros and self.script_ok):
            return
        req = SendScript.Request()
        req.id = 'ping'
        req.script = command
        self.send_script_client.call_async(req)

    # ---- Velocity Mode Helpers (return script strings) ----
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
            self,
            v_lin,
            v_rot=(0.0, 0.0, 0.0),
            joint: int = 6,
            axes_map: dict = None,
    ):
        """
        Express v_lin/v_rot in a frame tied to JOINT <joint>:
          joint=6 → TOOL frame (no removal)
          joint=5 → remove J6 (pre-roll)
          joint=4 → remove J6,J5
          joint=3 → remove J6,J5,J4
          joint=2 → remove J6..J3
          joint=1 → (alias to BASE in dispatcher)

        NOTE: Origin remains at TCP; we only change axes.
        """
        # Adjust once if TM5-900 joint axes differ
        if axes_map is None:
            axes_map = {6: 'z', 5: 'y', 4: 'z', 3: 'y', 2: 'z', 1: 'z'}

        pos_quat = self.get_current_tool_position()
        if not pos_quat:
            return self.set_end_effector_velocity(list(v_lin) + list(v_rot))

        # correct unpack: ((px,py,pz), (qw,qx,qy,qz))
        _, quat = pos_quat
        R = transforms3d.quaternions.quat2mat(quat)

        # TOOL frame needs no joint states
        if joint != 6:
            joints = self.get_current_positions()
            if not joints or len(joints) < 6:
                return self.set_end_effector_velocity(list(v_lin) + list(v_rot))

            # peel off downstream joints J6..J(joint+1)
            for j in range(6, joint, -1):  # 6,5,4,...,(joint+1)
                axis = axes_map.get(j, 'z').lower()
                q = float(joints[j - 1])  # 0-based index
                c, s = math.cos(-q), math.sin(-q)
                if axis == 'x':
                    R = R.dot(np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float))
                elif axis == 'y':
                    R = R.dot(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float))
                else:  # 'z'
                    R = R.dot(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float))

        v_lin_world = R.dot(np.asarray(v_lin, dtype=float))
        v_rot_world = R.dot(np.asarray(v_rot, dtype=float))
        return self.set_end_effector_velocity(list(v_lin_world) + list(v_rot_world))

    def set_end_effector_velocity_in_frame(self, v_lin, v_rot=(0.0, 0.0, 0.0), frame="tool"):
        """
        frame accepts:
          - "base" (world-fixed axes)
          - "tool", "tcp", or "joint6" (tool axes)
          - "joint5", "joint4", "joint3", "joint2"
          - "joint1" (alias to base)

        Examples:
          frame="base"   or "joint1"  -> base axes
          frame="tool"   or "joint6"  -> tool axes
          frame="joint5"               -> J5-aligned axes (pre-roll)
          frame="joint4"/"joint3"/"joint2" -> pre-wrist / pre-forearm / pre-shoulder
        """
        # normalize & aliases
        f = (frame or "tool").strip().lower()
        aliases = {
            "tcp": "joint6",
            "tool": "joint6",
            "world": "base",
            "j1": "base",
            "joint1": "base",
            "j6": "joint6",
        }
        f = aliases.get(f, f)

        if f == "base":
            return self.set_end_effector_velocity(list(v_lin) + list(v_rot))

        if f.startswith("joint"):
            n = int(f[5:])
            if not (1 <= n <= 6):
                raise ValueError("jointN must be between 1 and 6")
            if n == 1:
                # joint1 == base by your requirement
                return self.set_end_effector_velocity(list(v_lin) + list(v_rot))
            return self.set_end_effector_velocity_pre_joint_frame(v_lin, v_rot, joint=n)

        raise ValueError(f"Unknown frame '{frame}'")


