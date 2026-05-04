import math
import subprocess
import time
import re

import numpy as np
import transforms3d

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    from geometry_msgs.msg import PoseStamped
    from PyQt5.QtCore import QThread
    from sensor_msgs.msg import JointState
    from tm_msgs.srv import SendScript, SetEvent, SetPositions

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


class ROSNodeThread(QThread if ROS_AVAILABLE else object):
    """Spin a ROS node in the background using its own executor."""

    def __init__(self, node):
        if ROS_AVAILABLE:
            super().__init__()
        self.node = node
        self._stop_flag = False

    def run(self):
        if not ROS_AVAILABLE:
            return

        executor = SingleThreadedExecutor()
        executor.add_node(self.node)

        try:
            while rclpy.ok() and not self._stop_flag:
                try:
                    executor.spin_once(timeout_sec=0.0)
                    time.sleep(0.01)
                except Exception as exc:
                    print(f"[RobotController] Spin Error: {exc}")
                    break
        finally:
            try:
                executor.remove_node(self.node)
            except Exception:
                pass
            executor.shutdown()

    def stop(self):
        self._stop_flag = True
        if ROS_AVAILABLE:
            try:
                if self.isRunning():
                    self.wait(1000)
            except Exception:
                pass


class RobotController(Node if ROS_AVAILABLE else object):
    """Wrap TM Robot ROS2 services for motion, scripts, and feedback."""

    def __init__(self):
        self._set_offline_defaults()

        if not self._detect_ros_environment():
            print("[RobotController] OFFLINE MODE – skipping ROS2 calls")
            return

        self._initialize_ros_node()
        self._start_ros_thread()
        self._setup_service_clients()
        self._setup_feedback_subscriptions()

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------
    def _set_offline_defaults(self):
        self.use_ros = False
        self.service_ok = False
        self.script_ok = False
        self.event_ok = False
        self.current_positions = None
        self.current_tool_pose = None
        self.client = None
        self.send_script_client = None
        self.event_client = None
        self.ros_thread = None
        self._node_started = False

    def _detect_ros_environment(self):
        if not ROS_AVAILABLE:
            return False

        try:
            services = subprocess.check_output(
                ["ros2", "service", "list"],
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            ).decode()
        except Exception:
            return False

        self.use_ros = "/set_positions" in services
        return self.use_ros

    def _initialize_ros_node(self):
        if not rclpy.ok():
            rclpy.init()

        super().__init__("robot_controller")
        self._node_started = True

    def _start_ros_thread(self):
        self.ros_thread = ROSNodeThread(self)
        self.ros_thread.start()

    def _setup_service_clients(self):
        self.client, self.service_ok = self._create_client_checked(
            SetPositions,
            "/set_positions",
            "SetPositions unavailable – motion commands will be skipped",
        )
        self.send_script_client, self.script_ok = self._create_client_checked(
            SendScript,
            "send_script",
            "SendScript unavailable – script commands will be skipped",
        )
        self.event_client, self.event_ok = self._create_client_checked(
            SetEvent,
            "/set_event",
            "SetEvent unavailable – event commands will be skipped",
        )

    def _create_client_checked(self, srv_type, service_name, warning_text):
        client = self.create_client(srv_type, service_name)
        ok = client.wait_for_service(timeout_sec=0.5)
        if not ok:
            self.get_logger().warning(warning_text)
        return client, ok

    def _setup_feedback_subscriptions(self):
        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseStamped,
            "/tool_pose",
            self._tool_cb,
            qos_profile_sensor_data,
        )

    def shutdown(self):
        """Stop the spin thread and tear down the ROS node cleanly."""
        if self.ros_thread is not None:
            try:
                self.ros_thread.stop()
            except Exception:
                pass
            self.ros_thread = None

        if self._node_started:
            try:
                self.destroy_node()
            except Exception:
                pass
            self._node_started = False

    def stop(self):
        """Compatibility alias for callers that expect stop()."""
        self.shutdown()

    @property
    def is_available(self):
        return bool(self.use_ros and self._node_started)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Motion commands
    # ------------------------------------------------------------------
    def send_positions_joint_angle(self, positions):
        """Send a joint-space PTP via SetPositions service."""
        if not (self.use_ros and self.service_ok and self.client):
            return False

        req = SetPositions.Request()
        req.motion_type = 1
        req.positions = positions
        req.velocity = 3.14
        req.acc_time = 0.0
        req.blend_percentage = 100
        req.fine_goal = False
        self.client.call_async(req)
        return True

    def send_positions_tool_position(
        self,
        positions,
        quaternion,
        velocity=3.14,
        acc_time=0.0,
        blend_percentage=100,
        fine_goal=False,
    ):
        """Send a tool-space PTP via SetPositions service."""
        if not (self.use_ros and self.service_ok and self.client):
            return False

        euler = transforms3d.euler.quat2euler(quaternion, axes="sxyz")
        full = positions + list(euler)

        req = SetPositions.Request()
        req.motion_type = 2
        req.positions = full
        req.velocity = float(velocity)
        req.acc_time = float(acc_time)
        req.blend_percentage = int(blend_percentage)
        req.fine_goal = bool(fine_goal)
        self.client.call_async(req)
        return True

    # ------------------------------------------------------------------
    # Script commands
    # ------------------------------------------------------------------
    @staticmethod
    def convert_positions_to_script(positions):
        """Format joint angles (radians) to a PTP script command."""
        degs = [math.degrees(p) for p in positions]
        payload = ",".join(f"{d:.2f}" for d in degs)
        return f'PTP("JPP",{payload},100,0,100,true)'

    def send_request(self, command: str):
        """Send an arbitrary script command via SendScript service."""
        if not (self.use_ros and self.script_ok and self.send_script_client):
            return False

        req = SendScript.Request()
        req.id = "ping"
        req.script = command
        self.send_script_client.call_async(req)
        return True

    # ------------------------------------------------------------------
    # Velocity mode helpers (return script strings)
    # ------------------------------------------------------------------
    @staticmethod
    def enable_joint_velocity_mode():
        return "ContinueVJog()"

    @staticmethod
    def stop_joint_velocity_mode():
        return "StopContinueVmode()"

    @staticmethod
    def set_joint_velocity(v):
        return f"SetContinueVJog({','.join(map(str, v))})"

    @staticmethod
    def enable_end_effector_velocity_mode():
        return "ContinueVLine(20000,100000)"

    @staticmethod
    def suspend_end_effector_velocity_mode():
        return "SuspendContinueVmode()"

    @staticmethod
    def stop_end_effector_velocity_mode():
        return "StopContinueVmode()"

    @staticmethod
    def set_end_effector_velocity(v):
        return f"SetContinueVLine({','.join(map(str, v))})"

    @staticmethod
    def stop_and_clear_buffer():
        return "StopAndClearBuffer()"

    # ------------------------------------------------------------------
    # Feedback callbacks / getters
    # ------------------------------------------------------------------
    def _joint_cb(self, msg):
        self.current_positions = msg.position

    def get_current_positions(self):
        return self.current_positions

    def _tool_cb(self, msg):
        self.current_tool_pose = msg.pose

    def get_current_tool_position(self):
        if not self.current_tool_pose:
            return None, None

        position = self.current_tool_pose.position
        orientation = self.current_tool_pose.orientation
        return (
            (position.x, position.y, position.z),
            (orientation.w, orientation.x, orientation.y, orientation.z),
        )

    # ------------------------------------------------------------------
    # Frame-aware velocity helpers
    # ------------------------------------------------------------------
    def set_end_effector_velocity_pre_joint_frame(
        self,
        v_lin,
        v_rot=(0.0, 0.0, 0.0),
        joint=6,
        axes_map=None,
    ):
        if axes_map is None:
            axes_map = {6: "z", 5: "y", 4: "z", 3: "y", 2: "z", 1: "z"}

        pos_quat = self.get_current_tool_position()
        if not pos_quat or pos_quat == (None, None):
            return self.set_end_effector_velocity(list(v_lin) + list(v_rot))

        _, quat = pos_quat
        rotation_world_from_tool = transforms3d.quaternions.quat2mat(quat)

        if joint != 6:
            joints = self.get_current_positions()
            if not joints or len(joints) < 6:
                return self.set_end_effector_velocity(list(v_lin) + list(v_rot))

            for j in range(6, joint, -1):
                axis = axes_map.get(j, "z").lower()
                angle = float(joints[j - 1])
                c = math.cos(-angle)
                s = math.sin(-angle)

                if axis == "x":
                    correction = np.array(
                        [[1, 0, 0], [0, c, -s], [0, s, c]],
                        dtype=float,
                    )
                elif axis == "y":
                    correction = np.array(
                        [[c, 0, s], [0, 1, 0], [-s, 0, c]],
                        dtype=float,
                    )
                else:
                    correction = np.array(
                        [[c, -s, 0], [s, c, 0], [0, 0, 1]],
                        dtype=float,
                    )

                rotation_world_from_tool = rotation_world_from_tool.dot(correction)

        v_lin_world = rotation_world_from_tool.dot(np.asarray(v_lin, dtype=float))
        v_rot_world = rotation_world_from_tool.dot(np.asarray(v_rot, dtype=float))
        return self.set_end_effector_velocity(list(v_lin_world) + list(v_rot_world))

    @staticmethod
    def _normalize_frame_name(frame):
        raw = "tool" if frame is None else str(frame)
        normalized = raw.strip().lower().replace(" ", "")

        aliases = {
            "tcp": "joint6",
            "tool": "joint6",
            "world": "base",
            "base": "base",
            "j1": "joint1",
            "j2": "joint2",
            "j3": "joint3",
            "j4": "joint4",
            "j5": "joint5",
            "j6": "joint6",
        }
        normalized = aliases.get(normalized, normalized)

        if normalized in {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"}:
            return normalized

        if normalized.isdigit():
            return f"joint{normalized}"

        match = re.fullmatch(r"joint0*([1-6])", normalized)
        if match:
            return f"joint{match.group(1)}"

        return normalized

    def set_end_effector_velocity_in_frame(self, v_lin, v_rot=(0.0, 0.0, 0.0), frame="tool"):
        normalized = self._normalize_frame_name(frame)

        if normalized == "base":
            return self.set_end_effector_velocity(list(v_lin) + list(v_rot))

        if normalized.startswith("joint"):
            joint = int(normalized[5:])
            if not (1 <= joint <= 6):
                raise ValueError("jointN must be between 1 and 6")
            if joint == 1:
                return self.set_end_effector_velocity(list(v_lin) + list(v_rot))
            return self.set_end_effector_velocity_pre_joint_frame(v_lin, v_rot, joint=joint)

        raise ValueError(f"Unknown frame '{frame}'")
