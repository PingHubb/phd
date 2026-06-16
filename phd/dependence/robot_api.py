import math
import subprocess
import time
import re
from types import SimpleNamespace

import numpy as np
import transforms3d

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, qos_profile_sensor_data

    from geometry_msgs.msg import PoseStamped
    from PyQt5.QtCore import QThread
    from sensor_msgs.msg import JointState
    from tm_msgs.srv import SendScript, SetEvent, SetPositions

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

if ROS_AVAILABLE:
    try:
        from service_interfaces.srv import Getangleact, Setangle, Setforce, Setspeed

        HAND_SRVS_AVAILABLE = True
    except Exception:
        HAND_SRVS_AVAILABLE = False

    try:
        from service_interfaces.msg import TouchData1

        HAND_TOUCH_AVAILABLE = True
    except Exception:
        HAND_TOUCH_AVAILABLE = False

    try:
        from service_interfaces.msg import GetAngleAct1, SetAngle1, SetForce1, SetSpeed1

        HAND_TOPIC_CMDS_AVAILABLE = True
    except Exception:
        HAND_TOPIC_CMDS_AVAILABLE = False
else:
    HAND_SRVS_AVAILABLE = False
    HAND_TOUCH_AVAILABLE = False
    HAND_TOPIC_CMDS_AVAILABLE = False


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
                    time.sleep(0.05)
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
        self.hand_srv_ok = False
        self.hand_set_angle_client = None
        self.hand_set_speed_client = None
        self.hand_set_force_client = None
        self.hand_get_angle_client = None
        self.hand_topic_ok = False
        self.hand_topic_commands_detected = False
        self.hand_set_angle_pub = None
        self.hand_set_speed_pub = None
        self.hand_set_force_pub = None
        self.hand_angle_subscription = None
        self.latest_hand_angle_data = None
        self.latest_hand_angle_time = None
        self._hand_angle_target_cache = [1720, 1720, 1720, 1720, 1350, 1000]
        self.hand_touch_subscription = None
        self.latest_hand_touch_data = None
        self.latest_hand_touch_time = None
        self._end_effector_velocity_mode_active = False
        self._joint_velocity_mode_active = False

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

        try:
            topics = subprocess.check_output(
                ["ros2", "topic", "list"],
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            ).decode()
        except Exception:
            topics = ""

        robot_ready = "/set_positions" in services
        hand_service_ready = any(
            name in services
            for name in ("/Setangle", "/Setspeed", "/Setforce", "/Getangleact")
        )
        self.hand_topic_commands_detected = any(
            name in topics
            for name in ("/set_angle_data", "/set_speed_data", "/set_force_data", "/angle_data")
        )
        hand_ready = hand_service_ready or self.hand_topic_commands_detected or "/touch_data" in topics
        self.use_ros = bool(robot_ready or hand_ready)
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
        self._setup_hand_service_clients()
        self._setup_hand_topic_interfaces()

    def _setup_hand_service_clients(self):
        if not HAND_SRVS_AVAILABLE:
            self.hand_srv_ok = False
            return

        self.hand_set_angle_client = self.create_client(Setangle, "/Setangle")
        self.hand_set_speed_client = self.create_client(Setspeed, "/Setspeed")
        self.hand_set_force_client = self.create_client(Setforce, "/Setforce")
        self.hand_get_angle_client = self.create_client(Getangleact, "/Getangleact")

        ok_angle = self.hand_set_angle_client.wait_for_service(timeout_sec=0.3)
        ok_speed = self.hand_set_speed_client.wait_for_service(timeout_sec=0.3)
        ok_force = self.hand_set_force_client.wait_for_service(timeout_sec=0.3)
        ok_get = self.hand_get_angle_client.wait_for_service(timeout_sec=0.3)
        self.hand_srv_ok = bool(ok_angle and ok_speed and ok_force and ok_get)
        if not self.hand_srv_ok:
            self.get_logger().warning("RH56F1 service API unavailable; checking topic command interface.")

    def _setup_hand_topic_interfaces(self):
        if not HAND_TOPIC_CMDS_AVAILABLE:
            self.hand_topic_ok = False
            return

        try:
            qos = QoSProfile(depth=10)
            self.hand_set_angle_pub = self.create_publisher(SetAngle1, "/set_angle_data", qos)
            self.hand_set_speed_pub = self.create_publisher(SetSpeed1, "/set_speed_data", qos)
            self.hand_set_force_pub = self.create_publisher(SetForce1, "/set_force_data", qos)
            self.hand_topic_ok = True
        except Exception as exc:
            self.hand_topic_ok = False
            try:
                self.get_logger().warning(f"RH56F1 topic interface unavailable: {exc}")
            except Exception:
                pass

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
        self.enable_hand_tactile_subscription(False)
        self._set_hand_angle_subscription_enabled(False)
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
    def send_positions_joint_angle(
        self,
        positions,
        velocity=3.14,
        acc_time=0.0,
        blend_percentage=100,
        fine_goal=False,
    ):
        """Send a joint-space PTP via SetPositions service.

        ``blend_percentage`` / ``fine_goal`` let callers force a precise stop at
        the target (blend=0, fine=True), which is required when a follow-up
        command must start from the exact joint configuration (e.g. an explicit
        J6 unwrap inserted by Task 1 to avoid the J6 ±270° wrap-around limit).
        """
        if not (self.use_ros and self.service_ok and self.client):
            return False

        self.ensure_position_mode_ready()

        req = SetPositions.Request()
        req.motion_type = 1
        req.positions = positions
        req.velocity = float(velocity)
        req.acc_time = float(acc_time)
        req.blend_percentage = int(blend_percentage)
        req.fine_goal = bool(fine_goal)
        self.client.call_async(req)
        return True

    def ensure_position_mode_ready(self, wait_sec=0.08):
        """Leave ContinueVLine/ContinueVJog before sending position commands."""
        if not (self.use_ros and self.script_ok and self.send_script_client):
            return False
        self.send_request(self.suspend_end_effector_velocity_mode())
        self.send_request(self.stop_end_effector_velocity_mode())
        self._end_effector_velocity_mode_active = False
        self._joint_velocity_mode_active = False
        if wait_sec and wait_sec > 0.0:
            time.sleep(float(wait_sec))
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

        self.ensure_position_mode_ready()

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
        if not command or not (self.use_ros and self.script_ok and self.send_script_client):
            return False

        req = SendScript.Request()
        req.id = "ping"
        req.script = command
        self.send_script_client.call_async(req)
        self._note_script_mode_transition(command)
        return True

    def _note_script_mode_transition(self, command: str):
        text = str(command or "").strip()
        if text.startswith("ContinueVLine("):
            self._end_effector_velocity_mode_active = True
            self._joint_velocity_mode_active = False
        elif text.startswith("ContinueVJog("):
            self._joint_velocity_mode_active = True
            self._end_effector_velocity_mode_active = False
        elif text.startswith("SuspendContinueVmode") or text.startswith("StopContinueVmode"):
            self._end_effector_velocity_mode_active = False
            self._joint_velocity_mode_active = False

    def enter_end_effector_velocity_mode(self, suspend_existing=False):
        """Enter TM ContinueVLine once, then keep streaming SetContinueVLine."""
        if self._end_effector_velocity_mode_active:
            return True
        if suspend_existing or self._joint_velocity_mode_active:
            self.send_request(self.suspend_end_effector_velocity_mode())
        ok = self.send_request(self.enable_end_effector_velocity_mode())
        if ok:
            self._end_effector_velocity_mode_active = True
            self._joint_velocity_mode_active = False
        return ok

    def send_end_effector_velocity(self, velocity, ensure_mode=True):
        if ensure_mode and not self.enter_end_effector_velocity_mode():
            return False
        return self.send_request(self.set_end_effector_velocity(velocity))

    def send_end_effector_velocity_in_frame(self, v_lin, v_rot=(0.0, 0.0, 0.0), frame="tool", ensure_mode=True):
        if ensure_mode and not self.enter_end_effector_velocity_mode():
            return False
        return self.send_request(self.set_end_effector_velocity_in_frame(v_lin, v_rot, frame=frame))

    def exit_end_effector_velocity_mode(self, send_zero=True):
        """Stop tool velocity mode and clear cached mode state."""
        if send_zero:
            self.send_request(self.set_end_effector_velocity([0.0] * 6))
        ok_suspend = self.send_request(self.suspend_end_effector_velocity_mode())
        ok_stop = self.send_request(self.stop_end_effector_velocity_mode())
        self._end_effector_velocity_mode_active = False
        return bool(ok_suspend or ok_stop)

    # ------------------------------------------------------------------
    # Dexterous hand helpers (RH56F1 via service_interfaces)
    # ------------------------------------------------------------------
    def _hand_service_commands_available(self):
        return bool(self.use_ros and self.hand_srv_ok and HAND_SRVS_AVAILABLE)

    def _hand_topic_commands_available(self):
        return bool(self.use_ros and self._node_started and self.hand_topic_ok and HAND_TOPIC_CMDS_AVAILABLE)

    def hand_services_available(self):
        return bool(self._hand_service_commands_available() or self._hand_topic_commands_available())

    def hand_tactile_available(self):
        return bool(self.use_ros and self._node_started and HAND_TOUCH_AVAILABLE)

    def enable_hand_tactile_subscription(self, enabled=True):
        if not self.hand_tactile_available():
            return False

        if enabled:
            if self.hand_touch_subscription is None:
                self.hand_touch_subscription = self.create_subscription(
                    TouchData1,
                    "/touch_data",
                    self._hand_touch_cb,
                    QoSProfile(depth=10),
                )
            return True

        if self.hand_touch_subscription is not None:
            try:
                self.destroy_subscription(self.hand_touch_subscription)
            except Exception:
                pass
            self.hand_touch_subscription = None
        return True

    def _set_hand_angle_subscription_enabled(self, enabled=True):
        if not (self.use_ros and self._node_started and HAND_TOPIC_CMDS_AVAILABLE):
            return False

        if enabled:
            if self.hand_angle_subscription is None:
                self.hand_angle_subscription = self.create_subscription(
                    GetAngleAct1,
                    "/angle_data",
                    self._hand_angle_cb,
                    QoSProfile(depth=10),
                )
            return True

        if self.hand_angle_subscription is not None:
            try:
                self.destroy_subscription(self.hand_angle_subscription)
            except Exception:
                pass
            self.hand_angle_subscription = None
        return True

    @staticmethod
    def _msg_sequence(value):
        if value is None:
            return []
        try:
            return list(value)
        except TypeError:
            return [value]

    def _hand_touch_cb(self, msg):
        try:
            palm_values = self._msg_sequence(getattr(msg, "plam_data", None))
        except Exception:
            palm_values = []
        self.latest_hand_touch_data = {
            "finger_forces": self._msg_sequence(getattr(msg, "finger_forces", None)),
            "finger_tangentials": self._msg_sequence(getattr(msg, "finger_tangentials", None)),
            "finger_angles": self._msg_sequence(getattr(msg, "finger_angles", None)),
            "finger_proximity": self._msg_sequence(getattr(msg, "finger_proximity", None)),
            "palm_data": palm_values,
        }
        self.latest_hand_touch_time = time.time()

    def _hand_angle_cb(self, msg):
        data = {
            "finger_ids": self._msg_sequence(getattr(msg, "finger_ids", None)),
            "angle_values": self._msg_sequence(getattr(msg, "angle_values", None)),
            "finger_names": self._msg_sequence(getattr(msg, "finger_names", None)),
        }
        self.latest_hand_angle_data = data
        self.latest_hand_angle_time = time.time()

        for finger_id, angle in zip(data["finger_ids"], data["angle_values"]):
            try:
                idx = int(finger_id) - 1
                value = int(angle)
            except Exception:
                continue
            if 0 <= idx < len(self._hand_angle_target_cache) and value >= 0:
                self._hand_angle_target_cache[idx] = value

    def get_latest_hand_tactile(self):
        if self.latest_hand_touch_data is None:
            return None
        data = {key: list(value) for key, value in self.latest_hand_touch_data.items()}
        data["timestamp"] = self.latest_hand_touch_time
        return data

    def hand_tactile_publisher_count(self):
        if not (self.use_ros and self._node_started and HAND_TOUCH_AVAILABLE):
            return 0
        try:
            return int(self.count_publishers("/touch_data"))
        except Exception:
            return 0

    @staticmethod
    def _hand_full_command_values(values, count, fill_value):
        output = list(values or [])
        if len(output) < count:
            output += [fill_value] * (count - len(output))
        return [int(v) for v in output[:count]]

    def _hand_angle_topic_values(self, angles):
        incoming = self._hand_full_command_values(angles, 6, -1)
        output = list(self._hand_angle_target_cache)
        for idx, value in enumerate(incoming):
            if value >= 0:
                output[idx] = int(value)
        self._hand_angle_target_cache = list(output)
        return output

    def _publish_hand_angles_topic(self, angles):
        if not self._hand_topic_commands_available() or self.hand_set_angle_pub is None:
            return None
        msg = SetAngle1()
        msg.finger_ids = [1, 2, 3, 4, 5, 6]
        msg.angles = self._hand_angle_topic_values(angles)
        self.hand_set_angle_pub.publish(msg)
        return {"mode": "topic", "topic": "/set_angle_data", "angles": list(msg.angles)}

    def _publish_hand_speed_topic(self, speed):
        if not self._hand_topic_commands_available() or self.hand_set_speed_pub is None:
            return None
        msg = SetSpeed1()
        msg.finger_ids = [1, 2, 3, 4, 5, 6]
        msg.speeds = [int(speed)] * 6
        self.hand_set_speed_pub.publish(msg)
        return {"mode": "topic", "topic": "/set_speed_data", "speeds": list(msg.speeds)}

    def _publish_hand_force_topic(self, forces):
        if not self._hand_topic_commands_available() or self.hand_set_force_pub is None:
            return None
        msg = SetForce1()
        msg.finger_ids = [1, 2, 3, 4, 5, 6]
        msg.forces = self._hand_full_command_values(forces, 6, 2000)
        self.hand_set_force_pub.publish(msg)
        return {"mode": "topic", "topic": "/set_force_data", "forces": list(msg.forces)}

    def _latest_hand_angles_as_result(self):
        data = self.latest_hand_angle_data
        if not data:
            return None

        values = [None] * 6
        for finger_id, angle in zip(data.get("finger_ids") or [], data.get("angle_values") or []):
            try:
                fid = int(finger_id)
                value = int(angle)
            except Exception:
                continue
            one_based_idx = fid - 1
            if 0 <= one_based_idx < 6:
                values[one_based_idx] = value
            elif 0 <= fid < 6:
                values[fid] = value

        for idx, value in enumerate(values):
            if value is None:
                values[idx] = self._hand_angle_target_cache[idx]

        return SimpleNamespace(
            angle0=int(values[0]),
            angle1=int(values[1]),
            angle2=int(values[2]),
            angle3=int(values[3]),
            angle4=int(values[4]),
            angle5=int(values[5]),
        )

    @staticmethod
    def _wait_future(future, timeout_sec=1.5):
        started = time.time()
        while not future.done():
            if (time.time() - started) >= float(timeout_sec):
                return None
            time.sleep(0.01)
        try:
            return future.result()
        except Exception:
            return None

    def hand_set_angles(self, angles, hand_id=1, timeout_sec=1.5):
        if self._hand_service_commands_available():
            req = Setangle.Request()
            req.status = "set_angle"
            req.hand_id = int(hand_id)
            values = list(angles or [])
            if len(values) < 6:
                values = values + ([-1] * (6 - len(values)))
            req.angle0, req.angle1, req.angle2, req.angle3, req.angle4, req.angle5 = [int(v) for v in values[:6]]
            fut = self.hand_set_angle_client.call_async(req)
            return self._wait_future(fut, timeout_sec=timeout_sec)
        return self._publish_hand_angles_topic(angles)

    def hand_set_speed_all(self, speed, hand_id=1, timeout_sec=1.5):
        if self._hand_service_commands_available():
            req = Setspeed.Request()
            req.status = "set_speed"
            req.hand_id = int(hand_id)
            s = int(speed)
            req.speed0 = s
            req.speed1 = s
            req.speed2 = s
            req.speed3 = s
            req.speed4 = s
            req.speed5 = s
            fut = self.hand_set_speed_client.call_async(req)
            return self._wait_future(fut, timeout_sec=timeout_sec)
        return self._publish_hand_speed_topic(speed)

    def hand_set_force_all(self, forces, hand_id=1, timeout_sec=1.5):
        if self._hand_service_commands_available():
            req = Setforce.Request()
            req.status = "set_force"
            req.hand_id = int(hand_id)
            values = list(forces or [])
            if len(values) < 6:
                values = values + ([2000] * (6 - len(values)))
            req.force0, req.force1, req.force2, req.force3, req.force4, req.force5 = [int(v) for v in values[:6]]
            fut = self.hand_set_force_client.call_async(req)
            return self._wait_future(fut, timeout_sec=timeout_sec)
        return self._publish_hand_force_topic(forces)

    def hand_get_actual_angles(self, hand_id=1, timeout_sec=1.5):
        if self._hand_service_commands_available():
            req = Getangleact.Request()
            req.status = "get_angleact"
            req.hand_id = int(hand_id)
            fut = self.hand_get_angle_client.call_async(req)
            return self._wait_future(fut, timeout_sec=timeout_sec)

        if not self._hand_topic_commands_available():
            return None

        self.latest_hand_angle_data = None
        if not self._set_hand_angle_subscription_enabled(True):
            return None

        started = time.time()
        try:
            while self.latest_hand_angle_data is None:
                if (time.time() - started) >= float(timeout_sec):
                    return None
                time.sleep(0.02)
            return self._latest_hand_angles_as_result()
        finally:
            self._set_hand_angle_subscription_enabled(False)

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
        if self.current_tool_pose is None:
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
            if joints is None or len(joints) < 6:
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
