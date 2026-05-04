import time
from threading import Thread
from typing import Optional

try:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from control_msgs.action import GripperCommand
    from sensor_msgs.msg import JointState

    ROS_AVAILABLE = True
except Exception:
    rclpy = None
    ActionClient = None
    SingleThreadedExecutor = None
    Node = object
    QoSProfile = None
    ReliabilityPolicy = None
    DurabilityPolicy = None
    HistoryPolicy = None
    GripperCommand = None
    JointState = None
    ROS_AVAILABLE = False


class GripperHelper(Node if ROS_AVAILABLE else object):
    """Small helper around the Robotiq gripper ROS2 action + joint-state feedback.

    The class stays importable even when ROS2 or the gripper packages are not installed.
    In that case it enters offline mode and all command methods become safe no-ops.
    """

    STATUS_FAILED = "FAILED"
    STATUS_EMPTY = "EMPTY"
    STATUS_GRIPPED = "GRIPPED"

    ACTION_LIFT = "LIFT"
    ACTION_RETRY = "RETRY"
    ACTION_RETRY_OPEN = "RETRY_OPEN"
    ACTION_MANUAL = "MANUAL"

    TARGET_JOINT = "robotiq_85_left_knuckle_joint"
    OPEN_LIMIT = 0.05
    CLOSED_LIMIT = 0.78
    SOFT_CLOSE_FORCE = 35.0
    HARD_CLOSE_FORCE = 100.0
    DEFAULT_OPEN_POSITION = 0.0
    DEFAULT_CLOSED_POSITION = 1.0

    def __init__(
        self,
        action_name: str = "/robotiq_gripper_controller/gripper_cmd",
        joint_state_topic: str = "/joint_states",
        action_timeout_sec: float = 0.5,
        spin_sleep_sec: float = 0.01,
        auto_start_spin: bool = True,
    ):
        self.action_name = action_name
        self.joint_state_topic = joint_state_topic
        self.action_timeout_sec = action_timeout_sec
        self.spin_sleep_sec = spin_sleep_sec

        self.current_finger_pos = 0.0
        self.data_received = False
        self._stop_event = False
        self._spinner_thread: Optional[Thread] = None
        self._action_client = None
        self._subscriber = None
        self._is_online = False
        self._owns_rclpy_context = False

        if not ROS_AVAILABLE:
            print("[GripperHelper] OFFLINE MODE – ROS2 dependencies not available.")
            return

        self._ensure_rclpy()
        super().__init__(f"gripper_gui_client_{int(time.time())}")
        self._setup_ros_interfaces()

        if auto_start_spin:
            self.start_spinner()

    # ------------------------------------------------------------------
    # ROS setup / lifecycle
    # ------------------------------------------------------------------
    def _ensure_rclpy(self) -> None:
        if rclpy is None:
            return
        try:
            if not rclpy.ok():
                rclpy.init()
                self._owns_rclpy_context = True
        except Exception as exc:
            print(f"[GripperHelper] Failed to initialize ROS2: {exc}")

    def _setup_ros_interfaces(self) -> None:
        try:
            self._action_client = ActionClient(self, GripperCommand, self.action_name)
            qos_profile = self._build_qos_profile()
            self._subscriber = self.create_subscription(
                JointState,
                self.joint_state_topic,
                self._joint_state_callback,
                qos_profile,
            )
            self._is_online = True
        except Exception as exc:
            self._is_online = False
            print(f"[GripperHelper] Failed to create ROS interfaces: {exc}")

    @staticmethod
    def _build_qos_profile() -> QoSProfile:
        return QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

    def start_spinner(self) -> None:
        if not self._is_online:
            return
        if self._spinner_thread is not None and self._spinner_thread.is_alive():
            return

        self._stop_event = False
        self._spinner_thread = Thread(target=self._spin_node, daemon=True)
        self._spinner_thread.start()

    def _spin_node(self) -> None:
        executor = SingleThreadedExecutor()
        try:
            executor.add_node(self)
            while rclpy.ok() and not self._stop_event:
                try:
                    executor.spin_once(timeout_sec=0.0)
                    time.sleep(self.spin_sleep_sec)
                except Exception as exc:
                    print(f"[GripperHelper] Spin Error: {exc}")
                    break
        finally:
            try:
                executor.remove_node(self)
            except Exception:
                pass
            try:
                executor.shutdown()
            except Exception:
                pass

    def stop(self, shutdown_rclpy: bool = False) -> None:
        """Stop the background spinner and destroy the node safely.

        `shutdown_rclpy` defaults to False because the app may have other ROS nodes
        running (for example the robot controller).
        """
        self._stop_event = True

        if self._spinner_thread is not None and self._spinner_thread.is_alive():
            self._spinner_thread.join(timeout=1.0)

        if ROS_AVAILABLE and self._is_online:
            try:
                self.destroy_node()
            except Exception:
                pass

        if shutdown_rclpy and ROS_AVAILABLE and self._owns_rclpy_context:
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    def shutdown(self, shutdown_rclpy: bool = False) -> None:
        self.stop(shutdown_rclpy=shutdown_rclpy)

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------
    def _joint_state_callback(self, msg) -> None:
        if self.TARGET_JOINT not in msg.name:
            return

        if not self.data_received:
            print("✅ CONNECTION SUCCESS! Gripper Position found.")
            self.data_received = True

        index = msg.name.index(self.TARGET_JOINT)
        try:
            self.current_finger_pos = float(msg.position[index])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return bool(self._is_online and self._action_client is not None)

    @property
    def is_online(self) -> bool:
        return self.is_available

    def send_command(self, position: float, force: float = HARD_CLOSE_FORCE) -> bool:
        if not self.is_available:
            print("⚠️ [GripperHelper] Gripper is offline; command skipped.")
            return False

        try:
            if not self._action_client.wait_for_server(timeout_sec=self.action_timeout_sec):
                print("⚠️ [GripperHelper] ERROR: Action Server not found!")
                return False

            goal_msg = GripperCommand.Goal()
            goal_msg.command.position = float(position)
            goal_msg.command.max_effort = float(force)
            self._action_client.send_goal_async(goal_msg)
            return True
        except Exception as exc:
            print(f"⚠️ [GripperHelper] Failed to send gripper command: {exc}")
            return False

    def set_slider_pos(self, value_0_to_100: float) -> bool:
        value = max(0.0, min(100.0, float(value_0_to_100)))
        return self.send_command(value / 100.0, force=self.HARD_CLOSE_FORCE)

    def open(self) -> bool:
        print("[Gripper] Opening (0.0)...")
        return self.send_command(self.DEFAULT_OPEN_POSITION, force=self.HARD_CLOSE_FORCE)

    def close(self, soft: bool = False) -> bool:
        force = self.SOFT_CLOSE_FORCE if soft else self.HARD_CLOSE_FORCE
        print(f"[Gripper] Closing to 1.0 with Force {force}...")
        return self.send_command(self.DEFAULT_CLOSED_POSITION, force=force)

    # ------------------------------------------------------------------
    # Status / UI helpers
    # ------------------------------------------------------------------
    def get_state(self) -> str:
        pos = self.current_finger_pos
        if pos < self.OPEN_LIMIT:
            return self.STATUS_FAILED
        if pos > self.CLOSED_LIMIT:
            return self.STATUS_EMPTY
        return self.STATUS_GRIPPED

    def get_pos_string(self) -> str:
        return f"{self.current_finger_pos:.4f}"

    def evaluate_grip_attempt(self, current_fail_count: int):
        """Analyze the gripper result and tell the UI what to do next."""
        state = self.get_state()
        pos_str = self.get_pos_string()

        print(f"4. Position AFTER closing: {pos_str} -> State: {state}")

        if state == self.STATUS_GRIPPED:
            print("✅ SUCCESS: Object Gripped!")
            return self.ACTION_LIFT, 0

        if state == self.STATUS_FAILED:
            print("⚠️ FAIL: Gripper didn't move/close.")
            return self.ACTION_RETRY, current_fail_count

        if state == self.STATUS_EMPTY:
            print("❌ FAIL: Gripper Empty.")
            new_count = current_fail_count + 1
            print(f"   -> Fail Count: {new_count}/2")
            if new_count >= 2:
                print("❗ 2 Failed Attempts. Switching to MANUAL MODE.")
                return self.ACTION_MANUAL, new_count
            print("   -> Retrying Auto-Grip...")
            return self.ACTION_RETRY_OPEN, new_count

        return self.ACTION_RETRY, current_fail_count

    def __del__(self):
        try:
            self.stop(shutdown_rclpy=False)
        except Exception:
            pass
