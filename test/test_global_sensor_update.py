from phd.ui.ui_initial import MyMainWindow
from phd.ui.ui_ping_camera_control import CameraControlMixin
from phd.ui.ui_ping_ui_interactions import UiInteractionsMixin
from phd.dependence.gripper_api import GripperHelper


class _Status:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = str(text)


class _HumanoidViewer:
    def __init__(self):
        self.update_calls = 0

    def request_sensor_update(self):
        self.update_calls += 1
        return True


class _Ui:
    def __init__(self):
        self.humanoid_viewer = _HumanoidViewer()
        self.plotter_update_calls = 0
        self.disconnect_calls = 0
        self.stop_motion_calls = 0

    def _on_sensor_update(self):
        self.plotter_update_calls += 1

    def disconnect_sensor_scene(self):
        self.disconnect_calls += 1
        return True

    def stop_all_motion(self):
        self.stop_motion_calls += 1
        return []


class _Harness:
    def __init__(self):
        self.ui_ros = _Ui()
        self.info_process = _Status()
        self._sensor_scene_ready = True
        self._sensor_update_backend_enabled = True
        self.experiment_stop_calls = 0
        self.keyboard_stop_calls = 0

    @staticmethod
    def _require_ui_ros(_message):
        return True

    def set_sensor_scene_ready(self, ready=True):
        self._sensor_scene_ready = bool(ready)

    def set_sensor_update_enabled(self, enabled=True):
        self._sensor_update_backend_enabled = bool(enabled)

    def _stop_sidebar_control(self, **_kwargs):
        self.experiment_stop_calls += 1

    def _disable_keyboard_tool_velocity_internal(self):
        self.keyboard_stop_calls += 1

def test_global_update_calibrates_plotter_and_humanoid_sensors():
    harness = _Harness()

    MyMainWindow._trigger_global_sensor_update(harness)

    assert harness.ui_ros.humanoid_viewer.update_calls == 1
    assert harness.ui_ros.plotter_update_calls == 1
    assert "Sensor Plotter and humanoid" in harness.info_process.text


def test_global_disconnect_releases_the_built_sensor_scene():
    harness = _Harness()

    MyMainWindow._trigger_global_sensor_disconnect(harness)

    assert harness.ui_ros.disconnect_calls == 1
    assert not harness._sensor_scene_ready
    assert harness._sensor_update_backend_enabled
    assert harness.info_process.text == "Sensor disconnected"


def test_global_stop_all_motion_stops_every_command_source_but_not_sensor():
    harness = _Harness()

    MyMainWindow._trigger_global_stop_all_motion(harness)

    assert harness.experiment_stop_calls == 1
    assert harness.keyboard_stop_calls == 1
    assert harness.ui_ros.stop_motion_calls == 1
    assert harness._sensor_scene_ready
    assert "sensor streaming remains active" in harness.info_process.text


def test_robot_software_stop_exits_velocity_and_clears_motion_buffer():
    class _RobotApi:
        def __init__(self):
            self.exits = 0
            self.scripts = []
            self._end_effector_velocity_mode_active = True
            self._joint_velocity_mode_active = True

        def exit_end_effector_velocity_mode(self, send_zero=True):
            assert send_zero
            self.exits += 1
            return True

        def send_request(self, script):
            self.scripts.append(script)
            return True

        @staticmethod
        def stop_joint_velocity_mode():
            return "StopContinueVmode()"

        @staticmethod
        def stop_and_clear_buffer():
            return "StopAndClearBuffer()"

    api = _RobotApi()
    ui = type("_MotionUi", (), {})()
    ui.robot_api = api
    ui.features = {"robot_ready": True}

    issues = UiInteractionsMixin._stop_robot_motion_commands(ui)

    assert issues == []
    assert api.exits == 1
    assert api.scripts == ["StopContinueVmode()", "StopAndClearBuffer()"]
    assert not api._end_effector_velocity_mode_active
    assert not api._joint_velocity_mode_active


def test_gripper_software_stop_holds_latest_measured_position():
    gripper = object.__new__(GripperHelper)
    gripper.data_received = True
    gripper.current_finger_pos = 0.42
    sent = []
    gripper.send_command = lambda position, force: sent.append(
        (position, force)
    ) or True

    assert GripperHelper.stop_motion(gripper)
    assert sent == [(0.42, GripperHelper.HARD_CLOSE_FORCE)]


def test_delayed_camera_lift_is_blocked_after_software_stop():
    camera = type("_Camera", (), {})()
    camera._software_motion_stopped = True
    camera.is_lifting = True
    camera.send_velocity_command = lambda *_args: (_ for _ in ()).throw(
        AssertionError("a delayed lift command must not be sent")
    )

    CameraControlMixin.perform_lift_action(camera)

    assert not camera.is_lifting


def test_camera_requires_three_stable_depth_frames_before_gripping(monkeypatch):
    class _Gripper:
        def __init__(self):
            self.close_calls = 0

        @staticmethod
        def get_pos_string():
            return "open"

        def close(self, soft=False):
            assert soft
            self.close_calls += 1

    camera = type("_Camera", (), {})()
    camera._is_shutting_down = False
    camera._software_motion_stopped = False
    camera.centering_active = True
    camera.manual_mode_active = False
    camera.grab_triggered = False
    camera.is_lifting = False
    camera._target_reached_frames = 0
    camera.gripper = _Gripper()
    camera.velocities = []
    camera.send_velocity_command = lambda *values: camera.velocities.append(values)
    camera.check_grip_result = lambda: None
    monkeypatch.setattr(
        "phd.ui.ui_ping_camera_control.QTimer.singleShot",
        lambda *_args: None,
    )
    detection = {
        "name": "mouse",
        "box": [300, 220, 340, 260],
        "depth": 0.17,
        "conf": 0.9,
    }

    for _attempt in range(2):
        CameraControlMixin.process_yolo_data(camera, [detection], (640, 480))
    assert camera.gripper.close_calls == 0

    CameraControlMixin.process_yolo_data(camera, [detection], (640, 480))

    assert camera.gripper.close_calls == 1
    assert camera.grab_triggered
