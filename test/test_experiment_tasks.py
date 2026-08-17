import numpy as np

from phd.ui.experiment_tasks import (
    Sigraph2026ScanningTask,
    get_task,
)


class _RobotApi:
    def __init__(self):
        self.current = np.zeros(6, dtype=float)
        self.sent = []
        self.scripts = []

    def get_current_positions(self):
        return self.current

    def get_current_tool_position(self):
        return (0.1, 0.2, 0.3), (1.0, 0.0, 0.0, 0.0)

    def send_positions_joint_angle(self, positions, **kwargs):
        self.sent.append((list(positions), dict(kwargs)))
        return True

    def send_request(self, command):
        self.scripts.append(command)
        return True

    @staticmethod
    def stop_and_clear_buffer():
        return "StopAndClearBuffer()"


class _UiRos:
    def __init__(self, api):
        self.robot_api = api


class _Host:
    def __init__(self, api, task=None):
        self.ui_ros = _UiRos(api)
        self.task = task
        self.messages = []
        self.stops = []
        self.running_states = []
        self.active_indices = []

    def _append_sidebar_control_message(self, message):
        self.messages.append(str(message))

    def _refresh_sigraph_scanning_points(self, active_index=None):
        self.active_indices.append(active_index)

    def _set_sigraph_scanning_controls_running(self, running):
        self.running_states.append(bool(running))

    def _stop_sidebar_control(self, show_result=True, message="Control stopped"):
        self.stops.append((bool(show_result), str(message)))
        if self.task is not None:
            self.task.on_stop(self, show_result=show_result)


def test_sigraph_scanning_records_joint_and_tool_feedback():
    api = _RobotApi()
    api.current = np.arange(6, dtype=float) * 0.1
    task = Sigraph2026ScanningTask(storage_path=None)
    host = _Host(api, task)

    assert task.capture_current_point(host)

    assert len(task.recorded_points) == 1
    np.testing.assert_allclose(task.recorded_points[0]["joints"], api.current)
    np.testing.assert_allclose(
        task.recorded_points[0]["tool_position"], [0.1, 0.2, 0.3]
    )
    assert "Recorded Set 1 point 1" in host.messages[-1]


def test_sigraph_scanning_waits_for_feedback_then_runs_points_in_order():
    api = _RobotApi()
    task = Sigraph2026ScanningTask(storage_path=None)
    task.dwell_sec = 0.0
    task.recorded_points = [
        {"joints": [0.1] * 6, "tool_position": None, "tool_quaternion": None},
        {"joints": [0.2] * 6, "tool_position": None, "tool_quaternion": None},
    ]
    host = _Host(api, task)

    assert task.on_start(host)
    assert api.sent[0][0] == [0.1] * 6
    assert api.sent[0][1]["blend_percentage"] == 0
    assert api.sent[0][1]["fine_goal"] is True

    task.on_tick(host)
    assert len(api.sent) == 1
    api.current[:] = 0.1
    task.on_tick(host)
    task.on_tick(host)
    assert api.sent[1][0] == [0.2] * 6

    api.current[:] = 0.2
    task.on_tick(host)
    task.on_tick(host)
    assert host.stops[-1][0] is True
    assert "scanning complete" in host.stops[-1][1]
    assert api.scripts == []
    assert host.running_states == [True, False]


def test_sigraph_scanning_go_to_point_commands_only_the_selected_point():
    api = _RobotApi()
    task = Sigraph2026ScanningTask(storage_path=None)
    task.velocity_rad_s = 0.35
    task.recorded_points = [
        {"joints": [0.1] * 6, "tool_position": None, "tool_quaternion": None},
        {"joints": [0.4] * 6, "tool_position": None, "tool_quaternion": None},
    ]
    host = _Host(api, task)

    assert task.go_to_point(host, 1)

    assert len(api.sent) == 1
    assert api.sent[0][0] == [0.4] * 6
    assert api.sent[0][1]["velocity"] == 0.35
    assert api.sent[0][1]["blend_percentage"] == 0
    assert api.sent[0][1]["fine_goal"] is True
    assert task._running_index is None
    assert host.stops == []
    assert host.messages[-1] == "Moving to selected scanning point 2."


def test_sigraph_scanning_reverse_run_moves_from_last_point_to_first():
    api = _RobotApi()
    task = Sigraph2026ScanningTask(storage_path=None)
    task.dwell_sec = 0.0
    task.recorded_points = [
        {"joints": [0.1] * 6, "tool_position": None, "tool_quaternion": None},
        {"joints": [0.2] * 6, "tool_position": None, "tool_quaternion": None},
        {"joints": [0.3] * 6, "tool_position": None, "tool_quaternion": None},
    ]
    host = _Host(api, task)

    task.request_reverse_run()
    assert task.on_start(host)
    assert api.sent[-1][0] == [0.3] * 6

    for target in (0.3, 0.2, 0.1):
        api.current[:] = target
        task.on_tick(host)
        task.on_tick(host)

    assert [command[0] for command in api.sent] == [
        [0.3] * 6,
        [0.2] * 6,
        [0.1] * 6,
    ]
    assert "reverse scanning complete" in host.stops[-1][1]
    assert task._running_index is None


def test_sigraph_scanning_manual_stop_clears_robot_motion_buffer():
    api = _RobotApi()
    task = Sigraph2026ScanningTask(storage_path=None)
    task.recorded_points = [
        {"joints": [0.3] * 6, "tool_position": None, "tool_quaternion": None}
    ]
    host = _Host(api, task)

    assert task.on_start(host)
    task.on_stop(host, show_result=False)

    assert api.scripts == ["StopAndClearBuffer()"]
    assert task._running_index is None


def test_sigraph_scanning_points_persist_across_task_instances(tmp_path):
    storage_path = tmp_path / "sigraph_points.json"
    api = _RobotApi()
    api.current = np.arange(6, dtype=float) * 0.1
    task = Sigraph2026ScanningTask(storage_path=storage_path)
    host = _Host(api, task)

    assert task.capture_current_point(host)
    assert storage_path.exists()

    restored = Sigraph2026ScanningTask(storage_path=storage_path)
    assert len(restored.recorded_points) == 1
    np.testing.assert_allclose(restored.recorded_points[0]["joints"], api.current)
    np.testing.assert_allclose(
        restored.recorded_points[0]["tool_position"], [0.1, 0.2, 0.3]
    )


def test_sigraph_scanning_point_edits_are_persisted(tmp_path):
    storage_path = tmp_path / "sigraph_points.json"
    task = Sigraph2026ScanningTask(storage_path=storage_path)
    task.recorded_points = [
        {"joints": [0.1] * 6, "tool_position": None, "tool_quaternion": None},
        {"joints": [0.2] * 6, "tool_position": None, "tool_quaternion": None},
    ]
    assert task.save_recorded_points()

    assert task.move_point(1, -1) == 0
    restored = Sigraph2026ScanningTask(storage_path=storage_path)
    assert restored.recorded_points[0]["joints"] == [0.2] * 6

    assert restored.remove_point(0)
    assert len(Sigraph2026ScanningTask(storage_path=storage_path).recorded_points) == 1

    assert restored.clear_points()
    assert Sigraph2026ScanningTask(storage_path=storage_path).recorded_points == []


def test_sigraph_scanning_is_registered_with_requested_label():
    task = get_task("sigraph2026_scanning")

    assert task is not None
    assert task.label == "Sigraph2026 Scanning"
    assert task.start_button_label == "Run"
