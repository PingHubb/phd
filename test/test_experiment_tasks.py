import time

import numpy as np

from phd.ui.experiment_tasks import (
    CalibrationTask,
    Sigraph2026ScanningTask,
    get_task,
)


class _RobotApi:
    def __init__(self):
        self.current = np.zeros(6, dtype=float)
        self.tool_position = [0.1, 0.2, 0.3]
        self.sent = []
        self.scripts = []
        self.velocity_commands = []
        self.velocity_stops = 0

    def get_current_positions(self):
        return self.current

    def get_current_tool_position(self):
        return tuple(self.tool_position), (1.0, 0.0, 0.0, 0.0)

    def send_end_effector_velocity_in_frame(
        self, v_lin, v_rot, frame="tool", ensure_mode=True
    ):
        self.velocity_commands.append(
            (list(v_lin), list(v_rot), str(frame), bool(ensure_mode))
        )
        return True

    def exit_end_effector_velocity_mode(self, send_zero=True):
        self.velocity_stops += 1
        return True

    def send_positions_joint_angle(self, positions, **kwargs):
        self.sent.append((list(positions), dict(kwargs)))
        return True

    def send_request(self, command):
        self.scripts.append(command)
        return True

    @staticmethod
    def stop_and_clear_buffer():
        return "StopAndClearBuffer()"


class _SensorData:
    def __init__(self):
        self.frame_sequence = 1
        self.rawData = np.zeros((8, 10), dtype=float)
        self.calData = np.zeros((8, 10), dtype=float)
        self.diffData = np.zeros((8, 10), dtype=float)
        self.diffPerData = np.zeros((8, 10), dtype=float)
        self.rawDataAve = np.zeros((8, 10), dtype=float)
        self.diffDataAve = np.zeros((8, 10), dtype=float)
        self.diffPerDataAve = np.zeros((8, 10), dtype=float)


class _Sensor:
    def __init__(self):
        self.n_row = 8
        self.n_col = 10
        self.is_connected = True
        self._data = _SensorData()
        self._sensor_calibration_in_progress = False
        self.update_cal_calls = 0

    def updateCal(self):
        self.update_cal_calls += 1
        self._sensor_calibration_in_progress = True
        self.is_connected = False


class _UiRos:
    def __init__(self, api):
        self.robot_api = api
        self.sensor_functions = _Sensor()
        self._force_meter_last_raw_newtons = 0.0
        self._force_meter_last_sample_monotonic = time.monotonic()


class _Host:
    def __init__(self, api, task=None):
        self.ui_ros = _UiRos(api)
        self.task = task
        self.messages = []
        self.stops = []
        self.running_states = []
        self.active_indices = []
        self.calibration_running_states = []
        self.calibration_results = []

    def _append_sidebar_control_message(self, message):
        self.messages.append(str(message))

    def _refresh_sigraph_scanning_points(self, active_index=None):
        self.active_indices.append(active_index)

    def _set_sigraph_scanning_controls_running(self, running):
        self.running_states.append(bool(running))

    def _set_calibration_controls_running(self, running):
        self.calibration_running_states.append(bool(running))

    def _show_calibration_result(self, result):
        self.calibration_results.append(result)

    def _stop_sidebar_control(self, show_result=True, message="Control stopped"):
        self.stops.append((bool(show_result), str(message)))
        if self.task is not None:
            self.task.on_stop(self, show_result=show_result)


def test_calibration_remembers_joint_and_tool_pose_across_instances(tmp_path):
    storage_path = tmp_path / "calibration_initial_position.json"
    api = _RobotApi()
    api.current = np.arange(6, dtype=float) * 0.1
    task = CalibrationTask(storage_path=storage_path)
    host = _Host(api)

    assert task.capture_initial_position(host)
    assert storage_path.exists()
    np.testing.assert_allclose(task.initial_position["joints_rad"], api.current)
    np.testing.assert_allclose(
        task.initial_position["tool_position_m"], [0.1, 0.2, 0.3]
    )
    assert "Calibration initial position remembered" in host.messages[-1]

    restored = CalibrationTask(storage_path=storage_path)
    np.testing.assert_allclose(restored.initial_position["joints_rad"], api.current)
    np.testing.assert_allclose(
        restored.initial_position["tool_quaternion_wxyz"],
        [1.0, 0.0, 0.0, 0.0],
    )


def test_calibration_does_not_overwrite_pose_without_robot_feedback(tmp_path):
    storage_path = tmp_path / "calibration_initial_position.json"
    api = _RobotApi()
    task = CalibrationTask(storage_path=storage_path)
    host = _Host(api)
    assert task.capture_initial_position(host)
    original_position = dict(task.initial_position)

    api.current = None
    assert not task.capture_initial_position(host)
    assert task.initial_position == original_position
    assert "feedback is unavailable" in host.messages[-1]


def test_calibration_is_registered_as_one_shot_initial_position_task():
    task = get_task("calibration")

    assert task is not None
    assert task.label == "Calibration"
    assert task.start_button_label == "Remember Initial Position"
    assert task.tick_interval_ms == 50


def test_calibration_contact_approach_stops_and_saves_force_signal_data(tmp_path):
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=tmp_path)
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.contact_threshold_n = 0.1
    host = _Host(api, task)
    assert task.capture_initial_position(host)

    # Viewer index 14 in an 8x10 grid is display row 6, column 1. The
    # corresponding instantaneous source row is 1 after the top-down flip.
    data = host.ui_ros.sensor_functions._data
    data.rawData[1, 1] = 120.0
    data.calData[1, 1] = 100.0
    data.diffData[1, 1] = 20.0
    data.diffPerData[1, 1] = 20.0
    data.rawDataAve[6, 1] = 118.0
    data.diffDataAve[6, 1] = 18.0
    data.diffPerDataAve[6, 1] = 18.0

    task.request_contact_approach()
    assert task.on_start(host)
    task.on_tick(host)
    assert api.velocity_commands[-1][0] == [0.0, 0.0, -0.0002]
    assert api.velocity_commands[-1][2] == "base"

    host.ui_ros._force_meter_last_raw_newtons = 0.2
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)

    assert host.stops
    assert "Contact detected" in host.stops[-1][1]
    assert api.velocity_stops >= 1
    assert task.last_result["stop_reason"] == "contact_threshold"
    assert task.last_result["rows"][-1]["cell_index"] == 14
    assert task.last_result["rows"][-1]["display_row"] == 6
    assert task.last_result["rows"][-1]["column"] == 1
    assert task.last_result["rows"][-1]["diff_percent_ave"] == 18.0
    assert (tmp_path / task.last_result["csv_path"].split("/")[-1]).exists()
    assert host.calibration_results[-1] is task.last_result


def test_calibration_rejects_stale_force_meter_data_before_motion():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic() - 5.0

    task.request_contact_approach()
    assert not task.on_start(host)

    assert api.velocity_commands == []
    assert "Fresh HP-200 data is required" in host.messages[-1]


def test_calibration_batch_returns_recalibrates_and_repeats(tmp_path):
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=tmp_path)
    task.repeat_count = 2
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.return_settle_sec = 0.0
    task.contact_threshold_n = 0.1
    host = _Host(api, task)
    assert task.capture_initial_position(host)

    task.request_contact_approach()
    assert task.on_start(host)
    task.on_tick(host)
    assert task._phase == "approaching"

    host.ui_ros._force_meter_last_raw_newtons = 0.2
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)

    assert task._phase == "returning"
    assert len(task._batch_results) == 1
    assert api.sent[-1][0] == [0.0] * 6
    assert api.sent[-1][1]["fine_goal"] is True
    assert not host.stops

    host.ui_ros._force_meter_last_raw_newtons = 0.0
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    task.on_tick(host)
    sensor = host.ui_ros.sensor_functions
    assert task._phase == "recalibrating"
    assert sensor.update_cal_calls == 1

    sensor._sensor_calibration_in_progress = False
    sensor.is_connected = True
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    assert task._trial_number == 2
    assert task._phase == "force_baseline"

    task.on_tick(host)
    assert task._phase == "approaching"
    host.ui_ros._force_meter_last_raw_newtons = 0.25
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)

    assert host.stops
    assert "batch complete" in host.stops[-1][1].lower()
    assert task.last_result["batch_status"] == "complete"
    assert task.last_result["requested_trials"] == 2
    assert task.last_result["completed_trials"] == 2
    assert len(task.last_result["trial_results"]) == 2
    assert task.last_result["trial_results"][0]["trial_number"] == 1
    assert task.last_result["trial_results"][1]["trial_number"] == 2
    assert all(
        result["stop_reason"] == "contact_threshold"
        for result in task.last_result["trial_results"]
    )
    assert task.last_result["batch_summary_path"]
    assert task.last_result["batch_metadata_path"]
    assert (tmp_path / task.last_result["batch_summary_path"]).exists()
    assert "StopAndClearBuffer()" in api.scripts


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
