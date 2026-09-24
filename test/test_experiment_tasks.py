import time
from pathlib import Path

import numpy as np

from phd.ui.experiment_tasks import (
    CalibrationGraphAnalysisTask,
    CalibrationTask,
    Sigraph2026ScanningTask,
    get_task,
)


def test_graph_analysis_is_registered_as_one_shot_file_selection_task():
    task = get_task("graph_analysis")

    assert isinstance(task, CalibrationGraphAnalysisTask)
    assert task.label == "Graph Analysis"
    assert task.start_button_label == "Select Recording"
    assert task.tick_interval_ms is None


def test_graph_analysis_opens_host_dialog_without_starting_timer():
    class Host:
        def __init__(self):
            self.opened = 0

        def _open_calibration_graph_analysis(self):
            self.opened += 1

        def _append_sidebar_control_message(self, _message):
            return

    host = Host()
    task = CalibrationGraphAnalysisTask()

    assert task.on_start(host) is False
    assert host.opened == 1


class _RobotApi:
    def __init__(self):
        self.current = np.zeros(6, dtype=float)
        self.tool_position = [0.1, 0.2, 0.3]
        self.sent = []
        self.scripts = []
        self.velocity_commands = []
        self.velocity_stops = 0
        self.accept_velocity = True

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
        return self.accept_velocity

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


class _ForceMeterThread:
    def __init__(self, running=True):
        self.running = bool(running)

    def isRunning(self):
        return self.running


class _UiRos:
    def __init__(self, api):
        self.robot_api = api
        self.sensor_functions = _Sensor()
        self._force_meter_last_raw_newtons = 0.0
        self._force_meter_last_sample_monotonic = time.monotonic()
        self._force_meter_thread = _ForceMeterThread()
        self._force_meter_error_message = ""
        self.force_meter_refresh_calls = 0
        self.force_meter_start_calls = 0
        self.force_meter_stop_calls = 0
        self.force_meter_start_result = True

    def _refresh_force_meter_ports(self):
        self.force_meter_refresh_calls += 1

    def _start_force_meter(self):
        self.force_meter_start_calls += 1
        if not self.force_meter_start_result:
            return False
        self._force_meter_thread = _ForceMeterThread()
        self._force_meter_error_message = ""
        return True

    def _stop_force_meter(self):
        self.force_meter_stop_calls += 1
        self._force_meter_thread = None
        self._force_meter_last_sample_monotonic = None
        return True


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
        self.calibration_live_forces = []

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

    def _refresh_calibration_live_force(self, force_g=None, force_delta_g=None):
        self.calibration_live_forces.append((force_g, force_delta_g))

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


def test_calibration_is_registered_as_running_force_cycle():
    task = get_task("calibration")

    assert task is not None
    assert task.label == "Calibration"
    assert task.start_button_label == "Run"
    assert task.tick_interval_ms == 50


def test_calibration_operator_defaults():
    task = CalibrationTask(storage_path=None, output_directory=None)

    assert task.taxel_index == 35
    assert task.signal_field == "diff_ave"
    assert np.isclose(task.approach_speed_m_s, 0.0001)
    assert np.isclose(task.contact_threshold_g, 2000.0)
    assert task.max_travel_m == 0.0
    assert task.timeout_sec == 0.0


def test_calibration_folder_name_contains_time_and_all_seven_settings():
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.taxel_index = 35
    task.signal_field = "diff_ave"
    task.approach_speed_m_s = 0.0001
    task.contact_threshold_g = 600.0
    task.max_travel_m = 0.035
    task.timeout_sec = 120.0
    task.repeat_count = 4

    folder_name = task._experiment_folder_name("20260924_103015_123456")

    assert folder_name == (
        "calibration_20260924_103015_123456_"
        "taxel-35_signal-diff_ave_speed-0p1mmps_threshold-600g_"
        "travel-35mm_timeout-120s_reps-4"
    )


def test_calibration_count_difference_preserves_negative_sign():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    host = _Host(api, task)
    data = host.ui_ros.sensor_functions._data

    # Viewer taxel 35 maps to row 3 of the already flipped averaged matrix.
    data.diffDataAve[3, 4] = -27.0

    taxel = task._read_taxel(host)

    assert taxel is not None
    assert taxel[task.signal_field] == -27.0
    assert taxel["taxel_035_diff_ave"] == -27.0
    assert len(
        [key for key in taxel if key.startswith("taxel_")]
    ) == 80


def _advance_initial_return(task, host):
    """Reach the remembered position and enter force-baseline collection."""
    assert task._phase == "returning"
    task.on_tick(host)
    task.on_tick(host)
    assert task._phase == "force_baseline"


def test_calibration_run_returns_first_then_holds_and_saves(tmp_path):
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=tmp_path)
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.return_settle_sec = 0.0
    task.contact_threshold_n = 0.1
    host = _Host(api, task)
    assert task.capture_initial_position(host)

    # Viewer index 35 in an 8x10 grid is display row 3, column 4. The
    # corresponding instantaneous source row is 4 after the top-down flip.
    data = host.ui_ros.sensor_functions._data
    data.rawData[4, 4] = 120.0
    data.calData[4, 4] = 100.0
    data.diffData[4, 4] = 20.0
    data.diffPerData[4, 4] = 20.0
    data.rawDataAve[3, 4] = 118.0
    data.diffDataAve[3, 4] = 18.0
    data.diffPerDataAve[3, 4] = 18.0

    assert task.on_start(host)
    assert task._phase == "returning"
    assert api.sent[-1][0] == [0.0] * 6
    assert api.velocity_commands == []

    _advance_initial_return(task, host)
    task.on_tick(host)
    assert api.velocity_commands[-1][0] == [0.0, 0.0, -0.0001]
    assert api.velocity_commands[-1][2] == "base"
    assert host.calibration_live_forces[-1][0] == 0.0

    host.ui_ros._force_meter_last_raw_newtons = 0.2
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)

    assert task._phase == "contact_hold"
    assert host.calibration_live_forces[-1][1] > 20.0
    assert not host.stops
    assert api.velocity_stops >= 1
    assert task._batch_results == []

    # Simulate the TCP being 1 mm below its trial-start height at maximum force.
    api.tool_position[2] -= 0.001
    task._contact_hold_started_at -= 1.1
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    assert task._phase == "returning"
    assert task._batch_results == []
    assert api.velocity_commands[-1][0] == [0.0, 0.0, 0.0001]
    assert len(api.sent) == 1

    # The trial remains incomplete while the TCP is still below the start.
    task.on_tick(host)
    assert task._phase == "returning"
    assert not host.stops
    assert api.velocity_commands[-1][0] == [0.0, 0.0, 0.0001]

    api.tool_position[2] += 0.001
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    assert task._phase == "returning"
    assert not host.stops
    task.on_tick(host)
    assert host.stops
    assert len(task._batch_results) == 1
    assert "returned" in host.stops[-1][1]
    assert task.last_result["stop_reason"] == "contact_threshold"
    assert task.last_result["batch_status"] == "complete"
    assert task.last_result["rows"][-1]["cell_index"] == 35
    assert task.last_result["rows"][-1]["display_row"] == 3
    assert task.last_result["rows"][-1]["column"] == 4
    assert task.last_result["rows"][-1]["diff_percent_ave"] == 18.0
    assert task.last_result["rows"][-1]["taxel_035_diff_ave"] == 18.0
    assert len(task.last_result["recorded_taxel_indices"]) == 80
    assert task.last_result["rows"][-1]["force_delta_abs_g"] > 20.0
    assert task.last_result["rows"][-1]["phase"] == "returning"
    assert task.last_result["contact_threshold_g"] > 10.0
    experiment_directory = Path(task.last_result["experiment_directory"])
    assert experiment_directory.parent == tmp_path
    assert Path(task.last_result["csv_path"]).parent == experiment_directory
    assert Path(task.last_result["metadata_path"]).parent == experiment_directory
    assert Path(task.last_result["batch_summary_path"]).parent == experiment_directory
    assert Path(task.last_result["batch_metadata_path"]).parent == experiment_directory
    assert Path(task.last_result["graph_directory"]) == experiment_directory
    assert Path(task.last_result["csv_path"]).exists()
    assert host.calibration_results[-1] is task.last_result


def test_calibration_auto_connects_force_meter_before_robot_motion():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic() - 5.0
    host.ui_ros._force_meter_thread = None

    assert task.on_start(host)

    assert task._phase == "connecting_force_meter"
    assert host.ui_ros.force_meter_refresh_calls == 1
    assert host.ui_ros.force_meter_start_calls == 1
    assert api.velocity_commands == []
    assert api.sent == []

    host.ui_ros._force_meter_last_raw_newtons = 0.1
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)

    assert task._phase == "returning"
    assert api.sent[-1][0] == [0.0] * 6
    assert "live data received" in host.messages[-2]


def test_calibration_does_not_start_if_force_meter_auto_connect_fails():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic() - 5.0
    host.ui_ros._force_meter_thread = None
    host.ui_ros.force_meter_start_result = False

    assert not task.on_start(host)

    assert not task._test_active
    assert task._phase == "idle"
    assert api.velocity_commands == []
    assert api.sent == []
    assert "could not automatically start" in host.messages[-1]


def test_calibration_force_meter_auto_connect_timeout_never_moves_robot():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic() - 5.0
    host.ui_ros._force_meter_thread = None

    assert task.on_start(host)
    task._force_connection_started_at -= task.force_connection_timeout_sec + 1.0
    task.on_tick(host)

    assert host.stops
    assert "force transmitter unavailable" in host.stops[-1][1]
    assert api.sent == []
    assert api.velocity_commands == []
    assert not task._test_active
    assert task._phase == "idle"


def test_calibration_gram_threshold_converts_to_internal_newtons():
    task = CalibrationTask(storage_path=None, output_directory=None)

    task.contact_threshold_g = 50.0

    assert np.isclose(task.contact_threshold_n, 50.0 * 0.00980665)
    assert np.isclose(task.contact_threshold_g, 50.0)

    task.contact_threshold_g = 2000.0

    assert np.isclose(task.contact_threshold_n, 2000.0 * 0.00980665)
    assert np.isclose(task.contact_threshold_g, 2000.0)

    task.contact_threshold_g = 2500.0

    assert np.isclose(task.contact_threshold_g, 2000.0)


def test_calibration_stale_force_during_return_recovers_before_next_trial():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.repeat_count = 2
    task.return_settle_sec = 0.0
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.contact_hold_sec = 0.0
    task.contact_threshold_n = 0.1
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    assert task.on_start(host)

    _advance_initial_return(task, host)
    task.on_tick(host)
    api.tool_position[2] -= 0.001
    host.ui_ros._force_meter_last_raw_newtons = 0.2
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    task.on_tick(host)
    assert task._phase == "returning"

    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic() - 5.0
    task.on_tick(host)

    assert task._phase == "returning"
    assert not host.stops
    assert task._samples[-1]["force_sample_fresh"] is False
    assert task._samples[-1]["force_delta_abs_g"] is None

    api.tool_position[2] += 0.001
    task.on_tick(host)
    task.on_tick(host)
    assert task._phase == "recalibrating"
    assert len(task._batch_results) == 1

    sensor = host.ui_ros.sensor_functions
    sensor._sensor_calibration_in_progress = False
    sensor.is_connected = True
    task.on_tick(host)
    assert task._phase == "recovering_force_meter"

    task.on_tick(host)
    assert host.ui_ros.force_meter_stop_calls == 1
    assert host.ui_ros.force_meter_start_calls == 1
    assert not host.stops

    host.ui_ros._force_meter_last_raw_newtons = 0.0
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)

    assert task._trial_number == 2
    assert task._phase == "force_baseline"
    assert not host.stops


def test_calibration_stale_force_pauses_and_resumes_same_approach():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.return_settle_sec = 0.0
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    assert task.on_start(host)
    _advance_initial_return(task, host)
    task.on_tick(host)
    assert task._phase == "approaching"

    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic() - 5.0
    task.on_tick(host)

    assert not host.stops
    assert task._test_active
    assert task._phase == "paused_force_meter"
    assert task._force_paused_phase == "approaching"
    assert api.velocity_stops >= 1

    task.force_resume_stable_sec = 0.0
    task.force_resume_min_samples = 1
    host.ui_ros._force_meter_last_raw_newtons = 0.0
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)

    assert not host.stops
    assert task._test_active
    assert task._phase == "approaching"
    task.on_tick(host)
    assert api.velocity_commands[-1][0][2] < 0.0


def test_calibration_force_pause_reconnects_and_confirms_stream():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.return_settle_sec = 0.0
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.force_pause_reconnect_delay_sec = 0.0
    task.force_pause_retry_interval_sec = 0.0
    task.force_resume_stable_sec = 0.0
    task.force_resume_min_samples = 2
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    assert task.on_start(host)
    _advance_initial_return(task, host)
    task.on_tick(host)
    assert task._phase == "approaching"

    host.ui_ros._force_meter_thread.running = False
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic() - 5.0
    task.on_tick(host)
    assert task._phase == "paused_force_meter"

    task.on_tick(host)
    assert host.ui_ros.force_meter_stop_calls == 1
    assert host.ui_ros.force_meter_start_calls == 1
    assert task._phase == "paused_force_meter"

    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    assert task._phase == "paused_force_meter"

    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    assert task._phase == "approaching"
    assert task._force_pause_count == 1
    assert task._force_pause_total_sec >= 0.0


def test_calibration_slow_return_has_generous_timeout_and_stall_watchdog():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.approach_speed_m_s = 0.00001
    task._test_active = True
    task._trial_number = 1
    task._batch_total = 2
    task._start_tool_position = [0.1, 0.2, 0.34]
    api.tool_position[2] = 0.30
    host = _Host(api, task)

    assert task._begin_return_to_initial(
        host,
        time.monotonic(),
        purpose="next_trial",
    )

    expected_seconds = 0.04 / task.approach_speed_m_s
    assert task._current_return_timeout_sec >= expected_seconds * 4.0
    assert task._return_last_progress_at is not None


def test_calibration_rejected_velocity_command_stops_cycle():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.return_settle_sec = 0.0
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    assert task.on_start(host)
    _advance_initial_return(task, host)

    api.accept_velocity = False
    task.on_tick(host)

    assert host.stops
    assert "rejected" in host.stops[-1][1]
    assert api.velocity_stops >= 1


def test_calibration_max_travel_stops_before_more_motion():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.return_settle_sec = 0.0
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.max_travel_m = 0.001
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    assert task.on_start(host)
    _advance_initial_return(task, host)
    task.on_tick(host)
    sent_before_limit = len(api.velocity_commands)

    api.tool_position[2] -= 0.002
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)

    assert host.stops
    assert "travel limit" in host.stops[-1][1]
    assert len(api.velocity_commands) == sent_before_limit
    assert api.velocity_stops >= 1


def test_calibration_zero_timeout_disables_only_the_time_limit():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.return_settle_sec = 0.0
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.timeout_sec = 0.0
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    assert task.on_start(host)
    _advance_initial_return(task, host)
    task.on_tick(host)
    commands_before = len(api.velocity_commands)

    task._started_at -= 10000.0
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)

    assert not host.stops
    assert task._phase == "approaching"
    assert len(api.velocity_commands) == commands_before + 1


def test_calibration_zero_max_travel_disables_the_distance_limit():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.return_settle_sec = 0.0
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.max_travel_m = 0.0
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    assert task.on_start(host)
    _advance_initial_return(task, host)
    task.on_tick(host)
    commands_before = len(api.velocity_commands)

    api.tool_position[2] -= 1.0
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)

    assert not host.stops
    assert task._phase == "approaching"
    assert len(api.velocity_commands) == commands_before + 1


def test_calibration_batch_returns_recalibrates_and_repeats(tmp_path):
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=tmp_path)
    task.repeat_count = 2
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.return_settle_sec = 0.0
    task.contact_hold_sec = 0.0
    task.contact_threshold_n = 0.1
    host = _Host(api, task)
    assert task.capture_initial_position(host)

    assert task.on_start(host)
    _advance_initial_return(task, host)
    task.on_tick(host)
    assert task._phase == "approaching"

    host.ui_ros._force_meter_last_raw_newtons = 0.2
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    assert task._phase == "contact_hold"
    task.on_tick(host)

    assert task._phase == "returning"
    assert task._batch_results == []
    assert api.sent[-1][0] == [0.0] * 6
    assert api.sent[-1][1]["fine_goal"] is True
    assert not host.stops

    host.ui_ros._force_meter_last_raw_newtons = 0.0
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    task.on_tick(host)
    sensor = host.ui_ros.sensor_functions
    assert task._phase == "recalibrating"
    assert len(task._batch_results) == 1
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
    assert task._phase == "contact_hold"
    task.on_tick(host)
    assert task._phase == "returning"
    task.on_tick(host)
    task.on_tick(host)

    assert host.stops
    assert "batch complete" in host.stops[-1][1].lower()
    assert task.last_result["batch_status"] == "complete"
    assert task.last_result["requested_trials"] == 2
    assert task.last_result["completed_trials"] == 2
    assert len(task.last_result["trial_results"]) == 2
    assert all(
        result["rows"][-1]["phase"] == "returning"
        for result in task.last_result["trial_results"]
    )
    assert task.last_result["trial_results"][0]["trial_number"] == 1
    assert task.last_result["trial_results"][1]["trial_number"] == 2
    assert all(
        result["stop_reason"] == "contact_threshold"
        for result in task.last_result["trial_results"]
    )
    assert task.last_result["batch_summary_path"]
    assert task.last_result["batch_metadata_path"]
    experiment_directory = Path(task.last_result["experiment_directory"])
    assert experiment_directory.parent == tmp_path
    related_paths = [
        Path(task.last_result["batch_summary_path"]),
        Path(task.last_result["batch_metadata_path"]),
        *[
            Path(result[path_name])
            for result in task.last_result["trial_results"]
            for path_name in ("csv_path", "metadata_path")
        ],
    ]
    assert all(path.parent == experiment_directory for path in related_paths)
    assert all(path.exists() for path in related_paths)
    assert Path(task.last_result["graph_directory"]) == experiment_directory
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
