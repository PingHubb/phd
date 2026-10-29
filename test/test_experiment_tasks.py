import copy
import json
import time
from pathlib import Path

import numpy as np
import pytest

from phd.ui.calibration_analysis import with_trial_rows
from phd.ui.experiment_tasks import (
    CalibrationGraphAnalysisTask,
    CalibrationTask,
    FullTaxelCalibrationTask,
    Sigraph2026ScanningTask,
    TaxelPositionMappingTask,
    get_task,
    task_definitions,
)
from phd.ui.taxel_position_navigation import MappedTaxelMoveTask, prepare_taxel_map


def test_demo_tasks_are_not_exposed_in_experiment_controls():
    task_ids = {
        task_id for _label, task_id, _description, _category in task_definitions()
    }

    assert "hello_world" not in task_ids
    assert "sensor_peak_change" not in task_ids
    assert get_task("hello_world") is None
    assert get_task("sensor_peak_change") is None


def test_graph_analysis_is_registered_as_one_shot_file_selection_task():
    task = get_task("graph_analysis")

    assert isinstance(task, CalibrationGraphAnalysisTask)
    assert task.label == "Graph Analysis"
    assert task.start_button_label == "Select Graph"
    assert task.tick_interval_ms is None


def test_long_running_force_tasks_request_disconnect_on_stop():
    assert CalibrationTask().disconnect_force_meter_on_stop
    assert FullTaxelCalibrationTask().disconnect_force_meter_on_stop
    assert TaxelPositionMappingTask().disconnect_force_meter_on_stop
    assert not MappedTaxelMoveTask.disconnect_force_meter_on_stop


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
    @property
    def current_feedback_state_time(self):
        return getattr(self, "_feedback_stamp", time.monotonic())

    @current_feedback_state_time.setter
    def current_feedback_state_time(self, value):
        self._feedback_stamp = value

    def __init__(self):
        self.current = np.zeros(6, dtype=float)
        self.tool_position = [0.1, 0.2, 0.3]
        self.tool_quaternion = [1.0, 0.0, 0.0, 0.0]
        self.sent = []
        self.scripts = []
        self.velocity_commands = []
        self.velocity_stops = 0
        self.accept_velocity = True
        self.project_speed_percent = 100.0

    def get_current_positions(self):
        return self.current

    def get_current_tool_position(self):
        return tuple(self.tool_position), tuple(self.tool_quaternion)

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

    def send_positions_tool_position(self, positions, quaternion, **kwargs):
        self.sent.append(
            (list(positions), list(quaternion), dict(kwargs))
        )
        # Arrive instantly so feedback matches the commanded target.
        self.tool_position = list(positions)
        self.tool_quaternion = list(quaternion)
        return True

    def get_project_speed_percent(self):
        return self.project_speed_percent

    def send_request(self, command):
        self.scripts.append(command)
        return True

    @staticmethod
    def stop_and_clear_buffer():
        return "StopAndClearBuffer()"


class _SensorData:
    @property
    def frame_monotonic_sec(self):
        return getattr(self, "_frame_stamp", time.monotonic())

    @frame_monotonic_sec.setter
    def frame_monotonic_sec(self, value):
        self._frame_stamp = value

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
        # Existing lifecycle tests use a one-frame/zero-dwell baseline.
        # Production defaults and real time/freshness are exercised separately.
        if isinstance(task, CalibrationTask):
            task.sensor_baseline_frames = 1
            task.sensor_baseline_duration_sec = 0.0
            task.recovery_observation_sec = 0.0
            task.velocity_rest_sec = 0.0
            task.post_save_feedback_stable_sec = 0.0
            # These lifecycle tests cover the legacy velocity path; stepwise
            # motion (the default) has its own simulated tests.
            task.motion_mode = "velocity"
        self.messages = []
        self.stops = []
        self.running_states = []
        self.active_indices = []
        self.calibration_running_states = []
        self.calibration_results = []
        self.proximity_profile_results = []
        self.full_taxel_calibration_running_states = []
        self.full_taxel_calibration_progress = []
        self.full_taxel_calibration_results = []
        self.calibration_live_forces = []
        self.calibration_travels = []
        self.taxel_mapping_running_states = []
        self.taxel_mapping_progress = []
        self.taxel_mapping_live_maps = []
        self.taxel_mapping_results = []

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

    def _show_proximity_profile_result(self, result):
        self.proximity_profile_results.append(result)

    def _set_full_taxel_calibration_controls_running(self, running):
        self.full_taxel_calibration_running_states.append(bool(running))

    def _refresh_full_taxel_calibration_progress(self, **kwargs):
        self.full_taxel_calibration_progress.append(dict(kwargs))

    def _show_full_taxel_calibration_result(self, result):
        self.full_taxel_calibration_results.append(result)

    def _refresh_calibration_live_force(self, force_g=None, force_delta_g=None):
        self.calibration_live_forces.append((force_g, force_delta_g))

    def _refresh_calibration_travel(self, current_mm=None, maximum_mm=None):
        self.calibration_travels.append((current_mm, maximum_mm))

    def _set_taxel_mapping_controls_running(self, running):
        self.taxel_mapping_running_states.append(bool(running))

    def _refresh_taxel_mapping_progress(self, **kwargs):
        self.taxel_mapping_progress.append(dict(kwargs))

    def _refresh_taxel_mapping_live_taxels(self, **kwargs):
        self.taxel_mapping_live_maps.append(copy.deepcopy(kwargs))

    def _show_taxel_position_map(self, result):
        self.taxel_mapping_results.append(result)

    def _stop_sidebar_control(self, show_result=True, message="Control stopped"):
        self.stops.append((bool(show_result), str(message)))
        if self.task is not None:
            self.task.on_stop(self, show_result=show_result)


def test_five_by_five_coarse_scan_fits_its_own_neighbours_not_the_eight_by_ten_pairs():
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    task.sensor_rows = 5
    task.sensor_columns = 5
    task.anchor_taxel = 12
    centres = {
        12: np.array([0.0, 0.0, 0.15]),
        11: np.array([-0.0055, 0.0067, 0.15]),
        13: np.array([0.0058, -0.0046, 0.15]),
        7: np.array([0.0059, 0.0069, 0.15]),
        17: np.array([-0.0064, -0.0050, 0.15]),
    }
    guesses = {index: position for index, position in centres.items()}

    assert task.lattice_vectors_from_points(guesses) is None
    vectors = task.lattice_vectors_from_points(guesses, (11, 13), (7, 17))
    assert vectors is not None
    row_vector, column_vector, orthogonality = vectors
    assert np.linalg.norm(row_vector) > 0.0005
    assert np.linalg.norm(column_vector) > 0.0005
    assert orthogonality >= 0.25

    task._active = True
    task._current_taxel = 12
    task._current_context = "anchor"
    task._probe_onset_pose = np.array([0.0, 0.0, 0.15])
    task.anchor_pose = {
        "tool_position_m": [0.0, 0.0, 0.2],
        "tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    task._scan_results = []
    task._scan_baseline = np.zeros(25)
    task._far_baseline = np.zeros(25)

    def refine(_samples, _baseline, index):
        return {
            "position": centres[index],
            "signal_counts": 80.0,
            "competing_signal_counts": 10.0,
            "dominance_ratio": 2.0,
        }

    task.refine_xy_from_samples = staticmethod(refine)
    host = _Host(_RobotApi(), task)
    task._finish_coarse_scan(host)

    assert task._positions == {}
    assert task._initial_row_vector_xy is not None
    assert task._initial_column_vector_xy is not None
    assert set(task._neighbor_guesses) == {11, 13, 7, 17}
    assert not host.stops


def test_taxel_mapping_uses_the_centre_taxel_on_a_five_by_five_sensor(monkeypatch):
    selected = {"key": "sensor:testing_5x5_1"}

    def fake_sensor(_host):
        if selected["key"] == "sensor:testing_8x10":
            return "sensor:testing_8x10", "Testing Sensor", 8, 10
        return "sensor:testing_5x5_1", "Testing Sensor 1", 5, 5

    monkeypatch.setattr(
        "phd.ui.experiment_tasks.connected_calibration_sensor",
        fake_sensor,
    )
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    host = _Host(_RobotApi())
    task.use_connected_sensor(host)

    assert (task.sensor_rows, task.sensor_columns, task.anchor_taxel) == (5, 5, 12)
    assert task.anchor_neighbour_indices() == (11, 13, 7, 17)
    assert task.start_button_label == "Map 5x5 Sensor"

    selected["key"] = "sensor:testing_8x10"
    task.use_connected_sensor(host)
    assert task.anchor_taxel == 35
    assert task.anchor_neighbour_indices() == (34, 36, 27, 43)


def test_taxel_position_mapping_is_registered_with_safe_defaults():
    task = get_task("taxel_position_mapping")

    assert isinstance(task, TaxelPositionMappingTask)
    assert task.label == "Taxel Position Mapping"
    assert task.start_button_label == "Map 8x10 Sensor"
    assert task.tick_interval_ms == 50
    assert task.anchor_taxel == 35
    assert task.taxel_row_column(35) == (3, 4)
    assert task.taxel_index(3, 4) == 35
    assert task.proximity_detection_mode == "robust_shift"
    assert np.isclose(task.adaptive_margin_fraction, 0.30)
    assert np.isclose(task.adaptive_min_margin_counts, 6.0)
    assert task.baseline_frame_count == 100
    assert task.probe_window_frames == 7
    assert np.isclose(task.reference_probe_speed_m_s, 0.003)
    assert np.isclose(task.probe_speed_m_s, 0.003)
    assert np.isclose(task.probe_confirmation_speed_m_s, 0.0005)
    assert np.isclose(task.probe_step_m, 0.0005)
    assert task.probe_sample_frames == 20
    assert np.isclose(task.max_probe_descent_m, 0.100)
    assert np.isclose(task.reference_max_descent_m, 0.100)
    assert np.isclose(task.probe_start_margin_m, 0.002)
    assert task.probe_confirm_frames == 2
    assert task.centre_scan_signal_multiplier == pytest.approx(2.0)
    assert task.centre_scan_max_extra_descent_m == pytest.approx(0.020)
    assert task.final_min_dominance == pytest.approx(0.95)
    assert task.verify_frame_count == 20
    assert np.isclose(task.force_guard_g, 20.0)
    assert np.isclose(task.MAX_PROBE_SPEED_M_S, 0.005)
    assert np.isclose(task.MAX_PROBE_DESCENT_M, 0.100)


def test_taxel_mapping_adaptive_envelope_matches_recorded_noise_range():
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    task.proximity_detection_mode = "adaptive_range"
    mean = np.zeros(4, dtype=float)
    minimum = np.asarray([0.0, -5.0, 0.0, 0.0])
    maximum = np.asarray([0.0, 5.0, 0.0, 0.0])

    detection = task._detection_parameters(mean, minimum, maximum, 1)

    assert detection["baseline_range_counts"] == pytest.approx(10.0)
    assert detection["margin_counts"] == pytest.approx(6.0)
    assert detection["lower_trigger_counts"] == pytest.approx(-11.0)
    assert detection["upper_trigger_counts"] is None
    assert not task._below_detection_threshold(-10.9, detection)
    assert task._below_detection_threshold(-11.0, detection)
    assert not task._below_detection_threshold(8.0, detection)


def test_taxel_mapping_robust_shift_ignores_extremes_and_detects_median_drop():
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    pattern = np.asarray([-16.0, 16.0, 0.0, 1.0, -1.0, 0.0, 0.0])
    series = np.resize(pattern, task.baseline_frame_count)
    frames = series.reshape(-1, 1)
    mean = np.mean(frames, axis=0)

    detection = task._detection_parameters(
        mean,
        np.min(frames, axis=0),
        np.max(frames, axis=0),
        0,
        baseline_frames=frames,
    )

    assert detection["baseline_min_counts"] == pytest.approx(-16.0)
    assert detection["baseline_max_counts"] == pytest.approx(16.0)
    assert detection["effective_threshold_counts"] == pytest.approx(6.0)
    assert task._below_detection_threshold(-6.0, detection)
    assert not task._below_detection_threshold(3.0, detection)


def test_taxel_mapping_robust_stability_accepts_normal_wide_extremes():
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    pattern = np.asarray([-16.0, 16.0, 0.0, 1.0, -1.0, 0.0, 0.0])
    frames = np.zeros((task.baseline_frame_count, 80), dtype=float)
    frames[:, task.anchor_taxel] = np.resize(
        pattern, task.baseline_frame_count
    )

    stability = task._baseline_stability_stats(
        "reference_baseline", frames
    )

    assert stability["target_range_counts"] == pytest.approx(32.0)
    assert stability["target_rolling_median_range_counts"] == pytest.approx(0.0)


def test_taxel_mapping_estimates_moved_sensor_rotation_and_translation():
    angle = np.radians(37.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    translation = np.asarray([0.23, -0.17])
    old = np.asarray(
        [[0.0, 0.0], [0.02, 0.0], [0.0, 0.03], [0.02, 0.03]]
    )
    new = (rotation @ old.T).T + translation

    fitted = TaxelPositionMappingTask.estimate_rigid_transform_2d(old, new)

    assert fitted is not None
    np.testing.assert_allclose(fitted["rotation"], rotation, atol=1e-10)
    np.testing.assert_allclose(fitted["translation"], translation, atol=1e-10)
    assert fitted["angle_deg"] == pytest.approx(37.0)
    assert fitted["rms_residual_mm"] == pytest.approx(0.0, abs=1e-9)


def test_completed_map_can_start_relocation_from_a_new_anchor(tmp_path):
    api = _RobotApi()
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json", output_directory=None
    )
    host = _Host(api, task=task)
    sensor = host.ui_ros.sensor_functions
    sensor.n_row = 5
    sensor.n_col = 5
    sensor._data.diffDataAve = np.zeros((5, 5), dtype=float)
    positions = []
    for index in range(25):
        row, column = TaxelPositionMappingTask.taxel_row_column(index, 5)
        centre = [column * 0.004, row * 0.004, 0.2]
        positions.append(
            {
                "taxel_index": index,
                "row": row,
                "column": column,
                "status": "measured",
                "tool_position_m": centre,
                "safe_approach_position_m": [centre[0], centre[1], 0.25],
            }
        )
    source = {
        "task": "taxel_position_mapping",
        "completed": True,
        "settings": {
            "sensor_rows": 5,
            "sensor_columns": 5,
            "anchor_taxel": 12,
        },
        "positions": positions,
    }

    valid, message = task.validate_relocation_result(source)
    assert valid, message
    assert task.capture_anchor_pose(host)
    accepted, message = task.request_relocation(source, tmp_path / "old.json")
    assert accepted, message
    assert task.on_start(host)

    assert task._relocation_active
    assert not task._relocation_continues_mapping
    assert len(task._relocation_source_positions) == 25
    assert task._move_action == "anchor_safe_arrived"
    assert task._relocation_source_path == (tmp_path / "old.json").resolve()
    task.on_stop(host, show_result=False)


def _partial_relocation_source():
    positions = []
    for index in (12, 7, 11, 13, 17, 6):
        row, column = TaxelPositionMappingTask.taxel_row_column(index, 5)
        centre = [column * 0.004, row * 0.004, 0.2]
        positions.append(
            {
                "taxel_index": index,
                "row": row,
                "column": column,
                "status": "measured",
                "tool_position_m": centre,
                "safe_approach_position_m": [centre[0], centre[1], 0.25],
            }
        )
    return {
        "task": "taxel_position_mapping",
        "completed": False,
        "settings": {
            "sensor_rows": 5,
            "sensor_columns": 5,
            "anchor_taxel": 12,
            "skip_outer_border": True,
        },
        "initial_row_vector_xy_m": [0.0, 0.004],
        "initial_column_vector_xy_m": [0.004, 0.0],
        "row_vector_xy_m": [0.0, 0.004],
        "column_vector_xy_m": [0.004, 0.0],
        "positions": positions,
    }


def test_partial_map_can_be_relocated_and_keeps_its_target_set(tmp_path):
    api = _RobotApi()
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json", output_directory=None
    )
    host = _Host(api, task=task)
    sensor = host.ui_ros.sensor_functions
    sensor.n_row = 5
    sensor.n_col = 5
    sensor._data.diffDataAve = np.zeros((5, 5), dtype=float)
    source = _partial_relocation_source()
    task.skip_outer_border = False

    valid, message = task.validate_relocation_result(source)
    assert valid, message
    assert "continues mapping its remaining taxels" in message
    assert task.capture_anchor_pose(host)
    accepted, message = task.request_relocation(source, tmp_path / "old.json")
    assert accepted, message
    assert task.on_start(host)

    assert task._relocation_active and task._relocation_continues_mapping
    assert task.skip_outer_border is True
    task.on_stop(host, show_result=False)


def test_finished_partial_relocation_continues_from_remeasured_taxels(tmp_path):
    api = _RobotApi()
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json", output_directory=None
    )
    host = _Host(api, task=task)
    sensor = host.ui_ros.sensor_functions
    sensor.n_row = 5
    sensor.n_col = 5
    sensor._data.diffDataAve = np.zeros((5, 5), dtype=float)
    source = _partial_relocation_source()
    api.tool_position = [0.008, 0.008, 0.25]
    assert task.capture_anchor_pose(host)
    assert (task.sensor_rows, task.anchor_taxel) == (5, 12)
    task.skip_outer_border = True
    task._active = True
    task._mapping_run_started = True
    task._reference_onset_z = 0.2
    task._probe_start_z = 0.21
    task._relocation_active = True
    task._relocation_continues_mapping = True
    task._relocation_source_result = source
    task._relocation_source_positions = {
        int(record["taxel_index"]): record for record in source["positions"]
    }
    for record in task._relocation_source_positions.values():
        record.update(probe_start_z_m=0.21, proximity_onset_z_m=0.20, centre_scan_z_m=0.19)
    task._relocation_reference_indices = [7, 11, 12, 13, 17]
    task._positions = {
        index: dict(task._relocation_source_positions[index], method="cross_scan_verified")
        for index in task._relocation_reference_indices
    }
    fitted = {
        "rotation": np.eye(2),
        "translation": np.zeros(2),
        "z_shift_m": 0.03,
        "angle_deg": 0.0,
        "rms_residual_mm": 0.0,
        "max_residual_mm": 0.0,
    }

    task._finish_relocation(host, fitted)

    assert task._relocation_finished
    assert task._positions[6]["relocation_verified"] is False
    assert task._positions[6]["probe_start_z_m"] == pytest.approx(0.24)
    assert task._positions[6]["proximity_onset_z_m"] == pytest.approx(0.23)
    assert task._positions[6]["centre_scan_z_m"] == pytest.approx(0.22)
    assert task._positions[6]["tool_position_m"][2] == pytest.approx(0.23)
    assert task._positions[12]["relocation_verified"] is True
    assert task._positions[12]["probe_start_z_m"] == pytest.approx(0.21)
    assert any("Continuing the partial map" in message for message in host.messages)
    assert task._current_context == "regular"
    assert task._current_taxel in {8, 16, 18}
    assert task._move_action != "complete_anchor_arrived"
    # Transformed Taxel 6 is kept but never seeds motion.
    assert 6 not in task.plan_frontier_target()["verified_neighbours"]


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    (
        ("probe_speed_m_s", 0.0051, "no more than 5 mm/s"),
        ("reference_probe_speed_m_s", 0.0051, "no more than 5 mm/s"),
        ("max_probe_descent_m", 0.1001, "no more than 100 mm"),
        ("reference_max_descent_m", 0.1001, "no more than 100 mm"),
    ),
)
def test_taxel_mapping_rejects_probe_settings_above_ui_limits(
    tmp_path, setting, value, message
):
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json", output_directory=None
    )
    host = _Host(_RobotApi(), task=task)
    assert task.capture_anchor_pose(host)
    setattr(task, setting, value)

    assert not task.on_start(host)
    assert message in host.messages[-1]


def _tick(task, host, count=1):
    for _ in range(count):
        if (
            isinstance(task, TaxelPositionMappingTask)
            and task._phase == "sensor_calibrating"
        ):
            sensor = host.ui_ros.sensor_functions
            sensor._sensor_calibration_in_progress = False
            sensor.is_connected = True
        host.ui_ros.sensor_functions._data.frame_sequence += 1
        task.on_tick(host)


def _tick_until(task, host, predicate, limit=200):
    for _ in range(limit):
        if predicate():
            return True
        _tick(task, host)
    return predicate()


def _started_mapping(tmp_path, probe_motion="stepwise"):
    api = _RobotApi()
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json", output_directory=None
    )
    task.probe_motion = probe_motion
    task.move_settle_sec = 0.0
    task.probe_sample_frames = 3
    host = _Host(api, task=task)
    assert task.capture_anchor_pose(host)
    assert task.on_start(host)
    return api, task, host


def _mapping_mid_run(tmp_path, taxel, context, attempt):
    api = _RobotApi()
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json", output_directory=None
    )
    host = _Host(api, task=task)
    assert task.capture_anchor_pose(host)
    task._active = True
    task._mapping_run_started = True
    task._probe_start_z = 0.292
    task._current_taxel = taxel
    task._current_context = context
    task._attempt_counts[taxel] = attempt
    task._initial_row_vector_xy = np.asarray([0.004, 0.0])
    task._initial_column_vector_xy = np.asarray([0.0, 0.004])
    task._probe_baseline = np.zeros(80, dtype=float)
    task._far_baseline = np.zeros(80, dtype=float)
    return api, task, host


def _cross_samples(task, row_zero_m, column_zero_m, target, target_value=300.0):
    """Line-scan frames whose neighbour responses balance at the given offsets."""
    samples = []
    for axis in task._local_axes:
        zero = row_zero_m if axis["name"] == "row" else column_zero_m
        for offset in task.axis_offsets(axis["radius"], task.local_step_m):
            frame = np.zeros(80, dtype=float)
            frame[target] = target_value
            frame[axis["plus"]] = 200.0 + 20000.0 * (offset - zero)
            frame[axis["minus"]] = 200.0 - 20000.0 * (offset - zero)
            position = np.array(task._local_center, copy=True)
            position[:2] += offset * axis["unit"]
            samples.append(
                {"position": position, "frame": frame, "axis": axis["name"]}
            )
    return samples


def _prepare_local_scan(task, taxel, centre):
    task._local_center = np.asarray(centre, dtype=float)
    task._local_axes = task._local_axes_for(taxel)
    task._scan_kind = "local"
    task._scan_target = taxel
    task._scan_baseline = np.zeros(80, dtype=float)


def test_taxel_mapping_remembers_only_a_finite_tcp_anchor(tmp_path):
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json", output_directory=None
    )
    api = _RobotApi()
    host = _Host(api)

    assert task.capture_anchor_pose(host)
    assert task.storage_path.exists()
    assert task.anchor_pose["taxel_index"] == 35
    np.testing.assert_allclose(
        task.anchor_pose["tool_position_m"], [0.1, 0.2, 0.3]
    )

    restored = TaxelPositionMappingTask(
        storage_path=task.storage_path, output_directory=None
    )
    np.testing.assert_allclose(
        restored.anchor_pose["tool_quaternion_wxyz"], [1.0, 0.0, 0.0, 0.0]
    )

    api.tool_position = [np.nan, 0.2, 0.3]
    previous = dict(task.anchor_pose)
    assert not task.capture_anchor_pose(host)
    assert task.anchor_pose == previous


def test_taxel_mapping_reads_sensor_in_viewer_column_major_order():
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    host = _Host(_RobotApi())
    matrix = np.arange(80, dtype=float).reshape(8, 10)
    host.ui_ros.sensor_functions._data.diffDataAve = matrix

    sequence, flattened = task._read_sensor_frame(host)

    assert sequence == 1
    assert flattened[0] == matrix[0, 0]
    assert flattened[7] == matrix[7, 0]
    assert flattened[8] == matrix[0, 1]
    assert flattened[35] == matrix[3, 4]


def test_taxel_mapping_grid_is_serpentine_and_contains_centre():
    offsets = TaxelPositionMappingTask.grid_offsets(0.001, 0.001)

    assert (0.0, 0.0) in offsets
    assert offsets[:3] == [
        (-0.001, -0.001),
        (0.0, -0.001),
        (0.001, -0.001),
    ]
    assert offsets[3:6] == [
        (0.001, 0.0),
        (0.0, 0.0),
        (-0.001, 0.0),
    ]


def _fast_mapping_search():
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    api = _RobotApi()
    host = _Host(api, task)
    sensor = host.ui_ros.sensor_functions
    sensor.n_row = sensor.n_col = 20
    sensor._data.diffDataAve = np.zeros((20, 20))
    task.use_connected_sensor(host)
    task.anchor_pose = {
        "taxel_index": 189,
        "tool_position_m": [.1, .2, .3],
        "tool_quaternion_wxyz": [1., 0., 0., 0.],
    }
    task._active = True
    task._current_taxel = 189
    task._current_context = "anchor"
    task._far_baseline = np.zeros(400)
    task._probe_onset_pose = np.array([.1, .2, .29])
    task._probe_scan_pose = np.array([.1, .2, .285])
    task._start_coarse_scan(host, task._far_baseline, task._probe_scan_pose)
    return api, task, host


def test_fast_mapping_keeps_long_onset_baseline_and_verification_windows():
    _api, task, _host = _fast_mapping_search()
    assert task.scan_frame_count == 9
    assert task.probe_sample_frames == 20
    assert task.verify_frame_count == 20
    assert task.baseline_frame_count == 100
    assert task.force_guard_g == 20
    assert task.line_speed_m_s == .02
    assert task.local_step_m == .0005


def test_fast_twenty_by_twenty_search_reduces_441_positions_to_121():
    _api, task, host = _fast_mapping_search()
    assert len(task.grid_offsets(task.coarse_radius_m, task.coarse_step_m)) == 441
    assert task._coarse_sparse_pass
    assert len(task._scan_points) == 121
    assert len({tuple(point) for point in task._scan_points}) == 121
    task._dispatch_move_action(host, "scan_point_arrived")
    assert task._collection_count == 9
    assert "1/121" in host.taxel_mapping_progress[-1]["detail"]


def test_sparse_search_falls_back_before_retracting_for_missing_references():
    _api, task, host = _fast_mapping_search()
    task._scan_results = []
    task._finish_coarse_scan(host)
    assert not task._coarse_sparse_pass
    assert len(task._scan_points) == 441
    assert task._pending_recovery is None
    assert not host.stops
    task._dispatch_move_action(host, "scan_point_arrived")
    assert task._collection_count == 20
    assert "1/441" in host.taxel_mapping_progress[-1]["detail"]
    # A second failure uses the existing bounded retract/retry path.
    task._finish_coarse_scan(host)
    assert task._pending_recovery is not None


def test_sparse_anchor_centre_failure_uses_dense_grid_only_once():
    _api, task, host = _fast_mapping_search()
    task._handle_centre_scan_failure(host, "unclear centre")
    assert not task._coarse_sparse_pass
    assert task._pending_recovery is None
    assert len(task._scan_points) == 441
    task._handle_centre_scan_failure(host, "still unclear")
    assert task._pending_recovery is not None


def test_sparse_search_does_not_rescan_anchor_when_a_later_taxel_fails():
    _api, task, host = _fast_mapping_search()
    task._current_taxel = 190
    task._current_context = "neighbor"
    task._handle_centre_scan_failure(host, "weak target")
    assert task._pending_recovery["taxel_index"] == 190


def test_sparse_search_rejects_a_clipped_reference_peak():
    _api, task, host = _fast_mapping_search()
    centres = {
        189: [.1, .2, .285], 188: [.09, .2, .285],
        190: [.12, .2, .285], 169: [.1, .19, .285],
        209: [.1, .21, .285],
    }
    for index, position in centres.items():
        frame = np.zeros(400)
        frame[index] = -100
        task._scan_results.append({"position": np.array(position), "frame": frame})
    task._finish_coarse_scan(host)
    assert not task._coarse_sparse_pass
    assert len(task._scan_points) == 441
    assert any("boundary" in message for message in host.messages)


def test_short_line_scan_route_preserves_measurements_and_shortens_travel():
    lines = [
        [(np.array([x, 0., .3]), {"axis": "row", "offset": x}) for x in (-.0025, 0., .0025)],
        [(np.array([0., y, .3]), {"axis": "column", "offset": y}) for y in (-.0025, 0., .0025)],
    ]
    start = np.array([.004, .001, .3])
    old = lines[0] + lines[1]
    new = TaxelPositionMappingTask.shortest_line_scan_route(lines, start)
    def length(route):
        return np.linalg.norm(np.diff([start] + [item[0] for item in route], axis=0), axis=1).sum()
    assert length(new) < length(old)
    assert {(tuple(point), tuple(meta.items())) for point, meta in new} == {
        (tuple(point), tuple(meta.items())) for point, meta in old
    }
    assert len(new) == len(old)


def test_short_xy_window_uses_median_and_ignores_single_large_spike():
    _api, task, host = _fast_mapping_search()
    collected = []
    task._handle_collection = lambda host, action, frame, *args: collected.append(frame)
    task._begin_collection(host, "scan_sample", 9, "test")
    for sequence in range(9):
        task._tick_collection(host, (sequence, np.full(400, 10000. if sequence == 4 else -12.)), 0.)
    assert len(collected) == 1
    np.testing.assert_allclose(collected[0], -12.)


def test_fast_mapping_settings_round_trip_preserves_user_frame_count():
    _api, task, _host = _fast_mapping_search()
    task.scan_frame_count = 31
    task.adaptive_coarse_search = False
    settings = task._settings()
    restored = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    restored._apply_resume_settings({"settings": settings, "anchor_pose": task.anchor_pose})
    assert restored.scan_frame_count == 31
    assert restored.adaptive_coarse_search is False
    assert restored.anchor_taxel == 189
    assert settings["scan_sample_reducer"] == "median"


def test_taxel_mapping_refines_target_specific_xy_from_synthetic_scan():
    baseline = np.zeros(80, dtype=float)
    samples = []
    for x_m, signal in [(-0.001, 20.0), (0.0, 100.0), (0.001, 30.0)]:
        frame = np.zeros(80, dtype=float)
        frame[35] = signal
        frame[34] = 10.0
        samples.append(
            {"position": np.asarray([x_m, 0.002, 0.3]), "frame": frame}
        )

    refined = TaxelPositionMappingTask.refine_xy_from_samples(
        samples, baseline, 35
    )

    np.testing.assert_allclose(refined["position"], [0.0, 0.002, 0.3])
    assert refined["signal_counts"] == 100.0
    assert refined["dominance_ratio"] == 10.0


def test_taxel_mapping_fits_column_major_lattice_and_predicts_all_taxels():
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    anchor = np.asarray([0.100, 0.200, 0.300])
    row_vector = np.asarray([0.004, 0.001])
    column_vector = np.asarray([-0.001, 0.006])

    def record(index, offset):
        point = anchor.copy()
        point[:2] += np.asarray(offset)
        task._positions[index] = {"tool_position_m": point.tolist()}

    record(35, [0.0, 0.0])
    record(34, -row_vector)
    record(36, row_vector)
    record(27, -column_vector)
    record(43, column_vector)

    predicted, fitted_row, fitted_column, separation = (
        task.predict_lattice_positions()
    )

    assert len(predicted) == 80
    np.testing.assert_allclose(fitted_row, row_vector)
    np.testing.assert_allclose(fitted_column, column_vector)
    np.testing.assert_allclose(predicted[35], anchor)
    np.testing.assert_allclose(
        predicted[TaxelPositionMappingTask.taxel_index(4, 5)][:2],
        anchor[:2] + row_vector + column_vector,
    )
    assert separation > 0.9


def test_taxel_mapping_origin_uses_all_five_centres_and_reports_residuals():
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    anchor = np.asarray([0.100, 0.200, 0.290])
    row_vector = np.asarray([0.004, 0.0])
    column_vector = np.asarray([0.0, 0.005])

    def record(index, offset, error=(0.0, 0.0)):
        point = anchor.copy()
        point[:2] += np.asarray(offset) + np.asarray(error)
        task._positions[index] = {
            "status": "measured",
            "tool_position_m": point.tolist(),
        }

    record(35, [0.0, 0.0], error=(0.0005, 0.0))
    record(34, -row_vector)
    record(36, row_vector)
    record(27, -column_vector)
    record(43, column_vector)

    predicted, _row, _column, _score = task.predict_lattice_positions()
    np.testing.assert_allclose(predicted[35][:2], anchor[:2] + [0.0001, 0.0])

    quality = task.lattice_fit_quality()
    assert quality["max_residual_mm"] == pytest.approx(0.4)
    assert quality["rms_residual_mm"] == pytest.approx(0.2)
    assert quality["residual_mm_by_taxel"]["35"] == pytest.approx(0.4)


def test_taxel_mapping_requires_anchor_sensor_robot_and_fresh_force(tmp_path):
    api = _RobotApi()
    host = _Host(api)
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json", output_directory=None
    )

    assert not task.on_start(host)
    assert "Remember" in host.messages[-1]

    assert task.capture_anchor_pose(host)
    host.ui_ros._force_meter_thread = None
    host.ui_ros._force_meter_last_sample_monotonic = None
    assert task.on_start(host)
    assert task._phase == "connecting_force_meter"
    assert host.ui_ros.force_meter_refresh_calls == 1
    assert host.ui_ros.force_meter_start_calls == 1
    assert not api.sent

    host.ui_ros._force_meter_last_raw_newtons = 0.0
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    assert task._phase == "moving"
    assert api.sent

    sensor_task = TaxelPositionMappingTask(
        storage_path=tmp_path / "other-anchor.json", output_directory=None
    )
    sensor_host = _Host(_RobotApi())
    assert sensor_task.capture_anchor_pose(sensor_host)
    sensor_host.ui_ros.sensor_functions.is_connected = False
    assert not sensor_task.on_start(sensor_host)
    assert "8x10 tactile sensor" in sensor_host.messages[-1]


def test_taxel_mapping_restarts_a_running_but_stale_force_reader(tmp_path):
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json", output_directory=None
    )
    host = _Host(_RobotApi(), task=task)
    assert task.capture_anchor_pose(host)
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic() - 5.0

    assert task.on_start(host)

    assert task._phase == "connecting_force_meter"
    assert host.ui_ros.force_meter_stop_calls == 1
    assert host.ui_ros.force_meter_refresh_calls == 1
    assert host.ui_ros.force_meter_start_calls == 1


def test_taxel_mapping_runs_full_sensor_calibration_at_safe_anchor_before_baseline(
    tmp_path,
):
    api, task, host = _started_mapping(tmp_path)

    assert _tick_until(
        task,
        host,
        lambda: task._phase == "sensor_calibrating",
    )
    sensor = host.ui_ros.sensor_functions
    assert sensor.update_cal_calls == 1
    assert sensor._sensor_calibration_in_progress
    assert not api.velocity_commands
    assert task._collection_action == ""

    # The sensor is intentionally disconnected while its background full
    # calibration runs; that must not be mistaken for a sensor-loss failure.
    task.on_tick(host)
    assert task._active
    assert not host.stops

    sensor._sensor_calibration_in_progress = False
    sensor.is_connected = True
    sensor._data.frame_sequence = 1
    task.on_tick(host)

    assert task._phase == "collecting"
    assert task._collection_action == "reference_baseline"
    assert task._sensor_calibration_history[-1]["status"] == "complete"
    assert task._sensor_calibration_history[-1]["context"] == "fresh_start"
    task.on_stop(host, show_result=False)


def test_taxel_mapping_stops_if_full_sensor_calibration_fails(tmp_path):
    api, task, host = _started_mapping(tmp_path)
    assert _tick_until(
        task,
        host,
        lambda: task._phase == "sensor_calibrating",
    )
    sensor = host.ui_ros.sensor_functions
    sensor._sensor_calibration_in_progress = False
    sensor.is_connected = False
    task.on_tick(host)

    assert host.stops
    assert task.last_result["stop_reason"] == "sensor_calibration_failed"
    assert task.last_result["sensor_calibration_history"][-1]["status"] == "failed"
    assert not api.velocity_commands


def test_taxel_mapping_first_probe_learns_height_then_probes_from_above_it(
    tmp_path,
):
    # This test places the TCP itself and checks streamed descent speeds.
    api, task, host = _started_mapping(tmp_path, probe_motion="velocity")
    data = host.ui_ros.sensor_functions._data
    assert task._active
    assert host.taxel_mapping_running_states[-1] is True
    assert api.sent[-1][0] == [0.1, 0.2, 0.3]
    assert api.sent[-1][2]["motion_type"] == TaxelPositionMappingTask.LINE_MOTION
    assert api.sent[-1][2]["velocity"] == pytest.approx(task.line_speed_m_s)

    assert _tick_until(task, host, lambda: task._phase == "probing")
    assert task._probe_kind == "reference"
    assert task._current_context == "reference"
    assert task._baseline_checks[-1]["stable"] is True
    _tick(task, host)
    assert api.velocity_commands[-1][0][2] == pytest.approx(
        -task.reference_probe_speed_m_s
    )

    # The anchor may be high: the first probe descends 10 mm before Taxel 35
    # responds. Later probes get a local onset-independent height boundary.
    api.tool_position = [0.1, 0.2, 0.29]
    data.diffDataAve[3, 4] = -100.0
    _tick(task, host, 2 + task.probe_sample_frames)
    assert task._probe_hit_count == 1
    _tick(task, host)
    assert api.velocity_commands[-1][0][2] == pytest.approx(
        -task.probe_confirmation_speed_m_s
    )
    # Confirmation occurs at the next stationary level, but the learned
    # onset keeps the first credible crossing pose.
    api.tool_position = [0.1, 0.2, 0.2895]
    _tick(task, host, 2 + task.probe_sample_frames)
    assert task._reference_onset_z == pytest.approx(0.29)
    assert task._probe_start_z == pytest.approx(0.292)
    assert task._current_context == "anchor"
    assert task._attempt_counts[35] == 1
    # Back up to the anchor height for a far-field baseline, then descend.
    assert api.sent[-1][0] == pytest.approx([0.1, 0.2, 0.3])
    assert _tick_until(
        task, host, lambda: task._move_action == "target_start_arrived"
    )
    assert api.sent[-1][0] == pytest.approx([0.1, 0.2, 0.292])

    assert _tick_until(task, host, lambda: task._phase == "probing")
    assert task._probe_kind == "working"
    _tick(task, host)
    assert api.velocity_commands[-1][0][2] == pytest.approx(
        -min(task.probe_speed_m_s, task.probe_confirmation_speed_m_s)
    )
    assert task._probe_max_descent_used_m == pytest.approx(0.007)

    # This fixture also supplied -100 at clearance; require a sustained
    # 120-unit drop relative to that far-field reference for the 2x target.
    data.diffDataAve[3, 4] = -220.0
    api.tool_position = [0.1, 0.2, 0.2915]
    _tick(task, host, 2 + task.probe_sample_frames)
    _tick(task, host)
    api.tool_position = [0.1, 0.2, 0.2910]
    _tick(task, host, 2 + task.probe_sample_frames)
    assert task._scan_kind == "coarse"
    assert task._phase == "moving"
    task.on_stop(host, show_result=False)
    assert host.taxel_mapping_running_states[-1] is False


def test_taxel_mapping_probe_ignores_a_single_frame_spike(tmp_path):
    api, task, host = _started_mapping(tmp_path)
    data = host.ui_ros.sensor_functions._data
    assert _tick_until(task, host, lambda: task._phase == "probing")

    api.tool_position = [0.1, 0.2, 0.2995]
    _tick(task, host, 2)
    data.diffDataAve[3, 4] = -100.0
    _tick(task, host)
    data.diffDataAve[3, 4] = 0.0
    _tick(task, host, task.probe_sample_frames - 1)

    assert task._phase == "probing"
    assert task._reference_onset_z is None

    # Two independently sampled Z levels must sustain the downward change.
    api.tool_position = [0.1, 0.2, 0.2990]
    data.diffDataAve[3, 4] = -100.0
    _tick(task, host, 2 + task.probe_sample_frames)
    api.tool_position = [0.1, 0.2, 0.2985]
    _tick(task, host, 2 + task.probe_sample_frames)
    assert task._reference_onset_z is not None


def test_taxel_mapping_remembers_onset_then_descends_to_stronger_scan_level(
    tmp_path,
):
    api = _RobotApi()
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json", output_directory=None
    )
    host = _Host(api, task=task)
    task.probe_motion = "velocity"  # The test places the TCP at each level itself.
    task._active = True
    task.move_settle_sec = 0.0
    task.proximity_detection_mode = "fixed"
    task.signal_threshold_counts = 50.0
    task.probe_sample_frames = 3
    task.probe_confirm_frames = 2
    task.centre_scan_signal_multiplier = 2.0
    task._current_taxel = 35
    task._current_context = "regular"
    task.anchor_pose = {
        "tool_position_m": [0.1, 0.2, 0.3],
        "tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    task._initial_row_vector_xy = np.asarray([0.004, 0.0])
    task._initial_column_vector_xy = np.asarray([0.0, 0.005])
    task._far_baseline = np.zeros(80)
    baseline = np.zeros(80)
    task._start_probe(
        host,
        baseline,
        0.0,
        kind="working",
        baseline_min=baseline,
        baseline_max=baseline,
        baseline_frames=np.zeros((20, 80)),
    )

    sequence = 1

    def sample_level(z_m, sensor_value):
        nonlocal sequence
        api.tool_position = [0.1, 0.2, z_m]
        values = np.zeros(80)
        values[35] = sensor_value
        sequence += 1
        task._tick_probe(host, (sequence, values))  # stop at the Z level
        sequence += 1
        task._tick_probe(host, (sequence, values))  # enter sampling
        for _ in range(task.probe_sample_frames):
            sequence += 1
            task._tick_probe(host, (sequence, values))

    sample_level(0.2995, -60.0)
    sample_level(0.2990, -70.0)

    assert task._probe_onset_confirmed
    assert task._probe_onset_pose[2] == pytest.approx(0.2995)
    assert task._phase == "probing"
    assert task._scan_kind == ""

    sample_level(0.2985, -80.0)
    assert task._phase == "probing"
    sample_level(0.2980, -110.0)

    assert task._probe_onset_pose[2] == pytest.approx(0.2995)
    assert task._probe_scan_pose[2] == pytest.approx(0.2980)
    assert task._probe_scan_signal == pytest.approx(110.0)
    assert task._probe_scan_target_signal == pytest.approx(100.0)
    assert task._probe_scan_target_reached is True
    assert task._scan_kind == "local"
    assert task._phase == "moving"


def test_taxel_mapping_rejects_weak_scan_at_post_onset_safety_limit(
    tmp_path,
):
    api = _RobotApi()
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json", output_directory=None
    )
    host = _Host(api, task=task)
    task._active = True
    task.move_settle_sec = 0.0
    task.proximity_detection_mode = "fixed"
    task.signal_threshold_counts = 50.0
    task.probe_sample_frames = 3
    task.probe_confirm_frames = 2
    task.centre_scan_signal_multiplier = 2.0
    task.centre_scan_max_extra_descent_m = 0.001
    task._reference_onset_z = 0.2995
    task._current_taxel = 35
    task._current_context = "regular"
    task.anchor_pose = {
        "tool_position_m": [0.1, 0.2, 0.3],
        "tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    task._initial_row_vector_xy = np.asarray([0.004, 0.0])
    task._initial_column_vector_xy = np.asarray([0.0, 0.005])
    task._far_baseline = np.zeros(80)
    baseline = np.zeros(80)
    task._start_probe(
        host,
        baseline,
        0.0,
        kind="working",
        baseline_min=baseline,
        baseline_max=baseline,
        baseline_frames=np.zeros((20, 80)),
    )

    sequence = 1

    def sample_level(z_m, sensor_value):
        nonlocal sequence
        api.tool_position = [0.1, 0.2, z_m]
        values = np.zeros(80)
        values[35] = sensor_value
        sequence += 1
        task._tick_probe(host, (sequence, values))
        sequence += 1
        task._tick_probe(host, (sequence, values))
        for _ in range(task.probe_sample_frames):
            sequence += 1
            task._tick_probe(host, (sequence, values))

    sample_level(0.2995, -60.0)
    sample_level(0.2990, -70.0)
    sample_level(0.2985, -80.0)

    assert task._probe_onset_pose[2] == pytest.approx(0.2995)
    # The final line target now leaves 0.2 mm arrival headroom above the
    # unchanged 0.2985 m boundary; weak signals must still fail closed.
    assert task._probe_scan_pose[2] == pytest.approx(0.2987)
    assert task._probe_scan_signal == pytest.approx(80.0)
    assert task._probe_scan_target_signal == pytest.approx(100.0)
    assert task._probe_scan_target_reached is False
    assert task._scan_kind == ""
    assert task._pending_recovery is not None
    assert "refusing a weak-signal XY scan" in task._pending_recovery["reason"]


def test_taxel_mapping_waits_until_robot_stops_moving(tmp_path):
    api, task, host = _started_mapping(tmp_path)

    # Every reading is inside the 0.1 mm target tolerance, but the robot is
    # still settling between readings.
    for z_m in (0.30008, 0.29995, 0.30008):
        api.tool_position = [0.1, 0.2, z_m]
        _tick(task, host)
    assert task._phase == "moving"

    _tick(task, host, 2)
    assert task._phase == "sensor_calibrating"
    _tick(task, host)
    assert task._phase == "collecting"


def test_taxel_mapping_extends_long_move_timeout_at_low_project_speed(tmp_path):
    api, task, host = _started_mapping(tmp_path)
    api.project_speed_percent = 5.0
    api.tool_position = [0.1, 0.2, 0.254085]
    task._active = True

    assert task._command_move(
        host,
        [0.1, 0.2, 0.3],
        "test_arrived",
        "testing a slow safe-height retract",
    )

    # This is the real failed geometry: about 45.9 mm at an effective 1 mm/s.
    assert task._move_distance_m == pytest.approx(0.045915)
    assert task._move_expected_duration_sec == pytest.approx(45.915)
    assert task._move_timeout_current_sec == pytest.approx(78.8725)

    # The old fixed 45-second limit must no longer stop a healthy slow move.
    api.tool_position = [0.1, 0.2, 0.299]
    task._tick_move(host, task._move_started_at + 45.1)
    assert task._active
    assert not host.stops

    # A genuinely stuck move still stops once the calculated allowance ends.
    task._tick_move(
        host,
        task._move_started_at + task._move_timeout_current_sec + 0.1,
    )
    assert host.stops
    assert task.last_result["stop_reason"] == "move_timeout"


def test_taxel_mapping_route_travels_laterally_only_at_anchor_height(tmp_path):
    api, task, host = _mapping_mid_run(tmp_path, 36, "neighbor", 0)
    api.tool_position = [0.11, 0.21, 0.29]

    task._route_to(host, [0.12, 0.2], 0.292, "target_start_arrived", "test")
    assert api.sent[-1][0] == pytest.approx([0.11, 0.21, 0.3])
    task._dispatch_move_action(host, "route_waypoint")
    assert api.sent[-1][0] == pytest.approx([0.12, 0.2, 0.3])
    task._dispatch_move_action(host, "route_waypoint")
    assert api.sent[-1][0] == pytest.approx([0.12, 0.2, 0.292])
    assert task._move_action == "target_start_arrived"
    assert all(
        entry[2]["motion_type"] == TaxelPositionMappingTask.LINE_MOTION
        for entry in api.sent[-3:]
    )


def test_taxel_mapping_axis_centre_uses_neighbour_balance_or_edge_peak():
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    baseline = np.zeros(80, dtype=float)
    centre = np.asarray([0.1, 0.2, 0.29])
    axis = {
        "name": "row",
        "unit": np.asarray([1.0, 0.0]),
        "radius": 0.0025,
        "minus": 34,
        "plus": 36,
    }
    true_offset = 0.0004
    samples = []
    for offset in np.linspace(-0.0025, 0.0025, 11):
        frame = np.zeros(80, dtype=float)
        frame[35] = 100.0 - 4.0e6 * (offset - true_offset) ** 2
        frame[36] = 200.0 + 20000.0 * (offset - true_offset)
        frame[34] = 200.0 - 20000.0 * (offset - true_offset)
        samples.append({"position": centre + [offset, 0.0, 0.0], "frame": frame})

    balanced = task.estimate_axis_centre(samples, baseline, centre, axis, 35)
    assert balanced["method"] == "neighbour_balance"
    assert balanced["offset_m"] == pytest.approx(true_offset, abs=1e-7)
    assert balanced["peak_offset_m"] == pytest.approx(true_offset, abs=1e-7)

    edge = task.estimate_axis_centre(
        samples, baseline, centre, dict(axis, minus=None), 35
    )
    assert edge["method"] == "peak_quadratic"
    assert edge["offset_m"] == pytest.approx(true_offset, abs=1e-7)


@pytest.mark.parametrize(
    "missing, neighbour, expected_mode",
    [("plus", 34, "edge_balance_minus"), ("minus", 36, "edge_balance_plus")],
)
def test_taxel_mapping_edge_taxel_balances_against_its_only_neighbour(
    missing, neighbour, expected_mode
):
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    pitch = 0.004
    true_offset = 0.0003
    sigma = 0.005
    axis = {
        "name": "row",
        "unit": np.asarray([1.0, 0.0]),
        "pitch": pitch,
        "minus": 34 if missing == "plus" else None,
        "plus": 36 if missing == "minus" else None,
    }
    mode, window_centre = task.axis_balance_mode(axis)
    assert mode == expected_mode
    assert window_centre == pytest.approx(
        -0.5 * pitch if missing == "plus" else 0.5 * pitch
    )
    neighbour_offset = -pitch if missing == "plus" else pitch
    centre = np.asarray([0.1, 0.2, 0.29])
    samples = []
    for offset in window_centre + np.linspace(-0.0016, 0.0016, 9):
        frame = np.zeros(80, dtype=float)
        frame[35] = 100.0 * np.exp(-((offset - true_offset) ** 2) / (2 * sigma**2))
        frame[neighbour] = 100.0 * np.exp(
            -((offset - true_offset - neighbour_offset) ** 2) / (2 * sigma**2)
        )
        samples.append({"position": centre + [offset, 0.0, 0.0], "frame": frame})

    estimate = task.estimate_axis_centre(
        samples, np.zeros(80), centre, axis, 35
    )
    assert estimate["method"] == expected_mode
    assert estimate["offset_m"] == pytest.approx(true_offset, abs=1e-5)
    assert abs(estimate["bracket_m"]) < 0.0016


def test_taxel_mapping_recentres_when_the_target_only_leads_at_one_window_end(
    tmp_path,
):
    """A neighbour wins everywhere except one end: the centre is beyond it."""
    api, task, host = _mapping_mid_run(tmp_path, 36, "neighbor", 1)
    task._probe_detection_by_taxel[36] = {"effective_threshold_counts": 8.0}
    centre = [0.104, 0.2, 0.29]
    api.tool_position = list(centre)
    _prepare_local_scan(task, 36, centre)
    row_radius = next(axis["radius"] for axis in task._local_axes if axis["name"] == "row")
    samples = []
    for axis in task._local_axes:
        for offset in task.axis_offsets(axis["radius"], task.local_step_m):
            frame = np.zeros(80, dtype=float)
            frame[36] = 21.0
            if axis["name"] == "row":
                stronger = 24.0 + 1500.0 * offset
                frame[axis["plus"]] = stronger
                frame[axis["minus"]] = stronger
            else:
                frame[axis["plus"]] = 5.0 + 2000.0 * offset
                frame[axis["minus"]] = 5.0 - 2000.0 * offset
            position = np.array(task._local_center, copy=True)
            position[:2] += offset * axis["unit"]
            samples.append({"position": position, "frame": frame, "axis": axis["name"]})
    task._scan_results = samples

    task._finish_local_scan(host)

    assert task._local_recenters == 1
    assert "re-centring" in host.messages[-1]
    np.testing.assert_allclose(
        task._local_center, [0.104 - row_radius, 0.2, 0.29], atol=1e-6
    )


def test_taxel_mapping_recentres_once_then_retries_from_the_anchor(tmp_path):
    api, task, host = _mapping_mid_run(tmp_path, 36, "neighbor", 1)
    task._current_target_xy = np.asarray([0.104, 0.2])
    centre = [0.104, 0.2, 0.29]
    api.tool_position = list(centre)
    _prepare_local_scan(task, 36, centre)
    task._scan_results = _cross_samples(task, 0.004, 0.0, 36)

    task._finish_local_scan(host)
    assert task._local_recenters == 1
    np.testing.assert_allclose(task._local_center, [0.108, 0.2, 0.29])
    assert task._pending_recovery is None

    task._scan_kind = "local"
    task._scan_results = _cross_samples(task, 0.004, 0.0, 36)
    task._finish_local_scan(host)
    assert task._pending_recovery["retry"] is True
    assert "not bracketed" in task._pending_recovery["reason"]
    assert api.sent[-1][0][2] == pytest.approx(0.3)


def test_taxel_mapping_accepts_only_the_strongest_taxel_at_its_centre(tmp_path):
    api, task, host = _mapping_mid_run(tmp_path, 36, "neighbor", 1)
    task._current_target_xy = np.asarray([0.104, 0.2])
    task._neighbor_queue = [27]
    task._neighbor_guesses = {27: [0.1, 0.196]}
    centre = np.asarray([0.104, 0.2, 0.29])
    api.tool_position = centre.tolist()
    _prepare_local_scan(task, 36, centre)
    task._scan_results = _cross_samples(task, 0.0005, -0.0003, 36)

    task._finish_local_scan(host)
    refined = centre + [0.0005, -0.0003, 0.0]
    assert api.sent[-1][0] == pytest.approx(refined.tolist())
    assert task._move_action == "verify_arrived"

    weaker = np.zeros(80, dtype=float)
    weaker[36] = 80.0
    weaker[37] = 90.0
    task._finish_verify(host, weaker)
    assert task._pending_recovery["reason"].startswith("Taxel 37 was stronger")
    assert 36 not in task._positions

    task._pending_recovery = None
    api.tool_position = refined.tolist()
    task._verify_estimate = {
        "centre": refined,
        "axes": {"row": {"method": "neighbour_balance", "offset_m": 0.0005}},
    }
    stronger = np.zeros(80, dtype=float)
    stronger[36] = 110.0  # Clear dominance; borderline centres get a repeat scan.
    stronger[37] = 80.0
    task._finish_verify(host, stronger)
    assert task._positions[36]["method"] == "cross_scan_verified"
    assert task._positions[36]["row_axis_method"] == "neighbour_balance"
    np.testing.assert_allclose(task._positions[36]["tool_position_m"], refined)
    # The next neighbour is reached by lifting straight up first.
    assert api.sent[-1][0] == pytest.approx([refined[0], refined[1], 0.3])


class _SimRobotApi(_RobotApi):
    """Tool position follows streamed base-frame velocity between moves."""

    def __init__(self):
        super().__init__()
        self.velocity = np.zeros(3, dtype=float)

    def send_end_effector_velocity_in_frame(
        self, v_lin, v_rot, frame="tool", ensure_mode=True
    ):
        self.velocity = np.asarray(v_lin, dtype=float)
        return super().send_end_effector_velocity_in_frame(
            v_lin, v_rot, frame, ensure_mode
        )

    def exit_end_effector_velocity_mode(self, send_zero=True):
        self.velocity = np.zeros(3, dtype=float)
        return super().exit_end_effector_velocity_mode(send_zero)

    def send_positions_tool_position(self, positions, quaternion, **kwargs):
        self.velocity = np.zeros(3, dtype=float)
        return super().send_positions_tool_position(positions, quaternion, **kwargs)

    def step(self, dt):
        self.tool_position = (
            np.asarray(self.tool_position, dtype=float) + self.velocity * dt
        ).tolist()


_SIM_SURFACE_Z = 0.27


def _sim_taxel_xy(index):
    row, column = index % 8, index // 8
    taxel_35 = np.asarray([0.1005, 0.1996])
    row_vector = np.asarray([0.0040, 0.0003])
    column_vector = np.asarray([-0.0003, 0.0050])
    return taxel_35 + (row - 3) * row_vector + (column - 4) * column_vector


_SIM_TAXEL_XY = np.asarray([_sim_taxel_xy(index) for index in range(80)])


def _sim_frame(position):
    """Broad proximity response: neighbours see about 40% of the target."""
    position = np.asarray(position, dtype=float)
    height = max(float(position[2]) - _SIM_SURFACE_Z, 0.0)
    level = 400.0 / (1.0 + (height / 0.0025) ** 2)
    distance_sq = np.sum((_SIM_TAXEL_XY - position[:2]) ** 2, axis=1)
    flat = level * np.exp(-distance_sq / (2.0 * 0.003 ** 2))
    # Column-major taxel index -> (display row, column).
    return (-flat).reshape(10, 8).T.copy()


@pytest.mark.parametrize("angle_degrees", [0., 27.])
def test_fast_twenty_by_twenty_pilot_verifies_anchor_and_four_neighbours(angle_degrees):
    api = _SimRobotApi()
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    task.centre_scan_signal_multiplier = 2.  # Keep this geometry fixture's strength explicit.
    task.move_settle_sec = 0.
    host = _Host(api, task)
    sensor = host.ui_ros.sensor_functions
    sensor.n_row = sensor.n_col = 20
    theta = np.deg2rad(angle_degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    centres = np.array([
        np.array([.1005, .1996]) + rotation @ np.array([(index % 20 - 9) * .0111, (index // 20 - 9) * .011])
        for index in range(400)
    ])
    def frame_at_pose():
        position = np.asarray(api.tool_position)
        height = max(float(position[2]) - _SIM_SURFACE_Z, 0.)
        level = 400. / (1. + (height / .0025) ** 2)
        distance_sq = np.sum((centres - position[:2]) ** 2, axis=1)
        return (-level * np.exp(-distance_sq / (2. * .006 ** 2))).reshape(20, 20).T.copy()
    sensor._data.diffDataAve = frame_at_pose()
    assert task.capture_anchor_pose(host)
    assert task.on_start(host)
    saw_sparse = False
    for _ in range(60000):
        if not task._active or len(task._positions) >= 5:
            break
        api.step(.05)
        sensor._data.diffDataAve = frame_at_pose()
        host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
        _tick(task, host)
        saw_sparse |= task._scan_kind == "coarse" and task._coarse_sparse_pass
    assert task._active, host.messages[-5:]
    assert saw_sparse
    assert set(task._positions) == {189, 188, 190, 169, 209}, host.messages[-5:]
    assert not task._failures
    assert not any("denser check" in message for message in host.messages)
    for index, record in task._positions.items():
        error_mm = 1000 * np.linalg.norm(np.array(record["tool_position_m"][:2]) - centres[index])
        assert error_mm < .15, (index, error_mm)
    task.on_stop(host, show_result=False)
    assert task.last_result["mapped_count"] == 5
    assert task.last_result["settings"]["scan_frame_count"] == 9


def test_taxel_mapping_full_simulated_run_finds_every_true_centre(tmp_path):
    api = _SimRobotApi()
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json",
        output_directory=tmp_path / "maps",
    )
    task.move_settle_sec = 0.0
    # The synthetic field was tuned for the legacy 50-count onset; adaptive
    # envelope behavior is covered independently above.
    task.proximity_detection_mode = "fixed"
    task.centre_scan_signal_multiplier = 2.  # Keep this legacy Gaussian geometry fixture unchanged.
    task.reference_probe_speed_m_s = 0.002
    task.probe_speed_m_s = 0.0005
    task.coarse_radius_m = 0.006
    task.coarse_step_m = 0.0015
    host = _Host(api, task=task)
    data = host.ui_ros.sensor_functions._data
    data.diffDataAve = _sim_frame(api.tool_position)
    assert task.capture_anchor_pose(host)
    assert task.on_start(host)

    lowest_z = float(api.tool_position[2])
    for _ in range(300000):
        if not task._active:
            break
        api.step(0.05)
        lowest_z = min(lowest_z, float(api.tool_position[2]))
        data.diffDataAve = _sim_frame(api.tool_position)
        host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
        _tick(task, host)

    result = task.last_result
    assert result["completed"], (result["stop_reason"], host.messages[-5:])
    assert result["mapped_count"] == 80
    assert result["failed_count"] == 0
    assert result["settings"]["mapping_order"] == task.MAPPING_ORDER
    for record in result["positions"]:
        prediction = record.get("position_prediction")
        if prediction is not None:
            assert prediction["taxel_index"] == record["taxel_index"]
            for known in prediction["verified_neighbours"]:
                row, column = task._taxel_coords(record["taxel_index"])
                known_row, known_column = task._taxel_coords(known)
                assert abs(row-known_row)+abs(column-known_column) == 1
    interior_mm = []
    edge_mm = []
    for record in result["positions"]:
        error = 1000.0 * float(
            np.linalg.norm(
                np.asarray(record["tool_position_m"][:2])
                - _sim_taxel_xy(record["taxel_index"])
            )
        )
        on_edge = record["row"] in (0, 7) or record["column"] in (0, 9)
        (edge_mm if on_edge else interior_mm).append(error)
    assert max(interior_mm) < 0.05
    # An edge taxel balances against one neighbour, so the residual
    # proximity left in its far-field baseline shifts it slightly.
    assert max(edge_mm) < 0.1
    assert result["lattice_fit"]["rms_residual_mm"] < 0.05
    assert lowest_z > _SIM_SURFACE_Z + 0.003
    assert api.tool_position == pytest.approx([0.1, 0.2, 0.3])
    assert all(
        entry[2]["motion_type"] == TaxelPositionMappingTask.LINE_MOTION
        for entry in api.sent
    )
    assert Path(result["positions_csv_path"]).exists()
    assert host.taxel_mapping_results[-1] is result


def test_taxel_mapping_can_live_report_and_skip_the_complete_outer_border(
    tmp_path,
):
    api = _SimRobotApi()
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json",
        output_directory=tmp_path / "maps",
    )
    task.skip_outer_border = True
    task.move_settle_sec = 0.0
    # The synthetic field was tuned for the legacy 50-count onset.
    task.proximity_detection_mode = "fixed"
    task.centre_scan_signal_multiplier = 2.  # Legacy Gaussian fixture, not a default-strength test.
    task.reference_probe_speed_m_s = 0.002
    task.probe_speed_m_s = 0.0005
    task.coarse_radius_m = 0.006
    task.coarse_step_m = 0.0015
    host = _Host(api, task=task)
    data = host.ui_ros.sensor_functions._data
    data.diffDataAve = _sim_frame(api.tool_position)
    assert task.capture_anchor_pose(host)
    assert task.on_start(host)

    assert host.taxel_mapping_live_maps[-1] == {
        "measured": [],
        "skipped": task.skipped_taxel_indices(),
        "target_total": 48,
    }
    for _ in range(250000):
        if not task._active:
            break
        api.step(0.05)
        data.diffDataAve = _sim_frame(api.tool_position)
        host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
        _tick(task, host)

    result = task.last_result
    assert result["completed"], (result["stop_reason"], host.messages[-5:])
    assert result["mapped_count"] == 48
    assert result["measured_position_count"] == 48
    assert result["target_taxel_count"] == 48
    assert len(result["skipped_taxels"]) == 32
    assert all(
        row["row"] not in (0, 7) and row["column"] not in (0, 9)
        for row in result["positions"]
    )
    assert host.taxel_mapping_live_maps[-1]["measured"] == sorted(
        row["taxel_index"] for row in result["positions"]
    )
    assert host.taxel_mapping_progress[-1]["total"] == 48


def test_taxel_mapping_resumes_partial_json_without_remapping_saved_taxels(
    tmp_path,
):
    api = _SimRobotApi()
    original = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json",
        output_directory=tmp_path / "maps",
    )
    original.move_settle_sec = 0.0
    # Keep this resume/geometry simulation on its original 50-count field.
    original.proximity_detection_mode = "fixed"
    original.centre_scan_signal_multiplier = 2.
    original.reference_probe_speed_m_s = 0.002
    original.probe_speed_m_s = 0.0005
    original.coarse_radius_m = 0.006
    original.coarse_step_m = 0.0015
    first_host = _Host(api, task=original)
    first_data = first_host.ui_ros.sensor_functions._data
    first_data.diffDataAve = _sim_frame(api.tool_position)
    assert original.capture_anchor_pose(first_host)
    assert original.on_start(first_host)

    for _ in range(100000):
        api.step(0.05)
        first_data.diffDataAve = _sim_frame(api.tool_position)
        first_host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
        _tick(original, first_host)
        if len(original._positions) == 4 and original._current_taxel == 43:
            break
    else:
        pytest.fail("simulation did not reach an interruptible partial map")

    original.on_stop(first_host, show_result=False)
    partial = original.last_result
    assert partial["mapped_count"] == 4
    assert not partial["completed"]
    assert {row["taxel_index"] for row in partial["positions"]} == {
        27, 34, 35, 36,
    }
    partial_path = Path(partial["json_path"])
    preserved = {
        row["taxel_index"]: copy.deepcopy(row)
        for row in partial["positions"]
        if row["taxel_index"] != 35
    }
    valid, message = TaxelPositionMappingTask.validate_resume_result(partial)
    assert valid, message

    resumed = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json",
        output_directory=tmp_path / "maps",
    )
    resumed.move_settle_sec = 0.0
    accepted, message = resumed.request_resume(partial, partial_path)
    assert accepted, message
    second_host = _Host(api, task=resumed)
    second_data = second_host.ui_ros.sensor_functions._data
    second_data.diffDataAve = _sim_frame(api.tool_position)
    assert resumed.on_start(second_host)
    assert len(resumed._positions) == 4
    assert resumed._move_action == "resume_anchor_arrived"

    for _ in range(300000):
        if not resumed._active:
            break
        api.step(0.05)
        second_data.diffDataAve = _sim_frame(api.tool_position)
        second_host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
        _tick(resumed, second_host)

    result = resumed.last_result
    assert result["completed"], (result["stop_reason"], second_host.messages[-5:])
    assert result["mapped_count"] == 80
    assert Path(result["json_path"]) == partial_path
    assert result["resume_history"][-1]["mapped_count_before_resume"] == 4
    contexts = [row["context"] for row in result["sensor_calibration_history"]]
    assert contexts[0] == "fresh_start"
    assert contexts.count("resume") == 1
    assert "resume_anchor_taxel_35" in contexts
    assert len(contexts) >= 81  # each verified taxel also recalibrates
    completed = {row["taxel_index"]: row for row in result["positions"]}
    for index, record in preserved.items():
        assert completed[index]["tool_position_m"] == record["tool_position_m"]
        assert completed[index]["attempt_count"] == record["attempt_count"]


def test_taxel_mapping_resume_rejects_a_completed_or_too_early_map():
    result = _saved_navigation_map()
    result["completed"] = True
    valid, message = TaxelPositionMappingTask.validate_resume_result(result)
    assert not valid
    assert "already complete" in message

    result["completed"] = False
    valid, message = TaxelPositionMappingTask.validate_resume_result(result)
    assert not valid
    assert "sensor height" in message


def test_taxel_mapping_unstable_baseline_never_sends_downward_velocity(
    tmp_path,
):
    api, task, host = _started_mapping(tmp_path)
    assert _tick_until(task, host, lambda: task._phase == "collecting")

    for frame_index in range(task.baseline_frame_count):
        frame = host.ui_ros.sensor_functions._data.diffDataAve
        frame.fill(0.0)
        frame[3, 4] = 100.0 if frame_index % 2 else 0.0
        host.ui_ros.sensor_functions._data.frame_sequence += 1
        task.on_tick(host)

    assert task._baseline_checks[-1]["stable"] is False
    assert task._pending_recovery["retry"] is True
    assert task._phase == "moving"
    assert not api.velocity_commands
    assert api.sent[-1][0] == [0.1, 0.2, 0.3]


def test_taxel_mapping_returns_to_anchor_and_retries_one_time(tmp_path):
    api, task, host = _mapping_mid_run(tmp_path, 36, "neighbor", 1)
    task._phase = "probing"
    task._current_target_xy = np.asarray([0.104, 0.201])
    api.tool_position = [0.104, 0.201, 0.285]

    task._handle_probe_failure(host, "no proximity onset")

    assert task._pending_recovery["retry"] is True
    assert api.sent[-1][0] == pytest.approx([0.104, 0.201, 0.3])

    task._dispatch_move_action(host, "route_waypoint")
    assert api.sent[-1][0] == pytest.approx([0.1, 0.2, 0.3])
    assert task._move_action == "recovery_anchor_arrived"
    task._dispatch_move_action(host, "recovery_anchor_arrived")
    assert task._collection_action == "clearance_precheck"

    stable = np.zeros(80, dtype=float)
    for sequence in range(task.baseline_frame_count):
        task._tick_collection(
            host,
            (100 + sequence, stable),
            force_n=0.0,
        )

    assert task._phase == "sensor_calibrating"
    assert host.ui_ros.sensor_functions.update_cal_calls == 1
    assert _tick_until(task, host, lambda: task._pending_recovery is None, limit=400)

    assert task._pending_recovery is None
    assert task._attempt_counts[36] == 2
    assert task._current_taxel == 36
    assert api.sent[-1][0] == pytest.approx([0.104, 0.201, 0.3])


def test_taxel_mapping_first_probe_failure_retries_from_the_anchor(tmp_path):
    api, task, host = _started_mapping(tmp_path, probe_motion="velocity")
    assert _tick_until(task, host, lambda: task._phase == "probing")
    api.tool_position = [0.1, 0.2, 0.3 - task.reference_max_descent_m - 0.0001]
    _tick(task, host, 2 + task.probe_sample_frames)

    assert task._pending_recovery["kind"] == "reference"
    assert task._pending_recovery["retry"] is True
    assert api.sent[-1][0] == pytest.approx([0.1, 0.2, 0.3])
    assert _tick_until(task, host, lambda: task._phase == "probing", limit=400)
    assert task._probe_kind == "reference"
    assert task._reference_attempt_count == 2


def test_taxel_mapping_retries_missing_coarse_neighbours_from_anchor(
    tmp_path,
):
    api, task, host = _mapping_mid_run(tmp_path, 35, "anchor", 1)
    task._phase = "collecting"
    task._current_target_xy = np.asarray([0.1, 0.2])
    task._scan_baseline = np.zeros(80, dtype=float)
    task._scan_results = []
    api.tool_position = [0.108, 0.208, 0.285]

    task._finish_coarse_scan(host)

    assert task._pending_recovery["kind"] == "target"
    assert task._pending_recovery["taxel_index"] == 35
    assert task._pending_recovery["retry"] is True
    assert task._pending_recovery["next_attempt"] == 2
    assert "coarse scan" in task._pending_recovery["reason"]
    assert api.sent[-1][0] == pytest.approx([0.108, 0.208, 0.3])


def test_taxel_mapping_second_failure_returns_to_anchor_then_stops(tmp_path):
    api, task, host = _mapping_mid_run(tmp_path, 36, "neighbor", 2)
    task._phase = "probing"
    task._current_target_xy = np.asarray([0.104, 0.201])
    api.tool_position = [0.104, 0.201, 0.285]

    task._handle_probe_failure(host, "no proximity onset")
    assert task._pending_recovery["retry"] is False
    assert api.sent[-1][0] == pytest.approx([0.104, 0.201, 0.3])

    task._dispatch_move_action(host, "route_waypoint")
    assert api.sent[-1][0] == pytest.approx([0.1, 0.2, 0.3])
    task._dispatch_move_action(host, "recovery_anchor_arrived")

    assert not host.stops
    assert _tick_until(task, host, lambda: not task._active, limit=400)
    assert host.ui_ros.sensor_functions.update_cal_calls == 1

    assert host.stops
    assert "exhausted its 2 attempts" in host.stops[-1][1]
    assert "Remember Scan Height" not in host.stops[-1][1]
    assert task._active is False
    assert task.last_result["stop_reason"] == "mapping_target_failed"


def test_anchor_failing_in_automatic_mode_says_to_remember_a_scan_height(tmp_path):
    api, task, host = _mapping_mid_run(tmp_path, 35, "anchor", 2)
    task._phase = "probing"
    task._current_target_xy = np.asarray([0.1, 0.2])
    api.tool_position = [0.1, 0.2, 0.285]

    task._handle_probe_failure(host, "stronger non-contact scan signal was not reached")
    for _ in range(3):
        if task._move_action in {"route_waypoint", "recovery_anchor_arrived"}:
            task._dispatch_move_action(host, task._move_action)
    assert _tick_until(task, host, lambda: not task._active, limit=400)

    assert "Taxel 35 exhausted its 2 attempts" in host.stops[-1][1]
    assert "automatic descent found no usable scan plane" in host.stops[-1][1]
    assert "Remember Scan Height" in host.stops[-1][1]


def test_taxel_mapping_completion_returns_to_anchor_before_reporting(
    tmp_path,
):
    api = _RobotApi()
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json", output_directory=None
    )
    host = _Host(api, task=task)
    assert task.capture_anchor_pose(host)
    task._active = True
    task._mapping_queue = []
    task._positions = {
        index: {"tool_position_m": [0.1, 0.2, 0.295]}
        for index in range(80)
    }

    task._start_next_regular_taxel(host)

    assert not task._completed
    assert not host.stops
    assert api.sent[-1][0] == [0.1, 0.2, 0.3]

    task._dispatch_move_action(host, "complete_anchor_arrived")

    assert task._completed
    assert host.stops[-1][0] is True
    assert "returned to Taxel 35" in host.stops[-1][1]


def test_taxel_mapping_force_guard_stops_without_automatic_return(tmp_path):
    api = _RobotApi()
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json", output_directory=None
    )
    host = _Host(api, task=task)
    assert task.capture_anchor_pose(host)
    assert task.on_start(host)
    sent_before_guard = len(api.sent)

    host.ui_ros._force_meter_last_raw_newtons = (
        task.force_guard_g + 1.0
    ) * task.NEWTONS_PER_GRAM
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    host.ui_ros.sensor_functions._data.frame_sequence += 1
    task.on_tick(host)

    assert task._active is False
    assert task.last_result["stop_reason"] == "force_guard"
    assert len(api.sent) == sent_before_guard


def test_taxel_mapping_stale_force_stops_motion_and_saves_partial_state(tmp_path):
    api = _RobotApi()
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json", output_directory=None
    )
    host = _Host(api, task=task)
    assert task.capture_anchor_pose(host)
    assert task.on_start(host)

    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic() - 5.0
    host.ui_ros.sensor_functions._data.frame_sequence += 1
    task.on_tick(host)

    assert host.stops
    assert "force-transmitter reading became stale" in host.stops[-1][1]
    assert not task._active
    assert task.last_result["stop_reason"] == "force_stale"


def test_taxel_mapping_saves_partial_map_files_in_one_folder(tmp_path):
    task = TaxelPositionMappingTask(
        storage_path=None, output_directory=tmp_path
    )
    task.active_sensor_label = "Testing Sensor / #1"
    task.anchor_pose = {
        "saved_at": "2026-09-26T00:00:00+00:00",
        "taxel_index": 35,
        "tool_position_m": [0.1, 0.2, 0.3],
        "tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    task._started_at_iso = "2026-09-26T00:00:00+00:00"
    task._prepare_output_folder()
    task._positions[35] = {
        "taxel_index": 35,
        "row": 3,
        "column": 4,
        "status": "measured",
        "tool_position_m": [0.1, 0.2, 0.295],
        "safe_approach_position_m": [0.1, 0.2, 0.3],
        "tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "signal_counts": 100.0,
        "competing_signal_counts": 20.0,
        "dominance_ratio": 5.0,
        "confidence": 1.0,
        "method": "bounded_z_probe_and_local_xy_scan",
    }
    task._sample_log.append(
        {
            "scan": "local",
            "taxel_index": 35,
            "point_number": 1,
            "x_m": 0.1,
            "y_m": 0.2,
            "z_m": 0.295,
            "target_signal_counts": 100.0,
            "competing_signal_counts": 20.0,
            "dominance_ratio": 5.0,
            "dominant_taxel": 35,
        }
    )

    result = task._save_results()

    folder = Path(result["output_folder"])
    assert folder.parent == tmp_path
    assert "_sensor-Testing-Sensor-1_" in folder.name
    assert Path(result["json_path"]).parent == folder
    assert Path(result["positions_csv_path"]).parent == folder
    assert Path(result["samples_csv_path"]).parent == folder
    assert {path.name for path in folder.iterdir()} == {
        "baseline_checks.csv",
        "taxel_position_map.json",
        "taxel_positions.csv",
        "scan_samples.csv",
        "probe_levels.csv",
    }


def _saved_navigation_map():
    return {
        "task": "taxel_position_mapping",
        "version": 2,
        "completed": False,
        "settings": {"sensor_rows": 8, "sensor_columns": 10, "max_xy_offset_mm": 60.0},
        "anchor_pose": {
            "taxel_index": 35,
            "tool_position_m": [0.1, 0.2, 0.3],
            "tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "positions": [
            {
                "taxel_index": index,
                "row": index % 8,
                "column": index // 8,
                "status": "measured",
                "tool_position_m": [x, 0.2, 0.292],
                "safe_approach_position_m": [x, 0.2, 0.3],
                "tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
            for index, x in [(35, 0.1), (43, 0.104)]
        ],
    }


def _saved_full_calibration_map():
    result = _saved_navigation_map()
    result["completed"] = True
    result["probe_start_z_m"] = 0.295
    for record in result["positions"]:
        record["probe_start_z_m"] = 0.295
    return result


def test_full_taxel_calibration_is_registered_with_safe_defaults():
    task = get_task("full_taxel_calibration")

    assert isinstance(task, FullTaxelCalibrationTask)
    assert task.label == "Calibration (Full Taxel)"
    assert task.show_in_sidebar is False
    sidebar_ids = {task_id for _label, task_id, _description, _category in task_definitions()}
    assert "calibration" in sidebar_ids
    assert "full_taxel_calibration" not in sidebar_ids
    grouped = {}
    for _label, task_id, _description, category in task_definitions():
        grouped.setdefault(category, []).append(task_id)
    assert grouped["TRO2026"] == [
        "taxel_position_mapping",
        "calibration",
        "graph_analysis",
        "sensor_characterisation",
    ]
    assert grouped["Archive"] == ["sigraph2026_scanning"]
    assert task.start_button_label == "Run Full Taxel Calibration"
    assert task.tick_interval_ms == 50
    assert task.LINE_MOTION == 4
    assert np.isclose(task.line_speed_m_s, 0.02)
    assert np.isclose(task.route_force_guard_g, 20.0)


def test_full_taxel_calibration_loads_map_in_nearest_physical_order():
    task = FullTaxelCalibrationTask(output_directory=None)

    loaded, message = task.load_position_map(_saved_full_calibration_map())

    assert loaded
    assert "2 measured taxels" in message
    assert [
        item["taxel_index"] for item in task.position_map["positions"]
    ] == [35, 43]
    assert task.position_map["positions"][0][
        "calibration_start_position_m"
    ] == [0.1, 0.2, 0.3]


def test_full_taxel_live_read_keeps_neighbours_but_recording_keeps_only_target():
    task = FullTaxelCalibrationTask(output_directory=None)
    task.taxel_index = 35
    host = _Host(_RobotApi(), task)
    data = host.ui_ros.sensor_functions._data
    data.diffDataAve[3, 4] = -17.0
    data.diffDataAve[3, 5] = 91.0

    taxel = task._read_taxel(host)

    assert taxel["diff_ave"] == -17.0
    assert taxel["taxel_035_diff_ave"] == -17.0
    assert taxel["taxel_043_diff_ave"] == 91.0
    assert len([key for key in taxel if key.startswith("taxel_")]) == 400
    recorded = task._profile_trimmed_taxel(taxel)
    assert recorded["taxel_035_diff_ave"] == -17.0
    assert "taxel_043_diff_ave" not in recorded


def _navigation_task(result=None, index=43):
    return MappedTaxelMoveTask(
        result or _saved_navigation_map(), index,
        line_speed_m_s=0.02, force_guard_g=20.0,
    )


def test_navigation_map_loading_is_independent_and_ignores_unmeasured_taxels():
    saved = _saved_navigation_map()
    saved["positions"].append({"taxel_index": 0, "status": "predicted"})
    original = copy.deepcopy(saved)
    prepared = prepare_taxel_map(saved)

    assert [row["taxel_index"] for row in prepared["positions"]] == [35, 43]
    assert saved == original
    with pytest.raises(ValueError, match="no measured position"):
        _navigation_task(saved, 0)
    prepared["anchor_pose"]["tool_position_m"][2] = 99
    assert saved == original


@pytest.mark.parametrize("kind", ["duplicate", "nan", "low_clearance", "wrong_xy", "orientation", "index", "workspace"])
def test_navigation_rejects_invalid_saved_coordinates(kind):
    result = _saved_navigation_map()
    record = result["positions"][0]
    if kind == "duplicate":
        result["positions"].append(copy.deepcopy(record))
    elif kind == "nan":
        record["safe_approach_position_m"][0] = float("nan")
    elif kind == "low_clearance":
        record["safe_approach_position_m"][2] = 0.29
    elif kind == "wrong_xy":
        record["safe_approach_position_m"][0] += 0.01
    elif kind == "orientation":
        record["tool_quaternion_wxyz"] = [0.0, 1.0, 0.0, 0.0]
    elif kind == "index":
        record["taxel_index"] = 80
    elif kind == "workspace":
        record["safe_approach_position_m"][0] = 0.5
        record["tool_position_m"][0] = 0.5
    with pytest.raises(ValueError):
        _navigation_task(result)


@pytest.mark.parametrize("start_z,expected", [
    (0.29, [[0.1, 0.2, 0.3], [0.104, 0.2, 0.3]]),
    (0.2999, [[0.1, 0.2, 0.3], [0.104, 0.2, 0.3]]),
    (0.3, [[0.104, 0.2, 0.3]]),
    (0.31, [[0.104, 0.2, 0.3]]),
])
def test_navigation_routes_above_selected_taxel_without_probing_or_changing_map(start_z, expected):
    saved = _saved_navigation_map()
    original = copy.deepcopy(saved)
    task = _navigation_task(saved)
    task.move_settle_sec = 0
    api = _RobotApi()
    api.tool_position[2] = start_z
    host = _Host(api, task=task)
    assert task.on_start(host)
    assert api.sent == []  # A new tactile frame is required before moving.
    assert _tick_until(task, host, lambda: not task._active)

    np.testing.assert_allclose([entry[0] for entry in api.sent], expected)
    assert all(entry[2]["motion_type"] == task.LINE_MOTION for entry in api.sent)
    assert task._completed
    assert not api.velocity_commands
    assert "Arrived above Taxel 43" in host.messages[-1]
    assert saved == original
    assert task.last_result is None  # A move must not overwrite the mapping result.


@pytest.mark.parametrize("fault", ["force_stale", "force_guard", "sensor_stale", "timeout", "manual"])
def test_navigation_stop_cancels_remaining_waypoints(fault):
    task = _navigation_task()
    api = _RobotApi()
    api.tool_position[2] = 0.29
    host = _Host(api, task=task)
    assert task.on_start(host)
    _tick(task, host)
    assert len(api.sent) == 1
    if fault == "force_stale":
        host.ui_ros._force_meter_last_sample_monotonic -= 5
    elif fault == "force_guard":
        host.ui_ros._force_meter_last_raw_newtons = 30 * task.NEWTONS_PER_GRAM
    elif fault == "sensor_stale":
        task._last_sensor_advance_at -= 5
    elif fault == "timeout":
        api.tool_position[2] = 0.29
        task._move_started_at -= 50
    else:
        host._stop_sidebar_control(show_result=False)
    task.on_tick(host)
    assert not task._active
    _tick(task, host, 4)
    assert len(api.sent) == 1
    assert not task._completed
    assert task._route_queue == []
    assert task._navigation_queue == []


def test_navigation_auto_connects_stale_force_before_any_motion():
    task = _navigation_task()
    api = _RobotApi()
    host = _Host(api, task=task)
    host.ui_ros._force_meter_last_sample_monotonic -= 5

    assert task.on_start(host)
    assert task._phase == "connecting_force_meter"
    assert host.ui_ros.force_meter_start_calls == 1
    assert "connecting the force transmitter" in host.messages[-1]
    task.on_tick(host)
    assert task._phase == "connecting_force_meter" and not api.sent

    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    assert task._phase == "navigation_preflight"
    assert not api.sent  # Motion still waits for a new tactile frame.


def test_navigation_refuses_other_experiments_without_commands():
    task = _navigation_task()
    api = _RobotApi()
    host = _Host(api, task=task)
    host._sidebar_active_task = object()
    assert not task.on_start(host)
    assert "current experiment" in host.messages[-1]
    assert not api.sent


@pytest.mark.parametrize("far_xy,wrong_orientation,start_z", [
    (True, False, .35),
    (True, False, .3),
    (False, True, .29),
    (True, True, .29),
    (True, True, .35),
])
def test_navigation_goes_to_selected_taxel_without_visiting_anchor(far_xy, wrong_orientation, start_z):
    saved = _saved_navigation_map()
    original = copy.deepcopy(saved)
    task = _navigation_task(saved)
    task.move_settle_sec = 0
    api = _RobotApi()
    api.tool_position = [.35 if far_xy else .11, .2, start_z]
    angle = np.radians(33.0) / 2
    current_q = [np.cos(angle), 0., 0., np.sin(angle)] if wrong_orientation else [1., 0., 0., 0.]
    api.tool_quaternion = list(current_q)
    host = _Host(api, task)
    assert task.on_start(host)
    assert not api.sent
    assert any("without returning to the anchor" in message for message in host.messages)
    assert _tick_until(task, host, lambda: not task._active)
    assert task._completed
    points = [np.asarray(command[0]) for command in api.sent]
    assert not any(np.allclose(point[:2], [.1, .2]) for point in points)
    np.testing.assert_allclose(points[-1], [.104, .2, .3])
    assert all(point[2] >= .3 for point in points)
    # Only the last waypoint changes XY. Earlier lift/rotation stays in place.
    assert all(np.allclose(point[:2], [.35 if far_xy else .11, .2]) for point in points[:-1])
    if not wrong_orientation and start_z >= .3:
        assert len(points) == 1
    if start_z < .3:
        np.testing.assert_allclose(points[0], [.35 if far_xy else .11, .2, .3])
        np.testing.assert_allclose(api.sent[0][1], current_q)  # Lift without rotating the probe.
    if wrong_orientation:
        rotation_index = next(i for i, command in enumerate(api.sent) if np.allclose(command[1], [1, 0, 0, 0]))
        assert points[rotation_index][2] >= .3
        assert points[rotation_index][0] == pytest.approx(.35 if far_xy else .11)
    assert saved == original


def test_navigation_waits_for_actual_rotation_and_times_out_without_continuing(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("phd.ui.experiment_tasks.time.monotonic", lambda: clock[0])
    task = _navigation_task()
    task.move_settle_sec = 0
    api = _RobotApi()
    api.tool_position = [.11, .2, .3]
    angle = np.radians(33) / 2
    wrong_q = [np.cos(angle), 0, 0, np.sin(angle)]
    api.tool_quaternion = wrong_q.copy()
    original_send = api.send_positions_tool_position

    def ignore_rotation(*args, **kwargs):
        result = original_send(*args, **kwargs)
        api.tool_quaternion = wrong_q.copy()
        return result

    api.send_positions_tool_position = ignore_rotation
    host = _Host(api, task)
    assert task.on_start(host)
    _tick(task, host, 5)
    assert len(api.sent) == 1  # Rotate in place; no lateral motion yet.
    assert task._active
    clock[0] += task._move_timeout_current_sec + 1
    host.ui_ros._force_meter_last_sample_monotonic = clock[0]
    _tick(task, host)
    assert not task._active
    assert task._stop_reason == "orientation_timeout"
    assert len(api.sent) == 1
    assert not task._navigation_queue


@pytest.mark.parametrize("fault", ["manual", "force_stale", "force_guard", "sensor_stale", "robot_stale", "invalid_orientation"])
def test_navigation_clearance_preparation_stops_without_sending_remaining_waypoints(fault):
    task = _navigation_task()
    api = _RobotApi()
    api.tool_position = [.35, .2, .29]
    api.tool_quaternion = [np.cos(.3), 0, 0, np.sin(.3)]
    host = _Host(api, task)
    assert task.on_start(host)
    _tick(task, host)
    assert len(api.sent) == 1
    if fault == "manual":
        host._stop_sidebar_control(show_result=False)
    elif fault == "force_stale":
        host.ui_ros._force_meter_last_sample_monotonic -= 5
    elif fault == "force_guard":
        host.ui_ros._force_meter_last_raw_newtons = 30 * task.NEWTONS_PER_GRAM
    elif fault == "sensor_stale":
        task._last_sensor_advance_at -= 5
    elif fault == "robot_stale":
        api.current_feedback_state_time = time.monotonic() - 5
    else:
        api.tool_quaternion = [0, 0, 0, 0]
    task.on_tick(host)
    assert not task._active
    assert not task._navigation_queue
    assert len(api.sent) == 1


def test_navigation_rejects_invalid_feedback_before_direct_move():
    task = _navigation_task()
    api = _RobotApi()
    host = _Host(api, task)
    api.tool_position[0] = .35
    api.tool_quaternion = [0, 0, 0, 0]
    assert not task.on_start(host)
    api.tool_quaternion = [1, 0, 0, 0]
    api.current_feedback_state_time = time.monotonic() - 5
    assert not task.on_start(host)
    assert not api.sent


def test_navigation_refuses_paused_robot_without_sending_or_queueing_motion():
    task = _navigation_task()
    api = _RobotApi()
    api.current_project_paused = True
    host = _Host(api, task)
    assert not task.on_start(host)
    assert "robot project is paused" in host.messages[-1]
    assert not api.sent
    assert not api.scripts
    assert not task._navigation_queue


@pytest.mark.parametrize("pause_before_first_move", [True, False])
def test_navigation_pause_cancels_route_and_does_not_auto_resume(pause_before_first_move):
    task = _navigation_task()
    api = _RobotApi()
    api.tool_position = [.35, .2, .29]
    api.current_project_paused = False
    host = _Host(api, task)
    assert task.on_start(host)
    if not pause_before_first_move:
        _tick(task, host)
        assert len(api.sent) == 1
    count = len(api.sent)
    api.current_project_paused = True
    _tick(task, host)
    assert not task._active
    assert task._stop_reason == "robot_project_paused"
    assert not task._navigation_queue
    api.current_project_paused = False
    _tick(task, host, 5)
    assert len(api.sent) == count
    assert not task._completed


def test_twenty_by_twenty_mapping_uses_requested_anchor_and_radial_workspace():
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    host = _Host(_RobotApi(), task)
    sensor = host.ui_ros.sensor_functions
    sensor.n_row = sensor.n_col = 20
    sensor._data.diffDataAve = np.zeros((20, 20))
    task.use_connected_sensor(host)
    assert task.anchor_taxel == 189
    assert task.anchor_neighbour_indices() == (188, 190, 169, 209)
    assert task.mapping_target_indices() == [
        column * 20 + row for column in range(1, 19) for row in range(1, 19)
    ]
    assert len(task.mapping_target_indices()) == 324
    assert len(task.skipped_taxel_indices()) == 76
    assert set(task.mapping_target_indices()).isdisjoint(task.skipped_taxel_indices())
    assert task._settings()["skip_outer_border"] is True
    old_settings = task._settings()
    old_settings["skip_outer_border"] = False
    task._apply_resume_settings({
        "settings": old_settings,
        "anchor_pose": {
            "taxel_index": 189, "tool_position_m": [.1, .2, .3],
            "tool_quaternion_wxyz": [1., 0., 0., 0.],
        },
    })
    assert task.skip_outer_border
    assert len(task.mapping_target_indices()) == 324
    assert task.max_xy_offset_m == pytest.approx(.2)
    assert task.coarse_radius_m == pytest.approx(.02)
    assert task.coarse_step_m == pytest.approx(.002)
    assert task.start_button_label == "Map 20x20 Sensor"
    task.max_xy_offset_m = .18
    task.coarse_radius_m = .015
    task.use_connected_sensor(host)
    assert task.max_xy_offset_m == pytest.approx(.18)
    assert task.coarse_radius_m == pytest.approx(.015)
    sensor.n_row, sensor.n_col = 8, 10
    sensor._data.diffDataAve = np.zeros((8, 10))
    task.use_connected_sensor(host)
    assert task.anchor_taxel == 35
    assert task.max_xy_offset_m == pytest.approx(.1)
    assert task.coarse_radius_m == pytest.approx(.008)
    assert not task.skip_outer_border


def _settle_full_taxel_route(task, host, limit=40):
    for _ in range(limit):
        if task._phase != "full_moving":
            return
        host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
        task.on_tick(host)
    raise AssertionError("full-taxel route did not settle")


def _settle_post_save_feedback(task, host):
    for _ in range(4):
        if task._phase != "waiting_post_save_feedback":
            return
        host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
        host.ui_ros.sensor_functions._data.frame_sequence += 1
        task.on_tick(host)
    assert task._phase != "waiting_post_save_feedback"


def _complete_full_taxel_contact_trial(task, host, api, force_n):
    assert task._phase == "force_baseline"
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    assert task._phase == "approaching"
    host.ui_ros._force_meter_last_raw_newtons = float(force_n)
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    assert task._phase == "contact_hold"
    api.tool_position[2] -= 0.001
    task._contact_hold_started_at -= 2.0
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    assert task._phase == "contact_hold"  # Zero velocity until the TCP is still.
    task.on_tick(host)
    assert task._phase == "returning"
    api.tool_position[2] += 0.001
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    task.on_tick(host)
    _settle_post_save_feedback(task, host)


def test_full_taxel_calibration_routes_and_saves_each_taxel_separately(tmp_path):
    api = _RobotApi()
    task = FullTaxelCalibrationTask(output_directory=tmp_path)
    loaded, _message = task.load_position_map(
        _saved_full_calibration_map(), tmp_path / "taxel_position_map.json"
    )
    assert loaded
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.return_settle_sec = 0.0
    task.move_settle_sec = 0.0
    task.contact_threshold_n = 0.1
    host = _Host(api, task)
    exported = []

    def export_graph(result, path):
        exported.append((len(result["rows"]), Path(path)))
        Path(path).write_bytes(b"png")
        return True

    host._export_calibration_trial_graph = export_graph
    data = host.ui_ros.sensor_functions._data
    data.diffDataAve[3, 4] = 35.0
    data.diffDataAve[3, 5] = 43.0

    assert task.on_start(host)
    assert task._phase == "full_moving"
    assert task.taxel_index == 35
    _settle_full_taxel_route(task, host)
    assert task._phase == "force_baseline"
    _complete_full_taxel_contact_trial(task, host, api, 0.2)

    assert task._phase == "full_moving"
    assert task._completed_taxels == [35]
    assert task._samples == []
    assert exported[-1][0] > 0 and exported[-1][1].parent.name == "taxel_35"
    _settle_full_taxel_route(task, host)
    assert task._phase == "full_recalibrating"
    sensor = host.ui_ros.sensor_functions
    sensor._sensor_calibration_in_progress = False
    sensor.is_connected = True
    task.on_tick(host)
    assert task._phase == "full_moving"
    assert task.taxel_index == 43
    _settle_full_taxel_route(task, host)
    assert task._phase == "force_baseline"
    _complete_full_taxel_contact_trial(task, host, api, 0.4)

    assert task._completed_taxels == [35, 43]
    _settle_full_taxel_route(task, host)  # retract Taxel 43
    _settle_full_taxel_route(task, host)  # return to Taxel 35 clearance

    assert task.last_result is not None
    assert task.last_result["status"] == "complete"
    assert task.last_result["completed_taxels"] == 2
    assert task.last_result["home_position"]["source"] == "taxel_position_map_anchor"
    np.testing.assert_allclose(api.tool_position, task.position_map["anchor_pose"]["tool_position_m"])
    metadata = json.loads(Path(task.last_result["metadata_path"]).read_text())
    assert metadata["home_position"] == task.last_result["home_position"]
    assert host.full_taxel_calibration_results[-1] is task.last_result
    assert Path(task.last_result["summary_path"]).exists()
    assert Path(task.last_result["metadata_path"]).exists()
    assert Path(task.last_result["experiment_directory"]).parent == tmp_path
    assert [
        item["taxel_index"] for item in task.last_result["taxel_results"]
    ] == [35, 43]
    for expected, result in zip(
        (35, 43), task.last_result["taxel_results"]
    ):
        assert result["recorded_taxel_indices"] == [expected]
        assert result["recording_scope"] == "target_taxel_only"
        assert Path(result["experiment_directory"]).name == f"taxel_{expected:02d}"
        # Samples leave RAM once saved; only the pressed taxel reloads from CSV.
        assert result["rows"] == [] and result["row_count"] > 0
        assert Path(result["graph_path"]).name == "trial_0001.png"
        row = with_trial_rows(result)["rows"][-1]
        assert f"taxel_{expected:03d}_diff_ave" in row
        other = 43 if expected == 35 else 35
        assert f"taxel_{other:03d}_diff_ave" not in row
    # The only XY changes happened at the saved 0.300 m clearance height.
    tool_moves = [entry for entry in api.sent if len(entry) == 3]
    for previous, current in zip(tool_moves, tool_moves[1:]):
        previous_position = previous[0]
        current_position = current[0]
        if not np.allclose(previous_position[:2], current_position[:2]):
            assert np.isclose(current_position[2], 0.3)


def test_full_taxel_calibration_repeats_the_whole_set_before_the_next_pass(tmp_path):
    api = _RobotApi()
    task = FullTaxelCalibrationTask(output_directory=tmp_path)
    loaded, _message = task.load_position_map(
        _saved_full_calibration_map(), tmp_path / "taxel_position_map.json"
    )
    assert loaded
    task.repeat_count = 2
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.return_settle_sec = 0.0
    task.move_settle_sec = 0.0
    task.contact_threshold_n = 0.1
    host = _Host(api, task)
    visits = []

    def run_one_contact():
        _settle_full_taxel_route(task, host)
        assert task._phase == "force_baseline"
        visits.append(int(task.taxel_index))
        host.ui_ros._force_meter_last_raw_newtons = 0.0
        _complete_full_taxel_contact_trial(task, host, api, 0.2)

    assert task.on_start(host)
    run_one_contact()
    assert task._completed_taxels == []
    assert task._pass_number == 1
    _settle_full_taxel_route(task, host)
    sensor = host.ui_ros.sensor_functions
    sensor._sensor_calibration_in_progress = False
    sensor.is_connected = True
    task.on_tick(host)

    run_one_contact()
    assert task._completed_taxels == []
    _settle_full_taxel_route(task, host)
    assert task._pass_number == 2
    sensor._sensor_calibration_in_progress = False
    sensor.is_connected = True
    task.on_tick(host)

    run_one_contact()
    assert task._completed_taxels == [35]
    _settle_full_taxel_route(task, host)
    sensor._sensor_calibration_in_progress = False
    sensor.is_connected = True
    task.on_tick(host)

    run_one_contact()
    assert task._completed_taxels == [35, 43]
    _settle_full_taxel_route(task, host)
    _settle_full_taxel_route(task, host)

    assert visits == [35, 43, 35, 43]
    assert task.last_result["status"] == "complete"
    for taxel, result in zip((35, 43), task.last_result["taxel_results"]):
        assert result["taxel_index"] == taxel
        assert result["completed_trials"] == 2
        assert [item["trial_number"] for item in result["trial_results"]] == [1, 2]
        folder = Path(result["experiment_directory"])
        assert (folder / "trial_0001.csv").exists()
        assert (folder / "trial_0002.csv").exists()


def test_full_taxel_home_comes_from_map_not_legacy_file_or_single_taxel(tmp_path):
    single = CalibrationTask(
        storage_path=tmp_path / "single.json", output_directory=None
    )
    legacy_path = tmp_path / "full.json"
    legacy_path.write_text(json.dumps({"version": 2, "profiles": {"layout:8x10": {
        "saved_at": "2026-09-30T07:18:41Z", "joints_rad": [2.] * 6,
        "tool_position_m": [.4, .5, .6], "tool_quaternion_wxyz": [1., 0., 0., 0.],
    }}}))
    legacy_bytes = legacy_path.read_bytes()
    full = FullTaxelCalibrationTask(
        storage_path=legacy_path, output_directory=None
    )
    single_host = _Host(_RobotApi())
    single_host.ui_ros.robot_api.current = np.ones(6)
    single_host.ui_ros.robot_api.tool_position = [0.4, 0.5, 0.6]
    assert single.capture_initial_position(single_host)
    single_bytes = single.storage_path.read_bytes()
    api = _RobotApi()
    host = _Host(api, full)
    assert full.initial_position is None
    assert not full.capture_initial_position(host)
    loaded, message = full.load_position_map(_saved_full_calibration_map(), "today.json")
    assert loaded and "Home follows its safe anchor" in message
    full.use_connected_sensor(host)
    assert full.initial_position["tool_position_m"] == [.1, .2, .3]
    assert full.initial_position["source_path"] == "today.json"
    assert not api.sent and not api.scripts and not api.velocity_commands
    # Even a stale in-memory pose/profile cannot override the map at start.
    full.initial_position = copy.deepcopy(single.initial_position)
    full._position_profiles["layout:8x10"] = copy.deepcopy(single.initial_position)
    assert full.on_start(host)
    assert full._home_position["tool_position_m"] == [.1, .2, .3]
    assert legacy_path.read_bytes() == legacy_bytes
    assert single.storage_path.read_bytes() == single_bytes
    restored = FullTaxelCalibrationTask(storage_path=legacy_path, output_directory=None)
    assert restored.initial_position is None
    assert restored.load_position_map(_saved_full_calibration_map())[0]
    assert restored.initial_position["tool_position_m"] == [.1, .2, .3]


def test_full_taxel_home_follows_relocated_map_and_cannot_change_mid_run():
    task = FullTaxelCalibrationTask(output_directory=None)
    first = _saved_full_calibration_map()
    assert task.load_position_map(first, "first.json")[0]
    relocated = copy.deepcopy(first)
    shift = np.array([.01, -.02, .03])
    relocated["anchor_pose"]["tool_position_m"] = (
        np.asarray(first["anchor_pose"]["tool_position_m"]) + shift
    ).tolist()
    relocated["probe_start_z_m"] += shift[2]
    for record in relocated["positions"]:
        for field in ("tool_position_m", "safe_approach_position_m"):
            record[field] = (np.asarray(record[field]) + shift).tolist()
        record["probe_start_z_m"] += shift[2]
    assert task.load_position_map(relocated, "relocated.json")[0]
    anchor = relocated["anchor_pose"]["tool_position_m"]
    assert task.initial_position["tool_position_m"] == anchor
    host = _Host(_RobotApi(), task)
    task.use_connected_sensor(host)
    assert task.initial_position["tool_position_m"] == anchor
    assert task.on_start(host)
    assert task._home_position["tool_position_m"] == anchor
    assert not task.load_position_map(first, "first.json")[0]
    assert task.position_map_path == "relocated.json"
    assert task._home_position["tool_position_m"] == anchor


def test_full_taxel_home_still_rejects_wrong_orientation_without_motion():
    task = FullTaxelCalibrationTask(output_directory=None)
    assert task.load_position_map(_saved_full_calibration_map())[0]
    api = _RobotApi()
    api.tool_quaternion = [0., 1., 0., 0.]
    host = _Host(api, task)
    assert not task.on_start(host)
    assert "Restore the tool orientation" in host.messages[-1]
    assert not api.sent and not api.scripts and not api.velocity_commands


def test_full_taxel_routes_via_map_anchor_before_first_taxel():
    task = FullTaxelCalibrationTask(output_directory=None)
    result = _saved_full_calibration_map()
    result["anchor_pose"]["tool_position_m"] = [.099, .199, .3]
    assert task.load_position_map(result)[0]
    task.move_settle_sec = 0.
    api = _RobotApi()
    api.tool_position = [.11, .21, .25]
    host = _Host(api, task)
    assert task.on_start(host)
    _settle_full_taxel_route(task, host)
    moves = [entry[0] for entry in api.sent if len(entry) == 3]
    np.testing.assert_allclose(moves[0], [.11, .21, .3])  # lift before XY
    np.testing.assert_allclose(moves[1], [.099, .199, .3])  # common home first
    np.testing.assert_allclose(moves[2], [.1, .2, .3])  # then the first taxel
    assert task._phase == "force_baseline"
    assert task._home_position["tool_position_m"] != task.initial_position["tool_position_m"]


def test_full_taxel_cannot_replace_map_while_connecting_force():
    task = FullTaxelCalibrationTask(output_directory=None)
    result = _saved_full_calibration_map()
    assert task.load_position_map(result, "original.json")[0]
    task._phase = "connecting_force_meter"
    assert not task.load_position_map(result, "replacement.json")[0]
    assert task.position_map_path == "original.json"


def test_single_taxel_calibration_starts_at_the_selected_map_pose(tmp_path):
    task = CalibrationTask(storage_path=None, output_directory=tmp_path)
    task.return_settle_sec = 0.0
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.use_mapped_taxel(
        {
            "taxel_index": 12,
            "tool_position_m": [0.2, 0.3, 0.139],
            "safe_approach_position_m": [0.2, 0.3, 0.185],
            "probe_start_z_m": 0.144,
            "tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        rows=5,
        columns=5,
    )
    api = _RobotApi()
    api.tool_position = [0.1, 0.2, 0.3]
    host = _Host(api, task)
    for name in (
        "rawData",
        "calData",
        "diffData",
        "diffPerData",
        "rawDataAve",
        "diffDataAve",
        "diffPerDataAve",
    ):
        setattr(host.ui_ros.sensor_functions._data, name, np.zeros((5, 5)))
    host.ui_ros.sensor_functions.n_row = 5
    host.ui_ros.sensor_functions.n_col = 5

    assert task.on_start(host)
    assert task._phase == "returning"
    for _ in range(8):
        host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
        task.on_tick(host)
        if task._phase == "force_baseline":
            break

    assert task._phase == "force_baseline"
    assert task.taxel_index == 12
    assert [entry[0] for entry in api.sent] == [
        [0.2, 0.3, 0.3],
        [0.2, 0.3, 0.144],
    ]
    assert all(entry[2]["motion_type"] == 4 for entry in api.sent)


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


def test_calibration_keeps_a_separate_pose_for_each_sensor(tmp_path, monkeypatch):
    storage_path = tmp_path / "calibration_initial_position.json"
    storage_path.write_text(
        json.dumps(
            {
                "version": 1,
                "saved_at": "2026-09-24T00:00:00+00:00",
                "joints_rad": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "tool_position_m": [1.0, 2.0, 3.0],
                "tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    selected = {"key": "sensor:testing_8x10"}

    def fake_sensor(_host):
        if selected["key"] == "sensor:testing_5x5_1":
            return "sensor:testing_5x5_1", "Testing Sensor 1", 5, 5
        return "sensor:testing_8x10", "Testing Sensor", 8, 10

    monkeypatch.setattr(
        "phd.ui.experiment_tasks.connected_calibration_sensor",
        fake_sensor,
    )
    task = CalibrationTask(storage_path=storage_path)
    np.testing.assert_allclose(
        task.initial_position["joints_rad"],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )

    selected["key"] = "sensor:testing_5x5_1"
    host = _Host(_RobotApi())
    assert task.use_connected_sensor(host) == "sensor:testing_5x5_1"
    assert task.initial_position is None
    assert (task.sensor_rows, task.sensor_columns, task.taxel_index) == (5, 5, 12)

    api = host.ui_ros.robot_api
    api.current = np.arange(6, dtype=float) + 1.0
    assert task.capture_initial_position(host)

    selected["key"] = "sensor:testing_8x10"
    restored = CalibrationTask(storage_path=storage_path)
    restored.use_connected_sensor(host)
    np.testing.assert_allclose(
        restored.initial_position["joints_rad"],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    selected["key"] = "sensor:testing_5x5_1"
    restored.use_connected_sensor(host)
    np.testing.assert_allclose(
        restored.initial_position["joints_rad"],
        api.current,
    )


def test_full_taxel_calibration_accepts_a_five_by_five_position_map():
    result = _saved_full_calibration_map()
    result["settings"] = {
        "sensor_rows": 5,
        "sensor_columns": 5,
        "max_xy_offset_mm": 60.0,
    }
    result["anchor_pose"]["taxel_index"] = 12
    result["positions"] = [
        {
            "taxel_index": 12,
            "row": 2,
            "column": 2,
            "status": "measured",
            "tool_position_m": [0.1, 0.2, 0.292],
            "safe_approach_position_m": [0.1, 0.2, 0.3],
            "tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "probe_start_z_m": 0.295,
        }
    ]

    prepared = FullTaxelCalibrationTask.prepare_position_map(result)

    assert prepared["settings"]["sensor_rows"] == 5
    assert prepared["positions"][0]["taxel_index"] == 12


def test_full_taxel_calibration_starts_with_a_live_five_by_five_sensor():
    result = _saved_full_calibration_map()
    result["settings"] = {
        "sensor_rows": 5,
        "sensor_columns": 5,
        "max_xy_offset_mm": 60.0,
    }
    result["anchor_pose"]["taxel_index"] = 12
    result["positions"] = [
        {
            "taxel_index": 12,
            "row": 2,
            "column": 2,
            "status": "measured",
            "tool_position_m": [0.1, 0.2, 0.292],
            "safe_approach_position_m": [0.1, 0.2, 0.3],
            "tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "probe_start_z_m": 0.295,
        }
    ]
    task = FullTaxelCalibrationTask(output_directory=None)
    loaded, _message = task.load_position_map(result)
    assert loaded
    host = _Host(_RobotApi(), task)
    sensor = host.ui_ros.sensor_functions
    sensor.n_row = 5
    sensor.n_col = 5
    for name in (
        "rawData",
        "calData",
        "diffData",
        "diffPerData",
        "rawDataAve",
        "diffDataAve",
        "diffPerDataAve",
    ):
        setattr(sensor._data, name, np.zeros((5, 5), dtype=float))
    assert task.on_start(host)
    assert task.taxel_index == 12
    assert task._phase == "full_moving"
    assert not any("live 5x5" in message for message in host.messages)


def test_full_taxel_calibration_rejects_a_sensor_shape_different_from_the_map():
    result = _saved_full_calibration_map()
    result["settings"] = {
        "sensor_rows": 5,
        "sensor_columns": 5,
        "max_xy_offset_mm": 60.0,
    }
    result["anchor_pose"]["taxel_index"] = 12
    result["positions"] = [
        {
            "taxel_index": 12,
            "row": 2,
            "column": 2,
            "status": "measured",
            "tool_position_m": [0.1, 0.2, 0.292],
            "safe_approach_position_m": [0.1, 0.2, 0.3],
            "tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "probe_start_z_m": 0.295,
        }
    ]
    task = FullTaxelCalibrationTask(output_directory=None)
    loaded, _message = task.load_position_map(result)
    assert loaded
    host = _Host(_RobotApi(), task)

    assert not task.on_start(host)
    assert "live 5x5" in host.messages[-1]


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


def test_experiment_outputs_default_to_the_experiment_resource_folder():
    assert CalibrationTask().output_directory.parent.name == "experiment"
    assert CalibrationTask().output_directory.name == "calibration_recordings"
    assert TaxelPositionMappingTask().output_directory.parent.name == "experiment"
    assert TaxelPositionMappingTask().output_directory.name == "taxel_position_maps"


def test_calibration_operator_defaults():
    task = CalibrationTask(storage_path=None, output_directory=None)

    assert task.taxel_index == 35
    assert task.signal_field == "diff_ave"
    assert np.isclose(task.approach_speed_m_s, 0.0001)
    assert np.isclose(task.contact_threshold_g, 500.0)
    assert task.max_travel_m == 0.03
    assert task.timeout_sec == 600.0
    assert task.sensor_baseline_frames == 100
    assert task.recovery_observation_sec == 5.0


def test_calibration_folder_name_contains_time_and_all_seven_settings():
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.active_sensor_label = "Testing Sensor 1"
    task.active_sensor_key_name = "sensor:testing_5x5_1"
    task.protocol = "proximity_profile"
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
        "sensor-Testing-Sensor-1_protocol-proximity-profile_"
        "taxel-35_signal-diff_ave_speed-0p1mmps_threshold-600g_"
        "travel-35mm_timeout-120s_reps-4_probe-conductive"
    )
    task.probe_material_note = "copper foil"
    assert task._experiment_folder_name("20260924_103015_123456").endswith(
        "_probe-conductive-copper-foil"
    )
    task.probe_material = "other"
    task.probe_material_note = ""
    assert task._experiment_folder_name("20260924_103015_123456").endswith("_probe-other")


def test_insulative_full_taxel_keeps_the_far_step_until_force_contact():
    task = FullTaxelCalibrationTask(storage_path=None, output_directory=None)
    host = _Host(_RobotApi(), task)
    task.probe_material = "insulative"
    task._step_mode = "far"
    task._step_state = "dwell"
    task._step_zeroed = True
    task._step_contact_threshold_g = 0.5
    task._step_signal_centre = 0.0
    task._step_slow_threshold = 1.0
    task._step_raw_forces = [0.0]
    task._far_rest_raw_g = []
    task.signal_field = "diff_ave"
    task._samples = [{"step_state": "dwell", "diff_ave": -30.0}] * 7
    task._tick_step = lambda *_args, **_kwargs: 0.1
    task._next_approach_step = lambda *_args, **_kwargs: None

    task._tick_stepwise_approach(host, time.monotonic(), [0.1, 0.2, 0.2], 0.1)

    assert task._step_mode == "far"
    task._tick_step = lambda *_args, **_kwargs: 0.6
    task._tick_stepwise_approach(host, time.monotonic(), [0.1, 0.2, 0.2], 0.6)
    assert task._step_mode == "loading"
    single = CalibrationTask(storage_path=None, output_directory=None)
    single.probe_material = "insulative"
    assert not single._insulative_full_taxel_approach()


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
    ) == 400


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
    assert task._phase == "contact_hold"
    assert api.velocity_commands[-1][0] == [0.0, 0.0, 0.0]
    stops_before_return = api.velocity_stops
    task.on_tick(host)
    assert task._phase == "returning"
    assert api.velocity_stops > stops_before_return  # Velocity mode left only at rest.
    assert np.allclose(host.calibration_travels[-1], (1.0, 1.0))
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
    assert np.allclose(host.calibration_travels[-1], (0.0, 1.0))
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


def _profile_task(tmp_path):
    task = CalibrationTask(storage_path=None, output_directory=tmp_path)
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.return_settle_sec = 0.0
    task.PROFILE_SURFACE_SETTLE_SEC = 0.0
    task.velocity_rest_sec = 0.0
    task.protocol = "proximity_profile"
    task.profile_gaps_mm = [20.0, 8.0, 4.0, 2.0, 1.0]
    task.profile_sweeps = 2
    task.profile_frames_per_level = 10
    task.profile_settle_sec = 0.0
    task.profile_reference_every = 2
    task.profile_surface_probes = 2
    task.profile_search_step_far_mm = 1.0
    task.PROFILE_STEP_SETTLE_SEC = 0.0
    task.PROFILE_FORCE_LAG_SEC = 0.0
    task.FORCE_ZERO_WINDOW_SEC = 0.0
    task.max_travel_m = 0.05
    task.timeout_sec = 7200.0
    return task


def _run_simulated_profile(task, api, host, surface_z, limit=20000, force_offset_g=None):
    data = host.ui_ros.sensor_functions._data
    rng = np.random.default_rng(0)
    forces_in_windows = []
    for tick in range(limit):
        api.step(0.05)
        gap = float(api.tool_position[2]) - surface_z
        level = -400.0 / (1.0 + (max(gap, 0.0) / 0.0005) ** 2)
        data.diffDataAve[:, :] = 0.3 * level + rng.normal(0.0, 1.0, size=(8, 10))
        data.diffDataAve[3, 4] = level + rng.normal(0.0, 1.0)
        data.frame_sequence += 1
        data.frame_monotonic_sec = time.monotonic()
        force_g = 100.0 * max(0.0, -gap) * 1000.0
        if force_offset_g is not None:
            force_g += force_offset_g(tick)
        host.ui_ros._force_meter_last_raw_newtons = force_g * CalibrationTask.NEWTONS_PER_GRAM
        host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
        task.on_tick(host)
        if task._profile_kind in {"level", "reference"}:
            forces_in_windows.append(force_g)
        if not task._test_active:
            break
    return forces_in_windows


def test_proximity_profile_measures_surface_levels_and_drift_references(tmp_path):
    api = _SimRobotApi()
    task = _profile_task(tmp_path)
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    start_z = float(api.tool_position[2])
    surface_z = start_z - 0.025

    assert task.on_start(host)
    forces = _run_simulated_profile(task, api, host, surface_z)

    assert not task._test_active
    assert task._stop_reason == "profile_complete", host.messages[-5:]
    assert max(forces) < 1.0  # Never touched during stationary windows.
    assert abs(float(api.tool_position[2]) - start_z) < 0.0002
    profile = task.last_result["proximity_profile"]
    assert abs(profile["surface"]["surface_z_m"] - surface_z) < 0.00005
    assert profile["surface"]["final_shift_mm"] is not None
    assert [row["target_gap_mm"] for row in profile["profile"]] == [1.0, 2.0, 4.0, 8.0]
    assert all(row["n"] == 2 for row in profile["profile"])
    for row in profile["profile"]:
        assert abs(row["gap_mm"] - row["target_gap_mm"]) < 0.05
    assert 0.5 < profile["single_frame_sigma"] < 1.5
    three = profile["detection"]["3_sigma"]
    assert three["status"] == "interpolated" and 4.0 < three["distance_mm"] < 8.0
    assert profile["hysteresis"]["matched_gaps"] == 4
    assert all(row["dominance_median"] > 2.0 for row in profile["profile"] if row["gap_mm"] <= 4.0)
    files = task.last_result["proximity_profile_files"]
    assert Path(files["png"]).exists() and Path(files["profile_csv"]).exists()
    assert host.calibration_results == []
    assert host.proximity_profile_results[-1]["proximity_profile"] is profile

    rows = task.last_result["rows"]
    windows = [row for row in rows if row.get("profile_kind") in {"level", "reference"}]
    moves = [row for row in rows if row.get("profile_kind") == "move"]
    assert all("taxel_079_diff_ave" in row for row in windows)
    assert moves and all("taxel_079_diff_ave" not in row for row in moves)
    assert task.last_result["summary"]["profile_detection_3sigma_mm"] == three["distance_mm"]
    assert any("Proximity signal dropped" in message for message in host.messages)
    line_moves = [sent for sent in api.sent if len(sent) == 3 and sent[2].get("motion_type") == 4]
    assert len(line_moves) > 10
    assert all(abs(move[0][0] - 0.1) < 1e-9 and abs(move[0][1] - 0.2) < 1e-9 for move in line_moves)
    # The surface search and probes are position steps; velocity mode never moves down.
    assert not any(command[0][2] < 0 for command in api.velocity_commands)
    steps = np.diff([move[0][2] for move in line_moves])
    assert np.isclose(steps, -0.00001).sum() >= 10  # 0.01 mm light-contact probe steps.
    zeros = task.last_result["measurement_quality"]["proximity_profile_settings"]["force_zeros"]
    assert [zero["purpose"] for zero in zeros] == [
        "near surface search", "light-contact probe 1", "light-contact probe 2",
        "final light-contact probe 1", "final light-contact probe 2",
    ]
    assert not any(row.get("profile_kind") == "surface_probe" and row.get("phase") == "profile_zero" for row in rows)


def test_force_zero_waits_for_a_steady_reading_and_raises_the_threshold_if_it_never_settles():
    task = CalibrationTask(storage_path=None, output_directory=None)
    task._force_baseline_n = 0.7 * task.NEWTONS_PER_GRAM

    def poll(start, readings):
        task._start_force_zero(start, "light-contact probe 1")
        for tick in range(1, 1000):
            now = start + 0.1 * tick
            task._force_sample_source_at = now
            task._force_raw_g_now = readings[tick % len(readings)]
            zero = task._poll_force_zero(now)
            if zero is not None:
                return zero
        raise AssertionError("no zero")

    swinging = poll(0.0, [0.0, 0.4])  # A moving cable never settles.
    assert not swinging["stable"] and swinging["wait_s"] >= task.FORCE_ZERO_MAX_WAIT_SEC
    assert swinging["threshold_g"] >= 0.8
    assert "check the force-meter cable" in task._force_zero_message(swinging)

    steady = poll(100.0, [0.0, 0.1])  # One count of flicker is steady.
    assert steady["stable"] and steady["wait_s"] == pytest.approx(task.FORCE_ZERO_WINDOW_SEC, abs=0.11)
    assert steady["threshold_g"] == pytest.approx(0.5)
    assert task._force_baseline_n / task.NEWTONS_PER_GRAM == pytest.approx(steady["zero_g"])
    assert [zero["purpose"] for zero in task._force_zero_log] == ["light-contact probe 1"] * 2


def test_proximity_profile_surface_ignores_a_decaying_force_offset(tmp_path):
    api = _SimRobotApi()
    task = _profile_task(tmp_path)
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    surface_z = float(api.tool_position[2]) - 0.025
    assert task.on_start(host)

    # 0.7 g high at the start pose (recovering from an earlier press), zero after the far search.
    _run_simulated_profile(task, api, host, surface_z, force_offset_g=lambda tick: 0.7 * max(0.0, 1.0 - tick / 150.0))

    assert task._stop_reason == "profile_complete", host.messages[-5:]
    surface = task.last_result["proximity_profile"]["surface"]
    # 0.5 g at 100 g/mm is 5 µm deep; a 0.7 g zero error would put it in the air or 7 µm deeper.
    assert abs(surface["surface_z_m"] - (surface_z - 0.000005)) < 0.000004
    assert all(abs(probe["force_zero_g"]) < 0.05 for probe in surface["probes"])
    first_contact = next(m for m in host.messages if m.startswith("Light contact") and "starting" in m)
    assert float(first_contact.split(" at Z ")[1].split(" mm")[0]) < surface_z * 1000.0 + 0.01


def test_stepwise_calibration_loads_unloads_and_measures_only_at_rest(tmp_path):
    api = _SimRobotApi()
    task = CalibrationTask(storage_path=None, output_directory=tmp_path)
    host = _Host(api, task)
    task.motion_mode = "stepwise"
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.return_settle_sec = 0.0
    task.contact_threshold_g = 50.0
    task.step_far_mm = 1.0
    task.max_travel_m = 0.05
    task.STEP_SETTLE_SEC = 0.0
    task.STEP_FORCE_LAG_SEC = 0.0
    task.FORCE_ZERO_WINDOW_SEC = 0.0
    assert task.capture_initial_position(host)
    start_z = float(api.tool_position[2])
    assert task.on_start(host)
    task.step_dwell_sec = 0.0  # Validated at 0.5 s; the simulation does not wait in real time.

    _run_simulated_profile(task, api, host, start_z - 0.010)

    assert not task._test_active
    result = task.last_result
    assert result["stop_reason"] == "contact_threshold"
    assert result["batch_status"] == "complete"
    assert not any(command[0][2] < 0 for command in api.velocity_commands)
    assert abs(float(api.tool_position[2]) - start_z) < 0.0002
    rows = result["rows"]
    modes = {row.get("step_mode") for row in rows if row.get("step_state") == "dwell"}
    assert {"far", "near", "loading", "unloading"} <= modes
    loading_steps = [row for row in rows if row.get("step_mode") == "loading" and row.get("step_state") == "dwell"]
    assert len({row["step_index"] for row in loading_steps}) >= 10
    assert result["analysis"]["loading_samples"] == "at_rest_steps"
    assert result["analysis"]["force_signal_fit"] is not None
    motion = result["measurement_quality"]["motion"]
    assert motion["mode"] == "stepwise" and motion["measured_loading_rate_mm_s"] > 0
    assert any("Light contact" in message for message in host.messages)
    assert [zero["purpose"] for zero in motion["force_zeros"]] == ["near approach"]
    assert motion["contact_threshold_g"] == pytest.approx(0.5)


def test_trial_does_not_start_when_it_would_not_fit_in_free_memory(monkeypatch):
    from phd.ui import experiment_tasks as module
    task = CalibrationTask(storage_path=None, output_directory=None)
    host = _Host(_RobotApi(), task)
    task._test_active = True
    task._trial_memory_bytes = 3 * module.GIB
    monkeypatch.setattr(module, "available_memory_bytes", lambda: 4 * module.GIB)

    task._begin_trial(host, 10.0, [0.1, 0.2, 0.3])

    assert task._stop_reason == "low_memory"
    assert task._phase != "force_baseline"
    assert "4.0 GB of RAM is free" in host.stops[-1][1]

    task = CalibrationTask(storage_path=None, output_directory=None)
    host = _Host(_RobotApi(), task)
    task._test_active = True
    task._trial_memory_bytes = 3 * module.GIB
    monkeypatch.setattr(module, "available_memory_bytes", lambda: 5 * module.GIB)
    task._begin_trial(host, 10.0, [0.1, 0.2, 0.3])
    assert task._phase == "force_baseline" and not host.stops


def test_active_trial_stops_in_place_when_free_memory_runs_out(monkeypatch):
    from phd.ui import experiment_tasks as module
    task = CalibrationTask(storage_path=None, output_directory=None)
    host = _Host(_RobotApi(), task)
    task._test_active = True
    free = [5 * module.GIB]
    monkeypatch.setattr(module, "available_memory_bytes", lambda: free[0])

    assert task._memory_ok(host, 100.0)
    free[0] = module.GIB // 2
    assert task._memory_ok(host, 101.0)  # Read at most every 2 s.
    assert not task._memory_ok(host, 102.5)
    assert task._stop_reason == "low_memory"
    assert "no automatic retract" in host.stops[-1][1]


def test_stepwise_contact_uses_a_zero_re_taken_near_the_surface(tmp_path):
    api = _SimRobotApi()
    task = CalibrationTask(storage_path=None, output_directory=tmp_path)
    host = _Host(api, task)
    task.motion_mode = "stepwise"
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.return_settle_sec = 0.0
    task.contact_threshold_g = 50.0
    task.step_far_mm = 1.0
    task.max_travel_m = 0.05
    task.STEP_SETTLE_SEC = 0.0
    task.STEP_FORCE_LAG_SEC = 0.0
    task.FORCE_ZERO_WINDOW_SEC = 0.0
    assert task.capture_initial_position(host)
    surface_z = float(api.tool_position[2]) - 0.010
    assert task.on_start(host)
    task.step_dwell_sec = 0.0

    # The meter starts 0.7 g high (recovering from an earlier press) and decays to zero.
    _run_simulated_profile(task, api, host, surface_z, force_offset_g=lambda tick: 0.7 * max(0.0, 1.0 - tick / 150.0))

    result = task.last_result
    assert result["stop_reason"] == "contact_threshold", host.messages[-5:]
    contact = [row for row in result["rows"] if row.get("step_mode") == "loading"][0]
    assert float(contact["tool_z_m"]) < surface_z + 0.00002  # Not a drift-triggered contact in the air.
    zero = result["measurement_quality"]["motion"]["force_zeros"][0]
    # The far steps followed the decay, so the near zero only corrected the last few steps of it.
    assert zero["stable"] and zero["zero_g"] < 0.6 and abs(zero["previous_zero_g"] - zero["zero_g"]) < 0.1


def test_stepwise_full_taxel_calibration_visits_each_taxel_without_velocity_descents(tmp_path):
    api = _SimRobotApi()
    task = FullTaxelCalibrationTask(output_directory=tmp_path)
    loaded, _message = task.load_position_map(
        _saved_full_calibration_map(), tmp_path / "taxel_position_map.json"
    )
    assert loaded
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.return_settle_sec = 0.0
    task.move_settle_sec = 0.0
    task.contact_threshold_g = 30.0
    task.step_far_mm = 1.0
    task.max_travel_m = 0.05
    task.STEP_SETTLE_SEC = 0.0
    task.STEP_FORCE_LAG_SEC = 0.0
    task.FORCE_ZERO_WINDOW_SEC = 0.0
    host = _Host(api, task)
    task.motion_mode = "stepwise"
    assert task.on_start(host)
    task.step_dwell_sec = 0.0
    surface_z = 0.285
    sensor = host.ui_ros.sensor_functions
    data = sensor._data

    class _LiveForceUiRos(type(host.ui_ros)):
        # Device threads keep stamping samples while a long trial is saved.
        _force_meter_last_sample_monotonic = property(lambda self: time.monotonic(), lambda self, value: None)

    host.ui_ros.__class__ = _LiveForceUiRos
    for _ in range(40000):
        api.step(0.05)
        gap = float(api.tool_position[2]) - surface_z
        data.diffDataAve[:, :] = -400.0 / (1.0 + (max(gap, 0.0) / 0.0005) ** 2)
        data.frame_sequence += 1
        force_g = 100.0 * max(0.0, -gap) * 1000.0
        host.ui_ros._force_meter_last_raw_newtons = force_g * CalibrationTask.NEWTONS_PER_GRAM
        if sensor._sensor_calibration_in_progress:
            sensor._sensor_calibration_in_progress = False
            sensor.is_connected = True
        task.on_tick(host)
        if not task._test_active:
            break

    assert task._completed_taxels == [35, 43], host.messages[-5:]
    assert not any(command[0][2] < 0 for command in api.velocity_commands)
    for taxel in (35, 43):
        trials = task._trials_by_taxel[taxel]
        assert trials and trials[-1]["stop_reason"] == "contact_threshold"
        assert trials[-1]["analysis"]["loading_samples"] == "at_rest_steps"


def test_tiny_step_that_lands_within_robot_repeatability_is_accepted_after_a_second():
    task = CalibrationTask(storage_path=None, output_directory=None)
    host = _Host(_RobotApi(), task)
    task._step_state = "moving"
    task._step_target = np.asarray([0.1, 0.2, 0.30000])
    task._step_size_m = 0.00001
    task._step_started_at = 10.0
    landed = [0.1, 0.2, 0.300006]  # 6 µm off a 10 µm step, as seen on the TM5.

    task._tick_step(host, 10.2, landed, 0.0)
    task._tick_step(host, 10.6, landed, 0.0)
    assert task._step_state == "moving"  # Still, but not yet past the relaxed wait.
    task._tick_step(host, 11.1, landed, 0.0)
    assert task._step_state == "settling" and not host.stops

    profile = _profile_task(None)
    profile_host = _Host(_RobotApi(), profile)
    profile._profile_target_z_m = 0.30000
    profile._profile_move_tolerance_m = 0.000005
    profile._profile_move_hold_sec = 0.3
    profile._profile_move_timeout_sec = 30.0
    profile._enter_profile_stage("profile_move", 10.0)
    assert not profile._tick_profile_move(profile_host, 10.2, 0.300006)
    assert not profile._tick_profile_move(profile_host, 10.6, 0.300006)
    assert profile._tick_profile_move(profile_host, 11.1, 0.300006)


def test_stepwise_speed_comparison_rejects_unattainable_rates():
    task = CalibrationTask(storage_path=None, output_directory=None)
    host = _Host(_RobotApi(), task)
    task.motion_mode = "stepwise"
    task.protocol = "speed_sweep"
    task.speed_schedule_mm_s = [0.05, 0.1]

    assert not task._validate_measurement_settings(host)
    assert "fastest reliable rate" in host.messages[-1]

    task.speed_schedule_mm_s = [0.005, 0.01, 0.02]
    assert task._validate_measurement_settings(host)


def test_proximity_profile_step_check_uses_fresh_force_after_the_lag(tmp_path):
    task = _profile_task(tmp_path)
    api = _SimRobotApi()
    host = _Host(api, task)
    task.PROFILE_FORCE_LAG_SEC = 0.25
    task._start_tool_position = [0.1, 0.2, 0.3]
    task.initial_position = {"tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
    task._profile_search_mode = "probe"
    task._profile_step_target_z = 0.2995
    task._enter_profile_stage("profile_check", 10.0)
    task._profile_check_rows = len(task._samples)
    task._samples.append({"phase": "profile_check"})

    task._force_sample_source_at = 1.0
    task._tick_profile_check(host, 10.1, 0.2995, 50.0)  # Inside the force-meter lag: ignored.
    task._force_sample_source_at = 2.0
    task._tick_profile_check(host, 10.3, 0.2995, 0.2)
    assert task._phase == "profile_check" and not api.sent
    task._force_sample_source_at = 3.0
    task._tick_profile_check(host, 10.4, 0.2995, 0.3)

    assert task._phase == "profile_move"
    assert abs(api.sent[-1][0][2] - (0.2995 - 0.00001)) < 1e-12
    assert api.sent[-1][2]["motion_type"] == 4 and api.sent[-1][2]["fine_goal"]


def test_return_that_moves_down_instead_of_up_is_stopped():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    host = _Host(api, task)
    task._test_active = True
    task._phase = "returning"
    task._return_purpose = "complete"
    task._start_tool_position = [0.1, 0.2, 0.31]
    task._return_start_z = 0.30
    api.tool_position = [0.1, 0.2, 0.2998]

    task._tick_vertical_return_to_initial(host, time.monotonic(), api, "complete")

    assert task._stop_reason == "velocity_reversal"
    assert "moved down" in host.stops[-1][1]


def test_proximity_profile_rejects_full_taxel_and_short_timeouts(tmp_path):
    task = _profile_task(tmp_path)
    host = _Host(_RobotApi(), task)
    task.timeout_sec = 60.0
    assert not task._validate_measurement_settings(host)
    assert "raise Timeout" in host.messages[-1]

    full = FullTaxelCalibrationTask(output_directory=None)
    full.protocol = "proximity_profile"
    full_host = _Host(_RobotApi(), full)
    assert not full._validate_measurement_settings(full_host)
    assert "Single taxel" in full_host.messages[-1]


def test_proximity_profile_timeout_check_uses_the_reported_project_speed(tmp_path):
    api = _RobotApi()
    task = _profile_task(tmp_path)
    task.timeout_sec = 1200.0
    host = _Host(api, task)
    assert task._validate_measurement_settings(host)
    assert task._profile_project_speed_percent == 100.0

    api.project_speed_percent = 1.0
    assert not task._validate_measurement_settings(host)
    assert "1% TM project speed" in host.messages[-1] and "raise Timeout" in host.messages[-1]


def test_proximity_profile_estimate_matches_the_measured_22_oct_run():
    task = CalibrationTask(storage_path=None, output_directory=None)
    assert task._profile_speed_mm_s() == (2.5, 25.0)  # Before the robot reports a speed.
    task.profile_gaps_mm = [
        100, 70, 50, 40, 30, 25, 20, 16, 13, 10, 8, 6, 5, 4, 3, 2.5, 2, 1.5, 1.2, 1, .8, .6, .4,
    ]
    task.profile_sweeps = 3
    task.profile_reference_every = 3

    # That run took 40.8 min from the first surface probe to the end, at 25% project speed.
    estimate = task.estimate_profile_duration_sec(25.0)
    assert 40.8 * 60.0 <= estimate <= 1.25 * 40.8 * 60.0
    assert task.estimate_profile_duration_sec(100.0) < estimate


def test_proximity_profile_plan_alternates_and_brackets_with_references():
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.profile_gaps_mm = [20.0, 8.0, 4.0, 2.0, 1.0]
    task.profile_sweeps = 2
    task.profile_reference_every = 2

    plan = [(step["kind"][0], step["gap_mm"], step["direction"]) for step in task._build_profile_plan()]

    assert plan == [
        ("r", 20.0, ""), ("l", 8.0, "down"), ("l", 4.0, "down"), ("r", 20.0, ""),
        ("l", 2.0, "down"), ("l", 1.0, "down"), ("r", 20.0, ""),
        ("l", 1.0, "up"), ("l", 2.0, "up"), ("r", 20.0, ""),
        ("l", 4.0, "up"), ("l", 8.0, "up"), ("r", 20.0, ""),
    ]


def test_calibration_far_field_speed_only_above_previous_contact_height(tmp_path):
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=tmp_path)
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.return_settle_sec = 0.0
    task.contact_threshold_n = 0.1
    task.far_field_speed_mm_s = 1.0
    task.slow_zone_mm = 5.0
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    assert task.on_start(host)
    start_z = api.tool_position[2]
    task._contact_z_by_taxel[35] = start_z - 0.010

    _advance_initial_return(task, host)
    task.on_tick(host)
    assert api.velocity_commands[-1][0] == [0.0, 0.0, -0.001]

    api.tool_position[2] = start_z - 0.0051
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    assert api.velocity_commands[-1][0] == [0.0, 0.0, -0.0001]

    # The slow zone is latched for the rest of the approach.
    api.tool_position[2] = start_z
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    assert api.velocity_commands[-1][0] == [0.0, 0.0, -0.0001]
    assert any(row["approach_zone"] == "far_field" for row in task._samples)


def test_calibration_far_field_is_off_without_contact_memory_or_with_force():
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.far_field_speed_mm_s = 1.0
    task._trial_speed_m_s = 0.0001
    task._expected_contact_z_m = None
    assert not task._far_field_allowed([0.0, 0.0, 0.2], 0.0)

    task._slow_zone_entered = False
    task._expected_contact_z_m = 0.1
    assert task._far_field_allowed([0.0, 0.0, 0.2], 0.0)
    assert not task._far_field_allowed([0.0, 0.0, 0.2], 1.5)
    assert not task._far_field_allowed([0.0, 0.0, 0.2], 0.0)


def test_calibration_target_messages_explain_the_three_results():
    messages = CalibrationTask._calibration_target_messages({
        "proximity_range": {"status": "start_inside_range", "lower_bound_mm": 10.0},
        "turning_point": {
            "detected": True, "force_g": 12.0, "force_low_g": 10.0,
            "force_high_g": 14.0, "after_contact_mm": 0.1,
        },
        "baseline_return": {"reached": True, "force_g": 400.0},
    })

    assert "at least 10.00 mm" in messages[0] and "Raise the start pose" in messages[0]
    assert "12.0 g" in messages[1]
    assert "400.0 g" in messages[2]


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

    assert np.isclose(task.contact_threshold_n, 800.0 * 0.00980665)
    assert np.isclose(task.contact_threshold_g, 800.0)

    task.contact_threshold_g = 2500.0

    assert np.isclose(task.contact_threshold_g, 800.0)


def test_calibration_stale_force_during_return_stops_then_resumes_return():
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
    task.on_tick(host)
    assert task._phase == "returning"

    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic() - 5.0
    task.on_tick(host)

    assert task._phase == "paused_force_meter"
    assert not host.stops
    task.force_resume_stable_sec = 0
    task.force_resume_min_samples = 1
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    assert task._phase == "returning"

    api.tool_position[2] += 0.001
    task.on_tick(host)
    task.on_tick(host)
    _settle_post_save_feedback(task, host)
    assert task._phase == "recalibrating"
    assert len(task._batch_results) == 1

    sensor = host.ui_ros.sensor_functions
    sensor._sensor_calibration_in_progress = False
    sensor.is_connected = True
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
    task._trial_speed_m_s = 0.00001
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


def test_calibration_refuses_unlimited_time():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.return_settle_sec = 0.0
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.timeout_sec = 0.0
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    assert not task.on_start(host)
    assert not api.velocity_commands
    assert not api.sent


def test_calibration_refuses_unlimited_travel():
    api = _RobotApi()
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.return_settle_sec = 0.0
    task.force_baseline_duration_sec = 0.0
    task.force_baseline_min_samples = 1
    task.max_travel_m = 0.0
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    assert not task.on_start(host)
    assert not api.velocity_commands
    assert not api.sent


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
    _settle_post_save_feedback(task, host)
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
