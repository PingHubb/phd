import copy
import json
import time
from pathlib import Path

import numpy as np
import pytest

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
    assert task.start_button_label == "Select Recording"
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
    def __init__(self):
        self.current = np.zeros(6, dtype=float)
        self.tool_position = [0.1, 0.2, 0.3]
        self.sent = []
        self.scripts = []
        self.velocity_commands = []
        self.velocity_stops = 0
        self.accept_velocity = True
        self.project_speed_percent = 100.0

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

    def send_positions_tool_position(self, positions, quaternion, **kwargs):
        self.sent.append(
            (list(positions), list(quaternion), dict(kwargs))
        )
        # Arrive instantly so feedback matches the commanded target.
        self.tool_position = list(positions)
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
    assert task.centre_scan_max_extra_descent_m == pytest.approx(0.005)
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
    assert len(task._relocation_source_positions) == 25
    assert task._move_action == "anchor_safe_arrived"
    assert task._relocation_source_path == (tmp_path / "old.json").resolve()
    task.on_stop(host, show_result=False)


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


def _started_mapping(tmp_path):
    api = _RobotApi()
    task = TaxelPositionMappingTask(
        storage_path=tmp_path / "anchor.json", output_directory=None
    )
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
    api, task, host = _started_mapping(tmp_path)
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
    # responds, well past the 8 mm limit that applies to later probes.
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
    assert api.velocity_commands[-1][0][2] == pytest.approx(-task.probe_speed_m_s)
    assert task._probe_max_descent_used_m == pytest.approx(task.max_probe_descent_m)

    # The working baseline already contains the -100-count level.
    data.diffDataAve[3, 4] = -160.0
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


def test_taxel_mapping_uses_strongest_level_at_post_onset_safety_limit(
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
    assert task._probe_scan_pose[2] == pytest.approx(0.2985)
    assert task._probe_scan_signal == pytest.approx(80.0)
    assert task._probe_scan_target_signal == pytest.approx(100.0)
    assert task._probe_scan_target_reached is False
    assert task._scan_kind == "local"
    assert task._phase == "moving"


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
    stronger[36] = 90.0
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
    assert [
        row["context"] for row in result["sensor_calibration_history"]
    ] == ["fresh_start", "resume"]
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
    assert task._collection_action == "retry_anchor_stability"

    stable = np.zeros(80, dtype=float)
    for sequence in range(task.baseline_frame_count):
        task._tick_collection(
            host,
            (100 + sequence, stable),
            force_n=0.0,
        )

    assert task._pending_recovery is None
    assert task._attempt_counts[36] == 2
    assert task._current_taxel == 36
    assert api.sent[-1][0] == pytest.approx([0.104, 0.201, 0.3])


def test_taxel_mapping_first_probe_failure_retries_from_the_anchor(tmp_path):
    api, task, host = _started_mapping(tmp_path)
    assert _tick_until(task, host, lambda: task._phase == "probing")
    api.tool_position = [0.1, 0.2, 0.3 - task.reference_max_descent_m - 0.0001]
    _tick(task, host, 2 + task.probe_sample_frames)

    assert task._pending_recovery["kind"] == "reference"
    assert task._pending_recovery["retry"] is True
    assert api.sent[-1][0] == pytest.approx([0.1, 0.2, 0.3])
    assert _tick_until(task, host, lambda: task._phase == "probing")
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

    assert host.stops
    assert "failed twice" in host.stops[-1][1]
    assert task._active is False
    assert task.last_result["stop_reason"] == "mapping_target_failed"


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
    ] == [0.1, 0.2, 0.295]


def test_full_taxel_recording_keeps_only_the_active_taxel_signal():
    task = FullTaxelCalibrationTask(output_directory=None)
    task.taxel_index = 35
    host = _Host(_RobotApi(), task)
    data = host.ui_ros.sensor_functions._data
    data.diffDataAve[3, 4] = -17.0
    data.diffDataAve[3, 5] = 91.0

    taxel = task._read_taxel(host)

    assert taxel["diff_ave"] == -17.0
    assert taxel["taxel_035_diff_ave"] == -17.0
    assert "taxel_043_diff_ave" not in taxel
    assert [key for key in taxel if key.startswith("taxel_")] == [
        "taxel_035_diff_ave"
    ]


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
    (0.31, [[0.104, 0.2, 0.31], [0.104, 0.2, 0.3]]),
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


def test_navigation_refuses_stale_readings_and_other_experiments_without_commands():
    task = _navigation_task()
    api = _RobotApi()
    host = _Host(api, task=task)
    host.ui_ros._force_meter_last_sample_monotonic -= 5
    assert not task.on_start(host)
    assert "force transmitter" in host.messages[-1]
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    host._sidebar_active_task = object()
    assert not task.on_start(host)
    assert "current experiment" in host.messages[-1]
    assert not api.sent


def _remember_full_taxel_home(task, position=None):
    pose = {
        "saved_at": "",
        "joints_rad": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "tool_position_m": list(position or [0.1, 0.2, 0.3]),
        "tool_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    for key in ("layout:8x10", "layout:5x5", "layout:unknown"):
        task._position_profiles[key] = copy.deepcopy(pose)
    task.initial_position = copy.deepcopy(pose)


def _settle_full_taxel_route(task, host, limit=40):
    for _ in range(limit):
        if task._phase != "full_moving":
            return
        host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
        task.on_tick(host)
    raise AssertionError("full-taxel route did not settle")


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
    assert task._phase == "returning"
    api.tool_position[2] += 0.001
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    task.on_tick(host)
    task.on_tick(host)


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
    _remember_full_taxel_home(task)
    host = _Host(api, task)
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
        assert Path(result["experiment_directory"]).name == f"taxel_{expected:02d}"
        row = result["rows"][-1]
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
    _remember_full_taxel_home(task)
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


def test_full_taxel_initial_position_is_stored_apart_from_single_taxel(tmp_path):
    single = CalibrationTask(
        storage_path=tmp_path / "single.json", output_directory=None
    )
    full = FullTaxelCalibrationTask(
        storage_path=tmp_path / "full.json", output_directory=None
    )
    single_host = _Host(_RobotApi())
    full_host = _Host(_RobotApi())
    single_host.ui_ros.robot_api.current = np.ones(6)
    single_host.ui_ros.robot_api.tool_position = [0.1, 0.2, 0.3]
    full_host.ui_ros.robot_api.current = np.full(6, 2.0)
    full_host.ui_ros.robot_api.tool_position = [0.4, 0.5, 0.6]

    assert single.capture_initial_position(single_host)
    assert full.capture_initial_position(full_host)

    assert single.storage_path != full.storage_path
    np.testing.assert_allclose(single.initial_position["tool_position_m"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(full.initial_position["tool_position_m"], [0.4, 0.5, 0.6])
    restored_single = CalibrationTask(
        storage_path=single.storage_path, output_directory=None
    )
    restored_full = FullTaxelCalibrationTask(
        storage_path=full.storage_path, output_directory=None
    )
    restored_single.use_connected_sensor(single_host)
    restored_full.use_connected_sensor(full_host)
    np.testing.assert_allclose(
        restored_single.initial_position["joints_rad"], np.ones(6)
    )
    np.testing.assert_allclose(
        restored_full.initial_position["tool_position_m"], [0.4, 0.5, 0.6]
    )


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
    _remember_full_taxel_home(task)

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
