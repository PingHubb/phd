"""Hardware-free regression tests for safety, provenance and characterisation."""
import copy
import time

import numpy as np
import pytest

from phd.ui.experiment_tasks import CalibrationTask, FullTaxelCalibrationTask, TaxelPositionMappingTask
from phd.ui.calibration_analysis import analyze_calibration_rows, contact_referenced_rows
from phd.ui.characterisation_analysis import (
    trial_characterisation, batch_characterisation,
    comparable_trial_results, result_condition_groups,
)
from phd.ui.experiment_quality import compare_position_maps, downward_detector
from phd.dependence.sensor_data import SensorDataBuffer
from test_experiment_tasks import _Host, _RobotApi, _advance_initial_return, _saved_full_calibration_map


def running_task(monkeypatch, task_class=CalibrationTask):
    clock = [100.0]
    monkeypatch.setattr("phd.ui.experiment_tasks.time.monotonic", lambda: clock[0])
    task = task_class(storage_path=None, output_directory=None)
    api = _RobotApi()
    host = _Host(api, task)
    task.return_settle_sec = 0
    task.force_baseline_duration_sec = 0
    task.force_baseline_min_samples = 1
    assert task.capture_initial_position(host)
    assert task.on_start(host)
    _advance_initial_return(task, host)
    return clock, task, api, host


@pytest.mark.parametrize("fault", ["sensor_age", "sensor_disconnect", "robot_age", "invalid_frame"])
def test_required_feedback_fault_stops_before_another_downward_command(monkeypatch, fault):
    clock, task, api, host = running_task(monkeypatch)
    task.on_tick(host)
    before = len(api.velocity_commands)
    sensor = host.ui_ros.sensor_functions
    if fault == "sensor_age":
        sensor._data.frame_monotonic_sec = clock[0] - 2
    elif fault == "sensor_disconnect":
        sensor.is_connected = False
    elif fault == "invalid_frame":
        sensor._data.diffDataAve[0, 0] = np.nan
    else:
        api.current_feedback_state_time = clock[0] - 2
    task.on_tick(host)
    assert host.stops
    assert len(api.velocity_commands) == before
    assert api.velocity_stops


@pytest.mark.parametrize("phase", ["approaching", "contact_hold", "returning", "recovery"])
def test_emergency_force_is_independent_of_target_and_applies_during_every_phase(monkeypatch, phase):
    clock, task, api, host = running_task(monkeypatch)
    task.on_tick(host)
    task._phase = phase
    host.ui_ros._force_meter_last_raw_newtons = 950 * task.NEWTONS_PER_GRAM
    before = len(api.velocity_commands)
    task.on_tick(host)
    assert task._stop_reason == "force_safety_limit"
    assert len(api.velocity_commands) == before


def test_baseline_needs_distinct_frames_and_time(monkeypatch):
    clock, task, api, host = running_task(monkeypatch)
    task.sensor_baseline_frames = 20
    task.sensor_baseline_duration_sec = 1
    data = host.ui_ros.sensor_functions._data
    data.frame_monotonic_sec = clock[0]
    for _ in range(30):
        task.on_tick(host)
    assert len(task._baseline_rows) == 1
    assert task._phase == "force_baseline"
    assert not api.velocity_commands
    for _ in range(22):
        clock[0] += .05
        data.frame_sequence += 1
        data.frame_monotonic_sec = clock[0]
        host.ui_ros._force_meter_last_sample_monotonic = clock[0]
        task.on_tick(host)
    assert task._phase == "approaching"
    assert len([r for r in task._samples if r["phase"] == "force_baseline"]) >= 20


def test_duplicate_frames_are_not_recorded_twice(monkeypatch):
    clock, task, api, host = running_task(monkeypatch)
    data = host.ui_ros.sensor_functions._data
    data.frame_monotonic_sec = clock[0]
    task.on_tick(host)
    before = len(task._samples)
    task.on_tick(host)
    assert len(task._samples) == before
    clock[0] += .05
    data.frame_sequence += 1
    data.frame_monotonic_sec = clock[0]
    task.on_tick(host)
    assert len(task._samples) == before + 1


def test_pause_remains_in_wall_time_and_flags_saved_trial(monkeypatch):
    clock, task, api, host = running_task(monkeypatch)
    task.on_tick(host)
    task._begin_force_pause(host, clock[0])
    clock[0] += 5
    task._resume_from_force_pause(host, clock[0])
    host.ui_ros._force_meter_last_sample_monotonic = clock[0]
    task.on_tick(host)
    sample = task._samples[-1]
    assert sample["elapsed_s"] >= 5
    assert sample["active_elapsed_s"] < 1
    task._stop_reason = "contact_threshold"
    result = task._save_result()
    assert result["measurement_quality"]["interrupted"]
    assert not result["measurement_quality"]["eligible_for_comparison"]
    assert result["measurement_quality"]["pause_events"][0]["end_monotonic_s"] == clock[0]


def test_wall_timeout_still_applies_while_force_is_paused(monkeypatch):
    clock, task, api, host = running_task(monkeypatch)
    task.on_tick(host)
    task._begin_force_pause(host, clock[0])
    clock[0] += task.timeout_sec + 1
    host.ui_ros._force_meter_last_sample_monotonic = clock[0]
    task.on_tick(host)
    assert task._stop_reason == "timeout"


@pytest.mark.parametrize("mapped", [False, True])
def test_force_resume_reissues_cancelled_initial_pose_route(monkeypatch, mapped):
    clock = [100.0]
    monkeypatch.setattr("phd.ui.experiment_tasks.time.monotonic", lambda: clock[0])
    task = CalibrationTask(storage_path=None, output_directory=None)
    api = _RobotApi()
    host = _Host(api, task)
    assert task.capture_initial_position(host)
    if mapped:
        task.initial_position.pop("joints_rad")
        task.initial_position["safe_approach_position_m"] = [0.1, 0.2, 0.3]
    assert task.on_start(host)
    assert task._phase == "returning"
    task._begin_force_pause(host, clock[0])
    before = len(api.sent)
    clock[0] += 3
    host.ui_ros._force_meter_last_sample_monotonic = clock[0]
    task._resume_from_force_pause(host, clock[0])
    assert task._phase == "returning"
    assert task._return_purpose == "start"
    assert len(api.sent) == before + 1
    assert task._pause_events[-1]["end_monotonic_s"] == clock[0]


def test_force_resume_refuses_initial_route_if_pose_feedback_is_stale(monkeypatch):
    clock, task, api, host = running_task(monkeypatch)
    task._phase = "returning"
    task._return_purpose = "start"
    task._begin_force_pause(host, clock[0])
    before = len(api.sent)
    clock[0] += 3
    host.ui_ros._force_meter_last_sample_monotonic = clock[0]
    api.current_feedback_state_time = clock[0] - 2
    task._resume_from_force_pause(host, clock[0])
    assert task._stop_reason == "robot_feedback_stale"
    assert len(api.sent) == before


def test_a_future_force_timestamp_is_not_accepted_as_live(monkeypatch):
    clock, task, api, host = running_task(monkeypatch)
    host.ui_ros._force_meter_last_sample_monotonic = clock[0] + 60
    assert task._fresh_force_reading(host, clock[0]) is None


def test_stopping_after_contact_but_before_return_does_not_mark_trial_complete(monkeypatch):
    clock, task, api, host = running_task(monkeypatch)
    task.on_tick(host)
    task._stop_reason = "contact_threshold"
    result = task._save_result()
    assert not result["measurement_quality"]["trial_completed"]
    assert not result["measurement_quality"]["eligible_for_comparison"]


def test_speed_schedule_cycles_by_whole_pass(monkeypatch):
    clock, task, api, host = running_task(monkeypatch)
    task.protocol = "speed_sweep"
    observed = []
    for number in range(1, 7):
        task._trial_number = number
        task._begin_trial(host, clock[0], api.tool_position)
        observed.append(task._trial_speed_m_s * 1000)
    assert observed == [.05, .1, .2, .05, .1, .2]
    full = FullTaxelCalibrationTask(storage_path=None, output_directory=None)
    full.protocol = "speed_sweep"
    full._pass_number = 2
    full._trial_number = 9
    full._begin_trial(host, clock[0], api.tool_position)
    assert full._trial_speed_m_s == .0001


def test_contact_approach_slows_before_target(monkeypatch):
    clock, task, api, host = running_task(monkeypatch)
    task.on_tick(host)
    host.ui_ros._force_meter_last_raw_newtons = task.contact_threshold_n * .85
    clock[0] += .05
    host.ui_ros._force_meter_last_sample_monotonic = clock[0]
    task.on_tick(host)
    assert api.velocity_commands[-1][0][2] == -task._trial_speed_m_s * .25


def test_recovery_observation_precedes_trial_completion(monkeypatch):
    clock, task, api, host = running_task(monkeypatch)
    task.on_tick(host)
    task.recovery_observation_sec = 2
    task._stop_reason = "contact_threshold"
    task._finish_returned_trial(host, clock[0], "complete")
    assert task._phase == "recovery"
    assert not task._batch_results
    clock[0] += 1
    host.ui_ros._force_meter_last_sample_monotonic = clock[0]
    task.on_tick(host)
    assert task._phase == "recovery"
    clock[0] += 1.1
    host.ui_ros._force_meter_last_sample_monotonic = clock[0]
    task.on_tick(host)
    assert len(task._batch_results) == 1


def test_fixed_baseline_skips_recalibration_only(monkeypatch):
    clock, task, api, host = running_task(monkeypatch)
    task.recalibrate_between_trials = False
    previous = task._trial_number
    task._start_sensor_recalibration(host, clock[0])
    assert host.ui_ros.sensor_functions.update_cal_calls == 0
    assert task._trial_number == previous + 1
    assert task._phase == "force_baseline"


def test_sliding_requires_explicit_preflight_and_low_target():
    task = CalibrationTask(storage_path=None, output_directory=None)
    host = _Host(_RobotApi(), task)
    task.protocol = "sliding"
    assert not task._validate_measurement_settings(host)
    task.sliding_path_confirmed = True
    task.contact_threshold_g = 100
    assert task._validate_measurement_settings(host)
    task.sliding_distance_mm = 100
    assert not task._validate_measurement_settings(host)


def test_sliding_follows_bounded_out_and_back_path(monkeypatch):
    clock, task, api, host = running_task(monkeypatch)
    task.protocol = "sliding"
    task.contact_threshold_g = 100
    task._slide_origin = np.asarray(api.tool_position).copy()
    task._phase = "sliding"
    task.velocity_rest_sec = 0.0
    task._tick_sliding(host, clock[0], api.tool_position, 100)
    assert api.velocity_commands[-1][0] == [.0001, 0, 0]
    api.tool_position[0] += .002
    task._tick_sliding(host, clock[0], api.tool_position, 100)
    # The path end streams zero velocity and reverses only once at rest.
    assert api.velocity_commands[-1][0] == [0.0, 0.0, 0.0] and not task._slide_returning
    task._tick_sliding(host, clock[0], api.tool_position, 100)
    assert task._slide_returning
    task._tick_sliding(host, clock[0], api.tool_position, 100)
    assert api.velocity_commands[-1][0][0] < 0
    api.tool_position[0] -= .002
    task._tick_sliding(host, clock[0], api.tool_position, 100)
    task._tick_sliding(host, clock[0], api.tool_position, 100)
    assert task._phase == "returning"


def test_stepwise_sliding_steps_out_and_back_at_fixed_z(monkeypatch):
    clock, task, api, host = running_task(monkeypatch)
    task.motion_mode = "stepwise"
    task.STEP_SETTLE_SEC = 0.0
    task.STEP_FORCE_LAG_SEC = 0.0
    task.step_dwell_sec = 0.0
    task.protocol = "sliding"
    task.contact_threshold_g = 100
    task.sliding_distance_mm = 0.2
    task.sliding_step_mm = 0.05
    task.sliding_speed_mm_s = 0.1
    origin = np.asarray(api.tool_position, dtype=float).copy()
    task._slide_origin = origin.copy()
    task._slide_target_m = 0.0
    task._step_state = ""
    task._phase = "sliding"
    targets = []
    for _ in range(200):
        clock[0] += 0.1
        host.ui_ros._force_meter_last_sample_monotonic = clock[0]
        host.ui_ros.sensor_functions._data.frame_monotonic_sec = clock[0]
        api.current_feedback_state_time = clock[0]
        task._force_sample_source_at = clock[0]
        task._tick_sliding(host, clock[0], api.tool_position, 100)
        if api.sent and (not targets or api.sent[-1][0] != targets[-1]):
            targets.append(list(api.sent[-1][0]))
        if task._phase == "returning":
            break

    assert task._phase == "returning"
    slide = [target for target in targets if abs(target[2] - origin[2]) < 1e-9]
    np.testing.assert_allclose(
        [round((target[0] - origin[0]) * 1000.0, 6) for target in slide],
        [0.05, 0.10, 0.15, 0.20, 0.15, 0.10, 0.05, 0.0],
    )
    assert all(abs(target[1] - origin[1]) < 1e-9 for target in slide)
    assert not api.velocity_commands  # No velocity streaming at all.


@pytest.mark.parametrize("force", [0.0, 151.0])
def test_sliding_aborts_on_lost_contact_or_rising_normal_load(monkeypatch, force):
    clock, task, api, host = running_task(monkeypatch)
    task.protocol = "sliding"
    task.contact_threshold_g = 100
    task._slide_origin = np.asarray(api.tool_position).copy()
    task._phase = "sliding"
    task._tick_sliding(host, clock[0], api.tool_position, force)
    assert task._stop_reason == "sliding_guard"


def synthetic_dataset(name="a", speed=.0001):
    rows = []
    for i in range(180):
        phase = "force_baseline" if i < 30 else "approaching" if i < 130 else "contact_hold" if i < 140 else "recovery"
        step = max(0, i - 30)
        signal = -10 * max(0, step - 10) if phase == "approaching" else -900 if phase == "contact_hold" else -3 if phase == "recovery" else 0
        force = 10 * max(0, step - 50) if phase == "approaching" else 500 - (i - 130) if phase == "contact_hold" else 0
        rows.append({
            "phase": phase, "elapsed_s": i * .05, "frame_sequence": i + 1,
            "sensor_monotonic_s": 1000 + i * .05, "force_monotonic_s": 1000 + i * .05,
            "diff_ave": signal, "force_delta_abs_g": force, "force_delta_g": force,
            "travel_down_mm": step * .1 if phase == "approaching" else 0,
            "taxel_000_diff_ave": signal, "taxel_001_diff_ave": signal * .1,
        })
    return {
        "name": name, "source_path": f"/recordings/{name}.csv", "rows": rows,
        "signal_field": "diff_ave",
        "metadata": {
            "taxel_index": 0, "sensor_shape": [2, 2], "approach_speed_m_s": speed,
            "stop_reason": "contact_threshold", "contact_threshold_g": 500,
            "measurement_quality": {"interrupted": False, "eligible_for_comparison": True, "sensor_identity": "test", "protocol": "standard"},
        },
    }


def test_shared_proximity_rule_uses_stationary_baseline_and_decrease_only():
    dataset = synthetic_dataset()
    result = analyze_calibration_rows(dataset["rows"])
    assert result["baseline_source"] == "stationary_unique_frames"
    assert result["signal_onset"]["distance_mm"] < result["contact_onset"]["distance_mm"]
    assert result["signal_change_threshold"] == downward_detector([0] * 30)["margin"]
    positive = [{**r, "diff_ave": -r["diff_ave"]} for r in dataset["rows"]]
    assert analyze_calibration_rows(positive)["signal_onset"] is None
    assert contact_referenced_rows(dataset["rows"])[-1]["contact_relative_travel_mm"] < 0


def test_hold_does_not_bias_loading_fit():
    dataset = synthetic_dataset()
    before = analyze_calibration_rows(dataset["rows"])["force_signal_fit"]["slope"]
    for row in dataset["rows"]:
        if row["phase"] == "contact_hold":
            row["diff_ave"] = 1000000
    assert analyze_calibration_rows(dataset["rows"])["force_signal_fit"]["slope"] == before


def test_spatial_hold_and_recovery_metrics():
    report = trial_characterisation(synthetic_dataset())
    assert report["position_hold_force_g"]["change"] < 0
    assert report["unloaded_recovery"]["final_offset_from_stationary_baseline"] == -3
    assert report["spatial_response"][1]["response_ratio_to_pressed_taxel"] == pytest.approx(.1)


def test_held_out_equations_and_bands_use_whole_matching_trials():
    report = batch_characterisation([synthetic_dataset("a"), synthetic_dataset("b")])
    group = report["groups"][0]
    assert len(group["leave_one_trial_out"]) == 2
    assert group["held_out_rmse_g"] < 1e-6
    assert group["matched_force_signal_band"]["trials"] == 2


def test_different_speeds_targets_and_interrupted_trials_are_not_pooled():
    a, b, c = synthetic_dataset("a"), synthetic_dataset("b", .0002), synthetic_dataset("c")
    c["metadata"]["measurement_quality"]["interrupted"] = True
    report = batch_characterisation([a, b, c])
    assert len(report["groups"]) == 2
    assert report["excluded_trials"] == ["/recordings/c.csv"]
    assert all(not g["leave_one_trial_out"] for g in report["groups"])
    b["metadata"]["approach_speed_m_s"] = .0001
    b["metadata"]["physical_target_taxel_index"] = 1
    assert len(batch_characterisation([a, b])["groups"]) == 2


def test_live_result_comparisons_and_exports_use_the_same_condition_policy():
    results = []
    for name, speed in (("a", .0001), ("b", .0002), ("c", .0001), ("d", .0001)):
        data = synthetic_dataset(name, speed)
        result = dict(data["metadata"], rows=data["rows"], signal_field="diff_ave", name=name)
        result["measurement_quality"]["trial_completed"] = True
        results.append(result)
    results[-1]["measurement_quality"]["interrupted"] = True
    assert [r["name"] for r in comparable_trial_results(results, results[0])] == ["a", "c"]
    assert len(result_condition_groups(results)) == 2
    results[2]["measurement_quality"]["trial_completed"] = False
    assert [r["name"] for r in comparable_trial_results(results, results[0])] == ["a"]


def test_reverse_mapping_keeps_point_metadata_aligned(monkeypatch):
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    task.reverse_scan_order = True
    monkeypatch.setattr(task, "_move_to_current_scan_point", lambda host: None)
    task._start_scan(
        None, kind="local", points=[[1, 0, 0], [2, 0, 0]],
        baseline=np.zeros(2), target=0, meta=[{"axis": "first"}, {"axis": "second"}],
    )
    assert task._scan_points[0][0] == 2
    assert task._scan_point_meta[0]["axis"] == "second"


def test_legacy_data_still_analyzes_but_is_excluded_from_validated_comparisons():
    dataset = synthetic_dataset()
    del dataset["metadata"]["measurement_quality"]
    assert analyze_calibration_rows(dataset["rows"])["contact_onset"]
    assert batch_characterisation([dataset])["excluded_trials"]


def test_map_repeatability_is_not_absolute_accuracy():
    a = {"settings": {"sensor_rows": 2, "sensor_columns": 2}, "positions": [
        {"taxel_index": 0, "status": "measured", "tool_position_m": [0, 0, .1]}
    ]}
    b = copy.deepcopy(a)
    b["positions"][0]["tool_position_m"][0] = .001
    result = compare_position_maps([a, b])
    assert result["taxels"][0]["xy_rms_about_mean_mm"] == .5
    b["settings"]["sensor_rows"] = 3
    with pytest.raises(ValueError):
        compare_position_maps([a, b])


def test_completed_frame_snapshot_does_not_change_mid_record():
    data = SensorDataBuffer(2, 2, window_size=1)
    data.getRaw(np.full((2, 2), 10.0))
    data.calDiff()
    data.calDiffPer()
    data.getWin(1)
    snapshot = data.completed_frame_snapshot
    data.rawData[:] = 55
    assert np.all(snapshot.rawData == 10)
    with pytest.raises(ValueError):
        snapshot.rawData[0, 0] = 44
    data.clearData()
    assert data.completed_frame_snapshot is None
