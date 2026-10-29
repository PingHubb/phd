"""Clearance/recalibration and bounded-search regressions, using fake hardware."""

import csv
import json
import time

import numpy as np
import pytest

from phd.ui.experiment_tasks import TaxelPositionMappingTask
from phd.ui.mapping_safety import clearance_recovery, local_retry_xy, working_probe_boundary
from test_experiment_tasks import _Host, _RobotApi, _mapping_mid_run, _tick, _tick_until
from test_mapping_noncontact import probe


def record(x=.1, y=.2, z=.29, **extra):
    return dict(status="measured", tool_position_m=[x, y, z], **extra)


def test_boundary_uses_shallowest_cardinal_neighbour_not_remote_or_unverified():
    points = {34: record(z=.29), 36: record(z=.292), 27: record(z=.28),
              43: record(z=.299, relocation_verified=False), 0: record(z=.4)}
    boundary = working_probe_boundary(points, 35, 8, 10, .3)
    assert boundary["z_m"] == pytest.approx(.2915)
    assert set(boundary["taxels"]) == {34, 36, 27}
    assert boundary["source"] == "shallowest_verified_neighbour_plane"


def test_boundary_prefers_targets_own_recorded_response_plane():
    boundary = working_probe_boundary({35: record(centre_scan_z_m=.291), 34: record(z=.295)}, 35, 8, 10, .3)
    assert boundary["z_m"] == pytest.approx(.2905)
    assert boundary["taxels"] == [35]


@pytest.mark.parametrize("reference,expected", [(None, None), (float("nan"), None), (.29, .285)])
def test_boundary_reference_fallback_skips_invalid_positions(reference, expected):
    boundary = working_probe_boundary({34: record(z=float("nan"))}, 35, 8, 10, reference)
    assert (None if boundary is None else boundary["z_m"]) == expected


def test_clearance_recovery_accepts_noise_but_rejects_residual_downward_response():
    reference = np.tile(np.array([-5., 0., 5., 0.])[:, None], (25, 80))
    values = reference.copy() + 2.
    assert clearance_recovery(values, reference, 6.)["recovered"]
    values[:, 36] -= 50.
    values[:, 7] += 9.  # A rise within the limit is reported but does not fail.
    result = clearance_recovery(values, reference, 6.)
    assert not result["recovered"]
    assert result["unrecovered_taxels"] == [36]
    assert result["unrecovered_details"] == [
        {"taxel": 36, "shift_units": -48.0, "tolerance_units": pytest.approx(4 * 1.4826 * 2.5)}
    ]
    assert [item["taxel"] for item in result["largest_shifts"][:2]] == [36, 7]
    assert result["largest_shifts"][1]["shift_units"] == 11.0
    assert result["median_shift_units"] == 2.0


@pytest.mark.parametrize("frames,reference", [
    (np.full((5, 80), np.nan), None), (np.zeros((0, 80)), None),
    (np.zeros((5, 80)), np.zeros((5, 25))),
    (np.zeros((5, 80)), np.full((5, 80), np.nan)),
])
def test_invalid_clearance_evidence_never_allows_calibration(frames, reference):
    assert not clearance_recovery(frames, reference, 6.)["recovered"]


def test_missing_reference_is_explicit_operator_assumption_not_clearance_measurement():
    result = clearance_recovery(np.zeros((5, 80)), None, 6.)
    assert result["recovered"] and not result["reference_available"]
    assert "Operator-established" in result["interpretation"]


def test_retry_extrapolation_is_bounded_to_existing_local_search_radius():
    points = {34: record(x=.11), 33: record(x=.10)}
    xy = local_retry_xy(points, 35, [.1, .2], 8, 10, .0025)
    np.testing.assert_allclose(xy, [.1025, .2])
    np.testing.assert_allclose(local_retry_xy({}, 35, [.1, .2], 8, 10, .0025), [.1, .2])


def test_working_probe_has_independent_bound_before_any_candidate(tmp_path):
    task, api, host, _level = probe(tmp_path)
    task._positions[34] = record(z=.291)
    task._start_probe(host, np.zeros(80), 0., kind="working")
    assert not task._probe_onset_confirmed
    assert task._working_probe_boundary["z_m"] == pytest.approx(.2905)
    assert task._probe_max_descent_used_m == pytest.approx(.0095)
    api.tool_position[2] = .2905
    for sequence in range(10, 20):
        if task._pending_recovery:
            break
        task._tick_probe(host, (sequence, np.zeros(80)))
    assert task._pending_recovery is not None
    assert not api.velocity_commands  # no request to descend beyond the plane
    assert "no proximity onset" in task._pending_recovery["reason"]


def test_far_field_reference_does_not_zero_existing_near_start_proximity(tmp_path):
    task, _api, host, level = probe(tmp_path)
    near = np.zeros(80)
    near[35] = -120.
    task._far_baseline_frames = np.zeros((20, 80))
    task._start_probe(host, near, 0., kind="working", baseline_min=near,
                      baseline_max=near, baseline_frames=np.tile(near, (20, 1)))
    assert task._probe_baseline[35] == 0.
    level(.2995, 120.)
    level(.299, 120.)
    assert task._probe_scan_target_reached


def test_lost_reference_candidate_never_restores_fast_descent(tmp_path):
    task, api, host, level = probe(tmp_path)
    task._start_probe(host, np.zeros(80), 0., kind="reference")
    level(.2995, 60.)
    level(.299, 0.)
    assert not task._probe_onset_confirmed and task._probe_candidate_seen
    task._tick_probe(host, (100, np.zeros(80)))
    assert api.velocity_commands[-1][0][2] == -task.probe_confirmation_speed_m_s


def prepared(tmp_path):
    api, task, host = _mapping_mid_run(tmp_path, 36, "regular", 1)
    task.move_settle_sec = 0.
    task.baseline_frame_count = 5
    task._clearance_reference_frames = np.zeros((5, 80))
    api.tool_position = [.104, .2, .292]
    task._current_target_xy = np.array([.104, .2])
    return api, task, host


def test_verified_taxel_returns_vertically_then_calibrates_and_checkpoints_before_next(tmp_path, monkeypatch):
    api, task, host = prepared(tmp_path)
    task._output_folder = tmp_path
    events = []
    monkeypatch.setattr(task, "_start_next_regular_taxel", lambda _host: events.append("next"))
    task._accept_position(host, api.tool_position, 120., 10., 12.)
    assert 36 in task._positions
    assert api.sent[-1][0] == [.104, .2, .3]  # vertical lift first
    assert not events and host.ui_ros.sensor_functions.update_cal_calls == 0
    assert _tick_until(task, host, lambda: task._phase == "sensor_calibrating", limit=50)
    assert api.tool_position == [.1, .2, .3]
    assert not events
    assert _tick_until(task, host, lambda: bool(events), limit=50)
    assert host.ui_ros.sensor_functions.update_cal_calls == 1
    assert task._sensor_calibration_history[-1]["status"] == "complete"
    saved = json.loads((tmp_path/"taxel_position_map.json").read_text())
    assert saved["positions"][0]["taxel_index"] == 36
    assert saved["stop_reason"] == "checkpoint"
    assert not (tmp_path/"taxel_position_map.json.tmp").exists()
    assert saved["clearance_checks"][-1]["recovered"]


def test_residual_response_blocks_recalibration_instead_of_hiding_proximity(tmp_path):
    api, task, host = prepared(tmp_path)
    host.ui_ros.sensor_functions._data.diffDataAve[4, 4] = -50.
    task._accept_position(host, api.tool_position, 120., 10., 12.)
    assert _tick_until(task, host, lambda: not task._active, limit=50)
    assert task.last_result["stop_reason"] == "clearance_not_recovered"
    assert "taxel 36 (-50.0, limit 12.0)" in task.last_result["stop_message"]
    check = task.last_result["clearance_checks"][-1]
    assert check["unrecovered_details"][0]["taxel"] == 36
    assert check["median_shift_units"] == 0.0
    assert host.ui_ros.sensor_functions.update_cal_calls == 0
    assert api.tool_position == [.1, .2, .3]
    assert 36 in task._positions


def test_unstable_clearance_stops_without_calibrating(tmp_path):
    api, task, host = prepared(tmp_path)
    task.proximity_detection_mode = "fixed"
    task._begin_clearance_cycle(host, "regular")
    assert _tick_until(task, host, lambda: task._collection_action == "clearance_precheck", limit=50)
    for n in range(5):
        host.ui_ros.sensor_functions._data.diffDataAve.fill(100. if n % 2 else 0.)
        _tick(task, host)
    assert task.last_result["stop_reason"] == "clearance_unstable"
    assert host.ui_ros.sensor_functions.update_cal_calls == 0


def test_calibration_refuses_a_pose_below_clearance(tmp_path):
    api, task, host = prepared(tmp_path)
    task._begin_sensor_calibration(host, next_action="clearance_calibrated", context="test")
    assert task.last_result["stop_reason"] == "calibration_pose_not_clear"
    assert not api.sent
    assert host.ui_ros.sensor_functions.update_cal_calls == 0


def test_force_guard_stays_enabled_during_tactile_calibration(tmp_path):
    api, task, host = prepared(tmp_path)
    api.tool_position = [.1, .2, .3]
    task._force_baseline_n = .01
    task._begin_sensor_calibration(host, next_action="clearance_calibrated", context="test")
    _tick(task, host)  # finish fake asynchronous tactile calibration
    assert task._force_baseline_n == .01
    before = len(api.sent)
    host.ui_ros._force_meter_last_raw_newtons = .01 + 6.*task.NEWTONS_PER_GRAM
    task.on_tick(host)
    assert task.last_result["stop_reason"] == "possible_contact"
    assert len(api.sent) == before
    assert host.ui_ros.sensor_functions.update_cal_calls == 1


@pytest.mark.parametrize("fault,expected", [
    ("force", "force_stale"), ("robot", "robot_feedback_stale"),
    ("sensor", "sensor_lost"),
])
def test_connection_faults_stop_without_return_or_recalibration(tmp_path, fault, expected):
    api, task, host = prepared(tmp_path)
    task._phase = "probing"
    if fault == "force":
        host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()-20.
    elif fault == "robot":
        api.current_feedback_state_time = time.monotonic()-20.
    else:
        host.ui_ros.sensor_functions.is_connected = False
    task.on_tick(host)
    assert task.last_result["stop_reason"] == expected
    assert not api.sent and not api.velocity_commands
    assert host.ui_ros.sensor_functions.update_cal_calls == 0


def test_feedback_below_boundary_cannot_start_xy_scan(tmp_path):
    task, api, host, _level = probe(tmp_path)
    task._working_probe_boundary = dict(z_m=.295, source="test", taxels=[])
    api.tool_position[2] = .294
    assert not task._command_move(host, [.104, .2, .296], "scan_point_arrived", "scan")
    assert api.sent[-1][0] == [.1, .2, .3]  # recovery vertical, not forbidden XY
    assert "refusing XY motion" in task._pending_recovery["reason"]


def test_depth_log_and_force_stop_save_feedback_without_claiming_first_contact(tmp_path):
    task, api, host, level = probe(tmp_path)
    task._output_folder = tmp_path
    task._mapping_run_started = True
    level(.2995, 60.)
    host.ui_ros._force_meter_last_raw_newtons = 6.*task.NEWTONS_PER_GRAM
    task.on_tick(host)
    saved = json.loads((tmp_path/"taxel_position_map.json").read_text())
    snapshot = saved["stop_snapshot"]
    assert snapshot["tcp_z_m"] == .2995
    assert snapshot["probe_descent_mm"] == pytest.approx(.5)
    assert snapshot["force_change_g"] == pytest.approx(6.)
    assert snapshot["tcp_feedback_fresh"]
    assert "not an independently verified contact point" in snapshot["interpretation"]
    with (tmp_path/"probe_levels.csv").open() as stream:
        samples = list(csv.DictReader(stream))
    assert len(samples) == 1
    assert float(samples[0]["tcp_z_m"]) == .2995
    assert samples[0]["onset_confirmed_before_level"] == "False"
    assert not api.sent  # force stop did not auto-retract


@pytest.mark.parametrize("reason", ["possible_contact", "force_guard", "force_safety_limit"])
def test_resume_after_force_stop_below_anchor_refuses_all_automatic_motion(tmp_path, reason):
    api, original, host = prepared(tmp_path)
    original._current_taxel = 35
    original._current_context = "anchor"
    original._reference_onset_z = .29
    api.tool_position = [.1, .2, .292]
    original._accept_position(host, api.tool_position, 120., 10., 12.)
    partial = original._save_results()
    partial["stop_reason"] = reason
    resumed = TaxelPositionMappingTask(storage_path=tmp_path/"resume-anchor.json", output_directory=None)
    ok, message = resumed.request_resume(partial)
    assert ok, message
    second = _Host(api, resumed)
    api.sent.clear()
    api.tool_position = [.1, .2, .295]
    assert not resumed.on_start(second)
    refusal = second.messages[-1]
    assert "Resume refused after a force/contact stop" in refusal
    assert "Z 295.0 mm, 5.0 mm below the saved anchor height Z 300.0 mm" in refusal
    assert "raise the robot straight up to at least Z 300.0 mm" in refusal
    assert not api.sent and not api.velocity_commands
    assert second.ui_ros.sensor_functions.update_cal_calls == 0


def test_resume_alignment_failure_recalibrates_at_anchor_before_stopping(tmp_path):
    api, task, host = prepared(tmp_path)
    api.tool_position = [.1, .2, .3]
    task._resume_alignment_error = "Saved map alignment changed"
    task._dispatch_move_action(host, "resume_alignment_returned")
    assert not host.stops
    assert _tick_until(task, host, lambda: not task._active, limit=50)
    assert task.last_result["stop_reason"] == "resume_alignment_failed"
    assert task.last_result["stop_message"] == "Saved map alignment changed"
    assert host.ui_ros.sensor_functions.update_cal_calls == 1


def test_reference_retry_keeps_the_full_fresh_noise_batch(tmp_path):
    api, task, host = prepared(tmp_path)
    api.tool_position = [.1, .2, .3]
    task._pending_recovery = dict(kind="reference", taxel_index=35, retry=True)
    task._current_taxel = 35
    task._reference_attempt_count = 1
    frames = np.zeros((100, 80))
    frames[:, 35] = np.tile([-20., -15., 0., 15., 20.], 20)
    task._handle_collection(host, "retry_anchor_stability", frames.mean(axis=0), 0.,
                            frames.min(axis=0), frames.max(axis=0), frames)
    detection = task._probe_detection_by_taxel[35]
    assert detection["baseline_min_counts"] == -20.
    assert detection["baseline_max_counts"] == 20.
    assert task._reference_attempt_count == 2
