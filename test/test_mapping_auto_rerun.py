"""Automatic recovery never bypasses motion guards or restarts a stopped task."""

import copy
import json
import time

import numpy as np
import pytest

from phd.ui.experiment_tasks import TaxelPositionMappingTask
from test_experiment_tasks import _Host, _RobotApi, _saved_navigation_map, _tick, _tick_until
from test_mapping_recovery import prepared, record


def failed_round(tmp_path, limit=3):
    api, task, host = prepared(tmp_path)
    task.max_auto_reruns = limit
    task._reference_onset_z = .29
    task._attempt_counts[36] = 2
    task._positions[35] = record()
    task._handle_probe_failure(host, "target dominance too low")
    assert _tick_until(task, host, lambda: task._phase == "auto_rerun_wait" or not task._active, limit=80)
    return api, task, host


def expire_wait(task, host):
    task._auto_rerun_deadline = time.monotonic()-1.
    _tick(task, host)


def test_recoverable_failure_returns_calibrates_and_waits_without_ending(tmp_path):
    api, task, host = failed_round(tmp_path)
    assert task._active and not host.stops
    assert task._phase == "auto_rerun_wait"
    assert api.tool_position == [.1, .2, .3]
    assert host.ui_ros.sensor_functions.update_cal_calls == 1
    assert task._auto_rerun_count == 0
    assert task._auto_rerun_history[-1]["reason"] == "target dominance too low"
    before = len(api.sent), len(api.velocity_commands)
    _tick(task, host, 5)
    assert (len(api.sent), len(api.velocity_commands)) == before
    assert "in 30 s" in host.taxel_mapping_progress[-1]["detail"]


def test_expired_wait_rechecks_clearance_and_calibrates_before_retrying_only_failed_target(tmp_path):
    api, task, host = failed_round(tmp_path)
    preserved = copy.deepcopy(task._positions)
    expire_wait(task, host)
    assert task._collection_action == "clearance_precheck"
    assert task._auto_rerun_count == 0
    assert _tick_until(task, host, lambda: task._auto_rerun_count == 1, limit=80)
    assert host.ui_ros.sensor_functions.update_cal_calls == 2
    assert task._positions == preserved
    assert task._current_taxel == 36 and task._current_context == "regular"
    assert task._attempt_counts[36] == 1
    assert task._auto_rerun_history[-1]["status"] == "started"
    assert host.taxel_mapping_progress[-1]["auto_reruns"] == 1
    assert host.taxel_mapping_progress[-1]["max_auto_reruns"] == 3


def test_rerun_budget_is_global_not_reset_after_each_failed_taxel(tmp_path):
    api, task, host = failed_round(tmp_path, limit=1)
    expire_wait(task, host)
    assert _tick_until(task, host, lambda: task._auto_rerun_count == 1, limit=80)
    task._attempt_counts[36] = 2
    task._handle_probe_failure(host, "still not detectable")
    assert _tick_until(task, host, lambda: not task._active, limit=80)
    assert task._auto_rerun_count == 1
    assert task.last_result["stop_reason"] == "mapping_target_failed"
    assert "Auto reruns used: 1/1" in task.last_result["stop_message"]
    assert task._auto_rerun_pending is None and task._auto_rerun_deadline is None


def test_default_off_keeps_terminal_failure_stopped(tmp_path):
    _api, task, host = failed_round(tmp_path, limit=0)
    assert not task._active and host.stops
    assert task._auto_rerun_count == 0 and not task._auto_rerun_history


def test_manual_stop_cancels_wait_without_consuming_or_resurrecting_a_rerun(tmp_path):
    api, task, host = failed_round(tmp_path)
    task.on_stop(host, show_result=False)
    before = len(api.sent), len(api.velocity_commands)
    for _ in range(5):
        task.on_tick(host)
    assert (len(api.sent), len(api.velocity_commands)) == before
    assert task.last_result["stop_reason"] == "manual_stop"
    assert task._auto_rerun_history[-1]["status"] == "cancelled"
    assert task._auto_rerun_count == 0


@pytest.mark.parametrize("fault,expected", [
    ("force", "possible_contact"), ("force_stale", "force_stale"),
    ("robot_stale", "robot_feedback_stale"), ("sensor_lost", "sensor_lost"),
    ("moved", "auto_rerun_clearance_lost"), ("rotated", "auto_rerun_clearance_lost"),
])
def test_faults_during_cooldown_cancel_instead_of_auto_resuming(tmp_path, fault, expected):
    api, task, host = failed_round(tmp_path)
    if fault == "force":
        host.ui_ros._force_meter_last_raw_newtons = 6.*task.NEWTONS_PER_GRAM
    elif fault == "force_stale":
        host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()-10.
    elif fault == "robot_stale":
        api.current_feedback_state_time = time.monotonic()-10.
    elif fault == "sensor_lost":
        host.ui_ros.sensor_functions.is_connected = False
    elif fault == "moved":
        api.tool_position[0] += .001
    else:
        api.tool_quaternion = [0., 1., 0., 0.]
    before = len(api.sent), len(api.velocity_commands)
    task._auto_rerun_deadline = time.monotonic()-1.
    _tick(task, host)
    assert not task._active and task.last_result["stop_reason"] == expected
    assert (len(api.sent), len(api.velocity_commands)) == before
    assert task._auto_rerun_count == 0
    assert task._auto_rerun_history[-1]["status"] == "cancelled"


def test_pose_guard_remains_active_during_new_calibration(tmp_path):
    api, task, host = failed_round(tmp_path)
    expire_wait(task, host)
    assert _tick_until(task, host, lambda: task._phase == "sensor_calibrating", limit=30)
    before = len(api.sent)
    api.tool_position[2] -= .001
    _tick(task, host)
    assert task.last_result["stop_reason"] == "auto_rerun_clearance_lost"
    assert len(api.sent) == before and task._auto_rerun_count == 0


def test_residual_response_after_cooldown_cannot_be_zeroed_by_rerun(tmp_path):
    _api, task, host = failed_round(tmp_path)
    host.ui_ros.sensor_functions._data.diffDataAve.fill(-100.)
    expire_wait(task, host)
    assert _tick_until(task, host, lambda: not task._active, limit=30)
    assert task.last_result["stop_reason"] == "clearance_not_recovered"
    assert host.ui_ros.sensor_functions.update_cal_calls == 1
    assert task._auto_rerun_count == 0


def test_failed_calibration_does_not_schedule_another_rerun(tmp_path):
    _api, task, host = failed_round(tmp_path)
    expire_wait(task, host)
    assert _tick_until(task, host, lambda: task._phase == "sensor_calibrating", limit=30)
    sensor = host.ui_ros.sensor_functions
    sensor._sensor_calibration_in_progress = False
    sensor.is_connected = False
    task.on_tick(host)
    assert task.last_result["stop_reason"] == "sensor_calibration_failed"
    assert task._auto_rerun_count == 0
    assert len(task._auto_rerun_history) == 1


@pytest.mark.parametrize("kind,context,relocating", [
    ("reference", "reference", False), ("target", "resume_anchor", False),
    ("target", "relocation_verify", True), ("target", "regular", True),
])
def test_initial_height_alignment_and_relocation_never_auto_restart(tmp_path, kind, context, relocating):
    _api, task, host = prepared(tmp_path)
    task.max_auto_reruns = 3
    task._reference_onset_z = .29
    task._relocation_active = relocating
    assert not task._queue_auto_rerun(host, dict(kind=kind, context=context, taxel_index=36, reason="failed"))
    assert not task._auto_rerun_history


@pytest.mark.parametrize("limit,delay", [(-1, 30.), (11, 30.), (1.5, 30.), (float("nan"), 30.), (3, 0.), (3, 601.), (3, float("inf"))])
def test_invalid_automatic_recovery_settings_refuse_start(tmp_path, limit, delay):
    api = _RobotApi()
    task = TaxelPositionMappingTask(storage_path=tmp_path/"anchor.json", output_directory=None)
    host = _Host(api, task)
    assert task.capture_anchor_pose(host)
    task.max_auto_reruns, task.auto_rerun_delay_sec = limit, delay
    assert not task.on_start(host)
    assert not api.sent and "Invalid automatic recovery" in host.messages[-1]


def resumable_result():
    result = _saved_navigation_map()
    result.update(reference_onset_z_m=.29, probe_start_z_m=.292,
                  initial_row_vector_xy_m=[.004, 0.], initial_column_vector_xy_m=[0., .004],
                  auto_rerun_count=2, auto_rerun_history=[dict(status="waiting", rerun_number=3)])
    result["settings"].update(max_auto_reruns=10, auto_rerun_delay_sec=10.)
    return result


def test_loading_resume_restores_history_and_count_but_never_arms_saved_auto_settings(tmp_path):
    task = TaxelPositionMappingTask(storage_path=tmp_path/"anchor.json", output_directory=None)
    result = resumable_result()
    task._apply_resume_settings(result)
    task._restore_resume_state(result)
    assert task.max_auto_reruns == 0 and task.auto_rerun_delay_sec == 30.
    assert task._auto_rerun_count == 2
    assert task._auto_rerun_history[-1]["status"] == "interrupted_wait"
    assert task._auto_rerun_deadline is None and task._auto_rerun_pending is None
    task.max_auto_reruns = 3  # explicitly chosen before Resume
    task._apply_resume_settings(result)
    assert task.max_auto_reruns == 3


def test_checkpoint_records_count_and_failure_history(tmp_path):
    _api, task, host = failed_round(tmp_path)
    task._positions = {}  # this test only exercises JSON diagnostics, not geometry
    task._output_folder = tmp_path
    expire_wait(task, host)
    assert _tick_until(task, host, lambda: task._auto_rerun_count == 1, limit=80)
    saved = json.loads((tmp_path/"taxel_position_map.json").read_text())
    assert saved["auto_rerun_count"] == 1
    assert saved["auto_rerun_history"][-1]["taxel_index"] == 36
    assert saved["auto_rerun_history"][-1]["reason"] == "target dominance too low"
    assert saved["settings"]["max_auto_reruns"] == 3
    assert saved["settings"]["auto_rerun_delay_sec"] == 30.
