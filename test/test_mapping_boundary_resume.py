"""Regression for 26 Oct boundary jitter and inherited Resume retry counts.

All interfaces are fake: no serial devices, robot services or live maps touched.
"""
import copy
import time

import numpy as np
import pytest

from phd.ui.experiment_tasks import TaxelPositionMappingTask
from test_experiment_tasks import _Host, _RobotApi
from test_mapping_checkpoint import relocating_task
from test_mapping_noncontact import probe


BOUNDARY_Z = .23876129150390626
ACTUAL_Z = .23875872802734375


def boundary_probe(tmp_path):
    task, api, host, _level = probe(tmp_path, motion="stepwise")
    task._working_probe_boundary = dict(z_m=BOUNDARY_Z, source="saved_target_response_plane", taxels=[35])
    task._probe_start_pose = np.array([.1, .2, .262])
    task._probe_max_descent_used_m = .262 - BOUNDARY_Z
    task._probe_onset_pose = np.array([.1, .2, .239770187])
    task._probe_onset_confirmed = True
    task._probe_scan_target_reached = True
    task._probe_scan_target_signal = 100.
    task._probe_scan_signal = 120.
    task._probe_scan_pose = np.array([.1, .2, BOUNDARY_Z + .000004])
    task._probe_level_pose = task._probe_scan_pose.copy()
    api.tool_position = [.1, .2, ACTUAL_Z]
    return task, api, host


def sample_after_correction(task, api, host, signal):
    task._dispatch_move_action(host, "probe_boundary_corrected")
    frame = np.zeros(80)
    frame[35] = -signal
    for sequence in range(100, 140):
        if task._phase != "probing":
            break
        task._tick_probe(host, (sequence, frame))


def test_recorded_micrometre_undershoot_lifts_only_then_requires_fresh_signal(tmp_path):
    task, api, host = boundary_probe(tmp_path)
    before = list(api.tool_position)
    task._handle_probe_onset(host)
    assert task._move_action == "probe_boundary_corrected"
    assert len(api.sent) == 1
    target = api.sent[0][0]
    assert target[:2] == before[:2]
    assert target[2] == pytest.approx(BOUNDARY_Z + .0002)
    assert not task._probe_scan_target_reached
    assert task._probe_scan_pose is None and not task._probe_level_frames
    assert task._scan_kind == "" and task._pending_recovery is None
    assert task._working_probe_boundary["z_m"] == BOUNDARY_Z
    sample_after_correction(task, api, host, 120.)
    assert task._scan_kind == "local"
    assert task._probe_scan_target_reached
    assert task._probe_scan_pose[2] == pytest.approx(target[2])
    assert all(move[0][2] >= BOUNDARY_Z for move in api.sent)


def test_weak_signal_after_lift_cannot_reuse_old_success_or_descend_again(tmp_path):
    task, api, host = boundary_probe(tmp_path)
    task._handle_probe_onset(host)
    sample_after_correction(task, api, host, 80.)
    assert not task._probe_scan_target_reached
    assert task._scan_kind == ""
    assert task._pending_recovery["quality_failure"]
    assert "not reached reliably" in task._pending_recovery["reason"]
    assert all(move[0][:2] == [.1, .2] for move in api.sent)
    assert all(move[0][2] >= BOUNDARY_Z + .0002 - 1e-9 for move in api.sent)


@pytest.mark.parametrize("extra_drop", [0., .001])
def test_correction_uses_dispatch_feedback_not_previous_xy(tmp_path, monkeypatch, extra_drop):
    task, api, host = boundary_probe(tmp_path)
    read_pose = task._read_tool_pose
    reads = 0

    def changing_feedback(robot):
        nonlocal reads
        reads += 1
        if reads == 2:
            robot.tool_position[0] += .000003
            robot.tool_position[1] -= .000002
            robot.tool_position[2] -= extra_drop
        return read_pose(robot)

    monkeypatch.setattr(task, "_read_tool_pose", changing_feedback)
    task._handle_probe_onset(host)
    assert api.sent[0][0][:2] == pytest.approx([.100003, .199998])
    if extra_drop:
        assert "too far below" in task._pending_recovery["reason"]
        assert api.sent[0][0][2] == .3  # normal clearance recovery
    else:
        assert task._move_action == "probe_boundary_corrected"
        assert task._pending_recovery is None
        assert api.sent[0][0][2] == pytest.approx(BOUNDARY_Z + .0002)


@pytest.mark.parametrize("sample_below", [False, True])
def test_stationary_sampling_discards_below_boundary_frames(tmp_path, sample_below):
    task, api, host = boundary_probe(tmp_path)
    task._probe_stage = "sampling"
    task._probe_level_frames = [np.ones(80)]
    task._probe_strength_hold = {"batches": [120.]}
    if sample_below:
        task._probe_scan_pose[2] = ACTUAL_Z
        api.tool_position[2] = BOUNDARY_Z + .00001
        task._handle_probe_onset(host)
    else:
        task._tick_probe(host, (100, np.zeros(80)))
    assert task._move_action == "probe_boundary_corrected"
    assert task._probe_level_frames == []
    assert task._probe_strength_hold is None
    assert not task._probe_scan_target_reached


@pytest.mark.parametrize("undershoot,used", [(.0005, 0), (.000003, 1)])
def test_large_or_repeated_undershoot_recovers_instead_of_scanning(tmp_path, undershoot, used):
    task, api, host = boundary_probe(tmp_path)
    api.tool_position[2] = BOUNDARY_Z - undershoot
    task._probe_boundary_corrections = used
    task._handle_probe_onset(host)
    assert task._scan_kind == ""
    assert "excessive or repeated undershoot" in task._pending_recovery["reason"]
    assert task._move_action != "probe_boundary_corrected"
    assert api.sent[0][0] == [.1, .2, .3]  # normal vertical clearance recovery


def test_last_depth_step_leaves_headroom_instead_of_commanding_on_guard(tmp_path):
    task, api, host = boundary_probe(tmp_path)
    api.tool_position[2] = BOUNDARY_Z + .000504
    task._probe_line_target_z = None
    task._tick_probe_line_step(host, api, (np.array(api.tool_position), [1., 0., 0., 0.]),
                               time.monotonic(), 10, api.tool_position[2])
    assert api.sent[-1][0][2] == pytest.approx(BOUNDARY_Z + .0002)
    assert task._probe_line_at_limit
    assert task._working_probe_boundary["z_m"] == BOUNDARY_Z


def test_correction_arrival_below_boundary_does_not_resume_sampling(tmp_path):
    task, api, host = boundary_probe(tmp_path)
    task._handle_probe_onset(host)
    api.tool_position[2] = ACTUAL_Z
    task._dispatch_move_action(host, "probe_boundary_corrected")
    assert task._stop_reason == "probe_boundary_correction_failed"
    assert not task._active and task._scan_kind == ""
    assert len(api.sent) == 1


@pytest.mark.parametrize("fault,reason", [("force", "possible_contact"),
                                           ("force_stale", "force_stale"),
                                           ("robot_stale", "robot_feedback_stale")])
def test_correction_does_not_bypass_live_safety_guards(tmp_path, fault, reason):
    task, api, host = boundary_probe(tmp_path)
    task._handle_probe_onset(host)
    count = len(api.sent)
    if fault == "force":
        host.ui_ros._force_meter_last_raw_newtons = task._force_baseline_n + 6.*task.NEWTONS_PER_GRAM
    elif fault == "force_stale":
        host.ui_ros._force_meter_last_sample_monotonic = time.monotonic() - 20.
    else:
        api.current_feedback_state_time = time.monotonic() - 20.
    task.on_tick(host)
    assert task._stop_reason == reason
    assert len(api.sent) == count and task._scan_kind == ""


def test_resume_old_attempts_are_history_not_the_new_retry_budget(tmp_path, monkeypatch):
    original = relocating_task(tmp_path)
    original._positions[12]["attempt_count"] = 3
    original._attempt_counts = {12: 4, 7: 1}
    result = original._save_results(checkpoint=True)
    records = copy.deepcopy(result["positions"])
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    task._apply_resume_settings(result)
    task._restore_resume_state(result, tmp_path / "taxel_position_map.json")
    assert task._attempt_counts == {}
    assert task._resume_history[-1]["previous_attempt_counts"] == {"12": 4, "7": 1}
    assert sorted(task._positions.values(), key=lambda r: r["taxel_index"]) == records
    api = _RobotApi()
    host = _Host(api, task)
    monkeypatch.setattr(task, "_route_to", lambda *args, **kwargs: True)
    for attempt, retry in [(1, True), (2, False)]:
        task._start_target(host, 12, [.1, .2], "resume_anchor")
        assert task._attempt_counts[12] == attempt
        task._handle_probe_failure(host, "test failure")
        assert task._pending_recovery["retry"] is retry
    assert task._positions[12]["attempt_count"] == 3
    saved_again = task._save_results(checkpoint=True)
    assert saved_again["attempt_counts"]["12"] == 2
    assert saved_again["resume_history"][-1]["previous_attempt_counts"]["12"] == 4
