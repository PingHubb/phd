"""Mapping quality upgrades; all motion/force/sensor interfaces are simulated."""
import copy
import csv
import json
import time

import numpy as np
import pytest

from phd.ui.experiment_tasks import TaxelPositionMappingTask
from test_experiment_tasks import _cross_samples, _mapping_mid_run, _tick_until
from test_mapping_checkpoint import relocating_task
from test_mapping_frontier import make_grid
from test_mapping_noncontact import probe
from test_mapping_recovery import prepared


def at_boundary(tmp_path, monkeypatch, *, limit=True):
    task, api, host, level = probe(tmp_path)
    level(.2995, 60.)
    level(.2990, 70.)
    api.tool_position = [.1, .2, .298]
    task._probe_level_pose = np.array(api.tool_position)
    task._probe_stage = "sampling"
    task._probe_level_frames = []
    task._probe_level_at_max_depth = limit
    task._working_probe_boundary = {"z_m": .298, "source": "test", "taxels": [34]}
    sequence = 1000
    accepted = []
    monkeypatch.setattr(task, "_handle_probe_onset", lambda _host: accepted.append(True))

    def batch(signal, competitor=0.):
        nonlocal sequence
        for _ in range(task.probe_sample_frames):
            sequence += 1
            frame = np.zeros(80)
            frame[35], frame[34] = -signal, -competitor
            task._tick_probe(host, (sequence, frame))
    return task, api, host, accepted, batch


def test_borderline_success_requires_three_batches_without_any_new_motion(tmp_path, monkeypatch):
    task, api, host, accepted, batch = at_boundary(tmp_path, monkeypatch)
    before = len(api.sent), len(api.velocity_commands)
    for signal in [94., 102.]:
        batch(signal)
        assert not accepted and task._probe_strength_hold is not None
        assert (len(api.sent), len(api.velocity_commands)) == before
    batch(103.)
    assert accepted == [True]
    assert task._probe_scan_target_reached
    assert task._probe_scan_signal == 102.
    assert task._probe_scan_target_signal == 100.  # Never relaxed.
    assert task._probe_scan_pose[2] == pytest.approx(.298)
    assert (len(api.sent), len(api.velocity_commands)) == before
    check = task._strength_checks[-1]
    assert check["accepted"] and check["frame_count"] == 9
    assert check["at_depth_limit"]
    assert any("SAME height" in msg for msg in host.messages)


@pytest.mark.parametrize("signals", [[94., 95., 94.], [95., 200., 96.], [102., 103., 96.]])
def test_subthreshold_or_transient_strength_fails_closed_after_bounded_hold(tmp_path, monkeypatch, signals):
    task, api, host, accepted, batch = at_boundary(tmp_path, monkeypatch)
    for signal in signals:
        batch(signal)
    assert not accepted and not task._probe_scan_target_reached
    assert task._probe_strength_hold is None
    assert task._pending_recovery["quality_failure"]
    assert "stationary confirmation median" in task._pending_recovery["reason"]
    assert not task._strength_checks[-1]["accepted"]
    assert all(point[0][2] >= .298 for point in api.sent)  # Retraction only.


def test_confirmation_needs_dominance_not_just_a_large_target(tmp_path, monkeypatch):
    task, api, host, accepted, batch = at_boundary(tmp_path, monkeypatch)
    batch(95.)
    batch(120., competitor=300.)
    batch(120., competitor=300.)
    assert not accepted and task._pending_recovery
    assert not task._strength_checks[-1]["accepted"]


def test_duplicate_frames_do_not_finish_a_stationary_hold(tmp_path, monkeypatch):
    task, api, host, accepted, batch = at_boundary(tmp_path, monkeypatch)
    batch(95.)
    frame = np.zeros(80)
    frame[35] = -105.
    sequence = task._probe_last_sequence
    for _ in range(50):
        task._tick_probe(host, (sequence, frame))
    assert len(task._probe_strength_hold["batches"]) == 1
    assert not task._probe_level_frames and not accepted


def test_tcp_motion_during_hold_stops_instead_of_combining_different_heights(tmp_path, monkeypatch):
    task, api, host, accepted, batch = at_boundary(tmp_path, monkeypatch)
    batch(95.)
    api.tool_position[2] += .001
    batch(105.)
    assert task._stop_reason == "probe_hold_moved" and not task._active
    assert not accepted


def test_force_guard_remains_live_during_a_stationary_hold(tmp_path, monkeypatch):
    task, api, host, accepted, batch = at_boundary(tmp_path, monkeypatch)
    batch(95.)
    task._mapping_run_started = True
    host.ui_ros._force_meter_last_raw_newtons = 6. * task.NEWTONS_PER_GRAM
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    before = len(api.sent)
    task.on_tick(host)
    assert task._stop_reason == "possible_contact"
    assert not task._active and not accepted and len(api.sent) == before


def test_a_previous_peak_cannot_authorise_scan_at_a_weak_current_level(tmp_path):
    task, api, host, level = probe(tmp_path)
    level(.2995, 200.)  # First onset batch is strong, but not yet confirmed.
    level(.2990, 70.)
    assert task._probe_onset_confirmed
    assert not task._probe_scan_target_reached and task._scan_kind == ""


def local_scan(tmp_path, *, reverse=False):
    api, task, host = _mapping_mid_run(tmp_path, 36, "regular", 1)
    task.reverse_scan_order = reverse
    centre = np.array([.104, .2, .29])
    api.tool_position = centre.tolist()
    task._start_local_scan(host, centre)
    task._scan_results = _cross_samples(task, 0., 0., 36)
    task._finish_local_scan(host)
    return api, task, host


def weak_frame():
    frame = np.zeros(80)
    frame[36], frame[37] = 90., 85.
    return frame


@pytest.mark.parametrize("reverse", [False, True])
def test_ambiguous_centre_repeats_exact_footprint_reverse_order_then_saves(tmp_path, reverse):
    api, task, host = local_scan(tmp_path, reverse=reverse)
    first_points = np.array(task._scan_points, copy=True)
    task._finish_verify(host, weak_frame())
    assert 36 not in task._positions
    assert task._centre_repeat is not None
    np.testing.assert_array_equal(task._scan_points, first_points[::-1])
    task._dispatch_move_action(host, "scan_point_arrived")
    assert task._collection_count >= 20
    task._scan_results = _cross_samples(task, .0002, -.0001, 36)
    task._finish_local_scan(host)
    task._finish_verify(host, weak_frame())
    record = task._positions[36]
    assert record["centre_repeat_status"] == "repeatable"
    assert record["centre_repeat_separation_mm"] == pytest.approx(np.sqrt(.05))
    assert record["centre_repeat_check"]["final_verified"]
    assert len(task._centre_checks) == 1
    assert record["tool_position_m"][2] == pytest.approx(.29)


def test_disagreeing_centres_are_not_saved_or_averaged(tmp_path):
    api, task, host = local_scan(tmp_path)
    task._finish_verify(host, weak_frame())
    task._scan_results = _cross_samples(task, .001, 0., 36)
    task._finish_local_scan(host)
    assert 36 not in task._positions
    assert "centre not repeatable" in task._pending_recovery["reason"]
    assert not task._centre_checks[-1]["repeatable"]
    assert task._pending_recovery["quality_failure"]


def test_clear_centre_keeps_single_pass_fast_path(tmp_path):
    api, task, host = local_scan(tmp_path)
    frame = weak_frame()
    frame[36] = 120.
    task._finish_verify(host, frame)
    assert task._positions[36]["centre_repeat_status"] == "not_triggered"
    assert not task._centre_checks


def verify_anchor(tmp_path, competitor):
    api, task, host = _mapping_mid_run(tmp_path, 35, "anchor", 1)
    centre = np.array([.1, .2, .29])
    api.tool_position = centre.tolist()
    task._start_local_scan(host, centre)
    task._scan_results = _cross_samples(task, 0., 0., 35)
    task._finish_local_scan(host)
    frame = np.zeros(80)
    frame[35], frame[36] = 120., competitor
    task._finish_verify(host, frame)
    return api, task, host


def test_anchor_barely_above_its_neighbours_stops_at_the_anchor_unsaved(tmp_path):
    api, task, host = verify_anchor(tmp_path, competitor=90.)
    assert 35 not in task._positions
    if task._move_action == "route_waypoint":
        task._dispatch_move_action(host, "route_waypoint")
    assert task._move_action == "anchor_stop_returned"
    assert api.sent[-1][0] == pytest.approx(task.anchor_pose["tool_position_m"])
    task._dispatch_move_action(host, "anchor_stop_returned")
    assert task._stop_reason == "scan_plane_too_high" and not task._active
    assert "only 1.33x its strongest neighbour" in task._stop_message
    assert "Remember Scan Height" in task._stop_message


def test_clearly_dominant_anchor_is_saved(tmp_path):
    _api, task, _host = verify_anchor(tmp_path, competitor=50.)
    assert task._positions[35]["dominance_ratio"] == pytest.approx(2.4)
    assert task._stop_reason == ""


def test_flat_peak_repeats_even_with_clear_dominance(tmp_path):
    api, task, host = local_scan(tmp_path)
    task._verify_estimate["axes"]["row"]["method"] = "peak_centroid"
    frame = weak_frame()
    frame[36] = 120.
    task._finish_verify(host, frame)
    assert task._centre_repeat and 36 not in task._positions
    assert "broad/flat" in task._centre_repeat["reason"]


def fitted_recovery(tmp_path):
    api, task, host = prepared(tmp_path)
    grid, _points = make_grid(rows=8, columns=10, skip=False)
    task._positions = copy.deepcopy(grid._positions)
    task._current_taxel = 37  # Not an essential anchor neighbour.
    task._reference_onset_z = .29
    task._attempt_counts[37] = 2
    return api, task, host


def test_only_ordinary_quality_failure_with_valid_references_can_continue(tmp_path, monkeypatch):
    api, task, host = fitted_recovery(tmp_path)
    events = []
    monkeypatch.setattr(task, "_start_next_regular_taxel", lambda _host: events.append(True))
    task._handle_probe_failure(host, "not repeatable", quality_failure=True)
    assert _tick_until(task, host, lambda: bool(events) or not task._active, limit=80)
    assert events and task._active and 37 in task._skipped_failed_taxels
    assert api.tool_position == [.1, .2, .3]
    assert host.ui_ros.sensor_functions.update_cal_calls == 1
    assert 37 not in task.plan_frontier_target()["verified_neighbours"]


@pytest.mark.parametrize("context,quality,relocating", [
    ("neighbor", True, False), ("resume_anchor", True, False),
    ("regular", False, False), ("regular", True, True),
])
def test_essential_references_relocation_and_nonquality_failures_still_stop(tmp_path, context, quality, relocating):
    api, task, host = fitted_recovery(tmp_path)
    task._current_context = context
    task._relocation_active = relocating
    task._handle_probe_failure(host, "not usable", quality_failure=quality)
    assert _tick_until(task, host, lambda: not task._active, limit=80)
    assert task._stop_reason == "mapping_target_failed"
    assert not task._skipped_failed_taxels


def test_quality_failure_without_verified_lattice_still_stops(tmp_path):
    api, task, host = prepared(tmp_path)
    task._attempt_counts[36] = 2
    task._handle_probe_failure(host, "weak signal", quality_failure=True)
    assert _tick_until(task, host, lambda: not task._active, limit=80)
    assert task._stop_reason == "mapping_target_failed"


def test_partial_pass_is_not_claimed_complete_and_resume_retries_only_unresolved(tmp_path):
    api, task, host = fitted_recovery(tmp_path)
    task._skipped_failed_taxels = {37}
    task._failures = {37: "weak signal"}
    task._dispatch_move_action(host, "complete_anchor_arrived")
    saved = task.last_result
    assert not saved["completed"] and saved["stop_reason"] == "partial_complete"
    assert 37 in saved["unresolved_taxels"]
    before = copy.deepcopy(task._positions)
    task._restore_resume_state(saved)
    assert set(task._positions) == set(before)
    assert all(task._positions[index]["tool_position_m"] == record["tool_position_m"]
               for index, record in before.items())
    assert not task._skipped_failed_taxels
    assert 37 in task._remaining_target_indices() and 37 not in task._attempt_counts
    assert task._resume_history[-1]["previous_failures"] == {"37": "weak signal"}
    assert task._resume_history[-1]["retry_unresolved_taxels"] == [37]


def test_new_quality_evidence_survives_json_and_csv_checkpoint(tmp_path, monkeypatch):
    task, api, host, accepted, batch = at_boundary(tmp_path, monkeypatch)
    for value in [94., 102., 103.]:
        batch(value)
    saved_task = relocating_task(tmp_path)
    saved_task._strength_checks = task._strength_checks
    saved_task._probe_log = task._probe_log
    saved = saved_task._save_results(checkpoint=True)
    read_back = json.loads((tmp_path / "taxel_position_map.json").read_text())
    assert read_back == saved
    assert read_back["stationary_strength_checks"][0]["accepted"]
    assert read_back["settings"]["mapping_quality_policy"] == task.QUALITY_POLICY
    with (tmp_path / "probe_levels.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[-1]["strength_confirmation_batch"] == "3"
    assert rows[-1]["strength_confirmation_accepted"] == "True"
    saved_task._restore_resume_state(saved)
    assert saved_task._strength_checks == task._strength_checks
    assert saved_task._probe_strength_hold is None
