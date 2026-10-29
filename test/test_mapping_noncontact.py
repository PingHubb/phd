"""Stronger-signal mapping must fail closed, never chase the signal into load."""
import time

import numpy as np
import pytest

from phd.ui.experiment_tasks import TaxelPositionMappingTask
from test_experiment_tasks import _Host, _RobotApi, _mapping_mid_run


def probe(tmp_path, *, strength=None, context="regular", previous=None, motion="velocity"):
    task = TaxelPositionMappingTask(storage_path=tmp_path/"anchor.json", output_directory=None)
    api = _RobotApi()
    host = _Host(api, task)
    # level() below places the TCP itself, which matches the legacy velocity
    # probe; stepwise probes command the TCP and are tested separately.
    task.probe_motion = motion
    task.move_settle_sec = 0.
    task._active = True
    task._current_taxel = 35
    task._current_context = context
    task._resume_anchor_previous = previous
    task.proximity_detection_mode = "fixed"
    task.probe_sample_frames = 3
    task.probe_confirm_frames = 2
    task.signal_threshold_counts = 50.
    if strength is not None:
        task.centre_scan_signal_multiplier = strength
    task._initial_row_vector_xy = np.array([.004, 0.])
    task._initial_column_vector_xy = np.array([0., .005])
    task.anchor_pose = {"tool_position_m": [.1, .2, .3], "tool_quaternion_wxyz": [1., 0., 0., 0.]}
    baseline = np.zeros(80)
    task._far_baseline = baseline.copy()
    task._start_probe(host, baseline, 0., kind="working", baseline_min=baseline,
                      baseline_max=baseline, baseline_frames=np.zeros((20, 80)))
    sequence = [1]

    def level(z, signal):
        api.tool_position = [.1, .2, z]
        frame = np.zeros(80)
        frame[35] = -signal
        for _ in range(2+task.probe_sample_frames):
            sequence[0] += 1
            task._tick_probe(host, (sequence[0], frame))
        # A borderline level now holds still for up to three disjoint batches.
        # Complete that hold before this fake driver is allowed to change Z.
        for _ in range(task.STRENGTH_CONFIRM_BATCHES * task.probe_sample_frames):
            if task._probe_strength_hold is None or not task._active:
                break
            sequence[0] += 1
            task._tick_probe(host, (sequence[0], frame))
    return task, api, host, level


def _stepwise_levels(task, api, host, signal_at_mm, limit=300):
    """The task commands each 0.5 mm line step; the signal depends on the reached Z."""
    sequence = 1
    for _ in range(limit):
        if task._probe_scan_target_reached or task._pending_recovery or task._phase != "probing":
            return
        z_mm = round(float(api.tool_position[2]) * 1000.0, 3)
        frame = np.zeros(80)
        frame[35] = -signal_at_mm.get(z_mm, 0.0)
        sequence += 1
        task._tick_probe(host, (sequence, frame))


def test_stepwise_probe_finds_the_same_planes_with_line_steps(tmp_path):
    task, api, host, _level = probe(tmp_path, motion="stepwise")
    _stepwise_levels(task, api, host, {299.5: 60., 299.0: 70., 298.5: 90., 298.0: 110.})

    assert task._probe_onset_pose[2] == pytest.approx(.2995)
    assert task._probe_scan_pose[2] == pytest.approx(.2980)
    assert task._probe_scan_target_reached
    steps = [entry[0][2] for entry in api.sent if entry[2].get("motion_type") == task.LINE_MOTION]
    assert steps[:4] == pytest.approx([.2995, .2990, .2985, .2980])
    assert all(entry[2]["fine_goal"] for entry in api.sent[:4])
    assert not api.velocity_commands


def test_stepwise_probe_never_steps_below_the_neighbour_plane(tmp_path):
    from test_mapping_recovery import record
    task, api, host, _level = probe(tmp_path, motion="stepwise")
    task._positions[34] = record(z=.291)
    task._start_probe(host, np.zeros(80), 0., kind="working")
    _stepwise_levels(task, api, host, {})

    assert task._pending_recovery is not None
    assert "no proximity onset" in task._pending_recovery["reason"]
    assert min(entry[0][2] for entry in api.sent) >= .2905 - 1e-9
    assert not api.velocity_commands


def test_default_seeks_twofold_signal_without_losing_original_onset(tmp_path):
    task, _api, _host, level = probe(tmp_path)
    level(.2995, 60.)
    level(.2990, 70.)
    assert task._probe_onset_confirmed
    level(.2985, 90.)  # Confirmed proximity is not yet the 2× working plane.
    assert task._scan_kind == ""
    assert task._phase == "probing"
    assert task._probe_scan_target_signal == 100.
    level(.2980, 110.)
    assert task._probe_onset_pose[2] == pytest.approx(.2995)
    assert task._probe_scan_pose[2] == pytest.approx(.2980)
    assert task._probe_scan_target_reached
    assert task._scan_kind == "local"


@pytest.mark.parametrize("strength", [3., 4.])
def test_explicit_higher_target_still_requires_the_requested_strength(tmp_path, strength):
    task, _api, _host, level = probe(tmp_path, strength=strength)
    level(.2995, 60.)
    level(.2990, 70.)
    level(.2985, 110.)  # 2× is not enough when the operator requests more.
    assert task._phase == "probing" and task._scan_kind == ""
    assert task._probe_scan_target_signal == 50. * strength
    level(.2980, 50. * strength + 10.)
    assert task._probe_scan_target_reached and task._scan_kind == "local"


def test_resume_anchor_check_scans_at_the_saved_signal_strength(tmp_path):
    """A lower noise threshold today must not lift the resume scan plane."""
    task, _api, _host, level = probe(
        tmp_path, context="resume_anchor", previous={"centre_scan_target_units": 150.}
    )
    assert task._probe_scan_target_signal == 150.
    level(.2995, 60.)
    level(.2990, 70.)
    level(.2985, 110.)  # Today's 2× target, but weaker than the saved scan.
    assert task._phase == "probing" and task._scan_kind == ""
    level(.2980, 160.)
    assert task._probe_scan_target_reached and task._scan_kind == "local"


def test_resume_anchor_without_a_saved_strength_keeps_the_twofold_target(tmp_path):
    task, _api, _host, _level = probe(tmp_path, context="resume_anchor", previous={})
    assert task._probe_scan_target_signal == 100.


def test_cannot_enter_xy_scan_without_a_verified_strong_plane(tmp_path):
    task, _api, _host, level = probe(tmp_path)
    level(.2995, 60.)
    level(.2990, 70.)
    task._handle_probe_onset(_host)
    assert task._scan_kind == ""
    assert "has not been verified" in task._pending_recovery["reason"]


@pytest.mark.parametrize("limit", [.005, .020])
def test_over_limit_strong_sample_cannot_start_xy_motion(tmp_path, limit):
    task, api, host, level = probe(tmp_path)
    task.centre_scan_max_extra_descent_m = limit
    level(.2995, 60.)
    level(.2990, 70.)
    task._probe_scan_target_reached = True
    task._probe_scan_pose = np.array([.1, .2, task._probe_onset_pose[2] - limit - .0005])
    task._probe_scan_signal = 210.
    task._handle_probe_onset(host)
    assert task._scan_kind == ""
    assert "exceeded" in task._pending_recovery["reason"]
    assert any(text in task._pending_recovery["reason"] for text in ("allowed extra descent", "known-height boundary"))
    assert api.sent[-1][0][:2] == [.1, .2]  # Recovery starts with vertical lift.


@pytest.mark.parametrize("phase", ["probing", "moving", "collecting"])
def test_small_force_rise_stops_without_xy_scan_or_automatic_retract(tmp_path, phase):
    api, task, host = _mapping_mid_run(tmp_path, 36, "regular", 1)
    task._phase = phase
    task._force_baseline_n = 2.*task.NEWTONS_PER_GRAM
    host.ui_ros._force_meter_last_raw_newtons = 8.*task.NEWTONS_PER_GRAM  # +6 g, below old 20 g guard.
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    before = len(api.sent), len(api.velocity_commands)
    task.on_tick(host)
    assert task.last_result["stop_reason"] == "possible_contact"
    assert not task._active
    assert (len(api.sent), len(api.velocity_commands)) == before
    assert api.velocity_stops


def test_working_probe_does_not_rezero_away_force_at_the_lower_height(tmp_path):
    task, _api, host, _level = probe(tmp_path)
    task._force_baseline_n = .02
    task._start_probe(host, np.zeros(80), .06, kind="working")
    assert task._force_baseline_n == .02


@pytest.mark.parametrize("old_extra,expected", [(.001, .001), (.005, .005), (.02, .02), (.03, .02)])
def test_old_resume_settings_upgrade_strength_but_never_expand_depth(tmp_path, old_extra, expected):
    task, _api, _host, _level = probe(tmp_path)
    settings = task._settings()
    settings.pop("centre_scan_policy")
    settings["centre_scan_signal_multiplier"] = 2.
    settings["centre_scan_max_extra_descent_mm"] = old_extra*1000
    task._apply_resume_settings({"settings": settings, "anchor_pose": task.anchor_pose})
    assert task.centre_scan_signal_multiplier == 2.
    assert task.centre_scan_max_extra_descent_m == expected
    assert task._settings()["noncontact_force_stop_g"] == 5.


@pytest.mark.parametrize("saved,expected", [(1., 2.), (2., 2.), (4., 4.)])
def test_legacy_policy_uses_twofold_floor_without_erasing_larger_choices(tmp_path, saved, expected):
    task, _api, _host, _level = probe(tmp_path)
    settings = task._settings()
    settings.pop("centre_scan_policy")
    settings["centre_scan_signal_multiplier"] = saved
    task._apply_resume_settings({"settings": settings, "anchor_pose": task.anchor_pose})
    assert task.centre_scan_signal_multiplier == expected


def test_twofold_default_still_fails_closed_when_signal_is_too_weak_at_depth_limit(tmp_path):
    task, _api, _host, level = probe(tmp_path)
    level(.2995, 60.)
    level(.2990, 70.)
    # At the active working boundary (which is stricter than onset + 20 mm),
    # with a little arrival headroom; still below the 100-unit target.
    level(.2801, 80.)
    assert task._scan_kind == ""
    assert task._pending_recovery is not None
    assert "stronger non-contact scan signal was not reached" in task._pending_recovery["reason"]


def test_twenty_mm_allowance_can_pass_five_mm_but_stops_at_signal_target(tmp_path):
    task, _api, _host, level = probe(tmp_path)
    assert task.centre_scan_max_extra_descent_m == .020
    level(.2995, 60.)
    level(.2990, 70.)
    level(.29449, 80.)  # Over 5 mm below onset: continue, not a failure yet.
    assert task._phase == "probing" and task._pending_recovery is None
    level(.2915, 110.)  # The 2× target is reached after 8 mm, not 20 mm.
    assert task._probe_scan_target_reached and task._scan_kind == "local"
    assert task._probe_onset_pose[2] - task._probe_scan_pose[2] == pytest.approx(.008)


def test_smaller_explicit_allowance_still_stops_at_five_mm(tmp_path):
    task, _api, _host, level = probe(tmp_path)
    task.centre_scan_max_extra_descent_m = .005
    level(.2995, 60.)
    level(.2990, 70.)
    level(.29449, 80.)
    assert task._pending_recovery is not None and task._scan_kind == ""


@pytest.mark.parametrize("strength", [2., 3., 4., 5.])
def test_current_policy_preserves_explicit_operator_strength_choice(tmp_path, strength):
    task, _api, _host, _level = probe(tmp_path)
    settings = task._settings()
    settings["centre_scan_signal_multiplier"] = strength
    task._apply_resume_settings({"settings": settings, "anchor_pose": task.anchor_pose})
    assert task.centre_scan_signal_multiplier == strength


@pytest.mark.parametrize("extra,strength", [(.0201, 4.), (float("nan"), 4.), (.005, float("inf"))])
def test_invalid_noncontact_scan_limits_do_not_start_motion(tmp_path, extra, strength):
    task = TaxelPositionMappingTask(storage_path=tmp_path/"anchor.json", output_directory=None)
    api = _RobotApi()
    host = _Host(api, task)
    assert task.capture_anchor_pose(host)
    task.centre_scan_max_extra_descent_m = extra
    task.centre_scan_signal_multiplier = strength
    assert not task.on_start(host)
    assert not api.sent and not api.velocity_commands


def test_future_force_timestamp_is_not_accepted_for_noncontact_mapping(tmp_path):
    task, _api, host, _level = probe(tmp_path)
    now = time.monotonic()
    host.ui_ros._force_meter_last_sample_monotonic = now+30
    assert task._fresh_force(host, now) is None


def test_mapping_stop_preserves_full_message_in_saved_diagnostics_and_progress(tmp_path):
    task, _api, host, _level = probe(tmp_path)
    task._mapping_run_started = True
    detail = "Taxel 35 failed twice: stronger scan signal 24.5/34.7 sensor units at descent limit."
    task._request_stop(host, "mapping_target_failed", detail)
    assert task._stop_message == detail
    assert task.last_result["stop_reason"] == "mapping_target_failed"
    assert task.last_result["stop_message"] == detail
    assert detail in host.taxel_mapping_progress[-1]["detail"]
