"""Fixed scan plane: one operator-chosen height, guarded descent, skip-and-continue."""

import time

import numpy as np
import pytest

from phd.ui.experiment_tasks import TaxelPositionMappingTask
from test_experiment_tasks import _SIM_SURFACE_Z, _Host, _SimRobotApi, _tick, _tick_until
from test_mapping_recovery import prepared, record

PLANE_Z = _SIM_SURFACE_Z + 0.004


def _sim_run(raised=None, taxels=10):
    """20x20 proximity simulation; ``raised(index)`` lifts the skin under a taxel."""
    api = _SimRobotApi()
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    task.move_settle_sec = 0.
    host = _Host(api, task)
    sensor = host.ui_ros.sensor_functions
    sensor.n_row = sensor.n_col = 20
    centres = np.array([
        np.array([.1005, .1996]) + np.array([(index % 20 - 9) * .0111, (index // 20 - 9) * .011])
        for index in range(400)
    ])
    lift = np.array([0. if raised is None else raised(index) for index in range(400)])

    def frame_at_pose():
        position = np.asarray(api.tool_position)
        nearest = int(np.argmin(np.sum((centres - position[:2]) ** 2, axis=1)))
        height = max(float(position[2]) - (_SIM_SURFACE_Z + lift[nearest]), 0.)
        level = 400. / (1. + (height / .0025) ** 2)
        distance_sq = np.sum((centres - position[:2]) ** 2, axis=1)
        return (-level * np.exp(-distance_sq / (2. * .006 ** 2))).reshape(20, 20).T.copy()

    sensor._data.diffDataAve = frame_at_pose()
    assert task.capture_anchor_pose(host)
    api.tool_position = [.1005, .1996, PLANE_Z]
    assert task.capture_scan_height(host)
    api.tool_position = [.1, .2, .3]
    assert task.on_start(host)
    for _ in range(80000):
        if not task._active or len(task._positions) >= taxels:
            break
        api.step(.05)
        sensor._data.diffDataAve = frame_at_pose()
        host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
        _tick(task, host)
    return api, task, host, centres


def test_every_taxel_is_scanned_at_the_remembered_plane():
    _api, task, host, centres = _sim_run()

    assert task._active, host.messages[-5:]
    assert len(task._positions) >= 10
    assert not task._failures
    assert task._fixed_scan_reference_signal is not None
    for index, record_ in task._positions.items():
        assert record_["tool_position_m"][2] == pytest.approx(PLANE_Z, abs=2e-4)
        error_mm = 1000 * np.linalg.norm(np.array(record_["tool_position_m"][:2]) - centres[index])
        assert error_mm < .3, (index, error_mm)
    assert task._settings()["fixed_scan_z_mm"] == pytest.approx(PLANE_Z * 1000.)
    task.on_stop(host, show_result=False)


def test_raised_skin_stops_the_descent_above_the_plane():
    raised_taxels = {190}
    _api, task, host, _centres = _sim_run(
        raised=lambda index: .002 if index in raised_taxels else 0., taxels=5
    )

    assert task._active, host.messages[-5:]
    assert 190 in task._positions
    assert task._positions[190]["tool_position_m"][2] > PLANE_Z + .0003
    assert any("raised skin" in message for message in host.messages)
    task.on_stop(host, show_result=False)


def test_contact_like_response_during_a_scan_lifts_and_retries(tmp_path):
    api, task, host = prepared(tmp_path)
    task.fixed_scan_z_m = .29
    task._fixed_scan_reference_signal = 100.
    task._scan_kind = "local"
    task._scan_baseline = np.zeros(80)
    task._scan_points = [np.array([.104, .2, .29])]
    task._scan_point_meta = [{}]
    task._scan_index = 0
    task._scan_target = 36
    frame = np.zeros(80)
    frame[37] = -250.

    task._record_scan_sample(host, frame)

    assert task._pending_recovery["retry"] is True
    assert "contact guard" in task._pending_recovery["reason"]
    assert task._failed_attempt_total == 1
    # The first leg of the recovery route is straight up.
    assert api.sent[-1][0][:2] == pytest.approx([.104, .2])
    assert api.sent[-1][0][2] == pytest.approx(.3)


def test_exhausted_regular_taxel_is_skipped_and_the_run_continues(tmp_path, monkeypatch):
    api, task, host = prepared(tmp_path)
    task.fixed_scan_z_m = .29
    task._reference_onset_z = .29
    task._attempt_counts[36] = 2
    task._positions[35] = record()
    continued = []
    monkeypatch.setattr(task, "_start_next_regular_taxel", lambda _host: continued.append(True))

    task._handle_probe_failure(host, "target dominance too low")
    assert _tick_until(task, host, lambda: bool(continued) or not task._active, limit=80)

    assert continued and task._active and not host.stops
    assert 36 in task._skipped_failed_taxels
    assert 36 not in task._remaining_target_indices()
    assert task._failed_attempt_total == 1
    assert any("skipped" in message for message in host.messages)


def test_without_a_fixed_plane_an_exhausted_taxel_still_stops(tmp_path):
    api, task, host = prepared(tmp_path)
    task._reference_onset_z = .29
    task._attempt_counts[36] = 2
    task._positions[35] = record()

    task._handle_probe_failure(host, "target dominance too low")
    assert _tick_until(task, host, lambda: not task._active, limit=80)

    assert task.last_result["stop_reason"] == "mapping_target_failed"
    assert not task._skipped_failed_taxels


def test_overshoot_past_the_plane_lifts_back_then_scans_at_the_plane(tmp_path):
    """The arm settles slightly past a stop; that must not refuse the XY scan."""
    api, task, host = prepared(tmp_path)
    task.fixed_scan_z_m = .29
    task._fixed_scan_reference_signal = 100.
    task._probe_baseline = np.zeros(80)
    task._working_probe_boundary = {
        "z_m": .29 - task.FIXED_SCAN_BOUNDARY_MARGIN_M, "source": "operator_fixed_scan_height", "taxels": [],
    }
    api.tool_position = [.104, .2, .2897]
    frame = np.zeros(80)
    frame[36] = -110.

    assert task._tick_fixed_scan_level(host, frame, np.array(api.tool_position), 36, at_depth_limit=True)
    assert task._move_action == "fixed_scan_plane_arrived"
    assert api.sent[-1][0] == pytest.approx([.104, .2, .29])

    api.tool_position = [.104, .2, .29]
    task._dispatch_move_action(host, "fixed_scan_plane_arrived")
    assert task._scan_kind == "local"
    assert all(point[2] == pytest.approx(.29) for point in task._scan_points)
    assert not task._failures


def test_return_from_below_the_travel_height_always_starts_straight_up(tmp_path):
    api, task, host = prepared(tmp_path)
    task._working_probe_boundary = {"z_m": .2895, "source": "test", "taxels": []}
    api.tool_position = [.1001, .2, .2897]  # Within route tolerance, but not identical.

    task._route_to(host, [.1, .2], .3, "clearance_anchor_arrived", "test")

    assert api.sent[-1][0] == pytest.approx([.1001, .2, .3])
    assert not task._failures


def test_a_refused_return_route_stops_instead_of_looping(tmp_path, monkeypatch):
    api, task, host = prepared(tmp_path)

    def refuse(host_, *_args, **_kwargs):
        task._handle_probe_failure(host_, "return refused")
        return False

    monkeypatch.setattr(task, "_command_move", refuse)
    task._handle_probe_failure(host, "first failure")

    assert task.last_result["stop_reason"] == "recovery_route_refused"
    assert task._failed_attempt_total == 1


def test_scan_height_must_be_clearly_below_the_anchor(tmp_path):
    api, task, host = prepared(tmp_path)
    task._active = False
    api.tool_position = [.1, .2, .298]
    assert not task.capture_scan_height(host)
    assert task.fixed_scan_z_m is None
    api.tool_position = [.1, .2, .29]
    assert task.capture_scan_height(host)
    assert task.fixed_scan_z_m == pytest.approx(.29)
