"""Operator scan-plane choices survive Resume/restart. Fake hardware only."""
import copy
import json
import time

import numpy as np
import pytest

from phd.ui.experiment_tasks import TaxelPositionMappingTask
from test_experiment_tasks import _Host, _RobotApi, _SimRobotApi, _tick


def setup_task(tmp_path, api=None):
    api = api or _RobotApi()
    task = TaxelPositionMappingTask(storage_path=tmp_path / "anchor.json", output_directory=None)
    host = _Host(api, task)
    assert task.capture_anchor_pose(host)
    return task, api, host


def partial_map(task, saved_z=None):
    settings = task._settings()
    settings["fixed_scan_z_mm"] = None if saved_z is None else saved_z * 1000.
    settings["fixed_scan_reference_signal_counts"] = 300.
    return dict(
        task=task.id, completed=False, settings=settings,
        sensor_identity=task.active_sensor_key_name,
        anchor_pose=copy.deepcopy(task.anchor_pose),
        reference_onset_z_m=.290, probe_start_z_m=.292,
        initial_row_vector_xy_m=[.004, 0.], initial_column_vector_xy_m=[0., .004],
        positions=[dict(taxel_index=35, status="measured", tool_position_m=[.1, .2, .290],
                        safe_approach_position_m=[.1, .2, .3],
                        tool_quaternion_wxyz=[1., 0., 0., 0.],
                        centre_scan_target_units=12.5)],
    )


@pytest.mark.parametrize("saved_z", [None, .289, .291])
def test_newly_marked_plane_wins_over_saved_map_and_ui_refresh(tmp_path, saved_z):
    task, api, host = setup_task(tmp_path)
    result = partial_map(task, saved_z)
    original = copy.deepcopy(result)
    api.tool_position[2] = .291
    assert task.capture_scan_height(host)
    task._apply_resume_settings(result)
    assert task.fixed_scan_z_m == pytest.approx(.291)
    task.use_connected_sensor(host)  # idle UI refresh must not undo the choice
    assert task.fixed_scan_z_m == pytest.approx(.291)
    assert task._settings()["fixed_scan_z_mm"] == pytest.approx(291.)
    assert result == original
    assert not api.sent


def test_remember_persists_without_start_and_survives_restart_resume(tmp_path):
    task, api, host = setup_task(tmp_path)
    result = partial_map(task)
    api.tool_position[2] = .291
    assert task.capture_scan_height(host)
    saved = json.loads(task.storage_path.read_text())
    assert saved["profiles"][task.active_sensor_key_name]["scan_plane_choice"]["z_m"] == .291
    restarted = TaxelPositionMappingTask(storage_path=task.storage_path, output_directory=None)
    restarted.use_connected_sensor(host)
    assert restarted.fixed_scan_z_m == .291
    restarted._apply_resume_settings(result)
    assert restarted.fixed_scan_z_m == .291
    assert not api.sent


def test_explicit_clear_survives_resume_and_restart(tmp_path):
    task, api, host = setup_task(tmp_path)
    result = partial_map(task, .291)
    api.tool_position[2] = .291
    assert task.capture_scan_height(host)
    task.clear_scan_height(host)
    restarted = TaxelPositionMappingTask(storage_path=task.storage_path, output_directory=None)
    restarted.use_connected_sensor(host)
    restarted._apply_resume_settings(result)
    assert restarted.fixed_scan_z_m is None
    assert "automatic proximity" in restarted._scan_plane_status()
    assert not api.sent


def test_anchor_display_shows_scan_height_recorded_in_its_position_map(tmp_path):
    task = TaxelPositionMappingTask(storage_path=tmp_path / "anchor.json", output_directory=tmp_path / "maps")
    api = _RobotApi()
    host = _Host(api, task)
    assert task.capture_anchor_pose(host)
    api.tool_position[2] = .291
    assert task.capture_scan_height(host)
    recorded = partial_map(task, .291)
    folder = tmp_path / "maps" / "run"
    folder.mkdir(parents=True)
    (folder / "taxel_position_map.json").write_text(json.dumps(recorded))
    profile = task._anchor_profiles[task.active_sensor_key_name]
    profile.pop("scan_plane_choice", None)
    task.anchor_pose.pop("scan_plane_choice", None)
    task.fixed_scan_z_m = None
    task.save_anchor_pose()
    task.active_sensor_key_name = ""
    task.use_connected_sensor(host)
    assert task.fixed_scan_z_m == pytest.approx(.291)
    assert task.anchor_pose["scan_plane_choice"]["z_m"] == pytest.approx(.291)


def test_legacy_map_plane_is_restored_when_no_operator_choice(tmp_path):
    task, api, host = setup_task(tmp_path)
    task._apply_resume_settings(partial_map(task, .291))
    task.use_connected_sensor(host)
    assert task.fixed_scan_z_m == .291
    assert task._scan_plane_source == "saved_map"
    assert not api.sent


def test_new_anchor_invalidates_previous_plane_on_disk(tmp_path):
    task, api, host = setup_task(tmp_path)
    api.tool_position[2] = .291
    assert task.capture_scan_height(host)
    api.tool_position = [.15, .2, .32]
    assert task.capture_anchor_pose(host)
    assert task.fixed_scan_z_m is None
    restarted = TaxelPositionMappingTask(storage_path=task.storage_path, output_directory=None)
    assert restarted.fixed_scan_z_m is None
    assert "scan_plane_choice" not in restarted.anchor_pose
    assert any("previous scan plane (Z 291.00 mm)" in message for message in host.messages)


def test_other_sensor_never_inherits_plane_and_switch_back_restores_it(tmp_path, monkeypatch):
    import phd.ui.experiment_tasks as module
    task, api, host = setup_task(tmp_path)
    old_key = task.active_sensor_key_name
    api.tool_position[2] = .291
    assert task.capture_scan_height(host)
    monkeypatch.setattr(module, "connected_calibration_sensor", lambda h: ("sensor:other", "Other", 8, 10))
    task.use_connected_sensor(host)
    assert task.fixed_scan_z_m is None
    assert task.anchor_pose is None
    monkeypatch.setattr(module, "connected_calibration_sensor", lambda h: (old_key, "Original", 8, 10))
    task.use_connected_sensor(host)
    assert task.fixed_scan_z_m == .291


def test_resume_refuses_centres_saved_at_another_scan_height(tmp_path):
    task, api, host = setup_task(tmp_path)
    result = partial_map(task)
    result["positions"][0]["tool_position_m"][2] = .278
    api.tool_position[2] = .291
    assert task.capture_scan_height(host)
    assert task.request_resume(result)[0]
    assert not task.on_start(host)
    assert not api.sent
    assert any(
        "scanned at Z 278.0 mm, but the scan height is now Z 291.0 mm (-13.0 mm)" in message
        for message in host.messages
    )


def test_resume_at_the_same_scan_height_is_not_refused():
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    task.fixed_scan_z_m = .2915
    result = dict(settings={"anchor_taxel": 35}, positions=[
        dict(taxel_index=35, status="measured", tool_position_m=[.1, .2, .290]),
    ])
    assert task._resume_scan_plane_mismatch(result) == ""
    task.fixed_scan_z_m = None
    result["positions"][0]["tool_position_m"][2] = .250
    assert task._resume_scan_plane_mismatch(result) == ""


def test_resume_different_anchor_refuses_instead_of_silently_using_height(tmp_path):
    task, api, host = setup_task(tmp_path)
    result = partial_map(task)
    api.tool_position = [.12, .2, .3]
    assert task.capture_anchor_pose(host)
    api.tool_position[2] = .291
    assert task.capture_scan_height(host)
    assert task.request_resume(result)[0]
    assert not task.on_start(host)
    assert not api.sent
    assert any("different remembered anchor" in message for message in host.messages)


def test_changed_layout_with_same_sensor_key_does_not_restore_old_plane(tmp_path, monkeypatch):
    import phd.ui.experiment_tasks as module
    task, api, host = setup_task(tmp_path)
    api.tool_position[2] = .291
    assert task.capture_scan_height(host)
    key = task.active_sensor_key_name
    monkeypatch.setattr(module, "connected_calibration_sensor", lambda h: (key, "Reshaped", 5, 5))
    task.use_connected_sensor(host)
    assert task.fixed_scan_z_m is None


@pytest.mark.parametrize("different", ["sensor", "shape"])
def test_resume_choice_never_crosses_sensor_identity_or_layout(tmp_path, different):
    task, api, host = setup_task(tmp_path)
    result = partial_map(task, .289)
    api.tool_position[2] = .291
    assert task.capture_scan_height(host)
    if different == "sensor":
        result["sensor_identity"] = "sensor:other"
    else:
        result["settings"]["sensor_rows"] = 10
        result["settings"]["sensor_columns"] = 8
    assert task._resume_scan_plane_choice(result) is None


def test_resume_after_live_sensor_switch_does_not_overwrite_old_profile(tmp_path, monkeypatch):
    import phd.ui.experiment_tasks as module
    task, api, host = setup_task(tmp_path)
    api.tool_position[2] = .291
    assert task.capture_scan_height(host)
    old_key = task.active_sensor_key_name
    old_profile = copy.deepcopy(task._anchor_profiles[old_key])
    result = partial_map(task)
    result["sensor_identity"] = "sensor:other"
    result["anchor_pose"].pop("scan_plane_choice", None)
    result["anchor_pose"]["tool_position_m"][0] = .11
    monkeypatch.setattr(module, "connected_calibration_sensor", lambda h: ("sensor:other", "Other", 8, 10))
    api.tool_position[2] = .3
    assert task.request_resume(result)[0]
    assert task.on_start(host)
    assert task.active_sensor_key_name == "sensor:other"
    assert task.fixed_scan_z_m is None
    assert task._anchor_profiles[old_key] == old_profile


@pytest.mark.parametrize("z", [.291, .294, .288])
def test_resume_replans_start_and_discards_strength_from_another_plane(tmp_path, z):
    task, api, host = setup_task(tmp_path)
    result = partial_map(task, .290)
    api.tool_position[2] = z
    assert task.capture_scan_height(host)
    task._apply_resume_settings(result)
    task._restore_resume_state(result)
    assert task._probe_start_z == pytest.approx(min(.3, z + .005))
    assert task._probe_start_z > z
    assert task._fixed_scan_reference_signal is None
    assert task._positions[35] == result["positions"][0]
    history = task._resume_history[-1]
    assert history["previous_fixed_scan_z_mm"] == 290.
    assert history["fixed_scan_z_mm"] == pytest.approx(z * 1000.)
    assert not api.sent


def test_same_plane_keeps_saved_guard_reference(tmp_path):
    task, api, host = setup_task(tmp_path)
    result = partial_map(task, .291)
    task._apply_resume_settings(result)
    task._restore_resume_state(result)
    assert task._fixed_scan_reference_signal == 300.


def test_scan_plane_persistence_failure_is_visible(tmp_path, monkeypatch):
    task, api, host = setup_task(tmp_path)
    api.tool_position[2] = .291
    def fail():
        task.last_persistence_error = "disk write failed"
        return False
    monkeypatch.setattr(task, "save_anchor_pose", fail)
    assert not task.capture_scan_height(host)
    assert task.fixed_scan_z_m == .291
    assert any("could not be saved for restart" in message for message in host.messages)
    assert not api.sent


@pytest.mark.parametrize("z", [.299, .19, float("nan")])
def test_bad_height_is_not_remembered(tmp_path, z):
    task, api, host = setup_task(tmp_path)
    api.tool_position[2] = z
    assert not task.capture_scan_height(host)
    assert task.fixed_scan_z_m is None
    assert not api.sent


def test_stale_feedback_is_not_remembered(tmp_path):
    task, api, host = setup_task(tmp_path)
    api.tool_position[2] = .291
    api.current_feedback_state_time = time.monotonic() - 3.
    assert not task.capture_scan_height(host)
    assert task.fixed_scan_z_m is None


def test_full_resume_start_uses_marked_plane_and_keeps_force_guard(tmp_path):
    task, api, host = setup_task(tmp_path)
    result = partial_map(task)
    api.tool_position[2] = .291
    assert task.capture_scan_height(host)
    api.tool_position[2] = .3
    assert task.request_resume(result)[0]
    assert task.on_start(host)
    assert task.fixed_scan_z_m == .291
    assert task._probe_start_z == pytest.approx(.296)
    assert task._move_action == "resume_anchor_arrived"
    assert any("fixed scan plane Z 291.00" in message for message in host.messages)
    before = len(api.sent)
    host.ui_ros._force_meter_last_raw_newtons = (task.force_guard_g + 1.) * task.NEWTONS_PER_GRAM
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    _tick(task, host)
    assert not task._active
    assert len(api.sent) == before


def test_resume_reaches_new_plane_and_collects_new_reference_in_simulation(tmp_path, monkeypatch):
    task, api, host = setup_task(tmp_path, _SimRobotApi())
    result = partial_map(task)
    api.tool_position[2] = .291
    assert task.capture_scan_height(host)
    api.tool_position[2] = .3
    task.move_settle_sec = 0.
    reached = []
    monkeypatch.setattr(task, "_start_fixed_plane_scan", lambda h, pose: reached.append(np.array(pose)))
    assert task.request_resume(result)[0]
    assert task.on_start(host)
    for _ in range(15000):
        if reached or not task._active:
            break
        api.step(.05)
        frame = np.zeros(80)
        frame[35] = -max(0., (.296 - api.tool_position[2]) * 50000.)
        host.ui_ros.sensor_functions._data.diffDataAve = frame.reshape(10, 8).T
        host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
        _tick(task, host)
    assert reached, host.messages[-6:]
    assert reached[0][2] == pytest.approx(.291, abs=.0001)
    assert task._fixed_scan_reference_signal > 100.
    assert task._working_probe_boundary["source"] == "operator_fixed_scan_height"
    assert min(move[0][2] for move in api.sent) >= .291 - 1e-9
