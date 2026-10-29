"""Resume uses simulated robot/force/tactile feedback only; never real hardware."""
import copy
import json
import time
from pathlib import Path

import pytest

from phd.ui import full_taxel_resume as resume
from phd.ui.experiment_tasks import FullTaxelCalibrationTask
from test_experiment_tasks import (
    _Host, _RobotApi, _saved_full_calibration_map, _settle_full_taxel_route,
    _complete_full_taxel_contact_trial, _settle_post_save_feedback,
)


def stopped_run(tmp_path, passes=1):
    task = FullTaxelCalibrationTask(output_directory=tmp_path)
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps(_saved_full_calibration_map()))
    assert task.load_position_map(json.loads(map_path.read_text()), map_path)[0]
    api = _RobotApi()
    host = _Host(api, task)
    task.repeat_count = passes
    task.force_baseline_duration_sec = 0.
    task.force_baseline_min_samples = 1
    task.return_settle_sec = task.move_settle_sec = 0.
    task.contact_threshold_n = .1
    assert task.on_start(host)
    _settle_full_taxel_route(task, host)
    _complete_full_taxel_contact_trial(task, host, api, .2)
    task.on_stop(host, show_result=False)
    assert task._run_lock is None
    return task._full_directory, task, host


def staged(folder):
    task = FullTaxelCalibrationTask(output_directory=folder.parent)
    host = _Host(_RobotApi(), task)
    ok, message = task.load_resume(folder / resume.SUMMARY)
    assert ok, message
    task.resume_setup_confirmed = True
    host.ui_ros._force_meter_last_raw_newtons = 0.
    host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    return task, host


def finish_next(task, host):
    _settle_post_save_feedback(task, host)
    _settle_full_taxel_route(task, host)
    if task._phase == "full_recalibrating":
        sensor = host.ui_ros.sensor_functions
        sensor._sensor_calibration_in_progress = False
        sensor.is_connected = True
        task.on_tick(host)
        _settle_full_taxel_route(task, host)
    assert task._phase == "force_baseline"
    index = task.taxel_index
    host.ui_ros._force_meter_last_raw_newtons = 0.
    _complete_full_taxel_contact_trial(task, host, host.ui_ros.robot_api, .15)
    return index


def test_resume_preserves_complete_bytes_and_returns_to_anchor_with_fresh_baseline(tmp_path):
    folder, old, _host = stopped_run(tmp_path)
    originals = {p: p.read_bytes() for p in (folder / "taxel_35").glob("trial_*")}
    task, host = staged(folder)
    assert task.resume_plan["remaining"] == [(1, 43)]
    assert not host.ui_ros.robot_api.sent
    task._force_baseline_n = 42.  # Never reuse an old session's zero.
    assert task.on_start(host)
    assert task._full_directory == folder
    assert task._completed_taxels == [35]
    assert task._phase == "waiting_post_save_feedback"
    _settle_post_save_feedback(task, host)
    _settle_full_taxel_route(task, host)
    assert task._phase == "full_recalibrating"
    sensor = host.ui_ros.sensor_functions
    sensor._sensor_calibration_in_progress = False
    sensor.is_connected = True
    task.on_tick(host)
    _settle_full_taxel_route(task, host)
    assert task.taxel_index == 43 and task._force_baseline_n is None
    assert finish_next(task, host) == 43
    host.ui_ros._force_meter_last_raw_newtons = 0.
    _settle_full_taxel_route(task, host)
    _settle_full_taxel_route(task, host)
    assert task.last_result["status"] == "complete"
    assert task.last_result["completed_taxels"] == 2
    assert all(p.read_bytes() == data for p, data in originals.items())
    manifest = resume.read_json(folder / resume.MANIFEST)
    assert manifest["status"] == "stopped" and len(manifest["resume_history"]) == 1
    assert resume.read_json(folder / "taxel_43/trial_0001.json")["measurement_quality"]["resume_session"] == 1
    assert not task.load_resume(folder)[0]  # Entire run now finished.


def test_resume_multiple_passes_skips_only_saved_pairs_and_can_stop_again(tmp_path):
    folder, _, _ = stopped_run(tmp_path, passes=2)
    task, host = staged(folder)
    assert task.resume_plan["remaining"] == [(1, 43), (2, 35), (2, 43)]
    assert task.on_start(host)
    assert finish_next(task, host) == 43
    task.on_stop(host, show_result=False)
    task2, host2 = staged(folder)
    assert task2.resume_plan["remaining"] == [(2, 35), (2, 43)]
    assert task2.on_start(host2)
    assert finish_next(task2, host2) == 35
    host2.ui_ros._force_meter_last_raw_newtons = 0.
    assert finish_next(task2, host2) == 43
    task2.on_stop(host2, show_result=False)
    assert len(resume.read_json(folder / resume.MANIFEST)["resume_history"]) == 2
    assert len(list(folder.glob("taxel_*/trial_*.csv"))) == 4


def test_interrupted_attempt_archived_only_on_explicit_start(tmp_path):
    folder, _, _ = stopped_run(tmp_path)
    # Reuse the valid trial's schema to represent an interrupted second taxel.
    record = resume.read_json(folder / "taxel_35/trial_0001.json")
    manifest = resume.read_json(folder / resume.MANIFEST)
    target = manifest["position_map"]["positions"][1]
    record.update(taxel_index=43, batch_id=f"{manifest['run_id']}_taxel43")
    record["initial_position"]["tool_position_m"] = target["calibration_start_position_m"]
    record["measurement_quality"]["trial_completed"] = False
    partial = folder / "taxel_43"
    partial.mkdir()
    (partial / "trial_0001.json").write_text(json.dumps(record))
    (partial / "trial_0001.csv").write_text("unfinished")
    task, host = staged(folder)
    assert (partial / "trial_0001.csv").read_text() == "unfinished"
    assert not (folder / "resume_history").exists()
    assert task.on_start(host)
    archived = list(folder.glob("resume_history/*/interrupted_attempts/taxel_43/trial_0001.csv"))
    assert len(archived) == 1 and archived[0].read_text() == "unfinished"
    assert not (partial / "trial_0001.csv").exists()
    task.on_stop(host, show_result=False)


def test_resume_lifts_to_clearance_before_moving_toward_the_anchor(tmp_path):
    folder, _, _ = stopped_run(tmp_path)
    task, host = staged(folder)
    api = host.ui_ros.robot_api
    api.tool_position[2] -= .01
    assert task.on_start(host)
    assert task._phase == "waiting_post_save_feedback"
    assert not api.sent
    _settle_post_save_feedback(task, host)
    assert api.sent[0][0] == pytest.approx([0.1, 0.2, 0.3])
    assert api.tool_position[2] == pytest.approx(0.3)


@pytest.mark.parametrize("change", ["sensor", "force", "settings", "confirmation", "orientation", "workspace"])
def test_resume_safety_blocks_before_any_motion(tmp_path, change):
    folder, _, _ = stopped_run(tmp_path)
    task, host = staged(folder)
    api = host.ui_ros.robot_api
    if change == "sensor":
        host.ui_ros.sensor_functions.n_row = 5
    elif change == "force":
        host.ui_ros._force_meter_last_raw_newtons = 1.
    elif change == "settings":
        task.max_travel_m += .01
    elif change == "confirmation":
        task.resume_setup_confirmed = False
    elif change == "orientation":
        api.tool_quaternion = [0., 1., 0., 0.]
    else:
        api.tool_position[0] += 1.
    assert not task.on_start(host)
    assert not api.sent and not api.velocity_commands
    assert not (folder / "resume_history").exists()


def test_changed_files_and_active_writer_refused(tmp_path):
    folder, _, _ = stopped_run(tmp_path)
    task, host = staged(folder)
    lock = resume.acquire_run_lock(folder)
    try:
        assert not task.on_start(host)
        assert not host.ui_ros.robot_api.sent
    finally:
        lock.close()
    task, host = staged(folder)
    path = folder / "taxel_35/trial_0001.csv"
    path.write_text(path.read_text() + "\n")
    assert not task.on_start(host)
    assert "changed" in host.messages[-1]
    assert not (folder / "resume_history").exists()


def test_legacy_run_and_incomplete_data_guards(tmp_path):
    folder, task, _ = stopped_run(tmp_path)
    (folder / resume.MANIFEST).unlink()  # Emulate the currently running pre-resume app.
    restored, _ = staged(folder)
    assert restored.resume_plan["legacy"]
    assert restored.max_travel_m == task.max_travel_m
    assert restored.contact_threshold_g == pytest.approx(task.contact_threshold_g)
    assert restored.motion_mode == task.motion_mode
    (folder / "taxel_35/trial_0001.csv").unlink()
    assert "missing" in restored.load_resume(folder)[1]
    (folder / resume.SUMMARY).unlink()
    assert "Stop the experiment normally" in restored.load_resume(folder)[1]


def test_map_snapshot_not_latest_external_map_and_settings_locked(tmp_path):
    folder, old, _ = stopped_run(tmp_path)
    Path(old.position_map_path).write_text("not JSON anymore")
    task, _ = staged(folder)
    assert not task.load_position_map(_saved_full_calibration_map())[0]
    expected = copy.deepcopy(task.resume_plan["position_map"])
    assert task.position_map == expected
    task.clear_resume()
    assert task.resume_plan is None
    assert task.start_button_label == "Run Full Taxel Calibration"
    assert task.load_position_map(_saved_full_calibration_map())[0]


def test_live_manifest_refuses_stale_stopped_summary(tmp_path):
    folder, _, _ = stopped_run(tmp_path)
    task, host = staged(folder)
    assert task.on_start(host)
    other = FullTaxelCalibrationTask(output_directory=tmp_path)
    assert "active" in other.load_resume(folder)[1]
    task.on_stop(host, show_result=False)


def test_resume_rechecks_force_after_disk_io_before_routing(tmp_path):
    folder, _, _ = stopped_run(tmp_path)
    task, host = staged(folder)
    assert task.on_start(host)
    assert task._phase == "waiting_post_save_feedback"
    host.ui_ros._force_meter_last_raw_newtons = 1.
    _settle_post_save_feedback(task, host)
    assert task._phase == "idle"
    assert task.last_result["stop_reason"] == "resume_preflight_changed"
    assert not host.ui_ros.robot_api.sent
    assert resume.read_json(folder / resume.MANIFEST)["status"] == "stopped"


@pytest.mark.parametrize("problem", ["map_changed", "settings_changed", "corrupt_trial", "fixed_baseline"])
def test_incompatible_saved_run_fails_closed(tmp_path, problem):
    folder, _, _ = stopped_run(tmp_path)
    manifest = resume.read_json(folder / resume.MANIFEST)
    if problem == "map_changed":
        manifest["position_map"]["anchor_pose"]["tool_position_m"][0] += .01
    elif problem == "settings_changed":
        manifest["settings"]["contact_threshold_n"] += .2
    elif problem == "fixed_baseline":
        manifest["settings"]["recalibrate_between_trials"] = False
        manifest["settings_digest"] = resume.digest(manifest["settings"])
    else:
        (folder / "taxel_35/trial_0001.json").write_text("{")
    resume.atomic_json(folder / resume.MANIFEST, manifest)
    task = FullTaxelCalibrationTask(output_directory=tmp_path)
    assert not task.load_resume(folder)[0]
    assert task.resume_plan is None


def test_main_window_resume_load_clear_and_autoload_protection(tmp_path):
    from test_research_workspace import run_qt
    folder, _, _ = stopped_run(tmp_path)
    run_qt('''
import tempfile
import os
from types import SimpleNamespace
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox
from phd.ui.ui_initial import MyMainWindow
from phd.ui.full_taxel_resume import settings_snapshot
config = tempfile.TemporaryDirectory()
os.environ["XDG_CONFIG_HOME"] = config.name
app = QApplication([])
original = QTimer.singleShot
QTimer.singleShot = lambda *args: None
window = MyMainWindow()
QTimer.singleShot = original
assert window.select_sidebar_task("calibration")
window.calibration_mode_tabs.setCurrentIndex(window._calibration_full_taxel_tab_index)
window._choose_experiment_file = lambda *args: SOURCE
QMessageBox.warning = lambda *args: (_ for _ in ()).throw(AssertionError(str(args)))
task = window._full_taxel_calibration_task()
window._load_full_taxel_calibration_resume()
assert task.resume_plan is not None
assert window.sidebar_btn_start.text() == "Resume Full Taxel Calibration"
assert not window.sidebar_btn_start.isEnabled()  # No live sensor / confirmation.
assert "1 completed" in window.full_taxel_calibration_resume_info.text()
assert not window.full_taxel_calibration_resume_info.isHidden()
assert not window.full_taxel_calibration_speed_spin.isEnabled()
assert not window.full_taxel_calibration_load_map_button.isEnabled()
assert not window.full_taxel_calibration_readiness_panel.isEnabled()
before = settings_snapshot(task)
saved_map = task.position_map
window._auto_load_connected_sensor_map()
assert task.position_map is saved_map
assert settings_snapshot(task) == before
window.full_taxel_calibration_resume_clear.click()
assert task.resume_plan is None
assert window.full_taxel_calibration_load_map_button.isEnabled()
assert window.full_taxel_calibration_speed_spin.isEnabled()
assert window.full_taxel_calibration_resume_info.isHidden()
assert window.sidebar_btn_start.text() == "Run Full Taxel Calibration"
window.close()
'''.replace("SOURCE", repr(str(folder / resume.SUMMARY))))


def test_resume_ui_sync_keeps_task_values_and_locks_settings():
    from test_research_workspace import run_qt
    run_qt('''
from types import SimpleNamespace
from PyQt5.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox, QComboBox
from phd.ui.calibration_protocol_panel import build_protocol_panel
from phd.ui.calibration_readiness_panel import CalibrationReadinessPanel
from phd.ui.experiment_tasks import FullTaxelCalibrationTask
from phd.ui.full_taxel_resume import settings_snapshot
app = QApplication([])
task = FullTaxelCalibrationTask(output_directory=None)
force, repeats, speed = QDoubleSpinBox(), QSpinBox(), QDoubleSpinBox()
panel = build_protocol_panel(task, force, repeats, speed, allow_profile=False)
task.protocol = "hold_recovery"
task.contact_hold_sec = 17.5
task.step_dwell_sec = 1.3
task.experiment_notes = "saved notes"
before = settings_snapshot(task)
panel.sync_from_task()
assert settings_snapshot(task) == before
assert panel.findChild(QDoubleSpinBox, "contact_hold_sec").value() == 17.5
assert panel.protocol_selector.findChild(QComboBox, "characterisationProtocol").currentData() == "hold_recovery"
travel, timeout = QDoubleSpinBox(), QDoubleSpinBox()
readiness = CalibrationReadinessPanel(task, travel, timeout, panel, [])
task.resume_plan = {"loaded": True}
task.loading_auto_plan = False
readiness.refresh()
assert not readiness.auto_plan.isEnabled()
assert not readiness.apply_button.isEnabled()
''')
