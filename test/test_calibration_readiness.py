"""Loading setup planning never moves hardware or silently expands limits."""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from phd.ui.calibration_readiness import (
    loading_plan, loading_plan_error, suggested_loading_plan, mapped_descent, travel_error,
)
from phd.ui.experiment_tasks import CalibrationTask, FullTaxelCalibrationTask
from test_experiment_tasks import _Host, _RobotApi, _saved_full_calibration_map


def deep_map(depth_mm=70.):
    result = _saved_full_calibration_map()
    for record in result["positions"]:
        record["tool_position_m"][2] = .3 - depth_mm / 1000.
    return result


def full_task(depth_mm=70.):
    task = FullTaxelCalibrationTask(output_directory=None)
    assert task.load_position_map(deep_map(depth_mm))[0]
    return task


def test_map_is_a_travel_lower_bound_not_a_surface_height():
    task = full_task()
    before = (task.max_travel_m, task.timeout_sec, task.contact_threshold_g)
    assert mapped_descent(task)[0] == pytest.approx(70.)
    assert "cannot complete loading" in travel_error(task)
    plan = loading_plan(task)
    assert plan["suggested_travel_mm"] == 75.
    proposed = loading_plan(task, travel_mm=75.)
    assert 600 < proposed["suggested_timeout_sec"] <= 7200
    assert (task.max_travel_m, task.timeout_sec, task.contact_threshold_g) == before
    task.max_travel_m = .075
    assert not travel_error(task)
    task.repeat_count = 5
    assert loading_plan(task)["suggested_timeout_sec"] == proposed["suggested_timeout_sec"]  # per trial


@pytest.mark.parametrize("full", [False, True])
def test_impossible_travel_rejected_before_connection_or_robot_motion(full):
    task = full_task() if full else CalibrationTask(storage_path=None, output_directory=None)
    if not full:
        task.position_map = FullTaxelCalibrationTask.prepare_position_map(deep_map())
        task.use_mapped_taxel(task.position_map["positions"][0], rows=8, columns=10)
    api = _RobotApi()
    host = _Host(api, task)
    assert not task.on_start(host)
    assert "mapped non-contact" in host.messages[-1]
    assert not api.sent and not api.scripts and not api.velocity_commands
    assert task._phase != "connecting_force_meter"


def test_single_uses_selected_taxel_and_its_start_not_whole_map():
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.position_map = FullTaxelCalibrationTask.prepare_position_map(deep_map())
    task.position_map["positions"][1]["tool_position_m"][2] = .21
    assert mapped_descent(task) is None
    task.use_mapped_taxel(task.position_map["positions"][0], rows=8, columns=10)
    assert mapped_descent(task)[0] == pytest.approx(70.)
    task.use_mapped_taxel(task.position_map["positions"][1], rows=8, columns=10)
    assert mapped_descent(task)[0] == pytest.approx(90.)
    task.initial_position["tool_position_m"][2] = .29
    assert mapped_descent(task)[0] == pytest.approx(80.)


def test_plan_tracks_steps_dwell_project_speed_and_reports_unattainable_limits():
    task = full_task()
    task.max_travel_m = .075
    original = loading_plan(task)
    task.step_loading_mm = .005
    assert loading_plan(task)["suggested_timeout_sec"] > original["suggested_timeout_sec"]
    task.step_dwell_sec = 10.
    assert loading_plan(task)["suggested_timeout_sec"] > 7200.
    task = full_task(99.)
    assert loading_plan(task)["suggested_travel_mm"] > 100.
    task = full_task()
    assert loading_plan(task, project_speed_percent=5)["suggested_timeout_sec"] > loading_plan(task, project_speed_percent=100)["suggested_timeout_sec"]
    task.protocol = "proximity_profile"
    assert loading_plan(task) is None


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("motion_mode", ["stepwise", "velocity"])
def test_explicit_auto_mode_plans_once_on_run_without_changing_force(full, motion_mode):
    task = full_task() if full else CalibrationTask(storage_path=None, output_directory=None)
    if not full:
        task.position_map = FullTaxelCalibrationTask.prepare_position_map(deep_map())
        task.use_mapped_taxel(task.position_map["positions"][0], rows=8, columns=10)
    task.loading_auto_plan = True
    api = _RobotApi()
    api.get_project_speed_percent = lambda: 10.
    host = _Host(api, task)
    task.motion_mode = motion_mode
    before = (task.contact_threshold_g, task.approach_speed_m_s, task.repeat_count)
    # Stop at the acquisition boundary: planning itself must not issue motion.
    task._start_calibration_cycle = lambda host: True
    task._begin_force_meter_connection = lambda host: True
    assert task.on_start(host)
    assert task.max_travel_m == .075
    assert task.timeout_sec == suggested_loading_plan(task, project_speed_percent=10.)["suggested_timeout_sec"]
    assert task._loading_plan["limits_mode"] == "automatic"
    assert task._loading_plan["project_speed_percent"] == 10.
    assert before == (task.contact_threshold_g, task.approach_speed_m_s, task.repeat_count)
    assert not api.sent and not api.scripts and not api.velocity_commands
    previous = (task.max_travel_m, task.timeout_sec)
    task._test_active = True
    task.loading_extra_travel_mm = 10.
    assert not task.on_start(host)
    assert (task.max_travel_m, task.timeout_sec) == previous


@pytest.mark.parametrize("case", ["travel", "timeout", "missing_map", "allowance"])
def test_invalid_auto_plan_never_starts_or_partially_applies(case):
    task = full_task(99. if case == "travel" else 70.)
    task.loading_auto_plan = True
    if case == "timeout":
        task.step_dwell_sec = 30.
    elif case == "missing_map":
        task.position_map = None
    elif case == "allowance":
        task.loading_extra_travel_mm = 21.
    previous = (task.max_travel_m, task.timeout_sec)
    api = _RobotApi()
    host = _Host(api, task)
    task.motion_mode = "stepwise"
    assert not task.on_start(host)
    assert "Automatic loading plan blocked" in host.messages[-1]
    assert (task.max_travel_m, task.timeout_sec) == previous
    assert task._phase != "connecting_force_meter"
    assert not api.sent and not api.scripts and not api.velocity_commands


def test_auto_planning_ignores_previous_failure_and_other_protocols():
    task = full_task()
    expected = suggested_loading_plan(task)
    task._stop_reason = "max_travel"
    task._contact_z_by_taxel = {35: -.1}
    assert suggested_loading_plan(task) == expected
    assert not loading_plan_error(expected)
    task.protocol = "proximity_profile"
    assert suggested_loading_plan(task) is None


@pytest.mark.parametrize("mode,size", [("far", .0002), ("near", .00005), ("loading", .00002), ("near", 0.)])
def test_downward_step_and_fast_jump_targets_cannot_cross_travel_limit(mode, size):
    task = CalibrationTask(storage_path=None, output_directory=None)
    api = _RobotApi()
    host = _Host(api, task)
    task.initial_position = {"tool_quaternion_wxyz": [1., 0., 0., 0.]}
    task._start_tool_position = [.1, .2, .3]
    task.max_travel_m = .010
    assert task._send_step_move(host, time.monotonic(), [.1, .2, .289], size, mode)
    assert api.sent[-1][0][2] == pytest.approx(.290)
    count = len(api.sent)
    task._next_approach_step(host, time.monotonic(), api.tool_position)
    assert len(api.sent) == count
    assert task._stop_reason == "max_travel"


def test_readiness_panel_applies_only_limits_and_never_acts_during_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = r'''
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QDoubleSpinBox, QSpinBox, QComboBox
from phd.ui.calibration_protocol_panel import build_protocol_panel
from phd.ui.calibration_readiness_panel import CalibrationReadinessPanel
from test_calibration_readiness import full_task, deep_map
app = QApplication([])
task = full_task()
holder = QWidget()
layout = QVBoxLayout(holder)
force, travel, timeout, speed = [QDoubleSpinBox(holder) for _ in range(4)]
force.setRange(.1, 800); force.setValue(task.contact_threshold_g)
travel.setRange(.1, 100); travel.setValue(task.max_travel_m * 1000)
timeout.setRange(10, 7200); timeout.setValue(task.timeout_sec)
speed.setRange(.01, 2); speed.setValue(task.approach_speed_m_s * 1000)
repeat = QSpinBox(holder); repeat.setRange(1, 1000); repeat.setValue(1)
for spin, attr, scale in ((force, 'contact_threshold_g', 1), (travel, 'max_travel_m', .001),
                          (timeout, 'timeout_sec', 1), (speed, 'approach_speed_m_s', .001)):
    spin.valueChanged.connect(lambda value, attr=attr, scale=scale: setattr(task, attr, value * scale))
protocol = build_protocol_panel(task, force, repeat, speed, allow_profile=False)
layout.addWidget(protocol)
panel = CalibrationReadinessPanel(task, travel, timeout, protocol, [force, speed, repeat], auto_plan=False)
layout.insertWidget(0, panel)
assert 'BLOCKED' in panel.status.text()
assert '1,000 g target is not allowed' in panel.proposal.text()
assert panel.apply_button.isEnabled()
before = (task.contact_threshold_g, task.approach_speed_m_s, task.repeat_count, task.motion_mode)
panel.apply_button.click()
assert task.max_travel_m == .075 and task.timeout_sec > 600
assert 'BLOCKED' not in panel.status.text() and 'TIME WARNING' not in panel.status.text()
assert before == (task.contact_threshold_g, task.approach_speed_m_s, task.repeat_count, task.motion_mode)
assert task._phase == 'idle' and not task._test_active
panel.extra.setValue(10)
assert task.loading_extra_travel_mm == 10 and task.max_travel_m == .075
task._test_active = True
panel.refresh()
assert not panel.apply_button.isEnabled()
panel.apply_suggestions()
assert task.max_travel_m == .075
task._test_active = False; task._phase = 'connecting_force_meter'
panel.apply_suggestions(); assert task.max_travel_m == .075
task._phase = 'idle'
panel.extra.setValue(5)
assert task.load_position_map(deep_map(99.))[0]
panel.refresh()
assert not panel.apply_button.isEnabled() and 'will not be silently clipped' in panel.proposal.text()
assert task.load_position_map(deep_map())[0]
panel.refresh()
protocol.findChild(QComboBox, 'characterisationProtocol').setCurrentIndex(1)
assert panel.isHidden()
holder.close()
'''
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "test")))
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ui_auto_plans_both_tabs_manual_override_busy_and_invalid_maps():
    root = Path(__file__).resolve().parents[1]
    script = r'''
import os, tempfile
from types import SimpleNamespace
config = tempfile.TemporaryDirectory()
os.environ['XDG_CONFIG_HOME'] = config.name
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QComboBox
from phd.ui import experiment_tasks
from phd.ui.ui_initial import MyMainWindow
from test_calibration_readiness import deep_map
app = QApplication([])
QTimer.singleShot = lambda *a, **k: None
window = MyMainWindow()
window.ui_ros = SimpleNamespace(sensor_functions=SimpleNamespace(is_connected=True, n_row=8, n_col=10))
single, full = window._calibration_task(), window._full_taxel_calibration_task()
panel = window.full_taxel_calibration_readiness_panel
single_panel = window.calibration_readiness_panel
def no_motion(*a, **k):
    raise AssertionError('Planning must not start motion')
window._move_above_mapped_taxel = no_motion
window._start_sidebar_control = no_motion
before = (full.contact_threshold_g, full.approach_speed_m_s, full.repeat_count, full.route_force_guard_g)
window._show_taxel_position_map(deep_map(), source_path='/example/map.json')
assert full.loading_auto_plan and panel.auto_plan.isChecked()
assert full.max_travel_m == .075 and full.timeout_sec > 600
assert window.full_taxel_calibration_max_travel_spin.value() == 75.
assert window.full_taxel_calibration_max_travel_spin.isReadOnly()
assert 'Automatic limits: 75.0 mm' in panel.status.text()
assert panel.apply_button.isHidden()
assert single.initial_position is None and single.max_travel_m == .03
single.use_mapped_taxel(single.position_map['positions'][0], rows=8, columns=10)
window._refresh_calibration_controls()
assert single.max_travel_m == .075 and single_panel.travel_spin.value() == 75.
assert before == (full.contact_threshold_g, full.approach_speed_m_s, full.repeat_count, full.route_force_guard_g)
panel.extra.setValue(7)
assert full.max_travel_m == .077
panel.auto_plan.setChecked(False)
assert not full.loading_auto_plan and not panel.travel_spin.isReadOnly()
panel.travel_spin.setValue(80)
panel.timeout_spin.setValue(4000)
panel.extra.setValue(5)
panel.refresh()
assert full.max_travel_m == .080 and full.timeout_sec == 4000
panel.auto_plan.setChecked(True)
assert full.max_travel_m == .075
limits = (full.max_travel_m, full.timeout_sec)
full._test_active = True
window._sidebar_active_task = full
full.loading_extra_travel_mm = 10
panel.refresh()
panel.apply_suggestions()
panel.auto_plan.setChecked(False)  # even a programmatic toggle is refused while busy
assert full.loading_auto_plan and panel.auto_plan.isChecked()
assert (full.max_travel_m, full.timeout_sec) == limits
assert not panel.auto_plan.isEnabled() and not panel.extra.isEnabled()
full._test_active = False
full._phase = 'connecting_force_meter'
panel.refresh()
assert (full.max_travel_m, full.timeout_sec) == limits
full._phase = 'idle'
full.loading_extra_travel_mm = 5
window._sidebar_active_task = object()  # unrelated robot controller
panel.refresh()
assert (full.max_travel_m, full.timeout_sec) == limits
window._sidebar_active_task = None
panel.refresh()
assert panel.auto_plan.isEnabled()
window._show_taxel_position_map(deep_map(99.), source_path='/example/too-deep.json')
assert 'AUTO PLAN BLOCKED' in panel.status.text()
assert (full.max_travel_m, full.timeout_sec) == limits  # no 104 -> 100 clipping
window._show_taxel_position_map(deep_map(50.), source_path='/example/other-map.json')
assert full.max_travel_m == .055
mode = window.full_taxel_calibration_protocol_selector.findChild(QComboBox, 'characterisationProtocol')
mode.setCurrentIndex(mode.findData('hold_recovery'))
assert panel.isHidden() and not panel.travel_spin.isReadOnly()
panel.travel_spin.setValue(60)
panel.refresh()
assert full.max_travel_m == .060
mode.setCurrentIndex(mode.findData('standard'))
assert full.max_travel_m == .055
assert window._sidebar_active_task is None
window.ui_ros = None
window.close()
config.cleanup()
'''
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "test")))
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
