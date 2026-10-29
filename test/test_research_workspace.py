"""Layout and monitoring are presentation-only: no hardware is needed."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from phd.ui.experiment_run_view import ended_run_message, force_readout


def run_qt(script):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_end_status_keeps_stop_cause_and_save_error_without_duplication():
    detail = "Taxel 189 failed twice: signal 24.5/34.7 units at the descent limit."
    task = SimpleNamespace(_stop_reason="mapping_target_failed", _stop_message=detail)
    saved = "Taxel mapping files saved together in: /tmp/map"
    message = ended_run_message(task, "Control stopped", saved)
    assert "Stop reason: mapping target failed" in message
    assert detail in message and saved in message
    assert ended_run_message(task, detail, message) == message
    error = ended_run_message(task, detail, "Could not save: disk is full")
    assert detail in error and "disk is full" in error
    # A task without an internal detail still retains the message supplied by
    # the controller, even if cleanup posts a different message.
    task = SimpleNamespace(_stop_reason="sensor_stale")
    message = ended_run_message(task, "Tactile frames became stale", saved)
    assert "Tactile frames became stale" in message and saved in message


def test_end_status_can_read_new_saved_diagnostics_and_old_task_results():
    task = SimpleNamespace(last_result={
        "stop_reason": "mapping_target_failed", "stop_message": "Taxel 189: weak signal",
    })
    assert "Taxel 189: weak signal" in ended_run_message(task)
    assert ended_run_message(SimpleNamespace(), "Manual stop") == "Manual stop"
    assert "no stop reason" in ended_run_message(SimpleNamespace())


def test_latest_status_freezes_cause_after_cleanup_and_clears_for_new_run():
    run_qt('''
from types import SimpleNamespace
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QLabel
from phd.ui.experiment_run_view import ExperimentRunView
app = QApplication([])
task = SimpleNamespace(id="taxel_position_mapping", label="Taxel mapping", _phase="moving",
    _stop_reason="mapping_target_failed", _stop_message="Taxel 189 failed twice: 24.5/34.7 sensor units.",
    _output_folder=Path("/tmp/map"))
host = SimpleNamespace(_sidebar_active_task=task, ui_ros=None,
    info_process=QLabel("Returning to safe anchor"), taxel_mapping_progress_label=QLabel("5/324 located"))
view = ExperimentRunView(clock=lambda: 10.)
view.refresh(host)
view.finish(host, message=task._stop_message)
host._sidebar_active_task = None
task._phase = "idle"
host.info_process.setText("Taxel mapping files saved together in: /tmp/map")
view.update_end_details(host)
assert "24.5/34.7" in view.message.text()
assert "mapping target failed" in view.message.text()
assert "files saved" in view.message.text()
assert view.output.text() == "/tmp/map"
assert view.progress.text() == "5/324 located"
before = view.message.text()
host.info_process.setText("Other UI action")
view.refresh(host)
assert view.message.text() == before
task._stop_reason = ""
task._stop_message = ""
task._phase = "probing"
host._sidebar_active_task = task
host.info_process.setText("New run is probing")
assert view.refresh(host)
assert view.message.text() == "New run is probing"
assert view._end_message == ""
view.close()
''')


def test_real_stop_controller_keeps_reason_after_save_and_force_cleanup():
    run_qt('''
from types import SimpleNamespace
from PyQt5.QtWidgets import QApplication, QLabel
from phd.ui.ui_initial import MyMainWindow
from phd.ui.experiment_run_view import ExperimentRunView
app = QApplication([])
host = SimpleNamespace(info_process=QLabel("Running"), _sidebar_control_timer=None,
    _sidebar_active_control_id="test", ui_ros=None,
    taxel_mapping_progress_label=QLabel("5/324 located"))
host.experiment_run_view = ExperimentRunView(clock=lambda: 10.)
host._sigraph_heatmap_record_elapsed = lambda: 0.
host._refresh_experiment_status = lambda: None
host._refresh_taxel_mapping_controls = lambda: None
host._report_run_monitor_error = lambda exc: (_ for _ in ()).throw(exc)
host._stop_sigraph_heatmap_recording = lambda **kwargs: None
host._stop_sigraph_heatmap_playback = lambda **kwargs: None
def save(host, **kwargs):
    host.info_process.setText("Map files saved in /tmp/map")
task = SimpleNamespace(id="taxel_position_mapping", label="Taxel mapping", _phase="moving",
    _stop_reason="mapping_target_failed", _stop_message="Taxel 189 failed twice: weak scan signal.",
    on_stop=save)
host._sidebar_active_task = task
host._disconnect_force_meter_after_task = lambda task: host.info_process.setText("Force transmitter disconnected")
MyMainWindow._stop_sidebar_control(host, show_result=False, message=task._stop_message)
assert host._sidebar_active_task is None
assert "weak scan signal" in host.info_process.text()
assert "Map files saved" in host.info_process.text()
assert "Force transmitter disconnected" in host.info_process.text()
assert "weak scan signal" in host.experiment_run_view.message.text()
assert host.experiment_run_view.message.text().count("weak scan signal") == 1
host.experiment_run_view.close()
''')


@pytest.mark.parametrize("value,stamp", [(None, None), (float("nan"), 5), (1, float("inf")), (1, 20)])
def test_force_monitor_rejects_invalid_or_future_samples(value, stamp):
    ui = SimpleNamespace(_force_meter_last_raw_newtons=value, _force_meter_last_sample_monotonic=stamp)
    assert force_readout(ui, None, 10) == ("—", "No valid force sample", "idle")


def test_force_monitor_uses_raw_reading_and_task_staleness_limit():
    ui = SimpleNamespace(_force_meter_last_raw_newtons=0.980665, _force_meter_last_sample_monotonic=10)
    task = SimpleNamespace(force_sample_stale_sec=2)
    assert force_readout(ui, task, 11) == ("100.0 g", "Last sample 1.0 s ago", "active")
    assert force_readout(ui, task, 13) == ("100.0 g", "STALE · Last sample 3.0 s ago", "warning")
    mapping = SimpleNamespace(force_stale_sec=0.2)
    assert force_readout(ui, mapping, 10.5)[2] == "warning"


def test_auto_layout_restores_width_preserves_widgets_and_respects_manual_choice():
    run_qt('''
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QSplitter, QWidget, QTabWidget
from phd.ui.ui_ping import UI
app = QApplication([])
class Workspace(QSplitter):
    workspace_layout_changed = pyqtSignal(str)
    set_workspace_layout_mode = UI.set_workspace_layout_mode
    _automatic_workspace_layout = UI._automatic_workspace_layout
    _apply_workspace_layout = UI._apply_workspace_layout
    _fit_main_control_panel = UI._fit_main_control_panel
    _sync_main_control_panel_handle = UI._sync_main_control_panel_handle
    _set_main_control_panel_visible = UI._set_main_control_panel_visible
    _on_main_control_splitter_moved = UI._on_main_control_splitter_moved
    toggle_main_control_panel = UI.toggle_main_control_panel
view = Workspace(Qt.Horizontal)
view.splitter_1 = QWidget()
view.splitter_2 = QWidget()
view.addWidget(view.splitter_1)
view.addWidget(view.splitter_2)
view.resize(1400, 800)
view.show()
view._workspace_layout_mode = "auto"
view._workspace_layout_target = "split"
view._main_control_panel_last_width = 600
view._main_control_panel_user_sized = True
view._main_control_panel_collapsed = False
view._main_control_panel_state_syncing = False
view.tab_widget = QTabWidget()
for title in ("Sensor", "Robots", "Control", "AI", "Hand", "Tools"):
    view.tab_widget.addTab(QWidget(), title)
view.tools_tab_index, view.ai_tab_index, view.hand_tab_index = 5, 3, 4
view.ai_data_training_tab_index = 1
view.hand_sub_tabs = QTabWidget()
for i in range(2):
    view.hand_sub_tabs.addTab(QWidget(), str(i))
view.setSizes([800, 600])
app.processEvents()
saved_width = view.sizes()[1]
sensor_view = view.splitter_1
view.tab_widget.setCurrentIndex(3)  # AI before lazy construction.
view._apply_workspace_layout()
assert not sensor_view.isHidden()
assert view._workspace_layout_target == "split"
assert abs(view.sizes()[1] - saved_width) <= 5
view.tab_widget.setCurrentIndex(5)  # Tools must not take over the viewport.
view._apply_workspace_layout()
assert not sensor_view.isHidden()
assert abs(view.sizes()[1] - saved_width) <= 5
view.ai_sub_tabs = QTabWidget()
for i in range(2):
    view.ai_sub_tabs.addTab(QWidget(), str(i))
view.tab_widget.setCurrentIndex(3)
for index in (0, 1, 0, 1):  # Run and Data / training keep the same split.
    view.ai_sub_tabs.setCurrentIndex(index)
    view._apply_workspace_layout()
    app.processEvents()
    assert not sensor_view.isHidden()
    assert view._workspace_layout_target == "split"
    assert abs(view.sizes()[1] - saved_width) <= 5
view.tab_widget.setCurrentIndex(4)
view.hand_sub_tabs.setCurrentIndex(1)
view._apply_workspace_layout()
assert sensor_view.isHidden()
view.resize(1000, 700)
view._fit_main_control_panel()
assert sensor_view.isHidden()
view.tab_widget.setCurrentIndex(3)
view._apply_workspace_layout()
app.processEvents()
assert not sensor_view.isHidden()
assert abs(view.sizes()[1] - saved_width) <= 5
view.set_workspace_layout_mode("split")
view.tab_widget.setCurrentIndex(5)
view._apply_workspace_layout()
assert not sensor_view.isHidden()  # Explicit choice persists across tasks.
view.set_workspace_layout_mode("controls")
view.tab_widget.setCurrentIndex(3)
view._apply_workspace_layout()
assert sensor_view.isHidden()  # Only an explicit focus choice fills the window.
view.toggle_main_control_panel()
assert view._workspace_layout_mode == "sensor"
assert view.sizes()[1] == 0
assert not sensor_view.isHidden()
view.toggle_main_control_panel()
assert view._workspace_layout_mode == "split"
assert view.sizes()[1] > 0
view.set_workspace_layout_mode("auto")
assert not sensor_view.isHidden()
view.tab_widget.setCurrentIndex(5)
view._apply_workspace_layout()
assert not sensor_view.isHidden()
assert view.splitter_1 is sensor_view
try:
    view.set_workspace_layout_mode("unknown")
except ValueError:
    pass
else:
    raise AssertionError("Invalid mode was accepted")
''')


def test_run_monitor_freezes_telemetry_and_restarts_same_task_without_commands():
    run_qt('''
from types import SimpleNamespace
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QLabel
from phd.ui.experiment_run_view import ExperimentRunView
app = QApplication([])
now = [10.0]
view = ExperimentRunView(clock=lambda: now[0])
task = SimpleNamespace(id="calibration", label="Calibration", _phase="approaching",
                       _batch_directory=Path("/tmp/experiment"), force_sample_stale_sec=1.0)
host = SimpleNamespace(_sidebar_active_task=task,
    ui_ros=SimpleNamespace(_force_meter_last_raw_newtons=.980665, _force_meter_last_sample_monotonic=9.8),
    calibration_progress_label=QLabel("Trial 2/4 | Approaching"),
    status_calibration_travel=QLabel("Z travel: 4.00 mm | Max: 4.00 mm"),
    info_process=QLabel("Running"))
assert view.refresh(host)
assert view.progress.text() == "Trial 2/4 | Approaching"
assert view.force.text() == "100.0 g"
now[0] = 75
task._phase = "paused_force_meter"
assert not view.refresh(host)
assert view.elapsed.text() == "0 h 01 min 05 s"
assert view.state.state() == "warning"
assert "STALE" in view.force_age.text()
view.finish(host)
task._phase = "idle"
host._sidebar_active_task = None
host.status_calibration_travel.setText("Z travel: — mm | Max: — mm")
host.info_process.setText("Stopped: force unavailable; partial results saved")
view.update_end_details(host)
now[0] = 120
assert not view.refresh(host)
assert view.elapsed.text() == "0 h 01 min 05 s"
assert "4.00" in view.travel.text()
assert "Run ended" in view.state.text()
assert "completed" not in view.state.text().lower()
assert "not live" in view.force_age.text()
assert "partial results" in view.message.text()
assert view.output.text() == "/tmp/experiment"
host._sidebar_active_task = task
task._phase = "connecting_force_meter"
task._batch_directory = None
assert view.refresh(host)
assert view.elapsed.text() == "0 h 00 min 00 s"
assert "Not created" in view.output.text()
host._sidebar_active_task = SimpleNamespace(id="mapped_taxel_move", label="Move", _phase="moving")
assert view.refresh(host)
assert "not reported" in view.travel.text()
assert view.progress.text() == "Experiment is active."
host._sidebar_active_task = SimpleNamespace(id="full_taxel_calibration", label="Full calibration",
    _phase="returning", _full_directory=Path("/tmp/full"), _batch_directory=Path("/tmp/full/taxel_12"))
assert view.refresh(host)
assert view.output.text() == "/tmp/full"
view.finish(host)
host._sidebar_active_task._stop_reason = "manual_stop"
view.update_end_details(host)
assert view.state.text() == "Run ended · manual stop"
''')


def test_experiment_setup_run_tabs_switch_once_and_keep_stop_actions_reachable():
    run_qt('''
from types import SimpleNamespace
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from phd.ui.ui_initial import MyMainWindow
app = QApplication([])
QTimer.singleShot = lambda *_a, **_kw: None
window = MyMainWindow()
window.experiments_window.show()
app.processEvents()
tabs = window.experiment_view_tabs
assert tabs.currentIndex() == 0
assert window.experiment_stop_all_button.defaultAction() is window.action_stop_all_motion
assert not window.experiment_monitor_timer.isActive()
task = SimpleNamespace(id="calibration", label="Calibration", _phase="approaching")
window._sidebar_active_task = task
window._refresh_experiment_status()
assert tabs.currentIndex() == 1
assert window.experiment_monitor_timer.isActive()
assert window.sidebar_btn_stop.isVisible()
assert window.experiment_stop_all_button.isVisible()
tabs.setCurrentIndex(0)
window._refresh_run_monitor()
assert tabs.currentIndex() == 0  # Do not steal focus on every update.
window._sidebar_active_task = None
window._refresh_experiment_status()
assert not window.experiment_monitor_timer.isActive()
assert "Run ended" in window.experiment_run_view.state.text()
window.select_sidebar_task("graph_analysis")
assert tabs.currentIndex() == 0
window.select_sidebar_task("calibration")
window.experiments_window.resize(860, 620)
app.processEvents()
single_scroll = window.calibration_mode_tabs.widget(window._calibration_single_tab_index)
window.calibration_taxel_selector.toggle.setChecked(True)
from PyQt5.QtTest import QTest
from phd.ui.motion import DISCLOSURE_MS
QTest.qWait(DISCLOSURE_MS + 40)
app.processEvents()
assert single_scroll.verticalScrollBar().maximum() > 0
assert window.calibration_speed_spin.height() >= 20
assert window.calibration_taxel_map.height() >= 150
window.close()
''')


def test_monitor_failure_cannot_prevent_task_stop():
    run_qt('''
from types import SimpleNamespace
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from phd.ui.ui_initial import MyMainWindow
app = QApplication([])
QTimer.singleShot = lambda *_a, **_kw: None
window = MyMainWindow()
stopped = []
def broken(*_args):
    raise RuntimeError("test monitor failure")
window.experiment_run_view.refresh = broken
window.experiment_run_view.update_end_details = broken
window._sidebar_active_task = SimpleNamespace(id="test", label="Test", on_stop=lambda *_a, **_kw: stopped.append(True))
window._stop_sidebar_control(show_result=False)
assert stopped == [True]
assert window._sidebar_active_task is None
assert not window.experiment_monitor_timer.isActive()
assert "Monitor unavailable" in window.experiment_run_view.state.text()
window.close()
''')
