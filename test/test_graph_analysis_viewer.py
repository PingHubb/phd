"""Offline analysis never participates in the live experiment lifecycle."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from phd.ui import calibration_graph_viewer as viewer


def run_qt(script):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("started", [True, False])
def test_launcher_is_detached_and_limits_child_workers_only(monkeypatch, started):
    instances = []
    class Process:
        def __init__(self):
            instances.append(self)
        def setProgram(self, value):
            self.program = value
        def setArguments(self, value):
            self.arguments = value
        def setProcessEnvironment(self, value):
            self.environment = value
        def setWorkingDirectory(self, value):
            self.directory = value
        def startDetached(self):
            return started, 1234 if started else 0
    monkeypatch.setattr(viewer, "QProcess", Process)
    before = dict(os.environ)
    assert viewer.launch_graph_analysis("/saved records/sensor one") is started
    process = instances[0]
    assert process.program == sys.executable
    assert process.arguments == ["-m", "phd.ui.calibration_graph_viewer", "--start-directory", "/saved records/sensor one"]
    assert process.environment.value("OPENBLAS_NUM_THREADS") == "1"
    assert process.environment.value("OMP_NUM_THREADS") == "1"
    assert process.environment.value("PYTHONPATH").split(os.pathsep)[0] == process.directory
    assert os.environ == before


def test_analysis_buttons_preserve_running_task_timer_and_status():
    run_qt('''
import os
import tempfile
from types import SimpleNamespace
config = tempfile.TemporaryDirectory()
os.environ['XDG_CONFIG_HOME'] = config.name
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from phd.ui import ui_initial
app = QApplication([])
single_shot = QTimer.singleShot
QTimer.singleShot = lambda *args: None
window = ui_initial.MyMainWindow()
QTimer.singleShot = single_shot
launched = []
ui_initial.launch_graph_analysis = lambda directory: launched.append(directory) or True
ticks = []
task = SimpleNamespace(id='calibration', label='Live calibration', on_tick=lambda host: ticks.append(True))
window._sidebar_active_task = task
window._sidebar_active_control_id = 'calibration'
window._sidebar_control_timer = QTimer(window)
window._sidebar_control_timer.setInterval(10)
window._sidebar_control_timer.timeout.connect(window._run_sidebar_control_action)
window._sidebar_control_timer.start()
window.info_process.setText('Experiment running: do not replace this status')
try:
    assert window.select_sidebar_task('calibration')
    selection = window._current_sidebar_control_id()
    window.graph_analysis_button.click()
    assert len(launched) == 1
    assert window._current_sidebar_control_id() == selection
    assert window._sidebar_active_task is task
    for lock in (window._set_calibration_controls_running,
                 window._set_full_taxel_calibration_controls_running,
                 window._set_taxel_mapping_controls_running,
                 window._set_sigraph_scanning_controls_running):
        lock(True)
        assert window.graph_analysis_button.isEnabled()
    window.graph_analysis_button.click()
    assert len(launched) == 2
    # The legacy task route is safe too, even if called programmatically.
    assert window.select_sidebar_task('graph_analysis')
    window._on_sidebar_task_changed()
    assert window.sidebar_btn_start.isEnabled()
    window.sidebar_btn_start.click()
    assert len(launched) == 3
    assert window._sidebar_active_task is task
    assert window._sidebar_active_control_id == 'calibration'
    assert window._sidebar_control_timer.isActive()
    QTest.qWait(60)
    assert len(ticks) >= 2
    assert window.info_process.text() == 'Experiment running: do not replace this status'
    ui_initial.launch_graph_analysis = lambda directory: False
    window.graph_analysis_button.click()
    assert 'could not open' in window.graph_analysis_status.text()
    assert window._sidebar_active_task is task and window._sidebar_control_timer.isActive()
    assert window.info_process.text() == 'Experiment running: do not replace this status'
finally:
    window._sidebar_control_timer.stop()
    window._sidebar_active_task = None
    window._sidebar_active_control_id = None
    window.close()
''')


def test_standalone_viewer_opens_saved_data_and_imports_no_device_stack():
    run_qt('''
import csv
import sys
import tempfile
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from phd.ui.calibration_graph_viewer import open_recording
app = QApplication([])
folder = tempfile.TemporaryDirectory()
path = Path(folder.name) / 'synthetic.csv'
with path.open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['elapsed_s','phase','travel_down_mm','diff_ave','force_delta_abs_g'])
    for i in range(80):
        travel = i/10 if i < 40 else (79-i)/10
        writer.writerow([i/10, 'approaching' if i < 40 else 'returning', travel, -travel*10, max(0, travel-2)*100])
original = path.read_bytes()
dialog = open_recording(path)
dialog.show()
app.processEvents()
assert dialog.parent() is None and not dialog.isModal()
assert 'Independent offline viewer' in dialog.status_label.text()
assert dialog.charts['process'].rows
dialog.charts['process'].grab()
assert not any(name in sys.modules for name in ('rclpy', 'serial', 'pyrealsense2',
    'phd.ui.ui_initial', 'phd.ui.ui_ping', 'phd.ui.experiment_tasks'))
dialog.close()
app.processEvents()
assert path.read_bytes() == original
''')


def test_real_detached_viewer_startup_leaves_parent_timer_responsive():
    run_qt('''
import csv, json, os, sys, tempfile
from pathlib import Path
from PyQt5.QtCore import QProcess, QTimer
from PyQt5.QtWidgets import QApplication
from phd.ui import calibration_graph_viewer as viewer
folder = tempfile.TemporaryDirectory()
path = Path(folder.name) / 'trial.csv'
result_path = Path(folder.name) / 'child-result.json'
with path.open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['elapsed_s','phase','travel_down_mm','diff_ave','force_delta_abs_g'])
    for i in range(80):
        travel = i/10 if i < 40 else (79-i)/10
        writer.writerow([i/10,'approaching' if i<40 else 'returning',travel,-travel*10,max(0,travel-2)*100])
# Instrument only the child entry point so this real detached launch closes
# itself after drawing. The launch environment and startDetached are real Qt.
probe = """
import json, os, sys
from pathlib import Path
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from phd.ui.calibration_graph_viewer import main
once = QTimer.singleShot
def checked_start(delay, function):
    once(delay, function)
    def check():
        app = QApplication.instance()
        dialog = app.analysis_dialog
        assert dialog.charts['process'].rows
        assert not dialog.charts['process'].grab().isNull()
        Path(RESULT).write_text(json.dumps({'pid':os.getpid(), 'threads':os.environ.get('OPENBLAS_NUM_THREADS'),
            'device_imported':any(name in sys.modules for name in ('rclpy','serial','phd.ui.experiment_tasks'))}))
        dialog.close()
    once(250, check)
QTimer.singleShot = checked_start
raise SystemExit(main(['--recording', SOURCE]))
""".replace('RESULT', repr(str(result_path))).replace('SOURCE', repr(str(path)))
class InstrumentedProcess(QProcess):
    def setArguments(self, arguments):
        assert arguments[:2] == ['-m', 'phd.ui.calibration_graph_viewer']
        super().setArguments(['-c', probe])
viewer.QProcess = InstrumentedProcess
app = QApplication([])
ticks = []
timer = QTimer()
timer.setInterval(10)
def tick():
    ticks.append(1)
    if result_path.exists():
        app.quit()
timer.timeout.connect(tick)
timer.start()
assert viewer.launch_graph_analysis(folder.name)
QTimer.singleShot(15000, app.quit)
app.exec_()
assert result_path.exists(), 'Detached viewer did not complete startup'
result = json.loads(result_path.read_text())
assert result['pid'] != os.getpid() and result['threads'] == '1'
assert not result['device_imported']
assert len(ticks) >= 5, ticks
''')


@pytest.mark.parametrize("failure", [False, True])
def test_startup_cancel_and_bad_file_exit_cleanly_in_the_viewer(failure):
    run_qt('''
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from phd.ui.calibration_graph_viewer import main
QFileDialog.getOpenFileName = lambda *a: ('', '')
messages = []
QMessageBox.warning = lambda *a: messages.append(a)
failure = ''' + repr(failure) + '''
result = main(['--recording', '/missing/does-not-exist.json'] if failure else [])
assert result == (1 if failure else 0)
assert bool(messages) == failure
''')
