"""Experiment controls stay reusable when their modeless window is closed."""

import os
from pathlib import Path
import subprocess
import sys


def test_experiment_controls_window_hides_without_destroying_panel():
    project_root = Path(__file__).resolve().parents[1]
    script = """
import os
import tempfile

config_dir = tempfile.TemporaryDirectory()
os.environ["XDG_CONFIG_HOME"] = config_dir.name

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QFrame
from phd.ui.ui_initial import ExperimentControlsWindow, MyMainWindow

app = QApplication([])
panel = QFrame()
window = ExperimentControlsWindow(panel)

window.show()
app.processEvents()
assert window.isVisible()
assert panel.parent() is window

window.close()
app.processEvents()
assert not window.isVisible()
assert panel.parent() is window

window.show()
app.processEvents()
assert window.isVisible()
window.reject()
app.processEvents()
assert not window.isVisible()
assert panel.parent() is window

window.deleteLater()
app.processEvents()

original_single_shot = QTimer.singleShot
QTimer.singleShot = lambda *_args, **_kwargs: None
try:
    main_window = MyMainWindow()
    assert main_window.h_splitter.count() == 1
    assert main_window.sidebar.parent() is main_window.experiments_window
    main_window.open_experiments_window()
    app.processEvents()
    assert main_window.experiments_window.isVisible()
    main_window.experiments_window.close()
    app.processEvents()
    assert not main_window.experiments_window.isVisible()
    assert main_window.sidebar.parent() is main_window.experiments_window
    main_window.close()
finally:
    QTimer.singleShot = original_single_shot

config_dir.cleanup()
"""
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(project_root), env.get("PYTHONPATH", "")))
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
