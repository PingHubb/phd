"""The Humanoid tab exposes its fixed scene without source browsers."""

import os
from pathlib import Path
import subprocess
import sys


def test_humanoid_scene_source_browsers_are_not_shown():
    project_root = Path(__file__).resolve().parents[1]
    script = """
from PyQt5.QtWidgets import QApplication
from phd.ui.humanoid_viewer import HumanoidViewerWidget

app = QApplication([])
viewer = HumanoidViewerWidget(plotter=object())

assert not hasattr(viewer, "path_edit")
assert not hasattr(viewer, "signal_dir_edit")
assert viewer.load_button.text() == "Load Humanoid Scene"

viewer.deleteLater()
app.processEvents()
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
