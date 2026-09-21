"""The current-session summary stays compact and side-effect free."""

import os
from pathlib import Path
import subprocess
import sys


def test_session_strip_widget_contract():
    """Exercise Qt in isolation from tests that destroy QApplication objects."""
    project_root = Path(__file__).resolve().parents[1]
    script = """
from PyQt5.QtWidgets import QApplication
from phd.ui.components import SessionStrip
from phd.ui import theme

app = QApplication([])
strip = SessionStrip()
assert set(strip._values) == {"sensor", "control", "model", "force"}
assert all(strip.value(key) == "—" for key in strip._values)
assert all(strip.state(key) == "idle" for key in strip._values)
assert theme.STATE_COLORS["idle"][0] in strip._values["sensor"].styleSheet()

strip.set_value(
    "sensor",
    "ttyACM0 · 10x10 · 60 Hz",
    tooltip="Streaming: /dev/ttyACM0 · 10x10 · 60 Hz",
    state="active",
)
assert strip.value("sensor") == "ttyACM0 · 10x10 · 60 Hz"
assert strip.state("sensor") == "active"
assert theme.SUCCESS in strip._values["sensor"].styleSheet()
assert strip._values["sensor"].toolTip().startswith("Streaming")

try:
    strip.set_value("unknown", "value")
except KeyError:
    pass
else:
    raise AssertionError("unknown fields should not silently alter the strip")
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
