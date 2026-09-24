"""A short control page must scroll instead of crushing its rows.

Sensor, AI and Hand already wrap each sub-page in a scroll area. With
``widgetResizable`` set, Qt still shrinks that page down to its minimum
size. A list whose minimum is below its natural height is then crushed and
no scrollbar appears. The wrapper has to pin the page to its layout hint,
and to drop that pin again when a section is collapsed.
"""

import os
from pathlib import Path
import subprocess
import sys


def test_short_page_scrolls_and_follows_collapsed_sections():
    project_root = Path(__file__).resolve().parents[1]
    script = r"""
from PyQt5.QtWidgets import (
    QApplication, QListWidget, QPushButton, QVBoxLayout, QWidget,
)
from phd.ui.components import CollapsibleGroup
from phd.ui.ui_ping import _WidthFittingScrollArea

app = QApplication([])
page = QWidget()
layout = QVBoxLayout(page)
ports = QListWidget()
ports.addItems([f"port {i}" for i in range(8)])
layout.addWidget(ports)
advanced = CollapsibleGroup("Advanced", expanded=False)
for index in range(5):
    advanced.add(QPushButton(f"setting {index}"))
layout.addWidget(advanced)
layout.addWidget(QPushButton("Connect Sensor"))
layout.addStretch()

area = _WidthFittingScrollArea()
area.setWidgetResizable(True)
area.setWidget(page)
area.resize(360, 160)
area.show()
for _ in range(4):
    app.processEvents()

assert page.minimumHeight() == layout.sizeHint().height()
assert area.verticalScrollBar().maximum() > 0
assert ports.height() > 150
# The scroll area itself must stay short, or the panel could never shrink.
assert area.minimumSizeHint().height() < 120

advanced.toggle.setChecked(True)
for _ in range(4):
    app.processEvents()
assert page.minimumHeight() == layout.sizeHint().height()
assert page.minimumHeight() > 300

advanced.toggle.setChecked(False)
for _ in range(4):
    app.processEvents()
assert page.minimumHeight() == layout.sizeHint().height()
assert page.minimumHeight() < 300
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
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
