"""Presentation changes must not change the rig's behaviour or hide controls."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from phd.ui import theme
from phd.ui.ui_ping import UI


ROOT = Path(__file__).resolve().parents[1]


def run_qt(script):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_workspace_primitives_remain_compact_and_keyboard_accessible():
    run_qt('''
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from phd.ui import components, theme
from phd.ui.ui_initial import Resources
app = QApplication([])
host = QWidget()
host.setStyleSheet(Resources().get_stylesheet())
layout = QVBoxLayout(host)
strip = components.SessionStrip()
layout.addWidget(strip)
welcome = components.ViewportWelcome()
layout.addWidget(welcome, 1)
host.resize(500, 800)
host.show()
app.processEvents()
assert strip.height() < 80
events = []
welcome.setup_requested.connect(lambda: events.append("setup"))
welcome.tracking_requested.connect(lambda: events.append("tracking"))
welcome.setup_button.setFocus()
QTest.keyClick(welcome.setup_button, Qt.Key_Space)
QTest.mouseClick(welcome.tracking_button, Qt.LeftButton)
assert events == ["setup", "tracking"]
heading = components.WorkspaceHeader("Sensor", "First")
heading.set_context("Learning", "Second")
assert heading.title_label.text() == "Learning"
assert heading.subtitle_label.text() == "Second"
page = QWidget()
column = QVBoxLayout(page)
for index in range(30):
    column.addWidget(QPushButton(str(index)))
area = components.scroll_page(page)
area.resize(500, 250)
area.show()
app.processEvents()
assert area.verticalScrollBar().maximum() > 0
assert area.horizontalScrollBar().maximum() == 0
''')


def test_critical_toolbar_controls_fit_the_smallest_supported_window(tmp_path):
    run_qt(f'''
import os
os.environ["XDG_CONFIG_HOME"] = {str(tmp_path)!r}
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QToolBar
from phd.ui.ui_initial import MyMainWindow
app = QApplication([])
QTimer.singleShot = lambda *args, **kwargs: None
window = MyMainWindow()
for width, height in [(1440, 900), (1100, 720), (960, 540)]:
    window.resize(width, height)
    window.show()
    app.processEvents()
    toolbar = window.findChild(QToolBar)
    for button in (window.toolbar_stop_all_motion_button,
                   window.toolbar_disconnect_sensor_button,
                   window.toolbar_update_sensor_button):
        assert button.isVisible(), (width, button.text())
        assert toolbar.rect().contains(button.geometry()), (width, button.text())
    style = Qt.ToolButtonIconOnly if width < 1280 else Qt.ToolButtonTextBesideIcon
    assert all(button.toolButtonStyle() == style for button in window._toolbar_destinations)
''')


def test_scene_empty_state_never_covers_loaded_sensor_or_humanoid():
    class Label:
        def setText(self, text):
            self.text = text

    class Welcome:
        def setVisible(self, visible):
            self.visible = visible

    ui = SimpleNamespace(
        viewport_welcome=Welcome(), viewport_title=Label(), viewport_state=Label(),
        _humanoid_viewport_active=False,
    )
    UI.set_scene_presentation(ui, False)
    assert ui.viewport_welcome.visible
    UI.set_scene_presentation(ui, True)
    assert not ui.viewport_welcome.visible
    assert ui.viewport_state.text == "Scene loaded"
    ui._humanoid_viewport_active = True
    UI.set_scene_presentation(ui, False)
    assert not ui.viewport_welcome.visible
    assert ui.viewport_title.text == "Humanoid scene"
    ui._humanoid_viewport_active = False
    UI.set_scene_presentation(ui, False)
    assert ui.viewport_welcome.visible


def test_tracking_state_and_record_controls_stay_visible_on_a_laptop():
    run_qt('''
from PyQt5.QtWidgets import QApplication, QWidget
from phd.ui.touch_tracking_window import TouchTrackingWindow
from phd.ui.ui_initial import Resources
app = QApplication([])
owner = QWidget()
owner.setStyleSheet(Resources().get_stylesheet())
window = TouchTrackingWindow(owner)
window.resize(900, 700)
window.show()
app.processEvents()
assert window.height() == 700
assert window.record_setup_scroll.verticalScrollBar().maximum() > 0
assert window.record_setup_scroll.horizontalScrollBar().maximum() == 0
for button in [*window.protocol_buttons, window.record_button]:
    assert button.isVisible()
    assert window.rect().contains(button.mapTo(window, button.rect().bottomRight()))
window.close()
''')


def test_compact_geometry_selector_preserves_the_existing_selection_model():
    # Build the real Sensor page, without constructing VTK or device services.
    run_qt('''
from types import SimpleNamespace
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from phd.ui.ui_ping import UI
app = QApplication([])
owner = SimpleNamespace(widget_func=QWidget())
for name in dir(UI):
    value = getattr(UI, name)
    if callable(value) and name.startswith("_") and not name.startswith("__"):
        setattr(owner, name, lambda *args, **kwargs: None)
owner._refresh_sensor_port_empty_state = lambda *args: None
owner._scrollable = UI._scrollable
UI.setup_tab1(owner, QVBoxLayout(owner.widget_func))
combo = owner.sensor_geometry_combo
assert combo.model() is owner.sensor_choice.model()
for index in range(combo.count()):
    combo.setCurrentIndex(index)
    assert owner.sensor_choice.currentRow() == index
owner.sensor_choice.setCurrentRow(1)
assert combo.currentIndex() == 1
assert owner.sensor_choice.isHidden()
''')


def test_action_and_readout_palette_has_readable_contrast():
    def luminance(color):
        rgb = [int(color[i:i+2], 16) / 255 for i in (1, 3, 5)]
        linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgb]
        return sum(v * weight for v, weight in zip(linear, (.2126, .7152, .0722)))

    pairs = [(theme.TEXT_MUTED, theme.SURFACE), (theme.TEXT_SECONDARY, theme.SURFACE),
             (theme.VIEWPORT_MUTED, theme.VIEWPORT_BG)]
    pairs += [(theme.TEXT_ON_ACCENT, fill) for fill in (
        theme.ACCENT, theme.ACCENT_HOVER, theme.SUCCESS, theme.SUCCESS_HOVER,
        theme.DANGER, theme.DANGER_HOVER, theme.WARNING, theme.WARNING_HOVER,
    )]
    for foreground, background in pairs:
        a, b = sorted((luminance(foreground), luminance(background)))
        assert (b + .05) / (a + .05) >= 4.5, (foreground, background)
