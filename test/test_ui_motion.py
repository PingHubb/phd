"""Motion is paint/layout only: immediate actions, bounded work, safe teardown."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run_qt(script):
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="pinglab-motion-test-") as settings:
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", XDG_CONFIG_HOME=settings)
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                                capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback (most recent call last)" not in result.stderr, result.stderr


def test_navigation_selection_is_immediate_and_rapid_switches_retarget_one_tween():
    run_qt('''
from PyQt5.QtCore import Qt, QAbstractAnimation
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from phd.ui import nav_tabs, motion
app = QApplication([])
host = QWidget()
layout = QVBoxLayout(host)
tabs = nav_tabs.WorkspaceTabs()
nav_tabs.install(tabs)
for name in ("Sensor", "Robots", "Control", "AI", "Hand", "Tools"):
    tabs.addTab(QWidget(), name)
bar = tabs.create_external_navigation()
layout.addWidget(bar)
layout.addWidget(tabs)
host.resize(1000, 400)
host.show()
QTest.qWait(30)
events = []
tabs.currentChanged.connect(events.append)
start = bar._indicator
QTest.mouseClick(bar, Qt.LeftButton, pos=bar.tabRect(3).center())
assert tabs.currentIndex() == bar.currentIndex() == 3
assert events == [3]
assert bar._indicator == start
assert bar._indicator_motion.state() == QAbstractAnimation.Running
QTest.qWait(45)
assert start != bar._indicator != bar._selection_rect()
tabs.setCurrentIndex(1)  # blocked programmatic nav synchronization also animates
assert bar._indicator_motion.state() == QAbstractAnimation.Running
for i in (5, 2, 0, 4, 1):
    tabs.setCurrentIndex(i)
    QTest.qWait(12)
QTest.qWait(180)
assert bar._indicator == bar._selection_rect()
assert bar._indicator_motion.state() == QAbstractAnimation.Stopped
tabs.setCurrentIndex(5)
motion.preferences().set_reduced(True)
assert bar._indicator == bar._selection_rect()
tabs.setCurrentIndex(0)
assert bar._indicator == bar._selection_rect()
assert bar._indicator_motion.state() == QAbstractAnimation.Stopped
tabs.setTabVisible(2, False)
host.resize(700, 400)
QTest.qWait(20)
assert bar._indicator == bar._selection_rect()
tabs.setTabText(0, "Sensor setup")
QTest.qWait(20)
assert bar._indicator == bar._selection_rect()
host.close()
''')


def test_disclosure_interruptions_restore_layout_values_focus_and_scroll_height():
    run_qt('''
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSpinBox
from PyQt5.QtTest import QTest
from PyQt5.QtCore import QAbstractAnimation
from phd.ui import components, motion
from phd.ui.ui_ping import _WidthFittingScrollArea
app = QApplication([])
page = QWidget()
layout = QVBoxLayout(page)
group = components.CollapsibleGroup("Advanced options")
fields = [QSpinBox() for _ in range(8)]
for i, field in enumerate(fields):
    field.setValue(i + 1)
    group.add(field)
layout.addWidget(group)
layout.addStretch()
area = _WidthFittingScrollArea()
area.setWidgetResizable(True)
area.setWidget(page)
area.resize(500, 160)
area.show()
QTest.qWait(40)
collapsed = page.minimumHeight()
events = []
group.toggled.connect(events.append)
group.toggle.click()
assert events == [True]
assert group.content.isVisible()
assert group._motion.tween.state() == QAbstractAnimation.Running
QTest.qWait(50)
partial = group.content.maximumHeight()
assert 0 < partial < 16777215
QTest.qWait(160)
assert group.content.maximumHeight() == 16777215
assert page.minimumHeight() > collapsed
assert area.verticalScrollBar().maximum() > 0
fields[0].setFocus()
group.toggle.click()
assert group.toggle.hasFocus()
QTest.qWait(35)
group.toggle.click()
QTest.qWait(20)
group.toggle.click()
QTest.qWait(180)
assert not group.content.isVisible()
assert group.content.maximumHeight() == 16777215
assert page.minimumHeight() == collapsed, (page.minimumHeight(), collapsed)
assert [f.value() for f in fields] == list(range(1, 9))
group.toggle.click()
motion.preferences().set_reduced(True)
assert group.content.isVisible()
assert group.content.maximumHeight() == 16777215
group.toggle.click()
assert not group.content.isVisible()
motion.preferences().set_reduced(False)
group.toggle.click()
area.hide()
assert group._motion.tween.state() == QAbstractAnimation.Stopped
area.show()
QTest.qWait(20)
assert group.content.isVisible() and group.content.maximumHeight() == 16777215
area.close()
''')


def test_page_reveal_keeps_controls_live_and_cleans_up_on_hide_or_reduced_motion():
    run_qt('''
from PyQt5.QtWidgets import QApplication, QPushButton
from PyQt5.QtCore import Qt, QAbstractAnimation
from PyQt5.QtTest import QTest
from phd.ui import components, motion
app = QApplication([])
sections = components.TaskSections()
buttons = [QPushButton("Read state"), QPushButton("Open settings")]
for i, button in enumerate(buttons):
    sections.add_section(str(i), "Test", button)
sections.resize(650, 400)
sections.show()
QTest.qWait(180)
called = []
buttons[1].clicked.connect(lambda: called.append(True))
sections.setCurrentIndex(1)
QTest.qWait(20)
reveal = sections.pages[1]._page_reveal
assert reveal.tween.state() == QAbstractAnimation.Running
assert reveal.layer.testAttribute(Qt.WA_TransparentForMouseEvents)
assert reveal.layer.focusPolicy() == Qt.NoFocus
assert sections.pages[1].graphicsEffect() is None
QTest.mouseClick(buttons[1], Qt.LeftButton)
assert called == [True]
sections.setCurrentIndex(0)
assert not reveal.layer.isVisible() and not reveal.pending.isActive()
sections.setCurrentIndex(1)
QTest.qWait(20)
motion.preferences().set_reduced(True)
assert reveal.layer.isHidden()
assert reveal.tween.state() == QAbstractAnimation.Stopped
sections.setCurrentIndex(0)
QTest.qWait(20)
assert sections.pages[0]._page_reveal.layer.isHidden()
sections.close()
''')


def test_native_renderers_can_opt_out_of_page_effects():
    run_qt('''
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtTest import QTest
from phd.ui import components
app = QApplication([])
sections = components.TaskSections()
view = QLabel("Live view")
view.setProperty("motionExclude", True)
sections.add_section("Display", "Live", view)
sections.show()
QTest.qWait(30)
assert sections.pages[0]._page_reveal.layer.isHidden()
sections.close()
''')


def test_pending_animations_and_filters_are_safe_when_widgets_are_deleted():
    run_qt('''
from PyQt5.QtWidgets import QApplication, QPushButton
from PyQt5.QtCore import QEvent
from PyQt5.QtTest import QTest
from phd.ui import components, motion
app = QApplication([])
for _ in range(3):
    sections = components.TaskSections()
    group = components.CollapsibleGroup("More")
    group.add(QPushButton("Read state"))
    sections.add_section("Setup", "Settings", group)
    sections.show()
    QTest.qWait(20)
    group.toggle.click()
    app.sendEvent(sections.buttons[0], QEvent(QEvent.Enter))
    sections.deleteLater()
    app.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
motion.preferences().set_reduced(True)
QTest.qWait(200)
''')


def test_pages_with_safety_actions_are_not_faded():
    run_qt('''
from PyQt5.QtWidgets import QApplication, QPushButton
from PyQt5.QtTest import QTest
from phd.ui import components
app = QApplication([])
sections = components.TaskSections()
stop = QPushButton("Stop")
components.apply_variant(stop, "danger")
sections.add_section("Control", "Safety", stop)
sections.show()
QTest.qWait(30)
assert sections.pages[0]._page_reveal.layer.isHidden()
sections.close()
''')


def test_hover_does_not_delay_press_or_animate_danger_actions():
    run_qt('''
from PyQt5.QtCore import QEvent, QAbstractAnimation, Qt
from PyQt5.QtWidgets import QApplication, QPushButton
from PyQt5.QtTest import QTest
from phd.ui import components, motion
app = QApplication([])
button = QPushButton("Connect")
components.apply_variant(button, "primary")
button.show()
QTest.qWait(20)
app.sendEvent(button, QEvent(QEvent.Enter))
QTest.qWait(30)
feedback = button._hover_feedback
assert feedback.layer.amount > 0
called = []
button.clicked.connect(lambda: called.append(1))
QTest.mouseClick(button, Qt.LeftButton)
assert called == [1] and feedback.layer.isHidden()
components.apply_variant(button, "danger")
app.sendEvent(button, QEvent(QEvent.Enter))
assert feedback.layer.isHidden()
assert feedback.tween.state() == QAbstractAnimation.Stopped
components.apply_variant(button, "primary")
app.sendEvent(button, QEvent(QEvent.Enter))
button.setEnabled(False)
assert feedback.layer.isHidden()
button.setEnabled(True)
motion.preferences().set_reduced(True)
app.sendEvent(button, QEvent(QEvent.Enter))
assert feedback.layer.isHidden()
stop = QPushButton("Stop All Motion")
components.apply_variant(stop, "danger")
assert not hasattr(stop, "_hover_feedback")
button.close()
''')


def test_reduce_motion_menu_persists_and_mirrors_shared_preference():
    run_qt('''
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, QSettings
from phd.ui import motion
from phd.ui.ui_initial import MyMainWindow
app = QApplication([])
QTimer.singleShot = lambda *_a, **_kw: None
window = MyMainWindow()
action = window.action_reduce_motion
assert not action.isChecked()
action.trigger()
assert motion.preferences().reduced
motion.preferences().settings.sync()
assert QSettings("PingLab", "PingLab").value("appearance/reduce_motion", type=bool)
assert motion.MotionPreferences(app).reduced
motion.preferences().set_reduced(False)
assert not action.isChecked()
window.close()
''')
