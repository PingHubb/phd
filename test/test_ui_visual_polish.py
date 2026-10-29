"""Fixed navigation and visual refinements preserve the original actions."""

import os
from pathlib import Path
import subprocess
import sys


def run_qt(script):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_external_navigation_preserves_page_identity_indices_and_disabled_tabs():
    run_qt('''
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from phd.ui import nav_tabs
app = QApplication([])
host = QWidget()
layout = QVBoxLayout(host)
tabs = nav_tabs.WorkspaceTabs()
nav_tabs.install(tabs)
pages = [QWidget() for _ in range(3)]
for page, title in zip(pages, ("Sensor", "Robots", "AI")):
    tabs.addTab(page, title)
nav_tabs.set_glyphs(tabs, {"Sensor": "sensor", "Robots": "robot", "AI": "ai"})
bar = tabs.create_external_navigation()
layout.addWidget(bar)
layout.addWidget(tabs)
host.resize(900, 400)
host.show()
app.processEvents()
events = []
tabs.currentChanged.connect(events.append)
QTest.mouseClick(bar, Qt.LeftButton, pos=bar.tabRect(2).center())
assert tabs.currentIndex() == bar.currentIndex() == 2
assert events == [2]
assert tabs.widget(2) is pages[2]
assert tabs.tabBar().isHidden()
tabs.setCurrentIndex(0)
assert bar.currentIndex() == 0
tabs.setTabEnabled(1, False)
assert not bar.isTabEnabled(1)
events.clear()
QTest.mouseClick(bar, Qt.LeftButton, pos=bar.tabRect(1).center())
assert tabs.currentIndex() == 0
assert events == []
bar.setFocus()
QTest.keyClick(bar, Qt.Key_Right)
assert tabs.currentIndex() == 2
assert events == [2]
tabs.setTabText(2, "Learning")
tabs.setTabToolTip(2, "Models and recordings")
assert bar.tabText(2) == "Learning"
assert bar.tabToolTip(2) == "Models and recordings"
assert bar.tabData(2) == "ai"
tabs.setTabVisible(1, False)
assert not bar.isTabVisible(1)
tabs.addTab(QWidget(), "Tools")
assert bar.count() == tabs.count() == 4
tabs.removeTab(1)
assert bar.count() == 3
assert tabs.widget(1) is pages[2]
assert bar.currentIndex() == tabs.currentIndex()
''')


def test_action_cards_reflow_and_keep_each_existing_button_action_once():
    run_qt('''
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QPushButton
from phd.ui import components
from phd.ui.ui_initial import Resources
app = QApplication([])
cards = components.ActionCards()
cards.setStyleSheet(Resources().get_stylesheet())
clicked = []
buttons = []
for i in range(3):
    button = QPushButton("Open workspace " + str(i))
    button.clicked.connect(lambda _=False, i=i: clicked.append(i))
    cards.add_card(str(i), "Workspace title", "A useful description explaining this action.", button)
    buttons.append(button)
cards.resize(1000, 260)
cards.show()
app.processEvents()
assert cards._columns == 3
assert len({card.y() for card in cards.cards}) == 1
cards.resize(460, 700)
app.processEvents()
assert cards._columns == 1
assert len({card.y() for card in cards.cards}) == 3
assert clicked == []
for button in buttons:
    button.setFocus()
    QTest.keyClick(button, Qt.Key_Space)
assert clicked == [0, 1, 2]
''')


def test_task_navigation_width_is_bounded_and_resizing_does_not_select_a_task():
    run_qt('''
from PyQt5.QtWidgets import QApplication, QLabel
from phd.ui import components
app = QApplication([])
sections = components.TaskSections()
for i in range(4):
    sections.add_section("Task " + str(i), "Description", QLabel("Content"))
sections.setCurrentIndex(2)
changes = []
sections.currentChanged.connect(changes.append)
sections.show()
for width in (1400, 460, 1000, 380):
    sections.resize(width, 500)
    app.processEvents()
    assert sections.index.width() <= 1040
    assert sections.pages[2].widget().width() <= 1040
    assert sections.currentIndex() == 2
    assert all(button.isVisible() for button in sections.buttons)
assert not changes
''')


def test_calibration_disclosures_leave_settings_reachable_without_changing_values():
    run_qt('''
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from phd.ui.ui_initial import MyMainWindow
app = QApplication([])
QTimer.singleShot = lambda *_a, **_kw: None
window = MyMainWindow()
window.select_sidebar_task("calibration")
window.experiments_window.resize(860, 620)
window.experiments_window.show()
app.processEvents()
assert window.sidebar_btn_stop.text() == "Stop Experiment"
assert not window.sidebar_task_help.toggle.isChecked()
assert not window.calibration_taxel_selector.toggle.isChecked()
assert window.calibration_speed_spin.isVisible()
speed = window.calibration_speed_spin.value()
window.calibration_taxel_selector.toggle.click()
from PyQt5.QtTest import QTest
from phd.ui.motion import DISCLOSURE_MS
QTest.qWait(DISCLOSURE_MS + 40)
app.processEvents()
assert window.calibration_taxel_map.isVisible()
assert window.calibration_taxel_map.height() >= 150
assert window.calibration_speed_spin.value() == speed
assert window._sidebar_active_task is None
window.select_sidebar_task("graph_analysis")
assert window.sidebar_task_help.toggle.isChecked()
window.close()
''')


def test_sensor_welcome_adapts_to_a_small_viewport_without_clipped_text():
    run_qt('''
from PyQt5.QtWidgets import QApplication
from phd.ui.components import ViewportWelcome
from phd.ui.ui_initial import Resources
app = QApplication([])
view = ViewportWelcome()
view.setStyleSheet(Resources().get_stylesheet())
view.resize(320, 260)
view.show()
app.processEvents()
assert view.welcome_glyph.isHidden()
assert view.welcome_footer.isHidden()
for label in (view.welcome_title, view.welcome_note):
    assert label.height() >= label.heightForWidth(label.width())
for button in (view.setup_button, view.tracking_button):
    assert view.rect().contains(button.geometry())
view.resize(600, 650)
app.processEvents()
assert not view.welcome_glyph.isHidden()
assert not view.welcome_footer.isHidden()
''')


def test_settings_form_associates_labels_without_changing_field_values():
    run_qt('''
from PyQt5.QtWidgets import QApplication, QWidget, QSpinBox
from phd.ui.components import SettingsForm
app = QApplication([])
host = QWidget()
form = SettingsForm(host)
field = QSpinBox()
field.setRange(1, 200)
field.setValue(25)
events = []
field.valueChanged.connect(events.append)
form.addRow("Frames per probe level", field)
host.resize(340, 200)
host.show()
app.processEvents()
label = form.labelForField(field)
assert label.buddy() is field
assert label.wordWrap()
assert field.accessibleName() == "Frames per probe level"
assert field.value() == 25
assert not events
''')


def test_mapping_sections_keep_settings_values_and_progress_visible():
    run_qt('''
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QAbstractSpinBox, QScrollArea
from phd.ui.ui_initial import MyMainWindow
app = QApplication([])
QTimer.singleShot = lambda *_a, **_kw: None
window = MyMainWindow()
window.select_sidebar_task("taxel_position_mapping")
window.experiments_window.resize(860, 620)
window.experiments_window.show()
app.processEvents()
tabs = window.taxel_mapping_sections
assert [tabs.tabText(i) for i in range(tabs.count())] == ["Map / anchor", "Motion / safety", "Signal detection", "XY search"]
assert tabs.currentIndex() == 0
assert window.taxel_mapping_map.isVisible()
fields = tabs.findChildren(QAbstractSpinBox)
before = [field.value() for field in fields]
for i in range(tabs.count()):
    tabs.setCurrentIndex(i)
    app.processEvents()
    scroll = tabs.widget(i)
    assert isinstance(scroll, QScrollArea)
    assert scroll.horizontalScrollBar().maximum() == 0
    assert window.taxel_mapping_progress_label.isVisible()
    assert window.sidebar_btn_stop.isVisible()
assert [field.value() for field in fields] == before
assert window._sidebar_active_task is None
tabs.setCurrentIndex(1)
assert window.taxel_mapping_force_guard_spin.isVisible()
assert window.taxel_mapping_xy_limit_spin.isVisible()
tabs.setCurrentIndex(2)
assert window.taxel_mapping_baseline_frames_spin.isVisible()
combo = window.taxel_mapping_detection_combo
saved_mode = combo.currentIndex()
fixed_value = window.taxel_mapping_signal_threshold_spin.value()
legacy_value = window.taxel_mapping_adaptive_percent_spin.value()
for mode in ("robust_shift", "fixed", "adaptive_range"):
    combo.setCurrentIndex(combo.findData(mode))
    assert window.taxel_mapping_signal_threshold_spin.isVisible() == (mode == "fixed")
    assert window.taxel_mapping_adaptive_percent_spin.isVisible() == (mode == "adaptive_range")
assert window.taxel_mapping_signal_threshold_spin.value() == fixed_value
assert window.taxel_mapping_adaptive_percent_spin.value() == legacy_value
combo.setCurrentIndex(saved_mode)
tabs.setCurrentIndex(3)
assert window.taxel_mapping_coarse_radius_spin.isVisible()
window.close()
''')


def test_learning_setup_disclosures_and_pinned_training_actions_are_presentation_only():
    run_qt('''
from PyQt5.QtWidgets import QApplication, QWidget
from phd.ui.touch_tracking_window import TouchTrackingWindow
from phd.ui.ui_initial import Resources
app = QApplication([])
owner = QWidget()
window = TouchTrackingWindow(owner)
window.timer.stop()
window.setStyleSheet(Resources().get_stylesheet())
window.resize(900, 650)
window.show()
app.processEvents()
assert not window.sensor_dimensions.toggle.isChecked()
assert window.participant.isVisible()
assert not window.width_mm.isVisible()
assert window.record_button.isVisible()
window.sensor_dimensions.toggle.click()
assert window.width_mm.isVisible()
assert window.width_mm.value() == 0
window.tabs.setCurrentIndex(2)
app.processEvents()
before = (window.epochs.value(), window.history_frames.value(), window.seed.value())
window.training_advanced.toggle.click()
app.processEvents()
assert window.epochs.isVisible()
scroll = window.train_setup_scroll
scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
app.processEvents()
for button in (window.check_button, window.train_button, window.cancel_button):
    assert button.isVisible()
    assert window.rect().contains(button.mapTo(window, button.rect().bottomRight()))
assert scroll.horizontalScrollBar().maximum() == 0
assert (window.epochs.value(), window.history_frames.value(), window.seed.value()) == before
assert window.recorder is None and window._process is None
window.close()
''')


def test_camera_adjustment_disclosure_does_not_start_or_reconfigure_stream():
    run_qt('''
from PyQt5.QtWidgets import QApplication
from phd.ui import camera_preview_window as camera
from phd.ui.ui_initial import Resources
app = QApplication([])
starts = []
camera.realsense_color_candidates = lambda: []
camera.CameraPreviewWindow._start_realsense = lambda self: starts.append("start")
window = camera.CameraPreviewWindow()
window.setStyleSheet(Resources().get_stylesheet())
window.resize(700, 540)
window.show()
app.processEvents()
assert starts == ["start"]
assert not window.adjustments.toggle.isChecked()
large_preview = window.view.height()
window.adjustments.toggle.click()
app.processEvents()
assert window.exposure_slider.isVisible()
assert not window.exposure_slider.isEnabled()
assert window.view.height() < large_preview
assert window.height() == 540
assert starts == ["start"]
assert window.exposure_slider.accessibleName() == "exposure value"
assert window.save_button.isVisible() and window.reconnect_button.isVisible()
window.close()
''')


def test_return_to_split_restores_width_once_without_expanded_page_refit():
    run_qt('''
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QSplitter, QWidget, QVBoxLayout, QLabel
from phd.ui.ui_ping import UI
from phd.ui.components import TaskSections
app = QApplication([])
class Workspace(QSplitter):
    _apply_workspace_layout = UI._apply_workspace_layout
    def _automatic_workspace_layout(self):
        return "controls" if self.full_width else "split"
    def _fit_main_control_panel(self):
        raise AssertionError("Tab switches must not refit an expanded page")
    def _sync_main_control_panel_handle(self):
        pass
    def setSizes(self, sizes):
        self.resizes += 1
        super().setSizes(sizes)
view = Workspace(Qt.Horizontal)
view.resizes = 0
view.splitter_1, view.splitter_2 = QWidget(), QWidget()
view.addWidget(view.splitter_1)
view.addWidget(view.splitter_2)
controls = QVBoxLayout(view.splitter_2)
sections = TaskSections()
sections.add_section("Read state", "Read-only controls", QLabel("Robot state"))
sections.add_section("Move / send", "Command editors", QLabel("Motion controls"))
controls.addWidget(sections)
view.resize(1400, 700)
view.show()
app.processEvents()
view.setSizes([700, 700])
view._workspace_layout_mode = "auto"
view._workspace_layout_target = "split"
view._main_control_panel_last_width = view.sizes()[1]
view._main_control_panel_user_sized = False
saved_width = view.sizes()[1]
for _ in range(5):
    view.full_width = True
    view._apply_workspace_layout()
    app.processEvents()
    assert view.splitter_1.isHidden()
    assert sections.index.minimumWidth() == 0
    view.full_width = False
    view.resizes = 0
    view._apply_workspace_layout()
    app.processEvents()
    assert view.resizes == 1
    assert abs(view.sizes()[1] - saved_width) <= 1
    assert view.updatesEnabled()
    assert not view.splitter_1.isHidden()
view.setUpdatesEnabled(False)
view.full_width = True
view._apply_workspace_layout()
assert not view.updatesEnabled()  # Respect an outer update guard.
''')


def test_scroll_page_show_events_share_one_pending_layout_callback():
    run_qt('''
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from phd.ui.ui_ping import _WidthFittingScrollArea
app = QApplication([])
page = QWidget()
layout = QVBoxLayout(page)
layout.addWidget(QLabel("Robot controls"))
area = _WidthFittingScrollArea()
area.setWidgetResizable(True)
area.setWidget(page)
area.show()
app.processEvents()
app.processEvents()
calls = []
area._height_lock_timer.timeout.connect(lambda: calls.append(1))
for _ in range(100):
    area._schedule_content_height_lock()
assert area._height_lock_timer.isActive()
app.processEvents()
assert calls == [1]
assert page.minimumHeight() == page.layout().sizeHint().height()
''')


def test_unchanged_navigation_state_does_not_reassign_tab_geometry():
    run_qt('''
from PyQt5.QtWidgets import QApplication, QWidget
from phd.ui.nav_tabs import WorkspaceTabs
app = QApplication([])
tabs = WorkspaceTabs()
for title in ("Sensor", "Robots", "AI"):
    tabs.addTab(QWidget(), title)
bar = tabs.create_external_navigation()
calls = []
original = bar.setTabText
def set_text(index, text):
    calls.append((index, text))
    original(index, text)
bar.setTabText = set_text
for _ in range(10):
    tabs.setTabEnabled(1, True)
    tabs._sync_navigation()
assert calls == []
tabs.setTabText(1, "Robot platforms")
assert calls == [(1, "Robot platforms")]
''')
