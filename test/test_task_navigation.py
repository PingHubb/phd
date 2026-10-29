"""Task navigation changes presentation, never device or recording state."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from phd.ui.ui_ping_ai_controls import AiControlsMixin


def run_qt(script):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_task_index_stays_visible_preserves_values_and_never_runs_actions():
    run_qt('''
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit
from phd.ui import components
from phd.ui.ui_ping import UI
from phd.ui.ui_initial import Resources
app = QApplication([])
nav = components.TaskSections(scroll_factory=UI._scrollable)
nav.setStyleSheet(Resources().get_stylesheet())
page = QWidget()
layout = QVBoxLayout(page)
field = QLineEdit("retained value")
layout.addWidget(field)
actions = []
for i in range(25):
    button = QPushButton("Action " + str(i))
    button.clicked.connect(lambda: actions.append("unexpected action"))
    layout.addWidget(button)
nav.add_section("Setup", "Configure only", page)
other = QPushButton("Execute")
other.clicked.connect(lambda: actions.append("unexpected action"))
nav.add_section("Operate", "Execute only when explicitly pressed", other)
nav.resize(400, 360)
nav.show()
QTest.qWait(50)
assert nav.pages[0].verticalScrollBar().maximum() > 0
nav.pages[0].verticalScrollBar().setValue(150)
before = nav.pages[0].verticalScrollBar().value()
for selected in (1, 0, 1, 0):
    nav.buttons[selected].setFocus()
    QTest.keyClick(nav.buttons[selected], Qt.Key_Space)
    app.processEvents()
    assert nav.currentIndex() == selected
    assert sum(button.isChecked() for button in nav.buttons) == 1
    assert sum(area.isVisible() for area in nav.pages) == 1
    assert all(button.isVisible() for button in nav.buttons)
    assert nav.rect().contains(nav.index.geometry())
assert field.text() == "retained value"
assert nav.pages[0].verticalScrollBar().value() == before
assert not actions
''')


def test_dense_workspaces_partition_the_actual_widgets():
    run_qt('''
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from phd.ui.ui_ping import UI
from phd.ui import components
app = QApplication([])
owner = QWidget()
owner.widget_func = QWidget(owner)
for name in dir(UI):
    if name.startswith("_") and not name.startswith("__") and callable(getattr(UI, name)):
        setattr(owner, name, lambda *args, **kwargs: None)
owner._scrollable = UI._scrollable
owner._get_default_ai_execution_model_path = lambda: UI._get_default_ai_execution_model_path(owner)
owner.toggle_yolo_camera = lambda: None
owner.toggle_centering_mode = lambda: None
owner.set_gripper_manual = lambda value: None
owner.AI_PROXIMITY_ENVIRONMENT_TRIAL_COUNT = 10
owner.AI_PROXIMITY_ENVIRONMENT_TRIAL_SECONDS = 60
host = QWidget()
UI.setup_tab4(owner, QVBoxLayout(host))
nav = owner.tools_sections
assert [b.text() for b in nav.buttons] == ["Force meter", "Gripper", "Camera control"]
assert nav.pages[0].isAncestorOf(owner.force_meter_connect_button)
assert nav.pages[1].isAncestorOf(owner.btn_grip_close)
assert nav.pages[2].isAncestorOf(owner.live_yolo_button)
UI.setup_tab5(owner, QVBoxLayout(QWidget(owner)))
assert len(owner.hand_motion_sections.buttons) == 3
assert owner.hand_motion_sections.pages[1].isAncestorOf(owner.hand_send_custom_angles_button)
assert owner.hand_motion_sections.pages[2].isAncestorOf(owner.hand_pose_load_send_button)
UI._build_control_workspace(owner)
assert len(owner.control_sections.buttons) == 4
assert owner.control_sections.pages[0].isAncestorOf(owner.direct_finger_motion_button)
assert owner.control_sections.pages[3].isAncestorOf(owner.admittance_control_button)
UI.setup_tab3(owner, QVBoxLayout(QWidget(owner)))
assert owner.ai_run_sections.currentIndex() == owner.ai_gesture_section_index
assert owner.ai_run_sections.pages[owner.ai_gesture_section_index].isAncestorOf(owner.gesture_recognition_panel)
legacy = owner.ai_run_sections.pages[owner.ai_legacy_section_index]
assert legacy.isAncestorOf(owner.predict_threelevel_hierarchical_transformer_gesture_button)
assert legacy.isAncestorOf(owner.btn_toggle_3lvl_latch)
assert not owner.ai_advanced_controls_group.isAncestorOf(owner.predict_threelevel_hierarchical_transformer_gesture_button)
assert not hasattr(owner, 'ai_direct_execution_swipe_model_button')
assert not hasattr(owner, 'ai_direct_execution_continuous_button')
assert not hasattr(owner, 'ai_direct_execution_use_default_button')
assert owner.ai_direct_execution_dry_run_checkbox.isChecked()
assert owner.ai_direct_execution_velocity_scale_spin.value() == 1.0
assert owner.ai_direct_execution_model_path_input.text().endswith('six_gestures_single_touch_10x10.pt')
assert not owner.gesture_recognition_panel.active
assert not owner.gesture_recognition_panel.details.toggle.isChecked()
assert owner.ai_sub_tabs.currentIndex() == owner.ai_run_tab_index
assert owner.ai_data_sections.currentIndex() == owner.ai_robot_recording_section_index == 0
assert owner.record_trigger_mode_combo.currentData() == "trigger"
assert [owner.record_trigger_mode_combo.itemData(i) for i in range(owner.record_trigger_mode_combo.count())] == [
    "trigger", "no_trigger",
]
assert owner.ai_data_sections.pages[0].isAncestorOf(owner.ai_direct_finger_motion_button)
assert owner.ai_data_sections.pages[1].isAncestorOf(owner.touch_tracking_preview_button)
assert owner.ai_data_sections.pages[2].isAncestorOf(owner.ai_proximity_environment_record_button)
assert not owner.ai_advanced_controls_group.toggle.isChecked()
assert not owner.ai_proximity_setup_group.toggle.isChecked()
assert not hasattr(owner, "ai_direct_finger_motion_robot_button")
assert not hasattr(owner, "ai_teaching_label_buttons")
''')


def test_space_record_shortcut_only_applies_to_visible_robot_demonstrations():
    class Selection:
        def __init__(self, value):
            self.value = value

        def currentIndex(self):
            return self.value

    owner = SimpleNamespace(
        tab_widget=Selection(3), ai_tab_index=3,
        ai_sub_tabs=Selection(1), ai_data_training_tab_index=1,
        ai_data_sections=Selection(1), ai_robot_recording_section_index=1,
    )
    assert AiControlsMixin._is_data_training_subtab_active(owner)
    for selector, value in ((owner.ai_data_sections, 0), (owner.ai_data_sections, 2),
                            (owner.ai_sub_tabs, 0), (owner.tab_widget, 0)):
        previous = selector.value
        selector.value = value
        assert not AiControlsMixin._is_data_training_subtab_active(owner)
        selector.value = previous
    assert not AiControlsMixin._is_data_training_subtab_active(SimpleNamespace())


def test_tracking_shortcuts_only_open_the_requested_page():
    pages = []
    opens = []
    owner = SimpleNamespace(
        _open_touch_tracking_workbench=lambda: opens.append(True),
        touch_tracking_window=SimpleNamespace(
            tabs=SimpleNamespace(setCurrentIndex=pages.append)),
    )
    for index in (0, 2, 3):
        AiControlsMixin._open_touch_tracking_page(owner, index)
    assert pages == [0, 2, 3]
    assert len(opens) == 3


def test_compact_heading_recovers_space_and_survives_workspace_changes():
    run_qt('''
from PyQt5.QtWidgets import QApplication
from phd.ui.components import WorkspaceHeader
app = QApplication([])
header = WorkspaceHeader("Sensor", "Connection instructions")
header.show()
app.processEvents()
full_height = header.sizeHint().height()
header.set_compact(True)
header.set_context("Tools", "Instrument instructions")
app.processEvents()
assert header.eyebrow.isHidden()
assert header.subtitle_label.isHidden()
assert header.sizeHint().height() < full_height
assert header.title_label.text() == "Tools"
header.set_compact(False)
assert not header.subtitle_label.isHidden()
assert header.subtitle_label.text() == "Instrument instructions"
''')
