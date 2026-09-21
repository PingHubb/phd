from __future__ import annotations

import math
import json
import os
from typing import Any, Optional

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QPainter,
    QPen,
    QTextCursor,
)
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QHeaderView,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QSplitterHandle,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from phd.dependence.goodix_usb_sensor import (
    GOODIX_USB_COLUMNS,
    GOODIX_USB_ROWS,
    GOODIX_USB_SOURCE_ID,
    is_goodix_usb_source,
    set_goodix_desktop_touch_enabled,
)
from phd.dependence.paths import ai_resource_path, resource_path
from phd.dependence.humanoid_sensor_registry import (
    humanoid_sensor_extra_column_for_device,
    humanoid_sensor_grid_shape_for_device,
)
from phd.ui import components, icons, nav_tabs, theme
from phd.ui.force_meter_chart import ForceMeterChartWidget
from phd.ui.sensor_zero_mask_window import SensorZeroMaskPanel
from phd.ui.ui_ping_ai_controls import AiControlsMixin
from phd.ui.ui_ping_camera_control import CameraControlMixin
from phd.ui.ui_ping_direct_finger_motion import DirectFingerMotionMixin
from phd.ui.ui_ping_robot_sensor_controls import RobotSensorControlsMixin
from phd.ui.ui_ping_ui_interactions import UiInteractionsMixin


def _safe_import(module_path: str, symbol: str):
    try:
        module = __import__(module_path, fromlist=[symbol])
        return getattr(module, symbol), None
    except Exception as exc:
        return None, exc


def _sensor_source_from_item(item) -> str:
    if item is None:
        return ""
    try:
        source = item.data(Qt.UserRole)
    except (AttributeError, TypeError):
        source = None
    return str(source or item.text() or "").strip()


ArduinoCommander, _ARDUINO_IMPORT_ERROR = _safe_import('phd.dependence.sensor_api', 'ArduinoCommander')
RobotController, _ROBOT_IMPORT_ERROR = _safe_import('phd.dependence.robot_api', 'RobotController')
MyMeshLab, _MESHLAB_IMPORT_ERROR = _safe_import('phd.dependence.func_meshLab', 'MyMeshLab')
MySensor, _SENSOR_IMPORT_ERROR = _safe_import('phd.dependence.func_sensor', 'MySensor')
GripperHelper, _GRIPPER_IMPORT_ERROR = _safe_import('phd.dependence.gripper_api', 'GripperHelper')
YoloWorker, _YOLO_IMPORT_ERROR = _safe_import('phd.dependence.camera_api', 'YoloWorker')


class NullSensorApi:
    def __init__(self):
        self.ser = None

    def read_raw(self):
        return []

    def measure_read_raw_hz(self, duration_sec=1.0):
        return None

    def channel_check(self):
        return []

    def update_cal(self):
        return []


class NullRobotApi:
    use_ros = False

    def get_current_positions(self):
        return 'Robot API unavailable'

    def get_current_tool_position(self):
        return 'Robot API unavailable'

    def send_request(self, request=None):
        return None

    def suspend_end_effector_velocity_mode(self):
        return None

    def enable_end_effector_velocity_mode(self):
        return None

    def stop_end_effector_velocity_mode(self):
        return None

    def set_end_effector_velocity_in_frame(self, *args, **kwargs):
        return None

    def send_positions_joint_angle(self, *args, **kwargs):
        raise RuntimeError('Robot API unavailable')

    def send_positions_tool_position(self, *args, **kwargs):
        raise RuntimeError('Robot API unavailable')

    def hand_services_available(self):
        return False

    def hand_tactile_available(self):
        return False

    def enable_hand_tactile_subscription(self, enabled=True):
        return False

    def get_latest_hand_tactile(self):
        return None

    def hand_tactile_publisher_count(self):
        return 0


class NullGripper:
    ACTION_LIFT = 'lift'
    ACTION_RETRY = 'retry'
    ACTION_RETRY_OPEN = 'retry_open'
    ACTION_MANUAL = 'manual'

    def set_slider_pos(self, *_args, **_kwargs):
        return None

    def open(self, *_args, **_kwargs):
        return None

    def close(self, *_args, **_kwargs):
        return None

    def get_pos_string(self):
        return 'Gripper unavailable'

    def evaluate_grip_attempt(self, grip_fail_count: int):
        return self.ACTION_RETRY, grip_fail_count


class _NoOpRecorder:
    def set_trigger_mode(self, *_args, **_kwargs):
        return None

    def start_record_gesture(self, *_args, **_kwargs):
        return None


class _NoOpToggle:
    def __init__(self):
        self.is_recognizing_gesture = False
        self.last_gesture_time = 0.0
        self.latch_mode = False
        self.anchor_enabled = True

    def toggle_gesture_recognition(self):
        self.is_recognizing_gesture = not self.is_recognizing_gesture

    def toggle_prediction_mode(self):
        return None

    def toggle_model(self):
        return None

    def activate_rule_based(self):
        return None

    def toggle_latch_mode(self):
        self.latch_mode = not self.latch_mode

    def _set_anchor_from_current_frame(self):
        return None

    def toggle_direct_finger_motion(self):
        return None

    def toggle_proximity_control(self):
        return None

    def teach_proximity_reference(self):
        return False

    def toggle_recording(self):
        return False

    def apply_runtime_params(self, **_kwargs):
        return None

    def get_settings(self):
        return {}

    def apply_settings(self, *_args, **_kwargs):
        return None

    def toggle_ai_direct_finger_motion(self, *args, **kwargs):
        return None

    def toggle_ai_direct_finger_motion_execution(self, *args, **kwargs):
        return None

    def set_dry_run_predictions_only(self, *args, **kwargs):
        return None

    def set_teaching_override(self, *args, **kwargs):
        return None


class DisabledSensorFunctions:
    DEFAULT_AI_DIRECT_EXECUTION_MODEL_PATH = ai_resource_path(
        "models",
        "ai_direct_finger_motion",
        "latest_cnn_gru_model_10x10.pt",
    )
    DEFAULT_SENSOR_AVERAGE_WINDOW_SIZE = 3
    DEFAULT_VISUALIZATION_TARGET_HZ = 60.0

    def __init__(self, parent):
        self.parent = parent
        self.record_gesture_class = _NoOpRecorder()
        self.lstm_class = _NoOpToggle()
        self.rule_based_class = _NoOpToggle()
        self.hierarchical_transformer_class = _NoOpToggle()
        self.threelevel_hierarchical_transformer_class = _NoOpToggle()
        self.proximity_control_class = _NoOpToggle()
        self.direct_finger_motion_class = _NoOpToggle()
        self.ai_direct_finger_motion_class = _NoOpToggle()
        self.ai_direct_finger_motion_execution_class = _NoOpToggle()

    def read_sensor_raw_data(self):
        return []

    def read_sensor_raw_ave_data(self):
        return []

    def read_sensor_diff_data(self):
        return []

    def read_sensor_diff_debug_views(self):
        return "Sensor functions are not ready."

    def read_runtime_hz_report(self):
        return (
            "sensor_update_hz: 0.00\n"
            "direct_finger_motion_loop_hz: 0.00\n"
            "direct_finger_motion_running: False\n"
            f"sensor_average_window_size: {self.DEFAULT_SENSOR_AVERAGE_WINDOW_SIZE}\n"
            f"visualization_target_hz: {self.DEFAULT_VISUALIZATION_TARGET_HZ:.2f}"
        )

    def buildScene(self):
        return None

    def updateCal(self):
        return None

    def disconnect_sensor_scene(self):
        return False

    def set_touch_sensitivity(self, *_args, **_kwargs):
        return None

    def get_sensor_average_window_size(self):
        return self.DEFAULT_SENSOR_AVERAGE_WINDOW_SIZE

    def set_sensor_average_window_size(self, *_args, **_kwargs):
        return None

    def get_visualization_target_hz(self):
        return self.DEFAULT_VISUALIZATION_TARGET_HZ

    def set_visualization_target_hz(self, *_args, **_kwargs):
        return None

    def get_sensor_visualization_modes(self):
        return [
            ("point_grid", "Point Grid"),
            ("stereo_field", "Stereo Field"),
            ("heatmap_3d", "3D Heatmap"),
        ]

    def set_sensor_visualization_mode(self, *_args, **_kwargs):
        return None

    def get_stereo_field_settings(self):
        return {
            "ignore_noise_enabled": True,
            "deadband_pct": 0.35,
            "response_scale_pct": 2.0,
            "length_scale": 0.35,
        }

    def set_stereo_field_settings(self, *_args, **_kwargs):
        return None

    def set_saved_sensor_stereo_field_config(self, *_args, **_kwargs):
        return None

    def get_heatmap_settings(self):
        return {
            "palette_3d": "white_red",
            "response_mode": "linear_relative",
            "saturation_pct": 5.0,
            "noise_floor_pct": 0.5,
            "proximity_noise_floor": 20.0,
            "proximity_knee": 100.0,
            "proximity_saturation": 1000.0,
        }

    def set_heatmap_settings(self, *_args, **_kwargs):
        return None

    def set_saved_sensor_heatmap_config(self, *_args, **_kwargs):
        return None

    def set_sensor_point_labels_enabled(self, *_args, **_kwargs):
        return None

    def set_sensor_contact_force_scale(self, *_args, **_kwargs):
        return None

    def set_sensor_geometry_config(self, *_args, **_kwargs):
        return None

    def set_saved_sensor_geometry_config(self, *_args, **_kwargs):
        return None

    def save_current_sensor_perspective(self):
        return False

    def restore_saved_sensor_perspective(self, *_args, **_kwargs):
        return False

    def get_ai_direct_finger_motion_execution_default_model_path(self):
        return self.DEFAULT_AI_DIRECT_EXECUTION_MODEL_PATH


class NullMeshLab:
    def __init__(self, parent):
        self.parent = parent

    def addRobot(self):
        return None

    def addRobotInDialog(self):
        return None

    def addDexterousHandInDialog(self):
        return None

    def set_ai_admittance_control_enabled(self, *_args, **_kwargs):
        return None

    def start_ai_proximity_motion(self):
        return False, "Robot sensor mapping is unavailable"

    def update_ai_proximity_motion(self, *_args, **_kwargs):
        return {
            "ok": False,
            "error": "Robot sensor mapping is unavailable",
        }

    def stop_ai_proximity_motion(self):
        return None

    def is_ai_proximity_motion_active(self):
        return False

    def set_secondary_background_reference_enabled(self, *_args, **_kwargs):
        return None

    def updateDexterousHandTactile(self, *_args, **_kwargs):
        return None


class PlotterWidget(QWidget):
    filesDropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        # Style is now handled by QSS, but we can set object name for specific rules if needed
        # self.setObjectName("plotterWidget")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        file_paths = [str(url.toLocalFile()) for url in event.mimeData().urls()]
        self.filesDropped.emit(file_paths)


class LogConsoleWindow(QDialog):
    """Modeless log utility that never consumes visualization workspace."""

    _GEOMETRY_KEY = "windows/log_console_geometry"

    def __init__(self, log_display: QTextEdit, parent=None):
        super().__init__(parent, Qt.Window)
        self.log_display = log_display
        self._settings = QtCore.QSettings("PingLab", "PingLab")
        self.setWindowTitle("Application Log")
        self.setModal(False)
        self.setMinimumSize(520, 260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            theme.SPACE_MD,
            theme.SPACE_MD,
            theme.SPACE_MD,
            theme.SPACE_MD,
        )
        layout.setSpacing(theme.SPACE_SM)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(theme.SPACE_XS)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Find in log")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setMaximumWidth(280)
        toolbar.addWidget(self.search_input, 1)

        self.search_button = components.IconButton(
            "search", "Find the next matching log entry"
        )
        toolbar.addWidget(self.search_button)
        toolbar.addStretch(1)

        self.auto_scroll_check = QCheckBox("Auto-scroll")
        self.auto_scroll_check.setChecked(True)
        toolbar.addWidget(self.auto_scroll_check)
        toolbar.addWidget(
            components.divider(Qt.Vertical)
        )

        self.copy_button = components.IconButton(
            "copy", "Copy the selection, or copy the entire log"
        )
        self.clear_button = components.IconButton(
            "trash", "Clear the application log"
        )
        self.export_button = components.IconButton(
            "save", "Export the application log as a text file"
        )
        toolbar.addWidget(self.copy_button)
        toolbar.addWidget(self.clear_button)
        toolbar.addWidget(self.export_button)
        layout.addLayout(toolbar)
        layout.addWidget(self.log_display, 1)

        self.search_input.returnPressed.connect(self._find_next)
        self.search_button.clicked.connect(self._find_next)
        self.copy_button.clicked.connect(self._copy_log)
        self.clear_button.clicked.connect(self.log_display.clear)
        self.export_button.clicked.connect(self._export_log)
        self.log_display.textChanged.connect(self._scroll_to_latest)

        saved_geometry = self._settings.value(self._GEOMETRY_KEY)
        if not saved_geometry or not self.restoreGeometry(saved_geometry):
            components.size_to_screen(self, 760, 400)

    def _find_next(self):
        query = self.search_input.text().strip()
        if not query:
            return
        if self.log_display.find(query):
            return
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.Start)
        self.log_display.setTextCursor(cursor)
        self.log_display.find(query)

    def _copy_log(self):
        cursor = self.log_display.textCursor()
        if cursor.hasSelection():
            self.log_display.copy()
            return
        QApplication.clipboard().setText(self.log_display.toPlainText())

    def _export_log(self):
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Application Log",
            "pinglab_log.txt",
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".txt"
        try:
            with open(path, "w", encoding="utf-8") as output:
                output.write(self.log_display.toPlainText())
        except OSError as exc:
            self.log_display.append(f"Log export failed: {exc}")

    def _scroll_to_latest(self):
        if not self.auto_scroll_check.isChecked():
            return
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def remember_geometry(self):
        self._settings.setValue(self._GEOMETRY_KEY, self.saveGeometry())

    def closeEvent(self, event):
        self.remember_geometry()
        super().closeEvent(event)


class ProximityRecordingChartWidget(QWidget):
    def __init__(self, title: str, series, parent=None):
        super().__init__(parent)
        self.title = title
        self.series = list(series or [])
        self.setMinimumHeight(260)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(54, 30, -18, -38)
        painter.setPen(QPen(QColor("#d0d0d0")))
        painter.drawText(10, 20, self.title)
        painter.drawRect(rect)

        all_x = []
        all_y = []
        for _label, xs, ys, _color in self.series:
            all_x.extend([x for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)])
            all_y.extend([y for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)])
        if not all_x or not all_y:
            painter.drawText(rect, Qt.AlignCenter, "No finite data")
            return

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        if abs(x_max - x_min) < 1e-9:
            x_max = x_min + 1.0
        if abs(y_max - y_min) < 1e-9:
            pad = max(1.0, abs(y_min) * 0.1)
            y_min -= pad
            y_max += pad
        else:
            pad = (y_max - y_min) * 0.08
            y_min -= pad
            y_max += pad

        def map_xy(x, y):
            px = rect.left() + (float(x) - x_min) / (x_max - x_min) * rect.width()
            py = rect.bottom() - (float(y) - y_min) / (y_max - y_min) * rect.height()
            return px, py

        legend_x = rect.left()
        legend_y = rect.bottom() + 18
        for label, xs, ys, color in self.series:
            pen = QPen(QColor(color), 2)
            painter.setPen(pen)
            previous = None
            for x, y in zip(xs, ys):
                if not (np.isfinite(x) and np.isfinite(y)):
                    previous = None
                    continue
                point = map_xy(x, y)
                if previous is not None:
                    painter.drawLine(int(previous[0]), int(previous[1]), int(point[0]), int(point[1]))
                previous = point
            painter.drawLine(legend_x, legend_y - 4, legend_x + 18, legend_y - 4)
            painter.setPen(QPen(QColor("#d0d0d0")))
            painter.drawText(legend_x + 24, legend_y, label)
            legend_x += 130

        painter.setPen(QPen(QColor("#a0a0a0")))
        painter.drawText(8, rect.top() + 5, f"{y_max:.3g}")
        painter.drawText(8, rect.bottom(), f"{y_min:.3g}")
        painter.drawText(rect.left(), self.height() - 8, f"{x_min:.2f}s")
        painter.drawText(rect.right() - 60, self.height() - 8, f"{x_max:.2f}s")


class RobotScriptSendWidget(QWidget):
    """Script editor shown only after pressing Send Script (Robot tab)."""

    transmit_script = QtCore.pyqtSignal(str)

    def __init__(self, robot_api=None, parent=None):
        super().__init__(parent)
        self.robot_api = robot_api
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.SPACE_SM)
        script_section, script_list = components.section(
            "Robot script",
            "Send an explicit TM script command to the controller.",
        )
        self.robot_script_input = QPlainTextEdit()
        self.robot_script_input.setPlaceholderText(
            'Example: PTP("JPP",0,-30,-45,0,-90,0,100,0,100,true)'
        )
        self.robot_script_input.setMinimumHeight(88)
        self.robot_script_input.setTabChangesFocus(True)
        script_list.add_stacked("Script", self.robot_script_input)
        self.transmit_script_button = QPushButton("Transmit script")
        components.apply_variant(self.transmit_script_button, "primary")
        self.transmit_script_button.clicked.connect(self._on_transmit_clicked)
        script_list.add(self.transmit_script_button)
        outer.addWidget(script_section)

    def _on_transmit_clicked(self):
        self.transmit_script.emit(self.robot_script_input.toPlainText().strip())

    def toggle_visibility(self):
        self.setVisible(not components.is_visible_to_parent(self))


class RobotPositionWidget(QWidget):
    PRESET_CONFIG_PATH = resource_path("config", "robot_position_presets.json")

    def __init__(self, robot_api=None, parent=None):
        super().__init__(parent)
        self.robot_api = robot_api
        self.setLayout(QVBoxLayout())
        self.position_edits = []
        self.labels = []
        # --- MODIFICATION: Removed inline stylesheet ---

        # Preset 1 (deg): [-45, 0, -90, 0, -90, 0]
        self.presets = {
            1: [math.radians(d) for d in (-45.0, 0.0, -90.0, 0.0, -90.0, 0.0)],
            2: [-1.1, -0.43900, -1.005029724, -0.143107, -1.57, 0.0],
            3: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            4: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            5: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        self._load_presets_from_disk()
        self.current_preset_number = 1

        self.angle_group_box, angle_list = components.section(
            "Joint targets",
            "Angles are in radians; use the step buttons for precise changes.",
        )

        step_rad = math.radians(10.0)
        for i in range(6):
            row_widget = QWidget()
            row_h = QHBoxLayout(row_widget)
            row_h.setContentsMargins(0, 2, 0, 2)
            row_h.setSpacing(6)

            label = QLabel(f"Joint {i + 1}")
            line_edit = QLineEdit()
            line_edit.setText(f"{self.presets[1][i]:.4f}")
            line_edit.setMinimumWidth(72)

            btn_minus = QPushButton("-10°")
            btn_plus = QPushButton("+10°")
            btn_minus.setFixedWidth(52)
            btn_plus.setFixedWidth(52)
            btn_minus.clicked.connect(lambda checked, idx=i, dr=-step_rad: self._nudge_joint_rad(idx, dr))
            btn_plus.clicked.connect(lambda checked, idx=i, dr=step_rad: self._nudge_joint_rad(idx, dr))

            row_h.addWidget(line_edit, 1)
            row_h.addWidget(btn_minus)
            row_h.addWidget(btn_plus)

            angle_list.add(components.SettingsRow(f"Joint {i + 1}", row_widget))

            self.labels.append(label)
            self.position_edits.append(line_edit)

        # Initially hide the group box
        self.angle_group_box.setVisible(False)
        self.layout().addWidget(self.angle_group_box)

        action_widget, action_list = components.section("Send", "Choose or capture a preset, then send it.")
        preset_row = QWidget()
        preset_layout = QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(theme.SPACE_XS)

        self.preset_buttons = []
        for i in range(1, 6):
            btn = QPushButton(str(i))
            btn.clicked.connect(lambda checked, p=i: self.apply_preset(p))
            preset_layout.addWidget(btn)
            self.preset_buttons.append(btn)
        preset_layout.addStretch(1)
        action_list.add(components.SettingsRow("Preset", preset_row))

        command_row = QWidget()
        action_layout = QHBoxLayout(command_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        self.load_current_angle_button = QPushButton("Capture Current")
        self.load_current_angle_button.clicked.connect(self.load_current_robot_angle_into_current_preset)
        action_layout.addWidget(self.load_current_angle_button)
        self._update_load_current_angle_button_state()

        action_layout.addStretch()

        self.send_button = QPushButton("Send Positions")
        components.apply_variant(self.send_button, "primary")
        self.send_button.clicked.connect(self.send_positions)
        action_layout.addWidget(self.send_button)
        action_list.add(command_row)

        self.layout().addWidget(action_widget)
        self.action_widget = action_widget
        self.action_widget.setVisible(False)

    def apply_preset(self, preset_number):
        preset_values = self.presets.get(preset_number)
        if preset_values:
            self.current_preset_number = int(preset_number)
            for i, value in enumerate(preset_values):
                self.position_edits[i].setText(f"{value:.4f}")
            self._update_load_current_angle_button_state()

    def _update_load_current_angle_button_state(self):
        btn = getattr(self, "load_current_angle_button", None)
        if btn is None:
            return
        preset_number = int(getattr(self, "current_preset_number", 1))
        allow = preset_number >= 3
        btn.setEnabled(allow)
        if allow:
            btn.setToolTip(
                f"Load current robot joint angles and save into Preset {preset_number}."
            )
        else:
            btn.setToolTip(
                "Disabled for Preset 1/2 to avoid accidental overwrite. "
                "Switch to Preset 3 or 4 to enable."
            )

    def load_current_robot_angle_into_current_preset(self):
        api = self.robot_api
        if api is None or not hasattr(api, 'get_current_positions'):
            print("Robot API unavailable: cannot load current joint angle.")
            return

        try:
            current_positions = list(api.get_current_positions() or [])
        except Exception as e:
            print(f"Failed to read current robot joint angles: {e}")
            return

        if len(current_positions) < 6:
            print("Current robot joint angles unavailable or incomplete.")
            return

        current_positions = [float(v) for v in current_positions[:6]]
        preset_number = int(getattr(self, "current_preset_number", 1))
        if preset_number not in self.presets:
            preset_number = 1
            self.current_preset_number = 1

        self.presets[preset_number] = current_positions
        for i, value in enumerate(current_positions):
            self.position_edits[i].setText(f"{value:.4f}")
        self._save_presets_to_disk()

        print(
            f"Loaded current robot joint angles into Preset {preset_number}: "
            f"{[round(v, 4) for v in current_positions]}"
        )

    def _save_presets_to_disk(self):
        payload = {
            "version": 1,
            "presets": {
                str(k): [float(v) for v in values]
                for k, values in sorted(self.presets.items())
                if isinstance(k, int) and len(values) >= 6
            },
        }
        try:
            os.makedirs(os.path.dirname(self.PRESET_CONFIG_PATH), exist_ok=True)
            with open(self.PRESET_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"Failed to save robot position presets: {e}")

    def _load_presets_from_disk(self):
        try:
            if not os.path.isfile(self.PRESET_CONFIG_PATH):
                return
            with open(self.PRESET_CONFIG_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            loaded = payload.get("presets", {}) if isinstance(payload, dict) else {}
            if not isinstance(loaded, dict):
                return
            for key, values in loaded.items():
                try:
                    preset_number = int(key)
                    if not isinstance(values, list) or len(values) < 6:
                        continue
                    self.presets[preset_number] = [float(v) for v in values[:6]]
                except Exception:
                    continue
        except Exception as e:
            print(f"Failed to load robot position presets: {e}")

    def _nudge_joint_rad(self, index: int, delta_rad: float):
        edit = self.position_edits[index]
        try:
            v = float(edit.text())
        except ValueError:
            v = 0.0
        edit.setText(f"{v + float(delta_rad):.4f}")

    def send_positions(self):
        try:
            positions = [float(edit.text()) for edit in self.position_edits]
        except ValueError:
            print("Invalid input! Please enter valid numbers.")
            return

        api = self.robot_api
        if api is None or not hasattr(api, 'send_positions_joint_angle'):
            print("Robot API unavailable.")
            return

        try:
            print("Sending positions:", positions)
            ok = api.send_positions_joint_angle(positions)
            if not ok:
                print("Failed to queue joint positions: robot API returned False.")
        except Exception as e:
            print(f"Failed to send joint positions: {e}")

    def toggle_visibility(self):
        isVisible = not components.is_visible_to_parent(self)
        self.angle_group_box.setVisible(isVisible)
        self.action_widget.setVisible(isVisible)
        self.setVisible(isVisible)


class RobotToolPositionWidget(QWidget):
    def __init__(self, robot_api=None, parent=None):
        super().__init__(parent)
        self.robot_api = robot_api
        self.setLayout(QVBoxLayout())
        # --- MODIFICATION: Removed inline stylesheet ---

        self.presets = {
            1: ([-0.55, 0.1, 0.2], [0.0, -1.0, 0.0, 0.0]),
            2: ([0.115, 0.322, 0.443], [0.01, 0.348, 0.937, 0.033])
        }
        self.labels = {}
        self.line_edits = {}

        # Group inputs into Position and Orientation using the same inset-list
        # language as the rest of the control panel.
        self.input_container = QWidget()
        top_layout = QVBoxLayout(self.input_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(theme.SPACE_SM)

        position_group, position_layout = components.section(
            "Position", "Tool position in metres.",
        )
        orientation_group, orientation_layout = components.section(
            "Orientation", "Tool quaternion components.",
        )

        self.setupControls('X', self.presets[1][0][0], position_layout)
        self.setupControls('Y', self.presets[1][0][1], position_layout)
        self.setupControls('Z', self.presets[1][0][2], position_layout)

        self.setupControls('w', self.presets[1][1][0], orientation_layout)
        self.setupControls('i', self.presets[1][1][1], orientation_layout)
        self.setupControls('j', self.presets[1][1][2], orientation_layout)
        self.setupControls('k', self.presets[1][1][3], orientation_layout)

        top_layout.addWidget(position_group)
        top_layout.addWidget(orientation_group)

        self.layout().addWidget(self.input_container)
        self.input_container.setVisible(False)

        action_widget, action_list = components.section("Send")
        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)

        self.preset_buttons = []
        for i in range(1, 3):
            btn = QPushButton(f"Preset {i}")
            btn.clicked.connect(lambda checked, p=i: self.apply_preset(p))
            action_layout.addWidget(btn)
            self.preset_buttons.append(btn)

        action_layout.addStretch()

        self.send_button = QPushButton("Send Tool Position")
        components.apply_variant(self.send_button, "primary")
        self.send_button.clicked.connect(self.send_positions)
        action_layout.addWidget(self.send_button)

        action_list.add(action_row)

        self.layout().addWidget(action_widget)
        self.action_widget = action_widget
        self.action_widget.setVisible(False)

    def apply_preset(self, preset_number):
        preset_values = self.presets.get(preset_number)
        if preset_values:
            positions, quaternion = preset_values
            for i, coord in enumerate(['X', 'Y', 'Z']):
                self.line_edits[coord].setText(f"{positions[i]:.2f}")
            for i, part in enumerate(['w', 'i', 'j', 'k']):
                self.line_edits[part].setText(f"{quaternion[i]:.2f}")

    def setupControls(self, identifier, preset_value, layout):
        label = QLabel(f"{identifier}:")
        line_edit = QLineEdit()
        line_edit.setText(f"{preset_value:.2f}")
        layout.add(components.SettingsRow(identifier, line_edit))

        self.labels[identifier] = label
        self.line_edits[identifier] = line_edit

    def send_positions(self):
        api = self.robot_api
        if api is None or not hasattr(api, 'send_positions_tool_position'):
            print("Robot API unavailable.")
            return

        try:
            positions = [float(self.line_edits[coord].text()) for coord in ['X', 'Y', 'Z']]
            quaternion = tuple(float(self.line_edits[part].text()) for part in ['w', 'i', 'j', 'k'])
            print("Sending tool position:", positions, "with quaternion:", quaternion)
            api.send_positions_tool_position(positions, quaternion)
        except ValueError:
            print("Invalid input! Please enter valid numbers.")
        except Exception as e:
            print(f"Failed to send tool position: {e}")

    def toggle_visibility(self):
        isVisible = not components.is_visible_to_parent(self)
        self.input_container.setVisible(isVisible)
        self.action_widget.setVisible(isVisible)
        self.setVisible(isVisible)


class RobotToolFramePositionWidget(QWidget):
    """
    Button-based end-effector velocity control expressed in the TOOL frame.

    Improved layout:
      - speed sliders at top
      - 2 sub-tabs: Linear / Angular
      - direct 6D (x y z rx ry rz) send row
      - compact stop buttons at bottom
    """

    def __init__(self, robot_api, log_display=None, parent=None):
        super().__init__(parent)
        self.robot_api = robot_api
        self.log_display = log_display

        self.linear_speed = 0.02   # m/s
        # Angular velocity for Rx/Ry/Rz buttons (rad/s). Slider maps 1..10 →
        # 0.0001 .. 0.001 (fine control for slow tool-frame rotation).
        self.angular_speed = 0.0001
        self._velocity_mode_active = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        control_group, control_list = components.section(
            "Tool-frame velocity",
            "Set a speed, then send translation, rotation, or a direct 6D command.",
        )
        outer.addWidget(control_group)
        control_content = QWidget()
        main_layout = QVBoxLayout(control_content)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)
        control_list.add(control_content)

        # -------------------------
        # Speed sliders
        # -------------------------
        linear_row = QHBoxLayout()
        linear_row.addWidget(QLabel("XYZ speed:"))
        self.linear_slider = QSlider(Qt.Horizontal)
        self.linear_slider.setRange(1, 200)  # 0.001 -> 0.200 m/s
        self.linear_slider.setValue(int(self.linear_speed * 1000))
        self.linear_slider.valueChanged.connect(self._on_linear_slider)
        self.linear_label = QLabel()
        self.linear_label.setFixedWidth(90)
        linear_row.addWidget(self.linear_slider)
        linear_row.addWidget(self.linear_label)
        main_layout.addLayout(linear_row)

        angular_row = QHBoxLayout()
        angular_row.addWidget(QLabel("RXYZ speed:"))
        self.angular_slider = QSlider(Qt.Horizontal)
        # Integer 1..10 → 0.0001 .. 0.001 rad/s (step 0.0001)
        self.angular_slider.setRange(1, 10)
        self.angular_slider.setSingleStep(1)
        self.angular_slider.setPageStep(1)
        self.angular_slider.setValue(max(1, int(round(self.angular_speed / 1e-4))))
        self.angular_slider.setToolTip(
            "Angular velocity for Rx/Ry/Rz buttons: 0.0001 to 0.001 rad/s (step 0.0001)."
        )
        self.angular_slider.valueChanged.connect(self._on_angular_slider)
        self.angular_label = QLabel()
        self.angular_label.setFixedWidth(130)
        angular_row.addWidget(self.angular_slider)
        angular_row.addWidget(self.angular_label)
        main_layout.addLayout(angular_row)

        self._refresh_speed_labels()

        # -------------------------
        # Sub-tabs for buttons
        # -------------------------
        self.motion_tabs = QTabWidget()
        main_layout.addWidget(self.motion_tabs)

        # Linear tab
        linear_tab = QWidget()
        linear_tab_layout = QGridLayout(linear_tab)
        linear_tab_layout.setContentsMargins(8, 8, 8, 8)
        linear_tab_layout.setHorizontalSpacing(8)
        linear_tab_layout.setVerticalSpacing(8)

        linear_tab_layout.addWidget(
            self._make_btn("X+", lambda: self._send_linear(self.linear_speed, 0.0, 0.0)), 0, 0
        )
        linear_tab_layout.addWidget(
            self._make_btn("X-", lambda: self._send_linear(-self.linear_speed, 0.0, 0.0)), 0, 1
        )
        linear_tab_layout.addWidget(
            self._make_btn("Y+", lambda: self._send_linear(0.0, self.linear_speed, 0.0)), 1, 0
        )
        linear_tab_layout.addWidget(
            self._make_btn("Y-", lambda: self._send_linear(0.0, -self.linear_speed, 0.0)), 1, 1
        )
        linear_tab_layout.addWidget(
            self._make_btn("Z+", lambda: self._send_linear(0.0, 0.0, self.linear_speed)), 2, 0
        )
        linear_tab_layout.addWidget(
            self._make_btn("Z-", lambda: self._send_linear(0.0, 0.0, -self.linear_speed)), 2, 1
        )

        self.motion_tabs.addTab(linear_tab, "Linear")

        # Angular tab
        angular_tab = QWidget()
        angular_tab_layout = QGridLayout(angular_tab)
        angular_tab_layout.setContentsMargins(8, 8, 8, 8)
        angular_tab_layout.setHorizontalSpacing(8)
        angular_tab_layout.setVerticalSpacing(8)

        angular_tab_layout.addWidget(
            self._make_btn("Rx+", lambda: self._send_angular(self.angular_speed, 0.0, 0.0)), 0, 0
        )
        angular_tab_layout.addWidget(
            self._make_btn("Rx-", lambda: self._send_angular(-self.angular_speed, 0.0, 0.0)), 0, 1
        )
        angular_tab_layout.addWidget(
            self._make_btn("Ry+", lambda: self._send_angular(0.0, self.angular_speed, 0.0)), 1, 0
        )
        angular_tab_layout.addWidget(
            self._make_btn("Ry-", lambda: self._send_angular(0.0, -self.angular_speed, 0.0)), 1, 1
        )
        angular_tab_layout.addWidget(
            self._make_btn("Rz+", lambda: self._send_angular(0.0, 0.0, self.angular_speed)), 2, 0
        )
        angular_tab_layout.addWidget(
            self._make_btn("Rz-", lambda: self._send_angular(0.0, 0.0, -self.angular_speed)), 2, 1
        )

        self.motion_tabs.addTab(angular_tab, "Angular")

        # Direct 6D tab
        direct_tab = QWidget()
        direct_tab_layout = QGridLayout(direct_tab)
        direct_tab_layout.setContentsMargins(8, 8, 8, 8)
        direct_tab_layout.setHorizontalSpacing(8)
        direct_tab_layout.setVerticalSpacing(6)

        self.direct_vel_inputs = {}
        axes = ("x", "y", "z", "rx", "ry", "rz")
        for idx, axis in enumerate(axes):
            row = idx // 3
            col = (idx % 3) * 2
            label = QLabel(f"{axis}:")
            entry = QLineEdit("0.0")
            entry.setMaximumWidth(90)
            entry.setToolTip("Velocity in tool frame (m/s for x,y,z; rad/s for rx,ry,rz).")
            direct_tab_layout.addWidget(label, row, col)
            direct_tab_layout.addWidget(entry, row, col + 1)
            self.direct_vel_inputs[axis] = entry

        self.send_direct_6d_button = QPushButton("Send 6D Velocity")
        self.send_direct_6d_button.setMinimumHeight(32)
        self.send_direct_6d_button.clicked.connect(self._send_direct_6d_velocity)
        direct_tab_layout.addWidget(self.send_direct_6d_button, 2, 0, 1, 6)

        self.motion_tabs.addTab(direct_tab, "Direct 6D")

        # -------------------------
        # Stop buttons
        # -------------------------
        stop_row = QHBoxLayout()

        # Red, and always the widest thing in this panel: stopping must be the
        # easiest control to hit.
        self.btn_stop_all = QPushButton("STOP (All 0)")
        self.btn_stop_all.setMinimumHeight(theme.CONTROL_HEIGHT_LG)
        components.apply_variant(self.btn_stop_all, "danger")
        self.btn_stop_all.clicked.connect(self.stop_all_velocity)
        stop_row.addWidget(self.btn_stop_all, 2)

        self.btn_stop_velocity_mode = QPushButton("Stop Velocity Mode")
        self.btn_stop_velocity_mode.setMinimumHeight(theme.CONTROL_HEIGHT_LG)
        components.apply_variant(self.btn_stop_velocity_mode, "danger")
        self.btn_stop_velocity_mode.clicked.connect(self.stop_velocity_mode)
        stop_row.addWidget(self.btn_stop_velocity_mode, 1)

        main_layout.addLayout(stop_row)

        self.setVisible(False)

    def _make_btn(self, text, fn):
        btn = QPushButton(text)
        btn.setMinimumHeight(34)
        btn.clicked.connect(fn)
        return btn

    def _on_linear_slider(self, value: int):
        self.linear_speed = value / 1000.0
        self._refresh_speed_labels()

    def _on_angular_slider(self, value: int):
        self.angular_speed = max(1, min(10, int(value))) * 1e-4
        self._refresh_speed_labels()

    def _refresh_speed_labels(self):
        self.linear_label.setText(f"{self.linear_speed:.3f} m/s")
        self.angular_label.setText(f"{self.angular_speed:.4f} rad/s")

    def _append_log(self, message: str):
        if self.log_display is not None:
            try:
                self.log_display.append(message)
            except Exception:
                pass

    def _robot_velocity_mode_active(self):
        return bool(
            self._velocity_mode_active
            or getattr(self.robot_api, "_end_effector_velocity_mode_active", False)
        )

    @staticmethod
    def _velocity_is_zero(v_lin, v_rot):
        values = list(v_lin or []) + list(v_rot or [])
        return all(abs(float(value)) < 1e-12 for value in values)

    def _ensure_velocity_mode(self):
        if self._robot_velocity_mode_active():
            self._velocity_mode_active = True
            return True
        if hasattr(self.robot_api, "enter_end_effector_velocity_mode"):
            ok = self.robot_api.enter_end_effector_velocity_mode(suspend_existing=True)
            self._velocity_mode_active = bool(ok)
            return bool(ok)
        if not hasattr(self.robot_api, "send_request"):
            return False
        if hasattr(self.robot_api, "suspend_end_effector_velocity_mode"):
            self.robot_api.send_request(self.robot_api.suspend_end_effector_velocity_mode())
        if hasattr(self.robot_api, "enable_end_effector_velocity_mode"):
            ok = self.robot_api.send_request(self.robot_api.enable_end_effector_velocity_mode())
            self._velocity_mode_active = bool(ok)
            return bool(ok)
        return False

    def _send_linear(self, x, y, z):
        self._send_velocity([float(x), float(y), float(z)], [0.0, 0.0, 0.0])

    def _send_angular(self, rx, ry, rz):
        self._send_velocity([0.0, 0.0, 0.0], [float(rx), float(ry), float(rz)])

    def _send_velocity(self, v_lin, v_rot):
        try:
            if not hasattr(self.robot_api, "send_request"):
                print("[RobotToolFramePositionWidget] robot_api has no send_request()")
                return

            v_lin = [float(v_lin[0]), float(v_lin[1]), float(v_lin[2])]
            v_rot = [float(v_rot[0]), float(v_rot[1]), float(v_rot[2])]
            is_zero = self._velocity_is_zero(v_lin, v_rot)
            if not is_zero and not self._ensure_velocity_mode():
                self._append_log("❌ [VelocityControl] Could not enter tool velocity mode.")
                return
            if is_zero and not self._robot_velocity_mode_active():
                return

            if hasattr(self.robot_api, "send_end_effector_velocity_in_frame"):
                self.robot_api.send_end_effector_velocity_in_frame(
                    v_lin,
                    v_rot,
                    frame="tool",
                    ensure_mode=not is_zero,
                )
                return

            if hasattr(self.robot_api, "set_end_effector_velocity_in_frame"):
                self.robot_api.send_request(
                    self.robot_api.set_end_effector_velocity_in_frame(v_lin, v_rot, frame="tool")
                )
            elif hasattr(self.robot_api, "set_end_effector_velocity"):
                vel6 = [
                    float(v_lin[0]), float(v_lin[1]), float(v_lin[2]),
                    float(v_rot[0]), float(v_rot[1]), float(v_rot[2])
                ]
                self.robot_api.send_request(self.robot_api.set_end_effector_velocity(vel6))

        except Exception as e:
            msg = f"[VelocityControl] Failed to send velocity command: {e}"
            print(msg)
            self._append_log(f"❌ {msg}")

    def _send_direct_6d_velocity(self):
        try:
            x = float(self.direct_vel_inputs["x"].text())
            y = float(self.direct_vel_inputs["y"].text())
            z = float(self.direct_vel_inputs["z"].text())
            rx = float(self.direct_vel_inputs["rx"].text())
            ry = float(self.direct_vel_inputs["ry"].text())
            rz = float(self.direct_vel_inputs["rz"].text())
        except Exception:
            self._append_log("❌ [VelocityControl] Invalid 6D input. Please enter numeric x/y/z/rx/ry/rz.")
            return
        self._send_velocity([x, y, z], [rx, ry, rz])

    def stop_all_velocity(self):
        self._send_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def stop_velocity_mode(self):
        """Safely exit velocity mode: zero velocity, then Suspend -> Stop."""
        try:
            if not hasattr(self.robot_api, "send_request"):
                print("[RobotToolFramePositionWidget] robot_api has no send_request()")
                return

            if hasattr(self.robot_api, "exit_end_effector_velocity_mode"):
                self.robot_api.exit_end_effector_velocity_mode(send_zero=True)
            else:
                if hasattr(self.robot_api, "set_end_effector_velocity"):
                    self.robot_api.send_request(self.robot_api.set_end_effector_velocity([0.0] * 6))
                if hasattr(self.robot_api, "suspend_end_effector_velocity_mode"):
                    self.robot_api.send_request(self.robot_api.suspend_end_effector_velocity_mode())
                if hasattr(self.robot_api, "stop_end_effector_velocity_mode"):
                    self.robot_api.send_request(self.robot_api.stop_end_effector_velocity_mode())
            self._velocity_mode_active = False

        except Exception as e:
            self._velocity_mode_active = False
            msg = f"[VelocityControl] Failed to stop velocity mode: {e}"
            print(msg)
            self._append_log(f"❌ {msg}")

    def toggle_visibility(self):
        self.setVisible(not components.is_visible_to_parent(self))


class _MainControlPanelSplitterHandle(QSplitterHandle):
    """Outer splitter handle with a centered tabs-panel toggle."""

    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self.toggle_button = QToolButton(self)
        self.toggle_button.setAutoRaise(False)
        self.toggle_button.setFixedSize(18, 44)
        self.toggle_button.setIconSize(QtCore.QSize(12, 12))
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.clicked.connect(
            self._toggle_main_control_panel
        )
        self.toggle_button.setStyleSheet(
            "QToolButton {"
            f" background-color: {theme.SURFACE};"
            f" border: 1px solid {theme.BORDER};"
            f" border-radius: {theme.RADIUS_MD}px;"
            " padding: 0;"
            "}"
            "QToolButton:hover {"
            f" background-color: {theme.ACCENT_SOFT};"
            f" border-color: {theme.ACCENT_BORDER};"
            "}"
        )
        self.set_panel_visible(True)

    def _toggle_main_control_panel(self):
        splitter = self.splitter()
        callback = getattr(
            splitter,
            "toggle_main_control_panel",
            None,
        )
        if callable(callback):
            callback()

    def set_panel_visible(self, visible):
        visible = bool(visible)
        self.toggle_button.setIcon(
            icons.icon(
                "chevron-right" if visible else "chevron-left",
                color=theme.TEXT_SECONDARY,
                size=12,
            )
        )
        self.toggle_button.setToolTip(
            "Hide the control panel" if visible else "Show the control panel"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        x = max(0, (self.width() - self.toggle_button.width()) // 2)
        y = max(0, (self.height() - self.toggle_button.height()) // 2)
        self.toggle_button.move(x, y)


class _WidthFittingScrollArea(QScrollArea):
    """Scroll area that reports its page's natural width as its own.

    A plain QScrollArea advertises a small fixed width hint regardless of what
    it contains, which would let the control panel be sized narrower than its
    widest control row and clip the labels at the right edge. Height is left
    alone -- vertical scrolling is the whole point of the wrapper.
    """

    def sizeHint(self):
        hint = super().sizeHint()
        page = self.widget()
        if page is None:
            return hint
        extra = 2 * self.frameWidth()
        scrollbar = self.verticalScrollBar()
        if scrollbar is not None:
            extra += scrollbar.sizeHint().width()
        hint.setWidth(page.sizeHint().width() + extra)
        return hint


class UI(
    AiControlsMixin,
    CameraControlMixin,
    DirectFingerMotionMixin,
    RobotSensorControlsMixin,
    UiInteractionsMixin,
    QSplitter,
):
    def __init__(self, orientation: QtCore.Qt.Orientation):
        super().__init__(orientation)

        self._startup_messages = []
        self.features = {
            "sensor_driver": ArduinoCommander is not None,
            "robot_driver": RobotController is not None,
            "mesh_driver": MyMeshLab is not None,
            "sensor_module": MySensor is not None,
            "gripper_driver": GripperHelper is not None,
            "camera_driver": YoloWorker is not None,
        }

        self.sensor_api = None
        self.robot_api = None
        self.gripper = None
        self.mesh_functions = None
        self.sensor_functions = None
        self._yolo_worker_class = YoloWorker
        self.yolo_worker = None
        self.cam_window = None
        self.cam_label = None
        self.centering_active = False
        self._is_shutting_down = False
        self._software_motion_stopped = False
        self._main_control_panel_collapsed = False
        self._main_control_panel_last_width = theme.CONTROL_PANEL_WIDTH
        self._main_control_panel_state_syncing = False
        self._main_control_panel_user_sized = False
        self._log_unread_count = 0

        self.gripper_closed_flag = False
        self.grip_fail_count = 0
        self.manual_mode_active = False
        self.is_lifting = False
        self.stationary_mode = False
        self.grab_triggered = False

        self.manual_watchdog_timer = QTimer(self)
        self.manual_watchdog_timer.setInterval(500)
        self.manual_watchdog_timer.timeout.connect(self.check_manual_timeout)

        self._bootstrap_core_services()

        self.setHandleWidth(22)
        self.setup_layout()
        self._bootstrap_ui_services()
        self._flush_startup_messages()

        self.connect_function()
        self.adjust_splitter_sizes()
        self._apply_startup_feature_state()
        self._runtime_availability_timer = QTimer(self)
        self._runtime_availability_timer.setInterval(1000)
        self._runtime_availability_timer.timeout.connect(
            self._refresh_runtime_feature_state
        )
        self._refresh_runtime_feature_state()
        self._runtime_availability_timer.start()
        self._init_ai_toggle_states()
        self._install_keyboard_shortcuts()

    def _startup_log(self, message: str):
        if hasattr(self, 'log_display'):
            self.log_display.append(message)
        else:
            self._startup_messages.append(message)

    def _flush_startup_messages(self):
        if not hasattr(self, 'log_display'):
            return
        for msg in self._startup_messages:
            self.log_display.append(msg)
        self._startup_messages.clear()

    def _safe_create(self, cls, fallback, feature_key: str, label: str, *args, **kwargs):
        if cls is None:
            self.features[feature_key] = False
            self._startup_log(f"⚠️ {label} unavailable: dependency import failed.")
            return fallback
        try:
            obj = cls(*args, **kwargs)
            self.features[feature_key] = True
            return obj
        except Exception as exc:
            self.features[feature_key] = False
            self._startup_log(f"⚠️ {label} unavailable: {exc}")
            return fallback

    def _bootstrap_core_services(self):
        # Keep sensor startup lazy: the user selects ports/models later, so a failed
        # sensor API constructor should not block the Sensor tab from opening.
        self.sensor_api = NullSensorApi()
        self.features['sensor_driver'] = bool(ArduinoCommander is not None)

        self.robot_api = self._safe_create(RobotController, NullRobotApi(), 'robot_driver', 'Robot API')
        self.gripper = self._safe_create(GripperHelper, NullGripper(), 'gripper_driver', 'Gripper API')

        availability = {}
        refresh_availability = getattr(self.robot_api, 'refresh_availability', None)
        if callable(refresh_availability):
            try:
                availability = refresh_availability()
            except Exception:
                availability = {}
        self.features['robot_ready'] = bool(
            availability.get('robot', False)
            or getattr(self.robot_api, 'service_ok', False)
            or getattr(self.robot_api, 'script_ok', False)
        )
        hand_services_ready = False
        hand_tactile_ready = False
        try:
            hand_services_ready = bool(self.robot_api.hand_services_available())
        except Exception:
            hand_services_ready = False
        try:
            hand_tactile_ready = bool(self.robot_api.hand_tactile_available())
        except Exception:
            hand_tactile_ready = False
        self.features['hand_commands_ready'] = hand_services_ready
        self.features['hand_tactile_ready'] = hand_tactile_ready
        self.features['hand_ready'] = bool(hand_services_ready or hand_tactile_ready)
        self.features['sensor_ready'] = bool(self.features['sensor_module'])
        self.features['gripper_ready'] = bool(
            self.features['gripper_driver']
            and getattr(self.gripper, 'is_available', False)
        )
        self.features['camera_ready'] = bool(self.features['camera_driver'])

    def _refresh_runtime_feature_state(self):
        if self._is_shutting_down:
            return
        api = getattr(self, 'robot_api', None)
        availability = {}
        refresh = getattr(api, 'refresh_availability', None)
        if callable(refresh):
            try:
                availability = refresh()
            except Exception:
                availability = {}

        robot_ready = bool(availability.get('robot', False))
        hand_commands_ready = bool(availability.get('hand_commands', False))
        hand_tactile_ready = bool(availability.get('hand_tactile', False))
        hand_ready = bool(hand_commands_ready or hand_tactile_ready)
        gripper = getattr(self, 'gripper', None)
        gripper_ready = bool(
            self.features.get('gripper_driver', False)
            and getattr(gripper, 'is_available', False)
        )
        self.features['robot_ready'] = robot_ready
        self.features['hand_commands_ready'] = hand_commands_ready
        self.features['hand_tactile_ready'] = hand_tactile_ready
        self.features['hand_ready'] = hand_ready
        self.features['gripper_ready'] = gripper_ready

        if hasattr(self, 'robots_sub_tabs'):
            self.set_robot_subtab_enabled(robot_ready)
        if hasattr(self, 'auto_center_button'):
            self.auto_center_button.setEnabled(
                robot_ready and self.features.get('camera_ready', False)
            )
        if hasattr(self, 'hand_tab_index') and hasattr(self, 'tab_widget'):
            self.tab_widget.setTabEnabled(int(self.hand_tab_index), True)
        self._set_hand_controls_available(
            hand_commands_ready,
            hand_tactile_ready,
        )
        if all(
            hasattr(self, name)
            for name in ('gripper_slider', 'btn_grip_open', 'btn_grip_close')
        ):
            self._set_widgets_enabled(
                [self.gripper_slider, self.btn_grip_open, self.btn_grip_close],
                gripper_ready,
            )
        self._refresh_ai_proximity_model_preview()

    def _get_default_ai_execution_model_path(self) -> str:
        helper = getattr(self, "sensor_functions", None)
        if helper is None:
            return DisabledSensorFunctions.DEFAULT_AI_DIRECT_EXECUTION_MODEL_PATH

        try:
            path = helper.get_ai_direct_finger_motion_execution_default_model_path()
        except Exception:
            path = ""

        if isinstance(path, str) and path.strip():
            return path.strip()
        return DisabledSensorFunctions.DEFAULT_AI_DIRECT_EXECUTION_MODEL_PATH

    def ensure_sensor_api(
        self,
        connect_immediately: bool = True,
        serial_port: str | None = None,
    ) -> bool:
        """Create the sensor API only when it is first needed."""
        if self.sensor_api is not None and not isinstance(
            self.sensor_api, NullSensorApi
        ):
            if serial_port:
                set_serial_port = getattr(self.sensor_api, "set_serial_port", None)
                if callable(set_serial_port):
                    return bool(
                        set_serial_port(
                            serial_port,
                            reconnect=bool(connect_immediately),
                        )
                    )
                self.sensor_api.serial_port = str(serial_port)
            if not bool(connect_immediately):
                return True
            is_connected = getattr(self.sensor_api, "is_connected", None)
            if callable(is_connected) and is_connected():
                return True
            reconnect = getattr(self.sensor_api, "reconnect", None)
            return bool(reconnect()) if callable(reconnect) else True

        if ArduinoCommander is None:
            self.features['sensor_driver'] = False
            self._startup_log('⚠️ Sensor API unavailable: dependency import failed.')
            return False

        try:
            self.sensor_api = ArduinoCommander(
                serial_port=serial_port or "/dev/ttyACM0",
                connect_immediately=bool(connect_immediately)
            )
            self.features['sensor_driver'] = True
            if not bool(connect_immediately):
                return True
            is_connected = getattr(self.sensor_api, "is_connected", None)
            return bool(is_connected()) if callable(is_connected) else True
        except Exception as exc:
            self.sensor_api = NullSensorApi()
            self.features['sensor_driver'] = False
            self._startup_log(f"⚠️ Sensor API not ready yet: {exc}")
            return False

    def _bootstrap_ui_services(self):
        self.mesh_functions = self._safe_create(MyMeshLab, NullMeshLab(self), 'mesh_driver', 'Mesh functions', self)

        if MySensor is None:
            self.features['sensor_module'] = False
            self.sensor_functions = DisabledSensorFunctions(self)
            self._startup_log('⚠️ Sensor functions unavailable: dependency import failed.')
        else:
            try:
                self.sensor_functions = MySensor(self)
                self.features['sensor_module'] = True
            except Exception as exc:
                self.features['sensor_module'] = False
                self.sensor_functions = DisabledSensorFunctions(self)
                self._startup_log(f"⚠️ Sensor functions disabled: {exc}")

        if hasattr(self, "ai_direct_execution_model_path_input"):
            default_ai_model_path = self._get_default_ai_execution_model_path()
            if default_ai_model_path and not self.ai_direct_execution_model_path_input.text().strip():
                self.ai_direct_execution_model_path_input.setText(default_ai_model_path)

        if hasattr(self, "sensor_average_window_spin"):
            self.sensor_average_window_spin.setValue(
                int(self.sensor_functions.get_sensor_average_window_size())
            )
        if hasattr(self, "visualization_target_hz_spin"):
            self.visualization_target_hz_spin.setValue(
                float(self.sensor_functions.get_visualization_target_hz())
            )

        # The Sensor/AI tabs only need the sensor UI module to load. The serial/API
        # connection itself is established lazily when the user actually starts using it.
        self.features['sensor_ready'] = bool(self.features['sensor_module'])

    def _set_widgets_enabled(self, widgets, enabled: bool):
        for widget in widgets:
            widget.setEnabled(enabled)

    def _set_hand_controls_available(
        self,
        command_ready: bool,
        tactile_ready: bool,
    ):
        """Keep the Hand workspace visible while gating hardware operations."""
        command_names = (
            'hand_apply_speed_button',
            'hand_apply_force_button',
            'hand_open_all_button',
            'hand_close_all_button',
            'hand_read_angles_button',
            'hand_thumb_left_button',
            'hand_thumb_center_button',
            'hand_thumb_right_button',
            'hand_send_custom_angles_button',
            'hand_sliders_sync_button',
            'hand_sliders_live_check',
            'hand_pose_load_send_button',
        )
        command_widgets = [
            getattr(self, name)
            for name in command_names
            if hasattr(self, name)
        ]
        command_widgets.extend(
            getattr(self, '_hand_angle_send_buttons', []) or []
        )
        self._set_widgets_enabled(command_widgets, bool(command_ready))

        tactile_widgets = [
            getattr(self, name)
            for name in (
                'hand_tactile_live_button',
                'hand_tactile_refresh_button',
            )
            if hasattr(self, name)
        ]
        self._set_widgets_enabled(tactile_widgets, bool(tactile_ready))

        live_check = getattr(self, 'hand_sliders_live_check', None)
        if not command_ready and live_check is not None and live_check.isChecked():
            live_check.blockSignals(True)
            live_check.setChecked(False)
            live_check.blockSignals(False)
            timer = getattr(self, '_hand_live_timer', None)
            if timer is not None:
                timer.stop()
            self._hand_live_pending = {}

        connection_label = getattr(self, 'hand_connection_label', None)
        if connection_label is not None:
            if command_ready and tactile_ready:
                text = 'Connected · motion and tactile'
            elif command_ready:
                text = 'Connected · motion only'
            elif tactile_ready:
                text = 'Connected · tactile only'
            else:
                text = 'Offline · start the RH56F1 ROS nodes'
            connection_label.setText(text)

    def _apply_startup_feature_state(self):
        if not self.features.get('sensor_module', False):
            for index in (self.sensor_tab_index, self.ai_tab_index):
                self.tab_widget.setTabEnabled(int(index), False)

        if not self.features.get('robot_ready', False):
            self.set_robot_subtab_enabled(False)
            self.auto_center_button.setEnabled(False)

        if hasattr(self, "hand_tab_index"):
            self.tab_widget.setTabEnabled(int(self.hand_tab_index), True)
        self._set_hand_controls_available(
            self.features.get('hand_commands_ready', False),
            self.features.get('hand_tactile_ready', False),
        )

        if not self.features.get('camera_ready', False):
            self.live_yolo_button.setEnabled(False)
            self.auto_center_button.setEnabled(False)

        if not self.features.get('gripper_ready', False):
            self._set_widgets_enabled(
                [self.gripper_slider, self.btn_grip_open, self.btn_grip_close],
                False,
            )

    def createHandle(self):
        return _MainControlPanelSplitterHandle(
            self.orientation(),
            self,
        )

    def _sync_main_control_panel_handle(self):
        handle = self.handle(1)
        if hasattr(handle, "set_panel_visible"):
            handle.set_panel_visible(
                not self._main_control_panel_collapsed
            )

    def _set_main_control_panel_visible(self, visible):
        visible = bool(visible)
        sizes = self.sizes()
        total_width = max(
            int(sum(sizes)),
            int(self.width()),
            1,
        )
        if len(sizes) > 1 and int(sizes[1]) > 0:
            self._main_control_panel_last_width = int(sizes[1])
        self._main_control_panel_collapsed = not visible
        self._main_control_panel_state_syncing = True
        self.splitter_2.setVisible(True)
        if visible:
            restore_width = max(
                360,
                int(self._main_control_panel_last_width),
            )
            self.setSizes(
                [max(total_width - restore_width, 1), restore_width]
            )
        else:
            self.setSizes([total_width, 0])
        self._main_control_panel_state_syncing = False
        self._sync_main_control_panel_handle()

    def toggle_main_control_panel(self):
        self._set_main_control_panel_visible(
            self._main_control_panel_collapsed
        )

    def _on_main_control_splitter_moved(self, *_args):
        if self._main_control_panel_state_syncing:
            return
        sizes = self.sizes()
        if len(sizes) < 2:
            return
        panel_width = int(sizes[1])
        self._main_control_panel_collapsed = panel_width <= 0
        if panel_width > 0:
            self._main_control_panel_last_width = panel_width
            # The user has chosen a width: stop auto-fitting on window resize.
            self._main_control_panel_user_sized = True
        self._sync_main_control_panel_handle()

    def setup_layout(self):
        self.widget_plotter = PlotterWidget()
        layout_plotter = QGridLayout(self.widget_plotter)
        layout_plotter.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self.widget_plotter)
        self.plotter.background_color = theme.VIEWPORT_BG
        layout_plotter.addWidget(self.plotter.interactor)
        self.widget_plotter.setVisible(False)

        self.widget_plotter_2 = PlotterWidget()
        layout_plotter_2 = QGridLayout(self.widget_plotter_2)
        layout_plotter_2.setContentsMargins(0, 0, 0, 0)
        self.plotter_2 = QtInteractor(self.widget_plotter_2)
        self.plotter_2.background_color = theme.VIEWPORT_BG
        layout_plotter_2.addWidget(self.plotter_2.interactor)

        self.log_display = QTextEdit()
        self.log_display.setObjectName("logDisplay")  # Set object name for QSS
        self.log_display.setReadOnly(True)
        self.log_display.textChanged.connect(self.note_log_activity)
        self.log_display.textChanged.connect(self.publish_latest_log_message)
        self.log_console_window = LogConsoleWindow(
            self.log_display,
            parent=self,
        )

        self.splitter_1 = QSplitter(Qt.Horizontal, self)
        self.splitter_1.addWidget(self.widget_plotter)
        self.splitter_1.addWidget(self.widget_plotter_2)
        self.splitter_1.setHandleWidth(3)

        self.splitter_2 = QSplitter(Qt.Vertical, self)
        # --- MODIFICATION: Removed inline stylesheet ---
        self.splitter_2.setHandleWidth(3)

        self.position_entry_widget = RobotPositionWidget(robot_api=self.robot_api)
        self.position_quaternion_widget = RobotToolPositionWidget(robot_api=self.robot_api)
        self.position_toolframe_widget = RobotToolFramePositionWidget(self.robot_api, log_display=self.log_display)
        self.position_script_widget = RobotScriptSendWidget(robot_api=self.robot_api)

        self.position_entry_widget.setVisible(False)
        self.position_quaternion_widget.setVisible(False)
        self.position_toolframe_widget.setVisible(False)
        self.position_script_widget.setVisible(False)

        # The four robot send editors used to live here, above the tab widget,
        # which meant opening one left it hanging over every other workspace
        # and shifted the whole panel down. setup_tab2 now mounts them inside
        # the Robots page, under the buttons that open them.
        self.widget_func = QWidget()
        self.layout_func = QVBoxLayout(self.widget_func)

        # Always-visible operational context. This summarizes cached state
        # only; MyMainWindow refreshes it alongside the status bar.
        self.session_strip = components.SessionStrip(self.widget_func)
        self.layout_func.addWidget(self.session_strip)

        self.tab_widget = QTabWidget()
        self.tab_widget.setUsesScrollButtons(False)
        # The navigation rail must be installed before any tab is added:
        # QTabWidget stores its tabs *in* the tab bar, so replacing the bar
        # afterwards would silently discard every page.
        nav_tabs.install(self.tab_widget)
        self.setup_tabs()

        self.layout_func.addWidget(self.tab_widget)
        self.splitter_2.addWidget(self.widget_func)
        self.addWidget(self.splitter_1)
        self.addWidget(self.splitter_2)
        self.setCollapsible(1, True)
        self.splitterMoved.connect(
            self._on_main_control_splitter_moved
        )
        self._sync_main_control_panel_handle()

    def resizeEvent(self, event):
        """Keep the control panel at its content width as the window resizes.

        ``reLayout`` runs before this splitter has been laid out, so it can
        only set a ratio; a resize is the first point where the panel's own
        size hint (driven by the widest control row in the densest page) can be
        honoured. Re-fitting on every resize means growing the window widens
        the 3D view rather than stretching the panel. Dragging the handle opts
        out permanently.
        """
        super().resizeEvent(event)
        if self.width() <= 1 or self._main_control_panel_user_sized:
            return
        self._fit_main_control_panel()

    def _fit_main_control_panel(self):
        """Size the panel to its content, keeping a floor for the 3D view.

        The panel asks for the width its widest control row needs, so nothing
        is clipped; on a wide display that number stops growing and the
        viewport absorbs the rest.
        """
        if self._main_control_panel_collapsed:
            return
        total = self.width()
        wanted = max(
            self.splitter_2.sizeHint().width() + self.handleWidth(),
            theme.CONTROL_PANEL_MIN_WIDTH,
        )
        panel = min(wanted, max(total - theme.VIEWPORT_MIN_WIDTH, 1))
        panel = max(min(panel, total - 1), 1)
        self._main_control_panel_last_width = panel
        self._main_control_panel_state_syncing = True
        try:
            self.setSizes([max(total - panel, 1), panel])
        finally:
            self._main_control_panel_state_syncing = False

    # Nav bar glyph per workspace, keyed by tab label. Keep in step with the
    # labels passed to addTab in setup_tabs -- a stale key silently drops the
    # icon, since nav_tabs.set_glyphs matches on text.
    NAV_GLYPHS = {
        "Sensor": "sensor",
        "Robots": "robot",
        "Control": "sliders",
        "AI": "ai",
        "Hand": "hand",
        "Tools": "tools",
    }

    @staticmethod
    def _scrollable(page: QWidget) -> QScrollArea:
        """Wrap a dense control page so it scrolls instead of crushing its rows.

        Several pages contain widgets with a large minimum height (the force
        meter chart, the tactile tables). Without this, a short window makes Qt
        squeeze the column past its minimum and rows visibly overlap.
        """
        area = _WidthFittingScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.NoFrame)
        area.setWidget(page)
        return area

    def setup_tabs(self):
        # Every index is captured into a named attribute as its tab is added,
        # so the code that enables/disables a workspace and the code that
        # drives the humanoid viewport lifecycle never restate a literal
        # position. Reordering this method is then safe on its own.
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        self.setup_tab1(tab1_layout)
        self.sensor_tab_index = self.tab_widget.addTab(tab1, "Sensor")

        # Tab 2: ROBOTS (contains two subtabs)
        robots_tab = QWidget()
        robots_layout = QVBoxLayout(robots_tab)
        self.robots_sub_tabs = QTabWidget()
        self.robots_sub_tabs.setUsesScrollButtons(False)

        # ─── Subtab “TM Robot” ───
        robot_page = QWidget()
        robot_layout = QVBoxLayout(robot_page)
        self.setup_tab2(robot_layout)
        self.tm_robot_tab_index = self.robots_sub_tabs.addTab(
            self._scrollable(robot_page), "TM Robot"
        )

        # The humanoid controls are constructed only when this subtab is opened.
        # The G1 scene is loaded explicitly into the existing sensor viewport.
        self.humanoid_page = QWidget()
        self.humanoid_layout = QVBoxLayout(self.humanoid_page)
        self.humanoid_layout.setContentsMargins(0, 0, 0, 0)
        self.humanoid_viewer = None
        self._humanoid_viewer_loading = False
        self._humanoid_viewport_active = False
        self._humanoid_previous_sensor_visibility = None
        self.humanoid_placeholder = components.EmptyState(
            "Humanoid viewer is not loaded",
            glyph="robot",
            hint="Opening this workspace prepares the URDF and sensor-signal controls.",
        )
        self.humanoid_layout.addWidget(self.humanoid_placeholder)
        self.humanoid_tab_index = self.robots_sub_tabs.addTab(
            self.humanoid_page,
            "Humanoid",
        )
        self.robots_sub_tabs.currentChanged.connect(
            self._on_robot_subtab_changed
        )

        robots_layout.addWidget(self.robots_sub_tabs)
        self.robots_tab_index = self.tab_widget.addTab(robots_tab, "Robots")

        # Tabs 3 and 4: Control, then AI. setup_tab3 builds both pages -- the
        # deterministic controllers and the learned-policy controls share
        # dialogs and status labels -- and leaves the Control page on
        # self.control_page for us to mount as its own workspace. Control
        # comes first because it is what you reach for before a policy runs.
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        self.setup_tab3(tab3_layout)
        self.control_tab_index = self.tab_widget.addTab(
            self._scrollable(self.control_page), "Control"
        )
        self.ai_tab_index = self.tab_widget.addTab(tab3, "AI")

        # Tab 4: Dexterous Hand
        tab4 = QWidget()
        tab4_layout = QVBoxLayout(tab4)
        self.setup_tab5(tab4_layout)
        self.hand_tab_index = self.tab_widget.addTab(tab4, "Hand")

        # Tab 5: Tools (last)
        tab5 = QWidget()
        tab5_layout = QVBoxLayout(tab5)
        self.setup_tab4(tab5_layout)
        self.tools_tab_index = self.tab_widget.addTab(
            self._scrollable(tab5), "Tools"
        )
        self.tab_widget.currentChanged.connect(self._on_main_tab_changed)

        # Icons are keyed by tab label, so this has to run after every tab
        # exists -- and NAV_GLYPHS has to be updated alongside any rename.
        nav_tabs.set_glyphs(self.tab_widget, self.NAV_GLYPHS)

        # Related tools inside an area read as one segmented control rather
        # than a second row of page tabs competing with the rail.
        for sub_tabs in (
            getattr(self, "sensor_sub_tabs", None),
            self.robots_sub_tabs,
            getattr(self, "ai_sub_tabs", None),
            getattr(self, "hand_sub_tabs", None),
        ):
            if sub_tabs is not None:
                components.segmented(sub_tabs)

    def _on_robot_subtab_changed(self, index):
        if int(index) != int(getattr(self, "humanoid_tab_index", -1)):
            self._release_humanoid_viewport()
            return
        QTimer.singleShot(0, self._ensure_humanoid_viewer)

    def _on_main_tab_changed(self, index):
        if int(index) != int(getattr(self, "robots_tab_index", -1)):
            self._release_humanoid_viewport()
            return
        if int(self.robots_sub_tabs.currentIndex()) == int(
            getattr(self, "humanoid_tab_index", -1)
        ):
            QTimer.singleShot(0, self._ensure_humanoid_viewer)

    def _ensure_humanoid_viewer(self):
        if self._is_shutting_down:
            return
        if self._humanoid_viewer_loading:
            return
        if self.humanoid_viewer is not None:
            self.humanoid_viewer.restore_scene(reset_camera=False)
            return
        self._humanoid_viewer_loading = True
        self.humanoid_placeholder.set_message("Loading humanoid viewer...")
        try:
            # Keep this import lazy so normal phd_ui startup does not initialize
            # another PyVista render window or inspect the large G1 asset tree.
            from phd.ui.humanoid_viewer import HumanoidViewerWidget

            viewer = HumanoidViewerWidget(
                self.humanoid_page,
                plotter=self.plotter_2,
                sensor_functions=getattr(
                    self,
                    "sensor_functions",
                    None,
                ),
                before_scene_load=self._activate_humanoid_viewport,
            )
            self.humanoid_layout.replaceWidget(
                self.humanoid_placeholder,
                viewer,
            )
            self.humanoid_placeholder.hide()
            self.humanoid_viewer = viewer
        except Exception as exc:
            self.humanoid_placeholder.set_message(
                "Humanoid viewer could not be loaded.\n\n"
                f"{exc}\n\n"
                "Switch away and return to this tab to retry."
            )
            self._startup_log(f"Humanoid viewer unavailable: {exc}")
        finally:
            self._humanoid_viewer_loading = False

    def _activate_humanoid_viewport(self):
        if self._humanoid_viewport_active:
            return
        sensor = getattr(self, "sensor_functions", None)
        if sensor is not None:
            self._humanoid_previous_sensor_visibility = bool(
                getattr(sensor, "main_visualization_enabled", True)
            )
            sensor.set_main_visualization_enabled(False, render=False)
        self._humanoid_viewport_active = True
        self.widget_plotter_2.setVisible(True)

    def _release_humanoid_viewport(self):
        if not bool(getattr(self, "_humanoid_viewport_active", False)):
            return
        viewer = getattr(self, "humanoid_viewer", None)
        if viewer is not None:
            viewer.release_scene(render=False)
        sensor = getattr(self, "sensor_functions", None)
        previous = getattr(
            self,
            "_humanoid_previous_sensor_visibility",
            None,
        )
        if sensor is not None and previous is not None:
            sensor.set_main_visualization_enabled(bool(previous), render=False)
        self._humanoid_previous_sensor_visibility = None
        self._humanoid_viewport_active = False
        try:
            self.plotter_2.render()
        except Exception:
            pass

    def _shutdown_humanoid_viewer(self):
        self._release_humanoid_viewport()
        viewer = getattr(self, "humanoid_viewer", None)
        self.humanoid_viewer = None
        if viewer is not None:
            viewer.shutdown()

    def setup_tab1(self, layout):
        self.sensor_sub_tabs = QTabWidget()
        self.sensor_sub_tabs.setUsesScrollButtons(False)

        # ─── Subtab "Setup" ───
        # Two grouped inset lists (the macOS System Settings pattern): each
        # section is one rounded surface with hairline rules between its rows,
        # and its title sits outside and above. The page is a linear task --
        # choose a sensor, connect it, then adjust how it draws -- so a border
        # around every step added weight without adding meaning.
        send_page = QWidget()
        send_page_layout = QVBoxLayout(send_page)
        send_page_layout.setSpacing(theme.SPACE_LG)

        send_group, send_list = components.section(
            "Sensor",
            "Pick a sensor model and the port it is connected to.",
        )
        viz_group, viz_list = components.section(
            "Visualization",
            "How the live signal is drawn in the 3D view.",
        )

        # Sensor selection and connection widgets
        self.sensor_choice = QListWidget(self.widget_func)
        self.sensor_choice.setSelectionMode(QListWidget.SingleSelection)
        self.sensor_choice.addItems([
            "2D", "Elbow", "Kuka", "Double Curve", "Half Cylinder Surface"
        ])
        self.sensor_choice.setCurrentRow(0)
        self.sensor_choice.setToolTip(
            "Geometry of the sensor being read, which sets how the 3D grid is "
            "laid out."
        )
        components.fit_list_height(self.sensor_choice, max_rows=6)
        send_list.add(self.sensor_choice)

        self.sensor_source_mode_combo = QComboBox(self.widget_func)
        self.sensor_source_mode_combo.addItem("Single Sensor", "single")
        self.sensor_source_mode_combo.addItem("Multiple Ports", "multiple")
        self.sensor_source_mode_combo.setToolTip(
            "Single Sensor uses one selected source. Multiple Ports displays "
            "all selected serial sensors together. Grid rows, columns, and "
            "+1-column format are configured independently for each port; "
            "the first selected port is used by AI and robot-control features."
        )
        send_list.add_row("Show", self.sensor_source_mode_combo)

        self._sensor_port_profiles = {}
        self._sensor_port_profile_updating = False
        self.serial_channel = QListWidget(self.widget_func)
        self.serial_channel.setSelectionMode(QListWidget.SingleSelection)
        # An empty port list is the state this rig sits in most often (nothing
        # plugged in yet), and a blank box there reads as a broken panel.
        self.sensor_port_stack = components.EmptyStateStack(
            self.serial_channel,
            components.EmptyState(
                "No sensor ports found",
                glyph="plug",
                hint="Connect the sensor over USB, then press Connect Sensor "
                     "to rescan.",
            ),
        )
        self.serial_channel.model().rowsInserted.connect(
            self._refresh_sensor_port_empty_state
        )
        self.serial_channel.model().rowsRemoved.connect(
            self._refresh_sensor_port_empty_state
        )
        self._refresh_sensor_port_empty_state()
        send_list.add(self.sensor_port_stack)

        self.sensor_source_mode_combo.currentIndexChanged.connect(
            self._on_sensor_source_mode_changed
        )

        self.goodix_disable_desktop_touch_checkbox = QCheckBox(
            "Disable Goodix desktop touch"
        )
        self.goodix_disable_desktop_touch_checkbox.setChecked(True)
        self.goodix_disable_desktop_touch_checkbox.setEnabled(False)
        self.goodix_disable_desktop_touch_checkbox.setToolTip(
            "Prevent this USB sensor from moving or clicking the X11 desktop. "
            "Raw tactile-matrix reading remains enabled."
        )
        self.goodix_disable_desktop_touch_checkbox.toggled.connect(
            self._on_goodix_desktop_touch_toggled
        )
        send_list.add(self.goodix_disable_desktop_touch_checkbox)

        # The one primary action on this page: it opens the port, builds the
        # 3D grid and starts streaming.
        self.buildScene = QPushButton("Connect Sensor", self.widget_func)
        self.buildScene.setToolTip(
            "Open the selected port, build the 3D sensor grid and start "
            "streaming live data."
        )
        components.apply_variant(self.buildScene, "primary")
        send_list.add(self.buildScene)

        # 2D grid controls. Trailing-aligned so they line up with the other
        # controls in the list rather than hugging their label.
        grid_container = QWidget()
        grid_layout = QHBoxLayout(grid_container)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(theme.SPACE_XS)
        grid_layout.addStretch(1)

        # Both spin boxes hold the same kind of value, so give them the same
        # width -- otherwise the layout stretches them to different sizes.
        self.grid_rows_spin = QSpinBox()
        self.grid_rows_spin.setRange(2, 100)
        self.grid_rows_spin.setValue(10)
        self.grid_rows_spin.setFixedWidth(64)
        grid_layout.addWidget(self.grid_rows_spin)

        grid_layout.addWidget(QLabel("×"))

        self.grid_cols_spin = QSpinBox()
        self.grid_cols_spin.setRange(2, 100)
        self.grid_cols_spin.setValue(10)
        self.grid_cols_spin.setFixedWidth(64)
        grid_layout.addWidget(self.grid_cols_spin)

        self.serial_channel.currentItemChanged.connect(
            self._on_sensor_port_selection_changed
        )
        self.grid_rows_spin.valueChanged.connect(
            self._on_sensor_port_profile_controls_changed
        )
        self.grid_cols_spin.valueChanged.connect(
            self._on_sensor_port_profile_controls_changed
        )

        self.sensor_extra_column_checkbox = QCheckBox(
            "Raw packet includes +1 column"
        )
        self.sensor_extra_column_checkbox.setChecked(True)
        self.sensor_extra_column_checkbox.setToolTip(
            "Checked: expect rows x (columns + 1) values and ignore the extra "
            "column. Unchecked: expect exactly rows x columns values."
        )
        self.sensor_extra_column_checkbox.toggled.connect(
            self._on_sensor_port_profile_controls_changed
        )

        # Packet shape is a property of the hardware: set once per port, then
        # never touched again. Behind a disclosure so the controls you do use
        # every session stay visible without scrolling.
        self.sensor_packet_format_group = components.CollapsibleGroup(
            "Packet format"
        )
        packet_list = components.InsetList()
        packet_list.add(
            components.SettingsRow(
                "Rows × columns",
                grid_container,
                hint="In Multiple Ports mode this applies only to the "
                     "highlighted port.",
            )
        )
        packet_list.add(self.sensor_extra_column_checkbox)
        self.sensor_packet_format_group.add(packet_list)
        send_group.layout().addWidget(self.sensor_packet_format_group)

        # Style before sensitivity: the slider only makes sense once you know
        # which rendering it is scaling.
        self.sensor_visualization_mode_combo = QComboBox()
        self.sensor_visualization_mode_combo.addItem("Point Grid", "point_grid")
        self.sensor_visualization_mode_combo.addItem("Stereo Field", "stereo_field")
        self.sensor_visualization_mode_combo.addItem("3D Heatmap", "heatmap_3d")
        self.sensor_visualization_mode_combo.setToolTip(
            "Choose the live sensor rendering style."
        )
        viz_list.add_row("Style", self.sensor_visualization_mode_combo)

        # Sensitivity slider
        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(theme.SPACE_SM)

        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(0, 1000)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self.sensitivity_slider.setTickInterval(10)
        self.sensitivity_slider.setMinimumWidth(140)
        self.sensitivity_slider.setToolTip(
            "Maximum point-grid displacement when the signal reaches the "
            "configured full-colour level."
        )

        # Fixed-width figures: this value tracks the slider, and a proportional
        # font would shuffle the slider sideways on every digit change.
        self.sensitivity_value_label = QLabel("0.050")
        self.sensitivity_value_label.setStyleSheet(theme.readout_style())
        self.sensitivity_value_label.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        self.sensitivity_value_label.setFixedWidth(48)

        slider_layout.addWidget(self.sensitivity_slider, 1)
        slider_layout.addWidget(self.sensitivity_value_label)

        viz_list.add_row("Sensitivity", slider_container)

        self.sensor_transparent_screenshot_button = QPushButton(
            "Capture Transparent PNG"
        )
        self.sensor_transparent_screenshot_button.setToolTip(
            "Save the current sensor plotter view as a PNG with a transparent background."
        )
        viz_list.add(self.sensor_transparent_screenshot_button)

        send_page_layout.addWidget(send_group)
        send_page_layout.addWidget(viz_group)
        send_page_layout.addStretch()

        # ─── Subtab "Diagnostics" ───
        read_page = QWidget()
        read_page_layout = QVBoxLayout(read_page)

        read_group, read_list = components.section(
            "Readouts",
            "One-shot reads printed to the log, for checking wiring and "
            "throughput.",
        )

        self.read_sensor_api_button = QPushButton("Sensor API Raw Data")
        self.read_sensor_api_button.setToolTip(
            "Read from the port selected in Sensor > Setup."
        )
        self.read_sensor_api_hz_button = QPushButton("Sensor API Raw Hz")
        self.read_sensor_api_hz_button.setToolTip(
            "Measure the port selected in Sensor > Setup."
        )
        self.read_sensor_channel_button = QPushButton("Sensor API Channel")
        self.read_sensor_channel_button.setToolTip(
            "Check the port selected in Sensor > Setup."
        )
        self.read_sensor_raw_button = QPushButton("Sensor Raw Data")
        self.read_sensor_raw_ave_button = QPushButton("Sensor Raw Ave Data")
        self.read_sensor_diff_button = QPushButton("Sensor Diff Data")
        self.read_sensor_diff_debug_button = QPushButton("Sensor Diff Debug Views")
        self.read_runtime_hz_button = QPushButton("Sensor / DFM Runtime Hz")

        # Named rows rather than eight stacked buttons whose captions all begin
        # "Sensor ...": the row says what you get, the button says the verb.
        for label, hint, button in (
            (
                "Raw packet",
                "One frame straight off the port, unparsed.",
                self.read_sensor_api_button,
            ),
            (
                "Packet rate",
                "Frames per second measured over one second.",
                self.read_sensor_api_hz_button,
            ),
            (
                "Channel map",
                "Drive and sensor counts the device reports.",
                self.read_sensor_channel_button,
            ),
            ("Raw values", "", self.read_sensor_raw_button),
            ("Averaged values", "", self.read_sensor_raw_ave_button),
            ("Difference from calibration", "", self.read_sensor_diff_button),
            ("Difference debug views", "", self.read_sensor_diff_debug_button),
            (
                "Runtime rates",
                "Acquisition, visualization and DFM loop rates.",
                self.read_runtime_hz_button,
            ),
        ):
            button.setText("Read")
            button.setMinimumWidth(72)
            read_list.add(components.SettingsRow(label, button, hint=hint))

        read_page_layout.addWidget(read_group)
        read_page_layout.addStretch()

        self.sensor_setup_tab_index = self.sensor_sub_tabs.addTab(
            self._scrollable(send_page), "Setup"
        )
        self.sensor_diagnostics_tab_index = self.sensor_sub_tabs.addTab(
            self._scrollable(read_page), "Diagnostics"
        )
        layout.addWidget(self.sensor_sub_tabs)
        self._build_sensor_parameters_dialog()

    def _refresh_sensor_port_empty_state(self, *_args):
        """Swap the port list for its empty state when no ports are listed."""
        stack = getattr(self, "sensor_port_stack", None)
        if stack is None:
            return
        is_empty = self.serial_channel.count() == 0
        stack.set_empty(is_empty)
        # Ports arrive after this page is built, so the list is re-fitted here
        # rather than once at construction time.
        components.fit_list_height(self.serial_channel, max_rows=4)
        # QStackedWidget otherwise keeps an Expanding vertical policy even
        # though its visible list has a fixed height, creating a large blank
        # area that pushes Connect Sensor below the laptop fold.
        stack.setFixedHeight(
            max(112, stack.empty.minimumSizeHint().height())
            if is_empty
            else self.serial_channel.height()
        )

    @staticmethod
    def _default_sensor_grid_shape_for_port(port_name):
        if is_goodix_usb_source(port_name):
            return GOODIX_USB_ROWS, GOODIX_USB_COLUMNS
        humanoid_shape = humanoid_sensor_grid_shape_for_device(port_name)
        if humanoid_shape is not None:
            return humanoid_shape
        name = os.path.basename(str(port_name or "")).lower()
        if name == "ttyacm0":
            return 10, 10
        if name in ("ttyacm1", "ttyamc1"):
            return 8, 10
        return None

    @staticmethod
    def _sensor_port_profile_key(port_name):
        text = str(port_name or "").strip()
        if is_goodix_usb_source(text):
            return GOODIX_USB_SOURCE_ID
        return os.path.basename(text).lower()

    def _default_sensor_port_profile(self, port_name):
        shape = self._default_sensor_grid_shape_for_port(port_name)
        if shape is None:
            shape = (
                int(self.grid_rows_spin.value()),
                int(self.grid_cols_spin.value()),
            )
        port_basename = os.path.basename(str(port_name or "")).lower()
        recognized_extra_column = (
            humanoid_sensor_extra_column_for_device(port_name)
        )
        has_extra_column = (
            bool(recognized_extra_column)
            if recognized_extra_column is not None
            else port_basename == "ttyacm0"
        )
        return {
            "n_row": int(shape[0]),
            "n_col": int(shape[1]),
            "has_extra_column": has_extra_column,
        }

    def _save_sensor_port_profile_from_controls(self, port_name):
        if getattr(self, "_sensor_port_profile_updating", False):
            return
        key = self._sensor_port_profile_key(port_name)
        if not key:
            return
        self._sensor_port_profiles[key] = {
            "n_row": int(self.grid_rows_spin.value()),
            "n_col": int(self.grid_cols_spin.value()),
            "has_extra_column": bool(
                self.sensor_extra_column_checkbox.isChecked()
            ),
        }

    def get_sensor_port_profile(self, port_name):
        key = self._sensor_port_profile_key(port_name)
        current = self.serial_channel.currentItem()
        if (
            current is not None
            and self._sensor_port_profile_key(
                _sensor_source_from_item(current)
            ) == key
        ):
            self._save_sensor_port_profile_from_controls(
                _sensor_source_from_item(current)
            )
        profile = self._sensor_port_profiles.get(key)
        if profile is None:
            profile = self._default_sensor_port_profile(port_name)
            self._sensor_port_profiles[key] = dict(profile)
        return dict(profile)

    def _load_sensor_port_profile_into_controls(self, port_name):
        key = self._sensor_port_profile_key(port_name)
        profile = self._sensor_port_profiles.get(key)
        if profile is None:
            profile = self._default_sensor_port_profile(port_name)
            self._sensor_port_profiles[key] = dict(profile)
        widgets = (
            self.grid_rows_spin,
            self.grid_cols_spin,
            self.sensor_extra_column_checkbox,
        )
        self._sensor_port_profile_updating = True
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.grid_rows_spin.setValue(int(profile["n_row"]))
            self.grid_cols_spin.setValue(int(profile["n_col"]))
            self.sensor_extra_column_checkbox.setChecked(
                bool(profile["has_extra_column"])
            )
        finally:
            for widget in widgets:
                widget.blockSignals(False)
            self._sensor_port_profile_updating = False

    def _on_sensor_port_profile_controls_changed(self, *_args):
        if getattr(self, "_sensor_port_profile_updating", False):
            return
        current = self.serial_channel.currentItem()
        if current is not None:
            self._save_sensor_port_profile_from_controls(
                _sensor_source_from_item(current)
            )

    def _on_sensor_port_selection_changed(self, current, previous=None):
        if previous is not None:
            self._save_sensor_port_profile_from_controls(
                _sensor_source_from_item(previous)
            )
        port_name = _sensor_source_from_item(current)
        is_goodix = is_goodix_usb_source(port_name)
        checkbox = getattr(
            self,
            "goodix_disable_desktop_touch_checkbox",
            None,
        )
        if checkbox is not None:
            checkbox.setEnabled(is_goodix)
        if current is None:
            return
        self._load_sensor_port_profile_into_controls(port_name)
        if is_goodix and checkbox is not None and checkbox.isChecked():
            self._set_goodix_desktop_touch_enabled(False)

    def _on_sensor_source_mode_changed(self, *_args):
        port_list = getattr(self, "serial_channel", None)
        combo = getattr(self, "sensor_source_mode_combo", None)
        if port_list is None or combo is None:
            return

        multiple = combo.currentData() == "multiple"
        port_list.setSelectionMode(
            QListWidget.MultiSelection
            if multiple
            else QListWidget.SingleSelection
        )
        if multiple:
            return

        current = port_list.currentItem()
        if current is None:
            selected = list(port_list.selectedItems() or [])
            current = selected[0] if selected else None
        for index in range(port_list.count()):
            item = port_list.item(index)
            item.setSelected(item is current)

    def _set_goodix_desktop_touch_enabled(self, enabled):
        succeeded, message = set_goodix_desktop_touch_enabled(enabled)
        log_display = getattr(self, "log_display", None)
        if log_display is not None:
            log_display.append(message)
        return succeeded

    def _on_goodix_desktop_touch_toggled(self, disable_touch):
        current = self.serial_channel.currentItem()
        if current is None or not is_goodix_usb_source(
            _sensor_source_from_item(current)
        ):
            return
        self._set_goodix_desktop_touch_enabled(not bool(disable_touch))

    @staticmethod
    def _sensor_reorder_mode_label(mode):
        labels = {
            "factory": "Factory Default",
            "none": "No Reorder",
            "row_to_col": "Row to Column",
            "row_to_col_flipped": "Row to Column, Flip Row and Column",
            "row_to_col_c_flip_only": "Row to Column, Flip Column",
            "row_to_col_r_flip_only": "Row to Column, Flip Row",
            "col_to_row": "Column to Row",
            "col_to_row_flipped": "Column to Row, Flip Row and Column",
            "col_to_row_c_flip_only": "Column to Row, Flip Column",
            "col_to_row_r_flip_only": "Column to Row, Flip Row",
            "vertical_flip": "Vertical Flip",
            "horizontal_flip": "Horizontal Flip",
            "flip_and_rotate": "Flip and Rotate",
            "rotate_180": "Rotate 180",
        }
        return labels.get(str(mode), str(mode))

    @staticmethod
    def _set_combo_current_data(combo, data):
        for index in range(combo.count()):
            if combo.itemData(index) == data:
                combo.setCurrentIndex(index)
                return True
        return False

    def _build_sensor_parameters_dialog(self):
        self.sensor_parameters_dialog = QDialog(self)
        self.sensor_parameters_dialog.setWindowTitle("Sensor Parameters")
        self.sensor_parameters_dialog.setModal(False)
        components.size_to_screen(self.sensor_parameters_dialog, 860, 760)
        self.sensor_parameters_dialog.setMinimumSize(720, 540)

        outer_layout = QVBoxLayout(self.sensor_parameters_dialog)
        self.sensor_parameters_tabs = QTabWidget(self.sensor_parameters_dialog)
        sensor_parameters_general_tab = QWidget(self.sensor_parameters_tabs)
        sensor_parameters_general_layout = QVBoxLayout(
            sensor_parameters_general_tab
        )
        sensor_parameters_general_layout.setContentsMargins(0, 0, 0, 0)
        sensor_parameters_scroll = QScrollArea(sensor_parameters_general_tab)
        sensor_parameters_scroll.setWidgetResizable(True)
        sensor_parameters_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sensor_parameters_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sensor_parameters_content = QWidget(sensor_parameters_scroll)
        layout = QVBoxLayout(sensor_parameters_content)
        layout.setContentsMargins(8, 8, 8, 8)
        sensor_parameters_scroll.setWidget(sensor_parameters_content)
        sensor_parameters_general_layout.addWidget(sensor_parameters_scroll, 1)
        self.sensor_parameters_general_tab = sensor_parameters_general_tab
        self.sensor_parameters_tabs.addTab(
            sensor_parameters_general_tab, "General"
        )
        self.sensor_zero_mask_panel = SensorZeroMaskPanel(
            self.sensor_parameters_tabs,
            sensor_functions=getattr(self, "sensor_functions", None),
        )
        self.sensor_parameters_tabs.addTab(
            self.sensor_zero_mask_panel, "Zero Mask"
        )
        outer_layout.addWidget(self.sensor_parameters_tabs, 1)
        self.sensor_parameters_scroll = sensor_parameters_scroll
        layout.setSpacing(theme.SPACE_LG)

        self.sensor_parameter_model_combo = QComboBox(self.sensor_parameters_dialog)
        self.sensor_parameter_reorder_combo = QComboBox(self.sensor_parameters_dialog)
        self.sensor_parameter_force_scale_spin = QDoubleSpinBox(self.sensor_parameters_dialog)
        self.sensor_parameter_force_scale_spin.setDecimals(6)
        self.sensor_parameter_force_scale_spin.setRange(0.0, 1000.0)
        self.sensor_parameter_force_scale_spin.setSingleStep(0.001)
        self.sensor_parameter_force_scale_spin.setToolTip(
            "Calibration scale used to convert contact signal to Newtons.\n"
            "Force (N) = contact signal × this scale."
        )
        self.sensor_average_window_spin = QSpinBox(self.sensor_parameters_dialog)
        self.sensor_average_window_spin.setRange(1, 30)
        self.sensor_average_window_spin.setValue(
            DisabledSensorFunctions.DEFAULT_SENSOR_AVERAGE_WINDOW_SIZE
        )
        self.visualization_target_hz_spin = QDoubleSpinBox(self.sensor_parameters_dialog)
        self.visualization_target_hz_spin.setDecimals(1)
        self.visualization_target_hz_spin.setRange(1.0, 240.0)
        self.visualization_target_hz_spin.setSingleStep(1.0)
        self.visualization_target_hz_spin.setValue(
            DisabledSensorFunctions.DEFAULT_VISUALIZATION_TARGET_HZ
        )
        self.visualization_target_hz_spin.setSuffix(" Hz")

        self.sensor_parameter_shape_combo = QComboBox(self.sensor_parameters_dialog)
        self.sensor_parameter_shape_combo.addItem("Flat", "flat")
        self.sensor_parameter_shape_combo.addItem("Cylinder Bend", "cylinder")
        self.sensor_parameter_shape_combo.addItem("Custom Interactive", "custom")
        self._sensor_parameter_custom_points = []
        self._sensor_parameter_use_custom_heatmap_shape = False
        self._sensor_parameter_custom_heatmap_points = []
        self._sensor_parameter_custom_heatmap_corners = []
        self._sensor_parameter_use_curved_heatmap_edges = False
        self._sensor_parameter_custom_heatmap_edge_offsets = []

        self.sensor_parameter_bend_axis_combo = QComboBox(self.sensor_parameters_dialog)
        self.sensor_parameter_bend_axis_combo.addItem("Across Columns (X)", "columns")
        self.sensor_parameter_bend_axis_combo.addItem("Across Rows (Y)", "rows")

        self.sensor_parameter_arc_spin = QDoubleSpinBox(self.sensor_parameters_dialog)
        self.sensor_parameter_arc_spin.setDecimals(1)
        self.sensor_parameter_arc_spin.setRange(-180.0, 180.0)
        self.sensor_parameter_arc_spin.setSingleStep(5.0)
        self.sensor_parameter_arc_spin.setSuffix(" deg")

        self.sensor_parameter_rotation_x_spin = QDoubleSpinBox(self.sensor_parameters_dialog)
        self.sensor_parameter_rotation_x_spin.setDecimals(1)
        self.sensor_parameter_rotation_x_spin.setRange(-180.0, 180.0)
        self.sensor_parameter_rotation_x_spin.setSingleStep(5.0)
        self.sensor_parameter_rotation_x_spin.setSuffix(" deg")
        self.sensor_parameter_rotation_x_spin.setToolTip(
            "Rotate the complete 2D sensor around X after shape bending."
        )

        self.sensor_parameter_rotation_y_spin = QDoubleSpinBox(self.sensor_parameters_dialog)
        self.sensor_parameter_rotation_y_spin.setDecimals(1)
        self.sensor_parameter_rotation_y_spin.setRange(-180.0, 180.0)
        self.sensor_parameter_rotation_y_spin.setSingleStep(5.0)
        self.sensor_parameter_rotation_y_spin.setSuffix(" deg")
        self.sensor_parameter_rotation_y_spin.setToolTip(
            "Rotate the complete 2D sensor around Y after shape bending."
        )

        self.sensor_parameter_rotation_z_spin = QDoubleSpinBox(self.sensor_parameters_dialog)
        self.sensor_parameter_rotation_z_spin.setDecimals(1)
        self.sensor_parameter_rotation_z_spin.setRange(-180.0, 180.0)
        self.sensor_parameter_rotation_z_spin.setSingleStep(5.0)
        self.sensor_parameter_rotation_z_spin.setSuffix(" deg")
        self.sensor_parameter_rotation_z_spin.setToolTip(
            "Rotate the complete 2D sensor around Z after shape bending."
        )

        self.sensor_parameter_normal_flip_checkbox = QCheckBox("Flip normals")
        self.sensor_parameter_use_shape_checkbox = QCheckBox("Use selected shape")
        self.sensor_parameter_heatmap_follows_shape_checkbox = QCheckBox(
            "3D heatmap follows sensor shape"
        )
        self.sensor_parameter_heatmap_follows_shape_checkbox.setChecked(True)
        self.sensor_parameter_heatmap_follows_shape_checkbox.setToolTip(
            "Use the selected flat, cylinder, or custom sensor geometry for the "
            "3D heatmap. Uncheck this to use the independently edited heatmap shape."
        )
        self.sensor_parameter_shape_editor_button = QPushButton(
            "Edit custom shape"
        )
        self.sensor_parameter_shape_editor_button.setToolTip(
            "Open the interactive 3D taxel-grid editor for the currently built "
            "2D sensor."
        )
        self.sensor_parameter_stereo_ignore_noise_checkbox = QCheckBox("Ignore stereo field noise")
        self.sensor_parameter_stereo_ignore_noise_checkbox.setChecked(True)
        self.sensor_parameter_stereo_ignore_noise_checkbox.setToolTip(
            "Ignore small baseline sensor changes so the stereo field does not vibrate while idle."
        )
        self.sensor_parameter_stereo_deadband_spin = QDoubleSpinBox(self.sensor_parameters_dialog)
        self.sensor_parameter_stereo_deadband_spin.setDecimals(2)
        self.sensor_parameter_stereo_deadband_spin.setRange(0.0, 20.0)
        self.sensor_parameter_stereo_deadband_spin.setSingleStep(0.05)
        self.sensor_parameter_stereo_deadband_spin.setValue(0.35)
        self.sensor_parameter_stereo_deadband_spin.setToolTip(
            "Minimum signal before the stereo field begins to compress."
        )
        self.sensor_parameter_stereo_length_spin = QDoubleSpinBox(self.sensor_parameters_dialog)
        self.sensor_parameter_stereo_length_spin.setDecimals(2)
        self.sensor_parameter_stereo_length_spin.setRange(0.05, 2.0)
        self.sensor_parameter_stereo_length_spin.setSingleStep(0.05)
        self.sensor_parameter_stereo_length_spin.setValue(0.35)
        self.sensor_parameter_stereo_length_spin.setToolTip(
            "Stereo field line length relative to the sensor size."
        )
        self.sensor_parameter_signal_mode_combo = QComboBox(
            self.sensor_parameters_dialog
        )
        self.sensor_parameter_signal_mode_combo.addItem(
            "Magnitude (Red = |signal|)", True
        )
        self.sensor_parameter_signal_mode_combo.addItem(
            "Signed (Blue = negative, Red = positive)", False
        )
        self.sensor_parameter_signal_mode_combo.setToolTip(
            "Controls Point Grid, Stereo Field, and 3D Heatmap. Magnitude "
            "matches the original abs() display. Signed mode shows negative "
            "values in blue and positive values in red."
        )
        self.sensor_parameter_point_grid_response_combo = QComboBox(
            self.sensor_parameters_dialog
        )
        self.sensor_parameter_point_grid_response_combo.addItem(
            "Zero-Centred (Recommended)", "zero_centered"
        )
        self.sensor_parameter_point_grid_response_combo.addItem(
            "Legacy Offset", "legacy_offset"
        )
        self.sensor_parameter_point_grid_response_combo.setToolTip(
            "Choose how Point Grid height and colour respond to the signal. "
            "Zero-Centred keeps idle points on the sensor surface and bounds "
            "their motion. Legacy Offset reproduces the previous display."
        )
        self.sensor_parameter_heatmap_response_combo = QComboBox(
            self.sensor_parameters_dialog
        )
        self.sensor_parameter_heatmap_response_combo.addItem(
            "Linear Relative (%)", "linear_relative"
        )
        self.sensor_parameter_heatmap_response_combo.addItem(
            "Proximity Enhanced (absolute difference)", "proximity_enhanced"
        )
        self.sensor_parameter_heatmap_palette_combo = QComboBox(
            self.sensor_parameters_dialog
        )
        self.sensor_parameter_heatmap_palette_combo.addItem(
            "Original (White to Red)", "white_red"
        )
        self.sensor_parameter_heatmap_palette_combo.addItem(
            "White, Blue to Deep Red", "white_blue_red"
        )
        self.sensor_parameter_heatmap_palette_combo.addItem(
            "Light Blue to Deep Blue", "light_deep_blue"
        )
        self.sensor_parameter_heatmap_palette_combo.setToolTip(
            "Choose only the 3D heatmap colour progression. Signal response "
            "thresholds and the 2D signal viewer are unchanged."
        )
        self.sensor_parameter_heatmap_saturation_spin = QDoubleSpinBox(
            self.sensor_parameters_dialog
        )
        self.sensor_parameter_heatmap_saturation_spin.setDecimals(2)
        self.sensor_parameter_heatmap_saturation_spin.setRange(0.1, 50.0)
        self.sensor_parameter_heatmap_saturation_spin.setSingleStep(0.1)
        self.sensor_parameter_heatmap_saturation_spin.setValue(5.0)
        self.sensor_parameter_heatmap_saturation_spin.setSuffix(" %")
        self.sensor_parameter_heatmap_saturation_spin.setToolTip(
            "Signal change that produces the strongest selected colour. "
            "Lower values make the heatmap more sensitive."
        )
        self.sensor_parameter_heatmap_floor_spin = QDoubleSpinBox(
            self.sensor_parameters_dialog
        )
        self.sensor_parameter_heatmap_floor_spin.setDecimals(2)
        self.sensor_parameter_heatmap_floor_spin.setRange(0.0, 20.0)
        self.sensor_parameter_heatmap_floor_spin.setSingleStep(0.1)
        self.sensor_parameter_heatmap_floor_spin.setValue(0.5)
        self.sensor_parameter_heatmap_floor_spin.setSuffix(" %")
        self.sensor_parameter_heatmap_floor_spin.setToolTip(
            "Signal changes at or below this value remain at the palette's base colour."
        )

        def _heatmap_proximity_spin(default_value):
            spin = QDoubleSpinBox(self.sensor_parameters_dialog)
            spin.setDecimals(1)
            spin.setRange(0.0, 1000000.0)
            spin.setSingleStep(5.0)
            spin.setValue(float(default_value))
            spin.setKeyboardTracking(False)
            return spin

        self.sensor_parameter_heatmap_proximity_floor_spin = (
            _heatmap_proximity_spin(20.0)
        )
        self.sensor_parameter_heatmap_proximity_knee_spin = (
            _heatmap_proximity_spin(100.0)
        )
        self.sensor_parameter_heatmap_proximity_saturation_spin = (
            _heatmap_proximity_spin(1000.0)
        )
        self.sensor_parameter_heatmap_proximity_floor_spin.setToolTip(
            "Absolute signal difference treated as no-touch noise."
        )
        self.sensor_parameter_heatmap_proximity_knee_spin.setToolTip(
            "End of the emphasized proximity color range."
        )
        self.sensor_parameter_heatmap_proximity_saturation_spin.setToolTip(
            "Absolute signal difference rendered as the strongest selected colour."
        )
        self.sensor_parameter_heatmap_3d_color_gain_spin = QDoubleSpinBox(
            self.sensor_parameters_dialog
        )
        self.sensor_parameter_heatmap_3d_color_gain_spin.setDecimals(2)
        self.sensor_parameter_heatmap_3d_color_gain_spin.setRange(1.0, 3.0)
        self.sensor_parameter_heatmap_3d_color_gain_spin.setSingleStep(0.1)
        self.sensor_parameter_heatmap_3d_color_gain_spin.setValue(1.5)
        self.sensor_parameter_heatmap_3d_color_gain_spin.setToolTip(
            "Strengthen intermediate 3D heatmap colours without changing "
            "the proximity baseline or saturation threshold."
        )

        general_group, general_list = components.section(
            "Sensor",
            "Which sensor these settings apply to, and how it is sampled.",
        )
        general_list.add_row("Sensor", self.sensor_parameter_model_combo)
        general_list.add_row(
            "Reorder logic",
            self.sensor_parameter_reorder_combo,
            hint="Applied the next time the sensor connects.",
        )
        general_list.add_row(
            "Force scale", self.sensor_parameter_force_scale_spin,
            hint="Newtons per unit of signal.",
        )
        general_list.add_row("Average window", self.sensor_average_window_spin)
        general_list.add_row("Render rate", self.visualization_target_hz_spin)

        self.sensor_parameter_point_labels_checkbox = QCheckBox("Show point labels")
        self.sensor_parameter_point_labels_checkbox.setToolTip(
            "Show each coarse sensor point label in the 3D sensor scene."
        )
        general_list.add(self.sensor_parameter_point_labels_checkbox)

        self.sensor_parameter_background_reference_checkbox = QCheckBox(
            "Show background axes and grid"
        )
        self.sensor_parameter_background_reference_checkbox.setToolTip(
            "Show the red/green reference axes and gray XY wireframe plane behind the sensor."
        )
        self.sensor_parameter_background_reference_checkbox.setChecked(True)
        general_list.add(self.sensor_parameter_background_reference_checkbox)
        layout.addWidget(general_group)

        view_group, view_list = components.section("Sensor view")
        self.sensor_parameter_save_view_button = QPushButton("Save")
        self.sensor_parameter_save_view_button.setToolTip(
            "Save the current sensor plotter orientation and zoom for this "
            "sensor model and grid size."
        )
        self.sensor_parameter_save_view_button.setEnabled(False)
        self.sensor_parameter_restore_view_button = QPushButton("Restore")
        self.sensor_parameter_restore_view_button.setToolTip(
            "Restore the saved sensor plotter orientation and zoom."
        )
        self.sensor_parameter_restore_view_button.setEnabled(False)
        view_buttons = QWidget()
        view_layout = QHBoxLayout(view_buttons)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(theme.SPACE_SM)
        view_layout.addWidget(self.sensor_parameter_save_view_button)
        view_layout.addWidget(self.sensor_parameter_restore_view_button)
        view_list.add(
            components.SettingsRow(
                "Camera",
                view_buttons,
                hint="Per sensor model and grid size.",
            )
        )
        layout.addWidget(view_group)

        stereo_group, stereo_list = components.section("Stereo field")
        stereo_list.add(self.sensor_parameter_stereo_ignore_noise_checkbox)
        stereo_list.add_row(
            "Noise threshold", self.sensor_parameter_stereo_deadband_spin
        )
        stereo_list.add_row("Line length", self.sensor_parameter_stereo_length_spin)
        layout.addWidget(stereo_group)

        heatmap_group, heatmap_list = components.section(
            "3D signal and heatmap"
        )
        heatmap_list.add_row("Signal sign", self.sensor_parameter_signal_mode_combo)
        heatmap_list.add_row(
            "Point grid response", self.sensor_parameter_point_grid_response_combo
        )
        heatmap_list.add_row(
            "Colour scale", self.sensor_parameter_heatmap_palette_combo
        )
        heatmap_list.add_row("Response", self.sensor_parameter_heatmap_response_combo)
        heatmap_list.add_row(
            "Full colour at", self.sensor_parameter_heatmap_saturation_spin
        )
        heatmap_list.add_row(
            "Noise floor", self.sensor_parameter_heatmap_floor_spin
        )
        heatmap_list.add_row(
            "Proximity baseline",
            self.sensor_parameter_heatmap_proximity_floor_spin,
        )
        heatmap_list.add_row(
            "Proximity knee", self.sensor_parameter_heatmap_proximity_knee_spin
        )
        heatmap_list.add_row(
            "Proximity saturation",
            self.sensor_parameter_heatmap_proximity_saturation_spin,
        )
        heatmap_list.add_row(
            "3D colour strength",
            self.sensor_parameter_heatmap_3d_color_gain_spin,
        )
        layout.addWidget(heatmap_group)

        geometry_group, geometry_list = components.section("2D geometry")
        geometry_list.add(self.sensor_parameter_use_shape_checkbox)
        geometry_list.add_row("Shape", self.sensor_parameter_shape_combo)
        geometry_list.add_row("Bend direction", self.sensor_parameter_bend_axis_combo)
        geometry_list.add_row("Arc angle", self.sensor_parameter_arc_spin)
        geometry_list.add(self.sensor_parameter_normal_flip_checkbox)
        geometry_list.add_row("Rotation X", self.sensor_parameter_rotation_x_spin)
        geometry_list.add_row("Rotation Y", self.sensor_parameter_rotation_y_spin)
        geometry_list.add_row("Rotation Z", self.sensor_parameter_rotation_z_spin)
        geometry_list.add(self.sensor_parameter_heatmap_follows_shape_checkbox)
        geometry_list.add(self.sensor_parameter_shape_editor_button)
        # Enabling/disabling the container reaches every control inside it,
        # which is what _refresh_sensor_parameters_dialog relies on.
        self.sensor_parameter_geometry_group = geometry_group
        layout.addWidget(geometry_group)

        self.sensor_parameter_status_label = QLabel("")
        self.sensor_parameter_status_label.setWordWrap(True)
        self.sensor_parameter_status_label.setStyleSheet(theme.MUTED_LABEL_STYLE)
        layout.addWidget(self.sensor_parameter_status_label)

        self.sensor_parameter_save_button = QPushButton("Save for Sensor")
        self.sensor_parameter_reload_button = QPushButton("Reload Saved")

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_button = button_box.button(QDialogButtonBox.Close)
        if close_button is not None:
            # Qt ships a red cross on this button, which reads as a
            # destructive action on a dialog that only dismisses.
            close_button.setIcon(QIcon())
        button_box.rejected.connect(self.sensor_parameters_dialog.close)
        footer_layout = QHBoxLayout()
        footer_layout.addStretch()
        footer_layout.addWidget(self.sensor_parameter_save_button)
        footer_layout.addWidget(self.sensor_parameter_reload_button)
        footer_layout.addWidget(button_box)
        outer_layout.addLayout(footer_layout)

        self.sensor_parameter_model_combo.currentIndexChanged.connect(
            self._load_sensor_parameter_reorder_mode
        )
        self.sensor_parameter_point_labels_checkbox.toggled.connect(
            self._on_sensor_parameter_point_labels_toggled
        )
        self.sensor_parameter_background_reference_checkbox.toggled.connect(
            self._on_sensor_parameter_background_reference_toggled
        )
        self.sensor_parameter_save_view_button.clicked.connect(
            self._save_sensor_plotter_perspective
        )
        self.sensor_parameter_restore_view_button.clicked.connect(
            self._restore_sensor_plotter_perspective
        )
        self.sensor_parameter_force_scale_spin.valueChanged.connect(
            self._on_sensor_parameter_force_scale_changed
        )
        self.sensor_parameter_use_shape_checkbox.toggled.connect(
            self._on_sensor_parameter_geometry_changed
        )
        self.sensor_parameter_shape_combo.currentIndexChanged.connect(
            self._on_sensor_parameter_geometry_changed
        )
        self.sensor_parameter_bend_axis_combo.currentIndexChanged.connect(
            self._on_sensor_parameter_geometry_changed
        )
        self.sensor_parameter_arc_spin.valueChanged.connect(
            self._on_sensor_parameter_geometry_changed
        )
        self.sensor_parameter_normal_flip_checkbox.toggled.connect(
            self._on_sensor_parameter_geometry_changed
        )
        self.sensor_parameter_heatmap_follows_shape_checkbox.toggled.connect(
            self._on_sensor_parameter_geometry_changed
        )
        self.sensor_parameter_rotation_x_spin.valueChanged.connect(
            self._on_sensor_parameter_geometry_changed
        )
        self.sensor_parameter_rotation_y_spin.valueChanged.connect(
            self._on_sensor_parameter_geometry_changed
        )
        self.sensor_parameter_rotation_z_spin.valueChanged.connect(
            self._on_sensor_parameter_geometry_changed
        )
        self.sensor_parameter_shape_editor_button.clicked.connect(
            self._open_sensor_shape_editor
        )
        self.sensor_parameter_stereo_ignore_noise_checkbox.toggled.connect(
            self._on_sensor_parameter_stereo_field_changed
        )
        self.sensor_parameter_stereo_deadband_spin.valueChanged.connect(
            self._on_sensor_parameter_stereo_field_changed
        )
        self.sensor_parameter_stereo_length_spin.valueChanged.connect(
            self._on_sensor_parameter_stereo_field_changed
        )
        self.sensor_parameter_heatmap_saturation_spin.valueChanged.connect(
            self._on_sensor_parameter_heatmap_changed
        )
        self.sensor_parameter_heatmap_floor_spin.valueChanged.connect(
            self._on_sensor_parameter_heatmap_changed
        )
        self.sensor_parameter_heatmap_response_combo.currentIndexChanged.connect(
            self._on_sensor_parameter_heatmap_changed
        )
        self.sensor_parameter_heatmap_palette_combo.currentIndexChanged.connect(
            self._on_sensor_parameter_heatmap_changed
        )
        self.sensor_parameter_signal_mode_combo.currentIndexChanged.connect(
            self._on_sensor_parameter_heatmap_changed
        )
        self.sensor_parameter_point_grid_response_combo.currentIndexChanged.connect(
            self._on_sensor_parameter_heatmap_changed
        )
        self.sensor_parameter_heatmap_proximity_floor_spin.valueChanged.connect(
            self._on_sensor_parameter_heatmap_changed
        )
        self.sensor_parameter_heatmap_proximity_knee_spin.valueChanged.connect(
            self._on_sensor_parameter_heatmap_changed
        )
        self.sensor_parameter_heatmap_proximity_saturation_spin.valueChanged.connect(
            self._on_sensor_parameter_heatmap_changed
        )
        self.sensor_parameter_heatmap_3d_color_gain_spin.valueChanged.connect(
            self._on_sensor_parameter_heatmap_changed
        )
        self.sensor_parameter_save_button.clicked.connect(
            self._save_sensor_parameter_reorder_mode
        )
        self.sensor_parameter_reload_button.clicked.connect(
            self._reload_saved_sensor_parameters
        )

    def open_sensor_parameters_dialog(self, tab=None):
        if not hasattr(self, "sensor_parameters_dialog"):
            self._build_sensor_parameters_dialog()
        self._refresh_sensor_parameters_dialog()
        if tab == "zero_mask":
            self.sensor_parameters_tabs.setCurrentWidget(
                self.sensor_zero_mask_panel
            )
        elif tab == "general":
            self.sensor_parameters_tabs.setCurrentWidget(
                self.sensor_parameters_general_tab
            )
        self.sensor_parameters_dialog.show()
        self.sensor_parameters_dialog.raise_()
        self.sensor_parameters_dialog.activateWindow()

    def _refresh_sensor_parameters_dialog(self):
        helper = getattr(self, "sensor_functions", None)
        if hasattr(self, "sensor_zero_mask_panel"):
            self.sensor_zero_mask_panel.set_sensor_functions(helper)

        selected_model = None
        if helper is not None and hasattr(helper, "get_sensor_model_name_for_index"):
            try:
                selected_model = helper.get_sensor_model_name_for_index(self.sensor_choice.currentRow())
            except Exception:
                selected_model = None

        self.sensor_parameter_model_combo.blockSignals(True)
        self.sensor_parameter_model_combo.clear()
        if helper is not None and hasattr(helper, "get_sensor_model_choices"):
            try:
                choices = helper.get_sensor_model_choices()
            except Exception:
                choices = []
        else:
            choices = []
        if not choices:
            choices = [
                ("elbow", "Elbow"),
                ("kuka", "Kuka"),
                ("double_curve", "Double Curve"),
                ("2d", "2D"),
                ("half_cylinder_surface", "Half Cylinder Surface"),
            ]
        for model_name, label in choices:
            self.sensor_parameter_model_combo.addItem(str(label), str(model_name))
        self._set_combo_current_data(self.sensor_parameter_model_combo, selected_model)
        self.sensor_parameter_model_combo.blockSignals(False)

        self.sensor_parameter_reorder_combo.blockSignals(True)
        self.sensor_parameter_reorder_combo.clear()
        if helper is not None and hasattr(helper, "get_reorder_logic_options"):
            try:
                modes = helper.get_reorder_logic_options()
            except Exception:
                modes = []
        else:
            modes = []
        if not modes:
            modes = [
                "factory",
                "none",
                "row_to_col",
                "row_to_col_flipped",
                "vertical_flip",
                "horizontal_flip",
                "flip_and_rotate",
                "rotate_180",
            ]
        for mode in modes:
            self.sensor_parameter_reorder_combo.addItem(
                self._sensor_reorder_mode_label(mode),
                str(mode),
            )
        self.sensor_parameter_reorder_combo.blockSignals(False)
        self._load_sensor_parameter_reorder_mode()

    def _selected_sensor_parameter_model_name(self):
        data = self.sensor_parameter_model_combo.currentData()
        if data is None:
            text = self.sensor_parameter_model_combo.currentText().strip()
            return text.lower().replace(" ", "_") if text else "sensor"
        return str(data)

    def _sensor_parameter_geometry_from_ui(self):
        return {
            "use_selected_shape": bool(self.sensor_parameter_use_shape_checkbox.isChecked()),
            "shape": str(self.sensor_parameter_shape_combo.currentData() or "flat"),
            "bend_axis": str(self.sensor_parameter_bend_axis_combo.currentData() or "columns"),
            "arc_deg": float(self.sensor_parameter_arc_spin.value()),
            "normal_flip": bool(self.sensor_parameter_normal_flip_checkbox.isChecked()),
            "rotation_deg": [
                float(self.sensor_parameter_rotation_x_spin.value()),
                float(self.sensor_parameter_rotation_y_spin.value()),
                float(self.sensor_parameter_rotation_z_spin.value()),
            ],
            "custom_points": list(
                getattr(self, "_sensor_parameter_custom_points", []) or []
            ),
            "use_custom_heatmap_shape": bool(
                not self.sensor_parameter_heatmap_follows_shape_checkbox.isChecked()
            ),
            "custom_heatmap_points": list(
                getattr(self, "_sensor_parameter_custom_heatmap_points", [])
                or []
            ),
            "custom_heatmap_corners": list(
                getattr(self, "_sensor_parameter_custom_heatmap_corners", [])
                or []
            ),
            "use_curved_heatmap_edges": bool(
                getattr(
                    self,
                    "_sensor_parameter_use_curved_heatmap_edges",
                    False,
                )
            ),
            "custom_heatmap_edge_offsets": list(
                getattr(
                    self,
                    "_sensor_parameter_custom_heatmap_edge_offsets",
                    [],
                )
                or []
            ),
        }

    @staticmethod
    def _sensor_parameter_rotation_from_geometry(geometry):
        if not isinstance(geometry, dict):
            return [0.0, 0.0, 0.0]
        raw_rotation = geometry.get("rotation_deg")
        if isinstance(raw_rotation, dict):
            raw_values = [
                raw_rotation.get("x", raw_rotation.get("rx", 0.0)),
                raw_rotation.get("y", raw_rotation.get("ry", 0.0)),
                raw_rotation.get("z", raw_rotation.get("rz", 0.0)),
            ]
        elif isinstance(raw_rotation, (list, tuple, np.ndarray)) and len(raw_rotation) >= 3:
            raw_values = raw_rotation[:3]
        else:
            raw_values = [
                geometry.get("rotation_x_deg", 0.0),
                geometry.get("rotation_y_deg", 0.0),
                geometry.get("rotation_z_deg", 0.0),
            ]

        rotation = []
        for raw_value in raw_values:
            try:
                rotation.append(float(np.clip(float(raw_value), -180.0, 180.0)))
            except Exception:
                rotation.append(0.0)
        return rotation

    def _sensor_parameter_stereo_field_from_ui(self):
        return {
            "ignore_noise_enabled": bool(
                self.sensor_parameter_stereo_ignore_noise_checkbox.isChecked()
            ),
            "deadband_pct": float(self.sensor_parameter_stereo_deadband_spin.value()),
            "response_scale_pct": 2.0,
            "length_scale": float(self.sensor_parameter_stereo_length_spin.value()),
        }

    def _set_sensor_parameter_stereo_field_controls(self, stereo_field):
        stereo_field = stereo_field if isinstance(stereo_field, dict) else {}
        ignore_noise = bool(stereo_field.get("ignore_noise_enabled", True))
        try:
            deadband_pct = float(stereo_field.get("deadband_pct", 0.35))
        except Exception:
            deadband_pct = 0.35
        try:
            length_scale = float(stereo_field.get("length_scale", 0.35))
        except Exception:
            length_scale = 0.35

        widgets = [
            self.sensor_parameter_stereo_ignore_noise_checkbox,
            self.sensor_parameter_stereo_deadband_spin,
            self.sensor_parameter_stereo_length_spin,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        self.sensor_parameter_stereo_ignore_noise_checkbox.setChecked(ignore_noise)
        self.sensor_parameter_stereo_deadband_spin.setValue(float(np.clip(deadband_pct, 0.0, 20.0)))
        self.sensor_parameter_stereo_length_spin.setValue(float(np.clip(length_scale, 0.05, 2.0)))
        self.sensor_parameter_stereo_deadband_spin.setEnabled(ignore_noise)
        for widget in widgets:
            widget.blockSignals(False)

    def _sensor_parameter_heatmap_from_ui(self):
        proximity_floor = float(
            self.sensor_parameter_heatmap_proximity_floor_spin.value()
        )
        proximity_knee = max(
            proximity_floor + 0.1,
            float(self.sensor_parameter_heatmap_proximity_knee_spin.value()),
        )
        proximity_saturation = max(
            proximity_knee + 0.1,
            float(self.sensor_parameter_heatmap_proximity_saturation_spin.value()),
        )
        return {
            "use_absolute_signal": bool(
                self.sensor_parameter_signal_mode_combo.currentData()
            ),
            "point_grid_response_mode": str(
                self.sensor_parameter_point_grid_response_combo.currentData()
                or "zero_centered"
            ),
            "palette_3d": str(
                self.sensor_parameter_heatmap_palette_combo.currentData()
                or "white_red"
            ),
            "response_mode": str(
                self.sensor_parameter_heatmap_response_combo.currentData()
                or "linear_relative"
            ),
            "saturation_pct": float(
                self.sensor_parameter_heatmap_saturation_spin.value()
            ),
            "noise_floor_pct": float(
                self.sensor_parameter_heatmap_floor_spin.value()
            ),
            "proximity_noise_floor": proximity_floor,
            "proximity_knee": proximity_knee,
            "proximity_saturation": proximity_saturation,
            "color_gain_3d": float(
                self.sensor_parameter_heatmap_3d_color_gain_spin.value()
            ),
        }

    def _set_sensor_parameter_heatmap_controls(self, heatmap):
        heatmap = heatmap if isinstance(heatmap, dict) else {}
        use_absolute_signal = bool(
            heatmap.get("use_absolute_signal", True)
        )
        point_grid_response_mode = str(
            heatmap.get("point_grid_response_mode", "zero_centered")
        )
        if point_grid_response_mode not in ("zero_centered", "legacy_offset"):
            point_grid_response_mode = "zero_centered"
        response_mode = str(heatmap.get("response_mode", "linear_relative"))
        if response_mode not in ("linear_relative", "proximity_enhanced"):
            response_mode = "linear_relative"
        palette_3d = str(heatmap.get("palette_3d", "white_red"))
        if palette_3d not in (
            "white_red",
            "white_blue_red",
            "light_deep_blue",
        ):
            palette_3d = "white_red"
        try:
            saturation_pct = float(heatmap.get("saturation_pct", 5.0))
        except Exception:
            saturation_pct = 5.0
        try:
            noise_floor_pct = float(heatmap.get("noise_floor_pct", 0.5))
        except Exception:
            noise_floor_pct = 0.5
        try:
            proximity_floor = float(heatmap.get("proximity_noise_floor", 20.0))
        except Exception:
            proximity_floor = 20.0
        try:
            proximity_knee = float(heatmap.get("proximity_knee", 100.0))
        except Exception:
            proximity_knee = 100.0
        try:
            proximity_saturation = float(
                heatmap.get("proximity_saturation", 1000.0)
            )
        except Exception:
            proximity_saturation = 1000.0
        try:
            color_gain_3d = float(heatmap.get("color_gain_3d", 1.5))
        except Exception:
            color_gain_3d = 1.5
        proximity_floor = float(np.clip(proximity_floor, 0.0, 1000000.0))
        proximity_knee = float(
            np.clip(proximity_knee, proximity_floor + 0.1, 1000000.0)
        )
        proximity_saturation = float(
            np.clip(proximity_saturation, proximity_knee + 0.1, 1000000.0)
        )
        widgets = [
            self.sensor_parameter_signal_mode_combo,
            self.sensor_parameter_point_grid_response_combo,
            self.sensor_parameter_heatmap_palette_combo,
            self.sensor_parameter_heatmap_response_combo,
            self.sensor_parameter_heatmap_saturation_spin,
            self.sensor_parameter_heatmap_floor_spin,
            self.sensor_parameter_heatmap_proximity_floor_spin,
            self.sensor_parameter_heatmap_proximity_knee_spin,
            self.sensor_parameter_heatmap_proximity_saturation_spin,
            self.sensor_parameter_heatmap_3d_color_gain_spin,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        self._set_combo_current_data(
            self.sensor_parameter_signal_mode_combo,
            use_absolute_signal,
        )
        self._set_combo_current_data(
            self.sensor_parameter_point_grid_response_combo,
            point_grid_response_mode,
        )
        self._set_combo_current_data(
            self.sensor_parameter_heatmap_palette_combo, palette_3d
        )
        self._set_combo_current_data(
            self.sensor_parameter_heatmap_response_combo, response_mode
        )
        self.sensor_parameter_heatmap_saturation_spin.setValue(
            float(np.clip(saturation_pct, 0.1, 50.0))
        )
        self.sensor_parameter_heatmap_floor_spin.setValue(
            float(np.clip(noise_floor_pct, 0.0, 20.0))
        )
        self.sensor_parameter_heatmap_proximity_floor_spin.setValue(
            proximity_floor
        )
        self.sensor_parameter_heatmap_proximity_knee_spin.setValue(
            proximity_knee
        )
        self.sensor_parameter_heatmap_proximity_saturation_spin.setValue(
            proximity_saturation
        )
        self.sensor_parameter_heatmap_3d_color_gain_spin.setValue(
            float(np.clip(color_gain_3d, 1.0, 3.0))
        )
        for widget in widgets:
            widget.blockSignals(False)
        self._update_sensor_parameter_heatmap_controls()

    def _update_sensor_parameter_heatmap_controls(self):
        use_absolute_signal = bool(
            self.sensor_parameter_signal_mode_combo.currentData()
        )
        enhanced = (
            str(self.sensor_parameter_heatmap_response_combo.currentData())
            == "proximity_enhanced"
        )
        self.sensor_parameter_heatmap_palette_combo.setEnabled(
            use_absolute_signal
        )
        self.sensor_parameter_heatmap_saturation_spin.setEnabled(not enhanced)
        self.sensor_parameter_heatmap_floor_spin.setEnabled(not enhanced)
        self.sensor_parameter_heatmap_proximity_floor_spin.setEnabled(enhanced)
        self.sensor_parameter_heatmap_proximity_knee_spin.setEnabled(enhanced)
        self.sensor_parameter_heatmap_proximity_saturation_spin.setEnabled(enhanced)

    def _update_sensor_parameter_geometry_subcontrols(self):
        group_enabled = bool(
            getattr(self, "sensor_parameter_geometry_group", None) is not None
            and self.sensor_parameter_geometry_group.isEnabled()
        )
        shape_enabled = group_enabled and self.sensor_parameter_use_shape_checkbox.isChecked()
        shape = str(self.sensor_parameter_shape_combo.currentData() or "flat")
        self.sensor_parameter_shape_combo.setEnabled(shape_enabled)
        self.sensor_parameter_bend_axis_combo.setEnabled(
            shape_enabled and shape == "cylinder"
        )
        self.sensor_parameter_arc_spin.setEnabled(
            shape_enabled and shape == "cylinder"
        )
        self.sensor_parameter_normal_flip_checkbox.setEnabled(shape_enabled)
        self.sensor_parameter_heatmap_follows_shape_checkbox.setEnabled(group_enabled)
        self.sensor_parameter_shape_editor_button.setEnabled(group_enabled)
        for widget in (
            self.sensor_parameter_rotation_x_spin,
            self.sensor_parameter_rotation_y_spin,
            self.sensor_parameter_rotation_z_spin,
        ):
            widget.setEnabled(group_enabled)

    def _set_sensor_parameter_geometry_controls_enabled(self, enabled: bool):
        enabled = bool(enabled)
        group = getattr(self, "sensor_parameter_geometry_group", None)
        if group is not None:
            group.setEnabled(enabled)
        self._update_sensor_parameter_geometry_subcontrols()

    def _set_sensor_parameter_geometry_controls(self, geometry):
        geometry = geometry if isinstance(geometry, dict) else {}
        shape = str(geometry.get("shape", "flat") or "flat")
        bend_axis = str(geometry.get("bend_axis", "columns") or "columns")
        try:
            arc_deg = float(geometry.get("arc_deg", 0.0) or 0.0)
        except Exception:
            arc_deg = 0.0
        normal_flip = bool(geometry.get("normal_flip", False))
        custom_points = geometry.get("custom_points", [])
        self._sensor_parameter_custom_points = (
            list(custom_points) if isinstance(custom_points, list) else []
        )
        self._sensor_parameter_use_custom_heatmap_shape = bool(
            geometry.get("use_custom_heatmap_shape", False)
        )
        custom_heatmap_points = geometry.get("custom_heatmap_points", [])
        self._sensor_parameter_custom_heatmap_points = (
            list(custom_heatmap_points)
            if isinstance(custom_heatmap_points, list)
            else []
        )
        custom_heatmap_corners = geometry.get("custom_heatmap_corners", [])
        self._sensor_parameter_custom_heatmap_corners = (
            list(custom_heatmap_corners)
            if isinstance(custom_heatmap_corners, list)
            else []
        )
        self._sensor_parameter_use_curved_heatmap_edges = bool(
            geometry.get("use_curved_heatmap_edges", False)
        )
        custom_heatmap_edge_offsets = geometry.get(
            "custom_heatmap_edge_offsets", []
        )
        self._sensor_parameter_custom_heatmap_edge_offsets = (
            list(custom_heatmap_edge_offsets)
            if isinstance(custom_heatmap_edge_offsets, list)
            else []
        )
        use_selected_shape = bool(
            geometry.get(
                "use_selected_shape",
                shape != "flat" or abs(arc_deg) > 1e-6,
            )
        )
        rotation_deg = self._sensor_parameter_rotation_from_geometry(geometry)

        widgets = [
            self.sensor_parameter_use_shape_checkbox,
            self.sensor_parameter_shape_combo,
            self.sensor_parameter_bend_axis_combo,
            self.sensor_parameter_arc_spin,
            self.sensor_parameter_normal_flip_checkbox,
            self.sensor_parameter_heatmap_follows_shape_checkbox,
            self.sensor_parameter_rotation_x_spin,
            self.sensor_parameter_rotation_y_spin,
            self.sensor_parameter_rotation_z_spin,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        self.sensor_parameter_use_shape_checkbox.setChecked(use_selected_shape)
        self._set_combo_current_data(self.sensor_parameter_shape_combo, shape)
        self._set_combo_current_data(self.sensor_parameter_bend_axis_combo, bend_axis)
        self.sensor_parameter_arc_spin.setValue(float(np.clip(arc_deg, -180.0, 180.0)))
        self.sensor_parameter_normal_flip_checkbox.setChecked(normal_flip)
        self.sensor_parameter_heatmap_follows_shape_checkbox.setChecked(
            not self._sensor_parameter_use_custom_heatmap_shape
        )
        self.sensor_parameter_rotation_x_spin.setValue(rotation_deg[0])
        self.sensor_parameter_rotation_y_spin.setValue(rotation_deg[1])
        self.sensor_parameter_rotation_z_spin.setValue(rotation_deg[2])
        for widget in widgets:
            widget.blockSignals(False)
        self._update_sensor_parameter_geometry_subcontrols()

    def _load_sensor_parameter_reorder_mode(self):
        helper = getattr(self, "sensor_functions", None)
        model_name = self._selected_sensor_parameter_model_name()
        if helper is None or not hasattr(helper, "get_sensor_reorder_context"):
            self.sensor_parameter_status_label.setText("Sensor parameters are not ready.")
            return
        try:
            context = helper.get_sensor_reorder_context(model_name)
        except Exception as exc:
            self.sensor_parameter_status_label.setText(f"Failed to load sensor parameters: {exc}")
            return

        saved_mode = str(context.get("saved_mode", "factory"))
        self._set_combo_current_data(self.sensor_parameter_reorder_combo, saved_mode)
        self.sensor_parameter_point_labels_checkbox.blockSignals(True)
        self.sensor_parameter_point_labels_checkbox.setChecked(
            bool(context.get("point_labels_enabled", False))
        )
        self.sensor_parameter_point_labels_checkbox.blockSignals(False)
        self.sensor_parameter_background_reference_checkbox.blockSignals(True)
        self.sensor_parameter_background_reference_checkbox.setChecked(
            bool(context.get("background_reference_enabled", True))
        )
        self.sensor_parameter_background_reference_checkbox.blockSignals(False)
        self.sensor_parameter_force_scale_spin.blockSignals(True)
        self.sensor_parameter_force_scale_spin.setValue(
            float(context.get("force_scale_n_per_signal", 0.0) or 0.0)
        )
        self.sensor_parameter_force_scale_spin.blockSignals(False)
        geometry = context.get("geometry", {}) or {}
        self._set_sensor_parameter_geometry_controls(geometry)
        self._set_sensor_parameter_geometry_controls_enabled(model_name == "2d")
        stereo_field = context.get("stereo_field", {}) or {}
        self._set_sensor_parameter_stereo_field_controls(stereo_field)
        heatmap = context.get("heatmap", {}) or {}
        self._set_sensor_parameter_heatmap_controls(heatmap)
        saved_camera = context.get("camera")
        selected_is_current = self._sensor_parameter_selected_model_is_current()
        self.sensor_parameter_save_view_button.setEnabled(selected_is_current)
        self.sensor_parameter_restore_view_button.setEnabled(
            selected_is_current and bool(saved_camera)
        )
        default_logic = context.get("default_logic") or "none"
        effective_logic = context.get("effective_logic") or "none"
        point_labels = "on" if context.get("point_labels_enabled", False) else "off"
        background_reference = "on" if context.get("background_reference_enabled", True) else "off"
        force_scale = float(context.get("force_scale_n_per_signal", 0.0) or 0.0)
        geometry_shape = str(geometry.get("shape", "flat") or "flat")
        geometry_axis = str(geometry.get("bend_axis", "columns") or "columns")
        geometry_arc = float(geometry.get("arc_deg", 0.0) or 0.0)
        geometry_normals = "flipped" if geometry.get("normal_flip", False) else "normal"
        geometry_rotation = self._sensor_parameter_rotation_from_geometry(geometry)
        geometry_mode = (
            "selected shape"
            if geometry.get("use_selected_shape", geometry_shape != "flat" or abs(geometry_arc) > 1e-6)
            else "flat 2D"
        )
        stereo_noise = "ignored" if stereo_field.get("ignore_noise_enabled", True) else "raw"
        self.sensor_parameter_status_label.setText(
            f"Key: {context.get('key', model_name)}\n"
            f"Factory default: {default_logic}\n"
            f"Saved mode: {self._sensor_reorder_mode_label(saved_mode)}\n"
            f"Effective on next connect: {effective_logic}\n"
            f"Point labels: {point_labels}\n"
            f"Background axes/grid: {background_reference}\n"
            f"Saved sensor view: {'yes' if saved_camera else 'no'}\n"
            f"Force scale: {force_scale:.6f} N/signal\n"
            f"Stereo field: noise {stereo_noise}, threshold "
            f"{float(stereo_field.get('deadband_pct', 0.35) or 0.0):.2f}, "
            f"length {float(stereo_field.get('length_scale', 0.35) or 0.35):.2f}\n"
            f"3D heatmap: {heatmap.get('response_mode', 'linear_relative')}, "
            f"{'magnitude |signal|' if heatmap.get('use_absolute_signal', True) else 'signed blue-/red+'}, "
            f"linear full red "
            f"{float(heatmap.get('saturation_pct', 5.0) or 5.0):.2f}%, "
            f"floor {float(heatmap.get('noise_floor_pct', 0.5) or 0.0):.2f}%; "
            f"proximity {float(heatmap.get('proximity_noise_floor', 20.0) or 0.0):.1f}/"
            f"{float(heatmap.get('proximity_knee', 100.0) or 0.0):.1f}/"
            f"{float(heatmap.get('proximity_saturation', 1000.0) or 0.0):.1f}\n"
            f"Geometry: {geometry_mode}; {geometry_shape}, {geometry_axis}, "
            f"{geometry_arc:.1f} deg, {geometry_normals}, "
            f"rot XYZ=({geometry_rotation[0]:.1f}, {geometry_rotation[1]:.1f}, "
            f"{geometry_rotation[2]:.1f}) deg"
        )

    def _sensor_parameter_selected_model_is_current(self):
        helper = getattr(self, "sensor_functions", None)
        if helper is None:
            return False
        return str(getattr(helper, "current_model_name", "")) == self._selected_sensor_parameter_model_name()

    def _on_sensor_parameter_point_labels_toggled(self, checked: bool):
        helper = getattr(self, "sensor_functions", None)
        if (
            helper is not None
            and hasattr(helper, "set_sensor_point_labels_enabled")
            and self._sensor_parameter_selected_model_is_current()
        ):
            helper.set_sensor_point_labels_enabled(bool(checked), save_current_sensor=False)

    def _on_sensor_parameter_background_reference_toggled(self, checked: bool):
        helper = getattr(self, "sensor_functions", None)
        if (
            helper is not None
            and hasattr(helper, "set_sensor_background_reference_enabled")
            and self._sensor_parameter_selected_model_is_current()
        ):
            helper.set_sensor_background_reference_enabled(
                bool(checked),
                save_current_sensor=False,
                render=True,
            )

    def _on_sensor_parameter_force_scale_changed(self, value: float):
        helper = getattr(self, "sensor_functions", None)
        if (
            helper is not None
            and hasattr(helper, "set_sensor_contact_force_scale")
            and self._sensor_parameter_selected_model_is_current()
        ):
            helper.set_sensor_contact_force_scale(float(value), save_current_sensor=False)

    def _save_sensor_plotter_perspective(self):
        helper = getattr(self, "sensor_functions", None)
        if helper is None or not self._sensor_parameter_selected_model_is_current():
            self.sensor_parameter_status_label.setText(
                "Build the selected sensor before saving its current view."
            )
            return
        try:
            saved = bool(helper.save_current_sensor_perspective())
        except Exception as exc:
            self.sensor_parameter_status_label.setText(
                f"Failed to save the sensor view: {exc}"
            )
            return
        if not saved:
            self.sensor_parameter_status_label.setText(
                "The current sensor view could not be saved."
            )
            return
        self.sensor_parameter_status_label.setText(
            f"Current plotter view saved for "
            f"{helper.get_sensor_reorder_key(helper.current_model_name)}."
        )
        self.sensor_parameter_restore_view_button.setEnabled(True)

    def _restore_sensor_plotter_perspective(self):
        helper = getattr(self, "sensor_functions", None)
        if helper is None or not self._sensor_parameter_selected_model_is_current():
            self.sensor_parameter_status_label.setText(
                "Build the selected sensor before restoring its saved view."
            )
            return
        try:
            restored = bool(helper.restore_saved_sensor_perspective(render=True))
        except Exception as exc:
            self.sensor_parameter_status_label.setText(
                f"Failed to restore the sensor view: {exc}"
            )
            return
        if restored:
            self.sensor_parameter_status_label.setText(
                f"Saved plotter view restored for "
                f"{helper.get_sensor_reorder_key(helper.current_model_name)}."
            )
        else:
            self.sensor_parameter_status_label.setText(
                "No saved view is available for the current sensor."
            )

    def _reload_saved_sensor_parameters(self):
        self._load_sensor_parameter_reorder_mode()
        helper = getattr(self, "sensor_functions", None)
        if (
            helper is None
            or not self._sensor_parameter_selected_model_is_current()
            or not self.sensor_parameter_restore_view_button.isEnabled()
        ):
            return
        try:
            restored = bool(helper.restore_saved_sensor_perspective(render=True))
        except Exception as exc:
            self.sensor_parameter_status_label.setText(
                f"Sensor parameters loaded, but the saved view failed: {exc}"
            )
            return
        if restored:
            self.sensor_parameter_status_label.setText(
                f"Saved parameters and plotter view restored for "
                f"{helper.get_sensor_reorder_key(helper.current_model_name)}."
            )

    def _on_sensor_parameter_geometry_changed(self, *_args):
        self._update_sensor_parameter_geometry_subcontrols()
        helper = getattr(self, "sensor_functions", None)
        if (
            helper is not None
            and hasattr(helper, "set_sensor_geometry_config")
            and self._sensor_parameter_selected_model_is_current()
        ):
            helper.set_sensor_geometry_config(
                self._sensor_parameter_geometry_from_ui(),
                save_current_sensor=False,
                render=True,
            )

    def _open_sensor_shape_editor(self):
        helper = getattr(self, "sensor_functions", None)
        if self._selected_sensor_parameter_model_name() != "2d":
            self.sensor_parameter_status_label.setText(
                "The interactive shape editor currently supports the 2D sensor."
            )
            return
        if (
            helper is None
            or str(getattr(helper, "current_model_name", "") or "") != "2d"
            or not hasattr(helper, "get_sensor_geometry_editor_data")
        ):
            self.sensor_parameter_status_label.setText(
                "Build the 2D sensor scene before opening the shape editor."
            )
            return

        geometry = self._sensor_parameter_geometry_from_ui()
        try:
            editor_data = helper.get_sensor_geometry_editor_data(geometry)
        except Exception as exc:
            self.sensor_parameter_status_label.setText(
                f"Could not prepare the shape editor: {exc}"
            )
            return
        if not editor_data:
            self.sensor_parameter_status_label.setText(
                "The current 2D sensor geometry is not ready for editing."
            )
            return

        try:
            from phd.ui.sensor_shape_editor import SensorShapeEditorDialog

            dialog = SensorShapeEditorDialog(
                editor_data,
                geometry,
                parent=self.sensor_parameters_dialog,
            )
            accepted = dialog.exec_() == QDialog.Accepted
            if not accepted:
                return
            custom_geometry = dialog.result_geometry_config()
            self._set_sensor_parameter_geometry_controls(custom_geometry)
            applied = bool(
                helper.set_sensor_geometry_config(
                    custom_geometry,
                    save_current_sensor=True,
                    render=True,
                )
            )
        except Exception as exc:
            self.sensor_parameter_status_label.setText(
                f"Custom shape editor failed: {exc}"
            )
            return

        if applied:
            self.sensor_parameter_status_label.setText(
                f"Custom shape saved for 2d_{helper.n_row}x{helper.n_col}."
            )
        else:
            self.sensor_parameter_status_label.setText(
                "The custom shape could not be applied to the current sensor."
            )

    def _on_sensor_parameter_stereo_field_changed(self, *_args):
        self.sensor_parameter_stereo_deadband_spin.setEnabled(
            self.sensor_parameter_stereo_ignore_noise_checkbox.isChecked()
        )
        helper = getattr(self, "sensor_functions", None)
        if (
            helper is not None
            and hasattr(helper, "set_stereo_field_settings")
            and self._sensor_parameter_selected_model_is_current()
        ):
            helper.set_stereo_field_settings(
                self._sensor_parameter_stereo_field_from_ui(),
                save_current_sensor=False,
            )

    def _on_sensor_parameter_heatmap_changed(self, *_args):
        settings = self._sensor_parameter_heatmap_from_ui()
        self._set_sensor_parameter_heatmap_controls(settings)
        helper = getattr(self, "sensor_functions", None)
        if (
            helper is not None
            and hasattr(helper, "set_heatmap_settings")
            and self._sensor_parameter_selected_model_is_current()
        ):
            helper.set_heatmap_settings(
                settings,
                save_current_sensor=False,
            )

    def _save_sensor_parameter_reorder_mode(self):
        helper = getattr(self, "sensor_functions", None)
        model_name = self._selected_sensor_parameter_model_name()
        mode = self.sensor_parameter_reorder_combo.currentData()
        if helper is None or not hasattr(helper, "set_saved_sensor_reorder_mode"):
            self.sensor_parameter_status_label.setText("Sensor parameters are not ready.")
            return
        try:
            ok = bool(helper.set_saved_sensor_reorder_mode(model_name, mode))
            if hasattr(helper, "set_saved_sensor_point_labels_enabled"):
                ok = bool(
                    helper.set_saved_sensor_point_labels_enabled(
                        model_name,
                        self.sensor_parameter_point_labels_checkbox.isChecked(),
                    )
                ) and ok
            if hasattr(helper, "set_saved_sensor_background_reference_enabled"):
                ok = bool(
                    helper.set_saved_sensor_background_reference_enabled(
                        model_name,
                        self.sensor_parameter_background_reference_checkbox.isChecked(),
                    )
                ) and ok
            if hasattr(helper, "set_saved_sensor_contact_force_scale"):
                ok = bool(
                    helper.set_saved_sensor_contact_force_scale(
                        model_name,
                        self.sensor_parameter_force_scale_spin.value(),
                    )
                ) and ok
            if hasattr(helper, "set_saved_sensor_geometry_config"):
                ok = bool(
                    helper.set_saved_sensor_geometry_config(
                        model_name,
                        self._sensor_parameter_geometry_from_ui(),
                    )
                ) and ok
            if hasattr(helper, "set_saved_sensor_stereo_field_config"):
                ok = bool(
                    helper.set_saved_sensor_stereo_field_config(
                        model_name,
                        self._sensor_parameter_stereo_field_from_ui(),
                    )
                ) and ok
            if hasattr(helper, "set_saved_sensor_heatmap_config"):
                ok = bool(
                    helper.set_saved_sensor_heatmap_config(
                        model_name,
                        self._sensor_parameter_heatmap_from_ui(),
                    )
                ) and ok
            if (
                hasattr(helper, "set_sensor_point_labels_enabled")
                and self._sensor_parameter_selected_model_is_current()
            ):
                helper.set_sensor_point_labels_enabled(
                    self.sensor_parameter_point_labels_checkbox.isChecked(),
                    save_current_sensor=False,
                )
            if (
                hasattr(helper, "set_sensor_background_reference_enabled")
                and self._sensor_parameter_selected_model_is_current()
            ):
                helper.set_sensor_background_reference_enabled(
                    self.sensor_parameter_background_reference_checkbox.isChecked(),
                    save_current_sensor=False,
                    render=True,
                )
            if (
                hasattr(helper, "set_sensor_contact_force_scale")
                and self._sensor_parameter_selected_model_is_current()
            ):
                helper.set_sensor_contact_force_scale(
                    self.sensor_parameter_force_scale_spin.value(),
                    save_current_sensor=False,
                )
            if (
                hasattr(helper, "set_sensor_geometry_config")
                and self._sensor_parameter_selected_model_is_current()
            ):
                helper.set_sensor_geometry_config(
                    self._sensor_parameter_geometry_from_ui(),
                    save_current_sensor=False,
                    render=True,
                )
            if (
                hasattr(helper, "set_stereo_field_settings")
                and self._sensor_parameter_selected_model_is_current()
            ):
                helper.set_stereo_field_settings(
                    self._sensor_parameter_stereo_field_from_ui(),
                    save_current_sensor=False,
                )
            if (
                hasattr(helper, "set_heatmap_settings")
                and self._sensor_parameter_selected_model_is_current()
            ):
                helper.set_heatmap_settings(
                    self._sensor_parameter_heatmap_from_ui(),
                    save_current_sensor=False,
                )
            context = helper.get_sensor_reorder_context(model_name)
        except Exception as exc:
            self.sensor_parameter_status_label.setText(f"Failed to save sensor parameters: {exc}")
            return

        if ok:
            saved_geometry = context.get("geometry", {}) or {}
            saved_stereo = context.get("stereo_field", {}) or {}
            saved_heatmap = context.get("heatmap", {}) or {}
            saved_camera = context.get("camera")
            saved_shape = str(saved_geometry.get("shape", "flat") or "flat")
            saved_arc = float(saved_geometry.get("arc_deg", 0.0) or 0.0)
            saved_rotation = self._sensor_parameter_rotation_from_geometry(saved_geometry)
            saved_mode = (
                "selected shape"
                if saved_geometry.get(
                    "use_selected_shape",
                    saved_shape != "flat" or abs(saved_arc) > 1e-6,
                )
                else "flat 2D"
            )
            self.sensor_parameter_status_label.setText(
                f"Saved for {context.get('key', model_name)}.\n"
                f"Effective on next connect: {context.get('effective_logic') or 'none'}\n"
                f"Point labels: {'on' if context.get('point_labels_enabled', False) else 'off'}\n"
                f"Background axes/grid: {'on' if context.get('background_reference_enabled', True) else 'off'}\n"
                f"Saved sensor view: {'yes' if saved_camera else 'no'}\n"
                f"Force scale: {float(context.get('force_scale_n_per_signal', 0.0) or 0.0):.6f} N/signal\n"
                f"Stereo field: noise "
                f"{'ignored' if saved_stereo.get('ignore_noise_enabled', True) else 'raw'}, "
                f"threshold {float(saved_stereo.get('deadband_pct', 0.35) or 0.0):.2f}, "
                f"length {float(saved_stereo.get('length_scale', 0.35) or 0.35):.2f}\n"
                f"3D heatmap: {saved_heatmap.get('response_mode', 'linear_relative')}, "
                f"{'magnitude |signal|' if saved_heatmap.get('use_absolute_signal', True) else 'signed blue-/red+'}, "
                f"palette {saved_heatmap.get('palette_3d', 'white_red')}, "
                f"linear full red "
                f"{float(saved_heatmap.get('saturation_pct', 5.0) or 5.0):.2f}%, "
                f"floor {float(saved_heatmap.get('noise_floor_pct', 0.5) or 0.0):.2f}%; "
                f"proximity "
                f"{float(saved_heatmap.get('proximity_noise_floor', 20.0) or 0.0):.1f}/"
                f"{float(saved_heatmap.get('proximity_knee', 100.0) or 0.0):.1f}/"
                f"{float(saved_heatmap.get('proximity_saturation', 1000.0) or 0.0):.1f}\n"
                f"Geometry: {saved_mode}; {saved_shape}, "
                f"{saved_geometry.get('bend_axis', 'columns')}, "
                f"{saved_arc:.1f} deg, "
                f"rot XYZ=({saved_rotation[0]:.1f}, {saved_rotation[1]:.1f}, "
                f"{saved_rotation[2]:.1f}) deg"
            )
        else:
            self.sensor_parameter_status_label.setText("Failed to save sensor parameters.")

    def setup_tab2(self, layout):
        layout.setSpacing(theme.SPACE_LG)

        self.read_group_robot, read_list = components.section(
            "Read",
            "Query the arm's current state.",
        )
        send_group, send_list = components.section(
            "Command",
            "Pick what to send; its editor opens below.",
        )

        self.read_joint_angle_button = QPushButton("Read")
        self.read_tool_position_button = QPushButton("Read")
        self.show_robot_button = QPushButton("Open")

        # The row label names the thing; the button carries the verb, so a
        # column of "Read" reads as one action applied to several quantities.
        read_list.add(
            components.SettingsRow("Joint angles", self.read_joint_angle_button)
        )
        read_list.add(
            components.SettingsRow("Tool position", self.read_tool_position_button)
        )
        read_list.add(
            components.SettingsRow(
                "3D robot model",
                self.show_robot_button,
                hint="Opens the standalone viewer with sensor mapping and "
                     "admittance controls.",
            )
        )

        self.send_position_PTP_J_button = QPushButton("Edit")
        self.send_position_PTP_T_button = QPushButton("Edit")
        self.send_position_PTP_T_toolframe_button = QPushButton("Edit")
        self.send_script_button = QPushButton("Edit")

        for label, hint, button in (
            (
                "Joint angles",
                "Move each of the six joints to an absolute angle.",
                self.send_position_PTP_J_button,
            ),
            (
                "Tool position",
                "Absolute pose in the base frame.",
                self.send_position_PTP_T_button,
            ),
            (
                "Tool velocity",
                "Continuous motion in the tool frame.",
                self.send_position_PTP_T_toolframe_button,
            ),
            (
                "TM script",
                "Raw SendScript command.",
                self.send_script_button,
            ),
        ):
            send_list.add(components.SettingsRow(label, button, hint=hint))

        self.robot_basic_controls_group = components.CollapsibleGroup(
            "Basic controls",
            expanded=True,
        )
        self.robot_basic_controls_group.add(self.read_group_robot)
        self.robot_advanced_controls_group = components.CollapsibleGroup(
            "Advanced commands"
        )
        self.robot_advanced_controls_group.add(send_group)

        layout.addWidget(self.robot_basic_controls_group)
        layout.addWidget(self.robot_advanced_controls_group)

        # Editors are mounted here, below the buttons that open them, and only
        # one is visible at a time (see _toggle_robot_editor). Inside the page
        # rather than above the tab widget, so opening one no longer covers the
        # other workspaces.
        for editor in (
            self.position_entry_widget,
            self.position_quaternion_widget,
            self.position_toolframe_widget,
            self.position_script_widget,
        ):
            self.robot_advanced_controls_group.add(editor)

        # Keep the groups at their natural height at the top of the page.
        layout.addStretch(1)

    def setup_tab3(self, layout):
        self.ai_sub_tabs = QTabWidget()
        self.ai_sub_tabs.setUsesScrollButtons(False)

        # ─── Subtab “AI Model” ───
        ai_model_page = QWidget()
        ai_model_layout = QVBoxLayout(ai_model_page)

        # Keep deterministic controls separate from learned-policy controls.
        rule_based_page = QWidget()
        rule_based_page_layout = QVBoxLayout(rule_based_page)

        self.predict_threelevel_hierarchical_transformer_gesture_button = QPushButton("Predict")
        self.btn_toggle_3lvl_latch = QPushButton("Latch off")
        self.proximity_control_button = QPushButton("Proximity Control")
        self.proximity_record_button = QPushButton("Record Proximity Data")

        self.proximity_settings_dialog = QDialog(self.widget_func)
        self.proximity_settings_dialog.setWindowTitle("Proximity Control Settings")
        self.proximity_settings_dialog.setModal(False)
        components.size_to_screen(self.proximity_settings_dialog, 820, 560)
        proximity_settings_layout = QVBoxLayout(self.proximity_settings_dialog)
        proximity_settings_layout.setContentsMargins(10, 10, 10, 10)
        proximity_settings_layout.setSpacing(8)
        self.proximity_settings_group, proximity_params_list = components.section(
            "Proximity control",
            "Tune centering, hover distance, smoothing, and lost-signal recovery.",
        )

        def _mk_dspin(default: float, minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
            sp = QDoubleSpinBox()
            sp.setRange(minimum, maximum)
            sp.setSingleStep(step)
            sp.setDecimals(3)
            sp.setValue(default)
            sp.setMinimumWidth(120)
            return sp

        def _add_param_row(row: int, name: str, description: str, widget):
            tooltip = f"{name}\n\n{description}"
            widget.setToolTip(tooltip)
            proximity_params_list.add(
                components.SettingsRow(name, widget, hint=description)
            )

        self.proximity_frame_interval_spin = QSpinBox()
        self.proximity_frame_interval_spin.setRange(10, 500)
        self.proximity_frame_interval_spin.setSingleStep(10)
        self.proximity_frame_interval_spin.setValue(20)
        self.proximity_frame_interval_spin.setMinimumWidth(120)
        _add_param_row(
            0,
            "Control Loop Interval (ms)",
            "How often Proximity Control updates the sensor reading and robot velocity. Lower values react faster.",
            self.proximity_frame_interval_spin,
        )

        self.proximity_lateral_speed_spin = _mk_dspin(0.2, 0.001, 0.500, 0.01)
        _add_param_row(
            1,
            "Planar Tracking Speed (m/s)",
            "Maximum planar response used to move the sensor back under the finger in left/right/up/down directions.",
            self.proximity_lateral_speed_spin,
        )

        self.proximity_normal_speed_spin = _mk_dspin(0.10, 0.001, 0.300, 0.005)
        _add_param_row(
            2,
            "Normal Distance Speed (m/s)",
            "Speed scale for moving toward/away from the finger after planar centering is stable.",
            self.proximity_normal_speed_spin,
        )

        self.proximity_centroid_deadband_spin = _mk_dspin(0.040, 0.0, 0.300, 0.01)
        _add_param_row(
            3,
            "Planar Center Deadband",
            "Normalized row/column error treated as centered. Larger values are steadier but less precise.",
            self.proximity_centroid_deadband_spin,
        )

        self.proximity_strength_deadband_spin = _mk_dspin(0.080, 0.0, 0.300, 0.01)
        _add_param_row(
            4,
            "Normal Strength Deadband",
            "Relative signal-strength error ignored for distance control. Larger values reduce normal-axis jitter.",
            self.proximity_strength_deadband_spin,
        )

        self.proximity_max_linear_speed_spin = _mk_dspin(0.25, 0.01, 0.500, 0.01)
        _add_param_row(
            5,
            "Maximum Linear Speed (m/s)",
            "Final safety clamp applied to each linear velocity axis before sending the robot command.",
            self.proximity_max_linear_speed_spin,
        )

        self.proximity_center_window_spin = QSpinBox()
        self.proximity_center_window_spin.setRange(1, 15)
        self.proximity_center_window_spin.setValue(3)
        self.proximity_center_window_spin.setMinimumWidth(120)
        _add_param_row(
            6,
            "Center Strength Window Size",
            "Sensor-center patch size used to estimate hover-distance signal strength for normal control.",
            self.proximity_center_window_spin,
        )
        self.proximity_smoothing_alpha_spin = _mk_dspin(0.75, 0.0, 1.0, 0.05)
        _add_param_row(
            7,
            "Signal Smoothing Alpha",
            "EMA smoothing for finger center and strength. Higher values follow fast motion more quickly; lower values are steadier.",
            self.proximity_smoothing_alpha_spin,
        )

        self.proximity_lost_signal_recovery_frames_spin = QSpinBox()
        self.proximity_lost_signal_recovery_frames_spin.setRange(0, 100)
        self.proximity_lost_signal_recovery_frames_spin.setValue(12)
        self.proximity_lost_signal_recovery_frames_spin.setMinimumWidth(120)
        _add_param_row(
            8,
            "Lost-Signal Normal Recovery Frames",
            "When the signal becomes too weak, keep moving along the normal direction for this many frames to reacquire a quickly pulled-away finger.",
            self.proximity_lost_signal_recovery_frames_spin,
        )

        self.proximity_lost_signal_speed_ratio_spin = _mk_dspin(1.0, 0.0, 3.0, 0.1)
        _add_param_row(
            9,
            "Lost-Signal Recovery Speed Ratio",
            "Multiplier for normal speed during short lost-signal recovery. Higher values chase faster but can overshoot.",
            self.proximity_lost_signal_speed_ratio_spin,
        )
        proximity_settings_layout.addWidget(self.proximity_settings_group)

        proximity_button_row = QWidget(self.proximity_settings_dialog)
        proximity_button_layout = QHBoxLayout(proximity_button_row)
        proximity_button_layout.setContentsMargins(0, 0, 0, 0)
        self.apply_proximity_settings_button = QPushButton("Apply Proximity Params")
        self.reload_proximity_settings_button = QPushButton("Reload Saved Params")
        self.close_proximity_settings_button = QPushButton("Close")
        proximity_button_layout.addWidget(self.apply_proximity_settings_button)
        proximity_button_layout.addWidget(self.reload_proximity_settings_button)
        proximity_button_layout.addStretch()
        proximity_button_layout.addWidget(self.close_proximity_settings_button)
        proximity_settings_layout.addWidget(proximity_button_row)

        frame_row = QWidget()
        frame_grid = QGridLayout(frame_row)
        frame_grid.setContentsMargins(0, 0, 0, 0)
        frame_grid.setHorizontalSpacing(8)
        frame_grid.setVerticalSpacing(6)
        self.ai_frame_buttons = {}
        self.ai_selected_frame = "tool"

        # Keep a hidden compatibility field so older logic that still reads
        # ai_frame_input continues to work. The visible control is now the
        # button selector below.
        self.ai_frame_input = QLineEdit(self.widget_func)
        self.ai_frame_input.setText(self.ai_selected_frame)
        self.ai_frame_input.hide()

        frame_button_order = [
            ("joint1", "Base"),
            ("joint2", "Joint2"),
            ("joint3", "Joint3"),
            ("joint4", "Joint4"),
            ("joint5", "Joint5"),
            ("tool", "Tool"),
        ]
        frame_button_positions = {
            "joint1": (0, 0),
            "joint2": (0, 1),
            "joint3": (0, 2),
            "joint4": (1, 0),
            "joint5": (1, 1),
            "tool": (1, 2),
        }
        for frame_key, frame_label in frame_button_order:
            btn = QPushButton(frame_label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, key=frame_key: self._set_ai_frame(key))
            self.ai_frame_buttons[frame_key] = btn
            row, col = frame_button_positions[frame_key]
            frame_grid.addWidget(btn, row, col)

        self.btn_toggle_anchor_axes = QPushButton("Axes: Anchored ON")  # label will be synced on init

        frame_grid.setColumnStretch(3, 1)
        self._update_ai_frame_buttons()

        # Which link the predicted velocity is expressed in, plus whether the
        # axes stay anchored. Set once per session, so it leads the page as a
        # compact reference rather than as a wall of buttons.
        frame_group, frame_list = components.section(
            "Reference frame",
            "Link the AI's velocity command is applied in.",
        )
        frame_list.add(components.SettingsRow("End effector", frame_row))

        tactile_ai_group, tactile_ai_list = components.section(
            "Tactile policy",
            "Learned control from the tactile signal.",
        )
        tactile_ai_advanced_group, tactile_ai_advanced_list = components.section(
            "Tactile policy setup",
            "Checkpoint selection and gesture-recognition tools.",
        )
        proximity_ai_group, proximity_ai_list = components.section(
            "Proximity",
            "Detects approach before contact and can retreat the arm.",
        )
        proximity_ai_advanced_group, proximity_ai_advanced_list = components.section(
            "Proximity setup",
            "Detection tuning and checkpoint selection.",
        )

        rule_based_group, rule_based_list = components.section(
            "Tactile motion",
            "Deterministic controllers driven straight from the sensor.",
        )
        console_group, console_list = components.section(
            "Controller input",
            "Drive the arm from a gamepad or from the sensor itself.",
        )
        tool_path_group, tool_path_list = components.section(
            "Tool path",
            "Record and review where the tool actually travelled.",
        )
        admittance_group, admittance_list = components.section(
            "Pressure admittance",
            "The arm yields to pressure on the sensor.",
        )
        self.admittance_control_button = QPushButton("Start Admittance Control")
        self.admittance_control_button.setCheckable(True)
        self.admittance_control_button.setToolTip(
            "Start pressure-based robot admittance using the selected sensor and "
            "its saved robot-link mapping. The 3D robot window is not required."
        )
        self.admittance_control_button.setText("Start")
        self.admittance_control_status_label = QLabel("Idle")
        self.admittance_control_status_label.setStyleSheet(
            theme.type_style("caption")
        )
        self.admittance_control_status_label.setWordWrap(True)
        admittance_list.add(
            components.SettingsRow(
                "Admittance",
                self.admittance_control_button,
                hint="Uses the selected sensor and its saved robot-link "
                     "mapping; the 3D robot window is not required.",
            )
        )
        admittance_list.add(
            components.SettingsRow("State", self.admittance_control_status_label)
        )

        self.direct_finger_motion_button = QPushButton("Direct Finger Motion")
        self.console_control_button = QPushButton("Console Control (PS5)")
        self.console_control_sensor_button = QPushButton("Console Control (Sensor)")
        self.console_control_sensor_v2_button = QPushButton("Console Control (Sensor V2)")
        self.direct_finger_motion_tool_pose_record_menu_button = QPushButton("Tool Pose Recording")
        self.load_tool_pose_path_button = QPushButton("Load Tool Pose Path")
        self.clear_tool_pose_path_button = QPushButton("Clear Tool Pose Path")
        self.ai_direct_finger_motion_button = QPushButton("AI DFM Record (No Robot)")
        self.ai_direct_finger_motion_robot_button = QPushButton("AI DFM Record + Robot")
        self.ai_direct_finger_motion_execution_button = QPushButton("AI Direct Finger Motion (Execute)")
        self.ai_proximity_detection_button = QPushButton("Detect")
        self.ai_proximity_detection_button.setCheckable(True)
        self.ai_proximity_admittance_button = QPushButton("Retreat on approach")
        self.ai_proximity_admittance_button.setCheckable(True)
        self.ai_proximity_admittance_button.setChecked(False)
        self.ai_proximity_admittance_button.setEnabled(False)
        self.ai_proximity_admittance_button.setToolTip(
            "Arm robot retreat and return-to-start motion after AI Proximity "
            "Detection is running."
        )
        self.ai_proximity_detection_mode_combo = QComboBox()
        self.ai_proximity_detection_mode_combo.addItem("Hybrid", "hybrid")
        self.ai_proximity_detection_mode_combo.addItem(
            "CNN-GRU Only",
            "cnn_gru",
        )
        self.ai_proximity_detection_mode_combo.addItem(
            "Localized Only",
            "localized",
        )
        self.ai_proximity_detection_mode_combo.setCurrentIndex(0)
        self.ai_proximity_detection_mode_combo.setToolTip(
            "Choose whether detection uses the CNN-GRU, the localized "
            "statistical detector, or both."
        )
        self.ai_proximity_sensitivity_combo = QComboBox()
        self.ai_proximity_sensitivity_combo.addItem("Robust", 1.0)
        self.ai_proximity_sensitivity_combo.addItem("Sensitive", 0.85)
        self.ai_proximity_sensitivity_combo.addItem("Very sensitive", 0.70)
        self.ai_proximity_sensitivity_combo.setCurrentIndex(1)
        self.ai_proximity_sensitivity_combo.setToolTip(
            "Lower detection thresholds respond sooner to weak local changes "
            "but can produce more false detections."
        )
        self.ai_proximity_detection_status = QLabel("Proximity AI: idle")
        self.ai_proximity_detection_status.setWordWrap(True)
        self.ai_proximity_detection_status.setStyleSheet(
            theme.MUTED_LABEL_STYLE
        )
        self.ai_proximity_model_status = QLabel(
            "Automatic, from sensor size"
        )
        self.ai_proximity_model_status.setWordWrap(True)
        self.ai_proximity_model_status.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        self.ai_proximity_model_status.setStyleSheet(
            theme.MUTED_LABEL_STYLE
        )
        self.ai_proximity_model_status.setToolTip(
            "The size-specific proximity checkpoint selected when detection "
            "starts."
        )
        self.ai_proximity_select_model_button = QPushButton("Select Model")
        self.ai_proximity_select_model_button.setToolTip(
            "Manually choose the proximity checkpoint used the next time "
            "detection starts."
        )
        self.ai_proximity_use_auto_model_button = QPushButton("Use Auto")
        self.ai_proximity_use_auto_model_button.setEnabled(False)
        self.ai_proximity_use_auto_model_button.setToolTip(
            "Return to automatic model selection based on sensor size."
        )

        model_row = QWidget()
        model_row_layout = QHBoxLayout(model_row)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        self.ai_direct_execution_model_path_input = QLineEdit(model_row)
        self.ai_direct_execution_model_path_input.setPlaceholderText(
            DisabledSensorFunctions.DEFAULT_AI_DIRECT_EXECUTION_MODEL_PATH
        )
        default_ai_model_path = self._get_default_ai_execution_model_path()
        self.ai_direct_execution_model_path_input.setText(default_ai_model_path)
        self.ai_direct_execution_model_path_input.setVisible(False)
        # A checkpoint filename can be long. Let the label give up width and
        # elide rather than push the buttons beside it off the panel edge.
        self.ai_direct_execution_model_status = components.ElidedLabel()
        self.ai_direct_execution_select_model_button = QPushButton(
            "Select Model"
        )
        self.ai_direct_execution_select_model_button.setToolTip(
            "Choose the PyTorch checkpoint used the next time AI Direct "
            "Finger Motion execution starts."
        )
        self.ai_direct_execution_use_default_button = QPushButton(
            "Use Default"
        )
        self.ai_direct_execution_use_default_button.setToolTip(
            "Restore the current latest_cnn_gru_model_10x10.pt checkpoint."
        )
        self.ai_direct_execution_model_status.setStyleSheet(
            theme.type_style("caption")
        )
        model_row_layout.addWidget(
            self.ai_direct_execution_model_status,
            1,
        )
        model_row_layout.addWidget(
            self.ai_direct_execution_select_model_button
        )
        model_row_layout.addWidget(
            self.ai_direct_execution_use_default_button
        )
        self._set_ai_direct_execution_model_path(
            default_ai_model_path,
            is_default=True,
        )

        execute_safety_row = QWidget()
        execute_safety_layout = QHBoxLayout(execute_safety_row)
        execute_safety_layout.setContentsMargins(0, 0, 0, 0)
        execute_safety_layout.setSpacing(8)
        self.ai_direct_execution_dry_run_checkbox = QCheckBox("Dry run")
        self.ai_direct_execution_dry_run_checkbox.setChecked(False)
        self.ai_direct_execution_dry_run_checkbox.setToolTip(
            "When checked, AI predicts live velocity but does not send robot motion commands."
        )
        self.ai_direct_execution_prediction_status = QLabel("Prediction: idle")
        self.ai_direct_execution_prediction_status.setStyleSheet(
            theme.type_style("caption")
        )
        execute_safety_layout.addWidget(self.ai_direct_execution_dry_run_checkbox)
        execute_safety_layout.addStretch()
        execute_safety_layout.addWidget(self.ai_direct_execution_prediction_status)

        execute_action_row = QWidget()
        execute_action_layout = QHBoxLayout(execute_action_row)
        execute_action_layout.setContentsMargins(0, 0, 0, 0)
        execute_action_layout.setSpacing(6)
        self.ai_direct_finger_motion_execution_button.setText("Execute")
        self.ai_direct_finger_motion_execution_button.setToolTip(
            "Run the learned tactile policy and command robot motion."
        )
        execute_action_layout.addWidget(
            self.ai_direct_finger_motion_execution_button,
            1,
        )
        self.ai_direct_execution_velocity_scale_spin = QDoubleSpinBox()
        self.ai_direct_execution_velocity_scale_spin.setRange(0.10, 20.00)
        self.ai_direct_execution_velocity_scale_spin.setDecimals(2)
        self.ai_direct_execution_velocity_scale_spin.setSingleStep(0.10)
        self.ai_direct_execution_velocity_scale_spin.setValue(1.00)
        self.ai_direct_execution_velocity_scale_spin.setSuffix("x")
        self.ai_direct_execution_velocity_scale_spin.setToolTip(
            "Multiply the AI-predicted velocity. The execution safety limit "
            "still caps the total XYZ linear-speed magnitude at 0.30 m/s."
        )
        execute_action_layout.addWidget(
            self.ai_direct_execution_velocity_scale_spin
        )
        threelevel_row = QWidget()
        threelevel_row_layout = QHBoxLayout(threelevel_row)
        threelevel_row_layout.setContentsMargins(0, 0, 0, 0)
        threelevel_row_layout.setSpacing(6)
        self.predict_threelevel_hierarchical_transformer_gesture_button.setMinimumWidth(0)
        self.btn_toggle_3lvl_latch.setMinimumWidth(0)
        threelevel_row_layout.addWidget(
            self.predict_threelevel_hierarchical_transformer_gesture_button,
            1,
        )
        threelevel_row_layout.addWidget(self.btn_toggle_3lvl_latch, 1)
        proximity_row = QWidget()
        proximity_row_layout = QHBoxLayout(proximity_row)
        proximity_row_layout.setContentsMargins(0, 0, 0, 0)
        proximity_row_layout.setSpacing(6)
        self.proximity_control_button.setMinimumWidth(0)
        self.proximity_record_button.setMinimumWidth(0)
        self.proximity_control_button.setText("Follow")
        self.proximity_record_button.setText("Record data")
        proximity_row_layout.addWidget(self.proximity_control_button, 1)
        proximity_row_layout.addWidget(self.proximity_record_button, 1)
        rule_based_list.add(
            components.SettingsRow(
                "Proximity",
                proximity_row,
                hint="Keeps the finger centred over the sensor at the taught "
                     "hover distance.",
            )
        )
        self.direct_finger_motion_button.setText("Start")
        rule_based_list.add(
            components.SettingsRow(
                "Direct finger motion",
                self.direct_finger_motion_button,
                hint="Moves the arm from the tracked contact centroid.",
            )
        )

        console_row = QWidget()
        console_row_layout = QHBoxLayout(console_row)
        console_row_layout.setContentsMargins(0, 0, 0, 0)
        console_row_layout.setSpacing(6)
        # The shared "Console Control" prefix moves into the row label:
        # repeating it on all three buttons made this the widest row in the app.
        for button, short_text in (
            (self.console_control_button, "PS5"),
            (self.console_control_sensor_button, "Sensor"),
            (self.console_control_sensor_v2_button, "Sensor V2"),
        ):
            button.setToolTip(button.text())
            button.setText(short_text)
            button.setMinimumWidth(0)
            console_row_layout.addWidget(button, 1)
        console_list.add_stacked("Source", console_row)

        tool_pose_row = QWidget()
        tool_pose_row_layout = QHBoxLayout(tool_pose_row)
        tool_pose_row_layout.setContentsMargins(0, 0, 0, 0)
        tool_pose_row_layout.setSpacing(6)
        self.direct_finger_motion_tool_pose_record_menu_button.setText("Record")
        self.load_tool_pose_path_button.setText("Load")
        self.clear_tool_pose_path_button.setText("Clear")
        components.apply_variant(self.clear_tool_pose_path_button, "danger")
        self.direct_finger_motion_tool_pose_record_menu_button.setMinimumWidth(0)
        self.load_tool_pose_path_button.setMinimumWidth(0)
        self.clear_tool_pose_path_button.setMinimumWidth(0)
        tool_pose_row_layout.addWidget(self.direct_finger_motion_tool_pose_record_menu_button, 1)
        tool_pose_row_layout.addWidget(self.load_tool_pose_path_button, 1)
        tool_pose_row_layout.addWidget(self.clear_tool_pose_path_button, 1)
        tool_path_list.add_stacked("Path", tool_pose_row)
        self._build_direct_finger_motion_settings_dialog()
        self._build_console_control_settings_dialog()
        ai_proximity_row = QWidget()
        ai_proximity_row_layout = QHBoxLayout(ai_proximity_row)
        ai_proximity_row_layout.setContentsMargins(0, 0, 0, 0)
        ai_proximity_row_layout.setSpacing(6)
        ai_proximity_row_layout.addWidget(
            self.ai_proximity_detection_button,
            1,
        )
        ai_proximity_row_layout.addWidget(
            self.ai_proximity_admittance_button,
            1,
        )
        ai_proximity_options_row = QWidget()
        ai_proximity_options_layout = QHBoxLayout(
            ai_proximity_options_row
        )
        ai_proximity_options_layout.setContentsMargins(0, 0, 0, 0)
        ai_proximity_options_layout.setSpacing(6)
        ai_proximity_options_layout.addWidget(QLabel("Mode:"))
        ai_proximity_options_layout.addWidget(
            self.ai_proximity_detection_mode_combo
        )
        ai_proximity_options_layout.addWidget(QLabel("Sensitivity:"))
        ai_proximity_options_layout.addWidget(
            self.ai_proximity_sensitivity_combo
        )
        ai_proximity_options_layout.addStretch()
        ai_proximity_model_row = QWidget()
        ai_proximity_model_layout = QHBoxLayout(ai_proximity_model_row)
        ai_proximity_model_layout.setContentsMargins(0, 0, 0, 0)
        ai_proximity_model_layout.setSpacing(6)
        ai_proximity_model_layout.addWidget(
            self.ai_proximity_model_status,
            1,
        )
        ai_proximity_model_layout.addWidget(
            self.ai_proximity_select_model_button
        )
        ai_proximity_model_layout.addWidget(
            self.ai_proximity_use_auto_model_button
        )
        # ── Tactile policy section ──────────────────────────────────────
        # Execution first: it is the action you came here for. Its safety cap
        # rides along as the row's hint rather than as a separate grey line.
        execute_row = components.SettingsRow(
            "Direct finger motion",
            execute_action_row,
            hint="Safety cap: 0.30 m/s total linear speed",
        )
        tactile_ai_list.add(execute_row)
        tactile_ai_advanced_list.add(
            components.SettingsRow("Checkpoint", model_row)
        )
        tactile_ai_list.add(execute_safety_row)
        tactile_ai_advanced_list.add(
            components.SettingsRow("Gesture recognition", threelevel_row)
        )

        # ── Proximity section ───────────────────────────────────────────
        proximity_ai_list.add(
            components.SettingsRow("Detection", ai_proximity_row)
        )
        proximity_ai_advanced_list.add_stacked("Tuning", ai_proximity_options_row)
        proximity_ai_advanced_list.add(
            components.SettingsRow("Checkpoint", ai_proximity_model_row)
        )
        proximity_ai_list.add(
            components.SettingsRow("State", self.ai_proximity_detection_status)
        )

        self.ai_basic_controls_group = components.CollapsibleGroup(
            "Basic controls",
            expanded=True,
        )
        self.ai_basic_controls_group.add(tactile_ai_group)
        self.ai_basic_controls_group.add(proximity_ai_group)
        self.ai_advanced_controls_group = components.CollapsibleGroup(
            "Advanced setup"
        )
        self.ai_advanced_controls_group.add(frame_group)
        self.ai_advanced_controls_group.add(tactile_ai_advanced_group)
        self.ai_advanced_controls_group.add(proximity_ai_advanced_group)
        ai_model_layout.addWidget(self.ai_basic_controls_group)
        ai_model_layout.addWidget(self.ai_advanced_controls_group)
        ai_model_layout.addStretch()
        rule_based_page_layout.setSpacing(theme.SPACE_LG)
        rule_based_page_layout.addWidget(rule_based_group)
        rule_based_page_layout.addWidget(console_group)
        rule_based_page_layout.addWidget(tool_path_group)
        rule_based_page_layout.addWidget(admittance_group)
        rule_based_page_layout.addStretch()

        # ─── Subtab "Data" ───
        training_page = QWidget()
        training_layout = QVBoxLayout(training_page)
        training_layout.setSpacing(theme.SPACE_LG)

        record_group, record_list = components.section(
            "Record a dataset",
            "Capture tactile trials for training the policy.",
        )

        # One selection instead of four "Set ... Mode" buttons: these were
        # mutually exclusive settings of a single value, so a button each gave
        # no way to see which one was in force.
        trigger_mode_row = QWidget()
        trigger_mode_layout = QHBoxLayout(trigger_mode_row)
        trigger_mode_layout.setContentsMargins(0, 0, 0, 0)
        trigger_mode_layout.setSpacing(6)
        self.record_trigger_mode_combo = QComboBox()
        for label, mode, hint in (
            (
                "Manual start/stop",
                "no_trigger",
                "Record only while you hold the Record button.",
            ),
            (
                "Auto trials, recalibrate each",
                "no_trigger_auto",
                "Record repeated trials automatically, recalibrating the "
                "sensor between each one.",
            ),
            (
                "Auto trials, keep calibration",
                "no_trigger_no_updatecal_auto",
                "Record repeated trials automatically without recalibrating "
                "between them, so a slow drift is preserved in the data.",
            ),
            (
                "Start on touch",
                "trigger",
                "Begin each trial automatically when contact is detected.",
            ),
        ):
            self.record_trigger_mode_combo.addItem(label, mode)
            index = self.record_trigger_mode_combo.count() - 1
            self.record_trigger_mode_combo.setItemData(
                index, hint, Qt.ToolTipRole
            )
        self.record_trigger_mode_combo.setCurrentIndex(0)
        trigger_mode_layout.addWidget(self.record_trigger_mode_combo, 1)
        record_list.add(components.SettingsRow("Mode", trigger_mode_row))

        self.ai_dfm_session_input = QLineEdit()
        self.ai_dfm_session_input.setPlaceholderText(
            "Auto: ai_dfm_10x10_v1"
        )
        self.ai_dfm_session_input.setToolTip(
            "Optional custom dataset session. Leave blank to group trials "
            "automatically by sensor size."
        )
        record_list.add(
            components.SettingsRow(
                "Session",
                self.ai_dfm_session_input,
                hint="Leave blank to group trials by sensor size.",
            )
        )

        ai_dfm_record_row = QWidget()
        ai_dfm_record_row_layout = QHBoxLayout(ai_dfm_record_row)
        ai_dfm_record_row_layout.setContentsMargins(0, 0, 0, 0)
        ai_dfm_record_row_layout.setSpacing(6)
        self.ai_direct_finger_motion_button.setText("Without robot")
        self.ai_direct_finger_motion_robot_button.setText("With robot")
        self.ai_direct_finger_motion_button.setMinimumWidth(0)
        self.ai_direct_finger_motion_robot_button.setMinimumWidth(0)
        ai_dfm_record_row_layout.addWidget(self.ai_direct_finger_motion_button, 1)
        ai_dfm_record_row_layout.addWidget(self.ai_direct_finger_motion_robot_button, 1)
        record_list.add(
            components.SettingsRow(
                "Record trial",
                ai_dfm_record_row,
                hint="Space starts and stops recording on this page.",
            )
        )

        self.ai_proximity_environment_record_button = QPushButton(
            f"Record {self.AI_PROXIMITY_ENVIRONMENT_TRIAL_COUNT} trials"
        )
        self.ai_proximity_environment_record_button.setCheckable(True)
        self.ai_proximity_environment_record_button.setToolTip(
            "Automatically record ten separate one-minute normal-environment "
            "trials without sending robot commands."
        )
        self.ai_proximity_environment_record_status = QLabel(
            "Ready | Session will match the active sensor dimensions"
        )
        self.ai_proximity_environment_record_status.setWordWrap(True)
        self.ai_proximity_environment_record_status.setStyleSheet(
            theme.type_style("caption")
        )
        record_list.add(
            components.SettingsRow(
                "Proximity background",
                self.ai_proximity_environment_record_button,
                hint="Ten one-minute trials of the empty environment.",
            )
        )
        record_list.add(
            components.SettingsRow(
                "State", self.ai_proximity_environment_record_status
            )
        )

        # Legacy path kept but demoted behind a disclosure: it writes the old
        # gesture-number dataset that only the original Record button reads.
        legacy_group = components.CollapsibleGroup("Legacy gesture recording")
        legacy_list = components.InsetList()
        self.gesture_number_input = QLineEdit()
        self.gesture_number_input.setToolTip(
            "Label used only by the legacy Record button."
        )
        self.record_gesture_button = QPushButton("Record")
        legacy_list.add(
            components.SettingsRow("Gesture label", self.gesture_number_input)
        )
        legacy_list.add(
            components.SettingsRow("Capture", self.record_gesture_button)
        )
        legacy_group.add(legacy_list)
        record_group.layout().addWidget(legacy_group)

        # ── Teaching label ─────────────────────────────────────────────
        # Sixteen momentary buttons in one flat grid gave no clue that they
        # fall into three kinds, so they are grouped and the axis pairs are
        # laid out to read as pairs.
        self.ai_teaching_label_group, teaching_list = components.section(
            "Teaching label",
            "Tag what the demonstration is doing while it records.",
        )

        self.ai_teaching_label_buttons = {}
        self.ai_teaching_label_status = QLabel("Auto/DFM")
        self.ai_teaching_label_status.setStyleSheet(theme.type_style("caption"))

        # Each axis gets a stable row. This costs a little height, but is much
        # easier to scan and hit during a projected or touch-led session than
        # six equally weighted buttons in one line.
        teaching_rows = [
            (
                "Behaviour",
                [
                    ("auto", "Auto/DFM"),
                    ("stop", "Stop"),
                    ("normal_swipe", "Swipe"),
                    ("push", "Push"),
                    ("pull", "Pull"),
                ],
            ),
            (
                "Translation X",
                [
                    ("x_pos", "X+"), ("x_neg", "X−"),
                ],
            ),
            (
                "Translation Y",
                [
                    ("y_pos", "Y+"), ("y_neg", "Y−"),
                ],
            ),
            (
                "Translation Z",
                [
                    ("z_pos", "Z+"), ("z_neg", "Z−"),
                ],
            ),
            (
                "Rotation RX",
                [
                    ("rx_pos", "RX+"), ("rx_neg", "RX−"),
                ],
            ),
            (
                "Rotation RY",
                [
                    ("ry_pos", "RY+"), ("ry_neg", "RY−"),
                ],
            ),
            (
                "Rotation RZ",
                [
                    ("rz_pos", "RZ+"), ("rz_neg", "RZ−"),
                ],
            ),
        ]
        for row_label, specs in teaching_rows:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(theme.SPACE_XS)
            for label_key, label_text in specs:
                button = QPushButton(label_text)
                button.setCheckable(True)
                button.setMinimumWidth(0)
                self.ai_teaching_label_buttons[label_key] = button
                row_layout.addWidget(button, 1)
            teaching_list.add_stacked(row_label, row_widget)
        teaching_list.add(
            components.SettingsRow("Current", self.ai_teaching_label_status)
        )

        self.ai_data_basic_group = components.CollapsibleGroup(
            "Basic recording",
            expanded=True,
        )
        self.ai_data_basic_group.add(record_group)
        self.ai_data_advanced_group = components.CollapsibleGroup(
            "Advanced labels"
        )
        self.ai_data_advanced_group.add(self.ai_teaching_label_group)
        training_layout.addWidget(self.ai_data_basic_group)
        training_layout.addWidget(self.ai_data_advanced_group)
        training_layout.addStretch()

        self.ai_run_tab_index = self.ai_sub_tabs.addTab(
            self._scrollable(ai_model_page), "Run"
        )
        self.ai_data_training_tab_index = self.ai_sub_tabs.addTab(
            self._scrollable(training_page),
            "Data",
        )
        layout.addWidget(self.ai_sub_tabs)

        # The deterministic controllers live in their own Control workspace,
        # but they share dialogs and status labels with the learned-policy
        # controls above, so the page is built here and handed to setup_tabs.
        self.control_page = rule_based_page

    def setup_tab4(self, layout):
        layout.setSpacing(theme.SPACE_LG)

        # --- HP-200 Force Meter ---
        force_meter_group, force_meter_list = components.section(
            "Force meter",
            "HP-200 load cell, read over serial.",
        )
        self.force_meter_port_combo = QComboBox()
        self.force_meter_port_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLength
        )
        # Short enough that a verbose USB product string cannot widen the whole
        # control panel; the full text stays available as the item's tooltip.
        self.force_meter_port_combo.setMinimumContentsLength(18)
        self.force_meter_refresh_button = components.IconButton(
            "refresh", "Rescan serial ports"
        )
        port_row = QWidget()
        port_layout = QHBoxLayout(port_row)
        port_layout.setContentsMargins(0, 0, 0, 0)
        port_layout.setSpacing(theme.SPACE_SM)
        port_layout.addWidget(self.force_meter_port_combo, 1)
        port_layout.addWidget(self.force_meter_refresh_button)

        self.force_meter_connect_button = QPushButton("Connect")
        self.force_meter_connect_button.setCheckable(True)
        connect_row = QWidget()
        connect_layout = QHBoxLayout(connect_row)
        connect_layout.setContentsMargins(0, 0, 0, 0)
        connect_layout.setSpacing(theme.SPACE_SM)
        connect_layout.addWidget(self.force_meter_connect_button)

        self.force_meter_protocol_combo = QComboBox()
        self.force_meter_protocol_combo.addItem(
            "HP-200 Modbus RTU (official)", "modbus_rtu"
        )
        self.force_meter_protocol_combo.addItem(
            "Legacy text stream", "text_stream"
        )
        self.force_meter_baud_combo = QComboBox()
        for baud_rate in (9600, 19200, 38400, 115200, 4800, 2400):
            self.force_meter_baud_combo.addItem(str(baud_rate), baud_rate)

        # The big number first -- that is what you look at -- then the port and
        # the connect action. Protocol and baud rate are set once per meter, so
        # they go behind a disclosure.
        self.force_meter_value_label = QLabel("+0.0000 N")
        self.force_meter_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.force_meter_value_label.setStyleSheet(theme.readout_style("title"))
        self.force_meter_native_label = QLabel("No sample")
        self.force_meter_native_label.setStyleSheet(
            theme.type_style("caption", color=theme.INFO)
        )
        self.force_meter_stats_label = QLabel("Min -- N   Max -- N   Peak |F| -- N")
        self.force_meter_stats_label.setStyleSheet(theme.readout_style("caption"))

        force_meter_list.add(
            components.SettingsRow("Reading", self.force_meter_value_label)
        )
        force_meter_list.add(
            components.SettingsRow("Range", self.force_meter_stats_label)
        )
        force_meter_list.add(components.SettingsRow("Port", port_row))
        force_meter_list.add(
            components.SettingsRow("Connection", connect_row)
        )
        force_meter_list.add(
            components.SettingsRow("Meter", self.force_meter_native_label)
        )

        force_meter_link_group = components.CollapsibleGroup("Link settings")
        link_list = components.InsetList()
        link_list.add(
            components.SettingsRow("Protocol", self.force_meter_protocol_combo)
        )
        link_list.add(
            components.SettingsRow("Baud rate", self.force_meter_baud_combo)
        )
        force_meter_link_group.add(link_list)
        force_meter_group.layout().addWidget(force_meter_link_group)

        # --- Live force graph ---
        graph_group, graph_list = components.section("Force over time")
        graph_header = graph_group.layout().itemAt(0).widget()
        self.force_meter_clear_graph_button = components.IconButton(
            "trash", "Clear the graph"
        )
        graph_header.add_action(self.force_meter_clear_graph_button)

        self.force_meter_chart = ForceMeterChartWidget(time_window_seconds=30.0)
        graph_list.add(self.force_meter_chart)

        self.force_meter_graph_window_combo = QComboBox()
        for window_seconds in (10, 30, 60, 120):
            self.force_meter_graph_window_combo.addItem(
                f"{window_seconds} s", window_seconds
            )
        self.force_meter_graph_window_combo.setCurrentIndex(1)
        graph_list.add(
            components.SettingsRow("Window", self.force_meter_graph_window_combo)
        )

        force_controls = QWidget()
        force_controls_layout = QHBoxLayout(force_controls)
        force_controls_layout.setContentsMargins(0, 0, 0, 0)
        force_controls_layout.setSpacing(theme.SPACE_SM)
        self.force_meter_zero_button = QPushButton("Zero")
        self.force_meter_zero_button.setEnabled(False)
        self.force_meter_clear_zero_button = QPushButton("Clear zero")
        self.force_meter_clear_zero_button.setEnabled(False)
        self.force_meter_reset_stats_button = QPushButton("Reset range")
        force_controls_layout.addWidget(self.force_meter_zero_button)
        force_controls_layout.addWidget(self.force_meter_clear_zero_button)
        force_controls_layout.addWidget(self.force_meter_reset_stats_button)
        graph_list.add_stacked("Calibrate", force_controls)

        self.force_meter_status_label = QLabel("Disconnected")
        self.force_meter_status_label.setWordWrap(True)
        self.force_meter_status_label.setStyleSheet(theme.type_style("caption"))
        force_meter_list.add(
            components.SettingsRow("State", self.force_meter_status_label)
        )

        layout.addWidget(force_meter_group)
        layout.addWidget(graph_group)

        self._refresh_force_meter_ports()

        # --- Gripper Manual Control ---
        gripper_group, gripper_list = components.section(
            "Gripper",
            "Two-finger gripper on the arm's tool flange.",
        )

        slider_row = QWidget()
        slider_layout = QHBoxLayout(slider_row)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(theme.SPACE_SM)
        self.gripper_slider = QSlider(Qt.Horizontal)
        self.gripper_slider.setRange(0, 100)
        self.gripper_slider.setValue(0)
        self.gripper_slider.setTickPosition(QSlider.TicksBelow)
        self.gripper_slider.setTickInterval(10)
        self.gripper_slider.setMinimumWidth(140)

        self.gripper_label = QLabel("0.00 (Open)")
        self.gripper_label.setFixedWidth(88)
        self.gripper_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.gripper_label.setStyleSheet(theme.readout_style())

        slider_layout.addWidget(self.gripper_slider, 1)
        slider_layout.addWidget(self.gripper_label)
        gripper_list.add(
            components.SettingsRow(
                "Aperture", slider_row, hint="0 is fully open."
            )
        )

        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(theme.SPACE_SM)
        self.btn_grip_open = QPushButton("Open")
        self.btn_grip_close = QPushButton("Close")
        btn_layout.addWidget(self.btn_grip_open)
        btn_layout.addWidget(self.btn_grip_close)
        gripper_list.add(components.SettingsRow("Fully", btn_row))
        layout.addWidget(gripper_group)

        # Connections
        self.gripper_slider.valueChanged.connect(self._update_slider_label)
        self.gripper_slider.sliderReleased.connect(self._on_slider_released)
        self.btn_grip_open.clicked.connect(lambda: self.set_gripper_manual(0))
        self.btn_grip_close.clicked.connect(lambda: self.set_gripper_manual(100))

        # --- AI Camera Section ---
        camera_group, camera_list = components.section(
            "Camera",
            "YOLO object detection from the wrist camera.",
        )
        self.live_yolo_button = QPushButton("Start")
        self.live_yolo_button.clicked.connect(self.toggle_yolo_camera)

        self.auto_center_button = QPushButton("Enable")
        self.auto_center_button.setCheckable(True)
        self.auto_center_button.setEnabled(False)
        self.auto_center_button.clicked.connect(self.toggle_centering_mode)

        camera_list.add(
            components.SettingsRow("Object detection", self.live_yolo_button)
        )
        camera_list.add(
            components.SettingsRow(
                "Auto-centre",
                self.auto_center_button,
                hint="Move the arm to keep the detected object centred.",
            )
        )
        layout.addWidget(camera_group)
        layout.addStretch()

    def setup_tab5(self, layout):
        self.hand_sub_tabs = QTabWidget()
        self.hand_sub_tabs.setUsesScrollButtons(False)

        hand_monitor_page = QWidget()
        hand_monitor_layout = QVBoxLayout(hand_monitor_page)
        hand_monitor_layout.setContentsMargins(0, 0, 0, 0)
        hand_monitor_layout.setSpacing(8)

        hand_motion_page = QWidget()
        hand_motion_layout = QVBoxLayout(hand_motion_page)
        hand_motion_layout.setContentsMargins(0, 0, 0, 0)
        hand_motion_layout.setSpacing(8)

        hand_group, hand_list = components.section(
            "Hand",
            "RH56F1 connection and the 3D model used for visualization.",
        )
        self.hand_state_label = QLabel("Idle")
        self.hand_state_label.setWordWrap(True)
        self.hand_state_label.setStyleSheet(theme.type_style("caption"))
        self.hand_connection_label = QLabel(
            "Offline · start the RH56F1 ROS nodes"
        )
        self.hand_connection_label.setWordWrap(True)
        self.hand_connection_label.setStyleSheet(theme.type_style("caption"))
        hand_list.add(
            components.SettingsRow("Connection", self.hand_connection_label)
        )
        hand_list.add(
            components.SettingsRow("Activity", self.hand_state_label)
        )

        self.hand_model_show_button = QPushButton("Import")
        # Path length is not ours to control, so it elides rather than forcing
        # the panel wider.
        self.hand_model_status_label = components.ElidedLabel(
            "resource/dexterous_hand"
        )
        self.hand_model_status_label.setStyleSheet(theme.type_style("caption"))
        model_row = components.SettingsRow(
            "3D model", self.hand_model_show_button
        )
        model_row.set_hint("resource/dexterous_hand")
        hand_list.add(model_row)

        # Live-tactile controls belong in the section header, beside the title
        # they act on, rather than as another row of full-width buttons.
        tactile_group, tactile_list = components.section(
            "Tactile readout",
            "Per-region normal and tangential force from the hand's fingertips.",
        )
        tactile_header = tactile_group.layout().itemAt(0).widget()
        self.hand_tactile_live_button = QPushButton("Start Live")
        self.hand_tactile_live_button.setCheckable(True)
        self.hand_tactile_live_button.setToolTip(
            "Stream /touch_data continuously into the table below."
        )
        tactile_header.add_action(self.hand_tactile_live_button)
        self.hand_tactile_refresh_button = components.IconButton(
            "refresh", "Read /touch_data once and update the table"
        )
        tactile_header.add_action(self.hand_tactile_refresh_button)

        self.hand_tactile_status_label = QLabel("No tactile data")
        self.hand_tactile_status_label.setStyleSheet(theme.type_style("caption"))

        self.hand_tactile_table = QTableWidget(8, 5)
        self.hand_tactile_table.setHorizontalHeaderLabels(
            ["Region", "Normal (N)", "Tangential (N)", "Direction", "Proximity"]
        )
        self.hand_tactile_table.verticalHeader().hide()
        self.hand_tactile_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.hand_tactile_table.setSelectionMode(QTableWidget.NoSelection)
        header = self.hand_tactile_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for col in range(1, 5):
            header.setSectionResizeMode(col, QHeaderView.Stretch)

        self._hand_tactile_region_names = [
            "Little finger",
            "Ring finger",
            "Middle finger",
            "Index finger",
            "Thumb",
            "Palm left",
            "Palm middle",
            "Palm right",
        ]
        for row, name in enumerate(self._hand_tactile_region_names):
            self.hand_tactile_table.setItem(row, 0, QTableWidgetItem(name))
            for col in range(1, 5):
                item = QTableWidgetItem("--")
                item.setTextAlignment(Qt.AlignCenter)
                self.hand_tactile_table.setItem(row, col, item)
        components.fit_table_height(
            self.hand_tactile_table, len(self._hand_tactile_region_names)
        )

        # A grid of "--" reads as broken hardware. When nothing is publishing,
        # say so instead.
        self.hand_tactile_stack = components.EmptyStateStack(
            self.hand_tactile_table,
            components.EmptyState(
                "No tactile data",
                glyph="hand",
                hint="Start the hand's tactile node, then press Start Live.",
            ),
        )
        tactile_list.add(self.hand_tactile_stack)
        # Hidden while the empty state is up, which already carries the reason;
        # once frames flow this reports the topic state instead.
        self.hand_tactile_stream_row = tactile_list.add(
            components.SettingsRow("Stream", self.hand_tactile_status_label)
        )
        self.hand_tactile_stream_row.setVisible(False)

        hand_monitor_layout.addWidget(hand_group)
        hand_monitor_layout.addWidget(tactile_group)
        hand_monitor_layout.addStretch()

        self.hand_tactile_timer = QTimer(self)
        self.hand_tactile_timer.setInterval(100)
        self.hand_tactile_timer.timeout.connect(self._refresh_hand_tactile_display)

        # Three sections instead of five bordered groups: what the whole hand
        # does, where the individual fingers go, and saved poses.
        motion_group, motion_list = components.section(
            "Whole hand",
            "Applies to all six actuators at once.",
        )

        speed_row_widget = QWidget()
        speed_row = QHBoxLayout(speed_row_widget)
        speed_row.setContentsMargins(0, 0, 0, 0)
        speed_row.setSpacing(theme.SPACE_SM)
        self.hand_speed_spin = QSpinBox()
        self.hand_speed_spin.setRange(0, 3000)
        self.hand_speed_spin.setValue(300)
        self.hand_apply_speed_button = QPushButton("Apply")
        speed_row.addWidget(self.hand_speed_spin)
        speed_row.addWidget(self.hand_apply_speed_button)
        motion_list.add(components.SettingsRow("Speed", speed_row_widget))

        force_row_widget = QWidget()
        force_row = QHBoxLayout(force_row_widget)
        force_row.setContentsMargins(0, 0, 0, 0)
        force_row.setSpacing(theme.SPACE_SM)
        self.hand_force_spin = QSpinBox()
        self.hand_force_spin.setRange(0, 12000)
        self.hand_force_spin.setValue(2000)
        self.hand_apply_force_button = QPushButton("Apply")
        force_row.addWidget(self.hand_force_spin)
        force_row.addWidget(self.hand_apply_force_button)
        motion_list.add(components.SettingsRow("Force", force_row_widget))

        grip_row_widget = QWidget()
        grip_row = QHBoxLayout(grip_row_widget)
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.setSpacing(theme.SPACE_SM)
        self.hand_open_all_button = QPushButton("Open")
        self.hand_close_all_button = QPushButton("Close")
        self.hand_read_angles_button = QPushButton("Read angles")
        grip_row.addWidget(self.hand_open_all_button)
        grip_row.addWidget(self.hand_close_all_button)
        grip_row.addWidget(self.hand_read_angles_button)
        motion_list.add(components.SettingsRow("Grip", grip_row_widget))

        thumb_row_widget = QWidget()
        thumb_row = QHBoxLayout(thumb_row_widget)
        thumb_row.setContentsMargins(0, 0, 0, 0)
        thumb_row.setSpacing(theme.SPACE_SM)
        self.hand_thumb_left_button = QPushButton("Left")
        self.hand_thumb_center_button = QPushButton("Center")
        self.hand_thumb_right_button = QPushButton("Right")
        thumb_row.addWidget(self.hand_thumb_left_button)
        thumb_row.addWidget(self.hand_thumb_center_button)
        thumb_row.addWidget(self.hand_thumb_right_button)
        motion_list.add(
            components.SettingsRow("Thumb rotation", thumb_row_widget)
        )
        hand_motion_layout.addWidget(motion_group)

        # Per-finger sliders. Drag a slider to set the target angle; the
        # right-most "Send" button sends just that finger (others left
        # untouched). Range is tuned per actuator so the usable open/close
        # range falls in the middle of the slider.
        slider_group, slider_list = components.section(
            "Finger positions",
            "Drag to set a target angle, then send that finger or all six.",
        )

        # name, angle index, slider min, slider max, default open/close hints
        self._hand_slider_specs = [
            ("Little finger", 0, 800, 1900, 1720, 900),
            ("Ring finger", 1, 800, 1900, 1720, 900),
            ("Middle finger", 2, 800, 1900, 1720, 900),
            ("Index finger", 3, 800, 1900, 1720, 900),
            ("Thumb bending", 4, 1000, 1500, 1350, 1100),
            ("Thumb rotation", 5, 500, 2000, 1000, 1000),
        ]
        # Public list reused by the interaction layer (replaces the old
        # ``hand_angle_spins``). Each entry is a ``QSlider``.
        self.hand_angle_sliders = []
        self._hand_angle_value_labels = []
        self._hand_angle_send_buttons = []
        self._hand_angle_open_values = []
        self._hand_angle_close_values = []

        for name, idx, lo, hi, open_v, close_v in self._hand_slider_specs:
            slider = QSlider(Qt.Horizontal)
            slider.setRange(int(lo), int(hi))
            slider.setSingleStep(1)
            slider.setPageStep(max(10, (hi - lo) // 20))
            slider.setValue(int(open_v))
            slider.setTracking(True)
            slider.setMinimumWidth(120)

            # Fixed-width figures: this tracks the slider as you drag, and a
            # proportional font would shuffle the send button sideways.
            value_label = QLabel(str(int(open_v)))
            value_label.setMinimumWidth(44)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value_label.setStyleSheet(theme.readout_style())

            # Labelled, not an icon: a paper-plane glyph does not say "send
            # this one actuator", and in a list the repeated word forms a
            # readable column -- the same shape as the Read column in
            # Diagnostics. All six share a column, so it costs width once.
            send_btn = QPushButton("Send")
            send_btn.setMinimumWidth(64)
            send_btn.setToolTip(
                f"Send angle{idx} only; the other five actuators stay where "
                f"they are."
            )

            control = QWidget()
            control_row = QHBoxLayout(control)
            control_row.setContentsMargins(0, 0, 0, 0)
            control_row.setSpacing(theme.SPACE_SM)
            control_row.addWidget(slider, 1)
            control_row.addWidget(value_label)
            control_row.addWidget(send_btn)
            # angle index as the hint, so the label stays plain language.
            slider_list.add(
                components.SettingsRow(name, control, hint=f"angle{idx}")
            )

            self.hand_angle_sliders.append(slider)
            self._hand_angle_value_labels.append(value_label)
            self._hand_angle_send_buttons.append(send_btn)
            self._hand_angle_open_values.append(int(open_v))
            self._hand_angle_close_values.append(int(close_v))

        # Preset and send actions as two rows of the same list, so they read as
        # belonging to the sliders above rather than as loose buttons.
        preset_row_widget = QWidget()
        preset_row = QHBoxLayout(preset_row_widget)
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(theme.SPACE_SM)
        bottom_row_widget = QWidget()
        bottom_row = QHBoxLayout(bottom_row_widget)
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(theme.SPACE_SM)
        self.hand_sliders_live_check = QCheckBox("Live update (drag to send)")
        self.hand_sliders_live_check.setToolTip(
            "When enabled, first synchronize all sliders from the hand's\n"
            "actual angles, then continuously stream slider changes\n"
            "(throttled). Turn it off to dial in values without sending."
        )
        bottom_row.addWidget(self.hand_sliders_live_check)

        self.hand_sliders_load_open_button = QPushButton("Load Open")
        self.hand_sliders_load_open_button.setToolTip(
            "Snap all six sliders to their suggested OPEN position. Does not\n"
            "send anything until you press 'Send All'."
        )
        preset_row.addWidget(self.hand_sliders_load_open_button)

        self.hand_sliders_load_close_button = QPushButton("Load Close")
        self.hand_sliders_load_close_button.setToolTip(
            "Snap all six sliders to their suggested CLOSE position. Does\n"
            "not send anything until you press 'Send All'."
        )
        preset_row.addWidget(self.hand_sliders_load_close_button)

        self.hand_sliders_sync_button = QPushButton("Sync From Actual")
        self.hand_sliders_sync_button.setToolTip(
            "Read /Getangleact and set every slider to the live actual\n"
            "angle. Useful when starting from an unknown pose."
        )
        preset_row.addWidget(self.hand_sliders_sync_button)
        preset_row.addStretch(1)

        bottom_row.addStretch(1)

        self.hand_send_custom_angles_button = QPushButton("Send All")
        components.apply_variant(self.hand_send_custom_angles_button, "primary")
        self.hand_send_custom_angles_button.setToolTip(
            "Send the current value of all six sliders in one go."
        )
        bottom_row.addWidget(self.hand_send_custom_angles_button)

        slider_list.add_stacked("Presets", preset_row_widget)
        slider_list.add(bottom_row_widget)
        hand_motion_layout.addWidget(slider_group)

        pose_group, pose_list = components.section("Saved poses")

        pose_selection_widget = QWidget()
        pose_selection_row = QHBoxLayout(pose_selection_widget)
        pose_selection_row.setContentsMargins(0, 0, 0, 0)
        pose_selection_row.setSpacing(theme.SPACE_SM)
        self.hand_pose_preset_combo = QComboBox()
        self.hand_pose_preset_combo.setEditable(True)
        self.hand_pose_preset_combo.setInsertPolicy(QComboBox.NoInsert)
        self.hand_pose_preset_combo.setMinimumWidth(180)
        self.hand_pose_preset_combo.lineEdit().setMaxLength(64)
        self.hand_pose_preset_combo.lineEdit().setPlaceholderText(
            "Enter or select a pose name"
        )
        self.hand_pose_preset_combo.setToolTip(
            "Choose a saved pose, or enter a new name before saving the\n"
            "current six slider values."
        )
        pose_selection_row.addWidget(self.hand_pose_preset_combo, stretch=1)

        self.hand_pose_save_button = QPushButton("Save Current")
        self.hand_pose_save_button.setToolTip(
            "Save all six current slider values under the selected name."
        )
        pose_selection_row.addWidget(self.hand_pose_save_button)
        pose_list.add(components.SettingsRow("Pose", pose_selection_widget))

        pose_action_widget = QWidget()
        pose_action_row = QHBoxLayout(pose_action_widget)
        pose_action_row.setContentsMargins(0, 0, 0, 0)
        pose_action_row.setSpacing(theme.SPACE_SM)
        self.hand_pose_load_button = QPushButton("Load")
        self.hand_pose_load_button.setToolTip(
            "Load the saved values into the sliders without moving the hand."
        )
        pose_action_row.addWidget(self.hand_pose_load_button)

        self.hand_pose_load_send_button = QPushButton("Load && Send")
        self.hand_pose_load_send_button.setToolTip(
            "Load the selected pose and send all six values to the hand."
        )
        pose_action_row.addWidget(self.hand_pose_load_send_button)

        self.hand_pose_delete_button = QPushButton("Delete")
        components.apply_variant(self.hand_pose_delete_button, "danger")
        self.hand_pose_delete_button.setToolTip("Delete the selected saved pose.")
        pose_action_row.addWidget(self.hand_pose_delete_button)

        pose_action_row.addStretch(1)
        pose_list.add(pose_action_widget)

        self.hand_pose_status_label = QLabel("No saved poses")
        self.hand_pose_status_label.setStyleSheet(theme.type_style("caption"))
        pose_list.add(
            components.SettingsRow("Stored", self.hand_pose_status_label)
        )

        hand_motion_layout.addWidget(pose_group)
        self._load_hand_pose_presets()
        hand_motion_layout.addStretch()

        self.hand_sub_tabs.addTab(
            self._scrollable(hand_monitor_page), "Tactile / Model"
        )
        self.hand_sub_tabs.addTab(
            self._scrollable(hand_motion_page), "Motion Controls"
        )
        layout.addWidget(self.hand_sub_tabs)

    def _set_button_active(self, btn: QPushButton, active: bool):
        """Green when active; when inactive, revert to the default theme."""
        if active:
            btn.setStyleSheet(theme.active_button_style())
        else:
            btn.setStyleSheet("")  # clear → default OS/theme styling

    def open_proximity_recording_viewer(self, session):
        rows = list((session or {}).get("rows") or [])
        if not rows:
            return

        def col(name):
            values = []
            for row in rows:
                try:
                    values.append(float(row.get(name, float("nan"))))
                except Exception:
                    values.append(float("nan"))
            return values

        t = col("t_sec")
        dialog = QDialog(self)
        dialog.setWindowTitle("Proximity Recording Viewer")
        components.size_to_screen(dialog, 920, 720)
        layout = QVBoxLayout(dialog)
        tabs = QTabWidget(dialog)
        layout.addWidget(tabs)

        summary = QTextEdit()
        summary.setReadOnly(True)
        csv_path = (session or {}).get("csv_path", "")
        npz_path = (session or {}).get("npz_path", "")
        signal = (session or {}).get("sensor_signal", None)
        signal_shape = getattr(signal, "shape", None)
        summary.setText(
            "Proximity recording saved.\n\n"
            f"Samples: {len(rows)}\n"
            f"Duration: {t[-1] if t else 0.0:.3f} s\n"
            f"CSV: {csv_path}\n"
            f"NPZ: {npz_path}\n"
            f"Sensor stack shape: {signal_shape}\n\n"
            "CSV contains time-series values. NPZ contains full sensor_signal frames."
        )
        tabs.addTab(summary, "Summary")

        tabs.addTab(
            ProximityRecordingChartWidget(
                "Finger / Sensor State",
                [
                    ("center_col", t, col("center_col"), "#42a5f5"),
                    ("center_row", t, col("center_row"), "#66bb6a"),
                    ("strength", t, col("strength"), "#ffa726"),
                    ("sensor_max", t, col("sensor_max"), "#ab47bc"),
                ],
            ),
            "Finger",
        )
        tabs.addTab(
            ProximityRecordingChartWidget(
                "Tracking Errors",
                [
                    ("col_error", t, col("center_col_error"), "#42a5f5"),
                    ("row_error", t, col("center_row_error"), "#66bb6a"),
                    ("strength_error", t, col("strength_error"), "#ef5350"),
                ],
            ),
            "Errors",
        )
        tabs.addTab(
            ProximityRecordingChartWidget(
                "Robot Velocity Commands (pre frame/sign flip)",
                [
                    ("vx", t, col("cmd_vx_pre"), "#42a5f5"),
                    ("vy", t, col("cmd_vy_pre"), "#ef5350"),
                    ("vz", t, col("cmd_vz_pre"), "#66bb6a"),
                ],
            ),
            "Velocity",
        )
        tabs.addTab(
            ProximityRecordingChartWidget(
                "Robot Tool Position",
                [
                    ("tool_x", t, col("tool_x"), "#42a5f5"),
                    ("tool_y", t, col("tool_y"), "#ef5350"),
                    ("tool_z", t, col("tool_z"), "#66bb6a"),
                ],
            ),
            "Tool Pose",
        )
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)
        dialog.show()
        self._proximity_recording_viewer = dialog

    def send_velocity_command(self, v_x, v_y, v_z):
        """
        Sends a velocity vector to the robot in the TOOL frame.
        v_x, v_y, v_z are speeds in m/s (e.g., 0.02 or -0.02).
        """
        if not self.features.get('robot_ready', False):
            return

        # Create the lists expected by the API
        v_lin = [v_x, v_y, v_z]
        v_rot = [0.0, 0.0, 0.0]  # We don't want to rotate, just move

        try:
            if hasattr(self.robot_api, "send_end_effector_velocity_in_frame"):
                self.robot_api.send_end_effector_velocity_in_frame(
                    v_lin, v_rot, frame="tool"
                )
                return
            self.robot_api.send_request(
                self.robot_api.set_end_effector_velocity_in_frame(v_lin, v_rot, frame="tool")
            )
        except Exception as e:
            print(f"Error sending velocity: {e}")

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
