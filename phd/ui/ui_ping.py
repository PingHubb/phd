from __future__ import annotations

import math
import json
import os
from typing import Any, Optional

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QColor, QDragEnterEvent, QDropEvent, QPainter, QPen
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QHeaderView,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from phd.dependence.paths import ai_resource_path, resource_path
from phd.ui import theme
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
        "models", "ai_direct_finger_motion", "latest_cnn_gru_model.pt"
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

    def set_contact_normal_visualization_enabled(self, *_args, **_kwargs):
        return None

    def get_contact_normal_estimator_modes(self):
        return [
            ("motion_direction_v3", "Motion Direction (V3)"),
            ("touch_anchor_v4", "Touch Anchor Direction (V4)"),
        ]

    def set_contact_normal_estimator_mode(self, *_args, **_kwargs):
        return None

    def set_sensor_point_labels_enabled(self, *_args, **_kwargs):
        return None

    def set_sensor_contact_force_scale(self, *_args, **_kwargs):
        return None

    def set_sensor_geometry_config(self, *_args, **_kwargs):
        return None

    def set_saved_sensor_geometry_config(self, *_args, **_kwargs):
        return None

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
        outer.setSpacing(6)
        outer.addWidget(QLabel("TM script (SendScript):"))
        self.robot_script_input = QPlainTextEdit()
        self.robot_script_input.setPlaceholderText(
            'Example: PTP("JPP",0,-30,-45,0,-90,0,100,0,100,true)'
        )
        self.robot_script_input.setMinimumHeight(88)
        self.robot_script_input.setTabChangesFocus(True)
        outer.addWidget(self.robot_script_input)
        self.transmit_script_button = QPushButton("Transmit script")
        self.transmit_script_button.clicked.connect(self._on_transmit_clicked)
        outer.addWidget(self.transmit_script_button)

    def _on_transmit_clicked(self):
        self.transmit_script.emit(self.robot_script_input.toPlainText().strip())

    def toggle_visibility(self):
        self.setVisible(not self.isVisible())


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

        # --- MODIFICATION: Use QGroupBox for title and layout ---
        self.angle_group_box = QGroupBox("Angle")
        # You can use QSS to style this group box, e.g., self.angle_group_box.setObjectName("angleGroup")
        grid_layout = QGridLayout(self.angle_group_box)
        grid_layout.setContentsMargins(10, 10, 10, 10)

        step_rad = math.radians(10.0)
        for i in range(6):
            row = i % 3
            start_col = 0 if i < 3 else 2
            row_widget = QWidget()
            row_h = QHBoxLayout(row_widget)
            row_h.setContentsMargins(0, 2, 0, 2)
            row_h.setSpacing(6)

            label = QLabel(f"Joint {i + 1}:")
            line_edit = QLineEdit()
            line_edit.setText(f"{self.presets[1][i]:.4f}")
            line_edit.setMinimumWidth(72)

            btn_minus = QPushButton("-10°")
            btn_plus = QPushButton("+10°")
            btn_minus.setFixedWidth(52)
            btn_plus.setFixedWidth(52)
            btn_minus.clicked.connect(lambda checked, idx=i, dr=-step_rad: self._nudge_joint_rad(idx, dr))
            btn_plus.clicked.connect(lambda checked, idx=i, dr=step_rad: self._nudge_joint_rad(idx, dr))

            row_h.addWidget(label)
            row_h.addWidget(line_edit, 1)
            row_h.addWidget(btn_minus)
            row_h.addWidget(btn_plus)

            grid_layout.addWidget(row_widget, row, start_col, 1, 2)

            self.labels.append(label)
            self.position_edits.append(line_edit)

        # Initially hide the group box
        self.angle_group_box.setVisible(False)
        self.layout().addWidget(self.angle_group_box)

        # Group all buttons in a single horizontal layout to save vertical space
        action_widget = QWidget()
        action_layout = QHBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 10, 0, 0)

        self.preset_buttons = []
        for i in range(1, 6):
            btn = QPushButton(str(i))
            btn.clicked.connect(lambda checked, p=i: self.apply_preset(p))
            action_layout.addWidget(btn)
            self.preset_buttons.append(btn)

        self.load_current_angle_button = QPushButton("Load Current Robot Angle")
        self.load_current_angle_button.clicked.connect(self.load_current_robot_angle_into_current_preset)
        action_layout.addWidget(self.load_current_angle_button)
        self._update_load_current_angle_button_state()

        action_layout.addStretch()

        self.send_button = QPushButton("Send Positions")
        self.send_button.clicked.connect(self.send_positions)
        action_layout.addWidget(self.send_button)

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
        isVisible = not self.isVisible()
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

        # Group inputs into Position and Orientation
        self.input_container = QWidget()
        top_layout = QHBoxLayout(self.input_container)
        top_layout.setContentsMargins(0, 0, 0, 0)

        position_group = QGroupBox("Position (XYZ)")
        position_layout = QVBoxLayout(position_group)

        orientation_group = QGroupBox("Orientation (Quaternion)")
        orientation_layout = QVBoxLayout(orientation_group)

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

        # Group all buttons in a single horizontal layout to save vertical space
        action_widget = QWidget()
        action_layout = QHBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 10, 0, 0)

        self.preset_buttons = []
        for i in range(1, 3):
            btn = QPushButton(f"Preset {i}")
            btn.clicked.connect(lambda checked, p=i: self.apply_preset(p))
            action_layout.addWidget(btn)
            self.preset_buttons.append(btn)

        action_layout.addStretch()

        self.send_button = QPushButton("Send Tool Position")
        self.send_button.clicked.connect(self.send_positions)
        action_layout.addWidget(self.send_button)

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
        control_layout = QHBoxLayout()
        label = QLabel(f"{identifier}:")
        line_edit = QLineEdit()
        line_edit.setText(f"{preset_value:.2f}")

        control_layout.addWidget(label)
        control_layout.addWidget(line_edit)
        layout.addLayout(control_layout)

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
        isVisible = not self.isVisible()
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

        control_group = QGroupBox("Velocity Control (Tool Frame)")
        outer.addWidget(control_group)
        main_layout = QVBoxLayout(control_group)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        info = QLabel(
            "Set speed with the sliders, then click a direction button to send a tool-frame velocity command."
        )
        info.setWordWrap(True)
        main_layout.addWidget(info)

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

        self.btn_stop_all = QPushButton("STOP (All 0)")
        self.btn_stop_all.setMinimumHeight(34)
        self.btn_stop_all.clicked.connect(self.stop_all_velocity)
        stop_row.addWidget(self.btn_stop_all)

        self.btn_stop_velocity_mode = QPushButton("Stop Velocity Mode")
        self.btn_stop_velocity_mode.setMinimumHeight(34)
        self.btn_stop_velocity_mode.clicked.connect(self.stop_velocity_mode)
        stop_row.addWidget(self.btn_stop_velocity_mode)

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
        self.setVisible(not self.isVisible())


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

        self.setHandleWidth(3)
        self.setup_layout()
        self._bootstrap_ui_services()
        self._flush_startup_messages()

        self.connect_function()
        self.adjust_splitter_sizes()
        self._apply_startup_feature_state()
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

        self.features['robot_ready'] = bool(
            getattr(self.robot_api, 'service_ok', False)
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
        self.features['hand_ready'] = bool(hand_services_ready or hand_tactile_ready)
        self.features['sensor_ready'] = bool(self.features['sensor_module'])
        self.features['gripper_ready'] = bool(self.features['gripper_driver'])
        self.features['camera_ready'] = bool(self.features['camera_driver'])

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

    def ensure_sensor_api(self) -> bool:
        """Create the sensor API only when it is first needed."""
        if self.sensor_api is not None and not isinstance(self.sensor_api, NullSensorApi):
            return True

        if ArduinoCommander is None:
            self.features['sensor_driver'] = False
            self._startup_log('⚠️ Sensor API unavailable: dependency import failed.')
            return False

        try:
            self.sensor_api = ArduinoCommander()
            self.features['sensor_driver'] = True
            return True
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

    def _apply_startup_feature_state(self):
        if not self.features.get('sensor_module', False):
            self.tab_widget.setTabEnabled(0, False)
            self.tab_widget.setTabEnabled(2, False)

        if not self.features.get('robot_ready', False):
            self.set_robot_subtab_enabled(False)
            self.auto_center_button.setEnabled(False)

        if not self.features.get('hand_ready', False):
            if hasattr(self, "hand_tab_index"):
                self.tab_widget.setTabEnabled(int(self.hand_tab_index), False)

        if not self.features.get('camera_ready', False):
            self.live_yolo_button.setEnabled(False)
            self.auto_center_button.setEnabled(False)

        if not self.features.get('gripper_ready', False):
            self._set_widgets_enabled(
                [self.gripper_slider, self.btn_grip_open, self.btn_grip_close],
                False,
            )

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
        self.log_display.setVisible(False)
        self.log_display.textChanged.connect(self.show_log_if_hidden)

        self.splitter_1 = QSplitter(Qt.Horizontal, self)
        self.splitter_1.addWidget(self.widget_plotter)
        self.splitter_1.addWidget(self.widget_plotter_2)
        self.splitter_1.addWidget(self.log_display)
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

        self.widget_func = QWidget()
        self.layout_func = QVBoxLayout(self.widget_func)
        self.layout_func.addWidget(self.position_entry_widget)
        self.layout_func.addWidget(self.position_quaternion_widget)
        self.layout_func.addWidget(self.position_toolframe_widget)
        self.layout_func.addWidget(self.position_script_widget)

        self.tab_widget = QTabWidget()
        self.tab_widget.setUsesScrollButtons(False)
        self.setup_tabs()

        self.layout_func.addWidget(self.tab_widget)
        self.splitter_2.addWidget(self.widget_func)
        self.addWidget(self.splitter_1)
        self.addWidget(self.splitter_2)

    def setup_tabs(self):
        # Tab 1: Sensor
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        self.setup_tab1(tab1_layout)
        self.tab_widget.addTab(tab1, "Sensor")

        # Tab 2: ROBOTS (contains two subtabs)
        robots_tab = QWidget()
        robots_layout = QVBoxLayout(robots_tab)
        self.robots_sub_tabs = QTabWidget()
        self.robots_sub_tabs.setUsesScrollButtons(False)

        # ─── Subtab “TM Robot” ───
        robot_page = QWidget()
        robot_layout = QVBoxLayout(robot_page)
        self.setup_tab2(robot_layout)
        self.robots_sub_tabs.addTab(robot_page, "TM Robot")

        robots_layout.addWidget(self.robots_sub_tabs)
        self.tab_widget.addTab(robots_tab, "Robots")

        # Tab 3: AI
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        self.setup_tab3(tab3_layout)
        self.tab_widget.addTab(tab3, "AI")

        # Tab 4: Dexterous Hand
        tab4 = QWidget()
        tab4_layout = QVBoxLayout(tab4)
        self.setup_tab5(tab4_layout)
        self.hand_tab_index = self.tab_widget.addTab(tab4, "Dexterous Hand")

        # Tab 5: Extra (last)
        tab5 = QWidget()
        tab5_layout = QVBoxLayout(tab5)
        self.setup_tab4(tab5_layout)
        self.tab_widget.addTab(tab5, "Extra")

    def setup_tab1(self, layout):
        self.sensor_sub_tabs = QTabWidget()
        self.sensor_sub_tabs.setUsesScrollButtons(False)

        # ─── Subtab “Send Operation” ───
        send_page = QWidget()
        send_page_layout = QVBoxLayout(send_page)

        send_group = QGroupBox("Send Operations")
        send_layout = QVBoxLayout()

        viz_group = QGroupBox("Visualization Settings")
        viz_layout = QVBoxLayout(viz_group)

        # Sensor selection and connection widgets
        self.sensor_choice = QListWidget(self.widget_func)
        self.sensor_choice.setSelectionMode(QListWidget.SingleSelection)
        self.sensor_choice.addItems([
            "2D", "Elbow", "Kuka", "Double Curve", "Half Cylinder Surface"
        ])
        self.sensor_choice.setCurrentRow(0)
        send_layout.addWidget(self.sensor_choice)

        self.serial_channel = QListWidget(self.widget_func)
        self.serial_channel.setSelectionMode(QListWidget.MultiSelection)
        send_layout.addWidget(self.serial_channel)

        self.buildScene = QPushButton("Build Scene", self.widget_func)
        send_layout.addWidget(self.buildScene)

        # Keep a single backend update trigger object for compatibility with existing
        # enable/disable hooks; the visible Update Sensor control is now in top toolbar.
        self.sensor_update = QPushButton("Update Sensor", self.widget_func)
        self.sensor_update.setVisible(False)

        send_group.setLayout(send_layout)

        # 2D grid controls
        grid_container = QWidget()
        grid_layout = QHBoxLayout(grid_container)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        grid_layout.addWidget(QLabel("2D Grid (rows × cols):"))

        self.grid_rows_spin = QSpinBox()
        self.grid_rows_spin.setRange(2, 100)
        self.grid_rows_spin.setValue(10)
        grid_layout.addWidget(self.grid_rows_spin)

        grid_layout.addWidget(QLabel("×"))

        self.grid_cols_spin = QSpinBox()
        self.grid_cols_spin.setRange(2, 100)
        self.grid_cols_spin.setValue(10)
        grid_layout.addWidget(self.grid_cols_spin)

        viz_layout.addWidget(grid_container)

        # Sensitivity slider
        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)

        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(0, 1000)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self.sensitivity_slider.setTickInterval(10)

        self.sensitivity_value_label = QLabel("0.050")
        self.sensitivity_value_label.setFixedWidth(48)

        slider_layout.addWidget(QLabel("Sensitivity:"))
        slider_layout.addWidget(self.sensitivity_slider)
        slider_layout.addWidget(self.sensitivity_value_label)

        viz_layout.addWidget(slider_container)

        visual_mode_container = QWidget()
        visual_mode_layout = QHBoxLayout(visual_mode_container)
        visual_mode_layout.setContentsMargins(0, 0, 0, 0)
        self.sensor_visualization_mode_combo = QComboBox()
        self.sensor_visualization_mode_combo.addItem("Point Grid", "point_grid")
        self.sensor_visualization_mode_combo.addItem("Stereo Field", "stereo_field")
        self.sensor_visualization_mode_combo.setToolTip(
            "Choose the live sensor rendering style."
        )
        visual_mode_layout.addWidget(QLabel("Visual Mode:"))
        visual_mode_layout.addWidget(self.sensor_visualization_mode_combo)
        visual_mode_layout.addStretch()
        viz_layout.addWidget(visual_mode_container)

        normal_vector_container = QWidget()
        normal_vector_layout = QVBoxLayout(normal_vector_container)
        normal_vector_layout.setContentsMargins(0, 0, 0, 0)
        normal_vector_layout.setSpacing(4)
        self.contact_normal_checkbox = QCheckBox("Show Contact Vector")
        self.contact_normal_checkbox.setChecked(True)
        self.contact_normal_checkbox.setToolTip(
            "Show an estimated contact arrow in the sensor scene."
        )
        self.contact_normal_estimator_combo = QComboBox()
        self.contact_normal_estimator_combo.addItem("Motion Direction (V3)", "motion_direction_v3")
        self.contact_normal_estimator_combo.addItem("Touch Anchor Direction (V4)", "touch_anchor_v4")
        self.contact_normal_estimator_combo.setCurrentIndex(1)
        self.contact_normal_estimator_combo.setToolTip(
            "Choose whether the arrow shows pressure-derived normal tilt or contact motion direction."
        )
        self.contact_normal_status_label = QLabel("Normal vector: waiting for contact")
        self.contact_normal_status_label.setWordWrap(True)
        self.contact_normal_status_label.setStyleSheet(theme.MUTED_LABEL_STYLE)
        self.contact_force_status_label = QLabel("Contact force: waiting for contact")
        self.contact_force_status_label.setWordWrap(True)
        self.contact_force_status_label.setStyleSheet(theme.INFO_LABEL_STYLE)
        normal_vector_layout.addWidget(self.contact_normal_checkbox)
        normal_vector_layout.addWidget(self.contact_normal_estimator_combo)
        normal_vector_layout.addWidget(self.contact_normal_status_label)
        normal_vector_layout.addWidget(self.contact_force_status_label)
        viz_layout.addWidget(normal_vector_container)

        send_page_layout.addWidget(send_group)
        send_page_layout.addWidget(viz_group)
        send_page_layout.addStretch()

        # ─── Subtab “Read Operation” ───
        read_page = QWidget()
        read_page_layout = QVBoxLayout(read_page)

        read_group = QGroupBox("Read Operations")
        read_layout = QVBoxLayout()

        self.read_sensor_api_button = QPushButton("Sensor API Raw Data (ADJUST IN API!!)")
        self.read_sensor_api_hz_button = QPushButton("Sensor API Raw Hz")
        self.read_sensor_channel_button = QPushButton("Sensor API Channel (ADJUST IN API!!)")
        self.read_sensor_raw_button = QPushButton("Sensor Raw Data")
        self.read_sensor_raw_ave_button = QPushButton("Sensor Raw Ave Data")
        self.read_sensor_diff_button = QPushButton("Sensor Diff Data")
        self.read_sensor_diff_debug_button = QPushButton("Sensor Diff Debug Views")
        self.read_runtime_hz_button = QPushButton("Sensor / DFM Runtime Hz")

        read_layout.addWidget(self.read_sensor_api_button)
        read_layout.addWidget(self.read_sensor_api_hz_button)
        read_layout.addWidget(self.read_sensor_channel_button)
        read_layout.addWidget(self.read_sensor_raw_button)
        read_layout.addWidget(self.read_sensor_raw_ave_button)
        read_layout.addWidget(self.read_sensor_diff_button)
        read_layout.addWidget(self.read_sensor_diff_debug_button)
        read_layout.addWidget(self.read_runtime_hz_button)

        read_group.setLayout(read_layout)
        read_page_layout.addWidget(read_group)
        read_page_layout.addStretch()

        self.sensor_sub_tabs.addTab(send_page, "Send Operation")
        self.sensor_sub_tabs.addTab(read_page, "Read Operation")
        layout.addWidget(self.sensor_sub_tabs)
        self._build_sensor_parameters_dialog()

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
        self.sensor_parameters_dialog.resize(620, 500)

        layout = QVBoxLayout(self.sensor_parameters_dialog)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

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

        self.sensor_parameter_normal_flip_checkbox = QCheckBox("Flip Normals")
        self.sensor_parameter_use_shape_checkbox = QCheckBox("Use Selected Shape")
        self.sensor_parameter_stereo_ignore_noise_checkbox = QCheckBox("Ignore Stereo Field Noise")
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

        grid.addWidget(QLabel("Sensor:"), 0, 0)
        grid.addWidget(self.sensor_parameter_model_combo, 0, 1)
        grid.addWidget(QLabel("Reorder Logic:"), 1, 0)
        grid.addWidget(self.sensor_parameter_reorder_combo, 1, 1)
        grid.addWidget(QLabel("Force Scale (N/signal):"), 2, 0)
        grid.addWidget(self.sensor_parameter_force_scale_spin, 2, 1)
        grid.addWidget(QLabel("Average Window:"), 3, 0)
        grid.addWidget(self.sensor_average_window_spin, 3, 1)
        grid.addWidget(QLabel("Render Hz:"), 4, 0)
        grid.addWidget(self.visualization_target_hz_spin, 4, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self.sensor_parameter_point_labels_checkbox = QCheckBox("Show Point Labels")
        self.sensor_parameter_point_labels_checkbox.setToolTip(
            "Show each coarse sensor point label in the 3D sensor scene."
        )
        layout.addWidget(self.sensor_parameter_point_labels_checkbox)

        self.sensor_parameter_background_reference_checkbox = QCheckBox("Show Background Axes/Grid")
        self.sensor_parameter_background_reference_checkbox.setToolTip(
            "Show the red/green reference axes and gray XY wireframe plane behind the sensor."
        )
        self.sensor_parameter_background_reference_checkbox.setChecked(True)
        layout.addWidget(self.sensor_parameter_background_reference_checkbox)

        stereo_group = QGroupBox("Stereo Field")
        stereo_grid = QGridLayout(stereo_group)
        stereo_grid.setHorizontalSpacing(12)
        stereo_grid.setVerticalSpacing(8)
        stereo_grid.addWidget(self.sensor_parameter_stereo_ignore_noise_checkbox, 0, 1)
        stereo_grid.addWidget(QLabel("Noise Threshold:"), 1, 0)
        stereo_grid.addWidget(self.sensor_parameter_stereo_deadband_spin, 1, 1)
        stereo_grid.addWidget(QLabel("Line Length:"), 2, 0)
        stereo_grid.addWidget(self.sensor_parameter_stereo_length_spin, 2, 1)
        stereo_grid.setColumnStretch(1, 1)
        layout.addWidget(stereo_group)

        geometry_group = QGroupBox("2D Geometry")
        geometry_grid = QGridLayout(geometry_group)
        geometry_grid.setHorizontalSpacing(12)
        geometry_grid.setVerticalSpacing(8)
        geometry_grid.addWidget(self.sensor_parameter_use_shape_checkbox, 0, 1)
        geometry_grid.addWidget(QLabel("Shape:"), 1, 0)
        geometry_grid.addWidget(self.sensor_parameter_shape_combo, 1, 1)
        geometry_grid.addWidget(QLabel("Bend Direction:"), 2, 0)
        geometry_grid.addWidget(self.sensor_parameter_bend_axis_combo, 2, 1)
        geometry_grid.addWidget(QLabel("Arc Angle:"), 3, 0)
        geometry_grid.addWidget(self.sensor_parameter_arc_spin, 3, 1)
        geometry_grid.addWidget(self.sensor_parameter_normal_flip_checkbox, 4, 1)
        geometry_grid.addWidget(QLabel("Rotation X:"), 5, 0)
        geometry_grid.addWidget(self.sensor_parameter_rotation_x_spin, 5, 1)
        geometry_grid.addWidget(QLabel("Rotation Y:"), 6, 0)
        geometry_grid.addWidget(self.sensor_parameter_rotation_y_spin, 6, 1)
        geometry_grid.addWidget(QLabel("Rotation Z:"), 7, 0)
        geometry_grid.addWidget(self.sensor_parameter_rotation_z_spin, 7, 1)
        geometry_grid.setColumnStretch(1, 1)
        self.sensor_parameter_geometry_group = geometry_group
        layout.addWidget(geometry_group)

        self.sensor_parameter_status_label = QLabel("")
        self.sensor_parameter_status_label.setWordWrap(True)
        self.sensor_parameter_status_label.setStyleSheet(theme.MUTED_LABEL_STYLE)
        layout.addWidget(self.sensor_parameter_status_label)

        button_row = QHBoxLayout()
        self.sensor_parameter_save_button = QPushButton("Save for Sensor")
        self.sensor_parameter_reload_button = QPushButton("Reload Saved")
        button_row.addWidget(self.sensor_parameter_save_button)
        button_row.addWidget(self.sensor_parameter_reload_button)
        button_row.addStretch()
        layout.addLayout(button_row)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.sensor_parameters_dialog.close)
        layout.addWidget(button_box)

        self.sensor_parameter_model_combo.currentIndexChanged.connect(
            self._load_sensor_parameter_reorder_mode
        )
        self.sensor_parameter_point_labels_checkbox.toggled.connect(
            self._on_sensor_parameter_point_labels_toggled
        )
        self.sensor_parameter_background_reference_checkbox.toggled.connect(
            self._on_sensor_parameter_background_reference_toggled
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
        self.sensor_parameter_rotation_x_spin.valueChanged.connect(
            self._on_sensor_parameter_geometry_changed
        )
        self.sensor_parameter_rotation_y_spin.valueChanged.connect(
            self._on_sensor_parameter_geometry_changed
        )
        self.sensor_parameter_rotation_z_spin.valueChanged.connect(
            self._on_sensor_parameter_geometry_changed
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
        self.sensor_parameter_save_button.clicked.connect(
            self._save_sensor_parameter_reorder_mode
        )
        self.sensor_parameter_reload_button.clicked.connect(
            self._load_sensor_parameter_reorder_mode
        )

    def open_sensor_parameters_dialog(self):
        if not hasattr(self, "sensor_parameters_dialog"):
            self._build_sensor_parameters_dialog()
        self._refresh_sensor_parameters_dialog()
        self.sensor_parameters_dialog.show()
        self.sensor_parameters_dialog.raise_()
        self.sensor_parameters_dialog.activateWindow()

    def _refresh_sensor_parameters_dialog(self):
        helper = getattr(self, "sensor_functions", None)

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

    def _update_sensor_parameter_geometry_subcontrols(self):
        group_enabled = bool(
            getattr(self, "sensor_parameter_geometry_group", None) is not None
            and self.sensor_parameter_geometry_group.isEnabled()
        )
        shape_enabled = group_enabled and self.sensor_parameter_use_shape_checkbox.isChecked()
        for widget in (
            self.sensor_parameter_shape_combo,
            self.sensor_parameter_bend_axis_combo,
            self.sensor_parameter_arc_spin,
            self.sensor_parameter_normal_flip_checkbox,
        ):
            widget.setEnabled(shape_enabled)
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
            f"Effective on next Build Scene: {effective_logic}\n"
            f"Point labels: {point_labels}\n"
            f"Background axes/grid: {background_reference}\n"
            f"Force scale: {force_scale:.6f} N/signal\n"
            f"Stereo field: noise {stereo_noise}, threshold "
            f"{float(stereo_field.get('deadband_pct', 0.35) or 0.0):.2f}, "
            f"length {float(stereo_field.get('length_scale', 0.35) or 0.35):.2f}\n"
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
            context = helper.get_sensor_reorder_context(model_name)
        except Exception as exc:
            self.sensor_parameter_status_label.setText(f"Failed to save sensor parameters: {exc}")
            return

        if ok:
            saved_geometry = context.get("geometry", {}) or {}
            saved_stereo = context.get("stereo_field", {}) or {}
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
                f"Effective on next Build Scene: {context.get('effective_logic') or 'none'}\n"
                f"Point labels: {'on' if context.get('point_labels_enabled', False) else 'off'}\n"
                f"Background axes/grid: {'on' if context.get('background_reference_enabled', True) else 'off'}\n"
                f"Force scale: {float(context.get('force_scale_n_per_signal', 0.0) or 0.0):.6f} N/signal\n"
                f"Stereo field: noise "
                f"{'ignored' if saved_stereo.get('ignore_noise_enabled', True) else 'raw'}, "
                f"threshold {float(saved_stereo.get('deadband_pct', 0.35) or 0.0):.2f}, "
                f"length {float(saved_stereo.get('length_scale', 0.35) or 0.35):.2f}\n"
                f"Geometry: {saved_mode}; {saved_shape}, "
                f"{saved_geometry.get('bend_axis', 'columns')}, "
                f"{saved_arc:.1f} deg, "
                f"rot XYZ=({saved_rotation[0]:.1f}, {saved_rotation[1]:.1f}, "
                f"{saved_rotation[2]:.1f}) deg"
            )
        else:
            self.sensor_parameter_status_label.setText("Failed to save sensor parameters.")

    def setup_tab2(self, layout):
        self.read_group_robot = QGroupBox("Read Operations")
        send_group = QGroupBox("Send Operations")
        read_layout = QVBoxLayout()
        send_layout = QVBoxLayout()

        self.read_joint_angle_button = QPushButton("Read Joint Angle")
        self.read_tool_position_button = QPushButton("Read Tool Position ")

        self.send_position_PTP_J_button = QPushButton("Send Joint Angle")
        self.send_position_PTP_T_button = QPushButton("Send Tool Position (Base Frame)")
        self.send_position_PTP_T_toolframe_button = QPushButton("Send Tool Velocity (Tool Frame)")

        self.send_script_button = QPushButton("Send Script")
        self.show_robot_button = QPushButton("Import 3D Robot Model")

        read_layout.addWidget(self.read_joint_angle_button)
        read_layout.addWidget(self.read_tool_position_button)
        read_layout.addWidget(self.show_robot_button)
        send_layout.addWidget(self.send_position_PTP_J_button)
        send_layout.addWidget(self.send_position_PTP_T_button)
        send_layout.addWidget(self.send_position_PTP_T_toolframe_button)
        send_layout.addWidget(self.send_script_button)

        self.read_group_robot.setLayout(read_layout)
        send_group.setLayout(send_layout)
        layout.addWidget(self.read_group_robot)
        layout.addWidget(send_group)

    def setup_tab3(self, layout):
        self.ai_sub_tabs = QTabWidget()
        self.ai_sub_tabs.setUsesScrollButtons(False)

        # ─── Subtab “AI Model” ───
        ai_model_page = QWidget()
        ai_model_layout = QVBoxLayout(ai_model_page)

        self.predict_threelevel_hierarchical_transformer_gesture_button = QPushButton("Predict (ThreeLevel)")
        self.btn_toggle_3lvl_latch = QPushButton("3-Level: Latch OFF")
        self.proximity_control_button = QPushButton("Proximity Control")
        self.proximity_record_button = QPushButton("Record Proximity Data")

        self.proximity_settings_dialog = QDialog(self.widget_func)
        self.proximity_settings_dialog.setWindowTitle("Proximity Control Settings")
        self.proximity_settings_dialog.setModal(False)
        self.proximity_settings_dialog.resize(820, 560)
        proximity_settings_layout = QVBoxLayout(self.proximity_settings_dialog)
        proximity_settings_layout.setContentsMargins(10, 10, 10, 10)
        proximity_settings_layout.setSpacing(8)
        proximity_note = QLabel(
            "Tune how the robot keeps the finger centered over the sensor and maintains the taught hover distance."
        )
        proximity_note.setWordWrap(True)
        proximity_settings_layout.addWidget(proximity_note)

        self.proximity_settings_group = QGroupBox("Proximity Control Parameters")
        proximity_params_grid = QGridLayout(self.proximity_settings_group)
        proximity_params_grid.setContentsMargins(10, 10, 10, 10)
        proximity_params_grid.setHorizontalSpacing(12)
        proximity_params_grid.setVerticalSpacing(8)

        def _mk_dspin(default: float, minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
            sp = QDoubleSpinBox()
            sp.setRange(minimum, maximum)
            sp.setSingleStep(step)
            sp.setDecimals(3)
            sp.setValue(default)
            sp.setMinimumWidth(120)
            return sp

        def _add_param_row(row: int, name: str, description: str, widget):
            name_label = QLabel(name)
            desc_label = QLabel(description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet(theme.MUTED_LABEL_STYLE)
            tooltip = f"{name}\n\n{description}"
            name_label.setToolTip(tooltip)
            desc_label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            proximity_params_grid.addWidget(name_label, row, 0)
            proximity_params_grid.addWidget(desc_label, row, 1)
            proximity_params_grid.addWidget(widget, row, 2)

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
        proximity_params_grid.setColumnStretch(1, 1)
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
        frame_grid.addWidget(QLabel("EE frame:"), 0, 0)

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
            "joint1": (0, 1),
            "joint2": (0, 2),
            "joint3": (0, 3),
            "joint4": (1, 1),
            "joint5": (1, 2),
            "tool": (1, 3),
        }
        for frame_key, frame_label in frame_button_order:
            btn = QPushButton(frame_label)
            btn.setCheckable(True)
            btn.setMinimumWidth(90)
            btn.setMinimumHeight(32)
            btn.clicked.connect(lambda checked, key=frame_key: self._set_ai_frame(key))
            self.ai_frame_buttons[frame_key] = btn
            row, col = frame_button_positions[frame_key]
            frame_grid.addWidget(btn, row, col)

        self.btn_toggle_anchor_axes = QPushButton("Axes: Anchored ON")  # label will be synced on init
        self.btn_toggle_anchor_axes.setMinimumWidth(150)
        self.btn_toggle_anchor_axes.setMinimumHeight(32)
        frame_grid.addWidget(self.btn_toggle_anchor_axes, 0, 4, 2, 1)

        frame_grid.setColumnStretch(5, 1)
        self._update_ai_frame_buttons()
        ai_model_layout.addWidget(frame_row)

        ai_based_group = QGroupBox("AI-Based")
        ai_based_layout = QVBoxLayout(ai_based_group)
        ai_based_layout.setContentsMargins(10, 10, 10, 10)
        ai_based_layout.setSpacing(6)

        rule_based_group = QGroupBox("Rule-Based")
        rule_based_layout = QVBoxLayout(rule_based_group)
        rule_based_layout.setContentsMargins(10, 10, 10, 10)
        rule_based_layout.setSpacing(6)

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

        model_row = QWidget()
        model_row_layout = QHBoxLayout(model_row)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        model_row_layout.addWidget(QLabel("Model Path:"))
        self.ai_direct_execution_model_path_input = QLineEdit()
        self.ai_direct_execution_model_path_input.setPlaceholderText(
            DisabledSensorFunctions.DEFAULT_AI_DIRECT_EXECUTION_MODEL_PATH
        )
        default_ai_model_path = self._get_default_ai_execution_model_path()
        self.ai_direct_execution_model_path_input.setText(default_ai_model_path)
        model_row_layout.addWidget(self.ai_direct_execution_model_path_input)

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
        execute_safety_layout.addWidget(self.ai_direct_execution_dry_run_checkbox)
        execute_safety_layout.addWidget(self.ai_direct_execution_prediction_status)
        execute_safety_layout.addStretch()

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
        ai_based_layout.addWidget(threelevel_row)
        proximity_row = QWidget()
        proximity_row_layout = QHBoxLayout(proximity_row)
        proximity_row_layout.setContentsMargins(0, 0, 0, 0)
        proximity_row_layout.setSpacing(6)
        self.proximity_control_button.setMinimumWidth(0)
        self.proximity_record_button.setMinimumWidth(0)
        proximity_row_layout.addWidget(self.proximity_control_button, 1)
        proximity_row_layout.addWidget(self.proximity_record_button, 1)
        rule_based_layout.addWidget(proximity_row)
        rule_based_layout.addWidget(self.direct_finger_motion_button)
        console_row = QWidget()
        console_row_layout = QHBoxLayout(console_row)
        console_row_layout.setContentsMargins(0, 0, 0, 0)
        console_row_layout.setSpacing(6)
        self.console_control_button.setMinimumWidth(0)
        self.console_control_sensor_button.setMinimumWidth(0)
        self.console_control_sensor_v2_button.setMinimumWidth(0)
        console_row_layout.addWidget(self.console_control_button, 1)
        console_row_layout.addWidget(self.console_control_sensor_button, 1)
        console_row_layout.addWidget(self.console_control_sensor_v2_button, 1)
        rule_based_layout.addWidget(console_row)
        tool_pose_row = QWidget()
        tool_pose_row_layout = QHBoxLayout(tool_pose_row)
        tool_pose_row_layout.setContentsMargins(0, 0, 0, 0)
        tool_pose_row_layout.setSpacing(6)
        self.direct_finger_motion_tool_pose_record_menu_button.setMinimumWidth(0)
        self.load_tool_pose_path_button.setMinimumWidth(0)
        self.clear_tool_pose_path_button.setMinimumWidth(0)
        tool_pose_row_layout.addWidget(self.direct_finger_motion_tool_pose_record_menu_button, 1)
        tool_pose_row_layout.addWidget(self.load_tool_pose_path_button, 1)
        tool_pose_row_layout.addWidget(self.clear_tool_pose_path_button, 1)
        rule_based_layout.addWidget(tool_pose_row)
        self._build_direct_finger_motion_settings_dialog()
        self._build_console_control_settings_dialog()
        ai_based_layout.addWidget(model_row)
        ai_based_layout.addWidget(execute_safety_row)
        ai_based_layout.addWidget(self.ai_direct_finger_motion_execution_button)
        ai_model_layout.addWidget(ai_based_group)
        ai_model_layout.addWidget(rule_based_group)
        ai_model_layout.addStretch()

        # ─── Subtab “Data Training” ───
        training_page = QWidget()
        training_layout = QVBoxLayout(training_page)

        self.set_no_trigger_button = QPushButton("Set No Trigger Mode")
        training_layout.addWidget(self.set_no_trigger_button)
        self.set_no_trigger_auto_button = QPushButton("Set No Trigger Auto Mode")
        training_layout.addWidget(self.set_no_trigger_auto_button)
        self.set_no_trigger_no_updatecal_auto_button = QPushButton("Set No Trigger No UpdateCal Auto Mode")
        training_layout.addWidget(self.set_no_trigger_no_updatecal_auto_button)
        self.set_trigger_button = QPushButton("Set Trigger Mode")
        training_layout.addWidget(self.set_trigger_button)

        first_row_layout = QHBoxLayout()
        gesture_label = QLabel("Enter Gesture Number:")
        self.gesture_number_input = QLineEdit()
        self.gesture_number_input.setFixedSize(50, 40)
        first_row_layout.addWidget(gesture_label)
        first_row_layout.addWidget(self.gesture_number_input)
        self.record_gesture_button = QPushButton("Record")
        training_layout.addLayout(first_row_layout)
        training_layout.addWidget(self.record_gesture_button)
        ai_dfm_record_row = QWidget()
        ai_dfm_record_row_layout = QHBoxLayout(ai_dfm_record_row)
        ai_dfm_record_row_layout.setContentsMargins(0, 0, 0, 0)
        ai_dfm_record_row_layout.setSpacing(6)
        self.ai_direct_finger_motion_button.setMinimumWidth(0)
        self.ai_direct_finger_motion_robot_button.setMinimumWidth(0)
        ai_dfm_record_row_layout.addWidget(self.ai_direct_finger_motion_button, 1)
        ai_dfm_record_row_layout.addWidget(self.ai_direct_finger_motion_robot_button, 1)
        training_layout.addWidget(ai_dfm_record_row)

        self.ai_teaching_label_group = QGroupBox("AI Teaching Label")
        teaching_grid = QGridLayout(self.ai_teaching_label_group)
        teaching_grid.setContentsMargins(8, 8, 8, 8)
        teaching_grid.setHorizontalSpacing(6)
        teaching_grid.setVerticalSpacing(6)

        self.ai_teaching_label_buttons = {}
        self.ai_teaching_label_status = QLabel("Teaching: Auto/DFM")

        teaching_specs = [
            ("auto", "Auto/DFM"),
            ("stop", "Stop"),
            ("normal_swipe", "Normal Swipe"),
            ("push", "Push"),
            ("pull", "Pull"),
            ("x_pos", "X+"),
            ("x_neg", "X-"),
            ("y_pos", "Y+"),
            ("y_neg", "Y-"),
            ("z_pos", "Z+"),
            ("z_neg", "Z-"),
            ("rx_pos", "RX+"),
            ("rx_neg", "RX-"),
            ("ry_pos", "RY+"),
            ("ry_neg", "RY-"),
            ("rz_pos", "RZ+"),
            ("rz_neg", "RZ-"),
        ]
        for idx, (label_key, label_text) in enumerate(teaching_specs):
            button = QPushButton(label_text)
            button.setCheckable(True)
            button.setMinimumHeight(30)
            button.setMinimumWidth(96)
            self.ai_teaching_label_buttons[label_key] = button
            teaching_grid.addWidget(button, idx // 4, idx % 4)

        teaching_grid.addWidget(self.ai_teaching_label_status, 5, 0, 1, 4)
        training_layout.addWidget(self.ai_teaching_label_group)
        training_layout.addStretch()

        self.ai_sub_tabs.addTab(ai_model_page, "AI Model")
        self.ai_sub_tabs.addTab(training_page, "Data Training")
        layout.addWidget(self.ai_sub_tabs)

    def setup_tab4(self, layout):
        # --- Gripper Manual Control ---
        gripper_group = QGroupBox("Gripper Manual Control")
        gripper_layout = QVBoxLayout()

        slider_row = QHBoxLayout()
        self.gripper_slider = QSlider(Qt.Horizontal)
        self.gripper_slider.setRange(0, 100)
        self.gripper_slider.setValue(0)
        self.gripper_slider.setTickPosition(QSlider.TicksBelow)
        self.gripper_slider.setTickInterval(10)

        self.gripper_label = QLabel("0.00 (Open)")
        self.gripper_label.setFixedWidth(80)

        slider_row.addWidget(QLabel("Open"))
        slider_row.addWidget(self.gripper_slider)
        slider_row.addWidget(QLabel("Closed"))
        slider_row.addWidget(self.gripper_label)

        btn_row = QHBoxLayout()
        self.btn_grip_open = QPushButton("Fully Open")
        self.btn_grip_close = QPushButton("Fully Close")
        btn_row.addWidget(self.btn_grip_open)
        btn_row.addWidget(self.btn_grip_close)

        gripper_layout.addLayout(slider_row)
        gripper_layout.addLayout(btn_row)
        gripper_group.setLayout(gripper_layout)
        layout.addWidget(gripper_group)

        # Connections
        self.gripper_slider.valueChanged.connect(self._update_slider_label)
        self.gripper_slider.sliderReleased.connect(self._on_slider_released)
        self.btn_grip_open.clicked.connect(lambda: self.set_gripper_manual(0))
        self.btn_grip_close.clicked.connect(lambda: self.set_gripper_manual(100))

        # --- AI Camera Section ---
        camera_group = QGroupBox("AI Camera")
        camera_layout = QVBoxLayout()
        self.live_yolo_button = QPushButton("Start Live Object Detection")
        self.live_yolo_button.clicked.connect(self.toggle_yolo_camera)

        self.auto_center_button = QPushButton("Auto-Center on object")
        self.auto_center_button.setCheckable(True)
        self.auto_center_button.setEnabled(False)
        self.auto_center_button.clicked.connect(self.toggle_centering_mode)

        camera_layout.addWidget(self.live_yolo_button)
        camera_layout.addWidget(self.auto_center_button)
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

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

        self.hand_state_label = QLabel("RH56F1 status: idle")
        hand_monitor_layout.addWidget(self.hand_state_label)

        model_group = QGroupBox("3D Hand Model")
        model_layout = QHBoxLayout(model_group)
        self.hand_model_show_button = QPushButton("Import 3D Hand Model")
        model_layout.addWidget(self.hand_model_show_button)
        self.hand_model_status_label = QLabel("resource/dexterous_hand")
        self.hand_model_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        model_layout.addWidget(self.hand_model_status_label, stretch=1)
        hand_monitor_layout.addWidget(model_group)

        tactile_group = QGroupBox("Tactile Sensor Readout")
        tactile_layout = QVBoxLayout(tactile_group)

        tactile_control_row = QHBoxLayout()
        self.hand_tactile_live_button = QPushButton("Start Live Tactile")
        self.hand_tactile_live_button.setCheckable(True)
        tactile_control_row.addWidget(self.hand_tactile_live_button)

        self.hand_tactile_refresh_button = QPushButton("Refresh Display")
        tactile_control_row.addWidget(self.hand_tactile_refresh_button)

        self.hand_tactile_status_label = QLabel("No tactile data")
        self.hand_tactile_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        tactile_control_row.addWidget(self.hand_tactile_status_label, stretch=1)
        tactile_layout.addLayout(tactile_control_row)

        self.hand_tactile_table = QTableWidget(8, 5)
        self.hand_tactile_table.setHorizontalHeaderLabels(
            ["Region", "Normal (N)", "Tangential (N)", "Direction", "Proximity"]
        )
        self.hand_tactile_table.verticalHeader().hide()
        self.hand_tactile_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.hand_tactile_table.setSelectionMode(QTableWidget.NoSelection)
        self.hand_tactile_table.setMinimumHeight(245)
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

        tactile_layout.addWidget(self.hand_tactile_table)
        hand_monitor_layout.addWidget(tactile_group)
        hand_monitor_layout.addStretch()

        self.hand_tactile_timer = QTimer(self)
        self.hand_tactile_timer.setInterval(100)
        self.hand_tactile_timer.timeout.connect(self._refresh_hand_tactile_display)

        speed_force_group = QGroupBox("Global Speed / Force")
        sf_layout = QGridLayout(speed_force_group)
        sf_layout.addWidget(QLabel("Speed (all):"), 0, 0)
        self.hand_speed_spin = QSpinBox()
        self.hand_speed_spin.setRange(0, 3000)
        self.hand_speed_spin.setValue(300)
        sf_layout.addWidget(self.hand_speed_spin, 0, 1)
        self.hand_apply_speed_button = QPushButton("Apply Speed")
        sf_layout.addWidget(self.hand_apply_speed_button, 0, 2)
        sf_layout.addWidget(QLabel("Force (all):"), 1, 0)
        self.hand_force_spin = QSpinBox()
        self.hand_force_spin.setRange(0, 12000)
        self.hand_force_spin.setValue(2000)
        sf_layout.addWidget(self.hand_force_spin, 1, 1)
        self.hand_apply_force_button = QPushButton("Apply Force")
        sf_layout.addWidget(self.hand_apply_force_button, 1, 2)
        hand_motion_layout.addWidget(speed_force_group)

        open_close_group = QGroupBox("Quick Actions")
        oc_layout = QHBoxLayout(open_close_group)
        self.hand_open_all_button = QPushButton("Open All")
        self.hand_close_all_button = QPushButton("Close All")
        self.hand_read_angles_button = QPushButton("Read Actual Angles")
        oc_layout.addWidget(self.hand_open_all_button)
        oc_layout.addWidget(self.hand_close_all_button)
        oc_layout.addWidget(self.hand_read_angles_button)
        hand_motion_layout.addWidget(open_close_group)

        thumb_group = QGroupBox("Thumb Rotation Presets")
        thumb_layout = QHBoxLayout(thumb_group)
        self.hand_thumb_left_button = QPushButton("Thumb Left")
        self.hand_thumb_center_button = QPushButton("Thumb Center")
        self.hand_thumb_right_button = QPushButton("Thumb Right")
        thumb_layout.addWidget(self.hand_thumb_left_button)
        thumb_layout.addWidget(self.hand_thumb_center_button)
        thumb_layout.addWidget(self.hand_thumb_right_button)
        hand_motion_layout.addWidget(thumb_group)

        # Per-finger sliders. Drag a slider to set the target angle; the
        # right-most "Send" button sends just that finger (others left
        # untouched). Range is tuned per actuator so the usable open/close
        # range falls in the middle of the slider.
        slider_group = QGroupBox("Finger Position Sliders")
        slider_outer = QVBoxLayout(slider_group)

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

        sliders_grid = QGridLayout()
        sliders_grid.setHorizontalSpacing(8)
        sliders_grid.setVerticalSpacing(4)
        for row, (name, idx, lo, hi, open_v, close_v) in enumerate(
            self._hand_slider_specs
        ):
            name_label = QLabel(f"{name}\n(angle{idx})")
            name_label.setMinimumWidth(110)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(int(lo), int(hi))
            slider.setSingleStep(1)
            slider.setPageStep(max(10, (hi - lo) // 20))
            slider.setValue(int(open_v))
            slider.setTracking(True)

            value_label = QLabel(str(int(open_v)))
            value_label.setMinimumWidth(48)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            send_btn = QPushButton("Send")
            send_btn.setFixedWidth(56)
            send_btn.setToolTip(
                f"Send angle{idx}={slider.value()} only; the other five\n"
                f"actuators stay where they are."
            )

            sliders_grid.addWidget(name_label, row, 0)
            sliders_grid.addWidget(slider, row, 1)
            sliders_grid.addWidget(value_label, row, 2)
            sliders_grid.addWidget(send_btn, row, 3)

            self.hand_angle_sliders.append(slider)
            self._hand_angle_value_labels.append(value_label)
            self._hand_angle_send_buttons.append(send_btn)
            self._hand_angle_open_values.append(int(open_v))
            self._hand_angle_close_values.append(int(close_v))

        slider_outer.addLayout(sliders_grid)

        # Action / quick preset row.
        bottom_row = QHBoxLayout()
        self.hand_sliders_live_check = QCheckBox("Live update (drag to send)")
        self.hand_sliders_live_check.setToolTip(
            "When ticked, dragging a slider continuously streams the new\n"
            "angle to the hand (throttled). Turn off if you prefer to dial\n"
            "in the value first and press 'Send' yourself."
        )
        bottom_row.addWidget(self.hand_sliders_live_check)

        self.hand_sliders_load_open_button = QPushButton("Load Open")
        self.hand_sliders_load_open_button.setToolTip(
            "Snap all six sliders to their suggested OPEN position. Does not\n"
            "send anything until you press 'Send All'."
        )
        bottom_row.addWidget(self.hand_sliders_load_open_button)

        self.hand_sliders_load_close_button = QPushButton("Load Close")
        self.hand_sliders_load_close_button.setToolTip(
            "Snap all six sliders to their suggested CLOSE position. Does\n"
            "not send anything until you press 'Send All'."
        )
        bottom_row.addWidget(self.hand_sliders_load_close_button)

        self.hand_sliders_sync_button = QPushButton("Sync From Actual")
        self.hand_sliders_sync_button.setToolTip(
            "Read /Getangleact and set every slider to the live actual\n"
            "angle. Useful when starting from an unknown pose."
        )
        bottom_row.addWidget(self.hand_sliders_sync_button)

        bottom_row.addStretch(1)

        self.hand_send_custom_angles_button = QPushButton("Send All")
        self.hand_send_custom_angles_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 4px 12px; }"
        )
        self.hand_send_custom_angles_button.setToolTip(
            "Send the current value of all six sliders in one go."
        )
        bottom_row.addWidget(self.hand_send_custom_angles_button)

        slider_outer.addLayout(bottom_row)
        hand_motion_layout.addWidget(slider_group)
        hand_motion_layout.addStretch()

        self.hand_sub_tabs.addTab(hand_monitor_page, "Tactile / Model")
        self.hand_sub_tabs.addTab(hand_motion_page, "Motion Controls")
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
        dialog.resize(920, 720)
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
