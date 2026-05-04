from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QColor, QDragEnterEvent, QDropEvent, QPainter, QPen
from PyQt5.QtWidgets import (
    QDialog,
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
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
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

    def toggle_direct_finger_motion_v2(self):
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


class DisabledSensorFunctions:
    DEFAULT_AI_DIRECT_EXECUTION_MODEL_PATH = (
        "/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/ai_direct_finger_motion/best_model.pt"
    )
    DEFAULT_SENSOR_AVERAGE_WINDOW_SIZE = 3
    DEFAULT_VISUALIZATION_TARGET_HZ = 30.0

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

    def get_ai_direct_finger_motion_execution_default_model_path(self):
        return self.DEFAULT_AI_DIRECT_EXECUTION_MODEL_PATH


class NullMeshLab:
    def __init__(self, parent):
        self.parent = parent

    def addRobot(self):
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
    def __init__(self, robot_api=None, parent=None):
        super().__init__(parent)
        self.robot_api = robot_api
        self.setLayout(QVBoxLayout())
        self.position_edits = []
        self.labels = []
        # --- MODIFICATION: Removed inline stylesheet ---

        self.presets = {
            1: [-0.7242335, 0.28315391, -1.523286731370, -0.32650261753, -1.5700302685, 0.0],
            2: [-1.1, -0.43900, -1.005029724, -0.143107, -1.57, 0.0]
        }

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
        for i in range(1, 3):
            btn = QPushButton(f"Preset {i}")
            btn.clicked.connect(lambda checked, p=i: self.apply_preset(p))
            action_layout.addWidget(btn)
            self.preset_buttons.append(btn)

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
            for i, value in enumerate(preset_values):
                self.position_edits[i].setText(f"{value:.4f}")

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
            api.send_positions_joint_angle(positions)
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
      - compact stop buttons at bottom
    """

    def __init__(self, robot_api, log_display=None, parent=None):
        super().__init__(parent)
        self.robot_api = robot_api
        self.log_display = log_display

        self.linear_speed = 0.02   # m/s
        self.angular_speed = 0.01  # rad/s

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
        self.angular_slider.setRange(1, 200)  # 0.001 -> 0.200 rad/s
        self.angular_slider.setValue(int(self.angular_speed * 1000))
        self.angular_slider.valueChanged.connect(self._on_angular_slider)
        self.angular_label = QLabel()
        self.angular_label.setFixedWidth(90)
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
        self.angular_speed = value / 1000.0
        self._refresh_speed_labels()

    def _refresh_speed_labels(self):
        self.linear_label.setText(f"{self.linear_speed:.3f} m/s")
        self.angular_label.setText(f"{self.angular_speed:.3f} rad/s")

    def _append_log(self, message: str):
        if self.log_display is not None:
            try:
                self.log_display.append(message)
            except Exception:
                pass

    def _send_linear(self, x, y, z):
        self._send_velocity([float(x), float(y), float(z)], [0.0, 0.0, 0.0])

    def _send_angular(self, rx, ry, rz):
        self._send_velocity([0.0, 0.0, 0.0], [float(rx), float(ry), float(rz)])

    def _send_velocity(self, v_lin, v_rot):
        try:
            if not hasattr(self.robot_api, "send_request"):
                print("[RobotToolFramePositionWidget] robot_api has no send_request()")
                return

            if hasattr(self.robot_api, "suspend_end_effector_velocity_mode"):
                self.robot_api.send_request(self.robot_api.suspend_end_effector_velocity_mode())

            if hasattr(self.robot_api, "enable_end_effector_velocity_mode"):
                self.robot_api.send_request(self.robot_api.enable_end_effector_velocity_mode())

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

    def stop_all_velocity(self):
        self._send_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def stop_velocity_mode(self):
        """Safely exit velocity mode: Suspend -> Stop."""
        try:
            if not hasattr(self.robot_api, "send_request"):
                print("[RobotToolFramePositionWidget] robot_api has no send_request()")
                return

            if hasattr(self.robot_api, "suspend_end_effector_velocity_mode"):
                self.robot_api.send_request(self.robot_api.suspend_end_effector_velocity_mode())

            if hasattr(self.robot_api, "stop_end_effector_velocity_mode"):
                self.robot_api.send_request(self.robot_api.stop_end_effector_velocity_mode())

        except Exception as e:
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

        self.features['robot_ready'] = bool(getattr(self.robot_api, 'use_ros', False))
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
        self.plotter.background_color = '#202020'
        layout_plotter.addWidget(self.plotter.interactor)
        self.widget_plotter.setVisible(False)

        self.widget_plotter_2 = PlotterWidget()
        layout_plotter_2 = QGridLayout(self.widget_plotter_2)
        layout_plotter_2.setContentsMargins(0, 0, 0, 0)
        self.plotter_2 = QtInteractor(self.widget_plotter_2)
        self.plotter_2.background_color = '#202020'
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

        # Tab 4: Extra
        tab4 = QWidget()
        tab4_layout = QVBoxLayout(tab4)
        self.setup_tab4(tab4_layout)
        self.tab_widget.addTab(tab4, "Extra")

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
            "Elbow", "Kuka", "Double Curve", "2D", "Half Cylinder Surface"
        ])
        self.sensor_choice.setCurrentRow(0)
        send_layout.addWidget(self.sensor_choice)

        self.serial_channel = QListWidget(self.widget_func)
        self.serial_channel.setSelectionMode(QListWidget.MultiSelection)
        send_layout.addWidget(self.serial_channel)

        self.buildScene = QPushButton("Build Scene", self.widget_func)
        send_layout.addWidget(self.buildScene)

        self.sensor_update = QPushButton("Update Sensor", self.widget_func)
        send_layout.addWidget(self.sensor_update)

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

        avg_window_container = QWidget()
        avg_window_layout = QHBoxLayout(avg_window_container)
        avg_window_layout.setContentsMargins(0, 0, 0, 0)
        self.sensor_average_window_spin = QSpinBox()
        self.sensor_average_window_spin.setRange(1, 30)
        self.sensor_average_window_spin.setValue(DisabledSensorFunctions.DEFAULT_SENSOR_AVERAGE_WINDOW_SIZE)
        avg_window_layout.addWidget(QLabel("Average Window:"))
        avg_window_layout.addWidget(self.sensor_average_window_spin)
        avg_window_layout.addStretch()
        viz_layout.addWidget(avg_window_container)

        render_hz_container = QWidget()
        render_hz_layout = QHBoxLayout(render_hz_container)
        render_hz_layout.setContentsMargins(0, 0, 0, 0)
        self.visualization_target_hz_spin = QDoubleSpinBox()
        self.visualization_target_hz_spin.setDecimals(1)
        self.visualization_target_hz_spin.setRange(1.0, 240.0)
        self.visualization_target_hz_spin.setSingleStep(1.0)
        self.visualization_target_hz_spin.setValue(DisabledSensorFunctions.DEFAULT_VISUALIZATION_TARGET_HZ)
        render_hz_layout.addWidget(QLabel("Render Hz:"))
        render_hz_layout.addWidget(self.visualization_target_hz_spin)
        render_hz_layout.addStretch()
        viz_layout.addWidget(render_hz_container)

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
        self.continuous_read_button = QPushButton("Real-time Live 3D Robot Model", self.widget_func)

        read_layout.addWidget(self.read_joint_angle_button)
        read_layout.addWidget(self.read_tool_position_button)
        read_layout.addWidget(self.show_robot_button)
        read_layout.addWidget(self.continuous_read_button)
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
            desc_label.setStyleSheet("color: #b0b0b0;")
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

        frame_grid.setColumnStretch(4, 1)
        self._update_ai_frame_buttons()
        ai_model_layout.addWidget(frame_row)

        # ---- Axes anchor toggle row ----
        row_anchor = QWidget()
        ha = QHBoxLayout(row_anchor)
        ha.setContentsMargins(0, 0, 0, 0)
        self.btn_toggle_anchor_axes = QPushButton("Axes: Anchored ON")  # label will be synced on init
        ha.addWidget(self.btn_toggle_anchor_axes)
        ha.addStretch()
        ai_model_layout.addWidget(row_anchor)

        self.update_sensor_button = QPushButton("Update Sensor")
        self.direct_finger_motion_button = QPushButton("Direct Finger Motion")
        self.direct_finger_motion_v2_button = QPushButton("Direct Finger Motion (Version 2)")
        self.console_control_button = QPushButton("Console Control (PS5)")
        self.console_control_sensor_button = QPushButton("Console Control (Sensor)")
        self.direct_finger_motion_tool_pose_record_menu_button = QPushButton("Tool Pose Recording")
        self.load_tool_pose_path_button = QPushButton("Load Tool Pose Path")
        self.clear_tool_pose_path_button = QPushButton("Clear Tool Pose Path")
        self.ai_direct_finger_motion_button = QPushButton("AI Direct Finger Motion (Record)")
        self.ai_direct_finger_motion_execution_button = QPushButton("AI Direct Finger Motion (Execute)")

        model_row = QWidget()
        model_row_layout = QHBoxLayout(model_row)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        model_row_layout.addWidget(QLabel("Model Path:"))
        self.ai_direct_execution_model_path_input = QLineEdit()
        self.ai_direct_execution_model_path_input.setPlaceholderText(
            "/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/ai_direct_finger_motion/best_model.pt"
        )
        default_ai_model_path = self._get_default_ai_execution_model_path()
        self.ai_direct_execution_model_path_input.setText(default_ai_model_path)
        model_row_layout.addWidget(self.ai_direct_execution_model_path_input)

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
        ai_model_layout.addWidget(threelevel_row)
        ai_model_layout.addWidget(self.proximity_control_button)
        ai_model_layout.addWidget(self.proximity_record_button)
        ai_model_layout.addWidget(self.update_sensor_button)
        ai_model_layout.addWidget(self.direct_finger_motion_button)
        ai_model_layout.addWidget(self.direct_finger_motion_v2_button)
        console_row = QWidget()
        console_row_layout = QHBoxLayout(console_row)
        console_row_layout.setContentsMargins(0, 0, 0, 0)
        console_row_layout.setSpacing(6)
        self.console_control_button.setMinimumWidth(0)
        self.console_control_sensor_button.setMinimumWidth(0)
        console_row_layout.addWidget(self.console_control_button, 1)
        console_row_layout.addWidget(self.console_control_sensor_button, 1)
        ai_model_layout.addWidget(console_row)
        ai_model_layout.addWidget(self.direct_finger_motion_tool_pose_record_menu_button)
        tool_pose_path_row = QWidget()
        tool_pose_path_row_layout = QHBoxLayout(tool_pose_path_row)
        tool_pose_path_row_layout.setContentsMargins(0, 0, 0, 0)
        tool_pose_path_row_layout.addWidget(self.load_tool_pose_path_button)
        tool_pose_path_row_layout.addWidget(self.clear_tool_pose_path_button)
        ai_model_layout.addWidget(tool_pose_path_row)
        self._build_direct_finger_motion_settings_dialog()
        self._build_direct_finger_motion_v2_settings_dialog()
        self._build_console_control_settings_dialog()
        ai_model_layout.addWidget(model_row)
        ai_model_layout.addWidget(self.ai_direct_finger_motion_execution_button)
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
        training_layout.addWidget(self.ai_direct_finger_motion_button)
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

    def _set_button_active(self, btn: QPushButton, active: bool):
        """Green when active; when inactive, revert to the default theme."""
        if active:
            btn.setStyleSheet("QPushButton { background-color: #2e7d32; color: white; }")
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
            # We assume velocity mode is already enabled by the toggle button
            self.robot_api.send_request(
                self.robot_api.set_end_effector_velocity_in_frame(
                    v_lin, v_rot, frame="tool"
                )
            )
        except Exception as e:
            print(f"Error sending velocity: {e}")

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)