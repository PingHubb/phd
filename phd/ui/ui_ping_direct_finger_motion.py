from __future__ import annotations

import csv
import os
import struct
import time

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from pyvistaqt import QtInteractor


class DirectFingerMotionMixin:
    DIRECT_FINGER_MOTION_TOOL_POSE_RECORD_INTERVAL_MS = 200
    DIRECT_FINGER_MOTION_TOOL_POSE_PLAYBACK_INTERVAL_MS = 16
    DIRECT_FINGER_MOTION_TOOL_POSE_PLAYBACK_DURATION_SEC = 2.0
    DIRECT_FINGER_MOTION_TOOL_POSE_LOG_DIR = (
        "/home/ping2/ros2_ws/src/phd/phd/resource/robot/tool_pose_logs"
    )

    PS5_TEST_DEADBAND = 0.35

    def _build_ps5_controller_test_dialog(self):
        self.ps5_controller_test_dialog = QDialog(self)
        self.ps5_controller_test_dialog.setWindowTitle("PS5 Controller Test")
        self.ps5_controller_test_dialog.resize(760, 420)

        layout = QVBoxLayout(self.ps5_controller_test_dialog)
        self.ps5_controller_status_label = QLabel("Device: not connected")
        layout.addWidget(self.ps5_controller_status_label)
        self.ps5_controller_raw_label = QLabel("Raw: move or press a control to see active axis/button indices")
        self.ps5_controller_raw_label.setWordWrap(True)
        layout.addWidget(self.ps5_controller_raw_label)

        grid = QGridLayout()
        self.ps5_controller_indicator_labels = {}

        def add_indicator(key, text, row, col):
            label = QLabel(text)
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(110, 42)
            label.setStyleSheet(self._ps5_indicator_style(False))
            grid.addWidget(label, row, col)
            self.ps5_controller_indicator_labels[key] = label

        add_indicator("ls_up", "Left Stick Up", 0, 1)
        add_indicator("ls_left", "Left Stick Left", 1, 0)
        add_indicator("ls_down", "Left Stick Down", 1, 1)
        add_indicator("ls_right", "Left Stick Right", 1, 2)
        add_indicator("rs_up", "Right Stick Up", 0, 4)
        add_indicator("rs_left", "Right Stick Left", 1, 3)
        add_indicator("rs_down", "Right Stick Down", 1, 4)
        add_indicator("rs_right", "Right Stick Right", 1, 5)
        add_indicator("l1", "L1", 2, 0)
        add_indicator("r1", "R1", 2, 1)
        add_indicator("l2", "L2", 2, 2)
        add_indicator("r2", "R2", 2, 3)
        add_indicator("dpad_up", "D-Pad Up", 3, 1)
        add_indicator("dpad_left", "D-Pad Left", 4, 0)
        add_indicator("dpad_down", "D-Pad Down", 4, 1)
        add_indicator("dpad_right", "D-Pad Right", 4, 2)
        add_indicator("btn0", "Button 0 / Cross or A", 3, 3)
        add_indicator("btn1", "Button 1 / Circle or B", 3, 4)
        add_indicator("btn2", "Button 2 / Square or X", 4, 3)
        add_indicator("btn3", "Button 3 / Triangle or Y", 4, 4)

        layout.addLayout(grid)
        note = QLabel(
            "Linux often exposes PS5 controllers with Xbox-style axis/button indices. Use the Raw line to verify mappings."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.ps5_controller_test_dialog.close)
        layout.addWidget(close_button)

        self._ps5_test_fd = None
        self._ps5_test_device_path = None
        self._ps5_test_axes = {}
        self._ps5_test_buttons = {}
        self._ps5_test_timer = QTimer(self.ps5_controller_test_dialog)
        self._ps5_test_timer.setInterval(20)
        self._ps5_test_timer.timeout.connect(self._update_ps5_controller_test)
        self.ps5_controller_test_dialog.finished.connect(lambda *_: self._stop_ps5_controller_test())

    def _ps5_indicator_style(self, active):
        if active:
            return "background-color: #2ecc71; color: #111111; border: 1px solid #1e874b; border-radius: 6px;"
        return "background-color: #3a3a3a; color: #dddddd; border: 1px solid #666666; border-radius: 6px;"

    def _sensor_indicator_style(self, active, center=False):
        if not active:
            return "background-color: #3a3a3a; color: #dddddd; border: 1px solid #666666; border-radius: 6px;"
        if center:
            return "background-color: #3498db; color: #111111; border: 1px solid #21618c; border-radius: 6px;"
        return "background-color: #2ecc71; color: #111111; border: 1px solid #1e874b; border-radius: 6px;"

    def open_ps5_controller_test_dialog(self):
        if not hasattr(self, "ps5_controller_test_dialog"):
            self._build_ps5_controller_test_dialog()
        self.ps5_controller_test_dialog.show()
        self.ps5_controller_test_dialog.raise_()
        self.ps5_controller_test_dialog.activateWindow()
        self._ps5_test_timer.start()
        self._update_ps5_controller_test()

    def _build_sensor_controller_test_dialog(self):
        self.sensor_controller_test_dialog = QDialog(self)
        self.sensor_controller_test_dialog.setWindowTitle("Sensor Controller Test")
        self.sensor_controller_test_dialog.resize(760, 520)

        layout = QVBoxLayout(self.sensor_controller_test_dialog)
        self.sensor_controller_status_label = QLabel("Sensor mapping preview: ready")
        layout.addWidget(self.sensor_controller_status_label)
        self.sensor_controller_raw_label = QLabel("Raw: waiting for sensor touches")
        self.sensor_controller_raw_label.setWordWrap(True)
        layout.addWidget(self.sensor_controller_raw_label)

        grid = QGridLayout()
        self.sensor_controller_indicator_labels = {}

        def add_indicator(key, text, row, col, *, center=False):
            label = QLabel(text)
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(110, 42)
            label.setStyleSheet(self._sensor_indicator_style(False, center=center))
            label.setProperty("sensor_indicator_center", bool(center))
            grid.addWidget(label, row, col)
            self.sensor_controller_indicator_labels[key] = label

        # L1 / R1 on the top row (above stick-up). Sticks: 3x3 cross with center = press-to-center neutral.
        add_indicator("l1", "L1 (Top-Left)", 0, 0)
        add_indicator("r1", "R1 (Top-Right)", 0, 5)
        add_indicator("ls_up", "Left Stick Up", 1, 1)
        add_indicator("ls_left", "Left Stick Left", 2, 0)
        add_indicator(
            "ls_center",
            "L Stick Center",
            2,
            1,
            center=True,
        )
        add_indicator("ls_right", "Left Stick Right", 2, 2)
        add_indicator("ls_down", "Left Stick Down", 3, 1)
        add_indicator("rs_up", "Right Stick Up", 1, 4)
        add_indicator("rs_left", "Right Stick Left", 2, 3)
        add_indicator(
            "rs_center",
            "R Stick Center",
            2,
            4,
            center=True,
        )
        add_indicator("rs_right", "Right Stick Right", 2, 5)
        add_indicator("rs_down", "Right Stick Down", 3, 4)
        self.sensor_controller_indicator_labels["ls_center"].setToolTip(
            "Lit when the left virtual stick is anchored and neutral (your press defines center)."
        )
        self.sensor_controller_indicator_labels["rs_center"].setToolTip(
            "Lit when the right virtual stick is anchored and neutral (your press defines center)."
        )

        layout.addLayout(grid)
        note = QLabel(
            "Mapping: LHS / RHS are separate virtual sticks. First press in each half sets that stick's center (no output) "
            "until you move away from that point. Top two sensor rows (indices 0–1) are L1/R1 only "
            "(left half vs right half); virtual sticks read from row index 2 downward only. L2/R2 ignored for now."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.sensor_touch_grid_title_label = QLabel("Sensor Touch Grid")
        self.sensor_touch_grid_title_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.sensor_touch_grid_title_label)
        self.sensor_touch_grid_info_label = QLabel("Grid: waiting for sensor data")
        layout.addWidget(self.sensor_touch_grid_info_label)
        self.sensor_touch_grid_widget = QGroupBox()
        self.sensor_touch_grid_widget.setTitle("")
        self.sensor_touch_grid_layout = QGridLayout(self.sensor_touch_grid_widget)
        self.sensor_touch_grid_layout.setContentsMargins(4, 4, 4, 4)
        self.sensor_touch_grid_layout.setHorizontalSpacing(1)
        self.sensor_touch_grid_layout.setVerticalSpacing(1)
        layout.addWidget(self.sensor_touch_grid_widget)
        self._sensor_touch_cells = []
        self._sensor_touch_grid_shape = (0, 0)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.sensor_controller_test_dialog.close)
        layout.addWidget(close_button)

        self._sensor_controller_test_timer = QTimer(self.sensor_controller_test_dialog)
        self._sensor_controller_test_timer.setInterval(30)
        self._sensor_controller_test_timer.timeout.connect(self._update_sensor_controller_test)
        self.sensor_controller_test_dialog.finished.connect(
            lambda *_: self._stop_sensor_controller_test()
        )

    def open_sensor_controller_test_dialog(self):
        if not hasattr(self, "sensor_controller_test_dialog"):
            self._build_sensor_controller_test_dialog()
        self.sensor_controller_test_dialog.show()
        self.sensor_controller_test_dialog.raise_()
        self.sensor_controller_test_dialog.activateWindow()
        self._sensor_controller_test_timer.start()
        self._update_sensor_controller_test()

    def _stop_sensor_controller_test(self):
        timer = getattr(self, "_sensor_controller_test_timer", None)
        if timer is not None:
            timer.stop()

    def _set_sensor_indicator(self, key, active):
        label = self.sensor_controller_indicator_labels.get(key)
        if label is not None:
            center = bool(label.property("sensor_indicator_center"))
            label.setStyleSheet(self._sensor_indicator_style(bool(active), center=center))

    def _ensure_sensor_touch_grid(self, n_row, n_col):
        shape = (int(n_row), int(n_col))
        if shape == self._sensor_touch_grid_shape:
            return

        while self.sensor_touch_grid_layout.count():
            item = self.sensor_touch_grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._sensor_touch_cells = []
        for row in range(shape[0]):
            row_cells = []
            for col in range(shape[1]):
                cell = QLabel("")
                cell.setAlignment(Qt.AlignCenter)
                cell.setMinimumSize(18, 18)
                cell.setStyleSheet(
                    "background-color: #1f1f1f; border: 1px solid #3a3a3a; color: #d0d0d0;"
                )
                self.sensor_touch_grid_layout.addWidget(cell, row, col)
                row_cells.append(cell)
            self._sensor_touch_cells.append(row_cells)
        self._sensor_touch_grid_shape = shape
        self.sensor_touch_grid_info_label.setText(f"Grid: {shape[0]} x {shape[1]}")

    def _touch_cell_style(self, active_strength):
        strength = float(np.clip(active_strength, 0.0, 1.0))
        red = int(40 + 200 * strength)
        green = int(35 + 70 * strength)
        blue = int(35 + 50 * (1.0 - strength))
        return (
            f"background-color: rgb({red}, {green}, {blue}); "
            "border: 1px solid #3a3a3a; color: #ffffff;"
        )

    def _update_sensor_touch_grid(self, helper):
        my_sensor = getattr(helper, "my_sensor", None)
        if my_sensor is None or not hasattr(my_sensor, "_data"):
            return
        matrix = np.asarray(getattr(my_sensor._data, "diffPerDataAve", []), dtype=float)
        if matrix.ndim != 2 or matrix.size == 0:
            return

        n_row, n_col = int(matrix.shape[0]), int(matrix.shape[1])
        self._ensure_sensor_touch_grid(n_row, n_col)
        threshold = float(getattr(helper, "motion_threshold", -3.0))
        active_strength = np.maximum(threshold - matrix, 0.0)
        max_strength = float(np.max(active_strength)) if active_strength.size else 0.0
        norm = active_strength / max(1e-6, max_strength) if max_strength > 1e-6 else active_strength

        for row in range(n_row):
            for col in range(n_col):
                value = float(matrix[row, col])
                is_active = value < threshold
                cell = self._sensor_touch_cells[row][col]
                cell.setText("●" if is_active else "")
                cell.setStyleSheet(self._touch_cell_style(norm[row, col] if is_active else 0.0))

    def _update_sensor_controller_test(self):
        helper = self._get_sensor_helper("console_control_class")
        if helper is None or not hasattr(helper, "get_console_sensor_preview_inputs"):
            self.sensor_controller_status_label.setText("Sensor mapping preview: Console control helper unavailable")
            return

        try:
            lx, ly, rx, ry, l1, r1, _, _ = helper.get_console_sensor_preview_inputs()
            db = float(getattr(helper, "console_deadband", 0.08))
            center_state = (
                helper.get_console_sensor_stick_center_state(lx, ly, rx, ry)
                if hasattr(helper, "get_console_sensor_stick_center_state")
                else {"left_center": False, "right_center": False, "left_anchor": False, "right_anchor": False}
            )
            la = int(bool(center_state.get("left_anchor")))
            ra = int(bool(center_state.get("right_anchor")))
            lc = int(bool(center_state.get("left_center")))
            rc = int(bool(center_state.get("right_center")))
            self.sensor_controller_status_label.setText("Sensor mapping preview: connected")
            self.sensor_controller_raw_label.setText(
                f"Raw: LS({lx:+.2f}, {ly:+.2f}) | RS({rx:+.2f}, {ry:+.2f}) | L1={int(l1 > 0.5)} R1={int(r1 > 0.5)}"
                f" | anchor L={la} R={ra} | center L={lc} R={rc}"
            )
            states = {
                "ls_left": lx < -db,
                "ls_right": lx > db,
                "ls_up": ly < -db,
                "ls_down": ly > db,
                "rs_left": rx < -db,
                "rs_right": rx > db,
                "rs_up": ry < -db,
                "rs_down": ry > db,
                "ls_center": bool(center_state.get("left_center")),
                "rs_center": bool(center_state.get("right_center")),
                "l1": l1 > 0.5,
                "r1": r1 > 0.5,
            }
            for key, active in states.items():
                self._set_sensor_indicator(key, active)
            self._update_sensor_touch_grid(helper)
        except Exception as exc:
            self.sensor_controller_status_label.setText(f"Sensor mapping preview error: {exc}")

    def _find_ps5_test_device(self):
        input_dir = "/dev/input"
        preferred = os.path.join(input_dir, "js0")
        if os.path.exists(preferred):
            return preferred
        try:
            candidates = sorted(name for name in os.listdir(input_dir) if name.startswith("js"))
        except Exception:
            candidates = []
        return os.path.join(input_dir, candidates[0]) if candidates else None

    def _ensure_ps5_test_device(self):
        if getattr(self, "_ps5_test_fd", None) is not None:
            return True

        device_path = self._find_ps5_test_device()
        if device_path is None:
            self.ps5_controller_status_label.setText("Device: not found. Connect PS5 controller via USB/Bluetooth.")
            return False

        try:
            self._ps5_test_fd = os.open(device_path, os.O_RDONLY | os.O_NONBLOCK)
            self._ps5_test_device_path = device_path
            self.ps5_controller_status_label.setText(f"Device: connected ({device_path})")
            return True
        except Exception as exc:
            self.ps5_controller_status_label.setText(f"Device: failed to open {device_path}: {exc}")
            self._ps5_test_fd = None
            self._ps5_test_device_path = None
            return False

    def _stop_ps5_controller_test(self):
        timer = getattr(self, "_ps5_test_timer", None)
        if timer is not None:
            timer.stop()
        fd = getattr(self, "_ps5_test_fd", None)
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        self._ps5_test_fd = None
        self._ps5_test_device_path = None

    def _read_ps5_test_events(self):
        if not self._ensure_ps5_test_device():
            return
        try:
            while True:
                event = os.read(self._ps5_test_fd, 8)
                if len(event) < 8:
                    break
                _, value, event_type, number = struct.unpack("IhBB", event)
                event_type = event_type & ~0x80
                if event_type == 0x02:
                    self._ps5_test_axes[int(number)] = float(value) / 32767.0
                elif event_type == 0x01:
                    self._ps5_test_buttons[int(number)] = bool(value)
        except BlockingIOError:
            pass
        except OSError as exc:
            self.ps5_controller_status_label.setText(f"Device: disconnected/read error: {exc}")
            self._stop_ps5_controller_test()

    def _ps5_axis(self, index):
        return float(getattr(self, "_ps5_test_axes", {}).get(int(index), 0.0))

    def _ps5_button(self, index):
        return bool(getattr(self, "_ps5_test_buttons", {}).get(int(index), False))

    def _set_ps5_indicator(self, key, active):
        label = self.ps5_controller_indicator_labels.get(key)
        if label is not None:
            label.setStyleSheet(self._ps5_indicator_style(bool(active)))

    def _update_ps5_controller_test(self):
        self._read_ps5_test_events()
        db = float(self.PS5_TEST_DEADBAND)
        lx = self._ps5_axis(0)
        ly = self._ps5_axis(1)
        rx = self._ps5_axis(3)
        ry = self._ps5_axis(4)
        l2 = (self._ps5_axis(2) + 1.0) * 0.5
        r2 = (self._ps5_axis(5) + 1.0) * 0.5
        dpx = self._ps5_axis(6)
        dpy = self._ps5_axis(7)

        active_axes = [
            f"a{idx}={value:+.2f}"
            for idx, value in sorted(getattr(self, "_ps5_test_axes", {}).items())
            if abs(float(value)) > 0.10
        ]
        active_buttons = [
            f"b{idx}"
            for idx, active in sorted(getattr(self, "_ps5_test_buttons", {}).items())
            if active
        ]
        self.ps5_controller_raw_label.setText(
            "Raw: axes "
            + (", ".join(active_axes) if active_axes else "none")
            + " | buttons "
            + (", ".join(active_buttons) if active_buttons else "none")
        )

        states = {
            "ls_left": lx < -db,
            "ls_right": lx > db,
            "ls_up": ly < -db,
            "ls_down": ly > db,
            "rs_left": rx < -db,
            "rs_right": rx > db,
            "rs_up": ry < -db,
            "rs_down": ry > db,
            "l1": self._ps5_button(4),
            "r1": self._ps5_button(5),
            "l2": l2 > db,
            "r2": r2 > db,
            "dpad_left": dpx < -db,
            "dpad_right": dpx > db,
            "dpad_up": dpy < -db,
            "dpad_down": dpy > db,
            "btn0": self._ps5_button(0),
            "btn1": self._ps5_button(1),
            "btn2": self._ps5_button(2),
            "btn3": self._ps5_button(3),
        }
        for key, active in states.items():
            self._set_ps5_indicator(key, active)
    DFM_PARAMETER_TOOLTIPS = {
        "motion_threshold": (
            "Touch detection threshold.\n"
            "More negative → requires a stronger touch before motion starts.\n"
            "Less negative → responds earlier, but may pick up more noise.\n"
            "Single-finger swipe relevance: high."
        ),
        "no_touch_reset_limit": (
            "How many no-touch frames are required before the controller fully resets.\n"
            "Smaller → stop/release happens faster.\n"
            "Larger → more stable against brief dropouts, but adds release delay."
        ),
        "keep_margin": (
            "How strongly the tracker prefers to keep following the previous active touch.\n"
            "Larger → more stable, but can feel sticky during fast changes.\n"
            "Smaller → switches faster, but may jitter more."
        ),
        "robot_speed": (
            "Base linear speed used for single-finger swipe motion.\n"
            "Larger → robot feels faster.\n"
            "Smaller → robot feels softer/slower.\n"
            "This changes motion magnitude, not sensing delay."
        ),
        "centroid_deadband": (
            "Minimum normalized centroid movement per frame before motion is registered.\n"
            "Value is in sensor-fraction units (0 to 1).\n"
            "Smaller → more sensitive to tiny motion.\n"
            "Larger → more stable, but small swipes may be ignored.\n"
            "Single-finger swipe relevance: very high."
        ),
        "centroid_gain": (
            "Gain that maps normalized centroid movement into robot speed ratio.\n"
            "Value multiplies sensor-fraction delta (0 to 1).\n"
            "Larger → same finger movement produces stronger robot motion.\n"
            "Smaller → more gentle response.\n"
            "Single-finger swipe relevance: very high."
        ),
        "min_speed_ratio": (
            "Minimum non-zero speed ratio once motion is detected.\n"
            "Larger → motion starts more decisively.\n"
            "Smaller → motion starts more softly."
        ),
        "max_speed_ratio": (
            "Maximum speed ratio allowed after gain scaling.\n"
            "Larger → allows faster peak speed.\n"
            "Smaller → caps aggressive motion."
        ),
        "velocity_smoothing_alpha": (
            "EMA smoothing factor for robot velocity output.\n"
            "Smaller → smoother motion, less jitter, but slower response.\n"
            "Larger → faster response, but more noise passes through.\n"
            "0.0 = maximum smoothing (very slow). 1.0 = no smoothing (raw).\n"
            "Single-finger swipe relevance: very high."
        ),
        "push_value_threshold": (
            "Touch strength threshold for push detection.\n"
            "More negative → requires a deeper press.\n"
            "Less negative → push triggers more easily."
        ),
        "push_hold_deadband": (
            "Max normalized centroid drift allowed while counting as a held push.\n"
            "Value is in sensor-fraction units (0 to 1).\n"
            "Smaller → stricter hold detection.\n"
            "Larger → easier to trigger push, but more false positives."
        ),
        "push_hold_frames_required": (
            "How many consecutive frames must satisfy the push rule.\n"
            "Smaller → push triggers faster.\n"
            "Larger → push is more stable but adds delay."
        ),
        "push_speed": (
            "Robot speed used once push is triggered.\n"
            "Larger → stronger push motion.\n"
            "Smaller → gentler push motion."
        ),
        "push_exit_value_offset": (
            "How much above push_value_threshold before exiting push mode (hysteresis).\n"
            "Larger → push mode is more 'sticky' once entered.\n"
            "Smaller → push mode exits more easily.\n"
            "Prevents push/swipe flickering at threshold boundary."
        ),
        "pinch_axis_deadband": (
            "Minimum left/right finger motion needed for pinch detection.\n"
            "Smaller → pinch reacts sooner.\n"
            "Larger → pinch needs clearer motion."
        ),
        "pinch_distance_threshold": (
            "Minimum span reduction required to classify a pinch as pull.\n"
            "Smaller → pull triggers more easily.\n"
            "Larger → requires a more obvious pinch."
        ),
        "pinch_midpoint_deadband": (
            "How much the pinch midpoint may drift while still counting as a pull.\n"
            "Smaller → stricter pinch.\n"
            "Larger → more tolerant of hand drift."
        ),
        "pinch_frames_required": (
            "How many frames must satisfy the pinch rule before pull starts.\n"
            "Smaller → faster pull trigger.\n"
            "Larger → more stable but slower."
        ),
        "pull_speed": (
            "Robot speed used for pull after pinch detection.\n"
            "Larger → faster pull.\n"
            "Smaller → gentler pull."
        ),
        "rotation_speed": (
            "Robot angular speed used for two-finger swipe rotation.\n"
            "Larger → stronger rotation.\n"
            "Smaller → gentler rotation."
        ),
        "two_finger_swipe_deadband": (
            "Minimum two-finger midpoint motion needed before rotation begins.\n"
            "Smaller → more sensitive.\n"
            "Larger → steadier but slower to react."
        ),
        "two_finger_swipe_dominance_ratio": (
            "How much one axis must dominate before classifying a two-finger swipe direction.\n"
            "Larger → more selective direction locking.\n"
            "Smaller → easier to trigger either axis."
        ),
        "two_finger_swipe_axis_lock_frames": (
            "How many frames the chosen two-finger swipe axis stays locked before switching is allowed.\n"
            "Larger → horizontal/vertical is steadier and less likely to flip.\n"
            "Smaller → direction can switch more quickly."
        ),
        "two_finger_release_grace_frames": (
            "How many frames the controller waits after losing a two-finger contact.\n"
            "Smaller → stops faster.\n"
            "Larger → smoother transitions, but more delay."
        ),
        "frame_interval_ms": (
            "Timer interval for the DFM loop.\n"
            "0 means run as fast as the event loop allows.\n"
            "Larger values reduce CPU usage but increase control latency."
        ),
    }

    def _build_direct_finger_motion_settings_dialog(self):
        self.direct_finger_motion_settings_dialog = QDialog(self)
        self.direct_finger_motion_settings_dialog.setWindowTitle("Direct Finger Motion Parameters")
        self.direct_finger_motion_settings_dialog.resize(760, 560)
        self.direct_finger_motion_settings_dialog.setStyleSheet(
            "QToolTip {"
            " color: #111111;"
            " background-color: #fff7cc;"
            " border: 1px solid #5f5f5f;"
            " padding: 6px;"
            "}"
        )

        dialog_layout = QVBoxLayout(self.direct_finger_motion_settings_dialog)

        header_layout = QHBoxLayout()
        self.direct_finger_motion_logo = QLabel("🖐")
        self.direct_finger_motion_logo.setAlignment(Qt.AlignCenter)
        self.direct_finger_motion_logo.setFixedSize(56, 56)
        self.direct_finger_motion_logo.setStyleSheet(
            "font-size: 28px; border: 1px solid #8c8c8c; border-radius: 12px; background: rgba(255,255,255,0.08);"
        )

        header_text_layout = QVBoxLayout()
        self.direct_finger_motion_title_label = QLabel("Direct Finger Motion Control Panel")
        self.direct_finger_motion_title_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.direct_finger_motion_subtitle_label = QLabel(
            "Tune and save DFM parameters without editing the script."
        )
        self.direct_finger_motion_subtitle_label.setStyleSheet("color: #b0b0b0;")
        header_text_layout.addWidget(self.direct_finger_motion_title_label)
        header_text_layout.addWidget(self.direct_finger_motion_subtitle_label)
        header_text_layout.addStretch()

        header_layout.addWidget(self.direct_finger_motion_logo)
        header_layout.addLayout(header_text_layout)
        header_layout.addStretch()
        dialog_layout.addLayout(header_layout)

        self.direct_finger_motion_settings_group = QGroupBox("DFM Parameters")
        panel_layout = QVBoxLayout(self.direct_finger_motion_settings_group)
        grid = QGridLayout()

        self.direct_finger_motion_inputs = {}

        def add_double(name, label, row, col, minimum, maximum, step, decimals=4):
            widget = QDoubleSpinBox()
            widget.setDecimals(decimals)
            widget.setRange(minimum, maximum)
            widget.setSingleStep(step)
            widget.setMinimumWidth(120)
            label_widget = QLabel(label)
            tooltip = self.DFM_PARAMETER_TOOLTIPS.get(name, "")
            if tooltip:
                label_widget.setToolTip(tooltip)
                widget.setToolTip(tooltip)
            grid.addWidget(label_widget, row, col)
            grid.addWidget(widget, row, col + 1)
            self.direct_finger_motion_inputs[name] = widget

        def add_int(name, label, row, col, minimum, maximum, step=1):
            widget = QSpinBox()
            widget.setRange(minimum, maximum)
            widget.setSingleStep(step)
            widget.setMinimumWidth(120)
            label_widget = QLabel(label)
            tooltip = self.DFM_PARAMETER_TOOLTIPS.get(name, "")
            if tooltip:
                label_widget.setToolTip(tooltip)
                widget.setToolTip(tooltip)
            grid.addWidget(label_widget, row, col)
            grid.addWidget(widget, row, col + 1)
            self.direct_finger_motion_inputs[name] = widget

        add_double("motion_threshold", "Motion Threshold", 0, 0, -1000.0, 1000.0, 0.1, 3)
        add_int("no_touch_reset_limit", "No-touch Reset Frames", 1, 0, 0, 999)
        add_double("keep_margin", "Keep Margin", 2, 0, 0.0, 100.0, 0.05, 3)
        add_double("robot_speed", "Robot Speed", 3, 0, 0.0, 10.0, 0.01, 4)
        add_double("centroid_deadband", "Centroid Deadband", 4, 0, 0.0, 10.0, 0.001, 4)
        add_double("centroid_gain", "Centroid Gain", 5, 0, 0.0, 100.0, 0.1, 3)
        add_double("min_speed_ratio", "Min Speed Ratio", 6, 0, 0.0, 100.0, 0.05, 3)
        add_double("max_speed_ratio", "Max Speed Ratio", 7, 0, 0.0, 100.0, 0.05, 3)
        add_double("push_value_threshold", "Push Value Threshold", 0, 2, -1000.0, 1000.0, 0.5, 3)
        add_double("push_hold_deadband", "Push Hold Deadband", 1, 2, 0.0, 100.0, 0.01, 3)
        add_int("push_hold_frames_required", "Push Hold Frames", 2, 2, 0, 999)
        add_double("push_speed", "Push Speed", 3, 2, 0.0, 10.0, 0.01, 4)
        add_double("pinch_axis_deadband", "Pinch Axis Deadband", 4, 2, 0.0, 100.0, 0.001, 4)
        add_double("pinch_distance_threshold", "Pinch Distance Threshold", 5, 2, 0.0, 100.0, 0.005, 4)
        add_double("pinch_midpoint_deadband", "Pinch Midpoint Deadband", 6, 2, 0.0, 100.0, 0.05, 3)
        add_int("pinch_frames_required", "Pinch Frames", 7, 2, 0, 999)
        add_double("velocity_smoothing_alpha", "Velocity Smoothing α", 8, 0, 0.0, 1.0, 0.05, 2)
        add_double("push_exit_value_offset", "Push Exit Offset", 8, 2, 0.0, 50.0, 0.5, 1)
        add_double("pull_speed", "Pull Speed", 9, 0, 0.0, 10.0, 0.01, 4)
        add_double("rotation_speed", "Rotation Speed", 9, 2, 0.0, 10.0, 0.001, 4)
        add_double("two_finger_swipe_deadband", "2-Finger Swipe Deadband", 10, 0, 0.0, 100.0, 0.01, 3)
        add_double("two_finger_swipe_dominance_ratio", "2-Finger Swipe Dominance", 10, 2, 0.0, 100.0, 0.01, 3)
        add_int("two_finger_release_grace_frames", "2-Finger Release Grace", 11, 0, 0, 999)
        add_int("two_finger_swipe_axis_lock_frames", "2-Finger Axis Lock Frames", 11, 2, 0, 999)
        add_int("frame_interval_ms", "Timer Interval (ms)", 12, 2, 0, 10000)

        panel_layout.addLayout(grid)

        button_row = QHBoxLayout()
        self.apply_direct_finger_motion_settings_button = QPushButton("Apply DFM Params")
        self.reload_direct_finger_motion_settings_button = QPushButton("Reload Saved Params")
        button_row.addWidget(self.apply_direct_finger_motion_settings_button)
        button_row.addWidget(self.reload_direct_finger_motion_settings_button)
        button_row.addStretch()
        panel_layout.addLayout(button_row)

        dialog_layout.addWidget(self.direct_finger_motion_settings_group)
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self._close_direct_finger_motion_settings_dialog)
        button_box.accepted.connect(self._close_direct_finger_motion_settings_dialog)
        dialog_layout.addWidget(button_box)

        self._ensure_direct_finger_motion_tool_pose_timer()
        self._load_direct_finger_motion_settings_into_ui()

    def _build_direct_finger_motion_v2_settings_dialog(self):
        self.direct_finger_motion_v2_settings_dialog = QDialog(self)
        self.direct_finger_motion_v2_settings_dialog.setWindowTitle("Direct Finger Motion V2 Parameters")
        self.direct_finger_motion_v2_settings_dialog.resize(720, 520)

        dialog_layout = QVBoxLayout(self.direct_finger_motion_v2_settings_dialog)
        title = QLabel("Direct Finger Motion (Version 2) Parameters")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        subtitle = QLabel("Tune two-finger span control: pinch inward to pull, spread outward to move away.")
        subtitle.setStyleSheet("color: #b0b0b0;")
        subtitle.setWordWrap(True)
        dialog_layout.addWidget(title)
        dialog_layout.addWidget(subtitle)

        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Sensor Profile"))
        self.direct_finger_motion_v2_profile_combo = QComboBox()
        self.direct_finger_motion_v2_profile_combo.setEditable(True)
        self.direct_finger_motion_v2_profile_combo.setMinimumWidth(220)
        self.direct_finger_motion_v2_profile_combo.setToolTip(
            "Choose or type a profile name. Apply saves the current parameters to that profile."
        )
        self.load_direct_finger_motion_v2_profile_button = QPushButton("Load Profile")
        profile_row.addWidget(self.direct_finger_motion_v2_profile_combo)
        profile_row.addWidget(self.load_direct_finger_motion_v2_profile_button)
        profile_row.addStretch()
        dialog_layout.addLayout(profile_row)

        group = QGroupBox("DFM V2 Parameters")
        panel_layout = QVBoxLayout(group)
        grid = QGridLayout()
        self.direct_finger_motion_v2_inputs = {}

        def add_double(name, label, row, col, minimum, maximum, step, decimals=4, tooltip=""):
            widget = QDoubleSpinBox()
            widget.setDecimals(decimals)
            widget.setRange(minimum, maximum)
            widget.setSingleStep(step)
            widget.setMinimumWidth(120)
            label_widget = QLabel(label)
            if tooltip:
                label_widget.setToolTip(tooltip)
                widget.setToolTip(tooltip)
            grid.addWidget(label_widget, row, col)
            grid.addWidget(widget, row, col + 1)
            self.direct_finger_motion_v2_inputs[name] = widget

        def add_int(name, label, row, col, minimum, maximum, step=1, tooltip=""):
            widget = QSpinBox()
            widget.setRange(minimum, maximum)
            widget.setSingleStep(step)
            widget.setMinimumWidth(120)
            label_widget = QLabel(label)
            if tooltip:
                label_widget.setToolTip(tooltip)
                widget.setToolTip(tooltip)
            grid.addWidget(label_widget, row, col)
            grid.addWidget(widget, row, col + 1)
            self.direct_finger_motion_v2_inputs[name] = widget

        add_double("motion_threshold", "Touch Threshold", 0, 0, -1000.0, 1000.0, 0.1, 3, "Sensor threshold used to detect touched electrodes.")
        add_double("pull_speed", "Pull Speed", 1, 0, 0.0, 1.0, 0.01, 4, "Fixed speed when two fingers move toward center.")
        add_double("push_speed", "Move-Away Speed", 2, 0, 0.0, 1.0, 0.01, 4, "Fixed speed when two fingers move away from center.")
        add_double("robot_speed", "Planar Move Speed", 3, 0, 0.0, 1.0, 0.01, 4, "Speed for two fingers moving together left/right/up/down.")
        add_double("v2_span_deadband", "Two-Finger Span Deadband", 4, 0, 0.0, 1.0, 0.001, 4, "Minimum span change before v2 switches between pull and move-away.")
        add_double("v2_midpoint_deadband", "Midpoint Drift Deadband", 5, 0, 0.0, 1.0, 0.005, 4, "Allowed two-finger midpoint movement during pinch/spread detection.")
        add_double("v2_planar_span_tolerance", "Planar Span Tolerance", 6, 0, 0.0, 1.0, 0.005, 4, "Allowed two-finger span noise while detecting up/down/left/right movement.")
        add_double("v2_rotation_speed", "Cylinder Rotation Speed", 7, 0, 0.0, 1.0, 0.01, 4, "Angular speed for left-up/right-down cylinder rotation gestures.")
        add_double("v2_rotation_deadband", "Cylinder Rotation Deadband", 8, 0, 0.0, 1.0, 0.001, 4, "Minimum left/right vertical difference before rotation triggers.")
        add_double("v2_rotation_direction_sign", "Rotation Direction Sign", 9, 0, -1.0, 1.0, 1.0, 0, "Use -1 if clockwise/counter-clockwise is reversed on the robot.")
        add_double("v2_force_lateral_speed", "Side Press Lateral Speed", 10, 0, 0.0, 1.0, 0.01, 4, "Sideways speed when pressing the left or right side.")
        add_double("v2_force_lateral_deadband", "Side Press Force Threshold", 11, 0, 0.0, 100.0, 0.1, 3, "Minimum side press force before sideways motion triggers.")
        add_double("v2_force_lateral_center_deadband", "Side Press Center Deadband", 12, 0, 0.0, 0.5, 0.01, 3, "How far from the sensor center a single finger must be before side press triggers.")
        add_double("v2_force_lateral_direction_sign", "Side Press Direction Sign", 13, 0, -1.0, 1.0, 1.0, 0, "Use -1 if left-side / right-side sideways direction is reversed.")
        add_double("centroid_deadband", "Planar Motion Deadband", 0, 2, 0.0, 1.0, 0.001, 4, "Minimum two-finger center movement before left/right/up/down motion triggers.")
        add_double("centroid_gain", "Planar Motion Gain", 1, 2, 0.0, 100.0, 0.1, 3, "Sensitivity of two-finger center movement to planar robot speed.")
        add_double("v2_planar_dominance_ratio", "Planar Dominance Ratio", 2, 2, 0.0, 5.0, 0.1, 2, "Lower values make two-finger center movement win over span noise more easily.")
        add_double("v2_up_down_direction_sign", "Up/Down Direction Sign", 3, 2, -1.0, 1.0, 1.0, 0, "Use -1 if finger up/down makes the robot move in the reversed vertical direction.")
        add_double("v2_forward_backward_direction_sign", "Forward/Backward Direction Sign", 4, 2, -1.0, 1.0, 1.0, 0, "Use -1 if pinch/spread forward-backward motion is reversed.")
        add_double("velocity_smoothing_alpha", "Velocity Smoothing Alpha", 5, 2, 0.0, 1.0, 0.05, 2, "1.0 is most responsive; lower values smooth commands.")
        add_double("pinch_axis_deadband", "Finger Axis Deadband", 6, 2, 0.0, 1.0, 0.001, 4, "Compatibility threshold from DFM v1 two-finger detection.")
        add_double("pinch_distance_threshold", "Pinch Distance Threshold", 7, 2, 0.0, 1.0, 0.001, 4, "Compatibility distance threshold from DFM v1.")
        add_double("pinch_midpoint_deadband", "Pinch Midpoint Deadband", 8, 2, 0.0, 10.0, 0.05, 3, "Compatibility midpoint threshold from DFM v1.")
        add_int("frame_interval_ms", "Timer Interval (ms)", 9, 2, 0, 1000, tooltip="0 means run as fast as Qt event loop allows.")

        panel_layout.addLayout(grid)
        button_row = QHBoxLayout()
        self.apply_direct_finger_motion_v2_settings_button = QPushButton("Apply DFM V2 Params")
        self.reload_direct_finger_motion_v2_settings_button = QPushButton("Reload Saved Params")
        button_row.addWidget(self.apply_direct_finger_motion_v2_settings_button)
        button_row.addWidget(self.reload_direct_finger_motion_v2_settings_button)
        button_row.addStretch()
        panel_layout.addLayout(button_row)
        dialog_layout.addWidget(group)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.direct_finger_motion_v2_settings_dialog.close)
        button_box.accepted.connect(self.direct_finger_motion_v2_settings_dialog.close)
        dialog_layout.addWidget(button_box)
        self._load_direct_finger_motion_v2_settings_into_ui()

    def open_direct_finger_motion_settings_dialog(self):
        if not hasattr(self, "direct_finger_motion_settings_dialog"):
            self._build_direct_finger_motion_settings_dialog()
        self._load_direct_finger_motion_settings_into_ui()
        self.direct_finger_motion_settings_dialog.show()
        self.direct_finger_motion_settings_dialog.raise_()
        self.direct_finger_motion_settings_dialog.activateWindow()

    def open_direct_finger_motion_v2_settings_dialog(self):
        if not hasattr(self, "direct_finger_motion_v2_settings_dialog"):
            self._build_direct_finger_motion_v2_settings_dialog()
        self._refresh_direct_finger_motion_v2_profiles()
        self._load_direct_finger_motion_v2_settings_into_ui()
        self.direct_finger_motion_v2_settings_dialog.show()
        self.direct_finger_motion_v2_settings_dialog.raise_()
        self.direct_finger_motion_v2_settings_dialog.activateWindow()

    def _build_console_control_settings_dialog(self):
        self.console_control_settings_dialog = QDialog(self)
        self.console_control_settings_dialog.setWindowTitle("Console Control Parameters")
        self.console_control_settings_dialog.resize(720, 520)

        dialog_layout = QVBoxLayout(self.console_control_settings_dialog)
        title = QLabel("Console Control Parameters")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        subtitle = QLabel("Tune PS5/Linux joystick mapping, speeds, deadband, and direction signs.")
        subtitle.setStyleSheet("color: #b0b0b0;")
        subtitle.setWordWrap(True)
        dialog_layout.addWidget(title)
        dialog_layout.addWidget(subtitle)

        group = QGroupBox("Controller Mapping and Motion")
        panel_layout = QVBoxLayout(group)
        grid = QGridLayout()
        self.console_control_inputs = {}

        def add_double(name, label, row, col, minimum, maximum, step, decimals=4, tooltip=""):
            widget = QDoubleSpinBox()
            widget.setDecimals(decimals)
            widget.setRange(minimum, maximum)
            widget.setSingleStep(step)
            widget.setMinimumWidth(120)
            label_widget = QLabel(label)
            if tooltip:
                label_widget.setToolTip(tooltip)
                widget.setToolTip(tooltip)
            grid.addWidget(label_widget, row, col)
            grid.addWidget(widget, row, col + 1)
            self.console_control_inputs[name] = widget

        def add_int(name, label, row, col, minimum, maximum, step=1, tooltip=""):
            widget = QSpinBox()
            widget.setRange(minimum, maximum)
            widget.setSingleStep(step)
            widget.setMinimumWidth(120)
            label_widget = QLabel(label)
            if tooltip:
                label_widget.setToolTip(tooltip)
                widget.setToolTip(tooltip)
            grid.addWidget(label_widget, row, col)
            grid.addWidget(widget, row, col + 1)
            self.console_control_inputs[name] = widget

        add_int("console_device_index", "Joystick Device Index", 0, 0, 0, 9, tooltip="/dev/input/jsN device number.")
        add_double("console_deadband", "Joystick Deadband", 1, 0, 0.0, 0.9, 0.01, 3, "Ignore small joystick noise around center.")
        add_double("console_x_speed", "X Speed", 2, 0, 0.0, 1.0, 0.01, 4, "Maximum TCP speed for x motion.")
        add_double("console_y_speed", "Y Speed", 3, 0, 0.0, 1.0, 0.01, 4, "Maximum TCP speed for y motion.")
        add_double("console_z_speed", "Z Speed", 4, 0, 0.0, 1.0, 0.01, 4, "Maximum TCP speed for z motion.")
        add_double("console_rx_speed", "RX Rotation Speed", 5, 0, 0.0, 2.0, 0.01, 4, "Maximum TCP rotation speed around x.")
        add_double("console_ry_speed", "RY Rotation Speed", 6, 0, 0.0, 2.0, 0.01, 4, "Maximum TCP rotation speed around y.")
        add_double("console_rz_speed", "RZ Rotation Speed", 7, 0, 0.0, 2.0, 0.01, 4, "Maximum TCP rotation speed around z.")
        add_int("console_axis_left_x", "Left Stick X Axis", 8, 0, 0, 15)
        add_int("console_axis_left_y", "Left Stick Y Axis", 9, 0, 0, 15)
        add_int("console_axis_right_x", "Right Stick X Axis", 10, 0, 0, 15)
        add_int("console_axis_right_y", "Right Stick Y Axis", 11, 0, 0, 15)
        add_int("console_axis_l2", "L2 Axis", 12, 0, 0, 15)
        add_int("console_axis_r2", "R2 Axis", 13, 0, 0, 15)
        add_int("console_button_l1", "L1 Button", 14, 0, 0, 31)
        add_int("console_button_r1", "R1 Button", 15, 0, 0, 31)

        add_double("console_x_sign", "X Direction Sign", 0, 2, -1.0, 1.0, 1.0, 0, "Use -1 if x direction is reversed.")
        add_double("console_y_sign", "Y Direction Sign", 1, 2, -1.0, 1.0, 1.0, 0, "Use -1 if y direction is reversed.")
        add_double("console_z_sign", "Z Direction Sign", 2, 2, -1.0, 1.0, 1.0, 0, "Use -1 if L1/R1 z direction is reversed.")
        add_double("console_rx_sign", "RX Direction Sign", 3, 2, -1.0, 1.0, 1.0, 0, "Use -1 if rx direction is reversed.")
        add_double("console_ry_sign", "RY Direction Sign", 4, 2, -1.0, 1.0, 1.0, 0, "Use -1 if ry direction is reversed.")
        add_double("console_rz_sign", "RZ Direction Sign", 5, 2, -1.0, 1.0, 1.0, 0, "Use -1 if L2/R2 rz direction is reversed.")
        add_int("frame_interval_ms", "Timer Interval (ms)", 6, 2, 1, 200, tooltip="How often controller input is sent to the robot.")
        add_double("velocity_smoothing_alpha", "Velocity Smoothing Alpha", 7, 2, 0.0, 1.0, 0.05, 2, "1.0 is most responsive; lower values smooth commands.")

        panel_layout.addLayout(grid)
        button_row = QHBoxLayout()
        self.apply_console_control_settings_button = QPushButton("Apply Console Params")
        self.reload_console_control_settings_button = QPushButton("Reload Saved Params")
        button_row.addWidget(self.apply_console_control_settings_button)
        button_row.addWidget(self.reload_console_control_settings_button)
        button_row.addStretch()
        panel_layout.addLayout(button_row)
        dialog_layout.addWidget(group)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.console_control_settings_dialog.close)
        button_box.accepted.connect(self.console_control_settings_dialog.close)
        dialog_layout.addWidget(button_box)
        self._load_console_control_settings_into_ui()

    def open_console_control_settings_dialog(self):
        if not hasattr(self, "console_control_settings_dialog"):
            self._build_console_control_settings_dialog()
        self._load_console_control_settings_into_ui()
        self.console_control_settings_dialog.show()
        self.console_control_settings_dialog.raise_()
        self.console_control_settings_dialog.activateWindow()

    def _get_direct_finger_motion_helper(self):
        return self._get_sensor_helper("direct_finger_motion_class")

    def _get_direct_finger_motion_v2_helper(self):
        return self._get_sensor_helper("direct_finger_motion_v2_class")

    def _get_console_control_helper(self):
        return self._get_sensor_helper("console_control_class")

    def _get_ai_direct_finger_motion_helper(self):
        return self._get_sensor_helper("ai_direct_finger_motion_class")

    def _get_ai_direct_finger_motion_execution_helper(self):
        return self._get_sensor_helper("ai_direct_finger_motion_execution_class")

    def _ensure_direct_finger_motion_tool_pose_timer(self):
        if hasattr(self, "_direct_finger_motion_tool_pose_timer"):
            return

        self._direct_finger_motion_tool_pose_timer = QTimer(self)
        self._direct_finger_motion_tool_pose_timer.setInterval(
            self.DIRECT_FINGER_MOTION_TOOL_POSE_RECORD_INTERVAL_MS
        )
        self._direct_finger_motion_tool_pose_timer.timeout.connect(
            self._record_direct_finger_motion_tool_pose
        )
        self._direct_finger_motion_tool_pose_record_active = False
        self._direct_finger_motion_tool_pose_log_file = None
        self._direct_finger_motion_tool_pose_log_path = None
        self._direct_finger_motion_tool_pose_plot_actors = []
        self._direct_finger_motion_tool_pose_plot_static_actors = []
        self._direct_finger_motion_tool_pose_plot_dynamic_actors = []
        self._direct_finger_motion_tool_pose_plot_dialog = None
        self._direct_finger_motion_tool_pose_plotter = None
        self._direct_finger_motion_tool_pose_plot_points = None
        self._direct_finger_motion_tool_pose_animation_points = None
        self._direct_finger_motion_tool_pose_path_animation_index = 0
        self._direct_finger_motion_tool_pose_path_animation_timer = QTimer(self)
        self._direct_finger_motion_tool_pose_path_animation_timer.setInterval(
            self.DIRECT_FINGER_MOTION_TOOL_POSE_PLAYBACK_INTERVAL_MS
        )
        self._direct_finger_motion_tool_pose_path_animation_timer.timeout.connect(
            self._advance_direct_finger_motion_tool_pose_path_animation
        )

    def _close_direct_finger_motion_settings_dialog(self):
        self.direct_finger_motion_settings_dialog.close()

    def _set_direct_finger_motion_tool_pose_record_button_state(self):
        active = bool(getattr(self, "_direct_finger_motion_tool_pose_record_active", False))
        if hasattr(self, "direct_finger_motion_tool_pose_record_menu_button"):
            self.direct_finger_motion_tool_pose_record_menu_button.setText(
                "Stop Tool Pose Recording" if active else "Tool Pose Recording"
            )
            if hasattr(self, "_set_button_active"):
                self._set_button_active(self.direct_finger_motion_tool_pose_record_menu_button, active)

    def _append_direct_finger_motion_log(self, message):
        if hasattr(self, "log_display"):
            if not self.log_display.isVisible():
                self.log_display.setVisible(True)
                if hasattr(self, "adjust_splitter_sizes"):
                    self.adjust_splitter_sizes()
            self.log_display.append(message)
        else:
            print(message)

    def _toggle_direct_finger_motion_tool_pose_recording(self):
        if getattr(self, "_direct_finger_motion_tool_pose_record_active", False):
            self._stop_direct_finger_motion_tool_pose_recording()
        else:
            self._start_direct_finger_motion_tool_pose_recording()

    def _start_direct_finger_motion_tool_pose_recording(self):
        self._ensure_direct_finger_motion_tool_pose_timer()
        robot_api = getattr(self, "robot_api", None)
        if robot_api is None or not hasattr(robot_api, "get_current_tool_position"):
            self._append_direct_finger_motion_log("[DFM] Robot API is unavailable: cannot record tool pose.")
            return

        try:
            os.makedirs(self.DIRECT_FINGER_MOTION_TOOL_POSE_LOG_DIR, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            self._direct_finger_motion_tool_pose_log_path = os.path.join(
                self.DIRECT_FINGER_MOTION_TOOL_POSE_LOG_DIR,
                f"tool_pose_record_{timestamp}.csv",
            )
            self._direct_finger_motion_tool_pose_log_file = open(
                self._direct_finger_motion_tool_pose_log_path,
                "w",
                encoding="utf-8",
            )
            self._direct_finger_motion_tool_pose_log_file.write(
                "timestamp,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z\n"
            )
            self._direct_finger_motion_tool_pose_log_file.flush()
        except Exception as exc:
            self._direct_finger_motion_tool_pose_log_file = None
            self._direct_finger_motion_tool_pose_log_path = None
            self._append_direct_finger_motion_log(
                f"[DFM] Failed to create tool pose log file: {exc}"
            )
            return

        self._direct_finger_motion_tool_pose_record_active = True
        self._set_direct_finger_motion_tool_pose_record_button_state()
        self._append_direct_finger_motion_log(
            f"[DFM] Tool pose recording started. Saving to: {self._direct_finger_motion_tool_pose_log_path}"
        )
        self._direct_finger_motion_tool_pose_timer.start()
        self._record_direct_finger_motion_tool_pose()

    def _stop_direct_finger_motion_tool_pose_recording(self):
        if hasattr(self, "_direct_finger_motion_tool_pose_timer"):
            self._direct_finger_motion_tool_pose_timer.stop()

        was_active = bool(getattr(self, "_direct_finger_motion_tool_pose_record_active", False))
        saved_path = getattr(self, "_direct_finger_motion_tool_pose_log_path", None)
        self._direct_finger_motion_tool_pose_record_active = False
        self._set_direct_finger_motion_tool_pose_record_button_state()
        log_file = getattr(self, "_direct_finger_motion_tool_pose_log_file", None)
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass
        self._direct_finger_motion_tool_pose_log_file = None
        self._direct_finger_motion_tool_pose_log_path = None
        if was_active:
            if saved_path:
                self._append_direct_finger_motion_log(
                    f"[DFM] Tool pose recording stopped. Saved file: {saved_path}"
                )
            else:
                self._append_direct_finger_motion_log("[DFM] Tool pose recording stopped.")

    def _record_direct_finger_motion_tool_pose(self):
        robot_api = getattr(self, "robot_api", None)
        if robot_api is None or not hasattr(robot_api, "get_current_tool_position"):
            self._stop_direct_finger_motion_tool_pose_recording()
            self._append_direct_finger_motion_log("[DFM] Robot API became unavailable: stopped tool pose recording.")
            return

        pos_quat = robot_api.get_current_tool_position()
        if not pos_quat or pos_quat == (None, None):
            self._append_direct_finger_motion_log("[DFM] Tool pose unavailable yet.")
            return

        position, quaternion = pos_quat
        if position is None or quaternion is None:
            self._append_direct_finger_motion_log("[DFM] Tool pose unavailable yet.")
            return

        timestamp = time.time()
        log_file = getattr(self, "_direct_finger_motion_tool_pose_log_file", None)
        if log_file is not None:
            try:
                log_file.write(
                    f"{timestamp:.6f},"
                    f"{position[0]:.6f},{position[1]:.6f},{position[2]:.6f},"
                    f"{quaternion[0]:.6f},{quaternion[1]:.6f},{quaternion[2]:.6f},{quaternion[3]:.6f}\n"
                )
                log_file.flush()
            except Exception as exc:
                self._append_direct_finger_motion_log(
                    f"[DFM] Failed to write tool pose log file: {exc}"
                )
                self._stop_direct_finger_motion_tool_pose_recording()
                return

        self._append_direct_finger_motion_log(
            "[DFM] Tool pose | "
            f"pos=({position[0]:+.4f}, {position[1]:+.4f}, {position[2]:+.4f}) | "
            f"quat=({quaternion[0]:+.4f}, {quaternion[1]:+.4f}, {quaternion[2]:+.4f}, {quaternion[3]:+.4f})"
        )

    def _load_direct_finger_motion_tool_pose_path_from_dialog(self):
        start_dir = self.DIRECT_FINGER_MOTION_TOOL_POSE_LOG_DIR
        if not os.path.isdir(start_dir):
            start_dir = os.path.expanduser("~")

        dialog = QFileDialog(self, "Load Tool Pose Path", start_dir, "CSV Files (*.csv);;All Files (*)")
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setViewMode(QFileDialog.Detail)

        if dialog.exec_() != QFileDialog.Accepted:
            return

        selected_files = dialog.selectedFiles()
        if not selected_files:
            return
        self._load_direct_finger_motion_tool_pose_path(selected_files[0])

    def _load_direct_finger_motion_tool_pose_path(self, csv_path):
        try:
            points = self._read_direct_finger_motion_tool_pose_points(csv_path)
        except Exception as exc:
            self._append_direct_finger_motion_log(
                f"[DFM] Failed to load tool pose path file: {exc}"
            )
            return

        if not points:
            self._append_direct_finger_motion_log(
                f"[DFM] No tool pose samples found in file: {csv_path}"
            )
            return

        plotter = self._ensure_direct_finger_motion_tool_pose_plot_window()
        if plotter is None:
            self._append_direct_finger_motion_log(
                "[DFM] 3D popup plotter is unavailable: cannot draw tool pose path."
            )
            return
        points_np = self._compact_direct_finger_motion_tool_pose_points(
            np.asarray(points, dtype=float)
        )
        self._direct_finger_motion_tool_pose_plot_points = points_np
        self._direct_finger_motion_tool_pose_animation_points = (
            self._build_direct_finger_motion_tool_pose_animation_points(points_np)
        )
        self._stop_direct_finger_motion_tool_pose_path_animation(reset_button=False)
        self._render_direct_finger_motion_tool_pose_path_static(points_np)
        self._render_direct_finger_motion_tool_pose_path_frame(points_np[:1])
        self._set_direct_finger_motion_tool_pose_play_button_enabled(True)

        try:
            self._direct_finger_motion_tool_pose_plot_dialog.show()
            self._direct_finger_motion_tool_pose_plot_dialog.raise_()
            self._direct_finger_motion_tool_pose_plot_dialog.activateWindow()
            plotter.reset_camera()
            plotter.render()
        except Exception:
            pass

        self._append_direct_finger_motion_log(
            f"[DFM] Loaded tool pose path in popup 3D graph: {csv_path} ({len(points_np)} points)"
        )

    def _read_direct_finger_motion_tool_pose_points(self, csv_path):
        points = []
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                points.append(
                    (
                        float(row["pos_x"]),
                        float(row["pos_y"]),
                        float(row["pos_z"]),
                    )
                )
        return points

    def _compact_direct_finger_motion_tool_pose_points(self, points_np):
        if points_np is None or len(points_np) == 0:
            return points_np

        axis_min = np.min(points_np, axis=0)
        shifted_points = points_np - axis_min
        max_span = float(np.max(np.ptp(shifted_points, axis=0)))
        if max_span <= 1e-12:
            return shifted_points
        return shifted_points / max_span

    def _clear_direct_finger_motion_tool_pose_path_plot(
        self,
        log_message=True,
        clear_loaded_path=True,
        stop_animation=True,
        reset_animation_index=True,
    ):
        if stop_animation:
            self._stop_direct_finger_motion_tool_pose_path_animation(reset_button=False)
        plotter = getattr(self, "_direct_finger_motion_tool_pose_plotter", None)
        actors = list(getattr(self, "_direct_finger_motion_tool_pose_plot_actors", []))
        if plotter is not None:
            for actor in actors:
                try:
                    plotter.remove_actor(actor)
                except Exception:
                    pass
            try:
                plotter.render()
            except Exception:
                pass
        self._direct_finger_motion_tool_pose_plot_actors = []
        self._direct_finger_motion_tool_pose_plot_static_actors = []
        self._direct_finger_motion_tool_pose_plot_dynamic_actors = []
        if reset_animation_index:
            self._direct_finger_motion_tool_pose_path_animation_index = 0
        if clear_loaded_path:
            self._direct_finger_motion_tool_pose_plot_points = None
            self._direct_finger_motion_tool_pose_animation_points = None
            self._set_direct_finger_motion_tool_pose_play_button_enabled(False)
        if log_message:
            self._append_direct_finger_motion_log("[DFM] Cleared tool pose path from 3D graph.")

    def _ensure_direct_finger_motion_tool_pose_plot_window(self):
        dialog = getattr(self, "_direct_finger_motion_tool_pose_plot_dialog", None)
        plotter = getattr(self, "_direct_finger_motion_tool_pose_plotter", None)
        if dialog is not None and plotter is not None:
            try:
                dialog.windowTitle()
                return plotter
            except RuntimeError:
                self._direct_finger_motion_tool_pose_plot_dialog = None
                self._direct_finger_motion_tool_pose_plotter = None

        dialog = QDialog(self)
        dialog.setWindowTitle("Tool Pose Path Viewer")
        dialog.resize(900, 700)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(6, 6, 6, 6)

        control_row = QHBoxLayout()
        self.direct_finger_motion_tool_pose_play_button = QPushButton("Play Path Motion", dialog)
        self.direct_finger_motion_tool_pose_play_button.clicked.connect(
            self._play_direct_finger_motion_tool_pose_path_animation
        )
        self.direct_finger_motion_tool_pose_play_button.setEnabled(False)
        control_row.addWidget(self.direct_finger_motion_tool_pose_play_button)
        control_row.addStretch()
        layout.addLayout(control_row)

        plotter = QtInteractor(dialog)
        plotter.background_color = "#202020"
        layout.addWidget(plotter.interactor)
        try:
            plotter.add_axes()
            plotter.show_grid(
                xtitle="X",
                ytitle="Y",
                ztitle="Z",
                color="white",
            )
        except Exception:
            pass

        dialog.finished.connect(self._on_direct_finger_motion_tool_pose_plot_dialog_closed)

        self._direct_finger_motion_tool_pose_plot_dialog = dialog
        self._direct_finger_motion_tool_pose_plotter = plotter
        return plotter

    def _set_direct_finger_motion_tool_pose_play_button_enabled(self, enabled: bool):
        button = getattr(self, "direct_finger_motion_tool_pose_play_button", None)
        if button is not None:
            button.setEnabled(bool(enabled))
            if enabled:
                button.setText("Play Path Motion")

    def _build_direct_finger_motion_tool_pose_animation_points(self, points_np):
        if points_np is None or len(points_np) <= 1:
            return points_np

        target_frames = max(
            2,
            int(
                self.DIRECT_FINGER_MOTION_TOOL_POSE_PLAYBACK_DURATION_SEC * 1000
                / self.DIRECT_FINGER_MOTION_TOOL_POSE_PLAYBACK_INTERVAL_MS
            ),
        )
        deltas = np.diff(points_np, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = float(cumulative_lengths[-1])

        if total_length <= 1e-12:
            source_idx = np.linspace(0, len(points_np) - 1, target_frames)
            lower = np.floor(source_idx).astype(int)
            upper = np.ceil(source_idx).astype(int)
            blend = (source_idx - lower)[:, None]
            return points_np[lower] * (1.0 - blend) + points_np[upper] * blend

        sample_lengths = np.linspace(0.0, total_length, target_frames)
        segment_idx = np.searchsorted(cumulative_lengths, sample_lengths, side="right") - 1
        segment_idx = np.clip(segment_idx, 0, len(points_np) - 2)

        start_lengths = cumulative_lengths[segment_idx]
        end_lengths = cumulative_lengths[segment_idx + 1]
        denom = np.maximum(end_lengths - start_lengths, 1e-12)
        blend = ((sample_lengths - start_lengths) / denom)[:, None]
        return points_np[segment_idx] * (1.0 - blend) + points_np[segment_idx + 1] * blend

    def _clear_direct_finger_motion_tool_pose_dynamic_actors(self):
        plotter = getattr(self, "_direct_finger_motion_tool_pose_plotter", None)
        actors = list(getattr(self, "_direct_finger_motion_tool_pose_plot_dynamic_actors", []))
        if plotter is None:
            return

        for actor in actors:
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
        self._direct_finger_motion_tool_pose_plot_dynamic_actors = []
        self._direct_finger_motion_tool_pose_plot_actors = (
            list(getattr(self, "_direct_finger_motion_tool_pose_plot_static_actors", []))
        )

    def _render_direct_finger_motion_tool_pose_path_static(self, points_np, show_full_path=True):
        plotter = getattr(self, "_direct_finger_motion_tool_pose_plotter", None)
        if plotter is None or points_np is None or len(points_np) == 0:
            return

        self._clear_direct_finger_motion_tool_pose_path_plot(
            log_message=False,
            clear_loaded_path=False,
            stop_animation=False,
            reset_animation_index=False,
        )

        actors = []
        if show_full_path and len(points_np) >= 2:
            full_path_mesh = pv.lines_from_points(points_np)
            actors.append(
                plotter.add_mesh(
                    full_path_mesh,
                    color="#4fc3f7",
                    line_width=4,
                    name="dfm_tool_pose_path_full_line",
                )
            )
        actors.append(
            plotter.add_mesh(
                pv.PolyData(points_np[:1]),
                color="#66bb6a",
                point_size=14,
                render_points_as_spheres=True,
                name="dfm_tool_pose_path_start",
            )
        )
        actors.append(
            plotter.add_mesh(
                pv.PolyData(points_np[-1:]),
                color="#ef5350",
                point_size=14,
                render_points_as_spheres=True,
                name="dfm_tool_pose_path_end",
            )
        )
        self._direct_finger_motion_tool_pose_plot_static_actors = actors
        self._direct_finger_motion_tool_pose_plot_actors = list(actors)

        try:
            plotter.render()
        except Exception:
            pass

    def _render_direct_finger_motion_tool_pose_path_frame(self, current_points):
        plotter = getattr(self, "_direct_finger_motion_tool_pose_plotter", None)
        if plotter is None or current_points is None or len(current_points) == 0:
            return

        self._clear_direct_finger_motion_tool_pose_dynamic_actors()

        actors = []
        if len(current_points) >= 2:
            progress_mesh = pv.lines_from_points(current_points)
            actors.append(
                plotter.add_mesh(
                    progress_mesh,
                    color="#4fc3f7",
                    line_width=5,
                    name="dfm_tool_pose_path_progress_line",
                )
            )
        actors.append(
            plotter.add_mesh(
                pv.PolyData(current_points[-1:]),
                color="#ffd54f",
                point_size=18,
                render_points_as_spheres=True,
                name="dfm_tool_pose_path_current",
            )
        )
        self._direct_finger_motion_tool_pose_plot_dynamic_actors = actors
        self._direct_finger_motion_tool_pose_plot_actors = (
            list(getattr(self, "_direct_finger_motion_tool_pose_plot_static_actors", [])) + actors
        )

        try:
            plotter.render()
        except Exception:
            pass

    def _play_direct_finger_motion_tool_pose_path_animation(self):
        points_np = getattr(self, "_direct_finger_motion_tool_pose_animation_points", None)
        if points_np is None or len(points_np) == 0:
            self._append_direct_finger_motion_log("[DFM] No loaded path to animate yet.")
            return

        if len(points_np) == 1:
            original_points = getattr(self, "_direct_finger_motion_tool_pose_plot_points", points_np)
            self._render_direct_finger_motion_tool_pose_path_static(
                original_points,
                show_full_path=False,
            )
            self._render_direct_finger_motion_tool_pose_path_frame(points_np[:1])
            self._append_direct_finger_motion_log("[DFM] Tool pose path has only one point.")
            return

        self._stop_direct_finger_motion_tool_pose_path_animation(reset_button=False)
        self._direct_finger_motion_tool_pose_path_animation_index = 1
        original_points = getattr(self, "_direct_finger_motion_tool_pose_plot_points", points_np)
        self._render_direct_finger_motion_tool_pose_path_static(
            original_points,
            show_full_path=False,
        )
        self._render_direct_finger_motion_tool_pose_path_frame(points_np[:1])

        button = getattr(self, "direct_finger_motion_tool_pose_play_button", None)
        if button is not None:
            button.setText("Playing...")
            button.setEnabled(False)

        self._direct_finger_motion_tool_pose_path_animation_timer.start()

    def _advance_direct_finger_motion_tool_pose_path_animation(self):
        points_np = getattr(self, "_direct_finger_motion_tool_pose_animation_points", None)
        if points_np is None or len(points_np) == 0:
            self._stop_direct_finger_motion_tool_pose_path_animation()
            return

        next_count = self._direct_finger_motion_tool_pose_path_animation_index + 1
        self._direct_finger_motion_tool_pose_path_animation_index = next_count
        self._render_direct_finger_motion_tool_pose_path_frame(points_np[:next_count])

        if next_count >= len(points_np):
            self._stop_direct_finger_motion_tool_pose_path_animation()

    def _stop_direct_finger_motion_tool_pose_path_animation(self, reset_button=True):
        timer = getattr(self, "_direct_finger_motion_tool_pose_path_animation_timer", None)
        if timer is not None:
            timer.stop()

        if reset_button:
            button = getattr(self, "direct_finger_motion_tool_pose_play_button", None)
            points_np = getattr(self, "_direct_finger_motion_tool_pose_plot_points", None)
            if button is not None:
                button.setText("Play Path Motion")
                button.setEnabled(points_np is not None and len(points_np) > 0)

    def _on_direct_finger_motion_tool_pose_plot_dialog_closed(self, *_args):
        self._stop_direct_finger_motion_tool_pose_path_animation(reset_button=False)
        plotter = getattr(self, "_direct_finger_motion_tool_pose_plotter", None)
        if plotter is not None:
            try:
                plotter.close()
            except Exception:
                pass

        self._direct_finger_motion_tool_pose_plot_actors = []
        self._direct_finger_motion_tool_pose_plot_static_actors = []
        self._direct_finger_motion_tool_pose_plot_dynamic_actors = []
        self._direct_finger_motion_tool_pose_plot_points = None
        self._direct_finger_motion_tool_pose_animation_points = None
        self._direct_finger_motion_tool_pose_path_animation_index = 0
        self._direct_finger_motion_tool_pose_plotter = None
        self._direct_finger_motion_tool_pose_plot_dialog = None
        self.direct_finger_motion_tool_pose_play_button = None

    def _collect_direct_finger_motion_settings_from_ui(self):
        settings = {}
        for name, widget in self.direct_finger_motion_inputs.items():
            settings[name] = widget.value()
        return settings

    def _load_direct_finger_motion_settings_into_ui(self):
        helper = self._get_direct_finger_motion_helper()
        if helper is None or not hasattr(helper, "get_settings"):
            return

        try:
            settings = helper.get_settings()
            for name, widget in self.direct_finger_motion_inputs.items():
                if name in settings:
                    widget.setValue(settings[name])
        except Exception as exc:
            print(f"[UI] Failed to load direct finger motion settings into UI: {exc}")

    def _apply_direct_finger_motion_settings_from_ui(self):
        helper = self._get_direct_finger_motion_helper()
        if helper is None or not hasattr(helper, "apply_settings"):
            print("[UI] Direct finger motion helper is not ready yet.")
            return

        try:
            settings = self._collect_direct_finger_motion_settings_from_ui()
            helper.apply_settings(settings, save_to_file=True)
            print("[UI] Direct finger motion parameters applied.")
        except Exception as exc:
            print(f"[UI] Failed to apply direct finger motion settings: {exc}")

    def _collect_direct_finger_motion_v2_settings_from_ui(self):
        settings = {}
        for name, widget in self.direct_finger_motion_v2_inputs.items():
            settings[name] = widget.value()
        return settings

    def _current_direct_finger_motion_v2_profile_from_ui(self):
        combo = getattr(self, "direct_finger_motion_v2_profile_combo", None)
        if combo is None:
            return "default"
        return combo.currentText().strip() or "default"

    def _refresh_direct_finger_motion_v2_profiles(self):
        helper = self._get_direct_finger_motion_v2_helper()
        combo = getattr(self, "direct_finger_motion_v2_profile_combo", None)
        if helper is None or combo is None or not hasattr(helper, "list_profiles"):
            return

        current_profile = getattr(helper, "get_current_profile_name", lambda: "default")()
        profiles = helper.list_profiles()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(profiles)
        if current_profile not in profiles:
            combo.addItem(current_profile)
        combo.setCurrentText(current_profile)
        combo.blockSignals(False)

    def _load_direct_finger_motion_v2_profile_from_ui(self):
        helper = self._get_direct_finger_motion_v2_helper()
        if helper is None or not hasattr(helper, "set_profile"):
            print("[UI] Direct finger motion v2 helper is not ready yet.")
            return

        try:
            profile_name = self._current_direct_finger_motion_v2_profile_from_ui()
            active_profile = helper.set_profile(profile_name, load=True)
            self._refresh_direct_finger_motion_v2_profiles()
            if hasattr(self, "direct_finger_motion_v2_profile_combo"):
                self.direct_finger_motion_v2_profile_combo.setCurrentText(active_profile)
            self._load_direct_finger_motion_v2_settings_into_ui()
            print(f"[UI] Direct finger motion v2 profile loaded: {active_profile}")
        except Exception as exc:
            print(f"[UI] Failed to load direct finger motion v2 profile: {exc}")

    def _load_direct_finger_motion_v2_settings_into_ui(self):
        helper = self._get_direct_finger_motion_v2_helper()
        if helper is None or not hasattr(helper, "get_settings"):
            return

        try:
            if hasattr(helper, "set_profile"):
                helper.set_profile(self._current_direct_finger_motion_v2_profile_from_ui(), load=True)
            self._refresh_direct_finger_motion_v2_profiles()
            settings = helper.get_settings()
            for name, widget in self.direct_finger_motion_v2_inputs.items():
                if name in settings:
                    widget.setValue(settings[name])
        except Exception as exc:
            print(f"[UI] Failed to load direct finger motion v2 settings into UI: {exc}")

    def _apply_direct_finger_motion_v2_settings_from_ui(self):
        helper = self._get_direct_finger_motion_v2_helper()
        if helper is None or not hasattr(helper, "apply_settings"):
            print("[UI] Direct finger motion v2 helper is not ready yet.")
            return

        try:
            if hasattr(helper, "set_profile"):
                helper.set_profile(self._current_direct_finger_motion_v2_profile_from_ui(), load=False)
            settings = self._collect_direct_finger_motion_v2_settings_from_ui()
            helper.apply_settings(settings, save_to_file=True)
            self._refresh_direct_finger_motion_v2_profiles()
            print("[UI] Direct finger motion v2 parameters applied.")
        except Exception as exc:
            print(f"[UI] Failed to apply direct finger motion v2 settings: {exc}")

    def _collect_console_control_settings_from_ui(self):
        settings = {}
        for name, widget in self.console_control_inputs.items():
            settings[name] = widget.value()
        return settings

    def _load_console_control_settings_into_ui(self):
        helper = self._get_console_control_helper()
        if helper is None or not hasattr(helper, "get_settings"):
            return

        try:
            settings = helper.get_settings()
            for name, widget in self.console_control_inputs.items():
                if name in settings:
                    widget.setValue(settings[name])
        except Exception as exc:
            print(f"[UI] Failed to load console control settings into UI: {exc}")

    def _apply_console_control_settings_from_ui(self):
        helper = self._get_console_control_helper()
        if helper is None or not hasattr(helper, "apply_settings"):
            print("[UI] Console control helper is not ready yet.")
            return

        try:
            settings = self._collect_console_control_settings_from_ui()
            helper.apply_settings(settings, save_to_file=True)
            print("[UI] Console control parameters applied.")
        except Exception as exc:
            print(f"[UI] Failed to apply console control settings: {exc}")

    def _on_toggle_direct_finger_motion(self):
        self._direct_finger_active = not getattr(self, "_direct_finger_active", False)
        self._set_button_active(self.direct_finger_motion_button, self._direct_finger_active)

        try:
            helper = self._get_direct_finger_motion_helper()
            if helper is None:
                raise AttributeError("direct_finger_motion_class is not available")
            helper.toggle_direct_finger_motion()
        except Exception as exc:
            print(f"[UI] Direct finger motion toggle failed: {exc}")
            self._direct_finger_active = not self._direct_finger_active
            self._set_button_active(self.direct_finger_motion_button, self._direct_finger_active)

        self._update_anchor_button_label()

    def _on_toggle_ai_direct_finger_motion(self):
        self._ai_direct_finger_active = not getattr(self, "_ai_direct_finger_active", False)
        self._set_button_active(self.ai_direct_finger_motion_button, self._ai_direct_finger_active)

        session_tag = self.gesture_number_input.text().strip() if hasattr(self, "gesture_number_input") else ""

        try:
            helper = self._get_ai_direct_finger_motion_helper()
            if helper is None:
                raise AttributeError("ai_direct_finger_motion_class is not available")
            helper.toggle_ai_direct_finger_motion(
                session_tag=session_tag
            )
        except Exception as exc:
            print(f"[UI] AI direct finger motion toggle failed: {exc}")
            self._ai_direct_finger_active = not self._ai_direct_finger_active
            self._set_button_active(self.ai_direct_finger_motion_button, self._ai_direct_finger_active)

    def _on_toggle_ai_direct_finger_motion_execution(self):
        self._ai_direct_finger_execution_active = not getattr(
            self, "_ai_direct_finger_execution_active", False
        )
        self._set_button_active(
            self.ai_direct_finger_motion_execution_button,
            self._ai_direct_finger_execution_active,
        )

        model_path = ""
        if hasattr(self, "ai_direct_execution_model_path_input"):
            model_path = self.ai_direct_execution_model_path_input.text().strip()

        try:
            if hasattr(self.sensor_functions, "toggle_ai_direct_finger_motion_execution"):
                self.sensor_functions.toggle_ai_direct_finger_motion_execution(
                    model_checkpoint_path=model_path or None
                )
            else:
                helper = self._get_ai_direct_finger_motion_execution_helper()
                if helper is None:
                    raise AttributeError("ai_direct_finger_motion_execution_class is not available")
                helper.toggle_ai_direct_finger_motion_execution(
                    model_checkpoint_path=model_path or None
                )
        except Exception as exc:
            print(f"[UI] AI direct finger motion execution toggle failed: {exc}")
            self._ai_direct_finger_execution_active = not self._ai_direct_finger_execution_active
            self._set_button_active(
                self.ai_direct_finger_motion_execution_button,
                self._ai_direct_finger_execution_active,
            )
