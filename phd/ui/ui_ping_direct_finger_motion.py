from __future__ import annotations

import csv
import json
import os
import struct
import time

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from pyvistaqt import QtInteractor
from phd.dependence.paths import resource_path, robot_resource_path  # pyright: ignore[reportMissingImports]

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - optional plotting dependency
    FigureCanvas = None
    Figure = None


class DirectFingerMotionMixin:
    DIRECT_FINGER_MOTION_TOOL_POSE_RECORD_INTERVAL_MS = 200
    DIRECT_FINGER_MOTION_TOOL_POSE_PLAYBACK_INTERVAL_MS = 16
    DIRECT_FINGER_MOTION_TOOL_POSE_PLAYBACK_DURATION_SEC = 2.0
    DIRECT_FINGER_MOTION_PREVIEW_VIEW_ELEV_DEG = 35.264389682754654
    DIRECT_FINGER_MOTION_PREVIEW_VIEW_AZIM_DEG = 45.0
    DIRECT_FINGER_MOTION_TOOL_POSE_LOG_DIR = robot_resource_path("tool_pose_logs")
    DIRECT_FINGER_MOTION_TOOL_POSE_TARGET_ROTATION_FILE = resource_path(
        "config", "tool_pose_target_rotation.json"
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
        grid_flip_row = QHBoxLayout()
        self.sensor_grid_flip_lr_btn = QPushButton("Flip L/R")
        self.sensor_grid_flip_lr_btn.setCheckable(True)
        self.sensor_grid_flip_lr_btn.setToolTip(
            "Mirror the touch grid left-right. Saved in console_control.json with other console settings."
        )
        self.sensor_grid_flip_lr_btn.toggled.connect(self._on_sensor_touch_grid_flip_lr_toggled)
        self.sensor_grid_flip_ud_btn = QPushButton("Flip U/D")
        self.sensor_grid_flip_ud_btn.setCheckable(True)
        self.sensor_grid_flip_ud_btn.setToolTip(
            "Mirror the touch grid up-down. Saved in console_control.json with other console settings."
        )
        self.sensor_grid_flip_ud_btn.toggled.connect(self._on_sensor_touch_grid_flip_ud_toggled)
        grid_flip_row.addWidget(self.sensor_grid_flip_lr_btn)
        grid_flip_row.addWidget(self.sensor_grid_flip_ud_btn)
        grid_flip_row.addStretch()
        layout.addLayout(grid_flip_row)
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

        flip_lr = bool(int(getattr(helper, "console_sensor_touch_grid_flip_lr", 0)))
        flip_ud = bool(int(getattr(helper, "console_sensor_touch_grid_flip_ud", 0)))
        for matrix_row in range(n_row):
            for matrix_col in range(n_col):
                disp_row = (n_row - 1 - matrix_row) if flip_ud else matrix_row
                disp_col = (n_col - 1 - matrix_col) if flip_lr else matrix_col
                value = float(matrix[matrix_row, matrix_col])
                is_active = value < threshold
                cell = self._sensor_touch_cells[disp_row][disp_col]
                cell.setText("●" if is_active else "")
                cell.setStyleSheet(
                    self._touch_cell_style(norm[matrix_row, matrix_col] if is_active else 0.0)
                )

    def _apply_sensor_touch_grid_flip(self, key, checked):
        helper = self._get_sensor_helper("console_control_class")
        if helper is None or not hasattr(helper, "apply_settings"):
            return
        try:
            helper.apply_settings({key: 1 if checked else 0}, save_to_file=True)
        except Exception as exc:
            print(f"[UI] Failed to save touch grid flip: {exc}")

    def _on_sensor_touch_grid_flip_lr_toggled(self, checked):
        self._apply_sensor_touch_grid_flip("console_sensor_touch_grid_flip_lr", checked)

    def _on_sensor_touch_grid_flip_ud_toggled(self, checked):
        self._apply_sensor_touch_grid_flip("console_sensor_touch_grid_flip_ud", checked)

    def _update_sensor_controller_test(self):
        helper = self._get_sensor_helper("console_control_class")
        if helper is None or not hasattr(helper, "get_console_sensor_preview_inputs"):
            self.sensor_controller_status_label.setText("Sensor mapping preview: Console control helper unavailable")
            return

        try:
            if hasattr(self, "sensor_grid_flip_lr_btn"):
                lr = bool(int(getattr(helper, "console_sensor_touch_grid_flip_lr", 0)))
                ud = bool(int(getattr(helper, "console_sensor_touch_grid_flip_ud", 0)))
                self.sensor_grid_flip_lr_btn.blockSignals(True)
                self.sensor_grid_flip_ud_btn.blockSignals(True)
                self.sensor_grid_flip_lr_btn.setChecked(lr)
                self.sensor_grid_flip_ud_btn.setChecked(ud)
                self.sensor_grid_flip_lr_btn.blockSignals(False)
                self.sensor_grid_flip_ud_btn.blockSignals(False)
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
        "push_pinch_enabled": (
            "Enable hard-press push and two-finger pinch pull gestures.\n"
            "Off: hard press and pinch are ignored.\n"
            "On: hard press can push, and two-finger pinch can pull."
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
        "two_finger_swipe_enable_horizontal": (
            "Enable horizontal two-finger swipe rotation.\n"
            "Turn off to ignore left/right midpoint swipes."
        ),
        "two_finger_swipe_enable_vertical": (
            "Enable vertical two-finger swipe rotation.\n"
            "Turn off to ignore up/down midpoint swipes."
        ),
        "two_finger_swipe_up_add_push": (
            "Add push motion during a two-finger upward swipe.\n"
            "Off: two-finger vertical swipe only rotates as usual.\n"
            "On: two-finger upward swipe combines rotation with Push Speed."
        ),
        "two_finger_swipe_down_add_pull": (
            "Add pull motion during a two-finger downward swipe.\n"
            "Off: two-finger vertical swipe only rotates as usual.\n"
            "On: two-finger downward swipe combines rotation with Pull Speed."
        ),
        "single_finger_up_as_two_finger_swipe_up": (
            "Treat a one-finger upward swipe as a two-finger upward swipe.\n"
            "If 2-Finger Up Swipe + Push is enabled, this one-finger upward swipe\n"
            "also performs the same rotation plus push combination."
        ),
        "single_finger_vertical_to_y": (
            "Single-finger vertical swipe axis.\n"
            "Off: finger down/up controls +Z/-Z.\n"
            "On: finger down/up controls +Y/-Y."
        ),
        "single_finger_latch_motion": (
            "Keep the last one- or two-finger swipe velocity while touch remains.\n"
            "Off: robot stops when the finger stops moving.\n"
            "On: robot keeps moving in the previous direction until no touch is detected."
        ),
        "single_finger_magnitude_speed": (
            "Allow finger movement distance to change single-finger robot speed.\n"
            "On: larger finger movement commands faster robot motion.\n"
            "Off: finger movement only selects direction; speed stays at Robot Speed."
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
        "hand_control_enabled": (
            "Enable five-finger control for the RH56F1 dexterous hand.\n"
            "When enabled, five-finger motion controls only the hand:\n"
            "fingers moving inward closes; moving outward opens."
        ),
        "hand_five_finger_min_clusters": (
            "Minimum number of separated touch clusters required to treat the\n"
            "sensor input as a five-finger hand gesture."
        ),
        "hand_five_finger_min_cells": (
            "Minimum total active sensor cells required before sending a hand\n"
            "command. Increase this if accidental single contacts trigger it."
        ),
        "hand_five_finger_motion_threshold": (
            "Minimum normalized change in the five-finger spread per frame.\n"
            "Spread shrinking below this threshold closes the hand; growing\n"
            "above this threshold opens it."
        ),
        "hand_five_finger_close_frames": (
            "How many consecutive inward five-finger motion frames are\n"
            "required before RH56F1 closes."
        ),
        "hand_five_finger_open_frames": (
            "How many consecutive outward five-finger motion frames are\n"
            "required before RH56F1 opens."
        ),
        "hand_command_timeout_sec": (
            "Maximum wait time for each RH56F1 Setangle service call.\n"
            "Shorter values keep the DFM loop responsive if the service stalls."
        ),
    }

    def _build_direct_finger_motion_settings_dialog(self):
        self.direct_finger_motion_settings_dialog = QDialog(self)
        self.direct_finger_motion_settings_dialog.setWindowTitle("Direct Finger Motion Parameters")
        self.direct_finger_motion_settings_dialog.resize(780, 660)
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

        def add_bool(name, label, row, col):
            widget = QCheckBox()
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
        add_bool("two_finger_swipe_enable_horizontal", "Enable 2-Finger Horizontal Swipe", 12, 0)
        add_bool("two_finger_swipe_enable_vertical", "Enable 2-Finger Vertical Swipe", 12, 2)
        add_bool("two_finger_swipe_up_add_push", "2-Finger Up Swipe + Push", 13, 0)
        add_bool("two_finger_swipe_down_add_pull", "2-Finger Down Swipe + Pull", 13, 2)
        add_bool("single_finger_vertical_to_y", "Single-Finger Vertical → Y", 14, 0)
        add_bool("single_finger_latch_motion", "Hold Last Finger Motion", 14, 2)
        add_bool("single_finger_magnitude_speed", "Finger Magnitude Controls Speed", 15, 0)
        add_bool("push_pinch_enabled", "Enable Push/Pinch", 15, 2)
        add_bool("single_finger_up_as_two_finger_swipe_up", "1-Finger Up = 2-Finger Up", 16, 0)
        add_int("frame_interval_ms", "Timer Interval (ms)", 20, 0, 0, 10000)
        add_bool("hand_control_enabled", "RH56F1 5-Finger Control", 17, 0)
        add_int("hand_five_finger_min_clusters", "5-Finger Min Clusters", 18, 0, 1, 20)
        add_int("hand_five_finger_min_cells", "5-Finger Min Cells", 19, 0, 1, 999)
        add_double("hand_five_finger_motion_threshold", "5-Finger Motion Threshold", 17, 2, 0.0, 1.0, 0.005, 3)
        add_int("hand_five_finger_close_frames", "5-Finger Close Frames", 18, 2, 1, 999)
        add_int("hand_five_finger_open_frames", "5-Finger Open Frames", 19, 2, 1, 999)
        add_double("hand_command_timeout_sec", "Hand Command Timeout (s)", 20, 2, 0.1, 5.0, 0.1, 2)

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
        add_double("console_sensor_v2_slow_scale", "Sensor V2 Slow Scale", 8, 2, 0.05, 1.0, 0.05, 2, "L1 speed multiplier in Sensor V2.")
        add_double("console_sensor_v2_fast_scale", "Sensor V2 Fast Scale", 9, 2, 1.0, 4.0, 0.1, 2, "R1 speed multiplier in Sensor V2.")

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
        self._direct_finger_motion_tool_pose_raw_points = None
        self._direct_finger_motion_tool_pose_quaternions = None
        self._direct_finger_motion_tool_pose_comparison_paths = []
        self._direct_finger_motion_tool_pose_comparison_actors = []
        self._direct_finger_motion_tool_pose_compact_origin = None
        self._direct_finger_motion_tool_pose_compact_scale = None
        self._direct_finger_motion_tool_pose_target_display_origin = None
        self._direct_finger_motion_tool_pose_target_display_scale = None
        self._direct_finger_motion_tool_pose_target_points = None
        self._direct_finger_motion_tool_pose_target_raw_points = None
        self._direct_finger_motion_tool_pose_target_normals = None
        self._direct_finger_motion_tool_pose_target_csv_path = None
        self._direct_finger_motion_tool_pose_target_align_start = True
        self._direct_finger_motion_tool_pose_target_align_rotation = False
        self._direct_finger_motion_tool_pose_target_rotation_mode = "first_segment"
        self._direct_finger_motion_tool_pose_target_manual_rx_deg = 0.0
        self._direct_finger_motion_tool_pose_target_manual_ry_deg = 0.0
        self._direct_finger_motion_tool_pose_target_manual_rz_deg = 0.0
        self._direct_finger_motion_tool_pose_target_actors = []
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
            path_data = self._read_direct_finger_motion_tool_pose_path_data(csv_path)
            points = path_data["points"]
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
        raw_points_np = np.asarray(points, dtype=float)
        quaternions_np = None
        if path_data.get("quaternions"):
            quaternions_np = np.asarray(path_data["quaternions"], dtype=float)
        points_np = self._compact_direct_finger_motion_tool_pose_points(raw_points_np)
        self._direct_finger_motion_tool_pose_raw_points = raw_points_np
        self._direct_finger_motion_tool_pose_quaternions = quaternions_np
        self._direct_finger_motion_tool_pose_plot_points = points_np
        self._set_direct_finger_motion_primary_comparison_path(
            csv_path,
            raw_points_np,
            points_np,
            duration_s=path_data.get("duration_s"),
            time_source=path_data.get("time_source"),
            quaternions_np=quaternions_np,
        )
        self._refresh_direct_finger_motion_comparison_robot_display_points()
        self._direct_finger_motion_tool_pose_animation_points = (
            self._build_direct_finger_motion_tool_pose_animation_points(points_np)
        )
        self._stop_direct_finger_motion_tool_pose_path_animation(reset_button=False)
        self._render_direct_finger_motion_tool_pose_path_static(points_np)
        self._render_direct_finger_motion_comparison_robot_paths()
        if getattr(self, "_direct_finger_motion_tool_pose_target_raw_points", None) is not None:
            self._refresh_direct_finger_motion_target_path_display(log_message=True)
        self._render_direct_finger_motion_tool_pose_path_frame(points_np[:1])
        self._set_direct_finger_motion_tool_pose_play_button_enabled(True)

        try:
            self._direct_finger_motion_tool_pose_plot_dialog.show()
            self._direct_finger_motion_tool_pose_plot_dialog.raise_()
            self._direct_finger_motion_tool_pose_plot_dialog.activateWindow()
            self._set_direct_finger_motion_tool_pose_preview_camera(plotter)
            plotter.render()
        except Exception:
            pass

        self._append_direct_finger_motion_log(
            f"[DFM] Loaded tool pose path in popup 3D graph: {csv_path} ({len(points_np)} points)"
        )
        self._append_direct_finger_motion_target_error_summary()

    @staticmethod
    def _direct_finger_motion_robot_path_color(index):
        colors = (
            "#1e88e5",
            "#43a047",
            "#8e24aa",
            "#00acc1",
            "#f4511e",
            "#6d4c41",
            "#3949ab",
            "#c0ca33",
        )
        return colors[int(index) % len(colors)]

    @staticmethod
    def _direct_finger_motion_robot_path_label(csv_path, index):
        basename = os.path.basename(str(csv_path)).strip()
        if not basename:
            basename = f"robot_path_{int(index) + 1}"
        stem, extension = os.path.splitext(basename)
        if extension.lower() == ".csv" and stem:
            return stem
        return basename

    @staticmethod
    def _direct_finger_motion_analysis_display_label(label, fallback="Robot path"):
        text = str(label or "").strip()
        if not text:
            text = str(fallback)
        basename = os.path.basename(text)
        stem, extension = os.path.splitext(basename)
        if extension.lower() == ".csv" and stem:
            return stem
        return basename or text

    def _set_direct_finger_motion_primary_comparison_path(
        self,
        csv_path,
        raw_points_np,
        display_points_np,
        duration_s=None,
        time_source=None,
        quaternions_np=None,
    ):
        entry = {
            "label": self._direct_finger_motion_robot_path_label(csv_path, 0),
            "csv_path": os.path.abspath(str(csv_path)),
            "source_raw_points": np.asarray(raw_points_np, dtype=float),
            "raw_points": np.asarray(raw_points_np, dtype=float),
            "display_points": np.asarray(display_points_np, dtype=float),
            "source_quaternions": None
            if quaternions_np is None
            else np.asarray(quaternions_np, dtype=float),
            "quaternions": None
            if quaternions_np is None
            else np.asarray(quaternions_np, dtype=float),
            "color": self._direct_finger_motion_robot_path_color(0),
            "is_primary": True,
            "start_translation": np.zeros(3, dtype=float),
            "duration_s": duration_s,
            "time_source": time_source or "",
        }
        paths = list(getattr(self, "_direct_finger_motion_tool_pose_comparison_paths", []))
        if paths:
            paths[0] = entry
        else:
            paths = [entry]
        self._direct_finger_motion_tool_pose_comparison_paths = paths

    def _align_direct_finger_motion_robot_path_start_to_primary(self, points_np):
        points_np = np.asarray(points_np, dtype=float)
        if len(points_np) == 0:
            return points_np, np.zeros(3, dtype=float)

        primary_points_np = getattr(self, "_direct_finger_motion_tool_pose_raw_points", None)
        if primary_points_np is None or len(primary_points_np) == 0:
            return points_np, np.zeros(3, dtype=float)

        translation = np.asarray(primary_points_np[0], dtype=float) - points_np[0]
        return points_np + translation, translation

    def _refresh_direct_finger_motion_comparison_robot_display_points(self):
        paths = list(getattr(self, "_direct_finger_motion_tool_pose_comparison_paths", []))
        refreshed_paths = []
        for index, path in enumerate(paths):
            source_raw_points = path.get("source_raw_points", path.get("raw_points"))
            if source_raw_points is None:
                continue
            source_raw_points_np = np.asarray(source_raw_points, dtype=float)
            if len(source_raw_points_np) == 0:
                continue
            refreshed = dict(path)
            refreshed["source_raw_points"] = source_raw_points_np
            if bool(refreshed.get("is_primary", False)):
                aligned_raw_points_np = source_raw_points_np
                translation = np.zeros(3, dtype=float)
            else:
                aligned_raw_points_np, translation = (
                    self._align_direct_finger_motion_robot_path_start_to_primary(
                        source_raw_points_np
                    )
                )
            refreshed["raw_points"] = aligned_raw_points_np
            refreshed["start_translation"] = np.asarray(translation, dtype=float)
            refreshed["duration_s"] = path.get("duration_s")
            refreshed["time_source"] = path.get("time_source", "")
            source_quaternions = path.get("source_quaternions", path.get("quaternions"))
            if source_quaternions is not None:
                source_quaternions_np = np.asarray(source_quaternions, dtype=float)
                if len(source_quaternions_np) == len(source_raw_points_np):
                    refreshed["source_quaternions"] = source_quaternions_np
                    refreshed["quaternions"] = source_quaternions_np
                else:
                    refreshed["source_quaternions"] = None
                    refreshed["quaternions"] = None
            refreshed["display_points"] = self._transform_direct_finger_motion_tool_pose_points(
                aligned_raw_points_np
            )
            refreshed["color"] = (
                refreshed.get("color")
                or self._direct_finger_motion_robot_path_color(index)
            )
            refreshed_paths.append(refreshed)
        self._direct_finger_motion_tool_pose_comparison_paths = refreshed_paths

    def _load_direct_finger_motion_comparison_robot_paths_from_dialog(self):
        start_dir = self.DIRECT_FINGER_MOTION_TOOL_POSE_LOG_DIR
        if not os.path.isdir(start_dir):
            start_dir = os.path.expanduser("~")

        dialog = QFileDialog(
            self,
            "Add Robot Tool Path CSVs",
            start_dir,
            "CSV Files (*.csv);;All Files (*)",
        )
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setViewMode(QFileDialog.Detail)

        if dialog.exec_() != QFileDialog.Accepted:
            return

        selected_files = dialog.selectedFiles()
        if not selected_files:
            return
        for csv_path in selected_files:
            self._add_direct_finger_motion_comparison_robot_path(csv_path)

    def _add_direct_finger_motion_comparison_robot_path(self, csv_path):
        try:
            path_data = self._read_direct_finger_motion_tool_pose_path_data(csv_path)
            points = path_data["points"]
        except Exception as exc:
            self._append_direct_finger_motion_log(
                f"[DFM] Failed to load comparison robot path: {exc}"
            )
            return False

        if not points:
            self._append_direct_finger_motion_log(
                f"[DFM] No robot tool pose samples found in file: {csv_path}"
            )
            return False

        plotter = self._ensure_direct_finger_motion_tool_pose_plot_window()
        if plotter is None:
            self._append_direct_finger_motion_log(
                "[DFM] 3D popup plotter is unavailable: cannot draw comparison robot path."
            )
            return False

        raw_points_np = np.asarray(points, dtype=float)
        quaternions_np = None
        if path_data.get("quaternions"):
            quaternions_np = np.asarray(path_data["quaternions"], dtype=float)
        if getattr(self, "_direct_finger_motion_tool_pose_raw_points", None) is None:
            self._load_direct_finger_motion_tool_pose_path(csv_path)
            return True

        aligned_points_np, start_translation = (
            self._align_direct_finger_motion_robot_path_start_to_primary(raw_points_np)
        )
        display_points_np = self._transform_direct_finger_motion_tool_pose_points(
            aligned_points_np
        )
        paths = list(getattr(self, "_direct_finger_motion_tool_pose_comparison_paths", []))
        path_index = len(paths)
        paths.append(
            {
                "label": self._direct_finger_motion_robot_path_label(csv_path, path_index),
                "csv_path": os.path.abspath(str(csv_path)),
                "source_raw_points": raw_points_np,
                "raw_points": aligned_points_np,
                "display_points": display_points_np,
                "source_quaternions": quaternions_np,
                "quaternions": quaternions_np,
                "color": self._direct_finger_motion_robot_path_color(path_index),
                "is_primary": False,
                "start_translation": np.asarray(start_translation, dtype=float),
                "duration_s": path_data.get("duration_s"),
                "time_source": path_data.get("time_source"),
            }
        )
        self._direct_finger_motion_tool_pose_comparison_paths = paths
        self._render_direct_finger_motion_comparison_robot_paths()
        self._append_direct_finger_motion_log(
            f"[DFM] Added comparison robot path: {csv_path} ({len(raw_points_np)} points). "
            "Start translated by "
            f"dx={float(start_translation[0]):+.4f}, "
            f"dy={float(start_translation[1]):+.4f}, "
            f"dz={float(start_translation[2]):+.4f} m"
        )
        return True

    def _clear_direct_finger_motion_comparison_robot_paths(self, keep_primary=True, log_message=True):
        self._clear_direct_finger_motion_comparison_path_actors()
        paths = list(getattr(self, "_direct_finger_motion_tool_pose_comparison_paths", []))
        if keep_primary and paths:
            self._direct_finger_motion_tool_pose_comparison_paths = paths[:1]
        else:
            self._direct_finger_motion_tool_pose_comparison_paths = []
        if log_message:
            self._append_direct_finger_motion_log("[DFM] Cleared comparison robot paths.")

    def _read_direct_finger_motion_tool_pose_path_data(self, csv_path):
        points = []
        timestamps = []
        quaternions = []
        normals = []
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames:
                return {
                    "points": points,
                    "duration_s": None,
                    "time_source": "",
                    "timestamps": timestamps,
                    "quaternions": quaternions,
                    "normals": normals,
                }

            def normalized_field_name(name):
                text = str(name).strip().lower()
                for old, new in (
                    ("(", "_"),
                    (")", ""),
                    (" ", "_"),
                    ("-", "_"),
                    (".", "_"),
                ):
                    text = text.replace(old, new)
                while "__" in text:
                    text = text.replace("__", "_")
                return text.strip("_")

            field_lookup = {
                normalized_field_name(name): name
                for name in reader.fieldnames
            }
            x_key = field_lookup.get("pos_x") or field_lookup.get("x") or field_lookup.get("x_mm")
            y_key = field_lookup.get("pos_y") or field_lookup.get("y") or field_lookup.get("y_mm")
            z_key = field_lookup.get("pos_z") or field_lookup.get("z") or field_lookup.get("z_mm")
            quat_w_key = (
                field_lookup.get("quat_w")
                or field_lookup.get("qw")
                or field_lookup.get("orientation_w")
            )
            quat_x_key = (
                field_lookup.get("quat_x")
                or field_lookup.get("qx")
                or field_lookup.get("orientation_x")
            )
            quat_y_key = (
                field_lookup.get("quat_y")
                or field_lookup.get("qy")
                or field_lookup.get("orientation_y")
            )
            quat_z_key = (
                field_lookup.get("quat_z")
                or field_lookup.get("qz")
                or field_lookup.get("orientation_z")
            )
            normal_x_key = (
                field_lookup.get("normal_x")
                or field_lookup.get("nx")
                or field_lookup.get("n_x")
            )
            normal_y_key = (
                field_lookup.get("normal_y")
                or field_lookup.get("ny")
                or field_lookup.get("n_y")
            )
            normal_z_key = (
                field_lookup.get("normal_z")
                or field_lookup.get("nz")
                or field_lookup.get("n_z")
            )
            time_key = (
                field_lookup.get("timestamp")
                or field_lookup.get("time")
                or field_lookup.get("time_s")
                or field_lookup.get("elapsed_s")
                or field_lookup.get("elapsed_time_s")
                or field_lookup.get("elapsed_time")
                or field_lookup.get("t")
            )
            if not (x_key and y_key and z_key):
                raise ValueError(
                    "CSV must contain pos_x,pos_y,pos_z, x,y,z, or SolidWorks X (mm),Y (mm),Z (mm) columns."
                )
            unit_scale = 0.001 if any(
                "(mm)" in str(key).lower() or normalized_field_name(key).endswith("_mm")
                for key in (x_key, y_key, z_key)
            ) else 1.0
            for row in reader:
                point = (
                    float(row[x_key]) * unit_scale,
                    float(row[y_key]) * unit_scale,
                    float(row[z_key]) * unit_scale,
                )
                points.append(
                    point
                )
                if quat_w_key and quat_x_key and quat_y_key and quat_z_key:
                    try:
                        quaternions.append(
                            (
                                float(row[quat_w_key]),
                                float(row[quat_x_key]),
                                float(row[quat_y_key]),
                                float(row[quat_z_key]),
                            )
                        )
                    except Exception:
                        pass
                if normal_x_key and normal_y_key and normal_z_key:
                    try:
                        normals.append(
                            (
                                float(row[normal_x_key]),
                                float(row[normal_y_key]),
                                float(row[normal_z_key]),
                            )
                        )
                    except Exception:
                        pass
                if time_key:
                    try:
                        timestamps.append(float(row[time_key]))
                    except Exception:
                        pass

        duration_s = None
        if len(timestamps) >= 2:
            duration_s = float(timestamps[-1] - timestamps[0])
            if duration_s < 0.0:
                duration_s = float(max(timestamps) - min(timestamps))
        return {
            "points": points,
            "duration_s": duration_s,
            "time_source": time_key or "",
            "timestamps": timestamps,
            "quaternions": quaternions if len(quaternions) == len(points) else [],
            "normals": normals if len(normals) == len(points) else [],
        }

    def _read_direct_finger_motion_tool_pose_points(self, csv_path):
        return self._read_direct_finger_motion_tool_pose_path_data(csv_path)["points"]

    def _compact_direct_finger_motion_tool_pose_points(self, points_np):
        if points_np is None or len(points_np) == 0:
            return points_np

        axis_min = np.min(points_np, axis=0)
        shifted_points = points_np - axis_min
        max_span = float(np.max(np.ptp(shifted_points, axis=0)))
        self._direct_finger_motion_tool_pose_compact_origin = axis_min
        self._direct_finger_motion_tool_pose_compact_scale = max_span
        if max_span <= 1e-12:
            return shifted_points
        return shifted_points / max_span

    def _set_direct_finger_motion_tool_pose_target_display_frame(self, points_np):
        points_np = np.asarray(points_np, dtype=float)
        if len(points_np) == 0:
            return

        axis_min = np.min(points_np, axis=0)
        shifted_points = points_np - axis_min
        max_span = float(np.max(np.ptp(shifted_points, axis=0)))
        self._direct_finger_motion_tool_pose_target_display_origin = axis_min
        self._direct_finger_motion_tool_pose_target_display_scale = max(max_span, 1e-12)

    def _transform_direct_finger_motion_tool_pose_points(self, points_np):
        points_np = np.asarray(points_np, dtype=float)
        origin = getattr(self, "_direct_finger_motion_tool_pose_compact_origin", None)
        scale = getattr(self, "_direct_finger_motion_tool_pose_compact_scale", None)
        if origin is None or scale is None:
            origin = getattr(
                self, "_direct_finger_motion_tool_pose_target_display_origin", None
            )
            scale = getattr(
                self, "_direct_finger_motion_tool_pose_target_display_scale", None
            )
            if origin is None or scale is None:
                self._set_direct_finger_motion_tool_pose_target_display_frame(points_np)
                origin = getattr(
                    self, "_direct_finger_motion_tool_pose_target_display_origin", None
                )
                scale = getattr(
                    self, "_direct_finger_motion_tool_pose_target_display_scale", None
                )
        shifted_points = points_np - origin
        if float(scale) <= 1e-12:
            return shifted_points
        return shifted_points / float(scale)

    def _apply_direct_finger_motion_target_start_alignment(self, target_raw_np):
        """Translate target path so its first point matches the robot path start."""
        target_raw_np = np.asarray(target_raw_np, dtype=float)
        if not bool(getattr(self, "_direct_finger_motion_tool_pose_target_align_start", True)):
            return target_raw_np

        robot_raw_np = getattr(self, "_direct_finger_motion_tool_pose_raw_points", None)
        if robot_raw_np is None or len(robot_raw_np) == 0:
            return target_raw_np
        if len(target_raw_np) == 0:
            return target_raw_np

        offset = np.asarray(robot_raw_np[0], dtype=float) - np.asarray(
            target_raw_np[0], dtype=float
        )
        return target_raw_np + offset

    def _direct_finger_motion_target_rotation_settings_key(self, csv_path):
        return os.path.basename(os.path.abspath(str(csv_path)))

    def _load_direct_finger_motion_target_rotation_settings(self, csv_path):
        settings_file = self.DIRECT_FINGER_MOTION_TOOL_POSE_TARGET_ROTATION_FILE
        if not os.path.isfile(settings_file):
            return None

        try:
            with open(settings_file, "r", encoding="utf-8") as settings_handle:
                payload = json.load(settings_handle)
        except Exception:
            return None

        entry = (payload.get("by_target") or {}).get(
            self._direct_finger_motion_target_rotation_settings_key(csv_path)
        )
        if not isinstance(entry, dict):
            return None

        return (
            float(entry.get("rx_deg", 0.0)),
            float(entry.get("ry_deg", 0.0)),
            float(entry.get("rz_deg", 0.0)),
        )

    def _save_direct_finger_motion_target_rotation_settings(
        self,
        csv_path,
        rx_deg,
        ry_deg,
        rz_deg,
    ):
        settings_file = self.DIRECT_FINGER_MOTION_TOOL_POSE_TARGET_ROTATION_FILE
        settings_dir = os.path.dirname(settings_file)
        if settings_dir:
            os.makedirs(settings_dir, exist_ok=True)

        payload = {"version": 1, "by_target": {}}
        if os.path.isfile(settings_file):
            try:
                with open(settings_file, "r", encoding="utf-8") as settings_handle:
                    existing_payload = json.load(settings_handle)
                if isinstance(existing_payload, dict):
                    payload = existing_payload
            except Exception:
                pass

        by_target = payload.get("by_target")
        if not isinstance(by_target, dict):
            by_target = {}
            payload["by_target"] = by_target

        settings_key = self._direct_finger_motion_target_rotation_settings_key(csv_path)
        by_target[settings_key] = {
            "rx_deg": float(rx_deg),
            "ry_deg": float(ry_deg),
            "rz_deg": float(rz_deg),
            "csv_path": os.path.abspath(str(csv_path)),
        }
        payload["version"] = 1

        with open(settings_file, "w", encoding="utf-8") as settings_handle:
            json.dump(payload, settings_handle, indent=2)

    def _restore_direct_finger_motion_target_rotation_settings(self, csv_path):
        saved_rotation = self._load_direct_finger_motion_target_rotation_settings(csv_path)
        if saved_rotation is None:
            self._set_direct_finger_motion_target_manual_rotation_ui_values(
                0.0, 0.0, 0.0
            )
            return False

        rx_deg, ry_deg, rz_deg = saved_rotation
        self._set_direct_finger_motion_target_manual_rotation_ui_values(
            rx_deg, ry_deg, rz_deg
        )
        return True

    @staticmethod
    def _rotation_matrix_from_vectors(source_vec, target_vec):
        source_vec = np.asarray(source_vec, dtype=float).reshape(3)
        target_vec = np.asarray(target_vec, dtype=float).reshape(3)
        source_norm = float(np.linalg.norm(source_vec))
        target_norm = float(np.linalg.norm(target_vec))
        if source_norm <= 1e-12 or target_norm <= 1e-12:
            return np.eye(3)

        source_unit = source_vec / source_norm
        target_unit = target_vec / target_norm
        cross = np.cross(source_unit, target_unit)
        dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm <= 1e-12:
            if dot > 0.0:
                return np.eye(3)
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(source_unit[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=float)
            cross = np.cross(source_unit, axis)
            cross /= max(float(np.linalg.norm(cross)), 1e-12)
            cross_norm = 1.0

        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ],
            dtype=float,
        )
        return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (cross_norm ** 2))

    @staticmethod
    def _resample_direct_finger_motion_path_points(points_np, sample_count):
        points_np = np.asarray(points_np, dtype=float)
        if len(points_np) == 0:
            return points_np
        if len(points_np) == 1 or int(sample_count) <= 1:
            return points_np[:1]

        segment_lengths = np.linalg.norm(np.diff(points_np, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = float(cumulative[-1])
        if total_length <= 1e-12:
            return np.repeat(points_np[:1], int(sample_count), axis=0)

        sample_positions = np.linspace(0.0, total_length, int(sample_count))
        resampled = np.zeros((int(sample_count), 3), dtype=float)
        for axis in range(3):
            resampled[:, axis] = np.interp(sample_positions, cumulative, points_np[:, axis])
        return resampled

    @staticmethod
    def _kabsch_rotation_matrix(source_points_np, target_points_np):
        source_points_np = np.asarray(source_points_np, dtype=float)
        target_points_np = np.asarray(target_points_np, dtype=float)
        if len(source_points_np) == 0 or len(target_points_np) == 0:
            return np.eye(3)
        if len(source_points_np) != len(target_points_np):
            raise ValueError("Kabsch alignment requires the same number of samples.")

        covariance = source_points_np.T @ target_points_np
        u_matrix, _, vt_matrix = np.linalg.svd(covariance)
        rotation = vt_matrix.T @ u_matrix.T
        if np.linalg.det(rotation) < 0.0:
            vt_matrix[-1, :] *= -1.0
            rotation = vt_matrix.T @ u_matrix.T
        return rotation

    @staticmethod
    def _rotate_points_about_pivot(points_np, pivot, rotation_matrix):
        points_np = np.asarray(points_np, dtype=float)
        pivot = np.asarray(pivot, dtype=float).reshape(3)
        rotation_matrix = np.asarray(rotation_matrix, dtype=float).reshape(3, 3)
        return (points_np - pivot) @ rotation_matrix.T + pivot

    @staticmethod
    def _euler_xyz_rotation_matrix_deg(rx_deg, ry_deg, rz_deg):
        rx, ry, rz = np.radians(
            [float(rx_deg), float(ry_deg), float(rz_deg)],
            dtype=float,
        )
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        rotation_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
            dtype=float,
        )
        rotation_y = np.array(
            [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
            dtype=float,
        )
        rotation_z = np.array(
            [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        return rotation_z @ rotation_y @ rotation_x

    def _get_direct_finger_motion_target_manual_rotation_deg(self):
        rx_deg = float(
            getattr(self, "_direct_finger_motion_tool_pose_target_manual_rx_deg", 0.0)
        )
        ry_deg = float(
            getattr(self, "_direct_finger_motion_tool_pose_target_manual_ry_deg", 0.0)
        )
        rz_deg = float(
            getattr(self, "_direct_finger_motion_tool_pose_target_manual_rz_deg", 0.0)
        )
        spin_rx = getattr(self, "direct_finger_motion_target_manual_rx_spin", None)
        spin_ry = getattr(self, "direct_finger_motion_target_manual_ry_spin", None)
        spin_rz = getattr(self, "direct_finger_motion_target_manual_rz_spin", None)
        if spin_rx is not None:
            rx_deg = float(spin_rx.value())
        if spin_ry is not None:
            ry_deg = float(spin_ry.value())
        if spin_rz is not None:
            rz_deg = float(spin_rz.value())
        return rx_deg, ry_deg, rz_deg

    def _set_direct_finger_motion_target_manual_rotation_ui_values(
        self,
        rx_deg=0.0,
        ry_deg=0.0,
        rz_deg=0.0,
    ):
        self._direct_finger_motion_tool_pose_target_manual_rx_deg = float(rx_deg)
        self._direct_finger_motion_tool_pose_target_manual_ry_deg = float(ry_deg)
        self._direct_finger_motion_tool_pose_target_manual_rz_deg = float(rz_deg)
        for spin, value in (
            (
                getattr(self, "direct_finger_motion_target_manual_rx_spin", None),
                rx_deg,
            ),
            (
                getattr(self, "direct_finger_motion_target_manual_ry_spin", None),
                ry_deg,
            ),
            (
                getattr(self, "direct_finger_motion_target_manual_rz_spin", None),
                rz_deg,
            ),
        ):
            if spin is None:
                continue
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    def _direct_finger_motion_target_manual_rotation_matrix(self):
        rx_deg, ry_deg, rz_deg = self._get_direct_finger_motion_target_manual_rotation_deg()
        if (
            abs(rx_deg) <= 1e-9
            and abs(ry_deg) <= 1e-9
            and abs(rz_deg) <= 1e-9
        ):
            return np.eye(3)
        return self._euler_xyz_rotation_matrix_deg(rx_deg, ry_deg, rz_deg)

    def _apply_direct_finger_motion_target_manual_rotation(self, aligned_raw_np):
        """Rotate target path in world XYZ (metres) around its aligned start point."""
        aligned_raw_np = np.asarray(aligned_raw_np, dtype=float)
        if len(aligned_raw_np) == 0:
            return aligned_raw_np

        rotation_matrix = self._direct_finger_motion_target_manual_rotation_matrix()
        if np.allclose(rotation_matrix, np.eye(3)):
            return aligned_raw_np

        pivot = np.asarray(aligned_raw_np[0], dtype=float)
        return self._rotate_points_about_pivot(aligned_raw_np, pivot, rotation_matrix)

    def _commit_direct_finger_motion_target_manual_rotation_spinboxes(self):
        for widget_name in (
            "direct_finger_motion_target_manual_rx_spin",
            "direct_finger_motion_target_manual_ry_spin",
            "direct_finger_motion_target_manual_rz_spin",
        ):
            spinbox = getattr(self, widget_name, None)
            if spinbox is None:
                continue
            try:
                spinbox.interpretText()
            except Exception:
                pass

    def _apply_direct_finger_motion_target_manual_rotation_from_ui(self):
        self._commit_direct_finger_motion_target_manual_rotation_spinboxes()
        rx_deg, ry_deg, rz_deg = self._get_direct_finger_motion_target_manual_rotation_deg()
        self._direct_finger_motion_tool_pose_target_manual_rx_deg = rx_deg
        self._direct_finger_motion_tool_pose_target_manual_ry_deg = ry_deg
        self._direct_finger_motion_tool_pose_target_manual_rz_deg = rz_deg

        target_raw_np = getattr(
            self, "_direct_finger_motion_tool_pose_target_raw_points", None
        )
        if target_raw_np is None or len(target_raw_np) == 0:
            self._append_direct_finger_motion_log(
                "[DFM] Load a target CSV before applying manual rotation."
            )
            return False

        if not self._refresh_direct_finger_motion_target_path_display(log_message=False):
            self._append_direct_finger_motion_log(
                "[DFM] Failed to apply manual target rotation."
            )
            return False

        plotter = getattr(self, "_direct_finger_motion_tool_pose_plotter", None)
        if plotter is not None:
            try:
                self._set_direct_finger_motion_tool_pose_preview_camera(plotter)
                plotter.render()
            except Exception:
                pass

        self._append_direct_finger_motion_log(
            "[DFM] Applied manual target rotation: "
            f"Rx={rx_deg:.1f}, Ry={ry_deg:.1f}, Rz={rz_deg:.1f} deg"
        )
        csv_path = getattr(self, "_direct_finger_motion_tool_pose_target_csv_path", None)
        if csv_path:
            try:
                self._save_direct_finger_motion_target_rotation_settings(
                    csv_path, rx_deg, ry_deg, rz_deg
                )
                self._append_direct_finger_motion_log(
                    "[DFM] Saved target rotation settings for "
                    f"{self._direct_finger_motion_target_rotation_settings_key(csv_path)}"
                )
            except Exception as exc:
                self._append_direct_finger_motion_log(
                    f"[DFM] Failed to save target rotation settings: {exc}"
                )
        self._append_direct_finger_motion_target_error_summary()
        return True

    def _reset_direct_finger_motion_target_manual_rotation(self, refresh_display=True):
        self._set_direct_finger_motion_target_manual_rotation_ui_values(0.0, 0.0, 0.0)
        if refresh_display:
            self._apply_direct_finger_motion_target_manual_rotation_from_ui()

    def _create_direct_finger_motion_target_manual_rotation_spinbox(
        self,
        parent,
        default_value,
    ):
        spinbox = QDoubleSpinBox(parent)
        spinbox.setRange(-360.0, 360.0)
        spinbox.setDecimals(1)
        spinbox.setSingleStep(1.0)
        spinbox.setSuffix(" deg")
        spinbox.setValue(float(default_value))
        spinbox.setKeyboardTracking(False)
        spinbox.editingFinished.connect(
            self._apply_direct_finger_motion_target_manual_rotation_from_ui
        )
        return spinbox

    def _set_direct_finger_motion_target_manual_rotation_controls_enabled(
        self,
        enabled,
    ):
        enabled = bool(enabled)
        for widget_name in (
            "direct_finger_motion_target_manual_rx_spin",
            "direct_finger_motion_target_manual_ry_spin",
            "direct_finger_motion_target_manual_rz_spin",
            "direct_finger_motion_target_manual_rotation_apply_button",
            "direct_finger_motion_target_manual_rotation_reset_button",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(enabled)

    def _compute_direct_finger_motion_target_rotation_matrix(
        self,
        target_points_np,
        robot_points_np,
    ):
        mode = str(
            getattr(
                self,
                "_direct_finger_motion_tool_pose_target_rotation_mode",
                "first_segment",
            )
        ).strip().lower()
        if len(target_points_np) < 2 or len(robot_points_np) < 2:
            return np.eye(3), mode

        if mode == "best_fit":
            sample_count = min(len(target_points_np), len(robot_points_np), 100)
            sample_count = max(2, int(sample_count))
            target_samples = self._resample_direct_finger_motion_path_points(
                target_points_np, sample_count
            )
            robot_samples = self._resample_direct_finger_motion_path_points(
                robot_points_np, sample_count
            )
            pivot = np.asarray(target_points_np[0], dtype=float)
            target_centered = target_samples - pivot
            robot_centered = robot_samples - pivot
            rotation = self._kabsch_rotation_matrix(target_centered, robot_centered)
            return rotation, "best_fit"

        source_vec = target_points_np[1] - target_points_np[0]
        target_vec = robot_points_np[1] - robot_points_np[0]
        rotation = self._rotation_matrix_from_vectors(source_vec, target_vec)
        return rotation, "first_segment"

    def _apply_direct_finger_motion_target_rotation_alignment(self, target_points_np):
        target_points_np = np.asarray(target_points_np, dtype=float)
        if not bool(
            getattr(self, "_direct_finger_motion_tool_pose_target_align_rotation", False)
        ):
            return target_points_np, np.eye(3), None

        robot_raw_np = getattr(self, "_direct_finger_motion_tool_pose_raw_points", None)
        if robot_raw_np is None or len(robot_raw_np) < 2:
            return target_points_np, np.eye(3), None
        if len(target_points_np) < 2:
            return target_points_np, np.eye(3), None

        rotation_matrix, mode = self._compute_direct_finger_motion_target_rotation_matrix(
            target_points_np,
            robot_raw_np,
        )
        pivot = np.asarray(target_points_np[0], dtype=float)
        rotated_points = self._rotate_points_about_pivot(
            target_points_np, pivot, rotation_matrix
        )
        return rotated_points, rotation_matrix, mode

    def _apply_direct_finger_motion_target_path_alignment(self, target_raw_np):
        target_raw_np = np.asarray(target_raw_np, dtype=float)
        translated_np = self._apply_direct_finger_motion_target_start_alignment(target_raw_np)
        rotated_np, rotation_matrix, rotation_mode = (
            self._apply_direct_finger_motion_target_rotation_alignment(translated_np)
        )
        return rotated_np, rotation_matrix, rotation_mode

    def _compute_direct_finger_motion_target_display_points(self):
        target_raw_np = getattr(self, "_direct_finger_motion_tool_pose_target_raw_points", None)
        if target_raw_np is None or len(target_raw_np) == 0:
            return None

        if getattr(self, "_direct_finger_motion_tool_pose_compact_origin", None) is None:
            self._set_direct_finger_motion_tool_pose_target_display_frame(target_raw_np)

        aligned_raw_np, _, _ = self._apply_direct_finger_motion_target_path_alignment(
            target_raw_np
        )
        aligned_raw_np = self._apply_direct_finger_motion_target_manual_rotation(
            aligned_raw_np
        )
        return self._transform_direct_finger_motion_tool_pose_points(aligned_raw_np)

    def _compute_direct_finger_motion_target_aligned_raw_points(self):
        target_raw_np = getattr(self, "_direct_finger_motion_tool_pose_target_raw_points", None)
        if target_raw_np is None or len(target_raw_np) == 0:
            return None

        aligned_raw_np, _, _ = self._apply_direct_finger_motion_target_path_alignment(
            target_raw_np
        )
        return self._apply_direct_finger_motion_target_manual_rotation(aligned_raw_np)

    def _compute_direct_finger_motion_target_aligned_normals(self):
        target_raw_np = getattr(self, "_direct_finger_motion_tool_pose_target_raw_points", None)
        target_normals_np = getattr(
            self,
            "_direct_finger_motion_tool_pose_target_normals",
            None,
        )
        if (
            target_raw_np is None
            or target_normals_np is None
            or len(target_raw_np) == 0
            or len(target_raw_np) != len(target_normals_np)
        ):
            return None

        target_normals_np = self._normalize_direct_finger_motion_vectors(
            target_normals_np
        )
        if target_normals_np is None:
            return None

        translated_np = self._apply_direct_finger_motion_target_start_alignment(
            target_raw_np
        )
        _, rotation_matrix, _ = self._apply_direct_finger_motion_target_rotation_alignment(
            translated_np
        )
        manual_rotation_matrix = self._direct_finger_motion_target_manual_rotation_matrix()
        normals_np = target_normals_np @ rotation_matrix.T
        normals_np = normals_np @ manual_rotation_matrix.T
        return self._normalize_direct_finger_motion_vectors(normals_np)

    def _refresh_direct_finger_motion_target_path_display(self, log_message=False):
        target_points_np = self._compute_direct_finger_motion_target_display_points()
        if target_points_np is None:
            return False

        plotter = getattr(self, "_direct_finger_motion_tool_pose_plotter", None)
        if plotter is None:
            return False

        self._direct_finger_motion_tool_pose_target_points = target_points_np
        self._render_direct_finger_motion_target_path(target_points_np)
        try:
            plotter.render()
        except Exception:
            pass

        if log_message:
            robot_raw_np = getattr(self, "_direct_finger_motion_tool_pose_raw_points", None)
            target_raw_np = getattr(
                self, "_direct_finger_motion_tool_pose_target_raw_points", None
            )
            if (
                robot_raw_np is not None
                and len(robot_raw_np) > 0
                and target_raw_np is not None
                and len(target_raw_np) > 0
            ):
                _, rotation_matrix, rotation_mode = (
                    self._apply_direct_finger_motion_target_path_alignment(target_raw_np)
                )
                if bool(
                    getattr(self, "_direct_finger_motion_tool_pose_target_align_start", True)
                ):
                    offset = np.asarray(robot_raw_np[0], dtype=float) - np.asarray(
                        target_raw_np[0], dtype=float
                    )
                    self._append_direct_finger_motion_log(
                        "[DFM] Target path start aligned to robot start. "
                        f"Translation (m): dx={float(offset[0]):.4f}, "
                        f"dy={float(offset[1]):.4f}, dz={float(offset[2]):.4f}"
                    )
                if bool(
                    getattr(
                        self,
                        "_direct_finger_motion_tool_pose_target_align_rotation",
                        False,
                    )
                ) and rotation_mode is not None:
                    trace_value = float(np.trace(rotation_matrix))
                    angle_deg = float(
                        np.degrees(np.arccos(np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)))
                    )
                    mode_label = (
                        "Path best fit"
                        if rotation_mode == "best_fit"
                        else "First segment direction"
                    )
                    self._append_direct_finger_motion_log(
                        f"[DFM] Target path rotation aligned ({mode_label}). "
                        f"Rotation angle: {angle_deg:.2f} deg"
                    )
        return True

    def _on_direct_finger_motion_tool_pose_target_align_start_toggled(self, checked):
        self._direct_finger_motion_tool_pose_target_align_start = bool(checked)
        if not self._refresh_direct_finger_motion_target_path_display(
            log_message=bool(checked)
        ):
            return
        self._append_direct_finger_motion_target_error_summary()

    def _on_direct_finger_motion_tool_pose_target_align_rotation_toggled(self, checked):
        self._direct_finger_motion_tool_pose_target_align_rotation = bool(checked)
        if not self._refresh_direct_finger_motion_target_path_display(
            log_message=bool(checked)
        ):
            return
        self._append_direct_finger_motion_target_error_summary()

    def _on_direct_finger_motion_tool_pose_target_rotation_mode_changed(self, index):
        combo = getattr(self, "direct_finger_motion_target_rotation_mode_combo", None)
        if combo is None:
            return
        mode = combo.itemData(index)
        if mode is None:
            mode = combo.itemText(index)
        self._direct_finger_motion_tool_pose_target_rotation_mode = str(mode)
        if not self._refresh_direct_finger_motion_target_path_display(log_message=True):
            return
        self._append_direct_finger_motion_target_error_summary()

    def _load_direct_finger_motion_target_path_from_dialog(self):
        start_dir = self.DIRECT_FINGER_MOTION_TOOL_POSE_LOG_DIR
        if not os.path.isdir(start_dir):
            start_dir = os.path.expanduser("~")

        dialog = QFileDialog(
            self,
            "Load SolidWorks / Target Path CSV",
            start_dir,
            "CSV Files (*.csv);;All Files (*)",
        )
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setViewMode(QFileDialog.Detail)

        if dialog.exec_() != QFileDialog.Accepted:
            return

        selected_files = dialog.selectedFiles()
        if not selected_files:
            return
        self._load_direct_finger_motion_target_path(selected_files[0])

    def _load_direct_finger_motion_target_path(self, csv_path):
        try:
            path_data = self._read_direct_finger_motion_tool_pose_path_data(csv_path)
            points = path_data["points"]
        except Exception as exc:
            self._append_direct_finger_motion_log(
                f"[DFM] Failed to load target path file: {exc}"
            )
            return

        if not points:
            self._append_direct_finger_motion_log(
                f"[DFM] No target path samples found in file: {csv_path}"
            )
            return

        plotter = self._ensure_direct_finger_motion_tool_pose_plot_window()
        if plotter is None:
            self._append_direct_finger_motion_log(
                "[DFM] 3D popup plotter is unavailable: cannot draw target path."
            )
            return

        raw_points_np = np.asarray(points, dtype=float)
        normals_np = None
        if path_data.get("normals"):
            normals_np = np.asarray(path_data["normals"], dtype=float)
        self._direct_finger_motion_tool_pose_target_raw_points = raw_points_np
        self._direct_finger_motion_tool_pose_target_normals = normals_np
        self._direct_finger_motion_tool_pose_target_csv_path = os.path.abspath(csv_path)
        if getattr(self, "_direct_finger_motion_tool_pose_compact_origin", None) is None:
            self._set_direct_finger_motion_tool_pose_target_display_frame(raw_points_np)
        self._reset_direct_finger_motion_target_manual_rotation(refresh_display=False)
        restored_rotation = self._restore_direct_finger_motion_target_rotation_settings(
            csv_path
        )
        if not self._refresh_direct_finger_motion_target_path_display(log_message=True):
            return
        target_points_np = self._direct_finger_motion_tool_pose_target_points

        try:
            self._direct_finger_motion_tool_pose_plot_dialog.show()
            self._direct_finger_motion_tool_pose_plot_dialog.raise_()
            self._direct_finger_motion_tool_pose_plot_dialog.activateWindow()
            self._set_direct_finger_motion_tool_pose_preview_camera(plotter)
            plotter.render()
        except Exception:
            pass

        self._append_direct_finger_motion_log(
            f"[DFM] Loaded target path in popup 3D graph: {csv_path} ({len(target_points_np)} points)"
        )
        if restored_rotation:
            rx_deg, ry_deg, rz_deg = self._get_direct_finger_motion_target_manual_rotation_deg()
            self._append_direct_finger_motion_log(
                "[DFM] Restored saved target rotation: "
                f"Rx={rx_deg:.1f}, Ry={ry_deg:.1f}, Rz={rz_deg:.1f} deg"
            )
        self._set_direct_finger_motion_target_manual_rotation_controls_enabled(True)
        self._append_direct_finger_motion_target_error_summary()

    def _clear_direct_finger_motion_target_path_actors(self):
        plotter = getattr(self, "_direct_finger_motion_tool_pose_plotter", None)
        actors = list(getattr(self, "_direct_finger_motion_tool_pose_target_actors", []))
        if plotter is not None:
            for actor in actors:
                try:
                    plotter.remove_actor(actor)
                except Exception:
                    pass
        self._direct_finger_motion_tool_pose_target_actors = []

    def _clear_direct_finger_motion_target_path(self, log_message=True):
        self._clear_direct_finger_motion_target_path_actors()
        plotter = getattr(self, "_direct_finger_motion_tool_pose_plotter", None)
        if plotter is not None:
            try:
                plotter.render()
            except Exception:
                pass
        self._direct_finger_motion_tool_pose_target_points = None
        self._direct_finger_motion_tool_pose_target_raw_points = None
        self._direct_finger_motion_tool_pose_target_normals = None
        self._direct_finger_motion_tool_pose_target_csv_path = None
        self._direct_finger_motion_tool_pose_target_display_origin = None
        self._direct_finger_motion_tool_pose_target_display_scale = None
        self._reset_direct_finger_motion_target_manual_rotation(refresh_display=False)
        self._set_direct_finger_motion_target_manual_rotation_controls_enabled(False)
        if log_message:
            self._append_direct_finger_motion_log("[DFM] Cleared target path from 3D graph.")

    def _append_direct_finger_motion_target_error_summary(self):
        robot_points = getattr(self, "_direct_finger_motion_tool_pose_plot_points", None)
        target_points = getattr(self, "_direct_finger_motion_tool_pose_target_points", None)
        if robot_points is None or target_points is None:
            return
        if len(robot_points) == 0 or len(target_points) == 0:
            return

        robot_points = np.asarray(robot_points, dtype=float)
        target_points = np.asarray(target_points, dtype=float)
        nearest = []
        for point in robot_points:
            dists = np.linalg.norm(target_points - point, axis=1)
            nearest.append(float(np.min(dists)))
        if not nearest:
            return
        nearest_np = np.asarray(nearest, dtype=float)
        self._append_direct_finger_motion_log(
            "[DFM] Robot-vs-target nearest distance "
            f"(normalized): mean={float(np.mean(nearest_np)):.4f}, "
            f"max={float(np.max(nearest_np)):.4f}, "
            f"samples={len(nearest_np)}"
        )

    @staticmethod
    def _direct_finger_motion_path_cumulative_lengths(points_np):
        points_np = np.asarray(points_np, dtype=float)
        if len(points_np) <= 1:
            return np.zeros(len(points_np), dtype=float)
        segment_lengths = np.linalg.norm(np.diff(points_np, axis=0), axis=1)
        return np.concatenate(([0.0], np.cumsum(segment_lengths)))

    def _resample_direct_finger_motion_path_with_progress(self, points_np, sample_count):
        points_np = np.asarray(points_np, dtype=float)
        sample_count = max(2, int(sample_count))
        if len(points_np) == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), 0.0
        if len(points_np) == 1:
            return (
                np.repeat(points_np[:1], sample_count, axis=0),
                np.linspace(0.0, 1.0, sample_count),
                0.0,
            )

        cumulative = self._direct_finger_motion_path_cumulative_lengths(points_np)
        total_length = float(cumulative[-1])
        if total_length <= 1e-12:
            return (
                np.repeat(points_np[:1], sample_count, axis=0),
                np.linspace(0.0, 1.0, sample_count),
                0.0,
            )

        sample_positions = np.linspace(0.0, total_length, sample_count)
        resampled = np.zeros((sample_count, 3), dtype=float)
        for axis in range(3):
            resampled[:, axis] = np.interp(sample_positions, cumulative, points_np[:, axis])
        return resampled, sample_positions / total_length, total_length

    @staticmethod
    def _nearest_distances_to_path(query_points_np, reference_points_np):
        query_points_np = np.asarray(query_points_np, dtype=float)
        reference_points_np = np.asarray(reference_points_np, dtype=float)
        if len(query_points_np) == 0 or len(reference_points_np) == 0:
            return np.zeros(0, dtype=float)

        nearest = []
        for point in query_points_np:
            dists = np.linalg.norm(reference_points_np - point, axis=1)
            nearest.append(float(np.min(dists)))
        return np.asarray(nearest, dtype=float)

    @staticmethod
    def _nearest_distances_to_polyline(query_points_np, reference_points_np):
        query_points_np = np.asarray(query_points_np, dtype=float)
        reference_points_np = np.asarray(reference_points_np, dtype=float)
        if len(query_points_np) == 0 or len(reference_points_np) == 0:
            return np.zeros(0, dtype=float)
        if len(reference_points_np) == 1:
            return np.linalg.norm(query_points_np - reference_points_np[0], axis=1)

        segment_start = reference_points_np[:-1]
        segment_vec = reference_points_np[1:] - reference_points_np[:-1]
        segment_len2 = np.sum(segment_vec * segment_vec, axis=1)
        nearest = []
        for point in query_points_np:
            rel = point - segment_start
            t = np.zeros(len(segment_start), dtype=float)
            valid = segment_len2 > 1e-18
            t[valid] = np.sum(rel[valid] * segment_vec[valid], axis=1) / segment_len2[valid]
            t = np.clip(t, 0.0, 1.0)
            closest = segment_start + segment_vec * t[:, None]
            nearest.append(float(np.min(np.linalg.norm(closest - point, axis=1))))
        return np.asarray(nearest, dtype=float)

    @staticmethod
    def _discrete_frechet_distance(path_a_np, path_b_np):
        path_a_np = np.asarray(path_a_np, dtype=float)
        path_b_np = np.asarray(path_b_np, dtype=float)
        if len(path_a_np) == 0 or len(path_b_np) == 0:
            return 0.0

        distances = np.linalg.norm(path_a_np[:, None, :] - path_b_np[None, :, :], axis=2)
        cache = np.full(distances.shape, np.inf, dtype=float)
        cache[0, 0] = distances[0, 0]
        for i in range(len(path_a_np)):
            for j in range(len(path_b_np)):
                if i == 0 and j == 0:
                    continue
                previous = []
                if i > 0:
                    previous.append(cache[i - 1, j])
                if j > 0:
                    previous.append(cache[i, j - 1])
                if i > 0 and j > 0:
                    previous.append(cache[i - 1, j - 1])
                cache[i, j] = max(distances[i, j], min(previous))
        return float(cache[-1, -1])

    @staticmethod
    def _unit_vector_or_none(vector):
        vector = np.asarray(vector, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return None
        return vector / norm

    @classmethod
    def _target_path_coordinate_frame(cls, target_points_np):
        target_points_np = np.asarray(target_points_np, dtype=float)
        if len(target_points_np) == 0:
            return np.zeros(3, dtype=float), np.eye(3), "identity"

        origin = np.asarray(target_points_np[0], dtype=float)
        tangent = cls._unit_vector_or_none(target_points_np[-1] - target_points_np[0])
        frame_method = "start_to_end"
        centered = target_points_np - np.mean(target_points_np, axis=0)

        if tangent is None and len(target_points_np) >= 2:
            try:
                _, _, vh_matrix = np.linalg.svd(centered, full_matrices=False)
                tangent = cls._unit_vector_or_none(vh_matrix[0])
                frame_method = "pca_tangent"
            except Exception:
                tangent = None

        if tangent is None:
            tangent = np.array([1.0, 0.0, 0.0], dtype=float)
            frame_method = "identity_fallback"

        perpendicular = centered - (centered @ tangent)[:, None] * tangent
        normal_1 = None
        if np.linalg.norm(perpendicular) > 1e-12:
            try:
                _, _, vh_matrix = np.linalg.svd(perpendicular, full_matrices=False)
                normal_1 = cls._unit_vector_or_none(
                    vh_matrix[0] - float(np.dot(vh_matrix[0], tangent)) * tangent
                )
            except Exception:
                normal_1 = None

        if normal_1 is None:
            candidates = np.eye(3, dtype=float)
            seed = candidates[int(np.argmin(np.abs(candidates @ tangent)))]
            normal_1 = cls._unit_vector_or_none(seed - float(np.dot(seed, tangent)) * tangent)

        normal_2 = cls._unit_vector_or_none(np.cross(tangent, normal_1))
        if normal_2 is None:
            normal_2 = np.array([0.0, 0.0, 1.0], dtype=float)
        normal_1 = cls._unit_vector_or_none(np.cross(normal_2, tangent))
        if normal_1 is None:
            normal_1 = np.array([0.0, 1.0, 0.0], dtype=float)

        frame = np.column_stack((tangent, normal_1, normal_2))
        return origin, frame, frame_method

    @staticmethod
    def _project_points_to_frame(points_np, origin, frame):
        points_np = np.asarray(points_np, dtype=float)
        origin = np.asarray(origin, dtype=float).reshape(3)
        frame = np.asarray(frame, dtype=float).reshape(3, 3)
        return (points_np - origin) @ frame

    @staticmethod
    def _set_3d_axes_equal(ax, points_np):
        points_np = np.asarray(points_np, dtype=float)
        if len(points_np) == 0:
            return
        mins = np.min(points_np, axis=0)
        maxs = np.max(points_np, axis=0)
        center = 0.5 * (mins + maxs)
        radius = 0.5 * float(np.max(maxs - mins))
        radius = max(radius, 1e-6)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass

    @staticmethod
    def _normalize_direct_finger_motion_vectors(vectors_np):
        if vectors_np is None:
            return None
        vectors_np = np.asarray(vectors_np, dtype=float)
        if vectors_np.ndim != 2 or vectors_np.shape[1] != 3:
            return None
        norms = np.linalg.norm(vectors_np, axis=1)
        normalized = np.zeros_like(vectors_np, dtype=float)
        valid = norms > 1e-12
        if np.any(valid):
            normalized[valid] = vectors_np[valid] / norms[valid, None]
        return normalized

    @classmethod
    def _resample_direct_finger_motion_vectors_by_path_progress(
        cls,
        path_points_np,
        vectors_np,
        sample_progress,
    ):
        path_points_np = np.asarray(path_points_np, dtype=float)
        vectors_np = cls._normalize_direct_finger_motion_vectors(vectors_np)
        sample_progress = np.asarray(sample_progress, dtype=float)
        if (
            vectors_np is None
            or len(path_points_np) != len(vectors_np)
            or len(path_points_np) == 0
        ):
            return None

        if len(path_points_np) == 1:
            return np.repeat(vectors_np[:1], len(sample_progress), axis=0)

        cumulative = cls._direct_finger_motion_path_cumulative_lengths(path_points_np)
        total_length = float(cumulative[-1])
        if total_length <= 1e-12:
            return np.repeat(vectors_np[:1], len(sample_progress), axis=0)

        source_progress = cumulative / total_length
        resampled = np.zeros((len(sample_progress), 3), dtype=float)
        for axis in range(3):
            resampled[:, axis] = np.interp(
                sample_progress,
                source_progress,
                vectors_np[:, axis],
            )
        return cls._normalize_direct_finger_motion_vectors(resampled)

    @staticmethod
    def _prepare_direct_finger_motion_quaternion_series(quaternions_np):
        if quaternions_np is None:
            return None
        quaternions_np = np.asarray(quaternions_np, dtype=float)
        if quaternions_np.ndim != 2 or quaternions_np.shape[1] != 4:
            return None
        norms = np.linalg.norm(quaternions_np, axis=1)
        if np.any(norms <= 1e-12):
            return None
        prepared = quaternions_np / norms[:, None]
        for index in range(1, len(prepared)):
            if float(np.dot(prepared[index - 1], prepared[index])) < 0.0:
                prepared[index] *= -1.0
        return prepared

    @classmethod
    def _resample_direct_finger_motion_quaternions_by_path_progress(
        cls,
        path_points_np,
        quaternions_np,
        sample_progress,
    ):
        path_points_np = np.asarray(path_points_np, dtype=float)
        quaternions_np = cls._prepare_direct_finger_motion_quaternion_series(
            quaternions_np
        )
        sample_progress = np.asarray(sample_progress, dtype=float)
        if (
            quaternions_np is None
            or len(path_points_np) != len(quaternions_np)
            or len(path_points_np) == 0
        ):
            return None

        if len(path_points_np) == 1:
            return np.repeat(quaternions_np[:1], len(sample_progress), axis=0)

        cumulative = cls._direct_finger_motion_path_cumulative_lengths(path_points_np)
        total_length = float(cumulative[-1])
        if total_length <= 1e-12:
            return np.repeat(quaternions_np[:1], len(sample_progress), axis=0)

        source_progress = cumulative / total_length
        resampled = np.zeros((len(sample_progress), 4), dtype=float)
        for axis in range(4):
            resampled[:, axis] = np.interp(
                sample_progress,
                source_progress,
                quaternions_np[:, axis],
            )
        norms = np.linalg.norm(resampled, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            return None
        resampled[valid] = resampled[valid] / norms[valid, None]
        if not np.all(valid):
            resampled[~valid] = resampled[np.where(valid)[0][0]]
        return resampled

    @staticmethod
    def _rotate_direct_finger_motion_vectors_by_quaternions(vectors_np, quaternions_np):
        vectors_np = np.asarray(vectors_np, dtype=float)
        quaternions_np = np.asarray(quaternions_np, dtype=float)
        if vectors_np.ndim == 1:
            vectors_np = np.repeat(vectors_np.reshape(1, 3), len(quaternions_np), axis=0)
        q_vec = quaternions_np[:, 1:4]
        q_w = quaternions_np[:, 0:1]
        cross_1 = np.cross(q_vec, vectors_np)
        cross_2 = np.cross(q_vec, cross_1 + q_w * vectors_np)
        return vectors_np + 2.0 * cross_2

    @classmethod
    def _tool_normals_from_direct_finger_motion_quaternions(cls, quaternions_np):
        quaternions_np = cls._prepare_direct_finger_motion_quaternion_series(
            quaternions_np
        )
        if quaternions_np is None:
            return None
        local_tool_z = np.array([0.0, 0.0, 1.0], dtype=float)
        normals = cls._rotate_direct_finger_motion_vectors_by_quaternions(
            local_tool_z,
            quaternions_np,
        )
        return cls._normalize_direct_finger_motion_vectors(normals)

    @classmethod
    def _estimate_direct_finger_motion_target_normals(cls, target_points_np):
        target_points_np = np.asarray(target_points_np, dtype=float)
        if len(target_points_np) == 0:
            return np.zeros((0, 3), dtype=float)
        if len(target_points_np) == 1:
            return np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=float), 1, axis=0)

        tangents = np.gradient(target_points_np, axis=0)
        tangents = cls._normalize_direct_finger_motion_vectors(tangents)
        if tangents is None:
            tangents = np.repeat(np.array([[1.0, 0.0, 0.0]], dtype=float), len(target_points_np), axis=0)
        curvature = np.gradient(tangents, axis=0)
        curvature -= np.sum(curvature * tangents, axis=1)[:, None] * tangents
        normals = cls._normalize_direct_finger_motion_vectors(curvature)
        _, frame, _ = cls._target_path_coordinate_frame(target_points_np)
        fallback = np.asarray(frame[:, 1], dtype=float)

        fixed_normals = np.zeros((len(target_points_np), 3), dtype=float)
        previous = fallback
        for index in range(len(target_points_np)):
            candidate = normals[index] if normals is not None else fallback
            if float(np.linalg.norm(candidate)) <= 1e-12:
                candidate = previous
            if float(np.dot(previous, candidate)) < 0.0:
                candidate = -candidate
            fixed_normals[index] = candidate
            previous = candidate
        fixed_normals = cls._normalize_direct_finger_motion_vectors(fixed_normals)
        if fixed_normals is None:
            return np.repeat(np.array([[0.0, 0.0, 1.0]], dtype=float), len(target_points_np), axis=0)

        if len(fixed_normals) >= 5:
            window = min(9, len(fixed_normals))
            if window % 2 == 0:
                window -= 1
            if window >= 5:
                pad = window // 2
                padded = np.pad(fixed_normals, ((pad, pad), (0, 0)), mode="edge")
                smoothed = np.zeros_like(fixed_normals)
                kernel = np.ones(window, dtype=float) / float(window)
                for axis in range(3):
                    smoothed[:, axis] = np.convolve(
                        padded[:, axis],
                        kernel,
                        mode="valid",
                    )
                smoothed -= np.sum(smoothed * tangents, axis=1)[:, None] * tangents
                smoothed = cls._normalize_direct_finger_motion_vectors(smoothed)
                if smoothed is not None:
                    previous = smoothed[0]
                    for index in range(len(smoothed)):
                        candidate = smoothed[index]
                        if float(np.linalg.norm(candidate)) <= 1e-12:
                            candidate = previous
                        if float(np.dot(previous, candidate)) < 0.0:
                            candidate = -candidate
                        smoothed[index] = candidate
                        previous = candidate
                    fixed_normals = cls._normalize_direct_finger_motion_vectors(smoothed)
        return fixed_normals

    @staticmethod
    def _direct_finger_motion_normal_angle_errors_deg(
        robot_normals_np,
        target_normals_np,
        unsigned=True,
    ):
        robot_normals_np = np.asarray(robot_normals_np, dtype=float)
        target_normals_np = np.asarray(target_normals_np, dtype=float)
        if len(robot_normals_np) == 0 or len(robot_normals_np) != len(target_normals_np):
            return np.zeros(0, dtype=float)
        dots = np.sum(robot_normals_np * target_normals_np, axis=1)
        if unsigned:
            dots = np.abs(dots)
        dots = np.clip(dots, -1.0, 1.0)
        return np.degrees(np.arccos(dots))

    def _compute_direct_finger_motion_path_analysis_for_robot(
        self,
        robot_raw_np,
        target_raw_np,
        label="Robot path",
        csv_path=None,
        robot_quaternions_np=None,
        target_normals_np=None,
    ):
        if robot_raw_np is None or target_raw_np is None:
            return None
        robot_raw_np = np.asarray(robot_raw_np, dtype=float)
        target_raw_np = np.asarray(target_raw_np, dtype=float)
        if len(robot_raw_np) < 2 or len(target_raw_np) < 2:
            return None

        sample_count = min(max(len(robot_raw_np), len(target_raw_np)), 500)
        sample_count = max(50, int(sample_count))
        robot_samples, progress, robot_length = (
            self._resample_direct_finger_motion_path_with_progress(
                robot_raw_np, sample_count
            )
        )
        target_samples, _, target_length = (
            self._resample_direct_finger_motion_path_with_progress(
                target_raw_np, sample_count
            )
        )
        diff = robot_samples - target_samples
        progress_error = np.linalg.norm(diff, axis=1)
        nearest_error = self._nearest_distances_to_polyline(robot_samples, target_raw_np)
        target_to_robot_error = self._nearest_distances_to_polyline(target_samples, robot_raw_np)
        start_error = float(np.linalg.norm(robot_samples[0] - target_samples[0]))
        end_error = float(np.linalg.norm(robot_samples[-1] - target_samples[-1]))
        origin, frame, frame_method = self._target_path_coordinate_frame(target_samples)
        robot_frame_samples = self._project_points_to_frame(robot_samples, origin, frame)
        target_frame_samples = self._project_points_to_frame(target_samples, origin, frame)
        diff_frame = diff @ frame
        along_error = diff_frame[:, 0]
        normal_1_error = diff_frame[:, 1]
        normal_2_error = diff_frame[:, 2]
        cross_track_error = np.linalg.norm(diff_frame[:, 1:3], axis=1)
        target_length_safe = max(float(target_length), 1e-12)
        directed_hausdorff_robot_to_target = float(np.max(nearest_error))
        directed_hausdorff_target_to_robot = float(np.max(target_to_robot_error))
        symmetric_hausdorff = max(
            directed_hausdorff_robot_to_target,
            directed_hausdorff_target_to_robot,
        )
        robust_hausdorff_p95 = max(
            float(np.percentile(nearest_error, 95.0)),
            float(np.percentile(target_to_robot_error, 95.0)),
        )
        frechet_distance = self._discrete_frechet_distance(robot_samples, target_samples)
        robot_sample_quaternions = (
            self._resample_direct_finger_motion_quaternions_by_path_progress(
                robot_raw_np,
                robot_quaternions_np,
                progress,
            )
        )
        robot_tool_normals = None
        if robot_sample_quaternions is not None:
            robot_tool_normals = self._tool_normals_from_direct_finger_motion_quaternions(
                robot_sample_quaternions
            )

        target_normal_source = "curve_estimated"
        target_sample_normals = None
        if target_normals_np is not None:
            target_sample_normals = (
                self._resample_direct_finger_motion_vectors_by_path_progress(
                    target_raw_np,
                    target_normals_np,
                    progress,
                )
            )
            if target_sample_normals is not None:
                target_normal_source = "target_csv"
        if target_sample_normals is None:
            target_sample_normals = self._estimate_direct_finger_motion_target_normals(
                target_samples
            )

        normal_angle_error_deg = np.zeros(0, dtype=float)
        normal_compare_mode = "unavailable"
        if (
            robot_tool_normals is not None
            and target_sample_normals is not None
            and len(robot_tool_normals) == len(target_sample_normals)
        ):
            normal_compare_mode = (
                "directed" if target_normal_source == "target_csv" else "unsigned"
            )
            normal_angle_error_deg = (
                self._direct_finger_motion_normal_angle_errors_deg(
                    robot_tool_normals,
                    target_sample_normals,
                    unsigned=(target_normal_source != "target_csv"),
                )
            )

        if len(normal_angle_error_deg) > 0:
            normal_rms_deg = float(np.sqrt(np.mean(normal_angle_error_deg ** 2)))
            normal_mean_deg = float(np.mean(normal_angle_error_deg))
            normal_p95_deg = float(np.percentile(normal_angle_error_deg, 95.0))
            normal_max_deg = float(np.max(normal_angle_error_deg))
        else:
            normal_rms_deg = np.nan
            normal_mean_deg = np.nan
            normal_p95_deg = np.nan
            normal_max_deg = np.nan

        metrics = {
            "samples": sample_count,
            "rms_m": float(np.sqrt(np.mean(progress_error ** 2))),
            "mean_m": float(np.mean(progress_error)),
            "median_m": float(np.median(progress_error)),
            "std_m": float(np.std(progress_error)),
            "max_m": float(np.max(progress_error)),
            "p95_m": float(np.percentile(progress_error, 95.0)),
            "nearest_rms_m": float(np.sqrt(np.mean(nearest_error ** 2))),
            "nearest_mean_m": float(np.mean(nearest_error)),
            "nearest_max_m": float(np.max(nearest_error)),
            "nearest_p95_m": float(np.percentile(nearest_error, 95.0)),
            "target_to_robot_nearest_rms_m": float(np.sqrt(np.mean(target_to_robot_error ** 2))),
            "target_to_robot_nearest_mean_m": float(np.mean(target_to_robot_error)),
            "target_to_robot_nearest_p95_m": float(np.percentile(target_to_robot_error, 95.0)),
            "target_to_robot_nearest_max_m": directed_hausdorff_target_to_robot,
            "cross_track_rms_m": float(np.sqrt(np.mean(cross_track_error ** 2))),
            "cross_track_mean_m": float(np.mean(cross_track_error)),
            "cross_track_p95_m": float(np.percentile(cross_track_error, 95.0)),
            "cross_track_max_m": float(np.max(cross_track_error)),
            "along_track_bias_m": float(np.mean(along_error)),
            "along_track_rms_m": float(np.sqrt(np.mean(along_error ** 2))),
            "normal_1_rms_m": float(np.sqrt(np.mean(normal_1_error ** 2))),
            "normal_2_rms_m": float(np.sqrt(np.mean(normal_2_error ** 2))),
            "directed_hausdorff_robot_to_target_m": directed_hausdorff_robot_to_target,
            "directed_hausdorff_target_to_robot_m": directed_hausdorff_target_to_robot,
            "symmetric_hausdorff_m": symmetric_hausdorff,
            "robust_hausdorff_p95_m": robust_hausdorff_p95,
            "discrete_frechet_m": frechet_distance,
            "start_error_m": start_error,
            "end_error_m": end_error,
            "robot_length_m": float(robot_length),
            "target_length_m": float(target_length),
            "length_error_m": float(robot_length - target_length),
            "length_error_abs_m": float(abs(robot_length - target_length)),
            "length_ratio": float(robot_length / target_length_safe),
            "normalized_rms_percent": float(
                np.sqrt(np.mean(progress_error ** 2)) / target_length_safe * 100.0
            ),
            "normalized_p95_percent": float(
                np.percentile(progress_error, 95.0) / target_length_safe * 100.0
            ),
            "normal_angle_rms_deg": normal_rms_deg,
            "normal_angle_mean_deg": normal_mean_deg,
            "normal_angle_p95_deg": normal_p95_deg,
            "normal_angle_max_deg": normal_max_deg,
            "normal_target_source": target_normal_source,
            "normal_compare_mode": normal_compare_mode,
            "normal_tool_axis": "local_z",
            "frame_method": frame_method,
            "method_label": str(label),
            "csv_path": "" if csv_path is None else os.path.abspath(str(csv_path)),
        }
        return {
            "label": str(label),
            "csv_path": "" if csv_path is None else os.path.abspath(str(csv_path)),
            "robot_raw": np.asarray(robot_raw_np, dtype=float),
            "target_raw": np.asarray(target_raw_np, dtype=float),
            "robot_samples": robot_samples,
            "target_samples": target_samples,
            "robot_frame_samples": robot_frame_samples,
            "target_frame_samples": target_frame_samples,
            "progress": progress,
            "diff": diff,
            "diff_frame": diff_frame,
            "cross_track_error": cross_track_error,
            "progress_error": progress_error,
            "nearest_error": nearest_error,
            "target_to_robot_error": target_to_robot_error,
            "robot_tool_normals": robot_tool_normals,
            "target_normals": target_sample_normals,
            "normal_angle_error_deg": normal_angle_error_deg,
            "frame_origin": origin,
            "frame_matrix": frame,
            "frame_method": frame_method,
            "metrics": metrics,
        }

    def _direct_finger_motion_comparison_robot_entries(self):
        paths = list(getattr(self, "_direct_finger_motion_tool_pose_comparison_paths", []))
        valid_paths = [
            path for path in paths
            if path.get("raw_points") is not None and len(path.get("raw_points")) >= 2
        ]
        if valid_paths:
            return valid_paths

        robot_raw_np = getattr(self, "_direct_finger_motion_tool_pose_raw_points", None)
        if robot_raw_np is None or len(robot_raw_np) < 2:
            return []
        return [
            {
                "label": "Robot path",
                "csv_path": "",
                "raw_points": np.asarray(robot_raw_np, dtype=float),
                "quaternions": getattr(
                    self, "_direct_finger_motion_tool_pose_quaternions", None
                ),
                "color": self._direct_finger_motion_robot_path_color(0),
                "is_primary": True,
            }
        ]

    def _compute_direct_finger_motion_path_analyses(self):
        target_raw_np = self._compute_direct_finger_motion_target_aligned_raw_points()
        if target_raw_np is None or len(target_raw_np) < 2:
            return []
        target_normals_np = self._compute_direct_finger_motion_target_aligned_normals()

        analyses = []
        for index, entry in enumerate(self._direct_finger_motion_comparison_robot_entries()):
            analysis = self._compute_direct_finger_motion_path_analysis_for_robot(
                entry.get("raw_points"),
                target_raw_np,
                label=entry.get("label") or f"Robot path {index + 1}",
                csv_path=entry.get("csv_path"),
                robot_quaternions_np=entry.get("quaternions"),
                target_normals_np=target_normals_np,
            )
            if analysis is not None:
                analysis["color"] = (
                    entry.get("color")
                    or self._direct_finger_motion_robot_path_color(index)
                )
                start_translation = np.asarray(
                    entry.get("start_translation", np.zeros(3, dtype=float)),
                    dtype=float,
                )
                analysis["start_translation"] = start_translation
                analysis["metrics"]["start_translation_x_m"] = float(
                    start_translation[0]
                )
                analysis["metrics"]["start_translation_y_m"] = float(
                    start_translation[1]
                )
                analysis["metrics"]["start_translation_z_m"] = float(
                    start_translation[2]
                )
                analysis["metrics"]["start_translation_magnitude_m"] = float(
                    np.linalg.norm(start_translation)
                )
                duration_s = entry.get("duration_s")
                try:
                    duration_s = float(duration_s)
                except Exception:
                    duration_s = np.nan
                if not np.isfinite(duration_s) or duration_s < 0.0:
                    duration_s = np.nan
                analysis["duration_s"] = duration_s
                analysis["time_source"] = str(entry.get("time_source") or "")
                analysis["metrics"]["duration_s"] = float(duration_s)
                analysis["metrics"]["duration_source"] = str(
                    entry.get("time_source") or ""
                )
                if np.isfinite(duration_s) and duration_s > 1e-12:
                    analysis["metrics"]["average_speed_m_per_s"] = float(
                        analysis["metrics"]["robot_length_m"] / duration_s
                    )
                else:
                    analysis["metrics"]["average_speed_m_per_s"] = np.nan
                analyses.append(analysis)
        return analyses

    def _compute_direct_finger_motion_path_analysis(self):
        analyses = self._compute_direct_finger_motion_path_analyses()
        return analyses[0] if analyses else None

    @staticmethod
    def _finite_direct_finger_motion_metric(metrics, key):
        try:
            value = float(metrics.get(key, np.nan))
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return value

    @classmethod
    def _format_direct_finger_motion_duration(cls, metrics):
        duration_s = cls._finite_direct_finger_motion_metric(metrics, "duration_s")
        if duration_s is None:
            return "N/A"
        return f"{duration_s:.2f} s"

    @classmethod
    def _format_direct_finger_motion_speed(cls, metrics):
        speed = cls._finite_direct_finger_motion_metric(
            metrics,
            "average_speed_m_per_s",
        )
        if speed is None:
            return "N/A"
        return f"{speed:.4f} m/s"

    @staticmethod
    def _format_direct_finger_motion_path_analysis_summary(metrics):
        def mm(key):
            return metrics[key] * 1000.0

        lines = [
            "Tool Pose Path Analysis - Publication Metrics",
            "",
            "Coordinate frame:",
            "  T  = target path tangent axis",
            "  N1/N2 = perpendicular target-frame axes from target-path PCA",
            f"  Frame construction: {metrics.get('frame_method', 'unknown')}",
            "",
            f"Samples: {metrics['samples']}",
            f"Same-progress RMSE / MAE / median: "
            f"{mm('rms_m'):.2f} / {mm('mean_m'):.2f} / {mm('median_m'):.2f} mm",
            f"Same-progress SD / P95 / max: "
            f"{mm('std_m'):.2f} / {mm('p95_m'):.2f} / {mm('max_m'):.2f} mm",
            f"Normalized RMSE / P95: "
            f"{metrics['normalized_rms_percent']:.3f}% / {metrics['normalized_p95_percent']:.3f}% of target length",
            "",
            f"Cross-track RMSE / mean / P95 / max: "
            f"{mm('cross_track_rms_m'):.2f} / {mm('cross_track_mean_m'):.2f} / "
            f"{mm('cross_track_p95_m'):.2f} / {mm('cross_track_max_m'):.2f} mm",
            f"Along-track bias / RMSE: "
            f"{mm('along_track_bias_m'):+.2f} / {mm('along_track_rms_m'):.2f} mm",
            f"N1 / N2 RMSE: {mm('normal_1_rms_m'):.2f} / {mm('normal_2_rms_m'):.2f} mm",
            f"Tool-normal angle RMSE / mean / P95 / max: "
            f"{metrics.get('normal_angle_rms_deg', np.nan):.2f} / "
            f"{metrics.get('normal_angle_mean_deg', np.nan):.2f} / "
            f"{metrics.get('normal_angle_p95_deg', np.nan):.2f} / "
            f"{metrics.get('normal_angle_max_deg', np.nan):.2f} deg",
            f"Normal source / compare mode: "
            f"{metrics.get('normal_target_source', 'unknown')} / "
            f"{metrics.get('normal_compare_mode', 'unknown')}",
            "",
            f"Nearest-segment RMSE / mean / P95 / max (robot->target): "
            f"{mm('nearest_rms_m'):.2f} / {mm('nearest_mean_m'):.2f} / "
            f"{mm('nearest_p95_m'):.2f} / {mm('nearest_max_m'):.2f} mm",
            f"Nearest-segment RMSE / mean / P95 / max (target->robot): "
            f"{mm('target_to_robot_nearest_rms_m'):.2f} / "
            f"{mm('target_to_robot_nearest_mean_m'):.2f} / "
            f"{mm('target_to_robot_nearest_p95_m'):.2f} / "
            f"{mm('target_to_robot_nearest_max_m'):.2f} mm",
            f"Hausdorff robot->target / target->robot / symmetric: "
            f"{mm('directed_hausdorff_robot_to_target_m'):.2f} / "
            f"{mm('directed_hausdorff_target_to_robot_m'):.2f} / "
            f"{mm('symmetric_hausdorff_m'):.2f} mm",
            f"Robust Hausdorff P95 / discrete Frechet: "
            f"{mm('robust_hausdorff_p95_m'):.2f} / {mm('discrete_frechet_m'):.2f} mm",
            "",
            f"Start error: {mm('start_error_m'):.2f} mm",
            f"End error: {mm('end_error_m'):.2f} mm",
            f"Robot path length: {metrics['robot_length_m']:.4f} m",
            f"Target path length: {metrics['target_length_m']:.4f} m",
            f"Completion time: {DirectFingerMotionMixin._format_direct_finger_motion_duration(metrics)}",
            f"Average robot speed: {DirectFingerMotionMixin._format_direct_finger_motion_speed(metrics)}",
            f"Length difference (robot - target): {metrics['length_error_m']:.4f} m "
            f"({mm('length_error_abs_m'):.2f} mm abs)",
            f"Length ratio (robot / target): {metrics['length_ratio']:.4f}",
        ]
        return "\n".join(lines)

    def _export_direct_finger_motion_path_analysis_metrics(self, analysis):
        if isinstance(analysis, (list, tuple)):
            analyses = [item for item in analysis if isinstance(item, dict)]
        elif isinstance(analysis, dict):
            analyses = [analysis]
        else:
            analyses = []
        if not analyses:
            return

        default_name = time.strftime("path_analysis_metrics_%Y%m%d_%H%M%S.csv")
        default_path = os.path.join(
            self.DIRECT_FINGER_MOTION_TOOL_POSE_LOG_DIR,
            default_name,
        )
        csv_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Path Analysis Metrics",
            default_path,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not csv_path:
            return

        try:
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["method", "csv_path", "metric", "value", "unit"])
                for item in analyses:
                    metrics = item.get("metrics")
                    if not isinstance(metrics, dict):
                        continue
                    method = self._direct_finger_motion_analysis_display_label(
                        item.get("label") or metrics.get("method_label")
                    )
                    source_csv = str(item.get("csv_path") or metrics.get("csv_path") or "")
                    for key in sorted(metrics.keys()):
                        value = metrics[key]
                        unit = "m"
                        if key.endswith("_percent"):
                            unit = "%"
                        elif key.endswith("_m_per_s"):
                            unit = "m/s"
                        elif key.endswith("_s"):
                            unit = "s"
                        elif key.endswith("_deg"):
                            unit = "deg"
                        elif key in {"samples"}:
                            unit = "count"
                        elif key in {
                            "length_ratio",
                            "frame_method",
                            "method_label",
                            "csv_path",
                            "duration_source",
                            "normal_target_source",
                            "normal_compare_mode",
                            "normal_tool_axis",
                        }:
                            unit = ""
                        writer.writerow([method, source_csv, key, value, unit])
            self._append_direct_finger_motion_log(
                f"[DFM] Exported path analysis metrics: {csv_path}"
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Export Path Analysis",
                f"Failed to export metrics:\n{exc}",
            )

    def _format_direct_finger_motion_multi_path_analysis_summary(self, analyses):
        def mm(metrics, key):
            return float(metrics[key]) * 1000.0

        lines = [
            "Robot Tool Path Method Comparison",
            "",
            "Each robot path is compared against the same aligned target path.",
            (
                "Recommended paper columns: RMSE, Hausdorff distance, "
                "tool-normal RMSE, length ratio, completion time."
            ),
            "",
            (
                f"{'Method':<28} {'RMSE':>8} {'Hausdorff':>10} "
                f"{'NormRMSE':>9} {'LenRatio':>9} {'Time':>9}"
            ),
            (
                f"{'-' * 28} {'-' * 8:>8} {'-' * 10:>10} "
                f"{'-' * 9:>9} {'-' * 9:>9} {'-' * 9:>9}"
            ),
        ]

        rmse_ranked = []
        hausdorff_ranked = []
        timed = []
        for item in analyses:
            metrics = item["metrics"]
            label = self._direct_finger_motion_analysis_display_label(
                item.get("label") or metrics.get("method_label"),
            )
            short_label = label[:28]
            rmse_ranked.append((float(metrics["rms_m"]), label, metrics))
            hausdorff_ranked.append(
                (float(metrics["symmetric_hausdorff_m"]), label, metrics)
            )
            duration_s = DirectFingerMotionMixin._finite_direct_finger_motion_metric(
                metrics,
                "duration_s",
            )
            if duration_s is not None:
                timed.append((duration_s, label, metrics))
                duration_text = f"{duration_s:.2f}s"
            else:
                duration_text = "N/A"
            normal_rms = DirectFingerMotionMixin._finite_direct_finger_motion_metric(
                metrics,
                "normal_angle_rms_deg",
            )
            normal_text = f"{normal_rms:.2f}" if normal_rms is not None else "N/A"
            lines.append(
                f"{short_label:<28} "
                f"{mm(metrics, 'rms_m'):>8.2f} "
                f"{mm(metrics, 'symmetric_hausdorff_m'):>10.2f} "
                f"{normal_text:>9} "
                f"{float(metrics['length_ratio']):>9.4f} "
                f"{duration_text:>9}"
            )

        if rmse_ranked:
            best_rmse = min(rmse_ranked, key=lambda item: item[0])
            best_hausdorff = min(hausdorff_ranked, key=lambda item: item[0])
            lines.extend(
                [
                    "",
                    f"Best RMSE: {best_rmse[1]} ({best_rmse[0] * 1000.0:.2f} mm)",
                    (
                        "Best Hausdorff: "
                        f"{best_hausdorff[1]} "
                        f"({best_hausdorff[0] * 1000.0:.2f} mm)"
                    ),
                ]
            )
            if timed:
                fastest = min(timed, key=lambda item: item[0])
                lines.append(
                    f"Fastest completion time: {fastest[1]} ({fastest[0]:.2f} s)"
                )
            normal_ranked = [
                (
                    DirectFingerMotionMixin._finite_direct_finger_motion_metric(
                        item["metrics"],
                        "normal_angle_rms_deg",
                    ),
                    self._direct_finger_motion_analysis_display_label(
                        item.get("label"),
                    ),
                )
                for item in analyses
            ]
            normal_ranked = [
                (value, label) for value, label in normal_ranked if value is not None
            ]
            if normal_ranked:
                best_normal = min(normal_ranked, key=lambda item: item[0])
                source = analyses[0]["metrics"].get("normal_target_source", "unknown")
                compare_mode = analyses[0]["metrics"].get("normal_compare_mode", "unknown")
                lines.append(
                    f"Best tool-normal RMSE: {best_normal[1]} "
                    f"({best_normal[0]:.2f} deg)"
                )
                lines.append(
                    f"Normal reference: {source}; compare mode: {compare_mode}"
                )
        return "\n".join(lines)

    @staticmethod
    def _direct_finger_motion_multi_path_graph_specs():
        return [
            ("overlay", "Trajectory Following", "trajectory_following"),
            (
                "progress_error",
                "Trajectory Deviation",
                "trajectory_deviation",
            ),
            ("rmse", "RMSE", "rmse"),
            ("hausdorff", "Hausdorff Distance", "hausdorff_distance"),
            ("normal_angle", "Normal Direction Error", "normal_direction_error"),
            ("completion_time", "Completion Time", "completion_time"),
            ("length_ratio", "Path Length Ratio", "path_length_ratio"),
        ]

    @classmethod
    def _direct_finger_motion_preview_matplotlib_view(cls):
        return (
            cls.DIRECT_FINGER_MOTION_PREVIEW_VIEW_ELEV_DEG,
            cls.DIRECT_FINGER_MOTION_PREVIEW_VIEW_AZIM_DEG,
        )

    @staticmethod
    def _set_direct_finger_motion_tool_pose_preview_camera(plotter):
        if plotter is None:
            return
        try:
            plotter.view_isometric()
        except Exception:
            pass
        try:
            plotter.reset_camera()
        except Exception:
            pass

    @classmethod
    def _direct_finger_motion_trajectory_following_view_presets(cls):
        preview_elev, preview_azim = cls._direct_finger_motion_preview_matplotlib_view()
        return [
            ("default", "Viewer Match", preview_elev, preview_azim),
            ("front", "Front", 18, -90),
            ("side", "Side", 18, 0),
            ("top", "Top", 90, -90),
            ("isometric_left", "Isometric Left", 28, -35),
            ("isometric_right", "Isometric Right", 28, -125),
        ]

    @staticmethod
    def _direct_finger_motion_multi_path_export_figure_size(graph_key):
        if graph_key == "overlay":
            return (7.25, 6.3)
        return (8.8, 6.4)

    @staticmethod
    def _safe_direct_finger_motion_export_name(text):
        safe_chars = []
        for char in str(text).strip().lower():
            if char.isalnum():
                safe_chars.append(char)
            elif char in {" ", "-", "_"}:
                safe_chars.append("_")
        safe_name = "".join(safe_chars).strip("_")
        while "__" in safe_name:
            safe_name = safe_name.replace("__", "_")
        return safe_name or "analysis_graph"

    def _direct_finger_motion_multi_path_plot_context(self, analyses):
        labels = [
            self._direct_finger_motion_analysis_display_label(item.get("label"))
            for item in analyses
        ]
        return {
            "target_frame_samples": analyses[0]["target_frame_samples"],
            "labels": labels,
            "short_labels": labels,
            "colors": [
                str(item.get("color") or self._direct_finger_motion_robot_path_color(index))
                for index, item in enumerate(analyses)
            ],
            "x_pos": np.arange(len(analyses)),
        }

    @staticmethod
    def _style_direct_finger_motion_axes(ax, grid_axis="y"):
        ax.grid(True, axis=grid_axis, alpha=0.28)
        try:
            ax.set_axisbelow(True)
        except Exception:
            pass
        for side in ("top", "right"):
            try:
                ax.spines[side].set_visible(False)
            except Exception:
                pass

    @staticmethod
    def _annotate_direct_finger_motion_bars(ax, bars, labels):
        heights = [float(bar.get_height()) for bar in bars]
        max_height = max(heights) if heights else 0.0
        upper = max(max_height * 1.22, 1.0)
        ax.set_ylim(0.0, upper)
        offset = upper * 0.025
        for bar, label in zip(bars, labels):
            ax.text(
                bar.get_x() + bar.get_width() * 0.5,
                float(bar.get_height()) + offset,
                str(label),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    @staticmethod
    def _highlight_direct_finger_motion_best_bar(bars, values):
        known_values = [
            (index, float(value))
            for index, value in enumerate(values)
            if value is not None and np.isfinite(float(value))
        ]
        if not known_values:
            return
        best_index, _ = min(known_values, key=lambda item: item[1])
        for index, bar in enumerate(bars):
            if index == best_index:
                bar.set_edgecolor("#111111")
                bar.set_linewidth(2.0)
            else:
                bar.set_edgecolor("#ffffff")
                bar.set_linewidth(0.6)

    @staticmethod
    def _set_direct_finger_motion_horizontal_xticklabels(ax, context):
        ax.set_xticks(context["x_pos"])
        ax.set_xticklabels(
            context["short_labels"],
            rotation=0,
            ha="center",
            fontsize=7,
        )

    def _plot_direct_finger_motion_multi_path_overlay(
        self,
        ax,
        analyses,
        context,
        view_elev=None,
        view_azim=None,
    ):
        target_frame_samples = context["target_frame_samples"]
        ax.plot(
            target_frame_samples[:, 0],
            target_frame_samples[:, 1],
            target_frame_samples[:, 2],
            color="#111111",
            linewidth=3,
            label="Target",
        )
        all_frame_points = [target_frame_samples]
        for item, label, color in zip(analyses, context["labels"], context["colors"]):
            robot_frame = item["robot_frame_samples"]
            all_frame_points.append(robot_frame)
            ax.plot(
                robot_frame[:, 0],
                robot_frame[:, 1],
                robot_frame[:, 2],
                color=color,
                linewidth=2,
                label=label[:20],
            )
            ax.scatter(*robot_frame[-1], color=color, s=28, marker=".")
        ax.set_title("Trajectory Following")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        self._set_3d_axes_equal(ax, np.vstack(all_frame_points))
        default_elev, default_azim = self._direct_finger_motion_preview_matplotlib_view()
        if view_elev is None:
            view_elev = default_elev
        if view_azim is None:
            view_azim = default_azim
        ax.view_init(elev=view_elev, azim=view_azim)
        ax.legend(loc="best", fontsize=8)

    def _plot_direct_finger_motion_multi_path_progress_error(self, ax, analyses, context):
        for item, label, color in zip(analyses, context["labels"], context["colors"]):
            ax.plot(
                item["progress"] * 100.0,
                item["progress_error"] * 1000.0,
                color=color,
                linewidth=1.8,
                label=label[:24],
            )
        ax.set_title("Trajectory Deviation")
        ax.set_xlabel("Trajectory Progress (%)")
        ax.set_ylabel("Error (mm)")
        self._style_direct_finger_motion_axes(ax, grid_axis="both")
        ax.legend(loc="best", fontsize=8)

    def _plot_direct_finger_motion_multi_path_rmse(self, ax, analyses, context):
        values = [item["metrics"]["rms_m"] * 1000.0 for item in analyses]
        bars = ax.bar(context["x_pos"], values, color=context["colors"], alpha=0.9)
        self._highlight_direct_finger_motion_best_bar(bars, values)
        self._annotate_direct_finger_motion_bars(
            ax,
            bars,
            [f"{value:.2f}" for value in values],
        )
        ax.set_title("RMSE")
        ax.set_ylabel("Error (mm)")
        self._set_direct_finger_motion_horizontal_xticklabels(ax, context)
        self._style_direct_finger_motion_axes(ax)

    def _plot_direct_finger_motion_multi_path_hausdorff(self, ax, analyses, context):
        values = [item["metrics"]["symmetric_hausdorff_m"] * 1000.0 for item in analyses]
        bars = ax.bar(context["x_pos"], values, color="#d81b60", alpha=0.9)
        self._highlight_direct_finger_motion_best_bar(bars, values)
        self._annotate_direct_finger_motion_bars(
            ax,
            bars,
            [f"{value:.2f}" for value in values],
        )
        ax.set_title("Hausdorff Distance")
        ax.set_ylabel("Distance (mm)")
        self._set_direct_finger_motion_horizontal_xticklabels(ax, context)
        self._style_direct_finger_motion_axes(ax)

    def _plot_direct_finger_motion_multi_path_normal_angle(self, ax, analyses, context):
        plotted = False
        for item, label, color in zip(analyses, context["labels"], context["colors"]):
            angle_error = np.asarray(
                item.get("normal_angle_error_deg", []),
                dtype=float,
            )
            if len(angle_error) == 0:
                continue
            ax.plot(
                item["progress"] * 100.0,
                angle_error,
                color=color,
                linewidth=1.8,
                label=label[:24],
            )
            rms_deg = self._finite_direct_finger_motion_metric(
                item["metrics"],
                "normal_angle_rms_deg",
            )
            if rms_deg is not None:
                ax.axhline(rms_deg, color=color, linewidth=0.8, linestyle=":", alpha=0.65)
            plotted = True
        ax.set_title("Normal Direction Error")
        ax.set_xlabel("Trajectory Progress (%)")
        ax.set_ylabel("Angle error (deg)")
        self._style_direct_finger_motion_axes(ax, grid_axis="both")
        if plotted:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "No quaternion columns",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

    def _plot_direct_finger_motion_multi_path_completion_time(self, ax, analyses, context):
        values = [
            self._finite_direct_finger_motion_metric(item["metrics"], "duration_s")
            for item in analyses
        ]
        plotted_values = [value if value is not None else 0.0 for value in values]
        bar_colors = [
            color if value is not None else "#c7c7c7"
            for color, value in zip(context["colors"], values)
        ]
        bars = ax.bar(context["x_pos"], plotted_values, color=bar_colors, alpha=0.9)
        self._highlight_direct_finger_motion_best_bar(bars, values)
        self._annotate_direct_finger_motion_bars(
            ax,
            bars,
            [f"{value:.2f}" if value is not None else "N/A" for value in values],
        )
        ax.set_title("Completion Time")
        ax.set_ylabel("Time (s)")
        self._set_direct_finger_motion_horizontal_xticklabels(ax, context)
        self._style_direct_finger_motion_axes(ax)
        if not any(value is not None for value in values):
            ax.text(
                0.5,
                0.5,
                "No timestamp column",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

    def _plot_direct_finger_motion_multi_path_length_ratio(self, ax, analyses, context):
        values = [item["metrics"]["length_ratio"] for item in analyses]
        bars = ax.bar(context["x_pos"], values, color=context["colors"], alpha=0.9)
        ratio_errors = [abs(float(value) - 1.0) for value in values]
        self._highlight_direct_finger_motion_best_bar(bars, ratio_errors)
        self._annotate_direct_finger_motion_bars(
            ax,
            bars,
            [f"{value:.4f}" for value in values],
        )
        ax.axhline(1.0, color="#777777", linewidth=0.9, linestyle="--")
        ax.set_title("Path Length Ratio")
        ax.set_ylabel("Robot / target")
        self._set_direct_finger_motion_horizontal_xticklabels(ax, context)
        self._style_direct_finger_motion_axes(ax)

    def _plot_direct_finger_motion_multi_path_graph(
        self,
        figure,
        graph_key,
        analyses,
        subplot_position=None,
        view_elev=None,
        view_azim=None,
    ):
        context = self._direct_finger_motion_multi_path_plot_context(analyses)
        position = subplot_position or (1, 1, 1)
        if graph_key == "overlay":
            ax = figure.add_subplot(*position, projection="3d")
            self._plot_direct_finger_motion_multi_path_overlay(
                ax,
                analyses,
                context,
                view_elev=view_elev,
                view_azim=view_azim,
            )
        else:
            ax = figure.add_subplot(*position)
            if graph_key == "progress_error":
                self._plot_direct_finger_motion_multi_path_progress_error(
                    ax,
                    analyses,
                    context,
                )
            elif graph_key == "rmse":
                self._plot_direct_finger_motion_multi_path_rmse(ax, analyses, context)
            elif graph_key == "hausdorff":
                self._plot_direct_finger_motion_multi_path_hausdorff(ax, analyses, context)
            elif graph_key == "normal_angle":
                self._plot_direct_finger_motion_multi_path_normal_angle(
                    ax,
                    analyses,
                    context,
                )
            elif graph_key == "completion_time":
                self._plot_direct_finger_motion_multi_path_completion_time(
                    ax,
                    analyses,
                    context,
                )
            elif graph_key == "length_ratio":
                self._plot_direct_finger_motion_multi_path_length_ratio(
                    ax,
                    analyses,
                    context,
                )
            else:
                raise ValueError(f"Unknown analysis graph: {graph_key}")
        return ax

    def _save_direct_finger_motion_multi_path_graph(
        self,
        analyses,
        graph_key,
        file_path,
        view_elev=None,
        view_azim=None,
    ):
        figure = Figure(
            figsize=self._direct_finger_motion_multi_path_export_figure_size(graph_key),
            constrained_layout=True,
        )
        self._plot_direct_finger_motion_multi_path_graph(
            figure,
            graph_key,
            analyses,
            view_elev=view_elev,
            view_azim=view_azim,
        )
        save_kwargs = {"dpi": 300}
        if graph_key != "overlay":
            save_kwargs["bbox_inches"] = "tight"
        figure.savefig(file_path, **save_kwargs)

    def _preferred_direct_finger_motion_export_dir(self):
        candidates = [
            self.DIRECT_FINGER_MOTION_TOOL_POSE_LOG_DIR,
            robot_resource_path("tool_pose_logs"),
            os.path.join(os.path.expanduser("~"), "phd_exports", "tool_pose_logs"),
            os.path.expanduser("~"),
        ]
        for path in candidates:
            try:
                os.makedirs(path, exist_ok=True)
                if os.access(path, os.W_OK):
                    return path
            except Exception:
                continue
        return os.path.expanduser("~")

    def _choose_direct_finger_motion_export_directory(self, title, default_dir):
        dialog = QFileDialog(self, title, default_dir)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setViewMode(QFileDialog.Detail)

        if dialog.exec_() != QFileDialog.Accepted:
            return ""

        selected_dirs = dialog.selectedFiles()
        if not selected_dirs:
            return ""
        return selected_dirs[0]

    def _export_direct_finger_motion_multi_path_graph(self, analyses, graph_key):
        specs = {
            key: (label, filename)
            for key, label, filename in self._direct_finger_motion_multi_path_graph_specs()
        }
        label, filename = specs.get(graph_key, ("Analysis Graph", "analysis_graph"))
        default_dir = self._preferred_direct_finger_motion_export_dir()
        default_path = os.path.join(
            default_dir,
            f"{time.strftime('%Y%m%d_%H%M%S')}_{filename}.png",
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {label}",
            default_path,
            "PNG Image (*.png);;SVG Vector (*.svg);;PDF Document (*.pdf);;All Files (*)",
        )
        if not file_path:
            return
        if not os.path.splitext(file_path)[1]:
            file_path = f"{file_path}.png"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            self._save_direct_finger_motion_multi_path_graph(
                analyses,
                graph_key,
                file_path,
            )
            self._append_direct_finger_motion_log(
                f"[DFM] Exported analysis graph: {file_path}"
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Export Analysis Graph",
                f"Failed to export graph:\n{exc}",
            )

    def _export_all_direct_finger_motion_multi_path_graphs(self, analyses):
        default_dir = self._preferred_direct_finger_motion_export_dir()
        parent_dir = self._choose_direct_finger_motion_export_directory(
            "Choose Folder for Analysis Graphs",
            default_dir,
        )
        if not parent_dir:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(parent_dir, f"path_analysis_graphs_{timestamp}")
        exported = []
        try:
            os.makedirs(output_dir, exist_ok=True)
            for graph_key, label, filename in self._direct_finger_motion_multi_path_graph_specs():
                safe_filename = self._safe_direct_finger_motion_export_name(filename)
                if graph_key == "overlay":
                    for view_name, view_label, elev, azim in (
                        self._direct_finger_motion_trajectory_following_view_presets()
                    ):
                        view_filename = safe_filename
                        if view_name != "default":
                            view_filename = f"{safe_filename}_{view_name}"
                        file_path = os.path.join(output_dir, f"{view_filename}.png")
                        self._save_direct_finger_motion_multi_path_graph(
                            analyses,
                            graph_key,
                            file_path,
                            view_elev=elev,
                            view_azim=azim,
                        )
                        exported.append((f"{label} ({view_label})", file_path))
                    continue

                file_path = os.path.join(output_dir, f"{safe_filename}.png")
                self._save_direct_finger_motion_multi_path_graph(
                    analyses,
                    graph_key,
                    file_path,
                )
                exported.append((label, file_path))
            self._append_direct_finger_motion_log(
                f"[DFM] Exported {len(exported)} analysis graphs to: {output_dir}"
            )
            QMessageBox.information(
                self,
                "Export Analysis Graphs",
                f"Exported {len(exported)} graphs to:\n{output_dir}",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Export Analysis Graphs",
                f"Failed to export all graphs:\n{exc}",
            )

    def _show_direct_finger_motion_multi_path_analysis(self, analyses):
        if FigureCanvas is None or Figure is None:
            QMessageBox.warning(
                self,
                "Path Analysis",
                "Matplotlib is not available, so the graph cannot be displayed.",
            )
            return

        summary_text = self._format_direct_finger_motion_multi_path_analysis_summary(analyses)
        dialog = QDialog(self)
        dialog.setWindowTitle("Robot Tool Path Method Comparison")
        dialog.resize(1400, 1040)
        layout = QVBoxLayout(dialog)

        summary = QPlainTextEdit(dialog)
        summary.setReadOnly(True)
        summary.setMaximumHeight(240)
        summary.setPlainText(summary_text)
        layout.addWidget(summary)

        export_row = QHBoxLayout()
        export_button = QPushButton("Export Comparison Metrics CSV", dialog)
        export_button.clicked.connect(
            lambda _checked=False, payload=analyses: (
                self._export_direct_finger_motion_path_analysis_metrics(payload)
            )
        )
        export_row.addWidget(export_button)
        graph_selector = QComboBox(dialog)
        for graph_key, label, _filename in self._direct_finger_motion_multi_path_graph_specs():
            graph_selector.addItem(label, graph_key)
        export_row.addWidget(QLabel("Graph:", dialog))
        export_row.addWidget(graph_selector)
        export_graph_button = QPushButton("Export Selected Graph", dialog)
        export_graph_button.clicked.connect(
            lambda _checked=False, payload=analyses, selector=graph_selector: (
                self._export_direct_finger_motion_multi_path_graph(
                    payload,
                    selector.currentData(),
                )
            )
        )
        export_row.addWidget(export_graph_button)
        export_all_graphs_button = QPushButton("Export All Graphs", dialog)
        export_all_graphs_button.clicked.connect(
            lambda _checked=False, payload=analyses: (
                self._export_all_direct_finger_motion_multi_path_graphs(payload)
            )
        )
        export_row.addWidget(export_all_graphs_button)
        export_row.addStretch()
        layout.addLayout(export_row)

        graph_specs = self._direct_finger_motion_multi_path_graph_specs()
        subplot_columns = 3
        subplot_rows = int(np.ceil(len(graph_specs) / subplot_columns))
        figure = Figure(
            figsize=(14.5, max(8.8, 4.2 * subplot_rows)),
            constrained_layout=True,
        )
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
        for subplot_index, (graph_key, _label, _filename) in enumerate(graph_specs, start=1):
            self._plot_direct_finger_motion_multi_path_graph(
                figure,
                graph_key,
                analyses,
                subplot_position=(subplot_rows, subplot_columns, subplot_index),
            )

        canvas.draw()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._direct_finger_motion_tool_pose_analysis_dialog = dialog
        best = min(analyses, key=lambda item: item["metrics"]["rms_m"])
        self._append_direct_finger_motion_log(
            "[DFM] Multi-path comparison generated. "
            f"Best RMSE: {best.get('label', 'Robot path')} "
            f"({best['metrics']['rms_m'] * 1000.0:.2f} mm)"
        )

    def _show_direct_finger_motion_path_analysis(self):
        analyses = self._compute_direct_finger_motion_path_analyses()
        if not analyses:
            QMessageBox.information(
                self,
                "Path Analysis",
                "Please load both a robot tool pose path and a target CSV first.",
            )
            return
        if len(analyses) > 1:
            self._show_direct_finger_motion_multi_path_analysis(analyses)
            return

        analysis = analyses[0]
        if FigureCanvas is None or Figure is None:
            QMessageBox.warning(
                self,
                "Path Analysis",
                "Matplotlib is not available, so the graph cannot be displayed.",
            )
            return

        metrics = analysis["metrics"]
        summary_text = self._format_direct_finger_motion_path_analysis_summary(metrics)

        dialog = QDialog(self)
        dialog.setWindowTitle("Tool Pose Path Analysis")
        dialog.resize(1320, 920)
        layout = QVBoxLayout(dialog)

        summary = QPlainTextEdit(dialog)
        summary.setReadOnly(True)
        summary.setMaximumHeight(260)
        summary.setPlainText(summary_text)
        layout.addWidget(summary)

        export_row = QHBoxLayout()
        export_button = QPushButton("Export Metrics CSV", dialog)
        export_button.clicked.connect(
            lambda _checked=False, payload=analysis: (
                self._export_direct_finger_motion_path_analysis_metrics(payload)
            )
        )
        export_row.addWidget(export_button)
        export_row.addStretch()
        layout.addLayout(export_row)

        figure = Figure(figsize=(13, 7.8), tight_layout=True)
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)

        progress_pct = analysis["progress"] * 100.0
        error_mm = analysis["progress_error"] * 1000.0
        nearest_mm = analysis["nearest_error"] * 1000.0
        diff_frame_mm = analysis["diff_frame"] * 1000.0
        cross_track_mm = analysis["cross_track_error"] * 1000.0
        robot_samples = analysis["robot_samples"]
        target_samples = analysis["target_samples"]
        robot_frame_samples = analysis["robot_frame_samples"]
        target_frame_samples = analysis["target_frame_samples"]

        ax_path = figure.add_subplot(2, 3, 1, projection="3d")
        ax_path.plot(
            robot_frame_samples[:, 0],
            robot_frame_samples[:, 1],
            robot_frame_samples[:, 2],
            color="#1e88e5",
            label="Robot",
        )
        ax_path.plot(
            target_frame_samples[:, 0],
            target_frame_samples[:, 1],
            target_frame_samples[:, 2],
            color="#fb8c00",
            label="Target",
        )
        ax_path.scatter(*robot_frame_samples[0], color="#43a047", s=35, label="Start")
        ax_path.scatter(*robot_frame_samples[-1], color="#e53935", s=35, label="End")
        ax_path.set_title("Target-Frame Path Overlay")
        ax_path.set_xlabel("Tangent T (m)")
        ax_path.set_ylabel("Normal N1 (m)")
        ax_path.set_zlabel("Normal N2 (m)")
        self._set_3d_axes_equal(
            ax_path,
            np.vstack((robot_frame_samples, target_frame_samples)),
        )
        ax_path.legend(loc="best")

        ax_error = figure.add_subplot(2, 3, 2)
        ax_error.plot(progress_pct, error_mm, color="#d81b60", label="Same progress")
        ax_error.plot(progress_pct, nearest_mm, color="#5e35b1", linestyle="--", label="Nearest segment")
        ax_error.set_title("Position Error")
        ax_error.set_xlabel("Trajectory Progress (%)")
        ax_error.set_ylabel("Error (mm)")
        ax_error.grid(True, alpha=0.3)
        ax_error.legend(loc="best")

        ax_frame = figure.add_subplot(2, 3, 3)
        ax_frame.plot(progress_pct, diff_frame_mm[:, 0], label="T error")
        ax_frame.plot(progress_pct, diff_frame_mm[:, 1], label="N1 error")
        ax_frame.plot(progress_pct, diff_frame_mm[:, 2], label="N2 error")
        ax_frame.axhline(0.0, color="#777777", linewidth=0.8)
        ax_frame.set_title("Target-Frame Error Components")
        ax_frame.set_xlabel("Trajectory Progress (%)")
        ax_frame.set_ylabel("Robot - target (mm)")
        ax_frame.grid(True, alpha=0.3)
        ax_frame.legend(loc="best")

        ax_cross = figure.add_subplot(2, 3, 4)
        ax_cross.plot(progress_pct, np.abs(diff_frame_mm[:, 0]), color="#00897b", label="|Along-track|")
        ax_cross.plot(progress_pct, cross_track_mm, color="#3949ab", label="Cross-track")
        ax_cross.set_title("Along-Track vs Cross-Track")
        ax_cross.set_xlabel("Trajectory Progress (%)")
        ax_cross.set_ylabel("Error (mm)")
        ax_cross.grid(True, alpha=0.3)
        ax_cross.legend(loc="best")

        ax_hist = figure.add_subplot(2, 3, 5)
        ax_hist.hist(error_mm, bins=30, color="#26a69a", alpha=0.85)
        ax_hist.axvline(metrics["rms_m"] * 1000.0, color="#d81b60", label="RMS")
        ax_hist.axvline(metrics["p95_m"] * 1000.0, color="#fb8c00", label="P95")
        ax_hist.set_title("Error Distribution")
        ax_hist.set_xlabel("Error (mm)")
        ax_hist.set_ylabel("Count")
        ax_hist.grid(True, alpha=0.3)
        ax_hist.legend(loc="best")

        ax_metrics = figure.add_subplot(2, 3, 6)
        ax_metrics.axis("off")
        metric_text = (
            f"Normalized RMSE: {metrics['normalized_rms_percent']:.3f}%\n"
            f"Cross-track P95: {metrics['cross_track_p95_m'] * 1000.0:.2f} mm\n"
            f"Nearest-segment P95: {metrics['nearest_p95_m'] * 1000.0:.2f} mm\n"
            f"Robust Hausdorff P95: {metrics['robust_hausdorff_p95_m'] * 1000.0:.2f} mm\n"
            f"Discrete Frechet: {metrics['discrete_frechet_m'] * 1000.0:.2f} mm\n"
            f"Completion time: {self._format_direct_finger_motion_duration(metrics)}\n"
            f"Average speed: {self._format_direct_finger_motion_speed(metrics)}\n"
            f"Length ratio: {metrics['length_ratio']:.4f}"
        )
        ax_metrics.text(
            0.02,
            0.98,
            metric_text,
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )
        ax_metrics.set_title("Comparison Table Values")

        canvas.draw()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._direct_finger_motion_tool_pose_analysis_dialog = dialog
        self._append_direct_finger_motion_log(
            "[DFM] Path analysis generated. "
            f"RMS={metrics['rms_m'] * 1000.0:.2f} mm, "
            f"max={metrics['max_m'] * 1000.0:.2f} mm, "
            f"P95={metrics['p95_m'] * 1000.0:.2f} mm, "
            f"cross-track P95={metrics['cross_track_p95_m'] * 1000.0:.2f} mm"
        )

    def _clear_direct_finger_motion_tool_pose_path_plot(
        self,
        log_message=True,
        clear_loaded_path=True,
        stop_animation=True,
        reset_animation_index=True,
        clear_target=True,
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
        if clear_loaded_path:
            self._clear_direct_finger_motion_comparison_robot_paths(
                keep_primary=False,
                log_message=False,
            )
        if clear_target:
            self._clear_direct_finger_motion_target_path(log_message=False)
        if reset_animation_index:
            self._direct_finger_motion_tool_pose_path_animation_index = 0
        if clear_loaded_path:
            self._direct_finger_motion_tool_pose_plot_points = None
            self._direct_finger_motion_tool_pose_raw_points = None
            self._direct_finger_motion_tool_pose_compact_origin = None
            self._direct_finger_motion_tool_pose_compact_scale = None
            self._direct_finger_motion_tool_pose_target_display_origin = None
            self._direct_finger_motion_tool_pose_target_display_scale = None
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
                if getattr(self, "direct_finger_motion_target_manual_rx_spin", None) is None:
                    raise RuntimeError("Tool pose viewer is missing manual rotation controls.")
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
        self.direct_finger_motion_comparison_robot_path_add_button = QPushButton(
            "Add Robot CSV", dialog
        )
        self.direct_finger_motion_comparison_robot_path_add_button.clicked.connect(
            self._load_direct_finger_motion_comparison_robot_paths_from_dialog
        )
        control_row.addWidget(self.direct_finger_motion_comparison_robot_path_add_button)
        self.direct_finger_motion_comparison_robot_path_clear_button = QPushButton(
            "Clear Robot CSVs", dialog
        )
        self.direct_finger_motion_comparison_robot_path_clear_button.clicked.connect(
            lambda _checked=False: self._clear_direct_finger_motion_comparison_robot_paths(
                keep_primary=True
            )
        )
        control_row.addWidget(self.direct_finger_motion_comparison_robot_path_clear_button)
        self.direct_finger_motion_target_path_load_button = QPushButton("Load Target CSV", dialog)
        self.direct_finger_motion_target_path_load_button.clicked.connect(
            self._load_direct_finger_motion_target_path_from_dialog
        )
        control_row.addWidget(self.direct_finger_motion_target_path_load_button)
        self.direct_finger_motion_target_path_clear_button = QPushButton("Clear Target", dialog)
        self.direct_finger_motion_target_path_clear_button.clicked.connect(
            lambda _checked=False: self._clear_direct_finger_motion_target_path()
        )
        control_row.addWidget(self.direct_finger_motion_target_path_clear_button)
        self.direct_finger_motion_path_analysis_button = QPushButton("Analyze Path", dialog)
        self.direct_finger_motion_path_analysis_button.clicked.connect(
            self._show_direct_finger_motion_path_analysis
        )
        control_row.addWidget(self.direct_finger_motion_path_analysis_button)
        self.direct_finger_motion_target_align_start_checkbox = QCheckBox(
            "Align target start to robot start", dialog
        )
        self.direct_finger_motion_target_align_start_checkbox.setChecked(
            bool(getattr(self, "_direct_finger_motion_tool_pose_target_align_start", True))
        )
        self.direct_finger_motion_target_align_start_checkbox.toggled.connect(
            self._on_direct_finger_motion_tool_pose_target_align_start_toggled
        )
        control_row.addWidget(self.direct_finger_motion_target_align_start_checkbox)
        layout.addLayout(control_row)

        align_row = QHBoxLayout()
        self.direct_finger_motion_target_align_rotation_checkbox = QCheckBox(
            "Align target rotation", dialog
        )
        self.direct_finger_motion_target_align_rotation_checkbox.setChecked(
            bool(getattr(self, "_direct_finger_motion_tool_pose_target_align_rotation", False))
        )
        self.direct_finger_motion_target_align_rotation_checkbox.toggled.connect(
            self._on_direct_finger_motion_tool_pose_target_align_rotation_toggled
        )
        align_row.addWidget(self.direct_finger_motion_target_align_rotation_checkbox)
        align_row.addWidget(QLabel("Rotation mode:", dialog))
        self.direct_finger_motion_target_rotation_mode_combo = QComboBox(dialog)
        self.direct_finger_motion_target_rotation_mode_combo.addItem(
            "First segment direction", "first_segment"
        )
        self.direct_finger_motion_target_rotation_mode_combo.addItem(
            "Path best fit", "best_fit"
        )
        current_mode = str(
            getattr(
                self,
                "_direct_finger_motion_tool_pose_target_rotation_mode",
                "first_segment",
            )
        )
        mode_index = self.direct_finger_motion_target_rotation_mode_combo.findData(
            current_mode
        )
        if mode_index < 0:
            mode_index = 0
        self.direct_finger_motion_target_rotation_mode_combo.setCurrentIndex(mode_index)
        self.direct_finger_motion_target_rotation_mode_combo.currentIndexChanged.connect(
            self._on_direct_finger_motion_tool_pose_target_rotation_mode_changed
        )
        align_row.addWidget(self.direct_finger_motion_target_rotation_mode_combo)
        align_row.addStretch()
        layout.addLayout(align_row)

        manual_rotation_row = QHBoxLayout()
        manual_rotation_row.addWidget(QLabel("Manual target rotation:", dialog))
        self.direct_finger_motion_target_manual_rx_spin = (
            self._create_direct_finger_motion_target_manual_rotation_spinbox(
                dialog,
                getattr(self, "_direct_finger_motion_tool_pose_target_manual_rx_deg", 0.0),
            )
        )
        manual_rotation_row.addWidget(QLabel("Rx", dialog))
        manual_rotation_row.addWidget(self.direct_finger_motion_target_manual_rx_spin)
        self.direct_finger_motion_target_manual_ry_spin = (
            self._create_direct_finger_motion_target_manual_rotation_spinbox(
                dialog,
                getattr(self, "_direct_finger_motion_tool_pose_target_manual_ry_deg", 0.0),
            )
        )
        manual_rotation_row.addWidget(QLabel("Ry", dialog))
        manual_rotation_row.addWidget(self.direct_finger_motion_target_manual_ry_spin)
        self.direct_finger_motion_target_manual_rz_spin = (
            self._create_direct_finger_motion_target_manual_rotation_spinbox(
                dialog,
                getattr(self, "_direct_finger_motion_tool_pose_target_manual_rz_deg", 0.0),
            )
        )
        manual_rotation_row.addWidget(QLabel("Rz", dialog))
        manual_rotation_row.addWidget(self.direct_finger_motion_target_manual_rz_spin)
        self.direct_finger_motion_target_manual_rotation_apply_button = QPushButton(
            "Apply Rotation", dialog
        )
        self.direct_finger_motion_target_manual_rotation_apply_button.clicked.connect(
            self._apply_direct_finger_motion_target_manual_rotation_from_ui
        )
        manual_rotation_row.addWidget(
            self.direct_finger_motion_target_manual_rotation_apply_button
        )
        self.direct_finger_motion_target_manual_rotation_reset_button = QPushButton(
            "Reset Rx/Ry/Rz", dialog
        )
        self.direct_finger_motion_target_manual_rotation_reset_button.clicked.connect(
            lambda _checked=False: self._reset_direct_finger_motion_target_manual_rotation(
                refresh_display=True
            )
        )
        manual_rotation_row.addWidget(
            self.direct_finger_motion_target_manual_rotation_reset_button
        )
        manual_rotation_row.addStretch()
        layout.addLayout(manual_rotation_row)
        self._set_direct_finger_motion_target_manual_rotation_controls_enabled(
            getattr(self, "_direct_finger_motion_tool_pose_target_raw_points", None)
            is not None
        )

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
            clear_target=False,
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

    def _clear_direct_finger_motion_comparison_path_actors(self):
        plotter = getattr(self, "_direct_finger_motion_tool_pose_plotter", None)
        actors = list(getattr(self, "_direct_finger_motion_tool_pose_comparison_actors", []))
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
        self._direct_finger_motion_tool_pose_comparison_actors = []

    def _render_direct_finger_motion_comparison_robot_paths(self):
        plotter = getattr(self, "_direct_finger_motion_tool_pose_plotter", None)
        if plotter is None:
            return

        self._clear_direct_finger_motion_comparison_path_actors()
        actors = []
        paths = list(getattr(self, "_direct_finger_motion_tool_pose_comparison_paths", []))
        for index, path in enumerate(paths):
            if bool(path.get("is_primary", False)):
                continue
            points_np = np.asarray(path.get("display_points", []), dtype=float)
            if len(points_np) == 0:
                continue
            color = str(path.get("color") or self._direct_finger_motion_robot_path_color(index))
            if len(points_np) >= 2:
                try:
                    actors.append(
                        plotter.add_mesh(
                            pv.lines_from_points(points_np),
                            color=color,
                            line_width=3,
                            name=f"dfm_comparison_robot_path_{index}",
                        )
                    )
                except Exception:
                    pass
            try:
                actors.append(
                    plotter.add_mesh(
                        pv.PolyData(points_np[:1]),
                        color=color,
                        point_size=10,
                        render_points_as_spheres=True,
                        name=f"dfm_comparison_robot_path_start_{index}",
                    )
                )
            except Exception:
                pass
        self._direct_finger_motion_tool_pose_comparison_actors = actors
        try:
            plotter.render()
        except Exception:
            pass

    def _render_direct_finger_motion_target_path(self, target_points_np):
        plotter = getattr(self, "_direct_finger_motion_tool_pose_plotter", None)
        if plotter is None or target_points_np is None or len(target_points_np) == 0:
            return

        self._clear_direct_finger_motion_target_path_actors()

        actors = []
        if len(target_points_np) >= 2:
            target_mesh = pv.lines_from_points(target_points_np)
            actors.append(
                plotter.add_mesh(
                    target_mesh,
                    color="#ff9800",
                    line_width=4,
                    name="dfm_target_pose_path_line",
                )
            )
        actors.append(
            plotter.add_mesh(
                pv.PolyData(target_points_np[:1]),
                color="#fff176",
                point_size=14,
                render_points_as_spheres=True,
                name="dfm_target_pose_path_start",
            )
        )
        actors.append(
            plotter.add_mesh(
                pv.PolyData(target_points_np[-1:]),
                color="#f57c00",
                point_size=14,
                render_points_as_spheres=True,
                name="dfm_target_pose_path_end",
            )
        )
        self._direct_finger_motion_tool_pose_target_actors = actors

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
        self._direct_finger_motion_tool_pose_comparison_actors = []
        self._direct_finger_motion_tool_pose_comparison_paths = []
        self._direct_finger_motion_tool_pose_target_actors = []
        self._direct_finger_motion_tool_pose_plot_points = None
        self._direct_finger_motion_tool_pose_raw_points = None
        self._direct_finger_motion_tool_pose_quaternions = None
        self._direct_finger_motion_tool_pose_target_points = None
        self._direct_finger_motion_tool_pose_target_raw_points = None
        self._direct_finger_motion_tool_pose_target_normals = None
        self._direct_finger_motion_tool_pose_compact_origin = None
        self._direct_finger_motion_tool_pose_compact_scale = None
        self._direct_finger_motion_tool_pose_target_display_origin = None
        self._direct_finger_motion_tool_pose_target_display_scale = None
        self._direct_finger_motion_tool_pose_animation_points = None
        self._direct_finger_motion_tool_pose_path_animation_index = 0
        self._direct_finger_motion_tool_pose_plotter = None
        self._direct_finger_motion_tool_pose_plot_dialog = None
        self.direct_finger_motion_tool_pose_play_button = None
        self.direct_finger_motion_comparison_robot_path_add_button = None
        self.direct_finger_motion_comparison_robot_path_clear_button = None
        self.direct_finger_motion_target_manual_rx_spin = None
        self.direct_finger_motion_target_manual_ry_spin = None
        self.direct_finger_motion_target_manual_rz_spin = None
        self.direct_finger_motion_target_manual_rotation_apply_button = None
        self.direct_finger_motion_target_manual_rotation_reset_button = None

    def _collect_direct_finger_motion_settings_from_ui(self):
        settings = {}
        for name, widget in self.direct_finger_motion_inputs.items():
            if isinstance(widget, QCheckBox):
                settings[name] = widget.isChecked()
            else:
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
                    if isinstance(widget, QCheckBox):
                        widget.setChecked(bool(settings[name]))
                    else:
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
