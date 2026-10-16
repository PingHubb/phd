"""Robot-dog controls driven by simple tactile swipe gestures."""

from __future__ import annotations

import os
import time

import numpy as np
from PyQt5.QtWidgets import QLabel, QPushButton

from phd.dependence.sensor_protocol import write_serial_bytes
from phd.ui import components


ROBOT_DOG_SWIPE_COMMANDS = {
    "up": "kup",
    "down": "ksit",
    "left": "kstr",
    "right": "khi",
}
ROBOT_DOG_REST_COMMAND = "d"


def _connected_components(mask):
    rows, columns = mask.shape
    remaining = {
        (int(row), int(column))
        for row, column in np.argwhere(mask)
    }
    components_found = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component = [seed]
        while stack:
            row, column = stack.pop()
            for delta_row in (-1, 0, 1):
                for delta_column in (-1, 0, 1):
                    if delta_row == 0 and delta_column == 0:
                        continue
                    neighbor = (
                        row + delta_row,
                        column + delta_column,
                    )
                    if (
                        0 <= neighbor[0] < rows
                        and 0 <= neighbor[1] < columns
                        and neighbor in remaining
                    ):
                        remaining.remove(neighbor)
                        stack.append(neighbor)
                        component.append(neighbor)
        components_found.append(component)
    return components_found


def _single_touch_center(values, touch_threshold):
    touched = values < touch_threshold
    components_found = _connected_components(touched)
    if not components_found:
        return None, 0
    if len(components_found) != 1:
        return None, len(components_found)

    component = np.asarray(components_found[0], dtype=int)
    rows = component[:, 0]
    columns = component[:, 1]
    weights = np.maximum(
        touch_threshold - values[rows, columns],
        0.001,
    )
    return (
        float(np.average(rows, weights=weights)),
        float(np.average(columns, weights=weights)),
    ), 1


def _normalized_offset(shape, start, center):
    row_span = max(1.0, float(shape[0] - 1))
    column_span = max(1.0, float(shape[1] - 1))
    return (
        (center[0] - start[0]) / row_span,
        (center[1] - start[1]) / column_span,
    )


class SingleFingerSwipeDetector:
    """Recognize one four-way swipe from a single connected touch cluster."""

    def __init__(
        self,
        *,
        touch_threshold=-3.0,
        minimum_distance=0.25,
        dominance_ratio=1.25,
        minimum_frames=3,
        maximum_duration_sec=1.5,
        release_frames=2,
    ):
        self.touch_threshold = float(touch_threshold)
        self.minimum_distance = float(minimum_distance)
        self.dominance_ratio = float(dominance_ratio)
        self.minimum_frames = max(2, int(minimum_frames))
        self.maximum_duration_sec = max(0.1, float(maximum_duration_sec))
        self.release_frames = max(1, int(release_frames))
        self.reset()

    def reset(self):
        self._start_center = None
        self._started_at = None
        self._sample_count = 0
        self._no_touch_count = 0
        self._emitted = False
        self._blocked_until_release = False

    def _release(self):
        self._start_center = None
        self._started_at = None
        self._sample_count = 0
        self._emitted = False
        self._blocked_until_release = False

    def process(self, values, timestamp=None):
        """Return ``up``, ``down``, ``left``, or ``right`` once per swipe."""
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or min(matrix.shape) < 2:
            self.reset()
            return None
        now = time.monotonic() if timestamp is None else float(timestamp)
        center, cluster_count = _single_touch_center(
            matrix,
            self.touch_threshold,
        )

        if cluster_count == 0:
            self._no_touch_count += 1
            if self._no_touch_count >= self.release_frames:
                self._release()
            return None

        self._no_touch_count = 0
        if cluster_count != 1:
            self._blocked_until_release = True
            self._start_center = None
            return None
        if self._blocked_until_release:
            return None

        if self._start_center is None:
            self._start_center = center
            self._started_at = now
            self._sample_count = 1
            return None

        self._sample_count += 1
        if self._emitted:
            return None
        if now - self._started_at > self.maximum_duration_sec:
            self._start_center = center
            self._started_at = now
            self._sample_count = 1
            return None
        if self._sample_count < self.minimum_frames:
            return None

        delta_row, delta_column = _normalized_offset(
            matrix.shape,
            self._start_center,
            center,
        )
        vertical = abs(delta_row)
        horizontal = abs(delta_column)

        direction = None
        if (
            horizontal >= self.minimum_distance
            and horizontal >= vertical * self.dominance_ratio
        ):
            direction = "right" if delta_column > 0 else "left"
        elif (
            vertical >= self.minimum_distance
            and vertical >= horizontal * self.dominance_ratio
        ):
            direction = "down" if delta_row > 0 else "up"

        if direction is not None:
            self._emitted = True
        return direction


class SingleFingerDoubleTapDetector:
    """Recognize two short stationary taps inside one time window."""

    def __init__(
        self,
        *,
        touch_threshold=-3.0,
        maximum_travel=0.22,
        minimum_contact_frames=2,
        maximum_contact_sec=0.40,
        double_tap_window_sec=0.50,
        release_frames=2,
    ):
        self.touch_threshold = float(touch_threshold)
        self.maximum_travel = float(maximum_travel)
        self.minimum_contact_frames = max(1, int(minimum_contact_frames))
        self.maximum_contact_sec = max(0.05, float(maximum_contact_sec))
        self.double_tap_window_sec = max(0.05, float(double_tap_window_sec))
        self.release_frames = max(1, int(release_frames))
        self.reset()

    def reset(self):
        self._start_center = None
        self._started_at = None
        self._last_touch_at = None
        self._sample_count = 0
        self._max_travel = 0.0
        self._qualified = True
        self._no_touch_count = 0
        self._pending_tap_at = None

    def _clear_contact(self):
        self._start_center = None
        self._started_at = None
        self._last_touch_at = None
        self._sample_count = 0
        self._max_travel = 0.0
        self._qualified = True

    def _finish_contact(self):
        started_at = self._started_at
        last_touch_at = self._last_touch_at
        sample_count = self._sample_count
        qualified = bool(self._qualified and self._start_center is not None)
        pending_tap_at = self._pending_tap_at
        self._clear_contact()

        is_tap = (
            qualified
            and started_at is not None
            and last_touch_at is not None
            and sample_count >= self.minimum_contact_frames
            and (last_touch_at - started_at) <= self.maximum_contact_sec
        )
        if not is_tap:
            if sample_count > 0:
                self._pending_tap_at = None
            return None
        if (
            pending_tap_at is not None
            and (started_at - pending_tap_at) <= self.double_tap_window_sec
        ):
            self._pending_tap_at = None
            return "double_tap"
        self._pending_tap_at = last_touch_at
        return None

    def process(self, values, timestamp=None):
        """Return ``double_tap`` when the second tap lifts inside the window."""
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or min(matrix.shape) < 2:
            self.reset()
            return None
        now = time.monotonic() if timestamp is None else float(timestamp)
        center, cluster_count = _single_touch_center(
            matrix,
            self.touch_threshold,
        )

        if cluster_count == 0:
            self._no_touch_count += 1
            if self._start_center is None:
                return None
            if self._no_touch_count >= self.release_frames:
                return self._finish_contact()
            return None

        self._no_touch_count = 0
        if cluster_count != 1 or center is None:
            self._qualified = False
            self._pending_tap_at = None
            return None

        if self._start_center is None:
            self._start_center = center
            self._started_at = now
            self._last_touch_at = now
            self._sample_count = 1
            self._max_travel = 0.0
            self._qualified = True
            return None

        self._sample_count += 1
        self._last_touch_at = now
        delta_row, delta_column = _normalized_offset(
            matrix.shape,
            self._start_center,
            center,
        )
        travel = (delta_row * delta_row + delta_column * delta_column) ** 0.5
        self._max_travel = max(self._max_travel, travel)
        if self._max_travel > self.maximum_travel:
            self._qualified = False
        if now - self._started_at > self.maximum_contact_sec:
            self._qualified = False
        return None


class RobotDogControlsMixin:
    """Build and operate the log-only Robot Dog gesture subtab."""

    def setup_robot_dog_tab(self, layout):
        self._robot_dog_gesture_enabled = False
        self._robot_dog_sensor_signal = None
        self._robot_dog_swipe_detector = SingleFingerSwipeDetector()
        self._robot_dog_double_tap_detector = SingleFingerDoubleTapDetector()

        control_group, control_list = components.section(
            "Tactile gesture control",
            "Recognize one-finger swipes and a double tap from the connected tactile sensor.",
        )
        self.robot_dog_gesture_button = QPushButton("Enable Gesture Control")
        self.robot_dog_gesture_button.setCheckable(True)
        components.apply_variant(self.robot_dog_gesture_button, "primary")
        self.robot_dog_gesture_button.clicked.connect(
            self._toggle_robot_dog_gesture_control
        )
        control_list.add_row(
            "Gesture input",
            self.robot_dog_gesture_button,
            hint="Up: stand · Down: sit · Left: stretch · Right: greeting · Double tap: rest.",
        )

        self.robot_dog_gesture_status = QLabel(
            "Off · connect an 8×8 tactile sensor before enabling"
        )
        self.robot_dog_gesture_status.setWordWrap(True)
        control_list.add_row("Status", self.robot_dog_gesture_status)
        layout.addWidget(control_group)

        event_group, event_list = components.section(
            "Detected gesture",
            "Four swipe directions, plus a double tap for the rest pose.",
        )
        self.robot_dog_last_gesture = QLabel("No gesture detected")
        self.robot_dog_last_gesture.setWordWrap(True)
        event_list.add_row("Latest", self.robot_dog_last_gesture)
        layout.addWidget(event_group)
        layout.addStretch(1)

    def _toggle_robot_dog_gesture_control(self, checked):
        if checked:
            sensor = getattr(self, "sensor_functions", None)
            if sensor is None or not bool(getattr(sensor, "is_connected", False)):
                self.robot_dog_gesture_button.setChecked(False)
                self.robot_dog_gesture_status.setText(
                    "Off · connect the tactile sensor in Sensor → Setup first"
                )
                self.log_display.append(
                    "[Robot Dog] Gesture detection was not enabled: "
                    "connect a tactile sensor first."
                )
                return
            bridge = getattr(sensor, "_payload_bridge", None)
            signal = getattr(bridge, "frame_processed", None)
            if signal is None:
                self.robot_dog_gesture_button.setChecked(False)
                self.robot_dog_gesture_status.setText(
                    "Off · live sensor-frame signal is unavailable"
                )
                self.log_display.append(
                    "[Robot Dog] Gesture detection unavailable: "
                    "no live sensor-frame signal."
                )
                return
            try:
                signal.connect(self._on_robot_dog_sensor_frame)
            except Exception as exc:
                self.robot_dog_gesture_button.setChecked(False)
                self.robot_dog_gesture_status.setText("Off · connection failed")
                self.log_display.append(
                    f"[Robot Dog] Could not enable gesture detection: {exc}"
                )
                return

            self._robot_dog_sensor_signal = signal
            self._robot_dog_swipe_detector.reset()
            self._robot_dog_double_tap_detector.reset()
            self._robot_dog_gesture_enabled = True
            self.robot_dog_gesture_button.setText("Disable Gesture Control")
            self.robot_dog_gesture_status.setText(
                "On · swipe and double-tap commands enabled"
            )
            self.log_display.append(
                "[Robot Dog] Gesture control enabled: "
                "Up→kup, Down→ksit, Left→kstr, Right→khi, "
                "double tap→d."
            )
            return

        self._stop_robot_dog_gesture_control()

    def _stop_robot_dog_gesture_control(self):
        signal = getattr(self, "_robot_dog_sensor_signal", None)
        if signal is not None:
            try:
                signal.disconnect(self._on_robot_dog_sensor_frame)
            except (RuntimeError, TypeError):
                pass
        was_enabled = bool(getattr(self, "_robot_dog_gesture_enabled", False))
        self._robot_dog_sensor_signal = None
        self._robot_dog_gesture_enabled = False
        for detector_name in (
            "_robot_dog_swipe_detector",
            "_robot_dog_double_tap_detector",
        ):
            detector = getattr(self, detector_name, None)
            if detector is not None:
                detector.reset()
        button = getattr(self, "robot_dog_gesture_button", None)
        if button is not None:
            button.blockSignals(True)
            button.setChecked(False)
            button.setText("Enable Gesture Control")
            button.blockSignals(False)
        status = getattr(self, "robot_dog_gesture_status", None)
        if status is not None:
            status.setText("Off")
        if was_enabled and hasattr(self, "log_display"):
            self.log_display.append("[Robot Dog] Gesture control disabled.")

    def _robot_dog_serial_connection(self):
        sensor = getattr(self, "sensor_functions", None)
        serial_ports = list(getattr(sensor, "ser_list", None) or [])
        if not serial_ports:
            return None
        primary = os.path.basename(
            str(getattr(sensor, "_primary_sensor_port", "") or "")
        ).lower()
        for serial_port in serial_ports:
            name = os.path.basename(
                str(getattr(serial_port, "port", "") or "")
            ).lower()
            if primary and name == primary:
                return serial_port
        return serial_ports[0]

    def _send_robot_dog_command(self, command):
        serial_port = self._robot_dog_serial_connection()
        if serial_port is None or not bool(
            getattr(serial_port, "is_open", False)
        ):
            return False, "sensor/robot serial connection is not open"
        payload = (str(command) + "\n").encode("ascii")
        try:
            written = int(write_serial_bytes(serial_port, payload))
        except Exception as exc:
            return False, str(exc)
        if written != len(payload):
            return False, f"wrote {written}/{len(payload)} bytes"
        return True, ""

    def _on_robot_dog_sensor_frame(
        self,
        timestamp,
        _frame_sequence,
        _raw,
        _calibration,
    ):
        if not self._robot_dog_gesture_enabled:
            return
        sensor = getattr(self, "sensor_functions", None)
        data = getattr(sensor, "_data", None)
        matrix = getattr(data, "diffPerDataAve", None)
        if matrix is None:
            return
        direction = self._robot_dog_swipe_detector.process(
            matrix,
            timestamp=timestamp,
        )
        double_tap = self._robot_dog_double_tap_detector.process(
            matrix,
            timestamp=timestamp,
        )
        if direction is not None:
            self._report_robot_dog_gesture(
                f"Swipe {direction.upper()}",
                ROBOT_DOG_SWIPE_COMMANDS[direction],
            )
            return
        if double_tap == "double_tap":
            self._report_robot_dog_gesture(
                "Double tap",
                ROBOT_DOG_REST_COMMAND,
            )

    def _report_robot_dog_gesture(self, description, command):
        sent, error = self._send_robot_dog_command(command)
        if sent:
            self.robot_dog_last_gesture.setText(
                f"{description} → {command}"
            )
            self.log_display.append(
                f"[Robot Dog] {description} detected → sent {command}."
            )
            return

        self.robot_dog_last_gesture.setText(
            f"{description} → send failed"
        )
        self.log_display.append(
            f"[Robot Dog] {description} detected, but {command} was not sent: "
            f"{error}."
        )
