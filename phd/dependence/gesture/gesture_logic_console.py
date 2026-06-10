"""Console / PS5-style TCP velocity control (sensor or joystick)."""

import json
import os
import re
import struct
import time

import numpy as np

from phd.dependence.sensor_layout import (
    column_major_coords,
    flatten_column_major_view,
)
from phd.dependence.gesture.gesture_logic_direct_finger_motion import DirectFingerMotion

__all__ = ["ConsoleControl"]


class ConsoleControl(DirectFingerMotion):
    """PS5/DualSense-style joystick control for TCP velocity."""

    SETTINGS_FILE = os.path.join(DirectFingerMotion.CONFIG_DIR, "console_control.json")

    def _default_settings(self):
        settings = super()._default_settings()
        settings.update(
            {
                "config_version": 4,
                "motion_threshold": -3.0,
                "frame_interval_ms": 20,
                "velocity_smoothing_alpha": 1.0,
                "console_device_index": 0,
                "console_deadband": 0.08,
                "console_linear_speed": 0.05,
                "console_angular_speed": 0.20,
                "console_x_speed": 0.05,
                "console_y_speed": 0.05,
                "console_z_speed": 0.05,
                "console_rx_speed": 0.20,
                "console_ry_speed": 0.20,
                "console_rz_speed": 0.20,
                "console_axis_left_x": 0,
                "console_axis_left_y": 1,
                "console_axis_right_x": 3,
                "console_axis_right_y": 4,
                "console_axis_l2": 2,
                "console_axis_r2": 5,
                "console_button_l1": 4,
                "console_button_r1": 5,
                "console_x_sign": -1.0,
                "console_y_sign": 1.0,
                "console_z_sign": -1.0,
                "console_rx_sign": 1.0,
                "console_ry_sign": -1.0,
                "console_rz_sign": -1.0,
                "console_sensor_touch_grid_flip_lr": 0,
                "console_sensor_touch_grid_flip_ud": 0,
                "console_sensor_v2_slow_scale": 0.35,
                "console_sensor_v2_fast_scale": 1.8,
            }
        )
        return settings

    def load_settings_from_file(self):
        if not os.path.exists(self.settings_path):
            return
        try:
            with open(self.settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            config_version = int(settings.get("config_version", 0))
            migrated = config_version < 4
            if config_version < 2:
                settings.update(
                    {
                        "config_version": 2,
                        "console_axis_right_x": 3,
                        "console_axis_right_y": 4,
                        "console_axis_l2": 2,
                        "console_axis_r2": 5,
                    }
                )
                print("[Console Control] Migrated joystick mapping to Linux/Xbox-style axes.")
            if config_version < 3:
                settings.update(
                    {
                        "config_version": 3,
                        "console_x_sign": -1.0,
                        "console_y_sign": 1.0,
                        "console_z_sign": -1.0,
                        "console_rx_sign": 1.0,
                        "console_ry_sign": -1.0,
                        "console_rz_sign": -1.0,
                    }
                )
                print("[Console Control] Migrated direction signs to reversed defaults.")
            if config_version < 4:
                linear_speed = float(settings.get("console_linear_speed", 0.05))
                angular_speed = float(settings.get("console_angular_speed", 0.20))
                settings.update(
                    {
                        "config_version": 4,
                        "console_x_speed": linear_speed,
                        "console_y_speed": linear_speed,
                        "console_z_speed": linear_speed,
                        "console_rx_speed": angular_speed,
                        "console_ry_speed": angular_speed,
                        "console_rz_speed": angular_speed,
                    }
                )
                print("[Console Control] Migrated shared speeds to per-axis speeds.")
            self.apply_settings(settings, save_to_file=False)
            if migrated:
                self.save_settings_to_file()
            print(f"Console control settings loaded: {self.settings_path}")
        except Exception as exc:
            print(f"Failed to load Console control settings: {exc}")

    def apply_settings(self, settings: dict, save_to_file=False):
        super().apply_settings(settings, save_to_file=False)
        int_fields = (
            "console_device_index",
            "console_axis_left_x",
            "console_axis_left_y",
            "console_axis_right_x",
            "console_axis_right_y",
            "console_axis_l2",
            "console_axis_r2",
            "console_button_l1",
            "console_button_r1",
            "console_sensor_touch_grid_flip_lr",
            "console_sensor_touch_grid_flip_ud",
        )
        for field in int_fields:
            setattr(self, field, int(getattr(self, field)))
        if save_to_file:
            self.save_settings_to_file()

    def __init__(self, ros_splitter_instance, my_sensor_instance):
        super().__init__(ros_splitter_instance, my_sensor_instance)
        self.current_motion_mode = "console_control_idle"
        self.console_input_source = "ps5"
        self._console_fd = None
        self._console_device_path = None
        self._console_axes = {}
        self._console_buttons = {}
        self._console_last_logged_direction = None
        self._console_last_log_time = 0.0
        self._console_last_missing_log_time = 0.0
        self._console_sensor_left_stick_anchor = None
        self._console_sensor_right_stick_anchor = None
        self._console_sensor_v2_clutch_logged = False

    def toggle_console_control(self):
        self._toggle_console_control_mode("ps5")

    def toggle_console_control_sensor(self):
        self._toggle_console_control_mode("sensor")

    def toggle_console_control_sensor_v2(self):
        self._toggle_console_control_mode("sensor_v2")

    def _toggle_console_control_mode(self, source):
        source = str(source).strip().lower()
        if source not in {"ps5", "sensor", "sensor_v2"}:
            source = "ps5"

        if self.is_running and self.console_input_source == source:
            self._stop_console_control()
            return

        if self.is_running and self.console_input_source != source:
            self.console_input_source = source
            self._close_console_device()
            self._reset_state()
            self._append_console_log(
                f"[Console Control] switched input source -> {self.console_input_source.upper()} | frame={self._get_requested_frame()}"
            )
            return

        self.console_input_source = source
        self.is_running = True
        self._reset_state()
        self.control_timer.start(int(self.frame_interval_ms))
        if self.console_input_source == "ps5":
            self._append_console_log(
                f"[Console Control] STARTED (PS5) | frame={self._get_requested_frame()} | device=/dev/input/js{int(self.console_device_index)}"
            )
        elif self.console_input_source == "sensor":
            self._append_console_log(
                f"[Console Control] STARTED (SENSOR) | frame={self._get_requested_frame()}"
            )
        else:
            self._append_console_log(
                "[Console Control] STARTED (SENSOR V2 manual XYZ) | "
                f"frame={self._get_requested_frame()}"
            )

    def _stop_console_control(self):
        self.control_timer.stop()
        self._stop_robot_motion(stop_mode=True)
        self._close_console_device()
        self.is_running = False
        self._reset_state()
        self._append_console_log("[Console Control] STOPPED")

    def _reset_state(self):
        super()._reset_state()
        self.current_motion_mode = "console_control_idle"
        self._console_axes = {}
        self._console_buttons = {}
        self._console_last_logged_direction = None
        self._console_last_log_time = 0.0
        self._console_sensor_left_stick_anchor = None
        self._console_sensor_right_stick_anchor = None
        self._console_sensor_v2_clutch_logged = False

    def _append_console_log(self, message):
        log_display = getattr(self.ros_splitter, "log_display", None)
        if log_display is not None:
            try:
                log_display.append(message)
                return
            except Exception:
                pass
        print(message)

    def _find_console_device_path(self):
        device_index = int(getattr(self, "console_device_index", 0))
        preferred = f"/dev/input/js{device_index}"
        if os.path.exists(preferred):
            return preferred

        input_dir = "/dev/input"
        try:
            candidates = sorted(
                name for name in os.listdir(input_dir)
                if re.fullmatch(r"js\d+", name)
            )
        except Exception:
            candidates = []

        if candidates:
            return os.path.join(input_dir, candidates[0])
        return None

    def _close_console_device(self):
        if self._console_fd is not None:
            try:
                os.close(self._console_fd)
            except Exception:
                pass
        self._console_fd = None
        self._console_device_path = None

    def _ensure_console_device(self):
        if self._console_fd is not None:
            return True

        device_path = self._find_console_device_path()
        if device_path is None:
            now = time.perf_counter()
            if (now - self._console_last_missing_log_time) >= 2.0:
                self._append_console_log("[Console Control] No joystick found. Connect PS5 controller as /dev/input/js0.")
                self._console_last_missing_log_time = now
            return False

        try:
            self._console_fd = os.open(device_path, os.O_RDONLY | os.O_NONBLOCK)
            self._console_device_path = device_path
            self._append_console_log(f"[Console Control] Connected joystick: {device_path}")
            return True
        except Exception as exc:
            now = time.perf_counter()
            if (now - self._console_last_missing_log_time) >= 2.0:
                self._append_console_log(f"[Console Control] Failed to open {device_path}: {exc}")
                self._console_last_missing_log_time = now
            self._close_console_device()
            return False

    def _read_console_events(self):
        if not self._ensure_console_device():
            return False

        try:
            while True:
                event = os.read(self._console_fd, 8)
                if len(event) < 8:
                    break
                _, value, event_type, number = struct.unpack("IhBB", event)
                event_type = event_type & ~0x80
                if event_type == 0x02:
                    self._console_axes[int(number)] = float(value) / 32767.0
                elif event_type == 0x01:
                    self._console_buttons[int(number)] = 1.0 if int(value) else 0.0
        except BlockingIOError:
            pass
        except OSError as exc:
            self._append_console_log(f"[Console Control] Joystick disconnected/read error: {exc}")
            self._close_console_device()
            return False
        return True

    def _axis_value(self, axis_index):
        value = float(self._console_axes.get(int(axis_index), 0.0))
        deadband = float(self.console_deadband)
        if abs(value) < deadband:
            return 0.0
        scaled = (abs(value) - deadband) / max(1e-6, 1.0 - deadband)
        return scaled if value > 0.0 else -scaled

    def _trigger_value(self, axis_index):
        raw = float(self._console_axes.get(int(axis_index), -1.0))
        value = (raw + 1.0) * 0.5
        return 0.0 if value < float(self.console_deadband) else min(1.0, value)

    def _button_value(self, button_index):
        return float(self._console_buttons.get(int(button_index), 0.0))

    def _apply_console_deadband(self, value):
        value = float(value)
        deadband = float(self.console_deadband)
        if abs(value) < deadband:
            return 0.0
        scaled = (abs(value) - deadband) / max(1e-6, 1.0 - deadband)
        scaled = min(1.0, max(0.0, scaled))
        return scaled if value > 0.0 else -scaled

    def _sensor_weighted_centroid(self, values, cols, rows, region_mask):
        if not np.any(region_mask):
            return None
        region_values = values[region_mask]
        region_cols = cols[region_mask].astype(float)
        region_rows = rows[region_mask].astype(float)
        weights = np.maximum(float(self.motion_threshold) - region_values, 0.001)
        center_col = float(np.average(region_cols, weights=weights))
        center_row = float(np.average(region_rows, weights=weights))
        return center_col, center_row

    def _sensor_anchored_stick_axes(self, side, values, cols, rows, stick_mask, col_min, col_max, n_row):
        """First press in region sets neutral; output is delta from anchor (LHS/RHS separate)."""
        anchor_attr = (
            "_console_sensor_left_stick_anchor" if side == "left" else "_console_sensor_right_stick_anchor"
        )
        if not np.any(stick_mask):
            setattr(self, anchor_attr, None)
            return 0.0, 0.0

        centroid = self._sensor_weighted_centroid(values, cols, rows, stick_mask)
        if centroid is None:
            setattr(self, anchor_attr, None)
            return 0.0, 0.0

        anchor = getattr(self, anchor_attr)
        if anchor is None:
            setattr(self, anchor_attr, centroid)
            return 0.0, 0.0

        dc = centroid[0] - anchor[0]
        dr = centroid[1] - anchor[1]
        x_half = max(1.0, 0.5 * (float(col_max) - float(col_min)))
        y_half = max(1.0, 0.5 * max(1.0, float(n_row - 1)))
        axis_x = float(np.clip(dc / x_half, -1.0, 1.0))
        axis_y = float(np.clip(dr / y_half, -1.0, 1.0))
        return self._apply_console_deadband(axis_x), self._apply_console_deadband(axis_y)

    def _read_console_sensor_inputs(self):
        values = flatten_column_major_view(self.my_sensor._data.diffPerDataAve)
        indices = np.arange(values.size, dtype=int)
        cols, rows = column_major_coords(self.my_sensor.n_row, indices)
        cols = np.asarray(cols)
        rows = np.asarray(rows)

        n_col = max(2, int(self.my_sensor.n_col))
        n_row = max(2, int(self.my_sensor.n_row))
        left_min = 0
        left_max = max(left_min, n_col // 2 - 1)
        right_min = min(n_col - 1, left_max + 1)
        right_max = n_col - 1
        # L1/R1 use the two matrix rows that map to the top two *display* rows of the
        # Sensor Touch Grid (same rule as ui_ping_direct_finger_motion disp_row).
        # flip_ud off: disp_row == matrix row → shoulder rows 0,1.
        # flip_ud on: disp_row 0,1 ↔ matrix rows n_row-1, n_row-2 (were wrongly 0,1 before).
        flip_ud_setting = int(getattr(self, "console_sensor_touch_grid_flip_ud", 0))
        threshold = float(self.motion_threshold)
        touched = values < threshold
        if flip_ud_setting:
            btn_lo = max(0, n_row - 2)
            shoulder_rows = (rows >= btn_lo) & (rows <= (n_row - 1))
            stick_row_max = btn_lo - 1
            left_stick_mask = (cols >= left_min) & (cols <= left_max) & (rows <= stick_row_max) & touched
            right_stick_mask = (cols >= right_min) & (cols <= right_max) & (rows <= stick_row_max) & touched
        else:
            shoulder_rows = rows <= 1
            stick_row_min = min(2, n_row)
            left_stick_mask = (cols >= left_min) & (cols <= left_max) & (rows >= stick_row_min) & touched
            right_stick_mask = (cols >= right_min) & (cols <= right_max) & (rows >= stick_row_min) & touched

        l1_mask = shoulder_rows & (cols >= left_min) & (cols <= left_max) & touched
        r1_mask = shoulder_rows & (cols >= right_min) & (cols <= right_max) & touched
        l1 = 1.0 if np.any(l1_mask) else 0.0
        r1 = 1.0 if np.any(r1_mask) else 0.0

        lx, ly = self._sensor_anchored_stick_axes(
            "left", values, cols, rows, left_stick_mask, left_min, left_max, n_row
        )
        rx, ry = self._sensor_anchored_stick_axes(
            "right", values, cols, rows, right_stick_mask, right_min, right_max, n_row
        )
        if int(getattr(self, "console_sensor_touch_grid_flip_lr", 0)):
            lx, rx = -float(lx), -float(rx)
            # Mirror L1/R1 with grid: physical left top → display-right (R1) etc.
            l1, r1 = float(r1), float(l1)
        if int(getattr(self, "console_sensor_touch_grid_flip_ud", 0)):
            ly, ry = -float(ly), -float(ry)
        l2 = 0.0
        r2 = 0.0
        return lx, ly, rx, ry, l1, r1, l2, r2

    def get_console_sensor_preview_inputs(self):
        """Public preview API for UI test dialog."""
        return self._read_console_sensor_inputs()

    def get_console_sensor_stick_center_state(self, lx, ly, rx, ry):
        """After preview read: each virtual stick is at anchored neutral (press-to-center active)."""
        eps = 1e-5
        left_anchor = self._console_sensor_left_stick_anchor is not None
        right_anchor = self._console_sensor_right_stick_anchor is not None
        left_center = left_anchor and abs(float(lx)) < eps and abs(float(ly)) < eps
        right_center = right_anchor and abs(float(rx)) < eps and abs(float(ry)) < eps
        return {
            "left_center": left_center,
            "right_center": right_center,
            "left_anchor": left_anchor,
            "right_anchor": right_anchor,
        }

    def _console_velocity_direction_label(self, velocity):
        labels = []
        for axis_name, value in zip(("x", "y", "z", "rx", "ry", "rz"), velocity):
            value = float(value)
            if abs(value) < 1e-6:
                continue
            labels.append(f"{'+' if value > 0.0 else '-'}{axis_name}")
        return " ".join(labels) if labels else "stop"

    def _append_console_motion_log(self, velocity):
        direction = self._console_velocity_direction_label(velocity)
        now = time.perf_counter()
        should_log = (
            direction != self._console_last_logged_direction
            or (direction != "stop" and (now - self._console_last_log_time) >= 0.5)
        )
        if not should_log:
            return

        self._console_last_logged_direction = direction
        self._console_last_log_time = now
        self._append_console_log(
            f"[Console Control] moving: {direction} | frame={self._get_requested_frame()}"
        )

    def _sensor_v2_speed_scale(self, l1, r1):
        if l1 > 0.5 and r1 <= 0.5:
            return float(getattr(self, "console_sensor_v2_slow_scale", 0.35))
        if r1 > 0.5 and l1 <= 0.5:
            return float(getattr(self, "console_sensor_v2_fast_scale", 1.8))
        return 1.0

    def _sensor_v2_clutch_active(self, l1, r1):
        return bool(l1 > 0.5 and r1 > 0.5)

    def _reset_console_sensor_anchors(self):
        self._console_sensor_left_stick_anchor = None
        self._console_sensor_right_stick_anchor = None

    def run_step(self):
        if not self.is_running:
            return
        self._record_loop_tick()
        if self.console_input_source == "sensor_v2":
            lx, ly, rx, ry, l1, r1, l2, r2 = self._read_console_sensor_inputs()
            if self._sensor_v2_clutch_active(l1, r1):
                self._reset_console_sensor_anchors()
                self._apply_stop_output()
                if not self._console_sensor_v2_clutch_logged:
                    self._append_console_log(
                        "[Console Control V2] clutch/re-anchor active: robot stopped"
                    )
                    self._console_sensor_v2_clutch_logged = True
                return

            self._console_sensor_v2_clutch_logged = False
            speed_scale = self._sensor_v2_speed_scale(l1, r1)
            velocity = [
                speed_scale * float(self.console_x_sign) * float(self.console_x_speed) * lx,
                speed_scale * float(self.console_y_sign) * float(self.console_y_speed) * ly,
                speed_scale * float(self.console_z_sign) * float(self.console_z_speed) * ry,
                0.0,
                0.0,
                0.0,
            ]
        else:
            if self.console_input_source == "sensor":
                lx, ly, rx, ry, l1, r1, l2, r2 = self._read_console_sensor_inputs()
                # Sensor mode custom mapping:
                # - left stick up/down (ly) drives Z
                # - L1/R1 drives Y
                y_source = (r1 - l1)
                z_source = ly
            else:
                if not self._read_console_events():
                    self._apply_stop_output()
                    return
                lx = self._axis_value(self.console_axis_left_x)
                ly = self._axis_value(self.console_axis_left_y)
                rx = self._axis_value(self.console_axis_right_x)
                ry = self._axis_value(self.console_axis_right_y)
                l1 = self._button_value(self.console_button_l1)
                r1 = self._button_value(self.console_button_r1)
                l2 = self._trigger_value(self.console_axis_l2)
                r2 = self._trigger_value(self.console_axis_r2)
                y_source = ly
                z_source = (r1 - l1)

            velocity = [
                float(self.console_x_sign) * float(self.console_x_speed) * lx,
                float(self.console_y_sign) * float(self.console_y_speed) * y_source,
                float(self.console_z_sign) * float(self.console_z_speed) * z_source,
                float(self.console_rx_sign) * float(self.console_rx_speed) * ry,
                float(self.console_ry_sign) * float(self.console_ry_speed) * rx,
                float(self.console_rz_sign) * float(self.console_rz_speed) * (r2 - l2),
            ]

        if any(abs(v) > 1e-6 for v in velocity):
            self._apply_motion_output("console_control", velocity)
        else:
            self._apply_stop_output()
        self._append_console_motion_log(velocity)
