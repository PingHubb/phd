import json
import os
import re
import struct
import time

import numpy as np
from PyQt5.QtCore import QTimer
from phd.dependence.paths import ai_resource_path, resource_path

from phd.dependence.sensor_layout import (
    column_major_coords,
    column_major_idx,
    column_major_matrix_view,
    flatten_column_major_view,
)


class DirectFingerMotion:
    RESOURCE_ROOT = resource_path()
    CONFIG_DIR = resource_path("config")
    AI_DATA_DIR = ai_resource_path("data")
    AI_MODELS_DIR = ai_resource_path("models")
    SETTINGS_FILE = os.path.join(CONFIG_DIR, "direct_finger_motion.json")

    def __init__(self, ros_splitter_instance, my_sensor_instance):
        self.ros_splitter = ros_splitter_instance
        self.my_sensor = my_sensor_instance

        self.is_running = False
        self.control_timer = QTimer()
        self.control_timer.timeout.connect(self.run_step)

        self.no_touch_frames = 0
        self.settings_path = self.SETTINGS_FILE
        self.apply_settings(self._default_settings(), save_to_file=False)
        self.load_settings_from_file()

        self.last_active_flat_idx = None
        self.last_active_raw_index = None
        self.current_touch_center_row = None
        self.current_touch_center_col = None
        self.last_touch_center_row = None
        self.last_touch_center_col = None
        self.current_touch_value = None
        self.current_touch_peak_value = None
        self.current_touch_clusters = []
        self.current_two_peak_state = None
        self.last_two_peak_state = None
        self.push_hold_counter = 0
        self.pinch_hold_counter = 0
        self.two_finger_release_grace_frames = 3
        self.two_finger_grace_counter = 0
        self._two_finger_swipe_axis_lock = None
        self._two_finger_swipe_axis_lock_remaining = 0

        self._velocity_mode_enabled = False
        self.robot_command_output_enabled = True
        self.last_robot_velocity_cmd = None
        self.last_teacher_velocity_pre_flip = self._zero_velocity()
        self.current_motion_mode = "stop"
        self._smoothed_velocity = [0.0] * 6
        self._in_push_mode = False
        self.loop_hz = 0.0
        self._loop_tick_count = 0
        self._loop_tick_started_at = time.perf_counter()
        self._last_processed_sensor_frame = None
        self._last_sensor_frame_seen_at = 0.0
        self._last_motion_ratio_log_at = 0.0
        self._last_motion_ratio_log_signature = None

    def _default_settings(self):
        return {
            "config_version": 1,
            "motion_threshold": -3.0,
            "no_touch_reset_limit": 3,
            "keep_margin": 0.8,
            "robot_speed": 0.05,
            "centroid_deadband": 0.005,
            "centroid_gain": 20.0,
            "min_speed_ratio": 1.0,
            "max_speed_ratio": 1.0,
            "velocity_smoothing_alpha": 1.0,
            "push_value_threshold": -12.0,
            "push_hold_deadband": 0.01,
            "push_hold_frames_required": 2,
            "push_speed": 0.08,
            "push_exit_value_offset": 4.0,
            "pull_value_threshold": -12.0,
            "pinch_axis_deadband": 0.003,
            "pinch_distance_threshold": 0.02,
            "pinch_midpoint_deadband": 0.5,
            "pinch_frames_required": 1,
            "pull_speed": 0.08,
            "push_pinch_enabled": True,
            "rotation_speed": 0.002,
            "two_finger_swipe_deadband": 0.06,
            "two_finger_swipe_dominance_ratio": 1.0,
            "two_finger_swipe_axis_lock_frames": 5,
            "two_finger_swipe_enable_horizontal": True,
            "two_finger_swipe_enable_vertical": True,
            "two_finger_release_grace_frames": 3,
            "sensor_frame_timeout_sec": 0.2,
            "frame_interval_ms": 0,
            "debug_output": False,
            "motion_ratio_log_enabled": False,
        }

    def get_settings(self):
        keys = list(self._default_settings().keys())
        return {key: getattr(self, key) for key in keys}

    def apply_settings(self, settings: dict, save_to_file=False):
        defaults = self._default_settings()
        merged = {**defaults, **(settings or {})}
        if settings is not None and "pull_value_threshold" not in settings:
            merged["pull_value_threshold"] = merged.get("push_value_threshold", defaults["pull_value_threshold"])

        int_fields = {
            "config_version",
            "no_touch_reset_limit",
            "push_hold_frames_required",
            "pinch_frames_required",
            "two_finger_swipe_axis_lock_frames",
            "two_finger_release_grace_frames",
            "frame_interval_ms",
        }
        bool_fields = {
            "debug_output",
            "motion_ratio_log_enabled",
            "push_pinch_enabled",
            "two_finger_swipe_enable_horizontal",
            "two_finger_swipe_enable_vertical",
        }

        for key, default_value in defaults.items():
            value = merged.get(key, default_value)
            if key in int_fields:
                value = int(value)
            elif key in bool_fields:
                if isinstance(value, str):
                    value = value.strip().lower() in {"1", "true", "yes", "on"}
                else:
                    value = bool(value)
            else:
                value = float(value)
            setattr(self, key, value)

        if save_to_file:
            self.save_settings_to_file()

    def save_settings_to_file(self):
        try:
            os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(self.get_settings(), f, indent=2)
            print(f"Direct finger motion settings saved: {self.settings_path}")
        except Exception as exc:
            print(f"Failed to save Direct finger motion settings: {exc}")

    def load_settings_from_file(self):
        if not os.path.exists(self.settings_path):
            return
        try:
            with open(self.settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            if int(settings.get("config_version", 0)) < 1:
                print("[DFM] Config outdated — resetting to v1 defaults (normalized units + smoothing).")
                self.apply_settings(self._default_settings(), save_to_file=True)
                return
            self.apply_settings(settings, save_to_file=False)
            print(f"Direct finger motion settings loaded: {self.settings_path}")
        except Exception as exc:
            print(f"Failed to load Direct finger motion settings: {exc}")

    def _debug_print(self, *args, **kwargs):
        if getattr(self, "debug_output", False):
            print(*args, **kwargs)

    def toggle_direct_finger_motion(self):
        self.is_running = not self.is_running

        if self.is_running:
            self._reset_state()
            # Floor of 5 ms: run_step() already skips stale frames via
            # frame_sequence, so polling faster than the sensor delivers
            # frames only burns CPU (0 ms would busy-spin the GUI thread).
            self.control_timer.start(max(5, int(self.frame_interval_ms)))
            print("Direct finger motion STARTED")
        else:
            self.control_timer.stop()
            self._stop_robot_motion(stop_mode=True)
            self._reset_state()
            print("Direct finger motion STOPPED")

    def _reset_state(self):
        self.no_touch_frames = 0
        self.last_active_flat_idx = None
        self.last_active_raw_index = None
        self.current_touch_center_row = None
        self.current_touch_center_col = None
        self.last_touch_center_row = None
        self.last_touch_center_col = None
        self.current_touch_value = None
        self.current_touch_peak_value = None
        self.current_touch_clusters = []
        self.current_two_peak_state = None
        self.last_two_peak_state = None
        self.push_hold_counter = 0
        self.pinch_hold_counter = 0
        self.two_finger_grace_counter = 0
        self._two_finger_swipe_axis_lock = None
        self._two_finger_swipe_axis_lock_remaining = 0
        self.last_robot_velocity_cmd = None
        self.last_teacher_velocity_pre_flip = self._zero_velocity()
        self.current_motion_mode = "stop"
        self._smoothed_velocity = [0.0] * 6
        self._in_push_mode = False
        self._last_processed_sensor_frame = None
        self._last_sensor_frame_seen_at = 0.0
        self._last_motion_ratio_log_at = 0.0
        self._last_motion_ratio_log_signature = None

    def _record_loop_tick(self):
        self._loop_tick_count += 1
        now = time.perf_counter()
        elapsed = now - self._loop_tick_started_at
        if elapsed >= 1.0:
            self.loop_hz = self._loop_tick_count / elapsed
            self._loop_tick_count = 0
            self._loop_tick_started_at = now

    def run_step(self):
        if not self.is_running:
            return
        if self.my_sensor.n_row < 2 or self.my_sensor.n_col < 2:
            return

        self._record_loop_tick()
        data_obj = getattr(self.my_sensor, "_data", None)
        sensor_frame = getattr(data_obj, "frame_sequence", None)
        now = time.perf_counter()
        if sensor_frame is not None:
            if sensor_frame == self._last_processed_sensor_frame:
                timeout = max(0.0, float(getattr(self, "sensor_frame_timeout_sec", 0.2)))
                if (
                    timeout > 0.0
                    and self.last_robot_velocity_cmd != self._zero_velocity()
                    and (now - self._last_sensor_frame_seen_at) >= timeout
                ):
                    self._apply_stop_output()
                return
            self._last_processed_sensor_frame = sensor_frame
            self._last_sensor_frame_seen_at = now
        self.print_single_touch_map_with_motion(threshold=self.motion_threshold)

    def _flat_idx_to_raw_index(self, flat_idx):
        col, row = column_major_coords(self.my_sensor.n_row, flat_idx)
        j_unflipped = (self.my_sensor.n_row - 1) - row
        raw_index = column_major_idx(self.my_sensor.n_row, col, j_unflipped)
        return int(raw_index)

    def _direction_from_delta(self, delta_col, delta_row):
        step_c = 0 if abs(delta_col) < self.centroid_deadband else (1 if delta_col > 0 else -1)
        step_r = 0 if abs(delta_row) < self.centroid_deadband else (1 if delta_row > 0 else -1)

        arrow_map = {
            (-1, -1): "↖",
            (0, -1): "↑",
            (1, -1): "↗",
            (-1, 0): "←",
            (0, 0): "•",
            (1, 0): "→",
            (-1, 1): "↙",
            (0, 1): "↓",
            (1, 1): "↘",
        }

        name_map = {
            (-1, -1): "up-left",
            (0, -1): "up",
            (1, -1): "up-right",
            (-1, 0): "left",
            (0, 0): "stay",
            (1, 0): "right",
            (-1, 1): "down-left",
            (0, 1): "down",
            (1, 1): "down-right",
        }

        key = (step_c, step_r)
        return arrow_map[key], name_map[key]

    def _extract_touch_clusters(self, values, touched, threshold):
        if touched.size == 0:
            return []

        n_row = self.my_sensor.n_row
        remaining = {int(idx) for idx in touched.tolist()}
        clusters = []

        while remaining:
            seed = remaining.pop()
            stack = [seed]
            component = [seed]

            while stack:
                idx = stack.pop()
                col, row = column_major_coords(n_row, idx)

                for dc in (-1, 0, 1):
                    for dr in (-1, 0, 1):
                        if dc == 0 and dr == 0:
                            continue
                        nc = col + dc
                        nr = row + dr
                        if not (0 <= nc < self.my_sensor.n_col and 0 <= nr < self.my_sensor.n_row):
                            continue
                        neighbor = column_major_idx(n_row, nc, nr)
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            stack.append(neighbor)
                            component.append(neighbor)

            comp = np.array(component, dtype=int)
            comp_cols, comp_rows = column_major_coords(n_row, comp)
            comp_values = values[comp]
            weights = np.maximum(threshold - comp_values, 0.001)

            clusters.append(
                {
                    "indices": comp,
                    "center_row": float(np.average(comp_rows, weights=weights)),
                    "center_col": float(np.average(comp_cols, weights=weights)),
                    "peak_value": float(np.min(comp_values)),
                    "mean_value": float(np.mean(comp_values)),
                    "size": int(comp.size),
                }
            )

        clusters.sort(key=lambda c: c["peak_value"])
        return clusters

    def _build_two_peak_state(self):
        if len(self.current_touch_clusters) < 2:
            return None

        pair = sorted(self.current_touch_clusters[:2], key=lambda c: c["center_col"])
        left, right = pair

        denom_c = max(1.0, float(self.my_sensor.n_col - 1))
        denom_r = max(1.0, float(self.my_sensor.n_row - 1))
        horizontal_span = float((right["center_col"] - left["center_col"]) / denom_c)
        if horizontal_span <= 0.0:
            return None

        denom_c = max(1.0, float(self.my_sensor.n_col - 1))
        denom_r = max(1.0, float(self.my_sensor.n_row - 1))
        threshold = float(self.motion_threshold)
        left_force = max(0.0, threshold - float(left["peak_value"]))
        right_force = max(0.0, threshold - float(right["peak_value"]))
        return {
            "left_col": float(left["center_col"]) / denom_c,
            "left_row": float(left["center_row"]) / denom_r,
            "right_col": float(right["center_col"]) / denom_c,
            "right_row": float(right["center_row"]) / denom_r,
            "left_force": left_force,
            "right_force": right_force,
            "span": horizontal_span,
            "mid_col": 0.5 * ((left["center_col"] + right["center_col"]) / denom_c),
            "mid_row": 0.5 * ((left["center_row"] + right["center_row"]) / denom_r),
        }

    def _select_single_touch_flat_idx(self, threshold=-3):
        values = flatten_column_major_view(self.my_sensor._data.diffPerDataAve)
        touched = np.where(values < threshold)[0]

        if touched.size == 0:
            self.current_touch_center_row = None
            self.current_touch_center_col = None
            self.current_touch_value = None
            self.current_touch_peak_value = None
            self.current_touch_clusters = []
            self.current_two_peak_state = None
            return None, values, touched

        self.current_touch_clusters = self._extract_touch_clusters(values, touched, threshold)
        self.current_two_peak_state = self._build_two_peak_state()

        cols, rows = column_major_coords(self.my_sensor.n_row, touched)
        weights = np.maximum(threshold - values[touched], 0.001)

        center_row = float(np.average(rows, weights=weights))
        center_col = float(np.average(cols, weights=weights))
        self.current_touch_center_row = center_row
        self.current_touch_center_col = center_col

        dist2 = (rows - center_row) ** 2 + (cols - center_col) ** 2
        best_local = min(range(len(touched)), key=lambda k: (dist2[k], values[touched[k]]))
        chosen = int(touched[best_local])

        if self.last_active_flat_idx is not None:
            prev_matches = np.where(touched == self.last_active_flat_idx)[0]
            if prev_matches.size > 0:
                prev_local = int(prev_matches[0])
                prev_dist2 = dist2[prev_local]
                chosen_dist2 = dist2[best_local]
                prev_value = values[self.last_active_flat_idx]
                chosen_value = values[chosen]

                if prev_dist2 <= chosen_dist2 + 0.5 and prev_value <= chosen_value + self.keep_margin:
                    chosen = int(self.last_active_flat_idx)

        self.current_touch_value = float(values[chosen])
        self.current_touch_peak_value = float(np.min(values[touched]))
        return chosen, values, touched

    def print_single_touch_map_with_motion(self, threshold=-3, update_robot=True):
        dprint = self._debug_print
        red = "\033[91m"
        green = "\033[92m"
        reset = "\033[0m"

        chosen_flat_idx, values, touched = self._select_single_touch_flat_idx(threshold=threshold)

        if chosen_flat_idx is None:
            self.no_touch_frames += 1
            self.push_hold_counter = 0
            self.pinch_hold_counter = 0
            self.current_two_peak_state = None
            self.last_two_peak_state = None
            self.two_finger_grace_counter = 0

            dprint("\nTouch map (single active point):")
            for _ in range(self.my_sensor.n_row):
                dprint(" ".join(["0"] * self.my_sensor.n_col))

            if self.no_touch_frames >= self.no_touch_reset_limit:
                if self.last_active_raw_index is not None:
                    dprint("Motion: RELEASED")
                if update_robot:
                    self._stop_robot_motion()
                self.last_active_flat_idx = None
                self.last_active_raw_index = None
                self.last_touch_center_row = None
                self.last_touch_center_col = None
            else:
                dprint("Motion: no touch")
                if update_robot:
                    self._stop_robot_motion()

            return

        self.no_touch_frames = 0

        if len(self.current_touch_clusters) >= 2:
            self.two_finger_grace_counter = int(self.two_finger_release_grace_frames)
        elif self.two_finger_grace_counter > 0:
            self.two_finger_grace_counter -= 1

        previous_flat_idx = self.last_active_flat_idx
        previous_center_row = self.last_touch_center_row
        previous_center_col = self.last_touch_center_col
        current_raw_index = self._flat_idx_to_raw_index(chosen_flat_idx)
        current_center_row = self.current_touch_center_row
        current_center_col = self.current_touch_center_col

        if getattr(self, "debug_output", False):
            display_map = np.full((self.my_sensor.n_row, self.my_sensor.n_col), "0", dtype=object)
            if previous_flat_idx is not None:
                prev_col, prev_row = column_major_coords(self.my_sensor.n_row, previous_flat_idx)
                display_map[prev_row, prev_col] = f"{red}1{reset}"

            cur_col, cur_row = column_major_coords(self.my_sensor.n_row, chosen_flat_idx)
            display_map[cur_row, cur_col] = f"{green}1{reset}"

            dprint("\nTouch map (single active point):")
            for row in display_map:
                dprint(" ".join(row))

            touched_raw_indices = [self._flat_idx_to_raw_index(int(idx)) for idx in touched]
            dprint("Touched cluster raw_index:", touched_raw_indices)
            dprint(f"Selected raw_index={current_raw_index}, value={values[chosen_flat_idx]:.2f}")
            dprint(f"Peak touch value={self.current_touch_peak_value:.2f}")
            dprint(f"Centroid (row, col)=({current_center_row:.2f}, {current_center_col:.2f})")

            if self.current_two_peak_state is not None:
                state = self.current_two_peak_state
                dprint(
                    f"Two-peak pinch monitor: span={state['span']:.3f}, "
                    f"left=({state['left_row']:.2f}, {state['left_col']:.2f}), "
                    f"right=({state['right_row']:.2f}, {state['right_col']:.2f})"
                )

            if previous_center_row is None or previous_center_col is None:
                dprint(f"Motion: START at raw_index={current_raw_index}")
            else:
                denom_c = max(1.0, float(self.my_sensor.n_col - 1))
                denom_r = max(1.0, float(self.my_sensor.n_row - 1))
                delta_col = (current_center_col - previous_center_col) / denom_c
                delta_row = (current_center_row - previous_center_row) / denom_r
                arrow, direction_name = self._direction_from_delta(delta_col, delta_row)
                dprint(
                    f"Motion: centroid Δ(col,row)=({delta_col:+.3f}, {delta_row:+.3f})  {arrow}  {direction_name}"
                )

        if update_robot:
            self._update_robot_from_motion(
                previous_center=(previous_center_row, previous_center_col),
                current_center=(current_center_row, current_center_col),
            )

        self.last_active_flat_idx = chosen_flat_idx
        self.last_active_raw_index = current_raw_index
        self.last_touch_center_row = current_center_row
        self.last_touch_center_col = current_center_col
        self.last_two_peak_state = self.current_two_peak_state

    def _ensure_robot_velocity_mode(self):
        if not self._velocity_mode_enabled:
            robot_api = getattr(self.ros_splitter, "robot_api", None)
            if robot_api is None:
                return False
            if hasattr(robot_api, "enter_end_effector_velocity_mode"):
                self._velocity_mode_enabled = bool(
                    robot_api.enter_end_effector_velocity_mode()
                )
            else:
                self._velocity_mode_enabled = bool(
                    robot_api.send_request(robot_api.enable_end_effector_velocity_mode())
                )
        return self._velocity_mode_enabled

    def _get_requested_frame(self):
        selected = getattr(self.ros_splitter, "ai_selected_frame", None)
        if isinstance(selected, str) and selected.strip():
            return selected.strip().lower()

        le = getattr(self.ros_splitter, "ai_frame_input", None)
        if le is not None:
            try:
                txt = le.text().strip().lower()
            except Exception:
                txt = ""
            if txt:
                return txt

        return "tool"

    def _scaled_axis_component(self, delta_value):
        abs_delta = abs(float(delta_value))
        if abs_delta < self.centroid_deadband:
            return 0.0

        min_ratio, max_ratio = self._speed_ratio_bounds()
        scaled = abs_delta * self.centroid_gain
        scaled = max(min_ratio, scaled)
        scaled = min(max_ratio, scaled)
        return scaled if delta_value > 0 else -scaled

    def _speed_ratio_bounds(self):
        min_ratio = max(0.0, float(getattr(self, "min_speed_ratio", 1.0)))
        max_ratio = max(0.0, float(getattr(self, "max_speed_ratio", min_ratio)))
        if max_ratio < min_ratio:
            min_ratio, max_ratio = max_ratio, min_ratio
        return min_ratio, max_ratio

    def _pressure_speed_span(self, value_threshold=None):
        if value_threshold is None:
            value_threshold = self.push_value_threshold
        threshold_gap = abs(float(self.motion_threshold) - float(value_threshold))
        hysteresis_gap = abs(float(getattr(self, "push_exit_value_offset", 0.0)))
        return max(1e-6, hysteresis_gap, threshold_gap * 0.5)

    def _ratio_from_pressure_delta(self, pressure_delta, value_threshold=None):
        min_ratio, max_ratio = self._speed_ratio_bounds()
        if max_ratio <= min_ratio:
            return min_ratio
        normalized = max(0.0, min(1.0, float(pressure_delta) / self._pressure_speed_span(value_threshold)))
        return min_ratio + (max_ratio - min_ratio) * normalized

    def _push_pressure_speed_ratio(self):
        if self.current_touch_peak_value is None:
            return self._speed_ratio_bounds()[0]
        pressure_beyond_push = float(self.push_value_threshold) - float(self.current_touch_peak_value)
        return self._ratio_from_pressure_delta(max(0.0, pressure_beyond_push), self.push_value_threshold)

    def _two_finger_pull_pressure_delta(self, state=None):
        state = self.current_two_peak_state if state is None else state
        if state is None:
            return 0.0
        left_force = max(0.0, float(state.get("left_force", 0.0)))
        right_force = max(0.0, float(state.get("right_force", 0.0)))
        pull_entry_force = max(0.0, float(self.motion_threshold) - float(self.pull_value_threshold))
        average_force = 0.5 * (left_force + right_force)
        return max(0.0, average_force - pull_entry_force)

    def _two_finger_pull_pressure_ready(self, state=None):
        state = self.current_two_peak_state if state is None else state
        if state is None:
            return False
        left_force = max(0.0, float(state.get("left_force", 0.0)))
        right_force = max(0.0, float(state.get("right_force", 0.0)))
        average_force = 0.5 * (left_force + right_force)
        pull_entry_force = max(0.0, float(self.motion_threshold) - float(self.pull_value_threshold))
        return average_force >= pull_entry_force

    def _two_finger_pressure_speed_ratio(self):
        state = self.current_two_peak_state
        if state is None:
            return self._speed_ratio_bounds()[0]
        return self._ratio_from_pressure_delta(
            self._two_finger_pull_pressure_delta(state),
            self.pull_value_threshold,
        )

    def _sensor_axis_denominators(self):
        return (
            max(1.0, float(self.my_sensor.n_row - 1)),
            max(1.0, float(self.my_sensor.n_col - 1)),
        )

    def _zero_velocity(self):
        return [0.0] * 6

    def _apply_motion_output(self, mode, velocity):
        smoothed = self._apply_velocity_smoothing(velocity)
        self._set_teacher_output(mode, smoothed)
        self._send_robot_velocity(smoothed)
        self._log_motion_ratio(mode, smoothed)

    def _apply_stop_output(self):
        zero = self._zero_velocity()
        self._smoothed_velocity = list(zero)
        self._set_teacher_output("stop", zero)
        self._send_robot_velocity(zero)
        self._log_motion_ratio("stop", zero)

    def _apply_velocity_smoothing(self, target_velocity):
        alpha = max(0.0, min(1.0, self.velocity_smoothing_alpha))
        self._smoothed_velocity = [
            alpha * t + (1.0 - alpha) * s
            for t, s in zip(target_velocity, self._smoothed_velocity)
        ]
        if all(abs(v) < 1e-6 for v in self._smoothed_velocity):
            self._smoothed_velocity = [0.0] * 6
        return list(self._smoothed_velocity)

    def _motion_ratio_from_velocity(self, mode, velocity):
        mode = str(mode or "")
        values = [float(v) for v in velocity]
        if mode == "single_finger_swipe":
            base = max(1e-9, abs(float(self.robot_speed)))
            return max(abs(values[0]), abs(values[2])) / base
        if mode == "push":
            base = max(1e-9, abs(float(self.push_speed)))
            return abs(values[1]) / base
        if mode == "two_finger_pull":
            base = max(1e-9, abs(float(self.pull_speed)))
            return abs(values[1]) / base
        if mode == "two_finger_swipe":
            base = max(1e-9, abs(float(self.rotation_speed)))
            return max(abs(values[3]), abs(values[5])) / base
        return 0.0

    def _sent_velocity_from_pre_flip(self, velocity):
        values = [float(v) for v in velocity]
        return [-values[0], -values[1], -values[2], values[3], values[4], values[5]]

    def _motion_ratio_detail_text(self, mode, ratio, velocity):
        display_mode = "pull" if str(mode) == "two_finger_pull" else str(mode)
        return f"[DFM Ratio] mode={display_mode} | ratio={ratio:.3f}"

    def _log_motion_ratio(self, mode, velocity):
        if not bool(getattr(self, "motion_ratio_log_enabled", False)):
            return

        mode = str(mode or "stop")
        ratio = self._motion_ratio_from_velocity(mode, velocity)
        ratio = max(0.0, ratio)
        if mode == "stop":
            signature = (mode, 0.0)
        else:
            signature = (mode, round(ratio, 2))

        now = time.perf_counter()
        is_changed = signature != self._last_motion_ratio_log_signature
        if mode != "stop" and not is_changed and (now - self._last_motion_ratio_log_at) < 0.25:
            return
        if mode == "stop" and not is_changed:
            return

        message = self._motion_ratio_detail_text(mode, ratio, velocity)
        log_display = getattr(self.ros_splitter, "log_display", None)
        if log_display is not None:
            try:
                log_display.append(message)
            except Exception:
                print(message)
        else:
            print(message)

        self._last_motion_ratio_log_signature = signature
        self._last_motion_ratio_log_at = now

    def _compute_touch_motion_features(self, values, prev_row, prev_col):
        touch_mask = values < float(self.motion_threshold)
        active_values = values[touch_mask]
        current_row = self.current_touch_center_row
        current_col = self.current_touch_center_col
        denom_r, denom_c = self._sensor_axis_denominators()

        if prev_row is None or prev_col is None or current_row is None or current_col is None:
            delta_row = 0.0
            delta_col = 0.0
            delta_row_norm = 0.0
            delta_col_norm = 0.0
        else:
            delta_row = float(current_row - prev_row)
            delta_col = float(current_col - prev_col)
            delta_row_norm = float(delta_row / denom_r)
            delta_col_norm = float(delta_col / denom_c)

        return {
            "touch_mask": touch_mask,
            "active_values": active_values,
            "current_row": current_row,
            "current_col": current_col,
            "delta_row": delta_row,
            "delta_col": delta_col,
            "delta_row_norm": delta_row_norm,
            "delta_col_norm": delta_col_norm,
            "cluster_count": int(len(self.current_touch_clusters)),
            "finger_count_est": int(min(len(self.current_touch_clusters), 3)),
            "speed": float(np.hypot(delta_row_norm, delta_col_norm)),
            "peak_value": float(self.current_touch_peak_value) if self.current_touch_peak_value is not None else 0.0,
            "mean_active_value": float(np.mean(active_values)) if active_values.size else 0.0,
            "center_row_norm": 0.0 if current_row is None else float(current_row) / denom_r,
            "center_col_norm": 0.0 if current_col is None else float(current_col) / denom_c,
            "touch_present": 1.0 if active_values.size > 0 else 0.0,
            "selected_frame": str(self._get_requested_frame()),
        }

    def _centroid_delta_to_robot_velocity(self, delta_col, delta_row):
        scaled_col = self._scaled_axis_component(delta_col)
        scaled_row = self._scaled_axis_component(delta_row)

        if scaled_col == 0.0 and scaled_row == 0.0:
            return self._zero_velocity()

        vx = -self.robot_speed * scaled_col
        vz = -self.robot_speed * scaled_row
        return [float(vx), 0.0, float(vz), 0.0, 0.0, 0.0]

    def _push_velocity(self):
        speed = float(self.push_speed) * self._push_pressure_speed_ratio()
        return [0.0, speed, 0.0, 0.0, 0.0, 0.0]

    def _pull_velocity(self):
        speed = float(self.pull_speed) * self._two_finger_pressure_speed_ratio()
        return [0.0, -speed, 0.0, 0.0, 0.0, 0.0]

    def _two_finger_vertical_swipe_to_robot_velocity(self, delta_row):
        if not bool(getattr(self, "two_finger_swipe_enable_vertical", True)):
            return self._zero_velocity()

        scaled_row = self._scaled_axis_component(delta_row)
        if scaled_row == 0.0:
            return self._zero_velocity()

        rx = self.rotation_speed * scaled_row
        return [0.0, 0.0, 0.0, float(rx), 0.0, 0.0]

    def _two_finger_swipe_to_robot_velocity(self):
        curr = self.current_two_peak_state
        prev = self.last_two_peak_state
        if curr is None or prev is None:
            return self._zero_velocity()

        delta_mid_col = curr["mid_col"] - prev["mid_col"]
        delta_mid_row = curr["mid_row"] - prev["mid_row"]

        denom_c = max(1.0, float(self.my_sensor.n_col - 1))
        denom_r = max(1.0, float(self.my_sensor.n_row - 1))
        deadband_col = float(self.two_finger_swipe_deadband) / denom_c
        deadband_row = float(self.two_finger_swipe_deadband) / denom_r
        dominance = max(0.0, float(self.two_finger_swipe_dominance_ratio))

        abs_col = abs(float(delta_mid_col))
        abs_row = abs(float(delta_mid_row))
        use_vertical = abs_row >= deadband_row and abs_row >= (abs_col * dominance)
        use_horizontal = abs_col >= deadband_col and abs_col >= (abs_row * dominance)

        candidate_axis = None
        if use_vertical:
            candidate_axis = "vertical"
        elif use_horizontal:
            candidate_axis = "horizontal"
        else:
            if abs_row >= deadband_row and abs_col < deadband_col:
                candidate_axis = "vertical"
            elif abs_col >= deadband_col and abs_row < deadband_row:
                candidate_axis = "horizontal"

        lock_frames = int(max(0, self.two_finger_swipe_axis_lock_frames))
        if candidate_axis is None:
            if self._two_finger_swipe_axis_lock_remaining > 0:
                self._two_finger_swipe_axis_lock_remaining -= 1
            else:
                self._two_finger_swipe_axis_lock = None
            return self._zero_velocity()

        if self._two_finger_swipe_axis_lock is None:
            self._two_finger_swipe_axis_lock = candidate_axis
            self._two_finger_swipe_axis_lock_remaining = lock_frames
        elif candidate_axis == self._two_finger_swipe_axis_lock:
            self._two_finger_swipe_axis_lock_remaining = lock_frames
        elif self._two_finger_swipe_axis_lock_remaining > 0:
            candidate_axis = self._two_finger_swipe_axis_lock
            self._two_finger_swipe_axis_lock_remaining -= 1
        else:
            self._two_finger_swipe_axis_lock = candidate_axis
            self._two_finger_swipe_axis_lock_remaining = lock_frames

        allow_horizontal = bool(getattr(self, "two_finger_swipe_enable_horizontal", True))
        allow_vertical = bool(getattr(self, "two_finger_swipe_enable_vertical", True))
        if candidate_axis == "horizontal" and not allow_horizontal:
            return self._zero_velocity()
        if candidate_axis == "vertical" and not allow_vertical:
            return self._zero_velocity()

        if candidate_axis == "vertical":
            return self._two_finger_vertical_swipe_to_robot_velocity(delta_mid_row)

        scaled_col = self._scaled_axis_component(delta_mid_col)
        if scaled_col == 0.0:
            return self._zero_velocity()

        rz = -self.rotation_speed * scaled_col
        return [0.0, 0.0, 0.0, 0.0, 0.0, float(rz)]

    def _set_teacher_output(self, mode, velocity):
        self.current_motion_mode = str(mode)
        self.last_teacher_velocity_pre_flip = [float(v) for v in velocity]

    def _send_robot_velocity(self, velocity):
        velocity = [float(v) for v in velocity]
        velocity = [-velocity[0], -velocity[1], -velocity[2], velocity[3], velocity[4], velocity[5]]

        if self.last_robot_velocity_cmd == velocity:
            return

        if not bool(getattr(self, "robot_command_output_enabled", True)):
            self.last_robot_velocity_cmd = velocity
            self._debug_print(f"Virtual robot velocity command ({self._get_requested_frame()}): {velocity}")
            return

        robot_api = getattr(self.ros_splitter, "robot_api", None)
        if robot_api is None:
            return
        if not self._ensure_robot_velocity_mode():
            return

        frame = self._get_requested_frame()

        if hasattr(robot_api, "send_end_effector_velocity_in_frame"):
            robot_api.send_end_effector_velocity_in_frame(
                velocity[:3],
                velocity[3:],
                frame=frame,
                ensure_mode=False,
            )
        else:
            try:
                cmd = robot_api.set_end_effector_velocity_in_frame(velocity[:3], velocity[3:], frame=frame)
            except Exception:
                cmd = robot_api.set_end_effector_velocity(velocity)
            robot_api.send_request(cmd)
        self.last_robot_velocity_cmd = velocity
        self._debug_print(f"Robot velocity command ({frame}): {velocity}")

    def _two_peak_pinch_is_pull(self):
        curr = self.current_two_peak_state
        prev = self.last_two_peak_state
        if curr is None or prev is None:
            self.pinch_hold_counter = 0
            return False

        denom_c = max(1.0, float(self.my_sensor.n_col - 1))
        denom_r = max(1.0, float(self.my_sensor.n_row - 1))

        axis_deadband = float(self.pinch_axis_deadband) / denom_c
        distance_threshold = float(self.pinch_distance_threshold) / denom_c
        midpoint_deadband_col = float(self.pinch_midpoint_deadband) / denom_c
        midpoint_deadband_row = float(self.pinch_midpoint_deadband) / denom_r

        left_move = curr["left_col"] - prev["left_col"]
        right_move = curr["right_col"] - prev["right_col"]
        span_delta = curr["span"] - prev["span"]
        midpoint_shift_col = abs(curr["mid_col"] - prev["mid_col"])
        midpoint_shift_row = abs(curr["mid_row"] - prev["mid_row"])

        pinch_detected = (
            left_move >= axis_deadband
            and right_move <= -axis_deadband
            and span_delta <= -distance_threshold
            and midpoint_shift_col <= midpoint_deadband_col
            and midpoint_shift_row <= midpoint_deadband_row
        )

        if pinch_detected:
            if not self._two_finger_pull_pressure_ready(curr):
                self.pinch_hold_counter = 0
                return False
            self.pinch_hold_counter += 1
            return self.pinch_hold_counter >= self.pinch_frames_required

        self.pinch_hold_counter = 0
        return False

    def _update_robot_from_motion(self, previous_center=None, current_center=None):
        if self.current_two_peak_state is not None:
            push_pinch_enabled = bool(getattr(self, "push_pinch_enabled", True))
            if push_pinch_enabled and self._two_peak_pinch_is_pull():
                self.push_hold_counter = 0
                self._in_push_mode = False
                velocity = self._pull_velocity()
                self._apply_motion_output("two_finger_pull", velocity)
                return
            if not push_pinch_enabled:
                self.pinch_hold_counter = 0

            self.push_hold_counter = 0
            self._in_push_mode = False
            velocity = self._two_finger_swipe_to_robot_velocity()
            mode = "two_finger_swipe" if any(abs(v) > 1e-12 for v in velocity) else "stop"
            self._apply_motion_output(mode, velocity)
            return

        self.pinch_hold_counter = 0

        if self.two_finger_grace_counter > 0:
            self.push_hold_counter = 0
            self._in_push_mode = False
            self._apply_stop_output()
            return

        if previous_center is not None and current_center is not None:
            prev_row, prev_col = previous_center
            curr_row, curr_col = current_center

            if prev_row is None or prev_col is None or curr_row is None or curr_col is None:
                self.push_hold_counter = 0
                self._in_push_mode = False
                self._apply_stop_output()
                return

            denom_r, denom_c = self._sensor_axis_denominators()
            delta_col = (curr_col - prev_col) / denom_c
            delta_row = (curr_row - prev_row) / denom_r
            hold_distance = max(abs(delta_col), abs(delta_row))

            push_pinch_enabled = bool(getattr(self, "push_pinch_enabled", True))
            push_enter = (
                push_pinch_enabled
                and self.current_touch_peak_value is not None
                and self.current_touch_peak_value <= self.push_value_threshold
                and hold_distance <= self.push_hold_deadband
            )
            push_exit_threshold = self.push_value_threshold + self.push_exit_value_offset
            push_stay = (
                push_pinch_enabled
                and self._in_push_mode
                and self.current_touch_peak_value is not None
                and self.current_touch_peak_value <= push_exit_threshold
                and hold_distance <= self.push_hold_deadband * 2.0
            )

            if push_enter or push_stay:
                self.push_hold_counter += 1
                if self.push_hold_counter >= self.push_hold_frames_required:
                    self._in_push_mode = True
                    velocity = self._push_velocity()
                    self._apply_motion_output("push", velocity)
                    return
            else:
                self.push_hold_counter = 0
                self._in_push_mode = False

            velocity = self._centroid_delta_to_robot_velocity(delta_col, delta_row)
            mode = "single_finger_swipe" if any(abs(v) > 1e-12 for v in velocity) else "stop"
            self._apply_motion_output(mode, velocity)
            return

        self.push_hold_counter = 0
        self._in_push_mode = False
        self._apply_stop_output()

    def _stop_robot_motion(self, stop_mode=False):
        self.push_hold_counter = 0
        self.pinch_hold_counter = 0
        self.two_finger_grace_counter = 0
        self._two_finger_swipe_axis_lock = None
        self._two_finger_swipe_axis_lock_remaining = 0
        self._in_push_mode = False
        self.current_two_peak_state = None
        self.last_two_peak_state = None
        if stop_mode:
            self._smoothed_velocity = [0.0] * 6
        self._apply_stop_output()

        if stop_mode and self._velocity_mode_enabled:
            robot_api = getattr(self.ros_splitter, "robot_api", None)
            if robot_api is not None and hasattr(robot_api, "exit_end_effector_velocity_mode"):
                robot_api.exit_end_effector_velocity_mode(send_zero=False)
            elif robot_api is not None:
                robot_api.send_request(robot_api.stop_end_effector_velocity_mode())
            self._velocity_mode_enabled = False


class AI_DirectFingerMotion(DirectFingerMotion):
    RECORD_TARGET_HZ = 60.0
    RECORD_TIMER_INTERVAL_MS = 16
    TARGET_SEQ_LEN = 16
    TEACHING_AUTO_LABEL = "auto"
    TEACHING_LABELS = (
        "auto",
        "stop",
        "normal_swipe",
        "push",
        "pull",
        "x_pos",
        "x_neg",
        "y_pos",
        "y_neg",
        "z_pos",
        "z_neg",
        "rx_pos",
        "rx_neg",
        "ry_pos",
        "ry_neg",
        "rz_pos",
        "rz_neg",
    )

    def __init__(self, ros_splitter_instance, my_sensor_instance):
        super().__init__(ros_splitter_instance, my_sensor_instance)

        self.dataset_root = os.path.join(self.AI_DATA_DIR, "ai_direct_finger_motion")
        self.min_frames_to_save = 5
        self.session_tag = "default"
        self.trial_number = None
        self.episode_started_at = None
        self.episode_started_perf_at = None
        self._last_episode_perf_timestamp = None
        self.teaching_override_label = self.TEACHING_AUTO_LABEL
        self.send_robot_commands = False
        self.current_episode = self._create_empty_episode()

    def _create_empty_episode(self):
        return {
            "timestamps": [],
            "elapsed_sec": [],
            "dt_sec": [],
            "sensor_frame_sequence": [],
            "rawData": [],
            "diffData": [],
            "diffPerData": [],
            "diffDataAve": [],
            "diffPerDataAve": [],
            "touch_mask": [],
            "touch_present": [],
            "finger_count_est": [],
            "cluster_count": [],
            "center": [],
            "delta": [],
            "delta_norm": [],
            "speed": [],
            "peak_value": [],
            "mean_active_value": [],
            "mode": [],
            "teacher_velocity_pre_flip": [],
            "teacher_velocity_sent": [],
            "teacher_velocity_target": [],
            "intended_velocity_target": [],
            "intended_mode": [],
            "teaching_label": [],
            "teaching_source": [],
            "manual_override_active": [],
            "robot_tool_pose": [],
            "robot_joint_positions": [],
            "robot_feedback_valid": [],
            "control_frame_idx": [],
            "two_peak_state": [],
            "selected_frame": [],
        }

    @staticmethod
    def _safe_sequence(value, length):
        if value is None:
            return [float("nan")] * int(length)
        try:
            seq = list(value)
        except Exception:
            return [float("nan")] * int(length)

        out = []
        for idx in range(int(length)):
            try:
                out.append(float(seq[idx]))
            except Exception:
                out.append(float("nan"))
        return out

    def _safe_tool_pose(self):
        robot_api = getattr(self.ros_splitter, "robot_api", None)
        if robot_api is None or not hasattr(robot_api, "get_current_tool_position"):
            return [float("nan")] * 7
        try:
            pos, quat = robot_api.get_current_tool_position()
            return self._safe_sequence(pos, 3) + self._safe_sequence(quat, 4)
        except Exception:
            return [float("nan")] * 7

    def _safe_joint_positions(self):
        robot_api = getattr(self.ros_splitter, "robot_api", None)
        if robot_api is None or not hasattr(robot_api, "get_current_positions"):
            return [float("nan")] * 6
        try:
            return self._safe_sequence(robot_api.get_current_positions(), 6)
        except Exception:
            return [float("nan")] * 6

    @staticmethod
    def _selected_frame_index(frame_name):
        frame_name = str(frame_name or "").lower()
        if frame_name in {"tool", "tcp", "joint6", "j6"}:
            return 6.0
        if frame_name in {"base", "world", "joint1", "j1"}:
            return 1.0
        match = re.fullmatch(r"(?:joint|j)([1-6])", frame_name)
        if match:
            return float(match.group(1))
        return 0.0

    def _two_peak_state_vector(self):
        state = self.current_two_peak_state
        if state is None:
            return [float("nan")] * 9
        return [
            float(state.get("left_col", np.nan)),
            float(state.get("left_row", np.nan)),
            float(state.get("right_col", np.nan)),
            float(state.get("right_row", np.nan)),
            float(state.get("left_force", np.nan)),
            float(state.get("right_force", np.nan)),
            float(state.get("span", np.nan)),
            float(state.get("mid_col", np.nan)),
            float(state.get("mid_row", np.nan)),
        ]

    def set_teaching_override(self, label=None):
        label = str(label or self.TEACHING_AUTO_LABEL).strip().lower()
        if label in {"dfm", "auto_dfm", "auto/dfm", "none"}:
            label = self.TEACHING_AUTO_LABEL
        if label not in self.TEACHING_LABELS:
            label = self.TEACHING_AUTO_LABEL
        self.teaching_override_label = label
        return self.teaching_override_label

    def get_teaching_override_label(self):
        return str(getattr(self, "teaching_override_label", self.TEACHING_AUTO_LABEL))

    def _manual_teaching_velocity_target(self, label):
        label = str(label or self.TEACHING_AUTO_LABEL)
        linear_speed = float(
            max(
                abs(float(getattr(self, "robot_speed", 0.0))),
                abs(float(getattr(self, "push_speed", 0.0))),
                abs(float(getattr(self, "pull_speed", 0.0))),
            )
        )
        push_speed = abs(float(getattr(self, "push_speed", linear_speed)))
        pull_speed = abs(float(getattr(self, "pull_speed", linear_speed)))
        rotation_speed = abs(float(getattr(self, "rotation_speed", 0.0)))

        velocity = [0.0] * 6
        if label == "stop":
            return velocity
        if label == "push":
            velocity[1] = -push_speed * self._push_pressure_speed_ratio()
            return velocity
        if label == "pull":
            velocity[1] = pull_speed * self._two_finger_pressure_speed_ratio()
            return velocity

        axis_map = {
            "x_pos": (0, 1.0, linear_speed),
            "x_neg": (0, -1.0, linear_speed),
            "y_pos": (1, 1.0, linear_speed),
            "y_neg": (1, -1.0, linear_speed),
            "z_pos": (2, 1.0, linear_speed),
            "z_neg": (2, -1.0, linear_speed),
            "rx_pos": (3, 1.0, rotation_speed),
            "rx_neg": (3, -1.0, rotation_speed),
            "ry_pos": (4, 1.0, rotation_speed),
            "ry_neg": (4, -1.0, rotation_speed),
            "rz_pos": (5, 1.0, rotation_speed),
            "rz_neg": (5, -1.0, rotation_speed),
        }
        spec = axis_map.get(label)
        if spec is None:
            return None
        axis, sign, speed = spec
        velocity[axis] = float(sign * speed)
        return velocity

    def _normal_swipe_teaching_target(self, feature):
        if feature is None or not bool(feature.get("touch_present", 0.0)):
            return self._zero_velocity(), "stop"

        delta_col = float(feature.get("delta_col_norm", 0.0))
        delta_row = float(feature.get("delta_row_norm", 0.0))
        pre_flip_velocity = self._centroid_delta_to_robot_velocity(delta_col, delta_row)
        if not any(abs(value) > 1e-12 for value in pre_flip_velocity):
            return self._zero_velocity(), "stop"
        return self._sent_velocity_from_pre_flip(pre_flip_velocity), "single_finger_swipe"

    def _intended_teaching_target(self, dfm_target, dfm_mode, feature=None):
        label = self.get_teaching_override_label()
        if label == self.TEACHING_AUTO_LABEL:
            return list(dfm_target), str(dfm_mode), label, "dfm", 0
        if label == "normal_swipe":
            velocity, mode = self._normal_swipe_teaching_target(feature)
            return list(velocity), mode, label, "manual", 1

        velocity = self._manual_teaching_velocity_target(label)
        if velocity is None:
            return list(dfm_target), str(dfm_mode), self.TEACHING_AUTO_LABEL, "dfm", 0
        return list(velocity), str(label), label, "manual", 1

    def sanitize_session_tag(self, session_tag):
        session_tag = (session_tag or "default").strip()
        session_tag = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_tag)
        return session_tag or "default"

    def _session_dir(self):
        return os.path.join(self.dataset_root, f"session_{self.session_tag}")

    def _get_next_trial_number(self):
        session_dir = self._session_dir()
        os.makedirs(session_dir, exist_ok=True)
        existing = []
        for name in os.listdir(session_dir):
            if name.endswith(".npz") and name.startswith("trial_"):
                match = re.search(r"trial_(\d+)", name)
                if match:
                    existing.append(int(match.group(1)))
        return max(existing, default=0) + 1

    def toggle_ai_direct_finger_motion(self, session_tag=None, send_robot_commands=False):
        self.is_running = not self.is_running

        if self.is_running:
            self.session_tag = self.sanitize_session_tag(session_tag)
            self.send_robot_commands = bool(send_robot_commands)
            self.robot_command_output_enabled = bool(send_robot_commands)
            self.trial_number = self._get_next_trial_number()
            self.episode_started_at = time.time()
            self.episode_started_perf_at = time.perf_counter()
            self._last_episode_perf_timestamp = None
            self.current_episode = self._create_empty_episode()
            self._reset_state()
            self.control_timer.start(int(self.RECORD_TIMER_INTERVAL_MS))
            record_mode = "with_robot" if self.send_robot_commands else "no_robot"
            print(
                f"AI direct finger motion STARTED ({record_mode}) | session='{self.session_tag}' | "
                f"trial={self.trial_number} | target={self.RECORD_TARGET_HZ:.0f} Hz"
            )
        else:
            self.control_timer.stop()
            self._stop_robot_motion(stop_mode=True)
            self._save_episode()
            self.robot_command_output_enabled = True
            self.send_robot_commands = False
            self._reset_state()
            self.episode_started_perf_at = None
            self._last_episode_perf_timestamp = None
            print("AI direct finger motion STOPPED")

    def run_step(self):
        if not self.is_running:
            return
        if self.my_sensor.n_row < 2 or self.my_sensor.n_col < 2:
            return

        data_obj = getattr(self.my_sensor, "_data", None)
        sensor_frame_sequence = getattr(data_obj, "frame_sequence", None)
        perf_timestamp = time.perf_counter()
        if sensor_frame_sequence is not None:
            if sensor_frame_sequence == self._last_processed_sensor_frame:
                timeout = max(0.0, float(getattr(self, "sensor_frame_timeout_sec", 0.2)))
                if (
                    timeout > 0.0
                    and self.last_robot_velocity_cmd != self._zero_velocity()
                    and (perf_timestamp - self._last_sensor_frame_seen_at) >= timeout
                ):
                    self._apply_stop_output()
                return
            self._last_processed_sensor_frame = sensor_frame_sequence
            self._last_sensor_frame_seen_at = perf_timestamp

        prev_row = self.last_touch_center_row
        prev_col = self.last_touch_center_col
        timestamp = time.time()
        sensor_snapshot = self._snapshot_sensor_frame()

        self.print_single_touch_map_with_motion(threshold=self.motion_threshold)
        self._append_teacher_frame(
            timestamp,
            perf_timestamp,
            sensor_frame_sequence,
            sensor_snapshot,
            prev_row,
            prev_col,
        )

    def _snapshot_sensor_frame(self):
        data = self.my_sensor._data
        return {
            "rawData": column_major_matrix_view(data.rawData, dtype=np.float32, copy=True),
            "diffData": column_major_matrix_view(data.diffData, dtype=np.float32, copy=True),
            "diffPerData": column_major_matrix_view(data.diffPerData, dtype=np.float32, copy=True),
            "diffDataAve": column_major_matrix_view(data.diffDataAve, dtype=np.float32, copy=True),
            "diffPerDataAve": column_major_matrix_view(data.diffPerDataAve, dtype=np.float32, copy=True),
        }

    def _append_teacher_frame(
        self,
        timestamp,
        perf_timestamp,
        sensor_frame_sequence,
        sensor_snapshot,
        prev_row,
        prev_col,
    ):
        episode = self.current_episode
        values = sensor_snapshot["diffPerDataAve"]
        feature = self._compute_touch_motion_features(values, prev_row, prev_col)

        if self.episode_started_perf_at is None:
            self.episode_started_perf_at = float(perf_timestamp)
        elapsed_sec = float(perf_timestamp - self.episode_started_perf_at)
        if self._last_episode_perf_timestamp is None:
            dt_sec = 0.0
        else:
            dt_sec = float(perf_timestamp - self._last_episode_perf_timestamp)
        self._last_episode_perf_timestamp = float(perf_timestamp)

        teacher_pre_flip = self._safe_sequence(self.last_teacher_velocity_pre_flip, 6)
        teacher_sent = self._safe_sequence(self.last_robot_velocity_cmd or self._zero_velocity(), 6)
        (
            intended_velocity,
            intended_mode,
            teaching_label,
            teaching_source,
            manual_override_active,
        ) = self._intended_teaching_target(teacher_sent, self.current_motion_mode, feature)
        tool_pose = self._safe_tool_pose()
        joint_positions = self._safe_joint_positions()
        robot_feedback_valid = int(
            any(np.isfinite(tool_pose)) or any(np.isfinite(joint_positions))
        )
        selected_frame = feature["selected_frame"]

        episode["timestamps"].append(float(timestamp))
        episode["elapsed_sec"].append(np.float32(elapsed_sec))
        episode["dt_sec"].append(np.float32(dt_sec))
        episode["sensor_frame_sequence"].append(
            np.int64(-1 if sensor_frame_sequence is None else int(sensor_frame_sequence))
        )
        for key, value in sensor_snapshot.items():
            episode[key].append(value)
        episode["touch_mask"].append(feature["touch_mask"].astype(np.uint8))
        episode["touch_present"].append(np.uint8(feature["touch_present"] > 0.0))
        episode["finger_count_est"].append(np.int16(feature["finger_count_est"]))
        episode["cluster_count"].append(np.int16(feature["cluster_count"]))
        episode["center"].append(
            np.array(
                [
                    np.nan if feature["current_row"] is None else float(feature["current_row"]),
                    np.nan if feature["current_col"] is None else float(feature["current_col"]),
                ],
                dtype=np.float32,
            )
        )
        episode["delta"].append(np.array([feature["delta_row"], feature["delta_col"]], dtype=np.float32))
        episode["delta_norm"].append(np.array([feature["delta_row_norm"], feature["delta_col_norm"]], dtype=np.float32))
        episode["speed"].append(np.float32(feature["speed"]))
        episode["peak_value"].append(np.float32(feature["peak_value"]))
        episode["mean_active_value"].append(np.float32(feature["mean_active_value"]))
        episode["mode"].append(str(self.current_motion_mode))
        episode["teacher_velocity_pre_flip"].append(np.asarray(teacher_pre_flip, dtype=np.float32))
        episode["teacher_velocity_sent"].append(np.asarray(teacher_sent, dtype=np.float32))
        episode["teacher_velocity_target"].append(np.asarray(teacher_sent, dtype=np.float32))
        episode["intended_velocity_target"].append(np.asarray(intended_velocity, dtype=np.float32))
        episode["intended_mode"].append(str(intended_mode))
        episode["teaching_label"].append(str(teaching_label))
        episode["teaching_source"].append(str(teaching_source))
        episode["manual_override_active"].append(np.uint8(manual_override_active))
        episode["robot_tool_pose"].append(np.asarray(tool_pose, dtype=np.float32))
        episode["robot_joint_positions"].append(np.asarray(joint_positions, dtype=np.float32))
        episode["robot_feedback_valid"].append(np.uint8(robot_feedback_valid))
        episode["control_frame_idx"].append(np.float32(self._selected_frame_index(selected_frame)))
        episode["two_peak_state"].append(np.asarray(self._two_peak_state_vector(), dtype=np.float32))
        episode["selected_frame"].append(selected_frame)

    def _save_episode(self):
        frame_count = len(self.current_episode["timestamps"])
        if frame_count < self.min_frames_to_save:
            print(f"AI direct finger motion: only {frame_count} frame(s), skip save.")
            self.current_episode = self._create_empty_episode()
            return

        session_dir = self._session_dir()
        os.makedirs(session_dir, exist_ok=True)

        started_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.episode_started_at or time.time()))
        filename = os.path.join(session_dir, f"trial_{int(self.trial_number):04d}_{started_stamp}.npz")

        metadata = {
            "dataset_version": 3,
            "dataset_role": "ai_direct_finger_motion_stage2_teaching_recording",
            "session_tag": self.session_tag,
            "trial_number": int(self.trial_number or 0),
            "frame_count": int(frame_count),
            "started_at": float(self.episode_started_at or time.time()),
            "saved_at": float(time.time()),
            "sensor_rows": int(self.my_sensor.n_row),
            "sensor_cols": int(self.my_sensor.n_col),
            "sensor_model": str(getattr(self.my_sensor, "current_model_name", "")),
            "sensor_reorder_mode": str(getattr(self.my_sensor, "current_reorder_mode", "")),
            "sensor_average_window_size": int(getattr(self.my_sensor, "sensor_average_window_size", 0) or 0),
            "motion_threshold": float(self.motion_threshold),
            "teacher": "direct_finger_motion_v1_rule_based",
            "teacher_layout": "column_major_matrix_view",
            "record_mode": "with_robot" if bool(getattr(self, "send_robot_commands", False)) else "no_robot",
            "send_robot_commands": bool(getattr(self, "send_robot_commands", False)),
            "record_target_hz": float(self.RECORD_TARGET_HZ),
            "record_timer_interval_ms": int(self.RECORD_TIMER_INTERVAL_MS),
            "target_sequence_length": int(self.TARGET_SEQ_LEN),
            "target_window_sec_estimate": float(self.TARGET_SEQ_LEN / self.RECORD_TARGET_HZ),
            "fresh_sensor_frame_gated": True,
            "model_input_intent": "recent tactile sequence + robot state",
            "model_output_intent": "6d velocity [vx, vy, vz, rx, ry, rz] in selected EE frame",
            "training_target_key": "intended_velocity_target",
            "dfm_training_target_key": "teacher_velocity_target",
            "intended_training_target_key": "intended_velocity_target",
            "teaching_labels": list(self.TEACHING_LABELS),
            "teaching_auto_label": self.TEACHING_AUTO_LABEL,
            "two_peak_state_layout": [
                "left_col",
                "left_row",
                "right_col",
                "right_row",
                "left_force",
                "right_force",
                "span",
                "mid_col",
                "mid_row",
            ],
            "robot_tool_pose_layout": ["x", "y", "z", "qw", "qx", "qy", "qz"],
            "robot_joint_positions_layout": ["j1", "j2", "j3", "j4", "j5", "j6"],
            "dfm_settings": self.get_settings(),
        }

        episode = self.current_episode
        np.savez_compressed(
            filename,
            metadata_json=np.array(json.dumps(metadata)),
            timestamps=np.asarray(episode["timestamps"], dtype=np.float64),
            elapsed_sec=np.asarray(episode["elapsed_sec"], dtype=np.float32),
            dt_sec=np.asarray(episode["dt_sec"], dtype=np.float32),
            sensor_frame_sequence=np.asarray(episode["sensor_frame_sequence"], dtype=np.int64),
            rawData=np.stack(episode["rawData"], axis=0),
            diffData=np.stack(episode["diffData"], axis=0),
            diffPerData=np.stack(episode["diffPerData"], axis=0),
            diffDataAve=np.stack(episode["diffDataAve"], axis=0),
            diffPerDataAve=np.stack(episode["diffPerDataAve"], axis=0),
            touch_mask=np.stack(episode["touch_mask"], axis=0),
            touch_present=np.asarray(episode["touch_present"], dtype=np.uint8),
            finger_count_est=np.asarray(episode["finger_count_est"], dtype=np.int16),
            cluster_count=np.asarray(episode["cluster_count"], dtype=np.int16),
            center=np.stack(episode["center"], axis=0),
            delta=np.stack(episode["delta"], axis=0),
            delta_norm=np.stack(episode["delta_norm"], axis=0),
            speed=np.asarray(episode["speed"], dtype=np.float32),
            peak_value=np.asarray(episode["peak_value"], dtype=np.float32),
            mean_active_value=np.asarray(episode["mean_active_value"], dtype=np.float32),
            mode=np.asarray(episode["mode"], dtype="U32"),
            teacher_velocity_pre_flip=np.stack(episode["teacher_velocity_pre_flip"], axis=0),
            teacher_velocity_sent=np.stack(episode["teacher_velocity_sent"], axis=0),
            teacher_velocity_target=np.stack(episode["teacher_velocity_target"], axis=0),
            intended_velocity_target=np.stack(episode["intended_velocity_target"], axis=0),
            intended_mode=np.asarray(episode["intended_mode"], dtype="U32"),
            teaching_label=np.asarray(episode["teaching_label"], dtype="U32"),
            teaching_source=np.asarray(episode["teaching_source"], dtype="U16"),
            manual_override_active=np.asarray(episode["manual_override_active"], dtype=np.uint8),
            robot_tool_pose=np.stack(episode["robot_tool_pose"], axis=0),
            robot_joint_positions=np.stack(episode["robot_joint_positions"], axis=0),
            robot_feedback_valid=np.asarray(episode["robot_feedback_valid"], dtype=np.uint8),
            control_frame_idx=np.asarray(episode["control_frame_idx"], dtype=np.float32),
            two_peak_state=np.stack(episode["two_peak_state"], axis=0),
            selected_frame=np.asarray(episode["selected_frame"], dtype="U32"),
        )

        print(f"AI direct finger motion episode saved: {filename}")
        self.current_episode = self._create_empty_episode()
        self.episode_started_at = None
        self.trial_number = None


class AI_DirectFingerMotion_execution(DirectFingerMotion):
    MODE_TO_INDEX = {"stop": 0, "move": 1, "push": 2, "pull": 3}
    INDEX_TO_MODE = {v: k for k, v in MODE_TO_INDEX.items()}
    DEFAULT_MODEL_CHECKPOINT = os.path.join(
        DirectFingerMotion.AI_MODELS_DIR,
        "ai_direct_finger_motion",
        "latest_cnn_gru_model.pt",
    )

    def __init__(self, ros_splitter_instance, my_sensor_instance):
        super().__init__(ros_splitter_instance, my_sensor_instance)
        self._torch = None
        self.device = None
        self.model = None
        self.model_loaded = False
        self.model_kind = "unknown"
        self.model_checkpoint_path = self.DEFAULT_MODEL_CHECKPOINT
        self.model_conf_threshold = 0.55
        self.velocity_scale = 1.0
        self.max_linear_speed = 0.05
        self.max_angular_speed = 0.0
        self.prediction_interval_ms = 16
        self.seq_len = 16
        self.input_channels = ["diffPerData", "diffPerDataAve", "frameDiff", "touchMask"]
        self.use_aux_features = True
        self.aux_feature_names = [
            "center_row",
            "center_col",
            "delta_row",
            "delta_col",
            "delta_row_norm",
            "delta_col_norm",
            "speed",
            "peak_value",
            "mean_active_value",
            "touch_present",
            "control_frame_idx",
        ]
        self.scaler_mean = np.zeros(len(self.input_channels), dtype=np.float32)
        self.scaler_std = np.ones(len(self.input_channels), dtype=np.float32)
        self.aux_mean = np.zeros(len(self.aux_feature_names), dtype=np.float32)
        self.aux_std = np.ones(len(self.aux_feature_names), dtype=np.float32)
        self.target_scale = np.ones(6, dtype=np.float32)
        self.lock_rotation_axes = True
        self.dry_run_predictions_only = True
        self.frame_buffer = []
        self.aux_buffer = []
        self.last_prediction = None
        self.last_prediction_time = 0.0
        self._prev_diff_for_frame_diff = None
        self.zero_keepalive_sec = 0.5
        self.idle_reenable_sec = 1.0
        # Background inference (keeps torch forward passes off the GUI
        # thread). The robustness benchmark sets this to False because it
        # drives run_step() synchronously and reads last_prediction directly.
        self.inference_in_background = True
        self._inference_executor = None
        self._pending_inference = None  # (epoch, Future) or None
        self._inference_epoch = 0
        self._reset_execution_runtime_state()

    def _load_torch_runtime(self):
        if self._torch is None:
            import torch

            self._torch = torch
        if self.device is None:
            self.device = self._torch.device(
                "cuda" if self._torch.cuda.is_available() else "cpu"
            )
        return self._torch

    def set_dry_run_predictions_only(self, enabled=True):
        self.dry_run_predictions_only = bool(enabled)
        return self.dry_run_predictions_only

    def _reset_execution_buffers(self):
        self.frame_buffer = []
        self.aux_buffer = []
        self._prev_diff_for_frame_diff = None
        # Invalidate any in-flight background inference.
        self._inference_epoch += 1
        self._pending_inference = None

    def _reset_execution_runtime_state(self):
        self.last_robot_velocity_cmd = None
        self._velocity_mode_enabled = False
        self._last_velocity_send_time = 0.0
        self._last_nonzero_command_time = 0.0

    def toggle_ai_direct_finger_motion_execution(self, model_checkpoint_path=None):
        self.is_running = not self.is_running
        if self.is_running:
            if model_checkpoint_path:
                self.model_checkpoint_path = model_checkpoint_path
            if not self.model_loaded:
                self.load_model(self.model_checkpoint_path)
            if not self.model_loaded:
                self.is_running = False
                print("AI direct finger motion execution failed to start: model is not loaded.")
                return
            self._reset_state()
            self._reset_execution_buffers()
            self._reset_execution_runtime_state()
            if not self.dry_run_predictions_only:
                self._ensure_robot_velocity_mode()
            self.control_timer.start(self.prediction_interval_ms)
            mode_text = "DRY RUN" if self.dry_run_predictions_only else "ROBOT MOTION ENABLED"
            print(
                f"AI direct finger motion execution STARTED ({mode_text}) | "
                f"model={self.model_checkpoint_path}"
            )
        else:
            self.control_timer.stop()
            if self.dry_run_predictions_only:
                self._set_teacher_output("stop", self._zero_velocity())
            else:
                self._stop_robot_motion_execution(stop_mode=True)
            self._reset_state()
            self._reset_execution_buffers()
            self._reset_execution_runtime_state()
            print("AI direct finger motion execution STOPPED")

    def load_model(self, checkpoint_path=None):
        checkpoint_path = checkpoint_path or self.model_checkpoint_path
        try:
            torch = self._load_torch_runtime()

            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            config = checkpoint.get("config", {})
            model_class = str(checkpoint.get("model_class", ""))
            model_family = str(config.get("model_family", ""))
            if model_class == "TactileCNNGRUPolicy" or model_family == "tactile_cnn_gru_policy":
                self._load_cnn_gru_checkpoint(checkpoint, config)
            else:
                self._load_legacy_transformer_checkpoint(checkpoint, config)
            state = checkpoint.get("model_state", checkpoint)
            self.model.load_state_dict(state)
            self.model.eval()
            self.model_loaded = True
            print(
                f"AI_DirectFingerMotion_execution model loaded | kind={self.model_kind} | "
                f"seq_len={self.seq_len} | channels={self.input_channels} | "
                f"aux={self.aux_feature_names} | device={self.device}"
            )
        except Exception as exc:
            self.model_loaded = False
            self.model = None
            print(f"Failed to load AI direct finger motion model: {exc}")

    def _load_cnn_gru_checkpoint(self, _checkpoint, config):
        from phd.dependence.tactile_models import TactileCNNGRUPolicy

        self.model_kind = "cnn_gru"
        self.seq_len = int(config.get("seq_len", 16))
        self.input_channels = list(config.get("channels", self.input_channels))
        self.aux_feature_names = list(config.get("aux_feature_names", self.aux_feature_names))
        self.use_aux_features = True
        self.scaler_mean = np.asarray(
            config.get("channel_mean", [0.0] * len(self.input_channels)),
            dtype=np.float32,
        )
        self.scaler_std = np.asarray(
            config.get("channel_std", [1.0] * len(self.input_channels)),
            dtype=np.float32,
        )
        self.scaler_std[self.scaler_std < 1e-6] = 1.0
        self.aux_mean = np.asarray(
            config.get("aux_mean", [0.0] * len(self.aux_feature_names)),
            dtype=np.float32,
        )
        self.aux_std = np.asarray(
            config.get("aux_std", [1.0] * len(self.aux_feature_names)),
            dtype=np.float32,
        )
        self.aux_std[self.aux_std < 1e-6] = 1.0
        self.target_scale = np.asarray(config.get("target_scale", [1.0] * 6), dtype=np.float32)
        if self.target_scale.shape[0] < 6:
            self.target_scale = np.pad(self.target_scale, (0, 6 - self.target_scale.shape[0]), constant_values=1.0)
        self.target_scale[self.target_scale < 1e-6] = 1.0
        self.lock_rotation_axes = bool(config.get("lock_rotation_axes", True))
        if self.lock_rotation_axes:
            self.max_angular_speed = 0.0
        self.model = TactileCNNGRUPolicy(
            in_channels=len(self.input_channels),
            aux_dim=len(self.aux_feature_names),
            d_model=int(config.get("d_model", 96)),
            gru_hidden=int(config.get("gru_hidden", 128)),
            gru_layers=int(config.get("gru_layers", 1)),
            dropout=float(config.get("dropout", 0.12)),
            velocity_dim=int(config.get("velocity_dim", 6)),
            mode_classes=int(config.get("mode_classes", len(self.MODE_TO_INDEX))),
            encoder_type=str(config.get("encoder_type", "avgpool")),
        ).to(self.device)

    def _load_legacy_transformer_checkpoint(self, _checkpoint, config):
        from phd.dependence.tactile_models import _AI_DFM_CNNTactileTransformerAux

        self.model_kind = "legacy_transformer"
        self.seq_len = int(config.get("seq_len", 20))
        self.input_channels = list(config.get("input_channels", config.get("channels", self.input_channels)))
        self.use_aux_features = bool(config.get("use_aux_features", True))
        self.aux_feature_names = list(config.get("aux_feature_names", self.aux_feature_names))
        self.scaler_mean = np.asarray(config.get("scaler_mean", [0.0] * len(self.input_channels)), dtype=np.float32)
        self.scaler_std = np.asarray(config.get("scaler_std", [1.0] * len(self.input_channels)), dtype=np.float32)
        self.scaler_std[self.scaler_std < 1e-6] = 1.0
        self.aux_mean = np.asarray(config.get("aux_mean", [0.0] * len(self.aux_feature_names)), dtype=np.float32)
        self.aux_std = np.asarray(config.get("aux_std", [1.0] * len(self.aux_feature_names)), dtype=np.float32)
        self.aux_std[self.aux_std < 1e-6] = 1.0
        self.target_scale = np.ones(6, dtype=np.float32)
        self.lock_rotation_axes = True
        self.model = _AI_DFM_CNNTactileTransformerAux(
            in_channels=len(self.input_channels),
            aux_dim=len(self.aux_feature_names),
            seq_len=self.seq_len,
            d_model=int(config.get("d_model", 128)),
            nhead=int(config.get("nhead", 4)),
            num_layers=int(config.get("num_layers", 3)),
            dim_feedforward=int(config.get("dim_feedforward", 256)),
            dropout=float(config.get("dropout", 0.1)),
            num_mode_classes=len(self.MODE_TO_INDEX),
            num_finger_classes=4,
            use_aux_features=self.use_aux_features,
        ).to(self.device)

    def _reset_state(self):
        super()._reset_state()
        self.last_prediction = None
        self.last_prediction_time = 0.0

    def run_step(self):
        if not self.is_running or not self.model_loaded:
            return
        if self.my_sensor.n_row < 2 or self.my_sensor.n_col < 2:
            return

        # Apply any inference that finished since the previous tick before
        # deciding what to do with the current frame.
        if self.inference_in_background:
            self._collect_pending_prediction()

        data_obj = getattr(self.my_sensor, "_data", None)
        sensor_frame_sequence = getattr(data_obj, "frame_sequence", None)
        perf_timestamp = time.perf_counter()
        if sensor_frame_sequence is not None:
            if sensor_frame_sequence == self._last_processed_sensor_frame:
                timeout = max(0.0, float(getattr(self, "sensor_frame_timeout_sec", 0.2)))
                if (
                    timeout > 0.0
                    and self.last_robot_velocity_cmd != self._zero_velocity()
                    and (perf_timestamp - self._last_sensor_frame_seen_at) >= timeout
                ):
                    self._discard_pending_predictions()
                    self._apply_prediction(
                        {
                            "mode": "stop",
                            "mode_conf": 1.0,
                            "finger_idx": 0,
                            "finger_conf": 1.0,
                            "velocity_sent": np.zeros(6, dtype=np.float32),
                        }
                    )
                return
            self._last_processed_sensor_frame = sensor_frame_sequence
            self._last_sensor_frame_seen_at = perf_timestamp

        prev_row = self.last_touch_center_row
        prev_col = self.last_touch_center_col
        sensor_snapshot = self._snapshot_sensor_frame()
        self.print_single_touch_map_with_motion(
            threshold=self.motion_threshold,
            update_robot=False,
        )
        frame_tensor, aux_vec = self._build_live_features(sensor_snapshot, prev_row, prev_col)
        self.frame_buffer.append(frame_tensor)
        self.aux_buffer.append(aux_vec)
        if len(self.frame_buffer) > self.seq_len:
            self.frame_buffer.pop(0)
        if len(self.aux_buffer) > self.seq_len:
            self.aux_buffer.pop(0)
        if not bool(getattr(self, "_last_live_touch_present", False)):
            self._discard_pending_predictions()
            self._apply_prediction(
                {
                    "mode": "stop",
                    "mode_conf": 1.0,
                    "finger_idx": 0,
                    "finger_conf": 1.0,
                    "velocity_sent": np.zeros(6, dtype=np.float32),
                }
            )
            return
        if self.inference_in_background:
            # One inference in flight at a time; if the previous one is still
            # running, this frame still entered the buffers, so the next
            # submission covers it.
            if self._pending_inference is None:
                prepared = self._prepare_prediction_window()
                if prepared is not None:
                    future = self._ensure_inference_executor().submit(
                        self._run_model_inference, *prepared
                    )
                    self._pending_inference = (self._inference_epoch, future)
        else:
            prediction = self._predict_from_buffer()
            if prediction is not None:
                self._apply_prediction(prediction)

    def _snapshot_sensor_frame(self):
        data = self.my_sensor._data
        return {
            "diffPerData": column_major_matrix_view(data.diffPerData, dtype=np.float32, copy=True),
            "diffPerDataAve": column_major_matrix_view(data.diffPerDataAve, dtype=np.float32, copy=True),
        }

    def _selected_frame_index(self, frame_name):
        frame_name = str(frame_name).lower()
        if frame_name in {"tool", "tcp", "joint6", "j6"}:
            return 6.0
        if frame_name in {"base", "world", "joint1", "j1"}:
            return 1.0
        match = re.fullmatch(r"(?:joint|j)([1-6])", frame_name)
        if match:
            return float(match.group(1))
        return 0.0

    def _build_live_features(self, sensor_snapshot, prev_row, prev_col):
        channel_tensors = []
        diff = sensor_snapshot["diffPerData"].astype(np.float32)
        diff_ave = sensor_snapshot["diffPerDataAve"].astype(np.float32)
        if "diffPerData" in self.input_channels:
            channel_tensors.append(diff)
        if "diffPerDataAve" in self.input_channels:
            channel_tensors.append(diff_ave)
        if "frameDiff" in self.input_channels:
            if self._prev_diff_for_frame_diff is None:
                frame_diff = np.zeros_like(diff, dtype=np.float32)
            else:
                frame_diff = diff - self._prev_diff_for_frame_diff
            channel_tensors.append(frame_diff.astype(np.float32))
        if "touchMask" in self.input_channels:
            touch_mask = (diff_ave < float(self.motion_threshold)).astype(np.float32)
            channel_tensors.append(touch_mask)
        self._prev_diff_for_frame_diff = diff.copy()

        frame_tensor = np.stack(channel_tensors, axis=0).astype(np.float32)
        feature = self._compute_touch_motion_features(diff_ave, prev_row, prev_col)
        selected_frame_idx = self._selected_frame_index(feature["selected_frame"])
        self._last_live_touch_present = bool(feature["touch_present"] > 0.0)
        aux_map = {
            "center_row": feature["center_row_norm"],
            "center_col": feature["center_col_norm"],
            "delta_row": feature["delta_row"],
            "delta_col": feature["delta_col"],
            "delta_row_norm": feature["delta_row_norm"],
            "delta_col_norm": feature["delta_col_norm"],
            "speed": feature["speed"],
            "peak_value": feature["peak_value"],
            "mean_active_value": feature["mean_active_value"],
            "touch_present": feature["touch_present"],
            "selected_frame_idx": selected_frame_idx,
            "control_frame_idx": selected_frame_idx,
        }
        aux_vec = np.asarray([aux_map.get(name, 0.0) for name in self.aux_feature_names], dtype=np.float32)
        return frame_tensor, aux_vec

    def _clip_velocity6(self, velocity):
        velocity = np.asarray(velocity, dtype=np.float32).reshape(-1)
        if velocity.shape[0] < 6:
            velocity = np.pad(velocity, (0, 6 - velocity.shape[0]), constant_values=0.0)
        velocity = velocity[:6].astype(np.float32)
        if bool(getattr(self, "lock_rotation_axes", True)):
            velocity[3:] = 0.0
        linear_limit = abs(float(getattr(self, "max_linear_speed", 0.02)))
        angular_limit = abs(float(getattr(self, "max_angular_speed", 0.0)))
        velocity[:3] = np.clip(velocity[:3], -linear_limit, linear_limit)
        velocity[3:] = np.clip(velocity[3:], -angular_limit, angular_limit)
        return velocity

    def _prepare_prediction_window(self):
        """Snapshot + normalize the current input window (GUI thread, cheap)."""
        if self.model is None or not self.frame_buffer:
            return None
        if self._torch is None:
            return None

        window = np.stack(self.frame_buffer, axis=0)
        aux_window = np.stack(self.aux_buffer, axis=0)
        if window.shape[0] < self.seq_len:
            pad_len = self.seq_len - window.shape[0]
            window = np.concatenate([np.repeat(window[0:1], pad_len, axis=0), window], axis=0)
            aux_window = np.concatenate([np.repeat(aux_window[0:1], pad_len, axis=0), aux_window], axis=0)
        window = (window - self.scaler_mean[None, :, None, None]) / self.scaler_std[None, :, None, None]
        aux_window = (aux_window - self.aux_mean[None, :]) / self.aux_std[None, :]
        return window, aux_window

    def _predict_from_buffer(self):
        prepared = self._prepare_prediction_window()
        if prepared is None:
            return None
        return self._run_model_inference(*prepared)

    def _run_model_inference(self, window, aux_window):
        """Torch forward pass. Safe to run on a worker thread: it only reads
        model/scaler attributes that stay fixed while execution is running."""
        torch = self._torch

        x = torch.from_numpy(window[None, ...].astype(np.float32)).to(self.device)
        aux = torch.from_numpy(aux_window[None, ...].astype(np.float32)).to(self.device)
        with torch.no_grad():
            out = self.model(x, aux)
            mode_probs = torch.softmax(out["mode_logits"], dim=1)
            mode_idx = int(torch.argmax(mode_probs, dim=1).item())
            mode_conf = float(torch.max(mode_probs).item())
            if self.model_kind == "cnn_gru":
                velocity_norm = out["velocity_norm"][0].detach().cpu().numpy().astype(np.float32)
                target_scale = np.asarray(self.target_scale[: velocity_norm.shape[0]], dtype=np.float32)
                velocity_sent = velocity_norm * target_scale
                finger_idx = 0
                finger_conf = 1.0
            else:
                finger_probs = torch.softmax(out["finger_logits"], dim=1)
                finger_idx = int(torch.argmax(finger_probs, dim=1).item())
                finger_conf = float(torch.max(finger_probs).item())
                velocity_sent = out["velocity"][0].detach().cpu().numpy().astype(np.float32)

        velocity_sent = self._clip_velocity6(velocity_sent * float(self.velocity_scale))
        return {
            "mode": self.INDEX_TO_MODE.get(mode_idx, "stop"),
            "mode_conf": mode_conf,
            "finger_idx": finger_idx,
            "finger_conf": finger_conf,
            "velocity_sent": velocity_sent,
        }

    def _ensure_inference_executor(self):
        if self._inference_executor is None:
            from concurrent.futures import ThreadPoolExecutor

            self._inference_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ai-dfm-infer"
            )
        return self._inference_executor

    def _collect_pending_prediction(self):
        """Apply the result of a finished background inference, if any."""
        pending = self._pending_inference
        if pending is None:
            return
        epoch, future = pending
        if not future.done():
            return
        self._pending_inference = None
        try:
            prediction = future.result()
        except Exception as exc:
            print(f"[AI-DFM] Background inference failed: {exc}")
            return
        # Discard the result if a stop/reset happened after submission —
        # otherwise a stale motion command could override the stop.
        if epoch != self._inference_epoch or prediction is None:
            return
        self._apply_prediction(prediction)

    def _discard_pending_predictions(self):
        self._inference_epoch += 1
        self._pending_inference = None

    def _apply_prediction(self, prediction):
        mode = prediction["mode"]
        mode_conf = float(prediction["mode_conf"])
        finger_idx = int(prediction["finger_idx"])
        velocity_sent = self._clip_velocity6(prediction["velocity_sent"])
        if mode_conf < self.model_conf_threshold or mode == "stop":
            mode = "stop"
            velocity_sent = np.zeros(6, dtype=np.float32)

        velocity6_sent = [float(v) for v in velocity_sent[:6]]
        velocity6_pre_flip = [
            -velocity6_sent[0],
            -velocity6_sent[1],
            -velocity6_sent[2],
            -velocity6_sent[3],
            -velocity6_sent[4],
            -velocity6_sent[5],
        ]
        self._set_teacher_output(mode, velocity6_pre_flip)
        if not self.dry_run_predictions_only:
            self._send_robot_velocity_execution(velocity6_sent)
        self.last_prediction = {
            "mode": mode,
            "mode_conf": mode_conf,
            "finger_idx": finger_idx,
            "velocity_sent": velocity6_sent,
            "dry_run": bool(self.dry_run_predictions_only),
        }
        self.last_prediction_time = time.time()
        self._debug_print(
            f"[AI_EXEC{' DRY' if self.dry_run_predictions_only else ''}] "
            f"mode={mode} ({mode_conf:.3f}) | fingers={finger_idx} | "
            f"velocity_sent={[round(v, 4) for v in velocity6_sent]}"
        )

    def _send_robot_velocity_execution(self, velocity6_sent):
        if self.dry_run_predictions_only:
            return
        velocity6_sent = [float(v) for v in velocity6_sent]
        robot_api = getattr(self.ros_splitter, "robot_api", None)
        if robot_api is None:
            return

        now = time.time()
        is_zero_cmd = all(abs(v) < 1e-9 for v in velocity6_sent[:3]) and all(abs(v) < 1e-9 for v in velocity6_sent[3:])
        if is_zero_cmd:
            if self.last_robot_velocity_cmd == velocity6_sent and (now - self._last_velocity_send_time) < float(
                self.zero_keepalive_sec
            ):
                return
        else:
            if (now - self._last_nonzero_command_time) > float(self.idle_reenable_sec):
                self._velocity_mode_enabled = False
            if self.last_robot_velocity_cmd == velocity6_sent and (now - self._last_velocity_send_time) < 0.05:
                return

        if not self._ensure_robot_velocity_mode():
            return
        frame = self._get_requested_frame()
        if hasattr(robot_api, "send_end_effector_velocity_in_frame"):
            robot_api.send_end_effector_velocity_in_frame(
                velocity6_sent[:3],
                velocity6_sent[3:],
                frame=frame,
                ensure_mode=False,
            )
        else:
            try:
                cmd = robot_api.set_end_effector_velocity_in_frame(velocity6_sent[:3], velocity6_sent[3:], frame=frame)
            except Exception:
                cmd = robot_api.set_end_effector_velocity(velocity6_sent)
            robot_api.send_request(cmd)
        self.last_robot_velocity_cmd = velocity6_sent
        self._last_velocity_send_time = now
        if not is_zero_cmd:
            self._last_nonzero_command_time = now

    def _stop_robot_motion_execution(self, stop_mode=False):
        zero_velocity = self._zero_velocity()
        self._set_teacher_output("stop", zero_velocity)
        self._send_robot_velocity_execution(zero_velocity)
        robot_api = getattr(self.ros_splitter, "robot_api", None)
        if stop_mode and robot_api is not None:
            # Never swallow failures silently here: if the robot cannot leave
            # velocity mode it may keep executing the last velocity command.
            if hasattr(robot_api, "exit_end_effector_velocity_mode"):
                try:
                    robot_api.exit_end_effector_velocity_mode(send_zero=False)
                except Exception as exc:
                    print(f"[AI-DFM] WARNING: failed to exit velocity mode: {exc}")
            else:
                try:
                    robot_api.send_request(robot_api.suspend_end_effector_velocity_mode())
                except Exception as exc:
                    print(f"[AI-DFM] WARNING: failed to suspend velocity mode: {exc}")
                try:
                    robot_api.send_request(robot_api.stop_end_effector_velocity_mode())
                except Exception as exc:
                    print(f"[AI-DFM] WARNING: failed to stop velocity mode: {exc}")
        if stop_mode:
            self._velocity_mode_enabled = False
            self.last_robot_velocity_cmd = None
            self._last_velocity_send_time = 0.0
            self._last_nonzero_command_time = 0.0
