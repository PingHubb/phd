import json
import os
import re
import struct
import time

import numpy as np
import torch
from PyQt5.QtCore import QTimer

from phd.dependence.sensor_layout import (
    column_major_coords,
    column_major_idx,
    column_major_matrix_view,
    flatten_column_major_view,
)
from phd.dependence.transformer import _AI_DFM_CNNTactileTransformerAux


class DirectFingerMotion:
    RESOURCE_ROOT = "/home/ping2/ros2_ws/src/phd/phd/resource"
    CONFIG_DIR = os.path.join(RESOURCE_ROOT, "config")
    AI_DATA_DIR = os.path.join(RESOURCE_ROOT, "ai", "data")
    AI_MODELS_DIR = os.path.join(RESOURCE_ROOT, "ai", "models")
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
        self.last_robot_velocity_cmd = None
        self.last_teacher_velocity_pre_flip = self._zero_velocity()
        self.current_motion_mode = "stop"
        self._smoothed_velocity = [0.0] * 6
        self._in_push_mode = False
        self.loop_hz = 0.0
        self._loop_tick_count = 0
        self._loop_tick_started_at = time.perf_counter()

    def _default_settings(self):
        return {
            "config_version": 1,
            "motion_threshold": -3.0,
            "no_touch_reset_limit": 3,
            "keep_margin": 0.8,
            "robot_speed": 0.05,
            "centroid_deadband": 0.005,
            "centroid_gain": 20.0,
            "min_speed_ratio": 0.20,
            "max_speed_ratio": 1.50,
            "velocity_smoothing_alpha": 0.4,
            "push_value_threshold": -12.0,
            "push_hold_deadband": 0.01,
            "push_hold_frames_required": 2,
            "push_speed": 0.08,
            "push_exit_value_offset": 4.0,
            "pinch_axis_deadband": 0.003,
            "pinch_distance_threshold": 0.02,
            "pinch_midpoint_deadband": 0.5,
            "pinch_frames_required": 1,
            "pull_speed": 0.08,
            "rotation_speed": 0.005,
            "two_finger_swipe_deadband": 0.06,
            "two_finger_swipe_dominance_ratio": 0.1,
            "two_finger_swipe_axis_lock_frames": 3,
            "two_finger_release_grace_frames": 3,
            "frame_interval_ms": 0,
            "debug_output": False,
        }

    def get_settings(self):
        keys = list(self._default_settings().keys())
        return {key: getattr(self, key) for key in keys}

    def apply_settings(self, settings: dict, save_to_file=False):
        defaults = self._default_settings()
        merged = {**defaults, **(settings or {})}

        int_fields = {
            "config_version",
            "no_touch_reset_limit",
            "push_hold_frames_required",
            "pinch_frames_required",
            "two_finger_swipe_axis_lock_frames",
            "two_finger_release_grace_frames",
            "frame_interval_ms",
        }
        bool_fields = {"debug_output"}

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
            self.control_timer.start(int(self.frame_interval_ms))
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

    def print_single_touch_map_with_motion(self, threshold=-3):
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
                self._stop_robot_motion()
                self.last_active_flat_idx = None
                self.last_active_raw_index = None
                self.last_touch_center_row = None
                self.last_touch_center_col = None
            else:
                dprint("Motion: no touch")
                self._stop_robot_motion()

            return

        self.no_touch_frames = 0

        if len(self.current_touch_clusters) >= 2:
            self.two_finger_grace_counter = self.two_finger_release_grace_frames
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
            self.ros_splitter.robot_api.send_request(
                self.ros_splitter.robot_api.enable_end_effector_velocity_mode()
            )
            self._velocity_mode_enabled = True

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

        scaled = abs_delta * self.centroid_gain
        scaled = max(self.min_speed_ratio, scaled)
        scaled = min(self.max_speed_ratio, scaled)
        return scaled if delta_value > 0 else -scaled

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

    def _apply_stop_output(self):
        self._apply_motion_output("stop", self._zero_velocity())

    def _apply_velocity_smoothing(self, target_velocity):
        alpha = max(0.0, min(1.0, self.velocity_smoothing_alpha))
        self._smoothed_velocity = [
            alpha * t + (1.0 - alpha) * s
            for t, s in zip(target_velocity, self._smoothed_velocity)
        ]
        if all(abs(v) < 1e-6 for v in self._smoothed_velocity):
            self._smoothed_velocity = [0.0] * 6
        return list(self._smoothed_velocity)

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
        return [0.0, float(self.push_speed), 0.0, 0.0, 0.0, 0.0]

    def _pull_velocity(self):
        return [0.0, -float(self.pull_speed), 0.0, 0.0, 0.0, 0.0]

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

        scaled_col = 0.0
        scaled_row = 0.0
        if candidate_axis == "vertical":
            scaled_row = self._scaled_axis_component(delta_mid_row)
        elif candidate_axis == "horizontal":
            scaled_col = self._scaled_axis_component(delta_mid_col)

        if scaled_col == 0.0 and scaled_row == 0.0:
            return self._zero_velocity()

        rx = self.rotation_speed * scaled_row
        rz = -self.rotation_speed * scaled_col
        return [0.0, 0.0, 0.0, float(rx), 0.0, float(rz)]

    def _set_teacher_output(self, mode, velocity):
        self.current_motion_mode = str(mode)
        self.last_teacher_velocity_pre_flip = [float(v) for v in velocity]

    def _send_robot_velocity(self, velocity):
        self._ensure_robot_velocity_mode()
        velocity = [float(v) for v in velocity]
        velocity = [-velocity[0], -velocity[1], -velocity[2], velocity[3], velocity[4], velocity[5]]

        if self.last_robot_velocity_cmd == velocity:
            return

        robot_api = getattr(self.ros_splitter, "robot_api", None)
        if robot_api is None:
            return

        frame = self._get_requested_frame()

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
            self.pinch_hold_counter += 1
            return self.pinch_hold_counter >= self.pinch_frames_required

        self.pinch_hold_counter = 0
        return False

    def _update_robot_from_motion(self, previous_center=None, current_center=None):
        if self.current_two_peak_state is not None:
            if self._two_peak_pinch_is_pull():
                self.push_hold_counter = 0
                self._in_push_mode = False
                velocity = self._pull_velocity()
                self._apply_motion_output("pull", velocity)
                return

            self.push_hold_counter = 0
            self._in_push_mode = False
            velocity = self._two_finger_swipe_to_robot_velocity()
            mode = "move" if any(abs(v) > 1e-12 for v in velocity[3:]) else "stop"
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

            push_enter = (
                self.current_touch_peak_value is not None
                and self.current_touch_peak_value <= self.push_value_threshold
                and hold_distance <= self.push_hold_deadband
            )
            push_exit_threshold = self.push_value_threshold + self.push_exit_value_offset
            push_stay = (
                self._in_push_mode
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
            mode = "move" if any(abs(v) > 1e-12 for v in velocity[:3]) else "stop"
            self._apply_motion_output(mode, velocity)
            return

        self.push_hold_counter = 0
        self._in_push_mode = False
        self._two_finger_swipe_axis_lock = None
        self._two_finger_swipe_axis_lock_remaining = 0
        self._apply_stop_output()

    def _stop_robot_motion(self, stop_mode=False):
        self.push_hold_counter = 0
        self.pinch_hold_counter = 0
        self._in_push_mode = False
        self._two_finger_swipe_axis_lock = None
        self._two_finger_swipe_axis_lock_remaining = 0
        self.current_two_peak_state = None
        self.last_two_peak_state = None
        if stop_mode:
            self._smoothed_velocity = [0.0] * 6
        self._apply_stop_output()

        if stop_mode and self._velocity_mode_enabled:
            self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.stop_end_effector_velocity_mode())
            self._velocity_mode_enabled = False


class DirectFingerMotionV2(DirectFingerMotion):
    """
    Two-finger span control.

    Place one finger on the left side and one on the right side of the sensor:
    - both fingers move toward the center (span shrinks) -> pull robot toward user
    - both fingers move away from center (span grows) -> move robot away from user

    For this first version, magnitude is ignored; it sends fixed pull/push speeds.
    """
    SETTINGS_FILE = os.path.join(DirectFingerMotion.CONFIG_DIR, "direct_finger_motion_v2.json")
    PROFILE_DIR = os.path.join(DirectFingerMotion.CONFIG_DIR, "direct_finger_motion_v2_profiles")
    DEFAULT_PROFILE_NAME = "default"

    def __init__(self, ros_splitter_instance, my_sensor_instance):
        self.current_profile_name = self.DEFAULT_PROFILE_NAME
        super().__init__(ros_splitter_instance, my_sensor_instance)
        self.current_motion_mode = "v2_stop"
        self._v2_latched_mode = "v2_stop"
        self._v2_latched_velocity = self._zero_velocity()
        self._v2_last_logged_direction = None
        self._v2_last_log_time = 0.0

    def _profile_slug(self, profile_name):
        name = str(profile_name or self.DEFAULT_PROFILE_NAME).strip().lower()
        name = re.sub(r"[^a-z0-9_-]+", "_", name)
        name = name.strip("_")
        return name or self.DEFAULT_PROFILE_NAME

    def _profile_path(self, profile_name=None):
        profile = self._profile_slug(profile_name or self.current_profile_name)
        return os.path.join(self.PROFILE_DIR, f"{profile}.json")

    def get_current_profile_name(self):
        return self.current_profile_name

    def list_profiles(self):
        profiles = {self.DEFAULT_PROFILE_NAME}
        try:
            if os.path.isdir(self.PROFILE_DIR):
                for filename in os.listdir(self.PROFILE_DIR):
                    if filename.endswith(".json"):
                        profiles.add(os.path.splitext(filename)[0])
        except Exception as exc:
            print(f"Failed to list DFM v2 profiles: {exc}")
        return sorted(profiles)

    def set_profile(self, profile_name, load=True):
        self.current_profile_name = self._profile_slug(profile_name)
        self.settings_path = self._profile_path(self.current_profile_name)
        if load:
            self.apply_settings(self._default_settings(), save_to_file=False)
            self.load_settings_from_file()
        return self.current_profile_name

    def save_settings_to_file(self):
        self.settings_path = self._profile_path(self.current_profile_name)
        super().save_settings_to_file()

    def load_settings_from_file(self):
        self.settings_path = self._profile_path(self.current_profile_name)
        if not os.path.exists(self.settings_path) and self.current_profile_name == self.DEFAULT_PROFILE_NAME:
            # Migrate the previous single-file DFM v2 settings into the default profile.
            self.settings_path = self.SETTINGS_FILE
        super().load_settings_from_file()
        self.settings_path = self._profile_path(self.current_profile_name)

    def _default_settings(self):
        settings = super()._default_settings()
        settings.update(
            {
                "config_version": 2,
                "motion_threshold": -3.0,
                "frame_interval_ms": 0,
                "velocity_smoothing_alpha": 1.0,
                "robot_speed": 0.05,
                "centroid_deadband": 0.003,
                "centroid_gain": 20.0,
                "min_speed_ratio": 0.20,
                "max_speed_ratio": 1.50,
                "pull_speed": 0.12,
                "push_speed": 0.12,
                "pinch_axis_deadband": 0.003,
                "pinch_distance_threshold": 0.02,
                "pinch_midpoint_deadband": 0.5,
                "v2_span_deadband": 0.006,
                "v2_midpoint_deadband": 0.08,
                "v2_planar_span_tolerance": 0.04,
                "v2_planar_dominance_ratio": 1.0,
                "v2_up_down_direction_sign": 1.0,
                "v2_forward_backward_direction_sign": 1.0,
                "v2_rotation_speed": 0.08,
                "v2_rotation_deadband": 0.01,
                "v2_rotation_direction_sign": 1.0,
                "v2_force_lateral_speed": 0.05,
                "v2_force_lateral_deadband": 1.0,
                "v2_force_lateral_center_deadband": 0.08,
                "v2_force_lateral_direction_sign": 1.0,
            }
        )
        return settings

    def toggle_direct_finger_motion_v2(self):
        self.toggle_direct_finger_motion()

    def _reset_state(self):
        super()._reset_state()
        self._v2_latched_mode = "v2_stop"
        self._v2_latched_velocity = self._zero_velocity()
        self._v2_last_logged_direction = None
        self._v2_last_log_time = 0.0

    def _v2_velocity_direction_label(self, velocity):
        labels = []
        axis_names = ("x", "y", "z", "rx", "ry", "rz")
        for axis_name, value in zip(axis_names, velocity):
            value = float(value)
            if abs(value) < 1e-6:
                continue
            sign = "+" if value > 0.0 else "-"
            labels.append(f"{sign}{axis_name}")
        return " ".join(labels) if labels else "stop"

    def _v2_append_motion_log(self, mode, velocity):
        direction = self._v2_velocity_direction_label(velocity)
        now = time.perf_counter()
        should_log = (
            direction != self._v2_last_logged_direction
            or (direction != "stop" and (now - self._v2_last_log_time) >= 0.5)
        )
        if not should_log:
            return

        self._v2_last_logged_direction = direction
        self._v2_last_log_time = now
        frame = self._get_requested_frame()
        message = f"[DFM V2] moving: {direction} | mode={mode} | frame={frame}"

        log_display = getattr(self.ros_splitter, "log_display", None)
        if log_display is not None:
            try:
                log_display.append(message)
                return
            except Exception:
                pass
        print(message)

    def _apply_motion_output(self, mode, velocity):
        super()._apply_motion_output(mode, velocity)
        self._v2_append_motion_log(mode, velocity)

    def _v2_side_press_velocity(self, two_peak_state=None, held_side=None):
        velocity = self._zero_velocity()

        if two_peak_state is not None:
            if held_side not in {"left", "right"}:
                return velocity
            force_key = "left_force" if held_side == "left" else "right_force"
            force = float(two_peak_state.get(force_key, 0.0))
            if force < float(self.v2_force_lateral_deadband):
                return velocity
            side_sign = -1.0 if held_side == "left" else 1.0
        else:
            if not self.current_touch_clusters:
                return velocity
            strongest = self.current_touch_clusters[0]
            force = max(0.0, float(self.motion_threshold) - float(strongest["peak_value"]))
            if force < float(self.v2_force_lateral_deadband):
                return velocity

            denom_c = max(1.0, float(self.my_sensor.n_col - 1))
            side_delta = (float(strongest["center_col"]) / denom_c) - 0.5
            if abs(side_delta) < float(self.v2_force_lateral_center_deadband):
                return velocity
            side_sign = -1.0 if side_delta < 0.0 else 1.0

        velocity[0] = (
            float(self.v2_force_lateral_direction_sign)
            * float(self.v2_force_lateral_speed)
            * side_sign
        )
        return velocity

    def _update_robot_from_motion(self, previous_center=None, current_center=None):
        curr = self.current_two_peak_state
        prev = self.last_two_peak_state

        self.push_hold_counter = 0
        self.pinch_hold_counter = 0
        self._in_push_mode = False
        self._two_finger_swipe_axis_lock = None
        self._two_finger_swipe_axis_lock_remaining = 0

        if curr is None:
            force_lateral_velocity = self._v2_side_press_velocity()
            if any(abs(v) > 1e-6 for v in force_lateral_velocity):
                self._v2_latched_mode = "v2_single_side_press_move"
                self._v2_latched_velocity = force_lateral_velocity
                self._apply_motion_output(self._v2_latched_mode, self._v2_latched_velocity)
            else:
                self._v2_latched_mode = "v2_stop"
                self._v2_latched_velocity = self._zero_velocity()
                self._apply_stop_output()
            return

        if prev is None:
            self._apply_motion_output(self._v2_latched_mode, self._v2_latched_velocity)
            return

        left_move = curr["left_col"] - prev["left_col"]
        right_move = curr["right_col"] - prev["right_col"]
        left_row_move = curr["left_row"] - prev["left_row"]
        right_row_move = curr["right_row"] - prev["right_row"]
        span_delta = curr["span"] - prev["span"]
        mid_col_delta = curr["mid_col"] - prev["mid_col"]
        mid_row_delta = curr["mid_row"] - prev["mid_row"]
        midpoint_shift_col = abs(curr["mid_col"] - prev["mid_col"])
        midpoint_shift_row = abs(curr["mid_row"] - prev["mid_row"])

        midpoint_stable = (
            midpoint_shift_col <= float(self.v2_midpoint_deadband)
            and midpoint_shift_row <= float(self.v2_midpoint_deadband)
        )
        span_deadband = float(self.v2_span_deadband)
        span_abs = abs(float(span_delta))
        planar_axis_delta = max(abs(float(mid_col_delta)), abs(float(mid_row_delta)))
        planar_requested = (
            planar_axis_delta >= float(self.centroid_deadband)
            and (
                span_abs <= float(self.v2_planar_span_tolerance)
                or planar_axis_delta >= span_abs * float(self.v2_planar_dominance_ratio)
            )
        )

        edge_swipe_y_positive = span_delta >= span_deadband
        edge_swipe_y_negative = span_delta <= -span_deadband

        planar_velocity = self._zero_velocity()
        # Temporarily disable center-motion planar control for DFM v2 testing.
        # X is controlled by side press/hold; Y is controlled by edge swipe.

        normal_velocity = self._zero_velocity()
        if edge_swipe_y_positive:
            normal_velocity = self._push_velocity()
        elif edge_swipe_y_negative:
            normal_velocity = self._pull_velocity()
        normal_velocity[1] *= float(self.v2_forward_backward_direction_sign)

        rotation_velocity = self._zero_velocity()
        # Temporarily disable cylinder rotation for DFM v2 testing.

        side_hold_deadband = max(float(self.v2_span_deadband), float(self.centroid_deadband))
        swipe_active_deadband = float(self.v2_span_deadband)
        left_holding = abs(float(left_move)) <= side_hold_deadband
        right_holding = abs(float(right_move)) <= side_hold_deadband
        left_swiping_left = left_move <= -swipe_active_deadband
        left_swiping_right = left_move >= swipe_active_deadband
        right_swiping_left = right_move <= -swipe_active_deadband
        right_swiping_right = right_move >= swipe_active_deadband
        pure_y_gesture = (
            (left_swiping_left and right_swiping_right)
            or (left_swiping_right and right_swiping_left)
        )
        held_side = None
        if (edge_swipe_y_positive or edge_swipe_y_negative) and not pure_y_gesture:
            if left_holding and not right_holding:
                held_side = "left"
            elif right_holding and not left_holding:
                held_side = "right"

        force_lateral_velocity = self._v2_side_press_velocity(curr, held_side=held_side)

        mixed_velocity = [
            float(planar_velocity[i] + normal_velocity[i] + rotation_velocity[i] + force_lateral_velocity[i])
            for i in range(6)
        ]

        if any(abs(v) > 1e-6 for v in mixed_velocity):
            has_planar = any(abs(v) > 1e-6 for v in planar_velocity)
            has_normal = any(abs(v) > 1e-6 for v in normal_velocity)
            has_rotation = any(abs(v) > 1e-6 for v in rotation_velocity)
            has_force_lateral = any(abs(v) > 1e-6 for v in force_lateral_velocity)
            if sum([has_planar, has_normal, has_rotation, has_force_lateral]) > 1:
                self._v2_latched_mode = "v2_mixed_motion"
            elif has_planar:
                self._v2_latched_mode = "v2_two_finger_planar_move"
            elif has_rotation:
                self._v2_latched_mode = "v2_cylinder_rotation"
            elif has_force_lateral:
                self._v2_latched_mode = "v2_force_lateral_move"
            elif edge_swipe_y_negative:
                self._v2_latched_mode = "v2_pull_toward_user"
            else:
                self._v2_latched_mode = "v2_move_away_user"
            self._v2_latched_velocity = mixed_velocity
            self._apply_motion_output(self._v2_latched_mode, self._v2_latched_velocity)
            return

        self._apply_motion_output(self._v2_latched_mode, self._v2_latched_velocity)


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

    def toggle_console_control(self):
        self._toggle_console_control_mode("ps5")

    def toggle_console_control_sensor(self):
        self._toggle_console_control_mode("sensor")

    def _toggle_console_control_mode(self, source):
        source = str(source).strip().lower()
        if source not in {"ps5", "sensor"}:
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
        else:
            self._append_console_log(
                f"[Console Control] STARTED (SENSOR) | frame={self._get_requested_frame()}"
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
        # Top two sensor rows (0,1): only L1 / R1; virtual sticks never read those rows
        # so L1/R1 presses cannot be mistaken for stick center.
        top_button_row_max = 1
        stick_row_min = min(2, n_row)

        threshold = float(self.motion_threshold)
        touched = values < threshold
        top_two = rows <= top_button_row_max
        l1_mask = top_two & (cols >= left_min) & (cols <= left_max) & touched
        r1_mask = top_two & (cols >= right_min) & (cols <= right_max) & touched
        l1 = 1.0 if np.any(l1_mask) else 0.0
        r1 = 1.0 if np.any(r1_mask) else 0.0

        left_stick_mask = (cols >= left_min) & (cols <= left_max) & (rows >= stick_row_min) & touched
        right_stick_mask = (cols >= right_min) & (cols <= right_max) & (rows >= stick_row_min) & touched

        lx, ly = self._sensor_anchored_stick_axes(
            "left", values, cols, rows, left_stick_mask, left_min, left_max, n_row
        )
        rx, ry = self._sensor_anchored_stick_axes(
            "right", values, cols, rows, right_stick_mask, right_min, right_max, n_row
        )
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

    def run_step(self):
        if not self.is_running:
            return
        self._record_loop_tick()
        if self.console_input_source == "sensor":
            lx, ly, rx, ry, l1, r1, l2, r2 = self._read_console_sensor_inputs()
            if (l1 > 0.5) or (r1 > 0.5):
                # L1/R1 in sensor mode are reserved for pure Z motion.
                lx, ly, rx, ry = 0.0, 0.0, 0.0, 0.0
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

        velocity = [
            float(self.console_x_sign) * float(self.console_x_speed) * lx,
            float(self.console_y_sign) * float(self.console_y_speed) * ly,
            float(self.console_z_sign) * float(self.console_z_speed) * (r1 - l1),
            float(self.console_rx_sign) * float(self.console_rx_speed) * ry,
            float(self.console_ry_sign) * float(self.console_ry_speed) * rx,
            float(self.console_rz_sign) * float(self.console_rz_speed) * (r2 - l2),
        ]

        if any(abs(v) > 1e-6 for v in velocity):
            self._apply_motion_output("console_control", velocity)
        else:
            self._apply_stop_output()
        self._append_console_motion_log(velocity)


class AI_DirectFingerMotion(DirectFingerMotion):
    def __init__(self, ros_splitter_instance, my_sensor_instance):
        super().__init__(ros_splitter_instance, my_sensor_instance)

        self.dataset_root = os.path.join(self.AI_DATA_DIR, "ai_direct_finger_motion")
        self.min_frames_to_save = 5
        self.session_tag = "default"
        self.trial_number = None
        self.episode_started_at = None
        self.current_episode = self._create_empty_episode()

    def _create_empty_episode(self):
        return {
            "timestamps": [],
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
            "selected_frame": [],
        }

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

    def toggle_ai_direct_finger_motion(self, session_tag=None):
        self.is_running = not self.is_running

        if self.is_running:
            self.session_tag = self.sanitize_session_tag(session_tag)
            self.trial_number = self._get_next_trial_number()
            self.episode_started_at = time.time()
            self.current_episode = self._create_empty_episode()
            self._reset_state()
            self.control_timer.start(int(self.frame_interval_ms))
            print(
                f"AI direct finger motion STARTED | session='{self.session_tag}' | trial={self.trial_number}"
            )
        else:
            self.control_timer.stop()
            self._stop_robot_motion(stop_mode=True)
            self._save_episode()
            self._reset_state()
            print("AI direct finger motion STOPPED")

    def run_step(self):
        if not self.is_running:
            return
        if self.my_sensor.n_row < 2 or self.my_sensor.n_col < 2:
            return

        prev_row = self.last_touch_center_row
        prev_col = self.last_touch_center_col
        timestamp = time.time()
        sensor_snapshot = self._snapshot_sensor_frame()

        self.print_single_touch_map_with_motion(threshold=self.motion_threshold)
        self._append_teacher_frame(timestamp, sensor_snapshot, prev_row, prev_col)

    def _snapshot_sensor_frame(self):
        data = self.my_sensor._data
        return {
            "rawData": column_major_matrix_view(data.rawData, dtype=np.float32, copy=True),
            "diffData": column_major_matrix_view(data.diffData, dtype=np.float32, copy=True),
            "diffPerData": column_major_matrix_view(data.diffPerData, dtype=np.float32, copy=True),
            "diffDataAve": column_major_matrix_view(data.diffDataAve, dtype=np.float32, copy=True),
            "diffPerDataAve": column_major_matrix_view(data.diffPerDataAve, dtype=np.float32, copy=True),
        }

    def _append_teacher_frame(self, timestamp, sensor_snapshot, prev_row, prev_col):
        episode = self.current_episode
        values = sensor_snapshot["diffPerDataAve"]
        feature = self._compute_touch_motion_features(values, prev_row, prev_col)

        episode["timestamps"].append(float(timestamp))
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
        episode["teacher_velocity_pre_flip"].append(np.array(self.last_teacher_velocity_pre_flip, dtype=np.float32))
        episode["teacher_velocity_sent"].append(
            np.array(self.last_robot_velocity_cmd or self._zero_velocity(), dtype=np.float32)
        )
        episode["selected_frame"].append(feature["selected_frame"])

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
            "session_tag": self.session_tag,
            "trial_number": int(self.trial_number or 0),
            "frame_count": int(frame_count),
            "started_at": float(self.episode_started_at or time.time()),
            "saved_at": float(time.time()),
            "sensor_rows": int(self.my_sensor.n_row),
            "sensor_cols": int(self.my_sensor.n_col),
            "motion_threshold": float(self.motion_threshold),
            "teacher_layout": "column_major_matrix_view",
        }

        episode = self.current_episode
        np.savez_compressed(
            filename,
            metadata_json=np.array(json.dumps(metadata)),
            timestamps=np.asarray(episode["timestamps"], dtype=np.float64),
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
            mode=np.asarray(episode["mode"], dtype="U16"),
            teacher_velocity_pre_flip=np.stack(episode["teacher_velocity_pre_flip"], axis=0),
            teacher_velocity_sent=np.stack(episode["teacher_velocity_sent"], axis=0),
            selected_frame=np.asarray(episode["selected_frame"], dtype="U16"),
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
        "best_model.pt",
    )

    def __init__(self, ros_splitter_instance, my_sensor_instance):
        super().__init__(ros_splitter_instance, my_sensor_instance)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_loaded = False
        self.model_checkpoint_path = self.DEFAULT_MODEL_CHECKPOINT
        self.model_conf_threshold = 0.55
        self.velocity_scale = 1.0
        self.max_linear_speed = 0.20
        self.prediction_interval_ms = 30
        self.seq_len = 20
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
            "selected_frame_idx",
        ]
        self.scaler_mean = np.zeros(len(self.input_channels), dtype=np.float32)
        self.scaler_std = np.ones(len(self.input_channels), dtype=np.float32)
        self.aux_mean = np.zeros(len(self.aux_feature_names), dtype=np.float32)
        self.aux_std = np.ones(len(self.aux_feature_names), dtype=np.float32)
        self.frame_buffer = []
        self.aux_buffer = []
        self.last_prediction = None
        self.last_prediction_time = 0.0
        self._prev_diff_for_frame_diff = None
        self.zero_keepalive_sec = 0.5
        self.idle_reenable_sec = 1.0
        self._reset_execution_runtime_state()

    def _reset_execution_buffers(self):
        self.frame_buffer = []
        self.aux_buffer = []
        self._prev_diff_for_frame_diff = None

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
            self._ensure_robot_velocity_mode()
            self.control_timer.start(self.prediction_interval_ms)
            print(f"AI direct finger motion execution STARTED | model={self.model_checkpoint_path}")
        else:
            self.control_timer.stop()
            self._stop_robot_motion_execution(stop_mode=True)
            self._reset_state()
            self._reset_execution_buffers()
            self._reset_execution_runtime_state()
            print("AI direct finger motion execution STOPPED")

    def load_model(self, checkpoint_path=None):
        checkpoint_path = checkpoint_path or self.model_checkpoint_path
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            config = checkpoint.get("config", {})
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
            state = checkpoint.get("model_state", checkpoint)
            self.model.load_state_dict(state)
            self.model.eval()
            self.model_loaded = True
            print(
                f"AI_DirectFingerMotion_execution model loaded | seq_len={self.seq_len} | "
                f"channels={self.input_channels} | aux={self.aux_feature_names} | device={self.device}"
            )
        except Exception as exc:
            self.model_loaded = False
            self.model = None
            print(f"Failed to load AI direct finger motion model: {exc}")

    def _reset_state(self):
        super()._reset_state()
        self.last_prediction = None
        self.last_prediction_time = 0.0

    def run_step(self):
        if not self.is_running or not self.model_loaded:
            return
        if self.my_sensor.n_row < 2 or self.my_sensor.n_col < 2:
            return

        prev_row = self.last_touch_center_row
        prev_col = self.last_touch_center_col
        sensor_snapshot = self._snapshot_sensor_frame()
        self.print_single_touch_map_with_motion(threshold=self.motion_threshold)
        frame_tensor, aux_vec = self._build_live_features(sensor_snapshot, prev_row, prev_col)
        self.frame_buffer.append(frame_tensor)
        self.aux_buffer.append(aux_vec)
        if len(self.frame_buffer) > self.seq_len:
            self.frame_buffer.pop(0)
        if len(self.aux_buffer) > self.seq_len:
            self.aux_buffer.pop(0)
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
        if "tool" in frame_name:
            return 1.0
        if "base" in frame_name:
            return 2.0
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
        }
        aux_vec = np.asarray([aux_map.get(name, 0.0) for name in self.aux_feature_names], dtype=np.float32)
        return frame_tensor, aux_vec

    def _predict_from_buffer(self):
        if self.model is None or not self.frame_buffer:
            return None

        window = np.stack(self.frame_buffer, axis=0)
        aux_window = np.stack(self.aux_buffer, axis=0)
        if window.shape[0] < self.seq_len:
            pad_len = self.seq_len - window.shape[0]
            window = np.concatenate([np.repeat(window[0:1], pad_len, axis=0), window], axis=0)
            aux_window = np.concatenate([np.repeat(aux_window[0:1], pad_len, axis=0), aux_window], axis=0)
        window = (window - self.scaler_mean[None, :, None, None]) / self.scaler_std[None, :, None, None]
        aux_window = (aux_window - self.aux_mean[None, :]) / self.aux_std[None, :]

        x = torch.from_numpy(window[None, ...].astype(np.float32)).to(self.device)
        aux = torch.from_numpy(aux_window[None, ...].astype(np.float32)).to(self.device)
        with torch.no_grad():
            out = self.model(x, aux)
            mode_probs = torch.softmax(out["mode_logits"], dim=1)
            finger_probs = torch.softmax(out["finger_logits"], dim=1)
            mode_idx = int(torch.argmax(mode_probs, dim=1).item())
            finger_idx = int(torch.argmax(finger_probs, dim=1).item())
            mode_conf = float(torch.max(mode_probs).item())
            finger_conf = float(torch.max(finger_probs).item())
            velocity_sent = out["velocity"][0].detach().cpu().numpy().astype(np.float32)

        velocity_sent = np.clip(
            velocity_sent * float(self.velocity_scale),
            -float(self.max_linear_speed),
            float(self.max_linear_speed),
        )
        return {
            "mode": self.INDEX_TO_MODE.get(mode_idx, "stop"),
            "mode_conf": mode_conf,
            "finger_idx": finger_idx,
            "finger_conf": finger_conf,
            "velocity_sent": velocity_sent,
        }

    def _apply_prediction(self, prediction):
        mode = prediction["mode"]
        mode_conf = float(prediction["mode_conf"])
        finger_idx = int(prediction["finger_idx"])
        velocity_sent = prediction["velocity_sent"]
        if mode_conf < self.model_conf_threshold or mode == "stop":
            mode = "stop"
            velocity_sent = np.zeros(3, dtype=np.float32)

        velocity6_sent = [float(velocity_sent[0]), float(velocity_sent[1]), float(velocity_sent[2]), 0.0, 0.0, 0.0]
        velocity6_pre_flip = [-velocity6_sent[0], -velocity6_sent[1], -velocity6_sent[2], 0.0, 0.0, 0.0]
        self._set_teacher_output(mode, velocity6_pre_flip)
        self._send_robot_velocity_execution(velocity6_sent)
        self.last_prediction = {
            "mode": mode,
            "mode_conf": mode_conf,
            "finger_idx": finger_idx,
            "velocity_sent": velocity6_sent,
        }
        self.last_prediction_time = time.time()
        self._debug_print(
            f"[AI_EXEC] mode={mode} ({mode_conf:.3f}) | fingers={finger_idx} | "
            f"velocity_sent={[round(v, 4) for v in velocity6_sent[:3]]}"
        )

    def _send_robot_velocity_execution(self, velocity6_sent):
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

        self._ensure_robot_velocity_mode()
        frame = self._get_requested_frame()
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
            try:
                robot_api.send_request(robot_api.suspend_end_effector_velocity_mode())
            except Exception:
                pass
            try:
                robot_api.send_request(robot_api.stop_end_effector_velocity_mode())
            except Exception:
                pass
        if stop_mode:
            self._velocity_mode_enabled = False
            self.last_robot_velocity_cmd = None
            self._last_velocity_send_time = 0.0
            self._last_nonzero_command_time = 0.0
