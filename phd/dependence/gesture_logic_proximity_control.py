import json
import os
import time

import numpy as np
from PyQt5.QtCore import QTimer


class ProximityControl:
    RESOURCE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resource"))
    CONFIG_DIR = os.path.join(RESOURCE_ROOT, "config")
    RECORDING_DIR = os.path.join(RESOURCE_ROOT, "proximity_recordings")
    SETTINGS_FILE = os.path.join(CONFIG_DIR, "proximity_control.json")
    SETTING_NAMES = (
        "frame_interval_ms",
        "center_window_size",
        "lateral_speed",
        "normal_speed",
        "centroid_deadband",
        "strength_deadband",
        "max_linear_speed",
        "smoothing_alpha",
        "lost_signal_normal_recovery_frames",
        "lost_signal_normal_speed_ratio",
    )

    # True = only planar tracking (vx/vz); False = strength-based normal (vy) + two-phase.
    _PROXIMITY_PLANAR_ONLY_TEMP = False

    def __init__(self, ros_splitter_instance, my_sensor_instance):
        self.ros_splitter = ros_splitter_instance
        self.my_sensor = my_sensor_instance

        self.is_running = False
        self.control_timer = QTimer()
        self.control_timer.timeout.connect(self.run_step)

        self.frame_interval_ms = 20
        self.center_window_size = 3
        self.center_signal_floor = 0.05
        self.centroid_deadband = 0.04
        self.strength_deadband = 0.08
        self.lateral_speed = 0.2
        self.normal_speed = 0.10
        self.lateral_gain = 3.0
        self.normal_gain = 2.5
        self.max_linear_speed = 0.25
        self.normal_sign = 1.0
        self.no_signal_stop_frames = 20
        self.centroid_threshold_ratio = 0.20
        self.smoothing_alpha = 0.75
        self.lost_signal_normal_recovery_frames = 12
        self.lost_signal_normal_speed_ratio = 1.0
        self.target_geometric_center = True
        # Phase 1: lateral (vx/vz) only until centroid is back on reference; then phase 2 adds vy.
        self.lateral_center_stable_frames = 5

        self._velocity_mode_enabled = False
        self._proximity_phase = "lateral"
        self._lateral_center_stable_count = 0
        self._smoothed_state = None
        self.last_robot_velocity_cmd = None
        self.reference_strength = None
        self.reference_center_row = None
        self.reference_center_col = None
        self.no_signal_frames = 0
        self.settings_path = self.SETTINGS_FILE
        self.is_recording = False
        self._recording_started_at = None
        self._recording_rows = []
        self._recording_signals = []
        self.load_settings_from_file()

    def _append_log(self, message: str):
        if hasattr(self.ros_splitter, "log_display"):
            try:
                if not self.ros_splitter.log_display.isVisible():
                    self.ros_splitter.log_display.setVisible(True)
                    if hasattr(self.ros_splitter, "adjust_splitter_sizes"):
                        self.ros_splitter.adjust_splitter_sizes()
                self.ros_splitter.log_display.append(message)
                return
            except Exception:
                pass
        print(message)

    def toggle_proximity_control(self):
        self.is_running = not self.is_running
        if self.is_running:
            # Starting Proximity Control also teaches the current hover distance.
            # The user should hold the finger at the desired center distance, then
            # press the single Proximity Control button.
            if not self._capture_reference_state():
                self.is_running = False
                return
            self.no_signal_frames = 0
            self._reset_tracking_state()
            self.control_timer.start(int(self.frame_interval_ms))
            mode = "(planar-only TEMP)" if self._PROXIMITY_PLANAR_ONLY_TEMP else "(lateral-first)"
            self._append_log(
                f"[Proximity] Reference captured and STARTED {mode} | "
                f"reference_strength={self.reference_strength:.4f} | "
                f"reference_center=({self.reference_center_row:.2f}, {self.reference_center_col:.2f})"
            )
        else:
            self.control_timer.stop()
            if self.is_recording:
                self.stop_recording(show_viewer=True)
            self._reset_tracking_state()
            self._stop_robot_motion(stop_mode=True)
            self._append_log("[Proximity] STOPPED")

    def teach_proximity_reference(self):
        ok = self._capture_reference_state()
        if ok:
            self._append_log(
                "[Proximity] Reference taught | "
                f"strength={self.reference_strength:.4f} | "
                f"center=({self.reference_center_row:.2f}, {self.reference_center_col:.2f})"
            )
            if self.is_running:
                self._reset_tracking_state()
                if self._PROXIMITY_PLANAR_ONLY_TEMP:
                    self._append_log("[Proximity] Reference updated (planar-only TEMP).")
                else:
                    self._append_log("[Proximity] Re-centering lateral first, then normal tracking.")
        return ok

    def apply_runtime_params(
        self,
        *,
        frame_interval_ms=None,
        center_window_size=None,
        lateral_speed=None,
        normal_speed=None,
        centroid_deadband=None,
        strength_deadband=None,
        max_linear_speed=None,
        smoothing_alpha=None,
        lost_signal_normal_recovery_frames=None,
        lost_signal_normal_speed_ratio=None,
    ):
        if frame_interval_ms is not None:
            self.frame_interval_ms = int(max(10, min(500, int(frame_interval_ms))))
        if center_window_size is not None:
            self.center_window_size = int(max(1, min(15, int(center_window_size))))
        if lateral_speed is not None:
            self.lateral_speed = float(lateral_speed)
        if normal_speed is not None:
            self.normal_speed = float(normal_speed)
        if centroid_deadband is not None:
            self.centroid_deadband = float(max(0.0, centroid_deadband))
        if strength_deadband is not None:
            self.strength_deadband = float(max(0.0, strength_deadband))
        if max_linear_speed is not None:
            self.max_linear_speed = float(max(1e-6, max_linear_speed))
        if smoothing_alpha is not None:
            self.smoothing_alpha = float(np.clip(float(smoothing_alpha), 0.0, 1.0))
        if lost_signal_normal_recovery_frames is not None:
            self.lost_signal_normal_recovery_frames = int(max(0, int(lost_signal_normal_recovery_frames)))
            self.no_signal_stop_frames = max(self.no_signal_stop_frames, self.lost_signal_normal_recovery_frames + 1)
        if lost_signal_normal_speed_ratio is not None:
            self.lost_signal_normal_speed_ratio = float(max(0.0, lost_signal_normal_speed_ratio))

        if self.is_running and self.control_timer.isActive():
            self.control_timer.stop()
            self.control_timer.start(int(self.frame_interval_ms))

    def get_settings(self):
        settings = {"config_version": 2}
        for name in self.SETTING_NAMES:
            settings[name] = getattr(self, name)
        return settings

    def apply_settings(self, settings, save_to_file=False):
        if not isinstance(settings, dict):
            return
        self.apply_runtime_params(
            frame_interval_ms=settings.get("frame_interval_ms"),
            center_window_size=settings.get("center_window_size"),
            lateral_speed=settings.get("lateral_speed"),
            normal_speed=settings.get("normal_speed"),
            centroid_deadband=settings.get("centroid_deadband"),
            strength_deadband=settings.get("strength_deadband"),
            max_linear_speed=settings.get("max_linear_speed"),
            smoothing_alpha=settings.get("smoothing_alpha"),
            lost_signal_normal_recovery_frames=settings.get("lost_signal_normal_recovery_frames"),
            lost_signal_normal_speed_ratio=settings.get("lost_signal_normal_speed_ratio"),
        )
        if save_to_file:
            self.save_settings_to_file()

    def save_settings_to_file(self):
        try:
            os.makedirs(self.CONFIG_DIR, exist_ok=True)
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(self.get_settings(), f, indent=2)
            print(f"Proximity control settings saved: {self.settings_path}")
        except Exception as exc:
            print(f"Failed to save Proximity control settings: {exc}")

    def load_settings_from_file(self):
        if not os.path.exists(self.settings_path):
            return
        try:
            with open(self.settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            config_version = int(settings.get("config_version", 0))
            if config_version < 1:
                return
            if config_version < 2:
                # Older saved configs used slower defaults; keep user tuning for
                # deadbands/speeds where possible, but adopt the faster loop and
                # normal tracking defaults added for rapid pull-away following.
                settings.pop("frame_interval_ms", None)
                settings.pop("normal_speed", None)
            self.apply_settings(settings, save_to_file=False)
            print(f"Proximity control settings loaded: {self.settings_path}")
        except Exception as exc:
            print(f"Failed to load Proximity control settings: {exc}")

    def toggle_recording(self):
        if self.is_recording:
            return self.stop_recording(show_viewer=True)
        return self.start_recording()

    def start_recording(self):
        if not self.is_running:
            self._append_log("[ProximityRecord] Start Proximity Control before recording.")
            return False
        self.is_recording = True
        self._recording_started_at = time.time()
        self._recording_rows = []
        self._recording_signals = []
        self._append_log("[ProximityRecord] STARTED")
        return True

    def stop_recording(self, show_viewer=True):
        if not self.is_recording:
            return None
        self.is_recording = False
        session = self._save_recording()
        self._recording_rows = []
        self._recording_signals = []
        if session is not None and show_viewer:
            viewer = getattr(self.ros_splitter, "open_proximity_recording_viewer", None)
            if callable(viewer):
                try:
                    viewer(session)
                except Exception as exc:
                    self._append_log(f"[ProximityRecord] Viewer failed: {exc}")
        return session

    def _save_recording(self):
        if not self._recording_rows:
            self._append_log("[ProximityRecord] No samples recorded.")
            return None
        try:
            os.makedirs(self.RECORDING_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(self.RECORDING_DIR, f"proximity_record_{stamp}.csv")
            npz_path = os.path.join(self.RECORDING_DIR, f"proximity_record_{stamp}.npz")

            import csv

            headers = list(self._recording_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self._recording_rows)

            signal_stack = np.asarray(self._recording_signals, dtype=np.float32)
            np.savez_compressed(
                npz_path,
                sensor_signal=signal_stack,
                rows_json=json.dumps(self._recording_rows),
                settings_json=json.dumps(self.get_settings()),
            )

            session = {
                "csv_path": csv_path,
                "npz_path": npz_path,
                "rows": list(self._recording_rows),
                "sensor_signal": signal_stack,
            }
            self._append_log(
                f"[ProximityRecord] Saved {len(self._recording_rows)} samples | CSV: {csv_path} | NPZ: {npz_path}"
            )
            return session
        except Exception as exc:
            self._append_log(f"[ProximityRecord] Save failed: {exc}")
            return None

    def _safe_sequence(self, value, length):
        if value is None:
            return [float("nan")] * length
        try:
            seq = list(value)
        except Exception:
            return [float("nan")] * length
        out = []
        for idx in range(length):
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

    def _record_sample(
        self,
        *,
        state=None,
        center_row_error=float("nan"),
        center_col_error=float("nan"),
        strength_error=float("nan"),
        velocity_pre_flip=None,
        valid_signal=False,
    ):
        if not self.is_recording:
            return
        now = time.time()
        t0 = self._recording_started_at or now
        signal = self._get_signal_matrix()
        if signal is None:
            rows = int(getattr(self.my_sensor, "n_row", 1) or 1)
            cols = int(getattr(self.my_sensor, "n_col", 1) or 1)
            signal = np.full((rows, cols), np.nan, dtype=np.float32)
        else:
            signal = np.asarray(signal, dtype=np.float32)

        velocity_pre = self._safe_sequence(velocity_pre_flip, 6)
        velocity_sent = self._safe_sequence(self.last_robot_velocity_cmd, 6)
        tool = self._safe_tool_pose()
        joints = self._safe_joint_positions()
        state = state or {}
        row = {
            "t_sec": float(now - t0),
            "timestamp": float(now),
            "valid_signal": int(bool(valid_signal)),
            "phase": str(self._proximity_phase),
            "frame": str(self._get_requested_frame()),
            "center_row": float(state.get("center_row", np.nan)),
            "center_col": float(state.get("center_col", np.nan)),
            "strength": float(state.get("strength", np.nan)),
            "reference_center_row": float(self.reference_center_row if self.reference_center_row is not None else np.nan),
            "reference_center_col": float(self.reference_center_col if self.reference_center_col is not None else np.nan),
            "reference_strength": float(self.reference_strength if self.reference_strength is not None else np.nan),
            "center_row_error": float(center_row_error),
            "center_col_error": float(center_col_error),
            "strength_error": float(strength_error),
            "sensor_max": float(np.nanmax(signal)),
            "sensor_mean": float(np.nanmean(signal)),
            "cmd_vx_pre": velocity_pre[0],
            "cmd_vy_pre": velocity_pre[1],
            "cmd_vz_pre": velocity_pre[2],
            "cmd_vx_sent": velocity_sent[0],
            "cmd_vy_sent": velocity_sent[1],
            "cmd_vz_sent": velocity_sent[2],
            "tool_x": tool[0],
            "tool_y": tool[1],
            "tool_z": tool[2],
            "tool_qw": tool[3],
            "tool_qx": tool[4],
            "tool_qy": tool[5],
            "tool_qz": tool[6],
        }
        for idx, value in enumerate(joints, start=1):
            row[f"joint_{idx}"] = value
        self._recording_rows.append(row)
        self._recording_signals.append(signal)

    def _reset_tracking_state(self):
        self._proximity_phase = "lateral"
        self._lateral_center_stable_count = 0
        self._smoothed_state = None

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

    def _ensure_robot_velocity_mode(self):
        if not self._velocity_mode_enabled:
            self.ros_splitter.robot_api.send_request(
                self.ros_splitter.robot_api.enable_end_effector_velocity_mode()
            )
            self._velocity_mode_enabled = True

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

    def _zero_velocity(self):
        return [0.0] * 6

    def _stop_robot_motion(self, stop_mode=False):
        self.last_robot_velocity_cmd = None
        self._send_robot_velocity(self._zero_velocity())
        if stop_mode and self._velocity_mode_enabled:
            self.ros_splitter.robot_api.send_request(
                self.ros_splitter.robot_api.stop_end_effector_velocity_mode()
            )
            self._velocity_mode_enabled = False

    def _get_signal_matrix(self):
        data_obj = getattr(self.my_sensor, "_data", None)
        if data_obj is None:
            return None
        signal = getattr(data_obj, "diffDataAve", None)
        if signal is None:
            return None
        signal = np.asarray(signal, dtype=float)
        if signal.size == 0:
            return None
        return np.abs(signal)

    def _extract_center_patch(self, signal):
        rows, cols = signal.shape
        half = max(0, int(self.center_window_size) // 2)
        center_row = rows // 2
        center_col = cols // 2
        row_start = max(0, center_row - half)
        row_end = min(rows, center_row + half + 1)
        col_start = max(0, center_col - half)
        col_end = min(cols, center_col + half + 1)
        return signal[row_start:row_end, col_start:col_end]

    def _measure_proximity_state(self):
        signal = self._get_signal_matrix()
        if signal is None:
            return None

        max_signal = float(np.max(signal))
        if max_signal <= self.center_signal_floor:
            return None

        center_patch = self._extract_center_patch(signal)
        center_strength = float(np.mean(center_patch))

        centroid_floor = max(float(self.center_signal_floor), max_signal * float(self.centroid_threshold_ratio))
        centroid_signal = np.where(signal >= centroid_floor, signal, 0.0)
        total_signal = float(np.sum(centroid_signal))
        if total_signal <= 1e-9:
            return None

        row_coords, col_coords = np.indices(signal.shape)
        center_row = float(np.sum(row_coords * centroid_signal) / total_signal)
        center_col = float(np.sum(col_coords * centroid_signal) / total_signal)

        state = {
            "strength": center_strength,
            "center_row": center_row,
            "center_col": center_col,
            "rows": signal.shape[0],
            "cols": signal.shape[1],
        }
        return self._smooth_proximity_state(state)

    def _smooth_proximity_state(self, state):
        alpha = float(np.clip(self.smoothing_alpha, 0.0, 1.0))
        if self._smoothed_state is None or alpha >= 1.0:
            self._smoothed_state = dict(state)
            return state
        if alpha <= 0.0:
            return dict(self._smoothed_state)

        smoothed = dict(state)
        for key in ("strength", "center_row", "center_col"):
            smoothed[key] = alpha * float(state[key]) + (1.0 - alpha) * float(self._smoothed_state[key])
        self._smoothed_state = smoothed
        return smoothed

    def _capture_reference_state(self):
        self._smoothed_state = None
        state = self._measure_proximity_state()
        if state is None:
            self._append_log(
                "[Proximity] Reference capture failed. "
                "Hold your finger near the sensor center before starting."
            )
            return False

        self.reference_strength = state["strength"]
        if self.target_geometric_center:
            self.reference_center_row = (float(state["rows"]) - 1.0) / 2.0
            self.reference_center_col = (float(state["cols"]) - 1.0) / 2.0
        else:
            self.reference_center_row = state["center_row"]
            self.reference_center_col = state["center_col"]
        return True

    def _scale_error(self, error, deadband, gain):
        error = float(error)
        abs_error = abs(error)
        if abs_error <= deadband:
            return 0.0
        scaled = min(1.0, (abs_error - deadband) * gain)
        return scaled if error > 0.0 else -scaled

    def run_step(self):
        if not self.is_running:
            return

        state = self._measure_proximity_state()
        if state is None:
            self.no_signal_frames += 1
            recovery_velocity = self._zero_velocity()
            if (
                not self._PROXIMITY_PLANAR_ONLY_TEMP
                and self.reference_strength is not None
                and self.no_signal_frames <= int(self.lost_signal_normal_recovery_frames)
            ):
                # If the signal disappears while tracking, the most common cause
                # in this mode is that the finger moved too far away. Keep moving
                # along the same normal direction as a "too weak" strength error
                # for a short time to reacquire the finger instead of stopping.
                recovery_velocity[1] = float(
                    -self.normal_sign
                    * self.normal_speed
                    * self.lost_signal_normal_speed_ratio
                )
                recovery_velocity[1] = float(
                    np.clip(recovery_velocity[1], -self.max_linear_speed, self.max_linear_speed)
                )
            self._send_robot_velocity(recovery_velocity)
            self._record_sample(
                valid_signal=False,
                strength_error=-1.0 if recovery_velocity[1] else float("nan"),
                velocity_pre_flip=recovery_velocity,
            )
            if self.no_signal_frames >= self.no_signal_stop_frames:
                self._stop_robot_motion()
            return

        self.no_signal_frames = 0

        denom_r = max(1.0, float(self.my_sensor.n_row - 1))
        denom_c = max(1.0, float(self.my_sensor.n_col - 1))
        center_row_error = (state["center_row"] - self.reference_center_row) / denom_r
        center_col_error = (state["center_col"] - self.reference_center_col) / denom_c

        vx = -self.lateral_speed * self._scale_error(center_col_error, self.centroid_deadband, self.lateral_gain)
        vz = -self.lateral_speed * self._scale_error(center_row_error, self.centroid_deadband, self.lateral_gain)

        if self._PROXIMITY_PLANAR_ONLY_TEMP:
            # Finger left/right/up/down on the sensor -> planar EE motion only (no vy).
            vy = 0.0
        else:
            strength_error = (
                (state["strength"] - self.reference_strength)
                / max(abs(self.reference_strength), self.center_signal_floor, 1e-6)
            )
            lateral_centered = (
                abs(center_row_error) <= self.centroid_deadband
                and abs(center_col_error) <= self.centroid_deadband
            )
            if self._proximity_phase == "full" and not lateral_centered:
                self._proximity_phase = "lateral"
                self._lateral_center_stable_count = 0
                self._append_log("[Proximity] Lateral drift detected -> normal compensation paused.")

            if self._proximity_phase == "lateral":
                vy = 0.0
                if lateral_centered:
                    self._lateral_center_stable_count += 1
                else:
                    self._lateral_center_stable_count = 0
                if self._lateral_center_stable_count >= int(self.lateral_center_stable_frames):
                    self._proximity_phase = "full"
                    self._lateral_center_stable_count = 0
                    self._append_log("[Proximity] Lateral centered -> normal (in/out) compensation enabled.")
            else:
                vy = self.normal_sign * self.normal_speed * self._scale_error(
                    strength_error,
                    self.strength_deadband,
                    self.normal_gain,
                )

        velocity = [
            float(np.clip(vx, -self.max_linear_speed, self.max_linear_speed)),
            float(np.clip(vy, -self.max_linear_speed, self.max_linear_speed)),
            float(np.clip(vz, -self.max_linear_speed, self.max_linear_speed)),
            0.0,
            0.0,
            0.0,
        ]
        self._send_robot_velocity(velocity)
        self._record_sample(
            state=state,
            center_row_error=center_row_error,
            center_col_error=center_col_error,
            strength_error=strength_error if not self._PROXIMITY_PLANAR_ONLY_TEMP else float("nan"),
            velocity_pre_flip=velocity,
            valid_signal=True,
        )
