"""Registry of all Experiments-sidebar tasks for the main PingLab window.

Each task is implemented as a subclass of :class:`ExperimentTask` and exposes a
small lifecycle:

* ``on_start(host)`` — invoked when the user presses ``Start``. Return ``True``
  to keep the task running (with ``tick_interval_ms``); return ``False`` to
  abort or to mark the task as one-shot (already finished).
* ``on_tick(host)`` — invoked repeatedly by ``host._sidebar_control_timer``
  every ``tick_interval_ms`` milliseconds while the task is active.
* ``on_stop(host, show_result)`` — invoked when the user presses ``Stop`` or
  the task finishes naturally.

``host`` is the :class:`MyMainWindow` instance, so tasks can use:

* ``host.ui_ros``          — the embedded UI / ROS-bridge widget
* ``host._append_sidebar_control_message(msg)`` — write to log + status bar
* ``host._stop_sidebar_control(show_result, message)`` — request a stop
* ``host._show_sensor_capture_result(series)`` — open the capture-result dialog

Adding a new task: subclass :class:`ExperimentTask`, set ``id``, ``label``,
``description`` (and optionally ``tick_interval_ms``), implement the lifecycle
methods, then append a single instance of it to :data:`TASKS`.
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from phd.dependence.paths import resource_path


_DEFAULT_STORAGE_PATH = object()


class ExperimentTask:
    """Base class for Experiments-sidebar tasks."""

    id: str = ""
    label: str = ""
    description: str = ""
    start_button_label: str = "Start"
    # If None, the task is one-shot: ``on_start`` runs once and the timer is
    # not started. Otherwise, the host timer ticks every ``tick_interval_ms``.
    tick_interval_ms: Optional[int] = None

    def on_start(self, host) -> bool:
        """Return True to keep the task active (start the tick timer).

        Return False if the task could not start or is already done (one-shot).
        """
        return True

    def on_tick(self, host) -> None:
        """Called repeatedly by ``host._sidebar_control_timer``."""
        return

    def on_stop(self, host, show_result: bool = True) -> None:
        """Called when the task is stopped (manually or after completion)."""
        return


# ---------------------------------------------------------------------------
# Concrete tasks
# ---------------------------------------------------------------------------


class HelloWorldTask(ExperimentTask):
    """Prints ``hello world`` to the log every second until stopped."""

    id = "hello_world"
    label = "Hello World"
    description = "Every second, print 'hello world' to the log until you stop it."
    tick_interval_ms = 1000

    def on_start(self, host) -> bool:
        host._append_sidebar_control_message("hello world")
        return True

    def on_tick(self, host) -> None:
        host._append_sidebar_control_message("hello world")


class SensorPeakChangeTask(ExperimentTask):
    """Captures 10 s of sensor data and visualises the peak diffDataAve."""

    id = "sensor_peak_change"
    label = "Sensor Peak Change (10s)"
    description = (
        "Capture 10 s of sensor data, then plot the largest diffDataAve change over time."
    )
    tick_interval_ms = 50
    duration_sec: float = 10.0

    def __init__(self) -> None:
        super().__init__()
        self._series: list = []
        self._started_at: Optional[float] = None

    @staticmethod
    def _read_peak(host) -> Optional[float]:
        if host.ui_ros is None:
            return None
        sensor_functions = getattr(host.ui_ros, "sensor_functions", None)
        data_obj = getattr(sensor_functions, "_data", None) if sensor_functions is not None else None
        diff_data_ave = getattr(data_obj, "diffDataAve", None) if data_obj is not None else None
        if diff_data_ave is None:
            return None
        values = np.asarray(diff_data_ave, dtype=float)
        if values.size == 0:
            return None
        return float(np.max(np.abs(values)))

    def on_start(self, host) -> bool:
        peak_value = self._read_peak(host)
        if peak_value is None:
            host._append_sidebar_control_message(
                "Sensor data is not ready. Please build/update the sensor first."
            )
            return False
        self._started_at = time.time()
        self._series = [(0.0, peak_value)]
        host.info_process.setText(f"Control started: {self.label}")
        return True

    def on_tick(self, host) -> None:
        peak_value = self._read_peak(host)
        if peak_value is None:
            host._append_sidebar_control_message("Sensor data is not ready for capture.")
            host._stop_sidebar_control(show_result=False, message="Control stopped")
            return

        elapsed = 0.0
        if self._started_at is not None:
            elapsed = time.time() - self._started_at
        self._series.append((elapsed, peak_value))

        if elapsed >= self.duration_sec:
            host._stop_sidebar_control(
                show_result=True,
                message=f"Sensor capture finished ({len(self._series)} samples)",
            )

    def on_stop(self, host, show_result: bool = True) -> None:
        captured = list(self._series)
        self._started_at = None
        self._series = []
        if show_result and captured:
            host._show_sensor_capture_result(captured)


class CalibrationTask(ExperimentTask):
    """Characterize one tactile taxel against a force-meter reference."""

    id = "calibration"
    label = "Calibration"
    description = (
        "Approach one tactile taxel while recording synchronized sensor and "
        "force-meter values, then optionally return, recalibrate, and repeat."
    )
    start_button_label = "Remember Initial Position"
    tick_interval_ms = 50

    def __init__(
        self,
        storage_path=_DEFAULT_STORAGE_PATH,
        output_directory=_DEFAULT_STORAGE_PATH,
    ) -> None:
        super().__init__()
        if storage_path is _DEFAULT_STORAGE_PATH:
            storage_path = resource_path(
                "config", "calibration_initial_position.json"
            )
        if output_directory is _DEFAULT_STORAGE_PATH:
            output_directory = resource_path("calibration_recordings")
        self.storage_path = (
            Path(storage_path).expanduser() if storage_path is not None else None
        )
        self.output_directory = (
            Path(output_directory).expanduser()
            if output_directory is not None
            else None
        )
        self.initial_position: Optional[dict] = None
        self.last_persistence_error = ""
        self.sensor_rows = 8
        self.sensor_columns = 10
        self.taxel_index = 14
        self.signal_field = "diff_percent_ave"
        self.approach_speed_m_s = 0.0002
        self.contact_threshold_n = 0.10
        self.max_travel_m = 0.020
        self.timeout_sec = 120.0
        self.repeat_count = 1
        self.return_velocity_rad_s = 0.20
        self.return_timeout_sec = 90.0
        self.return_settle_sec = 0.75
        self.return_arrival_tolerance_rad = float(np.radians(0.5))
        self.recalibration_timeout_sec = 45.0
        self.start_position_tolerance_m = 0.003
        self.start_joint_tolerance_rad = float(np.radians(3.0))
        self.force_baseline_duration_sec = 1.0
        self.force_baseline_min_samples = 3
        self.force_sample_stale_sec = 1.0
        self._contact_approach_requested = False
        self._test_active = False
        self._phase = "idle"
        self._started_at: Optional[float] = None
        self._approach_started_at: Optional[float] = None
        self._start_tool_position = None
        self._force_baseline_samples: list[float] = []
        self._force_baseline_n: Optional[float] = None
        self._last_force_sample_at: Optional[float] = None
        self._samples: list[dict] = []
        self._stop_reason = ""
        self._trial_result_saved = False
        self._trial_number = 0
        self._batch_total = 1
        self._batch_id = ""
        self._batch_directory: Optional[Path] = None
        self._batch_results: list[dict] = []
        self._batch_completed = False
        self._return_started_at: Optional[float] = None
        self._return_arrived_at: Optional[float] = None
        self._recalibration_started_at: Optional[float] = None
        self.last_result: Optional[dict] = None
        self.load_initial_position()

    @staticmethod
    def _robot_api(host):
        ui_ros = getattr(host, "ui_ros", None)
        return getattr(ui_ros, "robot_api", None) if ui_ros is not None else None

    @staticmethod
    def _finite_values(values, count: int):
        if values is None:
            return None
        try:
            array = np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return None
        if array.size < int(count) or not np.all(np.isfinite(array[:count])):
            return None
        return [float(value) for value in array[:count]]

    @classmethod
    def _normalize_initial_position(cls, payload):
        if not isinstance(payload, dict):
            return None
        joints = cls._finite_values(payload.get("joints_rad"), 6)
        if joints is None:
            return None
        return {
            "saved_at": str(payload.get("saved_at", "")),
            "joints_rad": joints,
            "tool_position_m": cls._finite_values(
                payload.get("tool_position_m"), 3
            ),
            "tool_quaternion_wxyz": cls._finite_values(
                payload.get("tool_quaternion_wxyz"), 4
            ),
        }

    def load_initial_position(self) -> bool:
        self.last_persistence_error = ""
        if self.storage_path is None or not self.storage_path.exists():
            return True
        try:
            with self.storage_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            initial_position = self._normalize_initial_position(payload)
            if initial_position is None:
                raise ValueError("saved initial position is invalid")
            self.initial_position = initial_position
            return True
        except Exception as exc:
            self.initial_position = None
            self.last_persistence_error = (
                f"Could not load the saved calibration position: {exc}"
            )
            return False

    def save_initial_position(self) -> bool:
        self.last_persistence_error = ""
        if self.storage_path is None or self.initial_position is None:
            return True
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.storage_path.with_suffix(
                self.storage_path.suffix + ".tmp"
            )
            payload = {"version": 1, **self.initial_position}
            with temporary_path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
            os.replace(temporary_path, self.storage_path)
            return True
        except Exception as exc:
            self.last_persistence_error = (
                f"Could not save the calibration initial position: {exc}"
            )
            return False

    def capture_initial_position(self, host) -> bool:
        api = self._robot_api(host)
        if api is None or not hasattr(api, "get_current_positions"):
            host._append_sidebar_control_message(
                "Robot feedback is unavailable; initial position was not saved."
            )
            return False

        try:
            joints = self._finite_values(api.get_current_positions(), 6)
        except Exception:
            joints = None
        if joints is None:
            host._append_sidebar_control_message(
                "Current robot joint feedback is unavailable; initial position "
                "was not saved."
            )
            return False

        tool_position = None
        tool_quaternion = None
        if hasattr(api, "get_current_tool_position"):
            try:
                position, quaternion = api.get_current_tool_position()
                tool_position = self._finite_values(position, 3)
                tool_quaternion = self._finite_values(quaternion, 4)
            except Exception:
                pass

        self.initial_position = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "joints_rad": joints,
            "tool_position_m": tool_position,
            "tool_quaternion_wxyz": tool_quaternion,
        }
        saved = self.save_initial_position()
        joint_degrees = [round(value, 2) for value in np.degrees(joints)]
        message = f"Calibration initial position remembered: J={joint_degrees} deg"
        if tool_position is not None:
            message += f" | TCP={[round(value, 5) for value in tool_position]} m"
        host._append_sidebar_control_message(message)
        if not saved:
            host._append_sidebar_control_message(self.last_persistence_error)
        if hasattr(host, "_refresh_calibration_controls"):
            host._refresh_calibration_controls()
        return saved

    def request_contact_approach(self) -> None:
        self._contact_approach_requested = True

    def _update_progress(self, host, detail: str = "") -> None:
        callback = getattr(host, "_refresh_calibration_progress", None)
        if callable(callback):
            callback(
                current=int(self._trial_number),
                total=int(self._batch_total),
                phase=str(self._phase),
                detail=str(detail),
            )

    @staticmethod
    def _ui_ros(host):
        return getattr(host, "ui_ros", None)

    @classmethod
    def _read_tool_pose(cls, api):
        if api is None or not hasattr(api, "get_current_tool_position"):
            return None
        try:
            position, quaternion = api.get_current_tool_position()
        except Exception:
            return None
        position = cls._finite_values(position, 3)
        quaternion = cls._finite_values(quaternion, 4)
        if position is None or quaternion is None:
            return None
        return position, quaternion

    def _read_force_meter(self, host):
        ui_ros = self._ui_ros(host)
        if ui_ros is None:
            return None
        raw_force = getattr(ui_ros, "_force_meter_last_raw_newtons", None)
        sample_at = getattr(ui_ros, "_force_meter_last_sample_monotonic", None)
        try:
            raw_force = float(raw_force)
            sample_at = float(sample_at)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(raw_force) or not np.isfinite(sample_at):
            return None
        return raw_force, sample_at

    @staticmethod
    def _matrix_value(data_obj, name, row, column):
        matrix = getattr(data_obj, name, None)
        if matrix is None:
            return None
        try:
            value = float(np.asarray(matrix, dtype=float)[int(row), int(column)])
        except (IndexError, TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def _read_taxel(self, host):
        ui_ros = self._ui_ros(host)
        sensor = getattr(ui_ros, "sensor_functions", None) if ui_ros else None
        data_obj = getattr(sensor, "_data", None) if sensor is not None else None
        if sensor is None or data_obj is None:
            return None
        rows = int(getattr(sensor, "n_row", 0) or 0)
        columns = int(getattr(sensor, "n_col", 0) or 0)
        if rows != int(self.sensor_rows) or columns != int(self.sensor_columns):
            return None
        index = int(self.taxel_index)
        if not 0 <= index < rows * columns:
            return None

        # Signal-viewer indices are top-down and column-major. Instantaneous
        # matrices use source rows; averaged matrices are already flipped for
        # the shared display convention by SensorDataBuffer.
        display_row = index % rows
        column = index // rows
        source_row = rows - 1 - display_row
        return {
            "frame_sequence": int(getattr(data_obj, "frame_sequence", 0) or 0),
            "cell_index": index,
            "display_row": display_row,
            "column": column,
            "raw": self._matrix_value(
                data_obj, "rawData", source_row, column
            ),
            "calibration": self._matrix_value(
                data_obj, "calData", source_row, column
            ),
            "diff": self._matrix_value(
                data_obj, "diffData", source_row, column
            ),
            "diff_percent": self._matrix_value(
                data_obj, "diffPerData", source_row, column
            ),
            "raw_ave": self._matrix_value(
                data_obj, "rawDataAve", display_row, column
            ),
            "diff_ave": self._matrix_value(
                data_obj, "diffDataAve", display_row, column
            ),
            "diff_percent_ave": self._matrix_value(
                data_obj, "diffPerDataAve", display_row, column
            ),
        }

    @staticmethod
    def _send_base_z_velocity(api, velocity_m_s: float) -> bool:
        velocity = [0.0, 0.0, float(velocity_m_s)]
        if hasattr(api, "send_end_effector_velocity_in_frame"):
            return bool(
                api.send_end_effector_velocity_in_frame(
                    velocity,
                    [0.0, 0.0, 0.0],
                    frame="base",
                    ensure_mode=True,
                )
            )
        if not hasattr(api, "send_request"):
            return False
        if hasattr(api, "enter_end_effector_velocity_mode"):
            if not api.enter_end_effector_velocity_mode(suspend_existing=True):
                return False
        elif hasattr(api, "enable_end_effector_velocity_mode"):
            api.send_request(api.suspend_end_effector_velocity_mode())
            if not api.send_request(api.enable_end_effector_velocity_mode()):
                return False
        if hasattr(api, "set_end_effector_velocity_in_frame"):
            command = api.set_end_effector_velocity_in_frame(
                velocity, [0.0, 0.0, 0.0], frame="base"
            )
        elif hasattr(api, "set_end_effector_velocity"):
            command = api.set_end_effector_velocity(velocity + [0.0] * 3)
        else:
            return False
        return bool(api.send_request(command))

    @staticmethod
    def _stop_robot(api) -> None:
        if api is None:
            return
        try:
            if hasattr(api, "exit_end_effector_velocity_mode"):
                api.exit_end_effector_velocity_mode(send_zero=True)
            elif hasattr(api, "send_request"):
                if hasattr(api, "set_end_effector_velocity"):
                    api.send_request(api.set_end_effector_velocity([0.0] * 6))
                if hasattr(api, "suspend_end_effector_velocity_mode"):
                    api.send_request(api.suspend_end_effector_velocity_mode())
                if hasattr(api, "stop_end_effector_velocity_mode"):
                    api.send_request(api.stop_end_effector_velocity_mode())
            if hasattr(api, "send_request") and hasattr(
                api, "stop_and_clear_buffer"
            ):
                api.send_request(api.stop_and_clear_buffer())
        except Exception:
            pass

    def _fail_start(self, host, message: str) -> bool:
        self._test_active = False
        self._phase = "idle"
        host._append_sidebar_control_message(message)
        if hasattr(host, "_set_calibration_controls_running"):
            host._set_calibration_controls_running(False)
        return False

    def _begin_trial(self, host, now: float, tool_position) -> None:
        self._phase = "force_baseline"
        self._started_at = float(now)
        self._approach_started_at = None
        self._start_tool_position = list(tool_position)
        self._force_baseline_samples = []
        self._force_baseline_n = None
        self._last_force_sample_at = None
        self._samples = []
        self._stop_reason = ""
        self._trial_result_saved = False
        self._return_started_at = None
        self._return_arrived_at = None
        self._recalibration_started_at = None
        self._update_progress(host, "Measuring force baseline")
        host._append_sidebar_control_message(
            f"Calibration trial {self._trial_number}/{self._batch_total}: "
            "measuring the stationary force baseline. Keep the indenter "
            "clear of the sensor."
        )

    def _start_contact_approach(self, host) -> bool:
        if self.initial_position is None:
            return self._fail_start(
                host,
                "Remember the Calibration initial position before starting the approach.",
            )
        api = self._robot_api(host)
        tool_pose = self._read_tool_pose(api)
        if tool_pose is None:
            return self._fail_start(
                host, "Current robot TCP feedback is unavailable; approach not started."
            )
        try:
            current_joints = self._finite_values(api.get_current_positions(), 6)
        except Exception:
            current_joints = None
        saved_joints = self._finite_values(
            self.initial_position.get("joints_rad"), 6
        )
        saved_tool_position = self._finite_values(
            self.initial_position.get("tool_position_m"), 3
        )
        if current_joints is None or saved_joints is None:
            return self._fail_start(
                host,
                "Current or remembered joint feedback is unavailable. Remember "
                "the initial position again before approaching.",
            )
        joint_error = float(
            np.max(
                np.abs(
                    np.asarray(current_joints, dtype=float)
                    - np.asarray(saved_joints, dtype=float)
                )
            )
        )
        tool_error = (
            float(
                np.linalg.norm(
                    np.asarray(tool_pose[0], dtype=float)
                    - np.asarray(saved_tool_position, dtype=float)
                )
            )
            if saved_tool_position is not None
            else 0.0
        )
        if (
            joint_error > float(self.start_joint_tolerance_rad)
            or tool_error > float(self.start_position_tolerance_m)
        ):
            return self._fail_start(
                host,
                "Robot is not at the remembered Calibration initial position. "
                "Return it safely or remember the current position again.",
            )
        taxel = self._read_taxel(host)
        if taxel is None or int(taxel.get("frame_sequence", 0)) <= 0:
            return self._fail_start(
                host,
                f"A live {self.sensor_rows}x{self.sensor_columns} sensor is required "
                f"before testing taxel {self.taxel_index}.",
            )
        force_reading = self._read_force_meter(host)
        now = time.monotonic()
        if (
            force_reading is None
            or now - float(force_reading[1]) > float(self.force_sample_stale_sec)
        ):
            return self._fail_start(
                host,
                "Fresh HP-200 data is required. Connect the force meter in the Extra tab.",
            )

        sensor = getattr(self._ui_ros(host), "sensor_functions", None)
        if bool(getattr(sensor, "_sensor_calibration_in_progress", False)):
            return self._fail_start(
                host,
                "Wait for the active sensor calibration to finish before "
                "starting the Calibration experiment.",
            )

        self._test_active = True
        self._batch_total = max(1, int(self.repeat_count))
        self._trial_number = 1
        self._batch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._batch_directory = (
            self.output_directory
            / (
                f"calibration_taxel{int(self.taxel_index)}_batch_"
                f"{self._batch_id}"
            )
            if self.output_directory is not None and self._batch_total > 1
            else None
        )
        self._batch_results = []
        self._batch_completed = False
        self.last_result = None
        self._begin_trial(host, now, tool_pose[0])
        if hasattr(host, "_set_calibration_controls_running"):
            host._set_calibration_controls_running(True)
        return True

    def _append_sample(self, now, force_raw_n, taxel, tool_position) -> None:
        if self._started_at is None or self._force_baseline_n is None:
            return
        elapsed = max(0.0, float(now) - float(self._started_at))
        force_delta = float(force_raw_n) - float(self._force_baseline_n)
        start_z = float(self._start_tool_position[2])
        current_z = float(tool_position[2])
        sample = {
            "elapsed_s": elapsed,
            "phase": self._phase,
            "force_raw_n": float(force_raw_n),
            "force_baseline_n": float(self._force_baseline_n),
            "force_delta_n": force_delta,
            "force_delta_abs_n": abs(force_delta),
            "tool_z_m": current_z,
            "travel_down_mm": max(0.0, start_z - current_z) * 1000.0,
            **taxel,
        }
        self._samples.append(sample)

    def _request_stop(self, host, reason: str, message: str) -> None:
        self._stop_reason = str(reason)
        self._stop_robot(self._robot_api(host))
        host._stop_sidebar_control(show_result=True, message=message)

    def _begin_return_to_initial(self, host, now: float) -> bool:
        api = self._robot_api(host)
        saved_joints = self._finite_values(
            (self.initial_position or {}).get("joints_rad"), 6
        )
        if (
            api is None
            or saved_joints is None
            or not hasattr(api, "send_positions_joint_angle")
        ):
            self._request_stop(
                host,
                "return_unavailable",
                "Calibration batch stopped: the robot cannot return to the "
                "remembered initial position.",
            )
            return False
        self._stop_robot(api)
        try:
            sent = bool(
                api.send_positions_joint_angle(
                    saved_joints,
                    velocity=float(self.return_velocity_rad_s),
                    acc_time=0.2,
                    blend_percentage=0,
                    fine_goal=True,
                )
            )
        except Exception as exc:
            self._request_stop(
                host,
                "return_command_failed",
                f"Calibration batch stopped: return command failed ({exc}).",
            )
            return False
        if not sent:
            self._request_stop(
                host,
                "return_command_failed",
                "Calibration batch stopped: the robot rejected the return command.",
            )
            return False
        self._phase = "returning"
        self._stop_reason = ""
        self._samples = []
        self._return_started_at = float(now)
        self._return_arrived_at = None
        self._update_progress(host, "Returning to the initial position")
        host._append_sidebar_control_message(
            f"Calibration trial {self._trial_number}/{self._batch_total} "
            "complete. Returning to the remembered initial position."
        )
        return True

    def _tick_return_to_initial(self, host, now: float) -> None:
        api = self._robot_api(host)
        current_joints = None
        if api is not None and hasattr(api, "get_current_positions"):
            try:
                current_joints = self._finite_values(
                    api.get_current_positions(), 6
                )
            except Exception:
                current_joints = None
        saved_joints = self._finite_values(
            (self.initial_position or {}).get("joints_rad"), 6
        )
        if current_joints is None or saved_joints is None:
            self._request_stop(
                host,
                "return_feedback_unavailable",
                "Calibration batch stopped: joint feedback was unavailable "
                "during the return motion.",
            )
            return
        if (
            self._return_started_at is not None
            and now - float(self._return_started_at)
            >= float(self.return_timeout_sec)
        ):
            self._request_stop(
                host,
                "return_timeout",
                "Calibration batch stopped: returning to the initial position "
                "timed out.",
            )
            return
        joint_error = float(
            np.max(
                np.abs(
                    np.asarray(current_joints, dtype=float)
                    - np.asarray(saved_joints, dtype=float)
                )
            )
        )
        if joint_error > float(self.return_arrival_tolerance_rad):
            self._return_arrived_at = None
            return
        if self._return_arrived_at is None:
            self._return_arrived_at = float(now)
            self._update_progress(host, "Initial position reached; settling")
            return
        if now - float(self._return_arrived_at) < float(self.return_settle_sec):
            return
        self._start_sensor_recalibration(host, now)

    def _start_sensor_recalibration(self, host, now: float) -> None:
        sensor = getattr(self._ui_ros(host), "sensor_functions", None)
        if sensor is None or not hasattr(sensor, "updateCal"):
            self._request_stop(
                host,
                "sensor_calibration_unavailable",
                "Calibration batch stopped: sensor calibration is unavailable.",
            )
            return
        if bool(getattr(sensor, "_sensor_calibration_in_progress", False)):
            self._request_stop(
                host,
                "sensor_calibration_busy",
                "Calibration batch stopped: another sensor calibration is active.",
            )
            return
        self._phase = "recalibrating"
        self._recalibration_started_at = float(now)
        self._update_progress(host, "Updating sensor calibration")
        try:
            sensor.updateCal()
        except Exception as exc:
            self._request_stop(
                host,
                "sensor_calibration_failed",
                f"Calibration batch stopped: sensor calibration failed ({exc}).",
            )
            return
        if not bool(getattr(sensor, "_sensor_calibration_in_progress", False)):
            self._request_stop(
                host,
                "sensor_calibration_failed",
                "Calibration batch stopped: sensor calibration did not start.",
            )
            return
        host._append_sidebar_control_message(
            f"Initial position reached. Updating the sensor calibration before "
            f"trial {self._trial_number + 1}/{self._batch_total}."
        )

    def _tick_sensor_recalibration(self, host, now: float) -> None:
        sensor = getattr(self._ui_ros(host), "sensor_functions", None)
        if sensor is None:
            self._request_stop(
                host,
                "sensor_calibration_unavailable",
                "Calibration batch stopped: sensor feedback became unavailable.",
            )
            return
        if (
            self._recalibration_started_at is not None
            and now - float(self._recalibration_started_at)
            >= float(self.recalibration_timeout_sec)
        ):
            self._request_stop(
                host,
                "sensor_calibration_timeout",
                "Calibration batch stopped: sensor calibration timed out.",
            )
            return
        if bool(getattr(sensor, "_sensor_calibration_in_progress", False)):
            return
        if not bool(getattr(sensor, "is_connected", False)):
            detail = ""
            error_getter = getattr(sensor, "get_last_sensor_stream_error", None)
            if callable(error_getter):
                detail = str(error_getter() or "")
            message = "Calibration batch stopped: sensor calibration failed."
            if detail:
                message += f" {detail}"
            self._request_stop(host, "sensor_calibration_failed", message)
            return
        tool_pose = self._read_tool_pose(self._robot_api(host))
        if tool_pose is None:
            self._request_stop(
                host,
                "feedback_unavailable",
                "Calibration batch stopped: TCP feedback was unavailable after "
                "sensor calibration.",
            )
            return
        self._trial_number += 1
        self._begin_trial(host, now, tool_pose[0])

    def _save_result(self) -> Optional[dict]:
        if not self._samples:
            return None
        rows = list(self._samples)
        trial_number = max(1, int(self._trial_number))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = (
            f"trial_{trial_number:04d}"
            if self._batch_directory is not None
            else f"calibration_taxel{int(self.taxel_index)}_{timestamp}"
        )
        csv_path = None
        metadata_path = None
        if self.output_directory is not None:
            result_directory = self._batch_directory or self.output_directory
            result_directory.mkdir(parents=True, exist_ok=True)
            csv_path = result_directory / f"{stem}.csv"
            columns = list(rows[0].keys())
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            metadata_path = result_directory / f"{stem}.json"
            metadata = {
                "version": 2,
                "batch_id": self._batch_id,
                "trial_number": trial_number,
                "requested_trials": int(self._batch_total),
                "stop_reason": self._stop_reason,
                "sensor_shape": [self.sensor_rows, self.sensor_columns],
                "taxel_index": int(self.taxel_index),
                "signal_field": self.signal_field,
                "approach_speed_m_s": float(self.approach_speed_m_s),
                "contact_threshold_n": float(self.contact_threshold_n),
                "max_travel_m": float(self.max_travel_m),
                "timeout_sec": float(self.timeout_sec),
                "force_baseline_n": self._force_baseline_n,
                "initial_position": self.initial_position,
                "sample_count": len(rows),
                "csv_path": str(csv_path),
            }
            with metadata_path.open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2)
                stream.write("\n")

        result = {
            "rows": rows,
            "csv_path": str(csv_path) if csv_path is not None else "",
            "metadata_path": (
                str(metadata_path) if metadata_path is not None else ""
            ),
            "stop_reason": self._stop_reason,
            "trial_number": trial_number,
            "requested_trials": int(self._batch_total),
            "taxel_index": int(self.taxel_index),
            "signal_field": self.signal_field,
            "contact_threshold_n": float(self.contact_threshold_n),
        }
        result["summary"] = self._summarize_trial(result)
        return result

    @staticmethod
    def _finite_row_values(rows, field: str) -> list[float]:
        values = []
        for row in rows:
            try:
                value = float(row.get(field))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        return values

    def _summarize_trial(self, result: dict) -> dict:
        rows = list(result.get("rows") or [])
        force_values = self._finite_row_values(rows, "force_delta_abs_n")
        travel_values = self._finite_row_values(rows, "travel_down_mm")
        duration_values = self._finite_row_values(rows, "elapsed_s")
        signal_field = str(result.get("signal_field") or self.signal_field)
        signal_values = self._finite_row_values(rows, signal_field)
        return {
            "trial_number": int(result.get("trial_number", 0)),
            "stop_reason": str(result.get("stop_reason", "")),
            "sample_count": len(rows),
            "duration_s": max(duration_values, default=0.0),
            "peak_force_change_n": max(force_values, default=0.0),
            "contact_travel_mm": max(travel_values, default=0.0),
            "signal_min": min(signal_values, default=0.0),
            "signal_max": max(signal_values, default=0.0),
            "signal_peak_abs": max(
                (abs(value) for value in signal_values), default=0.0
            ),
            "signal_final": signal_values[-1] if signal_values else 0.0,
            "csv_path": str(result.get("csv_path", "")),
            "metadata_path": str(result.get("metadata_path", "")),
        }

    def _save_current_trial(self, host) -> Optional[dict]:
        if self._trial_result_saved:
            return self._batch_results[-1] if self._batch_results else None
        result = self._save_result()
        self._trial_result_saved = True
        if result is None:
            return None
        self._batch_results.append(result)
        csv_path = result.get("csv_path", "")
        if csv_path:
            host._append_sidebar_control_message(
                f"Calibration trial {self._trial_number}/{self._batch_total} "
                f"saved: {csv_path}"
            )
        return result

    def _save_batch_summary(self) -> tuple[str, str, str]:
        if not self._batch_results or self.output_directory is None:
            return "", "", ""
        result_directory = self._batch_directory or self.output_directory
        result_directory.mkdir(parents=True, exist_ok=True)
        if self._batch_directory is not None:
            summary_stem = "batch_summary"
        else:
            summary_stem = (
                f"calibration_taxel{int(self.taxel_index)}_"
                f"{self._batch_id}_summary"
            )
        summary_path = result_directory / f"{summary_stem}.csv"
        summary_rows = [
            dict(result.get("summary") or self._summarize_trial(result))
            for result in self._batch_results
        ]
        with summary_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(summary_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        metadata_path = result_directory / f"{summary_stem}.json"
        metadata = {
            "version": 1,
            "batch_id": self._batch_id,
            "status": (
                "complete"
                if self._batch_completed
                and len(self._batch_results) == int(self._batch_total)
                else "aborted"
            ),
            "requested_trials": int(self._batch_total),
            "recorded_trials": len(self._batch_results),
            "sensor_shape": [self.sensor_rows, self.sensor_columns],
            "taxel_index": int(self.taxel_index),
            "signal_field": self.signal_field,
            "approach_speed_m_s": float(self.approach_speed_m_s),
            "contact_threshold_n": float(self.contact_threshold_n),
            "max_travel_m": float(self.max_travel_m),
            "timeout_sec": float(self.timeout_sec),
            "return_velocity_rad_s": float(self.return_velocity_rad_s),
            "initial_position": self.initial_position,
            "summary_csv_path": str(summary_path),
            "trials": summary_rows,
        }
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")

        graph_directory = result_directory / (
            "graphs"
            if self._batch_directory is not None
            else f"{self._batch_id}_graphs"
        )
        return str(summary_path), str(metadata_path), str(graph_directory)

    def _build_final_result(self) -> Optional[dict]:
        if not self._batch_results:
            return None
        summary_path, metadata_path, graph_directory = self._save_batch_summary()
        common = {
            "batch_id": self._batch_id,
            "batch_status": (
                "complete"
                if self._batch_completed
                and len(self._batch_results) == int(self._batch_total)
                else "aborted"
            ),
            "requested_trials": int(self._batch_total),
            "completed_trials": len(self._batch_results),
            "batch_summary_path": summary_path,
            "batch_metadata_path": metadata_path,
            "graph_directory": graph_directory,
        }
        if self._batch_total == 1 and len(self._batch_results) == 1:
            result = dict(self._batch_results[0])
            result.update(common)
            return result
        return {
            **common,
            "taxel_index": int(self.taxel_index),
            "signal_field": self.signal_field,
            "contact_threshold_n": float(self.contact_threshold_n),
            "trial_results": list(self._batch_results),
            "rows": list(self._batch_results[-1].get("rows") or []),
            "stop_reason": str(
                self._batch_results[-1].get("stop_reason", "")
            ),
        }

    def on_start(self, host) -> bool:
        if self._contact_approach_requested:
            self._contact_approach_requested = False
            return self._start_contact_approach(host)
        self.capture_initial_position(host)
        return False

    def on_tick(self, host) -> None:
        if not self._test_active or self._started_at is None:
            return
        now = time.monotonic()
        force_reading = self._read_force_meter(host)
        if (
            force_reading is None
            or now - float(force_reading[1]) > float(self.force_sample_stale_sec)
        ):
            self._request_stop(
                host,
                "force_meter_stale",
                "Calibration stopped: HP-200 data became stale.",
            )
            return
        force_raw_n, force_sample_at = force_reading

        if self._phase == "returning":
            self._tick_return_to_initial(host, now)
            return
        if self._phase == "recalibrating":
            self._tick_sensor_recalibration(host, now)
            return

        if self._phase == "force_baseline":
            if force_sample_at != self._last_force_sample_at:
                self._force_baseline_samples.append(float(force_raw_n))
                self._last_force_sample_at = force_sample_at
            baseline_elapsed = now - self._started_at
            if baseline_elapsed < float(self.force_baseline_duration_sec):
                return
            if len(self._force_baseline_samples) < int(
                self.force_baseline_min_samples
            ):
                if baseline_elapsed < max(
                    3.0, float(self.force_baseline_duration_sec) + 2.0
                ):
                    return
                self._request_stop(
                    host,
                    "insufficient_force_samples",
                    "Calibration stopped: not enough fresh HP-200 baseline samples.",
                )
                return
            self._force_baseline_n = float(
                np.median(self._force_baseline_samples)
            )
            self._phase = "approaching"
            self._approach_started_at = now
            self._update_progress(host, "Approaching the selected taxel")
            host._append_sidebar_control_message(
                f"Force baseline: {self._force_baseline_n:+.4f} N. "
                f"Approaching taxel {self.taxel_index} at "
                f"{self.approach_speed_m_s * 1000.0:.3f} mm/s."
            )

        taxel = self._read_taxel(host)
        tool_pose = self._read_tool_pose(self._robot_api(host))
        if taxel is None or tool_pose is None:
            self._request_stop(
                host,
                "feedback_unavailable",
                "Calibration stopped: sensor or TCP feedback became unavailable.",
            )
            return
        tool_position, _quaternion = tool_pose
        self._append_sample(now, force_raw_n, taxel, tool_position)
        force_delta_abs = abs(float(force_raw_n) - float(self._force_baseline_n))
        travel_down_m = max(
            0.0,
            float(self._start_tool_position[2]) - float(tool_position[2]),
        )

        if force_delta_abs >= float(self.contact_threshold_n):
            self._stop_reason = "contact_threshold"
            self._stop_robot(self._robot_api(host))
            try:
                result = self._save_current_trial(host)
            except Exception as exc:
                self._request_stop(
                    host,
                    "save_failed",
                    f"Calibration batch stopped: trial data could not be saved ({exc}).",
                )
                return
            if result is None:
                self._request_stop(
                    host,
                    "save_failed",
                    "Calibration batch stopped: no trial samples were available to save.",
                )
                return
            if self._trial_number >= self._batch_total:
                self._batch_completed = True
                host._stop_sidebar_control(
                    show_result=True,
                    message=(
                        f"Contact detected at {force_delta_abs:.3f} N; robot "
                        f"stopped. Calibration batch complete: "
                        f"{self._batch_total} trial(s)."
                    ),
                )
                return
            self._begin_return_to_initial(host, now)
            return
        if travel_down_m >= float(self.max_travel_m):
            self._request_stop(
                host,
                "max_travel",
                f"Calibration stopped at the {self.max_travel_m * 1000.0:.1f} mm travel limit.",
            )
            return
        if now - float(self._started_at) >= float(self.timeout_sec):
            self._request_stop(
                host,
                "timeout",
                f"Calibration stopped at the {self.timeout_sec:.1f} s timeout.",
            )
            return
        if not self._send_base_z_velocity(
            self._robot_api(host), -abs(float(self.approach_speed_m_s))
        ):
            self._request_stop(
                host,
                "command_failed",
                "Calibration stopped: the robot rejected the base -Z velocity command.",
            )

    def on_stop(self, host, show_result: bool = True) -> None:
        was_active = self._test_active
        self._stop_robot(self._robot_api(host))
        if was_active and not self._stop_reason:
            self._stop_reason = "manual_stop"
        self._test_active = False
        self._phase = "idle"
        self._approach_started_at = None
        self._update_progress(
            host,
            "Complete" if self._batch_completed else "Stopped",
        )
        if hasattr(host, "_set_calibration_controls_running"):
            host._set_calibration_controls_running(False)
        if not was_active:
            return
        try:
            if self._samples and not self._trial_result_saved:
                self._save_current_trial(host)
            self.last_result = self._build_final_result()
        except Exception as exc:
            self.last_result = None
            host._append_sidebar_control_message(
                f"Could not save calibration recording: {exc}"
            )
        if self.last_result is not None:
            summary_path = self.last_result.get("batch_summary_path", "")
            if summary_path:
                host._append_sidebar_control_message(
                    f"Calibration batch summary saved: {summary_path}"
                )
            if show_result and hasattr(host, "_show_calibration_result"):
                host._show_calibration_result(self.last_result)


class Sigraph2026ScanningTask(ExperimentTask):
    """Record manually taught joint configurations and replay them in order."""

    id = "sigraph2026_scanning"
    label = "Sigraph2026 Scanning"
    description = (
        "Manually position the robot, record ordered taught points, then replay "
        "each point sequentially with measured-feedback arrival checks."
    )
    start_button_label = "Run"
    tick_interval_ms = 50

    def __init__(self, storage_path=_DEFAULT_STORAGE_PATH) -> None:
        super().__init__()
        self.point_sets: list[dict] = [{"name": "Set 1", "points": []}]
        self.active_set_index = 0
        if storage_path is _DEFAULT_STORAGE_PATH:
            storage_path = resource_path(
                "config", "sigraph2026_scanning_points.json"
            )
        self.storage_path = (
            Path(storage_path).expanduser() if storage_path is not None else None
        )
        self.last_persistence_error = ""
        self.velocity_rad_s = 0.2
        self.dwell_sec = 0.5
        self.arrival_tolerance_rad = float(np.radians(0.5))
        self.point_timeout_sec = 45.0
        self._running_index: Optional[int] = None
        self._target_sent_at: Optional[float] = None
        self._arrived_at: Optional[float] = None
        self._completed = False
        self._run_step = 1
        self._reverse_run_requested = False
        self._requested_start_index: Optional[int] = None
        self._run_request_pending = False
        self.load_recorded_points()

    @property
    def recorded_points(self) -> list:
        if not self.point_sets:
            self.point_sets = [{"name": "Set 1", "points": []}]
            self.active_set_index = 0
        self.active_set_index = int(
            np.clip(self.active_set_index, 0, len(self.point_sets) - 1)
        )
        return self.point_sets[self.active_set_index]["points"]

    @recorded_points.setter
    def recorded_points(self, points) -> None:
        if not self.point_sets:
            self.point_sets = [{"name": "Set 1", "points": []}]
            self.active_set_index = 0
        self.point_sets[self.active_set_index]["points"] = list(points)

    @property
    def active_set_name(self) -> str:
        if not self.point_sets:
            return "Set 1"
        return str(self.point_sets[self.active_set_index]["name"])

    def point_set_names(self) -> list[str]:
        return [str(point_set["name"]) for point_set in self.point_sets]

    @staticmethod
    def _finite_values(values, count):
        if values is None:
            return None
        try:
            array = np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return None
        if array.size < int(count) or not np.all(np.isfinite(array[:count])):
            return None
        return [float(value) for value in array[:count]]

    @classmethod
    def _normalize_recorded_point(cls, point):
        if not isinstance(point, dict):
            return None
        joints = cls._finite_values(point.get("joints"), 6)
        if joints is None:
            return None
        return {
            "joints": joints,
            "tool_position": cls._finite_values(point.get("tool_position"), 3),
            "tool_quaternion": cls._finite_values(
                point.get("tool_quaternion"), 4
            ),
        }

    @classmethod
    def _normalize_point_list(cls, raw_points):
        if not isinstance(raw_points, list):
            return []
        return [
            normalized
            for point in raw_points
            if (normalized := cls._normalize_recorded_point(point)) is not None
        ]

    @staticmethod
    def _unique_set_name(name, existing_names, fallback_index):
        candidate = str(name or "").strip() or f"Set {int(fallback_index)}"
        if candidate not in existing_names:
            return candidate
        suffix = 2
        while f"{candidate} ({suffix})" in existing_names:
            suffix += 1
        return f"{candidate} ({suffix})"

    def create_point_set(self, name=None) -> int:
        if self._running_index is not None:
            return -1
        existing_names = set(self.point_set_names())
        number = 1
        while f"Set {number}" in existing_names:
            number += 1
        set_name = self._unique_set_name(
            name or f"Set {number}",
            existing_names,
            len(self.point_sets) + 1,
        )
        self.point_sets.append({"name": set_name, "points": []})
        self.active_set_index = len(self.point_sets) - 1
        self.save_recorded_points()
        return self.active_set_index

    def select_point_set(self, index: int) -> bool:
        index = int(index)
        if (
            self._running_index is not None
            or not 0 <= index < len(self.point_sets)
        ):
            return False
        self.active_set_index = index
        self.save_recorded_points()
        return True

    def load_recorded_points(self) -> bool:
        self.last_persistence_error = ""
        if self.storage_path is None or not self.storage_path.exists():
            return True
        try:
            with self.storage_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            raw_sets = payload.get("sets")
            if isinstance(raw_sets, list):
                point_sets = []
                existing_names = set()
                for index, raw_set in enumerate(raw_sets, start=1):
                    if not isinstance(raw_set, dict):
                        continue
                    set_name = self._unique_set_name(
                        raw_set.get("name"),
                        existing_names,
                        index,
                    )
                    existing_names.add(set_name)
                    point_sets.append(
                        {
                            "name": set_name,
                            "points": self._normalize_point_list(
                                raw_set.get("points", [])
                            ),
                        }
                    )
                self.point_sets = point_sets or [{"name": "Set 1", "points": []}]
                requested_index = payload.get("active_set_index", 0)
                try:
                    requested_index = int(requested_index)
                except (TypeError, ValueError):
                    requested_index = 0
                self.active_set_index = int(
                    np.clip(requested_index, 0, len(self.point_sets) - 1)
                )
            else:
                raw_points = payload.get("points", [])
                if not isinstance(raw_points, list):
                    raise ValueError("the points field is not a list")
                self.point_sets = [
                    {
                        "name": "Set 1",
                        "points": self._normalize_point_list(raw_points),
                    }
                ]
                self.active_set_index = 0
            return True
        except Exception as exc:
            self.last_persistence_error = (
                f"Could not load saved scanning points: {exc}"
            )
            self.point_sets = [{"name": "Set 1", "points": []}]
            self.active_set_index = 0
            return False

    def save_recorded_points(self) -> bool:
        self.last_persistence_error = ""
        if self.storage_path is None:
            return True
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.storage_path.with_suffix(
                self.storage_path.suffix + ".tmp"
            )
            payload = {
                "version": 2,
                "active_set_index": int(self.active_set_index),
                "sets": [
                    {
                        "name": str(point_set["name"]),
                        "points": self._normalize_point_list(
                            point_set.get("points", [])
                        ),
                    }
                    for point_set in self.point_sets
                ],
            }
            with temporary_path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
            os.replace(temporary_path, self.storage_path)
            return True
        except Exception as exc:
            self.last_persistence_error = (
                f"Could not save scanning points: {exc}"
            )
            return False

    @staticmethod
    def _robot_api(host):
        ui_ros = getattr(host, "ui_ros", None)
        return getattr(ui_ros, "robot_api", None) if ui_ros is not None else None

    @staticmethod
    def _finite_joint_positions(api):
        if api is None or not hasattr(api, "get_current_positions"):
            return None
        try:
            positions = np.asarray(api.get_current_positions(), dtype=float).reshape(-1)
        except Exception:
            return None
        if positions.size < 6 or not np.all(np.isfinite(positions[:6])):
            return None
        return np.array(positions[:6], dtype=float, copy=True)

    def capture_current_point(self, host) -> bool:
        if self._running_index is not None:
            host._append_sidebar_control_message(
                "Stop the scanning replay before recording another point."
            )
            return False
        api = self._robot_api(host)
        joints = self._finite_joint_positions(api)
        if joints is None:
            host._append_sidebar_control_message(
                "Current robot joint feedback is unavailable; point not recorded."
            )
            return False

        tool_position = None
        tool_quaternion = None
        if hasattr(api, "get_current_tool_position"):
            try:
                position, quaternion = api.get_current_tool_position()
                if position is not None and quaternion is not None:
                    tool_position = [float(value) for value in position[:3]]
                    tool_quaternion = [float(value) for value in quaternion[:4]]
            except Exception:
                pass
        self.recorded_points.append(
            {
                "joints": joints.tolist(),
                "tool_position": tool_position,
                "tool_quaternion": tool_quaternion,
            }
        )
        saved = self.save_recorded_points()
        point_number = len(self.recorded_points)
        joint_degrees = np.degrees(joints)
        host._append_sidebar_control_message(
            f"Recorded {self.active_set_name} point {point_number}: "
            f"J={[round(value, 2) for value in joint_degrees]} deg"
        )
        if not saved:
            host._append_sidebar_control_message(self.last_persistence_error)
        if hasattr(host, "_refresh_sigraph_scanning_points"):
            host._refresh_sigraph_scanning_points()
        return True

    def remove_point(self, index: int) -> bool:
        index = int(index)
        if self._running_index is not None or not 0 <= index < len(self.recorded_points):
            return False
        del self.recorded_points[index]
        self.save_recorded_points()
        return True

    def copy_point(self, index: int) -> int:
        index = int(index)
        if (
            self._running_index is not None
            or not 0 <= index < len(self.recorded_points)
        ):
            return -1
        copied_point = self._normalize_recorded_point(
            self.recorded_points[index]
        )
        if copied_point is None:
            return -1
        self.recorded_points.append(copied_point)
        self.save_recorded_points()
        return len(self.recorded_points) - 1

    def move_point(self, index: int, offset: int) -> int:
        index = int(index)
        target = index + int(offset)
        if (
            self._running_index is not None
            or not 0 <= index < len(self.recorded_points)
            or not 0 <= target < len(self.recorded_points)
        ):
            return index
        point = self.recorded_points.pop(index)
        self.recorded_points.insert(target, point)
        self.save_recorded_points()
        return target

    def clear_points(self) -> bool:
        if self._running_index is not None:
            return False
        self.recorded_points.clear()
        self.save_recorded_points()
        return True

    def go_to_point(self, host, index: int) -> bool:
        index = int(index)
        if self._running_index is not None:
            host._append_sidebar_control_message(
                "Stop the scanning replay before moving to a selected point."
            )
            return False
        if not 0 <= index < len(self.recorded_points):
            host._append_sidebar_control_message(
                "Select a recorded scanning point first."
            )
            return False

        api = self._robot_api(host)
        if self._finite_joint_positions(api) is None:
            host._append_sidebar_control_message(
                "Robot joint feedback is unavailable; selected point was not sent."
            )
            return False
        if api is None or not hasattr(api, "send_positions_joint_angle"):
            host._append_sidebar_control_message("Robot position API is unavailable.")
            return False
        try:
            sent = bool(
                api.send_positions_joint_angle(
                    list(self.recorded_points[index]["joints"]),
                    velocity=float(self.velocity_rad_s),
                    acc_time=0.2,
                    blend_percentage=0,
                    fine_goal=True,
                )
            )
        except Exception as exc:
            host._append_sidebar_control_message(
                f"Failed to command selected scanning point {index + 1}: {exc}"
            )
            return False
        if not sent:
            host._append_sidebar_control_message(
                f"Robot rejected selected scanning point {index + 1}."
            )
            return False
        host._append_sidebar_control_message(
            f"Moving to selected scanning point {index + 1}."
        )
        return True

    def _send_current_target(self, host) -> bool:
        if self._running_index is None:
            return False
        api = self._robot_api(host)
        if api is None or not hasattr(api, "send_positions_joint_angle"):
            host._append_sidebar_control_message("Robot position API is unavailable.")
            return False
        point = self.recorded_points[self._running_index]
        try:
            sent = bool(
                api.send_positions_joint_angle(
                    list(point["joints"]),
                    velocity=float(self.velocity_rad_s),
                    acc_time=0.2,
                    blend_percentage=0,
                    fine_goal=True,
                )
            )
        except Exception as exc:
            host._append_sidebar_control_message(
                f"Failed to command scanning point {self._running_index + 1}: {exc}"
            )
            return False
        if not sent:
            host._append_sidebar_control_message(
                f"Robot rejected scanning point {self._running_index + 1}."
            )
            return False
        self._target_sent_at = time.monotonic()
        self._arrived_at = None
        host._append_sidebar_control_message(
            f"Moving to scanning point {self._running_index + 1}/"
            f"{len(self.recorded_points)}"
        )
        if hasattr(host, "_refresh_sigraph_scanning_points"):
            host._refresh_sigraph_scanning_points(
                active_index=self._running_index
            )
        return True

    def request_run(self, start_index: int, reverse: bool = False) -> None:
        """Queue a replay beginning at the selected recorded point."""
        self._requested_start_index = int(start_index)
        self._reverse_run_requested = bool(reverse)
        self._run_request_pending = True

    def request_reverse_run(self, start_index: Optional[int] = None) -> None:
        if start_index is None:
            start_index = len(self.recorded_points) - 1
        self.request_run(start_index, reverse=True)

    def on_start(self, host) -> bool:
        reverse = bool(self._reverse_run_requested)
        requested_start_index = self._requested_start_index
        self._reverse_run_requested = False
        self._requested_start_index = None
        self._run_request_pending = False
        if not self.recorded_points:
            host._append_sidebar_control_message(
                "Record at least one scanning point before running."
            )
            return False
        if requested_start_index is None:
            requested_start_index = (
                len(self.recorded_points) - 1 if reverse else 0
            )
        if not 0 <= int(requested_start_index) < len(self.recorded_points):
            host._append_sidebar_control_message(
                "Select a recorded scanning point before running."
            )
            return False
        if self._finite_joint_positions(self._robot_api(host)) is None:
            host._append_sidebar_control_message(
                "Robot joint feedback is unavailable; scanning cannot start."
            )
            return False
        self._run_step = -1 if reverse else 1
        self._running_index = int(requested_start_index)
        self._completed = False
        if hasattr(host, "_set_sigraph_scanning_controls_running"):
            host._set_sigraph_scanning_controls_running(True)
        if not self._send_current_target(host):
            self._running_index = None
            if hasattr(host, "_set_sigraph_scanning_controls_running"):
                host._set_sigraph_scanning_controls_running(False)
            return False
        return True

    def on_tick(self, host) -> None:
        if self._running_index is None:
            return
        now = time.monotonic()
        if (
            self._target_sent_at is not None
            and now - self._target_sent_at > self.point_timeout_sec
        ):
            host._stop_sidebar_control(
                show_result=False,
                message=f"Scanning stopped: point {self._running_index + 1} timed out",
            )
            return

        current = self._finite_joint_positions(self._robot_api(host))
        if current is None:
            return
        target = np.asarray(
            self.recorded_points[self._running_index]["joints"], dtype=float
        )
        if float(np.max(np.abs(current - target))) > self.arrival_tolerance_rad:
            self._arrived_at = None
            return
        if self._arrived_at is None:
            self._arrived_at = now
            host._append_sidebar_control_message(
                f"Reached scanning point {self._running_index + 1}."
            )
            return
        if now - self._arrived_at < float(self.dwell_sec):
            return

        next_index = self._running_index + self._run_step
        if not 0 <= next_index < len(self.recorded_points):
            self._completed = True
            direction = " reverse" if self._run_step < 0 else ""
            host._stop_sidebar_control(
                show_result=True,
                message=(
                    f"Sigraph2026{direction} scanning complete for "
                    f"{self.active_set_name} "
                    f"({len(self.recorded_points)} points)"
                ),
            )
            return
        self._running_index = next_index
        if not self._send_current_target(host):
            host._stop_sidebar_control(
                show_result=False,
                message="Sigraph2026 scanning stopped (command failed)",
            )

    def on_stop(self, host, show_result: bool = True) -> None:
        completed = bool(self._completed)
        api = self._robot_api(host)
        if not completed and api is not None and hasattr(api, "send_request"):
            try:
                api.send_request(api.stop_and_clear_buffer())
            except Exception:
                pass
        self._running_index = None
        self._target_sent_at = None
        self._arrived_at = None
        self._completed = False
        self._run_step = 1
        self._reverse_run_requested = False
        self._requested_start_index = None
        self._run_request_pending = False
        if hasattr(host, "_set_sigraph_scanning_controls_running"):
            host._set_sigraph_scanning_controls_running(False)
        if hasattr(host, "_refresh_sigraph_scanning_points"):
            host._refresh_sigraph_scanning_points()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# The order of this list controls the order in the sidebar.
TASKS: List[ExperimentTask] = [
    HelloWorldTask(),
    SensorPeakChangeTask(),
    Sigraph2026ScanningTask(),
    CalibrationTask(),
]


def task_definitions() -> List[tuple]:
    """Return list of ``(label, id, description)`` tuples for sidebar population."""
    return [(t.label, t.id, t.description) for t in TASKS]


def get_task(task_id: Optional[str]) -> Optional[ExperimentTask]:
    """Look up a task by its id; returns None if not found."""
    if not task_id:
        return None
    for task in TASKS:
        if task.id == task_id:
            return task
    return None
