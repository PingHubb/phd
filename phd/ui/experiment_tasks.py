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

import json
import os
import time
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
