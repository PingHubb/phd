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

import time
from typing import List, Optional

import numpy as np


class ExperimentTask:
    """Base class for Experiments-sidebar tasks."""

    id: str = ""
    label: str = ""
    description: str = ""
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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# The order of this list controls the order in the sidebar.
TASKS: List[ExperimentTask] = [
    HelloWorldTask(),
    SensorPeakChangeTask(),
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
