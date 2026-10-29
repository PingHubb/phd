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

import copy
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from phd.dependence.calibration_recording_names import (
    calibration_folder_name,
    sensor_name_for_recording,
)
from phd.dependence.paths import experiment_path, resource_path
from phd.ui.calibration_readiness import (
    loading_plan, loading_plan_error, suggested_loading_plan, travel_error,
)
from phd.ui.calibration_analysis import (
    analyze_calibration_rows,
    taxel_signal_column,
    contact_referenced_rows,
    release_trial_rows,
)
from phd.ui.experiment_quality import downward_detector, finite_number
from phd.ui.memory_guard import GIB, available_memory_bytes, process_rss_bytes
from phd.ui import full_taxel_resume
from phd.ui.taxel_map_revision import (
    normalize_setup, revision_motion_error, revision_setup, same_pose, validate_setup_binding,
)
from phd.ui.proximity_profile import (
    analyse_proximity_profile,
    report_lines as proximity_report_lines,
    save_profile_outputs,
    surface_reference,
)


_DEFAULT_STORAGE_PATH = object()
_LEGACY_CALIBRATION_SENSOR_KEY = "sensor:testing_8x10"


def _mapping_json_default(value):
    """Encode numeric NumPy values without hiding invalid/unknown objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def connected_calibration_sensor(host):
    """Identity used to keep each tactile sensor's calibration pose separate."""
    ui_ros = getattr(host, "ui_ros", None)
    sensor = getattr(ui_ros, "sensor_functions", None) if ui_ros is not None else None
    rows = int(getattr(sensor, "n_row", 0) or 0)
    columns = int(getattr(sensor, "n_col", 0) or 0)
    profile = {}
    port_name = str(getattr(sensor, "_primary_sensor_port", "") or "")
    if port_name:
        from phd.dependence.humanoid_sensor_registry import (
            assigned_sensor_profile,
        )

        try:
            profile = assigned_sensor_profile(port_name) or {}
        except Exception:
            profile = {}
    key = str(profile.get("key") or "")
    label = str(profile.get("label") or "")
    shape = profile.get("shape")
    if rows <= 0 or columns <= 0:
        if isinstance(shape, tuple) and len(shape) == 2:
            rows, columns = int(shape[0]), int(shape[1])
    if not key:
        key = (
            f"layout:{rows}x{columns}"
            if rows > 0 and columns > 0
            else "layout:unknown"
        )
    if not label:
        label = f"{rows}x{columns} sensor" if rows > 0 and columns > 0 else "this sensor"
    return key, label, rows, columns


def position_map_sensor_error(host, result) -> str:
    """Reject a known map from another sensor even before the UI refreshes."""
    saved_key = str((result or {}).get("sensor_identity") or "")
    if not saved_key or saved_key.startswith("layout:"):
        # Existing unlabelled maps can still be loaded explicitly by the user.
        return ""
    key, label, _rows, _columns = connected_calibration_sensor(host)
    if key != saved_key:
        return (
            f"The loaded map belongs to {saved_key}, but the connected sensor is {label} "
            f"({key}). Load the map for the connected sensor before starting."
        )
    return ""


class ExperimentTask:
    """Base class for Experiments-sidebar tasks."""

    id: str = ""
    label: str = ""
    description: str = ""
    start_button_label: str = "Start"
    show_in_sidebar: bool = True
    category: str = ""
    # If None, the task is one-shot: ``on_start`` runs once and the timer is
    # not started. Otherwise, the host timer ticks every ``tick_interval_ms``.
    tick_interval_ms: Optional[int] = None
    # Force-dependent, long-running tasks opt in so the shared lifecycle can
    # release the RS-485 reader as soon as the task ends.
    disconnect_force_meter_on_stop: bool = False

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


class CalibrationGraphAnalysisTask(ExperimentTask):
    """Open an existing calibration recording for detailed offline analysis."""

    id = "graph_analysis"
    label = "Graph Analysis"
    category = "TRO2026"
    description = (
        "Select a saved calibration CSV, JSON, or exported graph to examine "
        "event distances, fitted equations, residuals, hysteresis, acquisition "
        "quality, and repeat-trial consistency. Opens an independent offline "
        "viewer; it can be used while a robot experiment continues."
    )
    start_button_label = "Select Graph"
    tick_interval_ms = None

    def on_start(self, host) -> bool:
        opener = getattr(host, "_open_calibration_graph_analysis", None)
        if not callable(opener):
            host._append_sidebar_control_message(
                "Graph Analysis is unavailable in this window."
            )
            return False
        opener()
        return False


class SensorCharacterisationTask(ExperimentTask):
    """Open native recording/independent-reference analysis without devices."""
    id = "sensor_characterisation"
    label = "Sensor Characterisation"
    category = "TRO2026"
    description = (
        "Fixture-guided spatial fields, multi-touch, size/speed, mounting, noise "
        "and latency audits. Records native sensor frames and independent "
        "annotations; never sends robot commands. Offline analysis needs no devices."
    )
    start_button_label = "Open Workbench"
    tick_interval_ms = None

    def on_start(self, host):
        opener = getattr(host, "_open_sensor_characterisation", None)
        if callable(opener):
            opener()
        else:
            host._append_sidebar_control_message("Sensor Characterisation workbench is unavailable.")
        return False


class CalibrationTask(ExperimentTask):
    """Characterize one tactile taxel against a force-meter reference."""

    id = "calibration"
    label = "Calibration"
    category = "TRO2026"
    initial_position_name = "Calibration"
    description = (
        "Single taxel returns to a remembered pose and approaches one taxel "
        "in base -Z. Full taxel, on the second tab, loads a position map and "
        "scans every measured taxel. Choose that scan's graph signal on the "
        "Full taxel tab."
    )
    start_button_label = "Run"
    tick_interval_ms = 50
    disconnect_force_meter_on_stop = True
    NEWTONS_PER_GRAM = 0.00980665
    # Confirmed installed cell: 1 kg rated. Overload capacity is NOT its range.
    RATED_FORCE_G = 1000.0
    MAX_CONTACT_THRESHOLD_G = 800.0
    EMERGENCY_FORCE_G = 900.0
    # A trial holds all its frames in memory until saved; the kernel kills the
    # whole UI (mid-press, unsaved) if RAM runs out, so stop first.
    MEMORY_STOP_FREE_BYTES = 1 * GIB
    MEMORY_START_FREE_BYTES = 2 * GIB
    MEMORY_TRIAL_MARGIN = 1.5
    MEMORY_CHECK_INTERVAL_SEC = 2.0
    SIGNAL_MATRIX_SOURCES = {
        "raw": ("rawData", False),
        "calibration": ("calData", False),
        "diff": ("diffData", False),
        "diff_percent": ("diffPerData", False),
        "raw_ave": ("rawDataAve", True),
        "diff_ave": ("diffDataAve", True),
        "diff_percent_ave": ("diffPerDataAve", True),
    }

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
            output_directory = experiment_path("calibration_recordings")
        self.storage_path = (
            Path(storage_path).expanduser() if storage_path is not None else None
        )
        self.output_directory = (
            Path(output_directory).expanduser()
            if output_directory is not None
            else None
        )
        self.initial_position: Optional[dict] = None
        self._position_profiles: dict = {}
        self.active_sensor_key_name = ""
        self.active_sensor_label = ""
        self._taxel_by_sensor: dict = {}
        self.last_persistence_error = ""
        self.sensor_rows = 8
        self.sensor_columns = 10
        self.taxel_index = 35
        # Keep the plotted response in the sensor's native count unit.  The
        # underlying diffDataAve value is signed: raw - calibration.
        self.signal_field = "diff_ave"
        self.approach_speed_m_s = 0.0001
        # 0 disables it. Only used above a contact height already measured
        # on the same taxel in this run, and never once force is detected.
        self.far_field_speed_mm_s = 0.0
        self.slow_zone_mm = 5.0
        # "stepwise": TM line moves measured at rest; "velocity": legacy streaming.
        self.motion_mode = "stepwise"
        self.step_far_mm = 0.2
        self.step_near_mm = 0.05
        self.step_loading_mm = 0.02
        self.step_dwell_sec = 0.5
        self.sliding_step_mm = 0.05
        self._reset_step_state()
        self.velocity_rest_sec = 0.5
        self._rest_last_position = None
        self._rest_since = None
        self._slide_resting = False
        self._slide_rest_started_at = None
        self._slide_target_m = 0.0
        self._return_start_z = None
        self._memory_checked_at: Optional[float] = None
        self._trial_rss_start: Optional[int] = None
        self._trial_memory_bytes = 0
        self._contact_z_by_taxel: dict[int, float] = {}
        self._expected_contact_z_m: Optional[float] = None
        self._approach_zone = ""
        self._slow_zone_entered = False
        self.contact_threshold_n = (
            500.0 * self.NEWTONS_PER_GRAM
        )
        self.contact_hold_sec = 1.0
        self.max_travel_m = 0.03
        self.timeout_sec = 600.0
        self.loading_extra_travel_mm = 5.0  # Planning only; never extends motion by itself.
        # The UI opts in explicitly. Programmatic callers retain manual limits.
        self.loading_auto_plan = False
        self._loading_plan = None
        self.sensor_sample_stale_sec = 1.0
        # Only at a completed, stationary trial boundary, never during motion.
        self.post_save_feedback_timeout_sec = 10.0
        self.post_save_feedback_stable_sec = 0.25
        self._clear_post_save_wait()
        self.sensor_baseline_frames = 100
        self.sensor_baseline_duration_sec = 5.0
        self.sensor_baseline_timeout_sec = 30.0
        self.proximity_window_frames = 7
        self.proximity_min_margin = 6.0
        self.recalibrate_between_trials = True
        self.recovery_observation_sec = 5.0
        self.protocol = "standard"
        self.sliding_distance_mm = 2.0
        self.sliding_speed_mm_s = 0.1
        self.sliding_direction = "x+"
        self.sliding_path_confirmed = False
        self._slide_origin = None
        self._slide_returning = False
        self.speed_schedule_mm_s = [0.05, 0.1, 0.2]
        self.force_schedule_g = [5., 10., 20., 50., 100.]
        # Proximity profile: the largest gap is the far-field reference. Levels
        # are 2–3 mm apart over 10–40 mm, where the 3σ/5σ ranges lie.
        self.profile_gaps_mm = [
            100., 70., 50., 45., 40., 37., 34., 31., 28., 25., 22., 20., 18., 16., 14.,
            12., 10., 8., 6., 5., 4., 3., 2.5, 2., 1.5, 1.2, 1., .8, .6, .4,
        ]
        self.profile_sweeps = 5
        self.profile_order = "alternate"
        self.profile_frames_per_level = 40
        self.profile_settle_sec = 1.0
        # Common-mode rejection removes most drift between references.
        self.profile_reference_every = 6
        self.profile_surface_probes = 3
        self.profile_surface_stop_g = 3.0
        self.profile_search_step_far_mm = 0.2
        self.profile_search_step_near_mm = 0.05
        self.profile_probe_step_mm = 0.01
        self.profile_move_speed_mm_s = 10.0
        self.profile_remeasure_surface = True
        self._profile_project_speed_percent = None
        # Above the force zero re-taken at rest near the surface (stepwise and profile).
        self.light_contact_threshold_g = 0.5
        self._reset_force_zero_state()
        self._reset_profile_state()
        self.specimen_id = ""
        self.session_id = ""
        self.mounting_id = ""
        self.probe_geometry = "unspecified"
        self.probe_material = "conductive"
        self.probe_material_note = ""
        self.probe_diameter_mm = 0.
        self.contact_area_confirmed = False
        self.force_reference_verified = False
        self.experiment_notes = ""
        self._trial_speed_m_s = self.approach_speed_m_s
        self._baseline_rows = []
        self._baseline_last_sequence = None
        self._last_recorded_sequence = None
        self._trial_wall_started_at = None
        self._pause_events = []
        self._recovery_started_at = None
        self._pending_return_purpose = ""
        self._force_sample_source_at = None
        self._pose_sample_source_at = None
        self.repeat_count = 1
        self.return_velocity_rad_s = 0.20
        self.return_timeout_sec = 90.0
        self.return_timeout_factor = 4.0
        self.return_stall_timeout_sec = 120.0
        self.return_progress_epsilon_m = 0.00001
        self.return_settle_sec = 0.75
        self.return_arrival_tolerance_rad = float(np.radians(0.5))
        self.return_height_tolerance_m = 0.0001
        self.recalibration_timeout_sec = 45.0
        self.force_baseline_duration_sec = 1.0
        self.force_baseline_min_samples = 3
        self.force_sample_stale_sec = 1.0
        self.force_connection_timeout_sec = 10.0
        # A force dropout during a force-sensitive phase pauses the robot and
        # keeps the current trial alive. A living reader gets a short chance
        # to recover by itself; after that, reconnection is retried until the
        # stream is stable again or the operator presses Stop.
        self.force_pause_reconnect_delay_sec = 2.0
        self.force_pause_retry_interval_sec = 5.0
        self.force_resume_stable_sec = 0.5
        self.force_resume_min_samples = 3
        self._test_active = False
        self._phase = "idle"
        self._force_connection_started_at: Optional[float] = None
        self._started_at: Optional[float] = None
        self._approach_started_at: Optional[float] = None
        self._start_tool_position = None
        self._max_travel_down_m = 0.0
        self._force_baseline_samples: list[float] = []
        self._force_baseline_n: Optional[float] = None
        self._last_force_sample_at: Optional[float] = None
        self._samples: list[dict] = []
        self._stop_reason = ""
        self._stop_message = ""
        self._stop_phase = ""
        self._stopped_at = ""
        self._stop_taxel_index = None
        self._stop_pass_number = None
        self._trial_result_saved = False
        self._trial_completed = False
        self._trial_number = 0
        self._batch_total = 1
        self._batch_id = ""
        self._batch_directory: Optional[Path] = None
        self._batch_results: list[dict] = []
        self._batch_completed = False
        self._return_started_at: Optional[float] = None
        self._return_arrived_at: Optional[float] = None
        self._current_return_timeout_sec = float(self.return_timeout_sec)
        self._return_best_remaining_m: Optional[float] = None
        self._return_last_progress_at: Optional[float] = None
        self._return_purpose = ""
        self._contact_hold_started_at: Optional[float] = None
        self._recalibration_started_at: Optional[float] = None
        self._force_recovery_started_at: Optional[float] = None
        self._force_recovery_restart_attempted = False
        self._force_paused_phase = ""
        self._force_pause_started_at: Optional[float] = None
        self._force_pause_last_restart_at: Optional[float] = None
        self._force_pause_restart_attempts = 0
        self._force_resume_candidate_at: Optional[float] = None
        self._force_resume_last_sample_at: Optional[float] = None
        self._force_resume_sample_count = 0
        self._force_pause_count = 0
        self._force_pause_total_sec = 0.0
        self.last_result: Optional[dict] = None
        self.load_initial_position()

    @property
    def contact_threshold_g(self) -> float:
        """Contact threshold in the gram-force unit shown by the transmitter."""
        return float(self.contact_threshold_n) / self.NEWTONS_PER_GRAM

    @contact_threshold_g.setter
    def contact_threshold_g(self, value: float) -> None:
        self.contact_threshold_n = (
            min(
                self.MAX_CONTACT_THRESHOLD_G,
                max(0.0, float(value)),
            )
            * self.NEWTONS_PER_GRAM
        )

    @staticmethod
    def _folder_number(value: float) -> str:
        """Return a compact, filename-safe decimal value."""
        text = f"{float(value):.6f}".rstrip("0").rstrip(".")
        if text in {"", "-0"}:
            text = "0"
        return text.replace("-", "neg").replace(".", "p")

    @staticmethod
    def _folder_token(value) -> str:
        token = "".join(
            character
            if character.isalnum() or character in {"-", "_"}
            else "-"
            for character in str(value).strip()
        )
        while "--" in token:
            token = token.replace("--", "-")
        return token.strip("-_") or "unknown"

    def _recording_metadata(self) -> dict:
        return {
            "sensor_name": sensor_name_for_recording(
                self.active_sensor_label, self.active_sensor_key_name,
                (self.sensor_rows, self.sensor_columns),
            ),
            "sensor_identity": str(self.active_sensor_key_name or ""),
            "protocol": str(self.protocol),
        }

    def _experiment_folder_name(self, timestamp: str) -> str:
        """Name one folder from its sensor, protocol, time and seven settings."""
        speed_mm_s = abs(float(self.approach_speed_m_s)) * 1000.0
        travel_mm = max(0.0, float(self.max_travel_m)) * 1000.0
        timeout_s = max(0.0, float(self.timeout_sec))
        travel_token = (
            f"{self._folder_number(travel_mm)}mm"
            if travel_mm > 0.0
            else "unlimited"
        )
        timeout_token = (
            f"{self._folder_number(timeout_s)}s"
            if timeout_s > 0.0
            else "unlimited"
        )
        settings = (
            f"taxel-{int(self.taxel_index)}_"
            f"signal-{self._folder_token(self.signal_field)}_"
            f"speed-{self._folder_number(speed_mm_s)}mmps_"
            f"threshold-{self._force_folder_token()}g_"
            f"travel-{travel_token}_"
            f"timeout-{timeout_token}_"
            f"reps-{max(1, int(self.repeat_count))}_"
            f"{self._probe_material_folder_token()}"
        )
        return calibration_folder_name(
            "calibration", timestamp, settings,
            sensor_name=self._recording_metadata()["sensor_name"], protocol=self.protocol,
        )

    def _probe_material_folder_token(self) -> str:
        """Filename token for the probe category and any typed description."""
        tokens = {
            "conductive": "probe-conductive",
            "insulative": "probe-insulative",
            "human_like": "probe-human-like",
        }
        note = self._folder_token(self.probe_material_note)
        note = "" if note == "unknown" else note[:40].rstrip("-_")
        if self.probe_material == "other":
            return f"probe-{note}" if note else "probe-other"
        base = tokens.get(str(self.probe_material), "probe-unspecified")
        return f"{base}-{note}" if note else base

    def _force_folder_token(self):
        if self.protocol != "force_sweep":
            return self._folder_number(self.contact_threshold_g)
        # Keep a component within filesystem limits; the complete schedule is
        # always in the trial metadata, not the previously selected fixed target.
        levels = "-".join(self._folder_number(v) for v in self.force_schedule_g[:6])
        suffix = "-more" if len(self.force_schedule_g) > 6 else ""
        return f"sweep-{levels}{suffix}"

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

    def _read_position_profiles(self) -> dict:
        if self.storage_path is None or not self.storage_path.exists():
            return {}
        with self.storage_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("saved initial position is invalid")
        profiles = payload.get("profiles")
        if int(payload.get("version", 1) or 1) >= 2 and isinstance(profiles, dict):
            loaded = {}
            for key, item in profiles.items():
                position = self._normalize_initial_position(item)
                if position is not None:
                    loaded[str(key)] = position
            return loaded
        legacy = self._normalize_initial_position(payload)
        if legacy is None:
            raise ValueError("saved initial position is invalid")
        return {_LEGACY_CALIBRATION_SENSOR_KEY: legacy}

    def load_initial_position(self) -> bool:
        self.last_persistence_error = ""
        try:
            self._position_profiles = self._read_position_profiles()
        except Exception as exc:
            self._position_profiles = {}
            self.initial_position = None
            self.last_persistence_error = (
                f"Could not load the saved calibration position: {exc}"
            )
            return False
        if len(self._position_profiles) == 1:
            key, position = next(iter(self._position_profiles.items()))
            self.active_sensor_key_name = key
            self.initial_position = copy.deepcopy(position)
        return True

    def save_initial_position(self) -> bool:
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
                "profiles": self._position_profiles,
            }
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

    def use_connected_sensor(self, host) -> str:
        """Select the pose and taxel grid that belong to the live sensor."""
        if self._test_active:
            return str(self.active_sensor_key_name or "")
        key, label, rows, columns = connected_calibration_sensor(host)
        previous_key = str(self.active_sensor_key_name or "")
        if previous_key and previous_key != key:
            self._taxel_by_sensor[previous_key] = int(self.taxel_index)
        self.active_sensor_key_name = key
        self.active_sensor_label = label
        if (self.initial_position or {}).get("source") == "taxel_position_map":
            return key
        if rows > 0 and columns > 0:
            self.sensor_rows = int(rows)
            self.sensor_columns = int(columns)
            if key not in self._taxel_by_sensor:
                if (self.sensor_rows, self.sensor_columns) == (8, 10):
                    default_taxel = 35
                else:
                    default_taxel = (self.sensor_rows // 2) + (
                        self.sensor_columns // 2
                    ) * self.sensor_rows
                self._taxel_by_sensor[key] = int(default_taxel)
            self.taxel_index = int(self._taxel_by_sensor[key])
            limit = self.sensor_rows * self.sensor_columns
            if not 0 <= self.taxel_index < limit:
                self.taxel_index = 0
                self._taxel_by_sensor[key] = 0
        self.initial_position = copy.deepcopy(
            self._position_profiles.get(key)
        )
        if revision_motion_error(getattr(self, "position_map", None)):
            self.initial_position = None
        return key

    def use_mapped_taxel(self, record: dict, *, rows: int, columns: int) -> None:
        """Use one measured taxel's saved start pose instead of a remembered joint pose."""
        centre = self._finite_values(record.get("tool_position_m"), 3)
        safe = self._finite_values(record.get("safe_approach_position_m"), 3)
        start = self._finite_values(record.get("calibration_start_position_m"), 3)
        if start is None and centre is not None:
            try:
                probe_z = float(record.get("probe_start_z_m"))
            except (TypeError, ValueError):
                probe_z = float("nan")
            if np.isfinite(probe_z):
                start = [float(centre[0]), float(centre[1]), probe_z]
        if start is None:
            start = safe
        quaternion = self._finite_values(record.get("tool_quaternion_wxyz"), 4)
        if start is None or safe is None or quaternion is None:
            raise ValueError(
                f"Taxel {record.get('taxel_index')} has no saved start pose."
            )
        self.sensor_rows = int(rows)
        self.sensor_columns = int(columns)
        self.taxel_index = int(record["taxel_index"])
        self.initial_position = {
            "saved_at": "",
            "joints_rad": None,
            "tool_position_m": [float(value) for value in start],
            "safe_approach_position_m": [float(value) for value in safe],
            "tool_quaternion_wxyz": [float(value) for value in quaternion],
            "source": "taxel_position_map",
            "taxel_index": int(self.taxel_index),
            "sensor_rows": int(rows),
            "sensor_columns": int(columns),
        }

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

        self.use_connected_sensor(host)
        self.initial_position = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "joints_rad": joints,
            "tool_position_m": tool_position,
            "tool_quaternion_wxyz": tool_quaternion,
        }
        sensor_key = str(self.active_sensor_key_name or "layout:unknown")
        self._position_profiles[sensor_key] = copy.deepcopy(self.initial_position)
        saved = self.save_initial_position()
        joint_degrees = [round(value, 2) for value in np.degrees(joints)]
        sensor_label = str(self.active_sensor_label or "this sensor")
        message = (
            f"{self.initial_position_name} initial position remembered for "
            f"{sensor_label}: J={joint_degrees} deg"
        )
        if tool_position is not None:
            message += f" | TCP={[round(value, 5) for value in tool_position]} m"
        host._append_sidebar_control_message(message)
        if not saved:
            host._append_sidebar_control_message(self.last_persistence_error)
        if hasattr(host, "_refresh_calibration_controls"):
            host._refresh_calibration_controls()
        return saved

    def _update_progress(self, host, detail: str = "") -> None:
        callback = getattr(host, "_refresh_calibration_progress", None)
        if callable(callback):
            callback(
                current=int(self._trial_number),
                total=int(self._batch_total),
                phase=str(self._phase),
                detail=str(detail),
            )

    def _reset_travel_status(self, host, *, trial_started: bool = False) -> None:
        """Reset the live TCP-Z displacement shown for this trial."""
        self._max_travel_down_m = 0.0
        callback = getattr(host, "_refresh_calibration_travel", None)
        if callable(callback):
            if trial_started:
                callback(0.0, 0.0)
            else:
                callback(None, None)

    def _update_travel_status(self, host, tool_position) -> Optional[float]:
        """Publish measured downward TCP displacement and return it in metres."""
        start_position = self._finite_values(self._start_tool_position, 3)
        current_position = self._finite_values(tool_position, 3)
        if start_position is None or current_position is None:
            return None
        travel_down_m = max(
            0.0,
            float(start_position[2]) - float(current_position[2]),
        )
        self._max_travel_down_m = max(
            float(self._max_travel_down_m),
            travel_down_m,
        )
        callback = getattr(host, "_refresh_calibration_travel", None)
        if callable(callback):
            callback(
                travel_down_m * 1000.0,
                float(self._max_travel_down_m) * 1000.0,
            )
        return travel_down_m

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

    def _fresh_force_reading(self, host, now=None):
        """Return a force sample only while its reader and timestamp are live."""
        now = time.monotonic() if now is None else float(now)
        reading = self._read_force_meter(host)
        if (
            reading is None
            or not self._force_meter_reader_running(self._ui_ros(host))
            or not -0.1 <= now - float(reading[1]) <= float(self.force_sample_stale_sec)
        ):
            return None
        return reading

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

    def _read_sensor_frame(self, host):
        ui_ros = self._ui_ros(host)
        sensor = getattr(ui_ros, "sensor_functions", None) if ui_ros else None
        data_obj = getattr(sensor, "_data", None) if sensor is not None else None
        if sensor is None or data_obj is None or not bool(getattr(sensor, "is_connected", False)):
            return None
        if hasattr(data_obj, "completed_frame_snapshot"):
            data_obj = data_obj.completed_frame_snapshot
            if data_obj is None:
                return None
        stamp = finite_number(getattr(data_obj, "frame_monotonic_sec", None))
        age = None if stamp is None else time.monotonic() - stamp
        if age is None or not 0.0 <= age <= self.sensor_sample_stale_sec:
            return None
        rows = int(getattr(sensor, "n_row", 0) or 0)
        columns = int(getattr(sensor, "n_col", 0) or 0)
        if rows != int(self.sensor_rows) or columns != int(self.sensor_columns):
            return None
        matrix = np.asarray(getattr(data_obj, "diffDataAve", []), dtype=float)
        if matrix.shape != (rows, columns) or not np.all(np.isfinite(matrix)):
            return None
        return sensor, data_obj, rows, columns

    def _read_taxel(self, host):
        frame = self._read_sensor_frame(host)
        if frame is None:
            return None
        _sensor, data_obj, rows, columns = frame
        index = int(self.taxel_index)
        if not 0 <= index < rows * columns:
            return None

        # Signal-viewer indices are top-down and column-major. Instantaneous
        # matrices use source rows; averaged matrices are already flipped for
        # the shared display convention by SensorDataBuffer.
        display_row = index % rows
        column = index // rows
        source_row = rows - 1 - display_row
        values = {
            "frame_sequence": int(getattr(data_obj, "frame_sequence", 0) or 0),
            "sensor_monotonic_s": finite_number(getattr(data_obj, "frame_monotonic_sec", None)),
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
        source = self.SIGNAL_MATRIX_SOURCES.get(str(self.signal_field))
        if source is not None:
            matrix_name, uses_display_rows = source
            try:
                signal_matrix = np.asarray(
                    getattr(data_obj, matrix_name),
                    dtype=float,
                )
            except (AttributeError, TypeError, ValueError):
                signal_matrix = None
            for taxel_index in range(rows * columns):
                taxel_display_row = taxel_index % rows
                taxel_column = taxel_index // rows
                taxel_row = (
                    taxel_display_row
                    if uses_display_rows
                    else rows - 1 - taxel_display_row
                )
                value = None
                if signal_matrix is not None:
                    try:
                        candidate = float(
                            signal_matrix[taxel_row, taxel_column]
                        )
                    except (IndexError, TypeError, ValueError):
                        candidate = None
                    if candidate is not None and np.isfinite(candidate):
                        value = candidate
                column_name = taxel_signal_column(
                    self.signal_field,
                    taxel_index,
                )
                values[column_name] = value
        # Retain raw, baseline and signed data for EVERY node, independent of
        # the selected plot. Raw matrices use the source-row convention.
        for field in ("raw", "calibration", "diff", "diff_ave", "diff_percent_ave"):
            name, flipped = self.SIGNAL_MATRIX_SOURCES[field]
            for taxel_index in range(rows * columns):
                row = taxel_index % rows
                values[taxel_signal_column(field, taxel_index)] = self._matrix_value(
                    data_obj, name, row if flipped else rows - 1 - row, taxel_index // rows
                )
        return values

    def _validate_measurement_settings(self, host):
        settings = (
            self.contact_threshold_g, self.max_travel_m, self.timeout_sec,
            self.approach_speed_m_s, self.contact_hold_sec, self.recovery_observation_sec,
        )
        if not all(finite_number(value) is not None for value in settings):
            return self._fail_start(host, "Calibration settings must be finite.")
        if self.protocol not in {"standard", "hold_recovery", "speed_sweep", "force_sweep", "drift", "sliding", "proximity", "proximity_profile"} or self.signal_field not in self.SIGNAL_MATRIX_SOURCES:
            return self._fail_start(host, "Select a valid protocol and plotted signal.")
        if not 0 < self.contact_threshold_g <= self.MAX_CONTACT_THRESHOLD_G:
            return self._fail_start(host, "The installed 1 kg cell allows targets up to 800 g, not 2,000 g.")
        if not 0 < self.max_travel_m <= 0.1 or not 0 < self.timeout_sec <= 7200:
            return self._fail_start(host, "Set finite travel (up to 100 mm) and time (up to 7,200 s) limits.")
        if not 0.00001 <= self.approach_speed_m_s <= 0.002:
            return self._fail_start(host, "Approach speed must be 0.01–2 mm/s.")
        far_field = finite_number(self.far_field_speed_mm_s)
        slow_zone = finite_number(self.slow_zone_mm)
        if far_field is None or not (far_field == 0 or 0.01 <= far_field <= 2.0):
            return self._fail_start(host, "Far-field approach speed must be 0 (off) or 0.01–2 mm/s.")
        if slow_zone is None or not 1.0 <= slow_zone <= 50.0:
            return self._fail_start(host, "The slow zone above the previous contact must be 1–50 mm.")
        if not 0 <= self.contact_hold_sec <= 300 or not 0 <= self.recovery_observation_sec <= 300:
            return self._fail_start(host, "Hold/recovery duration must be 0–300 seconds.")
        slowest = 0.001 if self.motion_mode == "stepwise" else 0.01
        if self.protocol == "speed_sweep" and (
            not self.speed_schedule_mm_s or
            any(finite_number(v) is None or not slowest <= float(v) <= 2 for v in self.speed_schedule_mm_s)
        ):
            return self._fail_start(host, f"Speed schedule must contain values from {slowest:g} to 2 mm/s.")
        if self.protocol == "force_sweep" and (
            not self.force_reference_verified or not self.force_schedule_g
            or any(finite_number(v) is None or not 0 < float(v) <= self.MAX_CONTACT_THRESHOLD_G for v in self.force_schedule_g)
        ):
            return self._fail_start(host, "Verify the force reference against known loads and enter force levels above 0 and up to 800 g. Reader resolution/noise may prevent reliable low-force measurements.")
        if finite_number(self.probe_diameter_mm) is None or self.probe_diameter_mm < 0:
            return self._fail_start(host, "Probe diameter must be finite and nonnegative (0 = unknown).")
        if self.contact_area_confirmed and (self.probe_geometry != "flat_circular" or self.probe_diameter_mm <= 0 or not self.force_reference_verified):
            return self._fail_start(host, "Nominal pressure needs a verified force reference and a known flat circular full-contact probe area.")
        if self.protocol == "sliding" and (
            not self.sliding_path_confirmed or self.contact_threshold_g > 200
            or finite_number(self.sliding_distance_mm) is None
            or not 0.1 <= self.sliding_distance_mm <= 5
            or finite_number(self.sliding_speed_mm_s) is None
            or not 0.01 <= self.sliding_speed_mm_s <= 0.5
            or self.sliding_direction not in {"x+", "x-", "y+", "y-"}
        ):
            return self._fail_start(host, "Sliding needs a confirmed clear XY path, target ≤200 g, distance 0.1–5 mm and speed 0.01–0.5 mm/s. No shear-force protection is available.")
        if self.protocol == "proximity" and self.contact_threshold_g > 10:
            return self._fail_start(host, "Proximity approach requires a contact-stop target ≤10 g.")
        if self.protocol == "proximity_profile":
            return self._validate_profile_settings(host)
        if self.motion_mode not in {"stepwise", "velocity"}:
            return self._fail_start(host, "Select stepwise or velocity motion.")
        if self.motion_mode == "stepwise":
            for value, low, high, message in (
                (self.step_far_mm, 0.05, 1.0, "Far approach step must be 0.05–1 mm."),
                (self.step_near_mm, 0.01, 0.2, "Near approach step must be 0.01–0.2 mm."),
                (self.step_loading_mm, 0.005, 0.1, "Loading step must be 0.005–0.1 mm."),
                (self.step_dwell_sec, 0.3, 30.0, "Time at rest per step must be 0.3–30 s."),
                (self.sliding_step_mm, 0.01, 1.0, "Sliding step must be 0.01–1 mm."),
            ):
                number = finite_number(value)
                if number is None or not low <= number <= high:
                    return self._fail_start(host, message)
            if self.protocol == "speed_sweep":
                fastest = float(self.step_loading_mm) / self.STEP_MIN_PERIOD_SEC
                if any(float(v) > fastest for v in self.speed_schedule_mm_s):
                    return self._fail_start(
                        host,
                        "Stepwise loading rate is step size ÷ step period. With "
                        f"{self.step_loading_mm:.3f} mm steps the fastest reliable rate is about "
                        f"{fastest:.3f} mm/s; lower the speeds or enlarge the loading step.",
                    )
        return True

    def _profile_level_gaps(self) -> tuple[float, list[float]]:
        """Far reference gap and the measured levels below it, far to near."""
        gaps = sorted({round(float(value), 4) for value in self.profile_gaps_mm}, reverse=True)
        return gaps[0], gaps[1:]

    def _profile_speed_mm_s(self, project_speed_percent=None) -> tuple[float, float]:
        """Effective line-move speed and the TM project speed override it assumes."""
        percent = finite_number(project_speed_percent)
        if percent is None or percent <= 0:
            percent = finite_number(self._profile_project_speed_percent)
        if percent is None or percent <= 0:
            percent = self.PROFILE_ASSUMED_SPEED_PERCENT
        percent = min(float(percent), 100.0)
        return max(0.01, float(self.profile_move_speed_mm_s) * percent / 100.0), percent

    def estimate_profile_duration_sec(self, project_speed_percent=None) -> float:
        """Time from the surface to the end: probes, every planned window and the end check."""
        speed, _percent = self._profile_speed_mm_s(project_speed_percent)
        window = (
            float(self.profile_settle_sec) + int(self.profile_frames_per_level) / self.PROFILE_FRAME_RATE_HZ
            + self.PROFILE_WINDOW_OVERHEAD_SEC
        )
        probe = (
            self.PROFILE_PROBE_START_M * 1000.0 / float(self.profile_probe_step_mm) * self.PROFILE_SEARCH_STEP_SEC
            + 2.0 * self.PROFILE_MOVE_OVERHEAD_SEC
        )
        total, gap = int(self.profile_surface_probes) * probe, 0.0
        for step in self._build_profile_plan():
            total += self.PROFILE_MOVE_OVERHEAD_SEC + abs(float(step["gap_mm"]) - gap) / speed + window
            gap = float(step["gap_mm"])
        if self.profile_remeasure_surface:
            total += self.PROFILE_MOVE_OVERHEAD_SEC + gap / speed + int(self.profile_surface_probes) * probe
        return total

    def estimate_profile_descent_sec(self) -> float:
        """Upper bound of the stepped surface search within Maximum travel."""
        travel_mm = max(0.0, float(self.max_travel_m) * 1000.0)
        near_mm = min(travel_mm, self.PROFILE_NEAR_ZONE_MM)
        steps = (
            (travel_mm - near_mm) / float(self.profile_search_step_far_mm)
            + near_mm / float(self.profile_search_step_near_mm)
        )
        return steps * self.PROFILE_SEARCH_STEP_SEC

    def _validate_profile_settings(self, host) -> bool:
        if self.id != "calibration":
            return self._fail_start(host, "Run the proximity profile from the Single taxel tab.")
        try:
            gaps = [float(value) for value in self.profile_gaps_mm]
        except (TypeError, ValueError):
            gaps = []
        if not gaps or any(not math.isfinite(g) or not 0.2 <= g <= 100.0 for g in gaps):
            return self._fail_start(host, "Profile gaps must be 0.2–100 mm.")
        if len(self._profile_level_gaps()[1]) < 3:
            return self._fail_start(host, "Enter at least three gaps below the largest (far reference) gap.")
        checks = (
            (self.profile_sweeps, 1, 20, "Sweeps must be 1–20."),
            (self.profile_frames_per_level, 10, 500, "Frames per level must be 10–500."),
            (self.profile_settle_sec, 0, 10, "Settle time must be 0–10 s."),
            (self.profile_reference_every, 1, 20, "Reference interval must be 1–20 levels."),
            (self.profile_surface_probes, 1, 10, "Surface probes must be 1–10."),
            (self.profile_surface_stop_g, 1, 20, "Surface light-contact stop must be 1–20 g."),
            (self.profile_search_step_far_mm, 0.05, 1.0, "Far surface-search step must be 0.05–1 mm."),
            (self.profile_search_step_near_mm, 0.01, 0.2, "Near surface-search step must be 0.01–0.2 mm."),
            (self.profile_probe_step_mm, 0.005, 0.05, "Light-contact probe step must be 0.005–0.05 mm."),
            (self.profile_move_speed_mm_s, 1, 20, "Profile move speed must be 1–20 mm/s."),
        )
        for value, low, high, message in checks:
            number = finite_number(value)
            if number is None or not low <= number <= high:
                return self._fail_start(host, message)
        if self.profile_order not in {"alternate", "descending", "random"}:
            return self._fail_start(host, "Select a profile level order.")
        reported = None
        getter = getattr(self._robot_api(host), "get_project_speed_percent", None)
        if callable(getter):
            try:
                reported = finite_number(getter())
            except Exception:
                reported = None
        if reported is not None and reported > 0:
            self._profile_project_speed_percent = float(reported)
        percent = reported if reported is not None and reported > 0 else self.PROFILE_FALLBACK_SPEED_PERCENT
        speed, percent = self._profile_speed_mm_s(percent)
        profile = self.estimate_profile_duration_sec(percent)
        descent = self.estimate_profile_descent_sec()
        estimate = profile + descent + float(self.max_travel_m) * 1000.0 / speed + 60.0
        if self.timeout_sec < estimate:
            return self._fail_start(
                host,
                f"The proximity profile needs up to {estimate / 60.0:.0f} min at {percent:.0f}% TM project speed "
                f"({profile / 60.0:.0f} min from the surface, up to {descent / 60.0:.0f} min of descent within "
                f"Maximum travel); raise Timeout above {estimate:.0f} s (maximum 7,200 s), shorten Maximum "
                "travel or use fewer levels/sweeps.",
            )
        return True

    def _tick_sliding(self, host, now, position, force_g):
        """Fixed-Z bounded out-and-back path, not force or shear control."""
        api = self._robot_api(host)
        position = np.asarray(position, dtype=float)
        origin = self._slide_origin
        axis = 0 if self.sliding_direction.startswith("x") else 1
        sign = 1 if self.sliding_direction.endswith("+") else -1
        displacement = sign * (position[axis] - origin[axis])
        length = self.sliding_distance_mm / 1000
        if (abs(position[2] - origin[2]) > 0.0005
                or abs(position[1 - axis] - origin[1 - axis]) > 0.0005
                or displacement < -0.0002 or displacement > length + 0.0002
                or force_g > min(1.5 * self.contact_threshold_g, self.EMERGENCY_FORCE_G)
                or force_g < max(1.0, 0.1 * self.contact_threshold_g)):
            self._request_stop(host, "sliding_guard", "Sliding stopped: path deviation, lost contact or excessive normal force. Inspect manually.")
            return
        if self._stepwise():
            self._tick_stepwise_sliding(host, now, position, force_g, axis, sign, length)
            return
        reached = displacement <= 0.00005 if self._slide_returning else displacement >= length - 0.00005
        if reached or self._slide_resting:
            if not self._slide_resting:
                self._slide_resting = True
                self._slide_rest_started_at = float(now)
                self._reset_rest_tracking()
            if not self._send_zero_velocity(host):
                return
            if not self._at_rest(now, position):
                if now - float(self._slide_rest_started_at) > self.VELOCITY_REST_TIMEOUT_SEC:
                    self._request_stop(host, "not_at_rest", "Sliding stopped: the TCP did not come to rest at a path end.")
                return
            self._slide_resting = False
            self._stop_robot(api)  # Reverse only after a clean stop at rest.
            if not self._slide_returning:
                self._slide_returning = True
                return
            self._begin_return_to_initial(
                host, now, purpose="complete" if self._trial_number >= self._batch_total else "next_trial"
            )
            return
        velocity = [0.0, 0.0, 0.0]
        velocity[axis] = sign * self.sliding_speed_mm_s / 1000 * (-1 if self._slide_returning else 1)
        sender = getattr(api, "send_end_effector_velocity_in_frame", None)
        if not callable(sender) or not sender(velocity, [0.0] * 3, frame="base", ensure_mode=True):
            self._request_stop(host, "command_failed", "Sliding stopped: XY velocity command rejected.")

    def _tick_stepwise_sliding(self, host, now, position, force_g, axis, sign, length) -> None:
        """XY step-and-hold at fixed Z; rate = step size / step period."""
        step_m = min(float(self.sliding_step_mm) / 1000.0, length)
        if self._step_state:
            period = step_m / (float(self.sliding_speed_mm_s) / 1000.0)
            if self._tick_step(host, now, position, force_g, period_sec=period) is None:
                return
        if self._slide_returning and self._slide_target_m <= 1e-9:
            self._step_state = ""
            self._begin_return_to_initial(
                host, now, purpose="complete" if self._trial_number >= self._batch_total else "next_trial"
            )
            return
        if not self._slide_returning and self._slide_target_m >= length - 1e-9:
            self._slide_returning = True
        delta = -step_m if self._slide_returning else step_m
        self._slide_target_m = min(length, max(0.0, self._slide_target_m + delta))
        target = np.asarray(self._slide_origin, dtype=float).copy()
        target[axis] += sign * self._slide_target_m
        self._send_step_move(host, now, target, step_m, "sliding")

    def _measurement_guard(self, host, now, *, sensor_required=True):
        """Fail closed before issuing or continuing motion; never auto-retract on fault."""
        if sensor_required and self._read_sensor_frame(host) is None:
            self._request_stop(host, "sensor_stale", "Stopped: tactile data is disconnected, invalid or over 1 s old. Check the sensor before restarting.")
            return False
        api = self._robot_api(host)
        stamp = finite_number(getattr(api, "current_feedback_state_time", None))
        self._pose_sample_source_at = stamp
        # Feedback can arrive just after the tick's `now` was captured.
        if stamp is None or not -0.1 <= now - stamp <= 1.0:
            self._request_stop(host, "robot_feedback_stale", "Stopped: robot position feedback is missing or stale.")
            return False
        force = self._fresh_force_reading(host, now)
        self._force_sample_source_at = None if force is None else float(force[1])
        if force is not None:
            raw = abs(float(force[0]))
            delta = abs(float(force[0]) - float(self._force_baseline_n or 0))
            if max(raw, delta) >= self.EMERGENCY_FORCE_G * self.NEWTONS_PER_GRAM:
                self._request_stop(host, "force_safety_limit", "Emergency force stop at 900 g (raw or baseline-relative). Inspect the setup; no automatic return.")
                return False
        if self._trial_wall_started_at is not None and now - self._trial_wall_started_at >= self.timeout_sec:
            self._request_stop(host, "timeout", "Stopped: trial wall-clock time limit reached, including pauses.")
            return False
        if self._test_active and not self._memory_ok(host, now):
            return False
        return True

    def _memory_ok(self, host, now) -> bool:
        checked = self._memory_checked_at
        if checked is not None and now - checked < self.MEMORY_CHECK_INTERVAL_SEC:
            return True
        self._memory_checked_at = float(now)
        available = available_memory_bytes()
        if available is None or available >= self.MEMORY_STOP_FREE_BYTES:
            return True
        self._request_stop(
            host,
            "low_memory",
            f"Stopped: only {available / GIB:.1f} GB of RAM is free, so the computer would soon "
            "kill the UI. The data recorded so far is saved. The robot stopped where it is "
            "(no automatic retract): move the probe off the sensor if it is pressing, close "
            "other programs, then restart.",
        )
        return False

    def _memory_allows_trial(self, host) -> bool:
        """Stop at the start pose when the next trial would not fit in RAM."""
        available = available_memory_bytes()
        needed = max(
            float(self.MEMORY_START_FREE_BYTES),
            self.MEMORY_TRIAL_MARGIN * float(self._trial_memory_bytes),
        )
        if available is None or available >= needed:
            self._trial_rss_start = process_rss_bytes()
            return True
        self._request_stop(
            host,
            "low_memory",
            f"Stopped before the next trial: {available / GIB:.1f} GB of RAM is free and about "
            f"{needed / GIB:.1f} GB is needed. Saved trials are kept and the robot is at the "
            "start height. Close other programs or restart the UI, then continue.",
        )
        return False

    def _record_trial_memory(self) -> None:
        start, now = self._trial_rss_start, process_rss_bytes()
        if start is not None and now is not None:
            self._trial_memory_bytes = max(self._trial_memory_bytes, now - start)

    def _recorded_taxel_indices(self) -> list[int]:
        """Taxel signal columns intentionally retained in one recording."""
        return list(range(self.sensor_rows * self.sensor_columns))

    def _rows_for_storage(self, rows):
        """Storage projection, after any full-grid analysis has completed."""
        return rows

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

    # TM ContinueVLine: an abrupt stop while moving makes the next velocity
    # command start in the opposite direction. Normal-flow stops therefore
    # stream zero velocity until the TCP is still and only then leave velocity
    # mode; _stop_robot stays the immediate stop for faults and Stop.
    VELOCITY_REST_TOLERANCE_M = 0.000003
    VELOCITY_REST_TIMEOUT_SEC = 10.0
    RETURN_REVERSAL_LIMIT_M = 0.0001
    DESCENT_REVERSAL_LIMIT_M = 0.0005

    # Stepwise motion: TM line moves, each checked at rest before measuring.
    STEP_LINE_MOTION = 4
    STEP_MOVE_SPEED_M_S = 0.005
    STEP_RETRACT_SPEED_M_S = 0.01  # Existing clearance-return command, not the unloading rate.
    STEP_SETTLE_SEC = 0.3
    # TM5 line moves land within about ±0.02 mm; a still TCP this close to a
    # tiny step's target is accepted after STEP_RELAXED_AFTER_SEC.
    STEP_REPEATABILITY_M = 0.00003
    STEP_RELAXED_AFTER_SEC = 1.0
    STEP_FORCE_LAG_SEC = 0.25
    STEP_MIN_PERIOD_SEC = 0.8
    STEP_MOVE_TIMEOUT_SEC = 30.0
    STEP_STATIONARY_TOLERANCE_M = 0.000003
    STEP_RETURN_TOLERANCE_M = 0.0001
    # The trial-start force zero shifts by ~1 g with cable drag and after
    # earlier presses, so light contact is judged against a zero re-taken at
    # rest near the surface: steady within the range for the window, waiting
    # at most the maximum for recovery.
    FORCE_ZERO_WINDOW_SEC = 2.0
    FORCE_ZERO_MIN_SAMPLES = 3
    FORCE_ZERO_STABLE_RANGE_G = 0.15
    FORCE_ZERO_MAX_WAIT_SEC = 15.0
    FORCE_ZERO_TRACK_STEPS = 5

    def _reset_step_state(self) -> None:
        self._step_state = ""
        self._step_mode = ""
        self._step_index = 0
        self._step_size_m = 0.0
        self._step_target = None
        self._step_started_at = None
        self._step_arrived_at = None
        self._step_forces = []
        self._step_raw_forces = []
        self._step_force_at = None
        self._step_last_position = None
        self._step_still_since = None
        self._step_signal_centre = None
        self._step_slow_threshold = None
        self._step_contact_threshold_g = 1.0
        self._step_zeroed = False

    def _reset_force_zero_state(self) -> None:
        self._force_raw_g_now = None
        self._zero_samples = []
        self._zero_started_at = None
        self._zero_last_stamp = None
        self._zero_purpose = ""
        self._force_zero_log = []
        self._far_rest_raw_g = []

    def _start_force_zero(self, now: float, purpose: str) -> None:
        """Begin the stationary window that becomes the new light-contact zero."""
        self._zero_samples = []
        self._zero_started_at = float(now)
        self._zero_last_stamp = self._force_sample_source_at
        self._zero_purpose = str(purpose)

    def _poll_force_zero(self, now: float) -> Optional[dict]:
        """Set the force zero once the at-rest force is steady or the wait runs out."""
        stamp = self._force_sample_source_at
        if stamp is not None and stamp != self._zero_last_stamp and self._force_raw_g_now is not None:
            self._zero_last_stamp = stamp
            self._zero_samples.append((float(now), float(self._force_raw_g_now)))
        waited = float(now) - float(self._zero_started_at)
        if waited < self.FORCE_ZERO_WINDOW_SEC or len(self._zero_samples) < self.FORCE_ZERO_MIN_SAMPLES:
            return None
        window = [g for t, g in self._zero_samples if t >= float(now) - self.FORCE_ZERO_WINDOW_SEC]
        if len(window) < self.FORCE_ZERO_MIN_SAMPLES:
            window = [g for _t, g in self._zero_samples[-self.FORCE_ZERO_MIN_SAMPLES:]]
        values = np.asarray(window, dtype=float)
        spread = float(np.ptp(values))
        stable = spread <= self.FORCE_ZERO_STABLE_RANGE_G
        if not stable and waited < self.FORCE_ZERO_MAX_WAIT_SEC:
            return None
        sigma = float(1.4826 * np.median(np.abs(values - np.median(values))))
        zero = {
            "purpose": self._zero_purpose,
            "zero_g": float(np.median(values)),
            "previous_zero_g": float(self._force_baseline_n) / self.NEWTONS_PER_GRAM,
            "range_g": spread,
            "stable": stable,
            "wait_s": waited,
            "threshold_g": max(float(self.light_contact_threshold_g), 3.0 * sigma, 2.0 * spread),
        }
        self._force_baseline_n = zero["zero_g"] * self.NEWTONS_PER_GRAM
        self._force_zero_log.append(zero)
        return zero

    def _force_zero_message(self, zero: dict) -> str:
        text = (
            f"Force zero {zero['zero_g']:+.2f} g for the {zero['purpose']} (was {zero['previous_zero_g']:+.2f} g; "
            f"steady within {zero['range_g']:.2f} g after {zero['wait_s']:.1f} s); light contact at "
            f"{zero['threshold_g']:.2f} g above it."
        )
        if not zero["stable"]:
            text += (
                f" The force never settled within {self.FORCE_ZERO_STABLE_RANGE_G:.2f} g, so the threshold "
                "was raised; check the force-meter cable."
            )
        return text

    def _track_far_force_zero(self, raw_g: Optional[float]) -> None:
        """Follow slow drift at rest during the far approach (contact jumps within one step)."""
        if raw_g is None:
            return
        self._far_rest_raw_g.append(float(raw_g))
        recent = self._far_rest_raw_g[-self.FORCE_ZERO_TRACK_STEPS:]
        if len(recent) >= self.FORCE_ZERO_MIN_SAMPLES:
            self._force_baseline_n = float(np.median(recent)) * self.NEWTONS_PER_GRAM

    def _step_tags(self) -> dict:
        return {
            "step_state": self._step_state,
            "step_mode": self._step_mode,
            "step_index": self._step_index if self._step_state else "",
        }

    def _stepwise(self) -> bool:
        return self.motion_mode == "stepwise" and self.protocol != "proximity_profile"

    def _send_step_move(self, host, now: float, position, size_m: float, mode: str, speed_m_s=None) -> bool:
        """Command one TM line move; measurement waits until it is at rest."""
        position = np.asarray(position, dtype=float).copy()
        if mode in {"far", "near", "loading"} and self._start_tool_position is not None:
            # Check the next command, not just feedback after the robot has
            # already crossed the limit. Includes the optional far-field jump.
            floor_z = float(self._start_tool_position[2]) - float(self.max_travel_m)
            position[2] = max(float(position[2]), floor_z)
        api = self._robot_api(host)
        pose = self._read_tool_pose(api)
        quaternion = self._finite_values((self.initial_position or {}).get("tool_quaternion_wxyz"), 4)
        if quaternion is None and pose is not None:
            quaternion = self._finite_values(pose[1], 4)
        if api is None or not hasattr(api, "send_positions_tool_position") or quaternion is None:
            self._request_stop(host, "robot_unavailable", "Stopped: TCP position control is unavailable for a step.")
            return False
        self._stop_robot(api)
        try:
            sent = bool(api.send_positions_tool_position(
                [float(value) for value in position], list(quaternion),
                velocity=float(speed_m_s or self.STEP_MOVE_SPEED_M_S), acc_time=0.2,
                blend_percentage=0, fine_goal=True, motion_type=self.STEP_LINE_MOTION,
            ))
        except Exception:
            sent = False
        if not sent:
            self._request_stop(host, "command_failed", "Stopped: the robot rejected a line-move step.")
            return False
        self._step_target = np.asarray(position, dtype=float)
        self._step_size_m = float(size_m)
        self._step_mode = mode
        self._step_index += 1
        self._step_state = "moving"
        self._step_started_at = float(now)
        self._step_last_position = None
        self._step_still_since = None
        return True

    def _tick_step(self, host, now: float, position, force_g: float, *, period_sec: float = 0.0):
        """Advance moving → settling → dwell; return the median force once done."""
        position = np.asarray(position, dtype=float)[:3]
        if self._step_state == "moving":
            previous = self._step_last_position
            self._step_last_position = position
            if previous is None or float(np.max(np.abs(position - previous))) > self.STEP_STATIONARY_TOLERANCE_M:
                self._step_still_since = float(now)
            error = float(np.max(np.abs(position - self._step_target)))
            close = error <= max(self.STEP_STATIONARY_TOLERANCE_M, 0.5 * self._step_size_m) or (
                now - float(self._step_started_at) >= self.STEP_RELAXED_AFTER_SEC
                and error <= self.STEP_REPEATABILITY_M
            )
            arrived = close and now - float(self._step_still_since) >= self.STEP_SETTLE_SEC
            if arrived:
                self._step_state = "settling"
                self._step_arrived_at = float(now)
            elif now - float(self._step_started_at) > self.STEP_MOVE_TIMEOUT_SEC:
                self._request_stop(host, "step_move_timeout", "Stopped: a line-move step did not finish.")
            return None
        if self._step_state == "settling":
            if now - float(self._step_arrived_at) >= self.STEP_FORCE_LAG_SEC:
                self._step_state = "dwell"
                self._step_forces = []
                self._step_raw_forces = []
                self._step_force_at = self._force_sample_source_at
            return None
        if self._step_state != "dwell":
            return None
        stamp = self._force_sample_source_at
        if stamp is not None and stamp != self._step_force_at:
            self._step_force_at = stamp
            self._step_forces.append(float(force_g))
            if self._force_raw_g_now is not None:
                self._step_raw_forces.append(self._force_raw_g_now)
        done = (
            len(self._step_forces) >= 2
            and now - float(self._step_arrived_at) >= float(self.step_dwell_sec)
            and now - float(self._step_started_at) >= float(period_sec)
        )
        if not done:
            if now - float(self._step_arrived_at) > float(self.step_dwell_sec) + float(period_sec) + 5.0:
                self._request_stop(host, "step_no_force", "Stopped: no fresh force data while at rest.")
            return None
        return float(np.median(self._step_forces))

    def _loading_period_sec(self) -> float:
        """Step period that gives the nominal loading rate (0 when unattainable)."""
        speed_m_s = abs(float(self._trial_speed_m_s))
        if speed_m_s <= 0:
            return 0.0
        return float(self.step_loading_mm) / 1000.0 / speed_m_s

    def _insulative_full_taxel_approach(self) -> bool:
        """An insulative probe does not make a reliable proximity signal."""
        return self.id == "full_taxel_calibration" and self.probe_material == "insulative"

    def _begin_stepwise_approach(self, host, now: float) -> None:
        signals = [finite_number(row[2].get(self.signal_field)) for row in self._baseline_rows]
        signals = np.asarray([value for value in signals if value is not None], dtype=float)
        centre = float(np.median(signals)) if signals.size else 0.0
        sigma = float(1.4826 * np.median(np.abs(signals - centre))) if signals.size > 1 else 0.0
        self._step_signal_centre = centre
        self._step_slow_threshold = max(self.PROFILE_SLOW_SIGNAL_MIN, self.PROFILE_SLOW_SIGNAL_SIGMA * sigma)
        forces = np.asarray(self._force_baseline_samples, dtype=float) / self.NEWTONS_PER_GRAM
        noise_g = float(1.4826 * np.median(np.abs(forces - np.median(forces)))) if forces.size > 1 else 0.0
        self._step_contact_threshold_g = max(float(self.light_contact_threshold_g), 3.0 * noise_g)
        self._far_rest_raw_g = []
        self._step_zeroed = False
        start = np.asarray(self._start_tool_position, dtype=float)
        self._step_index = 0
        expected = self._expected_contact_z_m
        jump_z = None if expected is None else expected + float(self.slow_zone_mm) / 1000.0
        if float(self.far_field_speed_mm_s) > 0 and jump_z is not None and jump_z < start[2]:
            self._approach_zone = "far_field"
            host._append_sidebar_control_message(
                f"Stepwise approach: one line move to {self.slow_zone_mm:.1f} mm above the previous "
                "contact height, then near steps."
            )
            landing = "far" if self._insulative_full_taxel_approach() else "near"
            self._send_step_move(host, now, [start[0], start[1], jump_z], 0.0, landing,
                                 speed_m_s=float(self.far_field_speed_mm_s) / 1000.0)
            return
        self._step_mode = "far"
        self._next_approach_step(host, now, np.asarray(self._start_tool_position, dtype=float))

    def _next_approach_step(self, host, now: float, position, size_mm=None) -> None:
        mode = self._step_mode or "far"
        if size_mm is None:
            size_mm = {"far": self.step_far_mm, "near": self.step_near_mm, "loading": self.step_loading_mm}[mode]
        target = np.asarray(position, dtype=float).copy()
        base = self._step_target if self._step_target is not None else target
        floor_z = float(self._start_tool_position[2]) - float(self.max_travel_m)
        if float(base[2]) <= floor_z + 1e-9:
            self._request_stop(host, "max_travel", "Calibration stopped: the commanded downward travel limit was reached before the force target.")
            return
        target[:2] = np.asarray(self._start_tool_position, dtype=float)[:2]
        target[2] = float(base[2]) - float(size_mm) / 1000.0
        self._approach_zone = mode
        self._send_step_move(host, now, target, float(size_mm) / 1000.0, mode)

    def _tick_stepwise_approach(self, host, now: float, tool_position, force_g: float) -> None:
        if self._step_state == "zeroing":
            zero = self._poll_force_zero(now)
            if zero is None:
                return
            self._step_contact_threshold_g = zero["threshold_g"]
            self._step_zeroed = True
            self._step_state = ""
            host._append_sidebar_control_message(self._force_zero_message(zero))
            self._next_approach_step(host, now, tool_position)
            return
        period = self._loading_period_sec() if self._step_mode == "loading" else 0.0
        force = self._tick_step(host, now, tool_position, force_g, period_sec=period)
        if force is None:
            return
        if self._step_mode in {"far", "near"} and force >= self._step_contact_threshold_g:
            self._step_mode = "loading"
            host._append_sidebar_control_message(
                f"Light contact {force:.1f} g at Z {float(tool_position[2]) * 1000.0:.3f} mm; "
                f"loading in {self.step_loading_mm:.3f} mm steps."
            )
        elif self._step_mode == "far":
            self._track_far_force_zero(
                float(np.median(self._step_raw_forces)) if self._step_raw_forces else None
            )
            values = [
                finite_number(row.get(self.signal_field)) for row in self._samples[-7:]
                if row.get("step_state") == "dwell"
            ]
            values = [value for value in values if value is not None]
            if (
                not self._insulative_full_taxel_approach()
                and self._step_signal_centre is not None and values
                and self._step_signal_centre - float(np.median(values)) >= self._step_slow_threshold
            ):
                self._step_mode = "near"
                host._append_sidebar_control_message(
                    f"Proximity signal is strong at Z {float(tool_position[2]) * 1000.0:.2f} mm; "
                    f"continuing in {self.step_near_mm:.3f} mm steps."
                )
        if self._step_mode == "near" and not self._step_zeroed:
            self._start_force_zero(now, "near approach")
            self._step_state = "zeroing"
            return
        size_mm = None
        if self._step_mode == "loading" and force >= 0.8 * self.contact_threshold_g:
            size_mm = max(0.005, 0.5 * float(self.step_loading_mm))
        self._next_approach_step(host, now, tool_position, size_mm)

    def _send_zero_velocity(self, host) -> bool:
        if self._send_base_z_velocity(self._robot_api(host), 0.0):
            return True
        self._request_stop(host, "command_failed", "Stopped: the robot rejected the zero-velocity command.")
        return False

    def _reset_rest_tracking(self) -> None:
        self._rest_last_position = None
        self._rest_since = None

    def _at_rest(self, now: float, position) -> bool:
        """True once the TCP has not moved for ``velocity_rest_sec``."""
        position = np.asarray(position, dtype=float)[:3]
        previous = self._rest_last_position
        self._rest_last_position = position
        if previous is None or float(np.max(np.abs(position - previous))) > self.VELOCITY_REST_TOLERANCE_M:
            self._rest_since = float(now)
            return False
        return now - float(self._rest_since) >= float(self.velocity_rest_sec)

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
        self._force_connection_started_at = None
        host._append_sidebar_control_message(message)
        if hasattr(host, "_set_calibration_controls_running"):
            host._set_calibration_controls_running(False)
        return False

    def _validate_start_prerequisites(self, host) -> bool:
        if not self._validate_measurement_settings(host):
            return False
        revision_error = revision_motion_error(getattr(self, "position_map", None))
        if revision_error:
            return self._fail_start(host, revision_error)
        mismatch = position_map_sensor_error(host, getattr(self, "position_map", None))
        if mismatch:
            return self._fail_start(host, mismatch)
        self.use_connected_sensor(host)
        saved_joints = self._finite_values(
            (self.initial_position or {}).get("joints_rad"), 6
        )
        saved_tcp = self._finite_values(
            (self.initial_position or {}).get("tool_position_m"), 3
        )
        if saved_joints is None and saved_tcp is None:
            return self._fail_start(
                host,
                "Load a taxel position map and select a taxel before pressing Run.",
            )
        api = self._robot_api(host)
        if api is None or not hasattr(api, "get_current_positions"):
            return self._fail_start(
                host,
                "Robot joint feedback or position control is unavailable; "
                "Calibration did not start.",
            )
        if saved_joints is not None and not hasattr(api, "send_positions_joint_angle"):
            return self._fail_start(
                host,
                "Robot joint position control is unavailable; "
                "Calibration did not start.",
            )
        if saved_joints is None and not hasattr(api, "send_positions_tool_position"):
            return self._fail_start(
                host,
                "Robot TCP position control is unavailable; "
                "Calibration did not start.",
            )
        taxel = self._read_taxel(host)
        if taxel is None or int(taxel.get("frame_sequence", 0)) <= 0:
            return self._fail_start(
                host,
                f"A live {self.sensor_rows}x{self.sensor_columns} sensor is required "
                f"before testing taxel {self.taxel_index}.",
            )
        sensor = getattr(self._ui_ros(host), "sensor_functions", None)
        if bool(getattr(sensor, "_sensor_calibration_in_progress", False)):
            return self._fail_start(
                host,
                "Wait for the active sensor calibration to finish before "
                "starting the Calibration experiment.",
            )
        if self.protocol == "standard" and (error := travel_error(self)):
            return self._fail_start(host, error)
        return True

    @staticmethod
    def _force_meter_reader_running(ui_ros) -> bool:
        thread = getattr(ui_ros, "_force_meter_thread", None)
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except RuntimeError:
            return False

    def _begin_force_meter_connection(self, host) -> bool:
        ui_ros = self._ui_ros(host)
        if ui_ros is None:
            return self._fail_start(
                host,
                "Calibration could not access the force-transmitter controls.",
            )

        reader_running = self._force_meter_reader_running(ui_ros)
        if not reader_running:
            refresh_ports = getattr(ui_ros, "_refresh_force_meter_ports", None)
            if callable(refresh_ports):
                refresh_ports()
            start_reader = getattr(ui_ros, "_start_force_meter", None)
            if not callable(start_reader) or not bool(start_reader()):
                status_label = getattr(ui_ros, "force_meter_status_label", None)
                status = (
                    str(status_label.text()).strip()
                    if status_label is not None and hasattr(status_label, "text")
                    else ""
                )
                detail = f" ({status})" if status else ""
                return self._fail_start(
                    host,
                    "Calibration could not automatically start the force "
                    f"transmitter{detail}.",
                )

        self._test_active = False
        self._phase = "connecting_force_meter"
        self._force_connection_started_at = time.monotonic()
        self._stop_reason = ""
        self._samples = []
        self._trial_result_saved = False
        self._reset_travel_status(host)
        if hasattr(host, "_set_calibration_controls_running"):
            host._set_calibration_controls_running(True)
        self._update_progress(
            host,
            "Connecting force transmitter; waiting for a fresh sample",
        )
        host._append_sidebar_control_message(
            "Calibration is connecting the force transmitter automatically. "
            "Robot motion will begin only after fresh force data is received."
        )
        return True

    def _tick_force_meter_connection(self, host, now: float) -> None:
        force_reading = self._read_force_meter(host)
        if (
            force_reading is not None
            and self._force_meter_reader_running(self._ui_ros(host))
            and now - float(force_reading[1])
            <= float(self.force_sample_stale_sec)
        ):
            host._append_sidebar_control_message(
                "Force transmitter connected and live data received. "
                "Starting Calibration."
            )
            self._force_connection_started_at = None
            if not self._start_calibration_cycle(host):
                host._stop_sidebar_control(
                    show_result=False,
                    message="Calibration did not start",
                )
            return

        ui_ros = self._ui_ros(host)
        error_message = str(
            getattr(ui_ros, "_force_meter_error_message", "") or ""
        ).strip()
        started_at = self._force_connection_started_at
        elapsed = 0.0 if started_at is None else now - float(started_at)
        reader_stopped = (
            elapsed >= 0.5 and not self._force_meter_reader_running(ui_ros)
        )
        timed_out = elapsed >= float(self.force_connection_timeout_sec)
        if not error_message and not reader_stopped and not timed_out:
            return

        if error_message:
            detail = error_message
        elif reader_stopped:
            detail = "the force-transmitter reader stopped before sending data"
        else:
            detail = (
                "no fresh force sample was received within "
                f"{self.force_connection_timeout_sec:.0f} seconds"
            )
        self._phase = "idle"
        self._force_connection_started_at = None
        host._append_sidebar_control_message(
            f"Calibration did not start: {detail}. Check the serial port, "
            "transmitter power, RS-485 A/B wiring, protocol, and baud rate."
        )
        host._stop_sidebar_control(
            show_result=False,
            message="Calibration did not start (force transmitter unavailable)",
        )

    def _begin_trial(self, host, now: float, tool_position) -> None:
        if not self._memory_allows_trial(host):
            return
        self._phase = "force_baseline"
        self._started_at = float(now)
        self._trial_wall_started_at = float(now)
        self._baseline_rows = []
        self._baseline_last_sequence = None
        self._last_recorded_sequence = None
        self._pause_events = []
        self._recovery_started_at = None
        repetition = int(getattr(self, "_pass_number", self._trial_number))
        if self.protocol == "force_sweep":
            self.contact_threshold_g = float(self.force_schedule_g[(max(1, repetition)-1) % len(self.force_schedule_g)])
        self._trial_speed_m_s = (
            float(self.speed_schedule_mm_s[(max(1, repetition) - 1) % len(self.speed_schedule_mm_s)]) / 1000
            if self.protocol == "speed_sweep" else abs(float(self.approach_speed_m_s))
        )
        self._approach_started_at = None
        self._start_tool_position = list(tool_position)
        self._expected_contact_z_m = self._contact_z_by_taxel.get(int(self.taxel_index))
        self._approach_zone = "measurement"
        self._slow_zone_entered = False
        self._reset_profile_state()
        self._reset_step_state()
        self._reset_force_zero_state()
        self._reset_travel_status(host, trial_started=True)
        self._force_baseline_samples = []
        self._force_baseline_n = None
        self._last_force_sample_at = None
        self._samples = []
        self._stop_reason = ""
        self._trial_result_saved = False
        self._trial_completed = False
        self._return_started_at = None
        self._return_arrived_at = None
        self._return_best_remaining_m = None
        self._return_last_progress_at = None
        self._return_purpose = ""
        self._contact_hold_started_at = None
        self._recalibration_started_at = None
        self._force_recovery_started_at = None
        self._force_recovery_restart_attempted = False
        self._clear_force_pause_state()
        self._force_pause_count = 0
        self._force_pause_total_sec = 0.0
        self._update_progress(host, "Measuring force baseline")
        host._append_sidebar_control_message(
            f"Calibration trial {self._trial_number}/{self._batch_total}: "
            "measuring the stationary force baseline. Keep the indenter "
            "clear of the sensor."
        )

    def _start_calibration_cycle(self, host) -> bool:
        if not self._validate_start_prerequisites(host):
            return False
        force_reading = self._read_force_meter(host)
        now = time.monotonic()
        if (
            force_reading is None
            or not self._force_meter_reader_running(self._ui_ros(host))
            or now - float(force_reading[1]) > float(self.force_sample_stale_sec)
        ):
            return self._fail_start(
                host,
                "Fresh force-transmitter data is required. Connect the DYLY "
                "force meter in the Tools tab.",
            )

        self._trial_wall_started_at = None
        if not self._measurement_guard(host, now):
            return False
        self._test_active = True
        self._force_connection_started_at = None
        self._contact_z_by_taxel = {}
        self._trial_memory_bytes = 0
        self._batch_total = max(1, int(self.repeat_count))
        self._trial_number = 1
        self._batch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._batch_directory = (
            self.output_directory
            / self._experiment_folder_name(self._batch_id)
            if self.output_directory is not None
            else None
        )
        self._batch_results = []
        self._batch_completed = False
        self.last_result = None
        self._reset_travel_status(host)
        self._started_at = None
        self._approach_started_at = None
        self._contact_hold_started_at = None
        self._return_best_remaining_m = None
        self._return_last_progress_at = None
        self._return_purpose = ""
        self._force_recovery_started_at = None
        self._force_recovery_restart_attempted = False
        self._clear_force_pause_state()
        self._force_pause_count = 0
        self._force_pause_total_sec = 0.0
        if hasattr(host, "_set_calibration_controls_running"):
            host._set_calibration_controls_running(True)
        return self._begin_return_to_initial(
            host,
            now,
            purpose="start",
        )

    def _append_sample(self, now, force_raw_n, taxel, tool_position) -> None:
        if self._started_at is None or self._force_baseline_n is None:
            return
        sequence = (taxel.get("sensor_monotonic_s"), taxel.get("frame_sequence"))
        if sequence == self._last_recorded_sequence:
            return
        self._last_recorded_sequence = sequence
        elapsed = max(0.0, float(now) - float(self._trial_wall_started_at or self._started_at))
        force_sample_fresh = force_raw_n is not None
        force_raw_n = float(force_raw_n) if force_sample_fresh else None
        force_delta = (
            force_raw_n - float(self._force_baseline_n)
            if force_sample_fresh
            else None
        )
        force_raw_g = (
            force_raw_n / self.NEWTONS_PER_GRAM
            if force_sample_fresh
            else None
        )
        force_baseline_g = (
            float(self._force_baseline_n) / self.NEWTONS_PER_GRAM
        )
        force_delta_g = (
            force_delta / self.NEWTONS_PER_GRAM
            if force_delta is not None
            else None
        )
        start_z = float(self._start_tool_position[2])
        current_z = float(tool_position[2])
        sample = {
            "elapsed_s": elapsed,
            "active_elapsed_s": max(0.0, elapsed - self._force_pause_total_sec),
            "acquisition_monotonic_s": float(now),
            "force_monotonic_s": self._force_sample_source_at,
            "robot_monotonic_s": self._pose_sample_source_at,
            "sensor_age_s": max(0.0, now - float(taxel["sensor_monotonic_s"])),
            "force_age_s": None if self._force_sample_source_at is None else max(0.0, now - self._force_sample_source_at),
            "robot_age_s": None if self._pose_sample_source_at is None else max(0.0, now - self._pose_sample_source_at),
            "phase": self._phase,
            "approach_zone": self._approach_zone if self._phase == "approaching" else "",
            "force_sample_fresh": force_sample_fresh,
            "force_raw_n": force_raw_n,
            "force_baseline_n": float(self._force_baseline_n),
            "force_delta_n": force_delta,
            "force_delta_abs_n": (
                abs(force_delta) if force_delta is not None else None
            ),
            "force_raw_g": force_raw_g,
            "force_baseline_g": force_baseline_g,
            "force_delta_g": force_delta_g,
            "force_delta_abs_g": (
                abs(force_delta_g) if force_delta_g is not None else None
            ),
            "tool_z_m": current_z,
            "tool_x_m": float(tool_position[0]),
            "tool_y_m": float(tool_position[1]),
            "travel_down_mm": max(0.0, start_z - current_z) * 1000.0,
            **self._profile_trimmed_taxel(taxel),
        }
        if self.protocol == "proximity_profile":
            sample.update(self._profile_tags())
        elif self.motion_mode == "stepwise":
            sample.update(self._step_tags())
        self._samples.append(sample)

    def _request_stop(self, host, reason: str, message: str) -> None:
        self._stop_reason = str(reason)
        self._stop_message = str(message)
        self._stop_phase = self._phase
        self._stop_robot(self._robot_api(host))
        host._stop_sidebar_control(show_result=True, message=message)

    def _capture_stop_details(self, *, complete=False):
        self._stop_phase = self._stop_phase or self._phase
        self._stopped_at = datetime.now(timezone.utc).isoformat()
        self._stop_taxel_index = int(self.taxel_index)
        self._stop_pass_number = int(getattr(self, "_pass_number", self._trial_number))
        if not complete and self._stop_reason in {"", "contact_threshold", "profile_complete"}:
            self._stop_reason = "manual_stop"
            self._stop_message = "Stopped by the operator."

    def _stop_details(self):
        return {
            "stop_reason": self._stop_reason,
            "stop_message": self._stop_message,
            "stop_phase": self._stop_phase,
            "stopped_at": self._stopped_at,
        }

    def _clear_force_pause_state(self) -> None:
        self._force_paused_phase = ""
        self._force_pause_started_at = None
        self._force_pause_last_restart_at = None
        self._force_pause_restart_attempts = 0
        self._force_resume_candidate_at = None
        self._force_resume_last_sample_at = None
        self._force_resume_sample_count = 0

    def _begin_force_pause(self, host, now: float) -> None:
        """Stop motion while preserving the active calibration trial."""
        if self._phase == "sliding":
            self._request_stop(host, "force_stale_during_sliding", "Sliding aborted: force stream lost. Restart only after checking the setup.")
            return
        if self._phase == "paused_force_meter":
            return
        self._force_paused_phase = str(self._phase)
        self._force_pause_started_at = float(now)
        self._force_pause_last_restart_at = None
        self._force_pause_restart_attempts = 0
        self._force_resume_candidate_at = None
        self._force_resume_last_sample_at = None
        self._force_resume_sample_count = 0
        self._force_pause_count += 1
        self._pause_events.append({"start_monotonic_s": float(now), "phase": self._phase, "end_monotonic_s": None})
        self._stop_robot(self._robot_api(host))
        self._phase = "paused_force_meter"
        live_force_callback = getattr(
            host,
            "_refresh_calibration_live_force",
            None,
        )
        if callable(live_force_callback):
            live_force_callback(None, None)
        self._update_progress(
            host,
            "Paused safely; waiting for stable force data",
        )
        host._append_sidebar_control_message(
            "Force data became stale. Robot motion is paused and the current "
            "trial is preserved. Calibration will resume automatically after "
            "the force stream is stable again; press Stop to cancel."
        )

    def _restart_force_meter_for_pause(self, host, now: float) -> None:
        ui_ros = self._ui_ros(host)
        if ui_ros is None:
            return
        last_attempt = self._force_pause_last_restart_at
        if last_attempt is not None and (
            now - float(last_attempt)
            < float(self.force_pause_retry_interval_sec)
        ):
            return

        pause_started_at = self._force_pause_started_at
        paused_for = (
            0.0
            if pause_started_at is None
            else now - float(pause_started_at)
        )
        reader_running = self._force_meter_reader_running(ui_ros)
        if reader_running and (
            paused_for < float(self.force_pause_reconnect_delay_sec)
        ):
            return

        self._force_pause_last_restart_at = float(now)
        self._force_pause_restart_attempts += 1
        stop_reader = getattr(ui_ros, "_stop_force_meter", None)
        if callable(stop_reader):
            try:
                stop_reader()
            except Exception:
                pass
        refresh_ports = getattr(ui_ros, "_refresh_force_meter_ports", None)
        if callable(refresh_ports):
            try:
                refresh_ports()
            except Exception:
                pass
        start_reader = getattr(ui_ros, "_start_force_meter", None)
        started = False
        if callable(start_reader):
            try:
                started = bool(start_reader())
            except Exception:
                started = False
        status = "started" if started else "waiting for the port"
        self._update_progress(
            host,
            "Paused; force reconnection attempt "
            f"{self._force_pause_restart_attempts} {status}",
        )
        if self._force_pause_restart_attempts == 1:
            host._append_sidebar_control_message(
                "Automatic force-transmitter reconnection started. The robot "
                "will remain stopped until several fresh samples confirm a "
                "stable stream."
            )

    @staticmethod
    def _shift_timestamp(value, duration: float):
        return None if value is None else float(value) + float(duration)

    def _resume_from_force_pause(self, host, now: float) -> None:
        resume_phase = str(self._force_paused_phase or "approaching")
        pause_started_at = self._force_pause_started_at
        pause_duration = (
            0.0
            if pause_started_at is None
            else max(0.0, float(now) - float(pause_started_at))
        )
        self._force_pause_total_sec += pause_duration
        if self._pause_events:
            self._pause_events[-1]["end_monotonic_s"] = float(now)

        if resume_phase == "force_baseline":
            # Re-measure a complete stationary baseline after reconnection.
            self._started_at = float(now)
            self._force_baseline_samples = []
            self._last_force_sample_at = None
            self._baseline_rows = []
            self._baseline_last_sequence = None
        else:
            # Keep active timers, but elapsed_s uses an independent wall clock.
            self._started_at = self._shift_timestamp(
                self._started_at,
                pause_duration,
            )
            self._approach_started_at = self._shift_timestamp(
                self._approach_started_at,
                pause_duration,
            )
            self._contact_hold_started_at = self._shift_timestamp(
                self._contact_hold_started_at,
                pause_duration,
            )
            self._return_started_at = self._shift_timestamp(self._return_started_at, pause_duration)
            self._return_last_progress_at = self._shift_timestamp(self._return_last_progress_at, pause_duration)
            self._recovery_started_at = self._shift_timestamp(self._recovery_started_at, pause_duration)
            self._profile_stage_started_at = self._shift_timestamp(self._profile_stage_started_at, pause_duration)
            self._step_started_at = self._shift_timestamp(self._step_started_at, pause_duration)
            self._step_arrived_at = self._shift_timestamp(self._step_arrived_at, pause_duration)

        self._phase = resume_phase
        self._clear_force_pause_state()
        if self._stepwise() and self._step_state == "moving" and self._step_target is not None:
            # The pause stop cancelled the line-move step; command it again.
            self._step_index -= 1
            self._send_step_move(host, now, self._step_target, self._step_size_m, self._step_mode)
        if resume_phase == "profile_move":
            # The pause stop cancelled the TM line move; command it again.
            self._profile_move_to(
                host, now, self._profile_target_z_m, self._profile_after_move,
                kind=self._profile_kind, tolerance_m=self._profile_move_tolerance_m,
                hold_sec=self._profile_move_hold_sec,
            )
            return
        if resume_phase == "returning" and self._return_purpose == "start":
            # Pausing clears the controller's position trajectory. Rebuild the
            # clearance route from current feedback, rather than waiting for a
            # cancelled trajectory to arrive.
            self._begin_return_to_initial(host, now, purpose="start")
            return
        phase_detail = {
            "force_baseline": "Re-measuring force baseline",
            "approaching": "Force restored; resuming approach",
            "contact_hold": "Force restored; resuming contact hold",
        }.get(resume_phase, "Force restored; resuming calibration")
        self._update_progress(host, phase_detail)
        host._append_sidebar_control_message(
            "Force stream is stable again. Resuming calibration trial "
            f"{self._trial_number}/{self._batch_total} from the paused "
            f"position after {pause_duration:.1f} s."
        )

    def _tick_force_pause(self, host, now: float) -> None:
        reading = self._fresh_force_reading(host, now)
        if reading is None:
            self._force_resume_candidate_at = None
            self._force_resume_last_sample_at = None
            self._force_resume_sample_count = 0
            self._restart_force_meter_for_pause(host, now)
            return

        sample_at = float(reading[1])
        if self._force_resume_candidate_at is None:
            self._force_resume_candidate_at = float(now)
        if sample_at != self._force_resume_last_sample_at:
            self._force_resume_last_sample_at = sample_at
            self._force_resume_sample_count += 1
        stable_for = now - float(self._force_resume_candidate_at)
        required_samples = max(1, int(self.force_resume_min_samples))
        if stable_for < float(self.force_resume_stable_sec) or (
            self._force_resume_sample_count < required_samples
        ):
            self._update_progress(
                host,
                "Force data returned; confirming a stable stream",
            )
            return
        self._resume_from_force_pause(host, now)

    def _begin_return_to_initial(
        self,
        host,
        now: float,
        *,
        purpose: str,
    ) -> bool:
        api = self._robot_api(host)
        purpose = str(purpose)
        if not self._measurement_guard(host, now):
            return False
        if api is None:
            self._request_stop(
                host,
                "return_unavailable",
                "Calibration batch stopped: robot control is unavailable for "
                "the return motion.",
            )
            return False
        self._stop_robot(api)
        self._phase = "returning"
        self._return_purpose = purpose
        if purpose == "start":
            self._samples = []
        self._return_started_at = float(now)
        self._return_arrived_at = None
        self._return_best_remaining_m = None
        self._return_last_progress_at = float(now)
        self._current_return_timeout_sec = float(self.return_timeout_sec)
        if self._fresh_force_reading(host, now) is None:
            self._begin_force_pause(host, now)
            return True

        if purpose == "start" and self._finite_values(
            (self.initial_position or {}).get("joints_rad"), 6
        ) is None:
            self._begin_mapped_taxel_approach(host)
            return True

        if purpose == "start":
            saved_joints = self._finite_values(
                (self.initial_position or {}).get("joints_rad"), 6
            )
            if (
                saved_joints is None
                or not hasattr(api, "send_positions_joint_angle")
            ):
                self._request_stop(
                    host,
                    "return_unavailable",
                    "Calibration batch stopped: the robot cannot return to "
                    "the remembered initial pose before trial 1.",
                )
                return False
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
                    f"Calibration batch stopped: initial-pose return command "
                    f"failed ({exc}).",
                )
                return False
            if not sent:
                self._request_stop(
                    host,
                    "return_command_failed",
                    "Calibration batch stopped: the robot rejected the "
                    "initial-pose return command.",
                )
                return False
            self._update_progress(host, "Returning to the initial pose")
            message = (
                "Calibration Run started. Returning to the remembered "
                "initial position before trial 1."
            )
        else:
            target_position = self._finite_values(self._start_tool_position, 3)
            current_pose = self._read_tool_pose(api)
            if target_position is None or current_pose is None:
                self._request_stop(
                    host,
                    "return_unavailable",
                    "Calibration batch stopped: current or initial TCP height "
                    "is unavailable for the base +Z return.",
                )
                return False
            if self._stepwise():
                self._return_start_z = None
                self._current_return_timeout_sec = max(float(self.return_timeout_sec), 1800.0)
                self._step_target = np.asarray(current_pose[0], dtype=float)
                self._update_travel_status(host, current_pose[0])
                self._update_progress(host, "Stepwise unloading, then a line move to the start height")
                host._append_sidebar_control_message(
                    f"Calibration trial {self._trial_number}/{self._batch_total}: unloading in "
                    f"{self.step_loading_mm:.3f} mm steps measured at rest, then a line move to the "
                    "start height."
                )
                return self._next_unloading_step(host, now)
            return_speed_m_s = abs(float(self._trial_speed_m_s))
            if return_speed_m_s <= 0.0:
                self._request_stop(
                    host,
                    "return_unavailable",
                    "Calibration batch stopped: the base +Z return speed must "
                    "be greater than zero.",
                )
                return False
            self._return_start_z = float(current_pose[0][2])
            self._reset_rest_tracking()
            return_distance_m = max(
                0.0,
                float(target_position[2]) - float(current_pose[0][2]),
            )
            self._update_travel_status(host, current_pose[0])
            expected_return_sec = return_distance_m / return_speed_m_s
            self._current_return_timeout_sec = max(
                float(self.return_timeout_sec),
                expected_return_sec * float(self.return_timeout_factor) + 60.0,
            )
            self._return_best_remaining_m = return_distance_m
            if not self._send_base_z_velocity(
                api,
                return_speed_m_s,
            ):
                self._request_stop(
                    host,
                    "return_command_failed",
                    "Calibration batch stopped: the robot rejected the base "
                    "+Z return command.",
                )
                return False
            return_speed_mm_s = abs(float(self._trial_speed_m_s)) * 1000.0
            self._update_progress(
                host,
                f"Returning in base +Z at {return_speed_mm_s:.3f} mm/s",
            )
            message = (
                f"Calibration trial {self._trial_number}/{self._batch_total}: "
                f"returning in base +Z at {return_speed_mm_s:.3f} mm/s. "
                "The trial completes after the initial height is reached."
            )
        host._append_sidebar_control_message(message)
        return True

    def _mapped_start_waypoints(self, current) -> list:
        start = np.asarray(self.initial_position["tool_position_m"], dtype=float)
        safe_values = self._finite_values(
            self.initial_position.get("safe_approach_position_m"), 3
        )
        safe = start if safe_values is None else np.asarray(safe_values, dtype=float)
        current = np.asarray(current, dtype=float)
        tolerance = float(self.return_height_tolerance_m)
        travel_z = max(float(current[2]), float(safe[2]), float(start[2]))
        waypoints = []
        if current[2] < travel_z - tolerance:
            waypoints.append(np.array([current[0], current[1], travel_z]))
        if float(np.linalg.norm(current[:2] - start[:2])) > tolerance:
            waypoints.append(np.array([start[0], start[1], travel_z]))
        if abs(float(travel_z - start[2])) > tolerance:
            waypoints.append(np.array(start, copy=True))
        elif not waypoints or float(np.linalg.norm(waypoints[-1] - start)) > tolerance:
            waypoints.append(np.array(start, copy=True))
        route = []
        for point in waypoints:
            if route and float(np.linalg.norm(point - route[-1])) <= tolerance:
                continue
            route.append(point)
        return route

    def _send_mapped_start_waypoint(self, host, point) -> bool:
        now = time.monotonic()
        if not self._measurement_guard(host, now):
            return False
        if self._fresh_force_reading(host, now) is None:
            self._begin_force_pause(host, now)
            return True
        api = self._robot_api(host)
        quaternion = self._finite_values(
            (self.initial_position or {}).get("tool_quaternion_wxyz"), 4
        )
        if api is None or quaternion is None or not hasattr(
            api, "send_positions_tool_position"
        ):
            self._request_stop(
                host,
                "return_unavailable",
                "Calibration stopped: robot TCP position control is unavailable.",
            )
            return False
        self._stop_robot(api)
        try:
            sent = bool(
                api.send_positions_tool_position(
                    [float(value) for value in point],
                    list(quaternion),
                    velocity=0.02,
                    acc_time=0.2,
                    blend_percentage=0,
                    fine_goal=True,
                    motion_type=4,
                )
            )
        except Exception as exc:
            sent = False
            self._request_stop(
                host,
                "return_command_failed",
                f"Calibration stopped: moving to the mapped taxel failed ({exc}).",
            )
            return False
        if not sent:
            self._request_stop(
                host,
                "return_command_failed",
                "Calibration stopped: the robot rejected the move to the mapped taxel.",
            )
            return False
        self._mapped_start_target = np.asarray(point, dtype=float)
        self._return_arrived_at = None
        self._update_progress(
            host,
            f"Moving to the saved start pose of Taxel {self.taxel_index}",
        )
        return True

    def _begin_mapped_taxel_approach(self, host) -> bool:
        pose = self._read_tool_pose(self._robot_api(host))
        if pose is None:
            self._request_stop(
                host,
                "feedback_unavailable",
                "Calibration stopped: TCP feedback was unavailable before "
                "moving to the mapped taxel.",
            )
            return False
        self._mapped_start_queue = self._mapped_start_waypoints(pose[0])
        if not self._mapped_start_queue:
            self._begin_trial(host, time.monotonic(), pose[0])
            return True
        point = self._mapped_start_queue.pop(0)
        return self._send_mapped_start_waypoint(host, point)

    def _tick_mapped_taxel_approach(self, host, now: float) -> None:
        pose = self._read_tool_pose(self._robot_api(host))
        target = getattr(self, "_mapped_start_target", None)
        if pose is None or target is None:
            self._request_stop(
                host,
                "feedback_unavailable",
                "Calibration stopped: TCP feedback was lost while moving to "
                "the mapped taxel.",
            )
            return
        distance = float(np.linalg.norm(np.asarray(pose[0], dtype=float) - target))
        if distance > float(self.return_height_tolerance_m):
            self._return_arrived_at = None
            return
        queue = getattr(self, "_mapped_start_queue", None) or []
        if queue:
            point = queue.pop(0)
            self._mapped_start_queue = queue
            self._send_mapped_start_waypoint(host, point)
            return
        if self._return_arrived_at is None:
            self._stop_robot(self._robot_api(host))
            self._return_arrived_at = float(now)
            self._update_progress(host, "Mapped taxel start reached; settling")
            return
        if now - float(self._return_arrived_at) < float(self.return_settle_sec):
            return
        self._begin_trial(host, now, pose[0])

    def _tick_return_to_initial(self, host, now: float) -> None:
        api = self._robot_api(host)
        purpose = str(self._return_purpose)
        if (
            self._return_started_at is not None
            and now - float(self._return_started_at)
            >= float(self._current_return_timeout_sec)
        ):
            self._request_stop(
                host,
                "return_timeout",
                "Calibration batch stopped: returning to the initial position "
                "timed out.",
            )
            return

        if purpose != "start":
            self._tick_vertical_return_to_initial(host, now, api, purpose)
            return
        if self._finite_values(
            (self.initial_position or {}).get("joints_rad"), 6
        ) is None:
            self._tick_mapped_taxel_approach(host, now)
            return

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
        tool_pose = self._read_tool_pose(api)
        if tool_pose is None:
            self._request_stop(
                host,
                "feedback_unavailable",
                "Calibration stopped: TCP feedback was unavailable at "
                "the initial position.",
            )
            return
        self._begin_trial(host, now, tool_pose[0])

    def _next_unloading_step(self, host, now: float) -> bool:
        start = np.asarray(self._start_tool_position, dtype=float)
        target = np.asarray(self._step_target, dtype=float).copy()
        target[:2] = start[:2]
        target[2] = min(float(start[2]), float(target[2]) + float(self.step_loading_mm) / 1000.0)
        if target[2] >= float(start[2]) - 1e-9:
            return self._send_step_move(host, now, start, 2 * self.STEP_RETURN_TOLERANCE_M, "retract", self.STEP_RETRACT_SPEED_M_S)
        return self._send_step_move(host, now, target, float(self.step_loading_mm) / 1000.0, "unloading")

    def _tick_stepwise_return(self, host, now: float, position, purpose: str) -> None:
        reading = self._fresh_force_reading(host, now)
        force_g = (
            0.0 if reading is None or self._force_baseline_n is None
            else abs(float(reading[0]) - float(self._force_baseline_n)) / self.NEWTONS_PER_GRAM
        )
        if self._step_mode == "retract":
            if self._step_state == "moving":
                self._tick_step(host, now, position, force_g)
                return
            if now - float(self._step_arrived_at) < float(self.return_settle_sec):
                return
            self._step_state = ""
            self._step_mode = ""
            self._finish_returned_trial(host, now, purpose)
            return
        force = self._tick_step(host, now, position, force_g, period_sec=self._loading_period_sec())
        if force is None:
            return
        if force < self._step_contact_threshold_g:
            self._send_step_move(
                host, now, np.asarray(self._start_tool_position, dtype=float),
                2 * self.STEP_RETURN_TOLERANCE_M, "retract", self.STEP_RETRACT_SPEED_M_S,
            )
            return
        self._next_unloading_step(host, now)

    def _tick_vertical_return_to_initial(
        self,
        host,
        now: float,
        api,
        purpose: str,
    ) -> None:
        tool_pose = self._read_tool_pose(api)
        target_position = self._finite_values(self._start_tool_position, 3)
        if tool_pose is None or target_position is None:
            self._request_stop(
                host,
                "return_feedback_unavailable",
                "Calibration batch stopped: TCP feedback was unavailable "
                "during the base +Z return.",
            )
            return
        current_z = float(tool_pose[0][2])
        target_z = float(target_position[2])
        self._update_travel_status(host, tool_pose[0])
        if self._stepwise():
            self._tick_stepwise_return(host, now, tool_pose[0], purpose)
            return
        if self._return_start_z is not None and current_z < self._return_start_z - self.RETURN_REVERSAL_LIMIT_M:
            self._request_stop(
                host, "velocity_reversal",
                "Stopped: the robot moved down during the +Z return (TM velocity-mode reversal). "
                "It is stopped where it is; retract it manually before restarting.",
            )
            return
        remaining_m = target_z - current_z
        if remaining_m > float(self.return_height_tolerance_m):
            best_remaining = self._return_best_remaining_m
            if (
                best_remaining is None
                or remaining_m
                <= best_remaining - float(self.return_progress_epsilon_m)
            ):
                self._return_best_remaining_m = remaining_m
                self._return_last_progress_at = float(now)
            elif (
                self._return_last_progress_at is not None
                and now - float(self._return_last_progress_at)
                >= float(self.return_stall_timeout_sec)
            ):
                self._request_stop(
                    host,
                    "return_stalled",
                    "Calibration batch stopped: no upward TCP progress was "
                    "detected during the base +Z return.",
                )
                return
            self._return_arrived_at = None
            if not self._send_base_z_velocity(
                api,
                abs(float(self._trial_speed_m_s)),
            ):
                self._request_stop(
                    host,
                    "return_command_failed",
                    "Calibration batch stopped: the robot rejected the base "
                    "+Z return command.",
                )
            return

        if not self._send_zero_velocity(host):
            return
        at_rest = self._at_rest(now, tool_pose[0])
        if self._return_arrived_at is None:
            self._return_arrived_at = float(now)
            self._update_progress(
                host,
                "Initial height reached; settling before completing trial",
            )
            return
        settled_for = now - float(self._return_arrived_at)
        if settled_for < float(self.return_settle_sec) or not at_rest:
            if settled_for >= float(self.return_settle_sec) + self.VELOCITY_REST_TIMEOUT_SEC:
                self._request_stop(host, "not_at_rest", "Stopped: the TCP did not come to rest at the start height.")
            return
        self._stop_robot(api)  # Leave velocity mode only while still.
        self._return_start_z = None
        self._finish_returned_trial(host, now, purpose)

    def _finish_returned_trial(self, host, now: float, purpose: str) -> None:
        if self._phase != "recovery" and self.recovery_observation_sec > 0:
            self._phase = "recovery"
            self._recovery_started_at = float(now)
            self._pending_return_purpose = str(purpose)
            self._update_progress(host, "At initial height; observing baseline recovery")
            return
        self._complete_returned_trial(host, now, purpose)

    def _complete_returned_trial(self, host, now: float, purpose: str) -> None:
        self._trial_completed = True
        try:
            result = self._save_current_trial(host)
        except Exception as exc:
            self._request_stop(
                host,
                "save_failed",
                "Calibration batch stopped: trial data could not be "
                f"saved after the return ({exc}).",
            )
            return
        if result is None:
            self._request_stop(
                host,
                "save_failed",
                "Calibration batch stopped: no trial samples were available "
                "after the return.",
            )
            return

        if purpose == "complete":
            self._batch_completed = True
            host._stop_sidebar_control(
                show_result=True,
                message=(
                    "Calibration batch complete. Final trial completed after "
                    "the robot returned to the initial height using base +Z "
                    "motion."
                ),
            )
        else:
            self._begin_post_save_wait(host, purpose)

    def _clear_post_save_wait(self):
        self._post_save_started_at = None
        self._post_save_purpose = ""
        self._post_save_ready_since = None
        self._post_save_last_stamps = None
        self._post_save_frame_counts = [0, 0, 0]
        self._post_save_wait_detail = ""

    def _begin_post_save_wait(self, host, purpose):
        """Yield to Qt after saving; no route or calibration in the save callback.

        Serial payloads are delivered on the GUI thread. Even a healthy reader
        cannot refresh its completed frame while synchronous analysis/save runs.
        The normal next-motion guard must run on a later tick, on new feedback.
        """
        self._stop_robot(self._robot_api(host))
        self._clear_post_save_wait()
        self._post_save_started_at = time.monotonic()
        self._post_save_purpose = str(purpose)
        self._phase = "waiting_post_save_feedback"
        if purpose == "resume_home":
            self._update_progress(host, "Resume loaded; waiting for fresh feedback before routing")
            host._append_sidebar_control_message(
                "Resume files prepared. Robot remains stopped while new sensor, force and robot "
                f"feedback is confirmed ({self.post_save_feedback_timeout_sec:g} s maximum)."
            )
            return
        self._update_progress(host, "Trial saved; stopped at safe height, waiting for fresh feedback")
        host._append_sidebar_control_message(
            "Trial saved. Robot remains stopped; waiting for new tactile, force and robot "
            f"feedback before continuing ({self.post_save_feedback_timeout_sec:g} s maximum)."
        )

    def _tick_post_save_wait(self, host, now):
        """A bounded stationary wait, NOT a stale-data exception during motion."""
        if not self._test_active or self._phase != "waiting_post_save_feedback":
            return False
        started = self._post_save_started_at
        if started is None:
            self._request_stop(host, "post_save_state_lost", "Stopped: the post-save continuation was lost.")
            return True
        # A force alarm still stops immediately, even while awaiting other data.
        reading = self._read_force_meter(host)
        if reading is not None:
            raw, _stamp = reading
            delta = abs(raw - float(self._force_baseline_n or 0))
            if max(abs(raw), delta) >= self.EMERGENCY_FORCE_G * self.NEWTONS_PER_GRAM:
                self._request_stop(host, "force_safety_limit", "Emergency force stop at 900 g while waiting after saving. No automatic motion.")
                return True
        sensor_frame = self._read_sensor_frame(host)
        sensor_stamp = None if sensor_frame is None else finite_number(sensor_frame[1].frame_monotonic_sec)
        force = self._fresh_force_reading(host, now)
        force_stamp = None if force is None else float(force[1])
        api = self._robot_api(host)
        robot_stamp = finite_number(getattr(api, "current_feedback_state_time", None))
        if self._read_tool_pose(api) is None or robot_stamp is None or not -0.1 <= now - robot_stamp <= 1.0:
            robot_stamp = None
        stamps = (sensor_stamp, force_stamp, robot_stamp)
        missing = [name for name, stamp in zip(("tactile sensor", "force transmitter", "robot position"), stamps)
                   if stamp is None or stamp <= started]
        if now - started >= self.post_save_feedback_timeout_sec:
            detail = ", ".join(missing) if missing else "a stable sequence of new feedback samples"
            self._request_stop(
                host, "post_save_feedback_timeout",
                f"Stopped at safe height after saving: did not receive {detail} within "
                f"{self.post_save_feedback_timeout_sec:g} s. Completed trial data is preserved; no next move was sent.",
            )
            return True
        if missing:
            self._post_save_ready_since = None
            self._post_save_last_stamps = None
            self._post_save_frame_counts = [0, 0, 0]
            detail = "Waiting after save: " + ", ".join(missing)
            if detail != self._post_save_wait_detail:
                self._post_save_wait_detail = detail
                self._update_progress(host, detail)
            return True
        if self._post_save_ready_since is None:
            self._post_save_ready_since = now
        previous = self._post_save_last_stamps
        for i, stamp in enumerate(stamps):
            if previous is None or stamp > previous[i]:
                self._post_save_frame_counts[i] += 1
        self._post_save_last_stamps = stamps
        if (now - self._post_save_ready_since < self.post_save_feedback_stable_sec
                or min(self._post_save_frame_counts) < 3):
            return True
        # Preserve all ordinary freshness, force and wall-time checks before motion.
        if not self._measurement_guard(host, now):
            return True
        purpose = self._post_save_purpose
        self._clear_post_save_wait()
        host._append_sidebar_control_message("Fresh feedback confirmed after saving; continuing the calibration sequence.")
        self._continue_after_saved_trial(host, now, purpose)
        return True

    def _continue_after_saved_trial(self, host, now, purpose):
        self._start_sensor_recalibration(host, now)

    def _start_sensor_recalibration(self, host, now: float) -> None:
        if not self.recalibrate_between_trials:
            pose = self._read_tool_pose(self._robot_api(host))
            if pose is None:
                self._request_stop(host, "feedback_unavailable", "Robot pose unavailable.")
                return
            self._begin_next_trial_or_force_recovery(host, now, pose[0])
            return
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
        if self._read_sensor_frame(host) is None:
            return  # Wait for the first complete post-calibration frame.
        tool_pose = self._read_tool_pose(self._robot_api(host))
        if tool_pose is None:
            self._request_stop(
                host,
                "feedback_unavailable",
                "Calibration batch stopped: TCP feedback was unavailable after "
                "sensor calibration.",
            )
            return
        self._begin_next_trial_or_force_recovery(host, now, tool_pose[0])

    def _begin_next_trial_or_force_recovery(
        self,
        host,
        now: float,
        tool_position,
    ) -> None:
        if self._fresh_force_reading(host, now) is not None:
            self._trial_number += 1
            self._begin_trial(host, now, tool_position)
            return
        self._phase = "recovering_force_meter"
        self._force_recovery_started_at = float(now)
        self._force_recovery_restart_attempted = False
        self._update_progress(
            host,
            "Initial height reached; recovering force transmitter",
        )
        host._append_sidebar_control_message(
            "The force stream became stale during the completed return. "
            "The robot is stopped at the initial height; reconnecting the "
            f"force transmitter before trial {self._trial_number + 1}/"
            f"{self._batch_total}."
        )

    def _tick_force_recovery(self, host, now: float) -> None:
        reading = self._fresh_force_reading(host, now)
        if reading is not None:
            tool_pose = self._read_tool_pose(self._robot_api(host))
            if tool_pose is None:
                self._request_stop(
                    host,
                    "feedback_unavailable",
                    "Calibration batch stopped: TCP feedback was unavailable "
                    "after force-transmitter recovery.",
                )
                return
            self._force_recovery_started_at = None
            self._force_recovery_restart_attempted = False
            self._trial_number += 1
            host._append_sidebar_control_message(
                "Fresh force data received. Starting calibration trial "
                f"{self._trial_number}/{self._batch_total}."
            )
            self._begin_trial(host, now, tool_pose[0])
            return

        if not self._force_recovery_restart_attempted:
            self._force_recovery_restart_attempted = True
            ui_ros = self._ui_ros(host)
            stop_reader = getattr(ui_ros, "_stop_force_meter", None)
            if callable(stop_reader):
                try:
                    stop_reader()
                except Exception:
                    pass
            refresh_ports = getattr(ui_ros, "_refresh_force_meter_ports", None)
            if callable(refresh_ports):
                refresh_ports()
            start_reader = getattr(ui_ros, "_start_force_meter", None)
            if not callable(start_reader) or not bool(start_reader()):
                self._request_stop(
                    host,
                    "force_meter_recovery_failed",
                    "Calibration batch stopped: the force transmitter could "
                    "not be reconnected before the next trial.",
                )
                return

        started_at = self._force_recovery_started_at
        elapsed = 0.0 if started_at is None else now - float(started_at)
        if elapsed >= float(self.force_connection_timeout_sec):
            self._request_stop(
                host,
                "force_meter_recovery_timeout",
                "Calibration batch stopped safely at the initial height: no "
                "fresh force sample arrived after reconnection.",
            )

    def _save_result(self) -> Optional[dict]:
        if not self._samples:
            return None
        recording_metadata = self._recording_metadata()
        # The derived contact-distance column is not an input to this analysis.
        # Reuse one result instead of computing the same expensive fits twice.
        analysis = analyze_calibration_rows(self._samples, self.signal_field)
        rows = contact_referenced_rows(
            self._samples, self.signal_field, analysis=analysis, in_place=True
        )
        profile = None
        if self.protocol == "proximity_profile":
            profile = analyse_proximity_profile(
                rows, rows_n=self.sensor_rows, columns_n=self.sensor_columns,
                target=self.taxel_index, signal_field=self.signal_field,
                force_threshold_g=self._profile_force_threshold_g,
            )
            profile.update(recording_metadata)
        contact_travel_mm = (analysis.get("contact_onset") or {}).get("distance_mm")
        if contact_travel_mm is not None and self._start_tool_position is not None:
            self._contact_z_by_taxel[int(self.taxel_index)] = (
                float(self._start_tool_position[2]) - float(contact_travel_mm) / 1000.0
            )
        quality = {
            "resume_session": len(getattr(self, "_resume_history", [])),
            "interrupted": bool(self._force_pause_count),
            "trial_completed": self._trial_completed,
            "eligible_for_comparison": self._trial_completed and self._stop_reason == "contact_threshold" and not self._force_pause_count,
            "pause_events": copy.deepcopy(self._pause_events),
            "recording_clock": "monotonic wall time; pauses retained",
            "acquisition": "unique tactile frames polled at up to 20 Hz; latest timestamped force/robot readings",
            "rated_force_g": self.RATED_FORCE_G,
            "emergency_force_g": self.EMERGENCY_FORCE_G,
            "protocol": self.protocol,
            "near_target_speed_fraction": 0.25,
            "slowdown_at_target_fraction": 0.8,
            "motion": self._motion_metadata(),
            "loading_setup_plan": copy.deepcopy(self._loading_plan),
            "far_field_approach": {
                "speed_mm_s": float(self.far_field_speed_mm_s),
                "slow_zone_mm": float(self.slow_zone_mm),
                "expected_contact_z_m": self._expected_contact_z_m,
                "used": any(row.get("approach_zone") == "far_field" for row in self._samples),
                "rule": "Fast only above (previous contact Z of this taxel in this run + slow zone) and before any 1 g force change.",
            },
            "speed_schedule_mm_s": list(self.speed_schedule_mm_s),
            "force_schedule_g": list(self.force_schedule_g),
            "setup": {
                "specimen_id": self.specimen_id or self.active_sensor_key_name,
                "session_id": self.session_id,
                "mounting_id": self.mounting_id,
                "probe_geometry": self.probe_geometry,
                "probe_material": self.probe_material,
                "probe_material_note": str(self.probe_material_note).strip(),
                "probe_diameter_mm": self.probe_diameter_mm or None,
                "contact_area_confirmed": bool(self.contact_area_confirmed),
                "force_reference_verified": bool(self.force_reference_verified),
                "pressure_interpretation": "Nominal force divided by declared flat-tip area, not measured pressure distribution.",
            },
            "sliding": {
                "distance_mm": self.sliding_distance_mm,
                "speed_mm_s": self.sliding_speed_mm_s,
                "direction": self.sliding_direction,
                "mode": "fixed Z, no shear measurement or constant-force control",
            } if self.protocol == "sliding" else None,
            "proximity_profile_settings": self._profile_settings() if profile is not None else None,
            "notes": str(self.experiment_notes),
            "recalibrate_between_trials": bool(self.recalibrate_between_trials),
            "recovery_observation_s": float(self.recovery_observation_sec),
            "sensor_baseline_frames": len(self._baseline_rows),
            "sensor_identity": self.active_sensor_key_name,
            "sensor_name": recording_metadata["sensor_name"],
        }
        trial_number = max(1, int(self._trial_number))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = (
            f"trial_{trial_number:04d}"
            if self._batch_directory is not None
            else f"calibration_taxel{int(self.taxel_index)}_{timestamp}"
        )
        csv_path = None
        metadata_path = None
        profile_files = {}
        rows = self._rows_for_storage(rows)
        if self.output_directory is not None:
            result_directory = self._batch_directory or self.output_directory
            result_directory.mkdir(parents=True, exist_ok=True)
            if profile is not None:
                profile_files = save_profile_outputs(profile, result_directory, stem)
            csv_path = result_directory / f"{stem}.csv"
            columns = list(rows[0].keys())
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            metadata_path = result_directory / f"{stem}.json"
            metadata = {
                "version": 8,
                **recording_metadata,
                "measurement_quality": quality,
                "batch_id": self._batch_id,
                "trial_number": trial_number,
                "requested_trials": int(self._batch_total),
                "stop_reason": self._stop_reason,
                "sensor_shape": [self.sensor_rows, self.sensor_columns],
                "taxel_index": int(self.taxel_index),
                "signal_field": self.signal_field,
                "recorded_taxel_indices": self._recorded_taxel_indices(),
                "approach_speed_m_s": float(self._trial_speed_m_s),
                "return_speed_m_s": float(abs(self._trial_speed_m_s)),
                **self._return_speed_metadata(),
                "return_height_tolerance_m": float(
                    self.return_height_tolerance_m
                ),
                "contact_threshold_n": float(self.contact_threshold_n),
                "contact_threshold_g": float(self.contact_threshold_g),
                "contact_hold_sec": float(self.contact_hold_sec),
                "max_travel_m": float(self.max_travel_m),
                "timeout_sec": float(self.timeout_sec),
                "force_baseline_n": self._force_baseline_n,
                "force_baseline_g": (
                    None
                    if self._force_baseline_n is None
                    else float(self._force_baseline_n)
                    / self.NEWTONS_PER_GRAM
                ),
                "initial_position": self.initial_position,
                "experiment_directory": str(result_directory),
                "force_pause_count": int(self._force_pause_count),
                "force_pause_total_s": float(self._force_pause_total_sec),
                "sample_count": len(rows),
                "analysis": analysis,
                "proximity_profile": None if profile is None else {
                    key: profile.get(key) for key in (
                        "surface", "single_frame_sigma", "reference_drift", "detection",
                        "hysteresis", "profile", "warnings",
                    )
                },
                "proximity_profile_files": profile_files,
                "csv_path": str(csv_path),
            }
            with metadata_path.open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2)
                stream.write("\n")

        result = {
            **recording_metadata,
            "measurement_quality": quality,
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
            "sensor_shape": [self.sensor_rows, self.sensor_columns],
            "recorded_taxel_indices": self._recorded_taxel_indices(),
            "approach_speed_m_s": float(self._trial_speed_m_s),
            "return_speed_m_s": float(abs(self._trial_speed_m_s)),
            **self._return_speed_metadata(),
            "return_height_tolerance_m": float(
                self.return_height_tolerance_m
            ),
            "contact_threshold_n": float(self.contact_threshold_n),
            "contact_threshold_g": float(self.contact_threshold_g),
            "contact_hold_sec": float(self.contact_hold_sec),
            "experiment_directory": (
                str(self._batch_directory)
                if self._batch_directory is not None
                else ""
            ),
            "force_pause_count": int(self._force_pause_count),
            "force_pause_total_s": float(self._force_pause_total_sec),
            "analysis": analysis,
            "proximity_profile": profile,
            "proximity_profile_files": profile_files,
        }
        result["summary"] = self._summarize_trial(result)
        return result

    def _return_speed_metadata(self) -> dict:
        """Qualify the legacy nominal rate without changing any motion command."""
        return {
            "return_speed_m_s_scope": (
                "nominal_stepwise_unloading_rate" if self._stepwise() else
                "not_applicable_to_proximity_profile_steps" if self.protocol == "proximity_profile" else
                "velocity_return_command"
            ),
            "fast_return_command_speed_m_s": (
                self.STEP_RETRACT_SPEED_M_S if self._stepwise() else None
            ),
            "return_speed_note": (
                "Configured rates/commands, not measured TCP speeds. Stepwise unloading includes dwell; "
                "fast return includes arrival settling. Acceleration and robot project speed can affect motion."
            ),
        }

    def _motion_metadata(self) -> dict:
        if not self._stepwise():
            return {"mode": "proximity_profile_steps" if self.protocol == "proximity_profile" else "velocity"}
        loading = [
            row for row in self._samples
            if row.get("step_mode") == "loading" and row.get("step_state") == "dwell"
        ]
        rate = None
        if len(loading) >= 2:
            span = float(loading[-1]["elapsed_s"]) - float(loading[0]["elapsed_s"])
            travel = float(loading[0]["tool_z_m"]) - float(loading[-1]["tool_z_m"])
            rate = travel * 1000.0 / span if span > 0 else None
        return {
            "mode": "stepwise",
            "description": "TM line (position) moves; each step measured at rest",
            "far_step_mm": float(self.step_far_mm),
            "near_step_mm": float(self.step_near_mm),
            "loading_step_mm": float(self.step_loading_mm),
            "time_at_rest_per_step_s": float(self.step_dwell_sec),
            "force_lag_ignored_s": self.STEP_FORCE_LAG_SEC,
            "contact_threshold_g": float(self._step_contact_threshold_g),
            "light_contact_threshold_g": float(self.light_contact_threshold_g),
            "force_zeros": [dict(zero) for zero in self._force_zero_log],
            "nominal_loading_rate_mm_s": abs(float(self._trial_speed_m_s)) * 1000.0,
            "nominal_unloading_rate_mm_s": abs(float(self._trial_speed_m_s)) * 1000.0,
            "fast_return_command_speed_mm_s": self.STEP_RETRACT_SPEED_M_S * 1000.0,
            "measured_loading_rate_mm_s": rate,
            "sliding_step_mm": float(self.sliding_step_mm) if self.protocol == "sliding" else None,
        }

    def _profile_settings(self) -> dict:
        reference_gap, levels = self._profile_level_gaps()
        return {
            "reference_gap_mm": reference_gap,
            "level_gaps_mm": levels,
            "sweeps": int(self.profile_sweeps),
            "order": self.profile_order,
            "random_seed": self._profile_seed,
            "frames_per_level": int(self.profile_frames_per_level),
            "settle_s": float(self.profile_settle_sec),
            "reference_every_levels": int(self.profile_reference_every),
            "surface_probes": int(self.profile_surface_probes),
            "surface_stop_g": float(self.profile_surface_stop_g),
            "surface_contact_threshold_g": float(self._profile_force_threshold_g),
            "light_contact_threshold_g": float(self.light_contact_threshold_g),
            "force_zeros": [dict(zero) for zero in self._force_zero_log],
            "motion": "TM line (position) steps checked at rest; no velocity mode",
            "surface_search_step_far_mm": float(self.profile_search_step_far_mm),
            "surface_search_step_near_mm": float(self.profile_search_step_near_mm),
            "surface_slow_signal_drop": self._profile_slow_threshold,
            "surface_probe_step_mm": float(self.profile_probe_step_mm),
            "surface_probe_start_mm": self.PROFILE_PROBE_START_M * 1000.0,
            "force_lag_s": self.PROFILE_FORCE_LAG_SEC,
            "move_speed_mm_s": float(self.profile_move_speed_mm_s),
            "project_speed_percent": self._profile_project_speed_percent,
            "position_tolerance_mm": self.PROFILE_POSITION_TOLERANCE_M * 1000.0,
            "remeasure_surface": bool(self.profile_remeasure_surface),
        }

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
        force_values_g = self._finite_row_values(rows, "force_delta_abs_g")
        travel_values = self._finite_row_values(rows, "travel_down_mm")
        duration_values = self._finite_row_values(rows, "elapsed_s")
        signal_field = str(result.get("signal_field") or self.signal_field)
        signal_values = self._finite_row_values(rows, signal_field)
        analysis = dict(
            result.get("analysis")
            or analyze_calibration_rows(rows, signal_field)
        )
        signal_onset = analysis.get("signal_onset") or {}
        contact_onset = analysis.get("contact_onset") or {}
        signal_distance_fit = analysis.get("signal_distance_fit") or {}
        force_distance_fit = analysis.get("force_distance_fit") or {}
        force_signal_fit = analysis.get("force_signal_fit") or {}
        proximity_range = analysis.get("proximity_range") or {}
        turning = analysis.get("turning_point") or {}
        turning = turning if turning.get("detected") else {}
        baseline_return = analysis.get("baseline_return") or {}
        rise_fit = analysis.get("force_signal_rise_fit") or {}
        profile = result.get("proximity_profile")
        profile_fields = {}
        if profile is not None:
            detection = profile.get("detection") or {}
            localized = profile.get("detection_localized") or {}
            profile_fields = {
                "profile_surface_z_mm": (
                    None if (profile.get("surface") or {}).get("surface_z_m") is None
                    else profile["surface"]["surface_z_m"] * 1000.0
                ),
                "profile_surface_sd_mm": (profile.get("surface") or {}).get("surface_sd_mm"),
                "profile_single_frame_sigma": profile.get("single_frame_sigma"),
                "profile_detection_3sigma_mm": (detection.get("3_sigma") or {}).get("distance_mm"),
                "profile_detection_5sigma_mm": (detection.get("5_sigma") or {}).get("distance_mm"),
                "profile_localized_3sigma_mm": (localized.get("3_sigma") or {}).get("distance_mm"),
                "profile_localized_5sigma_mm": (localized.get("5_sigma") or {}).get("distance_mm"),
            }
        return {
            **profile_fields,
            "trial_number": int(result.get("trial_number", 0)),
            "stop_reason": str(result.get("stop_reason", "")),
            "force_pause_count": int(result.get("force_pause_count", 0)),
            "force_pause_total_s": float(
                result.get("force_pause_total_s", 0.0)
            ),
            "sample_count": len(rows),
            "duration_s": max(duration_values, default=0.0),
            "peak_force_change_n": max(force_values, default=0.0),
            "peak_force_change_g": max(force_values_g, default=0.0),
            "contact_travel_mm": max(travel_values, default=0.0),
            "signal_min": min(signal_values, default=0.0),
            "signal_max": max(signal_values, default=0.0),
            "signal_peak_abs": max(
                (abs(value) for value in signal_values), default=0.0
            ),
            "signal_final": signal_values[-1] if signal_values else 0.0,
            "signal_onset_distance_mm": signal_onset.get("distance_mm"),
            "contact_onset_distance_mm": contact_onset.get("distance_mm"),
            "signal_distance_slope": signal_distance_fit.get("slope"),
            "signal_distance_intercept": signal_distance_fit.get("intercept"),
            "signal_distance_r_squared": signal_distance_fit.get("r_squared"),
            "force_distance_slope": force_distance_fit.get("slope"),
            "force_distance_intercept": force_distance_fit.get("intercept"),
            "force_distance_r_squared": force_distance_fit.get("r_squared"),
            "force_signal_slope": force_signal_fit.get("slope"),
            "force_signal_intercept": force_signal_fit.get("intercept"),
            "force_signal_r_squared": force_signal_fit.get("r_squared"),
            "proximity_range_status": proximity_range.get("status"),
            "proximity_range_mm": proximity_range.get("distance_above_contact_mm"),
            "proximity_range_lower_bound_mm": proximity_range.get("lower_bound_mm"),
            "turning_force_g": turning.get("force_g"),
            "turning_force_low_g": turning.get("force_low_g"),
            "turning_force_high_g": turning.get("force_high_g"),
            "turning_after_contact_mm": turning.get("after_contact_mm"),
            "turning_signal": turning.get("signal"),
            "baseline_return_force_g": baseline_return.get("force_g"),
            "force_signal_rise_slope": rise_fit.get("slope"),
            "force_signal_rise_intercept": rise_fit.get("intercept"),
            "force_signal_rise_r_squared": rise_fit.get("r_squared"),
            "csv_path": str(result.get("csv_path", "")),
            "metadata_path": str(result.get("metadata_path", "")),
        }

    def _save_current_trial(self, host) -> Optional[dict]:
        if self._trial_result_saved:
            return self._batch_results[-1] if self._batch_results else None
        result = self._save_result()
        self._trial_result_saved = True
        self._record_trial_memory()
        if result is None:
            return None
        self._batch_results.append(result)
        csv_path = result.get("csv_path", "")
        if csv_path:
            host._append_sidebar_control_message(
                f"Calibration trial {self._trial_number}/{self._batch_total} "
                f"saved: {csv_path}"
            )
        if result.get("proximity_profile") is not None:
            lines = proximity_report_lines(result["proximity_profile"])
            host._append_sidebar_control_message("\n".join(
                line for line in lines[2:] if line.startswith((
                    "Surface", "Single", "Far-field", "Common-mode", "Localized", "Detection", "Target strongest",
                    "Warning",
                ))
            ))
            if result.get("proximity_profile_files", {}).get("png"):
                host._append_sidebar_control_message(
                    f"Proximity profile graph: {result['proximity_profile_files']['png']}"
                )
            return result
        for message in self._calibration_target_messages(result.get("analysis") or {}):
            host._append_sidebar_control_message(message)
        return result

    @staticmethod
    def _calibration_target_messages(analysis) -> list[str]:
        """Plain-language result of the three calibration targets for one trial."""
        messages = []
        proximity = analysis.get("proximity_range") or {}
        status = proximity.get("status")
        if status == "measured":
            messages.append(
                f"Proximity first detected {proximity['distance_above_contact_mm']:.2f} mm before force contact."
            )
        elif status == "start_inside_range":
            messages.append(
                "Proximity was already present at the start pose, so the range is at least "
                f"{proximity['lower_bound_mm']:.2f} mm. Raise the start pose (and recalibrate "
                "there) to measure the true range."
            )
        elif status == "proximity_not_detected":
            messages.append("No proximity response was detected before contact.")
        turning = analysis.get("turning_point") or {}
        if turning.get("detected"):
            messages.append(
                f"Drop-to-rise turning point at {turning['force_g']:.1f} g "
                f"(flat band {turning['force_low_g']:.1f}–{turning['force_high_g']:.1f} g), "
                f"{turning['after_contact_mm']:.2f} mm after force contact."
            )
        elif turning.get("reason") == "no_rise_within_loading":
            messages.append(
                "The signal had not started rising when the force target was reached; "
                "increase the force target to find the turning point."
            )
        baseline_return = analysis.get("baseline_return") or {}
        if baseline_return.get("reached"):
            messages.append(
                f"Signal returned to its no-touch baseline at {baseline_return['force_g']:.1f} g."
            )
        return messages

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
            "version": 2,
            **self._recording_metadata(),
            **self._stop_details(),
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
            "recorded_taxel_indices": self._recorded_taxel_indices(),
            "approach_speed_m_s": float(self.approach_speed_m_s),
            "return_speed_m_s": float(abs(self.approach_speed_m_s)),
            **self._return_speed_metadata(),
            "return_height_tolerance_m": float(
                self.return_height_tolerance_m
            ),
            "contact_threshold_n": float(self.contact_threshold_n),
            "contact_threshold_g": float(self.contact_threshold_g),
            "contact_hold_sec": float(self.contact_hold_sec),
            "max_travel_m": float(self.max_travel_m),
            "timeout_sec": float(self.timeout_sec),
            "return_velocity_rad_s": float(self.return_velocity_rad_s),
            "initial_position": self.initial_position,
            "experiment_directory": str(result_directory),
            "summary_csv_path": str(summary_path),
            "trials": summary_rows,
        }
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")

        # PNG exports live beside the CSV/JSON files: one experiment, one
        # self-contained directory with no separate graph output folder.
        graph_directory = result_directory
        return str(summary_path), str(metadata_path), str(graph_directory)

    def _build_final_result(self) -> Optional[dict]:
        if not self._batch_results:
            return None
        summary_path, metadata_path, graph_directory = self._save_batch_summary()
        common = {
            **self._recording_metadata(),
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
            "experiment_directory": (
                str(self._batch_directory)
                if self._batch_directory is not None
                else ""
            ),
        }
        if self._batch_total == 1 and len(self._batch_results) == 1:
            result = dict(self._batch_results[0])
            result.update(common)
            return result
        return {
            **common,
            "taxel_index": int(self.taxel_index),
            "signal_field": self.signal_field,
            "sensor_shape": [self.sensor_rows, self.sensor_columns],
            "recorded_taxel_indices": self._recorded_taxel_indices(),
            "contact_threshold_n": float(self.contact_threshold_n),
            "contact_threshold_g": float(self.contact_threshold_g),
            "contact_hold_sec": float(self.contact_hold_sec),
            "trial_results": list(self._batch_results),
            "rows": list(self._batch_results[-1].get("rows") or []),
            "stop_reason": str(
                self._batch_results[-1].get("stop_reason", "")
            ),
        }

    def _loading_project_speed(self, host):
        percent = None
        getter = getattr(self._robot_api(host), "get_project_speed_percent", None)
        if callable(getter):
            try:
                percent = getter()
            except Exception:
                pass
        return percent

    def on_start(self, host) -> bool:
        if not self._test_active:
            self._clear_post_save_wait()
            self._stop_message = self._stop_phase = self._stopped_at = ""
            self._stop_taxel_index = self._stop_pass_number = None
        revision_error = revision_motion_error(getattr(self, "position_map", None))
        if revision_error:
            return self._fail_start(host, revision_error)
        percent = self._loading_project_speed(host)
        if self.loading_auto_plan and self.protocol == "standard":
            # This runs only on an explicit Run, never on ticks, retries or
            # per-taxel transitions. Runtime limits remain fixed for the batch.
            if self._test_active or self._phase == "connecting_force_meter":
                host._append_sidebar_control_message("Stop the current calibration before replanning.")
                return False
            mismatch = position_map_sensor_error(host, self.position_map)
            if mismatch:
                return self._fail_start(host, mismatch)
            self.use_connected_sensor(host)
            proposed = suggested_loading_plan(self, project_speed_percent=percent)
            error = loading_plan_error(proposed)
            if error:
                return self._fail_start(host, "Automatic loading plan blocked: " + error)
            self.max_travel_m = proposed["suggested_travel_mm"] / 1000.
            self.timeout_sec = proposed["suggested_timeout_sec"]
            sync = getattr(host, "_refresh_calibration_planned_limits", None)
            if callable(sync):
                sync(self)
            host._append_sidebar_control_message(
                f"Automatic loading plan: travel {self.max_travel_m * 1000.:.1f} mm "
                f"(mapped scan + {self.loading_extra_travel_mm:g} mm allowance), "
                f"timeout {self.timeout_sec:.0f} s per taxel/trial. Force target unchanged. "
                "This is not a clearance approval; verify a supervised low-force pilot."
            )
        if not self._validate_start_prerequisites(host):
            return False
        self._loading_plan = loading_plan(
            self, extra_mm=self.loading_extra_travel_mm, project_speed_percent=percent,
        )
        if self._loading_plan is not None:
            self._loading_plan["limits_mode"] = "automatic" if self.loading_auto_plan else "manual"
        if self._loading_plan is not None and self.timeout_sec < self._loading_plan["suggested_timeout_sec"]:
            host._append_sidebar_control_message(
                f"Loading/unloading timeout warning: {self.timeout_sec:.0f} s is below the planning "
                f"allowance of {self._loading_plan['suggested_timeout_sec']:.0f} s per trial at "
                f"{self._loading_plan['project_speed_percent']:.0f}% project speed. "
                "This estimate is not a measured duration; keep the finite limit and verify a single-taxel pilot."
            )
        force_reading = self._read_force_meter(host)
        now = time.monotonic()
        if (
            force_reading is not None
            and self._force_meter_reader_running(self._ui_ros(host))
            and now - float(force_reading[1])
            <= float(self.force_sample_stale_sec)
        ):
            return self._start_calibration_cycle(host)
        return self._begin_force_meter_connection(host)

    def on_tick(self, host) -> None:
        if self._phase == "connecting_force_meter":
            self._tick_force_meter_connection(host, time.monotonic())
            return
        if not self._test_active:
            return
        now = time.monotonic()
        if self._tick_post_save_wait(host, now):
            return
        if not self._measurement_guard(
            host, now, sensor_required=self._phase not in {"recalibrating"}
        ):
            return
        if self._phase == "paused_force_meter":
            self._tick_force_pause(host, now)
            return
        if self._phase == "recovering_force_meter":
            self._tick_force_recovery(host, now)
            return
        if self._phase == "recalibrating":
            self._tick_sensor_recalibration(host, now)
            return

        force_reading = self._fresh_force_reading(host, now)
        if force_reading is None and self._phase in {"returning", "recovery"}:
            self._begin_force_pause(host, now)
            return
        if self._phase == "returning":
            force_raw_n = force_reading[0] if force_reading is not None else None
            live_force_callback = getattr(
                host,
                "_refresh_calibration_live_force",
                None,
            )
            if callable(live_force_callback):
                if force_raw_n is None:
                    live_force_callback(None, None)
                else:
                    force_delta_g = (
                        None
                        if self._force_baseline_n is None
                        else (
                            float(force_raw_n) - float(self._force_baseline_n)
                        )
                        / self.NEWTONS_PER_GRAM
                    )
                    live_force_callback(
                        float(force_raw_n) / self.NEWTONS_PER_GRAM,
                        force_delta_g,
                    )
            if self._return_purpose != "start":
                taxel = self._read_taxel(host)
                tool_pose = self._read_tool_pose(self._robot_api(host))
                if taxel is not None and tool_pose is not None:
                    self._append_sample(
                        now,
                        force_raw_n,
                        taxel,
                        tool_pose[0],
                    )
            self._tick_return_to_initial(host, now)
            return

        if force_reading is None:
            self._begin_force_pause(host, now)
            return
        force_raw_n, force_sample_at = force_reading
        self._force_raw_g_now = float(force_raw_n) / self.NEWTONS_PER_GRAM
        live_force_callback = getattr(
            host,
            "_refresh_calibration_live_force",
            None,
        )
        if callable(live_force_callback):
            force_delta_g = (
                None
                if self._force_baseline_n is None
                else (
                    float(force_raw_n) - float(self._force_baseline_n)
                )
                / self.NEWTONS_PER_GRAM
            )
            live_force_callback(
                float(force_raw_n) / self.NEWTONS_PER_GRAM,
                force_delta_g,
            )

        if self._phase == "force_baseline":
            taxel = self._read_taxel(host)
            pose = self._read_tool_pose(self._robot_api(host))
            if taxel is None or pose is None:
                self._request_stop(host, "feedback_unavailable", "Baseline stopped: missing tactile or robot feedback.")
                return
            sequence = (taxel["sensor_monotonic_s"], taxel["frame_sequence"])
            if sequence != self._baseline_last_sequence:
                self._baseline_last_sequence = sequence
                self._baseline_rows.append((now, force_raw_n, dict(taxel), list(pose[0]), force_sample_at, self._pose_sample_source_at))
            if force_sample_at != self._last_force_sample_at:
                self._force_baseline_samples.append(float(force_raw_n))
                self._last_force_sample_at = force_sample_at
            baseline_elapsed = now - self._started_at
            if baseline_elapsed < max(float(self.force_baseline_duration_sec), self.sensor_baseline_duration_sec):
                return
            if len(self._baseline_rows) < self.sensor_baseline_frames:
                # Large user-selected windows need longer than the default
                # 30 s timeout (the UI collector is capped at 20 Hz).
                baseline_timeout = max(
                    float(self.sensor_baseline_timeout_sec),
                    float(self.sensor_baseline_duration_sec) + 0.1 * self.sensor_baseline_frames,
                )
                if baseline_elapsed >= baseline_timeout:
                    self._request_stop(host, "insufficient_sensor_baseline", "Stopped: insufficient unique stationary tactile frames.")
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
                    "Calibration stopped: not enough fresh DYLY baseline samples.",
                )
                return
            self._force_baseline_n = float(
                np.median(self._force_baseline_samples)
            )
            baseline_values = [r[2]["diff_ave"] for r in self._baseline_rows]
            if any(finite_number(v) is None for v in baseline_values):
                self._request_stop(host, "invalid_sensor_baseline", "Stopped: non-finite tactile baseline.")
                return
            detector = downward_detector(baseline_values, self.proximity_window_frames, self.proximity_min_margin)
            half = max(1, len(baseline_values) // 2)
            drift = abs(float(np.median(baseline_values[-half:]) - np.median(baseline_values[:half])))
            if drift > detector["margin"]:
                self._request_stop(host, "unstable_sensor_baseline", "Stopped: stationary baseline is drifting. Let it settle and restart.")
                return
            for stamp, force, baseline_taxel, position, force_at, pose_at in self._baseline_rows:
                self._force_sample_source_at = force_at
                self._pose_sample_source_at = pose_at
                self._append_sample(stamp, force, baseline_taxel, position)
            self._force_sample_source_at = force_sample_at
            self._pose_sample_source_at = finite_number(getattr(self._robot_api(host), "current_feedback_state_time", None))
            self._phase = "approaching"
            self._approach_started_at = now
            if self.protocol == "proximity_profile":
                self._begin_profile(host, now)
                return
            if self._stepwise():
                self._update_progress(host, "Stepwise approach to the selected taxel")
                host._append_sidebar_control_message(
                    f"Force baseline: {self._force_baseline_n / self.NEWTONS_PER_GRAM:+.1f} g. Stepwise "
                    f"approach to taxel {self.taxel_index}: {self.step_far_mm:.2f} mm steps until the "
                    f"signal is strong, {self.step_near_mm:.3f} mm to contact, then "
                    f"{self.step_loading_mm:.3f} mm loading steps, each measured at rest for "
                    f"{self.step_dwell_sec:.2f} s."
                    + (
                        " Insulative probe: the far step continues until force contact, "
                        "then loading steps begin."
                        if self._insulative_full_taxel_approach() else ""
                    )
                )
                self._begin_stepwise_approach(host, now)
                return
            self._update_progress(host, "Approaching the selected taxel")
            baseline_g = self._force_baseline_n / self.NEWTONS_PER_GRAM
            host._append_sidebar_control_message(
                f"Force baseline: {baseline_g:+.1f} g "
                f"({self._force_baseline_n:+.4f} N). "
                f"Approaching taxel {self.taxel_index} at "
                f"{self._trial_speed_m_s * 1000.0:.3f} mm/s."
                + (
                    f" Far field at {float(self.far_field_speed_mm_s):.2f} mm/s until "
                    f"{self.slow_zone_mm:.1f} mm above the previous contact height."
                    if self._expected_contact_z_m is not None
                    and float(self.far_field_speed_mm_s) / 1000.0 > abs(float(self._trial_speed_m_s))
                    else ""
                )
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
        force_delta_abs_g = force_delta_abs / self.NEWTONS_PER_GRAM
        travel_down_m = self._update_travel_status(host, tool_position)
        if travel_down_m is None:
            self._request_stop(
                host,
                "feedback_unavailable",
                "Calibration stopped: TCP travel could not be measured.",
            )
            return

        if self._phase == "recovery":
            if now - float(self._recovery_started_at or now) >= self.recovery_observation_sec:
                self._complete_returned_trial(host, now, self._pending_return_purpose)
            return

        if self._phase == "sliding":
            self._tick_sliding(host, now, tool_position, force_delta_abs_g)
            return

        # Limits apply during holds as well as during descent.
        if travel_down_m >= float(self.max_travel_m):
            self._request_stop(host, "max_travel", "Stopped: maximum downward travel limit reached.")
            return

        if self._phase.startswith("profile_"):
            self._tick_profile(host, now, tool_position, force_delta_abs_g)
            return

        if self._phase == "contact_hold":
            if not self._stepwise() and not self._send_zero_velocity(host):
                return
            at_rest = self._at_rest(now, tool_position)
            held_for = (
                -1.0 if self._contact_hold_started_at is None
                else now - float(self._contact_hold_started_at)
            )
            if held_for < float(self.contact_hold_sec):
                return
            if not at_rest:
                if held_for >= float(self.contact_hold_sec) + self.VELOCITY_REST_TIMEOUT_SEC:
                    self._request_stop(host, "not_at_rest", "Stopped: the TCP did not come to rest after the contact stop.")
                return
            self._stop_robot(self._robot_api(host))  # Leave velocity mode only while still.
            if self.protocol == "sliding":
                self._slide_origin = np.asarray(tool_position, dtype=float)
                self._slide_returning = False
                self._slide_resting = False
                self._slide_target_m = 0.0
                self._step_state = ""
                self._phase = "sliding"
                self._update_progress(host, "Fixed-Z out-and-back sliding; normal-force guard active")
                return
            purpose = (
                "complete"
                if self._trial_number >= self._batch_total
                else "next_trial"
            )
            self._begin_return_to_initial(
                host,
                now,
                purpose=purpose,
            )
            return

        if (
            self._phase == "approaching"
            and force_delta_abs >= float(self.contact_threshold_n)
        ):
            self._stop_reason = "contact_threshold"
            if self._stepwise():
                self._stop_robot(self._robot_api(host))  # Cancels the current line-move step.
                self._step_state = ""
            elif not self._send_zero_velocity(host):
                return
            self._reset_rest_tracking()
            self._phase = "contact_hold"
            self._contact_hold_started_at = float(now)
            self._update_progress(
                host,
                f"Contact {force_delta_abs_g:.1f} g; holding "
                f"{self.contact_hold_sec:.1f} s",
            )
            host._append_sidebar_control_message(
                f"Contact threshold reached at {force_delta_abs_g:.1f} g "
                f"({force_delta_abs:.4f} N). "
                + ("Step stopped" if self._stepwise() else "Velocity set to zero")
                + f"; holding for {self.contact_hold_sec:.1f} s and returning once the TCP is still."
            )
            return
        if (
            float(self.max_travel_m) > 0.0
            and travel_down_m >= float(self.max_travel_m)
        ):
            self._request_stop(
                host,
                "max_travel",
                f"Calibration stopped at the {self.max_travel_m * 1000.0:.1f} mm travel limit.",
            )
            return
        if (
            float(self.timeout_sec) > 0.0
            and now - float(self._started_at) >= float(self.timeout_sec)
        ):
            self._request_stop(
                host,
                "timeout",
                f"Calibration stopped at the {self.timeout_sec:.1f} s timeout.",
            )
            return
        if self._stepwise():
            self._tick_stepwise_approach(host, now, tool_position, force_delta_abs_g)
            return
        if float(tool_position[2]) > float(self._start_tool_position[2]) + self.DESCENT_REVERSAL_LIMIT_M:
            self._request_stop(
                host, "velocity_reversal",
                "Stopped: the robot moved up while commanded down (TM velocity-mode reversal "
                "after an abrupt stop). Restart the run.",
            )
            return
        speed = abs(float(self._trial_speed_m_s))
        if force_delta_abs >= 0.8 * self.contact_threshold_n:
            speed *= 0.25
        elif self._far_field_allowed(tool_position, force_delta_abs_g):
            speed = float(self.far_field_speed_mm_s) / 1000.0
        if self._approach_zone == "far_field" and speed <= abs(float(self._trial_speed_m_s)):
            host._append_sidebar_control_message(
                f"Entered the slow measurement zone ({speed * 1000.0:.3f} mm/s) "
                f"{self.slow_zone_mm:.1f} mm above the previous contact height."
            )
        self._approach_zone = "far_field" if speed > abs(float(self._trial_speed_m_s)) else "measurement"
        if not self._send_base_z_velocity(self._robot_api(host), -speed):
            self._request_stop(
                host,
                "command_failed",
                "Calibration stopped: the robot rejected the base -Z velocity command.",
            )

    # TM velocity mode moves in stick-slip jumps of up to ~0.6 mm at low speed
    # and keeps moving after a stop, so the profile never uses it. Every move,
    # including the surface search, is a TM line (position) move that is
    # checked with the robot at rest.
    PROFILE_LINE_MOTION = 4
    PROFILE_POSITION_TOLERANCE_M = 0.0001
    PROFILE_STATIONARY_TOLERANCE_M = 0.000003
    PROFILE_STEP_SETTLE_SEC = 0.3
    PROFILE_FORCE_LAG_SEC = 0.25
    PROFILE_CHECK_FORCE_SAMPLES = 2
    PROFILE_CHECK_TIMEOUT_SEC = 5.0
    PROFILE_SEARCH_FORCE_LIMIT_G = 300.0
    PROFILE_SLOW_SIGNAL_SIGMA = 10.0
    PROFILE_SLOW_SIGNAL_MIN = 20.0
    PROFILE_PROBE_START_M = 0.0003
    PROFILE_SURFACE_SETTLE_SEC = 0.5
    PROFILE_COLLECT_TIMEOUT_SEC = 60.0
    # Duration model, slightly conservative against the TM rig: a line move
    # takes ~1 s plus distance / (speed limit × project speed override).
    PROFILE_ASSUMED_SPEED_PERCENT = 25.0
    PROFILE_FALLBACK_SPEED_PERCENT = 10.0
    PROFILE_MOVE_OVERHEAD_SEC = 1.0
    PROFILE_WINDOW_OVERHEAD_SEC = 1.5
    PROFILE_FRAME_RATE_HZ = 12.5
    PROFILE_SEARCH_STEP_SEC = 1.1
    PROFILE_NEAR_ZONE_MM = 15.0
    _PROFILE_CORE_FIELDS = (
        "frame_sequence", "sensor_monotonic_s", "cell_index", "display_row", "column",
        "raw", "calibration", "diff", "diff_percent", "raw_ave", "diff_ave", "diff_percent_ave",
    )

    def _reset_profile_state(self) -> None:
        self._profile_kind = ""
        self._profile_window_kind = ""
        self._profile_step = -1
        self._profile_plan = []
        self._profile_cursor = -1
        self._profile_sweep = 0
        self._profile_direction = ""
        self._profile_target_gap_mm = None
        self._profile_probe = -1
        self._profile_probe_final = False
        self._profile_surface_z_m = None
        self._profile_force_threshold_g = 1.0
        self._profile_target_z_m = None
        self._profile_after_move = ""
        self._profile_move_tolerance_m = self.PROFILE_POSITION_TOLERANCE_M
        self._profile_move_hold_sec = None
        self._profile_stage_started_at = None
        self._profile_move_timeout_sec = 0.0
        self._profile_last_z_m = None
        self._profile_still_since = None
        self._profile_collect_start = 0
        self._profile_seed = None
        self._profile_signal_centre = None
        self._profile_slow_threshold = None
        self._profile_search_mode = None
        self._profile_step_target_z = None
        self._profile_check_forces = []
        self._profile_check_raw = []
        self._profile_check_force_at = None
        self._profile_check_rows = 0

    def _profile_tags(self) -> dict:
        kind = self._profile_kind
        window = kind in {"level", "reference"}
        probe = kind == "surface_probe"
        return {
            "profile_kind": kind,
            "profile_step": self._profile_step if window else "",
            "profile_sweep": self._profile_sweep if window else "",
            "profile_direction": self._profile_direction if window else "",
            "profile_target_gap_mm": (
                self._profile_target_gap_mm if window and self._profile_target_gap_mm is not None else ""
            ),
            "profile_probe": self._profile_probe if probe else "",
            "profile_probe_final": self._profile_probe_final if probe else "",
            "profile_contact_threshold_g": self._profile_force_threshold_g if probe else "",
            "profile_surface_z_m": "" if self._profile_surface_z_m is None else self._profile_surface_z_m,
        }

    def _profile_trimmed_taxel(self, taxel) -> dict:
        """Keep every taxel only in stationary windows; elsewhere keep the target."""
        if not self._phase.startswith("profile_") or self._profile_kind in {"level", "reference"}:
            return taxel
        target = int(self.taxel_index)
        keep = set(self._PROFILE_CORE_FIELDS) | {
            taxel_signal_column(field, target)
            for field in (self.signal_field, "raw", "calibration", "diff", "diff_ave", "diff_percent_ave")
        }
        return {key: value for key, value in taxel.items() if key in keep}

    def _enter_profile_stage(self, phase: str, now: float) -> None:
        self._phase = phase
        self._profile_stage_started_at = float(now)
        self._profile_last_z_m = None
        self._profile_still_since = None

    def _profile_is_still(self, now: float, z_value: float, hold_sec: float) -> bool:
        """True once the TCP has not moved for ``hold_sec``."""
        previous = self._profile_last_z_m
        self._profile_last_z_m = float(z_value)
        if previous is None or abs(float(z_value) - previous) > self.PROFILE_STATIONARY_TOLERANCE_M:
            self._profile_still_since = float(now)
            return False
        return now - float(self._profile_still_since) >= float(hold_sec)

    def _profile_step_size_m(self) -> float:
        return float({
            "far": self.profile_search_step_far_mm,
            "near": self.profile_search_step_near_mm,
            "probe": self.profile_probe_step_mm,
        }[self._profile_search_mode]) / 1000.0

    def _begin_profile(self, host, now: float) -> None:
        baseline = np.asarray(self._force_baseline_samples, dtype=float) / self.NEWTONS_PER_GRAM
        noise_g = float(1.4826 * np.median(np.abs(baseline - np.median(baseline)))) if baseline.size > 1 else 0.0
        self._profile_force_threshold_g = max(float(self.light_contact_threshold_g), 3.0 * noise_g)
        self._far_rest_raw_g = []
        if self._profile_force_threshold_g >= float(self.profile_surface_stop_g):
            self._request_stop(
                host, "profile_force_noise",
                f"Proximity profile stopped: force noise needs a {self._profile_force_threshold_g:.1f} g "
                "contact threshold, at or above the level contact guard. Raise the guard.",
            )
            return
        signals = [
            finite_number(row[2].get(self.signal_field)) for row in self._baseline_rows
        ]
        signals = np.asarray([value for value in signals if value is not None], dtype=float)
        centre = float(np.median(signals)) if signals.size else 0.0
        sigma = float(1.4826 * np.median(np.abs(signals - centre))) if signals.size > 1 else 0.0
        self._profile_signal_centre = centre
        self._profile_slow_threshold = max(self.PROFILE_SLOW_SIGNAL_MIN, self.PROFILE_SLOW_SIGNAL_SIGMA * sigma)
        self._profile_seed = int(datetime.now().timestamp())
        self._profile_search_mode = "far"
        self._profile_step_target_z = float(self._start_tool_position[2])
        self._update_progress(host, "Proximity profile: stepping down to find the surface")
        host._append_sidebar_control_message(
            f"Proximity profile: stepping down {self.profile_search_step_far_mm:.2f} mm at a time until "
            f"the signal drops {self._profile_slow_threshold:.1f} counts, then "
            f"{self.profile_search_step_near_mm:.3f} mm steps until {self._profile_force_threshold_g:.1f} g "
            "light contact. Every step is a position move checked at rest. After the surface is found "
            f"the profile takes about {self.estimate_profile_duration_sec() / 60.0:.0f} min."
        )
        self._profile_next_search_step(host, now)

    def _profile_signal_drop(self) -> Optional[float]:
        """Median signal drop of the last seven recorded frames versus the start baseline."""
        values = [
            finite_number(row.get(self.signal_field)) for row in self._samples[-7:]
            if str(row.get("phase", "")).startswith("profile_")
        ]
        values = [value for value in values if value is not None]
        if len(values) < 3 or self._profile_signal_centre is None:
            return None
        return float(self._profile_signal_centre - np.median(values))

    def _build_profile_plan(self) -> list[dict]:
        reference_gap, levels = self._profile_level_gaps()
        every = max(1, int(self.profile_reference_every))
        rng = np.random.default_rng(self._profile_seed)
        plan = []

        def reference(sweep):
            if not plan or plan[-1]["kind"] != "reference":
                plan.append({"kind": "reference", "gap_mm": reference_gap, "sweep": sweep, "direction": ""})

        for sweep in range(1, max(1, int(self.profile_sweeps)) + 1):
            if self.profile_order == "random":
                ordered, direction = [float(g) for g in rng.permutation(levels)], "random"
            elif self.profile_order == "alternate" and sweep % 2 == 0:
                ordered, direction = list(reversed(levels)), "up"
            else:
                ordered, direction = list(levels), "down"
            reference(sweep)
            for count, gap in enumerate(ordered, 1):
                plan.append({"kind": "level", "gap_mm": gap, "sweep": sweep, "direction": direction})
                if count % every == 0 and count < len(ordered):
                    reference(sweep)
            reference(sweep)
        return plan

    def _profile_move_to(
        self, host, now: float, target_z: float, after: str, *,
        kind: str = "move", tolerance_m: Optional[float] = None, hold_sec: Optional[float] = None,
    ) -> bool:
        """Vertical TM line move at the start XY and orientation."""
        api = self._robot_api(host)
        start = self._finite_values(self._start_tool_position, 3)
        quaternion = self._finite_values((self.initial_position or {}).get("tool_quaternion_wxyz"), 4)
        pose = self._read_tool_pose(api)
        if api is None or not hasattr(api, "send_positions_tool_position") or start is None or pose is None:
            self._request_stop(host, "robot_unavailable", "Proximity profile stopped: TCP position control is unavailable.")
            return False
        if quaternion is None:
            quaternion = self._finite_values(pose[1], 4)
        target = [float(start[0]), float(start[1]), float(target_z)]
        speed_m_s = float(self.profile_move_speed_mm_s) / 1000.0
        percent = None
        getter = getattr(api, "get_project_speed_percent", None)
        if callable(getter):
            try:
                percent = finite_number(getter())
            except Exception:
                percent = None
        effective = speed_m_s * max(1.0, percent if percent is not None else 10.0) / 100.0
        distance = abs(float(target_z) - float(pose[0][2]))
        self._profile_move_timeout_sec = 30.0 + 3.0 * distance / max(effective, 1e-6)
        self._stop_robot(api)
        try:
            sent = bool(api.send_positions_tool_position(
                target, list(quaternion), velocity=speed_m_s, acc_time=0.2,
                blend_percentage=0, fine_goal=True, motion_type=self.PROFILE_LINE_MOTION,
            ))
        except Exception:
            sent = False
        if not sent:
            self._request_stop(host, "command_failed", "Proximity profile stopped: the robot rejected a vertical line move.")
            return False
        self._profile_target_z_m = float(target_z)
        self._profile_after_move = after
        self._profile_move_tolerance_m = (
            self.PROFILE_POSITION_TOLERANCE_M if tolerance_m is None else float(tolerance_m)
        )
        self._profile_move_hold_sec = hold_sec
        self._profile_kind = kind
        self._enter_profile_stage("profile_move", now)
        return True

    def _tick_profile_move(self, host, now: float, z_value: float) -> bool:
        """True once the TCP is still within tolerance of the line-move target."""
        hold = self.velocity_rest_sec if self._profile_move_hold_sec is None else self._profile_move_hold_sec
        still = self._profile_is_still(now, z_value, hold)
        error = abs(float(self._profile_target_z_m) - float(z_value))
        close = error <= self._profile_move_tolerance_m or (
            now - float(self._profile_stage_started_at) >= self.STEP_RELAXED_AFTER_SEC
            and error <= self.STEP_REPEATABILITY_M
        )
        if close and still:
            return True
        if now - float(self._profile_stage_started_at) > self._profile_move_timeout_sec:
            self._request_stop(
                host, "profile_move_timeout",
                f"Proximity profile stopped: the line move to Z {self._profile_target_z_m * 1000.0:.3f} mm "
                f"did not finish within {self._profile_move_timeout_sec:.0f} s.",
            )
        return False

    def _profile_next_search_step(self, host, now: float) -> None:
        """One downward position step of the surface search or a probe."""
        step = self._profile_step_size_m()
        self._profile_step_target_z -= step
        self._profile_move_to(
            host, now, self._profile_step_target_z, "check",
            kind="surface_probe" if self._profile_search_mode == "probe" else "move",
            tolerance_m=max(0.000003, 0.5 * step), hold_sec=self.PROFILE_STEP_SETTLE_SEC,
        )

    def _start_profile_probe(self, host, now: float, final: bool, index: int, contact_z: float) -> None:
        self._profile_probe_final = bool(final)
        self._profile_probe = int(index)
        self._profile_search_mode = "probe"
        self._profile_step_target_z = float(contact_z) + self.PROFILE_PROBE_START_M
        self._profile_move_to(host, now, self._profile_step_target_z, "probe_settle")

    def _finish_profile_probe(self, host, now: float, z_value: float) -> None:
        count = int(self.profile_surface_probes)
        if self._profile_probe + 1 < count:
            self._start_profile_probe(host, now, self._profile_probe_final, self._profile_probe + 1, z_value)
            return
        surface = surface_reference(self._samples, self._profile_force_threshold_g)
        if self._profile_probe_final:
            self._profile_search_mode = None
            host._append_sidebar_control_message(
                "Surface re-measured at the end"
                + (f": shift {surface['final_shift_mm']:+.3f} mm." if surface["final_shift_mm"] is not None else ".")
            )
            self._profile_move_to(host, now, float(self._start_tool_position[2]), "finish")
            return
        if surface["surface_z_m"] is None:
            self._request_stop(host, "profile_surface_unresolved", "Proximity profile stopped: no light-contact crossing was found.")
            return
        self._profile_search_mode = None
        self._profile_surface_z_m = float(surface["surface_z_m"])
        spread = surface["surface_sd_mm"]
        host._append_sidebar_control_message(
            f"Surface Z {self._profile_surface_z_m * 1000.0:.3f} mm from {count} probe(s)"
            + (f", SD {spread:.3f} mm." if spread is not None else ".")
        )
        self._profile_plan = self._build_profile_plan()
        self._profile_cursor = -1
        self._advance_profile_step(host, now)

    def _advance_profile_step(self, host, now: float) -> None:
        self._profile_cursor += 1
        if self._profile_cursor >= len(self._profile_plan):
            if self.profile_remeasure_surface:
                self._start_profile_probe(host, now, True, 0, self._profile_surface_z_m)
            else:
                self._profile_move_to(host, now, float(self._start_tool_position[2]), "finish")
            return
        step = self._profile_plan[self._profile_cursor]
        self._profile_step = self._profile_cursor
        self._profile_window_kind = step["kind"]
        self._profile_sweep = step["sweep"]
        self._profile_direction = step["direction"]
        self._profile_target_gap_mm = float(step["gap_mm"])
        self._profile_move_to(host, now, self._profile_surface_z_m + step["gap_mm"] / 1000.0, "settle")
        self._update_progress(
            host,
            f"Proximity profile sweep {step['sweep']}/{self.profile_sweeps}: "
            f"{step['kind']} at {step['gap_mm']:g} mm ({self._profile_cursor + 1}/{len(self._profile_plan)})",
        )

    def _tick_profile_check(self, host, now: float, z_value: float, force_g: float) -> None:
        """Decide the next search step from fresh force samples taken at rest."""
        stamp = self._force_sample_source_at
        if (
            stamp is not None and stamp != self._profile_check_force_at
            and now - float(self._profile_stage_started_at) >= self.PROFILE_FORCE_LAG_SEC
        ):
            self._profile_check_force_at = stamp
            self._profile_check_forces.append(float(force_g))
            if self._force_raw_g_now is not None:
                self._profile_check_raw.append(self._force_raw_g_now)
        if len(self._profile_check_forces) < self.PROFILE_CHECK_FORCE_SAMPLES or len(self._samples) <= self._profile_check_rows:
            if now - float(self._profile_stage_started_at) > self.PROFILE_CHECK_TIMEOUT_SEC:
                self._request_stop(host, "profile_check_timeout", "Proximity profile stopped: no fresh force and sensor data at rest.")
            return
        force = float(np.median(self._profile_check_forces))
        mode = self._profile_search_mode
        if force >= self._profile_force_threshold_g:
            if mode == "probe":
                self._finish_profile_probe(host, now, z_value)
                return
            self._profile_surface_z_m = float(z_value)
            host._append_sidebar_control_message(
                f"Light contact {force:.1f} g at Z {z_value * 1000.0:.3f} mm; starting "
                f"{int(self.profile_surface_probes)} light-contact probe(s) in "
                f"{self.profile_probe_step_mm:.3f} mm steps."
            )
            self._start_profile_probe(host, now, False, 0, z_value)
            return
        if mode == "far":
            self._track_far_force_zero(
                float(np.median(self._profile_check_raw)) if self._profile_check_raw else None
            )
            drop = self._profile_signal_drop()
            if drop is not None and drop >= self._profile_slow_threshold:
                self._profile_search_mode = "near"
                host._append_sidebar_control_message(
                    f"Proximity signal dropped {drop:.1f} counts at Z {z_value * 1000.0:.2f} mm; "
                    f"continuing in {self.profile_search_step_near_mm:.3f} mm steps to light contact."
                )
                self._begin_profile_force_zero(now, "near surface search")
                return
        self._profile_next_search_step(host, now)

    def _begin_profile_force_zero(self, now: float, purpose: str) -> None:
        self._start_force_zero(now, purpose)
        self._enter_profile_stage("profile_zero", now)

    def _finish_profile_force_zero(self, host, now: float) -> None:
        zero = self._poll_force_zero(now)
        if zero is None:
            return
        self._profile_force_threshold_g = zero["threshold_g"]
        host._append_sidebar_control_message(self._force_zero_message(zero))
        if self._profile_force_threshold_g >= float(self.profile_surface_stop_g):
            self._request_stop(
                host, "profile_force_noise",
                f"Proximity profile stopped: the force at rest needs a {self._profile_force_threshold_g:.1f} g "
                "light-contact threshold, at or above the level contact guard. Secure the force-meter cable.",
            )
            return
        if self._profile_search_mode == "probe":
            self._profile_kind = "surface_probe"
        self._profile_next_search_step(host, now)

    def _tick_profile(self, host, now: float, tool_position, force_g: float) -> None:
        phase = self._phase
        z_value = float(tool_position[2])
        searching = self._profile_search_mode is not None
        moving_up = phase == "profile_move" and float(self._profile_target_z_m) > z_value
        if searching and not moving_up and force_g >= self.PROFILE_SEARCH_FORCE_LIMIT_G:
            self._request_stop(
                host, "profile_search_force",
                f"Proximity profile stopped: {force_g:.0f} g during the surface search. "
                "Inspect the setup; there is no automatic retraction.",
            )
            return
        if not searching and not moving_up and force_g >= float(self.profile_surface_stop_g):
            self._request_stop(
                host, "profile_unexpected_contact",
                f"Proximity profile stopped: {force_g:.1f} g contact away from the surface probes. "
                "Inspect the setup; there is no automatic retraction.",
            )
            return
        if phase == "profile_move":
            if not self._tick_profile_move(host, now, z_value):
                return
            after = self._profile_after_move
            if after == "finish":
                self._profile_kind = ""
                self._stop_reason = "profile_complete"
                self._begin_return_to_initial(
                    host, now,
                    purpose="complete" if self._trial_number >= self._batch_total else "next_trial",
                )
            elif after == "check":
                self._profile_check_forces = []
                self._profile_check_raw = []
                self._profile_check_force_at = self._force_sample_source_at
                self._profile_check_rows = len(self._samples)
                self._enter_profile_stage("profile_check", now)
            elif after == "probe_settle":
                # Rows before the new zero use the old one and must not count as probe samples.
                self._profile_kind = "force_zero"
                self._enter_profile_stage("profile_settle_probe", now)
            else:
                self._profile_kind = "settle"
                self._enter_profile_stage("profile_settle", now)
            return
        if phase == "profile_check":
            self._tick_profile_check(host, now, z_value, force_g)
            return
        if phase == "profile_zero":
            self._finish_profile_force_zero(host, now)
            return
        if phase in {"profile_settle", "profile_settle_probe"}:
            probe = phase == "profile_settle_probe"
            settle = self.PROFILE_SURFACE_SETTLE_SEC if probe else float(self.profile_settle_sec)
            if now - float(self._profile_stage_started_at) < settle:
                return
            if probe:
                label = "final " if self._profile_probe_final else ""
                self._begin_profile_force_zero(now, f"{label}light-contact probe {self._profile_probe + 1}")
            else:
                self._profile_kind = self._profile_window_kind
                self._profile_collect_start = len(self._samples)
                self._enter_profile_stage("profile_collect", now)
            return
        if phase == "profile_collect":
            if len(self._samples) - self._profile_collect_start >= int(self.profile_frames_per_level):
                self._advance_profile_step(host, now)
            elif now - float(self._profile_stage_started_at) > max(
                self.PROFILE_COLLECT_TIMEOUT_SEC, int(self.profile_frames_per_level) / 2.0
            ):
                self._request_stop(host, "profile_frames_too_slow", "Proximity profile stopped: too few fresh sensor frames.")

    def _far_field_allowed(self, tool_position, force_delta_abs_g) -> bool:
        """Fast approach only well above this taxel's measured contact height."""
        if self._slow_zone_entered:
            return False
        allowed = (
            float(self.far_field_speed_mm_s) / 1000.0 > abs(float(self._trial_speed_m_s))
            and self._expected_contact_z_m is not None
            and force_delta_abs_g < 1.0
            and float(tool_position[2]) > self._expected_contact_z_m + float(self.slow_zone_mm) / 1000.0
        )
        # Once the slow zone is entered it is kept for the rest of the approach.
        self._slow_zone_entered = not allowed
        return allowed

    def on_stop(self, host, show_result: bool = True) -> None:
        was_active = self._test_active
        self._stop_robot(self._robot_api(host))
        if was_active:
            self._capture_stop_details(complete=self._batch_completed)
        self._clear_post_save_wait()
        self._test_active = False
        self._phase = "idle"
        self._force_connection_started_at = None
        self._approach_started_at = None
        self._contact_hold_started_at = None
        self._return_best_remaining_m = None
        self._return_last_progress_at = None
        self._return_purpose = ""
        self._force_recovery_started_at = None
        self._force_recovery_restart_attempted = False
        self._clear_force_pause_state()
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
            profile_trials = [
                trial for trial in (self.last_result.get("trial_results") or [self.last_result])
                if trial.get("proximity_profile") is not None
            ]
            if show_result and profile_trials and hasattr(host, "_show_proximity_profile_result"):
                host._show_proximity_profile_result(profile_trials[-1])
            elif show_result and hasattr(host, "_show_calibration_result"):
                host._show_calibration_result(self.last_result)


class FullTaxelCalibrationTask(CalibrationTask):
    """Run the force calibration at every measured pose in a saved map.

    The robot changes XY only at the map's saved clearance height.  One
    repetition visits every measured taxel once.  Further repetitions start
    again at the first taxel after that full scan finishes.  Recordings
    retain only the pressed taxel plus synchronized force/robot/time data.
    Offline comparisons group by physical target and acquisition conditions.
    """

    id = "full_taxel_calibration"
    label = "Calibration (Full Taxel)"
    show_in_sidebar = False
    description = (
        "Load a Taxel Position Mapping result, visit every measured taxel at "
        "the saved clearance height, and run the normal force calibration "
        "for each taxel. Repeated runs scan the whole set before starting "
        "again. Each trial saves only the tested taxel's signals plus force, "
        "position and timing, with a separate graph per taxel."
    )
    start_button_label = "Run Full Taxel Calibration"
    initial_position_name = "Full Taxel Calibration"
    # RobotController accepts 2 (PTP) or 4 (straight Cartesian line).
    # Clearance routing must remain Cartesian so XY motion stays at safe Z.
    LINE_MOTION = 4

    def __init__(
        self,
        output_directory=_DEFAULT_STORAGE_PATH,
        storage_path=_DEFAULT_STORAGE_PATH,
    ) -> None:
        # Keep the argument for callers of the old API, but neither read nor
        # overwrite the legacy home file. The loaded map owns the home pose.
        super().__init__(
            storage_path=None, output_directory=output_directory
        )
        self._home_position: Optional[dict] = None
        self.position_map: Optional[dict] = None
        self.position_map_path = ""
        self.line_speed_m_s = 0.02
        self.route_force_guard_g = 20.0
        self.position_tolerance_m = 0.0001
        self.move_settle_sec = 0.25
        self.move_timeout_sec = 120.0
        self.stationary_tolerance_m = 0.00001
        self._targets: list[dict] = []
        self._target_cursor = -1
        self._pass_number = 1
        self._current_target: Optional[dict] = None
        self._completed_taxels: list[int] = []
        self._trials_by_taxel: dict[int, list] = {}
        self._full_target_results: list[dict] = []
        self._full_run_id = ""
        self._full_directory: Optional[Path] = None
        self._full_completed = False
        self._full_move_target: Optional[np.ndarray] = None
        self._full_move_action = ""
        self._full_move_detail = ""
        self._full_move_started_at: Optional[float] = None
        self._full_move_arrived_at: Optional[float] = None
        self._full_move_last_position: Optional[np.ndarray] = None
        self._full_route_queue: list[tuple[np.ndarray, str, str]] = []
        self._routing_force_baseline_n: Optional[float] = None
        self._full_recalibration_started_at: Optional[float] = None
        self._pending_target_route: Optional[tuple[dict, str]] = None
        self.resume_plan = None
        self.resume_setup_confirmed = False
        self._resume_previous_settings = None
        self._resume_history = []
        self._resume_completed = set()
        self._resume_at_anchor = False
        self._run_lock = None
        self._run_settings = None

    def load_resume(self, source_path) -> tuple[bool, str]:
        """Stage a stopped run without changing files, connecting or moving."""
        if self._test_active or self._phase == "connecting_force_meter":
            return False, "Stop Full Taxel Calibration before loading a saved run."
        try:
            defaults = full_taxel_resume.settings_snapshot(
                FullTaxelCalibrationTask(output_directory=None)
            )
            plan = full_taxel_resume.load_plan(source_path, self.prepare_position_map, defaults)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return False, f"Cannot resume: {exc}"
        if self._resume_previous_settings is None:
            self._resume_previous_settings = (
                full_taxel_resume.settings_snapshot(self), self.loading_auto_plan
            )
        for name, value in plan["settings"].items():
            setattr(self, name, copy.deepcopy(value))
        # Never silently recompute saved travel/time limits when resuming.
        self.loading_auto_plan = False
        self.sliding_path_confirmed = False
        self.position_map = copy.deepcopy(plan["position_map"])
        self.position_map_path = plan["map_source"]
        self.sensor_rows, self.sensor_columns = plan["summary"]["sensor_shape"]
        self._home_position = None
        self._use_map_anchor_as_home()
        self.resume_plan = plan
        self.resume_setup_confirmed = False
        self.start_button_label = "Resume Full Taxel Calibration"
        self._targets = copy.deepcopy(self.position_map["positions"])
        self._trials_by_taxel = copy.deepcopy(plan["trials"])
        self._completed_taxels = [i for i, trials in plan["trials"].items() if len(trials) == self.repeat_count]
        self._current_target = None
        self._target_cursor = -1
        self._full_completed = False
        self._full_directory = None
        self._full_target_results = []
        self.last_result = None
        p, i = plan["remaining"][0]
        self._pass_number = self._trial_number = p
        self._batch_total = int(self.repeat_count)
        detail = (f"{len(plan['completed'])} completed trials preserved · "
                  f"{len(plan['remaining'])} remaining · next: pass {p}, Taxel {i}. "
                  "Interrupted trials restart from clearance. Keep the sensor, probe and mounting unchanged. "
                  "Resume lifts an unloaded probe to the safe anchor before continuing. Keep that path clear.")
        if plan["legacy"]:
            detail += " Older run: recorded settings recovered; unrecorded timing/safety options use current defaults."
        return True, detail

    def clear_resume(self):
        if self._test_active or self._phase == "connecting_force_meter":
            return
        if self._resume_previous_settings is not None:
            settings, auto = self._resume_previous_settings
            for name, value in settings.items():
                setattr(self, name, copy.deepcopy(value))
            self.loading_auto_plan = auto
        self.resume_plan = self._resume_previous_settings = None
        self.start_button_label = "Run Full Taxel Calibration"
        self._targets, self._trials_by_taxel, self._completed_taxels = [], {}, []

    def _write_run_manifest(self, status):
        if self._full_directory is None or self._run_settings is None:
            return
        full_taxel_resume.atomic_json(self._full_directory / full_taxel_resume.MANIFEST, {
            "version": 1, "run_id": self._full_run_id, "status": status,
            "settings": self._run_settings, "position_map": self.position_map,
            "settings_digest": full_taxel_resume.digest(self._run_settings),
            "map_digest": full_taxel_resume.digest(self.position_map),
            "resume_history": self._resume_history,
        })

    def _resume_selection_error(self, host):
        if self.resume_plan is not None:
            if not self.resume_setup_confirmed:
                return "Confirm the unchanged sensor/probe setup and clear unloaded route before Resume."
            if (full_taxel_resume.settings_snapshot(self) != self.resume_plan["settings"]
                    or self.loading_auto_plan
                    or self.position_map != self.resume_plan["position_map"]):
                return "Saved resume settings/map changed. Reload the saved run first."
            key, _label, rows, columns = connected_calibration_sensor(host)
            expected = self.resume_plan["summary"]
            if key != expected.get("sensor_identity") or [rows, columns] != expected["sensor_shape"]:
                return "Connect the same identified sensor used in this saved run before resuming."
            if self.output_directory is None:
                return "Resume requires recording storage to be enabled."
        return ""

    def on_start(self, host) -> bool:
        if self._test_active or self._phase == "connecting_force_meter":
            host._append_sidebar_control_message("Stop the active calibration before starting again.")
            return False
        if error := self._resume_selection_error(host):
            return self._fail_start(host, error)
        if self.resume_plan is not None:
            if self.protocol == "sliding":
                # The resume confirmation explicitly covers the sliding path.
                self.sliding_path_confirmed = True
        self._force_baseline_n = None
        self._routing_force_baseline_n = None
        return super().on_start(host)

    def capture_initial_position(self, host) -> bool:
        host._append_sidebar_control_message(
            "Full Taxel Calibration uses the loaded map's safe anchor as its home. "
            "Update or relocate the map to change this position."
        )
        return False

    def _use_map_anchor_as_home(self) -> None:
        """Select the validated map anchor without persisting or moving anything."""
        if self.position_map is None or revision_motion_error(self.position_map):
            self.initial_position = None
            return
        pose = copy.deepcopy(self.position_map["anchor_pose"])
        pose["source"] = "taxel_position_map_anchor"
        pose["source_path"] = self.position_map_path
        self.initial_position = pose

    def use_connected_sensor(self, host) -> str:
        if self._test_active:
            return str(self.active_sensor_key_name or "")
        key = super().use_connected_sensor(host)
        if position_map_sensor_error(host, self.position_map):
            self.initial_position = None
        else:
            self._use_map_anchor_as_home()
        return key

    @staticmethod
    def _map_vector(values, count: int, name: str) -> list[float]:
        try:
            array = np.asarray(values, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {name} in the taxel position map.") from None
        if array.size != int(count) or not np.all(np.isfinite(array)):
            raise ValueError(f"Invalid {name} in the taxel position map.")
        return [float(value) for value in array]

    @classmethod
    def prepare_position_map(cls, result: dict) -> dict:
        """Validate and normalize the motion data needed by this task."""
        if isinstance(result, dict) and revision_motion_error(result):
            raise ValueError(revision_motion_error(result))
        if not isinstance(result, dict) or result.get("task") != "taxel_position_mapping":
            raise ValueError("Select a taxel_position_map.json recording.")
        settings = result.get("settings")
        if not isinstance(settings, dict):
            raise ValueError("The position map must describe the sensor grid.")
        try:
            sensor_rows = int(settings.get("sensor_rows"))
            sensor_columns = int(settings.get("sensor_columns"))
        except (TypeError, ValueError):
            raise ValueError("The position map must describe the sensor grid.") from None
        taxel_count = sensor_rows * sensor_columns
        if sensor_rows <= 0 or sensor_columns <= 0 or taxel_count <= 0:
            raise ValueError("The position map must describe the sensor grid.")
        anchor = result.get("anchor_pose")
        try:
            anchor_index = int(anchor.get("taxel_index", -1)) if isinstance(anchor, dict) else -1
        except (TypeError, ValueError):
            anchor_index = -1
        if not isinstance(anchor, dict) or not 0 <= anchor_index < taxel_count:
            raise ValueError("The position map anchor taxel is outside the sensor grid.")
        if (sensor_rows, sensor_columns) == (8, 10) and anchor_index != 35:
            raise ValueError("The position map needs its original Taxel 35 anchor.")
        anchor_position = cls._map_vector(
            anchor.get("tool_position_m"), 3, "anchor position"
        )
        quaternion = np.asarray(
            cls._map_vector(
                anchor.get("tool_quaternion_wxyz"), 4, "tool orientation"
            ),
            dtype=float,
        )
        quaternion_norm = float(np.linalg.norm(quaternion))
        if not 0.99 <= quaternion_norm <= 1.01:
            raise ValueError("The saved tool orientation is not a unit quaternion.")
        quaternion /= quaternion_norm
        try:
            xy_limit_m = float(settings["max_xy_offset_mm"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            raise ValueError("The position map has no valid XY workspace limit.") from None
        if not np.isfinite(xy_limit_m) or not 0.0 < xy_limit_m <= 0.2:
            raise ValueError("The saved XY workspace limit must be within 200 mm.")
        try:
            default_probe_z = float(result["probe_start_z_m"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("The position map has no valid probe-start height.") from None
        if not np.isfinite(default_probe_z):
            raise ValueError("The position map has no valid probe-start height.")

        positions = result.get("positions")
        if not isinstance(positions, list):
            raise ValueError("The position map has no measured-position list.")
        measured = []
        seen = set()
        for source in positions:
            if not isinstance(source, dict) or source.get("status") != "measured":
                continue
            index = source.get("taxel_index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < taxel_count:
                raise ValueError(
                    f"Taxel indices must be integers from 0 to {taxel_count - 1}."
                )
            if index in seen:
                raise ValueError(f"Taxel {index} appears more than once in the map.")
            seen.add(index)
            if (source.get("row"), source.get("column")) != (
                index % sensor_rows,
                index // sensor_rows,
            ):
                raise ValueError(f"Taxel {index} has inconsistent row/column numbering.")
            centre = np.asarray(
                cls._map_vector(
                    source.get("tool_position_m"), 3, f"Taxel {index} centre"
                ),
                dtype=float,
            )
            safe = np.asarray(
                cls._map_vector(
                    source.get("safe_approach_position_m"),
                    3,
                    f"Taxel {index} clearance pose",
                ),
                dtype=float,
            )
            orientation = np.asarray(
                cls._map_vector(
                    source.get("tool_quaternion_wxyz"),
                    4,
                    f"Taxel {index} orientation",
                ),
                dtype=float,
            )
            orientation_norm = float(np.linalg.norm(orientation))
            if orientation_norm < 1e-9 or abs(
                float(np.dot(orientation / orientation_norm, quaternion))
            ) < 1.0 - 1e-6:
                raise ValueError(f"Taxel {index} has an inconsistent tool orientation.")
            if (
                float(np.linalg.norm(safe[:2] - centre[:2])) > 1e-6
                or abs(float(safe[2]) - float(anchor_position[2])) > 1e-6
                or float(safe[2]) < float(centre[2])
            ):
                raise ValueError(f"Taxel {index} has an inconsistent clearance pose.")
            if float(np.linalg.norm(safe[:2] - np.asarray(anchor_position[:2]))) > xy_limit_m + 1e-9:
                raise ValueError(f"Taxel {index} is outside the saved XY workspace limit.")
            try:
                probe_z = float(source.get("probe_start_z_m", default_probe_z))
            except (TypeError, ValueError):
                probe_z = float("nan")
            if (
                not np.isfinite(probe_z)
                or probe_z > float(safe[2]) + 1e-9
                or probe_z < float(centre[2]) - 1e-9
            ):
                raise ValueError(f"Taxel {index} has an invalid calibration start height.")
            item = copy.deepcopy(source)
            item["tool_position_m"] = centre.tolist()
            item["safe_approach_position_m"] = safe.tolist()
            item["calibration_start_position_m"] = [
                float(centre[0]),
                float(centre[1]),
                # A mapping probe plane can already contain proximity.
                # Start characterisation at the operator-provided clear anchor.
                float(safe[2]),
            ]
            item["tool_quaternion_wxyz"] = (orientation / orientation_norm).tolist()
            measured.append(item)
        if not measured:
            raise ValueError("The position map contains no measured taxels.")

        # Begin at the Taxel 35 anchor when it is available, then use a
        # deterministic nearest-neighbour walk over saved physical XY
        # coordinates. This shortens clearance travel without assuming the
        # fitted grid is perfectly regular.
        remaining = list(measured)
        ordered = []
        current_xy = np.asarray(anchor_position[:2], dtype=float)
        while remaining:
            next_item = min(
                remaining,
                key=lambda item: (
                    float(
                        np.linalg.norm(
                            np.asarray(item["safe_approach_position_m"][:2])
                            - current_xy
                        )
                    ),
                    int(item["taxel_index"]),
                ),
            )
            remaining.remove(next_item)
            ordered.append(next_item)
            current_xy = np.asarray(
                next_item["safe_approach_position_m"][:2], dtype=float
            )
        measured = ordered
        normalized = copy.deepcopy(result)
        normalized["anchor_pose"] = copy.deepcopy(anchor)
        normalized["anchor_pose"]["tool_position_m"] = anchor_position
        normalized["anchor_pose"]["tool_quaternion_wxyz"] = quaternion.tolist()
        normalized["positions"] = measured
        normalized["calibration_xy_limit_m"] = xy_limit_m
        return normalized

    def load_position_map(self, result: dict, source_path=None) -> tuple[bool, str]:
        if self._test_active or self._phase == "connecting_force_meter":
            return False, "Stop Full Taxel Calibration before replacing its map and home anchor."
        if self.resume_plan is not None:
            return False, "Clear the loaded resume before replacing its saved map."
        try:
            prepared = self.prepare_position_map(result)
        except (KeyError, TypeError, ValueError) as exc:
            return False, str(exc)
        self.position_map = prepared
        self.position_map_path = str(
            source_path or prepared.get("json_path") or ""
        )
        self._home_position = None
        self._use_map_anchor_as_home()
        count = len(prepared["positions"])
        state = "complete" if bool(prepared.get("completed")) else "partial"
        return True, (
            f"Loaded {count} measured taxels from a {state} position map. "
            f"Home follows its safe anchor (Taxel {prepared['anchor_pose']['taxel_index']})."
        )

    def _recording_metadata(self) -> dict:
        return {**super()._recording_metadata(), "recording_scope": "target_taxel_only"}

    def _recorded_taxel_indices(self) -> list[int]:
        return [int(self.taxel_index)]

    def _target_taxel_only(self, row):
        prefix = f"taxel_{int(self.taxel_index):03d}_"
        return {key: value for key, value in row.items()
                if not key.startswith("taxel_") or key.startswith(prefix)}

    def _profile_trimmed_taxel(self, taxel) -> dict:
        # The proximity-profile calculation still needs reference/neighbor
        # channels in memory. Only project its rows AFTER deriving that report.
        # Normal calibration can discard other channels at acquisition time.
        if self.protocol == "proximity_profile":
            return super()._profile_trimmed_taxel(taxel)
        return self._target_taxel_only(taxel)

    def _rows_for_storage(self, rows):
        for row in rows:
            prefix = f"taxel_{int(self.taxel_index):03d}_"
            for key in list(row):
                if key.startswith("taxel_") and not key.startswith(prefix):
                    del row[key]
        return rows

    def _full_folder_name(self, timestamp: str) -> str:
        travel_mm = max(0.0, float(self.max_travel_m)) * 1000.0
        timeout_s = max(0.0, float(self.timeout_sec))
        travel = (
            f"{self._folder_number(travel_mm)}mm" if travel_mm > 0.0 else "unlimited"
        )
        timeout = (
            f"{self._folder_number(timeout_s)}s" if timeout_s > 0.0 else "unlimited"
        )
        settings = (
            f"taxels-{len(self._targets)}_"
            f"signal-{self._folder_token(self.signal_field)}_"
            f"speed-{self._folder_number(abs(self.approach_speed_m_s) * 1000.0)}mmps_"
            f"threshold-{self._force_folder_token()}g_"
            f"travel-{travel}_timeout-{timeout}_"
            f"full-passes-{max(1, int(self.repeat_count))}_"
            f"{self._probe_material_folder_token()}"
        )
        return calibration_folder_name(
            "calibration_full_taxel", timestamp, settings,
            sensor_name=self._recording_metadata()["sensor_name"], protocol=self.protocol,
        )

    def _update_progress(self, host, detail: str = "") -> None:
        callback = getattr(host, "_refresh_full_taxel_calibration_progress", None)
        if callable(callback):
            callback(
                current_taxel=(
                    None if self._current_target is None else int(self.taxel_index)
                ),
                taxel_number=max(0, int(self._target_cursor) + 1),
                taxel_total=len(self._targets),
                trial=max(0, int(self._trial_number)),
                trials=max(1, int(self._batch_total)),
                phase=str(self._phase),
                detail=str(detail),
                completed=list(
                    dict.fromkeys(
                        [
                            *self._completed_taxels,
                            *self._trials_by_taxel.keys(),
                        ]
                    )
                ),
            )

    def _validate_start_prerequisites(self, host) -> bool:
        if not self._validate_measurement_settings(host):
            return False
        revision_error = revision_motion_error(getattr(self, "position_map", None))
        if revision_error:
            return self._fail_start(host, revision_error)
        mismatch = position_map_sensor_error(host, getattr(self, "position_map", None))
        if mismatch:
            return self._fail_start(host, mismatch)
        if self.position_map is None:
            return self._fail_start(
                host, "Load a Taxel Position Mapping JSON before pressing Run."
            )
        map_settings = self.position_map.get("settings") or {}
        try:
            self.sensor_rows = int(map_settings["sensor_rows"])
            self.sensor_columns = int(map_settings["sensor_columns"])
        except (KeyError, TypeError, ValueError):
            return self._fail_start(
                host, "The loaded position map has no sensor grid size."
            )
        self.use_connected_sensor(host)
        self.sensor_rows = int(map_settings["sensor_rows"])
        self.sensor_columns = int(map_settings["sensor_columns"])
        if not 0.001 <= float(self.line_speed_m_s) <= 0.1:
            return self._fail_start(
                host, "Set the safe routing speed between 1 and 100 mm/s."
            )
        if not 0.0 < float(self.route_force_guard_g) <= self.MAX_CONTACT_THRESHOLD_G:
            return self._fail_start(host, "Set a valid routing force guard.")
        api = self._robot_api(host)
        pose = self._read_tool_pose(api)
        if (
            pose is None
            or api is None
            or not callable(getattr(api, "send_positions_tool_position", None))
        ):
            return self._fail_start(
                host,
                "Robot TCP feedback and position control are required for Full Taxel Calibration.",
            )
        anchor = self.position_map["anchor_pose"]
        anchor_position = np.asarray(anchor["tool_position_m"], dtype=float)
        current_quaternion = np.asarray(pose[1], dtype=float)
        saved_quaternion = np.asarray(anchor["tool_quaternion_wxyz"], dtype=float)
        current_norm = float(np.linalg.norm(current_quaternion))
        alignment = (
            0.0
            if current_norm < 1e-9
            else abs(float(np.dot(current_quaternion / current_norm, saved_quaternion)))
        )
        if alignment < float(np.cos(np.radians(2.5))):
            return self._fail_start(
                host,
                "Restore the tool orientation used by the saved taxel map first.",
            )
        sensor_frame = self._read_sensor_frame(host)
        if (
            sensor_frame is None
            or int(getattr(sensor_frame[1], "frame_sequence", 0) or 0) <= 0
        ):
            return self._fail_start(
                host,
                f"A live {self.sensor_rows}x{self.sensor_columns} tactile sensor "
                "is required before calibration.",
            )
        sensor = getattr(self._ui_ros(host), "sensor_functions", None)
        if bool(getattr(sensor, "_sensor_calibration_in_progress", False)):
            return self._fail_start(
                host,
                "Wait for the active sensor calibration to finish before starting.",
            )
        home = self._finite_values(
            (self.initial_position or {}).get("tool_position_m"), 3
        )
        if home is None:
            return self._fail_start(
                host,
                "Load a position map with a valid safe anchor before pressing Run.",
            )
        home_position = np.asarray(home, dtype=float)
        if float(np.linalg.norm(home_position[:2] - anchor_position[:2])) > float(
            self.position_map["calibration_xy_limit_m"]
        ):
            return self._fail_start(
                host,
                "The Full Taxel home is outside this map's workspace. Reload the position map.",
            )
        if self.protocol == "standard" and (error := travel_error(self)):
            return self._fail_start(host, error)
        if self.resume_plan is not None:
            if np.linalg.norm(np.asarray(pose[0][:2]) - anchor_position[:2]) > self.position_map["calibration_xy_limit_m"]:
                return self._fail_start(host, "Resume blocked: move the robot into the recorded sensor workspace first.")
        return True

    def _start_calibration_cycle(self, host) -> bool:
        if error := self._resume_selection_error(host):
            return self._fail_start(host, error)
        if not self._validate_start_prerequisites(host):
            return False
        now = time.monotonic()
        force = self._fresh_force_reading(host, now)
        if force is None:
            return self._fail_start(
                host, "Fresh force-transmitter data is required before robot motion."
            )
        self._trial_wall_started_at = None
        if not self._measurement_guard(host, now):
            return False
        plan = self.resume_plan
        if plan is not None and abs(float(force[0])) / self.NEWTONS_PER_GRAM >= self.route_force_guard_g:
            return self._fail_start(host, "Resume blocked: unload the probe and check the force zero before routing.")
        self._targets = copy.deepcopy(self.position_map["positions"])
        self._contact_z_by_taxel = {}
        self._trial_memory_bytes = 0
        self._target_cursor = -1
        self._pass_number = 1
        self._current_target = None
        self._completed_taxels = []
        self._trials_by_taxel = {}
        self._full_target_results = []
        self._full_completed = False
        self._batch_results = []
        self._samples = []
        self._trial_result_saved = False
        self.last_result = None
        self._stop_reason = ""
        self._resume_completed = set()
        self._resume_history = []
        self._resume_at_anchor = plan is not None
        self._full_run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._full_directory = (
            self.output_directory / self._full_folder_name(self._full_run_id)
            if self.output_directory is not None
            else None
        )
        if plan is not None:
            self._full_run_id = plan["summary"]["run_id"]
            self._full_directory = plan["folder"]
        if self._full_directory is not None:
            try:
                self._full_directory.mkdir(parents=True, exist_ok=True)
                self._run_lock = full_taxel_resume.acquire_run_lock(self._full_directory)
                if plan is not None:
                    checked = full_taxel_resume.load_plan(
                        self._full_directory, self.prepare_position_map,
                        full_taxel_resume.settings_snapshot(FullTaxelCalibrationTask(output_directory=None)),
                    )
                    if checked["signature"] != plan["signature"]:
                        raise ValueError("Saved recordings changed since loading. Reload Resume first.")
                    self._resume_history = copy.deepcopy(plan["history"])
                    self._resume_history.append(full_taxel_resume.archive_attempts(plan))
                    self._resume_completed = set(plan["completed"])
                    self._trials_by_taxel = copy.deepcopy(plan["trials"])
                    self._pass_number, next_index = plan["remaining"][0]
                    self._target_cursor = next(i for i, t in enumerate(self._targets) if t["taxel_index"] == next_index) - 1
                    # Compact existing results: do not re-export completed trial files.
                    for index, trials in self._trials_by_taxel.items():
                        if len(trials) == int(self.repeat_count):
                            directory = self._full_directory / f"taxel_{index:02d}"
                            result = dict(trials[0]) if len(trials) == 1 else {"taxel_index": index, "trial_results": trials}
                            result.update(batch_status="complete", completed_trials=len(trials),
                                          requested_trials=int(self.repeat_count), experiment_directory=str(directory),
                                          batch_summary_path=str(directory / "batch_summary.csv"),
                                          batch_metadata_path=str(directory / "batch_summary.json"))
                            self._full_target_results.append(result)
                            self._completed_taxels.append(index)
                self._run_settings = full_taxel_resume.settings_snapshot(self)
                self._write_run_manifest("running")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                if self._run_lock is not None:
                    self._run_lock.close()
                    self._run_lock = None
                return self._fail_start(
                    host, f"Could not prepare the full-taxel recording folder: {exc}"
                )
        self._home_position = copy.deepcopy(self.initial_position)
        self._test_active = True
        self._force_connection_started_at = None
        self._clear_force_pause_state()
        self._reset_travel_status(host)
        setter = getattr(host, "_set_full_taxel_calibration_controls_running", None)
        if callable(setter):
            setter(True)
        host._append_sidebar_control_message(
            "Full Taxel Calibration will start and finish at the loaded map's safe anchor. "
            "The robot will change XY only at the saved clearance height; "
            "keep the workspace clear and remain beside the emergency stop."
        )
        if plan is not None:
            # Reading many trial files and archiving summaries can block Qt's
            # feedback delivery. Never route using the samples from before I/O.
            self._begin_post_save_wait(host, "resume_home")
            return True
        return self._route_to_saved_home(
            host,
            "full_home_arrived",
            "moving to the map's safe anchor",
        )

    def _start_next_target(self, host) -> bool:
        self._target_cursor += 1
        while True:
            if self._target_cursor >= len(self._targets):
                if self._pass_number >= max(1, int(self.repeat_count)):
                    return self._route_to_anchor_and_finish(host)
                self._pass_number += 1
                self._target_cursor = 0
            index = int(self._targets[self._target_cursor]["taxel_index"])
            if (self._pass_number, index) not in self._resume_completed:
                break
            self._target_cursor += 1
        target = self._targets[self._target_cursor]
        self._current_target = target
        self.taxel_index = int(target["taxel_index"])
        self._batch_total = max(1, int(self.repeat_count))
        self._trial_number = max(1, int(self._pass_number))
        self._batch_id = f"{self._full_run_id}_taxel{self.taxel_index:02d}"
        self._batch_directory = (
            self._full_directory / f"taxel_{self.taxel_index:02d}"
            if self._full_directory is not None
            else None
        )
        self._batch_results = list(self._trials_by_taxel.get(self.taxel_index, []))
        self._batch_completed = False
        self._samples = []
        self._trial_result_saved = False
        self._stop_reason = ""
        start_position = list(target["calibration_start_position_m"])
        self.initial_position = {
            "saved_at": str(self.position_map.get("saved_at", "")),
            "joints_rad": None,
            "tool_position_m": start_position,
            "tool_quaternion_wxyz": list(target["tool_quaternion_wxyz"]),
            "source": "taxel_position_map",
            "source_path": self.position_map_path,
        }
        detail = (
            f"Pass {self._pass_number}/{self._batch_total}: "
            f"Taxel {self.taxel_index} "
            f"({self._target_cursor + 1}/{len(self._targets)}): routing at clearance height"
        )
        return self._route_to_target_start(host, target, detail)

    def _route_to_target_start(self, host, target, detail: str) -> bool:
        force = self._fresh_force_reading(host)
        if force is None:
            self._pending_target_route = (copy.deepcopy(target), str(detail))
            self._phase = "full_waiting_force_route"
            self._begin_force_pause(host, time.monotonic())
            return True
        pose = self._read_tool_pose(self._robot_api(host))
        if pose is None:
            self._request_stop(
                host,
                "robot_feedback_lost",
                "Full Taxel Calibration stopped: TCP feedback was unavailable before routing.",
            )
            return False
        current = np.asarray(pose[0], dtype=float)
        safe = np.asarray(target["safe_approach_position_m"], dtype=float)
        start = np.asarray(target["calibration_start_position_m"], dtype=float)
        travel_z = max(float(current[2]), float(safe[2]))
        waypoints = []
        if current[2] < travel_z - self.position_tolerance_m:
            waypoints.append(np.array([current[0], current[1], travel_z]))
        if float(np.linalg.norm(current[:2] - safe[:2])) > self.position_tolerance_m:
            waypoints.append(np.array([safe[0], safe[1], travel_z]))
        if travel_z > float(safe[2]) + self.position_tolerance_m:
            waypoints.append(safe)
        elif not waypoints or float(np.linalg.norm(waypoints[-1] - safe)) > 1e-9:
            waypoints.append(safe)
        waypoints.append(start)
        self._full_route_queue = [
            (point, "full_route_waypoint", detail) for point in waypoints[:-1]
        ] + [(waypoints[-1], "full_target_ready", detail)]
        self._routing_force_baseline_n = float(force[0])
        return self._advance_full_route(host)

    def _set_routing_force_baseline(self, host) -> None:
        reading = self._fresh_force_reading(host)
        self._routing_force_baseline_n = (
            None if reading is None else float(reading[0])
        )

    def _route_current_target_to_clearance(self, host) -> bool:
        target = self._current_target
        if target is None:
            return False
        safe = np.asarray(target["safe_approach_position_m"], dtype=float)
        self._full_route_queue = [
            (
                safe,
                "full_target_retracted",
                f"Taxel {self.taxel_index}: retracting vertically to the saved clearance height",
            )
        ]
        self._set_routing_force_baseline(host)
        return self._advance_full_route(host)

    def _saved_home_position(self):
        source = self._home_position or self.initial_position
        values = self._finite_values((source or {}).get("tool_position_m"), 3)
        if values is None:
            return None
        return np.asarray(values, dtype=float)

    def _route_to_saved_home(self, host, action: str, detail: str) -> bool:
        """Travel at clearance height to the map anchor captured for this run."""
        home = self._saved_home_position()
        if home is None or self.position_map is None:
            self._request_stop(
                host,
                "internal_error",
                "Full Taxel Calibration stopped: its map anchor is missing.",
            )
            return False
        pose = self._read_tool_pose(self._robot_api(host))
        if pose is None:
            self._request_stop(
                host,
                "robot_feedback_lost",
                "Full Taxel Calibration stopped: TCP feedback was lost before "
                "returning to the map's safe anchor.",
            )
            return False
        current = np.asarray(pose[0], dtype=float)
        if float(np.linalg.norm(current - home)) <= float(self.position_tolerance_m):
            self._dispatch_full_move_action(host, action, time.monotonic())
            return True
        anchor_z = float(self.position_map["anchor_pose"]["tool_position_m"][2])
        travel_z = max(float(current[2]), float(home[2]), anchor_z)
        waypoints = []
        if current[2] < travel_z - self.position_tolerance_m:
            waypoints.append(np.array([current[0], current[1], travel_z]))
        if float(np.linalg.norm(current[:2] - home[:2])) > self.position_tolerance_m:
            waypoints.append(np.array([home[0], home[1], travel_z]))
        if abs(float(travel_z - home[2])) > self.position_tolerance_m:
            waypoints.append(home)
        elif not waypoints or float(np.linalg.norm(waypoints[-1] - home)) > 1e-9:
            waypoints.append(home)
        route = []
        for point in waypoints:
            if route and float(np.linalg.norm(point - route[-1])) <= 1e-9:
                continue
            route.append(point)
        self._full_route_queue = [
            (point, "full_route_waypoint", detail) for point in route[:-1]
        ] + [(route[-1], str(action), detail)]
        self._set_routing_force_baseline(host)
        return self._advance_full_route(host)

    def _route_to_anchor_and_finish(self, host) -> bool:
        if self.position_map is None:
            return False
        return self._route_to_saved_home(
            host,
            "full_final_anchor",
            "Returning to the map's safe anchor",
        )

    def _advance_full_route(self, host) -> bool:
        if not self._full_route_queue:
            self._request_stop(
                host,
                "internal_error",
                "Full Taxel Calibration stopped: the safe route was lost.",
            )
            return False
        position, action, detail = self._full_route_queue.pop(0)
        return self._command_full_move(host, position, action, detail)

    def _command_full_move(self, host, position, action: str, detail: str) -> bool:
        now = time.monotonic()
        if not self._measurement_guard(host, now):
            return False
        if self._fresh_force_reading(host, now) is None:
            self._request_stop(host, "force_stale_before_motion", "Stopped before a new route: force feedback is stale.")
            return False
        api = self._robot_api(host)
        quaternion = self._finite_values(
            (self.position_map or {}).get("anchor_pose", {}).get(
                "tool_quaternion_wxyz"
            ),
            4,
        )
        target = self._finite_values(position, 3)
        if (
            api is None
            or target is None
            or quaternion is None
            or not callable(getattr(api, "send_positions_tool_position", None))
        ):
            self._request_stop(
                host,
                "robot_unavailable",
                "Full Taxel Calibration stopped: robot TCP position control is unavailable.",
            )
            return False
        self._stop_robot(api)
        try:
            sent = bool(
                api.send_positions_tool_position(
                    list(target),
                    list(quaternion),
                    velocity=float(self.line_speed_m_s),
                    acc_time=0.2,
                    blend_percentage=0,
                    fine_goal=True,
                    motion_type=self.LINE_MOTION,
                )
            )
        except Exception as exc:
            sent = False
            detail = f"{detail} ({exc})"
        if not sent:
            self._request_stop(
                host,
                "route_command_failed",
                f"Full Taxel Calibration stopped: robot rejected the route command. {detail}",
            )
            return False
        self._phase = "full_moving"
        self._full_move_target = np.asarray(target, dtype=float)
        self._full_move_action = str(action)
        self._full_move_detail = str(detail)
        self._full_move_started_at = time.monotonic()
        self._full_move_arrived_at = None
        self._full_move_last_position = None
        self._update_progress(host, detail)
        return True

    def _tick_full_move(self, host, now: float) -> None:
        pose = self._read_tool_pose(self._robot_api(host))
        if pose is None or self._full_move_target is None:
            self._request_stop(
                host,
                "robot_feedback_lost",
                "Full Taxel Calibration stopped: TCP feedback was lost during routing.",
            )
            return
        position = np.asarray(pose[0], dtype=float)
        distance = float(np.linalg.norm(position - self._full_move_target))
        previous = self._full_move_last_position
        self._full_move_last_position = position.copy()
        still_moving = previous is None or float(np.linalg.norm(position - previous)) > float(
            self.stationary_tolerance_m
        )
        if distance > float(self.position_tolerance_m) or still_moving:
            self._full_move_arrived_at = None
            if (
                self._full_move_started_at is not None
                and now - float(self._full_move_started_at) > float(self.move_timeout_sec)
            ):
                self._request_stop(
                    host,
                    "route_timeout",
                    "Full Taxel Calibration stopped: a safe routing move timed out.",
                )
            return
        if self._full_move_arrived_at is None:
            self._full_move_arrived_at = float(now)
            return
        if now - float(self._full_move_arrived_at) < float(self.move_settle_sec):
            return
        action = self._full_move_action
        self._full_move_target = None
        self._full_move_action = ""
        self._dispatch_full_move_action(host, action, now)

    def _dispatch_full_move_action(self, host, action: str, now: float) -> None:
        if action == "full_route_waypoint":
            self._advance_full_route(host)
            return
        if action == "full_home_arrived":
            if self._resume_at_anchor:
                self._resume_at_anchor = False
                self._start_full_recalibration(host, now)
            else:
                self._start_next_target(host)
            return
        if action == "full_target_ready":
            pose = self._read_tool_pose(self._robot_api(host))
            if pose is None:
                self._request_stop(
                    host,
                    "robot_feedback_lost",
                    "Full Taxel Calibration stopped: TCP feedback was unavailable at the target.",
                )
                return
            self._begin_trial(host, now, pose[0])
            return
        if action == "full_target_retracted":
            passes = max(1, int(self.repeat_count))
            if self._target_cursor + 1 >= len(self._targets):
                if int(self._pass_number) < passes:
                    self._pass_number += 1
                    self._target_cursor = -1
                    host._append_sidebar_control_message(
                        f"Starting full scan {self._pass_number}/{passes} "
                        "from the first taxel."
                    )
                    self._start_full_recalibration(host, now)
                else:
                    self._route_to_anchor_and_finish(host)
            else:
                self._start_full_recalibration(host, now)
            return
        if action == "full_final_anchor":
            self._full_completed = True
            host._stop_sidebar_control(
                show_result=True,
                message=(
                    "Full Taxel Calibration complete. All mapped taxels were "
                    "saved separately and the robot returned to the map's safe anchor."
                ),
            )

    def _complete_returned_trial(self, host, now: float, purpose: str) -> None:
        self._trial_completed = True
        try:
            result = self._save_current_trial(host)
        except Exception as exc:
            self._request_stop(
                host,
                "save_failed",
                f"Full Taxel Calibration stopped: trial data could not be saved ({exc}).",
            )
            return
        if result is None:
            self._request_stop(
                host,
                "save_failed",
                "Full Taxel Calibration stopped: no trial samples were available after return.",
            )
            return
        index = int(self.taxel_index)
        self._export_trial_graph(host, result)
        self._batch_results.sort(key=lambda item: int(item["trial_number"]))
        self._trials_by_taxel[index] = list(self._batch_results)
        self._resume_completed.add((int(self._pass_number), index))
        passes = max(1, int(self.repeat_count))
        if len(self._batch_results) == passes:
            self._batch_completed = True
            target_result = self._build_final_result()
            if target_result is None:
                self._request_stop(
                    host,
                    "save_failed",
                    "Full Taxel Calibration stopped: the taxel result could not be built.",
                )
                return
            release_trial_rows(target_result)
            self._full_target_results.append(target_result)
            self._completed_taxels.append(index)
            host._append_sidebar_control_message(
                f"Taxel {index} complete "
                f"({len(self._completed_taxels)}/{len(self._targets)}); retracting to clearance."
            )
            self._update_progress(host, "Taxel complete; retracting to clearance")
        else:
            host._append_sidebar_control_message(
                f"Taxel {index} finished pass {self._pass_number}/{passes}; "
                "moving on to the next taxel."
            )
            self._update_progress(
                host,
                f"Pass {self._pass_number}/{passes} sample saved; moving to the next taxel",
            )
        release_trial_rows(result)
        self._samples = []
        self._begin_post_save_wait(host, purpose)

    def _export_trial_graph(self, host, result: dict) -> None:
        """Save the trial PNG while its samples are still in memory."""
        exporter = getattr(host, "_export_calibration_trial_graph", None)
        directory = str(result.get("experiment_directory") or "")
        if not callable(exporter) or not directory or not result.get("rows"):
            return
        number = int(result.get("trial_number", 1) or 1)
        path = Path(directory) / f"trial_{number:04d}.png"
        try:
            saved = exporter(result, path)
        except Exception as exc:
            host._append_sidebar_control_message(
                f"Trial graph not saved yet ({exc}); it is retried when the run ends."
            )
            return
        if saved:
            result["graph_path"] = str(path)

    def _continue_after_saved_trial(self, host, now, purpose):
        if purpose == "resume_home":
            pose = self._read_tool_pose(self._robot_api(host))
            anchor = self.position_map["anchor_pose"]
            key, _label, rows, columns = connected_calibration_sensor(host)
            expected = self.resume_plan["summary"]
            force = self._fresh_force_reading(host, now)
            valid = pose is not None and force is not None
            if valid:
                orientation = np.asarray(pose[1], dtype=float)
                norm = float(np.linalg.norm(orientation))
                alignment = 0. if norm < 1e-9 else abs(float(np.dot(orientation / norm, anchor["tool_quaternion_wxyz"])))
                valid = (
                    np.linalg.norm(np.asarray(pose[0][:2]) - np.asarray(anchor["tool_position_m"][:2])) <= self.position_map["calibration_xy_limit_m"]
                    and alignment >= float(np.cos(np.radians(2.5)))
                    and abs(float(force[0])) / self.NEWTONS_PER_GRAM < self.route_force_guard_g
                    and key == expected.get("sensor_identity")
                    and [rows, columns] == expected["sensor_shape"]
                )
            if not valid:
                self._request_stop(host, "resume_preflight_changed", "Resume stopped before motion: sensor, force, clearance or tool pose changed while loading. Check the setup and reload Resume.")
                return
            self._route_to_saved_home(host, "full_home_arrived", "moving to the original map's safe anchor")
            return
        self._route_current_target_to_clearance(host)

    def _start_full_recalibration(self, host, now: float) -> None:
        if not self.recalibrate_between_trials:
            self._start_next_target(host)
            return
        sensor = getattr(self._ui_ros(host), "sensor_functions", None)
        if sensor is None or not callable(getattr(sensor, "updateCal", None)):
            self._request_stop(
                host,
                "sensor_calibration_unavailable",
                "Full Taxel Calibration stopped: sensor calibration is unavailable.",
            )
            return
        if bool(getattr(sensor, "_sensor_calibration_in_progress", False)):
            self._request_stop(
                host,
                "sensor_calibration_busy",
                "Full Taxel Calibration stopped: another sensor calibration is active.",
            )
            return
        self._phase = "full_recalibrating"
        self._full_recalibration_started_at = float(now)
        self._update_progress(host, "Updating sensor calibration at clearance height")
        try:
            sensor.updateCal()
        except Exception as exc:
            self._request_stop(
                host,
                "sensor_calibration_failed",
                f"Full Taxel Calibration stopped: sensor calibration failed ({exc}).",
            )
            return
        if not bool(getattr(sensor, "_sensor_calibration_in_progress", False)):
            self._request_stop(
                host,
                "sensor_calibration_failed",
                "Full Taxel Calibration stopped: sensor calibration did not start.",
            )

    def _tick_full_recalibration(self, host, now: float) -> None:
        sensor = getattr(self._ui_ros(host), "sensor_functions", None)
        if sensor is None:
            self._request_stop(
                host,
                "sensor_calibration_unavailable",
                "Full Taxel Calibration stopped: sensor feedback became unavailable.",
            )
            return
        if (
            self._full_recalibration_started_at is not None
            and now - float(self._full_recalibration_started_at)
            >= float(self.recalibration_timeout_sec)
        ):
            self._request_stop(
                host,
                "sensor_calibration_timeout",
                "Full Taxel Calibration stopped: sensor calibration timed out.",
            )
            return
        if bool(getattr(sensor, "_sensor_calibration_in_progress", False)):
            return
        if not bool(getattr(sensor, "is_connected", False)):
            self._request_stop(
                host,
                "sensor_calibration_failed",
                "Full Taxel Calibration stopped: sensor calibration failed.",
            )
            return
        if self._read_sensor_frame(host) is None:
            return
        self._full_recalibration_started_at = None
        self._start_next_target(host)

    def _resume_from_force_pause(self, host, now: float) -> None:
        if self._force_paused_phase == "full_waiting_force_route":
            pause_started_at = self._force_pause_started_at
            duration = (
                0.0
                if pause_started_at is None
                else max(0.0, float(now) - float(pause_started_at))
            )
            self._force_pause_total_sec += duration
            pending = self._pending_target_route
            self._pending_target_route = None
            self._clear_force_pause_state()
            if pending is None:
                self._request_stop(
                    host,
                    "route_lost",
                    "Full Taxel Calibration stopped: the pending route could not be recovered.",
                )
                return
            host._append_sidebar_control_message(
                "Force stream is stable again. Starting the pending safe route "
                f"after {duration:.1f} s."
            )
            self._route_to_target_start(host, pending[0], pending[1])
            return
        if self._force_paused_phase != "full_moving":
            super()._resume_from_force_pause(host, now)
            return
        pause_started_at = self._force_pause_started_at
        duration = (
            0.0
            if pause_started_at is None
            else max(0.0, float(now) - float(pause_started_at))
        )
        self._force_pause_total_sec += duration
        target = None if self._full_move_target is None else self._full_move_target.copy()
        action = self._full_move_action
        detail = self._full_move_detail
        self._clear_force_pause_state()
        if target is None:
            self._request_stop(
                host,
                "route_lost",
                "Full Taxel Calibration stopped: the paused route could not be recovered.",
            )
            return
        self._routing_force_baseline_n = float(self._fresh_force_reading(host, now)[0])
        host._append_sidebar_control_message(
            f"Force stream is stable again. Resuming the saved clearance route after {duration:.1f} s."
        )
        self._command_full_move(host, target, action, detail)

    def on_tick(self, host) -> None:
        now = time.monotonic()
        if self._tick_post_save_wait(host, now):
            return
        if self._test_active and not self._measurement_guard(
            host, now, sensor_required=self._phase != "full_recalibrating"
        ):
            return
        if self._phase == "full_moving":
            force = self._fresh_force_reading(host, now)
            if force is None:
                self._begin_force_pause(host, now)
                return
            if self._routing_force_baseline_n is not None:
                delta_g = abs(float(force[0]) - self._routing_force_baseline_n) / self.NEWTONS_PER_GRAM
                if delta_g >= float(self.route_force_guard_g):
                    self._request_stop(
                        host,
                        "route_force_guard",
                        f"Full Taxel Calibration stopped: routing force changed by {delta_g:.1f} g.",
                    )
                    return
            self._tick_full_move(host, now)
            return
        if self._phase == "full_recalibrating":
            self._tick_full_recalibration(host, now)
            return
        super().on_tick(host)

    def _save_full_summary(self) -> tuple[str, str]:
        if self._full_directory is None:
            return "", ""
        rows = []
        for target_result in self._full_target_results:
            taxel = int(target_result.get("taxel_index", -1))
            trials = list(target_result.get("trial_results") or [target_result])
            for trial in trials:
                row = dict(trial.get("summary") or self._summarize_trial(trial))
                row = {"taxel_index": taxel, **row}
                rows.append(row)
        summary_path = self._full_directory / "full_taxel_summary.csv"
        if rows:
            with summary_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        metadata_path = self._full_directory / "full_taxel_summary.json"
        metadata = {
            "version": 1,
            "resume_history": copy.deepcopy(self._resume_history),
            **self._recording_metadata(),
            **self._stop_details(),
            "last_taxel_index": self._stop_taxel_index if self._stop_taxel_index is not None else int(self.taxel_index),
            "last_pass_number": self._stop_pass_number if self._stop_pass_number is not None else int(self._pass_number),
            "task": self.id,
            "run_id": self._full_run_id,
            "status": "complete" if self._full_completed else "aborted",
            "sensor_shape": [self.sensor_rows, self.sensor_columns],
            "position_map_path": self.position_map_path,
            "home_position": copy.deepcopy(self._home_position),
            "requested_taxels": [int(item["taxel_index"]) for item in self._targets],
            "completed_taxels": list(self._completed_taxels),
            "repetitions_per_taxel": int(self.repeat_count),
            "signal_field": self.signal_field,
            "approach_speed_m_s": float(self.approach_speed_m_s),
            "route_speed_m_s": float(self.line_speed_m_s),
            "route_force_guard_g": float(self.route_force_guard_g),
            "contact_threshold_g": float(self.contact_threshold_g),
            "max_travel_m": float(self.max_travel_m),
            "timeout_sec": float(self.timeout_sec),
            "summary_csv_path": str(summary_path) if rows else "",
            "taxel_results": [
                {
                    "taxel_index": int(item.get("taxel_index", -1)),
                    "batch_status": item.get("batch_status"),
                    "completed_trials": item.get("completed_trials"),
                    "requested_trials": item.get("requested_trials"),
                    "experiment_directory": item.get("experiment_directory"),
                    "batch_summary_path": item.get("batch_summary_path"),
                }
                for item in self._full_target_results
            ],
        }
        full_taxel_resume.atomic_json(metadata_path, metadata)
        return str(summary_path) if rows else "", str(metadata_path)

    def _build_full_result(self) -> Optional[dict]:
        if not self._full_target_results and self._full_directory is None:
            return None
        summary_path, metadata_path = self._save_full_summary()
        return {
            **self._recording_metadata(),
            **self._stop_details(),
            "task": self.id,
            "run_id": self._full_run_id,
            "status": "complete" if self._full_completed else "aborted",
            "sensor_shape": [self.sensor_rows, self.sensor_columns],
            "position_map_path": self.position_map_path,
            "home_position": copy.deepcopy(self._home_position),
            "requested_taxels": len(self._targets),
            "completed_taxels": len(self._completed_taxels),
            "completed_taxel_indices": list(self._completed_taxels),
            "repetitions_per_taxel": int(self.repeat_count),
            "signal_field": self.signal_field,
            "experiment_directory": (
                "" if self._full_directory is None else str(self._full_directory)
            ),
            "summary_path": summary_path,
            "metadata_path": metadata_path,
            "taxel_results": list(self._full_target_results),
        }

    def _flush_partial_taxel_results(self, host) -> None:
        """Write a summary for every taxel that has trials but no final result."""
        saved = {
            int(item.get("taxel_index", -1)) for item in self._full_target_results
        }
        passes = max(1, int(self.repeat_count))
        for index, trials in list(self._trials_by_taxel.items()):
            index = int(index)
            if index in saved or not trials:
                continue
            self.taxel_index = index
            self._batch_results = list(trials)
            self._batch_total = passes
            self._batch_completed = len(trials) >= passes
            self._batch_id = f"{self._full_run_id}_taxel{index:02d}"
            self._batch_directory = (
                self._full_directory / f"taxel_{index:02d}"
                if self._full_directory is not None
                else None
            )
            try:
                partial = self._build_final_result()
            except Exception as exc:
                partial = None
                host._append_sidebar_control_message(
                    f"Could not save the Taxel {index} summary: {exc}"
                )
            if partial is not None:
                self._full_target_results.append(partial)

    def on_stop(self, host, show_result: bool = True) -> None:
        was_active = bool(self._test_active)
        self._stop_robot(self._robot_api(host))
        if was_active:
            self._capture_stop_details(complete=self._full_completed)
        self._clear_post_save_wait()
        if was_active and self._samples and not self._trial_result_saved:
            try:
                self._save_current_trial(host)
            except Exception as exc:
                host._append_sidebar_control_message(
                    f"Could not save the active full-taxel trial: {exc}"
                )
        if was_active and self._batch_results:
            self._trials_by_taxel[int(self.taxel_index)] = list(self._batch_results)
        if was_active:
            self._flush_partial_taxel_results(host)
        try:
            if was_active:
                self.last_result = self._build_full_result()
                self._write_run_manifest("stopped")
        except Exception as exc:
            self.last_result = None
            host._append_sidebar_control_message(
                f"Could not save the full-taxel summary: {exc}"
            )
        finally:
            if self._run_lock is not None:
                self._run_lock.close()
                self._run_lock = None
        if was_active:
            self.resume_plan = self._resume_previous_settings = None
            self.start_button_label = "Run Full Taxel Calibration"
        self._test_active = False
        self._phase = "idle"
        self._force_connection_started_at = None
        self._full_move_target = None
        self._full_route_queue = []
        self._full_recalibration_started_at = None
        self._pending_target_route = None
        self._clear_force_pause_state()
        self._use_map_anchor_as_home()
        self._reset_travel_status(host)
        setter = getattr(host, "_set_full_taxel_calibration_controls_running", None)
        if callable(setter):
            setter(False)
        self._update_progress(
            host, "Complete" if self._full_completed else "Stopped"
        )
        if not was_active or self.last_result is None:
            return
        if self.last_result.get("summary_path"):
            host._append_sidebar_control_message(
                f"Full-taxel calibration summary saved: {self.last_result['summary_path']}"
            )
        if show_result and hasattr(host, "_show_full_taxel_calibration_result"):
            host._show_full_taxel_calibration_result(self.last_result)


class TaxelPositionMappingTask(ExperimentTask):
    """Locate robot TCP response centres for the connected sensor grid."""

    id = "taxel_position_mapping"
    label = "Taxel Position Mapping"
    category = "TRO2026"
    description = (
        "Starting from a remembered safe pose above the anchor taxel, use bounded "
        "proximity probes and non-contact XY scans to locate every taxel centre "
        "and save its robot TCP pose. The anchor follows the connected sensor grid."
    )
    start_button_label = "Map 8x10 Sensor"
    tick_interval_ms = 50
    disconnect_force_meter_on_stop = True
    NEWTONS_PER_GRAM = CalibrationTask.NEWTONS_PER_GRAM
    LINE_MOTION = 4
    FIRST_NEIGHBOURS = (34, 36, 27, 43)
    MAX_PROBE_SPEED_M_S = 0.005
    MAX_PROBE_DESCENT_M = 0.100
    MAPPING_ORDER = "verified_neighbour_frontier_v1"
    CENTRE_SCAN_POLICY = "strong_noncontact_v1"
    DEFAULT_CENTRE_SCAN_SIGNAL_MULTIPLIER = 2.0
    SEARCH_SAFETY_POLICY = "clearance_recalibration_bounded_search_v1"
    MAX_AUTO_RERUNS = 10
    NONCONTACT_FORCE_STOP_G = 5.0
    MAX_NONCONTACT_EXTRA_DESCENT_M = 0.020
    MIN_FIXED_SCAN_CLEARANCE_M = 0.005
    FIXED_SCAN_COARSE_LIFT_M = 0.003
    FIXED_SCAN_BOUNDARY_MARGIN_M = 0.0005
    QUALITY_POLICY = "stationary_strength_repeatable_centre_v1"
    STRENGTH_CONFIRM_BATCHES = 3
    STRENGTH_BORDERLINE_RATIO = 0.90
    STRENGTH_CLEAR_RATIO = 1.10
    CENTRE_REPEAT_MIN_DOMINANCE = 1.20
    # Every later scan plane descends from the anchor's, so an anchor that is
    # barely stronger than its neighbours dooms the whole map. Completed maps
    # had >= 1.8; maps scanned too high had 1.0-1.3.
    MIN_ANCHOR_DOMINANCE = 1.5

    def __init__(
        self,
        storage_path=_DEFAULT_STORAGE_PATH,
        output_directory=_DEFAULT_STORAGE_PATH,
    ) -> None:
        super().__init__()
        if storage_path is _DEFAULT_STORAGE_PATH:
            storage_path = resource_path(
                "config", "taxel_position_mapping_anchor.json"
            )
        if output_directory is _DEFAULT_STORAGE_PATH:
            output_directory = experiment_path("taxel_position_maps")
        self.storage_path = (
            Path(storage_path).expanduser() if storage_path is not None else None
        )
        self.output_directory = (
            Path(output_directory).expanduser()
            if output_directory is not None
            else None
        )

        self.sensor_rows = 8
        self.sensor_columns = 10
        self.anchor_taxel = 35
        self.skip_outer_border = False
        self.proximity_detection_mode = "robust_shift"
        self.signal_threshold_counts = 50.0
        self.adaptive_margin_fraction = 0.30
        # Keep the default detector clear of small baseline wander.  The
        # threshold remains configurable in the mapping UI for quieter or
        # lower-amplitude sensors.
        self.adaptive_min_margin_counts = 6.0
        self.probe_window_frames = 7
        # The anchor only sets the safe travel height. The first probe at
        # Taxel 35 descends from it to learn where proximity starts; every
        # later probe starts probe_start_margin_m above that height.
        self.reference_probe_speed_m_s = 0.003
        self.reference_max_descent_m = 0.100
        self.probe_start_margin_m = 0.002
        self.probe_speed_m_s = 0.003
        self.probe_confirmation_speed_m_s = 0.0005
        # Probe in short Z increments and measure while stationary at every
        # level.  This avoids mixing position change into the sensor window.
        self.probe_step_m = 0.0005
        # "stepwise": each depth step is a TM line move; "velocity": legacy
        # streaming, which stalls and jumps at low speed on this robot.
        self.probe_motion = "stepwise"
        self._probe_line_target_z = None
        self._probe_line_at_limit = False
        self._probe_line_sent_at = None
        self._probe_line_last_z = None
        self.probe_sample_frames = 20
        self.max_probe_descent_m = 0.100
        self.probe_confirm_frames = 2
        self.probe_min_dominance = 0.70
        # Preserve the first confirmed proximity pose as the onset, then move
        # deeper until the target's downward change is this multiple of the
        # learned onset threshold.  The stronger non-contact response gives
        # the subsequent XY centre scan a clearer spatial peak.
        self.centre_scan_signal_multiplier = self.DEFAULT_CENTRE_SCAN_SIGNAL_MULTIPLIER
        self.centre_scan_max_extra_descent_m = self.MAX_NONCONTACT_EXTRA_DESCENT_M
        # Final centre verification uses a longer sample and permits only a
        # small near-tie with the strongest neighbouring taxel.
        self.final_min_dominance = 0.95
        self.force_guard_g = 20.0
        self.coarse_radius_m = 0.008
        self.coarse_step_m = 0.001
        self.adaptive_coarse_search = True
        self.local_radius_m = 0.0025
        self.local_step_m = 0.0005
        self.local_radius_pitch_fraction = 0.6
        self.max_local_recenters = 1
        self.line_speed_m_s = 0.020
        self.max_xy_offset_m = 0.100
        self.position_tolerance_m = 0.0001
        self.stationary_tolerance_m = 0.00002
        self.route_xy_tolerance_m = 0.0002
        self.move_timeout_sec = 45.0
        self.move_timeout_scale = 1.5
        self.move_timeout_margin_sec = 10.0
        self.move_timeout_max_sec = 600.0
        self.move_timeout_fallback_project_speed_percent = 5.0
        self.move_settle_sec = 0.20
        self.sensor_stale_sec = 1.0
        self.sensor_calibration_timeout_sec = 45.0
        self.force_stale_sec = 1.0
        self.force_connection_timeout_sec = 10.0
        self.baseline_frame_count = 100
        # XY search uses a short robust median; onset/baseline/final checks
        # retain their longer stationary windows. Unclear sparse searches
        # automatically fall back to the full grid and at least 20 frames.
        self.scan_frame_count = 9
        self.reverse_scan_order = False
        self.verify_frame_count = 20
        self.max_attempts_per_taxel = 2
        # Explicit operator opt-in, never armed merely by loading a JSON.
        self.max_auto_reruns = 0
        self.auto_rerun_delay_sec = 30.
        self._auto_rerun_count = 0
        self._auto_rerun_history = []
        self._auto_rerun_pending = None
        self._auto_rerun_deadline = None
        self.baseline_stability_fraction = 0.50
        self.baseline_stability_floor_counts = 5.0
        self.resume_xy_tolerance_m = 0.0015
        self.resume_z_tolerance_m = 0.0020
        # Fixed scan plane: every taxel is measured at one operator-chosen Z
        # instead of a per-taxel, noise-derived proximity plane. A raised
        # patch of skin stops the descent early; a far stronger response
        # than the anchor's is treated as possible contact.
        self.fixed_scan_z_m: Optional[float] = None
        self._scan_plane_source = "proximity_threshold"
        self.fixed_scan_approach_window_m = 0.005
        self.fixed_scan_stop_ratio = 1.2
        self.fixed_scan_contact_ratio = 2.0
        self._fixed_scan_reference_signal: Optional[float] = None
        self._skipped_failed_taxels: set[int] = set()
        self._failed_attempt_total = 0

        self.anchor_pose: Optional[dict] = None
        self._anchor_profiles: dict = {}
        self.active_sensor_key_name = ""
        self.active_sensor_label = ""
        self.last_persistence_error = ""
        self.last_result: Optional[dict] = None
        self._active = False
        self._completed = False
        self._phase = "idle"
        self._stop_reason = ""
        self._stop_message = ""
        self._output_folder: Optional[Path] = None
        self._started_at_iso = ""
        self._positions: dict[int, dict] = {}
        self._failures: dict[int, str] = {}
        self._sample_log: list[dict] = []
        self._baseline_checks: list[dict] = []
        self._baseline_stats_by_taxel: dict[int, dict] = {}
        self._attempt_counts: dict[int, int] = {}
        self._reference_attempt_count = 0
        self._reference_onset_z: Optional[float] = None
        self._probe_start_z: Optional[float] = None
        self._mapping_queue: list[int] = []
        self._current_prediction: Optional[dict] = None
        self._neighbor_queue: list[int] = []
        self._neighbor_guesses: dict[int, list[float]] = {}
        self._predicted_positions: dict[int, np.ndarray] = {}
        self._initial_row_vector_xy: Optional[np.ndarray] = None
        self._initial_column_vector_xy: Optional[np.ndarray] = None
        self._row_vector_xy: Optional[np.ndarray] = None
        self._column_vector_xy: Optional[np.ndarray] = None
        self._current_taxel: Optional[int] = None
        self._current_context = ""
        self._probe_kind = ""
        self._probe_speed_used_m_s = 0.0
        self._probe_max_descent_used_m = 0.0
        self._probe_hit_count = 0
        self._probe_last_sequence: Optional[int] = None
        self._probe_baseline: Optional[np.ndarray] = None
        self._probe_baseline_min: Optional[np.ndarray] = None
        self._probe_baseline_max: Optional[np.ndarray] = None
        self._probe_detection_lower = 0.0
        self._probe_detection_upper: Optional[float] = None
        self._probe_detection_margin = 0.0
        self._probe_detection_threshold = float(self.signal_threshold_counts)
        self._probe_detection_by_taxel: dict[int, dict] = {}
        self._probe_recent_frames: list[np.ndarray] = []
        self._probe_candidate_pose: Optional[np.ndarray] = None
        self._probe_stage = "descending"
        self._probe_line_target_z = None
        self._probe_step_start_z: Optional[float] = None
        self._probe_level_pose: Optional[np.ndarray] = None
        self._probe_level_frames: list[np.ndarray] = []
        self._probe_settle_started_at: Optional[float] = None
        self._probe_level_at_max_depth = False
        self._probe_boundary_corrections = 0
        self._probe_strength_hold = None
        self._strength_checks = []
        self._target_strength_check = None
        self._far_baseline: Optional[np.ndarray] = None
        self._far_baseline_min: Optional[np.ndarray] = None
        self._far_baseline_max: Optional[np.ndarray] = None
        self._far_baseline_frames: Optional[np.ndarray] = None
        self._probe_start_pose: Optional[np.ndarray] = None
        self._probe_onset_pose: Optional[np.ndarray] = None
        self._probe_scan_pose: Optional[np.ndarray] = None
        self._probe_scan_signal = 0.0
        self._probe_scan_target_signal = 0.0
        self._probe_scan_target_reached = False
        self._probe_onset_confirmed = False
        self._probe_candidate_seen = False
        self._working_probe_boundary = None
        self._probe_log = []
        self._stop_snapshot = None
        self._clearance_reference_frames = None
        self._clearance_next_context = ""
        self._clearance_checks = []
        self._force_baseline_n: Optional[float] = None
        self._local_center: Optional[np.ndarray] = None
        self._local_axes: list[dict] = []
        self._local_recenters = 0
        self._verify_estimate: Optional[dict] = None
        self._centre_repeat = None
        self._centre_checks = []
        self._target_centre_check = None
        self._route_queue: list[tuple] = []
        self._move_target: Optional[np.ndarray] = None
        self._move_action = ""
        self._move_started_at: Optional[float] = None
        self._move_timeout_current_sec = float(self.move_timeout_sec)
        self._move_expected_duration_sec: Optional[float] = None
        self._move_distance_m = 0.0
        self._move_project_speed_percent: Optional[float] = None
        self._move_arrived_at: Optional[float] = None
        self._move_last_position: Optional[np.ndarray] = None
        self._collection_action = ""
        self._collection_frames: list[np.ndarray] = []
        self._collection_forces: list[float] = []
        self._collection_count = 0
        self._collection_last_sequence: Optional[int] = None
        self._scan_kind = ""
        self._scan_points: list[np.ndarray] = []
        self._scan_point_meta: list[dict] = []
        self._scan_results: list[dict] = []
        self._scan_index = 0
        self._scan_baseline: Optional[np.ndarray] = None
        self._scan_target: Optional[int] = None
        self._coarse_scan_centre: Optional[np.ndarray] = None
        self._coarse_sparse_pass = False
        self._current_target_xy: Optional[np.ndarray] = None
        self._pending_recovery: Optional[dict] = None
        self._last_sensor_sequence: Optional[int] = None
        self._last_sensor_advance_at: Optional[float] = None
        self._sensor_calibration_started_at: Optional[float] = None
        self._sensor_calibration_next_action = ""
        self._sensor_calibration_history: list[dict] = []
        self._force_connection_started_at: Optional[float] = None
        self._mapping_run_started = False
        self._resume_request: Optional[dict] = None
        self._resume_source_path: Optional[Path] = None
        self._resume_history: list[dict] = []
        self._resume_anchor_previous: Optional[dict] = None
        self._resume_alignment_error = ""
        self._anchor_stop: Optional[tuple[str, str]] = None
        self._resume_mapped_before = 0
        self._relocation_request: Optional[dict] = None
        self._relocation_source_path: Optional[Path] = None
        self._relocation_active = False
        self._relocation_continues_mapping = False
        self._relocation_finished = False
        self._relocation_source_result: Optional[dict] = None
        self._relocation_source_positions: dict[int, dict] = {}
        self._relocation_verification_queue: list[int] = []
        self._relocation_reference_indices: list[int] = []
        self._relocation_transform: Optional[dict] = None
        self.load_anchor_pose()

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

    @staticmethod
    def _robot_api(host):
        ui_ros = getattr(host, "ui_ros", None)
        return getattr(ui_ros, "robot_api", None) if ui_ros is not None else None

    @staticmethod
    def taxel_row_column(index: int, rows: int = 8) -> tuple[int, int]:
        """Return display row/column for the viewer's column-major index."""
        rows = int(rows)
        index = int(index)
        return index % rows, index // rows

    @staticmethod
    def taxel_index(row: int, column: int, rows: int = 8) -> int:
        return int(column) * int(rows) + int(row)

    @staticmethod
    def default_anchor_taxel(rows: int, columns: int) -> int:
        rows = int(rows)
        columns = int(columns)
        if (rows, columns) == (8, 10):
            return 35
        if (rows, columns) == (20, 20):
            return 189
        return (rows // 2) + (columns // 2) * rows

    def anchor_neighbour_indices(self) -> tuple[int, ...]:
        """Orthogonal neighbours of the anchor, in row-then-column order."""
        row, column = self.taxel_row_column(self.anchor_taxel, self.sensor_rows)
        neighbours = []
        for row_offset, column_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row = row + row_offset
            next_column = column + column_offset
            if (
                0 <= next_row < int(self.sensor_rows)
                and 0 <= next_column < int(self.sensor_columns)
            ):
                neighbours.append(
                    self.taxel_index(next_row, next_column, self.sensor_rows)
                )
        return tuple(neighbours)

    def _taxel_coords(self, index: int) -> tuple[int, int]:
        return self.taxel_row_column(index, self.sensor_rows)

    def _taxel_flat(self, row: int, column: int) -> int:
        return self.taxel_index(row, column, self.sensor_rows)

    def is_outer_border_taxel(self, index: int) -> bool:
        row, column = self._taxel_coords(index)
        return row in (0, self.sensor_rows - 1) or column in (
            0,
            self.sensor_columns - 1,
        )

    def mapping_target_indices(self) -> list[int]:
        indices = list(range(self.sensor_rows * self.sensor_columns))
        if not bool(self.skip_outer_border):
            return indices
        return [
            index for index in indices
            if not self.is_outer_border_taxel(index)
        ]

    def skipped_taxel_indices(self) -> list[int]:
        if not bool(self.skip_outer_border):
            return []
        return [
            index
            for index in range(self.sensor_rows * self.sensor_columns)
            if self.is_outer_border_taxel(index)
        ]

    def mapped_target_count(self) -> int:
        targets = set(self.mapping_target_indices())
        return sum(index in targets for index in self._positions)

    def _publish_live_map(self, host) -> None:
        callback = getattr(host, "_refresh_taxel_mapping_live_taxels", None)
        if callable(callback):
            callback(
                measured=sorted(self._positions),
                skipped=self.skipped_taxel_indices(),
                target_total=len(self.mapping_target_indices()),
            )

    @classmethod
    def _normalize_anchor_pose(cls, payload):
        if not isinstance(payload, dict):
            return None
        position = cls._finite_values(payload.get("tool_position_m"), 3)
        quaternion = cls._finite_values(
            payload.get("tool_quaternion_wxyz"), 4
        )
        if position is None or quaternion is None:
            return None
        anchor = {
            "saved_at": str(payload.get("saved_at", "")),
            "taxel_index": int(payload.get("taxel_index", 35)),
            "tool_position_m": position,
            "tool_quaternion_wxyz": quaternion,
        }
        choice = payload.get("scan_plane_choice")
        if isinstance(choice, dict) and "z_m" in choice:
            z = finite_number(choice["z_m"])
            if choice["z_m"] is None or z is not None:
                anchor["scan_plane_choice"] = copy.deepcopy(choice)
                anchor["scan_plane_choice"]["z_m"] = z
        return anchor

    def _restore_scan_plane_choice(self) -> None:
        """A choice belongs to one sensor's anchor, never to another setup."""
        choice = (self.anchor_pose or {}).get("scan_plane_choice")
        if choice is not None and (
            choice.get("sensor_identity") != self.active_sensor_key_name
            or (choice.get("sensor_rows"), choice.get("sensor_columns"))
            != (self.sensor_rows, self.sensor_columns)
        ):
            choice = None
        self.fixed_scan_z_m = choice["z_m"] if choice is not None else None
        if choice is None:
            self.fixed_scan_z_m = self._scan_height_recorded_for_this_anchor()
            if self.fixed_scan_z_m is not None and self.anchor_pose is not None:
                self._save_scan_plane_choice()
                return
        self._scan_plane_source = (
            "remembered_for_sensor_anchor" if self.fixed_scan_z_m is not None else "proximity_threshold"
        )

    def _scan_height_recorded_for_this_anchor(self):
        """The fixed scan height saved with this anchor, including inside its position map."""
        anchor = self.anchor_pose
        directory = self.output_directory
        if anchor is None or directory is None or not Path(directory).is_dir():
            return None
        identity = str(self.active_sensor_key_name or "")
        position = np.asarray(anchor.get("tool_position_m"), dtype=float)
        quaternion = np.asarray(anchor.get("tool_quaternion_wxyz"), dtype=float)
        if position.shape != (3,) or quaternion.shape != (4,) or not np.all(np.isfinite(position)):
            return None
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm < 1e-9:
            return None
        quaternion = quaternion / quaternion_norm
        newest = None
        for path in Path(directory).glob("*/taxel_position_map.json"):
            try:
                with path.open("r", encoding="utf-8") as stream:
                    result = json.load(stream)
            except (OSError, json.JSONDecodeError):
                continue
            if str(result.get("sensor_identity") or "") != identity:
                continue
            if revision_motion_error(result):
                continue
            settings = result.get("settings") if isinstance(result.get("settings"), dict) else {}
            if (int(settings.get("sensor_rows") or -1), int(settings.get("sensor_columns") or -1)) != (
                int(self.sensor_rows), int(self.sensor_columns)
            ):
                continue
            saved = result.get("anchor_pose") if isinstance(result.get("anchor_pose"), dict) else {}
            if int(saved.get("taxel_index", -1)) != int(anchor.get("taxel_index", -2)):
                continue
            saved_position = np.asarray(saved.get("tool_position_m"), dtype=float)
            saved_quaternion = np.asarray(saved.get("tool_quaternion_wxyz"), dtype=float)
            height = finite_number(settings.get("fixed_scan_z_mm"))
            if (saved_position.shape != (3,) or saved_quaternion.shape != (4,)
                    or height is None or not np.all(np.isfinite(saved_position))):
                continue
            saved_norm = float(np.linalg.norm(saved_quaternion))
            if saved_norm < 1e-9:
                continue
            alignment = abs(float(np.dot(quaternion, saved_quaternion / saved_norm)))
            if float(np.linalg.norm(saved_position - position)) > 0.010 or alignment < 0.99:
                continue
            anchor_z = float(position[2])
            scan_z = height / 1000.0
            if not self.MIN_FIXED_SCAN_CLEARANCE_M <= anchor_z - scan_z <= self.MAX_PROBE_DESCENT_M:
                continue
            stamp = path.stat().st_mtime
            if newest is None or stamp >= newest[0]:
                newest = (stamp, scan_z)
        return None if newest is None else newest[1]

    def _save_scan_plane_choice(self) -> bool:
        self.anchor_pose["scan_plane_choice"] = {
            "z_m": self.fixed_scan_z_m,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "sensor_identity": self.active_sensor_key_name,
            "sensor_rows": int(self.sensor_rows),
            "sensor_columns": int(self.sensor_columns),
        }
        self._anchor_profiles[self.active_sensor_key_name] = copy.deepcopy(self.anchor_pose)
        self._scan_plane_source = "remembered_for_sensor_anchor"
        return self.save_anchor_pose()

    def _resume_scan_plane_choice(self, result):
        """Keep an explicit choice (including Clear) only for the same setup."""
        choice = (self.anchor_pose or {}).get("scan_plane_choice")
        if choice is None:
            return None
        settings = result["settings"]
        saved_key = str(result.get("sensor_identity") or "")
        if choice.get("sensor_identity") != self.active_sensor_key_name or (
            saved_key and saved_key != choice.get("sensor_identity")
        ) or (
            (choice.get("sensor_rows"), choice.get("sensor_columns"))
            != (int(settings["sensor_rows"]), int(settings["sensor_columns"]))
        ):
            return None
        saved_anchor = self._normalize_anchor_pose(result["anchor_pose"])
        same_position = np.allclose(
            self.anchor_pose["tool_position_m"], saved_anchor["tool_position_m"],
            rtol=0., atol=1e-9,
        )
        current_q = np.asarray(self.anchor_pose["tool_quaternion_wxyz"], dtype=float)
        saved_q = np.asarray(saved_anchor["tool_quaternion_wxyz"], dtype=float)
        same_orientation = min(
            np.linalg.norm(current_q - saved_q), np.linalg.norm(current_q + saved_q)
        ) < 1e-9
        if (not same_position or not same_orientation
                or self.anchor_pose["taxel_index"] != saved_anchor["taxel_index"]):
            raise ValueError(
                "the selected scan plane belongs to a different remembered anchor. "
                "Use Relocate for a moved setup; do not apply this height to the old map."
            )
        return copy.deepcopy(choice)

    def _scan_plane_status(self) -> str:
        if self.fixed_scan_z_m is None:
            return (
                "Scan mode: automatic proximity-threshold descent (no fixed scan plane). A "
                "remembered scan height is more reliable: the automatic descent can stop too "
                "high to centre taxels."
            )
        source = (
            "your remembered height" if self._scan_plane_source == "remembered_for_sensor_anchor"
            else "saved map setting"
        )
        return (
            f"Scan mode: fixed scan plane Z {self.fixed_scan_z_m * 1000.0:.2f} mm "
            f"({source}). Force, travel and centre-quality guards remain active."
        )

    def _read_anchor_profiles(self) -> dict:
        if self.storage_path is None or not self.storage_path.exists():
            return {}
        with self.storage_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("saved anchor pose is invalid")
        profiles = payload.get("profiles")
        if int(payload.get("version", 1) or 1) >= 2 and isinstance(profiles, dict):
            loaded = {}
            for key, item in profiles.items():
                anchor = self._normalize_anchor_pose(item)
                if anchor is not None:
                    loaded[str(key)] = anchor
            return loaded
        legacy = self._normalize_anchor_pose(payload)
        if legacy is None:
            raise ValueError("saved anchor pose is invalid")
        return {_LEGACY_CALIBRATION_SENSOR_KEY: legacy}

    def load_anchor_pose(self) -> bool:
        self.last_persistence_error = ""
        try:
            self._anchor_profiles = self._read_anchor_profiles()
        except Exception as exc:
            self._anchor_profiles = {}
            self.anchor_pose = None
            self.last_persistence_error = f"Could not load mapping anchor: {exc}"
            return False
        if len(self._anchor_profiles) == 1:
            key, anchor = next(iter(self._anchor_profiles.items()))
            self.active_sensor_key_name = key
            self.anchor_pose = copy.deepcopy(anchor)
            self._restore_scan_plane_choice()
        return True

    def save_anchor_pose(self) -> bool:
        self.last_persistence_error = ""
        if self.storage_path is None:
            return True
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.storage_path.with_suffix(
                self.storage_path.suffix + ".tmp"
            )
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(
                    {"version": 2, "profiles": self._anchor_profiles},
                    stream,
                    indent=2,
                )
                stream.write("\n")
            os.replace(temporary, self.storage_path)
            return True
        except Exception as exc:
            self.last_persistence_error = f"Could not save mapping anchor: {exc}"
            return False

    def use_connected_sensor(self, host) -> str:
        """Match the grid, anchor taxel, and saved pose to the live sensor."""
        if self._active:
            return str(self.active_sensor_key_name or "")
        key, label, rows, columns = connected_calibration_sensor(host)
        previous_key = self.active_sensor_key_name
        self.active_sensor_key_name = key
        self.active_sensor_label = label
        if rows > 0 and columns > 0:
            previous_shape = (self.sensor_rows, self.sensor_columns)
            next_shape = (int(rows), int(columns))
            if next_shape == (20, 20) and previous_shape != next_shape:
                # Defaults for the 400-node sensor around Taxel 189. Keep the
                # fine 0.5 mm scan separate from this wider coarse discovery.
                self._smaller_grid_search_settings = (
                    self.max_xy_offset_m, self.coarse_radius_m, self.coarse_step_m,
                    self.skip_outer_border,
                )
                self.max_xy_offset_m = 0.200
                self.coarse_radius_m = 0.020
                self.coarse_step_m = 0.002
                self.skip_outer_border = True
            elif previous_shape == (20, 20) and next_shape != previous_shape:
                (
                    self.max_xy_offset_m, self.coarse_radius_m, self.coarse_step_m,
                    self.skip_outer_border,
                ) = getattr(self, "_smaller_grid_search_settings", (0.100, 0.008, 0.001, False))
            self.sensor_rows = int(rows)
            self.sensor_columns = int(columns)
            self.anchor_taxel = self.default_anchor_taxel(
                self.sensor_rows, self.sensor_columns
            )
            self.start_button_label = (
                f"Map {self.sensor_rows}x{self.sensor_columns} Sensor"
            )
        self.anchor_pose = copy.deepcopy(self._anchor_profiles.get(key))
        if key != previous_key or "scan_plane_choice" in (self.anchor_pose or {}):
            self._restore_scan_plane_choice()
        return key

    def preview_remembered_setup(self, host, kind, result=None):
        """Capture a proposal only. Cancel must not alter profiles or any files."""
        if self._active or getattr(host, "_sidebar_active_task", None) is not None:
            raise ValueError("Stop the current experiment before changing the mapping setup.")
        if kind not in {"anchor", "scan_height"}:
            raise ValueError("Unknown mapping setup control.")
        key, label, rows, columns = connected_calibration_sensor(host)
        if rows <= 0 or columns <= 0:
            raise ValueError("Connect/select the sensor before remembering its mapping setup.")
        if result is not None:
            settings = result.get("settings") or {}
            if (result.get("sensor_identity") != key or (rows, columns) != (
                    settings.get("sensor_rows"), settings.get("sensor_columns"))):
                raise ValueError("The loaded map does not match the connected sensor. Load its map first.")
        api = self._robot_api(host)
        pose = self._read_tool_pose(api)
        stamp = finite_number(getattr(api, "current_feedback_state_time", None))
        if pose is None or stamp is None or not -.1 <= time.monotonic() - stamp <= 1.:
            raise ValueError("Fresh robot TCP feedback is required; nothing was saved.")
        position, quaternion = pose
        stamp = datetime.now(timezone.utc).isoformat()
        requested = revision_setup(result)
        anchor = copy.deepcopy(
            (requested or {}).get("anchor_pose")
            or self._anchor_profiles.get(key)
            or (result or {}).get("anchor_pose")
        )
        index = self.default_anchor_taxel(rows, columns)
        z = None
        if anchor is not None:
            choice = anchor.get("scan_plane_choice")
            if choice is not None:
                z = choice.get("z_m")
            elif requested is not None:
                z = requested.get("fixed_scan_z_m")
            elif result is not None:
                mm = (result.get("settings") or {}).get("fixed_scan_z_mm")
                z = None if mm is None else float(mm) / 1000.
            elif self.active_sensor_key_name == key:
                z = self.fixed_scan_z_m
        captured = dict(saved_at=stamp, taxel_index=index,
                        tool_position_m=position.tolist(), tool_quaternion_wxyz=quaternion.tolist())
        if kind == "anchor":
            # A changed anchor does not inherit an old absolute scan plane.
            if anchor is None or not same_pose(anchor, captured):
                z = None
            anchor = captured
        else:
            if anchor is None:
                raise ValueError(f"Remember the safe anchor above Taxel {index} first.")
            distance = np.linalg.norm(position[:2] - np.asarray(anchor["tool_position_m"])[:2])
            if distance > self.max_xy_offset_m:
                raise ValueError("Move the robot over the sensor before remembering the scan height.")
            z = float(position[2])
        setup = normalize_setup(dict(
            saved_at=stamp, sensor_identity=key, sensor_label=label,
            sensor_rows=rows, sensor_columns=columns, anchor_pose=anchor, fixed_scan_z_m=z,
        ))
        return setup, captured

    def apply_remembered_setup(self, setup, *, persist=False):
        """Install a validated choice. Revisions themselves are the durable copy."""
        setup = normalize_setup(setup)
        before = copy.deepcopy(self._anchor_profiles)
        key = setup["sensor_identity"]
        self._anchor_profiles[key] = copy.deepcopy(setup["anchor_pose"])
        if persist and not self.save_anchor_pose():
            self._anchor_profiles = before
            raise OSError(self.last_persistence_error)
        self.active_sensor_key_name = key
        self.active_sensor_label = setup.get("sensor_label", key)
        self.sensor_rows, self.sensor_columns = setup["sensor_rows"], setup["sensor_columns"]
        self.anchor_taxel = setup["anchor_pose"]["taxel_index"]
        self.anchor_pose = copy.deepcopy(setup["anchor_pose"])
        self.fixed_scan_z_m = setup["fixed_scan_z_m"]
        self._scan_plane_source = "remembered_for_sensor_anchor" if self.fixed_scan_z_m is not None else "proximity_threshold"

    def capture_anchor_pose(self, host) -> bool:
        if self._active:
            host._append_sidebar_control_message(
                "Stop Taxel Position Mapping before changing its anchor."
            )
            return False
        pose = self._read_tool_pose(self._robot_api(host))
        if pose is None:
            host._append_sidebar_control_message(
                f"Current robot TCP feedback is unavailable; Taxel {self.anchor_taxel} anchor was not saved."
            )
            return False
        self.use_connected_sensor(host)
        position, quaternion = pose
        self.anchor_pose = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "taxel_index": int(self.anchor_taxel),
            "tool_position_m": position.tolist(),
            "tool_quaternion_wxyz": quaternion.tolist(),
        }
        sensor_key = str(self.active_sensor_key_name or "layout:unknown")
        self._anchor_profiles[sensor_key] = copy.deepcopy(self.anchor_pose)
        previous_scan_z = self.fixed_scan_z_m
        self._restore_scan_plane_choice()
        saved = self.save_anchor_pose()
        host._append_sidebar_control_message(
            f"Remembered the safe TCP pose above Taxel {self.anchor_taxel}: "
            f"{[round(value * 1000.0, 2) for value in position]} mm."
        )
        if not saved:
            host._append_sidebar_control_message(self.last_persistence_error)
        if previous_scan_z is not None and self.fixed_scan_z_m is None:
            host._append_sidebar_control_message(
                f"New anchor remembered: the previous scan plane (Z {previous_scan_z * 1000.0:.2f} mm) "
                "does not belong to this pose. Remember the scan height again for this anchor."
            )
        refresher = getattr(host, "_refresh_taxel_mapping_controls", None)
        if callable(refresher):
            refresher()
        return saved

    def capture_scan_height(self, host) -> bool:
        """Remember the current TCP Z as the fixed scan plane for every taxel."""
        if self._active:
            host._append_sidebar_control_message(
                "Stop Taxel Position Mapping before changing its scan height."
            )
            return False
        self.use_connected_sensor(host)
        if self.anchor_pose is None:
            host._append_sidebar_control_message(
                f"Remember the safe anchor above Taxel {self.anchor_taxel} first."
            )
            return False
        pose = self._read_tool_pose(self._robot_api(host))
        stamp = finite_number(getattr(self._robot_api(host), "current_feedback_state_time", None))
        if pose is None or stamp is None or not -0.1 <= time.monotonic() - stamp <= 1.0:
            host._append_sidebar_control_message(
                "Fresh robot TCP feedback is required; the scan height was not saved."
            )
            return False
        anchor = np.asarray(self.anchor_pose["tool_position_m"], dtype=float)
        position = np.asarray(pose[0], dtype=float)
        clearance = float(anchor[2] - position[2])
        if clearance < self.MIN_FIXED_SCAN_CLEARANCE_M:
            host._append_sidebar_control_message(
                f"The scan height must be at least {self.MIN_FIXED_SCAN_CLEARANCE_M * 1000.0:.0f} mm "
                f"below the anchor (currently {clearance * 1000.0:.1f} mm); not saved."
            )
            return False
        if clearance > self.MAX_PROBE_DESCENT_M:
            host._append_sidebar_control_message(
                "The scan height is more than 100 mm below the anchor; not saved."
            )
            return False
        if float(np.linalg.norm(position[:2] - anchor[:2])) > float(self.max_xy_offset_m):
            host._append_sidebar_control_message(
                "Move the robot over the sensor before remembering the scan height; not saved."
            )
            return False
        self.fixed_scan_z_m = float(position[2])
        saved = self._save_scan_plane_choice()
        host._append_sidebar_control_message(
            f"Fixed scan height remembered: Z {position[2] * 1000.0:.2f} mm, "
            f"{clearance * 1000.0:.1f} mm below the anchor. Every taxel is scanned at this height "
            "unless a raised patch of skin stops the descent earlier."
        )
        if not saved:
            host._append_sidebar_control_message(
                "Scan height is set for this session but could not be saved for restart: "
                + self.last_persistence_error
            )
        refresher = getattr(host, "_refresh_taxel_mapping_controls", None)
        if callable(refresher):
            refresher()
        return saved

    def clear_scan_height(self, host) -> None:
        if self._active:
            return
        self.use_connected_sensor(host)
        self.fixed_scan_z_m = None
        if self.anchor_pose is not None and not self._save_scan_plane_choice():
            host._append_sidebar_control_message(self.last_persistence_error)
        host._append_sidebar_control_message(
            "Fixed scan height cleared; mapping uses the proximity-threshold descent."
        )
        refresher = getattr(host, "_refresh_taxel_mapping_controls", None)
        if callable(refresher):
            refresher()

    @property
    def fixed_scan_active(self) -> bool:
        return self.fixed_scan_z_m is not None

    @classmethod
    def validate_resume_result(cls, result) -> tuple[bool, str]:
        """Return whether a partial mapping JSON has enough state to continue."""
        if not isinstance(result, dict) or result.get("task") != cls.id:
            return False, "Select a taxel_position_map.json recording."
        if revision_motion_error(result):
            return False, revision_motion_error(result)
        if bool(result.get("completed")):
            return False, "This map is already complete."
        settings = result.get("settings")
        if not isinstance(settings, dict):
            return False, "Resume requires a taxel position map with a sensor grid."
        try:
            rows = int(settings.get("sensor_rows"))
            columns = int(settings.get("sensor_columns"))
        except (TypeError, ValueError):
            return False, "Resume requires a taxel position map with a sensor grid."
        taxel_count = rows * columns
        if rows <= 0 or columns <= 0 or taxel_count <= 0:
            return False, "Resume requires a taxel position map with a sensor grid."
        anchor = cls._normalize_anchor_pose(result.get("anchor_pose"))
        try:
            anchor_index = int(anchor.get("taxel_index", -1)) if anchor else -1
        except (TypeError, ValueError):
            anchor_index = -1
        if anchor is None or not 0 <= anchor_index < taxel_count:
            return False, "The saved mapping anchor is missing or invalid."
        if (rows, columns) == (8, 10) and anchor_index != 35:
            return False, "The saved Taxel 35 anchor is missing or invalid."
        anchor_quaternion = np.asarray(
            anchor["tool_quaternion_wxyz"], dtype=float
        )
        if not 0.99 <= float(np.linalg.norm(anchor_quaternion)) <= 1.01:
            return False, "The saved tool orientation is invalid."
        try:
            onset_z = float(result.get("reference_onset_z_m"))
            probe_start_z = float(result.get("probe_start_z_m"))
        except (TypeError, ValueError):
            return False, "The saved sensor height is missing; this run cannot resume safely."
        anchor_z = float(anchor["tool_position_m"][2])
        if (
            not np.isfinite(onset_z)
            or not np.isfinite(probe_start_z)
            or not onset_z <= probe_start_z <= anchor_z
        ):
            return False, "The saved sensor/probe heights are inconsistent."
        positions = result.get("positions")
        if not isinstance(positions, list):
            return False, "The saved map has no measured-position list."
        measured = set()
        for record in positions:
            if not isinstance(record, dict) or record.get("status") != "measured":
                continue
            try:
                index = int(record.get("taxel_index"))
                centre = np.asarray(record.get("tool_position_m"), dtype=float)
                safe = np.asarray(record.get("safe_approach_position_m"), dtype=float)
            except (TypeError, ValueError):
                return False, "The saved map contains an invalid measured position."
            if (
                not 0 <= index < taxel_count
                or centre.shape != (3,)
                or safe.shape != (3,)
                or not np.all(np.isfinite(centre))
                or not np.all(np.isfinite(safe))
            ):
                return False, "The saved map contains an invalid measured position."
            measured.add(index)
        if anchor_index not in measured:
            return False, (
                f"Taxel {anchor_index} must have a measured centre before a run can resume."
            )
        anchor_row, anchor_column = cls.taxel_row_column(anchor_index, rows)
        neighbours = []
        for row_offset, column_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row = anchor_row + row_offset
            next_column = anchor_column + column_offset
            if 0 <= next_row < rows and 0 <= next_column < columns:
                neighbours.append(cls.taxel_index(next_row, next_column, rows))
        missing_neighbours = set(neighbours) - measured
        if missing_neighbours:
            for name in (
                "initial_row_vector_xy_m",
                "initial_column_vector_xy_m",
            ):
                try:
                    vector = np.asarray(result.get(name), dtype=float)
                except (TypeError, ValueError):
                    vector = np.asarray([])
                if vector.shape != (2,) or not np.all(np.isfinite(vector)):
                    return (
                        False,
                        "The initial taxel-grid directions were not saved; "
                        "this partial run must restart.",
                    )
        skip_border = bool(settings.get("skip_outer_border", False))

        def is_target(index: int) -> bool:
            if not skip_border:
                return True
            row, column = cls.taxel_row_column(index, rows)
            return row not in (0, rows - 1) and column not in (0, columns - 1)

        target_indices = [index for index in range(taxel_count) if is_target(index)]
        target_total = len(target_indices)
        mapped_target = sum(index in measured for index in target_indices)
        region = " interior" if skip_border else ""
        relocation_error = cls._relocation_resume_error(result)
        if relocation_error:
            return False, relocation_error
        relocating = bool(result.get("relocation_checkpoint"))
        return (
            True,
            f"{'Relocation resume' if relocating else 'Resume'} available from "
            f"{mapped_target}/{target_total} measured"
            f"{region} taxels.",
        )

    @classmethod
    def _relocation_resume_error(cls, result) -> str:
        state = result.get("relocation_checkpoint")
        if state is None:
            # Legacy intermediate transforms did not save their source map.
            # Never silently turn an interrupted relocation into a full scan.
            legacy_fit = isinstance(result.get("relocation"), dict) and "rotation" in result["relocation"]
            legacy_unfitted = result.get("relocation") is None and Path(
                str(result.get("output_folder") or "")
            ).name.startswith("taxel_position_map_relocated_")
            if legacy_fit or legacy_unfitted:
                return "This relocation checkpoint lacks its source map; recover it before resuming."
            return ""
        if not isinstance(state, dict) or state.get("version") != 1:
            return "The relocation checkpoint is invalid."
        source = state.get("source_map")
        valid, message = cls.validate_relocation_result(source)
        if not valid:
            return "The relocation source map is invalid: " + message
        for key in ("sensor_rows", "sensor_columns", "anchor_taxel"):
            if source["settings"].get(key) != result["settings"].get(key):
                return "The relocation source grid/anchor does not match the checkpoint."
        if source.get("sensor_identity") != result.get("sensor_identity"):
            return "The relocation source sensor does not match the checkpoint."
        if type(state.get("finished")) is not bool:
            return "The relocation checkpoint has no valid completion state."
        references = state.get("reference_taxels")
        measured = {record["taxel_index"] for record in result["positions"] if record.get("status") == "measured"}
        source_indices = {record["taxel_index"] for record in source["positions"] if record.get("status") == "measured"}
        if (not isinstance(references, list)
                or any(type(index) is not int for index in references)
                or len(set(references)) != len(references)
                or not set(references) <= measured.intersection(source_indices)):
            return "The relocation reference taxels are invalid."
        if not state["finished"] and not measured <= source_indices:
            return "The relocation measurements do not belong to its source map."
        rows = int(result["settings"]["sensor_rows"])
        columns = int(result["settings"]["sensor_columns"])
        anchor = int(result["settings"]["anchor_taxel"])
        row, column = cls.taxel_row_column(anchor, rows)
        expected = {anchor}
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if 0 <= row + dr < rows and 0 <= column + dc < columns:
                expected.add(cls.taxel_index(row + dr, column + dc, rows))
        if references:
            if set(references) != expected:
                return "The relocation checkpoint must retain its original centre/neighbour references."
        elif state["finished"] or not measured <= expected:
            return "The relocation checkpoint is missing its original reference taxels."
        return ""

    def _relocation_checkpoint(self):
        """Self-contained continuation state, including before a fit exists."""
        if not self._relocation_active:
            return None
        # Do not embed histories/checkpoints recursively when relocating a map
        # that was itself relocated. Only these fields seed the rigid fit.
        source = self._relocation_source_result or {}
        source_keys = (
            "task", "completed", "settings", "sensor_identity", "positions",
            "initial_row_vector_xy_m", "initial_column_vector_xy_m",
            "row_vector_xy_m", "column_vector_xy_m",
        )
        return {
            "version": 1,
            "source_json_path": str(self._relocation_source_path or ""),
            "source_map": {key: copy.deepcopy(source[key]) for key in source_keys if key in source},
            "finished": bool(self._relocation_finished),
            "reference_taxels": list(self._relocation_reference_indices),
        }

    def _restore_relocation_checkpoint(self, result) -> None:
        """Reset stale state and restore relocation without issuing motion."""
        state = result.get("relocation_checkpoint") or {}
        self._relocation_active = bool(state)
        self._relocation_finished = bool(state.get("finished", False))
        self._relocation_source_result = copy.deepcopy(state.get("source_map"))
        self._relocation_continues_mapping = bool(
            state and not self._relocation_source_result.get("completed")
        )
        self._relocation_source_path = (
            Path(state["source_json_path"]) if state.get("source_json_path") else None
        )
        self._relocation_source_positions = {
            int(record["taxel_index"]): copy.deepcopy(record)
            for record in (self._relocation_source_result or {}).get("positions", [])
            if record.get("status") == "measured"
        }
        self._relocation_transform = copy.deepcopy(result.get("relocation")) if state else None
        self._relocation_reference_indices = list(state.get("reference_taxels", []))
        self._relocation_verification_queue = []
        if self._relocation_reference_indices and not self._relocation_finished:
            # Rebuild the original deterministic plan, not just the queue:
            # an interrupted in-flight target was already popped from it.
            self._relocation_verification_queue = [
                index for index in self._choose_relocation_verification_indices(3)
                if index not in self._positions
            ]

    def request_resume(self, result, source_path=None) -> tuple[bool, str]:
        """Queue a validated partial result for the next ``on_start`` call."""
        valid, message = self.validate_resume_result(result)
        if not valid:
            self._resume_request = None
            self._resume_source_path = None
            return False, message
        self._resume_request = copy.deepcopy(result)
        self._resume_source_path = (
            Path(source_path).expanduser().resolve()
            if source_path
            else None
        )
        return True, message

    @classmethod
    def validate_relocation_result(cls, result) -> tuple[bool, str]:
        """Return whether a saved map can seed a moved-sensor map.

        A partial map is re-registered the same way and then continues
        mapping its remaining targets from the re-measured taxels.
        """
        if not isinstance(result, dict) or result.get("task") != cls.id:
            return False, "Select a taxel_position_map.json recording."
        if "setup_revision" in result:
            try:
                validate_setup_binding(result, revision_setup(result))
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                return False, f"Invalid map setup revision: {exc}"
        settings = result.get("settings")
        if not isinstance(settings, dict):
            return False, "The saved map has no sensor-grid settings."
        try:
            rows = int(settings.get("sensor_rows"))
            columns = int(settings.get("sensor_columns"))
            anchor_index = int(settings.get("anchor_taxel"))
        except (TypeError, ValueError):
            return False, "The saved map has invalid sensor-grid settings."
        if rows <= 0 or columns <= 0 or not 0 <= anchor_index < rows * columns:
            return False, "The saved map has invalid sensor-grid settings."
        measured = {}
        for record in result.get("positions") or []:
            if not isinstance(record, dict) or record.get("status") != "measured":
                continue
            try:
                index = int(record.get("taxel_index"))
                centre = np.asarray(record.get("tool_position_m"), dtype=float)
                safe = np.asarray(
                    record.get("safe_approach_position_m"), dtype=float
                )
            except (TypeError, ValueError):
                return False, "The saved map contains an invalid position."
            if (
                not 0 <= index < rows * columns
                or centre.shape != (3,)
                or safe.shape != (3,)
                or not np.all(np.isfinite(centre))
                or not np.all(np.isfinite(safe))
            ):
                return False, "The saved map contains an invalid position."
            measured[index] = record
        anchor_row, anchor_column = cls.taxel_row_column(anchor_index, rows)
        references = [anchor_index]
        for row_offset, column_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            row = anchor_row + row_offset
            column = anchor_column + column_offset
            if 0 <= row < rows and 0 <= column < columns:
                references.append(cls.taxel_index(row, column, rows))
        missing = [index for index in references if index not in measured]
        if missing:
            return False, (
                "Relocation requires the centre and its neighbouring reference "
                "taxels; missing " + ", ".join(map(str, missing)) + "."
            )
        if len(measured) < len(references) + 1:
            return False, "The saved map has too few positions to relocate."
        continuation = (
            ""
            if bool(result.get("completed"))
            else " The partial map then continues mapping its remaining taxels."
        )
        return True, (
            f"Relocation available for {len(measured)} saved taxels. Move the "
            f"robot to a safe pose approximately above Taxel {anchor_index}, "
            "remember that pose, then start relocation." + continuation
        )

    def request_relocation(self, result, source_path=None) -> tuple[bool, str]:
        """Queue a completed map to be registered at a new table pose."""
        valid, message = self.validate_relocation_result(result)
        if not valid:
            self._relocation_request = None
            self._relocation_source_path = None
            return False, message
        self._relocation_request = copy.deepcopy(result)
        self._relocation_source_path = (
            Path(source_path).expanduser().resolve()
            if source_path
            else None
        )
        return True, message

    @staticmethod
    def estimate_rigid_transform_2d(old_points, new_points) -> Optional[dict]:
        """Least-squares rotation and translation from old XY to new XY."""
        try:
            old = np.asarray(old_points, dtype=float)
            new = np.asarray(new_points, dtype=float)
        except (TypeError, ValueError):
            return None
        if (
            old.ndim != 2
            or new.shape != old.shape
            or old.shape[0] < 2
            or old.shape[1] != 2
            or not np.all(np.isfinite(old))
            or not np.all(np.isfinite(new))
        ):
            return None
        old_centre = np.mean(old, axis=0)
        new_centre = np.mean(new, axis=0)
        old_zero = old - old_centre
        new_zero = new - new_centre
        if np.linalg.matrix_rank(old_zero) < 2:
            return None
        left, _singular, right_t = np.linalg.svd(old_zero.T @ new_zero)
        rotation = right_t.T @ left.T
        if float(np.linalg.det(rotation)) < 0.0:
            right_t[-1, :] *= -1.0
            rotation = right_t.T @ left.T
        translation = new_centre - rotation @ old_centre
        predicted = (rotation @ old.T).T + translation
        residuals = np.linalg.norm(predicted - new, axis=1)
        angle_deg = float(
            np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
        )
        return {
            "rotation": rotation,
            "translation": translation,
            "angle_deg": angle_deg,
            "rms_residual_mm": float(
                np.sqrt(np.mean(residuals ** 2)) * 1000.0
            ),
            "max_residual_mm": float(np.max(residuals) * 1000.0),
        }

    @staticmethod
    def _saved_vector(result, name, count=2):
        value = result.get(name)
        if value is None:
            return None
        try:
            vector = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if vector.size != int(count) or not np.all(np.isfinite(vector)):
            return None
        return np.array(vector, dtype=float, copy=True)

    def _resume_scan_plane_mismatch(self, result) -> str:
        """Centres saved at another scan height can be neither verified nor extended."""
        if not self.fixed_scan_active:
            return ""
        anchor = int(result["settings"].get("anchor_taxel", self.anchor_taxel))
        positions = result.get("positions") or []
        if isinstance(positions, dict):
            positions = list(positions.values())
        saved_z = None
        for item in positions:
            if int(item.get("taxel_index", -1)) == anchor and item.get("status") == "measured":
                position = self._finite_values(item.get("tool_position_m"), 3)
                saved_z = None if position is None else float(position[2])
        if saved_z is None:
            return ""
        difference = saved_z - float(self.fixed_scan_z_m)
        if abs(difference) <= float(self.resume_z_tolerance_m):
            return ""
        return (
            f"the saved centres were scanned at Z {saved_z * 1000.0:.1f} mm, but the scan height is "
            f"now Z {self.fixed_scan_z_m * 1000.0:.1f} mm ({difference * 1000.0:+.1f} mm). Centres from "
            "another scan height cannot be verified or mixed: start a new map at this height, or "
            "clear the scan height to resume at the saved one."
        )

    def _apply_resume_settings(self, result) -> None:
        settings = result["settings"]
        scan_choice = self._resume_scan_plane_choice(result)
        # Recovery controls intentionally remain the operator's current choice.
        # Loading an old experiment must not enable automatic robot restarts.
        direct = {
            "skip_outer_border": "skip_outer_border",
            "proximity_detection_mode": "proximity_detection_mode",
            "signal_threshold_counts": "signal_threshold_counts",
            "adaptive_margin_fraction": "adaptive_margin_fraction",
            "adaptive_min_margin_counts": "adaptive_min_margin_counts",
            "probe_window_frames": "probe_window_frames",
            "probe_sample_frames": "probe_sample_frames",
            "probe_confirm_frames": "probe_confirm_frames",
            "probe_min_dominance": "probe_min_dominance",
            "centre_scan_signal_multiplier": "centre_scan_signal_multiplier",
            "final_min_dominance": "final_min_dominance",
            "force_guard_g": "force_guard_g",
            "local_radius_pitch_fraction": "local_radius_pitch_fraction",
            "max_local_recenters": "max_local_recenters",
            "verify_frame_count": "verify_frame_count",
            "scan_frame_count": "scan_frame_count",
            "adaptive_coarse_search": "adaptive_coarse_search",
            "reverse_scan_order": "reverse_scan_order",
            "baseline_frame_count": "baseline_frame_count",
            "baseline_stability_fraction": "baseline_stability_fraction",
            "baseline_stability_floor_counts": "baseline_stability_floor_counts",
            "max_attempts_per_taxel": "max_attempts_per_taxel",
            "sensor_calibration_timeout_sec": "sensor_calibration_timeout_sec",
        }
        millimetres = {
            "reference_probe_speed_mm_s": "reference_probe_speed_m_s",
            "reference_max_descent_mm": "reference_max_descent_m",
            "probe_start_margin_mm": "probe_start_margin_m",
            "probe_speed_mm_s": "probe_speed_m_s",
            "probe_confirmation_speed_mm_s": "probe_confirmation_speed_m_s",
            "probe_step_mm": "probe_step_m",
            "max_probe_descent_mm": "max_probe_descent_m",
            "centre_scan_max_extra_descent_mm": (
                "centre_scan_max_extra_descent_m"
            ),
            "coarse_radius_mm": "coarse_radius_m",
            "coarse_step_mm": "coarse_step_m",
            "local_radius_mm": "local_radius_m",
            "local_step_mm": "local_step_m",
            "line_speed_mm_s": "line_speed_m_s",
            "position_tolerance_mm": "position_tolerance_m",
            "max_xy_offset_mm": "max_xy_offset_m",
        }
        for key, attribute in direct.items():
            if key in settings:
                current = getattr(self, attribute)
                if isinstance(current, bool):
                    value = bool(settings[key])
                elif isinstance(current, int):
                    value = int(settings[key])
                elif isinstance(current, str):
                    value = str(settings[key])
                else:
                    value = float(settings[key])
                setattr(self, attribute, value)
        for key, attribute in millimetres.items():
            if key in settings:
                setattr(self, attribute, float(settings[key]) / 1000.0)
        fixed_z_mm = finite_number(settings.get("fixed_scan_z_mm"))
        self.fixed_scan_z_m = (
            scan_choice["z_m"] if scan_choice is not None
            else (None if fixed_z_mm is None else fixed_z_mm / 1000.0)
        )
        self._scan_plane_source = (
            "remembered_for_sensor_anchor" if scan_choice is not None else "saved_map"
        )
        window_mm = finite_number(settings.get("fixed_scan_approach_window_mm"))
        if window_mm is not None and window_mm > 0.0:
            self.fixed_scan_approach_window_m = window_mm / 1000.0
        for key in ("fixed_scan_stop_ratio", "fixed_scan_contact_ratio"):
            value = finite_number(settings.get(key))
            if value is not None and value > 1.0:
                setattr(self, key, value)
        if "probe_sample_frames" not in settings:
            # Older maps used overlapping frames while moving.  Resume them
            # with the safer step-and-sample defaults instead of interpreting
            # the old frame-confirmation value as stationary depth batches.
            self.probe_step_m = 0.0005
            self.probe_sample_frames = 20
            self.probe_confirm_frames = 2
        if (settings.get("centre_scan_policy") != self.CENTRE_SCAN_POLICY
                and np.isfinite(self.centre_scan_signal_multiplier)):
            # Migrate legacy weak-fallback maps to at least the 2× target,
            # preserving larger explicit settings and all measured poses.
            self.centre_scan_signal_multiplier = max(
                self.DEFAULT_CENTRE_SCAN_SIGNAL_MULTIPLIER, self.centre_scan_signal_multiplier
            )
        if np.isfinite(self.centre_scan_max_extra_descent_m):
            self.centre_scan_max_extra_descent_m = min(
                self.MAX_NONCONTACT_EXTRA_DESCENT_M, self.centre_scan_max_extra_descent_m
            )
        self.sensor_rows = int(settings["sensor_rows"])
        self.sensor_columns = int(settings["sensor_columns"])
        if (self.sensor_rows, self.sensor_columns) == (20, 20):
            # Apply the requested border exclusion when resuming an older
            # 400-node plan too; keep its already recorded data intact.
            self.skip_outer_border = True
        if "proximity_detection_mode" not in settings:
            # Maps saved before adaptive envelopes used the fixed-count rule.
            self.proximity_detection_mode = "fixed"
        self.anchor_pose = self._normalize_anchor_pose(result["anchor_pose"])
        # Keep the effective choice consistent with the profile used by idle UI
        # refreshes, without importing another map's obsolete operator choice.
        self.anchor_pose.pop("scan_plane_choice", None)
        if scan_choice is not None:
            self.anchor_pose["scan_plane_choice"] = scan_choice
        if self.active_sensor_key_name:
            self._anchor_profiles[self.active_sensor_key_name] = copy.deepcopy(self.anchor_pose)
        self.anchor_taxel = int(self.anchor_pose["taxel_index"])
        self.start_button_label = (
            f"Map {self.sensor_rows}x{self.sensor_columns} Sensor"
        )

    def _load_resume_samples(self, folder: Optional[Path]) -> list[dict]:
        if folder is None:
            return []
        path = folder / "scan_samples.csv"
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8", newline="") as stream:
                return list(csv.DictReader(stream))
        except (OSError, csv.Error):
            return []

    def _restore_resume_state(self, result, source_path=None) -> None:
        """Restore completed work while clearing only the interrupted step."""
        records = {
            int(record["taxel_index"]): copy.deepcopy(record)
            for record in result.get("positions", [])
            if isinstance(record, dict) and record.get("status") == "measured"
        }
        self._positions = records
        self._restore_relocation_checkpoint(result)
        self._resume_mapped_before = len(records)
        self._failures = {}
        self._baseline_checks = copy.deepcopy(result.get("baseline_checks") or [])
        self._sensor_calibration_history = copy.deepcopy(
            result.get("sensor_calibration_history") or []
        )
        self._baseline_stats_by_taxel = {}
        # Saved measurement attempts are historical evidence, not the budget
        # for this explicit Resume (including its required anchor re-check).
        # Keep record metadata unchanged; archive the prior live counters below.
        self._attempt_counts = {}
        self._reference_attempt_count = max(
            1, int(result.get("reference_attempt_count", 1) or 1)
        )
        self._reference_onset_z = float(result["reference_onset_z_m"])
        self._probe_start_z = float(result["probe_start_z_m"])
        if self.fixed_scan_active:
            # The previous threshold-derived start may lie BELOW the newly
            # chosen plane. Rebuild the guarded approach before any movement.
            self._reference_onset_z = float(self.fixed_scan_z_m)
            self._probe_start_z = min(
                float(self.anchor_pose["tool_position_m"][2]),
                float(self.fixed_scan_z_m) + self.fixed_scan_approach_window_m,
            )
        self._initial_row_vector_xy = self._saved_vector(
            result, "initial_row_vector_xy_m"
        )
        self._initial_column_vector_xy = self._saved_vector(
            result, "initial_column_vector_xy_m"
        )
        self._row_vector_xy = self._saved_vector(result, "row_vector_xy_m")
        self._column_vector_xy = self._saved_vector(result, "column_vector_xy_m")
        self._mapping_queue = []
        self._current_prediction = None
        self._neighbor_queue = []
        self._neighbor_guesses = {}
        self._predicted_positions = {}
        self._pending_recovery = None
        self._route_queue = []
        self._current_taxel = None
        self._current_context = ""
        self._current_target_xy = None
        self._far_baseline = None
        self._far_baseline_min = None
        self._far_baseline_max = None
        self._far_baseline_frames = None
        self._probe_detection_by_taxel = {
            int(index): {
                "mode": str(record.get("proximity_detection_mode") or self.proximity_detection_mode),
                "effective_threshold_counts": float(
                    record.get("proximity_threshold_counts")
                    or self.signal_threshold_counts
                ),
                "baseline_min_counts": record.get(
                    "proximity_baseline_min_counts"
                ),
                "baseline_max_counts": record.get(
                    "proximity_baseline_max_counts"
                ),
                "baseline_range_counts": record.get(
                    "proximity_baseline_range_counts"
                ),
                "margin_counts": record.get("proximity_margin_counts"),
                "lower_trigger_counts": record.get(
                    "proximity_lower_trigger_counts"
                ),
                "upper_trigger_counts": record.get(
                    "proximity_upper_trigger_counts"
                ),
            }
            for index, record in records.items()
        }
        self._verify_estimate = None
        self._local_recenters = 0
        self._scan_kind = ""
        self._sensor_calibration_started_at = None
        self._sensor_calibration_next_action = ""
        self._resume_anchor_previous = copy.deepcopy(records[int(self.anchor_taxel)])
        self._resume_alignment_error = ""
        if source_path is not None:
            source = Path(source_path).expanduser().resolve()
            self._output_folder = source.parent
        else:
            stored_path = result.get("json_path")
            self._output_folder = (
                Path(stored_path).expanduser().resolve().parent
                if stored_path
                else None
            )
        self._sample_log = self._load_resume_samples(self._output_folder)
        probe_path = None if self._output_folder is None else self._output_folder / "probe_levels.csv"
        if probe_path is not None and probe_path.is_file():
            with probe_path.open("r", encoding="utf-8", newline="") as stream:
                self._probe_log = list(csv.DictReader(stream))
        self._clearance_checks = copy.deepcopy(result.get("clearance_checks") or [])
        self._strength_checks = copy.deepcopy(result.get("stationary_strength_checks") or [])
        self._centre_checks = copy.deepcopy(result.get("centre_repeat_checks") or [])
        self._probe_strength_hold = None
        self._target_strength_check = None
        self._centre_repeat = None
        self._target_centre_check = None
        self._auto_rerun_history = copy.deepcopy(result.get("auto_rerun_history") or [])
        self._auto_rerun_count = max(
            0, int(result.get("auto_rerun_count", 0)),
            sum(row.get("status") == "started" for row in self._auto_rerun_history),
        )
        for row in self._auto_rerun_history:
            if row.get("status") == "waiting":
                row["status"] = "interrupted_wait"
                row["ended_at"] = datetime.now(timezone.utc).isoformat()
        self._started_at_iso = str(
            result.get("started_at") or datetime.now(timezone.utc).isoformat()
        )
        reference = finite_number(
            (result.get("settings") or {}).get("fixed_scan_reference_signal_counts")
        )
        self._fixed_scan_reference_signal = reference if reference and reference > 0.0 else None
        previous_fixed_mm = finite_number((result.get("settings") or {}).get("fixed_scan_z_mm"))
        if (not self.fixed_scan_active or previous_fixed_mm is None
                or not math.isclose(previous_fixed_mm / 1000., self.fixed_scan_z_m, abs_tol=1e-9, rel_tol=0.)):
            # A strength measured at a different Z cannot guard this plane.
            # The fixed-plane anchor check will collect a fresh reference.
            self._fixed_scan_reference_signal = None
        # An explicit Resume retries unresolved taxels with a fresh bounded
        # attempt budget. Keep measured positions and the old failure evidence.
        self._skipped_failed_taxels = set()
        self._failed_attempt_total = int(result.get("failed_attempt_total") or 0)
        self._resume_history = copy.deepcopy(result.get("resume_history") or [])
        self._resume_history.append(
            {
                "resumed_at": datetime.now(timezone.utc).isoformat(),
                "mapped_count_before_resume": len(records),
                "previous_stop_reason": str(result.get("stop_reason") or "unknown"),
                "previous_stop_snapshot": copy.deepcopy(result.get("stop_snapshot")),
                "previous_failures": copy.deepcopy(result.get("failures") or {}),
                "previous_attempt_counts": copy.deepcopy(result.get("attempt_counts") or {
                    str(index): record.get("attempt_count", 1)
                    for index, record in records.items()
                }),
                "attempt_budget_reset": True,
                "previous_fixed_scan_z_mm": previous_fixed_mm,
                "fixed_scan_z_mm": None if self.fixed_scan_z_m is None else self.fixed_scan_z_m * 1000.,
                "scan_plane_source": self._scan_plane_source,
                "retry_unresolved_taxels": list(result.get("skipped_failed_taxels") or []),
                "search_safety_policy": self.SEARCH_SAFETY_POLICY,
                "previous_mapping_order": (result.get("settings") or {}).get("mapping_order", "fixed_serpentine"),
                "mapping_order": self.MAPPING_ORDER,
                "previous_centre_scan_policy": (result.get("settings") or {}).get("centre_scan_policy", "weak_signal_fallback"),
                "previous_centre_scan_signal_multiplier": (result.get("settings") or {}).get("centre_scan_signal_multiplier"),
                "centre_scan_policy": self.CENTRE_SCAN_POLICY,
                "centre_scan_signal_multiplier": float(self.centre_scan_signal_multiplier),
            }
        )

    def _continue_after_resume_anchor_check(self, host) -> None:
        anchor = np.asarray(self._positions[self.anchor_taxel]["tool_position_m"], dtype=float)
        row_vector = self._initial_row_vector_xy
        column_vector = self._initial_column_vector_xy
        missing = [
            index
            for index in self.anchor_neighbour_indices()
            if index not in self._positions
        ]
        if missing and (row_vector is None or column_vector is None):
            self._request_stop(
                host,
                "resume_state_invalid",
                "Resume stopped: the saved initial taxel-grid directions are missing.",
            )
            return
        anchor_row, anchor_column = self._taxel_coords(self.anchor_taxel)
        self._neighbor_guesses = {}
        for index in missing:
            row, column = self._taxel_coords(index)
            xy = np.array(anchor[:2], copy=True)
            xy += (row - anchor_row) * row_vector
            xy += (column - anchor_column) * column_vector
            self._neighbor_guesses[index] = xy.tolist()
        self._neighbor_queue = missing
        host._append_sidebar_control_message(
            f"Resume verification passed. Keeping {self._resume_mapped_before} "
            f"saved taxels; continuing with Taxel {missing[0] if missing else 'grid fitting'}."
        )
        self._start_next_neighbor(host)

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
        return np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)

    def _read_sensor_frame(self, host):
        ui_ros = getattr(host, "ui_ros", None)
        sensor = getattr(ui_ros, "sensor_functions", None) if ui_ros else None
        data_obj = getattr(sensor, "_data", None) if sensor is not None else None
        if sensor is None or data_obj is None:
            return None
        if not bool(getattr(sensor, "is_connected", False)):
            return None
        rows = int(getattr(sensor, "n_row", 0) or 0)
        columns = int(getattr(sensor, "n_col", 0) or 0)
        if rows != self.sensor_rows or columns != self.sensor_columns:
            return None
        try:
            matrix = np.asarray(data_obj.diffDataAve, dtype=float)
            sequence = int(getattr(data_obj, "frame_sequence", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            return None
        if matrix.shape != (rows, columns) or not np.all(np.isfinite(matrix)):
            return None
        # diffDataAve already follows the Sensor Signal Viewer row direction.
        # Transpose before flattening to retain its column-major taxel IDs.
        return sequence, np.array(matrix.T.reshape(-1), dtype=float, copy=True)

    @staticmethod
    def _force_reader_running(ui_ros) -> bool:
        thread = getattr(ui_ros, "_force_meter_thread", None)
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except Exception:
            return False

    def _fresh_force(self, host, now=None):
        now = time.monotonic() if now is None else float(now)
        ui_ros = getattr(host, "ui_ros", None)
        if ui_ros is None or not self._force_reader_running(ui_ros):
            return None
        try:
            force_n = float(ui_ros._force_meter_last_raw_newtons)
            sample_at = float(ui_ros._force_meter_last_sample_monotonic)
        except (AttributeError, TypeError, ValueError):
            return None
        if (
            not np.isfinite(force_n)
            or not np.isfinite(sample_at)
            or not -0.1 <= now - sample_at <= float(self.force_stale_sec)
        ):
            return None
        return force_n, sample_at

    @staticmethod
    def grid_offsets(radius_m: float, step_m: float) -> list[tuple[float, float]]:
        """Return a deterministic serpentine square grid including its centre."""
        radius = max(0.0, float(radius_m))
        step = max(1e-6, float(step_m))
        count = max(1, int(np.ceil((2.0 * radius) / step)))
        axis = np.linspace(-radius, radius, count + 1)
        if not np.any(np.isclose(axis, 0.0, atol=1e-12)):
            axis = np.sort(np.append(axis, 0.0))
        offsets = []
        for row_index, dy in enumerate(axis):
            x_values = axis if row_index % 2 == 0 else axis[::-1]
            offsets.extend((float(dx), float(dy)) for dx in x_values)
        return offsets

    @staticmethod
    def _signal_metrics(frame, baseline, target_index: int):
        delta = np.abs(np.asarray(frame, dtype=float) - np.asarray(baseline, dtype=float))
        target = float(delta[int(target_index)])
        competitors = np.array(delta, copy=True)
        competitors[int(target_index)] = 0.0
        second = float(np.max(competitors)) if competitors.size else 0.0
        dominant = int(np.argmax(delta)) if delta.size else -1
        return target, second, target / max(second, 1e-9), dominant, delta

    @staticmethod
    def _rolling_medians(values, window: int) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        window = max(1, min(int(window), int(values.size)))
        return np.asarray(
            [
                np.median(values[index - window + 1 : index + 1])
                for index in range(window - 1, values.size)
            ],
            dtype=float,
        )

    def _detection_parameters(
        self,
        baseline,
        baseline_min,
        baseline_max,
        target_index: int,
        baseline_frames=None,
    ) -> dict:
        """Return the selected one-sided detector parameters for one taxel."""
        index = int(target_index)
        centre = float(np.asarray(baseline, dtype=float)[index])
        minimum = float(np.asarray(baseline_min, dtype=float)[index])
        maximum = float(np.asarray(baseline_max, dtype=float)[index])
        noise_range = max(0.0, maximum - minimum)
        robust_sigma = None
        rolling_low = None
        mode = str(self.proximity_detection_mode)
        if mode == "robust_shift":
            frames = np.asarray(baseline_frames, dtype=float)
            if frames.ndim == 2 and frames.shape[1] > index:
                series = frames[:, index]
            else:
                series = np.asarray([centre], dtype=float)
            detector = downward_detector(
                series, self.probe_window_frames, self.adaptive_min_margin_counts
            )
            centre, robust_sigma, rolling_low = detector["centre"], detector["sigma"], detector["low"]
            margin, lower = detector["margin"], detector["lower"]
        elif mode == "adaptive_range":
            margin = max(
                float(self.adaptive_min_margin_counts),
                noise_range * float(self.adaptive_margin_fraction),
            )
            lower = minimum - margin
        else:
            margin = float(self.signal_threshold_counts)
            lower = centre - margin
        # Proximity makes raw-calibration (diffDataAve) decrease for this
        # sensor, so a positive excursion must never count as an onset.
        effective_threshold = centre - lower
        return {
            "mode": mode,
            "baseline_mean_counts": centre,
            "baseline_min_counts": minimum,
            "baseline_max_counts": maximum,
            "baseline_range_counts": noise_range,
            "margin_counts": float(margin),
            "lower_trigger_counts": float(lower),
            "upper_trigger_counts": None,
            "effective_threshold_counts": float(effective_threshold),
            "rolling_window_frames": int(self.probe_window_frames),
            "rolling_low_percentile_counts": rolling_low,
            "rolling_robust_sigma_counts": robust_sigma,
        }

    @staticmethod
    def _below_detection_threshold(value: float, parameters: dict) -> bool:
        """Return true only for the physical, decreasing proximity direction."""
        return bool(
            float(value) <= float(parameters["lower_trigger_counts"])
        )

    def _signal_threshold_for_taxel(self, index: int) -> float:
        """Threshold used to validate scans after the probe onset."""
        saved = self._probe_detection_by_taxel.get(int(index))
        if saved is not None:
            return max(1e-9, float(saved["effective_threshold_counts"]))
        if (
            str(self.proximity_detection_mode) == "adaptive_range"
            and self._far_baseline is not None
            and self._far_baseline_min is not None
            and self._far_baseline_max is not None
        ):
            return max(
                1e-9,
                float(
                    self._detection_parameters(
                        self._far_baseline,
                        self._far_baseline_min,
                        self._far_baseline_max,
                        int(index),
                        baseline_frames=self._far_baseline_frames,
                    )["effective_threshold_counts"]
                ),
            )
        return max(1e-9, float(self.signal_threshold_counts))

    @classmethod
    def refine_xy_from_samples(cls, samples, baseline, target_index: int):
        """Estimate a taxel centre from the strongest target-specific samples."""
        candidates = []
        for sample in samples:
            target, second, dominance, _dominant, _delta = cls._signal_metrics(
                sample["frame"], baseline, target_index
            )
            candidates.append((target, second, dominance, np.asarray(sample["position"], dtype=float)))
        if not candidates:
            return None
        peak = max(item[0] for item in candidates)
        if peak <= 0.0:
            return None
        selected = [item for item in candidates if item[0] >= peak * 0.80]
        weights = np.asarray([max(item[0], 1e-9) for item in selected], dtype=float)
        positions = np.asarray([item[3] for item in selected], dtype=float)
        position = np.average(positions, axis=0, weights=weights)
        best = max(candidates, key=lambda item: item[0])
        return {
            "position": position,
            "signal_counts": float(peak),
            "competing_signal_counts": float(best[1]),
            "dominance_ratio": float(best[2]),
        }

    @staticmethod
    def lattice_vectors_from_points(
        points: dict,
        row_pair=(34, 36),
        column_pair=(27, 43),
    ):
        """Row/column step vectors from the anchor's orthogonal neighbours."""
        low_row, high_row = row_pair
        low_column, high_column = column_pair
        try:
            point = {
                index: np.asarray(points[index], dtype=float).reshape(-1)
                for index in (low_row, high_row, low_column, high_column)
            }
        except (KeyError, TypeError, ValueError):
            return None
        if any(
            value.size < 2 or not np.all(np.isfinite(value[:2]))
            for value in point.values()
        ):
            return None
        row_vector = (point[high_row][:2] - point[low_row][:2]) / 2.0
        column_vector = (point[high_column][:2] - point[low_column][:2]) / 2.0
        row_norm = float(np.linalg.norm(row_vector))
        column_norm = float(np.linalg.norm(column_vector))
        if row_norm < 0.0005 or column_norm < 0.0005:
            return None
        cross = abs(
            float(
                row_vector[0] * column_vector[1]
                - row_vector[1] * column_vector[0]
            )
        )
        orthogonality = cross / max(row_norm * column_norm, 1e-12)
        if orthogonality < 0.25:
            return None
        return row_vector, column_vector, orthogonality

    def fit_lattice_from_positions(self, positions: dict[int, dict]):
        """Fit row/column vectors from the anchor and its four neighbours."""
        neighbours = self.anchor_neighbour_indices()
        if len(neighbours) < 4:
            return None
        row_pair = (neighbours[0], neighbours[1])
        column_pair = (neighbours[2], neighbours[3])
        try:
            points = {
                index: positions[index]["tool_position_m"]
                for index in (
                    column_pair[0],
                    row_pair[0],
                    int(self.anchor_taxel),
                    row_pair[1],
                    column_pair[1],
                )
            }
        except (KeyError, TypeError):
            return None
        vectors = self.lattice_vectors_from_points(points, row_pair, column_pair)
        return vectors

    def predict_lattice_positions(self):
        fitted = self.fit_lattice_from_positions(self._positions)
        if fitted is None:
            return None
        row_vector, column_vector, orthogonality = fitted
        anchor_row, anchor_column = self._taxel_coords(self.anchor_taxel)
        origins = []
        neighbours = self.anchor_neighbour_indices()
        for index in (*neighbours, int(self.anchor_taxel)):
            row, column = self._taxel_coords(index)
            origin = np.asarray(
                self._positions[index]["tool_position_m"], dtype=float
            ).copy()
            origin[:2] -= (row - anchor_row) * row_vector
            origin[:2] -= (column - anchor_column) * column_vector
            origins.append(origin)
        anchor = np.mean(np.asarray(origins), axis=0)
        predicted = {}
        for index in range(self.sensor_rows * self.sensor_columns):
            row, column = self._taxel_coords(index)
            position = np.array(anchor, copy=True)
            position[:2] += (row - anchor_row) * row_vector
            position[:2] += (column - anchor_column) * column_vector
            predicted[index] = position
        return predicted, row_vector, column_vector, orthogonality

    def lattice_fit_quality(self) -> Optional[dict]:
        """Least-squares lattice through every measured centre."""
        measured = [
            (index, record)
            for index, record in sorted(self._positions.items())
            if record.get("status") == "measured"
        ]
        if len(measured) < 5:
            return None
        anchor_row, anchor_column = self._taxel_coords(self.anchor_taxel)
        design = []
        targets = []
        for index, record in measured:
            row, column = self._taxel_coords(index)
            design.append([1.0, row - anchor_row, column - anchor_column])
            targets.append(
                np.asarray(record["tool_position_m"], dtype=float)[:2]
            )
        design = np.asarray(design, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if np.linalg.matrix_rank(design) < 3:
            return None
        solution, *_ = np.linalg.lstsq(design, targets, rcond=None)
        distances = np.linalg.norm(targets - design @ solution, axis=1)
        return {
            "origin_xy_m": solution[0].tolist(),
            "row_vector_xy_m": solution[1].tolist(),
            "column_vector_xy_m": solution[2].tolist(),
            "rms_residual_mm": float(np.sqrt(np.mean(distances ** 2)) * 1000.0),
            "max_residual_mm": float(np.max(distances) * 1000.0),
            "residual_mm_by_taxel": {
                str(index): float(distance * 1000.0)
                for (index, _record), distance in zip(measured, distances)
            },
        }

    def _verified_mapping_positions(self) -> dict[int, np.ndarray]:
        """Only measured, finite centres may seed a new motion target."""
        verified = {}
        for index, record in self._positions.items():
            if record.get("status") != "measured" or record.get("relocation_verified") is False:
                continue
            try:
                point = np.asarray(record["tool_position_m"], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            if point.shape == (3,) and np.all(np.isfinite(point)):
                verified[int(index)] = point
        return verified

    def _cardinal_neighbours(self, index: int) -> list[int]:
        row, column = self._taxel_coords(index)
        candidates = (
            self._index_if_valid(row-1, column), self._index_if_valid(row+1, column),
            self._index_if_valid(row, column-1), self._index_if_valid(row, column+1),
        )
        return [candidate for candidate in candidates if candidate is not None]

    def plan_frontier_target(self, current_xy=None) -> Optional[dict]:
        """One grid step from verified neighbours, re-planned after each save.

        Never extrapolate across an unmapped gap. Choose the innermost available
        cardinal frontier, prefer multiple supports, then reduce travel. Nearby
        measured edges update the two step vectors; the five-centre fit is only
        a fallback/geometry sanity reference, not a distant target origin.
        """
        verified = self._verified_mapping_positions()
        remaining = self._remaining_target_indices()
        if not remaining:
            return None
        frontier = {}
        for known in sorted(verified):
            for index in self._cardinal_neighbours(known):
                if index in remaining:
                    frontier.setdefault(index, []).append(known)
        if not frontier:
            raise ValueError("No unmapped taxel has a verified cardinal neighbour; refusing a jump across unknown cells.")
        anchor_row, anchor_column = self._taxel_coords(self.anchor_taxel)
        current = None if current_xy is None else np.asarray(current_xy, dtype=float)

        def priority(index):
            row, column = self._taxel_coords(index)
            supported_xy = np.mean([verified[known][:2] for known in frontier[index]], axis=0)
            travel = 0. if current is None else float(np.linalg.norm(supported_xy-current))
            return (abs(row-anchor_row)+abs(column-anchor_column), -len(frontier[index]), travel, index)

        index = min(frontier, key=priority)
        row, column = self._taxel_coords(index)
        fitted = self.fit_lattice_from_positions({
            known: {"tool_position_m": point} for known, point in verified.items()
        })
        if fitted is None:
            raise ValueError("Verified anchor/reference neighbours no longer define a valid grid.")
        reference_vectors = fitted[:2]
        vectors, used_edges = [], []
        for axis, reference in enumerate(reference_vectors):
            reference = np.asarray(reference, dtype=float)
            reference_length = float(np.linalg.norm(reference))
            edges = []
            for known in sorted(verified):
                known_row, known_column = self._taxel_coords(known)
                next_index = self._index_if_valid(known_row+(axis == 0), known_column+(axis == 1))
                if next_index not in verified:
                    continue
                delta = verified[next_index][:2]-verified[known][:2]
                length = float(np.linalg.norm(delta))
                # Reject reversed/collapsed/implausibly long measured edges.
                # This does not enlarge the workspace or local scan radius.
                if not .5*reference_length <= length <= 2.*reference_length or np.dot(delta, reference) <= 0:
                    continue
                mid_row = known_row+.5*(axis == 0)
                mid_column = known_column+.5*(axis == 1)
                distance_sq = (mid_row-row)**2+(mid_column-column)**2
                if distance_sq <= 4.**2:
                    edges.append((distance_sq, known, next_index, delta))
            nearby = sorted(edges, key=lambda edge: edge[:3])[:8]
            vectors.append(np.median([edge[3] for edge in nearby], axis=0) if nearby else reference)
            used_edges.append([[edge[1], edge[2]] for edge in nearby])
        row_vector, column_vector = vectors
        row_length, column_length = np.linalg.norm(row_vector), np.linalg.norm(column_vector)
        separation = abs(row_vector[0]*column_vector[1]-row_vector[1]*column_vector[0]) / max(row_length*column_length, 1e-12)
        if min(row_length, column_length) < .0005 or separation < .25:
            raise ValueError("Local verified grid steps are collapsed or nearly parallel; mapping needs inspection.")
        predictions = []
        supports = sorted(frontier[index])
        for known in supports:
            known_row, known_column = self._taxel_coords(known)
            predictions.append(verified[known][:2]+(row-known_row)*row_vector+(column-known_column)*column_vector)
        xy = np.median(predictions, axis=0)
        return {
            "policy": self.MAPPING_ORDER, "taxel_index": int(index),
            "verified_neighbours": supports, "position_xy_m": xy.tolist(),
            "row_vector_xy_m": row_vector.tolist(), "column_vector_xy_m": column_vector.tolist(),
            "row_spacing_edges": used_edges[0], "column_spacing_edges": used_edges[1],
            "support_spread_mm": float(np.max(np.linalg.norm(np.asarray(predictions)-xy, axis=1))*1000.),
            "interpretation": "One-step local prediction, not an accepted centre or geometric accuracy estimate. Full probe/refinement/verification still required.",
        }

    def _index_if_valid(self, row: int, column: int) -> Optional[int]:
        if 0 <= int(row) < self.sensor_rows and 0 <= int(column) < self.sensor_columns:
            return self._taxel_flat(row, column)
        return None

    @staticmethod
    def axis_offsets(radius_m: float, step_m: float) -> list[float]:
        """Symmetric offsets along one axis, always including zero."""
        radius = max(0.0, float(radius_m))
        step = max(1e-6, float(step_m))
        count = max(1, int(np.ceil(radius / step - 1e-9)))
        return [float(value) for value in np.linspace(-radius, radius, 2 * count + 1)]

    @staticmethod
    def _peak_offset(offsets, values):
        offsets = np.asarray(offsets, dtype=float)
        values = np.asarray(values, dtype=float)
        if values.size < 3 or float(np.max(values)) <= 0.0:
            return None
        if float(np.ptp(offsets)) > 1e-9:
            curvature, slope, _intercept = np.polyfit(offsets, values, 2)
            if curvature < 0.0:
                return float(-slope / (2.0 * curvature)), "peak_quadratic"
        keep = values >= float(np.max(values)) * 0.80
        return (
            float(np.average(offsets[keep], weights=values[keep])),
            "peak_centroid",
        )

    @staticmethod
    def axis_balance_mode(axis) -> tuple[str, float]:
        """How one axis is centred, and where its scan window is centred.

        Interior: the two neighbours balance at the taxel centre. Edge: the
        taxel and its only neighbour balance half a pitch towards that
        neighbour, so the scan is centred there instead.
        """
        minus = axis.get("minus")
        plus = axis.get("plus")
        pitch = float(axis.get("pitch", 0.0))
        if minus is not None and plus is not None:
            return "neighbour_balance", 0.0
        if minus is not None and pitch > 0.0:
            return "edge_balance_minus", -0.5 * pitch
        if plus is not None and pitch > 0.0:
            return "edge_balance_plus", 0.5 * pitch
        return "peak", 0.0

    def estimate_axis_centre(self, samples, baseline, centre, axis, target_index):
        """Return the centre offset along one lattice axis, or None.

        ``bracket_m`` is how far the measured crossing (or peak) lies from the
        middle of the scan window; beyond the window radius it was not
        bracketed.
        """
        if len(samples) < 3:
            return None
        unit = np.asarray(axis["unit"], dtype=float)
        centre = np.asarray(centre, dtype=float)
        baseline = np.asarray(baseline, dtype=float)
        mode, window_centre = self.axis_balance_mode(axis)
        minus = axis.get("minus")
        plus = axis.get("plus")
        target_index = int(target_index)
        if mode == "neighbour_balance":
            high, low = plus, minus
        elif mode == "edge_balance_minus":
            high, low = target_index, minus
        elif mode == "edge_balance_plus":
            high, low = plus, target_index
        else:
            high = low = None
        offsets = []
        target = []
        balance = []
        for sample in samples:
            position = np.asarray(sample["position"], dtype=float)
            offsets.append(float(np.dot(position[:2] - centre[:2], unit)))
            delta = np.abs(np.asarray(sample["frame"], dtype=float) - baseline)
            target.append(float(delta[target_index]))
            if high is not None:
                balance.append(float(delta[int(high)] - delta[int(low)]))
        offsets = np.asarray(offsets, dtype=float)
        peak = self._peak_offset(offsets, target)
        result = {
            "peak_offset_m": None if peak is None else peak[0],
            "peak_method": None if peak is None else peak[1],
            "balance_offset_m": None,
            "balance_slope_counts_per_mm": None,
        }
        if balance and float(np.ptp(offsets)) > 1e-9:
            slope, intercept = np.polyfit(offsets, np.asarray(balance), 1)
            result["balance_slope_counts_per_mm"] = float(slope) / 1000.0
            # The balance must visibly change across the scan; a flat
            # difference makes -intercept/slope pure noise.
            if (
                slope > 0.0
                and slope * float(np.ptp(offsets))
                >= 0.25 * float(self.signal_threshold_counts)
            ):
                crossing = float(-intercept / slope)
                result["balance_offset_m"] = crossing - window_centre
                result["offset_m"] = crossing - window_centre
                result["bracket_m"] = crossing - window_centre
                result["method"] = mode
                return result
        if peak is None:
            return None
        result["offset_m"] = peak[0]
        result["bracket_m"] = peak[0] - window_centre
        result["method"] = peak[1]
        neighbours = [int(index) for index in (minus, plus) if index is not None]
        edge = self._lead_rises_to_window_edge(
            samples, baseline, offsets, window_centre, target_index,
            neighbours, self._signal_threshold_for_taxel(target_index),
        )
        if edge is not None and abs(result["bracket_m"]) <= float(axis.get("radius", 0.0)):
            # The peak sits inside the window only by noise: the target loses
            # to its own axis neighbour everywhere except at one end.
            result["offset_m"] = edge
            result["bracket_m"] = 1.5 * (edge - window_centre)
            result["method"] = "neighbour_lead_edge"
        return result

    @staticmethod
    def _lead_rises_to_window_edge(
        samples, baseline, offsets, window_centre, target_index,
        neighbours, threshold,
    ):
        """Offset of the window end the centre lies beyond, or ``None``.

        True only when the target loses to its axis neighbours in the middle
        of the window and first wins at one end.
        """
        if not neighbours or len(samples) < 5:
            return None
        baseline = np.asarray(baseline, dtype=float)
        leads = []
        for sample in samples:
            delta = np.abs(np.asarray(sample["frame"], dtype=float) - baseline)
            leads.append(
                float(delta[int(target_index)])
                - max(float(delta[index]) for index in neighbours)
            )
        order = np.argsort(offsets)
        ordered_offsets = np.asarray(offsets, dtype=float)[order]
        ordered = np.asarray(leads, dtype=float)[order]
        smoothed = np.convolve(
            np.pad(ordered, 1, mode="edge"), np.ones(3) / 3.0, mode="valid"
        )
        middle = int(np.argmin(np.abs(ordered_offsets - window_centre)))
        if smoothed[middle] > 0.0:
            return None
        best = int(np.argmax(smoothed))
        if best not in (0, smoothed.size - 1) or ordered[best] <= 0.0:
            return None
        if smoothed[best] - smoothed[middle] < max(1.0, 0.25 * float(threshold)):
            return None
        return float(ordered_offsets[best])

    def _progress(self, host, detail="") -> None:
        callback = getattr(host, "_refresh_taxel_mapping_progress", None)
        if callable(callback):
            current_taxel = self._current_taxel
            attempt = 0
            if current_taxel is not None:
                attempt = int(
                    self._attempt_counts.get(int(current_taxel), 0)
                )
            if self._current_context == "reference":
                attempt = int(self._reference_attempt_count)
            pending = self._pending_recovery
            if pending is not None:
                pending_index = pending.get("taxel_index")
                current_taxel = (
                    None
                    if pending_index is None
                    else int(pending_index)
                )
                attempt = int(pending.get("next_attempt", attempt))
            if self.fixed_scan_active or self._failed_attempt_total or self._skipped_failed_taxels:
                detail = (
                    f"Failed attempts {self._failed_attempt_total} · "
                    f"skipped {len(self._skipped_failed_taxels)} | {detail}"
                )
            callback(
                mapped=self.mapped_target_count(),
                total=len(self.mapping_target_indices()),
                phase=self._phase,
                detail=str(detail),
                current_taxel=current_taxel,
                attempt=attempt,
                max_attempts=int(self.max_attempts_per_taxel),
                auto_reruns=int(self._auto_rerun_count),
                max_auto_reruns=int(self.max_auto_reruns),
            )

    def _request_stop(self, host, reason: str, message: str) -> None:
        if not self._active:
            return
        self._stop_reason = str(reason)
        self._stop_message = str(message)
        CalibrationTask._stop_robot(self._robot_api(host))
        try:
            self._stop_snapshot = self._mapping_snapshot(host)
        except Exception as exc:
            self._stop_snapshot = {"diagnostic_error": str(exc)}
        host._append_sidebar_control_message(message)
        host._stop_sidebar_control(show_result=False, message=message)

    def _mapping_snapshot(self, host):
        """Feedback at a stopped/settled sample; never call it first contact."""
        pose = self._read_tool_pose(self._robot_api(host))
        force = self._fresh_force(host)
        z = None if pose is None else float(pose[0][2])
        baseline = self._force_baseline_n
        onset = self._probe_onset_pose
        start = self._probe_start_pose
        now = time.monotonic()
        pose_stamp = finite_number(getattr(self._robot_api(host), "current_feedback_state_time", None))
        force_stamp = finite_number(getattr(getattr(host, "ui_ros", None), "_force_meter_last_sample_monotonic", None))
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "taxel_index": self._current_taxel, "attempt": self._attempt_counts.get(self._current_taxel, 0),
            "phase": self._phase, "probe_stage": self._probe_stage,
            "tcp_position_m": None if pose is None else pose[0].tolist(),
            "tcp_x_m": None if pose is None else float(pose[0][0]),
            "tcp_y_m": None if pose is None else float(pose[0][1]),
            "tcp_z_m": z,
            "tcp_feedback_age_sec": None if pose_stamp is None else now-pose_stamp,
            "tcp_feedback_fresh": pose_stamp is not None and -.1 <= now-pose_stamp <= 1.,
            "force_sample_age_sec": None if force_stamp is None else now-force_stamp,
            "force_raw_g": None if force is None else float(force[0])/self.NEWTONS_PER_GRAM,
            "force_reference_g": None if baseline is None else float(baseline)/self.NEWTONS_PER_GRAM,
            "force_change_g": None if force is None or baseline is None else abs(float(force[0])-baseline)/self.NEWTONS_PER_GRAM,
            "probe_start_z_m": None if start is None else float(start[2]),
            "probe_descent_mm": None if z is None or start is None else max(0., float(start[2])-z)*1000.,
            "onset_confirmed": bool(self._probe_onset_confirmed),
            "onset_z_m": None if onset is None else float(onset[2]),
            "extra_descent_mm": None if z is None or onset is None else max(0., float(onset[2])-z)*1000.,
            "working_boundary": copy.deepcopy(self._working_probe_boundary),
            "working_boundary_z_m": None if self._working_probe_boundary is None else self._working_probe_boundary["z_m"],
            "working_boundary_source": None if self._working_probe_boundary is None else self._working_probe_boundary["source"],
            "interpretation": "Measured stop/sample feedback, not an independently verified contact point or air gap.",
        }

    def _begin_clearance_cycle(self, host, context):
        """Return to the operator-established clear anchor, never zero near-field."""
        self._clearance_next_context = str(context)
        anchor = self.anchor_pose["tool_position_m"]
        self._route_to(
            host, anchor[:2], anchor[2], "clearance_anchor_arrived",
            f"returning to Taxel {self.anchor_taxel} clearance anchor before full calibration",
        )

    def _continue_after_clearance(self, host):
        context = self._clearance_next_context
        self._clearance_next_context = ""
        if context == "resume_alignment_failed":
            self._request_stop(host, "resume_alignment_failed", self._resume_alignment_error)
        elif context == "resume_anchor":
            self._continue_after_resume_anchor_check(host)
        elif context == "anchor":
            self._neighbor_queue = list(self.anchor_neighbour_indices())
            self._start_next_neighbor(host)
        elif context == "neighbor":
            self._start_next_neighbor(host)
        elif context == "relocation_verify":
            self._start_next_relocation_verification(host)
        else:
            self._start_next_regular_taxel(host)

    def _queue_auto_rerun(self, host, pending):
        """Only an ordinary exhausted target can enter this guarded path."""
        if (self._auto_rerun_count >= self.max_auto_reruns
                or pending.get("kind") != "target"
                or (self._relocation_active and not self._relocation_finished)
                or pending.get("context") not in {"anchor", "neighbor", "regular"}
                or self._reference_onset_z is None or self._probe_start_z is None):
            return False
        CalibrationTask._stop_robot(self._robot_api(host))
        self._auto_rerun_pending = copy.deepcopy(pending)
        self._pending_recovery = None
        self._auto_rerun_deadline = time.monotonic() + self.auto_rerun_delay_sec
        self._phase = "auto_rerun_wait"
        self._auto_rerun_history.append({
            "rerun_number": self._auto_rerun_count+1,
            "status": "waiting", "queued_at": datetime.now(timezone.utc).isoformat(),
            "taxel_index": pending["taxel_index"], "reason": pending["reason"],
            "mapped_count": self.mapped_target_count(),
            "attempts_before_rerun": self._attempt_counts.get(pending["taxel_index"], 0),
            "delay_sec": float(self.auto_rerun_delay_sec),
        })
        message = (f"Taxel {pending['taxel_index']}: normal attempts exhausted ({pending['reason']}). "
                   f"At the clearance anchor; auto rerun {self._auto_rerun_count+1}/{self.max_auto_reruns} "
                   f"scheduled in {self.auto_rerun_delay_sec:.0f} s. Stop cancels it; safety guards remain active.")
        host._append_sidebar_control_message(message)
        self._progress(host, message)
        try:
            self._save_results(checkpoint=True)
        except Exception as exc:
            self._request_stop(host, "checkpoint_failed", f"Automatic rerun cancelled: checkpoint failed ({exc}).")
        return True

    def _at_auto_rerun_anchor(self, host):
        pose = self._read_tool_pose(self._robot_api(host))
        anchor = np.asarray(self.anchor_pose["tool_position_m"], dtype=float)
        orientation = np.asarray(self.anchor_pose["tool_quaternion_wxyz"], dtype=float)
        if pose is None:
            return False
        norm = float(np.linalg.norm(pose[1])*np.linalg.norm(orientation))
        return bool(np.linalg.norm(pose[0]-anchor) <= 2.*self.position_tolerance_m
                    and norm > 1e-9
                    and abs(float(np.dot(pose[1], orientation)))/norm >= np.cos(np.radians(2.5)))

    def _tick_auto_rerun_wait(self, host, now):
        # on_tick has already checked force, robot and tactile freshness.
        if not self._at_auto_rerun_anchor(host):
            self._request_stop(host, "auto_rerun_clearance_lost",
                               "Automatic rerun cancelled: TCP left the remembered clearance pose. Inspect before resuming.")
            return
        if self._auto_rerun_pending is None or self._auto_rerun_deadline is None:
            self._request_stop(host, "internal_error", "Automatic rerun cancelled: recovery state was lost.")
            return
        remaining = max(0., self._auto_rerun_deadline-now)
        self._progress(host, f"At clearance anchor: auto rerun {self._auto_rerun_count+1}/{self.max_auto_reruns} "
                       f"in {np.ceil(remaining):.0f} s; Stop cancels")
        if remaining > 0.:
            return
        self._auto_rerun_deadline = None
        self._clearance_next_context = "auto_rerun"
        self._begin_collection(host, "clearance_precheck", self.baseline_frame_count,
                               "Cooldown finished: checking clearance and recalibrating before automatic rerun")

    def _start_auto_rerun(self, host, frame, force_n, baseline_min, baseline_max, baseline_frames):
        pending = self._auto_rerun_pending
        if pending is None or self._auto_rerun_count >= self.max_auto_reruns or not self._at_auto_rerun_anchor(host):
            self._request_stop(host, "auto_rerun_limit", "Automatic rerun cancelled: recovery limit/state changed.")
            return
        index = int(pending["taxel_index"])
        self._auto_rerun_count += 1
        self._auto_rerun_history[-1].update(
            status="started", started_at=datetime.now(timezone.utc).isoformat(),
            clearance_pose_m=list(self.anchor_pose["tool_position_m"]),
        )
        self._auto_rerun_pending = None
        self._clearance_next_context = ""
        # Only the failed target's two-attempt round is reset. No measured
        # taxel, neighbour queue, geometry or previous failure history is lost.
        self._attempt_counts[index] = 0
        self._pending_recovery = dict(pending, retry=True, next_attempt=1)
        try:
            self._save_results(checkpoint=True)
        except Exception as exc:
            self._request_stop(host, "checkpoint_failed", f"Automatic rerun cancelled: checkpoint failed ({exc}).")
            return
        host._append_sidebar_control_message(
            f"Auto rerun {self._auto_rerun_count}/{self.max_auto_reruns}: continuing Taxel {index}; measured taxels retained.")
        self._handle_collection(host, "retry_anchor_stability", frame, force_n,
                                baseline_min, baseline_max, baseline_frames)

    @staticmethod
    def _project_speed_percent(api) -> Optional[float]:
        getter = getattr(api, "get_project_speed_percent", None)
        try:
            value = getter() if callable(getter) else getattr(
                api, "current_project_speed_percent", None
            )
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value) or value <= 0.0:
            return None
        return min(value, 100.0)

    def _move_timeout_for_distance(
        self,
        distance_m: float,
        project_speed_percent: Optional[float],
    ) -> tuple[float, float, float]:
        """Return timeout, expected duration, and speed used for one move.

        TM's project-speed override scales the Cartesian line velocity. Long
        safe-height routes therefore need more than the fixed minimum timeout
        when the controller is running at a low percentage.
        """
        distance_m = max(0.0, float(distance_m))
        try:
            speed_percent = float(project_speed_percent)
        except (TypeError, ValueError):
            speed_percent = float(
                self.move_timeout_fallback_project_speed_percent
            )
        if not np.isfinite(speed_percent) or speed_percent <= 0.0:
            speed_percent = float(
                self.move_timeout_fallback_project_speed_percent
            )
        speed_percent = min(max(speed_percent, 1.0), 100.0)
        effective_speed_m_s = max(
            1e-6,
            float(self.line_speed_m_s) * speed_percent / 100.0,
        )
        expected_duration_sec = distance_m / effective_speed_m_s
        calculated_timeout_sec = (
            expected_duration_sec * float(self.move_timeout_scale)
            + float(self.move_timeout_margin_sec)
        )
        timeout_sec = max(
            float(self.move_timeout_sec),
            min(float(self.move_timeout_max_sec), calculated_timeout_sec),
        )
        return timeout_sec, expected_duration_sec, speed_percent

    def _command_move(self, host, position, action: str, detail: str, *, quaternion=None,
                      vertical_only=False) -> bool:
        api = self._robot_api(host)
        quaternion = self._finite_values(
            (self.anchor_pose or {}).get("tool_quaternion_wxyz")
            if quaternion is None else quaternion, 4
        )
        position = self._finite_values(position, 3)
        if (
            api is None
            or not hasattr(api, "send_positions_tool_position")
            or position is None
            or quaternion is None
        ):
            self._request_stop(
                host, "robot_unavailable", "Mapping stopped: robot TCP position control is unavailable."
            )
            return False
        pose = self._read_tool_pose(api)
        start_position = None if pose is None else np.asarray(pose[0], dtype=float)
        target_position = np.asarray(position, dtype=float)
        boundary = self._working_probe_boundary
        if vertical_only:
            if pose is None:
                self._request_stop(host, "robot_feedback_lost",
                                   "Mapping stopped: TCP feedback was lost before vertical correction.")
                return False
            if boundary is not None and start_position[2] < boundary["z_m"] - self.position_tolerance_m:
                self._handle_probe_failure(host, "TCP moved too far below the known-height boundary before correction; refusing XY scanning")
                return False
            # Re-read XY/orientation at dispatch: tiny feedback changes since
            # the sampling tick must not turn an upward correction into XY.
            target_position[:2] = start_position[:2]
            target_position[2] = max(target_position[2], start_position[2])
            position = target_position.tolist()
            quaternion = pose[1].tolist()
        if boundary is not None and target_position[2] < boundary["z_m"]-1e-6:
            self._handle_probe_failure(
                host, f"requested scan Z {target_position[2]*1000.:.2f} mm is below "
                f"the onset-independent boundary {boundary['z_m']*1000.:.2f} mm",
            )
            return False
        if (boundary is not None and start_position is not None
                and start_position[2] < boundary["z_m"]-1e-6
                and np.linalg.norm(target_position[:2]-start_position[:2]) > 1e-6):
            self._handle_probe_failure(
                host, "TCP feedback is below the onset-independent boundary; "
                "refusing XY motion before a vertical clearance return",
            )
            return False
        distance_m = (
            0.0
            if start_position is None
            else float(np.linalg.norm(target_position - start_position))
        )
        reported_project_speed = self._project_speed_percent(api)
        (
            move_timeout_sec,
            expected_duration_sec,
            _speed_percent_used,
        ) = self._move_timeout_for_distance(
            distance_m,
            reported_project_speed,
        )
        CalibrationTask._stop_robot(api)
        try:
            sent = bool(
                api.send_positions_tool_position(
                    list(position),
                    list(quaternion),
                    velocity=float(self.line_speed_m_s),
                    acc_time=0.2,
                    blend_percentage=0,
                    fine_goal=True,
                    motion_type=self.LINE_MOTION,
                )
            )
        except Exception as exc:
            sent = False
            detail = f"{detail}: {exc}"
        if not sent:
            self._request_stop(
                host, "command_failed", f"Mapping stopped: robot rejected {detail}."
            )
            return False
        self._phase = "moving"
        self._move_target = np.asarray(position, dtype=float)
        self._move_action = str(action)
        self._move_started_at = time.monotonic()
        self._move_timeout_current_sec = float(move_timeout_sec)
        self._move_expected_duration_sec = float(expected_duration_sec)
        self._move_distance_m = float(distance_m)
        self._move_project_speed_percent = (
            None
            if reported_project_speed is None
            else float(reported_project_speed)
        )
        self._move_arrived_at = None
        self._move_last_position = None
        progress_detail = str(detail)
        if move_timeout_sec > float(self.move_timeout_sec) + 0.1:
            source = (
                "reported"
                if reported_project_speed is not None
                else "fallback"
            )
            progress_detail += (
                f"; allowing {move_timeout_sec:.0f} s for "
                f"{distance_m * 1000.0:.1f} mm at "
                f"{_speed_percent_used:.0f}% {source} project speed"
            )
        self._progress(host, progress_detail)
        return True

    def _tick_move(self, host, now: float) -> None:
        pose = self._read_tool_pose(self._robot_api(host))
        if pose is None or self._move_target is None:
            self._request_stop(
                host, "robot_feedback_lost", "Mapping stopped: robot TCP feedback was lost."
            )
            return
        position = np.asarray(pose[0], dtype=float)
        distance = float(np.linalg.norm(position - self._move_target))
        previous = self._move_last_position
        self._move_last_position = position.copy()
        still_moving = previous is None or float(
            np.linalg.norm(position - previous)
        ) > float(self.stationary_tolerance_m)
        if distance > float(self.position_tolerance_m) or still_moving:
            self._move_arrived_at = None
            if (
                self._move_started_at is not None
                and now - self._move_started_at
                > float(self._move_timeout_current_sec)
            ):
                speed_note = (
                    f" at {self._move_project_speed_percent:.0f}% project speed"
                    if self._move_project_speed_percent is not None
                    else (
                        " using the conservative "
                        f"{self.move_timeout_fallback_project_speed_percent:.0f}% "
                        "speed estimate"
                    )
                )
                self._request_stop(
                    host,
                    "move_timeout",
                    "Mapping stopped: robot did not reach its target within "
                    f"{self._move_timeout_current_sec:.1f} s for a "
                    f"{self._move_distance_m * 1000.0:.1f} mm move"
                    f"{speed_note}.",
                )
            return
        if self._move_arrived_at is None:
            self._move_arrived_at = now
            return
        if now - self._move_arrived_at < float(self.move_settle_sec):
            return
        action = self._move_action
        self._move_target = None
        self._move_action = ""
        self._dispatch_move_action(host, action)

    def _begin_collection(self, host, action: str, count: int, detail: str) -> None:
        self._phase = "collecting"
        self._collection_action = str(action)
        self._collection_frames = []
        self._collection_forces = []
        self._collection_count = max(1, int(count))
        self._collection_last_sequence = None
        self._progress(host, detail)

    def _begin_sensor_calibration(
        self,
        host,
        *,
        next_action: str,
        context: str,
    ) -> None:
        """Recalibrate at the motionless safe anchor before any baseline."""
        pose = self._read_tool_pose(self._robot_api(host))
        anchor = np.asarray(self.anchor_pose["tool_position_m"], dtype=float)
        if pose is None or np.linalg.norm(pose[0]-anchor) > 2.*self.position_tolerance_m:
            self._request_stop(host, "calibration_pose_not_clear", "Update Calibration refused: the robot has not reached the remembered clearance anchor.")
            return
        ui_ros = getattr(host, "ui_ros", None)
        sensor = getattr(ui_ros, "sensor_functions", None) if ui_ros else None
        if sensor is None or not callable(getattr(sensor, "updateCal", None)):
            self._request_stop(
                host,
                "sensor_calibration_unavailable",
                f"Mapping stopped at the safe Taxel {self.anchor_taxel} pose: full sensor "
                "calibration is unavailable.",
            )
            return
        if bool(getattr(sensor, "_sensor_calibration_in_progress", False)):
            self._request_stop(
                host,
                "sensor_calibration_busy",
                f"Mapping stopped at the safe Taxel {self.anchor_taxel} pose: another sensor "
                "calibration is already active.",
            )
            return
        CalibrationTask._stop_robot(self._robot_api(host))
        self._phase = "sensor_calibrating"
        self._sensor_calibration_started_at = time.monotonic()
        self._sensor_calibration_next_action = str(next_action)
        history = {
            "context": str(context),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "taxel_index": self._current_taxel,
            "clearance_pose_m": list(self.anchor_pose["tool_position_m"]),
            "clearance_interpretation": "Operator-established clear anchor; sensor stability is not independent clearance measurement.",
        }
        self._sensor_calibration_history.append(history)
        self._progress(
            host,
            f"Safe Taxel {self.anchor_taxel} pose reached; updating the full sensor calibration",
        )
        try:
            sensor.updateCal()
        except Exception as exc:
            history["status"] = "failed"
            history["error"] = str(exc)
            self._request_stop(
                host,
                "sensor_calibration_failed",
                f"Mapping stopped at the safe Taxel {self.anchor_taxel} pose: sensor "
                f"calibration failed ({exc}).",
            )
            return
        if not bool(getattr(sensor, "_sensor_calibration_in_progress", False)):
            history["status"] = "failed"
            history["error"] = "calibration did not start"
            self._request_stop(
                host,
                "sensor_calibration_failed",
                f"Mapping stopped at the safe Taxel {self.anchor_taxel} pose: the full sensor "
                "calibration did not start.",
            )
            return
        host._append_sidebar_control_message(
            f"Robot is stationary at the saved Taxel {self.anchor_taxel} safe pose. Full "
            "sensor calibration is running before the mapping baseline."
        )

    def _tick_sensor_calibration(self, host, now: float) -> None:
        ui_ros = getattr(host, "ui_ros", None)
        sensor = getattr(ui_ros, "sensor_functions", None) if ui_ros else None
        history = (
            self._sensor_calibration_history[-1]
            if self._sensor_calibration_history
            else None
        )
        if sensor is None:
            if history is not None:
                history["status"] = "failed"
                history["error"] = "sensor unavailable"
            self._request_stop(
                host,
                "sensor_calibration_unavailable",
                "Mapping stopped: sensor feedback became unavailable during "
                "the full calibration update.",
            )
            return
        started_at = self._sensor_calibration_started_at
        if (
            started_at is not None
            and now - float(started_at)
            >= float(self.sensor_calibration_timeout_sec)
        ):
            if history is not None:
                history["status"] = "failed"
                history["error"] = "calibration timed out"
            self._request_stop(
                host,
                "sensor_calibration_timeout",
                f"Mapping stopped at the safe Taxel {self.anchor_taxel} pose: full sensor "
                "calibration timed out.",
            )
            return
        if bool(getattr(sensor, "_sensor_calibration_in_progress", False)):
            return
        if not bool(getattr(sensor, "is_connected", False)):
            detail = ""
            error_getter = getattr(sensor, "get_last_sensor_stream_error", None)
            if callable(error_getter):
                detail = str(error_getter() or "").strip()
            if history is not None:
                history["status"] = "failed"
                history["error"] = detail or "sensor did not reconnect"
            message = (
                f"Mapping stopped at the safe Taxel {self.anchor_taxel} pose: full sensor "
                "calibration failed."
            )
            if detail:
                message += " " + detail
            self._request_stop(
                host,
                "sensor_calibration_failed",
                message,
            )
            return
        frame = self._read_sensor_frame(host)
        if frame is None or int(frame[0]) <= 0:
            self._progress(
                host,
                "Sensor calibration complete; waiting for a fresh tactile frame",
            )
            return
        action = self._sensor_calibration_next_action
        self._sensor_calibration_started_at = None
        self._sensor_calibration_next_action = ""
        self._last_sensor_sequence = int(frame[0])
        self._last_sensor_advance_at = float(now)
        # Tactile recalibration must never disable/re-zero the force guard.
        # The reference is refreshed only after a stable unloaded anchor batch.
        self._far_baseline = None
        self._far_baseline_frames = None
        if history is not None:
            history["status"] = "complete"
            history["completed_at"] = datetime.now(timezone.utc).isoformat()
            history["first_live_frame_sequence"] = int(frame[0])
        host._append_sidebar_control_message(
            f"Full sensor calibration completed at the saved Taxel {self.anchor_taxel} safe "
            "pose. Collecting the fresh mapping baseline."
        )
        detail = (
            f"Taxel {self.anchor_taxel} first probe: collecting the calibrated safe-anchor baseline"
            if action == "reference_baseline"
            else "Collecting the freshly calibrated clearance-anchor baseline"
        )
        self._begin_collection(
            host,
            action,
            self.baseline_frame_count,
            detail,
        )

    def _tick_collection(self, host, frame, force_n: float) -> None:
        sequence, values = frame
        if sequence == self._collection_last_sequence:
            return
        self._collection_last_sequence = int(sequence)
        self._collection_frames.append(np.asarray(values, dtype=float))
        self._collection_forces.append(float(force_n))
        if len(self._collection_frames) < self._collection_count:
            return
        frames = np.asarray(self._collection_frames, dtype=float)
        averaged = (
            np.median(frames, axis=0)
            if self._collection_action == "scan_sample"
            else np.mean(frames, axis=0)
        )
        baseline_min = np.min(frames, axis=0)
        baseline_max = np.max(frames, axis=0)
        averaged_force = float(np.mean(self._collection_forces))
        action = self._collection_action
        self._collection_action = ""
        if action in {
            "reference_baseline",
            "target_baseline",
            "retry_anchor_stability",
            "resume_anchor_stability",
            "clearance_precheck",
            "clearance_calibrated",
        }:
            stability = self._baseline_stability_stats(action, frames)
            self._baseline_checks.append(stability)
            target = stability.get("taxel_index")
            if action == "target_baseline" and target is not None:
                self._baseline_stats_by_taxel[int(target)] = stability
            if not bool(stability["stable"]):
                self._handle_unstable_baseline(host, action, stability)
                return
        self._handle_collection(
            host,
            action,
            averaged,
            averaged_force,
            baseline_min,
            baseline_max,
            frames,
        )

    def _baseline_stability_stats(self, action: str, frames) -> dict:
        """Summarize whether a baseline is quiet enough for a safe probe."""
        values = np.asarray(frames, dtype=float)
        peak_to_peak = np.ptp(values, axis=0)
        window = max(1, min(int(self.probe_window_frames), values.shape[0]))
        rolling_values = np.asarray(
            [
                np.median(values[index - window + 1 : index + 1], axis=0)
                for index in range(window - 1, values.shape[0])
            ],
            dtype=float,
        )
        rolling_peak_to_peak = np.ptp(rolling_values, axis=0)
        evaluation_values = (
            rolling_values
            if str(self.proximity_detection_mode) == "robust_shift"
            else values
        )
        split = max(1, evaluation_values.shape[0] // 2)
        first = np.median(evaluation_values[:split], axis=0)
        last = np.median(evaluation_values[-split:], axis=0)
        drift = np.abs(last - first)
        target = (
            int(self._current_taxel)
            if action == "target_baseline" and self._current_taxel is not None
            else int(self.anchor_taxel)
        )
        stability_limit = max(
            float(self.baseline_stability_floor_counts),
            float(self.signal_threshold_counts)
            * float(self.baseline_stability_fraction),
        )
        maximum_limit = max(
            float(self.signal_threshold_counts), stability_limit
        )
        target_range = float(peak_to_peak[target])
        grid_p90_range = float(np.percentile(peak_to_peak, 90.0))
        grid_max_range = float(np.max(peak_to_peak))
        target_rolling_range = float(rolling_peak_to_peak[target])
        grid_p90_rolling_range = float(
            np.percentile(rolling_peak_to_peak, 90.0)
        )
        grid_max_rolling_range = float(np.max(rolling_peak_to_peak))
        target_drift = float(drift[target])
        grid_p90_drift = float(np.percentile(drift, 90.0))
        target_mean = float(np.mean(values[:, target]))
        target_min = float(np.min(values[:, target]))
        target_max = float(np.max(values[:, target]))
        adaptive_margin = max(
            float(self.adaptive_min_margin_counts),
            target_range * float(self.adaptive_margin_fraction),
        )
        detection = self._detection_parameters(
            np.mean(values, axis=0),
            np.min(values, axis=0),
            np.max(values, axis=0),
            target,
            baseline_frames=values,
        )
        if str(self.proximity_detection_mode) == "robust_shift":
            stable = bool(
                target_rolling_range <= maximum_limit
                and grid_p90_rolling_range <= maximum_limit
                and grid_max_rolling_range <= maximum_limit
                and target_drift <= stability_limit
                and grid_p90_drift <= stability_limit
            )
        else:
            stable = bool(
                target_range <= stability_limit
                and grid_p90_range <= stability_limit
                and grid_max_range <= maximum_limit
                and target_drift <= stability_limit
                and grid_p90_drift <= stability_limit
            )
        attempt = (
            int(self._reference_attempt_count)
            if action == "reference_baseline"
            else int(self._attempt_counts.get(target, 0))
        )
        if action == "retry_anchor_stability":
            recovery = self._pending_recovery or {}
            attempt = int(recovery.get("next_attempt", attempt))
        return {
            "action": str(action),
            "taxel_index": target,
            "attempt_number": attempt,
            "frame_count": int(values.shape[0]),
            "stability_limit_counts": float(stability_limit),
            "maximum_limit_counts": float(maximum_limit),
            "target_range_counts": target_range,
            "grid_p90_range_counts": grid_p90_range,
            "grid_max_range_counts": grid_max_range,
            "target_rolling_median_range_counts": target_rolling_range,
            "grid_p90_rolling_median_range_counts": grid_p90_rolling_range,
            "grid_max_rolling_median_range_counts": grid_max_rolling_range,
            "target_drift_counts": target_drift,
            "grid_p90_drift_counts": grid_p90_drift,
            "target_mean_counts": target_mean,
            "target_min_counts": target_min,
            "target_max_counts": target_max,
            "adaptive_margin_counts": float(adaptive_margin),
            "adaptive_lower_trigger_counts": target_min - adaptive_margin,
            "adaptive_upper_trigger_counts": None,
            "robust_shift_threshold_counts": detection.get(
                "effective_threshold_counts"
            ),
            "rolling_robust_sigma_counts": detection.get(
                "rolling_robust_sigma_counts"
            ),
            "rolling_low_percentile_counts": detection.get(
                "rolling_low_percentile_counts"
            ),
            "stable": stable,
        }

    def _handle_unstable_baseline(
        self, host, action: str, stability: dict
    ) -> None:
        if str(self.proximity_detection_mode) == "robust_shift":
            detail = (
                "unstable tactile baseline "
                f"(target rolling-median range "
                f"{stability['target_rolling_median_range_counts']:.1f}, "
                f"grid P90 rolling range "
                f"{stability['grid_p90_rolling_median_range_counts']:.1f}, "
                f"maximum {stability['maximum_limit_counts']:.1f} sensor units)"
            )
        else:
            detail = (
                "unstable tactile baseline "
                f"(target range {stability['target_range_counts']:.1f}, "
                f"grid P90 {stability['grid_p90_range_counts']:.1f}, "
                f"limit {stability['stability_limit_counts']:.1f} sensor units)"
            )
        if action in {"clearance_precheck", "clearance_calibrated"}:
            self._request_stop(host, "clearance_unstable", "Mapping stopped at the clearance anchor: " + detail)
            return
        if action == "retry_anchor_stability":
            self._pending_recovery = None
            self._request_stop(
                host,
                "sensor_unstable",
                f"Mapping stopped at the Taxel {self.anchor_taxel} safe anchor: " + detail + ".",
            )
            return
        if action == "resume_anchor_stability":
            self._request_stop(
                host,
                "sensor_unstable",
                f"Resume stopped at the Taxel {self.anchor_taxel} safe anchor: " + detail + ".",
            )
            return
        if action == "reference_baseline":
            self._schedule_failure_recovery(
                host, kind="reference", reason=detail
            )
            return
        self._handle_probe_failure(host, detail)

    def _start_probe(
        self,
        host,
        baseline,
        force_baseline_n: float,
        kind: str,
        baseline_min=None,
        baseline_max=None,
        baseline_frames=None,
    ) -> None:
        pose = self._read_tool_pose(self._robot_api(host))
        if pose is None or self._current_taxel is None:
            self._request_stop(
                host, "robot_feedback_lost", "Mapping stopped: no TCP pose was available before probing."
            )
            return
        self._probe_kind = str(kind)
        if self._probe_kind == "reference":
            self._probe_speed_used_m_s = float(self.reference_probe_speed_m_s)
            self._probe_max_descent_used_m = float(self.reference_max_descent_m)
            detail = (
                f"Taxel {self.anchor_taxel}: first probe from the safe anchor to learn the "
                "sensor height"
            )
        else:
            # This is already the near-surface region. Never revert to the
            # fast search speed just because one candidate window was lost.
            self._probe_speed_used_m_s = min(
                float(self.probe_speed_m_s), float(self.probe_confirmation_speed_m_s),
            )
            self._probe_max_descent_used_m = float(self.max_probe_descent_m)
            detail = f"Taxel {self._current_taxel}: probing slowly in base -Z"
            # A near-start baseline may already contain proximity. Use the
            # far-field batch to detect onset; the near batch only checked
            # stability and must not zero away that existing response.
            if self._far_baseline is not None:
                baseline = self._far_baseline
                baseline_min = self._far_baseline_min
                baseline_max = self._far_baseline_max
                baseline_frames = self._far_baseline_frames
        self._probe_hit_count = 0
        self._probe_candidate_seen = False
        self._probe_last_sequence = None
        self._working_probe_boundary = None
        if self._probe_kind != "reference" and self.fixed_scan_active:
            # The descent itself stops at the plane (maximum descent below).
            # The guard sits a margin lower: the arm settles a fraction of a
            # millimetre past a commanded stop, which must not refuse the scan.
            self._working_probe_boundary = {
                "z_m": float(self.fixed_scan_z_m) - self.FIXED_SCAN_BOUNDARY_MARGIN_M,
                "source": "operator_fixed_scan_height",
                "taxels": [],
            }
            self._probe_max_descent_used_m = min(
                self._probe_max_descent_used_m,
                max(0.0, float(pose[0][2]) - float(self.fixed_scan_z_m)),
            )
            detail += (
                f"; fixed scan height Z {float(self.fixed_scan_z_m) * 1000.0:.2f} mm"
            )
        elif self._probe_kind != "reference":
            from phd.ui.mapping_safety import working_probe_boundary
            self._working_probe_boundary = working_probe_boundary(
                self._positions, self._current_taxel, self.sensor_rows, self.sensor_columns,
                self._reference_onset_z,
                discovery_allowance=min(.005, self.centre_scan_max_extra_descent_m),
            ) or {
                "z_m": float(pose[0][2])-min(self.max_probe_descent_m, self.centre_scan_max_extra_descent_m),
                "source": "onset_independent_total_working_allowance", "taxels": [],
            }
            self._probe_max_descent_used_m = min(
                self._probe_max_descent_used_m,
                max(0., float(pose[0][2])-self._working_probe_boundary["z_m"]),
            )
            detail += (
                f"; onset-independent Z boundary {self._working_probe_boundary['z_m']*1000.:.2f} mm "
                f"({self._working_probe_boundary['source']}); actual speed {self._probe_speed_used_m_s*1000.:.2f} mm/s"
            )
        baseline_frames_array = np.asarray(baseline_frames, dtype=float)
        if (
            str(self.proximity_detection_mode) == "robust_shift"
            and baseline_frames_array.ndim == 2
        ):
            self._probe_baseline = np.median(
                baseline_frames_array, axis=0
            )
        else:
            self._probe_baseline = np.asarray(baseline, dtype=float)
        self._probe_baseline_min = np.asarray(
            self._probe_baseline if baseline_min is None else baseline_min,
            dtype=float,
        )
        self._probe_baseline_max = np.asarray(
            self._probe_baseline if baseline_max is None else baseline_max,
            dtype=float,
        )
        detection = self._detection_parameters(
            self._probe_baseline,
            self._probe_baseline_min,
            self._probe_baseline_max,
            int(self._current_taxel),
            baseline_frames=baseline_frames_array,
        )
        self._probe_detection_lower = float(detection["lower_trigger_counts"])
        self._probe_detection_upper = None
        self._probe_detection_margin = float(detection["margin_counts"])
        self._probe_detection_threshold = float(
            detection["effective_threshold_counts"]
        )
        self._probe_detection_by_taxel[int(self._current_taxel)] = detection
        self._probe_recent_frames = []
        self._probe_candidate_pose = None
        self._probe_stage = "descending"
        self._probe_line_target_z = None
        self._probe_step_start_z = float(pose[0][2])
        self._probe_level_pose = None
        self._probe_level_frames = []
        self._probe_settle_started_at = None
        self._probe_level_at_max_depth = False
        self._probe_boundary_corrections = 0
        self._probe_strength_hold = None
        self._target_strength_check = None
        self._centre_repeat = None
        self._target_centre_check = None
        self._probe_onset_confirmed = False
        self._probe_onset_pose = None
        self._probe_scan_pose = None
        self._probe_scan_signal = 0.0
        self._probe_scan_target_signal = (
            float(self.centre_scan_signal_multiplier)
            * float(self._probe_detection_threshold)
        )
        saved_target = self._resume_anchor_saved_scan_target()
        if saved_target is not None:
            # The resume check is only comparable when it scans at the signal
            # strength the anchor was saved with; a new noise-derived target
            # moves the scan plane and makes the XY/Z comparison meaningless.
            self._probe_scan_target_signal = max(
                saved_target, float(self._probe_detection_threshold)
            )
            detail += (
                f"; resume check uses the saved {saved_target:.1f}-unit "
                "centre-scan signal"
            )
        self._probe_scan_target_reached = False
        if detection["mode"] == "robust_shift":
            detail += (
                f"; robust {int(self.probe_window_frames)}-frame median, "
                f"downward signal-change trigger "
                f"{detection['effective_threshold_counts']:.1f} sensor units"
            )
        elif detection["mode"] == "adaptive_range":
            detail += (
                f"; baseline {detection['baseline_min_counts']:.1f} to "
                f"{detection['baseline_max_counts']:.1f} sensor units, downward "
                f"trigger at ≤{detection['lower_trigger_counts']:.1f}"
            )
        else:
            detail += (
                f"; fixed downward change of "
                f"{float(self.signal_threshold_counts):.1f} sensor units"
            )
        detail += (
            f"; step {float(self.probe_step_m) * 1000.0:.2f} mm and sample "
            f"{int(self.probe_sample_frames)} stationary frames per level"
        )
        # Never re-zero force at the lower probe-start height: contact/load
        # picked up during the approach must remain visible to the guards.
        if self._probe_kind == "reference" or self._force_baseline_n is None:
            self._force_baseline_n = float(force_baseline_n)
        self._probe_start_pose = np.asarray(pose[0], dtype=float)
        self._phase = "probing"
        self._progress(host, detail)

    def _start_fixed_scan_mapping(self, host, force_n) -> None:
        """The operator chose the scan plane, so no reference probe is needed."""
        anchor_z = float(self.anchor_pose["tool_position_m"][2])
        scan_z = float(self.fixed_scan_z_m)
        self._force_baseline_n = float(force_n)
        self._reference_onset_z = scan_z
        self._probe_start_z = min(
            anchor_z, scan_z + float(self.fixed_scan_approach_window_m)
        )
        host._append_sidebar_control_message(
            f"Fixed scan height Z {scan_z * 1000.0:.2f} mm "
            f"({(anchor_z - scan_z) * 1000.0:.1f} mm below the anchor). Each taxel "
            f"descends in {float(self.probe_step_m) * 1000.0:.1f} mm steps from "
            f"{(self._probe_start_z - scan_z) * 1000.0:.1f} mm above it."
        )
        anchor_xy = np.asarray(self.anchor_pose["tool_position_m"], dtype=float)[:2]
        self._start_target(host, self.anchor_taxel, anchor_xy, "anchor")

    def _fixed_scan_strongest(self, frame, baseline) -> tuple[float, int]:
        delta = np.abs(np.asarray(frame, dtype=float) - np.asarray(baseline, dtype=float))
        index = int(np.argmax(delta))
        return float(delta[index]), index

    def _fixed_scan_contact_limit(self) -> Optional[float]:
        reference = self._fixed_scan_reference_signal
        if not self.fixed_scan_active or reference is None:
            return None
        return float(self.fixed_scan_contact_ratio) * float(reference)

    def _fixed_scan_contact(self, host, frame, baseline, where: str) -> bool:
        """Abort the attempt (lift, recalibrate, retry) on a contact-like response."""
        limit = self._fixed_scan_contact_limit()
        if limit is None:
            return False
        strongest, index = self._fixed_scan_strongest(frame, baseline)
        if strongest < limit:
            return False
        CalibrationTask._stop_robot(self._robot_api(host))
        self._handle_probe_failure(
            host,
            f"Taxel {index} reached {strongest:.0f} sensor units {where}, above the "
            f"{self.fixed_scan_contact_ratio:.1f}× contact guard ({limit:.0f}); "
            "possible contact, lifting to the anchor",
        )
        return True

    def _tick_fixed_scan_level(
        self, host, evaluation_frame, level_pose, target, at_depth_limit=False
    ) -> bool:
        """Decide one stationary level of a fixed-plane descent.

        Returns True when the level was handled (scan started, failure
        scheduled); False to keep descending.
        """
        if self._fixed_scan_contact(
            host, evaluation_frame, self._probe_baseline, "while descending"
        ):
            return True
        strongest, _index = self._fixed_scan_strongest(evaluation_frame, self._probe_baseline)
        reference = self._fixed_scan_reference_signal
        raised_skin = (
            reference is not None
            and strongest >= float(self.fixed_scan_stop_ratio) * float(reference)
        )
        at_plane = bool(at_depth_limit) or (
            float(level_pose[2]) <= float(self.fixed_scan_z_m) + float(self.position_tolerance_m)
        )
        if not (raised_skin or at_plane):
            return False
        CalibrationTask._stop_robot(self._robot_api(host))
        target_signal = max(
            0.0,
            abs(float(evaluation_frame[target]) - float(self._probe_baseline[target])),
        )
        if reference is None:
            # The anchor taxel defines the expected strength at this plane.
            threshold = self._signal_threshold_for_taxel(target)
            if target_signal < 3.0 * threshold:
                self._handle_probe_failure(
                    host,
                    f"the anchor signal at the fixed scan height is too weak "
                    f"({target_signal:.1f} sensor units, need {3.0 * threshold:.1f})",
                )
                return True
            self._fixed_scan_reference_signal = float(target_signal)
            host._append_sidebar_control_message(
                f"Reference strength at the fixed scan height: {target_signal:.1f} sensor units "
                f"on Taxel {target}. Descents stop early above "
                f"{self.fixed_scan_stop_ratio * target_signal:.0f}; any response above "
                f"{self.fixed_scan_contact_ratio * target_signal:.0f} is treated as possible contact."
            )
        pose = np.array(level_pose, dtype=float, copy=True)
        # An overshoot past the plane is not a deeper scan: every taxel is
        # scanned at the remembered height (or higher over raised skin).
        pose[2] = max(float(pose[2]), float(self.fixed_scan_z_m))
        self._probe_onset_pose = np.array(pose, copy=True)
        self._probe_scan_pose = np.array(pose, copy=True)
        self._probe_scan_signal = float(target_signal)
        self._probe_onset_confirmed = True
        self._probe_scan_target_reached = True
        if raised_skin and not at_plane:
            host._append_sidebar_control_message(
                f"Taxel {target}: raised skin, the descent stopped "
                f"{(pose[2] - float(self.fixed_scan_z_m)) * 1000.0:.2f} mm above the scan height."
            )
        current = self._read_tool_pose(self._robot_api(host))
        if (
            current is not None
            and float(current[0][2]) < float(pose[2]) - float(self.position_tolerance_m)
        ):
            self._fixed_scan_plane_pose = np.array(pose, copy=True)
            lift = np.array(current[0], dtype=float, copy=True)
            lift[2] = float(pose[2])
            self._command_move(
                host, lift, "fixed_scan_plane_arrived",
                f"Taxel {target}: settling back up to the scan height",
            )
            return True
        self._start_fixed_plane_scan(host, pose)
        return True

    def _start_fixed_plane_scan(self, host, pose) -> None:
        pose = np.asarray(pose, dtype=float)
        if self._current_context == "anchor":
            # The wide neighbour search crosses skin that may stand higher
            # than at the anchor; it only needs approximate neighbour centres.
            coarse_pose = np.array(pose, copy=True)
            coarse_pose[2] = min(
                float(self.anchor_pose["tool_position_m"][2]),
                float(pose[2]) + self.FIXED_SCAN_COARSE_LIFT_M,
            )
            self._start_coarse_scan(host, self._far_baseline, coarse_pose)
        else:
            self._start_local_scan(host, pose)

    def _resume_anchor_saved_scan_target(self) -> Optional[float]:
        if (
            self._current_context != "resume_anchor"
            or self._probe_kind == "reference"
            or self._current_taxel is None
            or int(self._current_taxel) != int(self.anchor_taxel)
        ):
            return None
        value = finite_number(
            (self._resume_anchor_previous or {}).get("centre_scan_target_units")
        )
        if value is None or value <= 0.0:
            return None
        return float(value)

    def _confirm_stationary_strength(self, host, pose, signal, qualified, sample):
        """Three disjoint batches at ONE pose, without moving or lowering the target.

        Return None while collecting, otherwise the acceptance decision. A
        transient peak cannot win: the combined median, two batch medians and
        the final batch must reach the unchanged target with valid dominance.
        """
        hold = self._probe_strength_hold
        if hold is None:
            hold = {"pose": np.array(pose, copy=True), "frames": [], "batches": []}
            self._probe_strength_hold = hold
            host._append_sidebar_control_message(
                f"Taxel {self._current_taxel}: confirming borderline scan strength "
                f"at the SAME height ({signal:.1f}/{self._probe_scan_target_signal:.1f} "
                f"sensor units): {self.STRENGTH_CONFIRM_BATCHES} stationary batches; no deeper move."
            )
        if np.linalg.norm(np.asarray(pose) - hold["pose"]) > 2.0 * self.position_tolerance_m:
            self._request_stop(host, "probe_hold_moved",
                               "Mapping stopped: TCP moved during stationary strength confirmation.")
            return None
        hold["frames"].extend(np.array(self._probe_level_frames, copy=True))
        hold["batches"].append({"signal_units": float(signal), "qualified": bool(qualified)})
        sample["strength_confirmation_batch"] = len(hold["batches"])
        if len(hold["batches"]) < self.STRENGTH_CONFIRM_BATCHES:
            self._probe_level_frames = []
            self._progress(host, f"Taxel {self._current_taxel}: stationary strength check "
                           f"{len(hold['batches'])}/{self.STRENGTH_CONFIRM_BATCHES} at "
                           f"Z {float(pose[2])*1000.:.2f} mm; no further descent")
            return None
        target = int(self._current_taxel)
        median = np.median(hold["frames"], axis=0)
        signal = max(0., float(self._probe_baseline[target] - median[target]))
        dominance = self._signal_metrics(median, self._probe_baseline, target)[2]
        threshold = float(self._probe_scan_target_signal)
        strong = [batch["qualified"] and batch["signal_units"] >= threshold for batch in hold["batches"]]
        accepted = bool(
            signal >= threshold and sum(strong) >= 2 and strong[-1]
            and dominance >= self.probe_min_dominance
            and self._below_detection_threshold(float(median[target]), self._probe_detection_by_taxel[target])
        )
        check = {
            "taxel_index": target, "attempt": int(self._attempt_counts.get(target, 1)),
            "tcp_position_m": hold["pose"].tolist(), "frame_count": len(hold["frames"]),
            "batches": hold["batches"], "signal_units": signal,
            "target_units": threshold, "dominance": float(dominance), "accepted": accepted,
            "at_depth_limit": bool(self._probe_level_at_max_depth),
        }
        self._strength_checks.append(check)
        self._target_strength_check = check
        sample.update(strength_confirmation_accepted=accepted,
                      strength_confirmation_median_units=signal)
        self._probe_strength_hold = None
        if accepted:
            self._probe_scan_signal = signal
            self._probe_scan_pose = np.array(pose, copy=True)
        return accepted

    def _tick_probe(self, host, frame) -> None:
        if self._probe_baseline is None or self._probe_start_pose is None:
            self._request_stop(host, "internal_error", "Mapping stopped: probe baseline is missing.")
            return
        api = self._robot_api(host)
        pose = self._read_tool_pose(api)
        if pose is None:
            self._request_stop(host, "robot_feedback_lost", "Mapping stopped: TCP feedback was lost during probing.")
            return
        target = int(self._current_taxel)
        descent = max(0.0, float(self._probe_start_pose[2] - pose[0][2]))
        sequence, values = frame
        now = time.monotonic()

        # Never evaluate a stationary batch below the unchanged working
        # boundary. A tiny arrival undershoot may be corrected upwards once;
        # all old strength evidence is discarded before trying the scan again.
        if self._probe_stage in ("settling", "sampling"):
            if self._correct_probe_boundary_undershoot(host, pose):
                return

        if self._probe_stage == "settling":
            started_at = self._probe_settle_started_at
            if (
                started_at is not None
                and now - float(started_at) < float(self.move_settle_sec)
            ):
                return
            self._probe_stage = "sampling"
            self._probe_level_frames = []
            # Start with the next fresh sequence so the measurement contains
            # only readings acquired after the robot stopped at this Z level.
            self._probe_last_sequence = int(sequence)
            return

        if self._probe_stage == "sampling":
            if int(sequence) == self._probe_last_sequence:
                return
            self._probe_last_sequence = int(sequence)
            self._probe_level_frames.append(np.asarray(values, dtype=float))
            if len(self._probe_level_frames) < max(
                1, int(self.probe_sample_frames)
            ):
                return

            evaluation_frame = np.median(
                np.asarray(self._probe_level_frames, dtype=float), axis=0
            )
            _target_signal, _second, dominance, _dominant, _delta = (
                self._signal_metrics(
                    evaluation_frame, self._probe_baseline, target
                )
            )
            downward_signal = max(
                0.0,
                float(self._probe_baseline[target])
                - float(evaluation_frame[target]),
            )
            qualified = bool(
                self._below_detection_threshold(
                    float(evaluation_frame[target]),
                    self._probe_detection_by_taxel[target],
                )
                and dominance >= float(self.probe_min_dominance)
            )
            if qualified:
                self._probe_candidate_seen = True
            sample = self._mapping_snapshot(host)
            sample["onset_confirmed_before_level"] = sample.pop("onset_confirmed")
            sample.update(
                signal_change_units=float(downward_signal),
                onset_threshold_units=float(self._probe_detection_threshold),
                stronger_target_units=float(self._probe_scan_target_signal),
                qualified=qualified, dominance=float(dominance),
                sensor_frame_sequence=int(sequence),
            )
            self._probe_log.append(sample)
            level_pose = np.asarray(
                self._probe_level_pose
                if self._probe_level_pose is not None
                else pose[0],
                dtype=float,
            )
            if self.fixed_scan_active and self._probe_kind != "reference":
                if self._tick_fixed_scan_level(
                    host, evaluation_frame, level_pose, target,
                    at_depth_limit=self._probe_level_at_max_depth,
                ):
                    return
                self._probe_level_frames = []
                self._probe_stage = "descending"
                self._probe_line_target_z = None
                self._probe_step_start_z = float(pose[0][2])
                self._probe_level_pose = None
                return
            if not self._probe_onset_confirmed:
                if qualified:
                    if self._probe_hit_count == 0:
                        self._probe_candidate_pose = np.array(
                            level_pose, copy=True
                        )
                        self._probe_scan_pose = np.array(level_pose, copy=True)
                        self._probe_scan_signal = float(downward_signal)
                    elif downward_signal > float(self._probe_scan_signal):
                        self._probe_scan_pose = np.array(level_pose, copy=True)
                        self._probe_scan_signal = float(downward_signal)
                    self._probe_hit_count += 1
                else:
                    self._probe_hit_count = 0
                    self._probe_candidate_pose = None
                    self._probe_scan_pose = None
                    self._probe_scan_signal = 0.0

                if self._probe_hit_count >= max(
                    1, int(self.probe_confirm_frames)
                ):
                    self._probe_onset_pose = np.asarray(
                        self._probe_candidate_pose
                        if self._probe_candidate_pose is not None
                        else level_pose,
                        dtype=float,
                    )
                    self._probe_onset_confirmed = True
                    # The first long reference probe only learns the onset
                    # height.  The later working probe chooses the stronger
                    # plane used by the XY centre scan.
                    if self._probe_kind == "reference":
                        CalibrationTask._stop_robot(api)
                        self._handle_probe_onset(host)
                        return
            elif qualified and downward_signal > float(self._probe_scan_signal):
                self._probe_scan_pose = np.array(level_pose, copy=True)
                self._probe_scan_signal = float(downward_signal)

            if self._probe_onset_confirmed and self._probe_kind != "reference":
                # Use the current stationary level, not a historical maximum
                # acquired at a different height or before onset confirmation.
                strong_enough = qualified and downward_signal >= self._probe_scan_target_signal
                borderline = bool(
                    qualified
                    and downward_signal >= self.STRENGTH_BORDERLINE_RATIO * self._probe_scan_target_signal
                    and (self._probe_level_at_max_depth or
                         1.0 <= downward_signal / max(self._probe_scan_target_signal, 1e-9)
                         < self.STRENGTH_CLEAR_RATIO)
                )
                if self._probe_strength_hold is not None or borderline:
                    strong_enough = self._confirm_stationary_strength(
                        host, pose[0], downward_signal, qualified, sample
                    )
                    if strong_enough is None:
                        return
                elif strong_enough:
                    self._probe_scan_pose = np.array(level_pose, copy=True)
                    self._probe_scan_signal = float(downward_signal)
                if strong_enough:
                    self._probe_scan_target_reached = True
                    CalibrationTask._stop_robot(api)
                    self._handle_probe_onset(host)
                    return

            self._probe_level_frames = []
            if self._probe_level_at_max_depth:
                if self._probe_onset_confirmed:
                    CalibrationTask._stop_robot(api)
                    self._handle_probe_failure(
                        host,
                        f"stronger non-contact scan signal was not reached reliably before the descent limit "
                        f"(latest batch {downward_signal:.1f}/{self._probe_scan_target_signal:.1f} sensor units); "
                        f"TCP Z {float(pose[0][2])*1000.:.2f} mm, total probe descent {descent*1000.:.2f} mm; "
                        f"boundary {self._working_probe_boundary}; "
                        + (f"stationary confirmation median {self._target_strength_check['signal_units']:.1f} "
                           f"from {self._target_strength_check['frame_count']} frames; "
                           if self._target_strength_check is not None else "")
                        + "refusing a weak-signal XY scan or further descent",
                        quality_failure=True,
                    )
                    return
                if qualified:
                    # Confirm again at the same level rather than exceeding
                    # the configured maximum descent.
                    self._probe_last_sequence = int(sequence)
                    return
                self._handle_probe_failure(
                    host,
                    "no proximity onset within "
                    f"{self._probe_max_descent_used_m * 1000.0:.2f} mm; "
                    f"TCP Z {float(pose[0][2])*1000.:.2f} mm, "
                    f"onset-independent boundary {self._working_probe_boundary}",
                    quality_failure=True,
                )
                return

            self._probe_stage = "descending"
            self._probe_line_target_z = None
            self._probe_step_start_z = float(pose[0][2])
            self._probe_level_pose = None
            return

        step_start_z = (
            float(self._probe_start_pose[2])
            if self._probe_step_start_z is None
            else float(self._probe_step_start_z)
        )
        if self.probe_motion == "stepwise":
            self._tick_probe_line_step(host, api, pose, now, sequence, step_start_z)
            return
        step_descent = max(0.0, step_start_z - float(pose[0][2]))
        boundary = self._working_probe_boundary
        boundary_headroom = (float(self.position_tolerance_m) if self.fixed_scan_active
                             else self._probe_boundary_headroom_m())
        reached_boundary = boundary is not None and float(pose[0][2]) <= boundary["z_m"] + boundary_headroom
        reached_probe_limit = descent >= float(self._probe_max_descent_used_m) or reached_boundary
        reached_scan_limit = bool(
            self._probe_onset_confirmed
            and self._probe_onset_pose is not None
            and max(
                0.0,
                float(self._probe_onset_pose[2]) - float(pose[0][2]),
            )
            >= float(self.centre_scan_max_extra_descent_m)
        )
        reached_limit = reached_probe_limit or reached_scan_limit
        if step_descent >= float(self.probe_step_m) or reached_limit:
            CalibrationTask._stop_robot(api)
            self._probe_stage = "settling"
            self._probe_settle_started_at = now
            self._probe_level_pose = np.asarray(pose[0], dtype=float)
            self._probe_level_at_max_depth = bool(reached_limit)
            self._probe_last_sequence = int(sequence)
            return

        speed_m_s = (
            min(
                abs(float(self._probe_speed_used_m_s)),
                abs(float(self.probe_confirmation_speed_m_s)),
            )
            if self._probe_candidate_seen or self._probe_onset_confirmed
            else abs(float(self._probe_speed_used_m_s))
        )
        if not CalibrationTask._send_base_z_velocity(api, -speed_m_s):
            self._request_stop(
                host, "command_failed", "Mapping stopped: robot rejected the base -Z probe command."
            )

    def _probe_boundary_headroom_m(self) -> float:
        # Command above, not on, the guard: arrival tolerance must not authorize
        # a scan below it. The second tolerance leaves room for feedback jitter.
        return max(2e-6, 2.0 * float(self.position_tolerance_m))

    def _correct_probe_boundary_undershoot(self, host, pose, *, sample_pose=None) -> bool:
        """Return True if XY must wait for correction/recovery; never lower a guard."""
        boundary = self._working_probe_boundary
        if boundary is None or self.fixed_scan_active or self._probe_kind == "reference":
            return False
        boundary_z = float(boundary["z_m"])
        measured_z = float(pose[0][2])
        if sample_pose is not None:
            measured_z = min(measured_z, float(sample_pose[2]))
        undershoot = boundary_z - measured_z
        if undershoot <= 1e-6:
            return False
        if undershoot > float(self.position_tolerance_m) or self._probe_boundary_corrections:
            self._handle_probe_failure(
                host, "TCP/sample exceeded the onset-independent known-height boundary by "
                f"{undershoot * 1000.:.4f} mm; excessive or repeated undershoot, "
                "refusing XY scanning",
            )
            return True
        lift = np.array(pose[0], dtype=float, copy=True)
        lift[2] = max(float(lift[2]), boundary_z + self._probe_boundary_headroom_m())
        self._probe_boundary_corrections += 1
        self._probe_scan_target_reached = False
        self._probe_scan_pose = None
        self._probe_scan_signal = 0.0
        self._probe_strength_hold = None
        self._target_strength_check = None
        self._probe_level_frames = []
        if not self._probe_onset_confirmed:
            self._probe_hit_count = 0
            self._probe_candidate_pose = None
        detail = (
            f"Taxel {self._current_taxel}: {undershoot * 1000.:.4f} mm boundary "
            "undershoot; correcting vertically upwards, then collecting fresh "
            "signal batches before any XY scan (1/1 correction)"
        )
        host._append_sidebar_control_message(detail)
        self._command_move(host, lift, "probe_boundary_corrected", detail,
                           quaternion=pose[1], vertical_only=True)
        return True

    def _tick_probe_line_step(self, host, api, pose, now, sequence, step_start_z) -> None:
        """One depth step as a TM line move, clamped to the same descent limits."""
        z_value = float(pose[0][2])
        if self._probe_line_target_z is None:
            lowest = float(self._probe_start_pose[2]) - float(self._probe_max_descent_used_m)
            boundary = self._working_probe_boundary
            if boundary is not None:
                lowest = max(lowest, float(boundary["z_m"]))
            if self._probe_onset_confirmed and self._probe_onset_pose is not None:
                lowest = max(lowest, float(self._probe_onset_pose[2]) - float(self.centre_scan_max_extra_descent_m))
            if boundary is not None and not self.fixed_scan_active:
                # Leave arrival headroom on the safe side of every active
                # working-depth limit, including the final short depth step.
                lowest = min(float(self._probe_start_pose[2]),
                             lowest + self._probe_boundary_headroom_m())
            target_z = max(lowest, float(step_start_z) - float(self.probe_step_m))
            at_limit = target_z <= lowest + 1e-9
            if z_value <= lowest + float(self.position_tolerance_m) or float(step_start_z) - target_z <= 1e-7:
                self._settle_probe_level(now, pose, sequence, at_limit=True)
                return
            position = [float(pose[0][0]), float(pose[0][1]), target_z]
            CalibrationTask._stop_robot(api)
            try:
                sent = bool(api.send_positions_tool_position(
                    position, list(pose[1]),
                    velocity=max(0.001, abs(float(self._probe_speed_used_m_s))), acc_time=0.2,
                    blend_percentage=0, fine_goal=True, motion_type=self.LINE_MOTION,
                ))
            except Exception:
                sent = False
            if not sent:
                self._request_stop(host, "command_failed", "Mapping stopped: robot rejected a depth line-move step.")
                return
            self._probe_line_target_z = target_z
            self._probe_line_at_limit = bool(at_limit)
            self._probe_line_sent_at = float(now)
            self._probe_line_last_z = None
            return
        previous = self._probe_line_last_z
        self._probe_line_last_z = z_value
        still = previous is not None and abs(z_value - previous) <= float(self.stationary_tolerance_m)
        tolerance = min(float(self.position_tolerance_m), 0.5 * float(self.probe_step_m))
        if abs(z_value - float(self._probe_line_target_z)) <= tolerance and still:
            self._settle_probe_level(now, pose, sequence, at_limit=self._probe_line_at_limit)
            return
        if now - float(self._probe_line_sent_at) > float(self.move_timeout_sec):
            self._request_stop(host, "move_timeout", "Mapping stopped: a depth line-move step did not finish.")

    def _settle_probe_level(self, now, pose, sequence, *, at_limit) -> None:
        self._probe_line_target_z = None
        self._probe_stage = "settling"
        self._probe_settle_started_at = now
        self._probe_level_pose = np.asarray(pose[0], dtype=float)
        self._probe_level_at_max_depth = bool(at_limit)
        self._probe_last_sequence = int(sequence)

    def _handle_probe_onset(self, host) -> None:
        onset = np.asarray(self._probe_onset_pose, dtype=float)
        if self._probe_kind == "reference":
            anchor_z = float(self.anchor_pose["tool_position_m"][2])
            self._reference_onset_z = float(onset[2])
            self._probe_start_z = min(
                anchor_z, float(onset[2]) + float(self.probe_start_margin_m)
            )
            host._append_sidebar_control_message(
                f"Sensor height learned at Taxel {self.anchor_taxel}: proximity starts "
                f"{(anchor_z - onset[2]) * 1000.0:.2f} mm below the anchor. "
                "Later probes start "
                f"{(self._probe_start_z - onset[2]) * 1000.0:.2f} mm above it."
            )
            self._start_target(host, self.anchor_taxel, onset[:2], "anchor")
            return
        if not self._probe_scan_target_reached:
            self._handle_probe_failure(host, "the requested stronger non-contact signal has not been verified")
            return
        scan_pose = np.asarray(
            self._probe_scan_pose
            if self._probe_scan_pose is not None
            else onset,
            dtype=float,
        )
        boundary = self._working_probe_boundary
        current = self._read_tool_pose(self._robot_api(host))
        if current is None:
            self._request_stop(host, "robot_feedback_lost", "Mapping stopped: TCP feedback was lost before XY scanning.")
            return
        if self._correct_probe_boundary_undershoot(host, current, sample_pose=scan_pose):
            return
        if boundary is not None and float(scan_pose[2]) < boundary["z_m"]-1e-6:
            self._handle_probe_failure(host, "the stronger-signal sample exceeded the onset-independent known-height boundary; refusing XY scanning")
            return
        extra_descent_mm = max(0.0, float(onset[2] - scan_pose[2])) * 1000.0
        if extra_descent_mm > self.centre_scan_max_extra_descent_m*1000.+1e-6:
            self._handle_probe_failure(host, "the sampled strong-signal plane exceeded the allowed extra descent; refusing XY scanning")
            return
        host._append_sidebar_control_message(
            f"Taxel {self._current_taxel}: proximity onset remembered at "
            f"Z={onset[2]:.6f} m. Centre scan uses "
            f"{self._probe_scan_signal:.1f} sensor units at "
            f"{extra_descent_mm:.2f} mm deeper; the "
            f"{self.centre_scan_signal_multiplier:.2f}× target "
            f"({self._probe_scan_target_signal:.1f} sensor units) was reached."
        )
        if self._current_context == "anchor":
            self._start_coarse_scan(host, self._far_baseline, scan_pose)
            return
        self._start_local_scan(host, scan_pose)

    def _local_axes_for(self, index: int) -> Optional[list[dict]]:
        row_vector = (
            self._row_vector_xy
            if self._row_vector_xy is not None
            else self._initial_row_vector_xy
        )
        column_vector = (
            self._column_vector_xy
            if self._column_vector_xy is not None
            else self._initial_column_vector_xy
        )
        if row_vector is None or column_vector is None:
            return None
        row, column = self._taxel_coords(index)
        axes = []
        for name, vector, minus, plus in (
            ("row", row_vector, (row - 1, column), (row + 1, column)),
            ("column", column_vector, (row, column - 1), (row, column + 1)),
        ):
            vector = np.asarray(vector, dtype=float)[:2]
            pitch = float(np.linalg.norm(vector))
            if pitch <= 0.0:
                return None
            axis = {
                "name": name,
                "unit": vector / pitch,
                "pitch": pitch,
                "minus": self._index_if_valid(*minus),
                "plus": self._index_if_valid(*plus),
            }
            mode, window_centre = self.axis_balance_mode(axis)
            # An edge window sits between the taxel and its neighbour, so it
            # must stay narrower than half a pitch on each side.
            fraction = (
                float(self.local_radius_pitch_fraction)
                if window_centre == 0.0
                else 0.4
            )
            radius = min(float(self.local_radius_m), fraction * pitch)
            axis["radius"] = max(radius, 2.0 * float(self.local_step_m))
            axis["mode"] = mode
            axis["window_centre"] = window_centre
            axes.append(axis)
        return axes

    def _start_local_scan(self, host, centre) -> None:
        index = int(self._current_taxel)
        axes = self._local_axes_for(index)
        if axes is None:
            self._handle_probe_failure(
                host, "row and column directions are not known yet"
            )
            return
        self._local_center = np.asarray(centre, dtype=float)
        self._local_axes = axes
        self._centre_repeat = None
        self._target_centre_check = None
        lines = []
        for axis in axes:
            line = []
            for offset in self.axis_offsets(axis["radius"], self.local_step_m):
                offset += float(axis["window_centre"])
                point = np.array(self._local_center, copy=True)
                point[:2] += offset * axis["unit"]
                line.append((point, {"axis": axis["name"], "nominal_offset_m": offset}))
            lines.append(line)
        pose = self._read_tool_pose(self._robot_api(host))
        route = self.shortest_line_scan_route(
            lines, self._local_center if pose is None else pose[0]
        )
        points, meta = zip(*route)
        # Line scans use the far-field baseline. The start-height baseline
        # already holds proximity signal whose left/right imbalance would
        # shift the neighbour balance point.
        self._start_scan(
            host,
            kind="local",
            points=points,
            baseline=self._far_baseline,
            target=index,
            meta=meta,
        )

    @staticmethod
    def shortest_line_scan_route(lines, start):
        """Visit both full measurement lines using the shortest endpoint route.

        Only traversal order changes: positions, offsets, axis labels and
        stationary measurement density are identical.
        """
        start = np.asarray(start, dtype=float)
        candidates = []
        for first, second in (lines, lines[::-1]):
            for first_line in (first, first[::-1]):
                for second_line in (second, second[::-1]):
                    route = list(first_line) + list(second_line)
                    positions = np.asarray([start] + [item[0] for item in route])
                    length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
                    candidates.append((length, route))
        return min(candidates, key=lambda item: item[0])[1]

    def _start_scan(self, host, *, kind, points, baseline, target, meta=None, preserve_order=False) -> None:
        self._scan_kind = str(kind)
        self._scan_points = [np.asarray(point, dtype=float) for point in points]
        self._scan_point_meta = list(meta or [])
        if self.reverse_scan_order and not preserve_order:
            self._scan_points.reverse()
            self._scan_point_meta.reverse()
        self._scan_results = []
        self._scan_index = 0
        self._scan_baseline = np.asarray(baseline, dtype=float)
        self._scan_target = None if target is None else int(target)
        self._move_to_current_scan_point(host)

    def _move_to_current_scan_point(self, host) -> None:
        if self._scan_index >= len(self._scan_points):
            self._finish_scan(host)
            return
        target_text = (
            ("fast neighbour discovery" if self._coarse_sparse_pass else "dense neighbour discovery")
            if self._scan_kind == "coarse"
            else f"Taxel {self._scan_target} " + ("repeat refinement" if self._centre_repeat else "refinement")
        )
        self._command_move(
            host,
            self._scan_points[self._scan_index],
            "scan_point_arrived",
            f"{target_text}: point {self._scan_index + 1}/{len(self._scan_points)}",
        )

    def _record_scan_sample(self, host, frame) -> None:
        pose = self._read_tool_pose(self._robot_api(host))
        if pose is None:
            self._request_stop(host, "robot_feedback_lost", "Mapping stopped: TCP feedback was lost during the XY scan.")
            return
        result = {"position": np.asarray(pose[0], dtype=float), "frame": np.asarray(frame, dtype=float)}
        if self._fixed_scan_contact(
            host, result["frame"], self._scan_baseline, f"during the {self._scan_kind} XY scan"
        ):
            return
        meta = (
            self._scan_point_meta[self._scan_index]
            if self._scan_index < len(self._scan_point_meta)
            else {}
        )
        result.update(meta)
        self._scan_results.append(result)
        target_index = self._scan_target
        if target_index is not None:
            target, second, dominance, dominant, _delta = self._signal_metrics(
                result["frame"], self._scan_baseline, target_index
            )
        else:
            delta = np.abs(result["frame"] - self._scan_baseline)
            dominant = int(np.argmax(delta))
            target = float(delta[dominant])
            second = float(np.partition(delta, -2)[-2]) if delta.size > 1 else 0.0
            dominance = target / max(second, 1e-9)
        attempt_taxel = (
            target_index if target_index is not None else self._current_taxel
        )
        nominal_offset = meta.get("nominal_offset_m")
        self._sample_log.append(
            {
                "scan": self._scan_kind,
                "scan_pass": (
                    ("sparse" if self._coarse_sparse_pass else "dense")
                    if self._scan_kind == "coarse" else
                    ("local_repeat" if self._centre_repeat is not None else "local")
                ),
                "sample_frames": int(self._collection_count),
                "taxel_index": target_index,
                "attempt_number": int(
                    self._attempt_counts.get(int(attempt_taxel), 0)
                )
                if attempt_taxel is not None
                else 0,
                "axis": meta.get("axis", ""),
                "nominal_offset_mm": None
                if nominal_offset is None
                else float(nominal_offset) * 1000.0,
                "point_number": self._scan_index + 1,
                "x_m": float(pose[0][0]),
                "y_m": float(pose[0][1]),
                "z_m": float(pose[0][2]),
                "target_signal_counts": float(target),
                "competing_signal_counts": float(second),
                "dominance_ratio": float(dominance),
                "dominant_taxel": int(dominant),
            }
        )
        self._scan_index += 1
        self._move_to_current_scan_point(host)

    def _finish_scan(self, host) -> None:
        if self._scan_kind == "local":
            self._finish_local_scan(host)
        elif self._scan_kind == "coarse":
            self._finish_coarse_scan(host)

    def _finish_local_scan(self, host) -> None:
        index = int(self._scan_target)
        estimates = {}
        for axis in self._local_axes:
            samples = [
                sample
                for sample in self._scan_results
                if sample.get("axis") == axis["name"]
            ]
            estimate = self.estimate_axis_centre(
                samples, self._scan_baseline, self._local_center, axis, index
            )
            if estimate is None:
                self._handle_centre_scan_failure(
                    host, f"the {axis['name']} scan gave no usable centre"
                )
                return
            estimates[axis["name"]] = estimate
        outside = []
        offsets = {}
        for axis in self._local_axes:
            offset = float(estimates[axis["name"]]["offset_m"])
            if abs(float(estimates[axis["name"]]["bracket_m"])) > float(axis["radius"]) + 1e-9:
                outside.append(axis["name"])
            offsets[axis["name"]] = float(
                np.clip(offset, -2.0 * axis["radius"], 2.0 * axis["radius"])
            )
        centre = np.array(self._local_center, copy=True)
        centre[:2] += self._shift_from_axis_offsets(offsets)
        if outside:
            if self._centre_repeat is None and self._local_recenters < int(self.max_local_recenters):
                self._local_recenters += 1
                host._append_sidebar_control_message(
                    f"Taxel {index}: the centre lies outside the local scan "
                    f"along the {' and '.join(outside)} direction; "
                    "re-centring the scan once."
                )
                self._start_local_scan(host, centre)
                return
            self._handle_centre_scan_failure(
                host, "the centre was not bracketed by the local scan"
            )
            return
        if self._centre_repeat is not None:
            first = self._centre_repeat
            separation = float(np.linalg.norm(centre[:2] - np.asarray(first["centre_m"])[:2]))
            tolerance = max(float(self.local_step_m), 2.0 * float(self.position_tolerance_m))
            check = {
                "taxel_index": index, "attempt": int(self._attempt_counts.get(index, 1)),
                "reason": first["reason"], "first_centre_m": first["centre_m"],
                "repeat_centre_m": centre.tolist(), "separation_mm": separation * 1000.,
                "tolerance_mm": tolerance * 1000., "repeatable": bool(separation <= tolerance),
                "final_verified": False,
                "first_axes": first["axes"], "repeat_axes": copy.deepcopy(estimates),
                "interpretation": "Same-height repeatability, not absolute geometric accuracy.",
            }
            self._centre_checks.append(check)
            self._target_centre_check = check
            if not check["repeatable"]:
                self._handle_centre_scan_failure(
                    host, f"centre not repeatable: opposite-order scans disagree by "
                    f"{separation*1000.:.2f} mm (limit {tolerance*1000.:.2f} mm)"
                )
                return
        self._verify_estimate = {"centre": centre, "axes": estimates}
        self._scan_kind = ""
        self._command_move(
            host,
            centre,
            "verify_arrived",
            f"Taxel {index}: moving to the refined centre to verify it",
        )

    def _shift_from_axis_offsets(self, offsets: dict) -> np.ndarray:
        """XY shift whose projections onto the scan axes equal ``offsets``.

        Each line scan measures the centre's projection onto its own axis.
        Row and column axes are rarely exactly perpendicular, so summing
        offset * unit would leak one axis into the other.
        """
        units = np.asarray(
            [np.asarray(axis["unit"], dtype=float)[:2] for axis in self._local_axes]
        )
        values = np.asarray(
            [offsets[axis["name"]] for axis in self._local_axes], dtype=float
        )
        try:
            return np.linalg.solve(units, values)
        except np.linalg.LinAlgError:
            return units.T @ values

    def _finish_verify(self, host, frame) -> None:
        index = int(self._current_taxel)
        pose = self._read_tool_pose(self._robot_api(host))
        if pose is None:
            self._request_stop(
                host,
                "robot_feedback_lost",
                "Mapping stopped: TCP feedback was lost during verification.",
            )
            return
        if self._fixed_scan_contact(
            host, frame, self._far_baseline, "at the refined centre"
        ):
            return
        target, second, dominance, dominant, _delta = self._signal_metrics(
            frame, self._far_baseline, index
        )
        signal_threshold = self._signal_threshold_for_taxel(index)
        if target < 0.5 * signal_threshold:
            self._handle_centre_scan_failure(
                host,
                f"the signal at the refined centre was too weak "
                f"({target:.1f} sensor units)",
            )
            return
        if dominance < float(self.final_min_dominance):
            self._handle_centre_scan_failure(
                host,
                f"Taxel {dominant} was stronger than the target at the refined "
                f"centre ({second:.1f} vs {target:.1f} sensor units)",
            )
            return
        if self._centre_repeat is None:
            axes = (self._verify_estimate or {}).get("axes", {})
            reasons = []
            if any(axis.get("method") == "peak_centroid" for axis in axes.values()):
                reasons.append("broad/flat peak")
            if dominance < self.CENTRE_REPEAT_MIN_DOMINANCE:
                reasons.append("small lead over neighbouring taxels")
            if target < signal_threshold:
                reasons.append("weak final signal")
            if self._target_strength_check and self._target_strength_check.get("accepted"):
                reasons.append("borderline probe strength")
            if reasons:
                if not self._scan_points or not axes:
                    self._request_stop(host, "internal_error",
                                       "Mapping stopped: centre repeat scan geometry is missing.")
                    return
                self._centre_repeat = {
                    "centre_m": np.asarray(self._verify_estimate["centre"]).tolist(),
                    "axes": copy.deepcopy(axes), "reason": "; ".join(reasons),
                }
                host._append_sidebar_control_message(
                    f"Taxel {index}: {self._centre_repeat['reason']}; repeating the SAME "
                    "XY lines in reverse order at the same height before saving the centre."
                )
                # These points already include the operator's scan direction.
                # Reverse exactly once; do not expand/recentre the footprint.
                self._start_scan(host, kind="local", points=list(reversed(self._scan_points)),
                                 baseline=self._far_baseline, target=index,
                                 meta=list(reversed(self._scan_point_meta)), preserve_order=True)
                return
        if self._target_centre_check is not None:
            self._target_centre_check["final_verified"] = True
            host._append_sidebar_control_message(
                f"Taxel {index}: centre repeat check passed, "
                f"{self._target_centre_check['separation_mm']:.2f} mm separation "
                f"(limit {self._target_centre_check['tolerance_mm']:.2f} mm); final signal verified."
            )
        if index == int(self.anchor_taxel) and float(dominance) < self.MIN_ANCHOR_DOMINANCE:
            self._stop_for_weak_anchor_plane(host, pose[0], target, dominance)
            return
        self._accept_position(host, pose[0], target, second, dominance)

    def _stop_for_weak_anchor_plane(self, host, position, target, dominance) -> None:
        """Retrying at the same plane cannot help; the operator must lower it."""
        plane = (
            f"the fixed scan height (Z {self.fixed_scan_z_m * 1000.0:.1f} mm)"
            if self.fixed_scan_active else "the automatic proximity-threshold scan plane"
        )
        self._anchor_stop = (
            "scan_plane_too_high",
            f"Mapping stopped: at {plane}, TCP Z {float(position[2]) * 1000.0:.1f} mm, Taxel "
            f"{self.anchor_taxel} reads {float(target):.1f} sensor units, only {float(dominance):.2f}x "
            f"its strongest neighbour (need {self.MIN_ANCHOR_DOMINANCE:.1f}x). The scan plane is too high "
            "to centre taxels reliably, so the anchor was not saved. Jog the probe down over the anchor "
            "until it clearly stands out from its neighbours (completed maps read about 100-200 units), "
            "use Remember Scan Height there, then start a new map.",
        )
        anchor = self.anchor_pose["tool_position_m"]
        self._route_to(
            host, anchor[:2], anchor[2], "anchor_stop_returned",
            f"scan plane too high; returning to the safe Taxel {self.anchor_taxel} anchor",
        )

    def _accept_position(self, host, position, target, second, dominance) -> None:
        index = int(self._current_taxel)
        row, column = self._taxel_coords(index)
        position = np.asarray(position, dtype=float)
        stability = self._baseline_stats_by_taxel.get(index, {})
        detection = self._probe_detection_by_taxel.get(index, {})
        axes = (self._verify_estimate or {}).get("axes", {})
        onset_z = (
            None
            if self._probe_onset_pose is None
            else float(self._probe_onset_pose[2])
        )
        scan_z = (
            None
            if self._probe_scan_pose is None
            else float(self._probe_scan_pose[2])
        )

        def axis_mm(name, key):
            value = axes.get(name, {}).get(key)
            return None if value is None else float(value) * 1000.0

        self._positions[index] = {
            "taxel_index": index,
            "row": row,
            "column": column,
            "status": "measured",
            "attempt_count": int(self._attempt_counts.get(index, 1)),
            "tool_position_m": position.tolist(),
            "safe_approach_position_m": [
                float(position[0]),
                float(position[1]),
                float(self.anchor_pose["tool_position_m"][2]),
            ],
            "probe_start_z_m": self._probe_start_z,
            "proximity_onset_z_m": onset_z,
            "centre_scan_z_m": scan_z,
            "centre_scan_extra_descent_mm": (
                None
                if onset_z is None or scan_z is None
                else max(0.0, onset_z - scan_z) * 1000.0
            ),
            "centre_scan_signal_units": float(self._probe_scan_signal),
            "centre_scan_target_units": float(
                self._probe_scan_target_signal
            ),
            "centre_scan_target_reached": bool(
                self._probe_scan_target_reached
            ),
            "centre_scan_policy": self.CENTRE_SCAN_POLICY,
            "quality_policy": self.QUALITY_POLICY,
            "stationary_strength_check": copy.deepcopy(self._target_strength_check),
            "centre_repeat_check": copy.deepcopy(self._target_centre_check),
            "centre_repeat_separation_mm": (self._target_centre_check or {}).get("separation_mm"),
            "centre_repeat_status": (
                "repeatable" if self._target_centre_check is not None else "not_triggered"
            ),
            "centre_scan_signal_multiplier": float(self.centre_scan_signal_multiplier),
            "centre_scan_max_extra_descent_mm": float(self.centre_scan_max_extra_descent_m)*1000.,
            "noncontact_force_stop_g": self.NONCONTACT_FORCE_STOP_G,
            "tool_quaternion_wxyz": list(self.anchor_pose["tool_quaternion_wxyz"]),
            "signal_counts": float(target),
            "competing_signal_counts": float(second),
            "dominance_ratio": float(dominance),
            "confidence": float(
                min(1.0, target / self._signal_threshold_for_taxel(index))
                * min(1.0, dominance)
            ),
            "confidence_interpretation": "Heuristic signal/dominance score, not a probability or position uncertainty.",
            "position_interpretation": "Verified response centre; absolute geometric accuracy requires independent validation.",
            "proximity_detection_mode": detection.get(
                "mode", self.proximity_detection_mode
            ),
            "proximity_threshold_counts": detection.get(
                "effective_threshold_counts",
                self._signal_threshold_for_taxel(index),
            ),
            "proximity_baseline_min_counts": detection.get(
                "baseline_min_counts"
            ),
            "proximity_baseline_max_counts": detection.get(
                "baseline_max_counts"
            ),
            "proximity_baseline_range_counts": detection.get(
                "baseline_range_counts"
            ),
            "proximity_margin_counts": detection.get("margin_counts"),
            "proximity_lower_trigger_counts": detection.get(
                "lower_trigger_counts"
            ),
            "proximity_upper_trigger_counts": detection.get(
                "upper_trigger_counts"
            ),
            "proximity_rolling_window_frames": detection.get(
                "rolling_window_frames"
            ),
            "proximity_rolling_robust_sigma_counts": detection.get(
                "rolling_robust_sigma_counts"
            ),
            "proximity_rolling_low_percentile_counts": detection.get(
                "rolling_low_percentile_counts"
            ),
            "method": "cross_scan_verified",
            "position_prediction": copy.deepcopy(self._current_prediction),
            "row_axis_method": axes.get("row", {}).get("method"),
            "column_axis_method": axes.get("column", {}).get("method"),
            "row_offset_mm": axis_mm("row", "offset_m"),
            "column_offset_mm": axis_mm("column", "offset_m"),
            "row_balance_offset_mm": axis_mm("row", "balance_offset_m"),
            "row_peak_offset_mm": axis_mm("row", "peak_offset_m"),
            "column_balance_offset_mm": axis_mm("column", "balance_offset_m"),
            "column_peak_offset_mm": axis_mm("column", "peak_offset_m"),
            "local_recenters": int(self._local_recenters),
            "baseline_target_range_counts": stability.get("target_range_counts"),
            "baseline_grid_p90_range_counts": stability.get("grid_p90_range_counts"),
            "baseline_grid_max_range_counts": stability.get("grid_max_range_counts"),
        }
        self._failures.pop(index, None)
        self._verify_estimate = None
        self._publish_live_map(host)
        self._progress(host, f"Taxel {index} located and verified")
        if self._current_context == "resume_anchor":
            previous = self._resume_anchor_previous or {}
            previous_position = np.asarray(
                previous.get("tool_position_m", []), dtype=float
            )
            new_position = np.asarray(
                self._positions[index]["tool_position_m"], dtype=float
            )
            xy_error = (
                float(np.linalg.norm(new_position[:2] - previous_position[:2]))
                if previous_position.shape == (3,)
                else float("inf")
            )
            z_error = (
                abs(float(new_position[2] - previous_position[2]))
                if previous_position.shape == (3,)
                else float("inf")
            )
            if (
                xy_error > float(self.resume_xy_tolerance_m)
                or z_error > float(self.resume_z_tolerance_m)
            ):
                measured = self._positions[index]
                self._positions[index] = previous
                heights = ""
                if previous_position.shape == (3,):
                    heights = (
                        f"Measured at Z {new_position[2] * 1000.0:.1f} mm with "
                        f"{float(measured.get('signal_counts') or 0.0):.1f} sensor units; "
                        f"saved at Z {previous_position[2] * 1000.0:.1f} mm with "
                        f"{float(previous.get('signal_counts') or 0.0):.1f}. "
                    )
                self._resume_alignment_error = (
                    f"Resume verification failed: Taxel {self.anchor_taxel} moved by "
                    f"{xy_error * 1000.0:.2f} mm in XY and "
                    f"{z_error * 1000.0:.2f} mm in Z relative to the saved map "
                    f"(limits {float(self.resume_xy_tolerance_m) * 1000.0:.1f} mm XY, "
                    f"{float(self.resume_z_tolerance_m) * 1000.0:.1f} mm Z). "
                    + heights
                    + "The sensor, robot base, or tool may have moved."
                )
                anchor = self.anchor_pose["tool_position_m"]
                self._route_to(
                    host,
                    anchor[:2],
                    anchor[2],
                    "resume_alignment_returned",
                    f"resume check failed; returning to the safe Taxel {self.anchor_taxel} anchor",
                )
                return
        self._begin_clearance_cycle(host, self._current_context)

    def _finish_coarse_scan(self, host) -> None:
        guesses = {}
        missing = []
        for index in (self.anchor_taxel,) + self.anchor_neighbour_indices():
            refined = self.refine_xy_from_samples(
                self._scan_results, self._scan_baseline, index
            )
            if (
                refined is None
                or refined["signal_counts"]
                < self._signal_threshold_for_taxel(index) * 0.25
            ):
                missing.append(index)
                continue
            guesses[index] = np.asarray(refined["position"], dtype=float)
        if missing:
            if self._retry_dense_coarse_scan(host, "some reference taxels were too weak"):
                return
            self._handle_probe_failure(
                host,
                "the coarse scan could not find Taxel "
                + ", ".join(str(index) for index in missing),
            )
            return
        neighbours = self.anchor_neighbour_indices()
        vectors = None
        if len(neighbours) >= 4 and all(index in guesses for index in neighbours):
            vectors = self.lattice_vectors_from_points(
                guesses,
                (neighbours[0], neighbours[1]),
                (neighbours[2], neighbours[3]),
            )
        if vectors is None:
            if self._retry_dense_coarse_scan(host, "the row/column directions were unclear"):
                return
            self._handle_probe_failure(
                host, "the coarse scan did not reveal a 2D taxel lattice"
            )
            return
        if self._coarse_sparse_pass:
            # A strong response clipped by the search boundary is not a
            # measured peak. Do not trust a sparse-grid centroid in that case.
            centre_xy = self._coarse_scan_centre[:2]
            for index in (self.anchor_taxel,) + neighbours:
                best = max(
                    self._scan_results,
                    key=lambda sample: self._signal_metrics(
                        sample["frame"], self._scan_baseline, index
                    )[0],
                )
                offset = np.abs(np.asarray(best["position"])[:2] - centre_xy)
                if np.any(offset >= float(self.coarse_radius_m) - 1e-6):
                    self._retry_dense_coarse_scan(host, f"Taxel {index}'s peak reached the search boundary")
                    return
        self._initial_row_vector_xy, self._initial_column_vector_xy, _score = vectors
        self._neighbor_guesses = {
            index: guesses[index][:2].tolist()
            for index in neighbours
        }
        host._append_sidebar_control_message(
            "Coarse scan: row pitch "
            f"{np.linalg.norm(self._initial_row_vector_xy) * 1000.0:.2f} mm, "
            "column pitch "
            f"{np.linalg.norm(self._initial_column_vector_xy) * 1000.0:.2f} mm."
        )
        centre = np.array(
            self._probe_scan_pose
            if self._probe_scan_pose is not None
            else self._probe_onset_pose,
            dtype=float,
        )
        centre[:2] = guesses[self.anchor_taxel][:2]
        self._start_local_scan(host, centre)

    def _retry_dense_coarse_scan(self, host, reason: str) -> bool:
        if not self._coarse_sparse_pass or self._coarse_scan_centre is None:
            return False
        host._append_sidebar_control_message(
            f"Fast neighbour search needs a denser check ({reason}); "
            "scanning the original grid with at least 20 readings per point."
        )
        self._local_recenters = 0
        self._verify_estimate = None
        self._start_coarse_scan(
            host, self._far_baseline, self._coarse_scan_centre, dense=True
        )
        return True

    def _handle_centre_scan_failure(self, host, reason: str) -> None:
        # A sparse initial estimate must not force a full retract/re-probe
        # cycle when the original denser discovery grid can resolve it.
        if (
            self._current_context == "anchor"
            and self.anchor_taxel not in self._positions
            and self._retry_dense_coarse_scan(host, reason)
        ):
            return
        self._handle_probe_failure(host, reason, quality_failure=True)

    def _route_to(self, host, xy, z, action: str, detail: str) -> bool:
        """Travel at the safe height, then move vertically to (xy, z).

        Lateral travel never happens below the remembered anchor height,
        except for the short scan moves made at the proximity height.
        """
        pose = self._read_tool_pose(self._robot_api(host))
        if pose is None:
            self._request_stop(
                host,
                "robot_feedback_lost",
                "Mapping stopped: TCP feedback was unavailable before a move.",
            )
            return False
        current = np.asarray(pose[0], dtype=float)
        travel_z = max(float(current[2]), float(self.anchor_pose["tool_position_m"][2]))
        target = np.array([float(xy[0]), float(xy[1]), float(z)])
        waypoints = []
        xy_distance = float(np.linalg.norm(current[:2] - target[:2]))
        # Any XY change from below the travel height starts straight up,
        # even a sub-tolerance one: diagonal moves near the skin are refused.
        if xy_distance > 1e-6 and current[2] < travel_z - float(self.position_tolerance_m):
            waypoints.append(np.array([current[0], current[1], travel_z]))
        if xy_distance > float(self.route_xy_tolerance_m):
            waypoints.append(np.array([target[0], target[1], travel_z]))
        waypoints.append(target)
        route = []
        for point in waypoints:
            if route and float(np.linalg.norm(point - route[-1])) <= 1e-9:
                continue
            route.append(point)
        self._route_queue = [
            (point, "route_waypoint", detail) for point in route[:-1]
        ] + [(route[-1], str(action), detail)]
        return self._advance_route(host)

    def _advance_route(self, host) -> bool:
        if not self._route_queue:
            self._request_stop(
                host, "internal_error", "Mapping stopped: the robot route was lost."
            )
            return False
        position, action, detail = self._route_queue.pop(0)
        return self._command_move(host, position, action, detail)

    def _schedule_failure_recovery(
        self, host, *, kind: str, reason: str, quality_failure=False
    ) -> None:
        """Return to the clearance anchor, recalibrate, then retry or stop."""
        if getattr(self, "_scheduling_recovery", False):
            # The return route itself was refused; retrying would recurse.
            self._request_stop(
                host,
                "recovery_route_refused",
                "Mapping stopped: the return to the clearance anchor was refused "
                f"({reason}). The robot was left where it is; move it up manually.",
            )
            return
        self._scheduling_recovery = True
        try:
            self._schedule_failure_recovery_route(host, kind=kind, reason=reason, quality_failure=quality_failure)
        finally:
            self._scheduling_recovery = False

    def _schedule_failure_recovery_route(
        self, host, *, kind: str, reason: str, quality_failure=False
    ) -> None:
        kind = "reference" if str(kind) == "reference" else "target"
        if kind == "reference":
            index = int(self.anchor_taxel)
            attempt = int(self._reference_attempt_count)
            label = f"The first Taxel {self.anchor_taxel} probe"
        else:
            index = int(self._current_taxel)
            attempt = int(self._attempt_counts.get(index, 0))
            label = f"Taxel {index}"
        will_retry = attempt < int(self.max_attempts_per_taxel)
        self._failures[index] = str(reason)
        self._failed_attempt_total += 1
        self._pending_recovery = {
            "kind": kind,
            "taxel_index": index,
            "xy": None
            if self._current_target_xy is None
            else np.asarray(self._current_target_xy, dtype=float).tolist(),
            "context": str(self._current_context),
            "reason": str(reason),
            "quality_failure": bool(quality_failure),
            "retry": bool(will_retry),
            "next_attempt": attempt + 1 if will_retry else attempt,
        }
        if will_retry:
            message = (
                f"{label} attempt {attempt}/{self.max_attempts_per_taxel} "
                f"failed: {reason}. Retracting and returning to Taxel {self.anchor_taxel} "
                "before one retry."
            )
        else:
            message = (
                f"{label} failed after {attempt} attempts: {reason}. "
                f"Retracting and returning to Taxel {self.anchor_taxel} for clearance "
                "checks before deciding whether other supported taxels can continue."
            )
        self._scan_kind = ""
        host._append_sidebar_control_message(message)
        self._progress(host, message)
        anchor = self.anchor_pose["tool_position_m"]
        self._route_to(
            host,
            anchor[:2],
            anchor[2],
            "recovery_anchor_arrived",
            f"returning to the remembered safe pose above Taxel {self.anchor_taxel}",
        )

    def _handle_probe_failure(self, host, reason: str, *, quality_failure=False) -> None:
        kind = "reference" if self._current_context == "reference" else "target"
        self._schedule_failure_recovery(host, kind=kind, reason=str(reason), quality_failure=quality_failure)

    def _dispatch_move_action(self, host, action: str) -> None:
        if action == "route_waypoint":
            self._advance_route(host)
        elif action == "probe_boundary_corrected":
            pose = self._read_tool_pose(self._robot_api(host))
            boundary = self._working_probe_boundary
            if pose is None or boundary is None or float(pose[0][2]) < float(boundary["z_m"]):
                self._request_stop(host, "probe_boundary_correction_failed",
                                   "Mapping stopped: vertical boundary correction was not confirmed; no XY scan.")
                return
            self._phase = "probing"
            # Do not chase the same boundary again if the raised signal is weak.
            # The normal stationary-sampling path must re-qualify it here.
            self._settle_probe_level(time.monotonic(), pose,
                                     self._probe_last_sequence or 0, at_limit=True)
        elif action == "fixed_scan_plane_arrived":
            pose = getattr(self, "_fixed_scan_plane_pose", None)
            self._fixed_scan_plane_pose = None
            if pose is None:
                self._request_stop(
                    host, "internal_error", "Mapping stopped: the fixed scan plane was lost."
                )
                return
            self._start_fixed_plane_scan(host, pose)
        elif action == "anchor_safe_arrived":
            self._current_taxel = self.anchor_taxel
            self._current_context = "reference"
            self._current_target_xy = np.asarray(
                self.anchor_pose["tool_position_m"], dtype=float
            )[:2]
            self._reference_attempt_count += 1
            self._begin_sensor_calibration(
                host,
                next_action="reference_baseline",
                context="fresh_start",
            )
        elif action == "resume_anchor_arrived":
            self._current_taxel = self.anchor_taxel
            self._current_context = "resume_anchor"
            self._current_target_xy = np.asarray(
                self._positions[self.anchor_taxel]["tool_position_m"],
                dtype=float,
            )[:2]
            self._begin_sensor_calibration(
                host,
                next_action="resume_anchor_stability",
                context="resume",
            )
        elif action == "target_over_arrived":
            self._begin_collection(
                host,
                "far_baseline",
                self.baseline_frame_count,
                f"Taxel {self._current_taxel}: recording the far-field "
                "baseline at the safe height",
            )
        elif action == "target_start_arrived":
            attempt = int(
                self._attempt_counts.get(int(self._current_taxel), 1)
            )
            self._begin_collection(
                host,
                "target_baseline",
                self.baseline_frame_count,
                f"Taxel {self._current_taxel} attempt "
                f"{attempt}/{self.max_attempts_per_taxel}: checking "
                "sensor stability at the probe start height",
            )
        elif action == "scan_point_arrived":
            count = int(self.scan_frame_count)
            if self._scan_kind == "local" and self._centre_repeat is not None:
                count = max(count, int(self.verify_frame_count), 20)
            if self._scan_kind == "coarse" and not self._coarse_sparse_pass:
                count = max(20, count)
            label = (
                ("Fast" if self._coarse_sparse_pass else "Dense") + " neighbour search"
                if self._scan_kind == "coarse"
                else f"Taxel {self._scan_target} " + ("repeat refinement" if self._centre_repeat else "refinement")
            )
            self._begin_collection(
                host, "scan_sample", count,
                f"{label}: point {self._scan_index + 1}/{len(self._scan_points)}, "
                f"collecting {count} stationary readings",
            )
        elif action == "verify_arrived":
            self._begin_collection(
                host,
                "verify_sample",
                self.verify_frame_count,
                f"Taxel {self._current_taxel}: checking that the refined "
                "centre belongs to the intended taxel",
            )
        elif action == "clearance_anchor_arrived":
            self._begin_collection(
                host, "clearance_precheck", self.baseline_frame_count,
                "At the clearance anchor: checking signal recovery BEFORE Update Calibration",
            )
        elif action == "recovery_anchor_arrived":
            pending = self._pending_recovery
            if pending is None:
                self._request_stop(
                    host,
                    "internal_error",
                    f"Mapping stopped at Taxel {self.anchor_taxel}: recovery state was lost.",
                )
            else:
                self._clearance_next_context = "failure"
                self._begin_collection(
                    host, "clearance_precheck", self.baseline_frame_count,
                    "Failed attempt returned to the clearance anchor: checking recovery before calibration",
                )
        elif action == "complete_anchor_arrived":
            self._completed = not bool(self._skipped_failed_taxels)
            self._stop_reason = "complete" if self._completed else "partial_complete"
            completion_name = (
                "Taxel map relocation"
                if self._relocation_active
                else "Taxel Position Mapping"
            )
            self._stop_message = (
                    f"{completion_name} {'complete' if self._completed else 'pass finished with unresolved taxels'} "
                    f"and returned to Taxel {self.anchor_taxel}: "
                    f"{self.mapped_target_count()}/"
                    f"{len(self.mapping_target_indices())} target taxels located, "
                    f"{len(self._failures)} failed."
                    + (" Review the failed taxels, then Resume to retry them; saved centres are retained."
                       if not self._completed else "")
            )
            host._stop_sidebar_control(show_result=True, message=self._stop_message)
        elif action == "anchor_stop_returned":
            reason, message = self._anchor_stop or (
                "scan_plane_too_high", f"Mapping stopped at the Taxel {self.anchor_taxel} anchor."
            )
            self._request_stop(host, reason, message)
        elif action == "resume_alignment_returned":
            self._resume_alignment_error = self._resume_alignment_error or (
                f"Resume verification failed because Taxel {self.anchor_taxel} no longer matches the saved map."
            )
            self._clearance_next_context = "resume_alignment_failed"
            self._begin_collection(
                host, "clearance_precheck", self.baseline_frame_count,
                "Resume alignment failed: checking recovery at the anchor before calibration and stopping",
            )

    def _handle_collection(
        self,
        host,
        action: str,
        frame,
        force_n: float,
        baseline_min,
        baseline_max,
        baseline_frames,
    ) -> None:
        if action in {"reference_baseline", "resume_anchor_stability"}:
            self._clearance_reference_frames = np.array(baseline_frames, dtype=float, copy=True)
        if action == "clearance_precheck":
            from phd.ui.mapping_safety import clearance_recovery
            check = clearance_recovery(
                baseline_frames, self._clearance_reference_frames, self.adaptive_min_margin_counts,
            )
            check.update(timestamp=datetime.now(timezone.utc).isoformat(), taxel_index=self._current_taxel)
            self._clearance_checks.append(check)
            if not check["recovered"]:
                details = check.get("unrecovered_details") or [
                    {"taxel": index} for index in check.get("unrecovered_taxels", [])
                ]
                taxels = ", ".join(
                    f"taxel {item['taxel']} ({item['shift_units']:+.1f}, limit {item['tolerance_units']:.1f})"
                    if "shift_units" in item else f"taxel {item['taxel']}"
                    for item in details[:8]
                )
                median_shift = check.get("median_shift_units")
                whole = "" if median_shift is None else f"; whole-sensor median shift {median_shift:+.1f}"
                self._request_stop(
                    host, "clearance_not_recovered",
                    f"Mapping stopped at the saved clearance anchor: the signal has not recovered on {taxels}{whole}. "
                    "Proximity lowers the signal, so a rise is baseline drift and a fall may be a residual "
                    "proximity response. No calibration was applied; inspect clearance and baseline drift "
                    "instead of zeroing a possible proximity response.",
                )
                return
            self._begin_sensor_calibration(
                host, next_action="clearance_calibrated",
                context=f"{self._clearance_next_context}_taxel_{self._current_taxel}",
            )
        elif action == "clearance_calibrated":
            self._clearance_reference_frames = np.array(baseline_frames, dtype=float, copy=True)
            self._force_baseline_n = float(force_n)
            if self._clearance_next_context == "auto_rerun":
                self._start_auto_rerun(host, frame, force_n, baseline_min, baseline_max, baseline_frames)
            elif self._clearance_next_context == "failure":
                self._clearance_next_context = ""
                pending = self._pending_recovery or {}
                if not pending.get("retry"):
                    if self._queue_auto_rerun(host, pending):
                        return
                    index = pending.get("taxel_index")
                    reason = pending.get("reason", "target not found")
                    self._pending_recovery = None
                    if (
                        pending.get("kind") == "target"
                        and pending.get("context") == "regular"
                        and index is not None
                        and (self.fixed_scan_active or (
                            pending.get("quality_failure")
                            and not (self._relocation_active and not self._relocation_finished)
                            and self.fit_lattice_from_positions({
                                known: {"tool_position_m": point}
                                for known, point in self._verified_mapping_positions().items()
                            }) is not None
                        ))
                    ):
                        self._skip_exhausted_taxel(host, int(index), str(reason))
                        return
                    guidance = ""
                    if index is not None and int(index) == int(self.anchor_taxel) and not self.fixed_scan_active:
                        guidance = (
                            " The automatic descent found no usable scan plane for the anchor. Set a "
                            "fixed one: jog the probe down over the anchor until it clearly stands out "
                            "from its neighbours (completed maps read about 100-200 units), press "
                            "Remember Scan Height, then start again."
                        )
                    self._request_stop(
                        host, "mapping_target_failed",
                        f"Mapping stopped at the Taxel {self.anchor_taxel} clearance anchor after calibration: "
                        f"Taxel {index} exhausted its {self.max_attempts_per_taxel} attempts ({reason}). "
                        f"Auto reruns used: {self._auto_rerun_count}/{self.max_auto_reruns}; "
                        "no further automatic recovery is enabled/eligible." + guidance,
                    )
                    return
                self._handle_collection(host, "retry_anchor_stability", frame, force_n,
                                        baseline_min, baseline_max, baseline_frames)
            else:
                try:
                    self._save_results(checkpoint=True)
                except Exception as exc:
                    self._request_stop(host, "checkpoint_failed", f"Mapping stopped at the clearance anchor: checkpoint could not be saved ({exc}).")
                    return
                self._continue_after_clearance(host)
        elif action == "reference_baseline":
            if self.fixed_scan_active:
                self._start_fixed_scan_mapping(host, force_n)
                return
            self._start_probe(
                host,
                frame,
                force_n,
                kind="reference",
                baseline_min=baseline_min,
                baseline_max=baseline_max,
                baseline_frames=baseline_frames,
            )
        elif action == "resume_anchor_stability":
            self._force_baseline_n = float(force_n)
            saved = self._positions[self.anchor_taxel]["tool_position_m"]
            self._start_target(
                host,
                self.anchor_taxel,
                saved[:2],
                "resume_anchor",
            )
        elif action == "target_baseline":
            self._start_probe(
                host,
                frame,
                force_n,
                kind="working",
                baseline_min=baseline_min,
                baseline_max=baseline_max,
                baseline_frames=baseline_frames,
            )
        elif action == "far_baseline":
            self._force_baseline_n = float(force_n)
            self._far_baseline = np.asarray(frame, dtype=float)
            self._far_baseline_min = np.asarray(baseline_min, dtype=float)
            self._far_baseline_max = np.asarray(baseline_max, dtype=float)
            self._far_baseline_frames = np.asarray(
                baseline_frames, dtype=float
            )
            target = np.array(
                [
                    float(self._current_target_xy[0]),
                    float(self._current_target_xy[1]),
                    float(self._probe_start_z),
                ]
            )
            self._command_move(
                host,
                target,
                "target_start_arrived",
                f"Taxel {self._current_taxel}: descending to the probe start height",
            )
        elif action == "scan_sample":
            self._record_scan_sample(host, frame)
        elif action == "verify_sample":
            self._finish_verify(host, frame)
        elif action == "retry_anchor_stability":
            pending = self._pending_recovery
            self._pending_recovery = None
            if pending is None:
                self._request_stop(
                    host,
                    "internal_error",
                    f"Mapping stopped at Taxel {self.anchor_taxel}: retry state was lost.",
                )
                return
            if str(pending.get("kind")) == "reference":
                self._current_taxel = self.anchor_taxel
                self._current_context = "reference"
                self._reference_attempt_count += 1
                self._start_probe(
                    host, frame, force_n, kind="reference",
                    baseline_min=baseline_min, baseline_max=baseline_max,
                    baseline_frames=baseline_frames,
                )
                return
            index = int(pending["taxel_index"])
            xy = pending.get("xy")
            if xy is None:
                self._request_stop(
                    host,
                    "internal_error",
                    f"Mapping stopped at Taxel {self.anchor_taxel}: retry XY target was lost.",
                )
                return
            from phd.ui.mapping_safety import local_retry_xy
            refined_xy = local_retry_xy(
                self._positions, index, xy, self.sensor_rows, self.sensor_columns, self.local_radius_m,
            )
            shift = float(np.linalg.norm(refined_xy-np.asarray(xy))) * 1000.
            if shift > .01:
                host._append_sidebar_control_message(
                    f"Taxel {index}: retry XY adjusted {shift:.2f} mm using measured local spacing, "
                    "within the existing local search radius."
                )
            self._start_target(
                host,
                index,
                refined_xy,
                str(pending.get("context") or "regular"),
            )

    def _start_coarse_scan(self, host, baseline, centre, *, dense=False) -> None:
        centre = np.asarray(centre, dtype=float)
        self._coarse_scan_centre = centre.copy()
        full_offsets = self.grid_offsets(self.coarse_radius_m, self.coarse_step_m)
        sparse_step = 2.0 * float(self.coarse_step_m)
        self._coarse_sparse_pass = bool(
            self.adaptive_coarse_search and not dense
            and len(full_offsets) > 169
            # Keep the sparse quantisation error within a local scan window.
            and sparse_step <= 2.0 * float(self.local_radius_m)
        )
        points = []
        offsets = (
            self.grid_offsets(self.coarse_radius_m, sparse_step)
            if self._coarse_sparse_pass else full_offsets
        )
        offsets = [(0.0, 0.0)] + [
            offset for offset in offsets if offset != (0.0, 0.0)
        ]
        for dx, dy in offsets:
            point = np.array(centre, copy=True)
            point[0] += dx
            point[1] += dy
            points.append(point)
        host._append_sidebar_control_message(
            f"Initial neighbour search: {len(points)} positions "
            f"({'fast first pass; automatic dense fallback' if self._coarse_sparse_pass else 'full-resolution grid'}). "
            "This one-time search learns the anchor and four neighbours; "
            "later taxels use only local centre scans."
        )
        self._start_scan(
            host,
            kind="coarse",
            points=points,
            baseline=baseline,
            target=None,
        )

    def _start_target(self, host, index: int, xy, context: str) -> None:
        anchor = np.asarray(self.anchor_pose["tool_position_m"], dtype=float)
        target_xy = np.asarray(xy, dtype=float)[:2]
        if float(np.linalg.norm(target_xy - anchor[:2])) > float(self.max_xy_offset_m):
            self._request_stop(
                host,
                "workspace_limit",
                f"Mapping stopped: predicted Taxel {index} is outside the {self.max_xy_offset_m * 1000.0:.1f} mm XY limit.",
            )
            return
        if self._probe_start_z is None:
            self._request_stop(
                host,
                "internal_error",
                "Mapping stopped: the sensor height has not been learned yet.",
            )
            return
        self._current_taxel = int(index)
        self._current_context = str(context)
        self._working_probe_boundary = None
        if context != "regular":
            self._current_prediction = None
        self._current_target_xy = np.array(target_xy, copy=True)
        self._local_recenters = 0
        self._attempt_counts[int(index)] = (
            int(self._attempt_counts.get(int(index), 0)) + 1
        )
        attempt = self._attempt_counts[int(index)]
        self._route_to(
            host,
            target_xy,
            float(anchor[2]),
            "target_over_arrived",
            f"moving above Taxel {index} for attempt "
            f"{attempt}/{self.max_attempts_per_taxel}",
        )

    def _fit_relocation_transform(self) -> Optional[dict]:
        matched = sorted(
            set(self._relocation_source_positions).intersection(self._positions)
        )
        if len(matched) < 3:
            return None
        old_xy = [
            self._relocation_source_positions[index]["tool_position_m"][:2]
            for index in matched
        ]
        new_xy = [self._positions[index]["tool_position_m"][:2] for index in matched]
        fitted = self.estimate_rigid_transform_2d(old_xy, new_xy)
        if fitted is None:
            return None
        z_shifts = [
            float(self._positions[index]["tool_position_m"][2])
            - float(self._relocation_source_positions[index]["tool_position_m"][2])
            for index in matched
        ]
        fitted["z_shift_m"] = float(np.median(z_shifts))
        fitted["matched_indices"] = matched
        return fitted

    @staticmethod
    def _apply_rigid_xy(position, fitted: dict) -> np.ndarray:
        value = np.asarray(position, dtype=float).copy()
        rotation = np.asarray(fitted["rotation"], dtype=float)
        translation = np.asarray(fitted["translation"], dtype=float)
        value[:2] = rotation @ value[:2] + translation
        value[2] += float(fitted.get("z_shift_m", 0.0))
        return value

    def _choose_relocation_verification_indices(self, count=3) -> list[int]:
        references = set(self._relocation_reference_indices)
        candidates = [
            index
            for index in sorted(self._relocation_source_positions)
            if index not in references
        ]
        if not candidates:
            return []
        source_xy = {
            index: np.asarray(
                self._relocation_source_positions[index]["tool_position_m"],
                dtype=float,
            )[:2]
            for index in candidates
        }
        anchor_xy = np.asarray(
            self._relocation_source_positions[self.anchor_taxel]["tool_position_m"],
            dtype=float,
        )[:2]
        selected = [
            max(candidates, key=lambda index: np.linalg.norm(source_xy[index] - anchor_xy))
        ]
        while len(selected) < min(int(count), len(candidates)):
            remaining = [index for index in candidates if index not in selected]
            selected.append(
                max(
                    remaining,
                    key=lambda index: min(
                        np.linalg.norm(source_xy[index] - source_xy[chosen])
                        for chosen in selected
                    ),
                )
            )
        return selected

    def _start_next_relocation_verification(self, host) -> None:
        if not self._relocation_active:
            self._request_stop(
                host,
                "internal_error",
                "Mapping stopped: relocation state was lost.",
            )
            return
        fitted = self._fit_relocation_transform()
        if fitted is None:
            self._request_stop(
                host,
                "relocation_fit_failed",
                "Relocation stopped: the new reference taxels did not provide a reliable 2D transform.",
            )
            return
        self._relocation_transform = fitted
        if not self._relocation_reference_indices:
            self._relocation_reference_indices = sorted(self._positions)
            self._relocation_verification_queue = (
                self._choose_relocation_verification_indices(3)
            )
            host._append_sidebar_control_message(
                "Moved-sensor transform estimated: rotation "
                f"{fitted['angle_deg']:+.2f}°, XY fit RMS "
                f"{fitted['rms_residual_mm']:.2f} mm. Verifying "
                f"{len(self._relocation_verification_queue)} distributed taxels."
            )
        if self._relocation_verification_queue:
            index = self._relocation_verification_queue.pop(0)
            predicted = self._apply_rigid_xy(
                self._relocation_source_positions[index]["tool_position_m"],
                fitted,
            )
            self._start_target(
                host,
                index,
                predicted[:2],
                "relocation_verify",
            )
            return
        self._finish_relocation(host, fitted)

    def _finish_relocation(self, host, fitted: dict) -> None:
        anchor_xy = np.asarray(self.anchor_pose["tool_position_m"], dtype=float)[:2]
        anchor_z = float(self.anchor_pose["tool_position_m"][2])
        quaternion = list(self.anchor_pose["tool_quaternion_wxyz"])
        verified = copy.deepcopy(self._positions)
        transformed = {}
        for index, source in sorted(self._relocation_source_positions.items()):
            centre = self._apply_rigid_xy(source["tool_position_m"], fitted)
            if float(np.linalg.norm(centre[:2] - anchor_xy)) > float(
                self.max_xy_offset_m
            ):
                self._request_stop(
                    host,
                    "workspace_limit",
                    f"Relocation stopped: transformed Taxel {index} is outside "
                    f"the {self.max_xy_offset_m * 1000.0:.1f} mm XY limit.",
                )
                return
            safe_source = source.get(
                "safe_approach_position_m", source["tool_position_m"]
            )
            safe = self._apply_rigid_xy(safe_source, fitted)
            safe[2] = anchor_z
            record = copy.deepcopy(source)
            record.update(
                {
                    "tool_position_m": centre.tolist(),
                    "safe_approach_position_m": safe.tolist(),
                    "tool_quaternion_wxyz": quaternion,
                    "method": "relocated_rigid_transform",
                    "relocation_verified": False,
                    "relocation_source_position_m": list(
                        source["tool_position_m"]
                    ),
                }
            )
            # These are absolute heights. The rigid move changes Z, so a copied
            # taxel must not keep the height from before the sensor was moved.
            for field in ("probe_start_z_m", "proximity_onset_z_m", "centre_scan_z_m"):
                try:
                    height = float(record.get(field))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(height):
                    record[field] = height + float(fitted["z_shift_m"])
            transformed[index] = record
        for index, record in verified.items():
            record = copy.deepcopy(record)
            record["relocation_verified"] = True
            record["relocation_source_position_m"] = list(
                self._relocation_source_positions[index]["tool_position_m"]
            )
            transformed[index] = record
        rotation = np.asarray(fitted["rotation"], dtype=float)

        def rotated_vector(name):
            value = (self._relocation_source_result or {}).get(name)
            try:
                vector = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                return None
            if vector.shape != (2,) or not np.all(np.isfinite(vector)):
                return None
            return rotation @ vector

        self._positions = transformed
        self._initial_row_vector_xy = rotated_vector("initial_row_vector_xy_m")
        self._initial_column_vector_xy = rotated_vector(
            "initial_column_vector_xy_m"
        )
        self._row_vector_xy = rotated_vector("row_vector_xy_m")
        self._column_vector_xy = rotated_vector("column_vector_xy_m")
        if self._row_vector_xy is None or self._column_vector_xy is None:
            lattice = self.fit_lattice_from_positions(verified)
            if lattice is not None:
                self._row_vector_xy, self._column_vector_xy, _score = lattice
        self._relocation_transform = {
            "source_json_path": str(self._relocation_source_path or ""),
            "rotation_deg": float(fitted["angle_deg"]),
            "rotation_matrix": np.asarray(
                fitted["rotation"], dtype=float
            ).tolist(),
            "translation_xy_m": np.asarray(
                fitted["translation"], dtype=float
            ).tolist(),
            "z_shift_m": float(fitted["z_shift_m"]),
            "rms_residual_mm": float(fitted["rms_residual_mm"]),
            "max_residual_mm": float(fitted["max_residual_mm"]),
            "reference_taxels": list(self._relocation_reference_indices),
            "verified_taxels": sorted(verified),
            "transformed_taxel_count": len(transformed),
        }
        self._failures = {}
        self._relocation_finished = True
        self._publish_live_map(host)
        host._append_sidebar_control_message(
            f"Relocation verified. Applied a {fitted['angle_deg']:+.2f}° rotation "
            f"and {np.linalg.norm(fitted['translation']) * 1000.0:.2f} mm XY "
            f"translation to {len(transformed)} taxels "
            f"(fit RMS {fitted['rms_residual_mm']:.2f} mm, max "
            f"{fitted['max_residual_mm']:.2f} mm)."
        )
        if self._relocation_continues_mapping:
            remaining = set(self.mapping_target_indices()).difference(self._positions)
            host._append_sidebar_control_message(
                f"Continuing the partial map: {len(remaining)} target taxels remain. "
                "Only re-measured taxels seed the next targets; transformed "
                "positions are kept but never used to plan motion."
            )
            self._start_next_regular_taxel(host)
            return
        self._current_taxel = self.anchor_taxel
        self._current_context = "complete_return"
        anchor = self.anchor_pose["tool_position_m"]
        self._route_to(
            host,
            anchor[:2],
            anchor[2],
            "complete_anchor_arrived",
            f"relocation finished; returning to the safe Taxel {self.anchor_taxel} anchor",
        )

    def _start_next_neighbor(self, host) -> None:
        if self._neighbor_queue:
            index = self._neighbor_queue.pop(0)
            self._start_target(host, index, self._neighbor_guesses[index], "neighbor")
            return
        if self._relocation_active and not self._relocation_finished:
            self._start_next_relocation_verification(host)
            return
        predicted = self.predict_lattice_positions()
        if predicted is None:
            self._request_stop(
                host,
                "invalid_lattice",
                "Mapping stopped: the four measured neighbours do not form a reliable 2D taxel lattice.",
            )
            return
        predicted_positions, self._row_vector_xy, self._column_vector_xy, orthogonality = predicted
        anchor_xy = np.asarray(self.anchor_pose["tool_position_m"], dtype=float)[:2]
        target_indices = set(self.mapping_target_indices())
        if any(
            float(np.linalg.norm(position[:2] - anchor_xy)) > float(self.max_xy_offset_m)
            for index, position in predicted_positions.items()
            if index in target_indices
        ):
            self._request_stop(
                host,
                "workspace_limit",
                f"Mapping stopped: the fitted {self.sensor_rows}x{self.sensor_columns} grid exceeds the configured XY workspace limit.",
            )
            return
        self._predicted_positions = predicted_positions
        self._mapping_queue = sorted(target_indices.difference(self._positions))
        host._append_sidebar_control_message(
            "Taxel lattice fitted: row pitch "
            f"{np.linalg.norm(self._row_vector_xy) * 1000.0:.2f} mm, column pitch "
            f"{np.linalg.norm(self._column_vector_xy) * 1000.0:.2f} mm, "
            f"2D separation score {orthogonality:.2f}. "
            "Continuing outwards from verified neighbours; local position estimates are updated after each verified taxel."
        )
        self._start_next_regular_taxel(host)

    def _remaining_target_indices(self) -> set[int]:
        """Unmapped targets, excluding taxels skipped after exhausting retries."""
        return (
            set(self.mapping_target_indices())
            .difference(self._positions)
            .difference(self._skipped_failed_taxels)
        )

    def _skip_exhausted_taxel(self, host, index: int, reason: str) -> None:
        self._skipped_failed_taxels.add(int(index))
        host._append_sidebar_control_message(
            f"Taxel {index} skipped after {self.max_attempts_per_taxel} attempts ({reason}). "
            f"Continuing with the next taxel; {len(self._skipped_failed_taxels)} skipped so far."
        )
        try:
            self._save_results(checkpoint=True)
        except Exception as exc:
            self._request_stop(
                host, "checkpoint_failed",
                f"Mapping stopped at the clearance anchor: checkpoint could not be saved ({exc}).",
            )
            return
        self._start_next_regular_taxel(host)

    def _start_next_regular_taxel(self, host) -> None:
        self._mapping_queue = sorted(self._remaining_target_indices())
        if not self._mapping_queue:
            self._current_prediction = None
            self._current_taxel = self.anchor_taxel
            self._current_context = "complete_return"
            anchor = self.anchor_pose["tool_position_m"]
            self._route_to(
                host,
                anchor[:2],
                anchor[2],
                "complete_anchor_arrived",
                f"mapping finished; returning to the safe Taxel {self.anchor_taxel} anchor",
            )
            return
        pose = self._read_tool_pose(self._robot_api(host))
        try:
            prediction = self.plan_frontier_target(None if pose is None else pose[0][:2])
        except ValueError as exc:
            self._request_stop(host, "invalid_frontier", f"Mapping stopped: {exc}")
            return
        index = prediction["taxel_index"]
        self._current_prediction = prediction
        self._row_vector_xy = np.asarray(prediction["row_vector_xy_m"])
        self._column_vector_xy = np.asarray(prediction["column_vector_xy_m"])
        predicted_pose = np.array(self._positions[prediction["verified_neighbours"][0]]["tool_position_m"], dtype=float)
        predicted_pose[:2] = prediction["position_xy_m"]
        self._predicted_positions[index] = predicted_pose
        self._mapping_queue.remove(index)
        host._append_sidebar_control_message(
            f"Taxel {index}: one-step estimate from verified neighbour(s) "
            + ", ".join(str(known) for known in prediction["verified_neighbours"])
            + f"; support spread {prediction['support_spread_mm']:.2f} mm (not position accuracy)."
        )
        self._start_target(
            host,
            index,
            self._predicted_positions[index][:2],
            "regular",
        )

    def _track_sensor_freshness(self, sequence: int, now: float) -> bool:
        if self._last_sensor_sequence != int(sequence):
            self._last_sensor_sequence = int(sequence)
            self._last_sensor_advance_at = float(now)
            return True
        if self._last_sensor_advance_at is None:
            self._last_sensor_advance_at = float(now)
            return True
        return now - self._last_sensor_advance_at <= float(self.sensor_stale_sec)

    def _settings(self) -> dict:
        return {
            "sensor_rows": int(self.sensor_rows),
            "sensor_columns": int(self.sensor_columns),
            "anchor_taxel": int(self.anchor_taxel),
            "mapping_order": self.MAPPING_ORDER,
            "centre_scan_policy": self.CENTRE_SCAN_POLICY,
            "mapping_quality_policy": self.QUALITY_POLICY,
            "stationary_strength_confirmation_batches": self.STRENGTH_CONFIRM_BATCHES,
            "stationary_strength_borderline_ratio": self.STRENGTH_BORDERLINE_RATIO,
            "centre_repeat_min_dominance": self.CENTRE_REPEAT_MIN_DOMINANCE,
            "centre_repeat_tolerance_mm": max(self.local_step_m, 2.*self.position_tolerance_m)*1000.,
            "search_safety_policy": self.SEARCH_SAFETY_POLICY,
            "probe_boundary_policy": "upward_headroom_fresh_sampling_v1",
            "probe_boundary_headroom_mm": self._probe_boundary_headroom_m() * 1000.,
            "max_boundary_corrections_per_probe": 1,
            "resume_attempt_budget": "fresh_per_explicit_resume_history_preserved",
            "recalibrate_after_each_taxel_and_recoverable_failure": True,
            "clearance_source": "operator-established remembered anchor, checked for signal recovery",
            "noncontact_force_stop_g": self.NONCONTACT_FORCE_STOP_G,
            "noncontact_interpretation": "No contact is intended. Force and travel guards cannot certify air gap; supervise with independently checked clearance.",
            "skip_outer_border": bool(self.skip_outer_border),
            "proximity_detection_mode": str(self.proximity_detection_mode),
            "signal_threshold_counts": float(self.signal_threshold_counts),
            "adaptive_margin_fraction": float(self.adaptive_margin_fraction),
            "adaptive_min_margin_counts": float(
                self.adaptive_min_margin_counts
            ),
            "probe_window_frames": int(self.probe_window_frames),
            "reference_probe_speed_mm_s": float(self.reference_probe_speed_m_s) * 1000.0,
            "reference_max_descent_mm": float(self.reference_max_descent_m) * 1000.0,
            "probe_start_margin_mm": float(self.probe_start_margin_m) * 1000.0,
            "probe_speed_mm_s": float(self.probe_speed_m_s) * 1000.0,
            "probe_confirmation_speed_mm_s": float(
                self.probe_confirmation_speed_m_s
            )
            * 1000.0,
            "probe_step_mm": float(self.probe_step_m) * 1000.0,
            "probe_motion": self.probe_motion,
            "probe_sample_frames": int(self.probe_sample_frames),
            "max_probe_descent_mm": float(self.max_probe_descent_m) * 1000.0,
            "probe_confirm_frames": int(self.probe_confirm_frames),
            "probe_min_dominance": float(self.probe_min_dominance),
            "centre_scan_signal_multiplier": float(
                self.centre_scan_signal_multiplier
            ),
            "centre_scan_max_extra_descent_mm": float(
                self.centre_scan_max_extra_descent_m
            )
            * 1000.0,
            "final_min_dominance": float(self.final_min_dominance),
            "force_guard_g": float(self.force_guard_g),
            "coarse_radius_mm": float(self.coarse_radius_m) * 1000.0,
            "coarse_step_mm": float(self.coarse_step_m) * 1000.0,
            "local_radius_mm": float(self.local_radius_m) * 1000.0,
            "local_step_mm": float(self.local_step_m) * 1000.0,
            "local_radius_pitch_fraction": float(self.local_radius_pitch_fraction),
            "max_local_recenters": int(self.max_local_recenters),
            "line_speed_mm_s": float(self.line_speed_m_s) * 1000.0,
            "position_tolerance_mm": float(self.position_tolerance_m) * 1000.0,
            "move_timeout_min_sec": float(self.move_timeout_sec),
            "move_timeout_scale": float(self.move_timeout_scale),
            "move_timeout_margin_sec": float(self.move_timeout_margin_sec),
            "move_timeout_max_sec": float(self.move_timeout_max_sec),
            "move_timeout_fallback_project_speed_percent": float(
                self.move_timeout_fallback_project_speed_percent
            ),
            "max_xy_offset_mm": float(self.max_xy_offset_m) * 1000.0,
            "verify_frame_count": int(self.verify_frame_count),
            "scan_frame_count": int(self.scan_frame_count),
            "scan_sample_reducer": "median",
            "adaptive_coarse_search": bool(self.adaptive_coarse_search),
            "reverse_scan_order": bool(self.reverse_scan_order),
            "baseline_frame_count": int(self.baseline_frame_count),
            "baseline_stability_fraction": float(
                self.baseline_stability_fraction
            ),
            "baseline_stability_floor_counts": float(
                self.baseline_stability_floor_counts
            ),
            "max_attempts_per_taxel": int(self.max_attempts_per_taxel),
            "max_auto_reruns": int(self.max_auto_reruns),
            "auto_rerun_delay_sec": float(self.auto_rerun_delay_sec),
            "sensor_calibration_timeout_sec": float(
                self.sensor_calibration_timeout_sec
            ),
            "resume_xy_tolerance_mm": float(self.resume_xy_tolerance_m) * 1000.0,
            "resume_z_tolerance_mm": float(self.resume_z_tolerance_m) * 1000.0,
            "fixed_scan_z_mm": (
                None if self.fixed_scan_z_m is None else float(self.fixed_scan_z_m) * 1000.0
            ),
            "scan_plane_source": self._scan_plane_source,
            "fixed_scan_approach_window_mm": float(self.fixed_scan_approach_window_m) * 1000.0,
            "fixed_scan_stop_ratio": float(self.fixed_scan_stop_ratio),
            "fixed_scan_contact_ratio": float(self.fixed_scan_contact_ratio),
            "fixed_scan_reference_signal_counts": self._fixed_scan_reference_signal,
        }

    def _prepare_output_folder(self) -> None:
        if self.output_directory is None:
            self._output_folder = None
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sensor_name = (
            str(self.active_sensor_label).strip()
            or str(self.active_sensor_key_name).strip()
            or f"{self.sensor_rows}x{self.sensor_columns}_sensor"
        )
        sensor_token = CalibrationTask._folder_token(sensor_name)
        if str(self.proximity_detection_mode) == "robust_shift":
            detection_token = (
                "robust-shift-"
                f"win{int(self.probe_window_frames)}-"
                f"min{CalibrationTask._folder_number(self.adaptive_min_margin_counts)}units"
            )
        elif str(self.proximity_detection_mode) == "adaptive_range":
            detection_token = (
                "adaptive-"
                f"{CalibrationTask._folder_number(self.adaptive_margin_fraction * 100.0)}pct-"
                f"min{CalibrationTask._folder_number(self.adaptive_min_margin_counts)}units"
            )
        else:
            detection_token = (
                "fixed-"
                f"{CalibrationTask._folder_number(self.signal_threshold_counts)}units"
            )
        folder_name = (
            f"taxel_position_map_"
            f"{'relocated_' if self._relocation_active else ''}"
            f"{timestamp}_sensor-{sensor_token}_"
            f"anchor-{self.anchor_taxel}_"
            f"detection-{detection_token}_"
            f"probe-{CalibrationTask._folder_number(self.probe_speed_m_s * 1000.0)}mmps_"
            f"step-{CalibrationTask._folder_number(self.probe_step_m * 1000.0)}mm_"
            f"samples-{int(self.probe_sample_frames)}_"
            f"scanstrength-"
            f"{CalibrationTask._folder_number(self.centre_scan_signal_multiplier)}x_"
            f"scanextra-"
            f"{CalibrationTask._folder_number(self.centre_scan_max_extra_descent_m * 1000.0)}mm"
        )
        self._output_folder = self.output_directory / folder_name
        self._output_folder.mkdir(parents=True, exist_ok=False)

    def _save_results(self, *, checkpoint=False) -> Optional[dict]:
        lattice_fit = self.lattice_fit_quality()
        residuals = (lattice_fit or {}).get("residual_mm_by_taxel", {})
        positions = []
        for index in sorted(self._positions):
            record = dict(self._positions[index])
            record["lattice_residual_mm"] = residuals.get(str(index))
            positions.append(record)

        def vector(value):
            return None if value is None else np.asarray(value, dtype=float).tolist()

        result = {
            "version": 3,
            "task": self.id,
            "sensor_shape": [self.sensor_rows, self.sensor_columns],
            "sensor_identity": self.active_sensor_key_name,
            "centre_interpretation": "Response centres, not certified geometric centres. Lattice residuals measure internal consistency, not absolute accuracy.",
            "started_at": self._started_at_iso,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "completed": bool(self._completed),
            "stop_reason": str(self._stop_reason or ("checkpoint" if checkpoint else "manual_stop")),
            "stop_message": str(self._stop_message),
            "stop_snapshot": copy.deepcopy(self._stop_snapshot),
            "search_safety_policy": self.SEARCH_SAFETY_POLICY,
            "clearance_checks": copy.deepcopy(self._clearance_checks),
            "stationary_strength_checks": copy.deepcopy(self._strength_checks),
            "centre_repeat_checks": copy.deepcopy(self._centre_checks),
            "auto_rerun_count": int(self._auto_rerun_count),
            "auto_rerun_history": copy.deepcopy(self._auto_rerun_history),
            "settings": self._settings(),
            "anchor_pose": self.anchor_pose,
            "reference_onset_z_m": self._reference_onset_z,
            "probe_start_z_m": self._probe_start_z,
            "reference_attempt_count": int(self._reference_attempt_count),
            "initial_row_vector_xy_m": vector(self._initial_row_vector_xy),
            "initial_column_vector_xy_m": vector(self._initial_column_vector_xy),
            "row_vector_xy_m": vector(self._row_vector_xy),
            "column_vector_xy_m": vector(self._column_vector_xy),
            "lattice_fit": lattice_fit,
            "mapped_count": self.mapped_target_count(),
            "measured_position_count": len(positions),
            "target_taxel_count": len(self.mapping_target_indices()),
            "skipped_taxels": self.skipped_taxel_indices(),
            "failed_count": len(self._failures),
            "failures": {str(index): reason for index, reason in sorted(self._failures.items())},
            "failed_attempt_total": int(self._failed_attempt_total),
            "skipped_failed_taxels": sorted(int(index) for index in self._skipped_failed_taxels),
            "unresolved_taxels": sorted(set(self.mapping_target_indices()).difference(self._positions)),
            "attempt_counts": {
                str(index): count
                for index, count in sorted(self._attempt_counts.items())
            },
            "baseline_checks": list(self._baseline_checks),
            "sensor_calibration_history": copy.deepcopy(
                self._sensor_calibration_history
            ),
            "resume_history": copy.deepcopy(self._resume_history),
            "relocation": copy.deepcopy(self._relocation_transform)
            if self._relocation_active
            else None,
            "relocation_checkpoint": self._relocation_checkpoint(),
            "positions": positions,
        }
        if self._output_folder is None:
            result = json.loads(json.dumps(result, default=_mapping_json_default, allow_nan=False))
            self.last_result = result
            return result
        json_path = self._output_folder / "taxel_position_map.json"
        csv_path = self._output_folder / "taxel_positions.csv"
        samples_path = self._output_folder / "scan_samples.csv"
        baseline_path = self._output_folder / "baseline_checks.csv"
        probe_path = self._output_folder / "probe_levels.csv"
        result.update(
            {
                "output_folder": str(self._output_folder),
                "json_path": str(json_path),
                "positions_csv_path": str(csv_path),
                "samples_csv_path": str(samples_path),
                "baseline_checks_csv_path": str(baseline_path),
                "probe_levels_csv_path": str(probe_path),
            }
        )
        # Validate the complete snapshot BEFORE overwriting any CSV or temp
        # JSON. Keep allow_nan=False: invalid coordinates must still fail.
        serialized = json.dumps(result, default=_mapping_json_default, indent=2, allow_nan=False) + "\n"
        result = json.loads(serialized)
        optional_position_fields = [
            "probe_start_z_m",
            "proximity_onset_z_m",
            "centre_scan_z_m",
            "centre_scan_extra_descent_mm",
            "centre_scan_signal_units",
            "centre_scan_target_units",
            "centre_scan_target_reached",
            "centre_scan_policy",
            "quality_policy",
            "centre_repeat_status",
            "centre_repeat_separation_mm",
            "centre_scan_signal_multiplier",
            "centre_scan_max_extra_descent_mm",
            "noncontact_force_stop_g",
            "row_axis_method",
            "column_axis_method",
            "row_offset_mm",
            "column_offset_mm",
            "row_balance_offset_mm",
            "row_peak_offset_mm",
            "column_balance_offset_mm",
            "column_peak_offset_mm",
            "local_recenters",
            "lattice_residual_mm",
            "baseline_target_range_counts",
            "baseline_grid_p90_range_counts",
            "baseline_grid_max_range_counts",
            "proximity_detection_mode",
            "proximity_threshold_counts",
            "proximity_baseline_min_counts",
            "proximity_baseline_max_counts",
            "proximity_baseline_range_counts",
            "proximity_margin_counts",
            "proximity_lower_trigger_counts",
            "proximity_upper_trigger_counts",
            "proximity_rolling_window_frames",
            "proximity_rolling_robust_sigma_counts",
            "proximity_rolling_low_percentile_counts",
        ]
        position_fields = [
            "taxel_index", "row", "column", "status", "attempt_count",
            "x_m", "y_m", "z_m",
            "safe_x_m", "safe_y_m", "safe_z_m", "signal_counts",
            "competing_signal_counts", "dominance_ratio", "confidence", "method",
        ] + optional_position_fields
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=position_fields)
            writer.writeheader()
            for record in positions:
                position = record["tool_position_m"]
                safe = record["safe_approach_position_m"]
                row = {
                    "taxel_index": record["taxel_index"],
                    "row": record["row"],
                    "column": record["column"],
                    "status": record["status"],
                    "attempt_count": record.get("attempt_count", 1),
                    "x_m": position[0], "y_m": position[1], "z_m": position[2],
                    "safe_x_m": safe[0], "safe_y_m": safe[1], "safe_z_m": safe[2],
                    "signal_counts": record["signal_counts"],
                    "competing_signal_counts": record["competing_signal_counts"],
                    "dominance_ratio": record["dominance_ratio"],
                    "confidence": record["confidence"],
                    "method": record["method"],
                }
                row.update(
                    {name: record.get(name) for name in optional_position_fields}
                )
                writer.writerow(row)
        sample_fields = [
            "scan", "scan_pass", "sample_frames", "taxel_index", "attempt_number", "axis",
            "nominal_offset_mm", "point_number",
            "x_m", "y_m", "z_m",
            "target_signal_counts", "competing_signal_counts", "dominance_ratio", "dominant_taxel",
        ]
        with samples_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sample_fields)
            writer.writeheader()
            writer.writerows(self._sample_log)
        baseline_fields = [
            "action",
            "taxel_index",
            "attempt_number",
            "frame_count",
            "stability_limit_counts",
            "maximum_limit_counts",
            "target_range_counts",
            "grid_p90_range_counts",
            "grid_max_range_counts",
            "target_rolling_median_range_counts",
            "grid_p90_rolling_median_range_counts",
            "grid_max_rolling_median_range_counts",
            "target_drift_counts",
            "grid_p90_drift_counts",
            "target_mean_counts",
            "target_min_counts",
            "target_max_counts",
            "adaptive_margin_counts",
            "adaptive_lower_trigger_counts",
            "adaptive_upper_trigger_counts",
            "robust_shift_threshold_counts",
            "rolling_robust_sigma_counts",
            "rolling_low_percentile_counts",
            "stable",
        ]
        with baseline_path.open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=baseline_fields)
            writer.writeheader()
            writer.writerows(self._baseline_checks)
        probe_fields = [
            "timestamp", "taxel_index", "attempt", "phase", "probe_stage",
            "tcp_x_m", "tcp_y_m", "tcp_z_m", "force_raw_g", "force_reference_g",
            "force_change_g", "probe_start_z_m", "probe_descent_mm",
            "onset_confirmed_before_level", "onset_z_m", "extra_descent_mm",
            "tcp_feedback_age_sec", "tcp_feedback_fresh", "force_sample_age_sec",
            "working_boundary_z_m", "working_boundary_source", "signal_change_units",
            "onset_threshold_units", "stronger_target_units", "qualified", "dominance",
            "sensor_frame_sequence",
            "strength_confirmation_batch", "strength_confirmation_accepted",
            "strength_confirmation_median_units",
        ]
        with probe_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=probe_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._probe_log)
        temporary_json = json_path.with_suffix(".json.tmp")
        with temporary_json.open("w", encoding="utf-8") as stream:
            stream.write(serialized)
        os.replace(temporary_json, json_path)
        self.last_result = result
        return result

    def _begin_force_meter_connection(self, host) -> bool:
        """Start the force reader and wait without allowing robot motion."""
        ui_ros = getattr(host, "ui_ros", None)
        if ui_ros is None:
            host._append_sidebar_control_message(
                f"{self.label} could not access the force-transmitter controls."
            )
            return False
        reader_running = self._force_reader_running(ui_ros)
        if reader_running:
            # This method is reached only when no fresh sample exists. Restart
            # a stale reader before the robot is allowed to move.
            stop_reader = getattr(ui_ros, "_stop_force_meter", None)
            if callable(stop_reader):
                try:
                    stop_reader()
                except Exception:
                    pass
            reader_running = self._force_reader_running(ui_ros)
        if not reader_running:
            refresh_ports = getattr(ui_ros, "_refresh_force_meter_ports", None)
            if callable(refresh_ports):
                refresh_ports()
            start_reader = getattr(ui_ros, "_start_force_meter", None)
            if not callable(start_reader) or not bool(start_reader()):
                status_label = getattr(ui_ros, "force_meter_status_label", None)
                status = (
                    str(status_label.text()).strip()
                    if status_label is not None and hasattr(status_label, "text")
                    else ""
                )
                detail = f" ({status})" if status else ""
                host._append_sidebar_control_message(
                    f"{self.label} could not automatically start the "
                    f"force transmitter{detail}."
                )
                return False
        self._active = True
        self._phase = "connecting_force_meter"
        self._force_connection_started_at = time.monotonic()
        setter = getattr(host, "_set_taxel_mapping_controls_running", None)
        if callable(setter):
            setter(True)
        self._progress(
            host,
            "Connecting force transmitter; waiting for a fresh safety sample",
        )
        host._append_sidebar_control_message(
            f"{self.label} is connecting the force transmitter "
            "automatically. Robot motion will start only after fresh force "
            "data is received."
        )
        return True

    def _tick_force_meter_connection(self, host, now: float) -> None:
        force = self._fresh_force(host, now)
        if force is not None:
            self._active = False
            self._phase = "idle"
            self._force_connection_started_at = None
            host._append_sidebar_control_message(
                "Force transmitter connected and live data received. "
                f"Starting {self.label}."
            )
            if not self.on_start(host):
                host._stop_sidebar_control(
                    show_result=False,
                    message=f"{self.label} did not start",
                )
            return
        ui_ros = getattr(host, "ui_ros", None)
        error_message = str(
            getattr(ui_ros, "_force_meter_error_message", "") or ""
        ).strip()
        started_at = self._force_connection_started_at
        elapsed = 0.0 if started_at is None else now - float(started_at)
        reader_stopped = (
            elapsed >= 0.5 and not self._force_reader_running(ui_ros)
        )
        timed_out = elapsed >= float(self.force_connection_timeout_sec)
        if not error_message and not reader_stopped and not timed_out:
            return
        if error_message:
            detail = error_message
        elif reader_stopped:
            detail = "the force-transmitter reader stopped before sending data"
        else:
            detail = (
                "no fresh force sample was received within "
                f"{self.force_connection_timeout_sec:.0f} seconds"
            )
        self._request_stop(
            host,
            "force_connection_failed",
            f"{self.label} did not start: "
            f"{detail}. Check the serial port, transmitter power, RS-485 "
            "wiring, protocol, and baud rate.",
        )

    def on_start(self, host) -> bool:
        self._active = False
        self._mapping_run_started = False
        self._coarse_scan_centre = None
        self._coarse_sparse_pass = False
        self._force_connection_started_at = None
        self._completed = False
        self._stop_reason = ""
        self._stop_message = ""
        self._stop_snapshot = None
        self._probe_log = []
        self._working_probe_boundary = None
        self._probe_candidate_seen = False
        self._clearance_reference_frames = None
        self._probe_strength_hold = None
        self._strength_checks = []
        self._target_strength_check = None
        self._centre_repeat = None
        self._centre_checks = []
        self._target_centre_check = None
        self._clearance_next_context = ""
        self._clearance_checks = []
        self._auto_rerun_count = 0
        self._auto_rerun_history = []
        self._auto_rerun_pending = None
        self._auto_rerun_deadline = None
        self._fixed_scan_reference_signal = None
        self._skipped_failed_taxels = set()
        self._failed_attempt_total = 0
        self.last_result = None
        resume_result = self._resume_request
        relocation_result = self._relocation_request
        mismatch = position_map_sensor_error(host, resume_result or relocation_result)
        if mismatch:
            host._append_sidebar_control_message(mismatch)
            return False
        if resume_result is not None and relocation_result is not None:
            host._append_sidebar_control_message(
                "Taxel mapping cannot resume and relocate at the same time."
            )
            return False
        # Recheck the live identity even for Resume: the sensor may have changed
        # since the last idle UI refresh. Choices are restored per sensor/anchor.
        self.use_connected_sensor(host)
        resume_source = self._resume_source_path
        relocation_source = (
            self._relocation_source_path
            if relocation_result is not None
            else None
        )
        if resume_result is not None:
            valid, message = self.validate_resume_result(resume_result)
            if not valid:
                host._append_sidebar_control_message(
                    f"Taxel mapping cannot resume: {message}"
                )
                return False
            try:
                self._apply_resume_settings(resume_result)
            except (KeyError, TypeError, ValueError) as exc:
                host._append_sidebar_control_message(
                    f"Taxel mapping cannot resume: invalid saved settings ({exc})."
                )
                return False
            plane_mismatch = self._resume_scan_plane_mismatch(resume_result)
            if plane_mismatch:
                host._append_sidebar_control_message(
                    f"Taxel mapping cannot resume: {plane_mismatch}"
                )
                return False
        if relocation_result is not None:
            valid, message = self.validate_relocation_result(relocation_result)
            if not valid:
                host._append_sidebar_control_message(
                    f"Taxel map cannot be relocated: {message}"
                )
                return False
            source_settings = relocation_result["settings"]
            source_rows = int(source_settings["sensor_rows"])
            source_columns = int(source_settings["sensor_columns"])
            source_anchor = int(source_settings["anchor_taxel"])
            if (source_rows, source_columns) != (
                int(self.sensor_rows),
                int(self.sensor_columns),
            ):
                host._append_sidebar_control_message(
                    "Taxel map cannot be relocated: the loaded map is "
                    f"{source_rows}x{source_columns}, but the live sensor is "
                    f"{self.sensor_rows}x{self.sensor_columns}."
                )
                return False
            if source_anchor != int(self.anchor_taxel):
                host._append_sidebar_control_message(
                    "Taxel map cannot be relocated: its reference taxel does "
                    "not match the live sensor's centre taxel."
                )
                return False
            proposed = revision_setup(relocation_result)
            if proposed is not None:
                try:
                    self.apply_remembered_setup(proposed)
                except (KeyError, TypeError, ValueError) as exc:
                    host._append_sidebar_control_message(f"Invalid map setup revision: {exc}")
                    return False
        if self.anchor_pose is None:
            host._append_sidebar_control_message(
                f"Remember the current safe pose above Taxel {self.anchor_taxel} before starting mapping."
            )
            return False
        rerun_limit = finite_number(self.max_auto_reruns)
        rerun_delay = finite_number(self.auto_rerun_delay_sec)
        if (rerun_limit is None or not rerun_limit.is_integer()
                or not 0 <= rerun_limit <= self.MAX_AUTO_RERUNS
                or rerun_delay is None or not 10. <= rerun_delay <= 600.):
            host._append_sidebar_control_message(
                "Invalid automatic recovery settings: choose 0–10 reruns and a 10–600 s cooldown.")
            return False
        if int(self.anchor_pose.get("taxel_index", -1)) != int(self.anchor_taxel):
            host._append_sidebar_control_message(f"The saved mapping anchor is not Taxel {self.anchor_taxel}; remember it again.")
            return False
        if self.fixed_scan_z_m is not None:
            scan_z = finite_number(self.fixed_scan_z_m)
            anchor_z = float(self.anchor_pose["tool_position_m"][2])
            if scan_z is None or scan_z > anchor_z - self.MIN_FIXED_SCAN_CLEARANCE_M:
                host._append_sidebar_control_message(
                    "The fixed scan height must be at least "
                    f"{self.MIN_FIXED_SCAN_CLEARANCE_M * 1000.0:.0f} mm below the "
                    "remembered anchor; remember it again."
                )
                return False
            if anchor_z - scan_z > self.MAX_PROBE_DESCENT_M:
                host._append_sidebar_control_message(
                    "The fixed scan height is more than 100 mm below the anchor; "
                    "remember a valid scan height before starting."
                )
                return False
        if (
            float(self.max_probe_descent_m) <= 0.0
            or float(self.probe_speed_m_s) <= 0.0
            or float(self.reference_max_descent_m) <= 0.0
            or float(self.reference_probe_speed_m_s) <= 0.0
            or float(self.probe_confirmation_speed_m_s) <= 0.0
            or float(self.probe_step_m) <= 0.0
            or float(self.probe_start_margin_m) < 0.0
            or float(self.signal_threshold_counts) <= 0.0
            or float(self.adaptive_margin_fraction) < 0.0
            or float(self.adaptive_min_margin_counts) <= 0.0
            or float(self.force_guard_g) <= 0.0
            or float(self.coarse_radius_m) <= 0.0
            or float(self.coarse_step_m) <= 0.0
            or float(self.local_radius_m) <= 0.0
            or float(self.local_step_m) <= 0.0
            or int(self.probe_confirm_frames) < 1
            or int(self.probe_sample_frames) < 3
            or int(self.probe_window_frames) < 3
            or int(self.baseline_frame_count) < 3
            or int(self.scan_frame_count) < 3
            or int(self.verify_frame_count) < 3
            or float(self.centre_scan_signal_multiplier) < 1.0
            or float(self.centre_scan_max_extra_descent_m) <= 0.0
        ):
            host._append_sidebar_control_message(
                "Mapping settings are invalid; distances, speeds, sensor-change settings, frame counts, centre-scan strength, and the force guard must be positive."
            )
            return False
        if str(self.proximity_detection_mode) not in {
            "robust_shift",
            "adaptive_range",
            "fixed",
        }:
            host._append_sidebar_control_message(
                "Mapping settings are invalid; select robust-shift, adaptive-range, or fixed-change proximity detection."
            )
            return False
        if (not np.isfinite(self.centre_scan_signal_multiplier)
                or not np.isfinite(self.centre_scan_max_extra_descent_m)
                or self.centre_scan_max_extra_descent_m > self.MAX_NONCONTACT_EXTRA_DESCENT_M):
            host._append_sidebar_control_message(
                f"Non-contact mapping needs finite scan settings and at most {self.MAX_NONCONTACT_EXTRA_DESCENT_M * 1000.:g} mm extra descent after onset. "
                "This is not a verified air gap; independently check clearance before use."
            )
            return False
        if (
            float(self.probe_speed_m_s) > self.MAX_PROBE_SPEED_M_S
            or float(self.reference_probe_speed_m_s) > self.MAX_PROBE_SPEED_M_S
            or float(self.probe_confirmation_speed_m_s)
            > self.MAX_PROBE_SPEED_M_S
        ):
            host._append_sidebar_control_message(
                "Mapping settings are invalid; probe speeds must be no more than 5 mm/s."
            )
            return False
        if (
            float(self.max_probe_descent_m) > self.MAX_PROBE_DESCENT_M
            or float(self.reference_max_descent_m) > self.MAX_PROBE_DESCENT_M
            or float(self.probe_step_m) > 0.005
        ):
            host._append_sidebar_control_message(
                "Mapping settings are invalid; probe maximum descent must be no more than 100 mm and the probe depth step no more than 5 mm."
            )
            return False
        # The TM driver truncates line speed to whole mm/s; 0 would not move.
        if float(self.line_speed_m_s) < 0.001:
            host._append_sidebar_control_message(
                "Mapping settings are invalid; the tool line speed must be at least 1 mm/s."
            )
            return False
        api = self._robot_api(host)
        current_pose = self._read_tool_pose(api)
        if current_pose is None or not hasattr(api, "send_positions_tool_position"):
            host._append_sidebar_control_message("Robot TCP feedback and position control must be available before mapping.")
            return False
        if resume_result is not None:
            anchor_position = np.asarray(
                self.anchor_pose["tool_position_m"], dtype=float
            )
            current_z_mm = float(current_pose[0][2]) * 1000.0
            anchor_z_mm = float(anchor_position[2]) * 1000.0
            if (
                resume_result.get("stop_reason") in {"possible_contact", "force_guard", "force_safety_limit"}
                and current_pose[0][2] < anchor_position[2]-self.position_tolerance_m
            ):
                host._append_sidebar_control_message(
                    "Resume refused after a force/contact stop: the TCP is at "
                    f"Z {current_z_mm:.1f} mm, {anchor_z_mm - current_z_mm:.1f} mm "
                    f"below the saved anchor height Z {anchor_z_mm:.1f} mm. "
                    "Inspect the probe, raise the robot straight up to at least "
                    f"Z {anchor_z_mm:.1f} mm, then press Resume again. "
                    "No automatic retraction or recalibration of a possibly loaded probe."
                )
                return False
            xy_offset_m = float(np.linalg.norm(current_pose[0][:2] - anchor_position[:2]))
            if xy_offset_m > float(self.max_xy_offset_m):
                host._append_sidebar_control_message(
                    "Taxel mapping cannot resume: the TCP is "
                    f"{xy_offset_m * 1000.0:.1f} mm from the saved anchor in XY "
                    f"(limit {float(self.max_xy_offset_m) * 1000.0:.0f} mm). "
                    "Move the robot back over the recorded sensor first."
                )
                return False
            current_quaternion = np.asarray(current_pose[1], dtype=float)
            saved_quaternion = np.asarray(
                self.anchor_pose["tool_quaternion_wxyz"], dtype=float
            )
            current_norm = float(np.linalg.norm(current_quaternion))
            saved_norm = float(np.linalg.norm(saved_quaternion))
            alignment = (
                0.0
                if current_norm < 1e-9 or saved_norm < 1e-9
                else abs(
                    float(
                        np.dot(
                            current_quaternion / current_norm,
                            saved_quaternion / saved_norm,
                        )
                    )
                )
            )
            if alignment < float(np.cos(np.radians(2.5))):
                tilt_deg = float(
                    np.degrees(2.0 * np.arccos(min(1.0, max(-1.0, alignment))))
                )
                host._append_sidebar_control_message(
                    "Taxel mapping cannot resume: the tool orientation is "
                    f"{tilt_deg:.1f}° away from the saved map (limit 5.0°). "
                    "Restore the saved orientation first."
                )
                return False
        frame = self._read_sensor_frame(host)
        if frame is None:
            host._append_sidebar_control_message(
                f"A live {self.sensor_rows}x{self.sensor_columns} tactile sensor is required before mapping."
            )
            return False
        now = time.monotonic()
        force = self._fresh_force(host, now)
        if force is None:
            return self._begin_force_meter_connection(host)
        pose_stamp = finite_number(getattr(api, "current_feedback_state_time", None))
        if pose_stamp is None or not -0.1 <= now - pose_stamp <= 1.0:
            host._append_sidebar_control_message("Mapping did not start: robot feedback is stale or untimestamped.")
            return False
        if abs(float(force[0])) / self.NEWTONS_PER_GRAM >= CalibrationTask.EMERGENCY_FORCE_G:
            host._append_sidebar_control_message("Mapping did not start: raw force exceeds the 900 g safety limit for the 1 kg cell.")
            return False
        host._append_sidebar_control_message(self._scan_plane_status())
        if resume_result is not None:
            try:
                self._restore_resume_state(resume_result, resume_source)
            except (KeyError, TypeError, ValueError, OSError) as exc:
                host._append_sidebar_control_message(
                    f"Taxel mapping cannot resume: the saved state is invalid ({exc})."
                )
                return False
            self._resume_request = None
            self._resume_source_path = None
            self._last_sensor_sequence = frame[0]
            self._last_sensor_advance_at = now
            self._force_baseline_n = float(force[0])
            self._mapping_run_started = True
            self._active = True
            self.save_anchor_pose()
            setter = getattr(host, "_set_taxel_mapping_controls_running", None)
            if callable(setter):
                setter(True)
            sync = getattr(host, "_sync_taxel_mapping_settings", None)
            if callable(sync):
                sync()
            host._append_sidebar_control_message(
                "Resuming Taxel Position Mapping from "
                f"{self.mapped_target_count()}/"
                f"{len(self.mapping_target_indices())} measured target "
                f"taxels. Verifying Taxel {self.anchor_taxel} before continuing."
            )
            self._publish_live_map(host)
            anchor = self.anchor_pose["tool_position_m"]
            return self._route_to(
                host,
                anchor[:2],
                anchor[2],
                "resume_anchor_arrived",
                f"resume: returning to the saved safe Taxel {self.anchor_taxel} anchor",
            )
        self._relocation_active = relocation_result is not None
        self._relocation_continues_mapping = bool(
            relocation_result is not None
            and not bool(relocation_result.get("completed"))
        )
        self._relocation_finished = False
        if self._relocation_continues_mapping:
            # The remaining targets must be the same set the partial map was
            # started with, whatever the sidebar currently shows.
            source_settings = relocation_result["settings"]
            if "skip_outer_border" in source_settings:
                self.skip_outer_border = bool(source_settings["skip_outer_border"])
        self._relocation_source_result = (
            copy.deepcopy(relocation_result)
            if relocation_result is not None
            else None
        )
        self._relocation_source_positions = {
            int(record["taxel_index"]): copy.deepcopy(record)
            for record in (relocation_result or {}).get("positions", [])
            if isinstance(record, dict) and record.get("status") == "measured"
        }
        self._relocation_source_path = relocation_source
        self._relocation_verification_queue = []
        self._relocation_reference_indices = []
        self._relocation_transform = None
        if relocation_result is not None:
            self._relocation_request = None
        self._positions = {}
        self._failures = {}
        self._sample_log = []
        self._baseline_checks = []
        self._sensor_calibration_history = []
        self._baseline_stats_by_taxel = {}
        self._attempt_counts = {}
        self._reference_attempt_count = 0
        self._reference_onset_z = None
        self._probe_start_z = None
        self._mapping_queue = []
        self._current_prediction = None
        self._neighbor_queue = []
        self._neighbor_guesses = {}
        self._predicted_positions = {}
        self._pending_recovery = None
        self._route_queue = []
        self._current_taxel = None
        self._current_context = ""
        self._current_target_xy = None
        self._far_baseline = None
        self._far_baseline_min = None
        self._far_baseline_max = None
        self._far_baseline_frames = None
        self._probe_detection_by_taxel = {}
        self._probe_recent_frames = []
        self._probe_candidate_pose = None
        self._probe_onset_pose = None
        self._probe_scan_pose = None
        self._probe_scan_signal = 0.0
        self._probe_scan_target_signal = 0.0
        self._probe_scan_target_reached = False
        self._probe_onset_confirmed = False
        self._verify_estimate = None
        self._local_recenters = 0
        self._scan_kind = ""
        self._sensor_calibration_started_at = None
        self._sensor_calibration_next_action = ""
        self._initial_row_vector_xy = None
        self._initial_column_vector_xy = None
        self._row_vector_xy = None
        self._column_vector_xy = None
        self._resume_history = []
        self._resume_anchor_previous = None
        self._resume_alignment_error = ""
        self._resume_mapped_before = 0
        self._last_sensor_sequence = frame[0]
        self._last_sensor_advance_at = now
        self._force_baseline_n = float(force[0])
        self._started_at_iso = datetime.now(timezone.utc).isoformat()
        try:
            self._prepare_output_folder()
        except Exception as exc:
            host._append_sidebar_control_message(f"Could not create the mapping output folder: {exc}")
            return False
        self._mapping_run_started = True
        self._active = True
        setter = getattr(host, "_set_taxel_mapping_controls_running", None)
        if callable(setter):
            setter(True)
        if self._relocation_active:
            continuation = (
                " It then continues mapping the remaining taxels."
                if self._relocation_continues_mapping
                else ""
            )
            host._append_sidebar_control_message(
                "Taxel map relocation started. The task will locate the new "
                "centre and neighbours, verify distributed taxels, and save a "
                "new transformed map." + continuation + " Keep the workspace "
                "clear and remain beside the emergency stop."
            )
        else:
            host._append_sidebar_control_message(
                "Taxel Position Mapping started. Keep the robot workspace clear and remain beside the emergency stop."
            )
        self._publish_live_map(host)
        anchor = self.anchor_pose["tool_position_m"]
        return self._route_to(
            host,
            anchor[:2],
            anchor[2],
            "anchor_safe_arrived",
            f"moving to the remembered safe pose above Taxel {self.anchor_taxel}"
            + (" for relocation" if self._relocation_active else ""),
        )

    def on_tick(self, host) -> None:
        if not self._active:
            return
        now = time.monotonic()
        if self._phase == "connecting_force_meter":
            self._tick_force_meter_connection(host, now)
            return
        force = self._fresh_force(host, now)
        if force is None:
            self._request_stop(host, "force_stale", "Mapping stopped: the force-transmitter reading became stale.")
            return
        pose_stamp = finite_number(getattr(self._robot_api(host), "current_feedback_state_time", None))
        if pose_stamp is None or not -0.1 <= now - pose_stamp <= 1.0:
            self._request_stop(host, "robot_feedback_stale", "Mapping stopped: robot feedback is stale.")
            return
        if abs(float(force[0])) / self.NEWTONS_PER_GRAM >= CalibrationTask.EMERGENCY_FORCE_G:
            self._request_stop(host, "force_safety_limit", "Mapping stopped: raw force reached 900 g.")
            return
        if self._force_baseline_n is not None:
            force_delta_g = abs(float(force[0]) - float(self._force_baseline_n)) / self.NEWTONS_PER_GRAM
            if force_delta_g >= float(self.force_guard_g):
                self._request_stop(
                    host,
                    "force_guard",
                    f"Mapping stopped: unexpected contact force changed by {force_delta_g:.1f} g.",
                )
                return
            if force_delta_g >= self.NONCONTACT_FORCE_STOP_G:
                self._request_stop(
                    host,
                    "possible_contact",
                    f"Non-contact mapping stopped: force changed by {force_delta_g:.1f} g "
                    f"(conservative {self.NONCONTACT_FORCE_STOP_G:.0f} g stop). "
                    "Possible contact; no automatic retraction or XY scanning. Inspect clearance before resuming.",
                )
                return
        if self._auto_rerun_pending is not None and not self._at_auto_rerun_anchor(host):
            self._request_stop(host, "auto_rerun_clearance_lost",
                               "Automatic rerun cancelled: TCP left the clearance pose during recovery. Inspect before resuming.")
            return
        if self._phase == "sensor_calibrating":
            self._tick_sensor_calibration(host, now)
            return
        frame = self._read_sensor_frame(host)
        if frame is None:
            self._request_stop(
                host,
                "sensor_lost",
                f"Mapping stopped: live {self.sensor_rows}x{self.sensor_columns} sensor data was lost.",
            )
            return
        if not self._track_sensor_freshness(frame[0], now):
            self._request_stop(host, "sensor_stale", "Mapping stopped: tactile frames became stale.")
            return
        if self._phase == "moving":
            self._tick_move(host, now)
        elif self._phase == "collecting":
            self._tick_collection(host, frame, force[0])
        elif self._phase == "probing":
            self._tick_probe(host, frame)
        elif self._phase == "auto_rerun_wait":
            self._tick_auto_rerun_wait(host, now)

    def on_stop(self, host, show_result: bool = True) -> None:
        was_active = bool(self._active)
        mapping_run_started = bool(self._mapping_run_started)
        CalibrationTask._stop_robot(self._robot_api(host))
        if was_active and not self._stop_reason:
            self._stop_reason = "manual_stop"
        self._auto_rerun_deadline = None
        self._auto_rerun_pending = None
        if self._auto_rerun_history and self._auto_rerun_history[-1].get("status") == "waiting":
            self._auto_rerun_history[-1].update(
                status="cancelled", ended_at=datetime.now(timezone.utc).isoformat(),
                stop_reason=self._stop_reason,
            )
        if was_active and self._stop_snapshot is None:
            try:
                self._stop_snapshot = self._mapping_snapshot(host)
            except Exception as exc:
                self._stop_snapshot = {"diagnostic_error": str(exc)}
        if (
            self._sensor_calibration_history
            and self._sensor_calibration_history[-1].get("status") == "running"
        ):
            self._sensor_calibration_history[-1]["status"] = "interrupted"
            self._sensor_calibration_history[-1]["stopped_at"] = (
                datetime.now(timezone.utc).isoformat()
            )
        self._active = False
        self._phase = "idle"
        self._sensor_calibration_started_at = None
        self._sensor_calibration_next_action = ""
        self._force_connection_started_at = None
        self._mapping_run_started = False
        setter = getattr(host, "_set_taxel_mapping_controls_running", None)
        if callable(setter):
            setter(False)
        self._progress(
            host,
            "Complete" if self._completed else (
                f"Stopped: {self._stop_message or self._stop_reason.replace('_', ' ') or 'reason not reported'}"
            ),
        )
        if not was_active or not mapping_run_started:
            return
        try:
            result = self._save_results()
        except Exception as exc:
            result = None
            host._append_sidebar_control_message(f"Could not save the taxel position map: {exc}")
        if result is not None and result.get("output_folder"):
            host._append_sidebar_control_message(
                f"Taxel mapping files saved together in: {result['output_folder']}"
            )
        callback = getattr(host, "_show_taxel_position_map", None)
        if result is not None and callable(callback):
            callback(result)


class Sigraph2026ScanningTask(ExperimentTask):
    """Record manually taught joint configurations and replay them in order."""

    id = "sigraph2026_scanning"
    label = "Sigraph2026 Scanning"
    category = "Archive"
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
    Sigraph2026ScanningTask(),
    TaxelPositionMappingTask(),
    CalibrationTask(),
    FullTaxelCalibrationTask(),
    CalibrationGraphAnalysisTask(),
    SensorCharacterisationTask(),
]


SIDEBAR_CATEGORIES = ("TRO2026", "Archive")
COLLAPSED_SIDEBAR_CATEGORIES = {"Archive"}


def task_definitions() -> List[tuple]:
    """Return ``(label, id, description, category)`` rows for the sidebar."""
    return [
        (t.label, t.id, t.description, str(getattr(t, "category", "") or ""))
        for t in TASKS
        if bool(getattr(t, "show_in_sidebar", True))
    ]


def get_task(task_id: Optional[str]) -> Optional[ExperimentTask]:
    """Look up a task by its id; returns None if not found."""
    if not task_id:
        return None
    for task in TASKS:
        if task.id == task_id:
            return task
    return None
