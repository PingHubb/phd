"""Embedded Unitree G1 URDF and woven-signal viewer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor

from phd.dependence.sensor_heatmap import (
    DEFAULT_HEATMAP_3D_COLOR_GAIN,
    DEFAULT_HEATMAP_3D_PALETTE,
    DEFAULT_HEATMAP_NOISE_FLOOR_PCT,
    DEFAULT_HEATMAP_RESPONSE_MODE,
    DEFAULT_HEATMAP_SATURATION_PCT,
    DEFAULT_PROXIMITY_KNEE,
    DEFAULT_PROXIMITY_NOISE_FLOOR,
    DEFAULT_PROXIMITY_SATURATION,
    HEATMAP_RESPONSE_PROXIMITY_ENHANCED,
    heatmap_3d_rgb,
)
from phd.dependence.sensor_api import ArduinoCommander
from phd.dependence.paths import resource_path
from phd.dependence.sensor_serial import SensorReadWorker
from phd.dependence.humanoid_sensor_registry import (
    HUMANOID_SENSOR_MAPPING_FILE,
    stable_usb_port_identity as _stable_usb_port_identity,
)
from phd.dependence.humanoid_signal_parts import (
    discover_signal_parts,
    load_colored_line_obj,
)
from phd.dependence.humanoid_urdf import load_urdf


DEFAULT_HUMANOID_AUTO_PORT_TARGETS = {
    "ttyACM1": ("1", "head_link"),
    "ttyACM2": ("10", "torso_link"),
    "ttyACM3": ("11", "torso_link"),
    "ttyACM5": ("3", "right_rubber_hand"),
}


def _ttyacm_port_details():
    """Return currently connected ACM ports with their USB metadata."""
    try:
        import serial.tools.list_ports
    except ImportError:
        return []
    return sorted(
        (
            port
            for port in serial.tools.list_ports.comports()
            if os.path.basename(str(port.device)).lower().startswith(
                "ttyacm"
            )
        ),
        key=lambda port: os.path.basename(str(port.device)).lower(),
    )


def _stable_identity_for_device(port_name) -> str:
    target_path = os.path.realpath(str(port_name or ""))
    for port in _ttyacm_port_details():
        if os.path.realpath(str(port.device)) == target_path:
            return _stable_usb_port_identity(port)
    return ""


def _write_humanoid_sensor_mapping_file(
    sensor_mappings,
    auto_device_assignments=None,
) -> bool:
    payload = {
        "version": 2,
        "auto_device_assignments": dict(
            auto_device_assignments or {}
        ),
        "mappings": dict(sensor_mappings or {}),
    }
    try:
        os.makedirs(
            os.path.dirname(HUMANOID_SENSOR_MAPPING_FILE),
            exist_ok=True,
        )
        with open(
            HUMANOID_SENSOR_MAPPING_FILE,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(payload, handle, indent=2)
        return True
    except OSError:
        return False


class HumanoidSensorAutoWorker(QtCore.QObject):
    """Discover and stream standalone ttyACM tactile sensors."""

    STARTUP_SETTLE_SECONDS = 1.5
    CALIBRATION_SETTLE_SECONDS = 0.2
    WARMUP_FRAME_COUNT = 3
    GUI_FRAME_TARGET_HZ = 30.0

    sensor_found = QtCore.pyqtSignal(
        str,
        str,
        int,
        int,
        int,
    )
    scan_summary = QtCore.pyqtSignal(str)
    frames_ready = QtCore.pyqtSignal(object)
    calibration_started = QtCore.pyqtSignal(int)
    calibration_finished = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal()

    def __init__(self, requirements, preferred_port_parts=None):
        super().__init__()
        self.requirements = list(requirements or [])
        self.preferred_port_parts = {
            os.path.basename(str(port_name)).lower(): str(part_key)
            for port_name, part_key in dict(
                preferred_port_parts or {}
            ).items()
        }
        self._running = False
        self._calibration_requested = False
        self._apis = []
        self._shared_reader = None
        self._stream_assignments_by_port = {}
        self._latest_frames = {}

    def stop(self):
        self._running = False
        self._calibration_requested = False
        reader = self._shared_reader
        if reader is not None:
            reader.stop()

    def request_calibration(self):
        """Queue calibration for the worker-owned serial connections."""
        self._calibration_requested = True

    def _refresh_calibration(self, assignment):
        if not self._running:
            return False
        api = assignment["api"]
        active_count = int(assignment["active_count"])
        try:
            command_calibration = api.update_cal()
            if not self._wait_while_running(
                self.CALIBRATION_SETTLE_SECONDS
            ):
                return False
            fresh_calibration = api.read_cal()
        except Exception:
            return False

        calibration = None
        for candidate in (
            fresh_calibration,
            command_calibration,
        ):
            try:
                if (
                    candidate is not None
                    and len(candidate) >= active_count
                    and self._frame_is_plausible(candidate)
                ):
                    calibration = list(candidate[:active_count])
                    break
            except (TypeError, ValueError):
                continue
        if calibration is None:
            return False
        assignment["calibration"] = calibration
        return True

    def _refresh_calibrations(self, assignments):
        updated = 0
        failed = 0
        for assignment in assignments:
            if not self._running:
                break
            if self._refresh_calibration(assignment):
                updated += 1
            else:
                failed += 1
        return updated, failed

    def _on_shared_reader_payload(
        self,
        _generation,
        port_name,
        values,
    ):
        assignment = self._stream_assignments_by_port.get(
            str(port_name)
        )
        if assignment is None:
            return
        active_count = int(assignment["active_count"])
        if values is None or len(values) < active_count:
            return
        self._latest_frames[assignment["part"]] = (
            assignment["port"],
            assignment["part"],
            list(values[:active_count]),
            assignment["calibration"],
        )

    def _wait_while_running(self, seconds):
        deadline = time.perf_counter() + max(0.0, float(seconds))
        while self._running and time.perf_counter() < deadline:
            time.sleep(
                min(
                    0.05,
                    max(0.0, deadline - time.perf_counter()),
                )
            )
        return self._running

    @staticmethod
    def _clear_serial_input(api):
        serial_port = getattr(api, "ser", None)
        reset = getattr(serial_port, "reset_input_buffer", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass

    def _drain_serial_until_quiet(
        self,
        api,
        *,
        quiet_seconds=0.12,
        max_wait_seconds=2.5,
    ):
        serial_port = getattr(api, "ser", None)
        if serial_port is None:
            return
        deadline = time.perf_counter() + max(
            0.0, float(max_wait_seconds)
        )
        quiet_deadline = time.perf_counter() + max(
            0.01, float(quiet_seconds)
        )
        while self._running and time.perf_counter() < deadline:
            try:
                waiting = int(getattr(serial_port, "in_waiting", 0))
            except Exception:
                return
            if waiting > 0:
                self._clear_serial_input(api)
                quiet_deadline = time.perf_counter() + max(
                    0.01, float(quiet_seconds)
                )
            elif time.perf_counter() >= quiet_deadline:
                return
            time.sleep(0.01)

    @staticmethod
    def _frame_is_plausible(values):
        if not values:
            return False
        try:
            array = np.asarray(values, dtype=np.int64).reshape(-1)
        except (TypeError, ValueError):
            return False
        if array.size == 0 or np.any(array < 0) or np.any(array > 65535):
            return False
        if np.all(array == 0) or np.all(array == 999) or np.all(array == 65535):
            return False
        return True

    @staticmethod
    def match_payload(payload_length, requirements):
        """Return part, active count, and trailing extra-row count."""
        payload_length = max(0, int(payload_length))
        for part_key, required in requirements:
            required = int(required)
            if payload_length == required:
                return part_key, required, 0
        for part_key, required in requirements:
            required = int(required)
            extra = payload_length - required
            if (
                2 <= extra <= 100
                and required % extra == 0
            ):
                return part_key, required, extra
        return None

    @staticmethod
    def _candidate_ports():
        return [port.device for port in _ttyacm_port_details()]

    def _candidate_port_targets(self):
        candidates = self._candidate_ports()
        if not self.preferred_port_parts:
            return [(port_name, None) for port_name in candidates]
        available = {
            os.path.basename(str(port_name)).lower(): port_name
            for port_name in candidates
        }
        return [
            (available[port_basename], part_key)
            for port_basename, part_key
            in self.preferred_port_parts.items()
            if port_basename in available
        ]

    @staticmethod
    def _read_frame(api, attempts=2):
        for _ in range(max(1, int(attempts))):
            values = api.read_raw()
            if values:
                return list(values)
        return None

    def _read_stable_frame(
        self,
        api,
        *,
        expected_lengths=None,
        frame_count=None,
        attempts=10,
    ):
        expected_lengths = {
            int(value) for value in (expected_lengths or [])
        }
        frame_count = max(
            1,
            int(
                self.WARMUP_FRAME_COUNT
                if frame_count is None
                else frame_count
            ),
        )
        stable_length = None
        stable_count = 0
        latest = None
        for _ in range(max(frame_count, int(attempts))):
            if not self._running:
                return None
            values = self._read_frame(api, attempts=1)
            if not self._frame_is_plausible(values):
                self._drain_serial_until_quiet(api)
                stable_length = None
                stable_count = 0
                continue
            current_length = len(values)
            if (
                expected_lengths
                and current_length not in expected_lengths
            ):
                self._drain_serial_until_quiet(api)
                stable_length = None
                stable_count = 0
                continue
            if current_length != stable_length:
                stable_length = current_length
                stable_count = 1
            else:
                stable_count += 1
            latest = list(values)
            if stable_count >= frame_count:
                return latest
        return None

    @staticmethod
    def _read_channel_shape(api, attempts=3):
        for _ in range(max(1, int(attempts))):
            try:
                values = list(api.channel_check() or [])
            except Exception:
                continue
            positive = [
                int(value) for value in values
                if 0 < int(value) <= 100
            ]
            if len(positive) >= 2:
                return positive[-2], positive[-1]
        return None

    @QtCore.pyqtSlot()
    def run(self):
        self._running = True
        remaining = list(self.requirements)
        assignments = []
        skipped = []
        try:
            port_targets = self._candidate_port_targets()
            if self.preferred_port_parts:
                available_names = {
                    os.path.basename(str(port_name)).lower()
                    for port_name, _part_key in port_targets
                }
                skipped.extend(
                    f"{port_name} not connected"
                    for port_name in self.preferred_port_parts
                    if port_name not in available_names
                )

            opened_ports = []
            for port_name, preferred_part in port_targets:
                if not self._running:
                    break
                api = ArduinoCommander(
                    serial_port=port_name,
                    connect_immediately=True,
                )
                if not api.is_connected():
                    skipped.append(
                        f"{os.path.basename(port_name)} busy"
                    )
                    api.close()
                    continue
                opened_ports.append(
                    (port_name, preferred_part, api)
                )
                self._apis.append(api)

            if opened_ports and not self._wait_while_running(
                self.STARTUP_SETTLE_SECONDS
            ):
                opened_ports = []

            for port_name, preferred_part, api in opened_ports:
                if not self._running or not remaining:
                    break
                preferred_requirement = next(
                    (
                        (part_key, int(required))
                        for part_key, required in remaining
                        if part_key == preferred_part
                    ),
                    None,
                )
                if (
                    preferred_part is not None
                    and preferred_requirement is None
                ):
                    api.close()
                    continue
                self._drain_serial_until_quiet(api)
                shape = self._read_channel_shape(api)
                expected_lengths = set()
                if shape is not None:
                    expected_active = int(shape[0]) * int(shape[1])
                    expected_target = (
                        preferred_requirement[1]
                        if preferred_requirement is not None
                        else None
                    )
                    if (
                        expected_target is not None
                        and expected_active != expected_target
                    ):
                        skipped.append(
                            f"{os.path.basename(port_name)} "
                            f"{shape[0]}x{shape[1]} does not match "
                            f"{preferred_part}"
                        )
                        api.close()
                        continue
                    if (
                        expected_target is None
                        and not any(
                            int(required) == expected_active
                            for _part_key, required in remaining
                        )
                    ):
                        skipped.append(
                            f"{os.path.basename(port_name)} "
                            f"{shape[0]}x{shape[1]} outside target parts"
                        )
                        api.close()
                        continue
                    expected_lengths = {
                        expected_active,
                        expected_active + int(shape[0]),
                    }
                    api.expected_payload_values = max(expected_lengths)
                raw = self._read_stable_frame(
                    api,
                    expected_lengths=expected_lengths,
                    frame_count=(
                        1 if preferred_part is not None else None
                    ),
                )
                if not raw:
                    skipped.append(f"{os.path.basename(port_name)} no data")
                    api.close()
                    continue
                match = None
                drive_count = 0
                sensor_count = 0
                extra_rows = 0
                if shape is not None:
                    drive_count, sensor_count = shape
                    active_count = drive_count * sensor_count
                    exact_part = (
                        preferred_part
                        if preferred_part is not None
                        else next(
                            (
                                part_key
                                for part_key, required in remaining
                                if int(required) == active_count
                            ),
                            None,
                        )
                    )
                    if (
                        exact_part is not None
                        and len(raw) >= active_count
                        and len(raw) - active_count
                        in (0, drive_count)
                    ):
                        extra_rows = len(raw) - active_count
                        match = (
                            exact_part,
                            active_count,
                            extra_rows,
                        )
                elif preferred_requirement is not None:
                    part_key, active_count = preferred_requirement
                    possible_extra = len(raw) - active_count
                    if (
                        possible_extra == 0
                        or (
                            2 <= possible_extra <= 100
                            and active_count % possible_extra == 0
                        )
                    ):
                        extra_rows = possible_extra
                        match = (
                            part_key,
                            active_count,
                            extra_rows,
                        )
                if match is None and preferred_part is None:
                    match = self.match_payload(len(raw), remaining)
                if match is None:
                    skipped.append(
                        f"{os.path.basename(port_name)} "
                        f"{len(raw)} values unmatched"
                    )
                    api.close()
                    continue
                part_key, active_count, extra_rows = match
                if drive_count <= 0 or sensor_count <= 0:
                    if extra_rows > 0:
                        drive_count = int(extra_rows)
                        sensor_count = int(active_count // extra_rows)
                api.expected_payload_values = len(raw)
                calibration = api.update_cal()
                if not self._wait_while_running(
                    self.CALIBRATION_SETTLE_SECONDS
                ):
                    api.close()
                    break
                fresh_calibration = api.read_cal()
                if (
                    fresh_calibration is not None
                    and len(fresh_calibration) >= active_count
                    and self._frame_is_plausible(fresh_calibration)
                ):
                    calibration = fresh_calibration
                if (
                    calibration is None
                    or len(calibration) < active_count
                    or not self._frame_is_plausible(calibration)
                ):
                    calibration = list(raw)
                post_calibration_frame = self._read_stable_frame(
                    api,
                    expected_lengths={len(raw)},
                    frame_count=(
                        1 if preferred_part is not None else None
                    ),
                )
                if post_calibration_frame is None:
                    skipped.append(
                        f"{os.path.basename(port_name)} unstable after calibration"
                    )
                    api.close()
                    continue
                assignment = {
                    "api": api,
                    "port": str(port_name),
                    "part": str(part_key),
                    "active_count": int(active_count),
                    "extra_rows": int(extra_rows),
                    "drive_count": int(drive_count),
                    "sensor_count": int(sensor_count),
                    "calibration": list(calibration[:active_count]),
                }
                assignments.append(assignment)
                remaining = [
                    item for item in remaining
                    if item[0] != part_key
                ]
                self.sensor_found.emit(
                    str(port_name),
                    str(part_key),
                    int(drive_count),
                    int(sensor_count),
                    int(extra_rows),
                )

            details = [
                f"{len(assignments)} matched",
                f"{len(remaining)} humanoid parts without sensors",
            ]
            if skipped:
                details.append("skipped: " + ", ".join(skipped))
            self.scan_summary.emit("; ".join(details))

            self._latest_frames.clear()
            self._stream_assignments_by_port = {
                str(assignment["port"]): assignment
                for assignment in assignments
            }
            expected_by_port = {
                str(assignment["port"]): (
                    int(assignment["active_count"])
                    + int(assignment["extra_rows"])
                )
                for assignment in assignments
            }
            reader = SensorReadWorker(
                [
                    assignment["api"].ser
                    for assignment in assignments
                ],
                expected_payload_values=max(
                    expected_by_port.values(),
                    default=0,
                ),
                expected_payload_values_by_port=expected_by_port,
            )
            reader.raw_payload_ready.connect(
                self._on_shared_reader_payload
            )
            reader._running = True
            self._shared_reader = reader
            last_frame_emit = 0.0
            while self._running and assignments:
                if self._calibration_requested:
                    self._calibration_requested = False
                    reader._running = False
                    self._latest_frames.clear()
                    self.calibration_started.emit(len(assignments))
                    updated, failed = self._refresh_calibrations(
                        assignments
                    )
                    self.calibration_finished.emit(updated, failed)
                    reader._running = self._running
                    continue

                emitted = reader._read_cycle()
                now = time.perf_counter()
                if (
                    last_frame_emit <= 0.0
                    or now - last_frame_emit
                    >= 1.0 / self.GUI_FRAME_TARGET_HZ
                ):
                    frames = list(self._latest_frames.values())
                    self._latest_frames.clear()
                    if frames:
                        self.frames_ready.emit(frames)
                        last_frame_emit = now
                if not emitted:
                    self._wait_while_running(reader.idle_sleep_sec)
        finally:
            self._running = False
            reader = self._shared_reader
            if reader is not None:
                reader.stop()
            self._shared_reader = None
            self._stream_assignments_by_port = {}
            self._latest_frames.clear()
            for api in self._apis:
                try:
                    api.close()
                except Exception:
                    pass
            self._apis = []
            self.finished.emit()


class HumanoidViewerWidget(QtWidgets.QWidget):
    """Reusable in-application version of ``view_g1.G1Viewer``."""

    @staticmethod
    def _is_fast_auto_detect_link(link_name) -> bool:
        link_name = str(link_name or "").strip().lower()
        return (
            link_name in {"head_link", "torso_link"}
            or "hand" in link_name
        )

    def _preferred_auto_port_parts(self) -> dict:
        available_keys = set(self._signal_keys)
        preferred = {}
        saved_assignments = dict(
            getattr(self, "_saved_auto_device_assignments", {})
        )
        bound_parts = {
            str(assignment.get("part", ""))
            for assignment in saved_assignments.values()
            if isinstance(assignment, dict)
            and str(assignment.get("part", "")) in available_keys
        }
        port_details = _ttyacm_port_details()
        details_by_basename = {
            os.path.basename(str(port.device)).lower(): port
            for port in port_details
        }

        for port in port_details:
            identity = _stable_usb_port_identity(port)
            assignment = saved_assignments.get(identity, {})
            part_key = str(
                assignment.get("part", "")
                if isinstance(assignment, dict)
                else ""
            )
            if identity and part_key in available_keys:
                preferred[
                    os.path.basename(str(port.device)).lower()
                ] = part_key

        assignments_changed = False
        for port_name, (part_number, link_name) in (
            DEFAULT_HUMANOID_AUTO_PORT_TARGETS.items()
        ):
            part_key = f"signal:{part_number}:{link_name}"
            port_basename = os.path.basename(port_name).lower()
            if (
                part_key not in available_keys
                or part_key in bound_parts
                or port_basename in preferred
            ):
                continue
            preferred[port_basename] = part_key

            port = details_by_basename.get(port_basename)
            identity = (
                _stable_usb_port_identity(port)
                if port is not None
                else ""
            )
            if identity and identity not in saved_assignments:
                saved_assignments[identity] = {
                    "part": part_key,
                    "serial_number": str(
                        getattr(port, "serial_number", "") or ""
                    ),
                    "vid": getattr(port, "vid", None),
                    "pid": getattr(port, "pid", None),
                    "last_port": os.path.basename(str(port.device)),
                }
                bound_parts.add(part_key)
                assignments_changed = True

        if assignments_changed:
            self._saved_auto_device_assignments = saved_assignments
            save_mapping = getattr(
                self,
                "_save_humanoid_sensor_mapping_file",
                None,
            )
            if callable(save_mapping):
                save_mapping()
        return preferred

    def __init__(
        self,
        parent=None,
        plotter=None,
        sensor_functions=None,
        before_scene_load: Callable[[], None] | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("humanoidViewer")
        self.plotter = plotter
        self._owns_plotter = plotter is None
        self._before_scene_load = before_scene_load
        self.sensor_functions = sensor_functions
        self.root = Path(resource_path("humanoid", "g1"))
        self._discover_signal_parts = discover_signal_parts
        self._load_colored_line_obj = load_colored_line_obj
        self._load_urdf = load_urdf

        self.default_urdf = self.root / "g1_custom_collision_29dof.urdf"
        self.default_signal_dir = self.root / "signals"

        self.model: Any | None = None
        self._actors: dict[str, Any] = {}
        self._world_meshes: dict[str, Any] = {}
        self._local_points: dict[str, np.ndarray] = {}
        self._local_normals: dict[str, np.ndarray] = {}
        self._item_link: dict[str, str] = {}
        self._robot_keys: list[str] = []
        self._signal_keys: list[str] = []
        self._signal_channel_indices: dict[str, np.ndarray] = {}
        self._signal_transposed_channel_indices: dict[
            str, np.ndarray
        ] = {}
        self._signal_required_channels: dict[str, int] = {}
        self._signal_expected_shapes: dict[
            str, tuple[int, int]
        ] = {}
        self._signal_static_colors: dict[str, np.ndarray] = {}
        self._signal_display_names: dict[str, str] = {}
        self._replaced_links: set[str] = set()
        self._sliders: dict[str, QtWidgets.QSlider] = {}
        self._updating = False
        self._loaded_once = False
        self._scene_active = False
        self._scene_suspended = False
        self._status_before_suspend = None
        self._closed = False
        self._live_frame_signal = None
        self._live_mapping: tuple[str, str] | None = None
        self._pending_live_frames: dict[
            str, tuple[np.ndarray, np.ndarray]
        ] = {}
        self._auto_sensor_thread = None
        self._auto_sensor_worker = None
        self._resume_auto_live_after_restore = False
        self._auto_restart_pending = False
        self._auto_live_parts: set[str] = set()
        self._auto_mapping_metadata: dict[str, dict] = {}
        self._pending_auto_frames: dict[
            str, tuple[np.ndarray, np.ndarray]
        ] = {}
        self._transformed_channel_index_cache: dict[
            tuple, np.ndarray
        ] = {}
        self._saved_sensor_mappings = (
            self._load_saved_sensor_mappings()
        )
        self._saved_auto_device_assignments = (
            self._load_saved_auto_device_assignments()
        )
        self._live_render_timer = QtCore.QTimer(self)
        self._live_render_timer.setInterval(100)
        self._live_render_timer.timeout.connect(
            self._render_pending_live_heatmap
        )

        self._build_ui()
        if self._owns_plotter:
            self._setup_studio()

    @staticmethod
    def _load_saved_sensor_mappings() -> dict:
        try:
            with open(
                HUMANOID_SENSOR_MAPPING_FILE,
                "r",
                encoding="utf-8",
            ) as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return {}
        mappings = (
            payload.get("mappings", {})
            if isinstance(payload, dict)
            else {}
        )
        return dict(mappings) if isinstance(mappings, dict) else {}

    @staticmethod
    def _load_saved_auto_device_assignments() -> dict:
        try:
            with open(
                HUMANOID_SENSOR_MAPPING_FILE,
                "r",
                encoding="utf-8",
            ) as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return {}
        assignments = (
            payload.get("auto_device_assignments", {})
            if isinstance(payload, dict)
            else {}
        )
        return (
            dict(assignments)
            if isinstance(assignments, dict)
            else {}
        )

    @staticmethod
    def _mapping_setting_key(
        part_key,
        port_name,
        drive_count,
        sensor_count,
        device_identity="",
    ) -> str:
        port_key = str(device_identity or "").strip()
        if not port_key:
            port_key = os.path.basename(str(port_name))
        return (
            f"{part_key}|{port_key}|"
            f"{int(drive_count)}x{int(sensor_count)}"
        )

    def _save_humanoid_sensor_mapping_file(self) -> bool:
        return _write_humanoid_sensor_mapping_file(
            self._saved_sensor_mappings,
            self._saved_auto_device_assignments,
        )

    def _save_sensor_mapping_setting(self, metadata) -> bool:
        key = self._mapping_setting_key(
            metadata["part"],
            metadata["port"],
            metadata["drive_count"],
            metadata["sensor_count"],
            metadata.get("device_identity", ""),
        )
        self._saved_sensor_mappings[key] = {
            "swap_drive_sensor": bool(
                metadata.get("swap_drive_sensor", False)
            ),
            "transpose_point_mapping": bool(
                metadata.get("transpose_point_mapping", False)
            ),
            "flip_horizontal_mapping": bool(
                metadata.get("flip_horizontal_mapping", False)
            ),
            "flip_vertical_mapping": bool(
                metadata.get("flip_vertical_mapping", False)
            ),
        }
        return _write_humanoid_sensor_mapping_file(
            self._saved_sensor_mappings,
            getattr(self, "_saved_auto_device_assignments", {}),
        )

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        splitter = None
        if self._owns_plotter:
            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
            layout.addWidget(splitter)
            controls = QtWidgets.QWidget(splitter)
            controls.setMinimumWidth(300)
            controls.setMaximumWidth(410)
        else:
            controls = QtWidgets.QWidget(self)
            layout.addWidget(controls)
        control_layout = QtWidgets.QVBoxLayout(controls)
        control_layout.setContentsMargins(
            0,
            0,
            8 if self._owns_plotter else 0,
            0,
        )
        control_layout.setSpacing(8)

        source_group = QtWidgets.QGroupBox("Scene Source")
        source_grid = QtWidgets.QGridLayout(source_group)
        source_grid.setContentsMargins(10, 10, 10, 10)
        source_grid.setHorizontalSpacing(8)
        source_grid.setVerticalSpacing(8)
        source_grid.setColumnStretch(1, 1)

        source_grid.addWidget(QtWidgets.QLabel("Robot model"), 0, 0)
        self.path_edit = QtWidgets.QLineEdit(str(self.default_urdf))
        self.path_edit.setToolTip(str(self.default_urdf))
        self.path_edit.setClearButtonEnabled(True)
        source_grid.addWidget(self.path_edit, 0, 1)
        browse_urdf = QtWidgets.QToolButton()
        browse_urdf.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        browse_urdf.setToolTip("Choose robot URDF")
        browse_urdf.setFixedSize(32, 32)
        browse_urdf.clicked.connect(self._browse_urdf)
        source_grid.addWidget(browse_urdf, 0, 2)

        source_grid.addWidget(QtWidgets.QLabel("Sensor signals"), 1, 0)
        self.signal_dir_edit = QtWidgets.QLineEdit(str(self.default_signal_dir))
        self.signal_dir_edit.setToolTip(
            "Directory containing N_part.obj and N/curves_col_signal.obj"
        )
        self.signal_dir_edit.setClearButtonEnabled(True)
        source_grid.addWidget(self.signal_dir_edit, 1, 1)
        browse_signal = QtWidgets.QToolButton()
        browse_signal.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon)
        )
        browse_signal.setToolTip("Choose sensor signal directory")
        browse_signal.setFixedSize(32, 32)
        browse_signal.clicked.connect(self._browse_signal_dir)
        source_grid.addWidget(browse_signal, 1, 2)
        control_layout.addWidget(source_group)

        self.load_button = QtWidgets.QPushButton("Load URDF and Sensor Signals")
        self.load_button.setObjectName("btnRunRecord")
        self.load_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        )
        self.load_button.setMinimumHeight(34)
        self.load_button.clicked.connect(self.reload)
        control_layout.addWidget(self.load_button)

        self.humanoid_control_tabs = QtWidgets.QTabWidget()
        self.sensor_controls_page = QtWidgets.QWidget()
        sensor_controls_layout = QtWidgets.QVBoxLayout(
            self.sensor_controls_page
        )
        sensor_controls_layout.setContentsMargins(4, 4, 4, 4)
        sensor_controls_layout.setSpacing(8)
        self.joint_controls_page = QtWidgets.QWidget()
        joint_controls_layout = QtWidgets.QVBoxLayout(
            self.joint_controls_page
        )
        joint_controls_layout.setContentsMargins(4, 4, 4, 4)
        joint_controls_layout.setSpacing(8)
        self.humanoid_control_tabs.addTab(
            self.sensor_controls_page,
            "Sensor Signals",
        )
        self.humanoid_control_tabs.addTab(
            self.joint_controls_page,
            "Joint Pose",
        )

        self.view_group = QtWidgets.QGroupBox("Visualization")
        mode_row = QtWidgets.QHBoxLayout(self.view_group)
        mode_row.setContentsMargins(10, 10, 10, 10)
        self.display_button_group = QtWidgets.QButtonGroup(self)
        self.display_button_group.setExclusive(True)
        self.mode_robot = QtWidgets.QPushButton("Robot")
        self.mode_signal = QtWidgets.QPushButton("Sensor Signals")
        self.mode_robot.setCheckable(True)
        self.mode_signal.setCheckable(True)
        self.mode_robot.setChecked(True)
        self.mode_robot.toggled.connect(self._apply_visibility)
        self.display_button_group.addButton(self.mode_robot)
        self.display_button_group.addButton(self.mode_signal)
        mode_row.addWidget(self.mode_robot, 1)
        mode_row.addWidget(self.mode_signal, 1)
        sensor_controls_layout.addWidget(self.view_group)

        self.live_group = QtWidgets.QGroupBox("Live Sensor Heatmap")
        live_grid = QtWidgets.QGridLayout(self.live_group)
        live_grid.setContentsMargins(10, 10, 10, 10)
        live_grid.setHorizontalSpacing(8)
        live_grid.setVerticalSpacing(8)
        live_grid.addWidget(QtWidgets.QLabel("Humanoid part"), 0, 0)
        self.live_part_combo = QtWidgets.QComboBox()
        self.live_part_combo.setToolTip(
            "Signal-mesh section that will receive the live sensor colours."
        )
        self.live_part_combo.currentIndexChanged.connect(
            self._on_live_part_changed
        )
        live_grid.addWidget(self.live_part_combo, 0, 1)
        live_grid.addWidget(QtWidgets.QLabel("Sensor port"), 1, 0)
        self.live_port_combo = QtWidgets.QComboBox()
        self.live_port_combo.setToolTip(
            "Active Sensor-tab port whose calibrated matrix drives this part."
        )
        self.live_port_combo.currentIndexChanged.connect(
            self._on_live_part_changed
        )
        live_grid.addWidget(self.live_port_combo, 1, 1)
        self.live_button = QtWidgets.QPushButton(
            "Use Sensor Tab Live Heatmap"
        )
        self.live_button.setCheckable(True)
        self.live_button.setObjectName("btnRunRecord")
        self.live_button.setToolTip(
            "Map the selected live sensor to the selected humanoid signal part."
        )
        self.live_button.toggled.connect(
            self._on_live_heatmap_toggled
        )
        live_grid.addWidget(self.live_button, 2, 0, 1, 2)
        self.auto_live_button = QtWidgets.QPushButton(
            "Auto-Detect Independent Sensors"
        )
        self.auto_live_button.setCheckable(True)
        self.auto_live_button.setToolTip(
            "Match each previously assigned physical USB sensor to its "
            "humanoid part, even if its ttyACM port number changes."
        )
        self.auto_live_button.toggled.connect(
            self._on_auto_live_toggled
        )
        live_grid.addWidget(
            self.auto_live_button,
            3,
            0,
            1,
            2,
        )
        live_grid.addWidget(
            QtWidgets.QLabel("Detected mapping"),
            4,
            0,
        )
        self.auto_mapping_combo = QtWidgets.QComboBox()
        self.auto_mapping_combo.setEnabled(False)
        self.auto_mapping_combo.currentIndexChanged.connect(
            self._on_auto_mapping_selected
        )
        live_grid.addWidget(self.auto_mapping_combo, 4, 1)
        self.swap_drive_sensor_checkbox = QtWidgets.QCheckBox(
            "Swap Drive ↔ Sensor axes"
        )
        self.swap_drive_sensor_checkbox.setEnabled(False)
        self.swap_drive_sensor_checkbox.setToolTip(
            "Transpose this matched sensor before applying channel IDs to "
            "the humanoid mesh. Saved for this part, port, and grid size."
        )
        self.swap_drive_sensor_checkbox.toggled.connect(
            self._on_swap_drive_sensor_toggled
        )
        live_grid.addWidget(
            self.swap_drive_sensor_checkbox,
            5,
            0,
            1,
            2,
        )
        self.transpose_point_mapping_checkbox = QtWidgets.QCheckBox(
            "Transpose Humanoid Point Mapping (Row ↔ Column)"
        )
        self.transpose_point_mapping_checkbox.setEnabled(False)
        self.transpose_point_mapping_checkbox.setToolTip(
            "Keep the raw sensor order unchanged, but exchange how humanoid "
            "mesh bands and regions map to 2D rows and columns. Saved for "
            "this part, port, and grid size."
        )
        self.transpose_point_mapping_checkbox.toggled.connect(
            self._on_transpose_point_mapping_toggled
        )
        live_grid.addWidget(
            self.transpose_point_mapping_checkbox,
            6,
            0,
            1,
            2,
        )
        self.flip_horizontal_mapping_checkbox = QtWidgets.QCheckBox(
            "Flip Humanoid Mapping Left ↔ Right"
        )
        self.flip_horizontal_mapping_checkbox.setEnabled(False)
        self.flip_horizontal_mapping_checkbox.setToolTip(
            "Mirror the mapped heatmap horizontally without changing the "
            "raw sensor data. Saved for this mapping."
        )
        self.flip_horizontal_mapping_checkbox.toggled.connect(
            self._on_flip_horizontal_mapping_toggled
        )
        live_grid.addWidget(
            self.flip_horizontal_mapping_checkbox,
            7,
            0,
            1,
            2,
        )
        self.flip_vertical_mapping_checkbox = QtWidgets.QCheckBox(
            "Flip Humanoid Mapping Top ↔ Bottom"
        )
        self.flip_vertical_mapping_checkbox.setEnabled(False)
        self.flip_vertical_mapping_checkbox.setToolTip(
            "Mirror the mapped heatmap vertically without changing the "
            "raw sensor data. Saved for this mapping."
        )
        self.flip_vertical_mapping_checkbox.toggled.connect(
            self._on_flip_vertical_mapping_toggled
        )
        live_grid.addWidget(
            self.flip_vertical_mapping_checkbox,
            8,
            0,
            1,
            2,
        )
        self.auto_live_status = QtWidgets.QLabel(
            "Independent sensors: off"
        )
        self.auto_live_status.setWordWrap(True)
        self.auto_live_status.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        live_grid.addWidget(
            self.auto_live_status,
            9,
            0,
            1,
            2,
        )
        live_grid.setColumnStretch(1, 1)
        sensor_controls_layout.addWidget(self.live_group)
        sensor_controls_layout.addStretch(1)

        self.status = QtWidgets.QLabel("Ready to load")
        self.status.setObjectName("humanoidStatus")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.status.setContentsMargins(4, 0, 4, 0)
        control_layout.addWidget(self.status)

        self.joint_group = QtWidgets.QGroupBox("Joint Pose")
        joint_layout = QtWidgets.QVBoxLayout(self.joint_group)
        joint_layout.setContentsMargins(10, 10, 10, 10)
        joint_layout.setSpacing(6)
        self.joint_summary = QtWidgets.QLabel("No joints loaded")
        joint_layout.addWidget(self.joint_summary)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.slider_host = QtWidgets.QWidget()
        self.slider_layout = QtWidgets.QVBoxLayout(self.slider_host)
        self.slider_layout.setContentsMargins(0, 0, 0, 0)
        self.slider_layout.setSpacing(4)
        scroll.setWidget(self.slider_host)
        joint_layout.addWidget(scroll, 1)

        self.reset_button = QtWidgets.QPushButton("Reset Joint Pose")
        self.reset_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
        )
        self.reset_button.clicked.connect(self._reset_joints)
        joint_layout.addWidget(self.reset_button)
        joint_controls_layout.addWidget(self.joint_group, 1)
        control_layout.addWidget(self.humanoid_control_tabs, 1)
        self._set_scene_controls_enabled(False)

        if splitter is not None:
            splitter.addWidget(controls)
            viewport = QtWidgets.QFrame(splitter)
            viewport_layout = QtWidgets.QVBoxLayout(viewport)
            viewport_layout.setContentsMargins(0, 0, 0, 0)
            self.plotter = QtInteractor(viewport)
            viewport_layout.addWidget(self.plotter.interactor)
            splitter.addWidget(viewport)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([350, 1000])

    def _set_scene_controls_enabled(self, enabled: bool) -> None:
        self.view_group.setEnabled(bool(enabled))
        self.joint_group.setEnabled(bool(enabled))
        self.live_group.setEnabled(bool(enabled))

    def load_default_scene(self) -> None:
        if self._loaded_once or self._closed:
            return
        self._loaded_once = True
        if self.default_urdf.is_file():
            self.reload()
        else:
            self.status.setText(f"Default URDF not found: {self.default_urdf}")

    def _setup_studio(self) -> None:
        plotter = self.plotter
        plotter.set_background("#3c4048", top="#8a909c")
        try:
            plotter.enable_anti_aliasing("ssaa")
        except Exception:
            try:
                plotter.enable_anti_aliasing("fxaa")
            except Exception:
                pass
        plotter.remove_all_lights()
        plotter.enable_3_lights()
        plotter.add_light(
            pv.Light(
                position=(2.6, -3.2, 3.8),
                focal_point=(0.04, 0.0, -0.1),
                color="#ffe7c2",
                intensity=1.6,
            )
        )
        plotter.add_light(
            pv.Light(
                position=(-2.8, 0.4, 1.8),
                focal_point=(0.04, 0.0, 0.0),
                color="#9db6d4",
                intensity=0.55,
            )
        )
        plotter.add_light(
            pv.Light(
                position=(0.2, 3.4, 1.2),
                focal_point=(0.04, 0.0, 0.2),
                color="#fff1dc",
                intensity=0.7,
            )
        )
        try:
            plotter.enable_ssao(radius=0.05, bias=0.004)
        except Exception:
            pass
        plotter.add_axes()

    @staticmethod
    def _phong(rgba: tuple[float, float, float, float]) -> dict[str, float]:
        luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        if luminance < 0.35:
            return {
                "ambient": 0.08,
                "diffuse": 0.55,
                "specular": 0.95,
                "specular_power": 80,
            }
        return {
            "ambient": 0.14,
            "diffuse": 0.70,
            "specular": 0.55,
            "specular_power": 28,
        }

    def _browse_urdf(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Humanoid URDF",
            str(Path(self.path_edit.text()).expanduser().parent),
            "URDF (*.urdf);;All files (*.*)",
        )
        if path:
            self.path_edit.setText(path)

    def _browse_signal_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Sensor Signal Directory",
            self.signal_dir_edit.text().strip() or str(self.default_signal_dir),
        )
        if path:
            self.signal_dir_edit.setText(path)

    def reload(self) -> None:
        path = Path(self.path_edit.text().strip()).expanduser()
        if not path.is_file():
            QtWidgets.QMessageBox.warning(self, "URDF not found", str(path))
            return
        try:
            model = self._load_urdf(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "URDF loading failed", str(exc))
            return

        if self._before_scene_load is not None:
            try:
                self._before_scene_load()
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Humanoid viewport unavailable",
                    str(exc),
                )
                return

        self.model = model
        self._rebuild_sliders()
        self._rebuild_scene(reset_camera=True)

    def _clear_sliders(self) -> None:
        while self.slider_layout.count():
            item = self.slider_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._sliders.clear()

    def _rebuild_sliders(self) -> None:
        if self.model is None:
            return
        self._clear_sliders()
        for joint in self.model.movable_joints:
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            display_name = (
                joint.name.replace("_joint", "").replace("_", " ").title()
            )
            name_label = QtWidgets.QLabel(display_name)
            name_label.setMinimumWidth(135)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            lower = int(round(np.degrees(joint.lower)))
            upper = int(round(np.degrees(joint.upper)))
            if upper <= lower:
                lower, upper = -180, 180
            slider.setRange(lower, upper)
            slider.setValue(0)
            slider.setToolTip(
                f"{lower} to {upper} degrees"
            )
            value_label = QtWidgets.QLabel("0 deg")
            value_label.setMinimumWidth(48)
            value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            slider.valueChanged.connect(
                lambda degrees, name=joint.name, label=value_label: (
                    self._on_slider(name, degrees, label)
                )
            )
            row_layout.addWidget(name_label)
            row_layout.addWidget(slider, 1)
            row_layout.addWidget(value_label)
            self.slider_layout.addWidget(row)
            self._sliders[joint.name] = slider
        self.slider_layout.addStretch(1)
        self.joint_summary.setText(
            f"{len(self._sliders)} controllable joints"
        )

    def _on_slider(self, _name: str, degrees: int, label) -> None:
        label.setText(f"{degrees} deg")
        if not self._updating:
            self._apply_pose()

    def _reset_joints(self) -> None:
        self._updating = True
        try:
            for slider in self._sliders.values():
                slider.setValue(0)
        finally:
            self._updating = False
        if self._scene_active:
            self._apply_pose()

    def _joint_positions(self) -> dict[str, float]:
        return {
            name: float(np.radians(slider.value()))
            for name, slider in self._sliders.items()
        }

    def _rebuild_scene(self, reset_camera: bool) -> None:
        if self.model is None:
            return
        self._stop_auto_sensors(restore_static=False)
        self._stop_live_heatmap(restore_static=False)
        if self._owns_plotter:
            self.plotter.clear()
            self._setup_studio()
        else:
            self._remove_scene_actors(render=False)
        self._actors.clear()
        self._world_meshes.clear()
        self._local_points.clear()
        self._local_normals.clear()
        self._item_link.clear()
        self._robot_keys.clear()
        self._signal_keys.clear()
        self._signal_channel_indices.clear()
        self._signal_transposed_channel_indices.clear()
        self._signal_required_channels.clear()
        self._signal_expected_shapes.clear()
        self._signal_static_colors.clear()
        self._signal_display_names.clear()
        self._transformed_channel_index_cache.clear()
        self._replaced_links.clear()
        self._scene_suspended = False

        missing: list[str] = []
        load_errors: list[str] = []
        loaded = 0
        bounds = []
        positions = self._joint_positions()
        for key, visual, pose in self.model.visual_poses(positions):
            if not visual.mesh_path.is_file():
                missing.append(visual.mesh_path.name)
                continue
            try:
                mesh = pv.read(str(visual.mesh_path))
                try:
                    mesh = mesh.compute_normals(
                        cell_normals=False,
                        point_normals=True,
                        auto_orient_normals=True,
                        inplace=False,
                    )
                except Exception:
                    pass
                local_points = np.asarray(mesh.points, dtype=np.float64)
                try:
                    local_normals = np.asarray(
                        mesh.point_normals,
                        dtype=np.float64,
                    )
                except Exception:
                    local_normals = np.zeros((0, 3), dtype=np.float64)
                self._place_mesh(mesh, local_points, local_normals, pose)
                actor = self.plotter.add_mesh(
                    mesh,
                    name=f"humanoid:{key}",
                    color=visual.rgba[:3],
                    opacity=visual.rgba[3],
                    smooth_shading=True,
                    **self._phong(visual.rgba),
                )
            except Exception as exc:
                load_errors.append(f"{visual.mesh_path.name}: {exc}")
                continue

            link_name = key.split(":", 1)[0]
            self._actors[key] = actor
            self._world_meshes[key] = mesh
            self._local_points[key] = local_points
            self._local_normals[key] = local_normals
            self._item_link[key] = link_name
            self._robot_keys.append(key)
            loaded += 1
            bounds.append(mesh.bounds)

        signal_count, skipped_signals = self._load_signals(positions, bounds)
        self._refresh_live_mapping_controls()

        message = (
            f"Humanoid ready: {loaded} robot meshes and "
            f"{signal_count} sensor signal sections"
        )
        details = []
        if skipped_signals:
            details.append("Unmatched signals: " + ", ".join(skipped_signals))
        if missing:
            details.append("Missing meshes: " + ", ".join(sorted(set(missing))))
        if load_errors:
            details.append("Mesh errors: " + "; ".join(load_errors))
        self.status.setText(message)
        self.status.setToolTip("\n".join(details))
        self._scene_active = True
        self._set_scene_controls_enabled(True)
        self._apply_visibility()
        if reset_camera and (loaded or signal_count):
            self.plotter.view_isometric()
            model_bounds = (
                min(item[0] for item in bounds),
                max(item[1] for item in bounds),
                min(item[2] for item in bounds),
                max(item[3] for item in bounds),
                min(item[4] for item in bounds),
                max(item[5] for item in bounds),
            )
            self.plotter.reset_camera(bounds=model_bounds)
        self.plotter.render()

    @staticmethod
    def _load_signal_channel_indices(
        signal_obj: Path,
        point_count: int,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        int,
        tuple[int, int],
    ]:
        """Load 1-based signal IDs aligned with signal OBJ vertices."""
        signal_txt = signal_obj.parent / "signal.txt"
        if not signal_txt.is_file():
            return None, None, 0, (0, 0)
        try:
            values = np.asarray(
                [
                    int(line.strip())
                    for line in signal_txt.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ],
                dtype=np.int64,
            )
        except (OSError, UnicodeError, ValueError):
            return None, None, 0, (0, 0)
        if values.shape != (int(point_count),):
            return None, None, 0, (0, 0)
        valid = values > 0
        if not np.any(valid):
            return None, None, 0, (0, 0)
        selected_weft_count = 0
        configured_warp_count = 0
        layout_path = signal_obj.parent.parent / "layouts.json"
        if layout_path.is_file():
            try:
                layouts = json.loads(
                    layout_path.read_text(encoding="utf-8")
                )
                layout = layouts.get(signal_obj.parent.name, {})
                selected_weft_count = int(
                    layout.get("selected_weft_num", 0)
                )
                configured_warp_count = int(
                    layout.get("selected_warp_num", 0)
                )
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                selected_weft_count = 0
                configured_warp_count = 0
        config_candidates = [
            signal_obj.parent / "pipeline.cfg",
            signal_obj.parent.parent
            / f"{signal_obj.parent.name}.cfg",
        ]
        for config_path in (
            config_candidates
            if selected_weft_count <= 0
            or configured_warp_count <= 0
            else []
        ):
            if not config_path.is_file():
                continue
            try:
                config_text = config_path.read_text(
                    encoding="utf-8"
                )
            except (OSError, UnicodeError):
                continue
            drive_match = re.search(
                r"(?m)^\s*selected_weft_num\s*=\s*(\d+)",
                config_text,
            )
            sensor_match = re.search(
                r"(?m)^\s*selected_warp_num\s*=\s*(\d+)",
                config_text,
            )
            if drive_match and sensor_match:
                selected_weft_count = int(drive_match.group(1))
                configured_warp_count = int(sensor_match.group(1))
                break
        drive_count = 0
        sensor_count = 0
        if selected_weft_count > 0:
            region_stride = selected_weft_count + 1
            observed_band_count = (
                int(np.max(values[valid] // region_stride)) + 1
            )
            drive_count = (
                min(observed_band_count, configured_warp_count)
                if configured_warp_count > 0
                else observed_band_count
            )
            sensor_count = selected_weft_count
        required_channels = (
            drive_count * sensor_count
            if drive_count > 0 and sensor_count > 0
            else int(np.max(values[valid]))
        )
        indices = np.full(values.shape, -1, dtype=np.int64)
        transposed_indices = np.full(
            values.shape,
            -1,
            dtype=np.int64,
        )
        if drive_count > 0 and sensor_count > 0:
            # region_signal IDs are not raw channel IDs. Each longitudinal
            # band reserves ``selected_weft_num + 1`` region slots,
            # including non-taxel boundary regions. Convert only the interior
            # regions to the same drive-major/sensor-minor order used by the
            # serial payload and the 2D viewer:
            #     raw_index = drive * sensor_count + sensor
            region_stride = sensor_count + 1
            signal_ids = values[valid]
            drive_indices = signal_ids // region_stride
            region_indices = signal_ids % region_stride
            mapped = (
                (region_indices >= 1)
                & (region_indices <= sensor_count)
                & (drive_indices >= 0)
                & (drive_indices < drive_count)
            )
            valid_positions = np.flatnonzero(valid)
            mapped_positions = valid_positions[mapped]
            indices[mapped_positions] = (
                drive_indices[mapped] * sensor_count
                + region_indices[mapped]
                - 1
            )
            transposed = (
                (region_indices >= 1)
                & (region_indices <= sensor_count)
                & (drive_indices >= 0)
                & (drive_indices < drive_count)
            )
            transposed_positions = valid_positions[transposed]
            transposed_indices[transposed_positions] = (
                (region_indices[transposed] - 1) * drive_count
                + drive_indices[transposed]
            )
        else:
            indices[valid] = values[valid] - 1
            transposed_indices[valid] = values[valid] - 1
        return (
            indices,
            transposed_indices,
            required_channels,
            (drive_count, sensor_count),
        )

    def _load_signals(
        self,
        positions: dict[str, float],
        bounds: list,
    ) -> tuple[int, list[str]]:
        signal_root = Path(self.signal_dir_edit.text().strip()).expanduser()
        if not signal_root.is_dir():
            return 0, [f"missing directory {signal_root}"]

        parts = self._discover_signal_parts(signal_root, self.model)
        if not parts:
            return 0, ["no matching N_part.obj / curves_col_signal.obj"]

        skipped: list[str] = []
        loaded = 0
        for part in parts:
            pose = self.model.link_visual_pose(part.link_name, positions)
            if pose is None:
                skipped.append(f"{part.number}:{part.part_name}")
                continue
            try:
                points, colors, lines = self._load_colored_line_obj(
                    part.signal_obj
                )
            except Exception:
                skipped.append(f"{part.number}:{part.part_name}")
                continue
            if len(points) == 0:
                skipped.append(f"{part.number}:{part.part_name}")
                continue

            mesh = (
                pv.PolyData(points, lines=lines)
                if lines.size
                else pv.PolyData(points)
            )
            mesh["RGB"] = np.clip(
                np.round(colors * 255.0),
                0,
                255,
            ).astype(np.uint8)
            empty_normals = np.zeros((0, 3), dtype=np.float64)
            self._place_mesh(mesh, points, empty_normals, pose)
            key = f"signal:{part.number}:{part.link_name}"
            actor = self.plotter.add_mesh(
                mesh,
                name=f"humanoid:{key}",
                scalars="RGB",
                rgb=True,
                render_lines_as_tubes=True,
                line_width=3.5,
            )
            self._actors[key] = actor
            self._world_meshes[key] = mesh
            self._local_points[key] = points
            self._local_normals[key] = empty_normals
            self._item_link[key] = part.link_name
            self._signal_keys.append(key)
            (
                channel_indices,
                transposed_channel_indices,
                required_channels,
                expected_shape,
            ) = (
                self._load_signal_channel_indices(
                    part.signal_obj,
                    len(points),
                )
            )
            if channel_indices is not None:
                self._signal_channel_indices[key] = channel_indices
                self._signal_transposed_channel_indices[key] = (
                    transposed_channel_indices
                )
                self._signal_required_channels[key] = required_channels
                self._signal_expected_shapes[key] = expected_shape
                self._signal_static_colors[key] = np.array(
                    mesh["RGB"],
                    dtype=np.uint8,
                    copy=True,
                )
                reused = (
                    f" (mirrored from {part.reused_from})"
                    if part.reused_from
                    else ""
                )
                self._signal_display_names[key] = (
                    f"{part.number}: {part.link_name} "
                    f"[{expected_shape[0]}D x {expected_shape[1]}S = "
                    f"{required_channels} ch]{reused}"
                    if expected_shape[0] > 0 and expected_shape[1] > 0
                    else (
                        f"{part.number}: {part.link_name} "
                        f"[{required_channels} ch]{reused}"
                    )
                )
            self._replaced_links.add(part.link_name)
            bounds.append(mesh.bounds)
            loaded += 1

        numbered = {part.number for part in parts}
        for source in signal_root.glob("*_*.obj"):
            if source.parent.resolve() != signal_root.resolve():
                continue
            number = source.stem.split("_", 1)[0]
            if number.isdigit() and number not in numbered:
                skipped.append(f"{number}:no signal")
        return loaded, skipped

    def _normalize_live_port(self, port_name: str) -> str:
        normalizer = getattr(
            self.sensor_functions,
            "_sensor_port_key",
            None,
        )
        if callable(normalizer):
            return str(normalizer(port_name))
        return str(port_name or "")

    def _active_sensor_profiles(self) -> dict[str, dict]:
        getter = getattr(
            self.sensor_functions,
            "get_live_sensor_port_profiles",
            None,
        )
        if not callable(getter):
            return {}
        try:
            return {
                self._normalize_live_port(port): dict(profile)
                for port, profile in dict(getter() or {}).items()
            }
        except Exception:
            return {}

    def _refresh_live_mapping_controls(self) -> None:
        previous_part = self.live_part_combo.currentData()
        previous_port = self.live_port_combo.currentData()
        self.live_part_combo.blockSignals(True)
        self.live_port_combo.blockSignals(True)
        self.live_part_combo.clear()
        self.live_port_combo.clear()
        for key in self._signal_keys:
            if key not in self._signal_channel_indices:
                continue
            self.live_part_combo.addItem(
                self._signal_display_names.get(key, key),
                key,
            )
        profiles = self._active_sensor_profiles()
        for port_name, profile in profiles.items():
            rows = int(profile.get("n_row", 0) or 0)
            columns = int(profile.get("n_col", 0) or 0)
            primary = " [primary]" if profile.get("is_primary") else ""
            self.live_port_combo.addItem(
                f"{os.path.basename(port_name)} — "
                f"{rows}x{columns} ({rows * columns} ch){primary}",
                port_name,
            )
        for combo, previous in (
            (self.live_part_combo, previous_part),
            (self.live_port_combo, previous_port),
        ):
            index = combo.findData(previous)
            if index >= 0:
                combo.setCurrentIndex(index)
        self.live_part_combo.blockSignals(False)
        self.live_port_combo.blockSignals(False)
        self._on_live_part_changed()

    def _on_live_part_changed(self, *_args) -> None:
        if self._live_mapping is not None:
            self._stop_live_heatmap(restore_static=True)
        part_key = self.live_part_combo.currentData()
        required = int(
            self._signal_required_channels.get(str(part_key), 0)
        )
        exact_index = -1
        sufficient_index = -1
        profiles = self._active_sensor_profiles()
        for index in range(self.live_port_combo.count()):
            port_name = str(self.live_port_combo.itemData(index) or "")
            profile = profiles.get(port_name, {})
            count = int(profile.get("n_row", 0) or 0) * int(
                profile.get("n_col", 0) or 0
            )
            if count == required:
                exact_index = index
                break
            if count >= required and sufficient_index < 0:
                sufficient_index = index
        preferred = exact_index if exact_index >= 0 else sufficient_index
        if preferred >= 0:
            self.live_port_combo.setCurrentIndex(preferred)
        self.live_button.setEnabled(
            bool(part_key)
            and self.live_port_combo.count() > 0
            and required > 0
        )

    def _connect_live_frame_signal(self) -> bool:
        if self._live_frame_signal is not None:
            return True
        bridge = getattr(
            self.sensor_functions,
            "_payload_bridge",
            None,
        )
        signal = getattr(bridge, "port_frame_processed", None)
        if signal is None:
            return False
        try:
            signal.connect(self._on_live_port_frame)
        except Exception:
            return False
        self._live_frame_signal = signal
        return True

    def _disconnect_live_frame_signal(self) -> None:
        signal = self._live_frame_signal
        if signal is None:
            return
        try:
            signal.disconnect(self._on_live_port_frame)
        except Exception:
            pass
        self._live_frame_signal = None

    def _on_live_heatmap_toggled(self, enabled: bool) -> None:
        if not enabled:
            self._stop_live_heatmap(restore_static=True)
            return
        if self.auto_live_button.isChecked():
            self._stop_auto_sensors(restore_static=True)
        self._refresh_live_mapping_controls()
        part_key = str(self.live_part_combo.currentData() or "")
        port_name = self._normalize_live_port(
            str(self.live_port_combo.currentData() or "")
        )
        required = int(
            self._signal_required_channels.get(part_key, 0)
        )
        profiles = self._active_sensor_profiles()
        profile = profiles.get(port_name)
        available = (
            int(profile.get("n_row", 0) or 0)
            * int(profile.get("n_col", 0) or 0)
            if profile is not None
            else 0
        )
        if (
            not self._scene_active
            or not part_key
            or profile is None
            or available < required
            or not self._connect_live_frame_signal()
        ):
            self.live_button.blockSignals(True)
            self.live_button.setChecked(False)
            self.live_button.blockSignals(False)
            self.status.setText(
                "Live heatmap unavailable. Start and calibrate a Sensor-tab "
                f"source with at least {required} channels, then reload."
            )
            return
        self._live_mapping = (port_name, part_key)
        self._pending_live_frames.clear()
        self.mode_signal.setChecked(True)
        self._live_render_timer.start()
        size_note = (
            ""
            if available == required
            else f"; first {required} of {available} channels used"
        )
        self.status.setText(
            f"Live heatmap: {os.path.basename(port_name)} → "
            f"{self._signal_display_names.get(part_key, part_key)}"
            f"{size_note}"
        )

    def _on_auto_live_toggled(self, enabled: bool) -> None:
        if not enabled:
            self._resume_auto_live_after_restore = False
            self._auto_restart_pending = False
            self._stop_auto_sensors(restore_static=True)
            return
        self._resume_auto_live_after_restore = False
        self._auto_restart_pending = False
        if not self._scene_active:
            self.auto_live_button.blockSignals(True)
            self.auto_live_button.setChecked(False)
            self.auto_live_button.blockSignals(False)
            self.auto_live_status.setText(
                "Load the humanoid scene before auto-detecting sensors."
            )
            return
        thread = self._auto_sensor_thread
        if thread is not None and thread.isRunning():
            self.auto_live_button.blockSignals(True)
            self.auto_live_button.setChecked(False)
            self.auto_live_button.blockSignals(False)
            self.auto_live_status.setText(
                "Previous sensor scan is still stopping."
            )
            return

        self._stop_live_heatmap(restore_static=True)
        requirements = [
            (
                key,
                int(self._signal_required_channels[key]),
            )
            for key in self._signal_keys
            if key in self._signal_required_channels
            and self._is_fast_auto_detect_link(
                self._item_link.get(key, "")
            )
        ]
        if not requirements:
            self.auto_live_button.blockSignals(True)
            self.auto_live_button.setChecked(False)
            self.auto_live_button.blockSignals(False)
            self.auto_live_status.setText(
                "No head, torso, or hand signal mappings were found."
            )
            return

        self._auto_live_parts.clear()
        self._auto_mapping_metadata.clear()
        self._pending_auto_frames.clear()
        self.auto_mapping_combo.blockSignals(True)
        self.auto_mapping_combo.clear()
        self.auto_mapping_combo.blockSignals(False)
        self.auto_mapping_combo.setEnabled(False)
        self.swap_drive_sensor_checkbox.setEnabled(False)
        self.swap_drive_sensor_checkbox.setChecked(False)
        self.transpose_point_mapping_checkbox.setEnabled(False)
        self.transpose_point_mapping_checkbox.setChecked(False)
        self.flip_horizontal_mapping_checkbox.setEnabled(False)
        self.flip_horizontal_mapping_checkbox.setChecked(False)
        self.flip_vertical_mapping_checkbox.setEnabled(False)
        self.flip_vertical_mapping_checkbox.setChecked(False)
        self._set_signal_sections_neutral()
        self.mode_signal.setChecked(True)
        worker = HumanoidSensorAutoWorker(
            requirements,
            preferred_port_parts=self._preferred_auto_port_parts(),
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.sensor_found.connect(self._on_auto_sensor_found)
        worker.scan_summary.connect(self._on_auto_scan_summary)
        worker.frames_ready.connect(self._on_auto_sensor_frames)
        worker.calibration_started.connect(
            self._on_auto_calibration_started
        )
        worker.calibration_finished.connect(
            self._on_auto_calibration_finished
        )
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_auto_sensor_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._auto_sensor_worker = worker
        self._auto_sensor_thread = thread
        self.auto_live_status.setText(
            "Opening remembered humanoid sensor devices..."
        )
        self._live_render_timer.start()
        thread.start()

    def request_sensor_update(self) -> bool:
        """Update calibration for active independent humanoid sensors."""
        worker = self._auto_sensor_worker
        thread = self._auto_sensor_thread
        if (
            not self.auto_live_button.isChecked()
            or worker is None
            or thread is None
            or not thread.isRunning()
        ):
            return False
        worker.request_calibration()
        self.auto_live_status.setText(
            "Humanoid sensor calibration queued..."
        )
        return True

    def _on_auto_calibration_started(self, sensor_count) -> None:
        self.auto_live_status.setText(
            f"Updating {int(sensor_count)} humanoid sensor"
            f"{'' if int(sensor_count) == 1 else 's'}..."
        )

    def _on_auto_calibration_finished(
        self,
        updated_count,
        failed_count,
    ) -> None:
        updated_count = int(updated_count)
        failed_count = int(failed_count)
        if failed_count:
            self.auto_live_status.setText(
                f"Humanoid sensor update: {updated_count} succeeded, "
                f"{failed_count} failed; live heatmap continues."
            )
        else:
            self.auto_live_status.setText(
                f"Humanoid sensor update complete: "
                f"{updated_count} calibrated."
            )

    def _set_signal_sections_neutral(self) -> None:
        neutral = np.array([92, 100, 112], dtype=np.uint8)
        for key in self._signal_keys:
            mesh = self._world_meshes.get(key)
            if mesh is None:
                continue
            mesh["RGB"] = np.tile(neutral, (mesh.n_points, 1))
            try:
                mesh.Modified()
            except Exception:
                pass
        try:
            self.plotter.render()
        except Exception:
            pass

    def _on_auto_sensor_found(
        self,
        port_name,
        part_key,
        drive_count,
        sensor_count,
        extra_rows,
    ) -> None:
        if not self.auto_live_button.isChecked():
            return
        part_key = str(part_key)
        drive_count = int(drive_count)
        sensor_count = int(sensor_count)
        active_count = (
            drive_count * sensor_count
            if drive_count > 0 and sensor_count > 0
            else int(
                self._signal_required_channels.get(part_key, 0)
            )
        )
        device_identity = _stable_identity_for_device(port_name)
        saved_key = self._mapping_setting_key(
            part_key,
            port_name,
            drive_count,
            sensor_count,
            device_identity,
        )
        saved = self._saved_sensor_mappings.get(saved_key, {})
        if not saved and device_identity:
            legacy_key = self._mapping_setting_key(
                part_key,
                port_name,
                drive_count,
                sensor_count,
            )
            saved = self._saved_sensor_mappings.get(legacy_key, {})
            if saved:
                self._saved_sensor_mappings[saved_key] = dict(saved)
                self._save_humanoid_sensor_mapping_file()
        metadata = {
            "part": part_key,
            "port": str(port_name),
            "device_identity": device_identity,
            "drive_count": drive_count,
            "sensor_count": sensor_count,
            "swap_drive_sensor": bool(
                saved.get("swap_drive_sensor", False)
            ),
            "transpose_point_mapping": bool(
                saved.get("transpose_point_mapping", False)
            ),
            "flip_horizontal_mapping": bool(
                saved.get("flip_horizontal_mapping", False)
            ),
            "flip_vertical_mapping": bool(
                saved.get("flip_vertical_mapping", False)
            ),
        }
        self._auto_mapping_metadata[part_key] = metadata
        self._auto_live_parts.add(part_key)
        axis_text = (
            f"{drive_count}D x {sensor_count}S"
            if drive_count > 0 and sensor_count > 0
            else "shape unknown"
        )
        self.auto_mapping_combo.addItem(
            f"{os.path.basename(str(port_name))} → "
            f"{self._signal_display_names.get(part_key, part_key)} "
            f"[{axis_text}]",
            part_key,
        )
        self.auto_mapping_combo.setEnabled(True)
        if self.auto_mapping_combo.count() == 1:
            self._on_auto_mapping_selected()
        extra_note = (
            f", +1 column with {extra_rows} rows"
            if int(extra_rows) > 0
            else ""
        )
        self.auto_live_status.setText(
            f"Matched {os.path.basename(str(port_name))} "
            f"({active_count} channels{extra_note}) → "
            f"{self._signal_display_names.get(str(part_key), part_key)}"
        )

    def _on_auto_mapping_selected(self, *_args) -> None:
        part_key = str(
            self.auto_mapping_combo.currentData() or ""
        )
        metadata = self._auto_mapping_metadata.get(part_key)
        valid_shape = bool(
            metadata is not None
            and int(metadata.get("drive_count", 0)) > 0
            and int(metadata.get("sensor_count", 0)) > 0
        )
        self.swap_drive_sensor_checkbox.blockSignals(True)
        self.swap_drive_sensor_checkbox.setChecked(
            bool(
                metadata.get("swap_drive_sensor", False)
                if metadata is not None
                else False
            )
        )
        self.swap_drive_sensor_checkbox.blockSignals(False)
        self.swap_drive_sensor_checkbox.setEnabled(valid_shape)
        self.transpose_point_mapping_checkbox.blockSignals(True)
        self.transpose_point_mapping_checkbox.setChecked(
            bool(
                metadata.get("transpose_point_mapping", False)
                if metadata is not None
                else False
            )
        )
        self.transpose_point_mapping_checkbox.blockSignals(False)
        self.transpose_point_mapping_checkbox.setEnabled(valid_shape)
        for checkbox, field_name in (
            (
                self.flip_horizontal_mapping_checkbox,
                "flip_horizontal_mapping",
            ),
            (
                self.flip_vertical_mapping_checkbox,
                "flip_vertical_mapping",
            ),
        ):
            checkbox.blockSignals(True)
            checkbox.setChecked(
                bool(
                    metadata.get(field_name, False)
                    if metadata is not None
                    else False
                )
            )
            checkbox.blockSignals(False)
            checkbox.setEnabled(valid_shape)

    def _on_swap_drive_sensor_toggled(self, checked) -> None:
        part_key = str(
            self.auto_mapping_combo.currentData() or ""
        )
        metadata = self._auto_mapping_metadata.get(part_key)
        if metadata is None:
            return
        metadata["swap_drive_sensor"] = bool(checked)
        saved = self._save_sensor_mapping_setting(metadata)
        state = "swapped" if checked else "normal"
        self.auto_live_status.setText(
            f"{self._signal_display_names.get(part_key, part_key)}: "
            f"Drive/Sensor order {state}"
            + (" and saved" if saved else "; save failed")
        )

    def _on_transpose_point_mapping_toggled(self, checked) -> None:
        part_key = str(
            self.auto_mapping_combo.currentData() or ""
        )
        metadata = self._auto_mapping_metadata.get(part_key)
        if metadata is None:
            return
        metadata["transpose_point_mapping"] = bool(checked)
        saved = self._save_sensor_mapping_setting(metadata)
        state = "transposed" if checked else "normal"
        self.auto_live_status.setText(
            f"{self._signal_display_names.get(part_key, part_key)}: "
            f"humanoid point mapping {state}"
            + (" and saved" if saved else "; save failed")
        )

    def _set_mapping_flip_option(
        self,
        field_name,
        checked,
        label,
    ) -> None:
        part_key = str(
            self.auto_mapping_combo.currentData() or ""
        )
        metadata = self._auto_mapping_metadata.get(part_key)
        if metadata is None:
            return
        metadata[str(field_name)] = bool(checked)
        saved = self._save_sensor_mapping_setting(metadata)
        state = "flipped" if checked else "normal"
        self.auto_live_status.setText(
            f"{self._signal_display_names.get(part_key, part_key)}: "
            f"{label} {state}"
            + (" and saved" if saved else "; save failed")
        )

    def _on_flip_horizontal_mapping_toggled(self, checked) -> None:
        self._set_mapping_flip_option(
            "flip_horizontal_mapping",
            checked,
            "left/right mapping",
        )

    def _on_flip_vertical_mapping_toggled(self, checked) -> None:
        self._set_mapping_flip_option(
            "flip_vertical_mapping",
            checked,
            "top/bottom mapping",
        )

    @staticmethod
    def _apply_drive_sensor_order(
        values,
        drive_count,
        sensor_count,
        swap_drive_sensor,
    ):
        array = np.asarray(values, dtype=float).reshape(-1)
        drive_count = int(drive_count)
        sensor_count = int(sensor_count)
        if (
            not bool(swap_drive_sensor)
            or drive_count <= 0
            or sensor_count <= 0
            or array.size != drive_count * sensor_count
        ):
            return array
        return array.reshape(
            drive_count,
            sensor_count,
        ).T.reshape(-1)

    @staticmethod
    def _transform_point_channel_indices(
        channel_indices,
        drive_count,
        sensor_count,
        *,
        flip_horizontal=False,
        flip_vertical=False,
    ):
        indices = np.asarray(
            channel_indices,
            dtype=np.int64,
        ).copy()
        drive_count = int(drive_count)
        sensor_count = int(sensor_count)
        valid = (
            (indices >= 0)
            & (indices < drive_count * sensor_count)
            & (drive_count > 0)
            & (sensor_count > 0)
        )
        if not np.any(valid):
            return indices
        drives = indices[valid] // sensor_count
        sensors = indices[valid] % sensor_count
        if bool(flip_horizontal):
            drives = (drive_count - 1) - drives
        if bool(flip_vertical):
            sensors = (sensor_count - 1) - sensors
        indices[valid] = drives * sensor_count + sensors
        return indices

    def _on_auto_scan_summary(self, summary) -> None:
        if self.auto_live_button.isChecked():
            self.auto_live_status.setText(str(summary))

    def _on_auto_sensor_frames(self, frames) -> None:
        if not self.auto_live_button.isChecked():
            return
        for frame in list(frames or []):
            if not isinstance(frame, (list, tuple)) or len(frame) != 4:
                continue
            self._on_auto_sensor_frame(*frame)

    def _on_auto_sensor_frame(
        self,
        _port_name,
        part_key,
        raw_values,
        calibration_values,
    ) -> None:
        if not self.auto_live_button.isChecked():
            return
        part_key = str(part_key)
        if part_key not in self._auto_live_parts:
            return
        self._pending_auto_frames[part_key] = (
            np.asarray(raw_values, dtype=float).copy(),
            np.asarray(calibration_values, dtype=float).copy(),
        )

    def _stop_auto_sensors(
        self,
        restore_static: bool,
        *,
        wait_for_thread: bool = False,
    ) -> None:
        self.auto_live_button.blockSignals(True)
        self.auto_live_button.setChecked(False)
        self.auto_live_button.blockSignals(False)
        worker = self._auto_sensor_worker
        if worker is not None:
            worker.stop()
        thread = self._auto_sensor_thread
        if (
            wait_for_thread
            and thread is not None
            and thread.isRunning()
        ):
            thread.quit()
            thread.wait(5000)
        if thread is not None and not thread.isRunning():
            self._auto_sensor_worker = None
            self._auto_sensor_thread = None
        self._pending_auto_frames.clear()
        self._auto_live_parts.clear()
        self.auto_mapping_combo.setEnabled(False)
        self.swap_drive_sensor_checkbox.setEnabled(False)
        self.transpose_point_mapping_checkbox.setEnabled(False)
        self.flip_horizontal_mapping_checkbox.setEnabled(False)
        self.flip_vertical_mapping_checkbox.setEnabled(False)
        if self._live_mapping is None:
            self._live_render_timer.stop()
        if restore_static:
            self._restore_static_signal_colors()
            if self._scene_active:
                try:
                    self.plotter.render()
                except Exception:
                    pass
        self.auto_live_status.setText("Independent sensors: off")

    def _on_auto_sensor_thread_finished(self) -> None:
        self._auto_sensor_worker = None
        self._auto_sensor_thread = None
        self.auto_live_button.blockSignals(True)
        self.auto_live_button.setChecked(False)
        self.auto_live_button.blockSignals(False)
        self.auto_live_button.setEnabled(self._scene_active)
        if self._live_mapping is None:
            self._live_render_timer.stop()
        if (
            self._scene_active
            and (
                self._resume_auto_live_after_restore
                or self._auto_restart_pending
            )
        ):
            self._schedule_auto_live_restore()

    def _resume_kept_auto_live_stream(self) -> bool:
        thread = getattr(self, "_auto_sensor_thread", None)
        auto_live_button = getattr(self, "auto_live_button", None)
        if auto_live_button is None:
            return False
        stream_is_active = bool(
            auto_live_button.isChecked()
            and getattr(self, "_auto_sensor_worker", None) is not None
            and thread is not None
            and thread.isRunning()
        )
        auto_live_button.setEnabled(
            self._scene_active
            and (
                stream_is_active
                or thread is None
                or not thread.isRunning()
            )
        )
        if not stream_is_active:
            return False
        self._resume_auto_live_after_restore = False
        self._auto_restart_pending = False
        live_render_timer = getattr(self, "_live_render_timer", None)
        if live_render_timer is not None:
            live_render_timer.start()
        auto_live_status = getattr(self, "auto_live_status", None)
        if auto_live_status is not None:
            auto_live_status.setText(
                "Independent sensors remained connected; "
                "live heatmap resumed."
            )
        return True

    def _schedule_auto_live_restore(self) -> None:
        if (
            self._closed
            or not self._scene_active
            or not self._resume_auto_live_after_restore
        ):
            return
        thread = self._auto_sensor_thread
        if thread is not None and thread.isRunning():
            self._auto_restart_pending = True
            self.auto_live_button.setEnabled(False)
            self.auto_live_status.setText(
                "Waiting for the previous sensor reader to stop..."
            )
            return
        self._auto_restart_pending = False
        self.auto_live_button.setEnabled(True)
        QtCore.QTimer.singleShot(
            0,
            self._restart_auto_live_after_restore,
        )

    def _restart_auto_live_after_restore(self) -> None:
        if (
            self._closed
            or not self._scene_active
            or not self._resume_auto_live_after_restore
        ):
            return
        self._resume_auto_live_after_restore = False
        self._auto_restart_pending = False
        self.auto_live_button.setEnabled(True)
        self.auto_live_button.setChecked(True)

    def _restore_static_signal_colors(self) -> None:
        for key, static_colors in self._signal_static_colors.items():
            mesh = self._world_meshes.get(key)
            if mesh is None or len(static_colors) != mesh.n_points:
                continue
            mesh["RGB"] = np.array(
                static_colors,
                dtype=np.uint8,
                copy=True,
            )
            try:
                mesh.Modified()
            except Exception:
                pass

    def _stop_live_heatmap(self, restore_static: bool) -> None:
        self._live_render_timer.stop()
        self._disconnect_live_frame_signal()
        self._pending_live_frames.clear()
        self._live_mapping = None
        self.live_button.blockSignals(True)
        self.live_button.setChecked(False)
        self.live_button.blockSignals(False)
        if restore_static:
            self._restore_static_signal_colors()
            if self._scene_active:
                try:
                    self.plotter.render()
                except Exception:
                    pass

    def _on_live_port_frame(
        self,
        port_name,
        _timestamp,
        _frame_sequence,
        raw_matrix,
        calibration_matrix,
    ) -> None:
        mapping = self._live_mapping
        if mapping is None:
            return
        normalized_port = self._normalize_live_port(port_name)
        if normalized_port != mapping[0]:
            return
        self._pending_live_frames[normalized_port] = (
            np.asarray(raw_matrix, dtype=float).copy(),
            np.asarray(calibration_matrix, dtype=float).copy(),
        )

    def _render_pending_live_heatmap(self) -> None:
        rendered = False
        mapping = self._live_mapping
        if mapping is not None:
            frame = self._pending_live_frames.pop(mapping[0], None)
            if frame is not None:
                rendered = self._apply_live_heatmap_frame(
                    mapping[1],
                    frame[0],
                    frame[1],
                    settings=self._sensor_tab_heatmap_settings(),
                ) or rendered
        pending_auto = getattr(self, "_pending_auto_frames", {})
        auto_frames = dict(pending_auto)
        pending_auto.clear()
        for part_key, frame in auto_frames.items():
            metadata = self._auto_mapping_metadata.get(part_key, {})
            ordered_raw = self._apply_drive_sensor_order(
                frame[0],
                metadata.get("drive_count", 0),
                metadata.get("sensor_count", 0),
                metadata.get("swap_drive_sensor", False),
            )
            ordered_calibration = self._apply_drive_sensor_order(
                frame[1],
                metadata.get("drive_count", 0),
                metadata.get("sensor_count", 0),
                metadata.get("swap_drive_sensor", False),
            )
            rendered = self._apply_live_heatmap_frame(
                part_key,
                ordered_raw,
                ordered_calibration,
                settings={"unmapped_neutral": True},
                transpose_point_mapping=bool(
                    metadata.get(
                        "transpose_point_mapping",
                        False,
                    )
                ),
                drive_count=metadata.get("drive_count", 0),
                sensor_count=metadata.get("sensor_count", 0),
                flip_horizontal_mapping=metadata.get(
                    "flip_horizontal_mapping",
                    False,
                ),
                flip_vertical_mapping=metadata.get(
                    "flip_vertical_mapping",
                    False,
                ),
            ) or rendered
        if rendered and self.mode_signal.isChecked():
            self.plotter.render()

    def _sensor_tab_heatmap_settings(self) -> dict:
        settings_getter = getattr(
            self.sensor_functions,
            "get_heatmap_settings",
            None,
        )
        if not callable(settings_getter):
            return {}
        try:
            return dict(settings_getter() or {})
        except Exception:
            return {}

    def _live_channel_indices(
        self,
        part_key,
        *,
        transpose_point_mapping,
        drive_count,
        sensor_count,
        flip_horizontal_mapping,
        flip_vertical_mapping,
    ):
        part_key = str(part_key)
        base_indices = (
            self._signal_transposed_channel_indices.get(part_key)
            if bool(transpose_point_mapping)
            else self._signal_channel_indices.get(part_key)
        )
        if base_indices is None:
            return None
        cache = getattr(
            self,
            "_transformed_channel_index_cache",
            None,
        )
        if cache is None:
            cache = {}
            self._transformed_channel_index_cache = cache
        cache_key = (
            part_key,
            bool(transpose_point_mapping),
            int(drive_count),
            int(sensor_count),
            bool(flip_horizontal_mapping),
            bool(flip_vertical_mapping),
        )
        cached = cache.get(cache_key)
        if cached is None:
            cached = (
                HumanoidViewerWidget._transform_point_channel_indices(
                    base_indices,
                    drive_count,
                    sensor_count,
                    flip_horizontal=flip_horizontal_mapping,
                    flip_vertical=flip_vertical_mapping,
                )
            )
            cache[cache_key] = cached
        return cached

    def _apply_live_heatmap_frame(
        self,
        part_key,
        raw_values,
        calibration_values,
        *,
        settings,
        transpose_point_mapping=False,
        drive_count=0,
        sensor_count=0,
        flip_horizontal_mapping=False,
        flip_vertical_mapping=False,
    ) -> bool:
        raw_array = np.asarray(raw_values, dtype=float)
        calibration_array = np.asarray(
            calibration_values,
            dtype=float,
        )
        if raw_array.shape != calibration_array.shape:
            return False
        if raw_array.ndim == 2:
            raw_flat = raw_array.T.reshape(-1)
            calibration_flat = calibration_array.T.reshape(-1)
        else:
            raw_flat = raw_array.reshape(-1)
            calibration_flat = calibration_array.reshape(-1)
        difference = raw_flat - calibration_flat
        settings = dict(settings or {})
        response_mode = str(
            settings.get(
                "response_mode",
                DEFAULT_HEATMAP_RESPONSE_MODE,
            )
        )
        if response_mode == HEATMAP_RESPONSE_PROXIMITY_ENHANCED:
            signal_values = difference
        else:
            signal_values = np.zeros_like(difference, dtype=float)
            np.divide(
                100.0 * difference,
                calibration_flat,
                out=signal_values,
                where=calibration_flat != 0.0,
            )

        part_key = str(part_key)
        mesh = self._world_meshes.get(part_key)
        channel_indices = HumanoidViewerWidget._live_channel_indices(
            self,
            part_key,
            transpose_point_mapping=transpose_point_mapping,
            drive_count=drive_count,
            sensor_count=sensor_count,
            flip_horizontal_mapping=flip_horizontal_mapping,
            flip_vertical_mapping=flip_vertical_mapping,
        )
        if mesh is None or channel_indices is None:
            return False
        valid = (
            (channel_indices >= 0)
            & (channel_indices < signal_values.size)
        )
        vertex_values = np.zeros(channel_indices.shape, dtype=float)
        vertex_values[valid] = signal_values[channel_indices[valid]]
        live_rgb = heatmap_3d_rgb(
            vertex_values,
            palette=settings.get(
                "palette_3d",
                DEFAULT_HEATMAP_3D_PALETTE,
            ),
            response_mode=response_mode,
            saturation_pct=settings.get(
                "saturation_pct",
                DEFAULT_HEATMAP_SATURATION_PCT,
            ),
            noise_floor_pct=settings.get(
                "noise_floor_pct",
                DEFAULT_HEATMAP_NOISE_FLOOR_PCT,
            ),
            proximity_noise_floor=settings.get(
                "proximity_noise_floor",
                DEFAULT_PROXIMITY_NOISE_FLOOR,
            ),
            proximity_knee=settings.get(
                "proximity_knee",
                DEFAULT_PROXIMITY_KNEE,
            ),
            proximity_saturation=settings.get(
                "proximity_saturation",
                DEFAULT_PROXIMITY_SATURATION,
            ),
            color_gain=settings.get(
                "color_gain_3d",
                DEFAULT_HEATMAP_3D_COLOR_GAIN,
            ),
            use_absolute_signal=settings.get(
                "use_absolute_signal",
                True,
            ),
        )
        existing_colors = np.asarray(mesh["RGB"])
        if existing_colors.shape == (len(channel_indices), 3):
            colors = existing_colors
        else:
            colors = np.empty(
                (len(channel_indices), 3),
                dtype=np.uint8,
            )
        if bool(settings.get("unmapped_neutral", False)):
            colors[:] = (92, 100, 112)
        else:
            static_colors = self._signal_static_colors.get(part_key)
            if (
                static_colors is not None
                and np.asarray(static_colors).shape == colors.shape
            ):
                colors[:] = static_colors
            else:
                colors[:] = live_rgb
        colors[valid] = live_rgb[valid]
        mesh["RGB"] = colors
        try:
            mesh.Modified()
        except Exception:
            pass
        return True

    def _apply_visibility(self) -> None:
        if self._closed or not self._scene_active:
            return
        use_signal = self.mode_signal.isChecked()
        for key in self._robot_keys:
            actor = self._actors.get(key)
            if actor is None:
                continue
            link_name = self._item_link.get(key)
            actor.SetVisibility(
                not (use_signal and link_name in self._replaced_links)
            )
        for key in self._signal_keys:
            actor = self._actors.get(key)
            if actor is not None:
                actor.SetVisibility(use_signal)
        self.plotter.render()

    @staticmethod
    def _place_mesh(
        mesh,
        local_points: np.ndarray,
        local_normals: np.ndarray,
        pose: np.ndarray,
    ) -> None:
        rotation = pose[:3, :3]
        mesh.points = local_points @ rotation.T + pose[:3, 3]
        if local_normals.size:
            mesh.point_data.set_array(local_normals @ rotation.T, "Normals")

    def _apply_pose(self) -> None:
        if self.model is None or self._closed or not self._scene_active:
            return
        positions = self._joint_positions()
        poses = {}
        for link_name in set(self._item_link.values()):
            pose = self.model.link_visual_pose(link_name, positions)
            if pose is not None:
                poses[link_name] = pose
        for key, mesh in self._world_meshes.items():
            link_name = self._item_link.get(key)
            if link_name and link_name in poses:
                self._place_mesh(
                    mesh,
                    self._local_points[key],
                    self._local_normals[key],
                    poses[link_name],
                )
        self.plotter.render()

    def _remove_scene_actors(self, render: bool = True) -> None:
        for actor in list(self._actors.values()):
            if actor is None:
                continue
            try:
                self.plotter.remove_actor(
                    actor,
                    reset_camera=False,
                    render=False,
                )
            except Exception:
                pass
        if render:
            try:
                self.plotter.render()
            except Exception:
                pass

    def _cached_scene_actors_attached(self) -> bool:
        renderer = getattr(self.plotter, "renderer", None)
        renderer_actors = getattr(renderer, "actors", {})
        try:
            attached_ids = {
                id(actor) for actor in renderer_actors.values()
            }
        except Exception:
            return False
        cached_actors = [
            actor for actor in self._actors.values()
            if actor is not None
        ]
        return bool(cached_actors) and all(
            id(actor) in attached_ids for actor in cached_actors
        )

    def restore_scene(self, reset_camera: bool = False) -> bool:
        """Restore a released shared-plotter scene without reloading the URDF."""
        if self._closed or self.model is None:
            return False
        if self._before_scene_load is not None:
            try:
                self._before_scene_load()
            except Exception as exc:
                self.status.setText(
                    f"Humanoid viewport could not be restored: {exc}"
                )
                return False
        if (
            bool(getattr(self, "_scene_suspended", False))
            and HumanoidViewerWidget._cached_scene_actors_attached(self)
        ):
            self._scene_suspended = False
            self._scene_active = True
            self._set_scene_controls_enabled(True)
            self._refresh_live_mapping_controls()
            previous_status = self._status_before_suspend
            if previous_status is not None:
                self.status.setText(previous_status[0])
                self.status.setToolTip(previous_status[1])
            self._status_before_suspend = None
            self._apply_visibility()
            kept_live_stream = (
                HumanoidViewerWidget._resume_kept_auto_live_stream(
                    self
                )
            )
            if (
                not kept_live_stream
                and bool(
                    getattr(
                        self,
                        "_resume_auto_live_after_restore",
                        False,
                    )
                )
            ):
                HumanoidViewerWidget._schedule_auto_live_restore(self)
            return True
        if not self._scene_active:
            self._scene_suspended = False
            self.status.setText("Restoring humanoid scene...")
            self._rebuild_scene(reset_camera=bool(reset_camera))
        kept_live_stream = (
            HumanoidViewerWidget._resume_kept_auto_live_stream(self)
        )
        if (
            not kept_live_stream
            and self._scene_active
            and bool(
                getattr(
                    self,
                    "_resume_auto_live_after_restore",
                    False,
                )
            )
        ):
            HumanoidViewerWidget._schedule_auto_live_restore(self)
        return bool(self._scene_active)

    def release_scene(
        self,
        render: bool = True,
        *,
        discard: bool = False,
    ) -> None:
        """Suspend or discard humanoid actors from a shared plotter."""
        if self._owns_plotter:
            return
        resume_auto_live = bool(
            not discard and self.auto_live_button.isChecked()
        )
        self._resume_auto_live_after_restore = resume_auto_live
        self._auto_restart_pending = False
        thread = self._auto_sensor_thread
        keep_auto_live = bool(
            resume_auto_live
            and self._auto_sensor_worker is not None
            and thread is not None
            and thread.isRunning()
        )
        if keep_auto_live:
            self._live_render_timer.stop()
        else:
            self._stop_auto_sensors(restore_static=False)
        self._stop_live_heatmap(restore_static=False)
        if not discard:
            if not keep_auto_live:
                self._restore_static_signal_colors()
            if self._scene_active:
                self._status_before_suspend = (
                    self.status.text(),
                    self.status.toolTip(),
                )
            for actor in self._actors.values():
                if actor is not None:
                    actor.SetVisibility(False)
            self._scene_active = False
            self._scene_suspended = bool(self._actors)
            self._set_scene_controls_enabled(False)
            if self.model is not None:
                self.status.setText(
                    "Humanoid scene paused while this tab is hidden."
                )
            if render:
                try:
                    self.plotter.render()
                except Exception:
                    pass
            return

        self._remove_scene_actors(render=False)
        self._actors.clear()
        self._world_meshes.clear()
        self._local_points.clear()
        self._local_normals.clear()
        self._item_link.clear()
        self._robot_keys.clear()
        self._signal_keys.clear()
        self._signal_channel_indices.clear()
        self._signal_transposed_channel_indices.clear()
        self._signal_required_channels.clear()
        self._signal_expected_shapes.clear()
        self._signal_static_colors.clear()
        self._signal_display_names.clear()
        self._transformed_channel_index_cache.clear()
        self._replaced_links.clear()
        self._scene_active = False
        self._scene_suspended = False
        self._status_before_suspend = None
        self._resume_auto_live_after_restore = False
        self._auto_restart_pending = False
        self._set_scene_controls_enabled(False)
        if self.model is not None:
            self.status.setText(
                "Humanoid scene hidden. Press Load URDF and Sensor Signals to show it."
            )
        if render:
            try:
                self.plotter.render()
            except Exception:
                pass

    def shutdown(self) -> None:
        if self._closed:
            return
        self._stop_auto_sensors(
            restore_static=False,
            wait_for_thread=True,
        )
        self._stop_live_heatmap(restore_static=False)
        if self._owns_plotter:
            try:
                self.plotter.clear()
            except Exception:
                pass
            try:
                self.plotter.close()
            except Exception:
                pass
        else:
            self.release_scene(render=False, discard=True)
        self._closed = True

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)
