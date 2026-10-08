"""Validate saved taxel maps and move above a measured taxel without probing."""

from __future__ import annotations

import copy
import time

import numpy as np

from phd.ui.experiment_tasks import CalibrationTask, TaxelPositionMappingTask


def prepare_taxel_map(result: dict) -> dict:
    """Validate motion coordinates before exposing a recording for navigation.

    Partial recordings are supported, but only measured positions are usable.
    Neither loading nor validating a recording sends a robot command.
    """
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

    def vector(values, size, name):
        try:
            array = np.array(values, dtype=float, copy=True)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {name} in the position map.") from None
        if array.shape != (size,) or not np.all(np.isfinite(array)):
            raise ValueError(f"Invalid {name} in the position map.")
        return array

    anchor = result.get("anchor_pose")
    try:
        anchor_index = int(anchor.get("taxel_index", -1)) if isinstance(anchor, dict) else -1
    except (TypeError, ValueError):
        anchor_index = -1
    if not isinstance(anchor, dict) or not 0 <= anchor_index < taxel_count:
        raise ValueError("The position map anchor taxel is outside the sensor grid.")
    if (sensor_rows, sensor_columns) == (8, 10) and anchor_index != 35:
        raise ValueError("The position map needs its original Taxel 35 anchor.")
    anchor_position = vector(anchor.get("tool_position_m"), 3, "anchor position")
    quaternion = vector(anchor.get("tool_quaternion_wxyz"), 4, "tool orientation")
    norm = float(np.linalg.norm(quaternion))
    if not 0.99 <= norm <= 1.01:
        raise ValueError("The saved tool orientation is not a unit quaternion.")
    quaternion /= norm
    try:
        xy_limit = float(settings["max_xy_offset_mm"]) / 1000.0
    except (KeyError, TypeError, ValueError):
        raise ValueError("The position map has no valid XY workspace limit.") from None
    if not np.isfinite(xy_limit) or not 0.0 < xy_limit <= 0.2:
        raise ValueError("The saved XY workspace limit must be within 200 mm.")

    records = result.get("positions")
    if not isinstance(records, list):
        raise ValueError("The position map has no position list.")
    normalized = copy.deepcopy(result)
    normalized["anchor_pose"]["tool_quaternion_wxyz"] = quaternion.tolist()
    measured = []
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid taxel record in the position map.")
        index = record.get("taxel_index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < taxel_count:
            raise ValueError(
                f"Taxel indices must be integers from 0 to {taxel_count - 1}."
            )
        if index in seen:
            raise ValueError(f"Taxel {index} appears more than once in the map.")
        seen.add(index)
        if record.get("status") != "measured":
            continue
        if (record.get("row"), record.get("column")) != (
            index % sensor_rows,
            index // sensor_rows,
        ):
            raise ValueError(f"Taxel {index} has inconsistent row/column numbering.")
        centre = vector(record.get("tool_position_m"), 3, f"Taxel {index} centre")
        safe = vector(
            record.get("safe_approach_position_m"), 3, f"Taxel {index} clearance pose"
        )
        if (
            np.linalg.norm(safe[:2] - centre[:2]) > 1e-6
            or abs(float(safe[2] - anchor_position[2])) > 1e-6
            or safe[2] < centre[2]
        ):
            raise ValueError(f"Taxel {index} has an inconsistent clearance pose.")
        if float(np.linalg.norm(safe[:2] - anchor_position[:2])) > xy_limit + 1e-9:
            raise ValueError(f"Taxel {index} is outside the saved XY workspace limit.")
        orientation = vector(
            record.get("tool_quaternion_wxyz"), 4, f"Taxel {index} orientation"
        )
        orientation_norm = float(np.linalg.norm(orientation))
        if not 0.99 <= orientation_norm <= 1.01 or abs(
            float(np.dot(orientation / orientation_norm, quaternion))
        ) < 1.0 - 1e-6:
            raise ValueError(f"Taxel {index} has an inconsistent tool orientation.")
        item = copy.deepcopy(record)
        item["safe_approach_position_m"] = safe.tolist()
        measured.append(item)
    normalized["positions"] = measured
    return normalized


class MappedTaxelMoveTask(TaxelPositionMappingTask):
    """A single monitored route to a saved clearance pose; never runs a probe.

    Reuse the mapper's straight-line routing, feedback arrival checks and
    sensor/force guards. This transient task uses the ordinary experiment
    timer and Stop lifecycle, without modifying or saving the original map.
    """

    id = "mapped_taxel_move"
    # This transient command finishes after every grid click. Keep the safety
    # stream available so another mapped taxel can be selected immediately.
    disconnect_force_meter_on_stop = False

    def __init__(self, result, taxel_index, *, line_speed_m_s, force_guard_g):
        super().__init__(storage_path=None, output_directory=None)
        self.map_result = prepare_taxel_map(result)
        settings = self.map_result.get("settings") or {}
        self.sensor_rows = int(settings.get("sensor_rows") or self.sensor_rows)
        self.sensor_columns = int(settings.get("sensor_columns") or self.sensor_columns)
        records = {record["taxel_index"]: record for record in self.map_result["positions"]}
        if taxel_index not in records:
            raise ValueError(f"Taxel {taxel_index} has no measured position.")
        self._current_taxel = int(taxel_index)
        self.label = f"Move Above Taxel {self._current_taxel}"
        self.target_position = list(records[taxel_index]["safe_approach_position_m"])
        self.anchor_pose = self.map_result["anchor_pose"]
        self.line_speed_m_s = float(line_speed_m_s)
        self.force_guard_g = float(force_guard_g)
        self.max_xy_offset_m = float(self.map_result["settings"]["max_xy_offset_mm"]) / 1000.0
        # Even a short lateral move begins after reaching clearance height.
        self.route_xy_tolerance_m = 1e-9

    def _progress(self, host, detail="") -> None:
        callback = getattr(host, "_refresh_taxel_navigation_status", None)
        if callable(callback):
            callback(detail)

    def on_start(self, host) -> bool:
        def reject(message):
            host._append_sidebar_control_message(message)
            self._progress(host, message)
            return False

        active = getattr(host, "_sidebar_active_task", None)
        if active is not None and active is not self:
            return reject("Stop the current experiment before selecting a mapped taxel.")
        if not np.isfinite(self.line_speed_m_s) or not 0.001 <= self.line_speed_m_s <= 0.1:
            return reject("Set the tool line speed between 1 and 100 mm/s.")
        if not np.isfinite(self.force_guard_g) or not 1.0 <= self.force_guard_g <= 2000.0:
            return reject("Set a valid unexpected-force stop threshold (1–2000 g).")
        api = self._robot_api(host)
        pose = self._read_tool_pose(api)
        if pose is None or not callable(getattr(api, "send_positions_tool_position", None)):
            return reject("Robot TCP feedback and position control are required to move above a taxel.")
        origin = np.asarray(self.anchor_pose["tool_position_m"], dtype=float)
        if np.linalg.norm(pose[0][:2] - origin[:2]) > self.max_xy_offset_m:
            return reject("Move the robot into the recorded sensor workspace before using the taxel map.")
        current_q = pose[1]
        q_norm = float(np.linalg.norm(current_q))
        target_q = np.asarray(self.anchor_pose["tool_quaternion_wxyz"], dtype=float)
        if q_norm < 1e-9 or abs(float(np.dot(current_q / q_norm, target_q))) < np.cos(np.radians(2.5)):
            return reject("Restore the tool orientation used during mapping before selecting a taxel.")
        frame = self._read_sensor_frame(host)
        if frame is None:
            return reject(
                f"Connect the {self.sensor_rows}x{self.sensor_columns} tactile sensor "
                "before moving to a mapped taxel."
            )
        now = time.monotonic()
        force = self._fresh_force(host, now)
        if force is None:
            return reject("Connect the force transmitter and wait for live data before moving to a mapped taxel.")
        self._force_baseline_n = float(force[0])
        self._last_sensor_sequence = frame[0]
        self._last_sensor_advance_at = now
        self._preflight_sequence = frame[0]
        self._active = True
        self._phase = "navigation_preflight"
        setter = getattr(host, "_set_taxel_mapping_controls_running", None)
        if callable(setter):
            setter(True)
        self._progress(host, f"Taxel {self._current_taxel}: checking live readings before moving")
        return True

    def on_tick(self, host) -> None:
        super().on_tick(host)
        if not self._active or self._phase != "navigation_preflight":
            return
        frame = self._read_sensor_frame(host)
        if frame is None or frame[0] == self._preflight_sequence:
            return
        self._route_to(
            host,
            self.target_position[:2],
            self.target_position[2],
            "navigation_arrived",
            f"Moving above Taxel {self._current_taxel} via the saved clearance height",
        )

    def _dispatch_move_action(self, host, action: str) -> None:
        if action == "route_waypoint":
            self._advance_route(host)
        elif action == "navigation_arrived":
            self._completed = True
            host._stop_sidebar_control(
                show_result=False,
                message=f"Robot reached the saved clearance pose above Taxel {self._current_taxel}.",
            )

    def on_stop(self, host, show_result=True) -> None:
        CalibrationTask._stop_robot(self._robot_api(host))
        self._active = False
        self._phase = "idle"
        self._route_queue = []
        setter = getattr(host, "_set_taxel_mapping_controls_running", None)
        if callable(setter):
            setter(False)
        if self._completed:
            xyz = ", ".join(f"{value * 1000.0:.2f}" for value in self.target_position)
            message = f"Arrived above Taxel {self._current_taxel} · TCP ({xyz}) mm"
        else:
            reason = (self._stop_reason or "manual stop").replace("_", " ")
            message = f"Move to Taxel {self._current_taxel} stopped: {reason}."
        self._progress(host, message)
        host._append_sidebar_control_message(message)
