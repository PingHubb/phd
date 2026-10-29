"""Validate saved taxel maps and move above a measured taxel without probing."""

from __future__ import annotations

import copy
import time

import numpy as np
from phd.ui.taxel_map_revision import (
    revision_motion_error, revision_setup, setup_changes_measurements, validate_setup_binding,
)

from phd.ui.experiment_tasks import (
    CalibrationTask, TaxelPositionMappingTask, position_map_sensor_error,
)


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
    if "setup_revision" in normalized:
        try:
            setup = validate_setup_binding(normalized, revision_setup(normalized))
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("The saved setup revision is invalid.") from exc
        normalized["setup_revision"]["requested_setup"] = setup
        if setup_changes_measurements(normalized, setup) and not revision_motion_error(normalized):
            raise ValueError("Changed map setup is missing its verification-required flag.")
    return normalized


class MappedTaxelMoveTask(TaxelPositionMappingTask):
    """A single monitored route to a saved clearance pose; never runs a probe.

    Go to the selected taxel without visiting the anchor XY. Lift vertically
    first when below clearance, and restore orientation there if needed. This
    transient task uses the ordinary experiment timer and Stop lifecycle.
    """

    id = "mapped_taxel_move"
    # This transient command finishes after every grid click. Keep the safety
    # stream available so another mapped taxel can be selected immediately.
    disconnect_force_meter_on_stop = False
    ORIENTATION_TOLERANCE_RAD = np.radians(2.0)
    ANGULAR_STATIONARY_TOLERANCE_RAD = np.radians(0.1)

    def __init__(self, result, taxel_index, *, line_speed_m_s, force_guard_g):
        if isinstance(result, dict) and revision_motion_error(result):
            raise ValueError(revision_motion_error(result))
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
        self._navigation_queue = []
        self._move_goal_quaternion = None
        self._move_last_quaternion = None

    @staticmethod
    def _orientation_error(current, target):
        current = np.asarray(current, dtype=float)
        target = np.asarray(target, dtype=float)
        if (current.shape != (4,) or target.shape != (4,)
                or not np.all(np.isfinite(current)) or not np.all(np.isfinite(target))):
            return float("inf")
        norm = float(np.linalg.norm(current) * np.linalg.norm(target))
        if norm < 1e-9:
            return float("inf")
        # q and -q describe the same physical orientation.
        dot = abs(float(np.dot(current, target))) / norm
        return float(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))

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
        mismatch = position_map_sensor_error(host, self.map_result)
        if mismatch:
            return reject(mismatch)
        if not np.isfinite(self.line_speed_m_s) or not 0.001 <= self.line_speed_m_s <= 0.1:
            return reject("Set the tool line speed between 1 and 100 mm/s.")
        if not np.isfinite(self.force_guard_g) or not 1.0 <= self.force_guard_g <= 2000.0:
            return reject("Set a valid unexpected-force stop threshold (1–2000 g).")
        api = self._robot_api(host)
        pose = self._read_tool_pose(api)
        if pose is None or not callable(getattr(api, "send_positions_tool_position", None)):
            return reject("Robot TCP feedback and position control are required to move above a taxel.")
        if not np.isfinite(self._orientation_error(pose[1], pose[1])):
            return reject("Valid robot tool-orientation feedback is required before moving.")
        now = time.monotonic()
        try:
            feedback_age = now - float(api.current_feedback_state_time)
        except (AttributeError, TypeError, ValueError):
            feedback_age = float("inf")
        if not -0.1 <= feedback_age <= 1.0:
            return reject("Live robot feedback is required before moving to a mapped taxel.")
        if getattr(api, "current_project_paused", None) is True:
            return reject(
                "Taxel move did not start: the robot project is paused. "
                "Check the robot controller and clear the pause only when safe, then click the taxel again."
            )
        frame = self._read_sensor_frame(host)
        if frame is None:
            return reject(
                f"Connect the {self.sensor_rows}x{self.sensor_columns} tactile sensor "
                "before moving to a mapped taxel."
            )
        force = self._fresh_force(host, now)
        if force is None:
            # Starts the reader and re-enters on_start once fresh data arrives;
            # the robot does not move before then.
            return self._begin_force_meter_connection(host)
        if abs(float(force[0])) / self.NEWTONS_PER_GRAM >= CalibrationTask.EMERGENCY_FORCE_G:
            return reject("Taxel move did not start: raw force reached the 900 g software limit.")
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
        host._append_sidebar_control_message(
            f"Taxel {self._current_taxel}: moving to its saved safe pose without returning to the anchor. "
            "A vertical lift or tool-orientation correction will be made first only if needed."
        )
        return True

    def on_tick(self, host) -> None:
        if self._active and getattr(self._robot_api(host), "current_project_paused", None) is True:
            self._request_stop(
                host, "robot_project_paused",
                "Taxel move stopped: the robot project is paused. "
                "Remaining waypoints were cancelled; check the controller before trying again.",
            )
            return
        super().on_tick(host)
        if not self._active or self._phase != "navigation_preflight":
            return
        frame = self._read_sensor_frame(host)
        if frame is None or frame[0] == self._preflight_sequence:
            return
        pose = self._read_tool_pose(self._robot_api(host))
        if pose is None:
            self._request_stop(host, "robot_feedback_lost", "Taxel move stopped: robot TCP feedback was lost.")
            return
        self._begin_selected_taxel_route(host, pose)

    def _begin_selected_taxel_route(self, host, pose):
        position = np.array(pose[0], dtype=float, copy=True)
        current_q = np.asarray(pose[1], dtype=float)
        current_q = current_q / np.linalg.norm(current_q)
        target = np.asarray(self.target_position, dtype=float)
        target_q = np.asarray(self.anchor_pose["tool_quaternion_wxyz"], dtype=float)
        queue = []
        # Never take a diagonal shortcut from the probing/contact depth, even
        # when the starting height is within the normal arrival tolerance.
        if position[2] < target[2] - 1e-6:
            position[2] = target[2]
            queue.append((position.copy(), current_q.copy(), "Lifting vertically to clearance before moving to the selected taxel"))
        if self._orientation_error(current_q, target_q) > self.ORIENTATION_TOLERANCE_RAD:
            queue.append((position.copy(), target_q.copy(), "Restoring the saved tool orientation at clearance height"))
        # Both endpoints are now at/above saved clearance. No anchor waypoint
        # and no descent to the taxel's lower non-contact scanning position.
        queue.append((target.copy(), target_q.copy(), f"Moving directly to the saved safe pose above Taxel {self._current_taxel}"))
        self._navigation_queue = queue
        self._advance_navigation_route(host)

    def _advance_navigation_route(self, host):
        if not self._navigation_queue:
            self._request_stop(host, "route_lost", "Taxel move stopped: the selected-taxel route was lost.")
            return
        position, quaternion, detail = self._navigation_queue.pop(0)
        action = "navigation_waypoint" if self._navigation_queue else "navigation_arrived"
        self._command_move(host, position, action, detail, quaternion=quaternion)

    def _command_move(self, host, position, action, detail, *, quaternion=None):
        target_q = self.anchor_pose["tool_quaternion_wxyz"] if quaternion is None else quaternion
        self._move_goal_quaternion = np.asarray(target_q, dtype=float).copy()
        self._move_last_quaternion = None
        return super()._command_move(host, position, action, detail, quaternion=target_q)

    def _tick_move(self, host, now):
        pose = self._read_tool_pose(self._robot_api(host))
        if pose is None:
            self._request_stop(host, "robot_feedback_lost", "Taxel move stopped: robot TCP feedback was lost.")
            return
        current_q = pose[1]
        angular_error = self._orientation_error(current_q, self._move_goal_quaternion)
        if not np.isfinite(angular_error):
            self._request_stop(
                host, "robot_feedback_lost",
                "Taxel move stopped: tool-orientation feedback became invalid.",
            )
            return
        still_rotating = self._move_last_quaternion is None or self._orientation_error(
            current_q, self._move_last_quaternion
        ) > self.ANGULAR_STATIONARY_TOLERANCE_RAD
        self._move_last_quaternion = current_q.copy()
        if angular_error > self.ORIENTATION_TOLERANCE_RAD or still_rotating:
            self._move_arrived_at = None
            if (self._move_started_at is not None
                    and now - self._move_started_at > self._move_timeout_current_sec):
                self._request_stop(
                    host, "orientation_timeout",
                    "Taxel move stopped: the tool did not reach and settle at its commanded orientation.",
                )
            return
        super()._tick_move(host, now)

    def _dispatch_move_action(self, host, action: str) -> None:
        if action == "navigation_waypoint":
            self._advance_navigation_route(host)
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
        self._navigation_queue = []
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
