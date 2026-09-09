"""Stable USB identities and display profiles for known tactile sensors."""

from __future__ import annotations

import json
import os

from phd.dependence.paths import resource_path


HUMANOID_SENSOR_MAPPING_FILE = resource_path(
    "config",
    "humanoid_sensor_mappings.json",
)

HUMANOID_PART_DISPLAY = {
    "sensor:robot_arm_end_effector": (
        "Robot Arm End-Effector Sensor",
        (10, 10),
    ),
    "sensor:testing_8x10": ("Testing Sensor", (8, 10)),
    "signal:1:head_link": ("Head", (9, 14)),
    "signal:10:torso_link": ("Torso Part 10", (15, 16)),
    "signal:11:torso_link": ("Torso Part 11", (16, 17)),
    "signal:3:left_rubber_hand": ("Left Hand", (7, 8)),
    "signal:3:right_rubber_hand": ("Right Hand", (7, 8)),
}

EXTRA_COLUMN_PARTS = {
    "sensor:robot_arm_end_effector",
}


def stable_usb_port_identity(port) -> str:
    """Build a port-independent identity from a USB serial number."""
    serial_number = str(
        getattr(port, "serial_number", "") or ""
    ).strip()
    if not serial_number:
        return ""
    vid = getattr(port, "vid", None)
    pid = getattr(port, "pid", None)
    vid_text = f"{int(vid):04x}" if vid is not None else "unknown"
    pid_text = f"{int(pid):04x}" if pid is not None else "unknown"
    return f"usb:{vid_text}:{pid_text}:{serial_number.lower()}"


def load_humanoid_device_assignments() -> dict:
    """Load physical-device-to-humanoid-part assignments."""
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
    return dict(assignments) if isinstance(assignments, dict) else {}


def humanoid_sensor_annotation(port, assignments=None) -> str:
    """Return a friendly humanoid part and grid-size label for a port."""
    identity = stable_usb_port_identity(port)
    if not identity:
        return ""
    if assignments is None:
        assignments = load_humanoid_device_assignments()
    assignment = dict(assignments or {}).get(identity, {})
    part_key = str(
        assignment.get("part", "")
        if isinstance(assignment, dict)
        else ""
    )
    display = HUMANOID_PART_DISPLAY.get(part_key)
    if display is None:
        return ""
    part_name, shape = display
    return f"{part_name} ({int(shape[0])}x{int(shape[1])})"


def humanoid_sensor_grid_shape(port, assignments=None):
    """Return the configured display grid shape for a recognized port."""
    identity = stable_usb_port_identity(port)
    if not identity:
        return None
    if assignments is None:
        assignments = load_humanoid_device_assignments()
    assignment = dict(assignments or {}).get(identity, {})
    part_key = str(
        assignment.get("part", "")
        if isinstance(assignment, dict)
        else ""
    )
    display = HUMANOID_PART_DISPLAY.get(part_key)
    if display is None:
        return None
    shape = display[1]
    return int(shape[0]), int(shape[1])


def humanoid_sensor_uses_extra_column(port, assignments=None):
    """Return the saved +1-column setting, or ``None`` if unrecognized."""
    identity = stable_usb_port_identity(port)
    if not identity:
        return None
    if assignments is None:
        assignments = load_humanoid_device_assignments()
    assignment = dict(assignments or {}).get(identity, {})
    part_key = str(
        assignment.get("part", "")
        if isinstance(assignment, dict)
        else ""
    )
    if part_key not in HUMANOID_PART_DISPLAY:
        return None
    return part_key in EXTRA_COLUMN_PARTS


def _current_serial_port(port_name):
    """Return pyserial metadata for a currently connected device name."""
    try:
        import serial.tools.list_ports
    except ImportError:
        return None
    requested = str(port_name or "").strip()
    requested_basename = os.path.basename(requested).lower()
    requested_realpath = (
        os.path.realpath(requested)
        if os.path.isabs(requested)
        else ""
    )
    try:
        ports = serial.tools.list_ports.comports()
    except Exception:
        return None
    for port in ports:
        device = str(getattr(port, "device", "") or "")
        name = str(getattr(port, "name", "") or "")
        if (
            os.path.basename(device).lower() == requested_basename
            or os.path.basename(name).lower() == requested_basename
            or (
                requested_realpath
                and os.path.realpath(device) == requested_realpath
            )
        ):
            return port
    return None


def humanoid_sensor_grid_shape_for_device(port_name):
    """Resolve a current device name, then return its humanoid grid shape."""
    port = _current_serial_port(port_name)
    if port is None:
        return None
    return humanoid_sensor_grid_shape(port)


def humanoid_sensor_extra_column_for_device(port_name):
    """Resolve a device name and return its saved +1-column setting."""
    port = _current_serial_port(port_name)
    if port is None:
        return None
    return humanoid_sensor_uses_extra_column(port)
