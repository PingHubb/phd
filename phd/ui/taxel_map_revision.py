"""Immutable setup revisions: measured geometry is never rewritten by Remember.

The root anchor/settings/positions describe the original measurements. A setup
revision stores the operator's requested anchor/plane separately until a real
relocation or fresh mapping run verifies the new geometry.
"""

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


VERIFICATION_MESSAGE = (
    "This map revision has changed setup geometry. Verify it using Relocate "
    "loaded map, or run a fresh mapping, before taxel navigation or calibration. "
    "The retained centres were measured with the original setup."
)


def revision_setup(result):
    revision = (result or {}).get("setup_revision") or {}
    return copy.deepcopy(revision.get("requested_setup"))


def revision_motion_error(result):
    revision = (result or {}).get("setup_revision") or {}
    return VERIFICATION_MESSAGE if revision.get("verification_required") else ""


def normalize_setup(setup):
    """Validate a persisted proposal without changing task state or hardware."""
    value = copy.deepcopy(setup)
    if not isinstance(value, dict):
        raise ValueError("The saved setup revision is invalid.")
    rows, columns = int(value["sensor_rows"]), int(value["sensor_columns"])
    anchor = value["anchor_pose"]
    index = int(anchor["taxel_index"])
    if rows <= 0 or columns <= 0 or not 0 <= index < rows * columns:
        raise ValueError("The proposed anchor is outside the sensor grid.")
    for field, length in (("tool_position_m", 3), ("tool_quaternion_wxyz", 4)):
        vector = [float(item) for item in anchor[field]]
        if len(vector) != length or not all(math.isfinite(item) for item in vector):
            raise ValueError("The proposed robot pose is invalid.")
        anchor[field] = vector
    norm = math.sqrt(sum(item ** 2 for item in anchor["tool_quaternion_wxyz"]))
    if not .99 <= norm <= 1.01:
        raise ValueError("The proposed robot orientation is not a unit quaternion.")
    anchor["tool_quaternion_wxyz"] = [item / norm for item in anchor["tool_quaternion_wxyz"]]
    z = value.get("fixed_scan_z_m")
    if z is not None:
        z = float(z)
        clearance = anchor["tool_position_m"][2] - z
        if not math.isfinite(z) or not .005 - 1e-9 <= clearance <= .100 + 1e-9:
            raise ValueError("The scan plane must be 5–100 mm below the safe anchor.")
    value["fixed_scan_z_m"] = z
    anchor["scan_plane_choice"] = dict(
        z_m=z, saved_at=value.get("saved_at"),
        sensor_identity=value["sensor_identity"], sensor_rows=rows, sensor_columns=columns,
    )
    return value


def same_pose(first, second):
    """Only numerical round-off/sign-equivalent quaternions count as unchanged."""
    if first.get("taxel_index") != second.get("taxel_index"):
        return False
    a, b = first["tool_position_m"], second["tool_position_m"]
    if any(abs(float(x) - float(y)) > 1e-8 for x, y in zip(a, b)):
        return False
    a, b = first["tool_quaternion_wxyz"], second["tool_quaternion_wxyz"]
    return min(sum((float(x) - sign * float(y)) ** 2 for x, y in zip(a, b))
               for sign in (1, -1)) < 1e-14


def setup_changes_measurements(source, setup):
    old_z = source["settings"].get("fixed_scan_z_mm")
    old_z = None if old_z is None else float(old_z) / 1000.
    z = setup["fixed_scan_z_m"]
    same_height = old_z == z or (old_z is not None and z is not None and abs(old_z - z) < 1e-8)
    return not same_pose(source["anchor_pose"], setup["anchor_pose"]) or not same_height


def validate_setup_binding(source, setup):
    setup = normalize_setup(setup)
    settings = source["settings"]
    if (setup["sensor_identity"] != source.get("sensor_identity")
            or (setup["sensor_rows"], setup["sensor_columns"])
            != (settings["sensor_rows"], settings["sensor_columns"])
            or setup["anchor_pose"]["taxel_index"] != source["anchor_pose"]["taxel_index"]):
        raise ValueError("The new setup does not belong to the loaded map's sensor/anchor.")
    return setup


def read_revision_source(path):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    result = json.loads(raw)
    if not isinstance(result, dict) or result.get("task") != "taxel_position_mapping":
        raise ValueError("Select a saved taxel position map first.")
    return result, hashlib.sha256(raw).hexdigest()


def save_setup_revision(source_path, setup, expected_digest, *, directory=None):
    """Write a unique, atomic JSON; never write into the parent recording."""
    source_path = Path(source_path).expanduser().resolve()
    source, digest = read_revision_source(source_path)
    if digest != expected_digest:
        raise ValueError("The loaded map changed while confirmation was open. Reload it and try again.")
    setup = validate_setup_binding(source, setup)
    changed = setup_changes_measurements(source, setup)
    now = datetime.now(timezone.utc)
    result = copy.deepcopy(source)
    result["saved_at"] = now.isoformat()
    result["setup_revision"] = dict(
        version=1, created_at=now.isoformat(), parent_json_path=str(source_path),
        parent_sha256=digest,
        measurement_saved_at=(source.get("setup_revision") or {}).get(
            "measurement_saved_at", source.get("saved_at")),
        verification_required=changed, requested_setup=setup,
    )
    label = source.get("sensor_label") or setup.get("sensor_label") or setup["sensor_identity"]
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", str(label)).strip("-")[:70] or "sensor"
    root = Path(directory) if directory is not None else source_path.parent.parent
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(
        prefix=f"taxel_position_map_revision_{now.strftime('%Y%m%d_%H%M%S_%f')}_sensor-{slug}_",
        dir=root,
    ))
    target = folder / "taxel_position_map.json"
    temporary = folder / "setup_revision.tmp"
    result["output_folder"] = str(folder)
    result["json_path"] = str(target)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        # Only our newly-created temporary file/empty folder, never source data.
        temporary.unlink(missing_ok=True)
        folder.rmdir()
        raise
    return result, target
