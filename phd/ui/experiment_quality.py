"""Pure measurement helpers shared by mapping, acquisition and offline analysis.

No hardware access. Signal units are ADC/digital sensor units, not capacitance
in pF. A response centre and a lattice residual are not geometric ground truth.
"""
from __future__ import annotations

import numpy as np


def rolling_medians(values, window=7):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("Finite baseline samples are required.")
    window = max(1, min(int(window), len(values)))
    return np.asarray([
        np.median(values[i - window + 1:i + 1])
        for i in range(window - 1, len(values))
    ])


def downward_detector(values, window=7, minimum_margin=6.0, noise_floor=1.0):
    """One-sided robust shift rule used identically by mapping and calibration."""
    rolling = rolling_medians(values, window)
    centre = float(np.median(rolling))
    sigma = float(1.4826 * np.median(np.abs(rolling - centre)))
    low = float(np.percentile(rolling, 1))
    margin = max(float(minimum_margin), centre - low + max(float(noise_floor), 1.5 * sigma))
    return {
        "centre": centre, "sigma": sigma, "low": low,
        "margin": margin, "lower": centre - margin,
    }


def finite_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def unique_sensor_rows(rows):
    """Drop explicitly repeated frames; leave legacy, untimestamped CSVs readable."""
    previous = None
    result = []
    for row in rows:
        sequence = finite_number(row.get("frame_sequence"))
        stamp = finite_number(row.get("sensor_monotonic_s"))
        key = (stamp, sequence) if stamp is not None else sequence
        if key is not None and key == previous:
            continue
        previous = key
        result.append(row)
    return result


def compare_position_maps(maps):
    """Within-setup centre repeatability, without claiming absolute accuracy."""
    if len(maps) < 2:
        raise ValueError("Select at least two maps from the same unchanged setup.")
    shapes = {
        tuple(m.get("sensor_shape") or (
            m.get("settings", {}).get("sensor_rows"),
            m.get("settings", {}).get("sensor_columns"),
        )) for m in maps
    }
    if len(shapes) != 1 or any(v is None for shape in shapes for v in shape):
        raise ValueError("Maps must use the same sensor layout.")
    identities = {m.get("sensor_identity") for m in maps if m.get("sensor_identity")}
    if len(identities) > 1 or any(m.get("relocation") for m in maps):
        raise ValueError("Do not mix different sensors or relocated maps in a repeatability test.")
    groups = {}
    for run, mapping in enumerate(maps):
        seen = set()
        for point in mapping.get("positions", []):
            if point.get("status") != "measured":
                continue
            index = int(point["taxel_index"])
            xyz = np.asarray(point["tool_position_m"], dtype=float)
            if index in seen or xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
                raise ValueError("Invalid or duplicate measured position.")
            seen.add(index)
            groups.setdefault(index, []).append((run, xyz * 1000))
    result = []
    for index, points in sorted(groups.items()):
        if len(points) < 2:
            continue
        positions = np.asarray([p[1] for p in points])
        mean = np.mean(positions, axis=0)
        radii = np.linalg.norm(positions[:, :2] - mean[:2], axis=1)
        result.append({
            "taxel_index": index, "runs": len(points),
            "mean_xyz_mm": mean.tolist(),
            "std_xyz_mm": np.std(positions, axis=0, ddof=1).tolist(),
            "xy_rms_about_mean_mm": float(np.sqrt(np.mean(radii ** 2))),
            "xy_max_about_mean_mm": float(np.max(radii)),
        })
    return {
        "interpretation": "Response-centre repeatability, not absolute geometric accuracy. "
        "Valid only for the same sensor, mounting, robot frame, tool and unchanged setup.",
        "map_count": len(maps), "taxels": result,
    }
