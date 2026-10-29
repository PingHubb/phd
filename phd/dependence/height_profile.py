"""Signal, neighbour contrast and noise versus probe height for one taxel.

Reads the rows of a Calibration "Proximity approach" trial (every taxel is
recorded) and answers: at which height does the target taxel clearly beat its
neighbours, and is that height safely above the first force change?
"""

from __future__ import annotations

import numpy as np

from phd.ui.calibration_analysis import taxel_signal_column

# Earlier sensors that mapped reliably were scanned where the target was about
# 1.6x its strongest neighbour; below about 1.3x centres were not repeatable.
MIN_DOMINANCE = 1.3
MIN_SNR = 10.0
CONTACT_FORCE_G = 2.0
CONTACT_MARGIN_MM = 1.0


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def cardinal_neighbours(index, rows, columns):
    row, column = int(index) % int(rows), int(index) // int(rows)
    found = []
    for r, c in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
        if 0 <= r < rows and 0 <= c < columns:
            found.append(c * int(rows) + r)
    return found


def _approach_frames(rows, rows_n, columns_n, signal_field):
    """Unique approach frames as (travel_mm, z_m, force_g, values[taxels])."""
    count = int(rows_n) * int(columns_n)
    names = [taxel_signal_column(signal_field, index) for index in range(count)]
    frames = []
    last_sequence = None
    for row in rows:
        if str(row.get("phase")) != "approaching":
            continue
        sequence = row.get("frame_sequence")
        if sequence is not None and sequence == last_sequence:
            continue
        last_sequence = sequence
        travel = _number(row.get("travel_down_mm"))
        if travel is None:
            continue
        values = np.array([_number(row.get(name)) for name in names], dtype=float)
        if not np.all(np.isfinite(values)):
            continue
        force = _number(row.get("force_delta_g"))
        frames.append((travel, _number(row.get("tool_z_m")), force, values))
    return frames


def analyse_height_profile(
    rows,
    rows_n,
    columns_n,
    target,
    signal_field="diff_ave",
    bin_mm=0.5,
    baseline_travel_mm=1.0,
):
    """Return per-height statistics and a recommended scan height."""
    frames = _approach_frames(rows, rows_n, columns_n, signal_field)
    target = int(target)
    baseline_frames = [f for f in frames if f[0] <= float(baseline_travel_mm)]
    if len(baseline_frames) < 5:
        baseline_frames = frames[: max(5, min(20, len(frames)))]
    if len(frames) < 10 or len(baseline_frames) < 5:
        raise ValueError("Not enough approach frames to analyse the height profile.")
    baseline_values = np.array([f[3] for f in baseline_frames])
    baseline = np.median(baseline_values, axis=0)
    noise = 1.4826 * np.median(np.abs(baseline_values - baseline), axis=0)
    target_noise = max(float(noise[target]), 1e-6)
    neighbours = cardinal_neighbours(target, rows_n, columns_n)

    contact_travel = None
    for travel, _z, force, _values in frames:
        if force is not None and abs(force) >= CONTACT_FORCE_G:
            contact_travel = travel
            break

    bins = {}
    for travel, z, force, values in frames:
        departure = np.abs(values - baseline)
        others = np.delete(departure, target)
        key = int(np.floor(travel / float(bin_mm)))
        bins.setdefault(key, []).append(
            (
                travel,
                z,
                force,
                float(departure[target]),
                max(float(departure[index]) for index in neighbours) if neighbours else 0.0,
                float(np.max(others)) if others.size else 0.0,
            )
        )

    profile = []
    for key in sorted(bins):
        samples = np.array(
            [[np.nan if value is None else value for value in sample] for sample in bins[key]],
            dtype=float,
        )
        target_signal = float(np.median(samples[:, 3]))
        strongest_other = float(np.median(samples[:, 5]))
        forces = samples[:, 2][np.isfinite(samples[:, 2])]
        z_values = samples[:, 1][np.isfinite(samples[:, 1])]
        profile.append(
            {
                "travel_mm": float(np.median(samples[:, 0])),
                "tool_z_mm": float(np.median(z_values) * 1000.0) if z_values.size else None,
                "frames": int(samples.shape[0]),
                "target_signal": target_signal,
                "strongest_neighbour": float(np.median(samples[:, 4])),
                "strongest_other": strongest_other,
                "dominance": target_signal / max(strongest_other, 1e-6),
                "snr": target_signal / target_noise,
                "force_g": float(np.median(np.abs(forces))) if forces.size else None,
            }
        )

    non_contact = [
        level
        for level in profile
        if contact_travel is None
        or level["travel_mm"] <= contact_travel - CONTACT_MARGIN_MM
    ]
    usable = [level for level in non_contact if level["snr"] >= MIN_SNR]
    best = max(usable, key=lambda level: level["dominance"]) if usable else None
    contact_levels = [
        level for level in profile
        if contact_travel is not None and level["travel_mm"] >= contact_travel
    ]
    contact_best = (
        max(contact_levels, key=lambda level: level["dominance"]) if contact_levels else None
    )
    feasible = bool(best is not None and best["dominance"] >= MIN_DOMINANCE)
    return {
        "target_taxel": target,
        "neighbours": neighbours,
        "signal_field": str(signal_field),
        "baseline_noise_counts": target_noise,
        "contact_travel_mm": contact_travel,
        "contact_force_g": CONTACT_FORCE_G,
        "profile": profile,
        "recommended_non_contact": best,
        "best_at_contact": contact_best,
        "non_contact_feasible": feasible,
        "criteria": {
            "min_dominance": MIN_DOMINANCE,
            "min_snr": MIN_SNR,
            "contact_margin_mm": CONTACT_MARGIN_MM,
        },
    }
