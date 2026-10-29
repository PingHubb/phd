"""Stepwise, contact-referenced proximity profile of one taxel.

Rows come from the Calibration "Proximity profile" protocol. They are tagged
with ``profile_kind``:

- ``surface_probe``: slow light-contact descents that define the surface Z.
- ``level``: stationary frames at one commanded gap above that surface.
- ``reference``: stationary frames at the far reference gap, repeated during
  each sweep so slow baseline drift can be interpolated and removed.
- ``move`` / ``settle``: transitions, kept for diagnostics only.

Proximity lowers the signed signal, so a positive *departure* means the
signal dropped relative to the drift-corrected far-field baseline.

The *localized* departure subtracts the median departure of taxels far from
the target. That removes drift and far-field coupling shared by the whole
sensor (for example the robot arm and tool), leaving proximity specific to
the target taxel. Both ranges are reported; they answer different questions.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from phd.ui.calibration_analysis import taxel_signal_column

DEFAULT_K_VALUES = (3.0, 5.0)
BOOTSTRAP_RESAMPLES = 1000
# A bootstrap over fewer sweeps only reshuffles a handful of curves.
MIN_BOOTSTRAP_SWEEPS = 5
DEFAULT_COMMON_MODE_DISTANCE = 6
MIN_COMMON_MODE_DISTANCE = 3
MIN_COMMON_MODE_TAXELS = 8
DOMINANCE_LEVELS = (1.0, 1.2)
# Two-sided 95% Student-t quantiles by degrees of freedom.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def _number(value):
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flag(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _t95(dof):
    if dof < 1:
        return None
    for limit in sorted(_T95):
        if dof <= limit:
            return _T95[limit]
    return 1.96


def _robust_sigma(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return None
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def contact_crossing(samples, threshold_g):
    """First sample at or above the threshold that the next sample confirms.

    The final sample of a probe counts as confirmed: a stepped probe stops at
    the first step whose readings at rest exceed the threshold.
    """
    for index, sample in enumerate(samples):
        if sample[2] >= threshold_g and (index + 1 == len(samples) or samples[index + 1][2] >= threshold_g):
            return index
    return None


def _crossing_z(samples, crossing, threshold_g):
    """Interpolate Z between the last sample below and the first at the threshold."""
    _t1, z1, f1 = samples[crossing]
    if crossing == 0:
        return z1
    _t0, z0, f0 = samples[crossing - 1]
    if f0 >= threshold_g or f1 <= f0:
        return z1
    return z0 + (threshold_g - f0) / (f1 - f0) * (z1 - z0)


def surface_reference(rows, threshold_g):
    """Surface Z from the light-contact probes, with zero-force extrapolation.

    Each probe uses the threshold recorded with its rows (relative to the force
    zero re-taken before it) and falls back to ``threshold_g`` for older trials.
    """
    probes = defaultdict(list)
    thresholds = defaultdict(list)
    zeros = defaultdict(list)
    for row in rows:
        if row.get("profile_kind") != "surface_probe":
            continue
        index = _number(row.get("profile_probe"))
        elapsed = _number(row.get("elapsed_s"))
        z_value = _number(row.get("tool_z_m"))
        force = _number(row.get("force_delta_abs_g"))
        if None in (index, elapsed, z_value, force):
            continue
        key = (_flag(row.get("profile_probe_final")), int(index))
        probes[key].append((elapsed, z_value, force))
        for store, column in ((thresholds, "profile_contact_threshold_g"), (zeros, "force_baseline_g")):
            value = _number(row.get(column))
            if value is not None:
                store[key].append(value)
    results = []
    for (final, index), samples in sorted(probes.items()):
        samples.sort()
        probe_threshold = float(np.median(thresholds[(final, index)])) if thresholds[(final, index)] else float(threshold_g)
        crossing = contact_crossing(samples, probe_threshold)
        payload = {
            "final": final,
            "probe": index,
            "samples": len(samples),
            "contact_threshold_g": probe_threshold,
            "force_zero_g": float(np.median(zeros[(final, index)])) if zeros[(final, index)] else None,
            "contact_z_m": None if crossing is None else _crossing_z(samples, crossing, probe_threshold),
            "zero_force_z_m": None,
        }
        loaded = samples[crossing:] if crossing is not None else []
        forces = np.asarray([item[2] for item in loaded], dtype=float)
        if len(loaded) >= 4 and float(np.ptp(forces)) > 0.5:
            slope, intercept = np.polyfit(forces, [item[1] for item in loaded], 1)
            if slope < 0:
                payload["zero_force_z_m"] = float(intercept)
        results.append(payload)

    def contacts(final):
        return [p["contact_z_m"] for p in results if p["final"] == final and p["contact_z_m"] is not None]

    initial, last = contacts(False), contacts(True)
    zero_force = [p["zero_force_z_m"] for p in results if not p["final"] and p["zero_force_z_m"] is not None]
    surface = float(np.median(initial)) if initial else None
    return {
        "force_threshold_g": float(threshold_g),
        "probes": results,
        "surface_z_m": surface,
        "surface_sd_mm": float(np.std(initial, ddof=1) * 1000.0) if len(initial) > 1 else None,
        "zero_force_surface_z_m": float(np.median(zero_force)) if zero_force else None,
        "zero_force_offset_mm": (
            float((np.median(zero_force) - surface) * 1000.0) if zero_force and surface is not None else None
        ),
        "final_shift_mm": (
            float((np.median(last) - surface) * 1000.0) if last and surface is not None else None
        ),
        "interpretation": (
            "Surface = median TCP Z where force first exceeds the threshold during slow "
            "light-contact probes. The zero-force extrapolation shows how far the fabric "
            "moved before the force reached that threshold."
        ),
    }


def common_mode_taxels(rows_n, columns_n, target, min_distance=DEFAULT_COMMON_MODE_DISTANCE):
    """Taxels far from the target and the Chebyshev cell distance actually used.

    Indices are top-down and column-major, as in the Sensor Signal Viewer. On
    small sensors the distance shrinks (not below MIN_COMMON_MODE_DISTANCE)
    until MIN_COMMON_MODE_TAXELS remain; otherwise the list is empty.
    """
    rows_n, columns_n, target = int(rows_n), int(columns_n), int(target)
    row, column = target % rows_n, target // rows_n
    distance = {
        index: max(abs(index % rows_n - row), abs(index // rows_n - column))
        for index in range(rows_n * columns_n)
    }
    for cells in range(int(min_distance), MIN_COMMON_MODE_DISTANCE - 1, -1):
        far = [index for index, value in distance.items() if value >= cells]
        if len(far) >= MIN_COMMON_MODE_TAXELS:
            return far, cells
    return [], None


def stationary_windows(rows, signal_field, taxel_count, target, surface_z_m, common_mode=None):
    """One summary per stationary level/reference window."""
    names = [taxel_signal_column(signal_field, index) for index in range(taxel_count)]
    groups = defaultdict(list)
    for row in rows:
        kind = row.get("profile_kind")
        step = _number(row.get("profile_step"))
        if kind not in {"level", "reference"} or step is None:
            continue
        values = [_number(row.get(name)) for name in names]
        elapsed = _number(row.get("elapsed_s"))
        z_value = _number(row.get("tool_z_m"))
        if elapsed is None or z_value is None or any(value is None for value in values):
            continue
        groups[int(step)].append((row, elapsed, z_value, _number(row.get("force_delta_abs_g")), values))
    windows = []
    for step in sorted(groups):
        frames = groups[step]
        first = frames[0][0]
        matrix = np.asarray([item[4] for item in frames], dtype=float)
        z_values = np.asarray([item[2] for item in frames], dtype=float)
        forces = [item[3] for item in frames if item[3] is not None]
        window = {
            "step": step,
            "kind": first.get("profile_kind"),
            "sweep": int(_number(first.get("profile_sweep")) or 0),
            "direction": str(first.get("profile_direction") or ""),
            "target_gap_mm": _number(first.get("profile_target_gap_mm")),
            "time_s": float(np.median([item[1] for item in frames])),
            "frames": len(frames),
            "gap_mm": float((np.median(z_values) - surface_z_m) * 1000.0),
            "z_sd_mm": float(np.std(z_values) * 1000.0),
            "force_g": float(np.median(forces)) if forces else None,
            "medians": np.median(matrix, axis=0),
            "target_values": matrix[:, int(target)],
        }
        if common_mode:
            window["localized_values"] = matrix[:, int(target)] - np.median(matrix[:, common_mode], axis=1)
        windows.append(window)
    return windows


def detection_distance(gaps, departures, threshold):
    """Largest gap from which the response stays at or above the threshold."""
    order = np.argsort(gaps)
    gaps = np.asarray(gaps, dtype=float)[order]
    departures = np.asarray(departures, dtype=float)[order]
    if not gaps.size or departures[0] < threshold:
        return {"distance_mm": None, "status": "not_detected_at_closest_gap"}
    last = 0
    while last + 1 < gaps.size and departures[last + 1] >= threshold:
        last += 1
    if last == gaps.size - 1:
        return {"distance_mm": float(gaps[last]), "status": "beyond_farthest_level"}
    near, far = departures[last], departures[last + 1]
    fraction = (near - threshold) / (near - far) if near > far else 0.0
    distance = gaps[last] + fraction * (gaps[last + 1] - gaps[last])
    return {
        "distance_mm": float(distance), "status": "interpolated",
        "bracket_mm": [float(gaps[last]), float(gaps[last + 1])],
    }


def _curve(levels, sweeps=None, key="departure"):
    """Mean departure per commanded gap, optionally from a subset of sweeps."""
    by_gap = defaultdict(list)
    for level in levels:
        if sweeps is None or level["sweep"] in sweeps:
            by_gap[level["target_gap_mm"]].append(level)
    gaps, means = [], []
    for gap in sorted(by_gap):
        gaps.append(float(np.mean([item["gap_mm"] for item in by_gap[gap]])))
        means.append(float(np.mean([item[key] for item in by_gap[gap]])))
    return np.asarray(gaps), np.asarray(means)


def _interpolation_error(times, values):
    """Leave-one-out error of predicting each interior reference from its neighbours.

    Removing a reference doubles the interpolation interval, so this bounds
    the drift error left between the references that were actually used.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.size < 3:
        return None
    errors = np.asarray([
        values[index] - np.interp(times[index], np.delete(times, index), np.delete(values, index))
        for index in range(1, times.size - 1)
    ])
    return {"rms": float(np.sqrt(np.mean(errors ** 2))), "max_abs": float(np.max(np.abs(errors)))}


def _detection_entries(levels, key, sigma, k_values, rng):
    """Range at each k·sigma from the mean curve and from every sweep separately."""
    sweeps = sorted({level["sweep"] for level in levels})
    gaps, means = _curve(levels, key=key)
    detection = {}
    for k_value in k_values:
        threshold = float(k_value) * sigma
        entry = {"k": float(k_value), "threshold": threshold, **detection_distance(gaps, means, threshold)}
        per_sweep, statuses = [], []
        for sweep in sweeps:
            crossing = detection_distance(*_curve(levels, {sweep}, key), threshold)
            if crossing["distance_mm"] is not None:
                per_sweep.append(crossing["distance_mm"])
                statuses.append(crossing["status"])
        entry["per_sweep_mm"] = per_sweep
        entry["per_sweep_status"] = statuses
        entry["per_sweep_mean_mm"] = float(np.mean(per_sweep)) if per_sweep else None
        entry["per_sweep_sd_mm"] = float(np.std(per_sweep, ddof=1)) if len(per_sweep) > 1 else None
        t_value = _t95(len(per_sweep) - 1)
        entry["per_sweep_ci95_mm"] = None
        if entry["per_sweep_sd_mm"] is not None and t_value is not None:
            half = t_value * entry["per_sweep_sd_mm"] / math.sqrt(len(per_sweep))
            entry["per_sweep_ci95_mm"] = [entry["per_sweep_mean_mm"] - half, entry["per_sweep_mean_mm"] + half]
        entry["bootstrap_ci95_mm"] = None
        if len(sweeps) >= MIN_BOOTSTRAP_SWEEPS:
            samples = []
            for _ in range(BOOTSTRAP_RESAMPLES):
                chosen = list(rng.choice(sweeps, size=len(sweeps), replace=True))
                boot_levels = [lv for s in chosen for lv in levels if lv["sweep"] == s]
                value = detection_distance(*_curve(boot_levels, key=key), threshold)["distance_mm"]
                if value is not None:
                    samples.append(value)
            if len(samples) >= 20:
                entry["bootstrap_ci95_mm"] = [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
        detection[f"{k_value:g}_sigma"] = entry
    return detection


def _frame_sigma(windows, key):
    """Robust single-frame noise of ``key`` within the stationary windows."""
    residuals = np.concatenate([w[key] - np.median(w[key]) for w in windows])
    sigma = _robust_sigma(residuals) or 0.0
    if sigma <= 0.0:
        sigma = float(np.std(residuals)) if residuals.size > 1 else 0.0
    return sigma


def analyse_proximity_profile(
    rows, *, rows_n, columns_n, target, signal_field="diff_ave",
    force_threshold_g=1.0, k_values=DEFAULT_K_VALUES, seed=0,
    common_mode_distance=DEFAULT_COMMON_MODE_DISTANCE,
):
    """Return the drift-corrected signal-versus-gap profile and its metrics."""
    rows = list(rows or [])
    taxel_count = int(rows_n) * int(columns_n)
    target = int(target)
    surface = surface_reference(rows, force_threshold_g)
    result = {
        "signal_field": str(signal_field),
        "target_taxel": target,
        "surface": surface,
        "warnings": [],
    }
    if surface["surface_z_m"] is None:
        result["warnings"].append("No surface contact was found; gaps cannot be referenced.")
        return result
    far, far_cells = common_mode_taxels(rows_n, columns_n, target, common_mode_distance)
    windows = stationary_windows(
        rows, signal_field, taxel_count, target, surface["surface_z_m"], common_mode=far or None,
    )
    references = [w for w in windows if w["kind"] == "reference"]
    levels = [w for w in windows if w["kind"] == "level" and w["target_gap_mm"] is not None]
    if not references or not levels:
        result["warnings"].append("Reference or level windows are missing.")
        return result

    ref_times = np.asarray([w["time_s"] for w in references])
    ref_medians = np.asarray([w["medians"] for w in references])
    ref_target = ref_medians[:, target]
    sigma = _frame_sigma(references, "target_values")
    local_sigma = _frame_sigma(references, "localized_values") if far else None
    others = [index for index in range(taxel_count) if index != target]
    for level in levels:
        baseline = np.asarray([
            np.interp(level["time_s"], ref_times, ref_medians[:, index]) for index in range(taxel_count)
        ])
        departure = baseline - level["medians"]
        level["departure"] = float(departure[target])
        strongest = float(np.max(departure[others])) if others else 0.0
        level["strongest_other"] = strongest
        level["dominance"] = level["departure"] / strongest if strongest > 1e-9 else None
        level["sigma"] = _robust_sigma(level["target_values"])
        if far:
            level["common_mode"] = float(np.median(departure[far]))
            level["localized_departure"] = level["departure"] - level["common_mode"]

    frames = int(np.median([level["frames"] for level in levels]))
    by_gap = defaultdict(list)
    for level in levels:
        by_gap[level["target_gap_mm"]].append(level)
    table = []
    for target_gap in sorted(by_gap):
        items = by_gap[target_gap]
        departures = np.asarray([item["departure"] for item in items])
        sd = float(np.std(departures, ddof=1)) if departures.size > 1 else None
        t_value = _t95(departures.size - 1)
        dominance = [item["dominance"] for item in items if item["dominance"] is not None]
        row = {
            "target_gap_mm": float(target_gap),
            "gap_mm": float(np.mean([item["gap_mm"] for item in items])),
            "gap_sd_mm": float(np.std([item["gap_mm"] for item in items])),
            "n": int(departures.size),
            "departure_mean": float(np.mean(departures)),
            "departure_sd": sd,
            "departure_ci95": None if sd is None or t_value is None else t_value * sd / math.sqrt(departures.size),
            "snr_single_frame": float(np.mean(departures) / sigma) if sigma > 0 else None,
            "dominance_median": float(np.median(dominance)) if dominance else None,
            "within_window_sigma": float(np.median([item["sigma"] for item in items if item["sigma"] is not None]))
            if any(item["sigma"] is not None for item in items) else None,
        }
        if far:
            local = np.asarray([item["localized_departure"] for item in items])
            row.update({
                "common_mode_mean": float(np.mean([item["common_mode"] for item in items])),
                "localized_mean": float(np.mean(local)),
                "localized_sd": float(np.std(local, ddof=1)) if local.size > 1 else None,
                "snr_localized": float(np.mean(local) / local_sigma) if local_sigma else None,
            })
        table.append(row)
    gaps = np.asarray([row["gap_mm"] for row in table])
    means = np.asarray([row["departure_mean"] for row in table])
    slopes = np.gradient(means, gaps) if gaps.size > 1 else np.zeros_like(gaps)
    for row, slope in zip(table, slopes):
        magnitude = abs(float(slope))
        row["sensitivity_per_mm"] = float(slope)
        row["resolution_single_frame_mm"] = sigma / magnitude if magnitude > 1e-9 and sigma > 0 else None
        row["resolution_window_mm"] = (
            sigma / math.sqrt(frames) / magnitude if magnitude > 1e-9 and sigma > 0 else None
        )
        row["resolution_repeatability_mm"] = (
            row["departure_sd"] / magnitude if magnitude > 1e-9 and row["departure_sd"] else None
        )

    sweeps = sorted({level["sweep"] for level in levels})
    detection = _detection_entries(levels, "departure", sigma, k_values, np.random.default_rng(seed))
    local_detection = (
        _detection_entries(levels, "localized_departure", local_sigma, k_values, np.random.default_rng(seed))
        if far else None
    )
    dominance_points = [(row["gap_mm"], row["dominance_median"]) for row in table if row["dominance_median"] is not None]
    dominance_range = {
        f"{level:g}": detection_distance(*zip(*dominance_points), level) for level in DOMINANCE_LEVELS
    } if dominance_points else {}

    hysteresis = None
    by_direction = {"down": defaultdict(list), "up": defaultdict(list)}
    for level in levels:
        bucket = by_direction.get(level["direction"])
        if bucket is not None:
            bucket[level["target_gap_mm"]].append(level["departure"])
    down, up = by_direction["down"], by_direction["up"]
    paired = [np.mean(down[gap]) - np.mean(up[gap]) for gap in down if gap in up]
    if paired:
        span = float(np.ptp(means)) if means.size else 0.0
        hysteresis = {
            "matched_gaps": len(paired),
            "mean_down_minus_up": float(np.mean(paired)),
            "max_abs": float(np.max(np.abs(paired))),
            "normalised_percent": 100.0 * float(np.mean(np.abs(paired))) / span if span > 0 else None,
        }

    drift_slope = (
        float(np.polyfit(ref_times, ref_target, 1)[0] * 60.0) if ref_times.size >= 2 and np.ptp(ref_times) > 0 else None
    )
    target_interpolation = _interpolation_error(ref_times, ref_target)
    result.update({
        "single_frame_sigma": sigma,
        "frames_per_level": frames,
        "sweeps": len(sweeps),
        "reference_gap_mm": float(np.median([w["gap_mm"] for w in references])),
        "reference_drift": {
            "windows": len(references),
            "range": float(np.ptp(ref_target)),
            "slope_per_min": drift_slope,
            "range_in_sigma": float(np.ptp(ref_target) / sigma) if sigma > 0 else None,
            "interpolation_error": target_interpolation,
        },
        "common_mode": {
            "available": bool(far),
            "requested_distance_cells": int(common_mode_distance),
            "distance_cells": far_cells,
            "taxels": len(far),
            "single_frame_sigma": local_sigma,
            "interpolation_error": (
                _interpolation_error(ref_times, [np.median(w["localized_values"]) for w in references])
                if far else None
            ),
            "interpretation": (
                "Localized signal = target minus the median of taxels at least the stated number of "
                "cells away, frame by frame. It removes drift and coupling shared by the whole sensor "
                "(for example the robot arm and tool), leaving proximity specific to the target taxel."
            ),
        },
        "levels": [
            {key: value for key, value in level.items() if key not in {"medians", "target_values", "localized_values"}}
            for level in levels
        ],
        "profile": table,
        "detection": detection,
        "detection_localized": local_detection,
        "dominance_range": dominance_range,
        "hysteresis": hysteresis,
    })
    warnings = result["warnings"]
    if surface["surface_sd_mm"] is not None and surface["surface_sd_mm"] > 0.05:
        warnings.append(
            f"Surface probes disagree by {surface['surface_sd_mm']:.3f} mm (SD); small gaps are uncertain."
        )
    if surface["final_shift_mm"] is not None and abs(surface["final_shift_mm"]) > 0.05:
        warnings.append(
            f"The surface moved {surface['final_shift_mm']:+.3f} mm between the start and the end."
        )
    if result["reference_drift"]["range_in_sigma"] and result["reference_drift"]["range_in_sigma"] > 3:
        warnings.append(
            "Far-field baseline drifted by more than 3 sigma; drift correction was essential."
        )
    if target_interpolation and sigma > 0 and target_interpolation["rms"] > sigma:
        warnings.append(
            f"Drift between references ({target_interpolation['rms']:.1f} counts RMS) exceeds single-frame "
            "noise, so the target-only range is drift-limited. "
            + ("Use the localized range." if far else "Revisit the far reference more often.")
        )
    if not far:
        warnings.append(
            f"Fewer than {MIN_COMMON_MODE_TAXELS} taxels lie {MIN_COMMON_MODE_DISTANCE}+ cells from the target; "
            "no localized (common-mode-rejected) range."
        )
    if len(sweeps) < MIN_BOOTSTRAP_SWEEPS:
        warnings.append(
            f"Only {len(sweeps)} sweep(s): range uncertainty is the Student-t interval of the per-sweep "
            f"distances. Use at least {MIN_BOOTSTRAP_SWEEPS} sweeps for a usable interval."
        )
    for name, entries in (("Localized", local_detection), ("Target-only", None if far else detection)):
        for entry in (entries or {}).values():
            bracket = entry.get("bracket_mm")
            if bracket and bracket[1] - bracket[0] > max(2.0, 0.15 * entry["distance_mm"]):
                warnings.append(
                    f"{name} {entry['k']:g} sigma range ({entry['distance_mm']:.1f} mm) lies between levels "
                    f"{bracket[0]:.1f} and {bracket[1]:.1f} mm; add levels there for a precise value."
                )
    return result


def _detection_line(label, entry):
    text = f"{label} at {entry['k']:g} sigma ({entry['threshold']:.2f} counts): "
    if entry["distance_mm"] is None:
        return text + "not detected even at the closest gap."
    text += f"{entry['distance_mm']:.2f} mm"
    if entry["status"] == "beyond_farthest_level":
        text += " or more (still detected at the farthest level)"
    per_sweep = entry.get("per_sweep_mm") or []
    if per_sweep:
        text += "; per sweep " + ", ".join(f"{value:.1f}" for value in per_sweep) + " mm"
    ci = entry.get("per_sweep_ci95_mm")
    if ci:
        text += f" (mean {entry['per_sweep_mean_mm']:.2f}, 95% CI {ci[0]:.2f}–{ci[1]:.2f}, SD {entry['per_sweep_sd_mm']:.2f})"
    bootstrap = entry.get("bootstrap_ci95_mm")
    if bootstrap:
        text += f"; bootstrap 95% CI {bootstrap[0]:.2f}–{bootstrap[1]:.2f} mm"
    return text + "."


def report_lines(analysis):
    """Plain-text summary for the sidebar and the result window."""
    surface = analysis.get("surface") or {}
    lines = ["PROXIMITY PROFILE", "=" * 60]

    def mm(value, digits=3):
        return "n/a" if value is None else f"{value:.{digits}f} mm"

    lines.append(
        f"Surface Z: {mm(None if surface.get('surface_z_m') is None else surface['surface_z_m'] * 1000.0)} "
        f"(SD {mm(surface.get('surface_sd_mm'))} over the initial probes; "
        f"end shift {mm(surface.get('final_shift_mm'))}; "
        f"zero-force offset {mm(surface.get('zero_force_offset_mm'))})."
    )
    probes = [p for p in surface.get("probes") or [] if p.get("force_zero_g") is not None]
    if probes:
        lines.append(
            "Light contact per probe (threshold above that probe's force zero): "
            + "; ".join(
                f"{'end ' if p['final'] else ''}#{p['probe'] + 1} {p['contact_threshold_g']:.2f} g "
                f"above {p['force_zero_g']:+.2f} g"
                for p in probes
            )
            + "."
        )
    if "profile" not in analysis:
        lines.extend(f"Warning: {warning}" for warning in analysis.get("warnings", []))
        return lines
    sigma = analysis["single_frame_sigma"]
    drift = analysis["reference_drift"]
    lines.append(
        f"Single-frame noise sigma: {sigma:.3f} counts; {analysis['sweeps']} sweep(s), "
        f"{analysis['frames_per_level']} frames per level, reference gap "
        f"{analysis['reference_gap_mm']:.1f} mm."
    )
    interpolation = drift.get("interpolation_error")
    lines.append(
        f"Far-field drift: range {drift['range']:.2f} counts over {drift['windows']} reference windows"
        + (f", {drift['slope_per_min']:+.3f} counts/min" if drift["slope_per_min"] is not None else "")
        + (
            f"; leave-one-out reference interpolation error {interpolation['rms']:.2f} counts RMS "
            f"(max {interpolation['max_abs']:.2f})." if interpolation else "."
        )
    )
    common = analysis.get("common_mode") or {}
    if common.get("available"):
        local_interpolation = common.get("interpolation_error")
        lines.append(
            f"Common-mode rejection: median of {common['taxels']} taxels at least {common['distance_cells']} "
            f"cells away; localized single-frame sigma {common['single_frame_sigma']:.3f} counts"
            + (
                f", reference interpolation error {local_interpolation['rms']:.2f} counts RMS."
                if local_interpolation else "."
            )
        )
    for entry in (analysis.get("detection_localized") or {}).values():
        lines.append(_detection_line("Localized detection distance", entry))
    for entry in analysis["detection"].values():
        lines.append(_detection_line("Detection distance", entry))
    dominance = analysis.get("dominance_range") or {}
    if dominance:
        parts = []
        for level, crossing in dominance.items():
            if crossing["distance_mm"] is None:
                parts.append(f"never {level}× at the closest gap")
            else:
                more = " or more" if crossing["status"] == "beyond_farthest_level" else ""
                parts.append(f"{level}× up to {crossing['distance_mm']:.1f} mm{more}")
        lines.append("Target strongest taxel (dominance): " + "; ".join(parts) + ".")
    lines.append("")
    localized = bool(common.get("available"))
    lines.append(
        "gap(mm)  drop(mean±SD)  "
        + ("local(mean±SD)  " if localized else "")
        + "SNR   dS/dgap   resolution: 1 frame / window / repeat (mm)  dominance"
    )
    for row in analysis["profile"]:
        def value(item, digits=3):
            return "  n/a " if item is None else f"{item:.{digits}f}"
        local = (
            f"{row['localized_mean']:7.2f}±{value(row['localized_sd'], 2):<6}  " if localized else ""
        )
        lines.append(
            f"{row['gap_mm']:7.2f}  {row['departure_mean']:7.2f}±{value(row['departure_sd'], 2):<6}  {local}"
            f"{value(row['snr_single_frame'], 1):>5}  {row['sensitivity_per_mm']:+8.2f}   "
            f"{value(row['resolution_single_frame_mm'])} / {value(row['resolution_window_mm'])} / "
            f"{value(row['resolution_repeatability_mm'])}   {value(row['dominance_median'], 2)}"
        )
    if analysis.get("hysteresis"):
        hysteresis = analysis["hysteresis"]
        lines.append(
            f"Down-minus-up sweep difference: mean {hysteresis['mean_down_minus_up']:+.2f} counts, "
            f"max |{hysteresis['max_abs']:.2f}| over {hysteresis['matched_gaps']} gaps."
        )
    lines.extend(f"Warning: {warning}" for warning in analysis.get("warnings", []))
    lines.extend([
        "",
        "Gap = TCP Z minus the light-contact surface Z (robot frame), not an independent air-gap "
        "measurement. Detection uses single-frame noise of the drift-corrected signal. The localized "
        "range uses the target minus distant taxels, so drift and coupling shared by the whole sensor "
        "(robot arm, tool) do not count; the target-only range includes them. The per-sweep interval "
        "is the spread of independent sweeps. Resolution is noise divided by local slope; it is the "
        "smallest gap change distinguishable at that gap.",
    ])
    return lines


def save_profile_figure(analysis, path):
    """Signal drop, distance resolution, neighbour dominance and per-sweep range."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(16, 4.2), dpi=150)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 4)
    table = analysis.get("profile") or []
    if not table:
        figure.text(0.5, 0.5, "No profile levels", ha="center")
        figure.savefig(path)
        return
    gaps = np.asarray([row["gap_mm"] for row in table])
    means = np.asarray([row["departure_mean"] for row in table])
    errors = np.asarray([row["departure_sd"] or 0.0 for row in table])
    for sweep in sorted({level["sweep"] for level in analysis["levels"]}):
        items = sorted((lv["gap_mm"], lv["departure"]) for lv in analysis["levels"] if lv["sweep"] == sweep)
        axes[0].plot(*zip(*items), color="0.75", linewidth=0.8)
    axes[0].errorbar(gaps, means, yerr=errors, fmt="o-", color="#0F8B8D", capsize=3, label="target ± SD")
    localized = analysis.get("detection_localized") or {}
    if localized:
        local_errors = np.asarray([row["localized_sd"] or 0.0 for row in table])
        axes[0].errorbar(
            gaps, [row["localized_mean"] for row in table], yerr=local_errors, fmt="s-", markersize=3,
            color="#6A4C93", capsize=2, linewidth=1, label="localized ± SD",
        )
    for name, series, colours in (
        ("target", analysis["detection"], ("#E07A1F", "#B23A48")),
        ("localized", localized, ("#9C7BC4", "#6A4C93")),
    ):
        for colour, entry in zip(colours, series.values()):
            axes[0].axhline(entry["threshold"], color=colour, linestyle="--", linewidth=0.8,
                            label=f"{name} {entry['k']:g}σ = {entry['threshold']:.1f}")
            if entry["distance_mm"] is not None:
                axes[0].axvline(entry["distance_mm"], color=colour, linestyle=":", linewidth=1)
    axes[0].set_xscale("log")
    # Linear below 10 counts keeps the threshold crossings visible next to large near-field drops.
    axes[0].set_yscale("symlog", linthresh=10.0)
    axes[0].set_xlabel("Gap above surface (mm)")
    axes[0].set_ylabel("Signal drop vs far field (counts)")
    axes[0].legend(fontsize=7)
    for key, label in (
        ("resolution_single_frame_mm", "single frame"),
        ("resolution_window_mm", "window median"),
        ("resolution_repeatability_mm", "sweep repeatability"),
    ):
        points = [(row["gap_mm"], row[key]) for row in table if row[key] is not None]
        if points:
            axes[1].plot(*zip(*points), "o-", label=label)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Gap above surface (mm)")
    axes[1].set_ylabel("Distance resolution (mm)")
    axes[1].legend(fontsize=8)
    points = [(row["gap_mm"], row["dominance_median"]) for row in table if row["dominance_median"] is not None]
    if points:
        axes[2].plot(*zip(*points), "o-", color="#142B4C")
    axes[2].axhline(1.0, color="0.6", linewidth=1)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Gap above surface (mm)")
    axes[2].set_ylabel("Target / strongest other taxel")
    from matplotlib.ticker import NullFormatter

    ticks = [value for value in (0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if gaps.min() * 0.9 <= value <= gaps.max() * 1.1]
    for axis in axes[:3]:
        axis.grid(alpha=0.3, which="both")
        axis.set_xticks(ticks)
        axis.set_xticklabels([f"{value:g}" for value in ticks])
        axis.xaxis.set_minor_formatter(NullFormatter())
    labels = []
    series = [("target", analysis["detection"], ("#E07A1F", "#B23A48")), ("localized", localized, ("#9C7BC4", "#6A4C93"))]
    for name, entries, colours in series:
        for colour, entry in zip(colours, entries.values()):
            position = len(labels)
            labels.append(f"{name}\n{entry['k']:g}σ")
            values = entry.get("per_sweep_mm") or []
            if values:
                offsets = np.linspace(-0.15, 0.15, len(values)) if len(values) > 1 else [0.0]
                axes[3].plot(position + np.asarray(offsets), values, "o", color=colour, alpha=0.6, markersize=4)
            ci = entry.get("per_sweep_ci95_mm")
            if ci:
                mean = entry["per_sweep_mean_mm"]
                axes[3].errorbar([position + 0.3], [mean], yerr=[[mean - ci[0]], [ci[1] - mean]],
                                 fmt="D", color=colour, capsize=3, markersize=4)
    axes[3].set_xticks(range(len(labels)))
    axes[3].set_xticklabels(labels, fontsize=8)
    axes[3].set_xlim(-0.5, len(labels) - 0.5)
    axes[3].set_ylabel("Detection distance per sweep (mm)")
    axes[3].set_title("dots: sweeps · ◆: mean ± 95% CI", fontsize=8)
    axes[3].grid(alpha=0.3, axis="y")
    figure.suptitle(f"Proximity profile · taxel {analysis['target_taxel']} · {analysis['signal_field']}")
    figure.tight_layout()
    figure.savefig(path)


def save_profile_outputs(analysis, directory, stem):
    """Write levels CSV, per-gap CSV, JSON and PNG beside the trial files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"json": directory / f"{stem}_proximity_profile.json"}
    if analysis.get("levels"):
        paths["levels_csv"] = directory / f"{stem}_proximity_levels.csv"
        with paths["levels_csv"].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(analysis["levels"][0].keys()))
            writer.writeheader()
            writer.writerows(analysis["levels"])
        paths["profile_csv"] = directory / f"{stem}_proximity_profile.csv"
        with paths["profile_csv"].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(analysis["profile"][0].keys()))
            writer.writeheader()
            writer.writerows(analysis["profile"])
        paths["png"] = directory / f"{stem}_proximity_profile.png"
        save_profile_figure(analysis, paths["png"])
    paths = {key: str(value) for key, value in paths.items()}
    with Path(paths["json"]).open("w", encoding="utf-8") as stream:
        json.dump({**analysis, "files": paths}, stream, indent=2)
        stream.write("\n")
    return paths
