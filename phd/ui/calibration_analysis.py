"""Analysis helpers for force, tactile-signal, and travel calibration data."""

from __future__ import annotations

import math

import numpy as np


APPROACH_PHASES = frozenset({"approaching", "contact_hold"})
RETURN_PHASE = "returning"


def taxel_signal_column(signal_field, taxel_index):
    """Return the stable CSV column name for one taxel signal."""
    return f"taxel_{int(taxel_index):03d}_{str(signal_field)}"


def available_taxel_indices(rows, signal_field, legacy_index=None):
    """Return taxels recorded for ``signal_field``, including legacy CSVs."""
    prefix = "taxel_"
    suffix = f"_{str(signal_field)}"
    indices = set()
    for row in list(rows or [])[:3]:
        for key in row:
            key = str(key)
            if not key.startswith(prefix) or not key.endswith(suffix):
                continue
            number = key[len(prefix) : -len(suffix)]
            if number.isdigit():
                indices.add(int(number))
    if indices:
        return sorted(indices)
    if legacy_index is not None and any(
        _finite_value(row, signal_field) is not None for row in rows or []
    ):
        return [int(legacy_index)]
    return []


def calibration_rows_for_taxel(
    rows,
    signal_field,
    taxel_index,
    legacy_index=None,
):
    """Project shared trial rows onto one recorded taxel signal."""
    rows = list(rows or [])
    signal_field = str(signal_field)
    taxel_index = int(taxel_index)
    column = taxel_signal_column(signal_field, taxel_index)
    if any(column in row for row in rows):
        projected = []
        for row in rows:
            item = dict(row)
            item[signal_field] = row.get(column)
            item["cell_index"] = taxel_index
            projected.append(item)
        return projected
    if legacy_index is not None and taxel_index == int(legacy_index):
        return rows
    return []


def _finite_value(row, field):
    try:
        value = float(row.get(field))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _linear_fit(points):
    """Return a serializable least-squares line fit for ``(x, y)`` points."""
    finite = []
    for x_value, y_value in points:
        try:
            x_value = float(x_value)
            y_value = float(y_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x_value) and math.isfinite(y_value):
            finite.append((x_value, y_value))
    if len(finite) < 2:
        return None
    x_values = np.asarray([point[0] for point in finite], dtype=float)
    y_values = np.asarray([point[1] for point in finite], dtype=float)
    if float(np.ptp(x_values)) <= 1e-12:
        return None
    slope, intercept = np.polyfit(x_values, y_values, 1)
    predicted = slope * x_values + intercept
    residuals = y_values - predicted
    residual_sum = float(np.sum(residuals ** 2))
    total_sum = float(np.sum((y_values - np.mean(y_values)) ** 2))
    r_squared = 1.0 if total_sum <= 1e-12 else 1.0 - residual_sum / total_sum
    if float(np.std(x_values)) <= 1e-12 or float(np.std(y_values)) <= 1e-12:
        pearson_r = 0.0
    else:
        pearson_r = float(np.corrcoef(x_values, y_values)[0, 1])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "sample_count": len(finite),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
        "max_abs_error": float(np.max(np.abs(residuals))),
        "pearson_r": pearson_r,
        "x_min": float(np.min(x_values)),
        "x_max": float(np.max(x_values)),
    }


def _first_sustained_position(values, threshold, count=2):
    required = max(1, int(count))
    run_start = None
    run_length = 0
    for position, value in enumerate(values):
        if value >= threshold:
            if run_start is None:
                run_start = position
            run_length += 1
            if run_length >= required:
                return run_start
        else:
            run_start = None
            run_length = 0
    return None


def analyze_calibration_rows(rows, signal_field="diff_ave"):
    """Detect proximity/contact onsets and fit the three main relationships.

    The proximity onset is based on signed signal departure from its initial
    approach baseline. Contact onset uses absolute force change in grams. Fits
    deliberately use loading/approach samples only so return hysteresis does
    not distort the reported equations.
    """
    rows = list(rows or [])
    signal_field = str(signal_field or "diff_ave")
    indexed_approach = [
        (index, row)
        for index, row in enumerate(rows)
        if str(row.get("phase", "")) in APPROACH_PHASES
        and _finite_value(row, "travel_down_mm") is not None
        and _finite_value(row, signal_field) is not None
        and _finite_value(row, "force_delta_abs_g") is not None
    ]
    if not indexed_approach:
        return {
            "signal_field": signal_field,
            "signal_onset": None,
            "contact_onset": None,
            "signal_distance_fit": None,
            "force_distance_fit": None,
            "force_signal_fit": None,
        }

    approach_rows = [item[1] for item in indexed_approach]
    signals = np.asarray(
        [_finite_value(row, signal_field) for row in approach_rows], dtype=float
    )
    forces_g = np.asarray(
        [_finite_value(row, "force_delta_abs_g") for row in approach_rows],
        dtype=float,
    )
    distances = np.asarray(
        [_finite_value(row, "travel_down_mm") for row in approach_rows],
        dtype=float,
    )

    baseline_count = min(len(signals), max(3, min(10, len(signals) // 5)))
    baseline_values = signals[:baseline_count]
    signal_baseline = float(np.median(baseline_values))
    median_deviation = float(
        np.median(np.abs(baseline_values - signal_baseline))
    )
    signal_threshold = max(1.0, 4.0 * 1.4826 * median_deviation)
    signal_departure = np.abs(signals - signal_baseline)
    signal_position = _first_sustained_position(
        signal_departure,
        signal_threshold,
        count=2,
    )

    peak_force_g = float(np.max(forces_g))
    contact_threshold_g = max(1.0, min(10.0, peak_force_g * 0.03))
    contact_position = _first_sustained_position(
        forces_g,
        contact_threshold_g,
        count=2,
    )
    if contact_position is None:
        crossing = np.flatnonzero(forces_g >= contact_threshold_g)
        contact_position = int(crossing[0]) if crossing.size else None

    def onset_payload(position):
        if position is None:
            return None
        row = approach_rows[int(position)]
        return {
            "row_index": int(indexed_approach[int(position)][0]),
            "elapsed_s": _finite_value(row, "elapsed_s"),
            "distance_mm": _finite_value(row, "travel_down_mm"),
            "signal": _finite_value(row, signal_field),
            "force_g": _finite_value(row, "force_delta_abs_g"),
        }

    precontact_end = (
        int(contact_position)
        if contact_position is not None
        else len(approach_rows)
    )
    signal_fit_start = int(signal_position or 0)
    signal_fit_rows = approach_rows[signal_fit_start:precontact_end]
    if len(signal_fit_rows) < 3:
        signal_fit_rows = approach_rows[:precontact_end]
    signal_distance_points = [
        (
            _finite_value(row, "travel_down_mm"),
            _finite_value(row, signal_field),
        )
        for row in signal_fit_rows
    ]

    contact_fit_start = int(contact_position or 0)
    contact_rows = approach_rows[contact_fit_start:]
    force_distance_points = [
        (
            _finite_value(row, "travel_down_mm"),
            _finite_value(row, "force_delta_abs_g"),
        )
        for row in contact_rows
    ]
    force_signal_points = [
        (
            _finite_value(row, signal_field),
            _finite_value(row, "force_delta_abs_g"),
        )
        for row in contact_rows
    ]

    return {
        "signal_field": signal_field,
        "signal_baseline": signal_baseline,
        "signal_change_threshold": signal_threshold,
        "contact_detection_threshold_g": contact_threshold_g,
        "peak_force_g": peak_force_g,
        "max_distance_mm": float(np.max(distances)),
        "signal_onset": onset_payload(signal_position),
        "contact_onset": onset_payload(contact_position),
        "signal_distance_fit": _linear_fit(signal_distance_points),
        "force_distance_fit": _linear_fit(force_distance_points),
        "force_signal_fit": _linear_fit(force_signal_points),
    }
