"""Analysis helpers for force, tactile-signal, and travel calibration data."""

from __future__ import annotations

import csv
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from phd.ui.experiment_quality import downward_detector, rolling_medians, unique_sensor_rows


APPROACH_PHASES = frozenset({"approaching"})
RETURN_PHASE = "returning"
# A response that starts within this travel of the start pose was probably
# already present there, so the measured range is only a lower bound.
START_INSIDE_RANGE_MM = 1.0
UNAVERAGED_FIELDS = {
    "diff_ave": "diff",
    "diff_percent_ave": "diff_percent",
    "raw_ave": "raw",
}
_TAXEL_COLUMN = re.compile(r"taxel_\d+_")


@lru_cache(maxsize=8192)
def taxel_signal_column(signal_field, taxel_index):
    """Return the stable CSV column name for one taxel signal.

    Cached so the thousands of recorded rows share one key string per column.
    """
    return f"taxel_{int(taxel_index):03d}_{str(signal_field)}"


def coerce_csv_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    return number if math.isfinite(number) else None


def read_calibration_csv_rows(path, *, taxel_columns=True):
    """Parse a saved trial CSV; ``taxel_columns=False`` skips the per-taxel grid."""
    with Path(path).expanduser().open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        keep = [
            (position, name) for position, name in enumerate(header)
            if taxel_columns or not _TAXEL_COLUMN.match(name)
        ]
        return [
            {
                name: coerce_csv_value(values[position]) if position < len(values) else None
                for position, name in keep
            }
            for values in reader
            if values
        ]


def release_trial_rows(result):
    """Drop the samples of saved trials from memory; ``with_trial_rows`` reloads them."""
    trials = list(result.get("trial_results") or [])
    for trial in trials or [result]:
        if trial.get("rows") and trial.get("csv_path"):
            trial["row_count"] = len(trial["rows"])
            trial["rows"] = []
    if trials and trials[-1].get("csv_path"):
        result["rows"] = []


def with_trial_rows(result, *, taxel_columns=True):
    """Copy of ``result`` whose trials carry their samples, reloaded from CSV if released."""
    loaded = dict(result)
    trials = result.get("trial_results")
    if trials:
        loaded["trial_results"] = [
            with_trial_rows(trial, taxel_columns=taxel_columns) for trial in trials
        ]
        loaded["rows"] = loaded["trial_results"][-1].get("rows") or []
        return loaded
    path = result.get("csv_path")
    if not result.get("rows") and path and Path(path).is_file():
        loaded["rows"] = read_calibration_csv_rows(path, taxel_columns=taxel_columns)
    return loaded


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
            for field in ("raw", "calibration", "diff", "diff_ave", "diff_percent_ave"):
                key = taxel_signal_column(field, taxel_index)
                if key in row:
                    item[field] = row[key]
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


def _smooth_polynomial_fit(points, maximum_degree=2):
    """Fit a stable low-order loading curve in the original axis units."""
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
    unique_x = np.unique(x_values)
    if unique_x.size < 2 or float(np.ptp(x_values)) <= 1e-12:
        return None
    degree = min(
        max(1, int(maximum_degree)),
        int(unique_x.size) - 1,
        len(finite) - 1,
    )
    if degree > 1 and len(finite) < 5:
        degree = 1
    try:
        polynomial = np.polynomial.Polynomial.fit(
            x_values,
            y_values,
            degree,
        ).convert()
    except (ValueError, np.linalg.LinAlgError):
        return None
    coefficients = np.asarray(polynomial.coef, dtype=float)
    predicted = polynomial(x_values)
    residuals = y_values - predicted
    residual_sum = float(np.sum(residuals ** 2))
    total_sum = float(np.sum((y_values - np.mean(y_values)) ** 2))
    r_squared = 1.0 if total_sum <= 1e-12 else 1.0 - residual_sum / total_sum
    return {
        "degree": int(degree),
        "coefficients": [float(value) for value in coefficients],
        "r_squared": float(r_squared),
        "sample_count": len(finite),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
        "max_abs_error": float(np.max(np.abs(residuals))),
        "x_min": float(np.min(x_values)),
        "x_max": float(np.max(x_values)),
    }


def _first_sustained_position(values, threshold, count=2, times=None, duration=0.0):
    required = max(1, int(count))
    run_start = None
    run_length = 0
    intervals = [] if times is None else [
        b - a for a, b in zip(times, times[1:]) if a is not None and b is not None and b > a
    ]
    max_gap = max(0.25, 3 * float(np.median(intervals))) if intervals else 0.25
    for position, value in enumerate(values):
        if times is not None and position and (
            times[position] is None or times[position - 1] is None or
            times[position] - times[position - 1] > max_gap
        ):
            run_start, run_length = None, 0
        if value >= threshold:
            if run_start is None:
                run_start = position
            run_length += 1
            if run_length >= required and (
                times is None or duration <= 0 or
                (times[position] is not None and times[run_start] is not None
                 and times[position] - times[run_start] >= duration)
            ):
                return run_start
        else:
            run_start = None
            run_length = 0
    return None


def _centred_median(values, window=7):
    """Offline zero-lag median smoothing; values must be finite."""
    values = np.asarray(values, dtype=float)
    half = max(0, int(window) // 2)
    return np.asarray([
        np.median(values[max(0, index - half):index + half + 1])
        for index in range(len(values))
    ])


def _start_gradient_change(signals, distances, contact_position):
    """Signal drop across the first approach millimetres (0 when unknown)."""
    start = float(distances[0])
    end = float(distances[contact_position]) if contact_position is not None else float(distances[-1])
    window = min(2.0, 0.2 * (end - start))
    if window < 0.5:
        return 0.0
    early = signals[distances <= start + 0.25 * window]
    late = signals[(distances >= start + 0.75 * window) & (distances <= start + window)]
    if early.size < 3 or late.size < 3:
        return 0.0
    return float(np.median(early) - np.median(late))


def _proximity_range(distances, signal_position, contact_position, start_change=0.0, threshold=np.inf):
    """Robot travel from first proximity response to first detected force."""
    start = float(distances[0])
    onset = None if signal_position is None else float(distances[signal_position])
    contact = None if contact_position is None else float(distances[contact_position])
    payload = {
        "distance_above_contact_mm": None,
        "lower_bound_mm": None,
        "onset_travel_mm": onset,
        "start_to_contact_mm": None if contact is None else contact - start,
        "start_signal_change": float(start_change),
        "status": "measured",
        "interpretation": (
            "Robot travel between the first sustained proximity response and "
            "the first sustained force change. Very light fabric contact can "
            "precede the force threshold, so this is not an independent air-gap "
            "measurement."
        ),
    }
    if contact is None:
        payload["status"] = "contact_not_reached"
    elif onset is None:
        payload["status"] = "proximity_not_detected"
    elif onset - start < START_INSIDE_RANGE_MM or start_change >= threshold:
        payload["status"] = "start_inside_range"
        payload["lower_bound_mm"] = contact - start
    else:
        payload["distance_above_contact_mm"] = contact - onset
    return payload


def _turning_point(
    signals, forces_g, distances, contact_position, baseline, sigma, threshold, percent
):
    """Locate the signal minimum where the proximity drop turns into a rise."""
    if contact_position is None or len(signals) < 5:
        return {"detected": False, "reason": "contact_not_reached"}
    smoothed = _centred_median(signals)
    index = int(np.argmin(smoothed))
    drop = float(baseline - smoothed[index])
    rise = float(np.max(smoothed[index:]) - smoothed[index])
    tolerance = max(2.0 * float(sigma), 0.02 * drop, 0.01 if percent else 1.0)
    low = high = index
    while low > 0 and smoothed[low - 1] <= smoothed[index] + tolerance:
        low -= 1
    while high < len(smoothed) - 1 and smoothed[high + 1] <= smoothed[index] + tolerance:
        high += 1
    # Median smoothing shifts a sharp asymmetric minimum towards its shallower
    # side; the raw minimum inside the smoothed flat band does not.
    index = low + int(np.argmin(signals[low:high + 1]))
    candidate = {
        "travel_mm": float(distances[index]),
        "after_contact_mm": float(distances[index] - distances[contact_position]),
        "force_g": float(forces_g[index]),
        "signal": float(smoothed[index]),
        "drop_from_baseline": drop,
        "rise_after_turning": rise,
    }
    if drop <= float(threshold):
        return {"detected": False, "reason": "no_signal_drop", "candidate": candidate}
    if rise < max(float(threshold), 0.1 * drop):
        return {"detected": False, "reason": "no_rise_within_loading", "candidate": candidate}
    band = forces_g[low:high + 1]
    return {
        "detected": True,
        "reason": "",
        "approach_index": index,
        **candidate,
        "force_low_g": float(np.min(band)),
        "force_high_g": float(np.max(band)),
        "flat_band_tolerance": tolerance,
        "before_detected_contact": index < contact_position,
        "interpretation": (
            "Minimum of the zero-lag median-smoothed loading signal. The force "
            "band covers samples within the stated tolerance of that minimum."
        ),
    }


def _baseline_return(smoothed_after, forces_g, distances, start, contact_position, baseline):
    """First loading sample after the turning point that is back at baseline."""
    for offset, value in enumerate(smoothed_after):
        if value >= baseline:
            index = start + offset
            return {
                "reached": True,
                "force_g": float(forces_g[index]),
                "travel_mm": float(distances[index]),
                "after_contact_mm": float(distances[index] - distances[contact_position]),
                "interpretation": (
                    "Below this force the reading overlaps the drop caused by "
                    "proximity/light contact, so signal alone is ambiguous."
                ),
            }
    return {"reached": False}


def _unaveraged_turning(rows, signal_field, distances, forces_g, contact_position):
    """Cross-check the turning location on the unaveraged signal, if recorded."""
    field = UNAVERAGED_FIELDS.get(signal_field)
    if field is None or contact_position is None:
        return None
    values = [_finite_value(row, field) for row in rows]
    if len(values) < 5 or any(value is None for value in values):
        return None
    index = int(np.argmin(_centred_median(values)))
    return {
        "signal_field": field,
        "travel_mm": float(distances[index]),
        "after_contact_mm": float(distances[index] - distances[contact_position]),
        "force_g": float(forces_g[index]),
    }


def analyze_calibration_rows(rows, signal_field="diff_ave"):
    """Detect proximity/contact onsets and fit the three main relationships.

    The proximity onset is based on signed signal departure from its initial
    approach baseline. Contact onset uses absolute force change in grams. Fits
    deliberately use loading/approach samples only so return hysteresis does
    not distort the reported equations.

    The proximity range, the drop-to-rise turning point and the force at which
    the signal returns to baseline are reported separately. Force-signal is
    also fitted separately on the drop branch (contact to turning point) and
    the rise branch (turning point onwards), because one line cannot describe
    a non-monotonic response.
    """
    original_rows = list(rows or [])
    rows = unique_sensor_rows(original_rows)
    original_indices = {id(row): index for index, row in enumerate(original_rows)}
    signal_field = str(signal_field or "diff_ave")
    # Stepwise trials: use only readings taken at rest after each step.
    at_rest_only = any(
        str(row.get("step_state", "")) == "dwell"
        for row in rows if str(row.get("phase", "")) in APPROACH_PHASES
    )
    indexed_approach = [
        (original_indices[id(row)], row)
        for index, row in enumerate(rows)
        if str(row.get("phase", "")) in APPROACH_PHASES
        and (not at_rest_only or str(row.get("step_state", "")) == "dwell")
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
            "signal_distance_smooth_fit": None,
            "force_distance_smooth_fit": None,
            "force_signal_smooth_fit": None,
            "segmented_smooth_fits": {},
            "proximity_range": None,
            "turning_point": {"detected": False, "reason": "no_loading_samples"},
            "baseline_return": {"reached": False},
            "force_signal_drop_fit": None,
            "force_signal_rise_fit": None,
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

    baseline_rows = [r for r in rows if r.get("phase") == "force_baseline"]
    baseline_values = [_finite_value(r, signal_field) for r in baseline_rows]
    baseline_values = [v for v in baseline_values if v is not None]
    measured_baseline = len(baseline_values) >= 20
    if not measured_baseline:
        baseline_count = min(len(signals), max(3, min(10, len(signals) // 5)))
        baseline_values = signals[:baseline_count]
    percent = "percent" in signal_field
    detector = downward_detector(
        baseline_values, 7, 0.1 if percent else 6.0, 0.01 if percent else 1.0
    )
    signal_baseline = detector["centre"]
    signal_threshold = detector["margin"]
    # Causal window; mark the first qualifying sample of a sustained run.
    smoothed = np.asarray([
        np.median(signals[max(0, i - 6):i + 1]) for i in range(len(signals))
    ])
    signal_departure = signal_baseline - smoothed
    times = [_finite_value(r, "elapsed_s") for r in approach_rows]
    signal_position = _first_sustained_position(
        signal_departure,
        signal_threshold,
        count=3,
        times=times,
        duration=0.1,
    )

    peak_force_g = max(
        (_finite_value(r, "force_delta_abs_g") or 0.0 for r in rows), default=0.0
    )
    baseline_force = np.asarray([
        _finite_value(r, "force_delta_g") for r in baseline_rows
        if _finite_value(r, "force_delta_g") is not None
    ])
    force_noise = (
        float(1.4826 * np.median(np.abs(baseline_force - np.median(baseline_force))))
        if baseline_force.size else 0.0
    )
    contact_threshold_g = max(
        1.0, float(np.percentile(np.abs(baseline_force), 99)) + 3 * force_noise
        if baseline_force.size else 1.0
    )
    contact_position = _first_sustained_position(
        forces_g,
        contact_threshold_g,
        count=2,
        times=times,
        duration=0.05,
    )
    if signal_position is not None and contact_position is not None and signal_position >= contact_position:
        signal_position = None  # A contact response is not evidence of proximity.

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
    if signal_position is None:
        signal_fit_rows = []
    signal_distance_points = [
        (
            _finite_value(row, "travel_down_mm"),
            _finite_value(row, signal_field),
        )
        for row in signal_fit_rows
    ]

    contact_fit_start = int(contact_position or 0)
    contact_rows = approach_rows[contact_fit_start:] if contact_position is not None else []
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

    proximity_start = int(signal_position or 0)
    proximity_end = (
        int(contact_position)
        if contact_position is not None
        else len(approach_rows)
    )
    segmented_rows = {
        "proximity": approach_rows[proximity_start:proximity_end] if signal_position is not None else [],
        "contact": (
            approach_rows[int(contact_position) :]
            if contact_position is not None
            else []
        ),
        "return": [
            row
            for row in rows
            if str(row.get("phase", "")) == RETURN_PHASE
        ],
    }
    relationship_fields = {
        "signal_distance": ("travel_down_mm", signal_field),
        "force_distance": ("travel_down_mm", "force_delta_abs_g"),
        "force_signal": (signal_field, "force_delta_abs_g"),
    }
    segmented_smooth_fits = {
        relationship: {
            segment: _smooth_polynomial_fit(
                [
                    (
                        _finite_value(row, x_field),
                        _finite_value(row, y_field),
                    )
                    for row in segment_rows
                ]
            )
            for segment, segment_rows in segmented_rows.items()
        }
        for relationship, (x_field, y_field) in relationship_fields.items()
    }

    turning_point = _turning_point(
        signals, forces_g, distances, contact_position, signal_baseline,
        detector["sigma"], signal_threshold, percent,
    )
    baseline_return = {"reached": False}
    drop_points, rise_points = [], []
    if turning_point["detected"]:
        turn = turning_point["approach_index"]
        baseline_return = _baseline_return(
            _centred_median(signals)[turn:], forces_g, distances, turn,
            contact_position, signal_baseline,
        )
        drop_points = list(zip(signals[contact_position:turn + 1], forces_g[contact_position:turn + 1]))
        rise_points = list(zip(signals[turn:], forces_g[turn:]))
    turning_point["unaveraged_cross_check"] = _unaveraged_turning(
        approach_rows, signal_field, distances, forces_g, contact_position
    )

    return {
        "signal_field": signal_field,
        "loading_samples": "at_rest_steps" if at_rest_only else "continuous",
        "signal_baseline": signal_baseline,
        "baseline_source": "stationary_unique_frames" if measured_baseline else "legacy_approach_estimate",
        "baseline_sample_count": len(baseline_values),
        "proximity_direction": "decrease",
        "baseline_robust_noise": detector["sigma"],
        "force_baseline_noise_g": force_noise,
        "warnings": ([] if measured_baseline else [
            "No adequate stationary baseline: onset estimates are exploratory, not validated."
        ]),
        "signal_change_threshold": signal_threshold,
        "contact_detection_threshold_g": contact_threshold_g,
        "peak_force_g": peak_force_g,
        "max_distance_mm": float(np.max(distances)),
        "signal_onset": onset_payload(signal_position),
        "contact_onset": onset_payload(contact_position),
        "signal_distance_fit": _linear_fit(signal_distance_points),
        "force_distance_fit": _linear_fit(force_distance_points),
        "force_signal_fit": _linear_fit(force_signal_points),
        "signal_distance_smooth_fit": _smooth_polynomial_fit(
            signal_distance_points
        ),
        "force_distance_smooth_fit": _smooth_polynomial_fit(
            force_distance_points
        ),
        "force_signal_smooth_fit": _smooth_polynomial_fit(
            force_signal_points
        ),
        "segmented_smooth_fits": segmented_smooth_fits,
        "proximity_range": _proximity_range(
            distances, signal_position, contact_position,
            _start_gradient_change(signals, distances, contact_position), signal_threshold,
        ),
        "turning_point": turning_point,
        "baseline_return": baseline_return,
        "force_signal_drop_fit": _linear_fit(drop_points),
        "force_signal_rise_fit": _linear_fit(rise_points),
    }


def contact_referenced_rows(rows, signal_field="diff_ave", *, analysis=None, in_place=False):
    """Derived TCP displacement relative to detected contact, not true compression.

    ``in_place`` adds the column to the given row dicts instead of copying them.
    """
    if analysis is None:
        analysis = analyze_calibration_rows(rows, signal_field)
    contact = (analysis.get("contact_onset") or {}).get("distance_mm")

    def relative(row):
        if contact is None or _finite_value(row, "travel_down_mm") is None:
            return None
        return float(row["travel_down_mm"]) - contact

    if in_place:
        rows = list(rows)
        for row in rows:
            row["contact_relative_travel_mm"] = relative(row)
        return rows
    return [{**row, "contact_relative_travel_mm": relative(row)} for row in rows]


def surface_referenced_rows(rows, signal_field="diff_ave", *, analysis=None, in_place=False):
    """Display-only signed height relative to detected first contact.

    Positive is above contact, zero is detected contact and negative is TCP
    travel below that reference, not independently measured fabric compression.
    Missing contact stays unknown; never substitute the start pose as a surface.
    The original travel column is retained unchanged.
    """
    result = contact_referenced_rows(
        rows, signal_field, analysis=analysis, in_place=in_place
    )
    for row in result:
        relative = _finite_value(row, "contact_relative_travel_mm")
        row["surface_distance_mm"] = None if relative is None else -relative
    return result
