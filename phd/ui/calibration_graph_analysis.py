"""Load saved calibration recordings and produce detailed offline analysis."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from phd.ui import components, theme
from phd.ui.calibration_analysis import (
    APPROACH_PHASES,
    RETURN_PHASE,
    analyze_calibration_rows,
    calibration_rows_for_taxel,
    read_calibration_csv_rows,
)
from phd.ui.calibration_result_dialog import (
    CalibrationCurveWidget,
    CalibrationGraphDetailDialog,
    CalibrationTaxelMap,
    SIGNAL_LABELS,
)
from phd.ui.calibration_comparison import (
    CalibrationComparisonDialog, comparison_group, shared_axis_ranges,
)
from phd.ui.calibration_recording_catalog import (
    catalog_full_taxel_run, dataset_rows, recorded_taxels,
)
from phd.ui.calibration_sequence import sequence_segments, RETURN_STAGE_LABELS


SIGNAL_FIELD_PREFERENCE = (
    "diff_ave",
    "diff",
    "diff_percent_ave",
    "diff_percent",
    "raw_ave",
    "raw",
)


def _dataset_sensor_shape(dataset, default=(8, 10)):
    """Return the physical taxel grid stored with one recording."""
    metadata = dict((dataset or {}).get("metadata") or {})
    shape = metadata.get("sensor_shape")
    try:
        rows, columns = map(int, shape[:2])
    except (TypeError, ValueError):
        return tuple(map(int, default))
    if rows <= 0 or columns <= 0:
        return tuple(map(int, default))
    return rows, columns


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _infer_signal_field(rows, preferred=None):
    candidates = []
    if preferred:
        candidates.append(str(preferred))
    candidates.extend(
        field for field in SIGNAL_FIELD_PREFERENCE if field not in candidates
    )
    for field in candidates:
        if any(_finite(row.get(field)) is not None for row in rows):
            return field
    raise ValueError(
        "The recording has no supported tactile signal column "
        "(expected diff_ave, diff, diff_percent_ave, diff_percent, raw_ave, or raw)."
    )


def _resolve_reference(reference, base_directory):
    reference = str(reference or "").strip()
    if not reference:
        return None
    path = Path(reference).expanduser()
    base_directory = Path(base_directory).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(base_directory / path)
    candidates.extend(
        [base_directory / path.name, base_directory.parent / path.name]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    for search_root in (base_directory, base_directory.parent):
        if not search_root.is_dir():
            continue
        matches = sorted(search_root.glob(f"**/{path.name}"))
        if matches:
            return matches[0].resolve()
    return None


def _read_csv_rows(path):
    path = Path(path).expanduser().resolve()
    rows = read_calibration_csv_rows(path)
    if not rows:
        raise ValueError(f"The CSV contains no data rows: {path}")
    return rows


def _trial_dataset(csv_path, metadata=None, preferred_signal=None, name=None):
    csv_path = Path(csv_path).expanduser().resolve()
    rows = _read_csv_rows(csv_path)
    signal_field = _infer_signal_field(rows, preferred_signal)
    metadata = dict(metadata or {})
    if _finite(metadata.get("taxel_index")) is None:
        recorded_index = _finite(rows[0].get("cell_index"))
        if recorded_index is not None:
            metadata["taxel_index"] = int(recorded_index)
    metadata.setdefault("sensor_shape", [8, 10])
    for row in rows:
        if _finite(row.get("force_delta_abs_g")) is None:
            force_n = _finite(row.get("force_delta_abs_n"))
            if force_n is not None:
                row["force_delta_abs_g"] = abs(force_n) / 0.00980665
    trial_number = metadata.get("trial_number")
    if name is None:
        name = (
            f"Trial {int(trial_number):04d}"
            if _finite(trial_number) is not None
            else csv_path.stem
        )
    return {
        "name": str(name),
        "rows": rows,
        "signal_field": signal_field,
        "source_path": str(csv_path),
        "metadata": metadata,
    }


def _load_csv(path, preferred_signal=None, metadata=None):
    rows = _read_csv_rows(path)
    # A batch summary contains references to the actual per-sample CSV files.
    if "csv_path" in rows[0] and "elapsed_s" not in rows[0]:
        datasets = []
        for index, summary in enumerate(rows, start=1):
            csv_path = _resolve_reference(summary.get("csv_path"), path.parent)
            if csv_path is None:
                raise FileNotFoundError(
                    f"Could not locate raw CSV for batch trial {index}: "
                    f"{summary.get('csv_path')}"
                )
            trial_metadata = dict(metadata or {})
            metadata_path = _resolve_reference(
                summary.get("metadata_path"), path.parent
            )
            if metadata_path is not None:
                try:
                    with metadata_path.open("r", encoding="utf-8") as stream:
                        trial_payload = json.load(stream)
                    if isinstance(trial_payload, dict):
                        trial_metadata.update(trial_payload)
                except (OSError, ValueError, TypeError):
                    pass
            trial_metadata.update(summary)
            datasets.append(
                _trial_dataset(
                    csv_path,
                    metadata=trial_metadata,
                    preferred_signal=preferred_signal,
                )
            )
        return datasets
    return [
        _trial_dataset(
            path,
            metadata=metadata,
            preferred_signal=preferred_signal,
        )
    ]


def _load_json(path):
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Calibration JSON must contain an object at its root.")

    preferred_signal = payload.get("signal_field")
    taxel_results = payload.get("taxel_results")
    if isinstance(taxel_results, list):
        datasets = []
        common_metadata = dict(payload)
        common_metadata.pop("taxel_results", None)
        for index, result in enumerate(taxel_results, start=1):
            if not isinstance(result, dict):
                continue
            summary_path = _resolve_reference(
                result.get("batch_summary_path"), path.parent
            )
            if summary_path is None:
                continue
            if summary_path.suffix.lower() == ".json":
                datasets.extend(_load_json(summary_path))
                continue
            taxel_metadata = dict(common_metadata)
            taxel_metadata.update(result)
            datasets.extend(
                _load_csv(
                    summary_path,
                    preferred_signal=preferred_signal,
                    metadata=taxel_metadata,
                )
            )
        if not datasets:
            summary_path = _resolve_reference(
                payload.get("summary_csv_path"), path.parent
            )
            if summary_path is not None:
                datasets = _load_csv(
                    summary_path,
                    preferred_signal=preferred_signal,
                    metadata=common_metadata,
                )
        if not datasets:
            raise ValueError(
                "The full-taxel summary contains no usable trial recordings."
            )
        return datasets

    trials = payload.get("trials")
    if isinstance(trials, list):
        datasets = []
        for index, trial in enumerate(trials, start=1):
            if not isinstance(trial, dict):
                continue
            csv_path = _resolve_reference(trial.get("csv_path"), path.parent)
            if csv_path is None:
                metadata_path = _resolve_reference(
                    trial.get("metadata_path"), path.parent
                )
                if metadata_path is not None:
                    nested = _load_json(metadata_path)
                    datasets.extend(nested)
                    continue
                raise FileNotFoundError(
                    f"Could not locate raw CSV for batch trial {index}: "
                    f"{trial.get('csv_path')}"
                )
            metadata = dict(payload)
            metadata.pop("trials", None)
            metadata.update(trial)
            datasets.append(
                _trial_dataset(
                    csv_path,
                    metadata=metadata,
                    preferred_signal=preferred_signal,
                )
            )
        if not datasets:
            raise ValueError("The batch JSON contains no usable trial recordings.")
        return datasets

    csv_path = _resolve_reference(payload.get("csv_path"), path.parent)
    if csv_path is None:
        raise FileNotFoundError(
            "The JSON does not point to an available raw calibration CSV."
        )
    return [
        _trial_dataset(
            csv_path,
            metadata=payload,
            preferred_signal=preferred_signal,
        )
    ]


def _raw_csv_for_graph(path):
    candidates = [path.with_suffix(".csv")]
    graph_directory = path.parent
    if graph_directory.name == "graphs":
        candidates.append(graph_directory.parent / f"{path.stem}.csv")
    if graph_directory.name.endswith("_graphs"):
        batch_id = graph_directory.name[: -len("_graphs")]
        recording_directory = graph_directory.parent
        candidates.extend(
            sorted(recording_directory.glob(f"*{batch_id}*.csv"))
        )
    for candidate in candidates:
        if candidate.is_file() and "summary" not in candidate.stem.lower():
            return candidate.resolve()
    return None


def load_calibration_recording(path, *, browse_full_run=False):
    """Load raw trial data selected through CSV, JSON, or an exported graph."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file does not exist: {path}")
    if browse_full_run:
        catalog = catalog_full_taxel_run(path)
        if catalog is not None:
            return catalog
    suffix = path.suffix.lower()
    if suffix == ".json":
        datasets = _load_json(path)
    elif suffix == ".csv":
        sibling_json = path.with_suffix(".json")
        metadata = None
        if sibling_json.is_file():
            try:
                with sibling_json.open("r", encoding="utf-8") as stream:
                    metadata = json.load(stream)
            except (OSError, ValueError, TypeError):
                metadata = None
        if path.stem == "batch_summary" and sibling_json.is_file():
            datasets = _load_json(sibling_json)
        else:
            datasets = _load_csv(
                path,
                preferred_signal=(metadata or {}).get("signal_field"),
                metadata=metadata,
            )
    elif suffix == ".png":
        batch_path = path.parent / "batch_summary.json"
        if path.stem == "repeatability" and batch_path.is_file():
            datasets = _load_json(batch_path)
        else:
            csv_path = _raw_csv_for_graph(path)
            if csv_path is None:
                raise FileNotFoundError(
                    "This graph image has no matching raw calibration CSV. Keep "
                    "the PNG beside its original calibration CSV/JSON, or select "
                    "the raw recording directly."
                )
            metadata = None
            metadata_path = csv_path.with_suffix(".json")
            if metadata_path.is_file():
                try:
                    with metadata_path.open("r", encoding="utf-8") as stream:
                        metadata = json.load(stream)
                except (OSError, ValueError, TypeError):
                    metadata = None
            datasets = _load_csv(
                csv_path,
                preferred_signal=(metadata or {}).get("signal_field"),
                metadata=metadata,
            )
    else:
        raise ValueError("Choose a calibration CSV, JSON, or exported PNG graph.")
    return {
        "selected_path": str(path),
        "datasets": datasets,
    }


def _series(rows, field, phases=None):
    values = []
    for row in rows:
        if phases is not None and str(row.get("phase", "")) not in phases:
            continue
        value = _finite(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _phase_metrics(rows):
    phase_values = defaultdict(list)
    for row in rows:
        phase = str(row.get("phase") or "unlabelled")
        elapsed = _finite(row.get("elapsed_s"))
        if elapsed is not None:
            phase_values[phase].append(elapsed)
    metrics = {}
    for phase, elapsed_values in phase_values.items():
        metrics[phase] = {
            "sample_count": len(elapsed_values),
            "duration_s": max(elapsed_values) - min(elapsed_values),
        }
    return dict(metrics)


def _averaged_xy(rows, x_field, y_field, phases):
    grouped = defaultdict(list)
    for row in rows:
        if str(row.get("phase", "")) not in phases:
            continue
        x_value = _finite(row.get(x_field))
        y_value = _finite(row.get(y_field))
        if x_value is not None and y_value is not None:
            grouped[x_value].append(y_value)
    x_values = sorted(grouped)
    y_values = [float(np.mean(grouped[value])) for value in x_values]
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def _hysteresis_metrics(rows, y_field):
    if any(str(row.get("step_state", "")) == "dwell" for row in rows):
        rows = [row for row in rows if str(row.get("step_state", "")) == "dwell"]
    load_x, load_y = _averaged_xy(
        rows, "travel_down_mm", y_field, APPROACH_PHASES
    )
    return_x, return_y = _averaged_xy(
        rows, "travel_down_mm", y_field, {RETURN_PHASE}
    )
    if len(load_x) < 2 or len(return_x) < 2:
        return None
    lower = max(float(load_x[0]), float(return_x[0]))
    upper = min(float(load_x[-1]), float(return_x[-1]))
    if upper - lower <= 1e-12:
        return None
    grid = np.linspace(lower, upper, 101)
    loading = np.interp(grid, load_x, load_y)
    returning = np.interp(grid, return_x, return_y)
    difference = returning - loading
    response_span = float(np.ptp(np.concatenate((loading, returning))))
    rms = float(np.sqrt(np.mean(difference ** 2)))
    return {
        "overlap_min_mm": lower,
        "overlap_max_mm": upper,
        "comparison_points": len(grid),
        "signed_mean": float(np.mean(difference)),
        "mean_absolute": float(np.mean(np.abs(difference))),
        "rms": rms,
        "maximum_absolute": float(np.max(np.abs(difference))),
        "normalized_rms_percent": (
            None if response_span <= 1e-12 else 100.0 * rms / response_span
        ),
    }


def analyze_calibration_dataset(dataset):
    """Return quantitative quality, event, model, and hysteresis metrics."""
    rows = list(dataset.get("rows") or [])
    signal_field = str(dataset.get("signal_field") or "diff_ave")
    core = analyze_calibration_rows(rows, signal_field)
    elapsed = _series(rows, "elapsed_s")
    intervals = np.diff(np.asarray(elapsed, dtype=float))
    intervals = intervals[np.isfinite(intervals) & (intervals > 0.0)]
    median_interval = float(np.median(intervals)) if intervals.size else None
    mean_interval = float(np.mean(intervals)) if intervals.size else None
    interval_cv = (
        None
        if not intervals.size or mean_interval <= 1e-12
        else float(np.std(intervals) / mean_interval)
    )
    distances = _series(rows, "travel_down_mm")
    signals = _series(rows, signal_field)
    forces = _series(rows, "force_delta_abs_g")
    approach_signals = _series(rows, signal_field, APPROACH_PHASES)
    baseline_values = _series(rows, signal_field, {"force_baseline"})
    if not baseline_values:
        baseline_values = approach_signals[: min(10, len(approach_signals))]
    baseline_median = (
        float(np.median(baseline_values)) if baseline_values else None
    )
    baseline_noise = (
        float(
            1.4826
            * np.median(np.abs(np.asarray(baseline_values) - baseline_median))
        )
        if baseline_values
        else None
    )
    peak_baseline_departure = (
        max(abs(value - baseline_median) for value in signals)
        if signals and baseline_median is not None
        else None
    )
    signal_onset = core.get("signal_onset") or {}
    contact_onset = core.get("contact_onset") or {}
    signal_distance = _finite(signal_onset.get("distance_mm"))
    contact_distance = _finite(contact_onset.get("distance_mm"))
    return_distances = _series(rows, "travel_down_mm", {RETURN_PHASE})
    first_distance = distances[0] if distances else None
    final_return_distance = return_distances[-1] if return_distances else None
    proximity_window = (
        contact_distance - signal_distance
        if signal_distance is not None and contact_distance is not None
        else None
    )
    invalid_core_samples = sum(
        1
        for row in rows
        if any(
            _finite(row.get(field)) is None
            for field in ("elapsed_s", "travel_down_mm", signal_field)
        )
    )
    return {
        "name": str(dataset.get("name") or "Calibration trial"),
        "source_path": str(dataset.get("source_path") or ""),
        "signal_field": signal_field,
        "signal_label": SIGNAL_LABELS.get(signal_field, signal_field),
        "metadata": dict(dataset.get("metadata") or {}),
        "sequence_segments": sequence_segments(rows),
        "onset_warnings": core.get("warnings", []),
        "acquisition": {
            "sample_count": len(rows),
            "invalid_core_sample_count": invalid_core_samples,
            "duration_s": (
                max(elapsed) - min(elapsed) if len(elapsed) >= 2 else 0.0
            ),
            "median_interval_s": median_interval,
            "median_sampling_rate_hz": (
                None
                if median_interval is None or median_interval <= 1e-12
                else 1.0 / median_interval
            ),
            "interval_cv": interval_cv,
            "interval_p95_s": (
                float(np.percentile(intervals, 95)) if intervals.size else None
            ),
            "phase_metrics": _phase_metrics(rows),
        },
        "events": {
            "signal_onset": core.get("signal_onset"),
            "contact_onset": core.get("contact_onset"),
            "proximity_window_mm": proximity_window,
            "signal_change_threshold": core.get("signal_change_threshold"),
            "contact_detection_threshold_g": core.get(
                "contact_detection_threshold_g"
            ),
            "proximity_range": core.get("proximity_range"),
            "turning_point": (
                core.get("turning_point")
                if (core.get("turning_point") or {}).get("detected") else None
            ),
            "turning_point_status": core.get("turning_point"),
            "baseline_return": (
                core.get("baseline_return")
                if (core.get("baseline_return") or {}).get("reached") else None
            ),
        },
        "transition_models": {
            "force_signal_drop": core.get("force_signal_drop_fit"),
            "force_signal_rise": core.get("force_signal_rise_fit"),
        },
        "response": {
            "peak_force_g": max(forces) if forces else None,
            "maximum_travel_mm": max(distances) if distances else None,
            "signal_min": min(signals) if signals else None,
            "signal_max": max(signals) if signals else None,
            "signal_span": max(signals) - min(signals) if signals else None,
            "signal_peak_absolute": (
                max(abs(value) for value in signals) if signals else None
            ),
            "signal_final": signals[-1] if signals else None,
            "baseline_median": baseline_median,
            "baseline_robust_noise": baseline_noise,
            "peak_baseline_departure": peak_baseline_departure,
            "peak_signal_to_noise_ratio": (
                peak_baseline_departure / baseline_noise
                if peak_baseline_departure is not None
                and baseline_noise is not None
                and baseline_noise > 1e-12
                else None
            ),
            "final_return_distance_mm": final_return_distance,
            "return_position_error_mm": (
                abs(final_return_distance - first_distance)
                if final_return_distance is not None and first_distance is not None
                else None
            ),
        },
        "models": {
            "signal_distance": core.get("signal_distance_fit"),
            "force_distance": core.get("force_distance_fit"),
            "force_signal": core.get("force_signal_fit"),
        },
        "approximations": {
            "signal_distance": core.get("signal_distance_smooth_fit"),
            "force_distance": core.get("force_distance_smooth_fit"),
            "force_signal": core.get("force_signal_smooth_fit"),
        },
        "segmented_approximations": dict(
            core.get("segmented_smooth_fits") or {}
        ),
        "hysteresis": {
            "signal_distance": _hysteresis_metrics(rows, signal_field),
            "force_distance": _hysteresis_metrics(
                rows, "force_delta_abs_g"
            ),
        },
    }


def _summary(values):
    finite = [value for value in (_finite(item) for item in values) if value is not None]
    if not finite:
        return None
    mean = float(np.mean(finite))
    standard_deviation = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return {
        "count": len(finite),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "minimum": min(finite),
        "maximum": max(finite),
        "coefficient_of_variation_percent": (
            None if abs(mean) <= 1e-12 else 100.0 * standard_deviation / abs(mean)
        ),
    }


def analyze_calibration_batch(analyses):
    """Summarize trial-to-trial repeatability for a loaded recording batch."""
    paths = {
        "signal_onset_distance_mm": (
            "events",
            "signal_onset",
            "distance_mm",
        ),
        "contact_onset_distance_mm": (
            "events",
            "contact_onset",
            "distance_mm",
        ),
        "proximity_window_mm": ("events", "proximity_window_mm"),
        "proximity_range_mm": ("events", "proximity_range", "distance_above_contact_mm"),
        "turning_force_g": ("events", "turning_point", "force_g"),
        "turning_after_contact_mm": ("events", "turning_point", "after_contact_mm"),
        "baseline_return_force_g": ("events", "baseline_return", "force_g"),
        "force_signal_rise_slope": ("transition_models", "force_signal_rise", "slope"),
        "peak_force_g": ("response", "peak_force_g"),
        "maximum_travel_mm": ("response", "maximum_travel_mm"),
        "signal_distance_slope": ("models", "signal_distance", "slope"),
        "force_distance_slope": ("models", "force_distance", "slope"),
        "force_signal_slope": ("models", "force_signal", "slope"),
    }

    def nested_value(payload, path):
        value = payload
        for key in path:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    return {
        "trial_count": len(analyses),
        "repeatability": {
            name: _summary([nested_value(item, path) for item in analyses])
            for name, path in paths.items()
        },
    }


def _number(value, digits=3, suffix=""):
    value = _finite(value)
    if value is None:
        return "not available"
    return f"{value:.{digits}f}{suffix}"


def _fit_quality(r_squared):
    r_squared = _finite(r_squared)
    if r_squared is None:
        return "unavailable"
    if r_squared >= 0.98:
        return "excellent linear agreement"
    if r_squared >= 0.90:
        return "strong linear agreement"
    if r_squared >= 0.75:
        return "moderate linear agreement"
    return "weak linear agreement; a linear calibration may be inappropriate"


def _fit_report(title, fit, x_unit, y_unit):
    if not fit:
        return [f"{title}: unavailable (not enough varying loading samples)."]
    slope = float(fit["slope"])
    intercept = float(fit["intercept"])
    sign = "+" if intercept >= 0.0 else "−"
    return [
        f"{title}: y = {slope:.6g}x {sign} {abs(intercept):.6g}",
        f"  Slope: {slope:.6g} {y_unit}/{x_unit}; intercept: "
        f"{intercept:.6g} {y_unit}",
        f"  R²: {float(fit['r_squared']):.5f} ({_fit_quality(fit['r_squared'])}); "
        f"Pearson r: {_number(fit.get('pearson_r'), 5)}",
        f"  RMSE: {_number(fit.get('rmse'), 4, ' ' + y_unit)}; "
        f"MAE: {_number(fit.get('mae'), 4, ' ' + y_unit)}; "
        f"maximum residual: {_number(fit.get('max_abs_error'), 4, ' ' + y_unit)}",
        f"  Valid observed x-range: {float(fit['x_min']):.4g} to "
        f"{float(fit['x_max']):.4g} {x_unit}; n = {int(fit['sample_count'])}",
    ]


_PROXIMITY_STATUS_TEXT = {
    "start_inside_range": (
        "the response was already present at the start pose, so the range is a "
        "lower bound. Raise the start pose and recalibrate there."
    ),
    "proximity_not_detected": "no proximity response was detected before contact.",
    "contact_not_reached": "force contact was not reached, so the range cannot be referenced.",
}
_TURNING_REASON_TEXT = {
    "contact_not_reached": "force contact was not reached.",
    "no_signal_drop": "the signal never dropped below its baseline noise threshold.",
    "no_rise_within_loading": (
        "the signal had not started rising when loading stopped; use a higher force target."
    ),
    "no_loading_samples": "no loading samples were recorded.",
}


def _calibration_target_report(analysis, signal_unit):
    events = analysis["events"]
    models = analysis.get("transition_models") or {}
    proximity = events.get("proximity_range") or {}
    turning = events.get("turning_point")
    turning_status = events.get("turning_point_status") or {}
    baseline_return = events.get("baseline_return")
    lines = ["4b. CALIBRATION TARGETS"]
    if proximity.get("status") == "measured":
        lines.append(
            "- Proximity range: first response "
            f"{_number(proximity.get('distance_above_contact_mm'), 3, ' mm')} of robot travel "
            "before force contact."
        )
    else:
        detail = _PROXIMITY_STATUS_TEXT.get(proximity.get("status"), "not available.")
        bound = proximity.get("lower_bound_mm")
        lines.append(
            "- Proximity range: "
            + (f"at least {_number(bound, 3, ' mm')}; " if bound is not None else "")
            + detail
        )
    if turning:
        cross = turning.get("unaveraged_cross_check") or {}
        lines.extend([
            f"- Drop-to-rise turning point: {_number(turning.get('force_g'), 2, ' g')} "
            f"(flat band {_number(turning.get('force_low_g'), 2)}–"
            f"{_number(turning.get('force_high_g'), 2, ' g')}), "
            f"{_number(turning.get('after_contact_mm'), 3, ' mm')} after force contact, "
            f"signal {_number(turning.get('signal'), 3, ' ' + signal_unit)}.",
            f"  Unaveraged cross-check ({cross.get('signal_field', 'not recorded')}): "
            f"{_number(cross.get('force_g'), 2, ' g')}, "
            f"{_number(cross.get('after_contact_mm'), 3, ' mm')} after contact.",
        ])
        if turning.get("before_detected_contact"):
            lines.append(
                "  The minimum precedes the detected force contact: the turning force "
                "is below the load-cell detection threshold."
            )
    else:
        lines.append(
            "- Drop-to-rise turning point: not found; "
            + _TURNING_REASON_TEXT.get(turning_status.get("reason"), "not available.")
        )
    lines.append(
        "- Signal back at no-touch baseline: "
        + (
            f"{_number(baseline_return.get('force_g'), 2, ' g')}; below this force the "
            "signal alone cannot separate proximity/light contact from no touch."
            if baseline_return else "not reached within this loading."
        )
    )
    lines.extend(_fit_report(
        "Force from sensor response, drop branch (contact to turning point)",
        models.get("force_signal_drop"), signal_unit, "g",
    ))
    lines.extend(_fit_report(
        "Force from sensor response, rise branch (turning point onwards)",
        models.get("force_signal_rise"), signal_unit, "g",
    ))
    return lines


def build_trial_report(analysis):
    acquisition = analysis["acquisition"]
    events = analysis["events"]
    response = analysis["response"]
    models = analysis["models"]
    hysteresis = analysis["hysteresis"]
    metadata = analysis.get("metadata") or {}
    approach_speed = _finite(metadata.get("approach_speed_m_s"))
    return_speed = _finite(metadata.get("return_speed_m_s"))
    travel_limit = _finite(metadata.get("max_travel_m"))
    physical_target = metadata.get(
        "physical_target_taxel_index",
        metadata.get("taxel_index", "not available"),
    )
    signal_unit = "%" if "%" in analysis["signal_label"] else "sensor units"
    signal_onset = events.get("signal_onset") or {}
    contact_onset = events.get("contact_onset") or {}
    warnings = list(analysis.get("onset_warnings") or [])
    quality = metadata.get("measurement_quality") or {}
    motion = quality.get("motion") or {}
    stages = analysis.get("sequence_segments") or []
    stepwise = motion.get("mode") == "stepwise" or any(stage["kind"] in {"unloading", "fast_return"} for stage in stages)
    profile = motion.get("mode") == "proximity_profile_steps" or metadata.get("protocol") == "proximity_profile"
    return_label = "nominal slow-unloading rate" if stepwise else "configured return speed"
    fast_return = _finite(metadata.get("fast_return_command_speed_m_s"))
    return_notes = []
    if profile:
        return_notes.append("- Proximity profile uses its own position-step speeds; the legacy return-speed field is not a profile speed.")
    elif stepwise:
        return_notes.extend([
            "- Fast return commanded speed: " + (
                f"{fast_return * 1000.0:.4f} mm/s" if fast_return is not None else "not recorded in this file"
            ) + ". This is separate from slow unloading; not a measured TCP speed.",
            "- Slow unloading includes dwell between steps. Fast return includes arrival settling; recovery is stationary observation at the anchor.",
        ])
    for stage in stages:
        if stage["kind"] in RETURN_STAGE_LABELS:
            return_notes.append(f"- {stage['label']}: {stage['start_s']:.2f}–{stage['end_s']:.2f} s (recorded stage boundaries).")
    if quality.get("interrupted"):
        warnings.append("Communication pause occurred. Wall time is retained; exclude this interrupted trial from uninterrupted loading comparisons.")
    if not quality:
        warnings.append("Legacy recording: acquisition timestamps and safety-range provenance are incomplete.")
    if not signal_onset:
        warnings.append("Proximity onset was not detected.")
    if not contact_onset:
        warnings.append("Contact onset was not detected.")
    if acquisition["sample_count"] < 30:
        warnings.append("The recording is short; event and fit estimates may be unstable.")
    if (acquisition.get("interval_cv") or 0.0) > 0.25:
        warnings.append("Sampling intervals have high timing variation.")
    for title, fit in models.items():
        if fit and float(fit.get("r_squared", 0.0)) < 0.75:
            warnings.append(
                f"The {title.replace('_', ' ')} linear model has low R²."
            )
    return_error = _finite(response.get("return_position_error_mm"))
    if return_error is not None and return_error > 0.1:
        warnings.append("The recorded return did not finish within 0.1 mm of its start.")

    lines = [
        "CALIBRATION GRAPH ANALYSIS",
        "=" * 72,
        f"Dataset: {analysis['name']}",
        f"Raw source: {analysis['source_path']}",
        f"Signal: {analysis['signal_label']} [{analysis['signal_field']}]",
        "",
        "1. EXECUTIVE ASSESSMENT",
        "- The analysis uses loading data for fitted equations and compares it "
        "with unloading data separately to quantify hysteresis.",
    ]
    if warnings:
        lines.extend(f"- Caution: {warning}" for warning in warnings)
    else:
        lines.append(
            "- No major automated quality warning was found. Confirm the plots "
            "and experimental conditions before using the equations for control."
        )

    lines.extend(
        [
            "",
            "2. EXPERIMENT CONFIGURATION",
            f"- Analysed taxel: {metadata.get('taxel_index', 'not available')}; "
            f"physical target taxel: {physical_target}; "
            f"trial number: {metadata.get('trial_number', 'not available')}; "
            f"stop reason: {metadata.get('stop_reason', 'not available')}",
            "- Approach speed: "
            f"{_number(None if approach_speed is None else approach_speed * 1000.0, 4, ' mm/s')}; "
            f"{return_label}: "
            f"{_number(None if return_speed is None or profile else return_speed * 1000.0, 4, ' mm/s')}",
            *return_notes,
            "- Requested maximum contact force: "
            f"{_number(metadata.get('contact_threshold_g'), 2, ' g')}; "
            f"contact hold: {_number(metadata.get('contact_hold_sec'), 3, ' s')}",
            "- Configured travel limit: "
            f"{_number(None if travel_limit is None else travel_limit * 1000.0, 4, ' mm')}; "
            f"time limit: {_number(metadata.get('timeout_sec'), 3, ' s')} "
            "(zero means disabled)",
            "",
            "3. ACQUISITION QUALITY",
            f"- Samples: {acquisition['sample_count']} "
            f"({acquisition['invalid_core_sample_count']} missing/invalid core samples)",
            f"- Duration: {_number(acquisition['duration_s'], 3, ' s')}",
            f"- Median sampling rate: "
            f"{_number(acquisition['median_sampling_rate_hz'], 2, ' Hz')}",
            f"- Median interval: {_number(acquisition['median_interval_s'], 5, ' s')}; "
            f"95th-percentile interval: {_number(acquisition['interval_p95_s'], 5, ' s')}",
            "- Interval coefficient of variation: "
            + _number(
                None
                if acquisition["interval_cv"] is None
                else acquisition["interval_cv"] * 100.0,
                2,
                "%",
            ),
        ]
    )
    for phase, metric in acquisition["phase_metrics"].items():
        lines.append(
            f"- Phase '{phase}': {metric['sample_count']} samples, "
            f"observed span {_number(metric['duration_s'], 3, ' s')}"
        )

    lines.extend(
        [
            "",
            "4. DETECTED EVENTS AND PHYSICAL INTERPRETATION",
            f"- Proximity onset: distance {_number(signal_onset.get('distance_mm'), 4, ' mm')}, "
            f"time {_number(signal_onset.get('elapsed_s'), 3, ' s')}, "
            f"signed signal {_number(signal_onset.get('signal'), 3, ' ' + signal_unit)}.",
            f"- Contact onset: distance {_number(contact_onset.get('distance_mm'), 4, ' mm')}, "
            f"time {_number(contact_onset.get('elapsed_s'), 3, ' s')}, "
            f"force {_number(contact_onset.get('force_g'), 2, ' g')}.",
            f"- Non-contact proximity travel window: "
            f"{_number(events.get('proximity_window_mm'), 4, ' mm')}. This is the "
            "additional robot travel from detected sensor response to detected force contact.",
            f"- Automatic signal-change threshold: "
            f"{_number(events.get('signal_change_threshold'), 3, ' ' + signal_unit)}; "
            f"contact detection threshold: "
            f"{_number(events.get('contact_detection_threshold_g'), 2, ' g')}.",
            "",
            *_calibration_target_report(analysis, signal_unit),
            "",
            "5. RANGE AND RESPONSE",
            f"- Maximum travel: {_number(response.get('maximum_travel_mm'), 4, ' mm')}",
            f"- Peak force change: {_number(response.get('peak_force_g'), 2, ' g')}",
            f"- Signed sensor range: {_number(response.get('signal_min'), 3)} to "
            f"{_number(response.get('signal_max'), 3)} {signal_unit}; span "
            f"{_number(response.get('signal_span'), 3, ' ' + signal_unit)}",
            "- Baseline median: "
            f"{_number(response.get('baseline_median'), 3, ' ' + signal_unit)}; "
            "robust baseline noise: "
            f"{_number(response.get('baseline_robust_noise'), 3, ' ' + signal_unit)}",
            "- Peak departure from baseline: "
            f"{_number(response.get('peak_baseline_departure'), 3, ' ' + signal_unit)}; "
            "peak-to-robust-noise ratio: "
            f"{_number(response.get('peak_signal_to_noise_ratio'), 2)}",
            "- Final return distance: "
            f"{_number(response.get('final_return_distance_mm'), 4, ' mm')}; "
            "return-position error: "
            f"{_number(response.get('return_position_error_mm'), 4, ' mm')}",
            "",
            "6. LOADING REGRESSION MODELS",
            "Equations are least-squares linear fits over their stated observed "
            "ranges. Do not extrapolate beyond those ranges.",
        ]
    )
    lines.extend(
        _fit_report(
            "Sensor response from distance",
            models.get("signal_distance"),
            "mm",
            signal_unit,
        )
    )
    lines.extend(
        _fit_report(
            "Force from distance",
            models.get("force_distance"),
            "mm",
            "g",
        )
    )
    lines.extend(
        _fit_report(
            "Force from sensor response",
            models.get("force_signal"),
            signal_unit,
            "g",
        )
    )

    lines.extend(["", "7. LOADING/RETURN HYSTERESIS"])
    for title, key, unit in (
        ("Sensor-distance", "signal_distance", signal_unit),
        ("Force-distance", "force_distance", "g"),
    ):
        metric = hysteresis.get(key)
        if not metric:
            lines.append(f"- {title}: unavailable (insufficient overlapping return data).")
            continue
        lines.append(
            f"- {title}: RMS separation {_number(metric['rms'], 4, ' ' + unit)}, "
            f"mean absolute separation {_number(metric['mean_absolute'], 4, ' ' + unit)}, "
            f"maximum {_number(metric['maximum_absolute'], 4, ' ' + unit)}."
        )
        lines.append(
            "  Signed return-minus-loading bias: "
            f"{_number(metric['signed_mean'], 4, ' ' + unit)}; normalized RMS: "
            f"{_number(metric['normalized_rms_percent'], 2, '%')} "
            f"over {metric['overlap_min_mm']:.4g}–{metric['overlap_max_mm']:.4g} mm."
        )

    lines.extend(
        [
            "",
            "8. INTERPRETATION AND USE",
            "- Proximity onset marks the first sustained signed departure from the "
            "initial sensor baseline; contact onset marks sustained force change.",
            "- Fit slope describes sensitivity. Its sign is meaningful: a negative "
            "sensor-distance slope means the signed sensor difference becomes more "
            "negative as the robot advances.",
            "- R² measures linear agreement, not causality or absolute accuracy. "
            "Residual errors and trial repeatability should be considered together.",
            "- Hysteresis is the loading/return difference at matched travel distances. "
            "Large hysteresis can arise from material memory, robot compliance, force "
            "offset drift, motion lag, or different contact geometry.",
            "- This automated report does not replace uncertainty analysis with known "
            "reference standards. For publication, report repeated trials, sensor/force "
            "meter uncertainty, approach speed, filtering, and environmental conditions.",
        ]
    )
    return "\n".join(lines)


def build_batch_report(batch):
    lines = [
        "BATCH REPEATABILITY ANALYSIS",
        "=" * 72,
        f"Loaded trials: {int(batch.get('trial_count', 0))}",
        "",
        "Mean ± sample standard deviation, range, and coefficient of variation "
        "(CV) are calculated across trials with an available metric.",
        "",
    ]
    labels = {
        "signal_onset_distance_mm": ("Proximity onset distance", "mm"),
        "contact_onset_distance_mm": ("Contact onset distance", "mm"),
        "proximity_window_mm": ("Non-contact proximity window", "mm"),
        "proximity_range_mm": ("Proximity range (start outside range only)", "mm"),
        "turning_force_g": ("Drop-to-rise turning force", "g"),
        "turning_after_contact_mm": ("Turning point after force contact", "mm"),
        "baseline_return_force_g": ("Force where signal returns to baseline", "g"),
        "force_signal_rise_slope": ("Force-signal slope, rise branch", "g/signal unit"),
        "peak_force_g": ("Peak force", "g"),
        "maximum_travel_mm": ("Maximum travel", "mm"),
        "signal_distance_slope": ("Signal-distance slope", "signal unit/mm"),
        "force_distance_slope": ("Force-distance slope", "g/mm"),
        "force_signal_slope": ("Force-signal slope", "g/signal unit"),
    }
    for key, (label, unit) in labels.items():
        metric = (batch.get("repeatability") or {}).get(key)
        if not metric:
            lines.append(f"- {label}: unavailable")
            continue
        lines.append(
            f"- {label}: {metric['mean']:.5g} ± "
            f"{metric['standard_deviation']:.5g} {unit}; range "
            f"{metric['minimum']:.5g} to {metric['maximum']:.5g}; "
            f"CV {_number(metric['coefficient_of_variation_percent'], 2, '%')}; "
            f"n={metric['count']}"
        )
    lines.extend(
        [
            "",
            "Interpret CV carefully when a signed mean is near zero; in that case "
            "the standard deviation and range are more informative.",
        ]
    )
    return "\n".join(lines)


class CalibrationGraphAnalysisDialog(QDialog):
    """Modeless report and interactive graphs for saved calibration data."""

    GRAPH_TITLES = {
        "process": "Full Calibration Sequence",
        "signal_distance": "Sensor Signal vs Travel Distance",
        "force_distance": "Force vs Travel Distance",
        "force_signal": "Force vs Sensor Signal",
        "repeatability": "Repeated-Trial Mean and Variation",
    }

    def __init__(self, loaded_recording, parent=None):
        super().__init__(parent, Qt.Window)
        self.loaded_recording = dict(loaded_recording or {})
        self.datasets = list(self.loaded_recording.get("datasets") or [])
        if not self.datasets:
            raise ValueError("No calibration trials were loaded.")
        initial_index = max(0, min(int(self.loaded_recording.get("initial_dataset_index", 0)), len(self.datasets) - 1))
        first_metadata = dict(self.datasets[initial_index].get("metadata") or {})
        self.selected_taxel_index = int(
            first_metadata.get("taxel_index", 0) or 0
        )
        self.projected_datasets = {}
        self.analyses_by_index = {}
        self.analyses = []
        self.batch_analysis = analyze_calibration_batch([])
        self._detail_dialog = None
        self._detail_dialogs = []
        self._comparison_dialog = None
        self._shared_range_cache = {}
        self._scale_error = ""
        self._displayed_index = initial_index
        self.setWindowTitle("Calibration Graph Analysis")
        self.setModal(False)
        components.size_to_screen(self, 1300, 850)
        self.setStyleSheet(
            f"QDialog {{ background: {theme.WINDOW_BG}; "
            f"color: {theme.TEXT_PRIMARY}; }}"
        )

        layout = QVBoxLayout(self)
        title = QLabel("Professional Calibration Graph Analysis")
        title.setStyleSheet(
            f"font-size: 18px; font-weight: 600; color: {theme.TEXT_PRIMARY};"
        )
        layout.addWidget(title)
        self.source_label = QLabel(
            f"Selected file: {self.loaded_recording.get('selected_path', '')}"
        )
        self.source_label.setWordWrap(True)
        self.source_label.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        layout.addWidget(self.source_label)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Analysed trial"))
        self.trial_selector = QComboBox(self)
        for dataset in self.datasets:
            self.trial_selector.addItem(
                f"{dataset.get('name') or 'Trial'} · probed taxel {self._original_taxel(dataset)}"
            )
            self.trial_selector.setItemData(
                self.trial_selector.count() - 1, str(dataset.get("source_path") or ""), Qt.ToolTipRole,
            )
        self.trial_selector.setEnabled(len(self.datasets) > 1)
        self.trial_selector.setCurrentIndex(initial_index)
        selector_layout.addWidget(self.trial_selector, 1)
        self.shared_scales_check = QCheckBox("Shared scales across nodes")
        self.shared_scales_check.setChecked(True)
        self.shared_scales_check.setToolTip(
            "Use the same time, travel, force and signed signal limits for all "
            "loaded trials/nodes with the same sensor identity, grid and signal units. "
            "Uncheck for independent auto-scaling. Reset restores shared limits after zooming."
        )
        selector_layout.addWidget(self.shared_scales_check)
        reset_scales = QPushButton("Reset View")
        reset_scales.clicked.connect(self._reset_chart_views)
        selector_layout.addWidget(reset_scales)
        layout.addLayout(selector_layout)

        sensor_rows, sensor_columns = _dataset_sensor_shape(self.datasets[initial_index])
        taxel_panel = QWidget(self)
        taxel_layout = QVBoxLayout(taxel_panel)
        taxel_layout.setContentsMargins(0, 0, 8, 0)
        taxel_title = QLabel("Recorded taxels")
        taxel_title.setStyleSheet(
            f"font-weight: 600; color: {theme.TEXT_PRIMARY};"
        )
        taxel_hint = QLabel(
            "Click a recorded node to open its own trial, graphs and report.\n"
            "Grey nodes have no saved data. No robot motion.\n"
            "Indices are top-down, column-major."
        )
        taxel_hint.setWordWrap(True)
        taxel_hint.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        self.taxel_context_label = QLabel("")
        self.taxel_context_label.setWordWrap(True)
        self.taxel_context_label.setStyleSheet(f"color: {theme.ACCENT};")
        self.taxel_map = CalibrationTaxelMap(
            sensor_rows,
            sensor_columns,
            taxel_panel,
        )
        taxel_layout.addWidget(taxel_title)
        taxel_layout.addWidget(taxel_hint)
        taxel_layout.addWidget(self.taxel_context_label)
        taxel_layout.addWidget(self.taxel_map)
        taxel_layout.addStretch()

        self.tabs = QTabWidget(self)
        self.report_view = QTextEdit(self)
        self.report_view.setReadOnly(True)
        self.report_view.setLineWrapMode(QTextEdit.NoWrap)
        self.tabs.addTab(self.report_view, "Professional Report")

        self.charts = {}
        for mode, tab_title in (
            ("process", "Full Sequence"),
            ("signal_distance", "Signal vs Distance"),
            ("force_distance", "Force vs Distance"),
            ("force_signal", "Force vs Signal"),
            ("repeatability", "Repeatability Curves"),
        ):
            chart = CalibrationCurveWidget([], "diff_ave", mode=mode)
            chart.detailRequested.connect(
                lambda mode=mode: self._open_chart_detail(mode)
            )
            self.charts[mode] = chart
            tab_index = self.tabs.addTab(chart, tab_title)
            if mode == "repeatability":
                self.tabs.setTabEnabled(tab_index, len(self.datasets) > 1)
                self.repeatability_tab_index = tab_index

        self.batch_report_view = QTextEdit(self)
        self.batch_report_view.setReadOnly(True)
        self.batch_report_view.setLineWrapMode(QTextEdit.NoWrap)
        self.tabs.addTab(self.batch_report_view, "Batch Repeatability")
        self.batch_tab_index = self.tabs.indexOf(self.batch_report_view)
        self.characterisation_view = QTextEdit(self)
        self.characterisation_view.setReadOnly(True)
        self.tabs.addTab(self.characterisation_view, "Quality & Characterisation")
        self.contact_chart = CalibrationCurveWidget([], "diff_ave", mode="signal_distance")
        self.contact_chart.contact_referenced = True
        self.contact_chart.distance_label = "TCP travel relative to detected contact (mm)"
        self.tabs.addTab(self.contact_chart, "Contact-referenced Signal")
        self.contact_chart.detailRequested.connect(self._open_contact_chart_detail)
        self.matched_force_chart = CalibrationCurveWidget([], "diff_ave", mode="repeatability")
        self.matched_force_chart.repeatability_x_field = "force_delta_abs_g"
        self.matched_force_tab_index = self.tabs.addTab(self.matched_force_chart, "Matched-force Variation")
        content_layout = QHBoxLayout()
        content_layout.addWidget(taxel_panel)
        content_layout.addWidget(self.tabs, 1)
        layout.addLayout(content_layout, 1)

        actions = QHBoxLayout()
        hint = QLabel(
            "Select a node, then add it to comparison. Detail windows stay open."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        actions.addWidget(hint, 1)
        open_button = QPushButton("Open Selected Graph")
        self.compare_button = QPushButton("Compare Nodes…")
        self.compare_button.setToolTip(
            "Pin this node/trial, then add more or import another sensor/experiment "
            "for linked side-by-side or overlay plots."
        )
        self.compare_button.clicked.connect(self._open_comparison)
        export_button = QPushButton("Export Analysis Report")
        close_button = QPushButton("Close")
        open_button.clicked.connect(self._open_selected_chart)
        export_button.clicked.connect(self._export_report)
        close_button.clicked.connect(self.close)
        actions.addWidget(open_button)
        actions.addWidget(self.compare_button)
        actions.addWidget(export_button)
        actions.addWidget(close_button)
        layout.addLayout(actions)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        layout.addWidget(self.status_label)

        self.trial_selector.currentIndexChanged.connect(self._choose_trial)
        self.taxel_map.taxelSelected.connect(self._select_taxel)
        self.shared_scales_check.toggled.connect(self._update_shared_scales)
        self._apply_taxel(self.selected_taxel_index)

    @staticmethod
    def _original_taxel(dataset):
        metadata = dict(dataset.get("metadata") or {})
        return int(metadata.get("taxel_index", 0) or 0)

    def _available_taxels(self, dataset):
        return recorded_taxels(dataset)

    def _map_taxels(self, dataset):
        group = comparison_group(dataset)
        if self.loaded_recording.get("full_taxel_run"):
            return sorted({self._original_taxel(item) for item in self.datasets
                           if comparison_group(item) == group})
        return sorted({taxel for item in self.datasets if comparison_group(item) == group
                       for taxel in self._available_taxels(item)})

    def _project_dataset(self, dataset, taxel_index):
        if taxel_index not in self._available_taxels(dataset):
            return None
        signal_field = str(dataset.get("signal_field") or "diff_ave")
        original_taxel = self._original_taxel(dataset)
        rows = calibration_rows_for_taxel(
            dataset_rows(dataset),
            signal_field,
            taxel_index,
            legacy_index=original_taxel,
        )
        if not rows:
            return None
        projected = dict(dataset)
        metadata = dict(dataset.get("metadata") or {})
        metadata["taxel_index"] = int(taxel_index)
        metadata["physical_target_taxel_index"] = original_taxel
        projected["metadata"] = metadata
        projected["rows"] = rows
        projected["name"] = (
            f"{dataset.get('name') or 'Calibration trial'} · "
            f"Taxel {int(taxel_index)}"
        )
        return projected

    def _apply_taxel(self, taxel_index):
        self.selected_taxel_index = int(taxel_index)
        self.projected_datasets = {}
        self.analyses_by_index = {}
        self.characterisation_trials = {}
        analyses = []
        group = comparison_group(self.datasets[self._current_index()])
        from phd.ui.characterisation_analysis import trial_characterisation, batch_characterisation
        for index, dataset in enumerate(self.datasets):
            if (comparison_group(dataset) != group
                    or (self.loaded_recording.get("full_taxel_run")
                        and self._original_taxel(dataset) != self.selected_taxel_index)
                    or self.selected_taxel_index not in self._available_taxels(dataset)):
                if dataset.get("deferred_csv"):
                    dataset["rows"] = []  # Keep only the currently analysed node's trials in RAM.
                continue
            projected = self._project_dataset(
                dataset,
                self.selected_taxel_index,
            )
            if projected is None:
                continue
            analysis = analyze_calibration_dataset(projected)
            self.projected_datasets[index] = projected
            self.analyses_by_index[index] = analysis
            analyses.append(analysis)
            self.characterisation_trials[index] = trial_characterisation(projected)
        self.analyses = analyses
        self.characterisation_batch = batch_characterisation(list(self.projected_datasets.values()))
        self._select_trial(self._current_index())

    def _refresh_condition_comparison(self, index):
        from phd.ui.characterisation_analysis import _condition_key, contact_points
        dataset = self.projected_datasets[index]
        condition = _condition_key(dataset)
        indices = [
            i for i, candidate in self.projected_datasets.items()
            if _condition_key(candidate) == condition
            and self.characterisation_trials[i]["eligible_for_comparison"]
        ]
        trials = [{"trial_number": i + 1, "rows": self.projected_datasets[i]["rows"]} for i in indices]
        self.charts["repeatability"].set_trials(trials, dataset["signal_field"])
        self.matched_force_chart.set_trials([
            {"trial_number": i + 1, "rows": [
                {"phase": "approaching", "elapsed_s": number,
                 "force_delta_abs_g": force, dataset["signal_field"]: signal}
                for number, (signal, force) in enumerate(contact_points(self.projected_datasets[i]))
            ]} for i in indices
        ], dataset["signal_field"])
        self.tabs.setTabEnabled(self.matched_force_tab_index, len(trials) > 1)
        self.batch_analysis = analyze_calibration_batch([self.analyses_by_index[i] for i in indices])
        self.batch_report_view.setPlainText(
            "Only uninterrupted, complete, matching-condition trials are included.\n\n"
            + build_batch_report(self.batch_analysis)
        )
        self.tabs.setTabEnabled(self.repeatability_tab_index, len(trials) > 1)
        self.tabs.setTabEnabled(self.batch_tab_index, len(trials) > 1)

    def _select_taxel(self, taxel_index):
        taxel_index = int(taxel_index)
        current_index = self._current_index()
        current = self.datasets[current_index]
        group = comparison_group(current)
        matching_indices = [
            index
            for index, dataset in enumerate(self.datasets)
            if comparison_group(dataset) == group and taxel_index in self._available_taxels(dataset)
            and (not self.loaded_recording.get("full_taxel_run") or self._original_taxel(dataset) == taxel_index)
        ]
        if not matching_indices:
            self.status_label.setText(
                f"Taxel {taxel_index} has no recorded calibration data."
            )
            return
        # A physical calibration of this node takes precedence over a neighbour's
        # signal captured while pressing a different node in an old full-grid CSV.
        physical = [i for i in matching_indices if self._original_taxel(self.datasets[i]) == taxel_index]
        candidates = physical or matching_indices
        trial_number = (current.get("metadata") or {}).get("trial_number")
        next_index = current_index if current_index in candidates else next(
            (i for i in candidates if (self.datasets[i].get("metadata") or {}).get("trial_number") == trial_number),
            candidates[0],
        )
        try:
            # Validate before changing selection; a removed/changed file must not
            # relabel the previous node's graph as the newly clicked node.
            for i in matching_indices:
                dataset_rows(self.datasets[i])
        except (OSError, ValueError) as exc:
            self.taxel_map.set_taxels(self._map_taxels(current), self.selected_taxel_index)
            self.status_label.setText(f"Could not load Taxel {taxel_index}: {exc}")
            return
        if current_index != next_index:
            self.trial_selector.blockSignals(True)
            self.trial_selector.setCurrentIndex(next_index)
            self.trial_selector.blockSignals(False)
        self._apply_taxel(taxel_index)

    def _select_trial(self, index):
        index = max(0, min(int(index), len(self.datasets) - 1))
        dataset = self.datasets[index]
        sensor_rows, sensor_columns = _dataset_sensor_shape(dataset)
        self.taxel_map.set_grid_shape(sensor_rows, sensor_columns)
        taxel_count = sensor_rows * sensor_columns
        available = [
            taxel
            for taxel in self._available_taxels(dataset)
            if 0 <= int(taxel) < taxel_count
        ]
        if available and (self.selected_taxel_index not in available or index not in self.projected_datasets):
            original_taxel = self._original_taxel(dataset)
            fallback = self.selected_taxel_index
            if self.loaded_recording.get("full_taxel_run") or fallback not in available:
                fallback = original_taxel if original_taxel in available else int(available[0])
            self._apply_taxel(fallback)
            return
        map_taxels = [i for i in self._map_taxels(dataset) if 0 <= i < taxel_count]
        self.taxel_map.set_taxels(map_taxels, self.selected_taxel_index)
        dataset = self.projected_datasets.get(index)
        analysis = self.analyses_by_index.get(index)
        if dataset is None or analysis is None:
            return
        self._refresh_condition_comparison(index)
        self.report_view.setPlainText(build_trial_report(analysis))
        from phd.ui.characterisation_analysis import characterisation_report
        from phd.ui.calibration_analysis import contact_referenced_rows
        self.characterisation_view.setPlainText(characterisation_report(
            self.characterisation_trials[index], self.characterisation_batch
        ))
        referenced = contact_referenced_rows(dataset["rows"], dataset["signal_field"])
        self.contact_chart.set_data(
            [{**r, "travel_down_mm": r["contact_relative_travel_mm"]} for r in referenced],
            dataset["signal_field"],
        )
        for mode, chart in self.charts.items():
            if mode == "repeatability":
                continue
            chart.set_data(dataset["rows"], dataset["signal_field"])
        self._update_shared_scales()
        original_taxel = self._original_taxel(self.datasets[index])
        self.taxel_context_label.setText(
            f"Viewing taxel {self.selected_taxel_index}\n"
            f"Physical target: taxel {original_taxel}\n"
            f"{len(map_taxels)} nodes with saved data"
        )
        partial = (dataset.get("metadata", {}).get("measurement_quality") or {}).get("trial_completed") is False
        warnings = self.loaded_recording.get("catalog_warnings") or []
        self.status_label.setText(
            ("Partial / interrupted trial — " if partial else "")
            + (self._scale_error + " " if self._scale_error else "")
            + (f"{len(warnings)} unreadable trial(s) skipped; see tooltip. " if warnings else "")
            + f"Taxel {self.selected_taxel_index} | Raw data: "
            f"{dataset['source_path']}"
        )
        self.status_label.setToolTip("\n".join(warnings))
        self._displayed_index = index

    def _choose_trial(self, index):
        previous, previous_taxel = self._displayed_index, self.selected_taxel_index
        try:
            dataset_rows(self.datasets[index])
            self._select_trial(index)
        except (OSError, ValueError) as exc:
            self.trial_selector.blockSignals(True)
            self.trial_selector.setCurrentIndex(previous)
            self.trial_selector.blockSignals(False)
            self._apply_taxel(previous_taxel)
            self.status_label.setText(f"Could not load this trial: {exc}")

    def _current_index(self):
        return max(0, min(self.trial_selector.currentIndex(), len(self.datasets) - 1))

    def _current_shared_ranges(self):
        if not self.shared_scales_check.isChecked():
            return {}
        group = comparison_group(self.datasets[self._current_index()])
        if group not in self._shared_range_cache:
            try:
                self._shared_range_cache[group] = shared_axis_ranges([
                    dataset for dataset in self.datasets if comparison_group(dataset) == group
                ])
                self._scale_error = ""
            except (OSError, ValueError) as exc:
                self._scale_error = f"Shared scales unavailable; using auto-scale. {exc}"
                return {}
        return self._shared_range_cache[group]

    def _update_shared_scales(self, *_args):
        ranges = self._current_shared_ranges()
        for chart in [*self.charts.values(), self.contact_chart, self.matched_force_chart]:
            chart.set_shared_axis_ranges(ranges)

    def _reset_chart_views(self):
        for chart in [*self.charts.values(), self.contact_chart, self.matched_force_chart]:
            chart.reset_view()

    def _open_comparison(self):
        if self._comparison_dialog is None:
            try:
                dialog = CalibrationComparisonDialog(self.datasets, self)
            except ImportError as exc:
                self.status_label.setText(f"Comparison needs Matplotlib: {exc}")
                return
            self._comparison_dialog = dialog
            dialog.setAttribute(Qt.WA_DeleteOnClose, True)
            dialog.destroyed.connect(lambda _=None: setattr(self, "_comparison_dialog", None))
        dialog = self._comparison_dialog
        dialog.add_node(self._current_index(), self.selected_taxel_index)
        dialog.select_dataset(self._current_index())
        selected = dialog.node_selector.findData(self.selected_taxel_index)
        if selected >= 0:
            dialog.node_selector.setCurrentIndex(selected)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _open_selected_chart(self):
        widget = self.tabs.currentWidget()
        if widget is self.contact_chart:
            self._open_contact_chart_detail()
            return
        if widget is self.matched_force_chart:
            dialog = CalibrationGraphDetailDialog(
                f"Signal variation at matched force — Taxel {self.selected_taxel_index} · "
                f"{self.datasets[self._current_index()].get('name', 'Trial')}",
                [], self.matched_force_chart.signal_field,
                "repeatability", parent=self, trial_results=self.matched_force_chart.trial_results,
            )
            dialog.chart.repeatability_x_field = "force_delta_abs_g"
            dialog.chart.set_shared_axis_ranges(self.matched_force_chart.shared_axis_ranges)
            self._show_extra_detail(dialog)
            return
        for mode, chart in self.charts.items():
            if widget is chart:
                self._open_chart_detail(mode)
                return
        self.status_label.setText(
            "Select one of the graph tabs to open it in detail."
        )

    def _show_extra_detail(self, dialog):
        self._detail_dialog = dialog
        self._detail_dialogs.append(dialog)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda _=None: self._detail_closed(dialog))
        dialog.status_label.setText(
            "Shared physical scales; Reset View restores common limits."
            if dialog.chart.shared_axis_ranges else "Independent auto-scales."
        )
        dialog.status_label.setToolTip(str(self.datasets[self._current_index()].get("source_path") or ""))
        dialog.show()

    def _open_contact_chart_detail(self):
        dialog = CalibrationGraphDetailDialog(
            f"Signal vs contact-relative TCP travel — Taxel {self.selected_taxel_index} · "
            f"{self.datasets[self._current_index()].get('name', 'Trial')}", self.contact_chart.rows,
            self.contact_chart.signal_field, "signal_distance", parent=self,
        )
        dialog.chart.distance_label = self.contact_chart.distance_label
        dialog.chart.contact_referenced = True
        dialog.chart.set_shared_axis_ranges(self.contact_chart.shared_axis_ranges)
        self._show_extra_detail(dialog)

    def _open_chart_detail(self, mode):
        if mode not in self.charts:
            return
        chart = self.charts[mode]
        trial_results = None
        if mode == "repeatability":
            trial_results = chart.trial_results
        dialog = CalibrationGraphDetailDialog(
            f"{self.GRAPH_TITLES[mode]} — Taxel "
            f"{self.selected_taxel_index} · {self.datasets[self._current_index()].get('name', 'Trial')}",
            chart.rows,
            chart.signal_field,
            mode,
            parent=self,
            trial_results=trial_results,
        )
        dialog.chart.set_shared_axis_ranges(chart.shared_axis_ranges)
        self._show_extra_detail(dialog)
        dialog.raise_()
        dialog.activateWindow()

    def _detail_closed(self, dialog):
        if dialog in self._detail_dialogs:
            self._detail_dialogs.remove(dialog)
        if self._detail_dialog is dialog:
            self._detail_dialog = self._detail_dialogs[-1] if self._detail_dialogs else None

    def _export_report(self):
        selected = Path(self.loaded_recording.get("selected_path") or "calibration")
        default_path = selected.with_name(
            f"{selected.stem}_taxel_{self.selected_taxel_index}_analysis.json"
        )
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export calibration analysis",
            str(default_path),
            "JSON report (*.json);;Text report (*.txt)",
        )
        if not path:
            return
        output_path = Path(path).expanduser()
        wants_text = "Text" in selected_filter or output_path.suffix.lower() == ".txt"
        if wants_text:
            if output_path.suffix.lower() != ".txt":
                output_path = output_path.with_suffix(".txt")
            sections = [build_trial_report(item) for item in self.analyses]
            sections.append(self.characterisation_view.toPlainText())
            if len(self.analyses) > 1:
                sections.append(build_batch_report(self.batch_analysis))
            content = "\n\n".join(sections) + "\n"
            output_path.write_text(content, encoding="utf-8")
        else:
            if output_path.suffix.lower() != ".json":
                output_path = output_path.with_suffix(".json")
            payload = {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "selected_path": self.loaded_recording.get("selected_path"),
                "selected_taxel_index": self.selected_taxel_index,
                "trials": self.analyses,
                "batch": self.batch_analysis,
                "characterisation": self.characterisation_batch,
                "trial_characterisation": self.characterisation_trials,
            }
            with output_path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write("\n")
        self.status_label.setText(f"Analysis report saved: {output_path}")

    def closeEvent(self, event):
        self._detail_dialog = None
        for detail in list(self._detail_dialogs):
            detail.close()
        self._detail_dialogs.clear()
        if self._comparison_dialog is not None:
            self._comparison_dialog.close()
        super().closeEvent(event)
