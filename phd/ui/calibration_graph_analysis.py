"""Load saved calibration recordings and produce detailed offline analysis."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
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
    available_taxel_indices,
    calibration_rows_for_taxel,
)
from phd.ui.calibration_result_dialog import (
    CalibrationCurveWidget,
    CalibrationGraphDetailDialog,
    CalibrationTaxelMap,
    SIGNAL_LABELS,
)


SIGNAL_FIELD_PREFERENCE = (
    "diff_ave",
    "diff",
    "diff_percent_ave",
    "diff_percent",
    "raw_ave",
    "raw",
)


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _coerce_csv_value(value):
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
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in reader
        ]
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


def load_calibration_recording(path):
    """Load raw trial data selected through CSV, JSON, or an exported graph."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file does not exist: {path}")
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
    signal_unit = "%" if "%" in analysis["signal_label"] else "counts"
    signal_onset = events.get("signal_onset") or {}
    contact_onset = events.get("contact_onset") or {}
    warnings = []
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
            "return speed: "
            f"{_number(None if return_speed is None else return_speed * 1000.0, 4, ' mm/s')}",
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
        first_metadata = dict(self.datasets[0].get("metadata") or {})
        self.selected_taxel_index = int(
            first_metadata.get("taxel_index", 0) or 0
        )
        self.projected_datasets = {}
        self.analyses_by_index = {}
        self.analyses = []
        self.batch_analysis = analyze_calibration_batch([])
        self._detail_dialog = None
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
            self.trial_selector.addItem(str(dataset.get("name") or "Trial"))
        self.trial_selector.setEnabled(len(self.datasets) > 1)
        selector_layout.addWidget(self.trial_selector, 1)
        layout.addLayout(selector_layout)

        sensor_shape = first_metadata.get("sensor_shape") or [8, 10]
        try:
            sensor_rows, sensor_columns = map(int, sensor_shape[:2])
        except (TypeError, ValueError):
            sensor_rows, sensor_columns = 8, 10
        taxel_panel = QWidget(self)
        taxel_layout = QVBoxLayout(taxel_panel)
        taxel_layout.setContentsMargins(0, 0, 8, 0)
        taxel_title = QLabel("Recorded taxels")
        taxel_title.setStyleSheet(
            f"font-weight: 600; color: {theme.TEXT_PRIMARY};"
        )
        taxel_hint = QLabel(
            "Click a node to recalculate this trial's graphs and report.\n"
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
        content_layout = QHBoxLayout()
        content_layout.addWidget(taxel_panel)
        content_layout.addWidget(self.tabs, 1)
        layout.addLayout(content_layout, 1)

        actions = QHBoxLayout()
        hint = QLabel(
            "Graphs: move the mouse for x/y values, use the wheel to zoom, "
            "drag to pan, or open a large detail view."
        )
        hint.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        actions.addWidget(hint, 1)
        open_button = QPushButton("Open Selected Graph")
        export_button = QPushButton("Export Analysis Report")
        close_button = QPushButton("Close")
        open_button.clicked.connect(self._open_selected_chart)
        export_button.clicked.connect(self._export_report)
        close_button.clicked.connect(self.close)
        actions.addWidget(open_button)
        actions.addWidget(export_button)
        actions.addWidget(close_button)
        layout.addLayout(actions)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        layout.addWidget(self.status_label)

        self.trial_selector.currentIndexChanged.connect(self._select_trial)
        self.taxel_map.taxelSelected.connect(self._select_taxel)
        self._apply_taxel(self.selected_taxel_index)

    @staticmethod
    def _original_taxel(dataset):
        metadata = dict(dataset.get("metadata") or {})
        return int(metadata.get("taxel_index", 0) or 0)

    def _available_taxels(self, dataset):
        return available_taxel_indices(
            dataset.get("rows") or [],
            dataset.get("signal_field") or "diff_ave",
            legacy_index=self._original_taxel(dataset),
        )

    def _project_dataset(self, dataset, taxel_index):
        signal_field = str(dataset.get("signal_field") or "diff_ave")
        original_taxel = self._original_taxel(dataset)
        rows = calibration_rows_for_taxel(
            dataset.get("rows") or [],
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
        repeatability_trials = []
        analyses = []
        for index, dataset in enumerate(self.datasets):
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
            repeatability_trials.append(
                {
                    "trial_number": index + 1,
                    "rows": projected["rows"],
                }
            )
        self.analyses = analyses
        self.batch_analysis = analyze_calibration_batch(analyses)
        signal_field = str(self.datasets[0].get("signal_field") or "diff_ave")
        self.charts["repeatability"].set_trials(
            repeatability_trials,
            signal_field,
        )
        repeatable = len(repeatability_trials) > 1
        self.tabs.setTabEnabled(self.repeatability_tab_index, repeatable)
        self.batch_report_view.setPlainText(
            build_batch_report(self.batch_analysis)
        )
        self.tabs.setTabEnabled(self.batch_tab_index, repeatable)
        self._select_trial(self._current_index())

    def _select_taxel(self, taxel_index):
        self._apply_taxel(int(taxel_index))

    def _select_trial(self, index):
        index = max(0, min(int(index), len(self.datasets) - 1))
        dataset = self.datasets[index]
        available = self._available_taxels(dataset)
        if self.selected_taxel_index not in available and available:
            original_taxel = self._original_taxel(dataset)
            fallback = (
                original_taxel
                if original_taxel in available
                else int(available[0])
            )
            self._apply_taxel(fallback)
            return
        self.taxel_map.set_taxels(available, self.selected_taxel_index)
        dataset = self.projected_datasets.get(index)
        analysis = self.analyses_by_index.get(index)
        if dataset is None or analysis is None:
            return
        self.report_view.setPlainText(build_trial_report(analysis))
        for mode, chart in self.charts.items():
            if mode == "repeatability":
                continue
            chart.set_data(dataset["rows"], dataset["signal_field"])
        original_taxel = self._original_taxel(self.datasets[index])
        self.taxel_context_label.setText(
            f"Viewing taxel {self.selected_taxel_index}\n"
            f"Physical target: taxel {original_taxel}"
        )
        self.status_label.setText(
            f"Taxel {self.selected_taxel_index} | Raw data: "
            f"{dataset['source_path']}"
        )

    def _current_index(self):
        return max(0, min(self.trial_selector.currentIndex(), len(self.datasets) - 1))

    def _open_selected_chart(self):
        widget = self.tabs.currentWidget()
        for mode, chart in self.charts.items():
            if widget is chart:
                self._open_chart_detail(mode)
                return
        self.status_label.setText(
            "Select one of the graph tabs to open it in detail."
        )

    def _open_chart_detail(self, mode):
        if mode not in self.charts:
            return
        if self._detail_dialog is not None:
            self._detail_dialog.close()
        chart = self.charts[mode]
        trial_results = None
        if mode == "repeatability":
            trial_results = chart.trial_results
        dialog = CalibrationGraphDetailDialog(
            f"{self.GRAPH_TITLES[mode]} — Taxel "
            f"{self.selected_taxel_index}",
            chart.rows,
            chart.signal_field,
            mode,
            parent=self,
            trial_results=trial_results,
        )
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.destroyed.connect(
            lambda _object=None, dialog=dialog: self._detail_closed(dialog)
        )
        self._detail_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _detail_closed(self, dialog):
        if self._detail_dialog is dialog:
            self._detail_dialog = None

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
            }
            with output_path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write("\n")
        self.status_label.setText(f"Analysis report saved: {output_path}")

    def closeEvent(self, event):
        detail = self._detail_dialog
        self._detail_dialog = None
        if detail is not None:
            detail.close()
        super().closeEvent(event)
