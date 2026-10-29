"""Auditable characterisation metrics; no hardware, filtering or file writes."""
from collections import defaultdict
import json
import numpy as np

from phd.ui.calibration_analysis import (
    analyze_calibration_rows, available_taxel_indices, taxel_signal_column,
    _smooth_polynomial_fit,
)
from phd.ui.experiment_quality import finite_number, unique_sensor_rows
from phd.dependence.characterisation_study import nominal_pressure_kpa


def phase_change(rows, phase, field):
    values = [finite_number(r.get(field)) for r in rows if r.get("phase") == phase]
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    count = max(1, min(5, len(values) // 2))
    start, end = float(np.median(values[:count])), float(np.median(values[-count:]))
    return {"start": start, "end": end, "change": end - start, "samples": len(values)}


def trial_characterisation(dataset):
    rows = list(dataset.get("rows") or [])
    fresh = unique_sensor_rows(rows)
    meta = dataset.get("metadata") or {}
    quality = meta.get("measurement_quality") or {}
    field = str(dataset.get("signal_field") or "diff_ave")
    baseline = [r for r in fresh if r.get("phase") == "force_baseline"]
    loading = [r for r in fresh if r.get("phase") == "approaching"]
    target = int(meta.get("physical_target_taxel_index", meta.get("taxel_index", 0)))
    reference = None
    reference_force = 0.8 * float(meta.get("contact_threshold_g") or 0)
    if loading:
        valid = [r for r in loading if finite_number(r.get("force_delta_abs_g")) is not None]
        if valid:
            reference = min(valid, key=lambda r: abs(float(r["force_delta_abs_g"]) - reference_force))
    spatial = []
    for index in available_taxel_indices(fresh, field, target):
        column = taxel_signal_column(field, index)
        values = [finite_number(r.get(column)) for r in baseline]
        values = [v for v in values if v is not None]
        if not values or reference is None or finite_number(reference.get(column)) is None:
            continue
        centre = float(np.median(values))
        response = float(reference[column]) - centre
        spatial.append({
            "taxel_index": index, "baseline": centre,
            "baseline_noise": float(1.4826 * np.median(np.abs(np.asarray(values) - centre))),
            "response_at_reference_force": response,
        })
    target_response = next((abs(r["response_at_reference_force"]) for r in spatial if r["taxel_index"] == target), None)
    for row in spatial:
        row["response_ratio_to_pressed_taxel"] = (
            abs(row["response_at_reference_force"]) / target_response
            if target_response is not None and target_response > 1e-9 else None
        )
    core = analyze_calibration_rows(fresh, field)
    warnings = list(core.get("warnings", []))
    interrupted = bool(quality.get("interrupted") or meta.get("force_pause_count", 0))
    complete = meta.get("stop_reason") == "contact_threshold"
    complete = complete and bool(quality.get("trial_completed", complete))
    if interrupted:
        warnings.append("Interrupted trial: excluded from automatic repeatability/model validation.")
    if not complete:
        warnings.append("Completion is unconfirmed or aborted: excluded from automatic comparisons.")
    if not quality:
        warnings.append("Legacy recording: source timestamps, load rating and interruption provenance may be incomplete.")
    recovery = phase_change(fresh, "recovery", field)
    if recovery and core.get("signal_baseline") is not None:
        recovery["final_offset_from_stationary_baseline"] = recovery["end"] - core["signal_baseline"]
    baseline_values = np.array([float(r[field]) for r in baseline if finite_number(r.get(field)) is not None])
    baseline_centre = float(np.median(baseline_values)) if len(baseline_values) else None
    baseline_sigma = float(1.4826*np.median(np.abs(baseline_values-baseline_centre))) if len(baseline_values) else None
    points = []
    contact_time = (core.get("contact_onset") or {}).get("elapsed_s")
    if baseline_centre is not None:
        for row in loading:
            force, value = finite_number(row.get("force_delta_abs_g")), finite_number(row.get(field))
            if (force is not None and value is not None and force > 0
                    and contact_time is not None and float(row.get("elapsed_s", 0)) >= float(contact_time)):
                points.append((force*.00980665, value-baseline_centre))
    sensitivity = None
    if len(points) >= 5 and np.ptp(np.asarray(points)[:, 0]) > 1e-6:
        sensitivity = float(np.polyfit(np.asarray(points)[:, 0], np.asarray(points)[:, 1], 1)[0])
    setup = quality.get("setup") or {}
    detection_threshold = None if baseline_sigma is None else max(3., 5.*baseline_sigma)
    detected = [] if detection_threshold is None else [force for force, value in points if abs(value) >= detection_threshold]
    return {
        "warnings": warnings,
        "eligible_for_comparison": complete and not interrupted and bool(quality) and quality.get("eligible_for_comparison", False),
        "duplicate_sensor_rows": len(rows) - len(fresh),
        "stationary_baseline_samples": len(baseline),
        "position_hold_signal": phase_change(fresh, "contact_hold", field),
        "position_hold_force_g": phase_change(fresh, "contact_hold", "force_delta_abs_g"),
        "unloaded_recovery": recovery,
        "reference_force_g": None if reference is None else reference["force_delta_abs_g"],
        "spatial_response": spatial,
        "force_response": {
            "baseline_robust_sigma_units": baseline_sigma,
            "linear_loading_slope_units_per_n": sensitivity,
            "exploratory_threshold_units": detection_threshold,
            "first_above_noise_force_n": min(detected) if detected and setup.get("force_reference_verified") else None,
            "nominal_pressure_at_reference_kpa": nominal_pressure_kpa(None if reference is None else float(reference["force_delta_abs_g"])*.00980665, setup),
            "interpretation": "Slope is a descriptive loading fit, not proof of linearity. First-above-noise force is exploratory, not a validated detection limit; independent repeated contacts and false alarms are required.",
        },
        "interpretation": "Hold is fixed position, not constant force. Spatial ratios include mechanical/electrical spread, not electrical crosstalk alone.",
    }


def _condition_key(dataset):
    meta = dataset.get("metadata") or {}
    quality = meta.get("measurement_quality") or {}
    return json.dumps({
        "sensor": quality.get("sensor_identity"),
        "shape": meta.get("sensor_shape"),
        "pressed_taxel": meta.get("physical_target_taxel_index", meta.get("taxel_index")),
        "viewed_taxel": meta.get("taxel_index"),
        "signal": dataset.get("signal_field"),
        "speed": meta.get("approach_speed_m_s"),
        "target_g": meta.get("contact_threshold_g"),
        "hold_s": meta.get("contact_hold_sec"),
        "recovery_s": quality.get("recovery_observation_s"),
        "recalibrate": quality.get("recalibrate_between_trials"),
        "protocol": quality.get("protocol"),
        "sliding": quality.get("sliding"),
        "notes": quality.get("notes"),
        "setup": quality.get("setup"),
    }, sort_keys=True)


def result_condition_groups(results):
    """Group in-memory acquisition results using the same policy as file analysis."""
    groups = defaultdict(list)
    for result in results:
        quality = result.get("measurement_quality") or {}
        if (
            result.get("stop_reason") != "contact_threshold"
            or not quality.get("trial_completed")
            or not quality.get("eligible_for_comparison")
            or quality.get("interrupted")
            or result.get("force_pause_count", 0)
        ):
            continue
        key = _condition_key({"metadata": result, "signal_field": result.get("signal_field")})
        groups[key].append(result)
    return dict(groups)


def comparable_trial_results(results, reference):
    key = _condition_key({"metadata": reference, "signal_field": reference.get("signal_field")})
    return result_condition_groups(results).get(key, [])


def contact_points(dataset):
    field = dataset.get("signal_field") or "diff_ave"
    rows = unique_sensor_rows(dataset.get("rows") or [])
    onset = analyze_calibration_rows(rows, field).get("contact_onset")
    if not onset:
        return []
    result, last_force = [], None
    for row in rows:
        if row.get("phase") != "approaching" or float(row.get("elapsed_s", 0)) < onset["elapsed_s"]:
            continue
        x, y = finite_number(row.get(field)), finite_number(row.get("force_delta_abs_g"))
        stamp = finite_number(row.get("force_monotonic_s"))
        if stamp is not None and stamp == last_force:
            continue
        last_force = stamp
        if x is not None and y is not None:
            result.append((x, y))
    return result


def matched_force_bands(curves):
    """Mean ± sample SD at matched force; never extrapolate."""
    ordered = []
    for curve in curves:
        grouped = defaultdict(list)
        for signal, force in curve:
            grouped[force].append(signal)
        values = [(f, float(np.mean(s))) for f, s in sorted(grouped.items())]
        if len(values) >= 2:
            ordered.append(np.asarray(values))
    if len(ordered) < 2:
        return None
    low, high = max(c[0, 0] for c in ordered), min(c[-1, 0] for c in ordered)
    if high <= low:
        return None
    force = np.linspace(low, high, 60)
    responses = np.asarray([np.interp(force, c[:, 0], c[:, 1]) for c in ordered])
    return {
        "force_g": force.tolist(), "mean_signal": responses.mean(axis=0).tolist(),
        "sd_signal": responses.std(axis=0, ddof=1).tolist(), "trials": len(ordered),
    }


def batch_characterisation(datasets):
    groups, excluded = defaultdict(list), []
    for dataset in datasets:
        if not trial_characterisation(dataset)["eligible_for_comparison"]:
            excluded.append(dataset.get("source_path") or dataset.get("name", "trial"))
            continue
        groups[_condition_key(dataset)].append(dataset)
    results = []
    for key, group in groups.items():
        curves = [contact_points(d) for d in group]
        errors, folds = [], []
        if len(curves) >= 2:
            for held_out, points in enumerate(curves):
                training = [point for i, curve in enumerate(curves) if i != held_out for point in curve]
                fit = _smooth_polynomial_fit(training)
                valid = [] if fit is None else [
                    p for p in points if fit["x_min"] <= p[0] <= fit["x_max"]
                ]
                if not valid:
                    continue
                predicted = np.polynomial.polynomial.polyval(
                    [p[0] for p in valid], fit["coefficients"]
                )
                residual = predicted - np.asarray([p[1] for p in valid])
                errors.extend(residual.tolist())
                folds.append({
                    "held_out_trial": group[held_out].get("source_path") or group[held_out].get("name"),
                    "training_trial_count": len(curves) - 1, "evaluated_samples": len(valid),
                    "outside_training_range_samples": len(points) - len(valid),
                    "rmse_g": float(np.sqrt(np.mean(residual ** 2))),
                    "mae_g": float(np.mean(np.abs(residual))),
                })
        results.append({
            "condition": json.loads(key), "trials": len(group),
            "matched_force_signal_band": matched_force_bands(curves),
            "leave_one_trial_out": folds,
            "held_out_rmse_g": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            "held_out_mae_g": float(np.mean(np.abs(errors))) if errors else None,
        })
    return {
        "groups": results, "excluded_trials": excluded,
        "interpretation": "Quadratic force-from-signal validation uses whole held-out trials, matching setup/speed/taxels, and training-range overlap only. Bands show between-trial SD, not confidence intervals. No extrapolation.",
    }


def characterisation_report(trial, batch):
    lines = ["CHARACTERISATION QUALITY", *trial["warnings"], "", trial["interpretation"]]
    lines.extend(["", "FORCE RESPONSE / DETECTION (EXPLORATORY)", json.dumps(trial.get("force_response"), ensure_ascii=False)])
    for label, key in (
        ("Fixed-position hold: signal", "position_hold_signal"),
        ("Fixed-position hold: force (g)", "position_hold_force_g"),
        ("Unloaded recovery", "unloaded_recovery"),
    ):
        lines.append(f"\n{label}: {json.dumps(trial.get(key), ensure_ascii=False)}")
    lines.extend(["", "FULL-GRID RESPONSE AT MATCHED INSTANT", f"Reference force: {trial['reference_force_g']} g"])
    for row in trial["spatial_response"]:
        ratio = row["response_ratio_to_pressed_taxel"]
        lines.append(
            f"Taxel {row['taxel_index']}: signal {row['response_at_reference_force']:+.3f}, "
            f"noise {row['baseline_noise']:.3f}, ratio to pressed taxel "
            + ("unavailable" if ratio is None else f"{ratio:.3f}")
        )
    lines.extend(["", "HELD-OUT TRIAL VALIDATION", batch["interpretation"],
                  f"Excluded recordings: {len(batch['excluded_trials'])}"])
    for group in batch["groups"]:
        lines.append(f"\nCondition: {json.dumps(group['condition'])}")
        lines.append(f"Trials: {group['trials']}; held-out RMSE: {group['held_out_rmse_g']} g; MAE: {group['held_out_mae_g']} g")
        band = group["matched_force_signal_band"]
        if band:
            lines.append("Matched force (g): mean signal ± between-trial SD")
            for f, m, s in zip(band["force_g"][::10], band["mean_signal"][::10], band["sd_signal"][::10]):
                lines.append(f"  {f:.2f}: {m:+.3f} ± {s:.3f}")
    return "\n".join(lines)
