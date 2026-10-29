"""Fixture-guided characterisation, native-frame capture and auditable analysis.

No robot command API is imported here. Positions/events are independent operator
annotations unless an external reference explicitly supplies them.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from phd.dependence.touch_tracking_data import TrialRecorder, load_trial, save_json


SCHEMA = "pinglab_characterisation_v1"
PROTOCOLS = {
    "spatial": ("Spatial response field", "Single probe/jig: move between known XY positions while unloaded; collect stationary contacts. This recorder never moves the robot."),
    "multi_touch": ("Multi-touch separation", "Manual fingers are allowed as labelled observations. Calibrated separation needs independently measured contact positions or a suitable fixture."),
    "size_speed": ("Finger size / gesture speed", "Repeat labelled gestures at measured or nominal size/speed. Nominal speed is not a physical velocity measurement."),
    "mounting": ("Mounting / remounting", "One unchanged mounting per recording. Record flat/cylinder/remount identity; compare independent trials offline."),
    "latency": ("Acquisition / latency audit", "Native host-buffer timestamps; manual event buttons are annotations, not independent physical latency ground truth."),
    "baseline": ("Noise / drift / robot interference", "Keep hands away. Record a declared robot state; this workbench does not power, move or stop the robot."),
}
EVENTS = {"condition_start", "release", "gesture_start", "gesture_end", "contact_reference", "command_reference", "motion_reference"}


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def display_vectors(matrices):
    """Native raw buffer -> existing Sensor signal taxel index row+column*rows."""
    values = np.asarray(matrices, dtype=float)
    return values[:, ::-1, :].transpose(0, 2, 1).reshape(len(values), -1)


def validate_condition(condition):
    result = dict(condition)
    count = result.get("contact_count", 1)
    if type(count) is not int or not 0 <= count <= 5:
        raise ValueError("Contact count must be an integer from 0 to 5.")
    result["contact_count"] = count
    for key in ("x_mm", "y_mm", "second_x_mm", "second_y_mm", "probe_diameter_mm", "nominal_speed_mm_s"):
        value = result.get(key)
        if value in (None, ""):
            result[key] = None
        elif finite(value) is None:
            raise ValueError(f"{key} must be finite or left unknown.")
        else:
            result[key] = float(value)
    for x, y in (("x_mm", "y_mm"), ("second_x_mm", "second_y_mm")):
        if (result[x] is None) != (result[y] is None):
            raise ValueError("Provide both coordinates of each known contact, or neither.")
    if result["probe_diameter_mm"] is not None and result["probe_diameter_mm"] <= 0:
        raise ValueError("Probe diameter must be positive or unknown.")
    if result["nominal_speed_mm_s"] is not None and result["nominal_speed_mm_s"] < 0:
        raise ValueError("Nominal speed cannot be negative.")
    source = str(result.get("position_source") or "unknown")
    if source not in {"unknown", "operator_estimate", "measured_fixture", "stationary_robot_tcp"}:
        raise ValueError("Choose a known position-reference source.")
    result["position_source"] = source
    if source != "unknown" and result["x_mm"] is None:
        raise ValueError("A position-reference source requires known XY coordinates.")
    return result


class CharacterisationRecorder:
    """Thread-safe native listener plus independent, timestamped annotations.

    Raw chunk writes are asynchronous/bounded through TrialRecorder. GUI code
    must finalize in a worker thread, not block acquisition on disk writes.
    """
    def __init__(self, root, metadata, *, baseline_sec=5., **writer_options):
        metadata = dict(metadata)
        if metadata.get("protocol") not in PROTOCOLS:
            raise ValueError("Select a characterisation protocol.")
        for key in ("specimen_id", "session_id", "mounting_id"):
            if not str(metadata.get(key, "")).strip():
                raise ValueError(f"Provide {key}; do not merge different specimens or mountings.")
        if not metadata.get("orientation_confirmed") or not metadata.get("baseline_confirmed"):
            raise ValueError("Confirm grid orientation and an untouched initial baseline.")
        if finite(baseline_sec) is None or not 2 <= baseline_sec <= 60:
            raise ValueError("Baseline duration must be 2–60 seconds.")
        metadata.update(
            sensor_id=metadata["specimen_id"], participant_id=metadata.get("participant_id") or "operator",
            supervision="fixture_guided_characterisation", robot_commands_enabled=False,
        )
        self.raw = TrialRecorder(root, metadata, **writer_options)
        self.path, self.started = self.raw.path, self.raw.started
        self.baseline_sec = float(baseline_sec)
        self.lock = threading.RLock()
        self._closed = False
        self.protocol = {
            "schema": SCHEMA, "status": "recording", "protocol": metadata["protocol"],
            "baseline_sec": self.baseline_sec, "baseline_min_frames": 20,
            "taxel_order": "display row + column*rows; native raw rows flipped",
            "robot_commands_enabled": False,
            "events": [], "references": [],
            "interpretation": "Manual events are not ground-truth physical onset. Raw chunks use native rows and host-buffer timestamps; upstream acquisition filtering/masking may already apply.",
        }
        try:
            self._save()
        except Exception:
            self.raw.close("Could not create characterisation manifest.")
            raise

    @property
    def error(self):
        return self.raw.error

    @property
    def sensor_count(self):
        return self.raw.sensor_count

    def _save(self):
        save_json(self.path / "characterisation.json", self.protocol)

    def sensor_frame(self, timestamp, sequence, raw, baseline, valid_taxels=None, *, filtered_signed=None):
        with self.lock:
            return self.raw.sensor_frame(timestamp, sequence, raw, baseline, valid_taxels, filtered_signed=filtered_signed)

    def mark(self, kind, condition=None, *, timestamp=None, source="operator_button"):
        now = time.monotonic() if timestamp is None else float(timestamp)
        with self.lock:
            if self._closed or self.error:
                raise ValueError("Recording is closed or failed.")
            if kind not in EVENTS or source not in {"operator_button", "external_host_monotonic"}:
                raise ValueError("Invalid event type or clock source.")
            elapsed = now - self.started
            if not math.isfinite(elapsed) or elapsed < self.baseline_sec or self.sensor_count < 20:
                raise ValueError("Wait for the untouched baseline and at least 20 native frames.")
            if self.protocol["events"] and elapsed <= self.protocol["events"][-1]["t"]:
                raise ValueError("Event times must increase.")
            event = {"t": elapsed, "kind": kind, "source": source}
            if kind in {"condition_start", "gesture_start"}:
                event["condition"] = validate_condition(condition or {})
            self.protocol["events"].append(event)
            try:
                self._save()
            except Exception as exc:
                self.raw.error = f"Could not checkpoint event: {exc}"
                raise
            return event

    def reference(self, values, *, timestamp=None):
        """Latest reference samples, not falsely labelled synchronous readings."""
        now = time.monotonic() if timestamp is None else float(timestamp)
        with self.lock:
            if self._closed:
                return
            row = {"t": now - self.started}
            for key in ("force_n", "force_source_monotonic", "robot_source_monotonic", "x_m", "y_m", "z_m"):
                row[key] = finite(values.get(key))
            row["force_age_s"] = None if row["force_source_monotonic"] is None else now - row["force_source_monotonic"]
            row["robot_age_s"] = None if row["robot_source_monotonic"] is None else now - row["robot_source_monotonic"]
            if not self.protocol["references"] or row["t"] > self.protocol["references"][-1]["t"]:
                self.protocol["references"].append(row)

    def close(self, reason=None):
        with self.lock:
            if self._closed:
                return self.path
            self._closed = True
        self.raw.close(reason)
        with self.lock:
            self.protocol.update(status="failed" if self.error else "complete", error=self.error)
            self._save()
            references = self.protocol["references"]
            if references:
                with (self.path / "references.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(references[0]))
                    writer.writeheader()
                    writer.writerows(references)
        return self.path


def peak_candidates(response, shape, threshold):
    """Algorithm-dependent local maxima, never certified physical contacts."""
    rows, cols = shape
    grid = np.asarray(response).reshape(cols, rows).T
    selected = []
    for index in np.argsort(response)[::-1]:
        row, col = int(index % rows), int(index // rows)
        if response[index] < threshold[index] or response[index] <= 0:
            continue
        neighbours = grid[max(0, row-1):row+2, max(0, col-1):col+2]
        if grid[row, col] < np.max(neighbours):
            continue
        if any(abs(row-r) <= 1 and abs(col-c) <= 1 for r, c in selected):
            continue
        selected.append((row, col))
    return [{"taxel_index": r+c*rows, "row": r, "column": c} for r, c in selected]


def _segments(events, end):
    current = None
    for event in events:
        if event["kind"] in {"condition_start", "gesture_start"}:
            if current is not None:
                yield current, float(event["t"])
            current = event
        elif event["kind"] in {"release", "gesture_end"} and current is not None:
            yield current, float(event["t"])
            current = None
    if current is not None:
        yield current, float(end)


def analyse_study(path):
    path = Path(path)
    metadata, arrays = load_trial(path)
    protocol = json.loads((path / "characterisation.json").read_text(encoding="utf-8"))
    if protocol.get("schema") != SCHEMA:
        raise ValueError("Select a Sensor Characterisation trial folder.")
    times = arrays["t"]
    signal = display_vectors(arrays["raw"] - arrays["baseline"])
    mask = display_vectors(arrays["valid_taxels"]).astype(bool)
    if not np.isfinite(signal).all():
        raise ValueError("Non-finite sensor samples; recording cannot be analysed as valid data.")
    baseline_mask = times < float(protocol["baseline_sec"])
    if np.count_nonzero(baseline_mask) < 20 or np.ptp(times[baseline_mask]) < .8 * float(protocol["baseline_sec"]):
        raise ValueError("Insufficient untouched baseline coverage; repeat the trial.")
    baseline = np.median(signal[baseline_mask], axis=0)
    noise = 1.4826 * np.median(np.abs(signal[baseline_mask] - baseline), axis=0)
    enabled = np.all(mask, axis=0)
    if not np.any(enabled):
        raise ValueError("No valid taxels.")
    threshold = np.maximum(3., 5. * noise)
    differences = np.diff(times)
    quality = {
        "native_host_buffer_frames": len(times),
        "observed_rate_hz": float((len(times)-1)/(times[-1]-times[0])),
        "median_interval_ms": float(1000*np.median(differences)),
        "p95_interval_ms": float(1000*np.percentile(differences, 95)),
        "sequence_gaps": int(np.maximum(np.diff(arrays["sequence"])-1, 0).sum()),
        "source": "Host completed-buffer callback, not hardware scan timestamps; gaps do not establish firmware-level dropped frames.",
        "complete": metadata.get("status") == protocol.get("status") == "complete",
        "filtered_signed_available": bool("filtered_signed" in arrays and np.isfinite(arrays["filtered_signed"]).all()),
        "baseline_duration_s": float(protocol["baseline_sec"]),
    }
    results = []
    shape = tuple(metadata["sensor_shape"])
    for event, end in _segments(protocol.get("events", []), float(times[-1])):
        selected = (times >= float(event["t"]) + .2) & (times < end)
        if np.count_nonzero(selected) < 5:
            continue
        signed = np.where(enabled, np.median(signal[selected], axis=0) - baseline, 0.)
        response = np.where(enabled, np.abs(signed), 0.)
        active = enabled & (response >= threshold)
        weights = np.where(active, response, 0.)
        mass = float(weights.sum())
        indices = np.arange(signal.shape[1])
        centroid = None if mass <= 0 else [float(np.dot(indices // shape[0], weights)/mass), float(np.dot(indices % shape[0], weights)/mass)]
        candidates = peak_candidates(response, shape, np.where(enabled, threshold, np.inf))
        condition = validate_condition(event.get("condition", {}))
        frame_responses = np.abs(signal[selected]-baseline)
        false_frames = np.any((frame_responses >= threshold) & enabled, axis=1)
        episodes = int(np.count_nonzero(np.diff(np.r_[False, false_frames].astype(int)) == 1))
        force = [r["force_n"] for r in protocol.get("references", [])
                 if float(event["t"]) <= r["t"] < end and r.get("force_n") is not None
                 and r.get("force_age_s") is not None and 0 <= r["force_age_s"] <= .25]
        results.append({
            "start_s": event["t"], "end_s": end, "condition": condition,
            "samples": int(selected.sum()), "peak_response_units": float(np.max(response)),
            "active_taxels": int(active.sum()), "centroid_column_row": centroid,
            "signed_response_by_taxel": signed.tolist(), "peak_candidates": candidates,
            "two_peak_candidates": len(candidates) >= 2,
            "median_latest_force_n": None if not force else float(np.median(force)),
            "declared_no_contact_false_activation_episodes_per_min": 60.*episodes/(end-float(event["t"])-.2) if condition["contact_count"] == 0 else None,
        })
    warnings = [
        "Response fields include mechanical spreading and electrical coupling; they are not electrical-crosstalk isolation.",
        "Centroids are grid coordinates, not independently validated millimetre localisation. Peak counts are algorithm-dependent, not a calibrated two-point resolution.",
        "Nominal gesture speed and operator event times are declarations, not independent physical measurements.",
    ]
    if not quality["complete"]:
        warnings.append("Failed/interrupted acquisition: exploratory analysis only; exclude from final comparisons.")
    # Native intervals set the available temporal granularity; UI-reference
    # samples and manual buttons cannot silently become a millisecond result.
    events = protocol.get("events", [])
    latency = []
    onset = None
    for event in sorted(events, key=lambda e: e["t"]):
        if event["kind"] in {"release", "condition_start", "gesture_start"}:
            onset = None
        if event.get("source") != "external_host_monotonic":
            continue
        if event["kind"] == "contact_reference":
            onset = event
        elif event["kind"] == "command_reference" and onset is not None:
            latency.append({"contact_to_command_ms": 1000*(event["t"]-onset["t"]),
                            "timing_uncertainty_ms": float(onset.get("uncertainty_ms", 0)) + float(event.get("uncertainty_ms", 0)),
                            "contact_s": onset["t"], "command_s": event["t"]})
            onset = None
    if not latency:
        warnings.append("No independently timestamped contact/command pair; end-to-end latency is unmeasured.")
    baseline_values = signal[baseline_mask]
    half = max(1, len(baseline_values)//2)
    drift = np.median(baseline_values[-half:], axis=0)-np.median(baseline_values[:half], axis=0)
    report = {"schema": SCHEMA, "metadata": metadata, "acquisition_quality": quality,
              "valid_taxels": enabled.tolist(),
              "baseline_median_units": baseline.tolist(), "baseline_robust_sigma_units": noise.tolist(),
              "baseline_first_to_last_half_shift_units": drift.tolist(),
              "segments": results, "independent_latency_events": latency, "warnings": warnings}
    return report


def export_study_report(path, report):
    path = Path(path)
    save_json(path / "analysis.json", report)
    with (path / "condition_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["start_s", "end_s", "samples", "peak_response_units", "active_taxels", "two_peak_candidates", "median_latest_force_n", "declared_no_contact_false_activation_episodes_per_min", "gesture_label", "mounting_id", "probe_diameter_mm", "nominal_speed_mm_s", "contact_count", "position_source", "x_mm", "y_mm"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for segment in report["segments"]:
            values = {**segment, **segment["condition"]}
            writer.writerow({key: values.get(key) for key in fields})


def compare_studies(reports):
    """Do not conflate different specimens/layouts; retain mounting conditions."""
    reports = list(reports)
    if not reports:
        raise ValueError("Select at least one analysed trial.")
    keys = {(r["metadata"]["specimen_id"], tuple(r["metadata"]["sensor_shape"])) for r in reports}
    if len(keys) != 1:
        raise ValueError("Compare one specimen and native grid at a time (10×10 and 20×20 are not interchangeable).")
    groups = {}
    excluded = []
    for report in reports:
        if not report["acquisition_quality"]["complete"]:
            excluded.append(report["metadata"]["trial_id"])
            continue
        for segment in report["segments"]:
            condition = segment["condition"]
            group = {key: condition.get(key) for key in ("gesture_label", "mounting_id", "probe_diameter_mm", "nominal_speed_mm_s", "contact_count", "position_source", "x_mm", "y_mm", "second_x_mm", "second_y_mm")}
            group["mounting_id"] = condition.get("mounting_id") or report["metadata"]["mounting_id"]
            for name in ("protocol", "robot_condition", "sensor_model", "reorder_mode", "notes"):
                group[name] = report["metadata"].get(name)
            group["baseline_duration_s"] = report["acquisition_quality"].get("baseline_duration_s")
            key = json.dumps(group, sort_keys=True)
            groups.setdefault(key, {}).setdefault(report["metadata"]["trial_id"], []).append(segment["peak_response_units"])
    output = []
    for key, trials in groups.items():
        # One value per trial: adjacent frames/segments are not independent runs.
        values = [float(np.mean(v)) for v in trials.values()]
        output.append({"condition": json.loads(key), "independent_trials": len(values),
                       "mean_peak_units": float(np.mean(values)),
                       "between_trial_sd_units": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    return {"groups": output, "excluded_trials": excluded, "interpretation": "Per-trial averages; between-trial SD is not a confidence interval. Nominal conditions remain declarations."}


def compare_map_reference(result, references, *, same_frame_confirmed=False):
    if not same_frame_confirmed:
        raise ValueError("Confirm references are independent measurements in the SAME robot-base frame and unchanged setup.")
    if result.get("task") != "taxel_position_mapping":
        raise ValueError("Select a taxel position map.")
    positions = {}
    for row in result.get("positions", []):
        if row.get("status") != "measured":
            continue
        index = int(row["taxel_index"])
        if index in positions:
            raise ValueError("Duplicate measured map taxel index.")
        positions[index] = row
    rows, seen = [], set()
    for reference in references:
        value = finite(reference["taxel_index"])
        if value is None or not value.is_integer() or value < 0:
            raise ValueError("Reference taxel index must be a nonnegative integer.")
        index = int(value)
        if index in seen:
            raise ValueError("Duplicate reference taxel index.")
        seen.add(index)
        if index not in positions:
            raise ValueError(f"Taxel {index} has no measured map position.")
        xy = [finite(reference.get(key)) for key in ("x_mm", "y_mm")]
        if any(v is None for v in xy):
            raise ValueError("Reference x_mm/y_mm must be finite robot-base millimetres.")
        measured = np.asarray(positions[index]["tool_position_m"][:2], dtype=float)*1000.
        if measured.shape != (2,) or not np.isfinite(measured).all():
            raise ValueError("Non-finite mapped position.")
        error = measured - np.asarray(xy)
        rows.append({"taxel_index": index, "dx_mm": float(error[0]), "dy_mm": float(error[1]), "xy_error_mm": float(np.linalg.norm(error))})
    if not rows:
        raise ValueError("No independent reference points supplied.")
    errors = np.array([r["xy_error_mm"] for r in rows])
    return {"points": rows, "count": len(rows), "xy_rmse_mm": float(np.sqrt(np.mean(errors**2))),
            "mean_xy_error_mm": float(errors.mean()), "max_xy_error_mm": float(errors.max()),
            "reference_policy": "Independent measured references; no alignment fitted to minimise the reported error. Only XY compared because mapping proximity Z is not geometric surface Z.",
            "warning": "Sparse references do not establish whole-sensor accuracy."}


def nominal_pressure_kpa(force_n, setup):
    if not setup.get("force_reference_verified") or not setup.get("contact_area_confirmed") or setup.get("probe_geometry") != "flat_circular":
        return None
    diameter = finite(setup.get("probe_diameter_mm"))
    force = finite(force_n)
    if diameter is None or diameter <= 0 or force is None:
        return None
    return force / (math.pi*(diameter/2.)**2)*1000.


def import_event_csv(trial_path, csv_path, *, clock_alignment_confirmed=False):
    """Import declared independent events in recording-relative host seconds."""
    if not clock_alignment_confirmed:
        raise ValueError("Confirm the external device events have been independently aligned to this recording's host-monotonic clock; video/device seconds alone are not sufficient.")
    path = Path(trial_path)
    metadata, arrays = load_trial(path)
    protocol = json.loads((path / "characterisation.json").read_text(encoding="utf-8"))
    if protocol.get("schema") != SCHEMA or metadata.get("status") != "complete":
        raise ValueError("Select a completed characterisation recording.")
    imported = []
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            stamp = finite(row.get("t_s"))
            if stamp is None or not float(protocol["baseline_sec"]) <= stamp <= float(arrays["t"][-1]):
                raise ValueError("External t_s must lie after the baseline and within the recorded native frames.")
            if row.get("kind") not in {"contact_reference", "command_reference", "motion_reference"}:
                raise ValueError("External kind must be contact_reference, command_reference or motion_reference.")
            uncertainty = finite(row.get("uncertainty_ms"))
            if uncertainty is None or uncertainty < 0:
                raise ValueError("Declare each external event's nonnegative timing uncertainty_ms.")
            imported.append({"t": stamp, "kind": row["kind"], "source": "external_host_monotonic", "uncertainty_ms": uncertainty, "external_source": str(Path(csv_path).resolve())})
    if not imported or any(b["t"] < a["t"] for a, b in zip(imported, imported[1:])):
        raise ValueError("Provide chronologically ordered external events.")
    # Preserve manual annotations and earlier imports. Duplicate device events
    # must not manufacture repeated latency observations.
    old = protocol["events"]
    existing = {(e["t"], e["kind"], e.get("source")) for e in old}
    additions = []
    for event in imported:
        key = (event["t"], event["kind"], event["source"])
        if key not in existing:
            additions.append(event)
            existing.add(key)
    protocol["events"] = sorted(old + additions, key=lambda e: e["t"])
    protocol["external_clock_alignment"] = "Operator declares an independently aligned host-monotonic reference; software cannot verify physical synchronisation."
    save_json(path / "characterisation.json", protocol)


def study_plot_data(path, report):
    """Prepare bounded plot arrays off the UI thread; full samples stay on disk."""
    metadata, arrays = load_trial(path)
    values = display_vectors(arrays["raw"] - arrays["baseline"])
    centre = np.asarray(report["baseline_median_units"])
    valid = np.asarray(report["valid_taxels"], bool)
    stride = max(1, math.ceil(len(values) / 4000))
    peak = np.max(np.where(valid, np.abs(values-centre), 0.), axis=1)
    packet = {"t": arrays["t"][::stride], "peak": peak[::stride],
              "shape": tuple(metadata["sensor_shape"])}
    if report["acquisition_quality"].get("filtered_signed_available"):
        averaged = display_vectors(arrays["filtered_signed"])
        baseline = np.median(averaged[arrays["t"] < report["acquisition_quality"]["baseline_duration_s"]], axis=0)
        packet["filtered_peak"] = np.max(np.where(valid, np.abs(averaged-baseline), 0.), axis=1)[::stride]
    return packet


def evaluate_prediction_csv(trial_path, csv_path):
    """Audit externally produced predictions against independent trial labels.

    This does not train a model or assert that predictions are held out. The
    prediction producer must preserve source/user split provenance separately.
    """
    path = Path(trial_path)
    metadata, arrays = load_trial(path)
    protocol = json.loads((path / "characterisation.json").read_text())
    if protocol.get("schema") != SCHEMA or metadata.get("status") != "complete":
        raise ValueError("Prediction evaluation needs a completed characterisation trial.")
    segments = list(_segments(protocol["events"], arrays["t"][-1]))
    comparisons = []
    last = -math.inf
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            stamp = finite(row.get("t_s"))
            if stamp is None or stamp <= last or not 0 <= stamp <= arrays["t"][-1]:
                raise ValueError("Prediction t_s must increase and lie within this trial.")
            last = stamp
            predicted = str(row.get("predicted_label") or "").strip()
            if not predicted:
                raise ValueError("Missing predicted_label.")
            for start, end in segments:
                # Boundary uncertainty excludes rapid label-transition timing.
                if start["t"] + .25 <= stamp <= end - .25:
                    truth = str(start["condition"].get("gesture_label") or "").strip()
                    if truth:
                        comparisons.append((truth, predicted))
                    break
    if not comparisons:
        raise ValueError("No predictions overlap independently labelled gesture intervals away from their boundaries.")
    labels = sorted({v for pair in comparisons for v in pair})
    matrix = np.zeros((len(labels), len(labels)), int)
    for truth, predicted in comparisons:
        matrix[labels.index(truth), labels.index(predicted)] += 1
    f1 = []
    for index in range(len(labels)):
        denominator = matrix[index].sum() + matrix[:, index].sum()
        f1.append(0. if not denominator else float(2.*matrix[index, index]/denominator))
    return {"labels": labels, "confusion": matrix.tolist(), "prediction_windows": len(comparisons),
            "accuracy": float(np.trace(matrix)/matrix.sum()), "macro_f1": float(np.mean(f1)),
            "trial_id": metadata["trial_id"], "participant_id": metadata.get("participant_id"),
            "session_id": metadata["session_id"],
            "interpretation": "Correlated prediction windows within one trial, not independent participants. Split original user/session/source recordings before augmentation; this import cannot verify model training provenance."}
