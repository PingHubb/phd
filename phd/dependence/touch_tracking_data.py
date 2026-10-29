"""Independent-reference touch tracking data, without robot/controller imports.

Coordinates are x=column, y=row, normalized to the measured sensor surface.
Frames use the native (row, column) layout, NOT the legacy transposed/flipped
AI layout. Timestamps are host monotonic receipt/processing times, not claimed
to be hardware-synchronised exposure times. No tactile threshold creates labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import queue
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np


SCHEMA = "pinglab.touch_tracking.v1"
LAYOUT = "native_row_col_unflipped"
REFERENCE_SOURCES = {"manual_camera", "external_tracking", "robot_probe"}
CHANNELS = ["signed_change_percent", "change_percent_per_second", "valid_taxel", "sample_age_seconds"]


def safe_name(value):
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value).strip()).strip("_")
    if not value:
        raise ValueError("Participant, session and sensor identifiers cannot be empty.")
    return value[:90]


def save_json(path, value):
    """Atomic sidecar replacement. Recorded raw chunks are never overwritten."""
    path = Path(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


class TrialRecorder:
    """Bounded disk writer. Overload fails visibly rather than silently losing data.

    A manifest and small independent sensor chunks allow partial trial recovery.
    Camera JPEGs retain their individual timestamps; assumed video FPS is never
    used to align labels. Call close in a background thread when using a GUI.
    """

    def __init__(self, root, metadata, *, chunk_size=128, queue_size=96):
        metadata = dict(metadata)
        for key in ("participant_id", "session_id", "sensor_id"):
            metadata[key] = safe_name(metadata.get(key, ""))
        shape = tuple(int(v) for v in metadata["sensor_shape"])
        if len(shape) != 2 or min(shape) < 2:
            raise ValueError("A sensor grid of at least 2 x 2 is required.")
        self.shape = shape
        self.started = time.monotonic()
        trial_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:6]
        self.path = Path(root) / (metadata["sensor_id"] + "_" + trial_id)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "sensor").mkdir()
        (self.path / "camera").mkdir()
        self.manifest = {
            **metadata, "schema": SCHEMA, "layout": LAYOUT, "trial_id": trial_id,
            "created_at": datetime.now().astimezone().isoformat(), "status": "recording",
            "clock": "host_monotonic", "started_monotonic": self.started,
            "sensor_timestamp_kind": "host_buffer_update",
            "camera_timestamp_kind": "host_color_frame_receipt",
            "robot_commands_enabled": False,
            "labels": ("operator_intervals_and_sensor_estimates" if metadata.get("supervision") == "camera_free_sensor_estimates_v1"
                       else "independent_reference_required"),
        }
        save_json(self.path / "manifest.json", self.manifest)
        self.error = None
        self.sensor_count = 0
        self.camera_count = 0
        self.chunk_size = int(chunk_size)
        self._queue = queue.Queue(maxsize=queue_size)
        self._last_sequence = None
        self._last_time = None
        self._last_camera_time = None
        self._camera_shape = None
        self._baseline = None
        self._mask = None
        self._closed = False
        self._close_lock = threading.Lock()
        self._worker = threading.Thread(target=self._write_loop, name="touch-trial-writer", daemon=True)
        self._worker.start()

    def _submit(self, item):
        if self.error or self._closed:
            return False
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self.error = "Recording writer overloaded. Trial stopped; partial data retained."
            return False

    def sensor_frame(self, timestamp, sequence, raw, baseline, valid_taxels=None, *, filtered_signed=None):
        if self.error or self._closed:
            return False
        if sequence == self._last_sequence:
            return False
        raw = np.array(raw, dtype=np.float32, copy=True)
        baseline = np.array(baseline, dtype=np.float32, copy=True)
        mask = np.ones(self.shape, bool) if valid_taxels is None else np.array(valid_taxels, bool, copy=True)
        if raw.shape != self.shape or baseline.shape != self.shape or mask.shape != self.shape:
            self.error = "Sensor geometry changed during recording; start a new trial."
            return False
        if self._last_sequence is not None and sequence < self._last_sequence:
            self.error = "Sensor frame counter restarted; reconnect/calibration requires a new trial."
            return False
        if not math.isfinite(timestamp) or timestamp < self.started:
            return False
        if self._last_time is not None and timestamp <= self._last_time:
            self.error = "Sensor timestamps are not increasing."
            return False
        if self._baseline is not None and (
            not np.array_equal(baseline, self._baseline, equal_nan=True)
            or not np.array_equal(mask, self._mask)
        ):
            self.error = "Calibration or disabled-taxel mask changed; start a new trial."
            return False
        self._baseline, self._mask = baseline, mask
        filtered = np.full(self.shape, np.nan, np.float32) if filtered_signed is None else np.array(filtered_signed, dtype=np.float32, copy=True)
        if filtered.shape != self.shape or (filtered_signed is not None and not np.isfinite(filtered).all()):
            self.error = "Filtered sensor array is invalid."
            return False
        if self._submit(("sensor", (float(timestamp - self.started), int(sequence), raw, baseline, mask, filtered))):
            self._last_time, self._last_sequence = timestamp, sequence
            self.sensor_count += 1
            return True
        return False

    def camera_frame(self, timestamp, rgb):
        if timestamp < self.started or not math.isfinite(timestamp):
            return False
        if self._last_camera_time is not None and timestamp <= self._last_camera_time:
            return False
        rgb = np.array(rgb, dtype=np.uint8, copy=True)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            self.error = "Camera reference must be an unannotated RGB color image."
            return False
        if self._camera_shape is not None and rgb.shape != self._camera_shape:
            self.error = "Reference camera resolution changed; start a new calibrated trial."
            return False
        self._camera_shape = rgb.shape
        if self._submit(("camera", (float(timestamp - self.started), rgb))):
            self._last_camera_time = timestamp
            self.camera_count += 1
            return True
        return False

    def _write_loop(self):
        rows, chunk_index, image_index = [], 0, 0
        def flush():
            nonlocal rows, chunk_index
            if not rows:
                return
            t, seq, raw, base, mask, filtered = zip(*rows)
            final = self.path / "sensor" / f"{chunk_index:06d}.npz"
            tmp = final.with_suffix(".tmp")
            with tmp.open("wb") as handle:
                np.savez_compressed(handle, t=np.array(t), sequence=np.array(seq, np.int64),
                                    raw=np.stack(raw), baseline=np.stack(base), valid_taxels=np.stack(mask),
                                    filtered_signed=np.stack(filtered))
            tmp.replace(final)
            rows = []
            chunk_index += 1
        try:
            with (self.path / "camera.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["t", "image"])
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    kind, payload = item
                    if kind == "sensor":
                        rows.append(payload)
                        if len(rows) >= self.chunk_size:
                            flush()
                    else:
                        import cv2
                        timestamp, rgb = payload
                        name = f"camera/{image_index:06d}.jpg"
                        if not cv2.imwrite(str(self.path / name), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92]):
                            raise OSError("Could not write reference image.")
                        writer.writerow([f"{timestamp:.9f}", name])
                        handle.flush()
                        image_index += 1
                flush()
        except Exception as exc:
            self.error = f"Disk recording failed: {exc}"
        finally:
            self.manifest.update(status="failed" if self.error else "complete", error=self.error,
                                 sensor_frames=self.sensor_count, camera_frames=image_index,
                                 duration_sec=time.monotonic() - self.started)
            try:
                save_json(self.path / "manifest.json", self.manifest)
            except Exception as exc:
                self.error = f"Could not finalize manifest: {exc}"

    def close(self, reason=None):
        with self._close_lock:
            if self._closed:
                return self.path
            self._closed = True
            if reason:
                self.error = self.error or str(reason)
            while self._worker.is_alive():
                try:
                    self._queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self._worker.join()
        return self.path


def load_trial(path):
    path = Path(path)
    meta = json.loads((path / "manifest.json").read_text())
    if meta.get("schema") != SCHEMA or meta.get("layout") != LAYOUT:
        raise ValueError(f"{path.name}: not an independent-reference touch tracking trial.")
    parts = []
    for file in sorted((path / "sensor").glob("*.npz")):
        with np.load(file, allow_pickle=False) as data:
            parts.append({key: np.array(data[key]) for key in ("t", "sequence", "raw", "baseline", "valid_taxels")})
            if "filtered_signed" in data:
                parts[-1]["filtered_signed"] = np.array(data["filtered_signed"])
    if not parts:
        raise ValueError(f"{path.name}: no completed sensor chunks.")
    keys = set.intersection(*(set(part) for part in parts))
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    n = len(arrays["t"])
    shape = tuple(meta["sensor_shape"])
    if n < 2 or arrays["t"].shape != (n,) or arrays["sequence"].shape != (n,):
        raise ValueError(f"{path.name}: invalid timestamp/sequence arrays.")
    if not np.isfinite(arrays["t"]).all() or np.any(np.diff(arrays["t"]) <= 0):
        raise ValueError(f"{path.name}: timestamps must be finite and increasing.")
    if np.any(np.diff(arrays["sequence"]) <= 0):
        raise ValueError(f"{path.name}: duplicate or reset sensor sequence.")
    if any(arrays[k].shape != (n, *shape) for k in ("raw", "baseline", "valid_taxels")):
        raise ValueError(f"{path.name}: sensor array shape mismatch.")
    if "filtered_signed" in arrays and arrays["filtered_signed"].shape != (n, *shape):
        raise ValueError(f"{path.name}: filtered sensor array shape mismatch.")
    return meta, arrays


def camera_index(path):
    path = Path(path).resolve()
    rows = []
    with (path / "camera.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            file = (path / row["image"]).resolve()
            if not file.is_relative_to(path) or not file.is_file():
                raise ValueError("Reference image missing or outside trial folder.")
            t = float(row["t"])
            if not math.isfinite(t) or (rows and t <= rows[-1][0]):
                raise ValueError("Camera timestamps must increase.")
            rows.append((t, file))
    return rows


def homography_from_corners(corners):
    """Four sensor-plane corners in order (0,0), (1,0), (1,1), (0,1)."""
    p = np.asarray(corners, float)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise ValueError("Select four finite, ordered sensor-plane corners.")
    edges = np.roll(p, -1, axis=0) - p
    cross = np.cross(edges, np.roll(edges, -1, axis=0))
    if not (np.all(cross > 1) or np.all(cross < -1)):
        raise ValueError("Corners must form a non-degenerate convex quadrilateral, in perimeter order.")
    a, b = [], []
    for (u, v), (x, y) in zip(p, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        a.extend([[u, v, 1, 0, 0, 0, -x*u, -x*v], [0, 0, 0, u, v, 1, -y*u, -y*v]])
        b.extend([x, y])
    h = np.append(np.linalg.solve(a, b), 1).reshape(3, 3)
    return h


def project_position(h, pixel):
    q = np.asarray(h, float) @ np.array([*pixel, 1.0])
    if not np.isfinite(q).all() or abs(q[2]) < 1e-10:
        raise ValueError("Invalid camera-to-sensor projection.")
    xy = q[:2] / q[2]
    if np.any(xy < -1e-7) or np.any(xy > 1 + 1e-7):
        raise ValueError("Point is outside the calibrated sensor surface.")
    return np.clip(xy, 0, 1).tolist()


def validate_reference(reference):
    if reference.get("schema") != SCHEMA:
        raise ValueError("Reference schema does not match touch tracking.")
    if reference.get("source") not in REFERENCE_SOURCES:
        raise ValueError("Reference must come from reviewed camera annotation, external tracking or a measured probe.")
    if reference.get("verified") is not True:
        raise ValueError("Reference timing, coordinates and contact labels must be reviewed and verified first.")
    offset = float(reference.get("time_offset_sec", 0.0))
    if not math.isfinite(offset):
        raise ValueError("Reference time offset must be finite.")
    samples = reference.get("samples", [])
    if not samples:
        raise ValueError("No independently labelled reference samples.")
    last = -float("inf")
    for sample in samples:
        t = float(sample["t"])
        if not math.isfinite(t) or t <= last:
            raise ValueError("Reference timestamps must be finite, unique and increasing.")
        last = t
        if sample.get("contact") not in (-1, 0, 1):
            raise ValueError("Contact labels are 1=touch, 0=no touch, -1=unknown.")
        q = float(sample.get("quality", 1.0))
        if not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError("Reference quality must be in [0,1].")
        if "reviewed" in sample and not isinstance(sample["reviewed"], bool):
            raise ValueError("Reference reviewed flags must be true/false booleans.")
        if sample["contact"] == 1:
            xy = np.array([sample.get("x"), sample.get("y")], dtype=float)
            if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy > 1):
                raise ValueError("Touch positions must be finite normalized sensor coordinates in [0,1].")
    return reference


def save_reference(path, reference):
    validate_reference(reference)
    path = Path(path)
    dest = path / "reference.json"
    if dest.exists():
        # Preserve previous annotations instead of overwriting the only copy.
        (path / "reference_history").mkdir(exist_ok=True)
        old = json.loads(dest.read_text())
        save_json(path / "reference_history" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"), old)
    save_json(dest, reference)


def import_reference_csv(path, csv_path, *, source="external_tracking", time_offset_sec=0.0):
    """Explicitly imported measurements, never controller-output columns.

    Required CSV columns: t,contact,x,y. Optional: quality,reviewed. t is seconds
    relative to recording start, before the explicit camera-to-sensor offset.
    Calling this function is the user's verification action.
    """
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"t", "contact", "x", "y"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires t,contact,x,y columns; robot-command targets are not accepted.")
        samples = []
        for row in reader:
            contact = int(row["contact"])
            samples.append({"t": float(row["t"]), "contact": contact,
                            "x": float(row["x"]) if contact == 1 else None,
                            "y": float(row["y"]) if contact == 1 else None,
                            "quality": float(row.get("quality") or 1.0),
                            "reviewed": row.get("reviewed", "1").lower() in {"1", "true", "yes"}})
    reference = {"schema": SCHEMA, "source": source, "verified": True,
                 "time_offset_sec": float(time_offset_sec), "samples": samples,
                 "import_file": Path(csv_path).name}
    save_reference(path, reference)
    return reference


def align_reference(reference, times, *, max_gap=0.12, minimum_quality=0.8):
    """Interpolate only between nearby, reviewed, same-contact measurements.

    No extrapolation across occlusion, release, unknown labels, or long gaps.
    Velocity targets are local measured-position slopes (surface widths/s).
    The reference may look ahead; the SENSOR input remains strictly causal.
    """
    validate_reference(reference)
    samples = reference["samples"]
    ts = np.array([s["t"] for s in samples]) + float(reference.get("time_offset_sec", 0))
    states = np.array([s["contact"] if s.get("reviewed") is True and s.get("quality", 1) >= minimum_quality else -1 for s in samples])
    pos = np.array([[s.get("x"), s.get("y")] for s in samples], float)
    result = {"contact": np.full(len(times), -1, np.int64), "xy": np.zeros((len(times), 2), np.float32),
              "velocity": np.zeros((len(times), 2), np.float32), "position_valid": np.zeros(len(times), bool),
              "velocity_valid": np.zeros(len(times), bool)}
    for k, t in enumerate(times):
        right = int(np.searchsorted(ts, t, side="left"))
        exact = right < len(ts) and abs(ts[right] - t) < 1e-7
        if exact and states[right] >= 0:
            result["contact"][k] = states[right]
            if states[right] == 0:
                result["velocity_valid"][k] = True
            else:
                result["xy"][k] = pos[right]
                result["position_valid"][k] = True
        left = right - 1
        if exact and right == 0:
            left, right = 0, 1
        if left < 0 or right >= len(ts):
            continue
        dt = ts[right] - ts[left]
        if dt <= 0 or dt > max_gap or states[left] < 0 or states[left] != states[right]:
            continue
        result["contact"][k] = states[left]
        result["velocity_valid"][k] = True
        if states[left] == 1:
            alpha = (t - ts[left]) / dt
            result["xy"][k] = pos[left] * (1 - alpha) + pos[right] * alpha
            result["velocity"][k] = (pos[right] - pos[left]) / dt
            result["position_valid"][k] = True
    return result


class CausalPreprocessor:
    """One acquisition-to-model implementation shared by offline and live use.

    Regular grid uses zero-order hold from past frames only. Missing data and
    long gaps invalidate windows. All signed changes retain their sign.
    """

    def __init__(self, hz=50.0, max_gap=0.10, clip_percent=200.0):
        if not 1 <= hz <= 250 or not 0 < max_gap <= 1:
            raise ValueError("Invalid resampling rate or gap limit.")
        self.hz, self.max_gap, self.clip_percent = float(hz), float(max_gap), float(clip_percent)
        self.previous = None
        self.origin = None
        self.step = 0

    def push(self, t, raw, baseline, mask):
        raw, baseline, mask = np.asarray(raw), np.asarray(baseline), np.asarray(mask, bool)
        if raw.ndim != 2 or raw.shape != baseline.shape or mask.shape != raw.shape or not np.isfinite(t):
            raise ValueError("Invalid frame geometry or timestamp.")
        if self.previous is not None and t <= self.previous[0]:
            raise ValueError("Frame time must strictly increase.")
        if self.previous is not None and raw.shape != self.previous[1].shape:
            raise ValueError("Sensor shape changed.")
        valid = mask & np.isfinite(raw) & np.isfinite(baseline) & (np.abs(baseline) > 1e-6)
        diff = np.zeros(raw.shape, np.float32)
        np.divide(100.0 * (raw - baseline), baseline, out=diff, where=valid)
        valid &= np.abs(diff) <= self.clip_percent
        diff = np.where(valid, diff, 0).astype(np.float32)
        derivative = np.zeros_like(diff)
        if self.previous is not None:
            dt = t - self.previous[0]
            if dt <= self.max_gap:
                derivative = np.where(valid & self.previous[2], (diff - self.previous[1]) / dt, 0)
        current = (float(t), diff, valid, derivative)
        if self.origin is None:
            self.origin = float(t)
        output = []
        # Avoid unbounded work after a disconnect; a single invalid sample resets
        # downstream history, then resume at the next regular timestamp.
        if self.origin + self.step / self.hz < t - 2 * self.max_gap:
            output.append((t - self.max_gap, np.zeros((4, *raw.shape), np.float32), False))
            self.step = max(self.step, int(math.ceil((t - self.origin) * self.hz - 1e-8)))
        while self.origin + self.step / self.hz <= t + 1e-9:
            grid_t = self.origin + self.step / self.hz
            source = current if grid_t >= t - 1e-9 else self.previous
            if source is not None:
                age = max(0.0, grid_t - source[0])
                frame = np.stack([source[1], source[3], source[2].astype(np.float32),
                                  np.full(raw.shape, age, np.float32)]).astype(np.float32)
                good = age <= self.max_gap and float(source[2].mean()) >= 0.5
                output.append((float(grid_t), frame, good))
            self.step += 1
        self.previous = current
        return output


def prepare_trial(path, *, hz=50.0, max_gap=0.10, label_gap=0.12):
    meta, arrays = load_trial(path)
    if meta.get("supervision", "independent") != "independent":
        raise ValueError("Camera-free estimates require explicit camera-free mode; they are not independent references.")
    reference = json.loads((Path(path) / "reference.json").read_text())
    pp = CausalPreprocessor(hz, max_gap)
    samples = []
    for t, raw, base, mask in zip(arrays["t"], arrays["raw"], arrays["baseline"], arrays["valid_taxels"]):
        samples.extend(pp.push(float(t), raw, base, mask))
    if not samples:
        raise ValueError("No resampled sensor frames.")
    t, x, valid = zip(*samples)
    labels = align_reference(reference, t, max_gap=label_gap)
    digest = hashlib.sha256()
    digest.update(arrays["raw"].tobytes())
    digest.update(np.round(arrays["t"] - arrays["t"][0], 6).tobytes())
    return {"path": str(Path(path).resolve()), "metadata": meta, "t": np.array(t),
            "frames": np.stack(x), "frame_valid": np.array(valid), "labels": labels,
            "data_hash": digest.hexdigest(),
            "reference_hash": hashlib.sha256((Path(path) / "reference.json").read_bytes()).hexdigest()}


def window_indices(trial, seq_len, *, require_labels=True):
    valid = trial["frame_valid"].astype(int)
    sums = np.convolve(valid, np.ones(seq_len, int), mode="valid") if len(valid) >= seq_len else []
    indices = np.flatnonzero(np.asarray(sums) == seq_len) + seq_len - 1
    if require_labels:
        indices = indices[trial["labels"]["contact"][indices] >= 0]
    return indices


def split_trials(trials, *, group_by="session", seed=42):
    if group_by not in {"session", "participant"}:
        raise ValueError("Split groups must be session or participant.")
    groups = {}
    for trial in trials:
        m = trial["metadata"]
        key = m["participant_id"] if group_by == "participant" else (m["participant_id"], m["session_id"])
        groups.setdefault(key, []).append(trial)
    if len(groups) < 3:
        raise ValueError(f"Need at least 3 independent {group_by} groups for train, validation and untouched test. Do not rename copies to manufacture groups.")
    keys = sorted(groups)
    np.random.default_rng(seed).shuffle(keys)
    hold = max(1, int(round(len(keys) * 0.2)))
    split_keys = {"train": keys[2*hold:], "validation": keys[hold:2*hold], "test": keys[:hold]}
    splits = {name: [trial for key in ks for trial in groups[key]] for name, ks in split_keys.items()}
    seen = {}
    for name, items in splits.items():
        for item in items:
            previous = seen.setdefault(item["data_hash"], name)
            if previous != name:
                raise ValueError("Duplicate raw recording appears in different splits; refusing leakage.")
    return splits


def dataset_report(root, seq_len=16, hz=50.0, max_gap=0.10, label_gap=0.12, *, supervision="independent"):
    if supervision not in {"independent", "camera_free"}:
        raise ValueError("Unknown supervision mode.")
    root = Path(root)
    paths = [root] if (root / "manifest.json").exists() else sorted(p.parent for p in root.glob("*/manifest.json"))
    reports, prepared = [], []
    for path in paths:
        item = {"trial": str(path), "errors": [], "warnings": []}
        try:
            meta, arrays = load_trial(path)
            item.update(frames=len(arrays["t"]), duration_sec=float(np.ptp(arrays["t"])),
                        sensor_shape=meta["sensor_shape"], participant=meta.get("participant_id"), session=meta.get("session_id"))
            if meta.get("status") != "complete":
                raise ValueError("Recording is incomplete/failed. Retained for recovery; not automatically used for training.")
            nonfinite = int((~np.isfinite(arrays["raw"])).sum())
            if nonfinite:
                item["warnings"].append(f"{nonfinite} nonfinite raw values; validity masking applies.")
            item["gaps_over_100ms"] = int((np.diff(arrays["t"]) > 0.1).sum())
            item["skipped_buffer_sequences"] = int(np.maximum(np.diff(arrays["sequence"]) - 1, 0).sum())
            prepare = prepare_trial
            if supervision == "camera_free":
                from phd.dependence.touch_tracking_camera_free import prepare_camera_free_trial
                prepare = prepare_camera_free_trial
            trial = prepare(path, hz=hz, max_gap=max_gap, label_gap=label_gap)
            if supervision == "camera_free":
                item["warnings"].append(trial["target_note"])
                item["withheld_position_frames"] = trial["withheld_frames"]
            ends = window_indices(trial, seq_len)
            labels = trial["labels"]
            item.update(labelled_windows=len(ends), contact_windows=int((labels["contact"][ends] == 1).sum()),
                        no_contact_windows=int((labels["contact"][ends] == 0).sum()),
                        position_windows=int(labels["position_valid"][ends].sum()),
                        velocity_windows=int(labels["velocity_valid"][ends].sum()))
            if supervision == "camera_free" and item["contact_windows"]:
                item["position_target_coverage"] = item["position_windows"] / item["contact_windows"]
                if item["position_target_coverage"] < .5:
                    item["warnings"].append("Less than half the declared touch windows have usable position estimates. Inspect sensor quality and collection states before training.")
            if not len(ends):
                raise ValueError("No valid labelled windows; annotate nearby frames, including contact and no-contact.")
            prepared.append(trial)
        except Exception as exc:
            item["errors"].append(str(exc))
        reports.append(item)
    return {"schema": SCHEMA, "root": str(root), "supervision": supervision, "trials": reports,
            "ready_trials": len(prepared), "invalid_trials": len(reports) - len(prepared)}, prepared
