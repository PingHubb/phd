"""Camera-free single-finger tracking and explicitly weak supervision.

No camera, torch, ROS, or robot dependencies. Positions are signal estimates,
NOT independent physical ground truth. Contact intervals are declared by the
operator; neither this estimator nor a neural confidence can certify contact.
"""

from collections import deque
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from phd.dependence.touch_tracking_data import CausalPreprocessor, load_trial


PROTOCOL_SCHEMA = "pinglab.camera_free_protocol.v1"
SUPERVISION = "camera_free_sensor_estimates_v1"
NOISE_SECONDS = 5.0
TRANSITION_MARGIN = .35


class MotionFilter:
    """Causal speed-adaptive smoothing, using actual elapsed time (1 Euro)."""

    def __init__(self):
        self.last_t = None
        self.xy = None
        self.raw_xy = None
        self.derivative = np.zeros(2)

    def update(self, t, xy):
        xy = np.asarray(xy, float)
        if self.last_t is None:
            self.last_t, self.xy, self.raw_xy = t, xy.copy(), xy.copy()
            return xy.copy(), np.zeros(2)
        dt = t - self.last_t
        a = lambda cutoff: 1 / (1 + 1 / (2 * np.pi * cutoff * dt))
        derivative = (xy - self.raw_xy) / dt
        self.derivative += a(1.) * (derivative - self.derivative)
        cutoff = 2. + 4. * np.linalg.norm(self.derivative)
        previous = self.xy.copy()
        self.xy += a(cutoff) * (xy - self.xy)
        velocity = (self.xy - previous) / dt
        if np.linalg.norm(velocity) < .015:
            velocity[:] = 0
        self.last_t, self.raw_xy = t, xy.copy()
        return self.xy.copy(), velocity


def _components(active):
    remaining = set(map(tuple, np.argwhere(active)))
    groups = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        todo, group = [seed], []
        while todo:
            r, c = todo.pop()
            group.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbour = (r+dr, c+dc)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        todo.append(neighbour)
        groups.append(group)
    return groups


class CameraFreeTracker:
    """Per-start robust noise calibration, artifact rejection and causal tracking.

    A fresh explicit hands-away calibration is required after resets/baseline
    changes; it never quietly learns a touching finger as the new idle state.
    Fixed noise statistics avoid adapting away sustained touch. The sign is
    unrestricted, supporting sensors whose response increases or decreases.
    """

    def __init__(self, shape, sensor_id, reorder_mode="", *, noise_seconds=NOISE_SECONDS, max_gap=.1):
        self.shape = tuple(shape)
        self.sensor_id = str(sensor_id)
        self.max_gap = float(max_gap)
        self.noise_seconds = float(noise_seconds)
        self.config = {"model": {"rows": shape[0], "cols": shape[1]},
                       "sensor_ids": [self.sensor_id], "reorder_mode": reorder_mode,
                       "training_reference": SUPERVISION, "robot_commands_enabled": False}
        self.last_t = None
        self.last_sequence = None
        self.baseline = None
        self.mask = None
        self.origin = None
        self.calibration = deque(maxlen=5000)
        self.offset = None
        self.sigma = None
        self.fault = None
        self.noise_frames = 0
        self._clear_motion()

    def _clear_motion(self):
        self.pending_xy = None
        self.pending_since = None
        self.last_position = None
        self.last_position_t = None
        self.filter = MotionFilter()

    def _result(self, state, *, valid=False, active=False, xy=None, velocity=None, quality=0., timestamp=None):
        return {"state": state, "valid": valid, "contact": active, "xy": xy,
                "velocity": [0., 0.] if velocity is None else velocity,
                "signal_quality": float(quality), "timestamp": timestamp,
                "contact_note": "Localized signal is not proof of physical contact.",
                "robot_commands_enabled": False}

    def update(self, timestamp, sequence, raw, baseline, mask, *, sensor_id, **_unused):
        started = time.perf_counter()
        result = self._update(float(timestamp), sequence, raw, baseline, mask, sensor_id)
        result["inference_ms"] = (time.perf_counter() - started)*1000
        return result

    def _update(self, t, sequence, raw, baseline, mask, sensor_id):
        invalid = lambda message: self._result(message, timestamp=t)
        if self.fault:
            return invalid(self.fault)
        raw, baseline, mask = np.asarray(raw), np.asarray(baseline), np.asarray(mask, bool)
        if sensor_id != self.sensor_id or raw.shape != self.shape or baseline.shape != self.shape or mask.shape != self.shape:
            self.fault = "Sensor identity/geometry changed; restart with hand away."
            return invalid(self.fault)
        if not np.isfinite(t):
            return invalid("Invalid timestamp")
        if self.last_t is not None:
            if sequence <= self.last_sequence or t <= self.last_t:
                self.fault = "Sensor sequence/time reset; restart with hand away."
                return invalid(self.fault)
            if not np.array_equal(baseline, self.baseline, equal_nan=True) or not np.array_equal(mask, self.mask):
                self.fault = "Baseline/mask changed; restart with hand away."
                return invalid(self.fault)
            if t-self.last_t > self.max_gap:
                self._clear_motion()
                if self.offset is None:
                    self.calibration.clear()
                    self.origin = t
        self.last_t, self.last_sequence = t, sequence
        self.baseline, self.mask = baseline.copy(), mask.copy()
        if self.origin is None:
            self.origin = t
        usable = mask & np.isfinite(raw) & np.isfinite(baseline) & (np.abs(baseline) > 1e-6)
        if usable.mean() < .5:
            self._clear_motion()
            return invalid("Too few usable taxels")
        diff = np.zeros(self.shape, np.float64)
        np.divide(100*(raw-baseline), baseline, out=diff, where=usable)
        if not np.isfinite(diff[usable]).all() or np.any(np.abs(diff[usable]) > 200):
            self._clear_motion()
            return invalid("Out-of-range signal — withheld")
        if self.offset is None:
            self.calibration.append(np.where(usable, diff, np.nan))
            if t-self.origin < self.noise_seconds or len(self.calibration) < 50:
                return invalid(f"Keep hand AWAY — noise calibration {t-self.origin:.1f}/{self.noise_seconds:.0f} s")
            values = np.stack(self.calibration)
            # Disabled cells are assigned benign constants, not all-NaN statistics.
            available = np.isfinite(values).all(axis=0)
            values[:, ~available] = 0
            self.offset = np.median(values, axis=0)
            self.sigma = np.maximum(1.4826*np.median(np.abs(values-self.offset), axis=0), .01)
            self.noise_mask = available & mask
            half = max(1, len(values)//2)
            drift = np.abs(np.median(values[:half], axis=0)-np.median(values[half:], axis=0))
            self.noise_frames = len(values)
            self.calibration.clear()
            if self.noise_mask.mean() < .5 or np.any(drift[self.noise_mask] > np.maximum(6*self.sigma[self.noise_mask], .2)):
                self.fault = "Baseline changed during hands-away calibration; settle/calibrate and restart."
                return invalid(self.fault)
            return self._result("Noise learned — ready for one finger", valid=True, timestamp=t)
        usable &= self.noise_mask
        amplitude = np.where(usable, np.abs(diff-self.offset), 0)
        z = amplitude/self.sigma
        active = usable & (z > 6) & (amplitude > .08)
        if not active.any():
            self._clear_motion()
            return self._result("No localized signal", valid=True, timestamp=t)
        # A complete row/column or broad response is ambiguous: never invent a
        # centre of several fingers, a line glitch, or global electrical drift.
        row_line = self.shape[1] >= 4 and np.any(active.sum(axis=1) >= .9*self.shape[1])
        col_line = self.shape[0] >= 4 and np.any(active.sum(axis=0) >= .9*self.shape[0])
        if row_line or col_line or active.mean() > .40:
            self._clear_motion()
            return invalid("Broad/row/column response — ambiguous, withheld")
        groups = _components(active)
        weighted = []
        for group in groups:
            rr, cc = np.array(group).T
            weighted.append((float(amplitude[rr, cc].sum()), rr, cc))
        weighted.sort(key=lambda v: v[0], reverse=True)
        if len(weighted) > 1 and weighted[1][0] > .25*weighted[0][0]:
            self._clear_motion()
            return invalid("Multiple responding regions — single-finger estimate withheld")
        mass, rr, cc = weighted[0]
        weights = amplitude[rr, cc]
        # Grid coordinates refer to cell centres, not the outside edges.
        xy = np.array([np.average((cc+.5)/self.shape[1], weights=weights),
                       np.average((rr+.5)/self.shape[0], weights=weights)])
        if self.pending_xy is None or np.linalg.norm(xy-self.pending_xy) > .15:
            self._clear_motion()
            self.pending_xy, self.pending_since = xy, t
            return invalid("Confirming stable localized signal")
        if t-self.pending_since < .06:
            self.pending_xy = xy
            return invalid("Confirming stable localized signal")
        if self.last_position is not None:
            dt = t-self.last_position_t
            if np.linalg.norm(xy-self.last_position) > .08 + 3*dt:
                self._clear_motion()
                return invalid("Abrupt position jump — withheld")
        self.pending_xy = xy
        self.last_position, self.last_position_t = xy, t
        smooth, velocity = self.filter.update(t, xy)
        quality = min(1., float(np.max(z[rr, cc]))/20) * mass/max(sum(w[0] for w in weighted), 1e-9)
        return self._result("Localized finger signal (touch/proximity not certified)", valid=True, active=True,
                            xy=smooth.tolist(), velocity=velocity.tolist(), quality=quality, timestamp=t)

    def update_many(self, frames, *, sensor_id):
        result = self._result("Waiting for sensor data")
        for frame in frames:
            result = self.update(*frame, sensor_id=sensor_id)
        return result


def validate_protocol(protocol):
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("complete") is not True:
        raise ValueError("A completed camera-free collection protocol is required.")
    if protocol.get("operator_confirmed") is not True:
        raise ValueError("Confirm the hands-away start and declared touch/no-touch intervals.")
    events = protocol.get("events", [])
    if not events or events[0] != {"t": 0., "state": "no_touch"}:
        raise ValueError("Camera-free trials must start with a hands-away interval.")
    last = -1.
    for event in events:
        t = float(event["t"])
        if not np.isfinite(t) or t <= last or event["state"] not in {"no_touch", "touch", "unknown"}:
            raise ValueError("Protocol times/states are invalid.")
        last = t
    if len(events) < 2 or events[1]["t"] < NOISE_SECONDS + TRANSITION_MARGIN:
        raise ValueError("Keep the hand away for at least 5.35 seconds before changing state.")
    return protocol


def prepare_camera_free_trial(path, *, hz=50., max_gap=.1, label_gap=.12):
    """Generate pseudo-position targets in memory; never overwrite reference.json.

    The initial hands-away interval estimates this session's noise only. The
    same explicit startup calibration is required in live use. No held-out
    motion data fits thresholds. Human state transitions have exclusion margins.
    """
    path = Path(path)
    meta, arrays = load_trial(path)
    if meta.get("supervision") != SUPERVISION:
        raise ValueError("Not a camera-free trial; independent and estimated labels cannot be silently mixed.")
    protocol = validate_protocol(json.loads((path / "camera_free_protocol.json").read_text()))
    if protocol["events"][-1]["t"] > arrays["t"][-1] + TRANSITION_MARGIN:
        raise ValueError("Protocol interval extends beyond the recorded sensor stream.")
    if np.any(arrays["baseline"] != arrays["baseline"][0]) or np.any(arrays["valid_taxels"] != arrays["valid_taxels"][0]):
        raise ValueError("Calibration/mask changed within the trial.")
    pp = CausalPreprocessor(hz, max_gap)
    samples = []
    for t, raw, base, mask in zip(arrays["t"], arrays["raw"], arrays["baseline"], arrays["valid_taxels"]):
        samples.extend(pp.push(float(t), raw, base, mask))
    times, frames, valid = zip(*samples)
    times, frames, valid = np.array(times), np.stack(frames), np.array(valid)
    tracker = CameraFreeTracker(meta["sensor_shape"], meta["sensor_id"], max_gap=max_gap)
    labels = {"contact": np.full(len(times), -1, np.int64), "xy": np.zeros((len(times), 2), np.float32),
              "velocity": np.zeros((len(times), 2), np.float32), "position_valid": np.zeros(len(times), bool),
              "velocity_valid": np.zeros(len(times), bool)}
    # Process on exactly the same resampled grid used for training. Predictions
    # use this grid as well, so filtering does not depend on USB burst timing.
    events = protocol["events"]
    event_times = np.array([e["t"] for e in events])
    withheld = 0
    for k, (t, frame, good) in enumerate(zip(times, frames, valid)):
        if tracker.offset is None and t >= event_times[1]-TRANSITION_MARGIN:
            raise ValueError("Initial hands-away interval was too short or interrupted; repeat with at least 6 seconds clear.")
        if not good:
            tracker._clear_motion()
            withheld += 1
            continue
        result = tracker.update(t, k+1, 100+frame[0], np.full(frame.shape[-2:], 100.), frame[2] > .5,
                                sensor_id=meta["sensor_id"])
        if tracker.fault:
            raise ValueError(tracker.fault)
        event_index = int(np.searchsorted(event_times, t, side="right"))-1
        if event_index < 0 or np.min(np.abs(event_times-t)) < TRANSITION_MARGIN:
            continue
        if times[-1]-t < TRANSITION_MARGIN:
            continue
        state = events[event_index]["state"]
        if state == "no_touch":
            labels["contact"][k] = 0
            labels["velocity_valid"][k] = True
        elif state == "touch":
            labels["contact"][k] = 1  # operator-declared, never inferred from strength
            if result.get("xy") is not None and result["signal_quality"] >= .6:
                labels["xy"][k] = result["xy"]
                labels["velocity"][k] = result["velocity"]
                labels["position_valid"][k] = labels["velocity_valid"][k] = True
            else:
                withheld += 1
    if tracker.offset is None:
        raise ValueError("Insufficient continuous hands-away frames for noise calibration.")
    digest = hashlib.sha256(arrays["raw"].tobytes())
    digest.update(np.round(arrays["t"]-arrays["t"][0], 6).tobytes())
    return {"path": str(path.resolve()), "metadata": meta, "t": times, "frames": frames,
            "frame_valid": valid, "labels": labels, "data_hash": digest.hexdigest(),
            "reference_hash": hashlib.sha256((path / "camera_free_protocol.json").read_bytes()).hexdigest(),
            "supervision": SUPERVISION, "estimated_position_targets": int(labels["position_valid"].sum()),
            "withheld_frames": withheld,
            "target_note": "Positions/velocities are sensor estimates; contact is operator-declared. Not independent accuracy."}


class CameraFreePreview:
    """Resample identically to offline pseudo-targets; no model is required."""

    def __init__(self, shape, sensor_id, reorder_mode="", hz=50., max_gap=.1):
        self.tracker = CameraFreeTracker(shape, sensor_id, reorder_mode, max_gap=max_gap)
        self.config = self.tracker.config
        self.processor = CausalPreprocessor(hz, max_gap)
        self.sequence = 0
        self.last_source = None
        self.fault = None
        self.last_result = self.tracker._result("Waiting for data")

    def update_many(self, frames, *, sensor_id):
        result = self.last_result
        for t, seq, raw, baseline, mask in frames:
            if self.fault:
                return self.tracker._result(self.fault, timestamp=t)
            if self.last_source is not None:
                previous_t, previous_seq, previous_base, previous_mask = self.last_source
                if t <= previous_t or seq <= previous_seq or not np.array_equal(baseline, previous_base, equal_nan=True) or not np.array_equal(mask, previous_mask):
                    self.fault = "Source reset/calibration changed; restart with hand away."
                    return self.tracker._result(self.fault, timestamp=t)
            self.last_source = (t, seq, np.array(baseline, copy=True), np.array(mask, copy=True))
            for grid_t, frame, good in self.processor.push(t, raw, baseline, mask):
                self.sequence += 1
                if not good:
                    self.tracker._clear_motion()
                    result = self.tracker._result("Signal gap — history cleared", timestamp=t)
                    continue
                result = self.tracker.update(grid_t, self.sequence, 100+frame[0], np.full(frame.shape[-2:], 100.),
                                             frame[2] > .5, sensor_id=sensor_id)
        self.last_result = result
        return result
