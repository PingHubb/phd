"""Human-declared surface-goal trials, independent of CIC/centroid labels.

The reusable touch recorder supplies raw transport only. intention.json is a
separate, mandatory protocol: labels are NOT passed to the feature extractor.
No camera, ROS, robot controller, or torch dependency is needed here.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from phd.dependence.touch_tracking_data import (
    CausalPreprocessor, TrialRecorder, load_trial, save_json, window_indices,
)

SCHEMA = "pinglab.surface_intention.v1"
SUPERVISION = "human_declared_surface_goal_v1"
CLASSES = ["No current goal", "A", "B", "C"]
# Native array coordinates: x=column/(cols-1), y=row/(rows-1).
POSITIONS = ((0.15, 0.2), (0.85, 0.2), (0.5, 0.85))
LAYOUTS = tuple(itertools.permutations(POSITIONS))


def validate_goals(goals):
    goals = np.asarray(goals, np.float32)
    if goals.shape != (3, 2) or not np.isfinite(goals).all() or np.any((goals < 0) | (goals > 1)):
        raise ValueError("Supply all three target coordinates in the normalized native sensor layout.")
    if min(np.linalg.norm(goals[i] - goals[j]) for i in range(3) for j in range(i)) < .1:
        raise ValueError("Targets must be distinct (at least 0.1 normalized distance apart).")
    return goals.copy()


class IntentionRecorder:
    """Capture independent operator declarations and immutable raw frames.

    declare() is called by the operator/UI, not a sensor threshold. Starting a
    recording is initially unlabelled, so forgetting a declaration is visible.
    """
    def __init__(self, root, metadata, goals, **writer_options):
        goals = validate_goals(goals)
        if not metadata.get("orientation_confirmed"):
            raise ValueError("Confirm native sensor orientation and physical target locations first.")
        self.raw = TrialRecorder(root, {**metadata, "supervision": SUPERVISION}, **writer_options)
        self.path, self.started = self.raw.path, self.raw.started
        self.protocol = {
            "schema": SCHEMA, "status": "recording", "task": "surface_target_selection",
            "coordinate_system": "x_column_y_row_normalized_taxel_centres",
            "goals": goals.tolist(), "classes": CLASSES,
            "label_source": "human_declaration", "robot_commands_enabled": False,
            "events": [{"t": 0.0, "goal": -1}],
        }
        self._closed = False
        try:
            self.raw.manifest["labels"] = "human_goal_declarations_in_intention_json"
            save_json(self.path / "manifest.json", self.raw.manifest)
            self._save()
        except Exception:
            self.raw.close("Could not create intention protocol.")
            raise

    @property
    def error(self):
        return self.raw.error

    @property
    def sensor_count(self):
        return self.raw.sensor_count

    def _save(self):
        save_json(self.path / "intention.json", self.protocol)

    def declare(self, goal, timestamp=None):
        if self._closed:
            raise ValueError("This recording has already closed.")
        if type(goal) is not int or goal not in (-1, 0, 1, 2, 3):
            raise ValueError("Choose unknown, no goal, A, B, or C.")
        t = (time.monotonic() if timestamp is None else float(timestamp)) - self.started
        if not np.isfinite(t) or t <= self.protocol["events"][-1]["t"]:
            raise ValueError("Declarations must have increasing host-monotonic timestamps.")
        # Repeated button presses do not manufacture a new intention onset.
        if self.protocol["events"][-1]["goal"] == goal:
            return
        event = {"t": t, "goal": goal}
        self.protocol["events"].append(event)
        try:
            self._save()
        except Exception as exc:
            self.raw.error = f"Could not save intention declaration: {exc}"
            raise

    def sensor_frame(self, *args, **kwargs):
        return self.raw.sensor_frame(*args, **kwargs)

    def close(self, reason=None):
        if self._closed:
            return self.path
        self._closed = True
        self.raw.close(reason)
        self.protocol["status"] = "failed" if self.error else "complete"
        self.protocol["error"] = self.error
        self._save()
        return self.path


def align_declarations(events, times, margin=.25):
    """Unknown around declaration boundaries; no interpolation of intention.

    The symmetric exclusion is an annotation-quality mask for training/eval,
    not future information in model features. Live inference uses no events.
    """
    times = np.asarray(times, float)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Boundary exclusion must be finite and nonnegative.")
    if not events:
        raise ValueError("Missing human declarations.")
    ts = np.asarray([e["t"] for e in events], float)
    goals = [e["goal"] for e in events]
    if (not np.isfinite(ts).all() or ts[0] != 0 or np.any(np.diff(ts) <= 0)
            or any(type(g) is not int or g not in (-1, 0, 1, 2, 3) for g in goals)):
        raise ValueError("Invalid declaration timeline.")
    labels = np.asarray(goals, np.int64)[np.maximum(np.searchsorted(ts, times, side="right") - 1, 0)]
    labels[times < 0] = -1
    for boundary in ts[1:]:
        labels[np.abs(times - boundary) < margin] = -1
    return labels


def prepare_intention_trial(path, *, hz=20., max_gap=.15, boundary_margin=.25):
    meta, arrays = load_trial(path)
    protocol = json.loads((Path(path) / "intention.json").read_text())
    if meta.get("supervision") != SUPERVISION or protocol.get("schema") != SCHEMA:
        raise ValueError("Not a human-declared intention trial. Old imitation/tracking labels are not accepted.")
    if meta.get("status") != "complete" or protocol.get("status") != "complete":
        raise ValueError("Incomplete/failed trial retained for recovery; excluded from training.")
    if (protocol.get("label_source") != "human_declaration" or protocol.get("classes") != CLASSES
            or not meta.get("orientation_confirmed")):
        raise ValueError("Missing label provenance or orientation confirmation.")
    goals = validate_goals(protocol["goals"])
    if not np.array_equal(arrays["baseline"], np.broadcast_to(arrays["baseline"][0], arrays["baseline"].shape), equal_nan=True):
        raise ValueError("Calibration changed within a trial.")
    if not np.array_equal(arrays["valid_taxels"], np.broadcast_to(arrays["valid_taxels"][0], arrays["valid_taxels"].shape)):
        raise ValueError("Disabled-taxel mask changed within a trial.")
    pp = CausalPreprocessor(hz=hz, max_gap=max_gap)
    samples = []
    for t, raw, base, mask in zip(arrays["t"], arrays["raw"], arrays["baseline"], arrays["valid_taxels"]):
        samples.extend(pp.push(float(t), raw, base, mask))
    if not samples:
        raise ValueError("No usable sensor samples.")
    t, x, valid = zip(*samples)
    labels = align_declarations(protocol["events"], t, boundary_margin)
    # Raw-only fingerprint deliberately catches copies with edited labels,
    # timestamps or participant/session names as well.
    digest = hashlib.sha256(arrays["raw"].tobytes()).hexdigest()
    return {"path": str(Path(path).resolve()), "metadata": meta, "protocol": protocol,
            "goals": goals, "t": np.asarray(t), "frames": np.stack(x),
            "frame_valid": np.asarray(valid), "labels": labels, "data_hash": digest,
            "protocol_hash": hashlib.sha256((Path(path) / "intention.json").read_bytes()).hexdigest(),
            "gaps": int(np.sum(np.diff(arrays["t"]) > max_gap))}


def intention_indices(trial, seq_len):
    if type(seq_len) is not int or seq_len < 2:
        raise ValueError("Sequence length must be at least two frames.")
    ends = window_indices(trial, seq_len, require_labels=False)
    # A change of intention can be inside the input history. Only the current
    # endpoint label is the target; cutting all such windows would hide switches.
    return ends[trial["labels"][ends] >= 0]


def dataset_report(root, *, seq_len=20, **prepare_options):
    root = Path(root)
    paths = [root] if (root / "manifest.json").exists() else sorted(p.parent for p in root.glob("*/manifest.json"))
    reports, trials = [], []
    for path in paths:
        row = {"trial": str(path), "errors": []}
        try:
            trial = prepare_intention_trial(path, **prepare_options)
            ends = intention_indices(trial, seq_len)
            counts = np.bincount(trial["labels"][ends], minlength=4)
            row.update(windows=len(ends), class_windows=counts.tolist(), gaps=trial["gaps"],
                       participant=trial["metadata"]["participant_id"], session=trial["metadata"]["session_id"])
            if not len(ends):
                raise ValueError("No valid labelled windows. Declare a goal/no-goal and record longer.")
            trials.append(trial)
        except Exception as exc:
            row["errors"].append(str(exc))
        reports.append(row)
    return {"schema": SCHEMA, "classes": CLASSES, "ready_trials": len(trials),
            "invalid_trials": len(paths)-len(trials), "trials": reports}, trials
