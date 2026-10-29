"""Folder-labelled gestures, inferred from the available touch prefix/window.

All public arrays are native (time, row, column) percentage differences. The
recorder's historic transpose is undone by robot_dog_gesture_model.load_trial.
The CNN-GRU/TCN sees only frames received so far, never the remainder of a trial.
Velocity is an explicit control mapping, not a learned human-speed estimate.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from phd.dependence.robot_dog_gesture_model import GestureClassifier, PREPROCESS, prepare_trial


FORMAT = "pinglab_touch_gesture_policy_v1"
CLASS_NAMES = ("up", "down", "left", "right", "push", "pull")
SWIPE_CLASS_NAMES = ("swipe_up", "swipe_down", "swipe_left", "swipe_right")
SIX_GESTURE_CLASS_NAMES = (*SWIPE_CLASS_NAMES, "two_finger_push", "two_finger_pull")
SUPPORTED_GESTURE_NAMES = frozenset((*CLASS_NAMES, *SIX_GESTURE_CLASS_NAMES))
PREPROCESS_CONFIG = {
    **PREPROCESS, "minimum_touch_frames": 5, "minimum_duration_sec": 0.08,
    "prediction_unit": "available_touch_prefix", "maximum_duration_sec": 2.0,
}
CONTINUOUS_PREPROCESS_CONFIG = {
    **PREPROCESS_CONFIG,
    "prediction_unit": "recent_touch_window",
    "window_duration_sec": 0.20,
    "maximum_duration_sec": 30.0,
    "motion_window_sec": 0.10,
    "motion_floor_percent": 0.25,
    "motion_noise_multiplier": 3.0,
}


def recent_window_start(times, config):
    """One shared, past-only crop for training and live inference."""
    if config.get("prediction_unit") != "recent_touch_window":
        return 0
    return int(np.searchsorted(times, times[-1] - config["window_duration_sec"], side="left"))
# Values are already in the final command convention used by Touch Policy.
# Preserve the existing direct-finger control's selected-frame axis mapping.
VELOCITY_DIRECTIONS = {
    "up": [0, 0, -1, 0, 0, 0], "down": [0, 0, 1, 0, 0, 0],
    "left": [-1, 0, 0, 0, 0, 0], "right": [1, 0, 0, 0, 0, 0],
    "push": [0, -1, 0, 0, 0, 0], "pull": [0, 1, 0, 0, 0, 0],
}
SWIPE_VELOCITY_DIRECTIONS = {
    name: VELOCITY_DIRECTIONS[name.removeprefix("swipe_")].copy()
    for name in SWIPE_CLASS_NAMES
}
SIX_GESTURE_VELOCITY_DIRECTIONS = {
    **SWIPE_VELOCITY_DIRECTIONS,
    "two_finger_push": VELOCITY_DIRECTIONS["push"].copy(),
    "two_finger_pull": VELOCITY_DIRECTIONS["pull"].copy(),
}


def gesture_control_mode(gesture):
    """Folder labels stay intact; the executor retains its existing mode names."""
    if gesture in ("push", "two_finger_push"):
        return "push"
    if gesture in ("pull", "two_finger_pull"):
        return "pull"
    return "move" if gesture in SUPPORTED_GESTURE_NAMES else "stop"


def policy_directions(class_names):
    """Explicit supported label sets; a checkpoint cannot introduce new axes."""
    if tuple(class_names) == CLASS_NAMES:
        return {name: value.copy() for name, value in VELOCITY_DIRECTIONS.items()}
    if tuple(class_names) == SWIPE_CLASS_NAMES:
        return {name: value.copy() for name, value in SWIPE_VELOCITY_DIRECTIONS.items()}
    if tuple(class_names) == SIX_GESTURE_CLASS_NAMES:
        return {name: value.copy() for name, value in SIX_GESTURE_VELOCITY_DIRECTIONS.items()}
    raise ValueError("Unsupported touch-policy gesture labels or ordering")


def prepare_prefix(raw, averaged, times, config=None):
    config = PREPROCESS_CONFIG if config is None else config
    raw, averaged, times = np.asarray(raw), np.asarray(averaged), np.asarray(times, dtype=np.float64)
    if averaged.ndim != 3 or times.shape != (len(averaged),):
        raise ValueError("Expected a tactile sequence and matching timestamps")
    if not len(times):
        raise ValueError("Collecting more touch frames")
    start = recent_window_start(times, config)
    raw, averaged, times = raw[start:], averaged[start:], times[start:]
    contact = np.flatnonzero(np.any(averaged < config["touch_threshold_percent"], axis=(1, 2)))
    if (len(contact) < config["minimum_touch_frames"]
            or times[contact[-1]] - times[contact[0]] < config["minimum_duration_sec"]):
        raise ValueError("Collecting more touch frames")
    if times[-1] - times[contact[0]] > config["maximum_duration_sec"]:
        raise ValueError("Touch exceeded the trained prefix time limit; release and retry")
    return prepare_trial(raw, averaged, times, config)


# 1× on the Touch Policy speed control. The scale is not capped.
TOUCH_POLICY_UNIT_SPEED_MPS = 0.05


def gesture_velocity(gesture, speed_mps=TOUCH_POLICY_UNIT_SPEED_MPS, scale=1.0):
    values = np.asarray([speed_mps, scale], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Velocity settings must be finite and nonnegative")
    speed = float(speed_mps) * float(scale)
    directions = {**VELOCITY_DIRECTIONS, **SIX_GESTURE_VELOCITY_DIRECTIONS}
    return np.asarray(directions.get(gesture, [0] * 6), dtype=np.float32) * speed


class TouchGesturePolicy(nn.Module):
    """Safe-loadable bundle with shared normalization and calibrated confidence."""

    def __init__(self, bundle):
        super().__init__()
        if bundle.get("format") != FORMAT:
            raise ValueError("Not a touch gesture policy checkpoint")
        self.class_names = tuple(bundle.get("class_names", ()))
        directions = policy_directions(self.class_names)
        control = bundle["control"]
        if (tuple(bundle["sensor_shape"]) != (10, 10)
                or control["directions"] != directions
                or not 0 < float(control["base_speed_mps"])
                or not 0.5 <= float(control["confidence_threshold"]) <= 1
                or not 0 <= float(control["margin_threshold"]) <= 1):
            raise ValueError("Invalid touch-policy sensor or control configuration")
        self.bundle = bundle
        preprocess = bundle["preprocess"]
        if preprocess.get("prediction_unit") == "recent_touch_window":
            if not (0.08 <= float(preprocess["window_duration_sec"]) <= 0.5
                    and 0.08 <= float(preprocess["motion_window_sec"]) <= 0.2
                    and 0 < float(preprocess["maximum_duration_sec"]) <= 30
                    and float(preprocess["motion_floor_percent"]) >= 0.25
                    and float(preprocess["motion_noise_multiplier"]) >= 3):
                raise ValueError("Invalid continuous-touch safety configuration")
        self.models = nn.ModuleList()
        for member in bundle["members"]:
            config = member["config"]
            if config["architecture"] not in ("cnn_gru", "cnn_tcn") or config["classes"] != len(self.class_names):
                raise ValueError("Touch policy requires a CNN-GRU/TCN matching its gesture labels")
            model = GestureClassifier(**config)
            model.load_state_dict(member["state_dict"], strict=True)
            self.models.append(model)
        if not self.models:
            raise ValueError("Checkpoint contains no models")
        self.register_buffer("mean", torch.tensor(bundle["normalization"]["mean"]).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(bundle["normalization"]["std"]).view(1, 1, 3, 1, 1))
        self.temperature = float(bundle["temperature"])
        if (not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all()
                or torch.any(self.std <= 0) or not np.isfinite(self.temperature) or self.temperature <= 0):
            raise ValueError("Invalid checkpoint normalization or temperature")

    def forward(self, x):
        normalized = (x - self.mean) / self.std
        probabilities = torch.stack([model(normalized).softmax(-1) for model in self.models]).mean(0)
        return (probabilities.clamp_min(1e-8).log() / self.temperature).softmax(-1)


class TouchPrefixBuffer:
    """Bounded capture with immediate motion release and debounced re-arming."""

    def __init__(self, shape=(10, 10), config=None):
        self.shape = tuple(shape)
        self.config = dict(PREPROCESS_CONFIG if config is None else config)
        self.continuous = self.config.get("prediction_unit") == "recent_touch_window"
        self.reset()

    def reset(self):
        self.state = "release"
        self.frames = []
        self.last_sequence = None
        self.last_time = None
        self.release_start = None
        self.release_count = 0
        self.touching = False
        self.episode = 0
        self.contact_start = None
        self.noise_samples = []
        self.previous_idle = None
        self.motion_threshold = float(self.config.get("motion_floor_percent", 0.25))
        self.motion_score = 0.0

    @staticmethod
    def _change_score(a, b):
        # Strongest 5% of taxels: reject diffuse small fluctuations without
        # diluting a small fingertip by averaging across all 100 taxels.
        changes = np.sort(np.abs(np.asarray(a) - np.asarray(b)).ravel())
        return float(np.mean(changes[-max(1, int(np.ceil(len(changes) * 0.05))):]))

    def motion_present(self):
        """A signal-change safety gate, NOT a learned stationary/idle class."""
        if not self.continuous:
            return True
        if len(self.frames) < 5:
            return False
        cutoff = self.frames[-1][2] - self.config["motion_window_sec"]
        recent = [frame for frame in self.frames if frame[2] >= cutoff]
        if len(recent) < 4 or recent[-1][2] - recent[0][2] < 0.06:
            return False
        start = np.mean([f[1] for f in recent[:2]], axis=0)
        end = np.mean([f[1] for f in recent[-2:]], axis=0)
        self.motion_score = self._change_score(start, end)
        return self.motion_score >= self.motion_threshold

    def feed(self, sequence, timestamp, raw, averaged):
        raw, averaged = np.asarray(raw, dtype=np.float32), np.asarray(averaged, dtype=np.float32)
        if (raw.shape != self.shape or averaged.shape != self.shape
                or not np.isfinite(raw).all() or not np.isfinite(averaged).all()
                or not np.isfinite(timestamp)):
            raise ValueError("Invalid sensor frame")
        if sequence == self.last_sequence:
            return False
        if self.last_sequence is not None and sequence < self.last_sequence:
            raise ValueError("Sensor frame counter reset")
        if self.last_time is not None and not 0 < timestamp - self.last_time <= 0.2:
            raise ValueError("Sensor stream interrupted")
        self.last_sequence, self.last_time = sequence, timestamp
        self.touching = bool(np.any(averaged < self.config["touch_threshold_percent"]))
        if self.touching:
            self.release_start, self.release_count = None, 0
            self.previous_idle = None
        else:
            if self.continuous and self.state != "recording":
                if self.previous_idle is not None:
                    self.noise_samples.append(self._change_score(averaged, self.previous_idle))
                    self.noise_samples = self.noise_samples[-50:]
                self.previous_idle = averaged.copy()
            self.release_count += 1
            if self.release_start is None:
                self.release_start = timestamp
            arm_time = 0.20 if self.continuous else 0.12
            if self.release_count >= (8 if self.continuous else 3) and timestamp - self.release_start >= arm_time:
                self.state, self.frames = "ready", []
                self.contact_start = None
                if self.continuous and self.noise_samples:
                    self.motion_threshold = max(self.config["motion_floor_percent"],
                        float(np.percentile(self.noise_samples, 90)) * self.config["motion_noise_multiplier"])
        if self.state == "release":
            return False
        if self.state == "ready":
            if not self.touching:
                return False
            self.state = "recording"
            self.contact_start = timestamp
            self.episode += 1
        self.frames.append((raw.copy(), averaged.copy(), float(timestamp)))
        if self.continuous:
            cutoff = timestamp - self.config["window_duration_sec"]
            self.frames = [frame for frame in self.frames if frame[2] >= cutoff]
        if len(self.frames) > 256 or timestamp - self.contact_start > self.config["maximum_duration_sec"]:
            self.state, self.frames, self.touching = "release", [], False
            return False
        return self.touching

    def snapshot(self):
        if self.state != "recording" or not self.touching or not self.frames:
            return None
        raw, averaged, times = zip(*self.frames)
        times = np.asarray(times, dtype=np.float64)
        try:
            return prepare_prefix(np.stack(raw), np.stack(averaged), times - times[0], self.config)
        except ValueError as exc:
            if str(exc) == "Collecting more touch frames":
                return None
            raise


class TouchGestureFeedback:
    """Display-only touch results. Never runs inference or commands hardware."""

    def __init__(self):
        self.completed = None
        self.abort()

    def begin(self, episode, continuous):
        self.episode = int(episode)
        self.continuous = bool(continuous)
        self.latest = None

    def observe(self, prediction, confirmed):
        # Stop/arming/paused messages have mode_conf=1.0, not classifier
        # confidence. They must never turn into a "100% Stop" gesture result.
        if self.episode is None or prediction.get("gesture") not in SUPPORTED_GESTURE_NAMES | {"uncertain"}:
            return
        self.latest = {
            "gesture": prediction["gesture"] if confirmed else "uncertain",
            "accepted": bool(confirmed),
            "confidence": float(prediction["mode_conf"]),
            "margin": float(prediction.get("gesture_margin", 0.0)),
            "top_prediction": prediction.get("top_gesture", prediction["gesture"]),
            "probabilities": dict(prediction.get("gesture_probabilities", {})),
        }

    def finish(self):
        if self.episode is None:
            return
        self.completed = {
            **(self.latest or {"gesture": "not_recognised", "accepted": False,
                              "confidence": None, "margin": None, "probabilities": {}}),
            "episode": self.episode, "continuous": self.continuous,
        }
        self.abort()

    def abort(self):
        # A user stop, timeout or fault is not a completed touch. Preserve the
        # previous completed result, but discard this unfinished candidate.
        self.episode = None
        self.latest = None
