"""Small causal finger trackers. This module has no robot API or command path."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from phd.dependence.touch_tracking_data import CausalPreprocessor, LAYOUT, SCHEMA


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation, padding=self.pad)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv(x)[..., :-self.pad]
        return x + self.dropout(self.activation(y))


class TouchTracker(nn.Module):
    def __init__(self, rows, cols, architecture="gru", hidden=64, dropout=0.1, context_frames=16):
        super().__init__()
        if architecture not in {"gru", "tcn"}:
            raise ValueError("Architecture must be gru or tcn.")
        self.architecture = architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.GELU(),
            nn.Flatten(), nn.Linear(24 * rows * cols, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        if architecture == "gru":
            self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        else:
            dilations, span, dilation = [], 1, 1
            while span < context_frames:
                dilations.append(dilation)
                span += 2*dilation
                dilation *= 2
            self.temporal = nn.Sequential(*(CausalBlock(hidden, d, dropout) for d in dilations))
        self.norm = nn.LayerNorm(hidden)
        self.position = nn.Linear(hidden, 2)
        self.velocity = nn.Linear(hidden, 2)
        self.contact = nn.Linear(hidden, 1)

    def forward(self, x):
        n, t, c, r, k = x.shape
        encoded = self.encoder(x.reshape(n*t, c, r, k)).reshape(n, t, -1)
        if self.architecture == "gru":
            encoded, _ = self.temporal(encoded)
        else:
            encoded = self.temporal(encoded.transpose(1, 2)).transpose(1, 2)
        final = self.norm(encoded[:, -1])
        return {"xy": torch.sigmoid(self.position(final)), "velocity": self.velocity(final),
                "contact_logits": self.contact(final).squeeze(-1)}


def normalize_frames(frames, normalization):
    mean = np.asarray(normalization["mean"], np.float32)[..., None, None]
    std = np.asarray(normalization["std"], np.float32)[..., None, None]
    result = (np.asarray(frames, np.float32) - mean) / std
    # Missing/disabled cells must not acquire a fictitious baseline signal.
    result[..., :2, :, :] *= np.asarray(frames)[..., 2:3, :, :]
    return result.astype(np.float32)


def load_tracker(path, device="cpu"):
    # New checkpoints contain tensors and plain metadata only. Never fall back
    # to unrestricted pickle loading for legacy/user-selected checkpoints.
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint.get("schema") != SCHEMA or checkpoint.get("layout") != LAYOUT:
        raise ValueError("Choose a Touch Tracking checkpoint, not an old robot policy.")
    config = checkpoint["config"]
    model = TouchTracker(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


class TrackingPreview:
    """Stateful sensor-only inference, deliberately incapable of robot motion.

    Call update serially on one worker. Stale, invalid, reset or geometry-changed
    streams clear temporal state. Probabilities are not certified uncertainty.
    """

    def __init__(self, path):
        # Small tactile tensors run poorly with a large server CPU thread pool.
        # This is only applied when the user explicitly starts preview.
        torch.set_num_threads(min(torch.get_num_threads(), 2))
        self.model, self.config = load_tracker(path)
        self.shape = (self.config["model"]["rows"], self.config["model"]["cols"])
        self.seq_len = int(self.config["seq_len"])
        self.reset()
        self.guard = None
        self.guard_result = None
        if self.config.get("training_reference") == "camera_free_sensor_estimates_v1":
            # Separate from reset(): calibration changes must fault the guard,
            # never silently re-learn an already touching finger as idle.
            self._guard_type = True
        else:
            self._guard_type = False

    def reset(self):
        self.processor = CausalPreprocessor(self.config["hz"], self.config["max_gap"])
        self.buffer = deque(maxlen=self.seq_len)
        self.last_sequence = None
        self.last_time = None
        self.last_baseline = None
        self.last_mask = None

    def update(self, timestamp, sequence, raw, baseline, mask, *, sensor_id, infer=True):
        if tuple(np.shape(raw)) != self.shape:
            self.reset()
            return {"state": "Sensor geometry does not match model", "valid": False}
        if sensor_id not in self.config["sensor_ids"]:
            self.reset()
            return {"state": "Sensor ID was not validated for this model", "valid": False}
        if self.last_sequence is not None and sequence == self.last_sequence:
            return {"state": "Waiting for fresh data", "valid": False}
        if self._guard_type:
            if self.guard is None:
                from phd.dependence.touch_tracking_camera_free import CameraFreePreview
                self.guard = CameraFreePreview(self.shape, sensor_id, self.config.get("reorder_mode", ""),
                                               self.config["hz"], self.config["max_gap"])
            self.guard_result = self.guard.update_many([(timestamp, sequence, raw, baseline, mask)], sensor_id=sensor_id)
        if self.last_time is not None and (
            timestamp <= self.last_time or timestamp - self.last_time > self.config["max_gap"]
            or sequence < self.last_sequence
            or not np.array_equal(baseline, self.last_baseline, equal_nan=True)
            or not np.array_equal(mask, self.last_mask)
        ):
            self.reset()
        self.last_sequence, self.last_time = sequence, timestamp
        self.last_baseline, self.last_mask = np.array(baseline, copy=True), np.array(mask, copy=True)
        for t, frame, valid in self.processor.push(timestamp, raw, baseline, mask):
            if not valid:
                self.buffer.clear()
            else:
                self.buffer.append(frame)
        if len(self.buffer) < self.seq_len:
            return {"state": f"Warming up {len(self.buffer)}/{self.seq_len}", "valid": False}
        if not infer:
            return {"state": "Ready", "valid": True, "timestamp": timestamp}
        return self.predict(timestamp)

    def update_many(self, frames, *, sensor_id):
        """Ingest EVERY fresh buffer frame, but infer once for the newest window."""
        result = {"state": "Waiting for data", "valid": False}
        for frame in frames:
            result = self.update(*frame, sensor_id=sensor_id, infer=False)
        if result.get("valid"):
            return self.predict(result["timestamp"])
        return result

    def predict(self, timestamp):
        if self._guard_type and (not self.guard_result or not self.guard_result.get("valid") or self.guard_result.get("xy") is None):
            return self.guard_result or {"state": "Waiting for hands-away calibration", "valid": False}
        x = normalize_frames(np.stack(self.buffer), self.config["normalization"])
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model(torch.from_numpy(x[None]))
        probability = float(torch.sigmoid(out["contact_logits"])[0])
        contact = probability >= self.config["contact_threshold"]
        xy = out["xy"][0].numpy()
        velocity = out["velocity"][0].numpy() * self.config["velocity_scale"]
        if not np.isfinite(xy).all() or not np.isfinite(velocity).all() or not np.isfinite(probability):
            return {"state": "Invalid model output — withheld", "valid": False, "timestamp": timestamp}
        if self._guard_type:
            disagreement = np.linalg.norm(xy-np.asarray(self.guard_result["xy"]))
            if disagreement > max(.08, 1.5/min(self.shape)) or np.linalg.norm(velocity) > 3:
                return {"state": "Model / signal disagree — withheld; compare the direct tracker", "valid": False, "timestamp": timestamp}
        return {"state": ("Estimated touch (camera-free)" if self._guard_type else "Touch") if contact else "No confident touch", "valid": True,
                "contact": contact, "contact_probability": probability,
                "xy": xy.tolist() if contact else None,
                "velocity": velocity.tolist() if contact else [0., 0.],
                "inference_ms": (time.perf_counter() - start) * 1000,
                "timestamp": timestamp, "robot_commands_enabled": False}
