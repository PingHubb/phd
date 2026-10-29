"""Whole-touch-trial gesture classification. No robot or camera dependencies.

NPZ recordings use the historic (time, column, row) layout. Public array inputs
use (time, row, column); this conversion happens exactly once in load_trial.
Predictions are intended for completed touch/release episodes, not single frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


CLASS_NAMES = (
    "anticlockwise", "clockwise", "diagonal_bottomleft_to_topright",
    "diagonal_bottomright_to_topleft", "diagonal_topleft_to_bottomright",
    "diagonal_topright_to_bottomleft", "swipe_down", "swipe_left",
    "swipe_right", "swipe_up",
)
ARCHITECTURES = ("cnn_gru", "cnn_tcn", "cnn_transformer")
PREPROCESS = {
    "version": 1, "sequence_length": 32, "signal_scale_percent": 3.0,
    "touch_threshold_percent": -3.0, "minimum_touch_frames": 2,
    "subtract_spatial_median": True,
    "channels": ["diffPerData", "diffPerDataAve", "averaged_temporal_delta"],
    "input_layout": "time_row_column", "prediction_unit": "completed_touch_trial",
}


def load_trial(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        raw = np.asarray(data["diffPerData"], dtype=np.float32)
        averaged = np.asarray(data["diffPerDataAve"], dtype=np.float32)
        times = np.asarray(data["elapsed_sec"], dtype=np.float64)
    if metadata.get("teacher_layout") != "column_major_matrix_view":
        raise ValueError(f"Unsupported or missing sensor layout: {path}")
    raw, averaged = raw.transpose(0, 2, 1).copy(), averaged.transpose(0, 2, 1).copy()
    expected = (int(metadata["sensor_rows"]), int(metadata["sensor_cols"]))
    if raw.shape[1:] != expected:
        raise ValueError(f"Sensor dimensions disagree with metadata: {path}")
    return raw, averaged, times, metadata


def prepare_trial(raw, averaged, times, config=None):
    """Remove idle tails; resample ordered contact motion, never reverse/rotate."""
    config = dict(PREPROCESS if config is None else config)
    raw = np.asarray(raw, dtype=np.float32)
    averaged = np.asarray(averaged, dtype=np.float32)
    times = np.asarray(times, dtype=np.float64)
    if raw.ndim != 3 or averaged.shape != raw.shape or times.shape != (len(raw),):
        raise ValueError("Expected matching (time, rows, columns) arrays and timestamps")
    if len(raw) < 2 or not all(np.isfinite(a).all() for a in (raw, averaged, times)):
        raise ValueError("Trial is too short or contains non-finite values")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Trial timestamps must increase strictly")
    touching = np.flatnonzero(np.any(averaged < config["touch_threshold_percent"], axis=(1, 2)))
    if len(touching) < config["minimum_touch_frames"]:
        raise ValueError("No valid touch episode (insufficient contact frames)")
    start, end = int(touching[0]), min(len(raw), int(touching[-1]) + 2)
    values = np.stack([raw[start:end], averaged[start:end]], axis=1)
    if config["subtract_spatial_median"]:
        values = values - np.median(values, axis=(2, 3), keepdims=True)
    values = np.arcsinh(values / float(config["signal_scale_percent"]))
    source_times = times[start:end]
    target_times = np.linspace(source_times[0], source_times[-1], config["sequence_length"])
    flat = values.reshape(len(values), -1)
    resampled = np.stack([
        np.interp(target_times, source_times, flat[:, i]) for i in range(flat.shape[1])
    ], axis=1).reshape(config["sequence_length"], 2, *raw.shape[1:])
    delta = np.zeros_like(resampled[:, 1:2])
    delta[1:] = np.diff(resampled[:, 1:2], axis=0)
    return np.concatenate([resampled, delta], axis=1).astype(np.float32)


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x):
        return self.norm(x + self.net(x))


class GestureClassifier(nn.Module):
    def __init__(self, architecture="cnn_gru", classes=10, width=96, dropout=0.2, sequence_length=32):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.architecture = architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(32 * 4 * 4, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout),
        )
        if architecture == "cnn_gru":
            self.temporal = nn.GRU(width, width // 2, batch_first=True, bidirectional=True)
        elif architecture == "cnn_tcn":
            self.temporal = nn.Sequential(*[TemporalBlock(width, d, dropout) for d in (1, 2, 4)])
        else:
            self.position = nn.Parameter(torch.zeros(1, sequence_length, width))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                width, 4, width * 2, dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.attention = nn.Linear(width, 1)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, classes))

    def forward(self, x):
        batch, steps, channels, rows, cols = x.shape
        encoded = self.encoder(x.reshape(batch * steps, channels, rows, cols)).reshape(batch, steps, -1)
        if self.architecture == "cnn_gru":
            sequence, _ = self.temporal(encoded)
        elif self.architecture == "cnn_tcn":
            sequence = self.temporal(encoded.transpose(1, 2)).transpose(1, 2)
        else:
            sequence = self.temporal(encoded + self.position[:, :steps])
        attention = self.attention(sequence).softmax(dim=1)
        return self.head((sequence * attention).sum(dim=1))


class GesturePredictor:
    """Load our own local checkpoint and classify a completed sensor-only trial."""

    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        self.bundle = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if self.bundle.get("format") != "pinglab_robot_dog_gesture_v1":
            raise ValueError("Not a robot-dog gesture classification checkpoint")
        self.class_names = self.bundle["class_names"]
        self.models = []
        for member in self.bundle["members"]:
            model = GestureClassifier(**member["config"])
            model.load_state_dict(member["state_dict"], strict=True)
            self.models.append(model.to(self.device).eval())
        self.mean = np.asarray(self.bundle["normalization"]["mean"], dtype=np.float32)[None, :, None, None]
        self.std = np.asarray(self.bundle["normalization"]["std"], dtype=np.float32)[None, :, None, None]

    @torch.inference_mode()
    def predict_arrays(self, raw, averaged, times, *, min_confidence=0.8, min_margin=0.15):
        if np.asarray(raw).shape[1:] != tuple(self.bundle["sensor_shape"]):
            raise ValueError(f"Model requires sensor shape {self.bundle['sensor_shape']}")
        x = prepare_trial(raw, averaged, times, self.bundle["preprocess"])
        x = torch.as_tensor((x - self.mean) / self.std, device=self.device).unsqueeze(0)
        probabilities = torch.stack([model(x).softmax(dim=-1) for model in self.models]).mean(dim=0)
        probabilities = (probabilities.clamp_min(1e-8).log() / self.bundle["temperature"]).softmax(dim=-1)[0]
        scores, indices = probabilities.topk(2)
        confidence, margin = float(scores[0]), float(scores[0] - scores[1])
        accepted = confidence >= min_confidence and margin >= min_margin
        return {
            "gesture": self.class_names[int(indices[0])] if accepted else "uncertain",
            "top_prediction": self.class_names[int(indices[0])],
            "confidence": confidence, "margin": margin, "accepted": accepted,
            "probabilities": dict(zip(self.class_names, probabilities.cpu().tolist())),
            "robot_command_sent": False,
        }

    def predict_file(self, path, **kwargs):
        raw, averaged, times, _ = load_trial(path)
        return self.predict_arrays(raw, averaged, times, **kwargs)
