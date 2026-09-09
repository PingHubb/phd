#!/usr/bin/env python3
"""Train a normal-only CNN-GRU tactile proximity-anomaly model.

The trainer never connects to the sensor or robot. It learns to predict the
next normal tactile frame from recent environmental history. A held-out
normal recording calibrates the anomaly threshold, while the checkpoint keeps
the spatial CNN and temporal GRU reusable for later gesture-learning stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset


warnings.filterwarnings(
    "ignore",
    message="CUDA initialization.*",
    category=UserWarning,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

from phd.dependence.tactile_models import TactileProximityCNNGRU  # noqa: E402
from phd.dependence.tactile_proximity import (  # noqa: E402
    localized_change_scores,
)


FORMAT_VERSION = 1
DEFAULT_SESSION = "environment_v1"
DEFAULT_ROWS = 10
DEFAULT_COLS = 10
DEFAULT_SEQUENCE_LENGTH = 8
AI_DFM_CHANNELS = ("diffPerData", "diffPerDataAve", "frameDiff")


def _resource_root() -> Path:
    configured = os.environ.get("PINGLAB_RESOURCE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return PACKAGE_ROOT / "resource"


def _ai_root() -> Path:
    configured = os.environ.get("PINGLAB_AI_RESOURCE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return _resource_root() / "ai"


def _default_dataset_root() -> Path:
    return _ai_root() / "data" / "ai_direct_finger_motion"


def _default_output_root() -> Path:
    return _ai_root() / "models" / "tactile_proximity"


def _session_path(session_or_path: str, dataset_root: Path) -> Path:
    value = Path(session_or_path).expanduser()
    if value.exists() or value.is_absolute() or "/" in session_or_path:
        return value
    tag = session_or_path
    if not tag.startswith("session_"):
        tag = f"session_{tag}"
    return dataset_root / tag


def _recording_paths(
    session_specs,
    dataset_root: Path,
) -> tuple[list[Path], list[Path]]:
    session_paths = [
        _session_path(str(session), dataset_root)
        for session in session_specs
    ]
    recordings = []
    for session_path in session_paths:
        if session_path.is_file() and session_path.suffix.lower() == ".npz":
            recordings.append(session_path)
            continue
        if not session_path.is_dir():
            raise FileNotFoundError(
                f"Normal-data session not found: {session_path}"
            )
        recordings.extend(sorted(session_path.glob("*.npz")))
    recordings = sorted(dict.fromkeys(path.resolve() for path in recordings))
    if not recordings:
        raise FileNotFoundError(
            "No normal-environment .npz recordings were found."
        )
    return recordings, session_paths


def _metadata(payload: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata_json" not in payload.files:
        return {}
    try:
        value = np.asarray(payload["metadata_json"])
        text = str(value.item() if value.shape == () else value.reshape(-1)[0])
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def _payload_sensor_shape(
    payload: np.lib.npyio.NpzFile,
) -> tuple[int, int] | None:
    metadata = _metadata(payload)
    try:
        physical_rows = int(metadata.get("sensor_rows", 0))
        physical_cols = int(metadata.get("sensor_cols", 0))
    except (TypeError, ValueError):
        physical_rows, physical_cols = 0, 0

    # AI-DFM recordings intentionally store column_major_matrix_view arrays,
    # whose model-facing shape is (sensor_cols, sensor_rows). The array shape
    # is therefore authoritative for the CNN input; metadata still describes
    # the physical sensor dimensions.
    for name in (
        "diffPerData",
        "diffPerDataAve",
        "raw_frames",
        "rawData",
        "heatmap_frames",
        "frames",
    ):
        if name not in payload.files:
            continue
        shape = tuple(np.asarray(payload[name]).shape)
        if len(shape) == 4 and shape[1] == 1:
            shape = (shape[0], shape[2], shape[3])
        if len(shape) == 3 and shape[1] > 0 and shape[2] > 0:
            return int(shape[1]), int(shape[2])
    if physical_rows > 0 and physical_cols > 0:
        return physical_rows, physical_cols
    return None


def resolve_sensor_shape(
    paths,
    *,
    rows: int | None = None,
    cols: int | None = None,
) -> tuple[int, int]:
    if (rows is None) != (cols is None):
        raise ValueError("Provide both --rows and --cols, or neither")
    requested = None
    if rows is not None and cols is not None:
        requested = (int(rows), int(cols))
        if requested[0] <= 0 or requested[1] <= 0:
            raise ValueError("--rows and --cols must be positive")

    detected = {}
    for path_value in paths:
        path = Path(path_value).expanduser()
        with np.load(path, allow_pickle=False) as payload:
            shape = _payload_sensor_shape(payload)
        if shape is not None:
            detected[path] = shape

    unique_shapes = sorted(set(detected.values()))
    if len(unique_shapes) > 1:
        summary = ", ".join(
            f"{path.name}={shape[0]}x{shape[1]}"
            for path, shape in detected.items()
        )
        raise ValueError(
            "Normal-environment recordings contain different sensor "
            f"dimensions: {summary}"
        )
    if requested is not None:
        if unique_shapes and unique_shapes[0] != requested:
            detected_shape = unique_shapes[0]
            raise ValueError(
                f"Requested sensor shape {requested[0]}x{requested[1]} does "
                f"not match recorded shape {detected_shape[0]}x"
                f"{detected_shape[1]}"
            )
        return requested
    if not unique_shapes:
        raise ValueError(
            "Could not infer sensor dimensions from the recordings. Provide "
            "both --rows and --cols."
        )
    return unique_shapes[0]


def resolve_physical_sensor_shape(paths, *, fallback=None):
    """Resolve the physical sensor dimensions recorded in NPZ metadata."""
    detected = {}
    for path_value in paths:
        path = Path(path_value).expanduser()
        with np.load(path, allow_pickle=False) as payload:
            metadata = _metadata(payload)
        try:
            shape = (
                int(metadata.get("sensor_rows", 0)),
                int(metadata.get("sensor_cols", 0)),
            )
        except (TypeError, ValueError):
            shape = (0, 0)
        if shape[0] > 0 and shape[1] > 0:
            detected[path] = shape

    unique_shapes = sorted(set(detected.values()))
    if len(unique_shapes) > 1:
        summary = ", ".join(
            f"{path.name}={shape[0]}x{shape[1]}"
            for path, shape in detected.items()
        )
        raise ValueError(
            "Recordings contain different physical sensor dimensions: "
            f"{summary}"
        )
    if unique_shapes:
        return unique_shapes[0]
    if fallback is not None:
        return int(fallback[0]), int(fallback[1])
    raise ValueError(
        "Could not infer physical sensor dimensions from recording metadata"
    )


def _matrix_series(value, *, name: str, rows: int, cols: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    elif array.ndim == 2 and array.shape[1] == rows * cols:
        array = array.reshape(-1, rows, cols)
    if array.ndim != 3 or tuple(array.shape[1:]) != (rows, cols):
        raise ValueError(
            f"{name} shape is {array.shape}; expected (frames, {rows}, {cols})"
        )
    if array.shape[0] == 0:
        raise ValueError(f"{name} contains no frames")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.array(array, dtype=np.float32, copy=True)


def _auto_channel_names(payload: np.lib.npyio.NpzFile) -> tuple[str, ...]:
    if all(name in payload.files for name in AI_DFM_CHANNELS[:2]):
        return AI_DFM_CHANNELS
    if "raw_frames" in payload.files:
        return ("raw_frames", "frameDiff")
    if "rawData" in payload.files:
        return ("rawData", "frameDiff")
    if "heatmap_frames" in payload.files:
        return ("heatmap_frames", "frameDiff")
    if "frames" in payload.files:
        return ("frames", "frameDiff")
    raise KeyError(
        "Recording has no supported tactile frame key. Expected AI-DFM "
        "arrays, raw_frames, rawData, heatmap_frames, or frames."
    )


def _load_channels(
    payload: np.lib.npyio.NpzFile,
    channel_names: tuple[str, ...] | None,
    *,
    rows: int,
    cols: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    names = channel_names or _auto_channel_names(payload)
    base = None
    channels = []
    for name in names:
        if name == "frameDiff":
            if base is None:
                preferred = (
                    "diffPerDataAve",
                    "raw_frames",
                    "rawData",
                    "heatmap_frames",
                    "frames",
                )
                base_name = next(
                    (
                        candidate
                        for candidate in preferred
                        if candidate in payload.files
                    ),
                    None,
                )
                if base_name is None:
                    raise KeyError(
                        "Cannot derive frameDiff without a frame array"
                    )
                base = _matrix_series(
                    payload[base_name],
                    name=base_name,
                    rows=rows,
                    cols=cols,
                )
            frame_difference = np.zeros_like(base)
            frame_difference[1:] = base[1:] - base[:-1]
            channels.append(frame_difference)
            continue

        if name not in payload.files:
            raise KeyError(f"Recording is missing requested channel: {name}")
        channel = _matrix_series(
            payload[name],
            name=name,
            rows=rows,
            cols=cols,
        )
        if base is None or name == "diffPerDataAve":
            base = channel
        channels.append(channel)

    frame_counts = {channel.shape[0] for channel in channels}
    if len(frame_counts) != 1:
        raise ValueError("Tactile channel frame counts do not match")
    return np.stack(channels, axis=1).astype(np.float32), tuple(names)


@dataclass
class EnvironmentRecording:
    path: Path
    frames: np.ndarray
    channels: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])


def load_environment_recordings(
    paths,
    *,
    rows=DEFAULT_ROWS,
    cols=DEFAULT_COLS,
    channels: tuple[str, ...] | None = None,
) -> tuple[list[EnvironmentRecording], tuple[str, ...]]:
    recordings = []
    selected_channels = channels
    for path_value in paths:
        path = Path(path_value).expanduser()
        with np.load(path, allow_pickle=False) as payload:
            frames, detected_channels = _load_channels(
                payload,
                selected_channels,
                rows=int(rows),
                cols=int(cols),
            )
            if selected_channels is None:
                selected_channels = detected_channels
            if detected_channels != selected_channels:
                raise ValueError(
                    f"{path.name} channels {detected_channels} do not match "
                    f"{selected_channels}"
                )
            recordings.append(
                EnvironmentRecording(
                    path=path,
                    frames=frames,
                    channels=detected_channels,
                    metadata=_metadata(payload),
                )
            )
    if not recordings or selected_channels is None:
        raise ValueError("No valid normal-environment recordings were loaded")
    return recordings, selected_channels


def split_recordings(
    recordings: list[EnvironmentRecording],
    *,
    validation_count=1,
    seed=20260824,
) -> tuple[list[EnvironmentRecording], list[EnvironmentRecording]]:
    validation_count = max(1, int(validation_count))
    if len(recordings) <= validation_count:
        raise ValueError(
            f"Need more than {validation_count} recordings for a file-level "
            "train/validation split."
        )
    order = np.random.default_rng(int(seed)).permutation(len(recordings))
    validation_indices = set(order[-validation_count:].tolist())
    train = [
        recording
        for index, recording in enumerate(recordings)
        if index not in validation_indices
    ]
    validation = [
        recording
        for index, recording in enumerate(recordings)
        if index in validation_indices
    ]
    return train, validation


def compute_normalization(
    recordings: list[EnvironmentRecording],
    *,
    std_floor_ratio=0.1,
    balance_recordings=True,
) -> dict[str, np.ndarray | float]:
    if balance_recordings:
        recording_means = np.stack(
            [recording.frames.mean(axis=0) for recording in recordings]
        )
        recording_second_moments = np.stack(
            [np.square(recording.frames).mean(axis=0) for recording in recordings]
        )
        mean = recording_means.mean(axis=0)
        variance = np.maximum(
            recording_second_moments.mean(axis=0) - np.square(mean),
            0.0,
        )
        std = np.sqrt(variance)
    else:
        frames = np.concatenate(
            [recording.frames for recording in recordings],
            axis=0,
        )
        mean = frames.mean(axis=0)
        std = frames.std(axis=0)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    positive = std[std > 1e-8]
    reference = float(np.median(positive)) if positive.size else 1.0
    std_floor = max(1e-6, reference * max(0.0, float(std_floor_ratio)))
    std = np.maximum(std, std_floor).astype(np.float32)
    return {
        "mean": mean,
        "std": std,
        "std_floor": float(std_floor),
    }


class NormalSequenceDataset(Dataset):
    """Normal frame histories paired with their immediately following frame."""

    def __init__(
        self,
        recordings,
        *,
        sequence_length,
        normalization,
        stride=1,
        augment=False,
        noise_std=0.02,
        clip=12.0,
        seed=20260824,
    ):
        self.recordings = list(recordings)
        self.sequence_length = int(sequence_length)
        self.normalization = normalization
        self.augment = bool(augment)
        self.noise_std = max(0.0, float(noise_std))
        self.clip = max(1.0, float(clip))
        self._rng = np.random.default_rng(int(seed))
        self.indices = []
        step = max(1, int(stride))
        for recording_index, recording in enumerate(self.recordings):
            for target_index in range(
                self.sequence_length,
                recording.frame_count,
                step,
            ):
                self.indices.append((recording_index, target_index))
        if not self.indices:
            raise ValueError(
                "Normal recordings are shorter than the requested sequence "
                "length"
            )

    def __len__(self):
        return len(self.indices)

    def _normalize(self, frames):
        normalized = (
            np.asarray(frames, dtype=np.float32) - self.normalization["mean"]
        ) / self.normalization["std"]
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def __getitem__(self, index):
        recording_index, target_index = self.indices[index]
        recording = self.recordings[recording_index]
        history = self._normalize(
            recording.frames[
                target_index - self.sequence_length:target_index
            ]
        )
        target = self._normalize(recording.frames[target_index])
        if self.augment and self.noise_std > 0.0:
            history = history + self._rng.normal(
                0.0,
                self.noise_std,
                size=history.shape,
            ).astype(np.float32)
        return {
            "history": torch.from_numpy(history),
            "target": torch.from_numpy(target),
            "recording_index": int(recording_index),
            "target_index": int(target_index),
        }


def _indices_by_recording(dataset):
    groups = {}
    for dataset_index, (recording_index, _target_index) in enumerate(
        dataset.indices
    ):
        groups.setdefault(int(recording_index), []).append(dataset_index)
    return {
        recording_index: np.asarray(indices, dtype=np.int64)
        for recording_index, indices in groups.items()
    }


def balanced_recording_subset_indices(dataset):
    """Return an equal, deterministic spread of windows from every trial."""
    groups = _indices_by_recording(dataset)
    if not groups:
        return []
    per_recording = min(len(indices) for indices in groups.values())
    selected = []
    for recording_index in sorted(groups):
        indices = groups[recording_index]
        positions = np.linspace(
            0,
            len(indices) - 1,
            per_recording,
            dtype=np.int64,
        )
        selected.extend(indices[positions].tolist())
    return selected


class RecordingBalancedSampler(Sampler):
    """Sample the same number of windows from each recording per epoch."""

    def __init__(self, dataset, *, seed=20260824):
        self.groups = _indices_by_recording(dataset)
        if not self.groups:
            raise ValueError("Cannot balance an empty sequence dataset")
        self.samples_per_recording = min(
            len(indices) for indices in self.groups.values()
        )
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return self.samples_per_recording * len(self.groups)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        selected = []
        for recording_index in sorted(self.groups):
            indices = self.groups[recording_index]
            if len(indices) == self.samples_per_recording:
                trial_selection = indices.copy()
            else:
                trial_selection = rng.choice(
                    indices,
                    size=self.samples_per_recording,
                    replace=False,
                )
            selected.extend(trial_selection.tolist())
        rng.shuffle(selected)
        return iter(selected)


def _seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _run_epoch(
    model,
    loader,
    *,
    device,
    optimizer=None,
    scaler=None,
    grad_clip=1.0,
):
    training = optimizer is not None
    model.train(training)
    criterion = nn.SmoothL1Loss()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        history = batch["history"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        batch_size = int(history.shape[0])
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            use_amp = scaler is not None and device.type == "cuda"
            amp_context = (
                torch.amp.autocast("cuda") if use_amp else nullcontext()
            )
            with amp_context:
                prediction = model(history)["predicted_frame"]
                loss = criterion(prediction, target)
            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip > 0.0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0.0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
    return total_loss / max(1, total_samples)


def _collect_components(
    model,
    loader,
    *,
    device,
    top_fraction,
    latent_mean=None,
    latent_std=None,
):
    model.eval()
    prediction_error = []
    embeddings = []
    recording_indices = []
    target_indices = []
    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            prediction = model(history)["predicted_frame"]
            absolute_error = (prediction - target).abs().flatten(1)
            top_count = max(
                1,
                int(math.ceil(absolute_error.shape[1] * float(top_fraction))),
            )
            local_error = absolute_error.topk(top_count, dim=1).values.mean(1)
            current_embedding = model.encode_frames(target)
            prediction_error.append(local_error.cpu().numpy())
            embeddings.append(current_embedding.cpu().numpy())
            recording_indices.append(batch["recording_index"].numpy())
            target_indices.append(batch["target_index"].numpy())

    result = {
        "prediction_error": np.concatenate(prediction_error).astype(
            np.float64
        ),
        "embedding": np.concatenate(embeddings).astype(np.float64),
        "recording_index": np.concatenate(recording_indices).astype(np.int64),
        "target_index": np.concatenate(target_indices).astype(np.int64),
    }
    if latent_mean is not None and latent_std is not None:
        standardized = (result["embedding"] - latent_mean) / latent_std
        result["latent_distance"] = np.sqrt(
            np.mean(np.square(standardized), axis=1)
        )
    return result


def _robust_location_scale(values):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad_scale = float(1.4826 * np.median(np.abs(values - median)))
    if mad_scale < 1e-8:
        mad_scale = max(float(np.std(values)), 1e-8)
    return median, mad_scale


def fit_anomaly_calibration(
    train_components,
    validation_components,
    *,
    threshold_quantile=0.995,
    prediction_weight=0.75,
):
    latent_mean = train_components["embedding"].mean(axis=0)
    latent_std = train_components["embedding"].std(axis=0)
    positive = latent_std[latent_std > 1e-8]
    floor = float(np.median(positive) * 0.1) if positive.size else 1e-6
    latent_std = np.maximum(latent_std, max(floor, 1e-6))

    def latent_distance(components):
        standardized = (components["embedding"] - latent_mean) / latent_std
        return np.sqrt(np.mean(np.square(standardized), axis=1))

    train_latent = latent_distance(train_components)
    validation_latent = latent_distance(validation_components)
    prediction_center, prediction_scale = _robust_location_scale(
        train_components["prediction_error"]
    )
    latent_center, latent_scale = _robust_location_scale(train_latent)
    prediction_weight = min(max(float(prediction_weight), 0.0), 1.0)

    def combined(prediction, latent):
        prediction_z = np.maximum(
            0.0,
            (prediction - prediction_center) / prediction_scale,
        )
        latent_z = np.maximum(0.0, (latent - latent_center) / latent_scale)
        return prediction_weight * prediction_z + (
            1.0 - prediction_weight
        ) * latent_z

    train_score = combined(
        train_components["prediction_error"],
        train_latent,
    )
    validation_score = combined(
        validation_components["prediction_error"],
        validation_latent,
    )
    quantile = min(max(float(threshold_quantile), 0.5), 0.99999)
    threshold = float(np.quantile(validation_score, quantile))
    calibration = {
        "latent_mean": latent_mean.astype(np.float32),
        "latent_std": latent_std.astype(np.float32),
        "prediction_center": float(prediction_center),
        "prediction_scale": float(prediction_scale),
        "latent_center": float(latent_center),
        "latent_scale": float(latent_scale),
        "prediction_weight": float(prediction_weight),
        "threshold_quantile": float(quantile),
        "threshold": float(threshold),
    }
    return calibration, train_score, validation_score, validation_latent


def fit_localized_change_calibration(
    recordings,
    normalization,
    channels,
    *,
    channel="diffPerDataAve",
    warmup_frames=30,
    top_taxels=3,
    spatial_window=2,
    allowance=0.3,
    decay=0.95,
    baseline_alpha=0.0005,
    threshold_quantile=0.999,
    input_clip=12.0,
):
    """Calibrate persistent local-change evidence from all normal trials."""
    if channel not in channels:
        return {}, np.empty(0, dtype=np.float64)
    channel_index = channels.index(channel)
    traces = []
    for recording in recordings:
        normalized = (
            recording.frames[:, channel_index]
            - normalization["mean"][channel_index]
        ) / normalization["std"][channel_index]
        normalized = np.clip(
            normalized,
            -float(input_clip),
            float(input_clip),
        )
        trace = localized_change_scores(
            normalized,
            warmup_frames=warmup_frames,
            top_taxels=top_taxels,
            spatial_window=spatial_window,
            allowance=allowance,
            decay=decay,
            baseline_alpha=baseline_alpha,
        )
        if trace.size:
            traces.append(trace)
    if not traces:
        return {}, np.empty(0, dtype=np.float64)

    scores = np.concatenate(traces).astype(np.float64)
    quantile = min(max(float(threshold_quantile), 0.5), 0.99999)
    calibration = {
        "localized_change_channel": str(channel),
        "localized_change_warmup_frames": int(warmup_frames),
        "localized_change_top_taxels": int(top_taxels),
        "localized_change_spatial_window": int(spatial_window),
        "localized_change_allowance": float(allowance),
        "localized_change_decay": float(decay),
        "localized_change_baseline_alpha": float(baseline_alpha),
        "localized_change_threshold_quantile": float(quantile),
        "localized_change_threshold": float(np.quantile(scores, quantile)),
    }
    return calibration, scores


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _checkpoint(model, config, normalization, calibration, metrics=None):
    return {
        "format_version": FORMAT_VERSION,
        "model_type": "tactile_proximity_cnn_gru",
        "model_state": model.state_dict(),
        "config": dict(config),
        "normalization": {
            "mean": np.asarray(normalization["mean"], dtype=np.float32),
            "std": np.asarray(normalization["std"], dtype=np.float32),
            "std_floor": float(normalization["std_floor"]),
        },
        "anomaly_calibration": calibration,
        "metrics": dict(metrics or {}),
    }


def _save_history(path, history):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _save_validation_scores(
    output_dir,
    validation_components,
    validation_score,
    validation_latent,
):
    np.savez_compressed(
        output_dir / "validation_scores.npz",
        anomaly_score=validation_score,
        prediction_error=validation_components["prediction_error"],
        latent_distance=validation_latent,
        recording_index=validation_components["recording_index"],
        target_index=validation_components["target_index"],
    )
    with (output_dir / "validation_scores.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "recording_index",
                "target_index",
                "prediction_error",
                "latent_distance",
                "anomaly_score",
            ]
        )
        for values in zip(
            validation_components["recording_index"],
            validation_components["target_index"],
            validation_components["prediction_error"],
            validation_latent,
            validation_score,
        ):
            writer.writerow(values)


def _try_save_plots(output_dir, history, validation_score, threshold):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")
        return

    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(epochs, [row["train_loss"] for row in history], label="Train")
    axis.plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normal-frame prediction loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.hist(validation_score, bins=40, alpha=0.8, label="Held-out normal")
    axis.axvline(threshold, color="red", linestyle="--", label="Threshold")
    axis.set_xlabel("Anomaly score")
    axis.set_ylabel("Normal samples")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "normal_score_distribution.png", dpi=180)
    plt.close(figure)


def train(args):
    _seed_everything(args.seed)
    recording_paths, session_paths = _recording_paths(
        args.sessions,
        args.dataset_root.expanduser(),
    )
    args.rows, args.cols = resolve_sensor_shape(
        recording_paths,
        rows=args.rows,
        cols=args.cols,
    )
    physical_rows, physical_cols = resolve_physical_sensor_shape(
        recording_paths,
        fallback=(args.rows, args.cols),
    )
    requested_channels = None
    if str(args.channels).strip().lower() != "auto":
        requested_channels = tuple(
            value.strip()
            for value in str(args.channels).split(",")
            if value.strip()
        )
    recordings, channel_names = load_environment_recordings(
        recording_paths,
        rows=args.rows,
        cols=args.cols,
        channels=requested_channels,
    )
    train_recordings, validation_recordings = split_recordings(
        recordings,
        validation_count=args.val_trials,
        seed=args.seed,
    )
    normalization = compute_normalization(
        train_recordings,
        std_floor_ratio=args.std_floor_ratio,
        balance_recordings=args.balance_trials,
    )
    train_dataset = NormalSequenceDataset(
        train_recordings,
        sequence_length=args.seq_len,
        normalization=normalization,
        stride=args.stride,
        augment=not args.no_augment,
        noise_std=args.aug_noise_std,
        clip=args.input_clip,
        seed=args.seed,
    )
    train_scoring_dataset = NormalSequenceDataset(
        train_recordings,
        sequence_length=args.seq_len,
        normalization=normalization,
        stride=args.stride,
        augment=False,
        clip=args.input_clip,
        seed=args.seed,
    )
    validation_dataset = NormalSequenceDataset(
        validation_recordings,
        sequence_length=args.seq_len,
        normalization=normalization,
        stride=args.stride,
        augment=False,
        clip=args.input_clip,
        seed=args.seed,
    )

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    pin_memory = device.type == "cuda"
    train_sampler = (
        RecordingBalancedSampler(train_dataset, seed=args.seed)
        if args.balance_trials
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    train_scoring_source = train_scoring_dataset
    if args.balance_trials:
        train_scoring_source = Subset(
            train_scoring_dataset,
            balanced_recording_subset_indices(train_scoring_dataset),
        )
    train_scoring_loader = DataLoader(
        train_scoring_source,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    session_name = "_".join(
        path.name.removeprefix("session_") for path in session_paths
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or (
        f"cnn_gru_proximity_{session_name}_{timestamp}"
    )
    output_dir = args.output_dir.expanduser() / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "task": "normal_only_tactile_proximity_anomaly_detection",
        "sensor_rows": int(args.rows),
        "sensor_cols": int(args.cols),
        "physical_sensor_rows": int(physical_rows),
        "physical_sensor_cols": int(physical_cols),
        "channels": list(channel_names),
        "in_channels": len(channel_names),
        "sequence_length": int(args.seq_len),
        "d_model": int(args.d_model),
        "gru_hidden": int(args.gru_hidden),
        "gru_layers": int(args.gru_layers),
        "dropout": float(args.dropout),
        "input_clip": float(args.input_clip),
        "top_error_fraction": float(args.top_error_fraction),
        "localized_change_channel": str(args.local_change_channel),
        "localized_change_warmup_frames": int(args.local_warmup_frames),
        "localized_change_top_taxels": int(args.local_top_taxels),
        "localized_change_spatial_window": int(
            args.local_spatial_window
        ),
        "recording_balanced_training": bool(args.balance_trials),
        "train_samples_available": int(len(train_dataset)),
        "train_samples_per_epoch": int(len(train_loader.sampler)),
        "samples_per_recording_per_epoch": (
            int(train_sampler.samples_per_recording)
            if train_sampler is not None
            else None
        ),
        "normal_sessions": [str(path) for path in session_paths],
        "normal_recordings": [str(path) for path in recording_paths],
        "train_recordings": [str(item.path) for item in train_recordings],
        "validation_recordings": [
            str(item.path) for item in validation_recordings
        ],
        "training_note": (
            "Normal-only training calibrates false alarms. Hand-proximity "
            "sensitivity must be measured with separate approach recordings."
        ),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    model = TactileProximityCNNGRU(
        in_channels=len(channel_names),
        sensor_rows=args.rows,
        sensor_cols=args.cols,
        d_model=args.d_model,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=max(2, args.patience // 3),
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and not args.no_amp
        else None
    )

    print("Training normal-only tactile proximity model")
    print(f"Device: {device}")
    print(f"Sensor: {args.rows}x{args.cols} | channels: {channel_names}")
    print(f"Train recordings: {[item.path.name for item in train_recordings]}")
    print(
        "Validation recordings: "
        f"{[item.path.name for item in validation_recordings]}"
    )
    if train_sampler is not None:
        print(
            f"Train samples available: {len(train_dataset)} | balanced "
            f"samples/epoch: {len(train_sampler)} "
            f"({train_sampler.samples_per_recording} per recording) | "
            f"validation samples: {len(validation_dataset)}"
        )
    else:
        print(
            f"Train samples: {len(train_dataset)} | "
            f"validation samples: {len(validation_dataset)}"
        )

    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    best_path = output_dir / "best_model.pt"
    latest_path = output_dir / "latest_model.pt"
    empty_calibration = {}
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
        )
        validation_loss = _run_epoch(
            model,
            validation_loader,
            device=device,
        )
        scheduler.step(validation_loss)
        row = {
            "epoch": int(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_loss),
            "val_loss": float(validation_loss),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train={train_loss:.6f} | "
            f"val={validation_loss:.6f} | lr={row['lr']:.2e}"
        )
        torch.save(
            _checkpoint(
                model,
                config,
                normalization,
                empty_calibration,
            ),
            latest_path,
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = float(validation_loss)
            best_epoch = int(epoch)
            bad_epochs = 0
            shutil.copy2(latest_path, best_path)
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; best validation loss "
                f"was epoch {best_epoch}."
            )
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    train_components = _collect_components(
        model,
        train_scoring_loader,
        device=device,
        top_fraction=args.top_error_fraction,
    )
    validation_components = _collect_components(
        model,
        validation_loader,
        device=device,
        top_fraction=args.top_error_fraction,
    )
    calibration, train_score, validation_score, validation_latent = (
        fit_anomaly_calibration(
            train_components,
            validation_components,
            threshold_quantile=args.threshold_quantile,
            prediction_weight=args.prediction_weight,
        )
    )
    local_calibration, local_scores = fit_localized_change_calibration(
        recordings,
        normalization,
        channel_names,
        channel=args.local_change_channel,
        warmup_frames=args.local_warmup_frames,
        top_taxels=args.local_top_taxels,
        spatial_window=args.local_spatial_window,
        allowance=args.local_cusum_allowance,
        decay=args.local_cusum_decay,
        baseline_alpha=args.local_baseline_alpha,
        threshold_quantile=args.local_threshold_quantile,
        input_clip=args.input_clip,
    )
    calibration.update(local_calibration)
    threshold = float(calibration["threshold"])
    metrics = {
        "best_epoch": int(best_epoch),
        "best_validation_prediction_loss": float(best_loss),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "train_samples": int(len(train_dataset)),
        "train_samples_per_epoch": int(len(train_loader.sampler)),
        "recording_balanced_training": bool(args.balance_trials),
        "validation_samples": int(len(validation_dataset)),
        "normal_threshold": threshold,
        "threshold_quantile": float(calibration["threshold_quantile"]),
        "validation_false_alarm_fraction": float(
            np.mean(validation_score > threshold)
        ),
        "train_score_mean": float(np.mean(train_score)),
        "validation_score_mean": float(np.mean(validation_score)),
        "validation_score_p95": float(np.quantile(validation_score, 0.95)),
        "validation_score_max": float(np.max(validation_score)),
    }
    if local_scores.size:
        local_threshold = float(
            calibration["localized_change_threshold"]
        )
        metrics.update(
            {
                "localized_change_threshold": local_threshold,
                "localized_change_threshold_quantile": float(
                    calibration[
                        "localized_change_threshold_quantile"
                    ]
                ),
                "localized_change_normal_fraction_above_threshold": float(
                    np.mean(local_scores > local_threshold)
                ),
                "localized_change_score_p95": float(
                    np.quantile(local_scores, 0.95)
                ),
                "localized_change_score_max": float(
                    np.max(local_scores)
                ),
            }
        )
    torch.save(
        _checkpoint(model, config, normalization, calibration, metrics),
        best_path,
    )
    labelled_best_path = output_dir / (
        f"best_model_{physical_rows}x{physical_cols}.pt"
    )
    shutil.copy2(best_path, labelled_best_path)
    _save_history(output_dir / "history.csv", history)
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=_json_value),
        encoding="utf-8",
    )
    _save_validation_scores(
        output_dir,
        validation_components,
        validation_score,
        validation_latent,
    )
    if not args.no_plots:
        _try_save_plots(output_dir, history, validation_score, threshold)

    if args.update_latest:
        alias = args.output_dir.expanduser() / "latest_proximity_cnn_gru.pt"
        shape_alias = args.output_dir.expanduser() / (
            "latest_proximity_cnn_gru_"
            f"{physical_rows}x{physical_cols}.pt"
        )
        alias.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, alias)
        shutil.copy2(best_path, shape_alias)
        print(f"Latest proximity-model alias updated: {alias}")
        print(f"Size-specific proximity-model alias updated: {shape_alias}")

    print("\nTraining complete")
    print(f"Best model: {best_path}")
    print(f"Size-labelled model: {labelled_best_path}")
    print(f"Metrics: {output_dir / 'metrics.json'}")
    print(f"Normal anomaly threshold: {threshold:.6f}")
    if local_scores.size:
        print(
            "Localized-change threshold: "
            f"{calibration['localized_change_threshold']:.6f}"
        )
    print(
        "Important: this threshold controls normal-data false alarms; use a "
        "separate controlled approach test before claiming a detection range."
    )
    return output_dir


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train a normal-only CNN-GRU tactile proximity detector."
        )
    )
    parser.add_argument(
        "sessions",
        nargs="*",
        default=[DEFAULT_SESSION],
        help="Normal-only session tags, directories, or .npz recording paths.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_root(),
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--channels",
        default="auto",
        help=(
            "auto, or comma-separated NPZ arrays. frameDiff is derived. "
            "AI-DFM recordings automatically use diffPerData, "
            "diffPerDataAve,frameDiff."
        ),
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Sensor rows; inferred from recordings when omitted.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Sensor columns; inferred from recordings when omitted.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
    )
    parser.add_argument("--val-trials", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--gru-hidden", type=int, default=96)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--std-floor-ratio", type=float, default=0.1)
    parser.add_argument("--input-clip", type=float, default=12.0)
    parser.add_argument("--aug-noise-std", type=float, default=0.02)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--no-balance-trials",
        dest="balance_trials",
        action="store_false",
        help=(
            "Disable the default equal-per-recording normalization and "
            "training sampler."
        ),
    )
    parser.set_defaults(balance_trials=True)
    parser.add_argument("--top-error-fraction", type=float, default=0.1)
    parser.add_argument("--prediction-weight", type=float, default=0.75)
    parser.add_argument("--threshold-quantile", type=float, default=0.995)
    parser.add_argument(
        "--local-change-channel",
        default="diffPerDataAve",
    )
    parser.add_argument("--local-warmup-frames", type=int, default=30)
    parser.add_argument("--local-top-taxels", type=int, default=3)
    parser.add_argument("--local-spatial-window", type=int, default=2)
    parser.add_argument("--local-cusum-allowance", type=float, default=0.3)
    parser.add_argument("--local-cusum-decay", type=float, default=0.95)
    parser.add_argument("--local-baseline-alpha", type=float, default=0.0005)
    parser.add_argument(
        "--local-threshold-quantile",
        type=float,
        default=0.9999,
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--update-latest", action="store_true")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if (args.rows is None) != (args.cols is None):
        raise ValueError("Provide both --rows and --cols, or neither")
    if args.rows is not None and (args.rows <= 0 or args.cols <= 0):
        raise ValueError("--rows and --cols must be positive")
    if args.seq_len < 2:
        raise ValueError("--seq-len must be at least 2")
    if not 0.0 < args.top_error_fraction <= 1.0:
        raise ValueError("--top-error-fraction must be in (0, 1]")
    if args.local_warmup_frames < 2:
        raise ValueError("--local-warmup-frames must be at least 2")
    if args.local_top_taxels < 1:
        raise ValueError("--local-top-taxels must be positive")
    if args.local_spatial_window < 1:
        raise ValueError("--local-spatial-window must be positive")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
