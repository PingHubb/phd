#!/usr/bin/env python3
"""Train the first practical AI Direct Finger Motion tactile policy.

The script trains an offline CNN-GRU imitation model from recorded tactile
episodes. It is intentionally conservative: split by trial, normalize with
training data only, save metrics/plots, and never connect to the robot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


warnings.filterwarnings("ignore", message="CUDA initialization.*", category=UserWarning)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Single source of truth for the model architecture: the runtime module.
# Importing it (instead of keeping a copy here) guarantees that trained
# checkpoints always load on the execution side.
try:
    from phd.dependence.tactile_models import TactileCNNGRUPolicy
except ModuleNotFoundError:  # running directly from the source tree
    import sys

    sys.path.insert(0, str(PACKAGE_ROOT.parent))
    from phd.dependence.tactile_models import TactileCNNGRUPolicy

DEFAULT_SESSION = "first_trial_no_rotation"
DEFAULT_TARGET_KEY = "intended_velocity_target"
DEFAULT_SEQ_LEN = 16
DEFAULT_CHANNELS = ("diffPerData", "diffPerDataAve", "frameDiff", "touchMask")
DEFAULT_AUX_FEATURES = (
    "center_row",
    "center_col",
    "delta_row",
    "delta_col",
    "delta_row_norm",
    "delta_col_norm",
    "speed",
    "peak_value",
    "mean_active_value",
    "touch_present",
    "control_frame_idx",
)
AXIS_NAMES = ("vx", "vy", "vz", "rx", "ry", "rz")
LINEAR_AXIS_COUNT = 3
MODE_TO_INDEX = {
    "stop": 0,
    "single_finger_swipe": 1,
    "move": 1,
    "push": 2,
    "two_finger_pull": 3,
    "pull": 3,
}
INDEX_TO_MODE = {0: "stop", 1: "move", 2: "push", 3: "pull"}


def _resource_root() -> Path:
    env_resource = os.environ.get("PINGLAB_RESOURCE_ROOT", "").strip()
    if env_resource:
        return Path(env_resource).expanduser()
    return PACKAGE_ROOT / "resource"


def _ai_root() -> Path:
    explicit_ai = os.environ.get("PINGLAB_AI_RESOURCE_ROOT", "").strip()
    if explicit_ai:
        return Path(explicit_ai).expanduser()
    return _resource_root() / "ai"


def _default_dataset_root() -> Path:
    return _ai_root() / "data" / "ai_direct_finger_motion"


def _default_output_root() -> Path:
    return _ai_root() / "models" / "ai_direct_finger_motion"


def _session_path(session_or_path: str, dataset_root: Path) -> Path:
    value = Path(session_or_path).expanduser()
    if value.exists() or value.is_absolute() or "/" in session_or_path:
        return value
    tag = session_or_path
    if not tag.startswith("session_"):
        tag = f"session_{tag}"
    return dataset_root / tag


def _safe_array(data: np.lib.npyio.NpzFile, key: str, dtype=np.float32) -> np.ndarray:
    array = np.asarray(data[key], dtype=dtype)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _load_metadata(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata_json" not in data.files:
        return {}
    try:
        raw = data["metadata_json"]
        text = str(raw.item() if raw.shape == () else raw.reshape(-1)[0])
        return json.loads(text)
    except Exception:
        return {}


@dataclass
class Episode:
    path: Path
    frames: np.ndarray
    aux: np.ndarray
    target: np.ndarray
    mode: np.ndarray
    elapsed_sec: np.ndarray
    metadata: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.target.shape[0])


def _build_channels(data: np.lib.npyio.NpzFile, channels: tuple[str, ...]) -> np.ndarray:
    if "diffPerData" in data.files:
        base = _safe_array(data, "diffPerData")
    elif "diffPerDataAve" in data.files:
        base = _safe_array(data, "diffPerDataAve")
    else:
        raise KeyError("recording has neither diffPerData nor diffPerDataAve")

    channel_arrays: list[np.ndarray] = []
    for name in channels:
        if name == "frameDiff":
            frame_diff = np.zeros_like(base, dtype=np.float32)
            if base.shape[0] > 1:
                frame_diff[1:] = base[1:] - base[:-1]
            channel_arrays.append(frame_diff)
        elif name == "touchMask":
            if "touch_mask" in data.files:
                touch_mask = _safe_array(data, "touch_mask")
            else:
                touch_mask = (base < 0.0).astype(np.float32)
            channel_arrays.append(touch_mask.astype(np.float32))
        else:
            if name not in data.files:
                raise KeyError(f"missing channel array: {name}")
            channel_arrays.append(_safe_array(data, name))

    return np.stack(channel_arrays, axis=1).astype(np.float32)


def _normalize_center(center: np.ndarray, rows: int, cols: int) -> np.ndarray:
    center = np.nan_to_num(center.astype(np.float32), nan=-1.0, posinf=-1.0, neginf=-1.0)
    out = np.zeros_like(center, dtype=np.float32)
    row_den = max(1.0, float(rows - 1))
    col_den = max(1.0, float(cols - 1))
    out[:, 0] = np.where(center[:, 0] >= 0.0, center[:, 0] / row_den, 0.0)
    out[:, 1] = np.where(center[:, 1] >= 0.0, center[:, 1] / col_den, 0.0)
    return out


def _build_aux(data: np.lib.npyio.NpzFile, aux_features: tuple[str, ...], rows: int, cols: int) -> np.ndarray:
    frame_count = int(np.asarray(data["intended_velocity_target"]).shape[0])
    center = _safe_array(data, "center") if "center" in data.files else np.full((frame_count, 2), -1.0)
    center_norm = _normalize_center(center, rows, cols)
    delta = _safe_array(data, "delta") if "delta" in data.files else np.zeros((frame_count, 2), dtype=np.float32)
    delta_norm = (
        _safe_array(data, "delta_norm")
        if "delta_norm" in data.files
        else np.zeros((frame_count, 2), dtype=np.float32)
    )

    values: dict[str, np.ndarray] = {
        "center_row": center_norm[:, 0],
        "center_col": center_norm[:, 1],
        "delta_row": delta[:, 0],
        "delta_col": delta[:, 1],
        "delta_row_norm": delta_norm[:, 0],
        "delta_col_norm": delta_norm[:, 1],
        "speed": _safe_array(data, "speed") if "speed" in data.files else np.zeros(frame_count, dtype=np.float32),
        "peak_value": _safe_array(data, "peak_value")
        if "peak_value" in data.files
        else np.zeros(frame_count, dtype=np.float32),
        "mean_active_value": _safe_array(data, "mean_active_value")
        if "mean_active_value" in data.files
        else np.zeros(frame_count, dtype=np.float32),
        "touch_present": _safe_array(data, "touch_present")
        if "touch_present" in data.files
        else np.zeros(frame_count, dtype=np.float32),
        "control_frame_idx": _safe_array(data, "control_frame_idx")
        if "control_frame_idx" in data.files
        else np.zeros(frame_count, dtype=np.float32),
    }

    aux = np.stack([np.asarray(values.get(name, np.zeros(frame_count)), dtype=np.float32) for name in aux_features], axis=1)
    return np.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _mode_indices(data: np.lib.npyio.NpzFile, target: np.ndarray) -> np.ndarray:
    if "intended_mode" in data.files:
        names = [str(value) for value in np.asarray(data["intended_mode"]).reshape(-1)]
        mode = np.asarray([MODE_TO_INDEX.get(name, 1 if np.linalg.norm(target[idx]) > 1e-8 else 0) for idx, name in enumerate(names)])
    else:
        mode = np.asarray([1 if np.linalg.norm(row) > 1e-8 else 0 for row in target])
    mode = mode.astype(np.int64)
    mode[np.linalg.norm(target, axis=1) <= 1e-8] = 0
    return mode


def load_episodes(
    session_path: Path,
    *,
    target_key: str,
    channels: tuple[str, ...],
    aux_features: tuple[str, ...],
) -> list[Episode]:
    paths = sorted(session_path.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {session_path}")

    episodes: list[Episode] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in (target_key, "diffPerDataAve") if key not in data.files]
            if missing:
                raise KeyError(f"{path.name} missing required key(s): {', '.join(missing)}")
            target = _safe_array(data, target_key)
            frames = _build_channels(data, channels)
            if frames.shape[0] != target.shape[0]:
                raise ValueError(f"{path.name} frame/target count mismatch: {frames.shape[0]} vs {target.shape[0]}")
            if target.ndim != 2 or target.shape[1] != 6:
                raise ValueError(f"{path.name} {target_key} shape is {target.shape}, expected (N, 6)")
            rows, cols = int(frames.shape[-2]), int(frames.shape[-1])
            aux = _build_aux(data, aux_features, rows, cols)
            elapsed_sec = (
                _safe_array(data, "elapsed_sec")
                if "elapsed_sec" in data.files
                else np.arange(target.shape[0], dtype=np.float32)
            )
            episodes.append(
                Episode(
                    path=path,
                    frames=frames,
                    aux=aux,
                    target=target.astype(np.float32),
                    mode=_mode_indices(data, target),
                    elapsed_sec=elapsed_sec.reshape(-1).astype(np.float32),
                    metadata=_load_metadata(data),
                )
            )
    return episodes


def load_episodes_from_sessions(
    session_specs: list[str],
    *,
    dataset_root: Path,
    target_key: str,
    channels: tuple[str, ...],
    aux_features: tuple[str, ...],
) -> tuple[list[Episode], list[Path]]:
    session_paths = [_session_path(session, dataset_root) for session in session_specs]
    episodes: list[Episode] = []
    for session_path in session_paths:
        episodes.extend(
            load_episodes(
                session_path,
                target_key=target_key,
                channels=channels,
                aux_features=aux_features,
            )
        )
    return episodes, session_paths


def split_episodes(
    episodes: list[Episode],
    val_trials: int,
    *,
    split_each_session: bool = False,
) -> tuple[list[Episode], list[Episode]]:
    train, validation, _test = split_episodes_with_test(
        episodes,
        val_trials,
        test_trials=0,
        split_each_session=split_each_session,
    )
    return train, validation


def split_episodes_with_test(
    episodes: list[Episode],
    val_trials: int,
    test_trials: int,
    *,
    split_each_session: bool = False,
) -> tuple[list[Episode], list[Episode], list[Episode]]:
    if len(episodes) < 2:
        raise ValueError("Need at least two trials for a train/validation split.")

    def split_group(group: list[Episode]):
        validation_count = max(1, int(val_trials))
        test_count = max(0, int(test_trials))
        reserved = validation_count + test_count
        if len(group) <= reserved:
            raise ValueError(
                f"Need more than {reserved} trials to retain training data "
                f"with {validation_count} validation and {test_count} test trials."
            )
        train_group = group[:-reserved]
        validation_end = len(group) - test_count if test_count else len(group)
        validation_group = group[-reserved:validation_end]
        test_group = group[-test_count:] if test_count else []
        return train_group, validation_group, test_group

    if split_each_session:
        train_episodes: list[Episode] = []
        val_episodes: list[Episode] = []
        test_episodes: list[Episode] = []
        session_order: list[Path] = []
        by_session: dict[Path, list[Episode]] = {}
        for episode in episodes:
            session_path = episode.path.parent
            if session_path not in by_session:
                session_order.append(session_path)
                by_session[session_path] = []
            by_session[session_path].append(episode)

        for session_path in session_order:
            session_episodes = by_session[session_path]
            try:
                train_group, validation_group, test_group = split_group(
                    session_episodes
                )
            except ValueError as exc:
                raise ValueError(f"{session_path}: {exc}") from exc
            train_episodes.extend(train_group)
            val_episodes.extend(validation_group)
            test_episodes.extend(test_group)
        return train_episodes, val_episodes, test_episodes

    return split_group(episodes)


def _stack_train_frames(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.concatenate([episode.frames for episode in episodes], axis=0)
    aux = np.concatenate([episode.aux for episode in episodes], axis=0)
    target = np.concatenate([episode.target for episode in episodes], axis=0)
    return frames, aux, target


def compute_normalization(episodes: list[Episode]) -> dict[str, np.ndarray]:
    frames, aux, target = _stack_train_frames(episodes)
    channel_mean = frames.mean(axis=(0, 2, 3)).astype(np.float32)
    channel_std = frames.std(axis=(0, 2, 3)).astype(np.float32)
    channel_std[channel_std < 1e-6] = 1.0

    aux_mean = aux.mean(axis=0).astype(np.float32)
    aux_std = aux.std(axis=0).astype(np.float32)
    aux_std[aux_std < 1e-6] = 1.0

    target_scale = np.max(np.abs(target), axis=0).astype(np.float32)
    target_scale[target_scale < 1e-6] = 1.0
    return {
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "aux_mean": aux_mean,
        "aux_std": aux_std,
        "target_scale": target_scale,
    }


class SequenceDataset(Dataset):
    # Axis semantics from the rule-based teacher (gesture_logic_direct_finger_motion):
    # sensor column motion -> vx (and two-finger horizontal swipe -> rz),
    # sensor row motion -> vz (and two-finger vertical swipe -> rx),
    # push/pull -> vy (spatially direction-invariant).
    COL_FLIP_TARGET_AXES = (0, 5)  # vx, rz
    ROW_FLIP_TARGET_AXES = (2, 3)  # vz, rx

    def __init__(
        self,
        episodes: list[Episode],
        *,
        seq_len: int,
        normalization: dict[str, np.ndarray],
        stride: int = 1,
        channels: tuple[str, ...] = DEFAULT_CHANNELS,
        aux_features: tuple[str, ...] = DEFAULT_AUX_FEATURES,
        augment: bool = False,
        aug_flip_prob: float = 0.5,
        aug_max_shift: int = 2,
        aug_noise_std: float = 0.02,
        aug_sensor_noise_max: float = 0.0,
        aug_drift_max: float = 0.0,
        aug_touch_threshold: float = -3.0,
        seed: int = 0,
    ):
        self.episodes = episodes
        self.seq_len = int(seq_len)
        self.normalization = normalization
        self.channels = tuple(channels)
        self.aux_features = tuple(aux_features)
        self.augment = bool(augment)
        self.aug_flip_prob = float(aug_flip_prob)
        self.aug_max_shift = int(aug_max_shift)
        self.aug_noise_std = float(aug_noise_std)
        self.aug_sensor_noise_max = float(aug_sensor_noise_max)
        self.aug_drift_max = float(aug_drift_max)
        self.aug_touch_threshold = float(aug_touch_threshold)
        self._rng = np.random.default_rng(seed)
        self._channel_idx = {name: idx for idx, name in enumerate(self.channels)}
        self._aux_idx = {name: idx for idx, name in enumerate(self.aux_features)}
        self._touch_mask_channel = self.channels.index("touchMask") if "touchMask" in self.channels else None
        self._noise_channel_indices = [
            idx for idx, name in enumerate(self.channels) if name != "touchMask"
        ]
        self.indices: list[tuple[int, int]] = []
        for episode_idx, episode in enumerate(episodes):
            for end_idx in range(self.seq_len - 1, episode.frame_count, max(1, int(stride))):
                self.indices.append((episode_idx, end_idx))

    def __len__(self) -> int:
        return len(self.indices)

    def _aux_col(self, aux: np.ndarray, name: str) -> np.ndarray | None:
        idx = self._aux_idx.get(name)
        return aux[:, idx] if idx is not None else None

    def _touch_present_mask(self, aux: np.ndarray) -> np.ndarray:
        touch = self._aux_col(aux, "touch_present")
        if touch is not None:
            return touch > 0.5
        return np.ones(aux.shape[0], dtype=bool)

    def _flip_aux_axis(self, aux: np.ndarray, *, center: str, deltas: tuple[str, ...]) -> None:
        touch = self._touch_present_mask(aux)
        center_col = self._aux_col(aux, center)
        if center_col is not None:
            center_col[touch] = 1.0 - center_col[touch]
        for name in deltas:
            delta_col = self._aux_col(aux, name)
            if delta_col is not None:
                delta_col *= -1.0

    def _active_cell_mask(self, x: np.ndarray) -> np.ndarray:
        if self._touch_mask_channel is not None:
            return (x[:, self._touch_mask_channel] > 0.5).any(axis=0)
        return (np.abs(x[:, 0]) > 1e-6).any(axis=0)

    def _shift_bounds(self, active: np.ndarray, axis: int, size: int) -> tuple[int, int]:
        occupied = np.where(active.any(axis=1 - axis))[0]
        if occupied.size == 0:
            return 0, 0
        low = max(-self.aug_max_shift, -int(occupied[0]))
        high = min(self.aug_max_shift, int(size - 1 - occupied[-1]))
        return low, max(low, high)

    def _augment_sample(
        self, x: np.ndarray, aux: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows, cols = int(x.shape[-2]), int(x.shape[-1])

        if self._rng.random() < self.aug_flip_prob:
            x = x[..., ::-1]
            for axis in self.COL_FLIP_TARGET_AXES:
                y[axis] = -y[axis]
            self._flip_aux_axis(aux, center="center_col", deltas=("delta_col", "delta_col_norm"))

        if self._rng.random() < self.aug_flip_prob:
            x = x[..., ::-1, :]
            for axis in self.ROW_FLIP_TARGET_AXES:
                y[axis] = -y[axis]
            self._flip_aux_axis(aux, center="center_row", deltas=("delta_row", "delta_row_norm"))

        if self.aug_max_shift > 0:
            active = self._active_cell_mask(x)
            row_low, row_high = self._shift_bounds(active, axis=0, size=rows)
            col_low, col_high = self._shift_bounds(active, axis=1, size=cols)
            shift_r = int(self._rng.integers(row_low, row_high + 1))
            shift_c = int(self._rng.integers(col_low, col_high + 1))
            if shift_r != 0 or shift_c != 0:
                shifted = np.zeros_like(x)
                src_r = slice(max(0, -shift_r), rows - max(0, shift_r))
                dst_r = slice(max(0, shift_r), rows - max(0, -shift_r))
                src_c = slice(max(0, -shift_c), cols - max(0, shift_c))
                dst_c = slice(max(0, shift_c), cols - max(0, -shift_c))
                shifted[..., dst_r, dst_c] = x[..., src_r, src_c]
                x = shifted
                touch = self._touch_present_mask(aux)
                center_row = self._aux_col(aux, "center_row")
                if center_row is not None and shift_r != 0:
                    center_row[touch] = np.clip(
                        center_row[touch] + shift_r / max(1.0, rows - 1.0), 0.0, 1.0
                    )
                center_col = self._aux_col(aux, "center_col")
                if center_col is not None and shift_c != 0:
                    center_col[touch] = np.clip(
                        center_col[touch] + shift_c / max(1.0, cols - 1.0), 0.0, 1.0
                    )

        if self.aug_noise_std > 0.0 and self._noise_channel_indices:
            x = np.ascontiguousarray(x)
            for channel_idx in self._noise_channel_indices:
                scale = self.aug_noise_std * float(self.normalization["channel_std"][channel_idx])
                x[:, channel_idx] += self._rng.normal(
                    0.0, scale, size=x[:, channel_idx].shape
                ).astype(np.float32)

        if self.aug_sensor_noise_max > 0.0 or self.aug_drift_max > 0.0:
            x = self._apply_sensor_perturbation(x)

        return np.ascontiguousarray(x), aux, y

    def _apply_sensor_perturbation(self, x: np.ndarray) -> np.ndarray:
        """Emulate physical sensor corruption on the input window only.

        Mirrors the deployment-time failure modes (i.i.d. electrical noise and
        baseline drift toward the touch threshold) consistently across the
        derived channels: the averaged channel receives the moving average of
        the same noise (spatially flipped, matching the live pipeline), the
        temporal-difference channel receives the frame-to-frame noise delta,
        and the touch mask is recomputed from the corrupted averaged channel.
        Velocity/mode targets are deliberately unchanged: the operator's
        finger is doing the same thing regardless of sensor health.
        """
        diff_idx = self._channel_idx.get("diffPerData")
        ave_idx = self._channel_idx.get("diffPerDataAve")
        frame_diff_idx = self._channel_idx.get("frameDiff")
        mask_idx = self._touch_mask_channel

        x = np.ascontiguousarray(x)
        frames = x.shape[0]
        rows, cols = int(x.shape[-2]), int(x.shape[-1])

        # Apply at most one perturbation family per window (mutually exclusive,
        # matching the one-family-at-a-time robustness benchmark). Stacking
        # noise and drift on the same window compounds the ambiguity and makes
        # the policy hedge velocity magnitudes on clean input.
        sigma = 0.0
        drift = 0.0
        enabled = [name for name, mx in (("noise", self.aug_sensor_noise_max), ("drift", self.aug_drift_max)) if mx > 0.0]
        if enabled and self._rng.random() < 0.5:
            choice = enabled[int(self._rng.integers(len(enabled)))]
            if choice == "noise":
                sigma = float(self._rng.uniform(0.0, self.aug_sensor_noise_max))
            else:
                drift = float(self._rng.uniform(0.0, self.aug_drift_max))
        if sigma <= 0.0 and drift <= 0.0:
            return x

        if sigma > 0.0:
            noise = self._rng.normal(0.0, sigma, size=(frames, rows, cols)).astype(np.float32)
            moving_avg = np.empty_like(noise)
            for t in range(frames):
                lo = max(0, t - 2)
                moving_avg[t] = noise[lo : t + 1].mean(axis=0)
            moving_avg = moving_avg[:, ::-1, :]
            if diff_idx is not None:
                x[:, diff_idx] += noise
            if ave_idx is not None:
                x[:, ave_idx] += moving_avg
            if frame_diff_idx is not None:
                frame_diff_noise = np.zeros_like(noise)
                frame_diff_noise[1:] = noise[1:] - noise[:-1]
                x[:, frame_diff_idx] += frame_diff_noise

        if drift > 0.0:
            # Toward the (negative) touch threshold; constant offsets cancel in
            # the temporal-difference channel, so frameDiff is unaffected.
            for idx in (diff_idx, ave_idx):
                if idx is not None:
                    x[:, idx] -= drift

        if mask_idx is not None and ave_idx is not None:
            x[:, mask_idx] = (x[:, ave_idx] < self.aug_touch_threshold).astype(np.float32)

        return x

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode_idx, end_idx = self.indices[idx]
        episode = self.episodes[episode_idx]
        start = end_idx - self.seq_len + 1
        x = episode.frames[start : end_idx + 1].copy()
        aux = episode.aux[start : end_idx + 1].copy()
        y = episode.target[end_idx].copy()
        mode = int(episode.mode[end_idx])

        if self.augment:
            x, aux, y = self._augment_sample(x, aux, y)

        x = (x - self.normalization["channel_mean"][None, :, None, None]) / self.normalization["channel_std"][
            None, :, None, None
        ]
        aux = (aux - self.normalization["aux_mean"][None, :]) / self.normalization["aux_std"][None, :]
        y_norm = y / self.normalization["target_scale"]

        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "aux": torch.from_numpy(aux.astype(np.float32)),
            "target": torch.from_numpy(y.astype(np.float32)),
            "target_norm": torch.from_numpy(y_norm.astype(np.float32)),
            "mode": torch.tensor(mode, dtype=torch.long),
            "episode_idx": torch.tensor(episode_idx, dtype=torch.long),
            "frame_idx": torch.tensor(end_idx, dtype=torch.long),
        }


def _cuda_available_safely() -> bool:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA initialization.*")
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False


def _seed_everything(seed: int, *, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def _mode_class_weights(episodes: list[Episode], device: torch.device) -> torch.Tensor:
    modes = np.concatenate([episode.mode for episode in episodes], axis=0)
    counts = np.bincount(modes, minlength=len(INDEX_TO_MODE)).astype(np.float32)
    counts[counts <= 0.0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = np.clip(weights, 0.25, 4.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _velocity_sample_weight(mode: torch.Tensor) -> torch.Tensor:
    weights = torch.ones_like(mode, dtype=torch.float32)
    weights = torch.where(mode == 0, weights * 0.75, weights)
    weights = torch.where(mode == 3, weights * 1.25, weights)
    return weights


def _apply_velocity_constraints(velocity: torch.Tensor, *, lock_rotation_axes: bool) -> torch.Tensor:
    if not lock_rotation_axes or velocity.shape[1] <= LINEAR_AXIS_COUNT:
        return velocity
    constrained = velocity.clone()
    constrained[:, LINEAR_AXIS_COUNT:] = 0.0
    return constrained


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    mode_criterion: nn.Module,
    velocity_weight: float,
    mode_weight: float,
    grad_clip: float,
    lock_rotation_axes: bool,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_velocity_loss = 0.0
    total_mode_loss = 0.0
    total_count = 0
    correct_mode = 0
    velocity_criterion = nn.SmoothL1Loss(reduction="none")

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        aux = batch["aux"].to(device, non_blocking=True)
        target_norm = batch["target_norm"].to(device, non_blocking=True)
        mode = batch["mode"].to(device, non_blocking=True)
        batch_size = int(x.shape[0])

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            use_amp = scaler is not None and device.type == "cuda"
            amp_context = torch.amp.autocast("cuda") if use_amp else nullcontext()
            with amp_context:
                output = model(x, aux)
                velocity_norm = _apply_velocity_constraints(
                    output["velocity_norm"],
                    lock_rotation_axes=lock_rotation_axes,
                )
                velocity_loss_per_axis = velocity_criterion(velocity_norm, target_norm)
                sample_weight = _velocity_sample_weight(mode).unsqueeze(1)
                velocity_loss = (velocity_loss_per_axis * sample_weight).mean()
                mode_loss = mode_criterion(output["mode_logits"], mode)
                loss = velocity_weight * velocity_loss + mode_weight * mode_loss

            if is_train:
                if scaler is not None and device.type == "cuda":
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
        total_velocity_loss += float(velocity_loss.detach().cpu()) * batch_size
        total_mode_loss += float(mode_loss.detach().cpu()) * batch_size
        total_count += batch_size
        correct_mode += int((output["mode_logits"].argmax(dim=1) == mode).sum().detach().cpu())

    denom = max(1, total_count)
    return {
        "loss": total_loss / denom,
        "velocity_loss": total_velocity_loss / denom,
        "mode_loss": total_mode_loss / denom,
        "mode_accuracy": correct_mode / denom,
    }


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    target_scale: np.ndarray,
    lock_rotation_axes: bool,
) -> dict[str, np.ndarray]:
    model.eval()
    pred_velocity: list[np.ndarray] = []
    target_velocity: list[np.ndarray] = []
    pred_mode: list[np.ndarray] = []
    true_mode: list[np.ndarray] = []
    episode_idx: list[np.ndarray] = []
    frame_idx: list[np.ndarray] = []
    target_scale_tensor = torch.tensor(target_scale, dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            aux = batch["aux"].to(device, non_blocking=True)
            output = model(x, aux)
            velocity_norm = _apply_velocity_constraints(
                output["velocity_norm"],
                lock_rotation_axes=lock_rotation_axes,
            )
            velocity = velocity_norm * target_scale_tensor[None, :]
            pred_velocity.append(velocity.detach().cpu().numpy().astype(np.float32))
            target_velocity.append(batch["target"].numpy().astype(np.float32))
            pred_mode.append(output["mode_logits"].argmax(dim=1).detach().cpu().numpy().astype(np.int64))
            true_mode.append(batch["mode"].numpy().astype(np.int64))
            episode_idx.append(batch["episode_idx"].numpy().astype(np.int64))
            frame_idx.append(batch["frame_idx"].numpy().astype(np.int64))

    return {
        "pred_velocity": np.concatenate(pred_velocity, axis=0),
        "target_velocity": np.concatenate(target_velocity, axis=0),
        "pred_mode": np.concatenate(pred_mode, axis=0),
        "true_mode": np.concatenate(true_mode, axis=0),
        "episode_idx": np.concatenate(episode_idx, axis=0),
        "frame_idx": np.concatenate(frame_idx, axis=0),
    }


def _metrics_from_predictions(preds: dict[str, np.ndarray], *, active_axis_count: int) -> dict[str, Any]:
    pred = preds["pred_velocity"]
    target = preds["target_velocity"]
    error = pred - target
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error**2, axis=0))
    active_axis_count = max(1, min(int(active_axis_count), pred.shape[1]))
    total_rmse = float(np.sqrt(np.mean(error[:, :active_axis_count] ** 2)))
    target_var = np.var(target, axis=0)
    raw_r2 = 1.0 - np.mean(error**2, axis=0) / np.maximum(target_var, 1e-8)
    r2_values: list[float | None] = []
    for axis_idx, value in enumerate(raw_r2):
        if target_var[axis_idx] < 1e-8:
            r2_values.append(None)
        else:
            r2_values.append(float(value))

    pred_mode = preds["pred_mode"]
    true_mode = preds["true_mode"]
    mode_accuracy = float(np.mean(pred_mode == true_mode)) if true_mode.size else 0.0
    confusion = np.zeros((len(INDEX_TO_MODE), len(INDEX_TO_MODE)), dtype=np.int64)
    for truth, pred_idx in zip(true_mode, pred_mode):
        if 0 <= int(truth) < confusion.shape[0] and 0 <= int(pred_idx) < confusion.shape[1]:
            confusion[int(truth), int(pred_idx)] += 1

    moving_true = np.linalg.norm(target[:, :active_axis_count], axis=1) > 1e-8
    moving_pred = np.linalg.norm(pred[:, :active_axis_count], axis=1) > 0.01
    stop_false_move = float(np.mean(moving_pred[~moving_true])) if np.any(~moving_true) else 0.0
    move_false_stop = float(np.mean(~moving_pred[moving_true])) if np.any(moving_true) else 0.0

    return {
        "mae": dict(zip(AXIS_NAMES, (float(value) for value in mae))),
        "rmse": dict(zip(AXIS_NAMES, (float(value) for value in rmse))),
        "r2": dict(zip(AXIS_NAMES, r2_values)),
        "total_rmse": total_rmse,
        "active_axis_count": active_axis_count,
        "mode_accuracy": mode_accuracy,
        "mode_confusion": confusion.tolist(),
        "stop_false_move_rate": stop_false_move,
        "move_false_stop_rate": move_false_stop,
    }


def _save_predictions_csv(path: Path, preds: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample", "episode_idx", "frame_idx", "true_mode", "pred_mode"]
            + [f"target_{axis}" for axis in AXIS_NAMES]
            + [f"pred_{axis}" for axis in AXIS_NAMES]
        )
        for idx in range(preds["target_velocity"].shape[0]):
            writer.writerow(
                [
                    idx,
                    int(preds["episode_idx"][idx]),
                    int(preds["frame_idx"][idx]),
                    INDEX_TO_MODE.get(int(preds["true_mode"][idx]), str(preds["true_mode"][idx])),
                    INDEX_TO_MODE.get(int(preds["pred_mode"][idx]), str(preds["pred_mode"][idx])),
                ]
                + [f"{value:.8g}" for value in preds["target_velocity"][idx]]
                + [f"{value:.8g}" for value in preds["pred_velocity"][idx]]
            )


def _try_save_plots(output_dir: Path, history: list[dict[str, float]], preds: dict[str, np.ndarray], metrics: dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "training_history.png", dpi=180)
    plt.close()

    target = preds["target_velocity"]
    pred = preds["pred_velocity"]
    limit = min(800, target.shape[0])
    sample_axis = np.arange(limit)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, axis_idx in zip(axes, range(3)):
        ax.plot(sample_axis, target[:limit, axis_idx], label=f"target {AXIS_NAMES[axis_idx]}", linewidth=1.5)
        ax.plot(sample_axis, pred[:limit, axis_idx], label=f"pred {AXIS_NAMES[axis_idx]}", linewidth=1.1, alpha=0.8)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        ax.set_ylabel("Velocity")
    axes[-1].set_xlabel("Validation sequence sample")
    fig.suptitle("Validation Velocity Prediction")
    fig.tight_layout()
    fig.savefig(figures_dir / "validation_velocity_timeseries.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis_idx, ax in enumerate(axes.reshape(-1)):
        ax.scatter(target[:, axis_idx], pred[:, axis_idx], s=8, alpha=0.35)
        min_value = float(min(target[:, axis_idx].min(), pred[:, axis_idx].min()))
        max_value = float(max(target[:, axis_idx].max(), pred[:, axis_idx].max()))
        if math.isclose(min_value, max_value):
            min_value -= 0.01
            max_value += 0.01
        ax.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1)
        ax.set_title(f"{AXIS_NAMES[axis_idx]} RMSE={metrics['rmse'][AXIS_NAMES[axis_idx]]:.4f}")
        ax.set_xlabel("Target")
        ax.set_ylabel("Prediction")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Predicted vs Target Velocity")
    fig.tight_layout()
    fig.savefig(figures_dir / "velocity_scatter.png", dpi=180)
    plt.close(fig)

    confusion = np.asarray(metrics["mode_confusion"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    labels = [INDEX_TO_MODE[idx] for idx in range(len(INDEX_TO_MODE))]
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_title("Mode Confusion Matrix")
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            ax.text(col, row, str(int(confusion[row, col])), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figures_dir / "mode_confusion.png", dpi=180)
    plt.close(fig)


def _save_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _checkpoint_payload(
    *,
    model: nn.Module,
    config: dict[str, Any],
    normalization: dict[str, np.ndarray],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    serializable_config = dict(config)
    serializable_config.update(
        {
            "channel_mean": normalization["channel_mean"].tolist(),
            "channel_std": normalization["channel_std"].tolist(),
            "aux_mean": normalization["aux_mean"].tolist(),
            "aux_std": normalization["aux_std"].tolist(),
            "target_scale": normalization["target_scale"].tolist(),
            "axis_names": list(AXIS_NAMES),
            "mode_to_index": dict(MODE_TO_INDEX),
            "index_to_mode": dict(INDEX_TO_MODE),
        }
    )
    return {
        "model_state": model.state_dict(),
        "model_class": "TactileCNNGRUPolicy",
        "config": serializable_config,
        "metrics": metrics or {},
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def train(args: argparse.Namespace) -> Path:
    cuda_available = (not args.cpu) and _cuda_available_safely()
    _seed_everything(int(args.seed), seed_cuda=cuda_available)
    channels = tuple(args.channels.split(","))
    aux_features = tuple(args.aux_features.split(","))
    session_specs = list(args.sessions or [DEFAULT_SESSION])
    dataset_root = args.dataset_root.expanduser()
    episodes, session_paths = load_episodes_from_sessions(
        session_specs,
        dataset_root=dataset_root,
        target_key=args.target_key,
        channels=channels,
        aux_features=aux_features,
    )
    train_episodes, val_episodes, test_episodes = split_episodes_with_test(
        episodes,
        args.val_trials,
        args.test_trials,
        split_each_session=len(session_paths) > 1,
    )
    normalization = compute_normalization(train_episodes)
    augment_enabled = not args.no_augment
    train_dataset = SequenceDataset(
        train_episodes,
        seq_len=args.seq_len,
        normalization=normalization,
        stride=args.stride,
        channels=channels,
        aux_features=aux_features,
        augment=augment_enabled,
        aug_flip_prob=args.aug_flip_prob,
        aug_max_shift=args.aug_max_shift,
        aug_noise_std=args.aug_noise_std,
        aug_sensor_noise_max=args.aug_sensor_noise_max,
        aug_drift_max=args.aug_drift_max,
        seed=int(args.seed),
    )
    val_dataset = SequenceDataset(
        val_episodes,
        seq_len=args.seq_len,
        normalization=normalization,
        stride=args.stride,
        channels=channels,
        aux_features=aux_features,
        augment=False,
    )
    test_dataset = None
    if test_episodes:
        test_dataset = SequenceDataset(
            test_episodes,
            seq_len=args.seq_len,
            normalization=normalization,
            stride=args.stride,
            channels=channels,
            aux_features=aux_features,
            augment=False,
        )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Not enough sequence samples for training/validation.")
    if test_dataset is not None and len(test_dataset) == 0:
        raise ValueError("Not enough sequence samples in the independent test split.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if len(session_paths) == 1:
        default_run_stem = session_paths[0].name
    else:
        session_stems = [path.name.removeprefix("session_") for path in session_paths]
        default_run_stem = "multi_" + "_".join(session_stems)
    run_name = args.run_name or f"cnn_gru_{default_run_stem}_{stamp}"
    output_dir = args.output_dir.expanduser() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Training AI Direct Finger Motion model")
    print(f"Sessions: {[str(path) for path in session_paths]}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Train trials: {[episode.path.name for episode in train_episodes]}")
    print(f"Validation trials: {[episode.path.name for episode in val_episodes]}")
    print(f"Test trials: {[episode.path.name for episode in test_episodes]}")
    print(
        f"Train samples: {len(train_dataset)} | "
        f"Validation samples: {len(val_dataset)} | "
        f"Test samples: {len(test_dataset) if test_dataset is not None else 0}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    sample_episode = episodes[0]
    config = {
        "model_family": "tactile_cnn_gru_policy",
        "dataset_session": str(session_paths[0]) if len(session_paths) == 1 else "",
        "dataset_sessions": [str(path) for path in session_paths],
        "dataset_files": [str(episode.path) for episode in episodes],
        "train_files": [str(episode.path) for episode in train_episodes],
        "val_files": [str(episode.path) for episode in val_episodes],
        "test_files": [str(episode.path) for episode in test_episodes],
        "target_key": args.target_key,
        "seq_len": int(args.seq_len),
        "channels": list(channels),
        "aux_feature_names": list(aux_features),
        "sensor_shape": list(sample_episode.frames.shape[-2:]),
        "in_channels": len(channels),
        "aux_dim": len(aux_features),
        "d_model": int(args.d_model),
        "gru_hidden": int(args.gru_hidden),
        "gru_layers": int(args.gru_layers),
        "dropout": float(args.dropout),
        "velocity_dim": 6,
        "mode_classes": len(INDEX_TO_MODE),
        "encoder_type": args.encoder,
        "lock_rotation_axes": bool(not args.allow_rotation_output),
        "augment": augment_enabled,
        "aug_flip_prob": float(args.aug_flip_prob),
        "aug_max_shift": int(args.aug_max_shift),
        "aug_noise_std": float(args.aug_noise_std),
        "aug_sensor_noise_max": float(args.aug_sensor_noise_max),
        "aug_drift_max": float(args.aug_drift_max),
        "recording_note": "offline training only; do not execute on robot until validated",
    }

    model = TactileCNNGRUPolicy(
        in_channels=len(channels),
        aux_dim=len(aux_features),
        d_model=args.d_model,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        velocity_dim=6,
        mode_classes=len(INDEX_TO_MODE),
        encoder_type=args.encoder,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, args.patience // 3),
    )
    mode_criterion = nn.CrossEntropyLoss(weight=_mode_class_weights(train_episodes, device))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and not args.no_amp else None

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, default=_json_default), encoding="utf-8")

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float]] = []
    best_path = output_dir / "best_model.pt"
    latest_path = output_dir / "latest_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            mode_criterion=mode_criterion,
            velocity_weight=args.velocity_loss_weight,
            mode_weight=args.mode_loss_weight,
            grad_clip=args.grad_clip,
            lock_rotation_axes=not args.allow_rotation_output,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            scaler=None,
            mode_criterion=mode_criterion,
            velocity_weight=args.velocity_loss_weight,
            mode_weight=args.mode_loss_weight,
            grad_clip=args.grad_clip,
            lock_rotation_axes=not args.allow_rotation_output,
        )
        scheduler.step(val_metrics["loss"])
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics["loss"],
            "train_velocity_loss": train_metrics["velocity_loss"],
            "train_mode_loss": train_metrics["mode_loss"],
            "train_mode_accuracy": train_metrics["mode_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_velocity_loss": val_metrics["velocity_loss"],
            "val_mode_loss": val_metrics["mode_loss"],
            "val_mode_accuracy": val_metrics["mode_accuracy"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={row['train_loss']:.5f} | val={row['val_loss']:.5f} | "
            f"val_mode_acc={row['val_mode_accuracy']:.3f} | lr={row['lr']:.2e}"
        )

        torch.save(_checkpoint_payload(model=model, config=config, normalization=normalization), latest_path)
        if val_metrics["loss"] < best_val - 1e-6:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(_checkpoint_payload(model=model, config=config, normalization=normalization), best_path)
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best validation loss was epoch {best_epoch}.")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_preds = _collect_predictions(
        model,
        val_loader,
        device=device,
        target_scale=normalization["target_scale"],
        lock_rotation_axes=not args.allow_rotation_output,
    )
    final_metrics = _metrics_from_predictions(
        val_preds,
        active_axis_count=6 if args.allow_rotation_output else LINEAR_AXIS_COUNT,
    )
    test_preds = None
    test_metrics = None
    if test_loader is not None:
        test_preds = _collect_predictions(
            model,
            test_loader,
            device=device,
            target_scale=normalization["target_scale"],
            lock_rotation_axes=not args.allow_rotation_output,
        )
        test_metrics = _metrics_from_predictions(
            test_preds,
            active_axis_count=(
                6 if args.allow_rotation_output else LINEAR_AXIS_COUNT
            ),
        )
    final_metrics.update(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_trials": len(train_episodes),
            "val_trials": len(val_episodes),
            "test_samples": len(test_dataset) if test_dataset is not None else 0,
            "test_trials": len(test_episodes),
            "test_metrics": test_metrics,
        }
    )

    torch.save(
        _checkpoint_payload(model=model, config=config, normalization=normalization, metrics=final_metrics),
        best_path,
    )
    sensor_rows, sensor_cols = (int(value) for value in config["sensor_shape"])
    labelled_best_path = output_dir / (
        f"best_model_{sensor_rows}x{sensor_cols}.pt"
    )
    shutil.copy2(best_path, labelled_best_path)
    _save_history_csv(output_dir / "history.csv", history)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "validation_predictions.npz", **val_preds)
    _save_predictions_csv(output_dir / "validation_predictions.csv", val_preds)
    if test_preds is not None:
        np.savez_compressed(output_dir / "test_predictions.npz", **test_preds)
        _save_predictions_csv(output_dir / "test_predictions.csv", test_preds)
    if not args.no_plots:
        _try_save_plots(output_dir, history, val_preds, final_metrics)

    if args.update_latest:
        latest_alias = args.output_dir.expanduser() / "latest_cnn_gru_model.pt"
        shape_alias = args.output_dir.expanduser() / (
            f"latest_cnn_gru_model_{sensor_rows}x{sensor_cols}.pt"
        )
        shutil.copy2(best_path, latest_alias)
        shutil.copy2(best_path, shape_alias)
        print(f"Latest CNN-GRU alias updated: {latest_alias}")
        print(f"Size-specific CNN-GRU alias updated: {shape_alias}")

    print("\nTraining complete")
    print(f"Best model: {best_path}")
    print(f"Size-labelled model: {labelled_best_path}")
    print(f"Metrics: {output_dir / 'metrics.json'}")
    print(f"Validation RMSE: {final_metrics['total_rmse']:.6f}")
    print(f"Validation mode accuracy: {final_metrics['mode_accuracy']:.3f}")
    if test_metrics is not None:
        print(f"Independent test RMSE: {test_metrics['total_rmse']:.6f}")
        print(
            f"Independent test mode accuracy: "
            f"{test_metrics['mode_accuracy']:.3f}"
        )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the AI Direct Finger Motion CNN-GRU tactile policy.")
    parser.add_argument(
        "sessions",
        nargs="*",
        default=[DEFAULT_SESSION],
        help=(
            "One or more session tags or paths. Example: "
            "first_trial_no_rotation push_focus_v1"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=_default_dataset_root(), help="Root containing session_* folders.")
    parser.add_argument("--output-dir", type=Path, default=_default_output_root(), help="Directory for model run outputs.")
    parser.add_argument("--run-name", default="", help="Optional output subfolder name.")
    parser.add_argument("--target-key", default=DEFAULT_TARGET_KEY, help="Velocity target key in each .npz file.")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Number of tactile frames per input sequence.")
    parser.add_argument("--channels", default=",".join(DEFAULT_CHANNELS), help="Comma-separated tactile input channels.")
    parser.add_argument("--aux-features", default=",".join(DEFAULT_AUX_FEATURES), help="Comma-separated auxiliary features.")
    parser.add_argument("--val-trials", type=int, default=1, help="Number of final trials reserved for validation.")
    parser.add_argument(
        "--test-trials",
        type=int,
        default=1,
        help=(
            "Number of final trials reserved as an independent test split. "
            "These trials are never used for optimization or early stopping."
        ),
    )
    parser.add_argument("--stride", type=int, default=1, help="Sequence sampling stride.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--velocity-loss-weight", type=float, default=1.0)
    parser.add_argument("--mode-loss-weight", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument(
        "--encoder",
        choices=("spatial_softmax", "avgpool"),
        default="spatial_softmax",
        help="Frame encoder. spatial_softmax preserves touch location; avgpool is the legacy encoder.",
    )
    parser.add_argument("--no-augment", action="store_true", help="Disable training-time data augmentation.")
    parser.add_argument(
        "--aug-flip-prob",
        type=float,
        default=0.5,
        help="Probability of each geometric flip (row/col) with matching velocity-target sign flips.",
    )
    parser.add_argument(
        "--aug-max-shift",
        type=int,
        default=2,
        help="Max random translation (cells) of the touch pattern; velocity target unchanged.",
    )
    parser.add_argument(
        "--aug-noise-std",
        type=float,
        default=0.02,
        help="Gaussian sensor-noise std as a fraction of each channel's training std (0 disables).",
    )
    parser.add_argument(
        "--aug-sensor-noise-max",
        type=float,
        default=0.0,
        help=(
            "Max i.i.d. sensor-noise std in raw signal units, applied consistently across "
            "channels (instantaneous, moving-average, temporal-difference, recomputed touch "
            "mask). Sampled uniformly per window with 50%% probability. 0 disables."
        ),
    )
    parser.add_argument(
        "--aug-drift-max",
        type=float,
        default=0.0,
        help=(
            "Max baseline-drift magnitude in raw signal units toward the touch threshold, "
            "with the touch mask recomputed. Sampled uniformly per window with 50%% "
            "probability. 0 disables."
        ),
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    parser.add_argument(
        "--allow-rotation-output",
        action="store_true",
        help="Train/evaluate rx/ry/rz outputs. Default locks rotation output to zero for no-rotation datasets.",
    )
    parser.add_argument(
        "--update-latest",
        action="store_true",
        help=(
            "Also copy the best checkpoint to generic and sensor-size-labelled "
            "latest_cnn_gru_model aliases in the output root."
        ),
    )
    args = parser.parse_args()

    if args.seq_len < 2:
        raise ValueError("--seq-len must be at least 2")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
