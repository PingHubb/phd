"""Streaming inference for normal-only tactile proximity checkpoints."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading

import numpy as np
import torch

from phd.dependence.tactile_models import TactileProximityCNNGRU


@dataclass(frozen=True)
class TactileProximityResult:
    ready: bool
    detected: bool
    anomaly_score: float
    threshold: float
    prediction_error: float
    latent_distance: float
    consecutive_anomalies: int
    model_anomaly_score: float = 0.0
    localized_change_score: float = 0.0
    localized_change_threshold: float = 0.0
    model_threshold: float = 0.0
    detection_mode: str = "hybrid"
    center_row: float | None = None
    center_col: float | None = None
    localization_source: str = "none"


def _strongest_spatial_region(values, spatial_window=2, top_taxels=3):
    """Return the strongest local score and its weighted grid center."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        return 0.0, None, None
    matrix = np.maximum(matrix, 0.0)
    window = min(max(1, int(spatial_window)), *matrix.shape)

    if window > 1:
        patches = np.lib.stride_tricks.sliding_window_view(
            matrix,
            (window, window),
        )
        patch_scores = patches.mean(axis=(-2, -1))
        start_row, start_col = np.unravel_index(
            int(np.argmax(patch_scores)),
            patch_scores.shape,
        )
        score = float(patch_scores[start_row, start_col])
        selected = matrix[
            start_row : start_row + window,
            start_col : start_col + window,
        ]
        rows, cols = np.indices(selected.shape, dtype=np.float32)
        rows += float(start_row)
        cols += float(start_col)
        weight_sum = float(selected.sum())
        if weight_sum <= 1e-12:
            return score, None, None
        center_row = float(np.sum(rows * selected) / weight_sum)
        center_col = float(np.sum(cols * selected) / weight_sum)
        return score, center_row, center_col

    flattened = matrix.reshape(-1)
    count = min(max(1, int(top_taxels)), int(flattened.size))
    selected_indices = np.argpartition(flattened, -count)[-count:]
    selected_weights = flattened[selected_indices]
    score = float(selected_weights.mean())
    weight_sum = float(selected_weights.sum())
    if weight_sum <= 1e-12:
        return score, None, None
    rows, cols = np.unravel_index(selected_indices, matrix.shape)
    center_row = float(np.sum(rows * selected_weights) / weight_sum)
    center_col = float(np.sum(cols * selected_weights) / weight_sum)
    return score, center_row, center_col


class LocalizedChangeAccumulator:
    """Accumulate weak, spatially local signal decreases over time."""

    def __init__(
        self,
        *,
        warmup_frames=30,
        top_taxels=3,
        spatial_window=2,
        allowance=0.3,
        decay=0.95,
        baseline_alpha=0.0005,
    ):
        self.warmup_frames = max(2, int(warmup_frames))
        self.top_taxels = max(1, int(top_taxels))
        self.spatial_window = max(1, int(spatial_window))
        self.allowance = max(0.0, float(allowance))
        self.decay = float(np.clip(decay, 0.0, 0.99999))
        self.baseline_alpha = float(
            np.clip(baseline_alpha, 0.0, 1.0)
        )
        self.reset()

    def reset(self):
        self._warmup = []
        self._baseline = None
        self._cusum = None
        self.center_row = None
        self.center_col = None

    @property
    def ready(self):
        return self._baseline is not None

    def update(self, frame):
        current = np.asarray(frame, dtype=np.float32)
        if current.ndim != 2 or not np.all(np.isfinite(current)):
            raise ValueError(
                "Localized-change input must be a finite 2D taxel frame"
            )

        if self._baseline is None:
            self._warmup.append(current.copy())
            if len(self._warmup) < self.warmup_frames:
                return None
            self._baseline = np.median(
                np.stack(self._warmup, axis=0),
                axis=0,
            ).astype(np.float32)
            self._cusum = np.zeros_like(self._baseline)
            self._warmup.clear()
            return None

        # Capacitive proximity in this sensor appears mainly as a decrease.
        # Removing the spatial median rejects common-mode environmental drift.
        residual = self._baseline - current
        residual = residual - np.median(residual)
        self._cusum = np.maximum(
            0.0,
            self.decay * self._cusum + residual - self.allowance,
        )
        score, self.center_row, self.center_col = _strongest_spatial_region(
            self._cusum,
            spatial_window=self.spatial_window,
            top_taxels=self.top_taxels,
        )
        alpha = self.baseline_alpha
        self._baseline = (
            (1.0 - alpha) * self._baseline + alpha * current
        ).astype(np.float32)
        return score


def localized_change_scores(frames, **kwargs):
    """Return the detector-matched local-change trace for one recording."""
    accumulator = LocalizedChangeAccumulator(**kwargs)
    scores = []
    for frame in np.asarray(frames, dtype=np.float32):
        score = accumulator.update(frame)
        if score is not None:
            scores.append(score)
    return np.asarray(scores, dtype=np.float64)


def build_tactile_proximity_channel_frame(
    channels,
    diff_frame,
    averaged_diff_frame,
    previous_averaged_diff_frame=None,
):
    """Build the live channel stack in the same form used for training."""
    diff_frame = np.asarray(diff_frame, dtype=np.float32)
    averaged_diff_frame = np.asarray(
        averaged_diff_frame,
        dtype=np.float32,
    )
    if diff_frame.shape != averaged_diff_frame.shape:
        raise ValueError(
            "Live diffPerData and diffPerDataAve shapes do not match"
        )

    if previous_averaged_diff_frame is None:
        frame_difference = np.zeros_like(averaged_diff_frame)
    else:
        previous = np.asarray(
            previous_averaged_diff_frame,
            dtype=np.float32,
        )
        if previous.shape != averaged_diff_frame.shape:
            raise ValueError(
                "Previous averaged tactile frame shape does not match"
            )
        frame_difference = averaged_diff_frame - previous

    available = {
        "diffPerData": diff_frame,
        "diffPerDataAve": averaged_diff_frame,
        "frameDiff": frame_difference,
    }
    unknown = [name for name in channels if name not in available]
    if unknown:
        raise ValueError(
            "Unsupported live proximity channel(s): " + ", ".join(unknown)
        )
    return np.stack(
        [available[name] for name in channels],
        axis=0,
    ).astype(np.float32)


class TactileProximityDetector:
    """Apply a trained CNN-GRU checkpoint to a rolling tactile stream."""

    DETECTION_MODES = ("hybrid", "cnn_gru", "localized")

    def __init__(
        self,
        checkpoint_path,
        *,
        device=None,
        consecutive_required=2,
        sensitivity=1.0,
        detection_mode="hybrid",
        release_required=3,
        release_ratio=1.0,
        localized_warmup_frames=None,
    ):
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        selected_device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(selected_device)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        if checkpoint.get("model_type") != "tactile_proximity_cnn_gru":
            raise ValueError("Checkpoint is not a tactile proximity CNN-GRU")

        self.config = dict(checkpoint["config"])
        self.channels = tuple(self.config["channels"])
        self.sequence_length = int(self.config["sequence_length"])
        self.rows = int(self.config["sensor_rows"])
        self.cols = int(self.config["sensor_cols"])
        self.input_clip = float(self.config.get("input_clip", 12.0))
        self.top_error_fraction = float(
            self.config.get("top_error_fraction", 0.1)
        )
        self.model = TactileProximityCNNGRU(
            in_channels=int(self.config["in_channels"]),
            sensor_rows=self.rows,
            sensor_cols=self.cols,
            d_model=int(self.config["d_model"]),
            gru_hidden=int(self.config["gru_hidden"]),
            gru_layers=int(self.config["gru_layers"]),
            dropout=float(self.config["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        normalization = checkpoint["normalization"]
        self.normalization_mean = np.asarray(
            normalization["mean"],
            dtype=np.float32,
        )
        self.normalization_std = np.asarray(
            normalization["std"],
            dtype=np.float32,
        )
        calibration = checkpoint.get("anomaly_calibration") or {}
        required = {
            "latent_mean",
            "latent_std",
            "prediction_center",
            "prediction_scale",
            "latent_center",
            "latent_scale",
            "prediction_weight",
            "threshold",
        }
        missing = sorted(required.difference(calibration))
        if missing:
            raise ValueError(
                "Checkpoint has no completed anomaly calibration: "
                + ", ".join(missing)
            )
        self.latent_mean = np.asarray(
            calibration["latent_mean"],
            dtype=np.float64,
        )
        self.latent_std = np.asarray(
            calibration["latent_std"],
            dtype=np.float64,
        )
        self.prediction_center = float(calibration["prediction_center"])
        self.prediction_scale = float(calibration["prediction_scale"])
        self.latent_center = float(calibration["latent_center"])
        self.latent_scale = float(calibration["latent_scale"])
        self.prediction_weight = float(calibration["prediction_weight"])
        self.model_threshold = float(calibration["threshold"])
        self.localized_change_channel = calibration.get(
            "localized_change_channel"
        )
        self.localized_change_threshold = float(
            calibration.get("localized_change_threshold", 0.0)
        )
        self._localized_channel_index = None
        self._localized_change = None
        self.warmup_frames = self.sequence_length
        if (
            self.localized_change_channel in self.channels
            and self.localized_change_threshold > 0.0
        ):
            self._localized_channel_index = self.channels.index(
                self.localized_change_channel
            )
            calibrated_warmup_frames = calibration.get(
                "localized_change_warmup_frames", 30
            )
            active_warmup_frames = (
                calibrated_warmup_frames
                if localized_warmup_frames is None
                else localized_warmup_frames
            )
            self._localized_change = LocalizedChangeAccumulator(
                warmup_frames=active_warmup_frames,
                top_taxels=calibration.get(
                    "localized_change_top_taxels", 3
                ),
                spatial_window=calibration.get(
                    "localized_change_spatial_window", 2
                ),
                allowance=calibration.get(
                    "localized_change_allowance", 0.3
                ),
                decay=calibration.get("localized_change_decay", 0.95),
                baseline_alpha=calibration.get(
                    "localized_change_baseline_alpha", 0.0005
                ),
            )
            self.warmup_frames = max(
                self.sequence_length,
                self._localized_change.warmup_frames,
            )
        self._lock = threading.Lock()
        self.sensitivity = 1.0
        self.detection_mode = "hybrid"
        self.set_sensitivity(sensitivity)
        self.set_detection_mode(detection_mode)
        # Scores are normalized by their calibrated normal-data thresholds.
        self.threshold = 1.0
        self.consecutive_required = max(1, int(consecutive_required))
        self.release_required = max(1, int(release_required))
        self.release_ratio = float(np.clip(release_ratio, 0.1, 1.0))
        self._history = deque(maxlen=self.sequence_length)
        self._consecutive_anomalies = 0
        self._consecutive_normal = 0
        self._detected = False

    def reset(self):
        with self._lock:
            self._history.clear()
            self._consecutive_anomalies = 0
            self._consecutive_normal = 0
            self._detected = False
            if self._localized_change is not None:
                self._localized_change.reset()

    def set_sensitivity(self, value):
        """Scale only the local-change threshold; lower is more sensitive."""
        with self._lock:
            self.sensitivity = float(np.clip(value, 0.5, 1.5))
            self.threshold = 1.0

    @classmethod
    def normalize_detection_mode(cls, value):
        mode = str(value or "hybrid").strip().lower().replace("-", "_")
        aliases = {
            "cnn": "cnn_gru",
            "cnn_gru_only": "cnn_gru",
            "ai": "cnn_gru",
            "local": "localized",
            "localized_only": "localized",
        }
        mode = aliases.get(mode, mode)
        if mode not in cls.DETECTION_MODES:
            raise ValueError(
                "Detection mode must be hybrid, cnn_gru, or localized"
            )
        return mode

    def set_detection_mode(self, value):
        """Select which normalized branch can trigger detection."""
        mode = self.normalize_detection_mode(value)
        with self._lock:
            changed = mode != self.detection_mode
            self.detection_mode = mode
            if changed and hasattr(self, "_consecutive_anomalies"):
                self._consecutive_anomalies = 0
                self._consecutive_normal = 0
                self._detected = False
        return mode

    @property
    def active_warmup_frames(self):
        if self.detection_mode == "cnn_gru":
            return self.sequence_length
        return self.warmup_frames

    def _active_localized_change_threshold(self):
        if self.localized_change_threshold <= 0.0:
            return 0.0
        return self.localized_change_threshold * self.sensitivity

    def _normalize(self, channel_frame):
        frame = np.asarray(channel_frame, dtype=np.float32)
        expected = (len(self.channels), self.rows, self.cols)
        if frame.shape != expected:
            raise ValueError(
                f"Tactile channel frame shape is {frame.shape}; expected "
                f"{expected}"
            )
        if not np.all(np.isfinite(frame)):
            raise ValueError("Tactile channel frame contains invalid values")
        normalized = (
            frame - self.normalization_mean
        ) / self.normalization_std
        return np.clip(
            normalized,
            -self.input_clip,
            self.input_clip,
        ).astype(np.float32)

    def _empty_result(self):
        return TactileProximityResult(
            ready=False,
            detected=False,
            anomaly_score=0.0,
            threshold=self.threshold,
            prediction_error=0.0,
            latent_distance=0.0,
            consecutive_anomalies=0,
            model_anomaly_score=0.0,
            localized_change_score=0.0,
            localized_change_threshold=(
                self._active_localized_change_threshold()
            ),
            model_threshold=self.model_threshold,
            detection_mode=self.detection_mode,
        )

    def update(self, channel_frame):
        """Score the current frame against its preceding normal history."""
        normalized = self._normalize(channel_frame)
        with self._lock:
            localized_score = 0.0
            localized_ready = True
            localized_center = (None, None)
            if self._localized_change is not None:
                value = self._localized_change.update(
                    normalized[self._localized_channel_index]
                )
                localized_ready = value is not None
                if value is not None:
                    localized_score = float(value)
                    localized_center = (
                        self._localized_change.center_row,
                        self._localized_change.center_col,
                    )
            if len(self._history) < self.sequence_length:
                self._history.append(normalized)
                return self._empty_result()
            if not localized_ready and self.detection_mode != "cnn_gru":
                self._history.append(normalized)
                return self._empty_result()

            prediction_error = 0.0
            latent_distance = 0.0
            model_anomaly_score = 0.0
            model_ratio = 0.0
            model_center = (None, None)
            if self.detection_mode != "localized":
                history = torch.from_numpy(
                    np.stack(tuple(self._history), axis=0)[None]
                ).to(self.device)
                current = torch.from_numpy(normalized[None]).to(self.device)
                with torch.no_grad():
                    prediction = self.model(history)["predicted_frame"]
                    absolute_error_frame = (prediction - current).abs()
                    absolute_error = absolute_error_frame.flatten(1)
                    top_count = max(
                        1,
                        int(
                            np.ceil(
                                absolute_error.shape[1]
                                * self.top_error_fraction
                            )
                        ),
                    )
                    prediction_error = float(
                        absolute_error.topk(top_count, dim=1)
                        .values.mean()
                        .cpu()
                    )
                    embedding = (
                        self.model.encode_frames(current)
                        .cpu()
                        .numpy()[0]
                        .astype(np.float64)
                    )
                location_channel = (
                    self._localized_channel_index
                    if self._localized_channel_index is not None
                    else 0
                )
                _, model_center_row, model_center_col = (
                    _strongest_spatial_region(
                        absolute_error_frame[
                            0,
                            location_channel,
                        ].cpu().numpy(),
                        spatial_window=2,
                        top_taxels=3,
                    )
                )
                model_center = (model_center_row, model_center_col)

                latent_z = (embedding - self.latent_mean) / self.latent_std
                latent_distance = float(
                    np.sqrt(np.mean(np.square(latent_z)))
                )
                prediction_z = max(
                    0.0,
                    (prediction_error - self.prediction_center)
                    / self.prediction_scale,
                )
                latent_score = max(
                    0.0,
                    (latent_distance - self.latent_center)
                    / self.latent_scale,
                )
                model_anomaly_score = (
                    self.prediction_weight * prediction_z
                    + (1.0 - self.prediction_weight) * latent_score
                )
                model_ratio = model_anomaly_score / max(
                    self.model_threshold,
                    1e-8,
                )
            localized_ratio = 0.0
            if self._localized_change is not None:
                localized_ratio = localized_score / max(
                    self._active_localized_change_threshold(),
                    1e-8,
                )
            if self.detection_mode == "cnn_gru":
                anomaly_score = model_ratio
            elif self.detection_mode == "localized":
                anomaly_score = localized_ratio
            else:
                anomaly_score = max(model_ratio, localized_ratio)
            if self.detection_mode == "localized":
                center_row, center_col = localized_center
                localization_source = "localized"
            elif self.detection_mode == "cnn_gru":
                center_row, center_col = model_center
                localization_source = "cnn_gru"
            elif localized_ratio >= model_ratio and all(
                value is not None for value in localized_center
            ):
                center_row, center_col = localized_center
                localization_source = "localized"
            else:
                center_row, center_col = model_center
                localization_source = "cnn_gru"
            is_anomaly = anomaly_score > self.threshold
            if is_anomaly:
                self._consecutive_anomalies += 1
                self._consecutive_normal = 0
                if (
                    self._consecutive_anomalies
                    >= self.consecutive_required
                ):
                    self._detected = True
            else:
                self._consecutive_anomalies = 0
                if (
                    self._detected
                    and anomaly_score
                    < self.threshold * self.release_ratio
                ):
                    self._consecutive_normal += 1
                    if self._consecutive_normal >= self.release_required:
                        self._detected = False
                        self._consecutive_normal = 0
                else:
                    self._consecutive_normal = 0
            detected = self._detected
            self._history.append(normalized)
            return TactileProximityResult(
                ready=True,
                detected=bool(detected),
                anomaly_score=float(anomaly_score),
                threshold=self.threshold,
                prediction_error=prediction_error,
                latent_distance=latent_distance,
                consecutive_anomalies=self._consecutive_anomalies,
                model_anomaly_score=float(model_anomaly_score),
                localized_change_score=float(localized_score),
                localized_change_threshold=float(
                    self._active_localized_change_threshold()
                ),
                model_threshold=self.model_threshold,
                detection_mode=self.detection_mode,
                center_row=center_row,
                center_col=center_col,
                localization_source=localization_source,
            )
