"""Timestamped tactile frames for reproducible 3D heatmap playback."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np


HEATMAP_RECORDING_FORMAT_VERSION = 1


def _matrix(value, name, expected_shape=None):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 2D matrix")
    if expected_shape is not None and array.shape != tuple(expected_shape):
        raise ValueError(
            f"{name} shape {array.shape} does not match {tuple(expected_shape)}"
        )
    return np.array(array, dtype=np.float32, copy=True)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class HeatmapSignalRecorder:
    """Collect heatmap/raw sensor snapshots aligned with constant-FPS video frames."""

    def __init__(self, fps=60.0, clock=None):
        self.fps = max(1.0, float(fps))
        self._clock = clock or time.perf_counter
        self.output_path = None
        self.video_path = None
        self.metadata = {}
        self.calibration = None
        self._heatmap_frames = []
        self._raw_frames = []
        self._capture_times = []
        self._started_at = None
        self.last_error = ""

    @property
    def is_recording(self):
        return self._started_at is not None

    @property
    def frame_count(self):
        return len(self._heatmap_frames)

    def _append_snapshot(self, snapshot):
        heatmap = _matrix(snapshot.get("heatmap_values"), "heatmap_values")
        expected_shape = heatmap.shape
        if self._heatmap_frames:
            expected_shape = self._heatmap_frames[0].shape
            heatmap = _matrix(
                snapshot.get("heatmap_values"),
                "heatmap_values",
                expected_shape=expected_shape,
            )

        raw_value = snapshot.get("raw_values")
        raw = (
            _matrix(raw_value, "raw_values", expected_shape=expected_shape)
            if raw_value is not None
            else np.full(expected_shape, np.nan, dtype=np.float32)
        )
        self._heatmap_frames.append(heatmap)
        self._raw_frames.append(raw)
        self._capture_times.append(float(self._clock()) - self._started_at)

    def start(self, output_path, first_snapshot, metadata=None, video_path=None):
        self.last_error = ""
        if self.is_recording:
            self.last_error = "A tactile recording is already active."
            return False
        try:
            path = Path(output_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            calibration_value = first_snapshot.get("calibration_values")
            first_heatmap = _matrix(
                first_snapshot.get("heatmap_values"), "heatmap_values"
            )
            self.calibration = (
                _matrix(
                    calibration_value,
                    "calibration_values",
                    expected_shape=first_heatmap.shape,
                )
                if calibration_value is not None
                else np.full(first_heatmap.shape, np.nan, dtype=np.float32)
            )
            self.output_path = path
            self.video_path = Path(video_path).expanduser() if video_path else None
            self.metadata = dict(metadata or {})
            self._heatmap_frames = []
            self._raw_frames = []
            self._capture_times = []
            self._started_at = float(self._clock())
            self._append_snapshot(first_snapshot)
            return True
        except Exception as exc:
            self.last_error = f"Could not start tactile recording: {exc}"
            self._started_at = None
            return False

    def capture(self, snapshot):
        if not self.is_recording:
            return False
        try:
            self._append_snapshot(snapshot)
            return True
        except Exception as exc:
            self.last_error = f"Could not capture tactile frame: {exc}"
            return False

    def align_timeline_start(self, started_at):
        """Align capture timestamps with an externally defined run start."""
        if not self.is_recording:
            return False
        self._started_at = float(started_at)
        if self._capture_times:
            self._capture_times[0] = 0.0
        return True

    def capture_to_frame_count(self, snapshot, target_frame_count):
        """Append one current sample and hold the prior sample across gaps."""
        if not self.is_recording:
            return False
        target = max(0, int(target_frame_count))
        additional = target - self.frame_count
        if additional <= 0:
            return True

        initial_count = self.frame_count
        try:
            if initial_count == 0:
                raise RuntimeError("No initial tactile frame is available")
            for _ in range(additional - 1):
                self._heatmap_frames.append(
                    np.array(self._heatmap_frames[-1], copy=True)
                )
                self._raw_frames.append(np.array(self._raw_frames[-1], copy=True))
                self._capture_times.append(float(self._capture_times[-1]))
            self._append_snapshot(snapshot)
            return True
        except Exception as exc:
            del self._heatmap_frames[initial_count:]
            del self._raw_frames[initial_count:]
            del self._capture_times[initial_count:]
            self.last_error = f"Could not capture tactile frame: {exc}"
            return False

    def discard_last_frames(self, count=1):
        remove_count = min(max(0, int(count)), self.frame_count)
        if remove_count <= 0:
            return False
        del self._heatmap_frames[-remove_count:]
        del self._raw_frames[-remove_count:]
        del self._capture_times[-remove_count:]
        return True

    def discard_last_frame(self):
        return self.discard_last_frames(1)

    def stop(self, discard=False):
        if not self.is_recording:
            return self.output_path
        self._started_at = None
        path = self.output_path
        if discard:
            try:
                if path is not None:
                    path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        try:
            heatmap_frames = np.stack(self._heatmap_frames).astype(
                np.float32, copy=False
            )
            raw_frames = np.stack(self._raw_frames).astype(np.float32, copy=False)
            video_times = np.arange(self.frame_count, dtype=np.float64) / self.fps
            capture_times = np.asarray(self._capture_times, dtype=np.float64)
            metadata_json = json.dumps(
                self.metadata,
                default=_json_default,
                sort_keys=True,
                separators=(",", ":"),
            )
            with path.open("wb") as output_file:
                np.savez_compressed(
                    output_file,
                    format_version=np.asarray(
                        HEATMAP_RECORDING_FORMAT_VERSION, dtype=np.int16
                    ),
                    fps=np.asarray(self.fps, dtype=np.float64),
                    video_timestamps_s=video_times,
                    capture_timestamps_s=capture_times,
                    heatmap_frames=heatmap_frames,
                    raw_frames=raw_frames,
                    calibration_values=np.asarray(
                        self.calibration, dtype=np.float32
                    ),
                    metadata_json=np.asarray(metadata_json),
                    video_filename=np.asarray(
                        self.video_path.name if self.video_path else ""
                    ),
                )
            return path
        except Exception as exc:
            self.last_error = f"Could not save tactile recording: {exc}"
            return None


@dataclass(frozen=True)
class HeatmapSignalPlayback:
    path: Path
    fps: float
    timestamps_s: np.ndarray
    capture_timestamps_s: np.ndarray
    heatmap_frames: np.ndarray
    raw_frames: np.ndarray
    calibration_values: np.ndarray
    metadata: dict
    video_filename: str

    @classmethod
    def load(cls, path):
        recording_path = Path(path).expanduser()
        with np.load(recording_path, allow_pickle=False) as payload:
            required = {
                "format_version",
                "fps",
                "video_timestamps_s",
                "capture_timestamps_s",
                "heatmap_frames",
                "raw_frames",
                "calibration_values",
                "metadata_json",
                "video_filename",
            }
            missing = sorted(required.difference(payload.files))
            if missing:
                raise ValueError(
                    "Tactile recording is missing: " + ", ".join(missing)
                )
            version = int(np.asarray(payload["format_version"]).item())
            if version != HEATMAP_RECORDING_FORMAT_VERSION:
                raise ValueError(f"Unsupported tactile recording version: {version}")

            frames = np.asarray(payload["heatmap_frames"], dtype=np.float32)
            raw_frames = np.asarray(payload["raw_frames"], dtype=np.float32)
            timestamps = np.asarray(
                payload["video_timestamps_s"], dtype=np.float64
            )
            capture_timestamps = np.asarray(
                payload["capture_timestamps_s"], dtype=np.float64
            )
            if frames.ndim != 3 or frames.shape[0] == 0:
                raise ValueError("Tactile recording contains no valid heatmap frames")
            if raw_frames.shape != frames.shape:
                raise ValueError("Raw and heatmap frame shapes do not match")
            if timestamps.shape != (frames.shape[0],):
                raise ValueError("Video timestamps do not match the frame count")
            if capture_timestamps.shape != timestamps.shape:
                raise ValueError("Capture timestamps do not match the frame count")
            if not np.all(np.isfinite(frames)):
                raise ValueError("Heatmap recording contains invalid values")
            if not np.all(np.diff(timestamps) >= 0.0):
                raise ValueError("Heatmap recording timestamps are not ordered")

            metadata_text = str(np.asarray(payload["metadata_json"]).item())
            metadata = json.loads(metadata_text) if metadata_text else {}
            if not isinstance(metadata, dict):
                raise ValueError("Heatmap recording metadata is invalid")

            return cls(
                path=recording_path,
                fps=max(1.0, float(np.asarray(payload["fps"]).item())),
                timestamps_s=np.array(timestamps, copy=True),
                capture_timestamps_s=np.array(capture_timestamps, copy=True),
                heatmap_frames=np.array(frames, copy=True),
                raw_frames=np.array(raw_frames, copy=True),
                calibration_values=np.array(
                    payload["calibration_values"], dtype=np.float32, copy=True
                ),
                metadata=metadata,
                video_filename=str(np.asarray(payload["video_filename"]).item()),
            )

    @property
    def frame_count(self):
        return int(self.heatmap_frames.shape[0])

    @property
    def frame_shape(self):
        return tuple(int(value) for value in self.heatmap_frames.shape[1:])

    @property
    def duration_s(self):
        return float(self.timestamps_s[-1]) if self.frame_count else 0.0

    def frame_index_at(self, elapsed_s):
        elapsed = max(0.0, float(elapsed_s))
        index = int(np.searchsorted(self.timestamps_s, elapsed, side="right") - 1)
        return min(max(index, 0), self.frame_count - 1)

    def frame_at(self, elapsed_s):
        return self.heatmap_frames[self.frame_index_at(elapsed_s)]
