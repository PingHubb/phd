"""Bounded touch/release segmentation, independent of Qt and robot control."""

import numpy as np


class GestureTrialSegmenter:
    """Consume fresh native-layout percentage frames, never absolute capacitance.

    An initial confirmed release prevents recognising a gesture already underway.
    A pause, reset or invalid frame is an error, not evidence of finger release.
    """

    RELEASE_FRAMES = 3
    RELEASE_SECONDS = 0.12
    MAX_GAP_SECONDS = 0.5
    MAX_SECONDS = 30.0
    MAX_FRAMES = 2000

    def __init__(self, shape=(8, 8), threshold=-3.0, minimum_touch_frames=2):
        self.shape = tuple(shape)
        self.threshold = float(threshold)
        self.minimum_touch_frames = int(minimum_touch_frames)
        self.state = "release"
        self.last_sequence = None
        self.last_time = None
        self.frames = []
        self.touch_frames = 0
        self.release_frames = 0
        self.release_start = None

    def _released(self, touching, timestamp):
        if touching:
            self.release_frames = 0
            self.release_start = None
        else:
            self.release_frames += 1
            if self.release_start is None:
                self.release_start = timestamp
        return (self.release_frames >= self.RELEASE_FRAMES
                and timestamp - self.release_start >= self.RELEASE_SECONDS)

    def feed(self, sequence, timestamp, instantaneous, averaged):
        """Return (instantaneous, averaged, times) once per completed gesture."""
        sequence, timestamp = int(sequence), float(timestamp)
        if sequence == self.last_sequence:
            return None
        raw = np.asarray(instantaneous, dtype=np.float32)
        avg = np.asarray(averaged, dtype=np.float32)
        if (raw.shape != self.shape or avg.shape != self.shape
                or not np.isfinite(raw).all() or not np.isfinite(avg).all()
                or not np.isfinite(timestamp)):
            raise ValueError("Invalid sensor frame; this gesture was discarded.")
        if self.last_sequence is not None and sequence < self.last_sequence:
            raise ValueError("Sensor frame counter reset; restart recognition after calibration.")
        if self.last_time is not None:
            if timestamp <= self.last_time:
                raise ValueError("Sensor timestamps did not increase.")
            if timestamp - self.last_time > self.MAX_GAP_SECONDS:
                raise ValueError("Sensor stream paused; incomplete gesture discarded. Restart recognition.")
        self.last_sequence, self.last_time = sequence, timestamp
        touching = bool(np.any(avg < self.threshold))
        released = self._released(touching, timestamp)
        if self.state == "release":
            if released:
                self.state = "ready"
            return None
        if self.state == "ready":
            if not touching:
                return None
            self.state = "recording"
        self.frames.append((raw.copy(), avg.copy(), timestamp))
        self.touch_frames += int(touching)
        if (len(self.frames) > self.MAX_FRAMES
                or timestamp - self.frames[0][2] > self.MAX_SECONDS):
            self.frames.clear()
            self.touch_frames = 0
            self.state = "release"
            return None
        if not released:
            return None
        frames, count = self.frames, self.touch_frames
        self.frames, self.touch_frames, self.state = [], 0, "ready"
        if count < self.minimum_touch_frames:
            return None
        raw, avg, times = zip(*frames)
        times = np.asarray(times, dtype=np.float64)
        return np.stack(raw), np.stack(avg), times - times[0]
