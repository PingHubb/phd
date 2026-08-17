"""MP4 recording for a live PyVista/QtInteractor viewport."""

from __future__ import annotations

from importlib import import_module
import math
from pathlib import Path

import numpy as np


class PlotterVideoRecorder:
    def __init__(self, fps=20.0, cv2_module=None):
        self.fps = max(1.0, float(fps))
        self._cv2 = cv2_module
        self._plotter = None
        self._writer = None
        self._frame_size = None
        self._last_rgb_frame = None
        self.output_path = None
        self.frame_count = 0
        self.last_error = ""

    @property
    def is_recording(self):
        return self._writer is not None

    def _opencv(self):
        if self._cv2 is None:
            self._cv2 = import_module("cv2")
        return self._cv2

    def _read_rgb_frame(self):
        if self._plotter is None:
            raise RuntimeError("3D plotter is unavailable")
        frame = np.asarray(
            self._plotter.screenshot(return_img=True), dtype=np.uint8
        )
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise RuntimeError("3D plotter returned an invalid image")
        frame = np.ascontiguousarray(frame[..., :3])
        height = int(frame.shape[0]) - int(frame.shape[0]) % 2
        width = int(frame.shape[1]) - int(frame.shape[1]) % 2
        if height < 2 or width < 2:
            raise RuntimeError("3D plotter image is too small to record")
        return np.ascontiguousarray(frame[:height, :width])

    def _write_rgb_frame(self, frame):
        cv2 = self._opencv()
        target_width, target_height = self._frame_size
        if frame.shape[:2] != (target_height, target_width):
            frame = cv2.resize(
                frame,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._writer.write(bgr_frame)
        self.frame_count += 1

    def frame_count_for_duration(self, elapsed_s):
        """Number of encoded frames needed to cover real elapsed time."""
        elapsed = max(0.0, float(elapsed_s))
        # The epsilon prevents an exact frame boundary represented as a tiny
        # floating-point overshoot from adding an unwanted extra frame.
        return max(1, int(math.ceil((elapsed * self.fps) - 1e-9)))

    def frames_due_for_elapsed(self, elapsed_s):
        if not self.is_recording:
            return 0
        return max(0, self.frame_count_for_duration(elapsed_s) - self.frame_count)

    def start(self, plotter, output_path) -> bool:
        self.last_error = ""
        if self.is_recording:
            self.last_error = "A 3D viewport recording is already active."
            return False
        try:
            self._plotter = plotter
            first_frame = self._read_rgb_frame()
            height, width = first_frame.shape[:2]
            path = Path(output_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2 = self._opencv()
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
            if not writer.isOpened():
                writer.release()
                raise RuntimeError("OpenCV could not open the MP4 encoder")
            self._writer = writer
            self._frame_size = (width, height)
            self.output_path = path
            self.frame_count = 0
            self._write_rgb_frame(first_frame)
            self._last_rgb_frame = np.array(first_frame, copy=True)
            return True
        except Exception as exc:
            self.last_error = f"Could not start 3D heatmap recording: {exc}"
            if self._writer is not None:
                try:
                    self._writer.release()
                except Exception:
                    pass
            self._writer = None
            self._plotter = None
            self._frame_size = None
            self._last_rgb_frame = None
            return False

    def capture_frame(self, repeat_count=1) -> bool:
        if not self.is_recording:
            return False
        repeat_count = max(0, int(repeat_count))
        if repeat_count == 0:
            return True
        try:
            current_frame = self._read_rgb_frame()
            previous_frame = self._last_rgb_frame
            if previous_frame is None:
                previous_frame = current_frame
            for _ in range(repeat_count - 1):
                self._write_rgb_frame(previous_frame)
            self._write_rgb_frame(current_frame)
            self._last_rgb_frame = np.array(current_frame, copy=True)
            return True
        except Exception as exc:
            self.last_error = f"Could not capture 3D heatmap frame: {exc}"
            return False

    def stop(self, capture_final=True):
        if not self.is_recording:
            return self.output_path
        if capture_final:
            self.capture_frame()
        writer = self._writer
        self._writer = None
        try:
            writer.release()
        finally:
            self._plotter = None
            self._frame_size = None
            self._last_rgb_frame = None
        return self.output_path
